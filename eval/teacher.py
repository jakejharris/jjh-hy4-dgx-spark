#!/usr/bin/env python3
"""Full-precision reference, prefix control, and paired end-to-end null."""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))
from eval_bf16_teacher import (
    ShardManager, atomic_save_safetensors, atomic_write_json, disk_name, free_gpu,
    indexed_load_device, load_layer_weights, read_json, ref, run_decoder_layer_batches,
    run_embed_tokens, sha256_file, text_config, utc_now, weight_dtype,
)
from capture import layer_capture
from compare import compare_tensor

DEVICE = "cuda"
PANEL_SHA = "c7091d0c42e93e157fbbed90aa7487b3f84417438b4021529127d8a82c2c6598"
NULL_SHORT = ("prose-01", "code-01", "prose-02", "code-02")


def text_keys(shards):
    return [key for key in shards.weight_map if key.startswith(
        ("model.embed_tokens.", "model.layers.", "model.hc_head.", "model.norm.", "lm_head.")
    )]


class VerifiedShards(ShardManager):
    """Verify surviving cache files on first use after each process restart."""

    def __init__(self, source, inputs, cache, keep_layers=None):
        super().__init__(source, inputs, cache, keep_layers)
        self.checked = set()
        if any(not self.manifest[self.weight_map[k]].get("sha256") for k in text_keys(self)):
            raise ValueError("Missing text shard hash")
        for cached, pinned in ((self.index_path, inputs / "index.json"),
                               (self.config_path, inputs / "config.json")):
            if sha256_file(cached) != sha256_file(pinned):
                raise ValueError(f"Stale cache metadata {cached}")

    def _download_http(self, rel_path, dst):
        if dst.is_file() and rel_path not in self.checked:
            if sha256_file(dst) != self.manifest[rel_path]["sha256"]:
                raise ValueError(f"Cached shard hash mismatch: {dst}")
        super()._download_http(rel_path, dst)
        self.checked.add(rel_path)


def build_layer(shards, config, index):
    with torch.device("meta"):
        layer = ref.HYV4VLDecoderLayer(config, index).eval()
    names = {disk_name(k, index) for k in layer.state_dict()}
    indexed = {k for k in shards.weight_map if k.startswith(f"model.layers.{index}.")}
    if names != indexed:
        raise ValueError(f"L{index} tensor map mismatch: {names ^ indexed}")
    shards.ensure_shards(names)
    load_layer_weights(shards.index_path, layer, torch.bfloat16, DEVICE)
    return layer


def project(shards, config, groups):
    """Apply trained iHC head, RMSNorm, and FP32 vocabulary projection."""
    with torch.device("meta"):
        head = ref.HYV4VLHyperHead(config).eval()
        norm = ref.HYV4VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps).eval()
    names = {disk_name(f"hc_head.{k}"): k for k in head.state_dict()}
    norm_names = {disk_name(f"norm.{k}"): k for k in norm.state_dict()}
    lm_key = "lm_head.weight"
    keys = [*names, *norm_names, lm_key]
    shards.ensure_shards(keys)
    weights = dict(indexed_load_device(shards.index_path, keys, DEVICE))
    for module, mapping in ((head, names), (norm, norm_names)):
        module.load_state_dict({k: weights[disk].to(weight_dtype(k, torch.bfloat16))
                                for disk, k in mapping.items()}, assign=True)
    lm = weights[lm_key].float()
    return [torch.nn.functional.linear(norm(head(s.to(DEVICE).unsqueeze(0))).float(), lm)
            .squeeze(0).cpu() for s in groups]


def select_positions(streams, rows, positions):
    return streams[rows, torch.tensor(positions, dtype=torch.long) - 1]


def check_records(reference, candidate, bands, exact=()):
    if candidate.keys() != reference.keys():
        raise ValueError(f"Control key mismatch: {candidate.keys() ^ reference.keys()}")
    results = {}
    for name in sorted(reference):
        a, b = reference[name], candidate[name].cpu()
        if name in exact:
            results[name] = 0 if torch.equal(a, b) else 2
            print(f"{'PASS' if results[name] == 0 else 'MISS'} exact {name}")
        else:
            results[name] = compare_tensor(name, a, b, bands[name.rsplit(".", 1)[-1]])
    return results


def control(shards, config, out, identity):
    refs = ROOT / "controls"
    locked = read_json(refs / "bands-cuda.json")
    defaults = read_json(refs / "tolerances.json")
    rot = ref.HYV4VLRotaryEmbedding(config).to(DEVICE)
    port = load_file(refs / "real_l0_prompt_head_ref_cuda.safetensors")
    shards.keep_shards.add(shards.weight_map["model.embed_tokens.weight"])
    (embedded,) = run_embed_tokens(shards, config, [port["input_ids"]], DEVICE)
    levels = {}
    for i in range(3):
        layer = build_layer(shards, config, i)
        records = {}
        layer_capture(layer, records)
        expected_path = refs / f"ref_cuda_L{i}_prompt16.safetensors"
        if i == 0:
            a = {k: v for k, v in port.items() if k.startswith("layer.000.")}
            fixture = {"streams_in": a["layer.000.streams_in"], "position_ids": torch.arange(16)[None, :]}
        else:
            a = load_file(expected_path)
            fixture = load_file(refs / f"L{i}_in_prompt16.safetensors")
        if not torch.equal(fixture["position_ids"], torch.arange(16).reshape(1, -1)):
            raise ValueError("Unexpected control position IDs")
        streams = fixture["streams_in"]
        previous = fixture.get("prev_topk_indices")
        if previous is not None:
            previous = previous.masked_fill(previous == -1, 15)
        if i == 0:
            if not torch.equal(embedded, streams):
                raise ValueError("Embedding differs from the frozen oracle input")
            streams = embedded
        expected_indices = previous.clone() if previous is not None else None
        final, indices = run_decoder_layer_batches(layer, config, rot, streams.clone(), previous, DEVICE, 1)
        if i == 2 and not torch.equal(indices, expected_indices):
            raise ValueError("L2 changed its shared DSA selection")
        atomic_save_safetensors(out / f"control-l{i}.safetensors", records)
        bands = {**defaults, **locked["real_layer_pass"][f"l{i}"]}
        levels.update(check_records(a, records, bands, exact=(f"layer.{i:03d}.streams_in",)))
        del layer, records
        free_gpu(DEVICE)
        if i == 0:
            positions = list(range(1, 17))
            shifted = positions[1:] + positions[:1]
            clean, planted = project(shards, config, [
                select_positions(final[0], slice(None), p) for p in (positions, shifted)
            ])
            clean_level = compare_tensor("clean L0+head logits", port["logits"], clean.unsqueeze(0), locked["logits"])
            plant_level = compare_tensor("planted one-position shift", port["logits"], planted.unsqueeze(0), locked["logits"])
            levels["L0+head"] = clean_level
            if plant_level != 2:
                raise ValueError("Control failed to reject the planted one-position shift")
    if any(levels.values()):
        raise ValueError(f"Teacher control outside locked PASS bands: {levels}")
    atomic_write_json(out / "control.json", {"identity": identity, "utc": utc_now(),
                      "checks": levels, "planted_position_shift": "MISS"})


def frozen_rows():
    path = ROOT / "inputs/teacher-prompts.json"
    if sha256_file(path) != PANEL_SHA:
        raise ValueError("Panel digest mismatch")
    panel = read_json(path)
    rows = panel["prompts"]
    if sorted(len(row["input_ids"]) for row in rows) != [1024] * 16 + [4096] * 4:
        raise ValueError("Unexpected panel rows")
    probes = read_json(ROOT / "thinking-probes.json")["renders"]
    alt = [i for i, r in enumerate(rows) if r["id"] in NULL_SHORT or len(r["input_ids"]) == 4096]
    return rows, alt, [probes["high"], probes["no_think"]]


def grouping(rows, selected, batch_sizes):
    groups = {}
    for i in selected:
        n = len(rows[i]["input_ids"])
        groups.setdefault(n, []).append(i)
    return [(n, ids, batch_sizes[n]) for n, ids in sorted(groups.items())]


def score_inputs(rows, groups, stream):
    by_row = {}
    for length, indices, _ in groups:
        for offset, i in enumerate(indices):
            by_row[i] = stream[length][offset]
    ids, pos, vectors = [], [], []
    for i, state in sorted(by_row.items()):
        positions = rows[i]["score_positions_1based"]
        vectors.append(select_positions(state, slice(None), positions))
        ids.extend([i] * len(positions))
        pos.extend(positions)
    return torch.cat(vectors), torch.tensor(ids, dtype=torch.int32), torch.tensor(pos, dtype=torch.int32)


def read_state(path, identity, groups, config):
    with safe_open(path, framework="pt") as f:
        meta = f.metadata()
        if meta["identity"] != identity:
            raise ValueError("Teacher checkpoint identity differs from the active inputs/runtime")
        last = int(meta["last_layer"])
    if not -1 <= last < config.num_hidden_layers:
        raise ValueError(f"Invalid checkpoint layer: {last}")
    state = load_file(path)
    expected = set()
    for arm, layout in groups.items():
        for n, rows, _ in layout:
            key = f"{arm}_streams_{n}"
            expected.add(key)
            assert state[key].shape == (len(rows), n, config.hc_mult, config.hidden_size)
            assert state[key].dtype == torch.bfloat16
            if last >= 0:
                key = f"{arm}_topk_{n}"
                expected.add(key)
                assert state[key].shape == (len(rows), n, min(n, config.index_topk))
                assert state[key].dtype == torch.int32
    if set(state) != expected:
        raise ValueError("Checkpoint tensor roster mismatch")
    return last, state


def commit_state(path, identity, index, state):
    temp = path.with_suffix(".part")
    save_file(state, temp, metadata={"identity": identity, "last_layer": str(index), "utc": utc_now()})
    with temp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_logits(out, stem, v, ids, pos, names, identity):
    assert v.dtype == torch.float32 and v.shape == (len(ids), 120832)
    assert ids.shape == pos.shape and torch.isfinite(v).all()
    path = out / f"{stem}.safetensors"
    atomic_save_safetensors(path, {"logits": v, "prompt_index": ids, "position_1based": pos})
    atomic_write_json(path.with_suffix(".json"), {
        "identity": identity, "utc": utc_now(), "panel_digest": PANEL_SHA,
        "logits_shape": list(v.shape), "logits_dtype": str(v.dtype),
        "prompt_ids": names, "sha256": sha256_file(path),
    })


def teacher(shards, config, out, identity):
    if read_json(out / "control.json")["identity"] != identity:
        raise ValueError("Teacher control is missing or stale")
    if (out / "complete.json").exists():
        raise ValueError("Teacher artifacts are already sealed")
    rows, alternate, probes = frozen_rows()
    cases = {"panel": rows, "alternate": rows, "probe": probes}
    groups = {
        "panel": grouping(rows, range(len(rows)), {1024: 4, 4096: 1}),
        "alternate": grouping(rows, alternate, {1024: 1, 4096: 2}),
        "probe": grouping(probes, range(2), {35: 1, 38: 1}),
    }
    checkpoint = out / "teacher_state.safetensors"
    if checkpoint.exists():
        last, state = read_state(checkpoint, identity, groups, config)
        print(f"[{utc_now()}] resume after L{last}", flush=True)
        shards.evict_before(last)
    else:
        last, state = -1, {}
        batches = [(f"{arm}_streams_{n}", torch.tensor([cases[arm][i]["input_ids"] for i in ids]))
                   for arm, layout in groups.items() for n, ids, _ in layout]
        states = run_embed_tokens(shards, config, [tokens for _, tokens in batches], DEVICE)
        state.update((key, s) for (key, _), s in zip(batches, states, strict=True))
        del states
        commit_state(checkpoint, identity, last, state)
    rot = ref.HYV4VLRotaryEmbedding(config).to(DEVICE)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = None
        for i in range(last + 1, config.num_hidden_layers):
            start = time.monotonic()
            if future is not None:
                future.result()
            layer = build_layer(shards, config, i)
            if i + 1 < config.num_hidden_layers:
                keys = [k for k in shards.weight_map if k.startswith(f"model.layers.{i + 1}.")]
                future = executor.submit(shards.ensure_shards, keys)
            elapsed = {}
            for arm, layout in groups.items():
                t = time.monotonic()
                for length, _, bs in layout:
                    stream = f"{arm}_streams_{length}"
                    topk = f"{arm}_topk_{length}"
                    state[stream], state[topk] = run_decoder_layer_batches(
                        layer, config, rot, state[stream], state.get(topk), DEVICE, bs
                    )
                elapsed[arm] = round(time.monotonic() - t, 2)
            del layer
            free_gpu(DEVICE)
            commit_state(checkpoint, identity, i, state)
            shards.evict_before(i)
            print(f"[{utc_now()}] L{i:02d} committed after {time.monotonic() - start:.1f}s "
                  f"forward_s={elapsed}", flush=True)
    scored = [score_inputs(cases[arm], layout, {n: state[f"{arm}_streams_{n}"] for n, _, _ in layout})
              for arm, layout in groups.items()]
    assert [len(s[1]) for s in scored] == [2048, 1280, 2]
    assert int((scored[0][2] > 2048).sum()) == 768
    stems = ("bf16_ref_logits", "bf16_alt_logits", "thinking_logits")
    for arm, stem, logit, (_, ids, pos) in zip(groups, stems, project(shards, config, [s[0] for s in scored]), scored, strict=True):
        write_logits(out, stem, logit, ids, pos, [p["id"] for p in cases[arm]], identity)
    score(out, rows, probes)
    sealed = [f"{stem}.{ext}" for stem in stems for ext in ("json", "safetensors")]
    sealed += ["control.json", "null.json", "thinking.json"]
    atomic_write_json(out / "complete.json", {"identity": identity, "panel_digest": PANEL_SHA,
                      "utc": utc_now(), "outputs": {name: sha256_file(out / name) for name in sealed}})


def summarize_null(values, ids, pos, rows):
    rng = torch.Generator().manual_seed(20261120)
    summary = {}
    for name, mask in (("all", torch.ones_like(pos, dtype=torch.bool)),
                       ("unpruned", pos <= 2048), ("pruned", pos > 2048)):
        per_prompt = {}
        for i in ids[mask].unique().tolist():
            d = values[mask & (ids == i)]
            r = rows[i]
            per_prompt.setdefault((r["genre"], len(r["input_ids"])), []).append((d.sum().item(), d.numel()))
        total = torch.zeros(1000, dtype=torch.float64)
        count = torch.zeros_like(total)
        for stratum in per_prompt.values():
            data = torch.tensor(stratum, dtype=torch.float64)
            draw = data[torch.randint(len(data), (1000, len(data)), generator=rng)]
            total += draw[:, :, 0].sum(dim=1)
            count += draw[:, :, 1].sum(dim=1)
        low, high = (total / count).quantile(torch.tensor([.025, .975], dtype=torch.float64)).tolist()
        summary[name] = {"positions": int(mask.sum()), "prompts": int(ids[mask].unique().numel()),
                         "mean": values[mask].mean().item(), "ci95": [low, high]}
    summary["N_KL"] = max(max(s["mean"], s["ci95"][1]) for s in summary.values())
    summary["effective_KL_pass"] = max(0.085, summary["N_KL"] + 0.04)
    return summary


def score(out, rows, probes):
    main = load_file(out / "bf16_ref_logits.safetensors")
    alt = load_file(out / "bf16_alt_logits.safetensors")
    ids, pos = alt["prompt_index"], alt["position_1based"]
    lookup = {(i, p): j for j, (i, p) in enumerate(zip(main["prompt_index"].tolist(), main["position_1based"].tolist()))}
    matches = [lookup[pair] for pair in zip(ids.tolist(), pos.tolist())]
    assert len(set(matches)) == len(matches) == 1280 and len(lookup) == 2048
    kl = []
    for j in range(0, len(matches), 16):
        a = main["logits"][matches[j:j + 16]].double().log_softmax(dim=-1)
        b = alt["logits"][j:j + 16].double().log_softmax(dim=-1)
        kl.append((a.exp() * (a - b)).sum(dim=-1).clamp_min(0))
    atomic_write_json(out / "null.json", summarize_null(torch.cat(kl), ids, pos, rows))
    del main, alt
    p = load_file(out / "thinking_logits.safetensors")["logits"].double().softmax(dim=-1)
    tok = Tokenizer.from_file(str(ROOT / "inputs/tokenizer.json"))
    report = {}
    for spec, probs in zip(probes, p, strict=True):
        _, indices = probs.topk(5)
        report[spec["id"]] = {
            "top5": [{"id": i, "text": tok.decode([i], skip_special_tokens=False), "p": probs[i].item()}
                     for i in indices.tolist()],
            "targets": {i: {"p": probs[i].item(), "rank": int((probs > probs[i]).sum()) + 1}
                        for i in (120029, 120030, 19, 220)},
        }
    hi = report["high"]["top5"][0]["id"]
    off = report["no_think"]["top5"][0]["id"]
    report["direct_first_token"] = off == 19
    report["paired_top1_support"] = off == 19 and hi not in (19, 120029, 120030, 220) and (
        report["no_think"]["targets"][19]["p"] > report["high"]["targets"][19]["p"]
    )
    atomic_write_json(out / "thinking.json", report)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("control", "teacher"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    identity = os.environ["TEACHER_IDENTITY"]
    keep = {0, 1, 2} if args.step == "control" else None
    shards = VerifiedShards(args.source, ROOT / "inputs", args.out / "cache", keep_layers=keep)
    config = text_config(shards.config_path)
    assert (config.num_hidden_layers, config.hidden_size, config.hc_mult, config.vocab_size, config.index_topk) == (
        78, 6144, 4, 120832, 2048
    )
    print(f"[{utc_now()}] {args.step} identity={identity} torch={torch.__version__} "
          f"cuda={torch.version.cuda} gpu={torch.cuda.get_device_name()}", flush=True)
    if args.step == "control":
        control(shards, config, args.out, identity)
    else:
        teacher(shards, config, args.out, identity)


if __name__ == "__main__":
    main()
