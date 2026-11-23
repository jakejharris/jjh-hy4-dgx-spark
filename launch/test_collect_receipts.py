#!/usr/bin/env python3
"""Offline test of launch/collect_receipts.py against excerpts of real rank logs from two 2026-11-21 boots.

    python3 -I launch/test_collect_receipts.py

Fixtures (launch/fixtures/): the lines the collector reads from each rank log of two boots (folder names keep the run's label), the full
quant at 131,072 tokens (winc-128k) and the lossless trunk at 32,768 tokens (winc-lossless-32k). Checks:
  1. The release engine's load line ends with ", repeat_stable=on|off"; the drafter, head and repeat option
     still parse whole, and a further field after it changes nothing.
  2. Every rank's weights come from the same field (weights= on the load line). Rank 0's serve banner figure
     (weights plus the KV pool) is kept as serve_resident_gib and never becomes weights.
  3. A full collection over the four fixture logs writes the same weights on every rank, and the fill reads
     the right maximum and drafter label from it.
  4. The per-claim files and their claim ids carry the build (systems.*.exl3.json for the full quant,
     systems.*.lossless.json for the lossless trunk), and a --build that names the other build is refused.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

LAUNCH = Path(__file__).resolve().parent
KIT = LAUNCH.parent
FIX = LAUNCH / "fixtures"
FAILS: list[str] = []


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(cond: bool, what: str) -> None:
    print(f"  {'ok  ' if cond else 'FAIL'} {what}")
    if not cond:
        FAILS.append(what)


def parse_line(cr, line: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="hy4-receipt-test-") as td:
        p = Path(td) / "rank0_node1.log"
        p.write_text(line + "\n", encoding="utf-8")
        return cr.parse_rank_log(p)


def main() -> int:
    cr = load("collect_receipts", LAUNCH / "collect_receipts.py")
    fc = load("fill_claims", KIT / "tools" / "fill_claims.py")

    print("== image decoders from rank 0's serve log")
    warn = ("[tensorfold] warning: libturbojpeg.so.0 is required for JPEG inputs (--vision); install libjpeg-turbo 3.1.4.1 "
            "and set TENSORFOLD_LIBTURBOJPEG to its file, or put it on the documented loader path")
    r0 = parse_line(cr, warn)
    dec = cr.image_decoders(r0, {"SERVE_VISION": "1"})
    check(dec["jpeg"].startswith("missing: [tensorfold] warning: libturbojpeg.so.0") and dec["webp"] == "found",
          f"a JPEG warning reads missing, no WebP warning reads found: {dec}")
    check(cr.image_decoders(parse_line(cr, "NCCL INFO NCCL version 2.30.7+cuda13.3"), {"SERVE_VISION": "1"}) == {"jpeg": "found", "webp": "found"},
          "no warnings with vision on: both found")
    check(cr.image_decoders(r0, {"SERVE_VISION": "0"})["jpeg"] == "not checked (vision off)", "vision off: not checked, whatever the log says")

    cases = (("winc-128k", "4b6b", 75.22, 80764690564, 87.29), ("winc-lossless-32k", "lossless", 83.72, 89891496068, 87.07))
    for name, trunk, weights, wbytes, banner in cases:
        print(f"== {name}: each rank log parsed alone")
        ranks = {r: cr.parse_rank_log(FIX / name / f"rank{r}_node{r + 1}.log") for r in range(4)}
        for r, info in ranks.items():
            check(info.get("drafter_loaded") == "DFly2 4-bit", f"rank {r} drafter is {info.get('drafter_loaded')!r}, want 'DFly2 4-bit'")
            check(info.get("head_loaded") == "fp32" and info.get("repeat_stable_loaded") == "on" and info.get("trunk_loaded") == trunk,
                  f"rank {r} head {info.get('head_loaded')!r}, repeat_stable {info.get('repeat_stable_loaded')!r}, trunk {info.get('trunk_loaded')!r}")
            check(info.get("weights_gib") == weights and info.get("weight_bytes") == wbytes,
                  f"rank {r} weights_gib {info.get('weights_gib')} from the load line (want {weights}, {wbytes} bytes)")
        check(ranks[0].get("serve_resident_gib") == banner, f"rank 0 serve banner kept as serve_resident_gib {ranks[0].get('serve_resident_gib')} (want {banner})")
        check(all("serve_resident_gib" not in ranks[r] for r in (1, 2, 3)), "ranks 1 to 3 print no serve banner")
        check(ranks[0].get("sampling") == "temperature 0.9, top_k -1, top_p 1" and all("sampling" not in ranks[r] for r in (1, 2, 3)),
              f"rank 0's serving line sampling read (no --temperature flag on that boot): {ranks[0].get('sampling')!r}")

    print("== the serving line's sampling part")
    line = (FIX / "winc-128k" / "rank0_node1.log").read_text(encoding="utf-8").splitlines()
    line = next(ln for ln in line if "] serving " in ln)
    s07 = parse_line(cr, line.replace("temperature 0.9,", "temperature 0.7,"))
    check(s07.get("sampling") == "temperature 0.7, top_k -1, top_p 1" and s07.get("drafts") == "DFly2" and s07.get("serve_context") == 131072,
          f"--temperature 0.7: {s07.get('sampling')!r}, drafts and context still read")
    sg = parse_line(cr, line.replace("temperature 0.9, top_k -1, top_p 1", "greedy"))
    check(sg.get("sampling") == "greedy", f"--temperature 0: {sg.get('sampling')!r}")
    si = parse_line(cr, line.replace("top_p 1;", "top_p 1; thinking off: temperature 0.7, top_p 0.8;"))
    check(si.get("sampling") == "temperature 0.9, top_k -1, top_p 1" and si.get("drafts") == "DFly2",
          f"a thinking-off set after it is not taken as the defaults: {si.get('sampling')!r}")

    print("== the load line with a further field, and without the repeat option (older engine)")
    real = (FIX / "winc-128k" / "rank1_node2.log").read_text(encoding="utf-8").splitlines()[-1]
    more = parse_line(cr, real + ", spec_depth=6, note=a b")
    check(more.get("drafter_loaded") == "DFly2 4-bit" and more.get("head_loaded") == "fp32"
          and more.get("repeat_stable_loaded") == "on" and more.get("weights_gib") == 75.22,
          f"extra fields after repeat_stable: drafter {more.get('drafter_loaded')!r}, repeat_stable {more.get('repeat_stable_loaded')!r}")
    older = parse_line(cr, real.replace(", repeat_stable=on", ""))
    check(older.get("drafter_loaded") == "DFly2 4-bit" and older.get("head_loaded") == "fp32" and "repeat_stable_loaded" not in older,
          f"no repeat_stable field: drafter {older.get('drafter_loaded')!r}, head {older.get('head_loaded')!r}")
    off = parse_line(cr, real.replace("repeat_stable=on", "repeat_stable=off").replace("drafter=DFly2 4-bit", "drafter=none"))
    check(off.get("drafter_loaded") == "none" and off.get("repeat_stable_loaded") == "off",
          f"drafts off: drafter {off.get('drafter_loaded')!r}, repeat_stable {off.get('repeat_stable_loaded')!r}")

    for (name, trunk, weights, _, banner), build in zip(cases, ("exl3", "lossless")):
        print(f"== {name}: full collection over the four fixture logs, then the fill's extractor")
        with tempfile.TemporaryDirectory(prefix="hy4-receipt-test-") as td:
            out = Path(td) / "receipts"
            cmd = [sys.executable, "-I", str(LAUNCH / "collect_receipts.py"), "--logdir", str(FIX / name),
                   "--out-dir", str(out), "--label", f"test-{name}", "--isolation", "quiet"]
            p = subprocess.run(cmd, capture_output=True, text=True)
            check(p.returncode == 0, f"collect_receipts.py rc {p.returncode} {p.stderr.strip()[-200:]}")
            if p.returncode != 0:
                continue
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            files = sorted(f.name for f in out.glob("systems.*.json"))
            ids = {json.loads((out / f).read_text(encoding="utf-8"))["claim_id"] for f in files}
            check(summary.get("build") == build and len(files) == 12 and all(f.endswith(f".{build}.json") for f in files)
                  and ids == {f[: -len(".json")] for f in files},
                  f"build {summary.get('build')!r}: {len(files)} files named systems.*.{build}.json, each claim_id its own file name")
            other = "lossless" if build == "exl3" else "exl3"
            q = subprocess.run(cmd[:-6] + ["--out-dir", str(Path(td) / "other"), "--build", other], capture_output=True, text=True)
            check(q.returncode != 0 and "but the boot is" in q.stderr and not (Path(td) / "other").exists(),
                  f"--build {other} on the {build} boot refused: rc {q.returncode} {q.stderr.strip()[-90:]!r}")
            per_rank = {r: json.loads((out / f"systems.weights.r{r}.{build}.json").read_text(encoding="utf-8")) for r in range(4)}
            check({d["value"] for d in per_rank.values()} == {weights} and {d["source"] for d in per_rank.values()} == {"rank_log"},
                  f"systems.weights.r0..r3 values {[d['value'] for d in per_rank.values()]}, one source")
            check(summary["ranks"]["0"].get("serve_resident_gib") == banner, "summary keeps rank 0's serve banner under its own name")
            value, spread = fc.x_ranks_max(summary, {"field": "weights_gib"}, {}, "summary.json")
            check(float(value) == weights and spread.endswith("drafter DFly2 4-bit"),
                  f"fill reads max weights {value} with spread ending {spread.split(';')[-1].strip()!r}")

    print("== engine receipt: the binary hash per rank, switches, clock cap and snapshots, engine commit")
    sha = "40188e7f905724278693a05dfa25f257ab8ba88a1db4ccf6eaaab894dc13570c"
    meta = {"RING_HOSTS": "node1 node2 node3 node4", "TENSORFOLD_BIN": "/opt/tensorfold/bin/tensorfold-native",
            "TF_HY4_FAST_PREFILL": "1", "TF_HY4_PREFILL_AR": "fp32", "SERVE_VISION": "1",
            "TF_HY4_VISION_DIR": "/data/models/hy4-vit-bf16", "GPU_CLOCK_CAP_MHZ": "2200", "ENGINE_COMMIT": "abc1234", "SERVE_TEMPERATURE": "0.7",
            "bin_sha256_by_rank": {str(r): sha for r in range(4)},
            "gpu_clock_mhz_at_launch_by_rank": {"0": "2184", "1": "2184", "2": "2177", "3": "2184"},
            "gpu_clock_mhz_at_ready_by_rank": {"0": "2184", "1": "2184", "2": "2184", "3": "2184"}}
    with tempfile.TemporaryDirectory(prefix="hy4-receipt-test-") as td:
        logdir = Path(td) / "logs"
        logdir.mkdir()
        for f in (FIX / "winc-128k").iterdir():
            (logdir / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
        (logdir / "launch_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        r0 = logdir / "rank0_node1.log"
        r0.write_text("spark-267e:1:1 [0] NCCL INFO NCCL version 2.30.7+cuda13.3\n" + r0.read_text(encoding="utf-8"), encoding="utf-8")
        (logdir / "vision_check.json").write_text(json.dumps({"tool": "bench/vision_check.py", "known_pass": 3, "known_total": 3,
                                                             "server_alive_after": True, "url": "http://127.0.0.1:8080"}), encoding="utf-8")
        out = Path(td) / "receipts"
        p = subprocess.run([sys.executable, "-I", str(LAUNCH / "collect_receipts.py"), "--logdir", str(logdir),
                            "--out-dir", str(out), "--label", "test-engine", "--isolation", "quiet"], capture_output=True, text=True)
        check(p.returncode == 0, f"collect_receipts.py rc {p.returncode} {p.stderr.strip()[-200:]}")
        if p.returncode == 0:
            s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            e, ls = s.get("engine", {}), s.get("launch_settings", {})
            check(e.get("bin_sha256_by_rank") == meta["bin_sha256_by_rank"] and e.get("bin_sha256_same_on_all_ranks") is True,
                  f"per-rank binary sha256 kept and agreeing: {e.get('bin_sha256_same_on_all_ranks')!r}")
            check(e.get("engine_commit") == "abc1234" and e.get("gpu_clock_cap_mhz") == "2200"
                  and e.get("gpu_clock_mhz_at_launch_by_rank", {}).get("2") == "2177",
                  f"commit {e.get('engine_commit')!r}, cap {e.get('gpu_clock_cap_mhz')!r}, launch snapshot kept")
            check(all(ls.get(k) == meta[k] for k in ("TF_HY4_FAST_PREFILL", "TF_HY4_PREFILL_AR", "SERVE_VISION", "TF_HY4_VISION_DIR", "SERVE_TEMPERATURE")),
                  "switches in force kept in launch_settings, SERVE_TEMPERATURE among them")
            check(e.get("problems") == [] and "WARNING" not in p.stderr, f"no problems on a full read: {e.get('problems')!r}")
            check(e.get("nccl_version_by_rank", {}).get("0") == "2.30.7+cuda13.3" and e["nccl_version_by_rank"].get("1") == "not printed",
                  f"NCCL version read from rank 0's log: {e.get('nccl_version_by_rank')!r}")
            vc = s.get("vision_check") or {}
            check(vc.get("known_pass") == 3 and vc.get("server_alive_after") is True and (out / "vision_check.json").is_file(),
                  f"vision_check.json picked up and copied: {vc!r}")
        meta2 = {k: v for k, v in meta.items() if k not in ("bin_sha256_by_rank", "GPU_CLOCK_CAP_MHZ", "ENGINE_COMMIT")}
        meta2["bin_sha256_by_rank"] = {"0": sha, "1": sha, "2": "unknown", "3": sha}
        (logdir / "launch_meta.json").write_text(json.dumps(meta2), encoding="utf-8")
        p = subprocess.run([sys.executable, "-I", str(LAUNCH / "collect_receipts.py"), "--logdir", str(logdir),
                            "--out-dir", str(Path(td) / "r2"), "--label", "test-engine2", "--isolation", "quiet"], capture_output=True, text=True)
        if p.returncode == 0:
            e = json.loads((Path(td) / "r2" / "summary.json").read_text(encoding="utf-8")).get("engine", {})
            check(e.get("bin_sha256_same_on_all_ranks") is False and e.get("engine_commit") == "not declared"
                  and e.get("gpu_clock_cap_mhz") == "none declared (stock clocks)",
                  f"an unread rank, no commit and no cap are said so: {e.get('bin_sha256_same_on_all_ranks')!r}, {e.get('engine_commit')!r}")
            check("rank 2: binary hash not read (unknown)" in e.get("problems", []) and "WARNING: engine receipt: rank 2" in p.stderr,
                  f"the unread rank is named in problems and on stderr: {e.get('problems')!r}")
        else:
            check(False, f"second collection rc {p.returncode}")

    print("== " + ("PASS" if not FAILS else f"FAIL ({len(FAILS)}): " + "; ".join(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
