#!/usr/bin/env python3
"""Score existing natural router dumps using unordered top-eight expert sets."""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def get(path, name):
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(name)


def ids(path, prefix):
    picks = get(path, prefix + "router_experts")
    if picks.ndim != 3 or picks.shape[-1] != 8:
        raise ValueError(f"Expected [prompt,position,8] expert IDs in {path}")
    if picks.dtype not in (torch.int32, torch.int64) or ((picks < 0) | (picks >= 256)).any():
        raise ValueError(f"Invalid expert IDs in {path}")
    if (picks.sort(dim=-1).values.diff(dim=-1) == 0).any():
        raise ValueError(f"Duplicate expert ID in {path}")
    return picks


def replacements(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError("Router row shapes differ")
    shared = (reference.unsqueeze(-1) == candidate.unsqueeze(-2)).any(-1).sum(-1)
    return 8 - shared


def summarize(counts, mask):
    selected = counts[mask]
    n = selected.numel()
    changed = int(selected.sum())
    return {
        "replacements": changed, "picks": 8 * n,
        "affected_tokens": int((selected > 0).sum()), "tokens": n,
        "flip_rate": changed / (8 * n) if n else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--unquant", type=Path, required=True)
    p.add_argument("--score-dump", type=Path)
    a = p.parse_args()
    torch.set_num_threads(2)
    prefix = f"layer.{a.layer:03d}."
    ref, cand, unquant = [ids(path, prefix) for path in (a.reference, a.candidate, a.unquant)]
    pairs = {
        "candidate_vs_reference": replacements(ref, cand),
        "unquant_vs_reference": replacements(ref, unquant),
        "candidate_vs_unquant": replacements(unquant, cand),
    }
    all_rows = torch.ones(ref.shape[:-1], dtype=torch.bool)
    bins = {"all": all_rows}
    if a.score_dump:
        scores = get(a.score_dump, prefix + "router_choice_scores")
        if scores.shape != (*ref.shape[:-1], 256) or scores.dtype != torch.float32 or not torch.isfinite(scores).all():
            raise ValueError("Expected finite FP32 choice scores aligned to reference rows")
        top = scores.topk(9, dim=-1).values
        margin = top[..., 7] - top[..., 8]
        selected = torch.gather(scores, -1, ref.long()).amin(-1)
        if not torch.equal(selected, top[..., 7]):
            raise ValueError("Choice scores and reference IDs disagree at the cutoff")
        bins.update({
            "margin_le_0.000125": margin <= .000125,
            "margin_gt_0.000125_le_0.001": (margin > .000125) & (margin <= .001),
            "margin_gt_0.001": margin > .001,
        })
    report = {label: {bucket: summarize(counts, mask) for bucket, mask in bins.items()}
              for label, counts in pairs.items()}
    print(json.dumps({"layer": a.layer, "reference": str(a.reference), "candidate": str(a.candidate),
                      "unquant": str(a.unquant), "margin": str(a.score_dump) if a.score_dump else "UNVERIFIED",
                      "pairs": report}, indent=2))


if __name__ == "__main__":
    main()
