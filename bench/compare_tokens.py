#!/usr/bin/env python3
"""Compare token sequences across two bench.py JSON runs (e.g. plain vs drafted, or stock vs relay).

Reports per-prompt and thinking-on/off aggregates: exact equality, first differing
position, differing tokens out of total, and drafted kept-per-round statistics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _extract_run(path: Path) -> tuple[dict, list[int], str | None, str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    runs = obj.get("runs") or []
    if not runs:
        raise RuntimeError(f"{path}: no runs found")
    r0 = runs[0]
    tf = r0.get("tensorfold") or {}
    tok_ids = r0.get("token_ids")
    if tok_ids is None:
        tok_ids = tf.get("token_ids")
    if not isinstance(tok_ids, list) or len(tok_ids) == 0:
        raise RuntimeError(f"{path}: missing or empty token_ids (ensure bench.py runs with return_token_ids=true)")
    sha = r0.get("token_sha256") or tf.get("token_sha256")
    effort = obj.get("reasoning_effort", "omit")
    thinking = "off" if effort == "no_think" else "on"
    return r0, [int(x) for x in tok_ids], sha, thinking


def compare_seqs(a: list[int], b: list[int]) -> dict:
    total = max(len(a), len(b))
    first_diff = None
    diffs = 0
    for i in range(total):
        va = a[i] if i < len(a) else None
        vb = b[i] if i < len(b) else None
        if va != vb:
            if first_diff is None:
                first_diff = i
            diffs += 1
    return {
        "equal": diffs == 0,
        "first_diff_pos": first_diff,
        "diff_tokens": diffs,
        "total_tokens": total,
        "plain_len": len(a),
        "draft_len": len(b),
    }


def _spec_summary(r0: dict) -> dict | None:
    spec = r0.get("speculative") or (r0.get("tensorfold") or {}).get("speculative")
    if not isinstance(spec, dict):
        return None
    rounds = spec.get("rounds") or 0
    accepted = spec.get("accepted") or 0
    offered = spec.get("offered") or 0
    return {
        "rounds": rounds,
        "offered": offered,
        "accepted": accepted,
        "kept_per_round": round(accepted / rounds, 3) if rounds > 0 else 0.0,
        "tokens_per_round": spec.get("tokens_per_round"),
        "acceptance_rate": spec.get("acceptance_rate"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pair",
        action="append",
        required=True,
        help="name:plain_json:cand_json (repeatable)",
    )
    ap.add_argument("--require-exact", action="store_true", help="Exit non-zero if any pair differs")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    per_prompt: dict[str, dict] = {}
    by_thinking: dict[str, dict] = {
        "on": {"equal": True, "first_diff_pos": None, "diff_tokens": 0, "total_tokens": 0, "prompts": []},
        "off": {"equal": True, "first_diff_pos": None, "diff_tokens": 0, "total_tokens": 0, "prompts": []},
    }
    all_equal = True

    for spec in args.pair:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            print(f"COMPARE_TOKENS FAIL: bad --pair {spec!r} (expected name:plain:cand)", file=sys.stderr)
            return 2
        name, p_str, c_str = parts
        p_path, c_path = Path(p_str), Path(c_str)
        if not p_path.is_file() or not c_path.is_file():
            print(f"COMPARE_TOKENS FAIL: missing file for {name}: plain={p_path} cand={c_path}", file=sys.stderr)
            return 1
        rp, seq_p, sha_p, th_p = _extract_run(p_path)
        rc, seq_c, sha_c, th_c = _extract_run(c_path)
        cmp_res = compare_seqs(seq_p, seq_c)
        if cmp_res["equal"] and sha_p and sha_c and sha_p != sha_c:
            print(f"COMPARE_TOKENS FAIL: {name}: token_ids matched but token_sha256 differed ({sha_p} vs {sha_c})", file=sys.stderr)
            return 1
        entry = {
            **cmp_res,
            "thinking": th_p,
            "plain_sha256": sha_p,
            "cand_sha256": sha_c,
            "plain_tg_tps": rp.get("tg_tps"),
            "cand_tg_tps": rc.get("tg_tps"),
            "cand_speculative": _spec_summary(rc),
        }
        per_prompt[name] = entry
        if not entry["equal"]:
            all_equal = False
        bucket = by_thinking[th_p]
        bucket["prompts"].append(name)
        bucket["diff_tokens"] += entry["diff_tokens"]
        bucket["total_tokens"] += entry["total_tokens"]
        if not entry["equal"]:
            bucket["equal"] = False
            if bucket["first_diff_pos"] is None or entry["first_diff_pos"] < bucket["first_diff_pos"]:
                bucket["first_diff_pos"] = entry["first_diff_pos"]

        print(
            f"TOKEN_DIFF [{name}] thinking={th_p} equal={entry['equal']} "
            f"first_diff={entry['first_diff_pos']} diff={entry['diff_tokens']}/{entry['total_tokens']} "
            f"plain_tg={entry['plain_tg_tps']} cand_tg={entry['cand_tg_tps']} "
            f"sha_match={sha_p == sha_c if (sha_p and sha_c) else None}"
        )

    out_obj = {
        "all_equal": all_equal,
        "thinking_on": by_thinking["on"],
        "thinking_off": by_thinking["off"],
        "prompts": per_prompt,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out_obj, indent=2) + "\n", encoding="utf-8")

    if args.require_exact and not all_equal:
        print("COMPARE_TOKENS FAIL: --require-exact violated", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
