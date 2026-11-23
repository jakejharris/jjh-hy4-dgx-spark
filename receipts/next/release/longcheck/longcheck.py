#!/usr/bin/env python3
"""Long-prompt paired check runner and comparer for Hy4 fast prefill (stdlib only).

Subcommands:
  dry-check  Build all payloads from manifest.json, print sizes and token counts, send nothing.
  run        Send prompts at T=0, thinking off, drafts off to /v1/chat/completions and save outputs + hashes.
  compare    Compare two saved tags (e.g. off vs on) and return 0 on PASS or 1 on BLOCK.
  selftest   Exercise dry-check and every PASS/BLOCK rule on synthetic outputs in a temp dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "manifest.json"
DEFAULT_RESULTS = HERE / "results"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_ids(ids: list[int]) -> str:
    h = hashlib.sha256()
    for tid in ids:
        h.update(int(tid).to_bytes(4, "little", signed=False))
    return h.hexdigest()


def detect_loop_or_broken(content: str, token_ids: list[int] | None) -> str | None:
    if not content or not content.strip():
        return "empty_output"
    if "\ufffd" in content:
        return "utf8_replacement_char"
    bad_ctrl = [ch for ch in content if ord(ch) < 0x20 and ch not in ("\n", "\r", "\t")]
    if bad_ctrl:
        return f"control_char_0x{ord(bad_ctrl[0]):02x}"
    if token_ids:
        n = len(token_ids)
        run = 1
        for i in range(1, n):
            if token_ids[i] == token_ids[i - 1]:
                run += 1
                if run >= 8:
                    return f"token_repeat_x{run}_id={token_ids[i]}"
            else:
                run = 1
        for period in range(2, 9):
            if n < period * 4:
                continue
            for start in range(0, n - period * 4 + 1):
                pat = token_ids[start : start + period]
                reps = 1
                pos = start + period
                while pos + period <= n and token_ids[pos : pos + period] == pat:
                    reps += 1
                    pos += period
                if reps >= 4 and reps * period >= 16:
                    return f"ngram_loop_p{period}_x{reps}"
    # Character-level fallback loop check if token_ids absent or degenerate.
    if re.search(r"(.{4,24}?)\1{4,}", content, flags=re.DOTALL):
        return "text_repeat_loop"
    return None


def score_facts(content: str, facts: list[dict]) -> list[dict]:
    out = []
    for f in facts:
        exp = f["expected"]
        out.append({
            "key": f["key"],
            "expected": exp,
            "right": exp in content,
        })
    return out


def build_payload(model: str, prompt_text: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "draft": False,
        "reasoning_effort": "no_think",
        "return_token_ids": True,
        "cache_prompt": False,
        "stream": False,
    }


def load_manifest(path: Path, group: str = "all") -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if group != "all":
        data = dict(data)
        data["prompts"] = [p for p in data["prompts"] if p["group"] == group]
    return data


def first_unit(rec: dict):
    tids = rec.get("token_ids")
    if isinstance(tids, list) and len(tids) > 0:
        return ("id", int(tids[0]))
    txt = rec.get("content") or ""
    toks = txt.split()
    return ("word", toks[0]) if toks else ("empty", "")


def compare_token_lists(a: list[int] | None, b: list[int] | None, a_txt: str, b_txt: str) -> dict:
    if isinstance(a, list) and isinstance(b, list) and (a or b):
        total = max(len(a), len(b))
        same = 0
        first_diff = None
        for i in range(total):
            va = a[i] if i < len(a) else None
            vb = b[i] if i < len(b) else None
            if va == vb:
                same += 1
            elif first_diff is None:
                first_diff = i
        n8 = min(8, total)
        first8_match = all(
            (a[i] if i < len(a) else None) == (b[i] if i < len(b) else None)
            for i in range(n8)
        ) and n8 > 0
        return {
            "mode": "token_ids",
            "first_token_match": (len(a) > 0 and len(b) > 0 and a[0] == b[0]),
            "first8_match": first8_match,
            "matched_tokens": same,
            "total_tokens": total,
            "first_diff_pos": first_diff,
        }
    aw, bw = a_txt.split(), b_txt.split()
    total = max(len(aw), len(bw))
    same = sum(1 for i in range(total) if (aw[i] if i < len(aw) else None) == (bw[i] if i < len(bw) else None))
    first_diff = next((i for i in range(total) if (aw[i] if i < len(aw) else None) != (bw[i] if i < len(bw) else None)), None)
    return {
        "mode": "words_fallback",
        "first_token_match": (len(aw) > 0 and len(bw) > 0 and aw[0] == bw[0]),
        "first8_match": aw[:8] == bw[:8] and len(aw) > 0,
        "matched_tokens": same,
        "total_tokens": total,
        "first_diff_pos": first_diff,
    }


def cmd_dry_check(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest, args.group)
    base_dir = args.manifest.parent
    print(f"DRY-CHECK manifest={args.manifest} prompts={len(manifest['prompts'])} tokenizer={manifest.get('tokenizer', '?')}")
    print(f"{'ID':20s} {'GRP':11s} {'KIND':6s} {'RAW':>6s} {'REN':>6s} {'2048_SPLIT':>12s} {'MAX':>3s} {'FACTS':>5s} {'PAYLOAD_B':>10s}")
    total_bytes = 0
    for p in manifest["prompts"]:
        ptxt = (base_dir / p["file"]).read_text(encoding="utf-8")
        payload = build_payload(args.model, ptxt, p["max_tokens"])
        raw_bytes = json.dumps(payload).encode("utf-8")
        total_bytes += len(raw_bytes)
        ren = int(p["rendered_tokens_no_think"])
        split = f"{ren // 2048}x2048+{ren % 2048}"
        print(
            f"{p['id']:20s} {p['group']:11s} {p['kind']:6s} "
            f"{p['raw_tokens']:6d} {ren:6d} {split:>12s} "
            f"{p['max_tokens']:3d} {len(p.get('facts', [])):5d} {len(raw_bytes):10d}"
        )
    print(f"DRY-CHECK OK: {len(manifest['prompts'])} payloads built, {total_bytes:,} total JSON bytes, 0 requests sent.")
    return 0


def send_chat(base_url: str, payload: dict, timeout_s: float) -> tuple[dict, float]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body_bytes, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        status = getattr(resp, "status", resp.getcode())
        raw = resp.read().decode("utf-8", errors="replace")
    dt = time.monotonic() - t0
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {raw[:300]}")
    return json.loads(raw), dt


def cmd_run(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest, args.group)
    base_dir = args.manifest.parent
    out_dir = args.results_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for p in manifest["prompts"]:
        ptxt = (base_dir / p["file"]).read_text(encoding="utf-8")
        payload = build_payload(args.model, ptxt, p["max_tokens"])
        resp, wall_s = send_chat(args.url, payload, args.timeout)
        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        usage = resp.get("usage") or {}
        tf = resp.get("tensorfold") or {}
        token_ids = tf.get("token_ids")
        if isinstance(token_ids, list):
            token_ids = [int(x) for x in token_ids]
        token_sha = tf.get("token_sha256") or (sha256_ids(token_ids) if token_ids else None)
        text_sha = sha256_text(content)
        facts_res = score_facts(content, p.get("facts", []))
        broken = detect_loop_or_broken(content, token_ids)
        rec = {
            "id": p["id"],
            "group": p["group"],
            "kind": p["kind"],
            "tag": args.tag,
            "utc": utc_now(),
            "model": args.model,
            "max_tokens": p["max_tokens"],
            "expected_rendered_tokens": p["rendered_tokens_no_think"],
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": choice.get("finish_reason"),
            "wall_s": round(wall_s, 3),
            "content": content,
            "text_sha256": text_sha,
            "token_ids": token_ids,
            "token_sha256": token_sha,
            "facts": facts_res,
            "loop_or_broken": broken,
        }
        (out_dir / f"{p['id']}.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        records.append(rec)
        f_ok = sum(1 for x in facts_res if x["right"])
        f_str = f"{f_ok}/{len(facts_res)}" if facts_res else "-"
        print(
            f"[{args.tag}] {p['id']:20s} grp={p['group']:11s} p_tok={rec['prompt_tokens']} "
            f"out={rec['completion_tokens']} sha={(token_sha or text_sha)[:12]} "
            f"facts={f_str} broken={broken} wall={wall_s:.2f}s",
            flush=True,
        )

    summary = {
        "tag": args.tag,
        "utc": utc_now(),
        "url": args.url,
        "model": args.model,
        "manifest": str(args.manifest),
        "count": len(records),
        "prompts": [
            {
                "id": r["id"],
                "group": r["group"],
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "text_sha256": r["text_sha256"],
                "token_sha256": r["token_sha256"],
                "facts_right": sum(1 for x in r["facts"] if x["right"]),
                "facts_total": len(r["facts"]),
                "loop_or_broken": r["loop_or_broken"],
            }
            for r in records
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"RUN DONE tag={args.tag} saved={out_dir}")
    return 0


def compare_dirs(manifest: dict, base_dir: Path, cand_dir: Path) -> dict:
    blocks: list[str] = []
    rows: list[dict] = []
    main_first_match = 0
    main_total = 0
    corr_first8_diff = 0
    corr_total = 0
    fact_regressions: list[str] = []

    for p in manifest["prompts"]:
        pid = p["id"]
        bp = base_dir / f"{pid}.json"
        cp = cand_dir / f"{pid}.json"
        if not bp.is_file() or not cp.is_file():
            blocks.append(f"missing_result:{pid}")
            continue
        b = json.loads(bp.read_text(encoding="utf-8"))
        c = json.loads(cp.read_text(encoding="utf-8"))
        cmp_res = compare_token_lists(b.get("token_ids"), c.get("token_ids"), b.get("content", ""), c.get("content", ""))
        b_facts = score_facts(b.get("content", ""), p.get("facts", []))
        c_facts = score_facts(c.get("content", ""), p.get("facts", []))
        b_broken = detect_loop_or_broken(b.get("content", ""), b.get("token_ids"))
        c_broken = detect_loop_or_broken(c.get("content", ""), c.get("token_ids"))
        if c_broken:
            blocks.append(f"cand_broken:{pid}:{c_broken}")
        for bf, cf in zip(b_facts, c_facts):
            if bf["right"] and not cf["right"]:
                reg = f"fact_regression:{pid}:{bf['key']}:expected={bf['expected']}"
                fact_regressions.append(reg)
                blocks.append(reg)
        if p["group"] == "main":
            main_total += 1
            if cmp_res["first_token_match"]:
                main_first_match += 1
        elif p["group"] == "correctness":
            corr_total += 1
            if not cmp_res["first8_match"]:
                corr_first8_diff += 1
        rows.append({
            "id": pid,
            "group": p["group"],
            "first_token_match": cmp_res["first_token_match"],
            "first8_match": cmp_res["first8_match"],
            "matched_tokens": cmp_res["matched_tokens"],
            "total_tokens": cmp_res["total_tokens"],
            "first_diff_pos": cmp_res["first_diff_pos"],
            "base_facts_right": sum(1 for x in b_facts if x["right"]),
            "cand_facts_right": sum(1 for x in c_facts if x["right"]),
            "facts_total": len(b_facts),
            "base_broken": b_broken,
            "cand_broken": c_broken,
            "base_sha": (b.get("token_sha256") or b.get("text_sha256") or "")[:12],
            "cand_sha": (c.get("token_sha256") or c.get("text_sha256") or "")[:12],
        })

    if main_total > 0 and main_first_match < (main_total - 1 if main_total == 10 else main_total):
        blocks.append(f"main_first_token_agreement:{main_first_match}/{main_total}<9/10")
    if corr_total >= 3 and corr_first8_diff >= 2:
        blocks.append(f"correctness_first8_diffs:{corr_first8_diff}/{corr_total}>=2/3")

    return {
        "verdict": "BLOCK" if blocks else "PASS",
        "blocks": blocks,
        "main_first_token_agreement": f"{main_first_match}/{main_total}",
        "correctness_first8_diffs": f"{corr_first8_diff}/{corr_total}",
        "fact_regressions": fact_regressions,
        "rows": rows,
    }


def cmd_compare(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest, args.group)
    base_dir = args.results_dir / args.base
    cand_dir = args.results_dir / args.cand
    report = compare_dirs(manifest, base_dir, cand_dir)
    print(f"COMPARE base={args.base} cand={args.cand} verdict={report['verdict']}")
    print(f"{'ID':20s} {'GRP':11s} {'1ST':>4s} {'F8':>4s} {'TOK_EQ':>9s} {'DIFF@':>5s} {'FACTS(B->C)':>11s} {'BROKEN(C)':>16s}")
    for r in report["rows"]:
        f_col = f"{r['base_facts_right']}/{r['facts_total']}->{r['cand_facts_right']}/{r['facts_total']}" if r["facts_total"] else "-"
        print(
            f"{r['id']:20s} {r['group']:11s} "
            f"{'OK' if r['first_token_match'] else 'DIFF':>4s} "
            f"{'OK' if r['first8_match'] else 'DIFF':>4s} "
            f"{r['matched_tokens']:>3d}/{r['total_tokens']:<5d} "
            f"{str(r['first_diff_pos']) if r['first_diff_pos'] is not None else '-':>5s} "
            f"{f_col:>11s} {str(r['cand_broken'] or '-'):>16s}"
        )
    print(f"main first-token agreement: {report['main_first_token_agreement']} (blocks < 9/10)")
    print(f"correctness first-8 diffs:  {report['correctness_first8_diffs']} (blocks >= 2/3)")
    if report["blocks"]:
        print("BLOCKS:")
        for blk in report["blocks"]:
            print(f"  - {blk}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["verdict"] == "PASS" else 1


def cmd_selftest(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest, "all")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        base_d = root / "off"
        cand_d = root / "on"
        base_d.mkdir()
        cand_d.mkdir()
        for idx, p in enumerate(manifest["prompts"]):
            facts = p.get("facts", [])
            txt = " | ".join(f"{f['key']}={f['expected']}" for f in facts) if facts else f"Clean deterministic answer for {p['id']}."
            ids = [100 + idx * 10 + j for j in range(p["max_tokens"])]
            rec = {"id": p["id"], "group": p["group"], "content": txt, "token_ids": ids, "token_sha256": sha256_ids(ids), "text_sha256": sha256_text(txt)}
            (base_d / f"{p['id']}.json").write_text(json.dumps(rec), encoding="utf-8")
            (cand_d / f"{p['id']}.json").write_text(json.dumps(rec), encoding="utf-8")
        r_ok = compare_dirs(manifest, base_d, cand_d)
        assert r_ok["verdict"] == "PASS", f"expected PASS, got {r_ok}"

        # Planted failure 1: loop on one prompt -> BLOCK
        loop_p = manifest["prompts"][0]["id"]
        rec = json.loads((cand_d / f"{loop_p}.json").read_text(encoding="utf-8"))
        rec["token_ids"] = [42] * 32
        (cand_d / f"{loop_p}.json").write_text(json.dumps(rec), encoding="utf-8")
        r_loop = compare_dirs(manifest, base_d, cand_d)
        assert r_loop["verdict"] == "BLOCK" and any("cand_broken" in b for b in r_loop["blocks"]), r_loop
        (cand_d / f"{loop_p}.json").write_text((base_d / f"{loop_p}.json").read_text(encoding="utf-8"), encoding="utf-8")

        # Planted failure 2: planted fact right -> wrong -> BLOCK
        fact_p = next(p for p in manifest["prompts"] if p.get("facts"))
        rec = json.loads((cand_d / f"{fact_p['id']}.json").read_text(encoding="utf-8"))
        rec["content"] = "WRONG-VALUE"
        (cand_d / f"{fact_p['id']}.json").write_text(json.dumps(rec), encoding="utf-8")
        r_fact = compare_dirs(manifest, base_d, cand_d)
        assert r_fact["verdict"] == "BLOCK" and any("fact_regression" in b for b in r_fact["blocks"]), r_fact
        (cand_d / f"{fact_p['id']}.json").write_text((base_d / f"{fact_p['id']}.json").read_text(encoding="utf-8"), encoding="utf-8")

        # Planted failure 3: main first-token agreement 8/10 -> BLOCK
        mains = [p for p in manifest["prompts"] if p["group"] == "main"]
        for p in mains[:2]:
            rec = json.loads((cand_d / f"{p['id']}.json").read_text(encoding="utf-8"))
            rec["token_ids"][0] += 999
            (cand_d / f"{p['id']}.json").write_text(json.dumps(rec), encoding="utf-8")
        r_ft = compare_dirs(manifest, base_d, cand_d)
        assert r_ft["verdict"] == "BLOCK" and any("main_first_token_agreement:8/10" in b for b in r_ft["blocks"]), r_ft
        for p in mains[:2]:
            (cand_d / f"{p['id']}.json").write_text((base_d / f"{p['id']}.json").read_text(encoding="utf-8"), encoding="utf-8")

        # Planted failure 4: correctness first-8 diff on 2 of 3 -> BLOCK
        corrs = [p for p in manifest["prompts"] if p["group"] == "correctness"]
        for p in corrs[:2]:
            rec = json.loads((cand_d / f"{p['id']}.json").read_text(encoding="utf-8"))
            rec["token_ids"][5] += 777
            (cand_d / f"{p['id']}.json").write_text(json.dumps(rec), encoding="utf-8")
        r_c8 = compare_dirs(manifest, base_d, cand_d)
        assert r_c8["verdict"] == "BLOCK" and any("correctness_first8_diffs:2/3" in b for b in r_c8["blocks"]), r_c8

    print("SELFTEST OK: PASS baseline and all 4 planted BLOCK rules (loop, fact regression, <9/10 first-token, >=2/3 correctness first-8) caught.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--group", choices=("all", "main", "correctness"), default="all")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dry-check", help="Build payloads and print sizes/tokens without sending requests")
    d.add_argument("--model", default="frozen-78l-dfly")
    d.set_defaults(func=cmd_dry_check)

    r = sub.add_parser("run", help="Run prompts against a server and save outputs + hashes under results/<tag>")
    r.add_argument("--url", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--model", default="frozen-78l-dfly")
    r.add_argument("--timeout", type=float, default=1800.0)
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="Compare two saved result tags")
    c.add_argument("--base", required=True)
    c.add_argument("--cand", required=True)
    c.add_argument("--out", type=Path, default=None)
    c.set_defaults(func=cmd_compare)

    s = sub.add_parser("selftest", help="Run offline PASS/BLOCK checks in a temporary directory")
    s.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
