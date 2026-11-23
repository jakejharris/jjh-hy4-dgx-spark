#!/usr/bin/env python3
"""Top-1 agreement with the BF16 reference, measured on a running server.

collect: send raw token-ID prefixes to a live TensorFold server (/v1/completions,
max_tokens 1, temperature 0) and save its greedy next token at each (prefix, cut).
score: compare those picks with the reference's picks at the same positions
(overall, prose, code) and, given the lossless build's picks, count the paired
outcomes on exactly the same positions.

Reference picks file, either layout:
  - picks (I64 [N, L]) and optional margin (F32 [N, L]): the reference's greedy
    next token after each prefix length 1..L
  - bonus_tok (I64 [N]): one pick per prefix at cut 384

Usage (see eval/README.md):
  python3 eval/agreement.py selftest
  python3 eval/agreement.py collect --url http://127.0.0.1:8080 --model MODEL \
    --prefixes eval/prefixes/prefixes.safetensors --label lossless --out OUT/lossless_picks.json
  python3 eval/agreement.py collect --url http://127.0.0.1:8080 --model MODEL \
    --prefixes eval/prefixes/prefixes.safetensors --match-grid OUT/lossless_picks.json \
    --label 4b --out OUT/4b_picks.json
  python3 eval/agreement.py score --teacher eval/prefixes/reference_picks.safetensors \
    --cand OUT/4b_picks.json --lossless OUT/lossless_picks.json --json-out OUT/4b_score.json
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_safetensors(path: Path) -> dict[str, tuple[str, list[int], bytes]]:
    """Minimal zero-dependency safetensors reader returning {name: (dtype, shape, raw_bytes)}."""
    with open(path, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hdr_len).decode("utf-8"))
        data_start = 8 + hdr_len
        out: dict[str, tuple[str, list[int], bytes]] = {}
        for k, meta in hdr.items():
            if k == "__metadata__":
                continue
            off0, off1 = meta["data_offsets"]
            f.seek(data_start + off0)
            out[k] = (meta["dtype"], list(meta["shape"]), f.read(off1 - off0))
    return out


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    hdr: dict[str, dict] = {}
    chunks: list[bytes] = []
    off = 0
    for k in sorted(tensors):
        dtype, shape, raw = tensors[k]
        hdr[k] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(raw)]}
        chunks.append(raw)
        off += len(raw)
    hdr_bytes = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        for c in chunks:
            f.write(c)


def unpack_2d_i64(dtype: str, shape: list[int], raw: bytes) -> list[list[int]]:
    if len(shape) != 2:
        raise ValueError(f"Expected 2D tensor, got shape {shape}")
    rows, cols = shape
    fmt_ch = {"I64": "q", "I32": "i", "U32": "I"}.get(dtype)
    if fmt_ch is None:
        raise ValueError(f"Unsupported integer dtype {dtype}")
    vals = struct.unpack(f"<{rows * cols}{fmt_ch}", raw)
    return [list(vals[r * cols : (r + 1) * cols]) for r in range(rows)]


def unpack_1d_i64(dtype: str, shape: list[int], raw: bytes) -> list[int]:
    if len(shape) != 1:
        raise ValueError(f"Expected 1D tensor, got shape {shape}")
    n = shape[0]
    fmt_ch = {"I64": "q", "I32": "i", "U32": "I", "U8": "B"}.get(dtype)
    if fmt_ch is None:
        raise ValueError(f"Unsupported dtype {dtype}")
    return list(struct.unpack(f"<{n}{fmt_ch}", raw))


def unpack_2d_f32(dtype: str, shape: list[int], raw: bytes) -> list[list[float]]:
    if len(shape) != 2 or dtype != "F32":
        raise ValueError(f"Expected 2D F32 tensor, got {dtype} {shape}")
    rows, cols = shape
    vals = struct.unpack(f"<{rows * cols}f", raw)
    return [list(vals[r * cols : (r + 1) * cols]) for r in range(rows)]


def load_prefixes(path: Path) -> tuple[list[list[int]], list[int]]:
    st = read_safetensors(path)
    if "input_ids" not in st:
        raise ValueError(f"input_ids missing from {path}")
    ids = unpack_2d_i64(*st["input_ids"])
    if "is_code" in st:
        is_code = unpack_1d_i64(*st["is_code"])
    else:
        is_code = [i % 2 for i in range(len(ids))]
    if len(is_code) != len(ids):
        raise ValueError(f"is_code len {len(is_code)} != input_ids rows {len(ids)}")
    return ids, is_code


def query_one_pick(url: str, model: str, prompt_ids: list[int], timeout_s: float) -> tuple[int, float]:
    endpoint = url.rstrip("/") + "/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "ignore_eos": True,
        "return_token_ids": True,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {endpoint}: {err_body[:400]}") from e
    dt_ms = (time.monotonic() - t0) * 1000.0
    tf = body.get("tensorfold") or {}
    tids = tf.get("token_ids")
    if not isinstance(tids, list) or len(tids) < 1:
        raise RuntimeError(f"Missing tensorfold.token_ids in response: {json.dumps(body)[:400]}")
    return int(tids[0]), dt_ms


def cmd_collect(args: argparse.Namespace) -> int:
    ids, is_code = load_prefixes(args.prefixes)
    n_prefixes = len(ids)
    seq_len = len(ids[0]) if ids else 0

    if args.match_grid is not None:
        ref = json.loads(args.match_grid.read_text(encoding="utf-8"))
        grid = [(int(p), int(c)) for p, c in ref["grid"]]
        cuts_order = [int(c) for c in ref.get("cuts", sorted({c for _, c in grid}))]
    else:
        raw_cuts = [int(x.strip()) for x in args.cuts.split(",") if x.strip()]
        cuts_order = []
        for c in raw_cuts:
            if c < 1 or c > seq_len:
                raise ValueError(f"Cut {c} out of range 1..{seq_len}")
            if c not in cuts_order:
                cuts_order.append(c)
        if args.max_prefixes > 0:
            n_use = min(n_prefixes, args.max_prefixes)
        else:
            n_use = n_prefixes
        grid = [(p, c) for c in cuts_order for p in range(n_use)]

    # Group by cut so --max-seconds always stops cleanly on a complete cut boundary
    by_cut: dict[int, list[int]] = {}
    ordered_cuts: list[int] = []
    for p, c in grid:
        if c not in by_cut:
            by_cut[c] = []
            ordered_cuts.append(c)
        by_cut[c].append(p)

    t_start = time.monotonic()
    completed_grid: list[list[int]] = []
    picks: list[int] = []
    latencies_ms: list[float] = []
    cuts_completed: list[int] = []

    for cut_idx, c in enumerate(ordered_cuts):
        if args.match_grid is None and args.max_seconds > 0 and cut_idx > 0:
            elapsed = time.monotonic() - t_start
            est_cut_s = (elapsed / cut_idx)
            if elapsed + est_cut_s > args.max_seconds:
                print(
                    f"[{utc_now()}] time budget {args.max_seconds:.0f}s reached after "
                    f"{len(cuts_completed)} cuts ({len(picks)} positions); stopping cleanly",
                    flush=True,
                )
                break
        t_cut0 = time.monotonic()
        for p in by_cut[c]:
            tok, dt_ms = query_one_pick(args.url, args.model, ids[p][:c], args.timeout)
            completed_grid.append([p, c])
            picks.append(tok)
            latencies_ms.append(round(dt_ms, 2))
        cuts_completed.append(c)
        dt_cut = time.monotonic() - t_cut0
        print(
            f"[{utc_now()}] label={args.label} cut={c:3d} prefixes={len(by_cut[c])} "
            f"cut_s={dt_cut:.2f} total_pos={len(picks)} total_s={time.monotonic() - t_start:.2f}",
            flush=True,
        )

    total_s = time.monotonic() - t_start
    mean_ms = (sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0
    out_obj = {
        "schema": 1,
        "utc": utc_now(),
        "label": args.label,
        "url": args.url,
        "model": args.model,
        "prefixes_path": str(args.prefixes),
        "n_prefixes": n_prefixes,
        "seq_len": seq_len,
        "is_code": is_code,
        "cuts": cuts_completed,
        "grid": completed_grid,
        "picks": picks,
        "positions": len(picks),
        "elapsed_s": round(total_s, 3),
        "mean_ms_per_req": round(mean_ms, 2),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out_obj, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, args.out)
    print(
        f"[{utc_now()}] COLLECT_DONE label={args.label} positions={len(picks)} "
        f"cuts={cuts_completed} elapsed_s={total_s:.2f} mean_ms={mean_ms:.2f} out={args.out}",
        flush=True,
    )
    return 0


def load_teacher(path: Path) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], float], str]:
    st = read_safetensors(path)
    picks_map: dict[tuple[int, int], int] = {}
    margin_map: dict[tuple[int, int], float] = {}
    if "picks" in st:
        arr = unpack_2d_i64(*st["picks"])
        for p, row in enumerate(arr):
            for pos, tok in enumerate(row):
                # pos is 0-based index into [0..seq_len-1], corresponding to cut = pos + 1
                picks_map[(p, pos + 1)] = tok
        if "margin" in st:
            marr = unpack_2d_f32(*st["margin"])
            for p, row in enumerate(marr):
                for pos, val in enumerate(row):
                    margin_map[(p, pos + 1)] = val
        return picks_map, margin_map, f"teacher_all({len(arr)}x{len(arr[0])})"
    if "bonus_tok" in st:
        arr1 = unpack_1d_i64(*st["bonus_tok"])
        for p, tok in enumerate(arr1):
            picks_map[(p, 384)] = tok
        return picks_map, margin_map, f"pass1_bonus_tok({len(arr1)}x1@384)"
    raise ValueError(f"Unrecognized teacher safetensors keys in {path}: {sorted(st.keys())}")


def cmd_score(args: argparse.Namespace) -> int:
    teacher_picks, teacher_margins, teacher_kind = load_teacher(args.teacher)
    cand = json.loads(args.cand.read_text(encoding="utf-8"))
    is_code = [int(x) for x in cand["is_code"]]
    cand_map = {(int(p), int(c)): int(t) for (p, c), t in zip(cand["grid"], cand["picks"])}

    lossless_map: dict[tuple[int, int], int] | None = None
    lossless_label = ""
    if args.lossless is not None:
        ll = json.loads(args.lossless.read_text(encoding="utf-8"))
        lossless_label = ll.get("label", "lossless")
        lossless_map = {(int(p), int(c)): int(t) for (p, c), t in zip(ll["grid"], ll["picks"])}

    keys = sorted(k for k in cand_map if k in teacher_picks and (lossless_map is None or k in lossless_map))
    if not keys:
        raise RuntimeError(
            f"Zero overlapping positions between cand ({len(cand_map)}), "
            f"teacher ({len(teacher_picks)}, {teacher_kind}), and lossless"
        )

    tot = len(keys)
    cand_ok = 0
    prose_tot = prose_ok = 0
    code_tot = code_ok = 0
    low_m_tot = low_m_ok = 0
    high_m_tot = high_m_ok = 0
    ll_ok = 0
    both_match = ll_only = cand_only = neither = 0

    for k in keys:
        p, _c = k
        t_tok = teacher_picks[k]
        c_hit = (cand_map[k] == t_tok)
        cand_ok += int(c_hit)
        if is_code[p]:
            code_tot += 1
            code_ok += int(c_hit)
        else:
            prose_tot += 1
            prose_ok += int(c_hit)
        if k in teacher_margins:
            if teacher_margins[k] < 1.0:
                low_m_tot += 1
                low_m_ok += int(c_hit)
            else:
                high_m_tot += 1
                high_m_ok += int(c_hit)
        if lossless_map is not None:
            l_hit = (lossless_map[k] == t_tok)
            ll_ok += int(l_hit)
            if l_hit and c_hit:
                both_match += 1
            elif l_hit and not c_hit:
                ll_only += 1
            elif (not l_hit) and c_hit:
                cand_only += 1
            else:
                neither += 1

    def pct(a: int, b: int) -> float:
        return round((100.0 * a / b), 3) if b > 0 else 0.0

    result: dict = {
        "utc": utc_now(),
        "label": cand.get("label", str(args.cand)),
        "teacher_kind": teacher_kind,
        "positions_scored": tot,
        "overall_matched": cand_ok,
        "overall_pct": pct(cand_ok, tot),
        "prose_matched": prose_ok,
        "prose_total": prose_tot,
        "prose_pct": pct(prose_ok, prose_tot),
        "code_matched": code_ok,
        "code_total": code_tot,
        "code_pct": pct(code_ok, code_tot),
    }
    if low_m_tot or high_m_tot:
        result["margin_lt1_matched"] = low_m_ok
        result["margin_lt1_total"] = low_m_tot
        result["margin_lt1_pct"] = pct(low_m_ok, low_m_tot)
        result["margin_ge1_matched"] = high_m_ok
        result["margin_ge1_total"] = high_m_tot
        result["margin_ge1_pct"] = pct(high_m_ok, high_m_tot)

    if lossless_map is not None:
        ll_pct = pct(ll_ok, tot)
        delta_pts = round(result["overall_pct"] - ll_pct, 3)
        result["paired_vs"] = lossless_label
        result["lossless_matched"] = ll_ok
        result["lossless_pct"] = ll_pct
        result["both_match"] = both_match
        result["lossless_only"] = ll_only
        result["cand_only"] = cand_only
        result["neither"] = neither
        result["delta_pts_vs_lossless"] = delta_pts
        result["within_1pt"] = delta_pts >= -1.0

    line = (
        f"TEACHER_AGREE label={result['label']} teacher={teacher_kind} "
        f"pos={tot} overall={cand_ok}/{tot} ({result['overall_pct']:.2f}%) "
        f"prose={prose_ok}/{prose_tot} ({result['prose_pct']:.2f}%) "
        f"code={code_ok}/{code_tot} ({result['code_pct']:.2f}%)"
    )
    if lossless_map is not None:
        line += (
            f" | paired_vs_{lossless_label}: both={both_match} "
            f"lossless_only={ll_only} cand_only={cand_only} neither={neither} "
            f"delta={result['delta_pts_vs_lossless']:+.2f}pt within_1pt={result['within_1pt']}"
        )
    print(line, flush=True)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.json_out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, args.json_out)
    return 0


def cmd_selftest(_args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="hy4-teacher-selftest-") as td:
        d = Path(td)
        n_pref, seq_len = 64, 384
        is_code = [i % 2 for i in range(n_pref)]
        ids_flat = [((i * 384 + j) % 1000) + 1 for i in range(n_pref) for j in range(seq_len)]
        picks_flat = [((i * 13 + j * 7) % 500) + 10 for i in range(n_pref) for j in range(seq_len)]
        margins_flat = [0.5 if (j % 2 == 0) else 2.5 for _ in range(n_pref) for j in range(seq_len)]

        pref_p = d / "prefixes.safetensors"
        write_safetensors(
            pref_p,
            {
                "input_ids": ("I64", [n_pref, seq_len], struct.pack(f"<{len(ids_flat)}q", *ids_flat)),
                "is_code": ("U8", [n_pref], struct.pack(f"<{n_pref}B", *is_code)),
            },
        )
        t_all = d / "teacher_all.safetensors"
        write_safetensors(
            t_all,
            {
                "picks": ("I64", [n_pref, seq_len], struct.pack(f"<{len(picks_flat)}q", *picks_flat)),
                "margin": ("F32", [n_pref, seq_len], struct.pack(f"<{len(margins_flat)}f", *margins_flat)),
            },
        )
        cuts = [384, 48, 96, 144, 192, 240, 288, 336]
        grid = [[p, c] for c in cuts for p in range(n_pref)]  # 512 positions

        # Construct known paired counts on 512 positions:
        # idx 0..489 (490): both match teacher
        # idx 490..499 (10): lossless matches, cand misses (lossless_only=10)
        # idx 500..505 (6):  cand matches, lossless misses (cand_only=6)
        # idx 506..511 (6):  neither matches (neither=6)
        ll_picks = []
        cand_picks = []
        for idx, (p, c) in enumerate(grid):
            t_tok = picks_flat[p * seq_len + (c - 1)]
            if idx < 490:
                ll_picks.append(t_tok)
                cand_picks.append(t_tok)
            elif idx < 500:
                ll_picks.append(t_tok)
                cand_picks.append(t_tok + 1)
            elif idx < 506:
                ll_picks.append(t_tok + 2)
                cand_picks.append(t_tok)
            else:
                ll_picks.append(t_tok + 2)
                cand_picks.append(t_tok + 3)

        ll_p = d / "lossless.json"
        ca_p = d / "cand.json"
        sc_p = d / "score.json"
        base = {"n_prefixes": n_pref, "seq_len": seq_len, "is_code": is_code, "cuts": cuts, "grid": grid}
        ll_p.write_text(json.dumps({**base, "label": "lossless", "picks": ll_picks}), encoding="utf-8")
        ca_p.write_text(json.dumps({**base, "label": "trunk-4b", "picks": cand_picks}), encoding="utf-8")

        rc = cmd_score(argparse.Namespace(teacher=t_all, cand=ca_p, lossless=ll_p, json_out=sc_p))
        assert rc == 0
        res = json.loads(sc_p.read_text(encoding="utf-8"))
        assert res["positions_scored"] == 512
        assert res["overall_matched"] == 496
        assert res["lossless_matched"] == 500
        assert res["both_match"] == 490
        assert res["lossless_only"] == 10
        assert res["cand_only"] == 6
        assert res["neither"] == 6
        assert res["within_1pt"] is True  # (496-500)/512 = -0.781 pt >= -1.0
        assert res["prose_total"] == 256 and res["code_total"] == 256
        assert res["margin_lt1_total"] == 0 or res["margin_ge1_total"] > 0

        # Also test pass1_state.safetensors fallback (bonus_tok [64] at cut=384)
        p1 = d / "pass1_state.safetensors"
        bonus = [picks_flat[i * seq_len + 383] for i in range(n_pref)]
        write_safetensors(p1, {"bonus_tok": ("I64", [n_pref], struct.pack(f"<{n_pref}q", *bonus))})
        sc2 = d / "score_p1.json"
        rc2 = cmd_score(argparse.Namespace(teacher=p1, cand=ca_p, lossless=ll_p, json_out=sc2))
        assert rc2 == 0
        res2 = json.loads(sc2.read_text(encoding="utf-8"))
        assert res2["positions_scored"] == 64
        print("SELFTEST: PASS (512-pos paired counts + 64-pos pass1 fallback verified)", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp_c = sub.add_parser("collect", help="Collect greedy next-token picks from a live server")
    sp_c.add_argument("--url", required=True, help="Base server URL, e.g. http://127.0.0.1:8080")
    sp_c.add_argument("--model", required=True, help="Model name for /v1/completions")
    sp_c.add_argument("--prefixes", required=True, type=Path, help="Path to prefixes.safetensors")
    sp_c.add_argument("--cuts", default="384,192,288,96,336,144,240,48", help="Comma-separated prefix cut lengths (384 first, then spread across span)")
    sp_c.add_argument("--max-prefixes", type=int, default=0, help="Optional cap on prefixes (0 = all)")
    sp_c.add_argument("--max-seconds", type=float, default=270.0, help="Wall-clock cap in seconds (stops on complete cut boundary)")
    sp_c.add_argument("--match-grid", type=Path, default=None, help="Match the exact (prefix, cut) grid of a prior lossless JSON")
    sp_c.add_argument("--timeout", type=float, default=120.0, help="Per-request HTTP timeout seconds")
    sp_c.add_argument("--label", required=True, help="Build label")
    sp_c.add_argument("--out", required=True, type=Path, help="Output picks JSON")

    sp_s = sub.add_parser("score", help="Score picks against teacher safetensors and optional lossless baseline")
    sp_s.add_argument("--teacher", required=True, type=Path, help="Reference picks safetensors (picks/margin, or bonus_tok)")
    sp_s.add_argument("--cand", required=True, type=Path, help="Candidate picks JSON")
    sp_s.add_argument("--lossless", type=Path, default=None, help="Lossless baseline picks JSON for paired counts")
    sp_s.add_argument("--json-out", type=Path, default=None, help="Optional JSON summary path")

    sub.add_parser("selftest", help="Run offline fixture self-test")

    args = ap.parse_args()
    if args.cmd == "collect":
        return cmd_collect(args)
    if args.cmd == "score":
        return cmd_score(args)
    return cmd_selftest(args)


if __name__ == "__main__":
    raise SystemExit(main())
