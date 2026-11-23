#!/usr/bin/env python3
"""Parse tf-hy4-oracle --bench-window output into a TF_HY4_ROW_COST_TABLE spec.

Format expected by TensorFold draft_policy.CostParams.parseTableSpec:
  v1:<ms>,v2:<ms>,v3:<ms>,v4:<ms>,v5:<ms>,v6:<ms>,v7:<ms>,d:<ms>,e:<ms>

vN is the median time of one whole verify step over N rows (all 78 layers and
the head), read from the rank-0 log's verifyN_med fields. Widths the bench did
not run (v4 to v6 with --bench-widths 1,2,3,7) are interpolated linearly
between the nearest measured widths. d (draft round overhead) and e (per draft
step) are read from the log when it carries draft_round= and draft_each=;
otherwise the transport's defaults are used and the script says so.
tools/row_cost_table.sh runs the bench and this parser in one step.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Matches e.g.:
#   width=1 ... median=124.81 ms
#   W=1: 124.81 ms
#   bench-window width 1: 124.81 ms (or verify1=124.81 ms)
RE_PATTERNS = [
    re.compile(r"\bverify(?P<w>[1-7])_med\s*=\s*(?P<ms>[0-9]+(?:\.[0-9]+)?)\s*ms", re.I),
    re.compile(r"\bwidth[=\s]+(?P<w>[1-7])\b[^\n]*?(?:median|med|mean|ms)[=:\s]+(?P<ms>[0-9]+(?:\.[0-9]+)?)\s*ms", re.I),
    re.compile(r"\bW[=\s]*(?P<w>[1-7])\b[^\n]*?[:=\s](?P<ms>[0-9]+(?:\.[0-9]+)?)\s*ms", re.I),
    re.compile(r"\bverify(?P<w>[1-7])\s*=\s*(?P<ms>[0-9]+(?:\.[0-9]+)?)\s*ms", re.I),
    re.compile(r"\bv(?P<w>[1-7])\s*:\s*(?P<ms>[0-9]+(?:\.[0-9]+)?)", re.I),
]
RE_DRAFT_ROUND = re.compile(r"draft_round\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.I)
RE_DRAFT_EACH = re.compile(r"draft_each\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.I)


def parse_widths(text: str) -> dict[int, float]:
    found: dict[int, float] = {}
    for pat in RE_PATTERNS:
        for m in pat.finditer(text):
            w = int(m.group("w"))
            ms = float(m.group("ms"))
            found[w] = ms
    return found


def interpolate_v(measured: dict[int, float]) -> dict[int, float]:
    if 1 not in measured or 7 not in measured:
        raise SystemExit(f"ERROR: need at least W=1 and W=7 measured; got {sorted(measured)}")
    pts = sorted(measured.items())
    out: dict[int, float] = dict(measured)
    for w in range(1, 8):
        if w in out:
            continue
        # Linear interpolation between nearest measured points bracketing w
        lo_w, lo_v = max((pw, pv) for pw, pv in pts if pw < w)
        hi_w, hi_v = min((pw, pv) for pw, pv in pts if pw > w)
        frac = (w - lo_w) / float(hi_w - lo_w)
        out[w] = lo_v + frac * (hi_v - lo_v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, default=None, help="Rank-0 oracle log containing --bench-window output")
    ap.add_argument("--v1", type=float, default=None)
    ap.add_argument("--v2", type=float, default=None)
    ap.add_argument("--v3", type=float, default=None)
    ap.add_argument("--v7", type=float, default=None)
    ap.add_argument("--draft-round-ms", type=float, default=None, help="Draft round overhead d (ms); default from log or 7.63 (stock) / 6.44 (relay)")
    ap.add_argument("--draft-each-ms", type=float, default=None, help="Per-draft-step e (ms); default from log or 2.76 (stock) / 0.41 (relay)")
    ap.add_argument("--transport", choices=("stock", "relay"), default="stock")
    ap.add_argument("--out", type=Path, required=True, help="Output row-cost-table .txt file")
    args = ap.parse_args()

    measured: dict[int, float] = {}
    d_ms = args.draft_round_ms
    e_ms = args.draft_each_ms

    if args.log:
        if not args.log.is_file():
            raise SystemExit(f"ERROR: log not found: {args.log}")
        txt = args.log.read_text(encoding="utf-8", errors="replace")
        measured.update(parse_widths(txt))
        if d_ms is None:
            md = RE_DRAFT_ROUND.search(txt)
            if md:
                d_ms = float(md.group(1))
        if e_ms is None:
            me = RE_DRAFT_EACH.search(txt)
            if me:
                e_ms = float(me.group(1))

    for w, val in ((1, args.v1), (2, args.v2), (3, args.v3), (7, args.v7)):
        if val is not None:
            measured[w] = val

    if d_ms is None:
        d_ms = 6.44 if args.transport == "relay" else 7.63
        print(f"note: d not in the log; using the {args.transport} default {d_ms} ms")
    if e_ms is None:
        e_ms = 0.41 if args.transport == "relay" else 2.76
        print(f"note: e not in the log; using the {args.transport} default {e_ms} ms")

    v = interpolate_v(measured)
    spec = ",".join(f"v{w}:{v[w]:.2f}" for w in range(1, 8)) + f",d:{d_ms:.2f},e:{e_ms:.2f}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(spec + "\n", encoding="utf-8")
    print(f"ROW_COST_TABLE_OK out={args.out} spec={spec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
