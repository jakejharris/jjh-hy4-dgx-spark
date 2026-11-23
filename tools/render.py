#!/usr/bin/env python3
"""Render claim slots in markdown files from card/claims.json.

Slots:
  {{claim:ID}}   value and unit, then any status note and the spread in brackets
  {{value:ID}}   value and unit (and a status note such as "component"), no spread
  {{bare:ID}}    value without its unit
  {{spread:ID}}  the spread text alone (empty when a measured claim has none)
  {{rows:ID}}    the claim's per-prompt table, on a line of its own (falls back to the value)

A pending claim renders as "(pending)", in every kind of slot. An interrupted claim (its run started but did
not finish, so nothing was measured) and a not_run claim (its run was planned and dropped before it started)
render their status_note instead, and an empty spread. Numbers are written one way everywhere:
at least two decimals, a percent sign or "x" right after the number, a difference in points with its sign,
byte and token counts with thousands separators, and a whole number of seconds without decimals (load to ready
is counted in whole seconds).

Every page with slots is written from a template in card/templates/ (TEMPLATES below). Edit the template,
then write the public pages:

  python3 tools/render.py --all      write every public page from its template
  python3 tools/render.py --check    exit 1 if a public page differs from a fresh render (tools/check_kit.py checks it too)
  python3 tools/render.py FILE...    print FILE rendered (or rewrite it with --in-place)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SLOT_RE = re.compile(r"\{\{(claim|value|bare|spread|rows):([a-z0-9_.]+)\}\}")
# Template (where the slots are written) -> public page (what a reader opens). Paths from the kit root.
# Links in a template are written for its public page.
TEMPLATES = {
    "card/templates/README.md": "README.md",
    "card/templates/MODEL_CARD.md": "card/MODEL_CARD.md",
    "card/templates/QUALITY_CARD.md": "card/QUALITY_CARD.md",
    "card/templates/FINDINGS.md": "FINDINGS.md",
    "card/templates/RING-RDMA.md": "launch/RING-RDMA.md",
    "card/templates/quant-README.md": "quant/README.md",
}
ONE_DECIMAL = re.compile(r"(?<![\d.])([+-]?\d+\.\d)(?!\d|\.\d)")
SPACE_PERCENT = re.compile(r"(\d) %")
BYTE_UNITS = {"B", "B/token", "bytes", "tokens"}  # counts written with thousands separators
TIGHT_UNITS = {"%": "%", "×": "x", "x": "x"}
TWO_DECIMAL_UNITS = {"%", "tok/s", "GiB", "bpw", "points"}  # a whole number in these units still shows two decimals
SIGNED_UNITS = {"points"}  # a difference always shows its sign
WHOLE_UNITS = {"s"}  # a whole number of seconds is written whole: "79 s" (load to ready); 220.79 s keeps its decimals
PLAIN_NUMBER = re.compile(r"[+-]?\d+(\.\d+)?")


def fmt_text(text: str) -> str:
    """Numbers inside ledger text: two decimals at least, no space before %, ASCII x."""
    text = ONE_DECIMAL.sub(r"\g<1>0", str(text))
    text = SPACE_PERCENT.sub(r"\1%", text)
    return text.replace("×", "x")


def fmt_number(value: object, unit: str) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if unit in BYTE_UNITS and (isinstance(value, int) or (isinstance(value, str) and value.isdigit())):
        return f"{int(value):,}"
    if unit == "GB" and isinstance(value, float) and round(value, 2) == value:  # sizes in GB: "1,539.81 GB"
        return f"{value:,.2f}"
    if unit in WHOLE_UNITS and isinstance(value, (int, float)) and float(value).is_integer():
        return str(int(value))
    if unit in TWO_DECIMAL_UNITS and PLAIN_NUMBER.fullmatch(str(value)):
        value = float(value)
    if unit in SIGNED_UNITS and isinstance(value, float) and round(value, 2) == value:
        return f"{value:+.2f}"
    if isinstance(value, float):
        return f"{value:.2f}" if round(value, 2) == value else repr(value)
    return fmt_text(value)


def with_unit(number: str, unit: str) -> str:
    if not unit:
        return number
    if unit in TIGHT_UNITS:
        return number + TIGHT_UNITS[unit]
    return f"{number} {fmt_text(unit)}"


def is_pending(entry: dict) -> bool:
    return entry.get("status", "pending") == "pending" or entry.get("value") is None


def status_note(entry: dict) -> str:
    status = entry.get("status", "pending")
    return status if status in ("provisional", "component") else ""


def format_claim(entry: dict, kind: str = "claim") -> str:
    if entry.get("status") == "interrupted":
        return "" if kind == "spread" else fmt_text(entry.get("status_note") or "not measured (run interrupted)")
    if entry.get("status") == "not_run":
        return "" if kind == "spread" else fmt_text(entry.get("status_note") or "not run")
    if is_pending(entry):
        return "(pending)"
    unit = entry.get("unit") or ""
    spread = entry.get("spread")
    spread = fmt_text(spread) if spread is not None and str(spread).strip() else ""
    if kind == "spread":
        return spread
    if kind == "rows":
        table = entry.get("table")
        if isinstance(table, dict) and table.get("columns") and table.get("rows"):
            cols = [fmt_text(c) for c in table["columns"]]
            lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
            lines += ["| " + " | ".join(fmt_text(c) for c in row) + " |" for row in table["rows"]]
            return "\n".join(lines)
        kind = "value"
    number = fmt_number(entry.get("value"), unit)
    base = number if kind == "bare" else with_unit(number, unit)
    notes = [n for n in (status_note(entry),) if n]
    if kind == "claim" and spread:
        notes.append(f"spread: {spread}")
    return f"{base} ({'; '.join(notes)})" if notes else base


def render_text(text: str, by_id: dict[str, dict], source_name: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        kind, cid = match.group(1), match.group(2)
        if cid not in by_id:
            raise KeyError(f"Unknown claim id '{cid}' in {source_name}")
        return format_claim(by_id[cid], kind)

    return SLOT_RE.sub(_replace, text)


def load_claims(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {c["id"]: c for c in data.get("claims", []) if isinstance(c, dict) and "id" in c}


def render_templates(root: Path, by_id: dict[str, dict]) -> dict[str, str]:
    """Public page path -> its text rendered from the template."""
    return {pub: render_text((root / tpl).read_text(encoding="utf-8"), by_id, tpl) for tpl, pub in TEMPLATES.items()}


def drifted(root: Path, pages: dict[str, str]) -> list[str]:
    """Public pages that are missing or differ from their fresh render."""
    return [pub for pub, text in pages.items()
            if not (root / pub).is_file() or (root / pub).read_text(encoding="utf-8") != text]


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--claims",
        type=Path,
        default=root / "card" / "claims.json",
        help="Path to claims.json",
    )
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite target files in place instead of printing to stdout",
    )
    ap.add_argument("--all", action="store_true", help="Write every public page from its template (TEMPLATES)")
    ap.add_argument("--check", action="store_true", help="Exit 1 if a public page differs from a fresh render")
    ap.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Markdown file(s) to render",
    )
    args = ap.parse_args()
    if not (args.all or args.check or args.files):
        ap.error("give --all, --check or one or more files")

    if not args.claims.is_file():
        print(f"ERROR: claims ledger not found: {args.claims}", file=sys.stderr)
        return 1

    by_id = load_claims(args.claims)

    if args.all or args.check:
        try:
            pages = render_templates(root, by_id)
        except KeyError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        stale = drifted(root, pages)
        if args.check:
            for pub in stale:
                print(f"ERROR: {pub} is missing or differs from a fresh render of its template; run python3 tools/render.py --all",
                      file=sys.stderr)
            return 1 if stale else 0
        for pub in stale:
            (root / pub).write_text(pages[pub], encoding="utf-8")
        print(f"rendered {len(pages)} pages, rewrote {len(stale)}: {', '.join(stale) or 'none'}")
        return 0

    for path in args.files:
        text = path.read_text(encoding="utf-8")
        try:
            rendered = render_text(text, by_id, str(path))
        except KeyError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.in_place:
            path.write_text(rendered, encoding="utf-8")
        else:
            if len(args.files) > 1:
                print(f"=== {path} ===")
            print(rendered, end="" if rendered.endswith("\n") else "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
