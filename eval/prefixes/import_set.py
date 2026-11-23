#!/usr/bin/env python3
"""Copy a public prefix set into eval/prefixes/NAME/ and record its licenses.

    python3 eval/prefixes/import_set.py NAME SRC_DIR [--expect FILE=SHA256 ...]

SRC_DIR holds the set as it was built:
  manifest.json          one item per prefix: source, URL, license, genre, source sha256
  prefixes.safetensors   input_ids I64 [N, 384], is_code U8 [N]
  reference picks        optional; the reference's greedy pick per position (picks, top2,
                         margin, input_ids), found as reference_picks.safetensors or
                         teacher_*.safetensors unless --picks names it

What it does, in order, and it writes nothing until every check passes:
  1. Every --expect FILE=SHA256 (and every entry of a "files" table in the source
     manifest) must match the source file's sha256.
  2. Every manifest item must name a license. One item without a license refuses the set.
     A set whose items carry byline, license_basis_url and license_basis_sentence must fill
     all three on every shipped item, and the items of one source must agree on them.
  3. The manifest must agree with the tensors: count and seq_len against input_ids,
     genre against is_code, and the reference picks' input_ids must equal the prefixes.
  4. Items marked "shipped": false stay out of the copy (their rows are cut from the
     prefixes and the reference picks alike).
  5. Writes eval/prefixes/NAME/: manifest.json (with the sha256 of every file written and
     of every source file read), prefixes.safetensors, reference_picks.safetensors when
     picks exist, and stats.json: the reference's own next-token accuracy on the set and
     the share of close calls (top-two logit gap under 0.5), the two numbers printed
     beside every agreement row.
  6. Rewrites this set's rows in THIRD_PARTY.md (between its begin and end markers): one
     row per source with its URL, license and prefix count, plus its byline and license
     basis (the sentence and the page it is quoted from) when the manifest carries them.

Standard library only. Exit 0 on success, 2 when a check refuses the set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CLOSE_CALL = 0.5
ITEMSIZE = {"I64": 8, "F32": 4, "U8": 1, "I32": 4, "BF16": 2, "F16": 2}


class Refuse(Exception):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_st(path: Path) -> dict[str, tuple[str, list[int], bytes]]:
    data = path.read_bytes()
    n = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8 : 8 + n])
    base = 8 + n
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        out[name] = (meta["dtype"], list(meta["shape"]), data[base + a : base + b])
    return out


def write_st(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header, blobs, off = {}, [], 0
    for name in sorted(tensors):
        dtype, shape, raw = tensors[name]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + b"".join(blobs))


def keep_rows(t: tuple[str, list[int], bytes], rows: list[int]) -> tuple[str, list[int], bytes]:
    dtype, shape, raw = t
    width = len(raw) // shape[0]
    return dtype, [len(rows)] + shape[1:], b"".join(raw[r * width : (r + 1) * width] for r in rows)


def ints(t: tuple[str, list[int], bytes]) -> list[int]:
    dtype, _, raw = t
    fmt = {"I64": "q", "I32": "i", "U8": "B"}[dtype]
    return list(struct.unpack(f"<{len(raw) // ITEMSIZE[dtype]}{fmt}", raw))


def floats(t: tuple[str, list[int], bytes]) -> list[float]:
    dtype, _, raw = t
    if dtype != "F32":
        raise Refuse(f"margin is {dtype}, expected F32")
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def find_picks(src: Path, named: str | None) -> Path | None:
    if named:
        p = src / named
        if not p.is_file():
            raise Refuse(f"--picks {named}: not found in {src}")
        return p
    for cand in ["reference_picks.safetensors"] + sorted(x.name for x in src.glob("teacher_*.safetensors")):
        if (src / cand).is_file():
            return src / cand
    return None


def ref_stats(picks: dict, prefixes: dict, is_code: list[int]) -> dict:
    """Reference next-token accuracy (picks[:, :-1] == input_ids[:, 1:]) and close-call share."""
    n, s = picks["picks"][1]
    p = ints(picks["picks"])
    ids = ints(prefixes["input_ids"])
    m = floats(picks["margin"])
    acc = {"prose": [0, 0], "code": [0, 0]}
    close = {"prose": [0, 0], "code": [0, 0]}
    for r in range(n):
        g = "code" if is_code[r] else "prose"
        for c in range(s):
            if c < s - 1:
                acc[g][0] += int(p[r * s + c] == ids[r * s + c + 1])
                acc[g][1] += 1
            close[g][0] += int(m[r * s + c] < CLOSE_CALL)
            close[g][1] += 1

    def pct(a: int, b: int) -> float | None:
        return round(100.0 * a / b, 2) if b else None

    out = {
        "definition": {
            "reference_next_token_accuracy": "share of positions where the reference's greedy pick equals the next token of the published text (picks[:, :-1] == input_ids[:, 1:])",
            "close_calls": f"share of positions where the reference's top-two logit gap (margin) is under {CLOSE_CALL}",
        },
        "prefixes": n,
        "seq_len": s,
    }
    for g in ("prose", "code"):
        out[f"reference_next_token_accuracy_{g}_pct"] = pct(*acc[g])
        out[f"close_calls_{g}_pct"] = pct(*close[g])
    out["reference_next_token_accuracy_pct"] = pct(acc["prose"][0] + acc["code"][0], acc["prose"][1] + acc["code"][1])
    out["close_calls_pct"] = pct(close["prose"][0] + close["code"][0], close["prose"][1] + close["code"][1])
    out["close_calls"] = close["prose"][0] + close["code"][0]
    out["positions"] = close["prose"][1] + close["code"][1]
    return out


CREDIT_FIELDS = ("byline", "license_basis_url", "license_basis_sentence")


def cell(value: object) -> str:
    """One markdown table cell: no line breaks, pipes escaped."""
    return " ".join(str(value or "").split()).replace("|", "\\|")


def byline_cell(value: object) -> str:
    """The byline without an address: names only. The full copyright line stays in the license text (LICENSES/)."""
    text = re.sub(r"\s*<[^<>\s]+@[^<>\s]+>", "", str(value or ""))
    text = re.sub(r",\s*,", ",", text)
    return cell(text)


def check_credit_fields(items: list[dict]) -> bool:
    """True when the set carries byline and license-basis fields. Each source's items must agree on them,
    since THIRD_PARTY.md has one row per source; a set that uses them must fill them on every item."""
    used = any(str(it.get(k) or "").strip() for it in items for k in CREDIT_FIELDS)
    if not used:
        return False
    empty = [it.get("idx", i) for i, it in enumerate(items) if not all(str(it.get(k) or "").strip() for k in CREDIT_FIELDS)]
    if empty:
        raise Refuse(f"manifest items missing one of {', '.join(CREDIT_FIELDS)}: {empty}")
    seen: dict[str, tuple] = {}
    for it in items:
        key = tuple(cell(it.get(k)) for k in CREDIT_FIELDS)
        if seen.setdefault(it["source_id"], key) != key:
            raise Refuse(f"source {it['source_id']}: items disagree on {', '.join(CREDIT_FIELDS)}")
    return True


def third_party_block(name: str, items: list[dict]) -> str:
    credits = check_credit_fields(items)
    rows: dict[str, dict] = {}
    for it in items:
        row = rows.setdefault(it["source_id"], {"title": it.get("title") or it["source_id"], "url": it.get("url", ""),
                                                "license": it["license"], "n": 0,
                                                **{k: it.get(k, "") for k in CREDIT_FIELDS}})
        row["n"] += 1
    head = f"<!-- prefix-set:{name} begin (written by eval/prefixes/import_set.py; do not edit by hand) -->"
    if credits:
        lines = [head,
                 f"| Prefix set `{name}`: source | URL | License | Byline | License basis | Prefixes |",
                 "| --- | --- | --- | --- | --- | ---: |"]
    else:
        lines = [head,
                 f"| Prefix set `{name}`: source | URL | License | Prefixes |",
                 "| --- | --- | --- | ---: |"]
    for sid in sorted(rows):
        r = rows[sid]
        if credits:
            basis = f"{cell(r['license_basis_sentence'])} ({cell(r['license_basis_url'])})"
            lines.append(f"| {cell(r['title'])} | {cell(r['url'])} | {cell(r['license'])} | {byline_cell(r['byline'])} | {basis} | {r['n']} |")
        else:
            lines.append(f"| {cell(r['title'])} | {cell(r['url'])} | {cell(r['license'])} | {r['n']} |")
    lines.append(f"<!-- prefix-set:{name} end -->")
    return "\n".join(lines)


def update_third_party(path: Path, name: str, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    begin = f"<!-- prefix-set:{name} begin"
    end = f"<!-- prefix-set:{name} end -->"
    if begin in text and end in text:
        a = text.index(begin)
        b = text.index(end) + len(end)
        text = text[:a] + block + text[b:]
    else:
        text = text.rstrip("\n") + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    src = args.src_dir
    man_path = src / args.manifest
    pre_path = src / args.prefixes
    for p in (man_path, pre_path):
        if not p.is_file():
            raise Refuse(f"{p.name}: not found in {src}")
    picks_path = None if args.no_picks else find_picks(src, args.picks)
    manifest = json.loads(man_path.read_text(encoding="utf-8"))

    # 1. hashes
    expect = dict(manifest.get("files") or {})
    for e in args.expect:
        if "=" not in e:
            raise Refuse(f"--expect {e!r}: use FILE=SHA256")
        f, h = e.split("=", 1)
        expect[f] = h.lower()
    read = {p.name: sha256(p) for p in (man_path, pre_path, picks_path) if p}
    for f, h in expect.items():
        if f not in read:
            if not (src / f).is_file():
                raise Refuse(f"expected file {f} is not in {src}")
            read[f] = sha256(src / f)
        if read[f] != h:
            raise Refuse(f"{f}: sha256 {read[f]} does not match the expected {h}")

    # 2. licenses
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise Refuse("manifest has no items")
    missing = [it.get("idx", i) for i, it in enumerate(items) if not str(it.get("license") or "").strip()]
    if missing:
        raise Refuse(f"manifest items without a license: {missing}")

    # 3. manifest against tensors
    pre = read_st(pre_path)
    if "input_ids" not in pre:
        raise Refuse(f"{pre_path.name} has no input_ids")
    n, s = pre["input_ids"][1]
    if manifest.get("count") != n or manifest.get("seq_len") != s or len(items) != n:
        raise Refuse(f"manifest says {manifest.get('count')}x{manifest.get('seq_len')} with {len(items)} items; "
                     f"{pre_path.name} holds {n}x{s}")
    is_code = ints(pre["is_code"]) if "is_code" in pre else [int(it.get("genre") == "code") for it in items]
    for i, it in enumerate(items):
        if it.get("idx", i) != i:
            raise Refuse(f"manifest item {i} has idx {it.get('idx')}")
        if (it.get("genre") == "code") != bool(is_code[i]):
            raise Refuse(f"item {i}: genre {it.get('genre')!r} but is_code {is_code[i]}")
    picks = None
    if picks_path:
        picks = read_st(picks_path)
        for need in ("picks", "margin"):
            if need not in picks or picks[need][1] != [n, s]:
                raise Refuse(f"{picks_path.name}: {need} missing or not [{n}, {s}]")
        if "input_ids" in picks and picks["input_ids"][2] != pre["input_ids"][2]:
            raise Refuse(f"{picks_path.name}: input_ids differ from {pre_path.name}; the picks belong to another set")

    # 4. shipped rows only
    rows = [i for i, it in enumerate(items) if it.get("shipped", True) is not False]
    dropped = [i for i in range(n) if i not in rows]
    if not rows:
        raise Refuse("no item is marked shipped")
    out_pre = {k: keep_rows(v, rows) for k, v in pre.items()}
    if "is_code" not in out_pre:
        out_pre["is_code"] = ("U8", [len(rows)], bytes(is_code[i] for i in rows))
    out_picks = {k: keep_rows(v, rows) for k, v in picks.items()} if picks else None
    out_items = []
    for new, i in enumerate(rows):
        it = dict(items[i])
        it["source_idx"] = it.get("idx", i)
        it["idx"] = new
        out_items.append(it)
    credits = check_credit_fields(out_items)

    dest = ROOT / "eval" / "prefixes" / args.name
    if args.dry_run:
        print(f"would write {dest.relative_to(ROOT)}: {len(rows)} of {n} prefixes (dropped not shipped: {dropped}); "
              f"THIRD_PARTY.md rows {'with' if credits else 'without'} byline and license basis")
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    write_st(dest / "prefixes.safetensors", out_pre)
    files = {"prefixes.safetensors": sha256(dest / "prefixes.safetensors")}
    stats = None
    if out_picks:
        write_st(dest / "reference_picks.safetensors", out_picks)
        files["reference_picks.safetensors"] = sha256(dest / "reference_picks.safetensors")
        stats = ref_stats(out_picks, out_pre, [is_code[i] for i in rows])
        (dest / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        files["stats.json"] = sha256(dest / "stats.json")
    out_man = {k: v for k, v in manifest.items() if k not in ("items", "files")}
    out_man.update({
        "name": args.name,
        "count": len(rows),
        "seq_len": s,
        "prose_count": sum(1 for i in rows if not is_code[i]),
        "code_count": sum(1 for i in rows if is_code[i]),
        "dropped_not_shipped": [items[i].get("source_id") for i in dropped],
        "imported_from": read,
        "files": files,
        "items": out_items,
    })
    out_man.pop("shipped_prose_count", None)
    out_man.pop("shipped_code_count", None)
    (dest / "manifest.json").write_text(json.dumps(out_man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    update_third_party(ROOT / "THIRD_PARTY.md", args.name, third_party_block(args.name, out_items))

    print(f"imported {args.name}: {len(rows)} of {n} prefixes x {s} tokens "
          f"({out_man['prose_count']} prose, {out_man['code_count']} code); dropped not shipped: {len(dropped)}")
    for f, h in read.items():
        print(f"  source {f} sha256 {h}{' (checked)' if f in expect else ''}")
    for f, h in files.items():
        print(f"  wrote  {args.name}/{f} sha256 {h}")
    if stats:
        print(f"  reference next-token accuracy {stats['reference_next_token_accuracy_pct']}% "
              f"(prose {stats['reference_next_token_accuracy_prose_pct']}%, code {stats['reference_next_token_accuracy_code_pct']}%); "
              f"close calls {stats['close_calls_pct']}% ({stats['close_calls']} of {stats['positions']})")
    else:
        print("  no reference picks in the source: agreement on this set stays pending until they exist")
    print(f"  THIRD_PARTY.md rows for {args.name}: {len({it['source_id'] for it in out_items})} sources")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="set name; the copy goes to eval/prefixes/NAME/")
    ap.add_argument("src_dir", type=Path, help="folder holding the set's manifest and tensors")
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--prefixes", default="prefixes.safetensors")
    ap.add_argument("--picks", default=None, help="reference picks file in SRC_DIR (default: reference_picks.safetensors or teacher_*.safetensors)")
    ap.add_argument("--no-picks", action="store_true", help="import without reference picks (they do not exist yet); agreement on the set stays pending")
    ap.add_argument("--expect", action="append", default=[], help="FILE=SHA256 a source file must match (repeatable)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    try:
        return run(args)
    except Refuse as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
