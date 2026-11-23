#!/usr/bin/env python3
"""Fill card/claims.json from a folder of boot outputs, using card/fill-map.json.

For each claim in the map, the script reads the output file the map names,
checks that the labels in that file (and in the boot record next to it) match
the claim's build, trunk width, context window and draft setting, takes the
value and spread, marks the claim measured and copies the file into receipts/.

It prints one line per claim:
  filled   value and spread written, receipt copied; "[forced over X]" when --force
           replaced an earlier measured value X
  skipped  the output file is missing, the claim is already measured
           (pass --force to overwrite), or the trunk width is not picked yet
  refused  the file is there, but a field is absent or not a finite number,
           or a label does not match the claim

It never writes a value it did not read. After writing the ledger it rewrites
the public pages from their templates in card/templates/ (as tools/render.py
--all does), so the README and cards show the new values. --dry-run prints the
same lines and writes nothing.

    python3 tools/fill_claims.py --outputs OUTDIR [--pick 4b|4b6b|6b] [--dry-run] [--force]

The RigMark claims (rigmark.*) read the JSON receipt of RigMark's whole standard suite with thinking off,
rigmark/<label>.json under OUTDIR; a quick or partial run is refused:

    python3 tools/fill_claims.py --outputs OUTDIR --only rigmark.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WIDTHS = ("4b", "4b6b", "6b")

# Receipt copies are redacted the way tools/check_kit.py requires: machine host
# names become node1..node4, home folders become ~/. Values are not touched.
# (Patterns are assembled so this file does not contain the strings it removes.)
REDACT = [
    ("host names", re.compile(r"(?<![A-Za-z0-9])spark" + r"([1-4])(?![0-9])", re.IGNORECASE), r"node\1"),
    ("home paths", re.compile("/ho" + r"me/[^/\s\"']+/"), "~/"),
    ("home paths", re.compile("/ho" + r"me/[^/\s\"']+\b"), "~"),
]


def redacted_copy(src: Path, dest: Path) -> list[str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = src.read_text()
    except UnicodeDecodeError:
        shutil.copyfile(src, dest)
        return []
    notes = {}
    for label, pat, repl in REDACT:
        text, n = pat.subn(repl, text)
        if n:
            notes[label] = notes.get(label, 0) + n
    dest.write_text(text)
    return [f"{n} {label}" for label, n in notes.items()]


class Refuse(Exception):
    """The file is present but cannot back the claim."""


class Skip(Exception):
    """Nothing to read yet."""


# ---------------------------------------------------------------- helpers


def load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise Refuse(f"{path.name} is not readable JSON ({exc.__class__.__name__})") from exc


def finite(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refuse(f"{what} absent or not a number ({value!r})")
    if not math.isfinite(float(value)):
        raise Refuse(f"{what} is not finite ({value!r})")
    return float(value)


def get(obj: object, dotted: str, what: str) -> object:
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise Refuse(f"{what}: field {dotted} absent")
        cur = cur[part]
    return cur


def num(x: float) -> float | int:
    r = round(x, 2)
    return int(r) if r == int(r) and abs(r) >= 1000 else r


def spread_text(values: list[float], what: str = "runs") -> str:
    return f"{num(min(values))} to {num(max(values))} over {len(values)} {what}"


def table(columns: list[str], rows: list[list[str]]) -> dict:
    """A per-prompt table kept on the claim (tools/render.py draws it for a {{rows:ID}} slot)."""
    return {"columns": columns, "rows": rows}


def r2(x: float) -> str:
    return f"{x:.2f}"


def prefix_set_kind(kit: Path, name: str = "familiar") -> str | None:
    """Reference shape agreement.py reports for one of the kit's prefix sets, e.g. teacher_all(84x384)."""
    manifest = kit / "eval" / "prefixes" / name / "manifest.json"
    if not manifest.is_file():
        return None
    m = json.loads(manifest.read_text())
    return f"teacher_all({m['count']}x{m['seq_len']})"


def resolve(value: object, pick: str | None, pick_drafter: dict, kit: Path) -> object:
    if value == "@pick":
        return pick
    if value == "@pick_drafter":
        return pick_drafter.get(pick or "", None)
    if isinstance(value, str) and (value == "@prefix_set" or value.startswith("@prefix_set:")):
        return prefix_set_kind(kit, value.partition(":")[2] or "familiar")
    return value


# ---------------------------------------------------------------- label checks


def check_drafter(seen: object, want: object, where: str) -> None:
    if want is None:
        return
    if want == "on":
        if not (isinstance(seen, str) and seen.startswith("DFly2")):
            raise Refuse(f"{where} drafter is {seen!r}, claim needs drafts on")
    elif seen != want:
        raise Refuse(f"{where} drafter is {seen!r}, claim needs {want!r}")


def check_boot_meta(meta: dict, exp: dict, where: str) -> None:
    if meta.get("status") != "PASS":
        raise Refuse(f"{where} boot status is {meta.get('status')!r}, not PASS")
    if "trunk" in exp and meta.get("trunk") != exp["trunk"]:
        raise Refuse(f"{where} trunk is {meta.get('trunk')!r}, claim needs {exp['trunk']!r}")
    check_drafter(meta.get("drafter"), exp.get("drafter"), where)
    if "context" in exp and meta.get("context_window") != exp["context"]:
        raise Refuse(
            f"{where} context window is {meta.get('context_window')!r}, claim needs {exp['context']}"
        )
    if "layers" in exp and meta.get("layers") != exp["layers"]:
        raise Refuse(f"{where} layers is {meta.get('layers')!r}, claim needs {exp['layers']}")
    if "transport" in exp and meta.get("transport") != exp["transport"]:
        raise Refuse(f"{where} transport is {meta.get('transport')!r}, claim needs {exp['transport']!r}")
    if "repeat_stable" in exp and bool(meta.get("repeat_stable")) is not exp["repeat_stable"]:
        raise Refuse(
            f"{where} repeat_stable is {meta.get('repeat_stable')!r}, claim needs {exp['repeat_stable']} "
            "(the boot record carries what the server's startup line said)"
        )


def check_summary(summary: dict, exp: dict, where: str) -> None:
    ranks = summary.get("ranks")
    if not isinstance(ranks, dict) or len(ranks) != exp.get("ranks", 4):
        raise Refuse(f"{where} has {len(ranks) if isinstance(ranks, dict) else 0} ranks, claim needs {exp.get('ranks', 4)}")
    for r, info in sorted(ranks.items()):
        if "trunk" in exp and info.get("trunk_loaded") != exp["trunk"]:
            raise Refuse(f"{where} rank {r} trunk is {info.get('trunk_loaded')!r}, claim needs {exp['trunk']!r}")
        if "context" in exp and info.get("context_window") != exp["context"]:
            raise Refuse(f"{where} rank {r} context window is {info.get('context_window')!r}, claim needs {exp['context']}")
        check_drafter(info.get("drafter_loaded"), exp.get("drafter"), f"{where} rank {r}")
        if "layers" in exp and info.get("layers") != exp["layers"]:
            raise Refuse(f"{where} rank {r} layers is {info.get('layers')!r}, claim needs {exp['layers']}")


THINKING = {"on": "omit", "off": "no_think"}


def check_bench(doc: dict, exp: dict, name: str) -> list[dict]:
    runs = doc.get("runs")
    if not isinstance(runs, list) or not runs:
        raise Refuse(f"{name}: no runs")
    if "prompt" in exp:
        prompt = str(doc.get("prompt", ""))
        if Path(prompt).name != exp["prompt"]:
            raise Refuse(f"{name}: prompt is {Path(prompt).name or None!r}, claim needs {exp['prompt']!r}")
    if "thinking" in exp:
        want = THINKING[exp["thinking"]]
        if doc.get("reasoning_effort") != want:
            raise Refuse(f"{name}: reasoning_effort is {doc.get('reasoning_effort')!r}, claim needs {want!r}")
    if exp.get("runs") and len(runs) < exp["runs"]:
        raise Refuse(f"{name}: {len(runs)} runs, claim needs {exp['runs']}")
    for i, run in enumerate(runs):
        status = run.get("http_status")
        if status is not None and status != 200:
            raise Refuse(f"{name}: run {i} returned HTTP {status}")
        if "drafts" in exp:
            seen = (run.get("tensorfold") or {}).get("drafts")
            if seen is not exp["drafts"]:
                raise Refuse(f"{name}: run {i} drafts={seen!r}, claim needs drafts={exp['drafts']}")
    return runs


# ---------------------------------------------------------------- extractors


def x_bench_median(doc, ent, exp, name):
    runs = check_bench(doc, exp, name)
    field = ent["field"]
    vals = [finite(r.get(field), f"{name}: run {i} {field}") for i, r in enumerate(runs)]
    return num(statistics.median(vals)), spread_text(vals)


def x_bench_kept(doc, ent, exp, name):
    runs = check_bench(doc, exp, name)
    vals = []
    for i, r in enumerate(runs):
        spec = r.get("speculative") or (r.get("tensorfold") or {}).get("speculative")
        if not isinstance(spec, dict):
            raise Refuse(f"{name}: run {i} has no speculative counts")
        rounds = finite(spec.get("rounds"), f"{name}: run {i} speculative.rounds")
        accepted = finite(spec.get("accepted"), f"{name}: run {i} speculative.accepted")
        if rounds <= 0:
            raise Refuse(f"{name}: run {i} speculative.rounds is {rounds}")
        vals.append(accepted / rounds)
    return num(statistics.median(vals)), spread_text(vals)


DRAFT_PROMPTS = {
    "prose_default": "prose, thinking on",
    "prose_nothink": "prose, thinking off",
    "code_default": "code, thinking on",
}


def x_draft_same(doc, ent, exp, name):
    """Per prompt: tokens that differ, drafts on against off, and the first differing position."""
    prompts = doc.get("prompts")
    if not isinstance(prompts, dict) or not prompts:
        raise Refuse(f"{name}: no compared prompts")
    want = exp.get("pairs")
    if want and len(prompts) != want:
        raise Refuse(f"{name}: {len(prompts)} prompts compared, claim needs {want}")
    parts, rows = [], []
    for key in sorted(prompts, key=lambda k: list(DRAFT_PROMPTS).index(k) if k in DRAFT_PROMPTS else 99):
        row = prompts[key]
        if not isinstance(row, dict):
            raise Refuse(f"{name}: prompt {key} is not an object")
        diff = int(finite(row.get("diff_tokens"), f"{name}: prompt {key} diff_tokens"))
        total = int(finite(row.get("total_tokens"), f"{name}: prompt {key} total_tokens"))
        first = row.get("first_diff_pos")
        if (diff == 0) != (first is None):
            raise Refuse(f"{name}: prompt {key} has diff_tokens {diff} but first_diff_pos {first!r}")
        if first is not None:
            first = int(finite(first, f"{name}: prompt {key} first_diff_pos"))
        label = DRAFT_PROMPTS.get(key, key)
        parts.append(f"{label}: {diff} of {total} differ" + (f", first at {first}" if first is not None else ""))
        rows.append([label, f"{diff} of {total}", str(first) if first is not None else "none"])
    tbl = table(["Prompt", "Tokens that differ", "First position that differs"], rows)
    return "; ".join(parts), ent.get("note", "first run of each prompt, drafts on against off, same build"), tbl


PROMPT_LABEL = dict(DRAFT_PROMPTS, pp4k_default="4K prompt, thinking on")


def prompt_key(rel: str) -> str:
    """prose_default from exl3-4k-rs/4b6b-4k-rs_prose_default.json (the known prompt the name ends with)."""
    stem = Path(rel).stem
    return next((k for k in PROMPT_LABEL if stem.endswith(k)), stem)


def load_more(ent, key="more"):
    """The extra files an entry names, relative to --outputs, as (rel, doc); a missing one refuses."""
    out = []
    for rel in ent.get(key, []):
        rel = rel.replace("{pick}", ent["_pick"] or "")
        path = ent["_outputs"] / rel
        if not path.is_file():
            raise Refuse(f"file missing: {rel} (the claim needs every listed prompt)")
        doc = load_json(path)
        if not isinstance(doc, dict):
            raise Refuse(f"{path.name} is not a JSON object")
        out.append((rel, doc))
    return out


def first_token_diff(a: list, b: list) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def x_repeats_equal(doc, ent, exp, name):
    """Per prompt: are all runs token-equal (token_sha256, else token_ids) and offered-equal."""
    files = [(ent["_rel"], doc)] + load_more(ent)
    parts, rows, equal = [], [], 0
    for rel, d in files:
        runs = check_bench(d, exp, Path(rel).name)
        shas = [r.get("token_sha256") or (r.get("tensorfold") or {}).get("token_sha256") for r in runs]
        if any(not s for s in shas):
            raise Refuse(f"{Path(rel).name}: a run has no token_sha256")
        offered = [(r.get("speculative") or {}).get("offered") for r in runs]
        offered_txt = ", ".join(str(o) for o in offered)
        label = PROMPT_LABEL.get(prompt_key(rel), prompt_key(rel))
        if len(set(shas)) == 1:
            equal += 1
            parts.append(f"{label}: {len(runs)} of {len(runs)} runs equal" + ("" if len(set(offered)) == 1 else f" (offered {offered})"))
            rows.append([label, f"same tokens on {len(runs)} of {len(runs)}", "none", offered_txt])
        else:
            ids = [r.get("token_ids") or [] for r in runs]
            firsts = [first_token_diff(ids[0], x) for x in ids[1:]]
            pos = min((f for f in firsts if f is not None), default=None)
            parts.append(f"{label}: runs differ" + (f", first at {pos}" if pos is not None else "") + f" (offered {offered})")
            rows.append([label, "differ", str(pos) if pos is not None else "unknown", offered_txt])
    tbl = table(["Prompt", "Repeats", "First position that differs", "Drafts offered, per run"], rows)
    return f"{equal} of {len(files)} prompts", "; ".join(parts), tbl


def x_on_off_median(doc, ent, exp, name):
    """Median tg_tps, option on (the entry's file and 'more') against off ('off' files), per prompt."""
    on = [(ent["_rel"], doc)] + load_more(ent)
    off = load_more(ent, "off")
    if len(on) != len(off):
        raise Refuse(f"{name}: {len(on)} files with the option on, {len(off)} off")
    parts, rows = [], []
    for (rel_on, d_on), (rel_off, d_off) in zip(on, off):
        k_on, k_off = prompt_key(rel_on), prompt_key(rel_off)
        if k_on != k_off:
            raise Refuse(f"{name}: prompt {k_on} paired with {k_off}")
        v_on = [finite(r.get("tg_tps"), f"{Path(rel_on).name}: tg_tps") for r in check_bench(d_on, exp, Path(rel_on).name)]
        v_off = [finite(r.get("tg_tps"), f"{Path(rel_off).name}: tg_tps") for r in check_bench(d_off, exp, Path(rel_off).name)]
        m_on, m_off = statistics.median(v_on), statistics.median(v_off)
        pct = (m_on - m_off) / m_off * 100 if m_off else float("nan")
        parts.append(f"{PROMPT_LABEL.get(k_on, k_on)}: {num(m_off)} off, {num(m_on)} on ({pct:+.2f}%)")
        rows.append([PROMPT_LABEL.get(k_on, k_on), r2(m_off), r2(m_on), f"{pct:+.2f}%",
                     f"{r2(min(v_on))} to {r2(max(v_on))}"])
    tbl = table(["Prompt", "Off (tok/s)", "On (tok/s)", "Off to on", "On, slowest to fastest run"], rows)
    return "; ".join(parts), f"median of {exp.get('runs', 3)} or more runs each, drafted, same build and row-cost table", tbl


def x_cutoff_summary(doc, ent, exp, name):
    """A b3_7_cutoff_report.py summary: margin, both boot records, per prompt off and on medians and token diffs."""
    want = exp.get("margin")
    if want is not None and float(doc.get("margin", "nan")) != float(want):
        raise Refuse(f"{name}: margin is {doc.get('margin')!r}, claim needs {want}")
    for side in ("off", "on"):
        meta = doc.get(f"{side}_boot_meta")
        if not isinstance(meta, dict):
            raise Refuse(f"{name}: no {side}_boot_meta")
        check_boot_meta(meta, exp, f"{name} {side}_boot_meta")
    rows = doc.get("rows")
    if not isinstance(rows, list) or not rows:
        raise Refuse(f"{name}: no rows")
    parts, trows = [], []
    for row in rows:
        key = row.get("key")
        vals = {}
        for side in ("off", "on"):
            s = row.get(side) or {}
            if exp.get("runs") and int(finite(s.get("runs"), f"{name}: {key} {side} runs")) < exp["runs"]:
                raise Refuse(f"{name}: {key} {side} has {s.get('runs')} runs, claim needs {exp['runs']}")
            vals[side] = finite(s.get("tg_tps_med"), f"{name}: {key} {side} tg_tps_med")
        pct = (vals["on"] - vals["off"]) / vals["off"] * 100
        parts.append(f"{DRAFT_PROMPTS.get(key, key)} {num(vals['off'])} off, {num(vals['on'])} on ({pct:+.2f}%)")
        dd = row.get("off_vs_on_diff") or {}
        diff_cell = "not compared"
        if "diff_tokens" in dd:
            first = dd.get("first_diff_pos")
            diff_cell = f"{dd['diff_tokens']} of {dd.get('total_tokens')}" + (f", from {first}" if first is not None else "")
        trows.append([DRAFT_PROMPTS.get(key, key), r2(vals["off"]), r2(vals["on"]), f"{pct:+.2f}%", diff_cell])
    # the tokens that differ, on against off, are in the table (they are not repeated in the spread)
    spread = f"median of {exp.get('runs', 3)} runs each, drafted, both boots the same build"
    if doc.get("engine"):  # an engine commit, when the report names one (never bin_src: it is a machine path)
        spread += f"; engine {doc['engine']}"
    tbl = table(["Prompt", "Cutoff off (tok/s)", "Cutoff on (tok/s)", "Off to on", "Tokens that differ, on against off"], trows)
    return f"cutoff {doc.get('margin')}: " + "; ".join(parts) + " tok/s", spread, tbl


def x_grid_top1(doc, ent, exp, name):
    """drafted_grid_agree.py score: the drafted path's top-1 against the reference, plain path beside it."""
    if "teacher_kind" in exp and doc.get("teacher_kind") != exp["teacher_kind"]:
        raise Refuse(f"{name}: scored against {doc.get('teacher_kind')!r}, claim needs {exp['teacher_kind']!r}")
    pos = int(finite(doc.get("positions_scored"), f"{name}: positions_scored"))
    if pos < exp.get("positions_min", 0):
        raise Refuse(f"{name}: {pos} positions scored, claim needs at least {exp['positions_min']}")
    d = finite(doc.get("draft_pct"), f"{name}: draft_pct")
    p = finite(doc.get("plain_pct"), f"{name}: plain_pct")
    eq = finite(doc.get("draft_equals_plain_pct"), f"{name}: draft_equals_plain_pct")
    same = doc.get("draft_equals_plain_count")
    same_txt = f" ({int(finite(same, f'{name}: draft_equals_plain_count'))} of {pos})" if same is not None else ""
    gap = doc.get("max_margin_at_draft_vs_plain_diff")
    gap_txt = ""
    if gap is not None:
        gap_txt = f"; largest reference top-two gap where they differ {finite(gap, f'{name}: max_margin_at_draft_vs_plain_diff'):.3f}"
    counts = ""
    d_cell, p_cell = f"{r2(d)}%", f"{r2(p)}%"
    if doc.get("draft_matched") is not None and doc.get("plain_matched") is not None:
        dm = int(finite(doc["draft_matched"], f"{name}: draft_matched"))
        pm = int(finite(doc["plain_matched"], f"{name}: plain_matched"))
        counts = f"; {dm} of {pos} drafted, {pm} of {pos} plain"
        d_cell, p_cell = f"{d_cell} ({dm} of {pos})", f"{p_cell} ({pm} of {pos})"
    rows = [["Top-1 agreement, drafted path", d_cell],
            ["Top-1 agreement, plain path, same positions", p_cell],
            ["Drafted minus plain", f"{d - p:+.2f} points"],
            ["Drafted pick equals plain pick", (f"{int(same)} of {pos} ({r2(eq)}%)" if same is not None else f"{r2(eq)}%")]]
    if gap is not None:
        rows.append(["Largest reference top-two gap where they differ", f"{float(gap):.3f}"])
    tbl = table(["Measure", "Value"], rows)  # reader text; the run label stays in the spread only
    return num(d), (f"plain path {num(p)}% on the same {pos} positions ({d - p:+.2f} points){counts}; "
                    f"drafted pick equals plain at {num(eq)}% of positions{same_txt}{gap_txt}; label {doc.get('label')}"), tbl


def check_score_label(doc, exp, name):
    if "label" in exp and doc.get("label") != exp["label"]:
        raise Refuse(f"{name}: score label is {doc.get('label')!r}, claim needs {exp['label']!r}")
    if "paired_vs" in exp and doc.get("paired_vs") != exp["paired_vs"]:
        raise Refuse(f"{name}: paired_vs is {doc.get('paired_vs')!r}, claim needs {exp['paired_vs']!r}")
    if "teacher_kind" in exp and doc.get("teacher_kind") != exp["teacher_kind"]:
        raise Refuse(
            f"{name}: scored against {doc.get('teacher_kind')!r}, but the kit's prefix set gives {exp['teacher_kind']!r}"
        )
    if "positions_min" in exp:
        pos = finite(doc.get("positions_scored"), f"{name}: positions_scored")
        if pos < exp["positions_min"]:
            raise Refuse(f"{name}: {int(pos)} positions scored, claim needs at least {exp['positions_min']}")


def x_teacher_top1(doc, ent, exp, name):
    check_score_label(doc, exp, name)
    overall = finite(doc.get("overall_pct"), f"{name}: overall_pct")
    prose = finite(doc.get("prose_pct"), f"{name}: prose_pct")
    code = finite(doc.get("code_pct"), f"{name}: code_pct")
    pos = int(finite(doc.get("positions_scored"), f"{name}: positions_scored"))
    return num(overall), f"prose {num(prose)}%, code {num(code)}%; {pos} positions"


def x_teacher_delta(doc, ent, exp, name):
    check_score_label(doc, exp, name)
    delta = finite(doc.get("delta_pts_vs_lossless"), f"{name}: delta_pts_vs_lossless")
    lo = int(finite(doc.get("lossless_only"), f"{name}: lossless_only"))
    co = int(finite(doc.get("cand_only"), f"{name}: cand_only"))
    return num(delta), f"{lo} position{'' if lo == 1 else 's'} only the lossless build gets right, {co} only the full quant"


def check_niah(doc, exp, name):
    if "thinking" in exp:
        want = THINKING[exp["thinking"]]
        if doc.get("reasoning_effort") != want:
            raise Refuse(f"{name}: reasoning_effort is {doc.get('reasoning_effort')!r}, claim needs {want!r}")
    if "context" in exp:
        seen = get(doc, "meta.context_window", name)
        if seen != exp["context"]:
            raise Refuse(f"{name}: needle prompt is for a {seen!r}-token window, claim needs {exp['context']}")
    status = get(doc, "response.http_status", name)
    if status != 200:
        raise Refuse(f"{name}: request returned HTTP {status}")


def x_niah_field(doc, ent, exp, name):
    check_niah(doc, exp, name)
    field = ent["field"]
    return num(finite(get(doc, f"response.{field}", name), f"{name}: response.{field}")), None


def x_niah_pass(doc, ent, exp, name):
    check_niah(doc, exp, name)
    verdict = doc.get("verdict")
    if verdict not in ("PASS", "FAIL"):
        raise Refuse(f"{name}: verdict is {verdict!r}")
    depth = (doc.get("meta") or {}).get("requested_depth")
    return verdict.lower(), (f"one needle at depth {depth}" if depth is not None else None)


def x_receipt_value(doc, ent, exp, name):
    want = exp.get("claim_id")
    if want and doc.get("claim_id") != want:
        raise Refuse(f"{name}: claim_id is {doc.get('claim_id')!r}, expected {want!r}")
    if doc.get("status") != "measured":
        raise Refuse(f"{name}: status is {doc.get('status')!r}, not measured")
    if ent.get("field") == "verdict":
        verdict = doc.get("verdict")
        if verdict not in ("PASS", "FAIL"):
            raise Refuse(f"{name}: verdict is {verdict!r}")
        return verdict.lower(), None
    return num(finite(doc.get("value"), f"{name}: value")), None


def rank_values(doc, field, name):
    ranks = doc.get("ranks")
    if not isinstance(ranks, dict) or not ranks:
        raise Refuse(f"{name}: no ranks")
    out = {}
    for r, info in sorted(ranks.items(), key=lambda kv: int(kv[0])):
        out[r] = finite(get(info, field, f"{name} rank {r}"), f"{name}: rank {r} {field}")
    return out


def drafter_note(doc):
    seen = sorted({str(i.get("drafter_loaded")) for i in (doc.get("ranks") or {}).values()})
    return "drafter " + "/".join(seen)


def x_ranks_max(doc, ent, exp, name):
    vals = rank_values(doc, ent["field"], name)
    listing = ", ".join(f"r{r} {num(v)}" for r, v in vals.items())
    return num(max(vals.values())), f"{listing}; {drafter_note(doc)}"


def x_ranks_min(doc, ent, exp, name):
    vals = rank_values(doc, ent["field"], name)
    listing = ", ".join(f"r{r} {num(v)}" for r, v in vals.items())
    return num(min(vals.values())), f"{listing}; {drafter_note(doc)}"


# RigMark (github.com/alexellis/rigmark), a third-party benchmark. Only its whole standard suite fills: the
# settings its report.py calls DEFAULT SETTINGS (SUITE_DEFAULTS, with the 8,192-token decode cap of protocol
# 1.3.0), no error, a clean source tree, and every completion gate passed (its card says DO NOT HEADLINE
# otherwise). A quick or partial run is refused. The claims name the thinking-off run (label, comparison id and
# reasoning_effort no_think in card/fill-map.json): with thinking on, a code reply ran to the 8,192-token cap
# before its answer (FINDINGS.md, finding 6).
RIGMARK_SUITE = {
    "runs": 5, "decode_tokens": 8192, "temperature": 0.0, "top_p": 1.0, "seed": 20260905,
    "prefill_depths": [8192, 32768, 65536], "prefill_runs": 3, "concurrency": [1, 2, 4],
    "concurrency_runs": 3, "concurrency_tokens": 256, "concurrency_workload": "code",
}


def check_rigmark(doc, exp, name):
    """Refuse unless the receipt is the whole standard suite of the run the claim names. Every reason is listed."""
    if "error" in doc:
        raise Refuse(f"{name}: the run stopped with an error, so the suite is not whole ({str(doc['error'])[:80]!r})")
    run = doc.get("run") if isinstance(doc.get("run"), dict) else {}
    if not run.get("finished_at"):
        raise Refuse(f"{name}: no run.finished_at, so the suite did not finish")
    bad = []
    proto = doc.get("protocol") if isinstance(doc.get("protocol"), dict) else {}
    if proto.get("version") != exp.get("protocol"):
        bad.append(f"protocol {proto.get('version')!r}, claim needs {exp.get('protocol')!r}")
    if proto.get("repository_revision") != exp.get("revision"):
        bad.append(f"RigMark revision {proto.get('repository_revision')!r}, claim needs {exp.get('revision')!r}")
    if proto.get("repository_dirty") is not False:
        bad.append(f"RigMark source tree not recorded clean (repository_dirty {proto.get('repository_dirty')!r})")
    for key in ("label", "comparison_id"):
        if key in exp and run.get(key) != exp[key]:
            bad.append(f"{key} {run.get(key)!r}, claim needs {exp[key]!r}")
    settings = doc.get("settings") if isinstance(doc.get("settings"), dict) else {}
    changed = [k for k, v in RIGMARK_SUITE.items() if settings.get(k) != v]
    if changed:
        bad.append(f"not the standard suite (RigMark CUSTOM SETTINGS: {', '.join(changed)})")
    if "extra_body" in exp and settings.get("extra_body") != exp["extra_body"]:
        bad.append(f"extra_body {settings.get('extra_body')!r}, claim needs {exp['extra_body']!r}")
    app = run.get("appliance") if isinstance(run.get("appliance"), dict) else {}
    if "gpu_clocks" in exp and app.get("gpu_clocks") != exp["gpu_clocks"]:
        bad.append(f"gpu_clocks {app.get('gpu_clocks')!r}, claim needs {exp['gpu_clocks']!r}")
    if "context" in exp and app.get("context_limit") != exp["context"]:
        bad.append(f"context_limit {app.get('context_limit')!r}, claim needs {exp['context']!r}")
    if "trunk" in exp and f"{exp['trunk']} trunk" not in str(app.get("quantisation", "")):
        bad.append(f"quantisation does not say '{exp['trunk']} trunk'")
    if "drafter" in exp and not str(app.get("drafter", "")).startswith(str(exp["drafter"])):
        bad.append(f"drafter {app.get('drafter')!r}, claim needs {exp['drafter']!r}")
    passed = total = 0
    for workload in ("code", "prose", "structured"):
        gate = get(doc, f"decode.{workload}.completion_gate", name)
        passed += int(finite(gate.get("passed") if isinstance(gate, dict) else None, f"{name}: {workload} gate passed"))
        total += int(finite(gate.get("total") if isinstance(gate, dict) else None, f"{name}: {workload} gate total"))
    if total == 0 or passed != total:
        bad.append(f"{passed} of {total} completion gates passed (RigMark: DO NOT HEADLINE)")
    if bad:
        raise Refuse(f"{name}: " + "; ".join(bad))


def x_rigmark(doc, ent, exp, name):
    """One median from a RigMark receipt; spread is RigMark's own min and max over its runs or rounds."""
    check_rigmark(doc, exp, name)
    value = finite(get(doc, ent["field"], name), f"{name}: {ent['field']}")
    rows = get(doc, ent["runs"], name)
    if not isinstance(rows, list) or not rows:
        raise Refuse(f"{name}: {ent['runs']} holds no runs")
    if ent.get("cold"):  # RigMark itself prints "Cold cache UNVERIFIED" unless every cold request saw 0 cached tokens
        cached = [r.get("cached_prompt_tokens") if isinstance(r, dict) else None for r in rows]
        if any(c != 0 for c in cached):
            raise Refuse(f"{name}: cold requests reported cached prompt tokens {cached}; cold is not verified")
    span = get(doc, ent["range"], name)
    lo = finite(span.get("minimum") if isinstance(span, dict) else None, f"{name}: {ent['range']}.minimum")
    hi = finite(span.get("maximum") if isinstance(span, dict) else None, f"{name}: {ent['range']}.maximum")
    spread = f"{num(lo)} to {num(hi)} over {len(rows)} {ent.get('count_word', 'runs')}"
    if ent.get("gate"):
        gate = get(doc, ent["gate"], name)
        spread += f"; completion gate {gate.get('passed')} of {gate.get('total')}"
    return num(value), spread


EXTRACT = {
    "rigmark": x_rigmark,
    "bench_median": x_bench_median,
    "bench_kept": x_bench_kept,
    "draft_same": x_draft_same,
    "teacher_top1": x_teacher_top1,
    "teacher_delta": x_teacher_delta,
    "niah_field": x_niah_field,
    "niah_pass": x_niah_pass,
    "receipt_value": x_receipt_value,
    "ranks_max": x_ranks_max,
    "ranks_min": x_ranks_min,
    "repeats_equal": x_repeats_equal,
    "on_off_median": x_on_off_median,
    "cutoff_summary": x_cutoff_summary,
    "grid_top1": x_grid_top1,
}


# ---------------------------------------------------------------- main loop


def fill_pick(ent, claim, outputs, pick, args):
    """The trunk width is a decision; the width table is its receipt."""
    if pick is None:
        raise Skip("no width given (pass --pick 4b, 4b6b or 6b)")
    if pick not in WIDTHS:
        raise Refuse(f"width {pick!r} is not one of {', '.join(WIDTHS)}")
    src = outputs / ent["file"]
    if not src.is_file():
        raise Skip(f"file missing: {ent['file']}")
    table = load_json(src)
    rows = table.get("rows") if isinstance(table, dict) else None
    trunks = [r.get("trunk") for r in rows or [] if isinstance(r, dict)]
    if pick not in trunks:
        raise Refuse(f"{src.name} has no row for width {pick!r} (rows: {trunks})")
    return pick, None, src, []


def run_entry(ent, claim, outputs, pick, pick_drafter, kit):
    exp = {k: resolve(v, pick, pick_drafter, kit) for k, v in (ent.get("expect") or {}).items()}
    if exp.get("teacher_kind", "") is None:
        raw = str((ent.get("expect") or {}).get("teacher_kind", ""))
        name = raw.partition(":")[2] or "familiar"
        raise Skip(f"eval/prefixes/{name}/manifest.json is not in the kit yet (agreement is scored on that set)")
    if any(v is None for v in exp.values()) or ("{pick}" in ent["file"] and pick is None):
        raise Skip("trunk width not picked yet (build.full.trunk pending; pass --pick)")
    rel = ent["file"].replace("{pick}", pick or "")
    src = outputs / rel
    if not src.is_file():
        raise Skip(f"file missing: {rel}")
    doc = load_json(src)
    if not isinstance(doc, dict):
        raise Refuse(f"{src.name} is not a JSON object")
    extras = []
    if ent.get("meta"):
        meta_rel = ent["meta"].replace("{pick}", pick or "")
        meta_path = outputs / meta_rel
        if not meta_path.is_file():
            raise Refuse(f"boot record missing: {meta_rel} (labels cannot be checked)")
        meta = load_json(meta_path)
        if not isinstance(meta, dict):
            raise Refuse(f"{meta_path.name} is not a JSON object")
        if ent.get("meta_kind") == "summary":
            check_summary(meta, exp, meta_path.name)
        else:
            check_boot_meta(meta, exp, meta_path.name)
        if ent.get("meta_receipt"):
            extras.append((meta_path, ent["meta_receipt"]))
    if ent.get("meta_off"):  # the other side of an on-against-off claim has its own boot record
        off_rel = ent["meta_off"].replace("{pick}", pick or "")
        off_path = outputs / off_rel
        if not off_path.is_file():
            raise Refuse(f"boot record missing: {off_rel} (labels cannot be checked)")
        off_exp = dict(exp)
        if "repeat_stable" in exp:  # an on-against-off claim: the off boot must say the opposite
            off_exp["repeat_stable"] = not exp["repeat_stable"]
        off_meta = load_json(off_path)
        if not isinstance(off_meta, dict):
            raise Refuse(f"{off_path.name} is not a JSON object")
        check_boot_meta(off_meta, off_exp, off_path.name)
    ent = dict(ent, _outputs=outputs, _pick=pick, _rel=rel)
    out = EXTRACT[ent["kind"]](doc, ent, exp, src.name)
    value, spread = out[0], out[1]
    tbl = out[2] if len(out) > 2 else None  # per-prompt kinds also give a small table
    # every extra file the claim read is kept beside its receipt
    receipt_dir = str(Path(claim.get("receipt") or "receipts/x").parent)
    for key in ("more", "off"):
        for more_rel in ent.get(key, []):
            more_rel = more_rel.replace("{pick}", pick or "")
            extras.append((outputs / more_rel, f"{receipt_dir}/{Path(more_rel).name}"))
    return value, spread, src, extras, tbl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", type=Path, required=True, help="folder of boot outputs laid out as card/fill-map.json describes")
    ap.add_argument("--map", type=Path, default=ROOT / "card" / "fill-map.json")
    ap.add_argument("--claims", type=Path, default=ROOT / "card" / "claims.json")
    ap.add_argument("--kit", type=Path, default=ROOT, help="kit root that receipts/ paths are relative to")
    ap.add_argument("--pick", choices=WIDTHS, help="trunk width of the full quant, when build.full.trunk is still pending")
    ap.add_argument("--only", action="append", default=[], help="fill only claim ids starting with this (repeatable)")
    ap.add_argument("--force", action="store_true", help="overwrite claims that are already measured")
    ap.add_argument("--dry-run", action="store_true", help="print what would happen; write nothing")
    args = ap.parse_args()

    fmap = json.loads(args.map.read_text())
    ledger = json.loads(args.claims.read_text())
    by_id = {c["id"]: c for c in ledger["claims"]}
    pick_drafter = fmap.get("pick_drafter", {})

    trunk_claim = by_id.get("build.full.trunk", {})
    pick = None
    if trunk_claim.get("status") == "measured" and trunk_claim.get("value") in WIDTHS:
        pick = trunk_claim["value"]
        if args.pick and args.pick != pick and not args.force:
            print(f"refused  build.full.trunk: already {pick!r}; --pick {args.pick} needs --force")
            return 2
    if args.pick:
        pick = args.pick

    counts = {"filled": 0, "skipped": 0, "refused": 0}
    copies: list[tuple[Path, Path]] = []
    for ent in fmap["claims"]:
        cid = ent["id"]
        if args.only and not any(cid.startswith(p) for p in args.only):
            continue
        claim = by_id.get(cid)
        if claim is None:
            print(f"refused  {cid}: not in claims.json")
            counts["refused"] += 1
            continue
        if claim.get("status") == "measured" and not args.force:
            print(f"skipped  {cid}: already measured ({claim.get('value')}); --force to overwrite")
            counts["skipped"] += 1
            continue
        try:
            tbl = None
            if ent["kind"] == "pick":
                value, spread, src, extras = fill_pick(ent, claim, args.outputs, pick, args)
            else:
                value, spread, src, extras, tbl = run_entry(ent, claim, args.outputs, pick, pick_drafter, args.kit)
        except Skip as exc:
            print(f"skipped  {cid}: {exc}")
            counts["skipped"] += 1
            continue
        except Refuse as exc:
            print(f"refused  {cid}: {exc}")
            counts["refused"] += 1
            continue
        receipt = claim.get("receipt")
        if not receipt or receipt.startswith("S:"):
            print(f"refused  {cid}: claim has no kit receipt path")
            counts["refused"] += 1
            continue
        shown = f"{value}" + (f" ({spread})" if spread else "")
        forced = ""
        if claim.get("status") == "measured":
            forced = f" [forced over {claim.get('value')!r}]"
        print(f"filled   {cid}: {shown} <- {src.relative_to(args.outputs)} -> {receipt}{forced}")
        counts["filled"] += 1
        claim["value"] = value
        claim["spread"] = spread
        claim["status"] = "measured"
        claim.pop("status_note", None)  # an interrupted claim's reason ends when a rerun fills it
        if tbl:
            claim["table"] = tbl
        else:
            claim.pop("table", None)
        copies.append((src, args.kit / receipt))
        for extra_src, extra_dest in extras:
            copies.append((extra_src, args.kit / extra_dest))

    print(
        f"total    {'would fill' if args.dry_run else 'filled'} {counts['filled']}, "
        f"skipped {counts['skipped']}, refused {counts['refused']}"
    )
    if pick and by_id.get("build.full.trunk", {}).get("value") == pick:
        env = args.kit / "launch" / "ring.env.example"
        if env.is_file() and f'TF_HY4_TRUNK="{pick}"' not in env.read_text():
            print(f'todo     set TF_HY4_TRUNK="{pick}" in launch/ring.env.example (tools/check_kit.py checks it)')
    if args.dry_run:
        return 1 if counts["refused"] else 0

    done = set()
    for src, dest in copies:
        if dest in done:
            continue
        done.add(dest)
        notes = redacted_copy(src, dest)  # also redacts in place when src is already the kit file
        if notes:
            print(f"redacted {dest.relative_to(args.kit)}: {', '.join(notes)}")
    args.claims.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n")
    rerender(args.kit, args.claims)
    return 1 if counts["refused"] else 0


def rerender(kit: Path, claims: Path) -> None:
    """Rewrite the public pages from their templates with the new values (what tools/render.py --all does)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # run with -I, this folder is off the path
    try:
        import render
    finally:
        sys.path.pop(0)
    if not all((kit / tpl).is_file() for tpl in render.TEMPLATES):
        print(f"render   skipped: {kit} has no card/templates/; run python3 tools/render.py --all in the kit")
        return
    pages = render.render_templates(kit, render.load_claims(claims))
    stale = render.drifted(kit, pages)
    for pub in stale:
        (kit / pub).write_text(pages[pub], encoding="utf-8")
    print(f"render   rewrote {len(stale)} of {len(pages)} pages: {', '.join(stale) or 'none'}")


if __name__ == "__main__":
    sys.exit(main())
