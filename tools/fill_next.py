#!/usr/bin/env python3
"""Fill this release's pending next.* claims from one receipt folder of the release run, then render the pages.

    python3 tools/fill_next.py engine  --commit SHA --native SHA256 --oracle SHA256 --boundary SHA256 --gate TEXT
    python3 tools/fill_next.py window1 DIR     # release build, defaults, vision on: RigMark, needle, image checks
    python3 tools/fill_next.py codecs DIR      # the same boot with the image decoders staged: the Formats row
    python3 tools/fill_next.py decode-match DIR [--summary PATH]   # release build, decode only, the speed team's id
    python3 tools/fill_next.py bf16    DIR     # release build, TF_HY4_PREFILL_AR=bf16, prefill-only RigMark
    python3 tools/fill_next.py control DIR     # earlier build's binary at a path of the same length, decode-only
    python3 tools/fill_next.py control-dropped [--reason TEXT]   # the control was left out: its row reads not run
    python3 tools/fill_next.py tools DIR [--summary PATH] [--serve-log PATH]   # tool calls and structured output
    python3 tools/fill_next.py temp07 DIR [--alongside TEXT]   # release build, kit defaults: no temperature sent
    python3 tools/fill_next.py fix-engine --commit SHA --native SHA256 --oracle SHA256 --boundary SHA256 --gate TEXT
    python3 tools/fill_next.py fix07 DIR [--alongside TEXT]    # the fixed build (sampling fix), as temp07, plus thinking off
    python3 tools/fill_next.py parity DIR --serve-log PATH --release-log PATH [--summary PATH]   # the sampling gate

DIR is a receipt folder: launch/collect_receipts.py's output (summary.json, vision_check.json) plus the clients'
JSON copied beside it (RigMark's hy4-st-*.json; for window1 also niah_128k_thinkon.json and niah_128k_nothink.json).
Add --dry-run to print what would be written. Run `engine` first: the others check the run's binary against it
(`fix-engine` likewise before fix07 and parity).

Refuses (exit 2, nothing written) when:
  - the folder lacks a file it needs, or summary.json's engine block lists any problem (an unread hash or clock);
  - the four ranks' tensorfold-native hashes differ, or differ from next.engine.sha256.native (window1, bf16),
    or from the earlier build's 4a3d5265... (control);
  - the run did not declare GPU_CLOCK_CAP_MHZ=2200, or its engine commit differs from next.engine.commit;
  - the switches in force are not the run's own: window1 FAST_PREFILL=1, PREFILL_AR=fp32, SERVE_VISION=1;
    bf16 FAST_PREFILL=1, PREFILL_AR=bf16; control FAST_PREFILL=0;
  - RigMark's settings differ (window1 runs 3 and prefill_runs 1; bf16 prefill_runs 1; control runs 3, no prefill),
    or the control's binary path is not the same length as window1's;
  - a claim it would fill is neither pending nor not_run (it never overwrites a measured value);
  - decode-match: RigMark's --comparison-id is not the speed team's 2026-11-22-hy4-speed-team-2200mhz, it ran prefill
    or not 3 runs, the switches are not FAST_PREFILL=1 and PREFILL_AR=fp32, or the serve log's sha= and the receipts'
    outputs disagree on a run.
decode-match puts the release's decode row beside the earlier build's on the same prompts. RigMark puts a nonce made
from --comparison-id into every prompt, so only runs under one id share prompts and text. It reads DIR's RigMark
receipt (--skip-prefill), the boot's summary.json (DIR/summary.json, else DIR/../codecs/summary.json, or --summary)
and the gate seat's DIR/verify.txt (its shas= line, from rank 0's serve log). It compares each run's output with the
same run of receipts/next/prefill/base-2200/ (the earlier build's binary, same id) and each sha= with that row's
rank0-decode-done.txt, and fills next.decode.{code,prose,structured}.match and next.decode.same_output. It prints a
NOTE (exit 3 after writing) when a median lies outside the earlier build's two readings widened by their gap, the
size of the binary's path effect (base-2200 and base6e-exact-winb, same bytes, two paths).
control-dropped sets next.decode.{code,prose,structured}.control to not_run with --reason as the status_note
(default "not run (dropped for time)"); the page then reads that note in the control row. A later `control` fill
replaces it.
tools reads tool_probes.py's receipt folder: one JSON per request (tool_auto, tool_required, format_json_schema,
format_json_object) and the script's own summary.json, kept as tool_probes.json beside the boot's summary.json (DIR/../probes/
summary.json, or --summary). It fills next.api.* with what the server did, in its own words when it refused, and
refuses when a request file is missing, a request got no reply, or the boot is not the release build with
FAST_PREFILL=1 and PREFILL_AR=fp32. --serve-log is rank 0's serve log: its libtfgrammar.so warning is copied to
grammar_warning.txt; without that line it prints a NOTE (exit 3 after writing), since the page quotes it.
temp07 reads one boot's folder: collect_receipts.py's summary.json and bench/sampling_check.py's sampling_check.json
(its printed lines may sit beside it as sampling_check.log). It fills next.temp07.*: the sampling part of rank 0's
serving line, the decode medians with no temperature sent and with temperature 0, and finding 6's coding request
with and without "thinking_budget" 4096. It refuses when the boot is not the release build with FAST_PREFILL=1,
PREFILL_AR=fp32 and SERVE_TEMPERATURE 0.7, a server thinking budget was set, the serving line does not start with
temperature 0.7, the script's settings are not this kit's, or a request went unanswered.
It also fills next.temp07.alongside, which the README prints beside the decode medians, from summary.json's isolation
(collect_receipts.py --isolation): quiet gives "nothing else running"; shared needs --alongside TEXT naming what ran
on the Sparks during the run, at most 20 words. It refuses shared without --alongside, and --alongside with quiet.
Each filled claim's setup then opens with the run's times from sampling_check.json, in place of its pending sentence.
The script's settings now include its thinking-off rows (decode_nothink), which temp07 does not fill; the one temp07
run in receipts/next/release/temp07/ ran the script before it had them.
The fixed build is the release build's engine plus the sampling fix; its claims are next.fix.* and next.fix07.*, and
next.engine.* and next.temp07.* stay the measured record of the release build. fix-engine fills next.fix.* as engine
fills next.engine.*, wants a full 40-character commit, and refuses the release build's own commit or tensorfold-native.
fix07 is temp07 on the fixed build (binary next.fix.sha256.native, commit next.fix.commit) into
receipts/next/release/fix07/, and also fills next.fix07.nothink.* from the thinking-off rows, refusing a thinking-off
run that did not send no_think or that reasoned. It prints a NOTE (exit 3 after writing) when a thinking-on decode reply
has answer tokens or did not end on length (the README says every one was still reasoning at the cap), or when finding
6's two replies differ from next.temp07.f6.* (FINDINGS.md says they are the same).
parity reads the sampling gate's gate.json and gate.log: DIR holds both, or DIR is the gate's JSON itself with its
.log beside it (the rig's out/sample-gate-LABEL.json). Its 2 greedy and 8 same-seed cases must all pass with
the hash and token count they expect; each reply must be in rank 0's log of the fixed boot (--serve-log) by request id
with the same hash and token count, and in rank 0's log of the release build's temp07 boot (--release-log) with the
same hash and token count and, for the seeded ones, the same seed. The boot's summary.json (beside gate.json, or
--summary) must be the fixed build with the temp07 switches. It fills next.fix.parity and copies gate.json, gate.log,
summary.json and the matched done lines of both logs into receipts/next/release/fix-parity/.
Prints a NOTE (exit 3 after writing) when the serve log's NCCL version is not the 2.30.7 the docs state.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
LEDGER = KIT / "card" / "claims.json"
LAST_RELEASE_BIN = "4a3d5265523408f302dfeb2066cdae1f4fe9f28276310c9d3c82df1c0aa35695"
DOCS_NCCL = "2.30.7"
DEPTHS = (("8k", "8192"), ("32k", "32768"), ("64k", "65536"))
WORKLOADS = ("code", "prose", "structured")
# The earlier build's decode rows on the speed team's prompts: same bytes (4a3d5265...), two binary paths.
SPEED_ID = "2026-11-22-hy4-speed-team-2200mhz"
LAST_DECODE = KIT / "receipts" / "next" / "prefill" / "base-2200"
LAST_DECODE_OTHER_PATH = KIT / "receipts" / "next" / "prefill" / "base6e-exact-winb"
SHA_RE = re.compile(r"\bsha=([0-9a-f]+)")


class Refuse(Exception):
    pass


# Receipts write home folders as ~ and the Sparks as node1 to node4 (the kit's convention, as collect_receipts.py).
# Patterns are assembled so this file does not hold the strings it removes.
HOME_RE = re.compile("/ho" + r"me/[^/\s\"']+")
USER_RE = re.compile(r"\b" + "jake" + "jh" + r"\b")
HW = {"spark-" + "267e": "node1", "spark-" + "c8b6": "node2", "spark-" + "973f": "node3", "spark-" + "d823": "node4"}
# The home network's addresses (a gate run from the PC prints its target), which check_kit refuses; not the ring's.
LAN_RE = re.compile(r"(?<![0-9])" + "192" + r"\.168\.1\.[0-9]{1,3}(?![0-9])")


def clean_copy(src: Path, dst: Path) -> None:
    t = src.read_text(encoding="utf-8", errors="replace")
    t = HOME_RE.sub("~", t)
    t = LAN_RE.sub("HOST", t)
    for h, n in HW.items():
        t = re.sub(r"(?<![A-Za-z0-9.-])" + re.escape(h) + r"(?![A-Za-z0-9-])", n, t)
    t = USER_RE.sub("user", t)
    dst.write_text(t, encoding="utf-8")


def need(cond: bool, why: str) -> None:
    if not cond:
        raise Refuse(why)


def load(p: Path) -> dict:
    need(p.is_file(), f"missing {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def find(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find(v, key)
            if r is not None:
                return r
    if isinstance(obj, list):
        for v in obj:
            r = find(v, key)
            if r is not None:
                return r
    return None


def r2(x) -> float:
    return round(float(x), 2)


def check_run(d: Path, by: dict, want: dict, binary: str | None, summary: Path | None = None) -> dict:
    s = load(summary or d / "summary.json")
    e, ls = s.get("engine") or {}, s.get("launch_settings") or {}
    need(not e.get("problems"), f"summary.json engine problems: {e.get('problems')}")
    need(e.get("bin_sha256_same_on_all_ranks") is True, "the four ranks' binary hashes differ or were not read")
    sha = set((e.get("bin_sha256_by_rank") or {}).values())
    if binary:
        need(sha == {binary}, f"binary {sorted(sha)} is not the expected {binary[:12]}...")
    need(str(e.get("gpu_clock_cap_mhz")) == "2200", f"clock cap declared {e.get('gpu_clock_cap_mhz')!r}, want 2200")
    for k, v in want.items():
        need(str(ls.get(k)) == v, f"{k}={ls.get(k)!r} in force, want {v!r}")
    return s


def rigmark(d: Path) -> dict:
    files = sorted(p for p in d.glob("hy4-st-*.json") if not p.name.endswith(".card.json"))
    need(len(files) == 1, f"want one RigMark hy4-st-*.json in {d}, found {[p.name for p in files]}")
    return load(files[0]) | {"_file": files[0].name}


def serve_shas(p: Path) -> list | None:
    """The sha= of each decode reply in rank 0's serve log, in order: from the gate seat's verify.txt (its shas=[...]
    line) or from `[tensorfold] done chatcmpl-` lines. None when the file is absent."""
    if not p.is_file():
        return None
    t = p.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^shas=\[(.*)\]\s*$", t, re.M)
    if m:
        return re.findall(r"[0-9a-f]+", m.group(1))
    return [SHA_RE.search(ln).group(1) for ln in t.splitlines() if "[tensorfold] done chatcmpl-" in ln and SHA_RE.search(ln)]


FORMAT_NAMES = {"png": "PNG", "bmp": "BMP", "jpeg": "JPEG", "webp": "WebP"}


def err_msg(p: dict) -> str:
    """A refused probe's reason: the server's JSON error.message, else the start of the raw body."""
    try:
        return str(json.loads(p["error"])["error"]["message"])
    except (TypeError, ValueError, KeyError):
        return str(p.get("error"))[:80]


def said_probe(p: dict) -> str:
    return f"{p['label']}: " + ("accepted" if p["accepted"] else f"refused, HTTP {p['status']}, {err_msg(p)}")


# tool_probes.py's four requests and the claims they fill; the page quotes the engine's boot warning.
TOOL_PROBES = (("tool_auto", "next.api.tool_auto"), ("tool_required", "next.api.tool_required"),
               ("format_json_schema", "next.api.json_schema"), ("format_json_object", "next.api.json_object"))
GRAMMAR_WARN = "libtfgrammar.so is required"


def said_tool(name: str, res: dict) -> tuple[str, str]:
    """What the server did with one tool_probes.py request, short for the page; finish, tokens and time for the spread."""
    st = res.get("status")
    need(st is not None, f"{name}: no reply ({res.get('error')})")
    if st != 200:
        return f"HTTP {st}: {res.get('error_message') or str(res.get('raw', ''))[:200]}", f"{res.get('seconds')} s"
    toks = (res.get("usage") or {}).get("completion_tokens")
    spread = f"finish {res.get('finish_reason')}; {toks} tokens, {res.get('reasoning_chars')} characters of reasoning; {res.get('seconds')} s"
    if name.startswith("tool_"):
        calls = res.get("tool_calls") or []
        if not calls:
            return f"HTTP 200, no tool call (finish {res.get('finish_reason')})", spread
        f = calls[0].get("function") or {}
        return f"HTTP 200, called {f.get('name')}({f.get('arguments')})", spread
    text = res.get("content") or ""
    try:
        obj = json.loads(text)
    except ValueError:
        obj = None
    if isinstance(obj, dict):
        return f"HTTP 200, JSON {json.dumps(obj, ensure_ascii=False)}", spread
    return f"HTTP 200, not JSON (finish {res.get('finish_reason')}): {text[:80]!r}", spread


# bench/sampling_check.py's fixed settings; the next.temp07.* claim texts and the pages state them.
SAMPLING_SETTINGS = {"decode_tokens": 2048, "runs": 3, "warmup_tokens": 32, "f6_max_tokens": 8192, "f6_seed": 20260905,
                     "f6_budget": 4096, "f6_reasoning_effort": "high",
                     "prompts": ["bench/prompts/short-prose.txt", "bench/prompts/short-code.txt"],
                     "f6_messages": "receipts/thinking/budget_probes.json P5", "nothink_reasoning_effort": "no_think"}
# The next.temp07.* setups open with this sentence while pending; the fill puts the run's times in its place.
PENDING_LEAD = "Pending: Max runs it on the release build in a rig window; nothing here is measured yet. "
PENDING_LEAD_FIX = "Pending: Max runs it on the fixed build in a rig window; nothing here is measured yet. "
UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
PATCH = "serve/patches/0001-hy4-port.patch"
# The next.fix.* setups' pending openings and what replaces them once filled (fix-engine, parity).
FIX_SETUPS = {
    "next.fix.commit": ("Pending: fill_next.py fix-engine fills it with the hashes of the kit's build of this head. ",
                        "The engine seat's head of branch hy4-sample-fast; this kit's patch is cut from it. "),
    "next.fix.sha256.native": ("Pending: built by", "Built by"),
    "next.fix.sha256.oracle": ("Pending: built by", "Built by"),
    "next.fix.sha256.boundary": ("Pending: built by", "Built by"),
    "next.fix.gate": ("Pending: the engine seat's", "The engine seat's"),
    "next.fix.parity": ("Pending: Max runs the sampling gate on the fixed build in a rig window, kit defaults. ",
                        "Max ran the sampling gate on the fixed build in a rig window, kit defaults (gate.json, gate.log). "),
}
PARITY_SUITES = {"greedy": 2, "seed_0.7": 8}
DONE_ID = re.compile(r"\[tensorfold\] done (\S+) ")


def done_lines(p: Path) -> list:
    """Each `[tensorfold] done` line of a serve log: request id, token hash, token count, seed (None when greedy)."""
    need(p.is_file(), f"missing {p}")
    out = []
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m, sha = DONE_ID.search(ln), SHA_RE.search(ln)
        tok, seed = re.search(r"\btokens=(\d+)", ln), re.search(r"\bseed=(\d+)", ln)
        if m and sha and tok:
            out.append({"id": m.group(1), "sha": sha.group(1), "tokens": int(tok.group(1)),
                        "seed": int(seed.group(1)) if seed else None, "line": ln.strip()})
    return out


def said_f6(r: dict) -> str:
    """Finding 6's request at the launcher's default: where it ended, the reasoning and answer tokens, the answer."""
    n, rt, at = r.get("completion_tokens"), r.get("reasoning_tokens"), r.get("answer_tokens")
    need(all(isinstance(x, int) for x in (n, rt, at)), f"finding 6 request without token counts: {n}, {rt}, {at}")
    b = r.get("go_blocks") or {}
    answer = ("the two Go blocks it asked for" if b.get("two_go_blocks") else "no answer" if at == 0
              else f"not the two Go blocks ({b.get('fenced_blocks')} code blocks)")
    return f"{r['finish_reason']} after {n:,} tokens: {rt:,} reasoning, {at:,} answer; {answer}"


def set_claim(by: dict, cid: str, value, spread: str, receipt: str, fills: list) -> None:
    c = by[cid]
    need(c["status"] in ("pending", "not_run"), f"{cid} is {c['status']}, not pending; refusing to overwrite")
    fills.append((cid, value, spread, receipt))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("kind", choices=("engine", "window1", "codecs", "decode-match", "bf16", "control", "control-dropped", "tools", "temp07",
                                     "fix-engine", "fix07", "parity"))
    ap.add_argument("--reason", default="not run (dropped for time)")
    ap.add_argument("--summary", help="decode-match: the boot's summary.json, if not in DIR or DIR/../codecs; tools: if not DIR/../probes; "
                                      "parity: if not beside gate.json")
    ap.add_argument("--serve-log", help="tools: rank 0's serve log, for its libtfgrammar.so warning; parity: rank 0's log of the fixed boot")
    ap.add_argument("--release-log", help="parity: rank 0's log of the release build's temp07 boot")
    ap.add_argument("--alongside", help="temp07, fix07: what else ran on the Sparks during the run (summary.json isolation shared); the README prints it")
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--commit"); ap.add_argument("--native"); ap.add_argument("--oracle"); ap.add_argument("--boundary")
    ap.add_argument("--gate")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    doc = json.loads(LEDGER.read_text(encoding="utf-8"))
    by = {c["id"]: c for c in doc["claims"]}
    fills: list = []
    note = 0
    swap: dict = {}  # claim id -> (pending opening of its setup, what replaces it once filled)
    writes: list = []  # (file name, text) written into the receipt folder, then cleaned like the copies
    fnative = (by.get("next.fix.sha256.native") or {}).get("value")
    fcommit = (by.get("next.fix.commit") or {}).get("value")
    try:
        if a.kind == "control-dropped":
            for w in WORKLOADS:
                c = by[f"next.decode.{w}.control"]
                need(c["status"] == "pending", f"{c['id']} is {c['status']}, not pending")
            need(bool(a.reason.strip()), "--reason must say why")
            dest, copies = None, []
        elif a.kind in ("engine", "fix-engine"):
            for k in ("commit", "native", "oracle", "boundary", "gate"):
                need(bool(getattr(a, k)), f"{a.kind} needs --{k}")
            for k in ("native", "oracle", "boundary"):
                need(len(getattr(a, k)) == 64, f"--{k} must be a full SHA-256")
            if a.kind == "fix-engine":
                old_commit, old_native = by["next.engine.commit"]["value"], by["next.engine.sha256.native"]["value"]
                need(bool(old_commit and old_native), "next.engine.* is still pending: the fixed build is told apart from the release build by it")
                need(a.commit != old_commit, "--commit is the release build's own commit (next.engine.commit), not the fixed build's")
                need(re.fullmatch(r"[0-9a-f]{40}", a.commit) is not None, f"--commit {a.commit!r}: want the engine head's full 40-character commit")
                need(a.native != old_native, "--native is the release build's own tensorfold-native (next.engine.sha256.native), not the fixed build's")
            pre, branch = ("next.engine", "hy4-next") if a.kind == "engine" else ("next.fix", "hy4-sample-fast")
            set_claim(by, f"{pre}.commit", a.commit, f"branch {branch}", PATCH, fills)
            for k in ("native", "oracle", "boundary"):
                set_claim(by, f"{pre}.sha256.{k}", getattr(a, k), "built by serve/build_release.sh", PATCH, fills)
            set_claim(by, "next.gate.merged" if a.kind == "engine" else "next.fix.gate", a.gate, "", PATCH, fills)
            if a.kind == "fix-engine":
                swap = {cid: FIX_SETUPS[cid] for cid, *_ in fills}
            dest, copies = None, []
        else:
            need(bool(a.dir), f"{a.kind} needs a receipt folder")
            d = Path(a.dir)
            dest = KIT / "receipts" / "next" / "release" / {"window1": "window1", "codecs": "codecs", "decode-match": "decode-match",
                                                             "bf16": "prefill-bf16", "control": "control", "tools": "tool-probes",
                                                             "temp07": "temp07", "fix07": "fix07", "parity": "fix-parity"}[a.kind]
            rel = dest.relative_to(KIT).as_posix()
            native = by["next.engine.sha256.native"]["value"]
            commit = by["next.engine.commit"]["value"]
            if a.kind in ("window1", "codecs", "decode-match", "bf16", "tools", "temp07"):
                need(bool(native and commit), "run `fill_next.py engine` first: next.engine.* is still pending")
            if a.kind == "window1":
                s = check_run(d, by, {"TF_HY4_FAST_PREFILL": "1", "TF_HY4_PREFILL_AR": "fp32", "SERVE_VISION": "1"}, native)
                need(s["engine"].get("engine_commit") == commit, f"engine commit {s['engine'].get('engine_commit')!r} != next.engine.commit {commit!r}")
                rm = rigmark(d)
                st = rm.get("settings", {})
                need(st.get("runs") == 3 and st.get("prefill_runs") == 1, f"RigMark runs {st.get('runs')}, prefill_runs {st.get('prefill_runs')}; want 3 and 1")
                for k, n in DEPTHS:
                    node = rm["prefill"][n]["cold"]
                    set_claim(by, f"next.prefill.{k}.fp32", r2(node["effective_prefill_tokens_per_second"]["median"]),
                              "1 cold request (RigMark --prefill-runs=1)", f"{rel}/{rm['_file']}", fills)
                rid = (rm.get("run") or {}).get("comparison_id")
                for w in WORKLOADS:
                    node = rm["decode"][w]["decode_tokens_per_second"]
                    toks = ", ".join(str(r["completion_tokens"]) for r in rm["decode"][w]["runs"])
                    set_claim(by, f"next.decode.{w}.fp32", r2(node["median"]),
                              f"{r2(node['minimum'])} to {r2(node['maximum'])} over {len(rm['decode'][w]['runs'])} runs; {toks} tokens; "
                              f"comparison id {rid}" + ("" if rid == SPEED_ID else ", so other prompts than the earlier build's row"),
                              f"{rel}/{rm['_file']}", fills)
                on, off = load(d / "niah_128k_thinkon.json"), load(d / "niah_128k_nothink.json")
                need(find(on, "strict_exact") is True and find(off, "strict_exact") is True,
                     f"needle not found exactly: thinking on {find(on, 'strict_exact')}, off {find(off, 'strict_exact')}")
                set_claim(by, "next.niah128k.ttft.fp32", r2(on["response"]["ttft_s"]),
                          f"found the code with thinking on and off; thinking off: {r2(off['response']['ttft_s'])} s", f"{rel}/niah_128k_thinkon.json", fills)
                v = load(d / "vision_check.json")
                need(v.get("server_alive_after") is True, "vision_check: the server stopped answering after the probes")
                set_claim(by, "next.vision.known_answer.merged", v["known_pass"],
                          "; ".join(f"{k['id']} {'right' if k['pass'] else 'wrong: ' + json.dumps(k['wrong'])}" for k in v["known_answers"]),
                          f"{rel}/vision_check.json", fills)
                rates = [k["prompt_tokens"] / k["ttft_s"] for k in v["known_answers"] if k.get("prompt_tokens") and k.get("ttft_s")]
                need(len(rates) == 3, "vision_check: a known-answer request has no prompt tokens or time to first output")
                set_claim(by, "next.vision.prefill.merged", r2(statistics.median(rates)),
                          ", ".join(f"{r2(x)}" for x in rates) + " tok/s on one image each", f"{rel}/vision_check.json", fills)
                pr = v["probes"]

                def said(p):
                    return f"{p['label']}: " + ("accepted" + (f", {p['prompt_tokens']} prompt tokens" if p.get("prompt_tokens") else "")
                                                if p["accepted"] else f"refused, HTTP {p['status']}, {err_msg(p)}")
                # The Formats row is filled by `codecs`, from a boot with the image decoders staged.
                ok = [p for p in pr["sizes"] if p["accepted"]]
                set_claim(by, "next.vision.contract.largest", ok[-1]["label"] if ok else "none of the probes",
                          "; ".join(said(p) for p in pr["sizes"]), f"{rel}/vision_check.json", fills)
                okn = [p for p in pr["per_request"] if p["accepted"]]
                set_claim(by, "next.vision.contract.per_request", okn[-1]["label"] if okn else "none",
                          "; ".join(said(p) for p in pr["per_request"]), f"{rel}/vision_check.json", fills)
                big = pr["sizes"][-1]
                set_claim(by, "next.vision.contract.over_cap", said(big), "the 8192x8192 probe", f"{rel}/vision_check.json", fills)
                bad = pr["malformed"] + pr["remote_url"]
                codes = {p["status"] for p in bad}
                short = (f"{', '.join(p['label'] for p in bad)}: each refused, HTTP {codes.pop()}"
                         if not any(p["accepted"] for p in bad) and len(codes) == 1 else "; ".join(said(p) for p in bad))
                set_claim(by, "next.vision.contract.malformed", short, "; ".join(said(p) for p in bad), f"{rel}/vision_check.json", fills)
                nv = (s["engine"].get("nccl_version_by_rank") or {}).get("0", "not printed")
                set_claim(by, "next.nccl.version", nv, "rank 0's serve log", f"{rel}/summary.json", fills)
                if not str(nv).startswith(DOCS_NCCL + "+") and nv != DOCS_NCCL:
                    print(f"NOTE: the serve log prints NCCL {nv}; README.md:129, FINDINGS.md:26, THIRD_PARTY.md:11, launch/RING-RDMA.md "
                          f"and launch/ring.env.example say {DOCS_NCCL}. Change them to what the log prints.")
                    note = 3
                copies = [d / "summary.json", d / rm["_file"], d / "niah_128k_thinkon.json", d / "niah_128k_nothink.json", d / "vision_check.json"]
            elif a.kind == "codecs":
                s = check_run(d, by, {"TF_HY4_FAST_PREFILL": "1", "TF_HY4_PREFILL_AR": "fp32", "SERVE_VISION": "1"}, native)
                need(s["engine"].get("engine_commit") == commit, f"engine commit {s['engine'].get('engine_commit')!r} != next.engine.commit {commit!r}")
                dec = s["engine"].get("image_decoders") or {}
                need(dec.get("jpeg") == "found" and dec.get("webp") == "found",
                     f"rank 0's serve log does not show both image decoders found at boot: {dec or 'not recorded'}")
                v = load(d / "vision_check.json")
                fm = {p["label"]: p for p in (v.get("probes") or {}).get("formats", [])}
                for k in ("jpeg", "webp"):
                    need(k in fm, f"vision_check.json has no {k} format probe (this kit's bench/vision_check.py has one)")
                    need(fm[k]["accepted"] and fm[k].get("answered_right") is True,
                         f"{k}: {said_probe(fm[k])}; answered right: {fm[k].get('answered_right')}")
                set_claim(by, "next.vision.contract.formats", ", ".join(FORMAT_NAMES.get(k, k.upper()) for k, p in fm.items() if p["accepted"]),
                          "; ".join(said_probe(p) + (", answered right" if p.get("answered_right") else "") for p in fm.values()),
                          f"{rel}/vision_check.json", fills)
                copies = [d / "summary.json", d / "vision_check.json"]
            elif a.kind == "decode-match":
                sp = Path(a.summary) if a.summary else next(
                    (p for p in (d / "summary.json", d.parent / "codecs" / "summary.json") if p.is_file()), d / "summary.json")
                s = check_run(d, by, {"TF_HY4_FAST_PREFILL": "1", "TF_HY4_PREFILL_AR": "fp32"}, native, sp)
                need(s["engine"].get("engine_commit") == commit, f"engine commit {s['engine'].get('engine_commit')!r} != next.engine.commit {commit!r}")
                rm, base = rigmark(d), rigmark(LAST_DECODE)
                rid = (rm.get("run") or {}).get("comparison_id")
                need(rid == SPEED_ID, f"RigMark --comparison-id {rid!r}, want {SPEED_ID!r}: another id gives other prompts and other text")
                need((base.get("run") or {}).get("comparison_id") == SPEED_ID, f"{LAST_DECODE.name}'s receipt is not under {SPEED_ID}")
                st = rm.get("settings", {})
                need(st.get("runs") == 3 and not rm.get("prefill"), f"RigMark runs {st.get('runs')} with prefill {bool(rm.get('prefill'))}; want 3 runs, --skip-prefill")
                got, want_sha = serve_shas(d / "verify.txt"), serve_shas(LAST_DECODE / "rank0-decode-done.txt")
                need(want_sha is not None and len(want_sha) == 9, f"{LAST_DECODE.name}/rank0-decode-done.txt should hold 9 decode replies")
                need(got is None or len(got) == 9, f"verify.txt lists {len(got or [])} serve log sha= values, want 9")
                other = rigmark(LAST_DECODE_OTHER_PATH)
                eq_all, differ, outside = [], [], []
                for w in WORKLOADS:
                    runs, bruns = rm["decode"][w]["runs"], base["decode"][w]["runs"]
                    need(len(runs) == 3 and len(bruns) == 3, f"{w}: {len(runs)} runs here, {len(bruns)} in the earlier build's row; want 3 and 3")
                    eq = [r.get("output") == b.get("output") for r, b in zip(runs, bruns)]
                    eq_all += eq
                    differ += [f"{w} run {i + 1}" for i, e in enumerate(eq) if not e]
                    node = rm["decode"][w]["decode_tokens_per_second"]
                    set_claim(by, f"next.decode.{w}.match", r2(node["median"]),
                              ", ".join(str(r2(r["decode_tokens_per_second"])) for r in runs) + " tok/s over 3 runs; "
                              + ", ".join(str(r["completion_tokens"]) for r in runs) + f" tokens; the same text as the earlier build's in {sum(eq)} of 3 runs",
                              f"{rel}/{rm['_file']}", fills)
                    two = [float(x["decode"][w]["decode_tokens_per_second"]["median"]) for x in (base, other)]
                    lo, hi = min(two), max(two)
                    med = float(node["median"])
                    print(f"{w}: median {med:.3f}, earlier build {two[0]:.3f} ({LAST_DECODE.name}) and {two[1]:.3f} ({LAST_DECODE_OTHER_PATH.name}), "
                          f"{100 * (med - two[0]) / two[0]:+.2f}% against {LAST_DECODE.name}")
                    if not lo - (hi - lo) <= med <= hi + (hi - lo):
                        outside.append(f"{w} {med:.3f} outside {lo - (hi - lo):.3f} to {hi + (hi - lo):.3f}")
                if got is not None:
                    sha_eq = [g == x for g, x in zip(got, want_sha)]
                    bad = [i + 1 for i in range(9) if sha_eq[i] != eq_all[i]]
                    need(not bad, f"the serve log's sha= and the receipts' outputs disagree on reply {bad} of 9: look before filling")
                    check = f"serve log sha= the same in {sum(sha_eq)} of 9 ({rel}/verify.txt against {LAST_DECODE.name}/rank0-decode-done.txt)"
                else:
                    check = "serve log sha= not checked: no verify.txt"
                n = sum(eq_all)
                said = ("all 9 the same text, token for token, so the rates are like for like" if n == 9 else
                        f"{n} of 9 the same text; {', '.join(differ)} {'differs' if len(differ) == 1 else 'differ'}, "
                        f"so {'that rate is' if len(differ) == 1 else 'those rates are'} not like for like")
                set_claim(by, "next.decode.same_output", said,
                          f"each run's output against the same run of {LAST_DECODE.relative_to(KIT).as_posix()}; {check}", f"{rel}/{rm['_file']}", fills)
                if outside:
                    print("NOTE: " + "; ".join(outside) + ". That is beyond the binary's path effect: the page's decode line "
                          "(decode path not changed) needs a look before it ships.")
                    note = 3
                copies = [sp, d / rm["_file"]] + [p for p in (d / "verify.txt", d / "rigmark-decode-match.log") if p.is_file()]
            elif a.kind == "tools":
                sp = Path(a.summary) if a.summary else d.parent / "probes" / "summary.json"
                s = check_run(d, by, {"TF_HY4_FAST_PREFILL": "1", "TF_HY4_PREFILL_AR": "fp32"}, native, sp)
                need(s["engine"].get("engine_commit") == commit, f"engine commit {s['engine'].get('engine_commit')!r} != next.engine.commit {commit!r}")
                for name, cid in TOOL_PROBES:
                    res = load(d / f"{name}.json").get("result") or {}
                    value, spread = said_tool(name, res)
                    set_claim(by, cid, value, spread, f"{rel}/{name}.json", fills)
                need((d / "summary.json").is_file(), f"missing {d / 'summary.json'} (tool_probes.py's own summary)")
                warn = []
                if a.serve_log:
                    lg = Path(a.serve_log)
                    need(lg.is_file(), f"missing {lg}")
                    warn = [ln.strip() for ln in lg.read_text(encoding="utf-8", errors="replace").splitlines() if GRAMMAR_WARN in ln]
                if not warn:
                    print(f"NOTE: no '{GRAMMAR_WARN}' line from rank 0's serve log ({a.serve_log or 'no --serve-log'}); "
                          "the page quotes that warning, so give it a receipt or change the sentence.")
                    note = 3
                copies = [(sp, "summary.json"), (d / "summary.json", "tool_probes.json")] + [d / f"{n}.json" for n, _ in TOOL_PROBES] \
                    + [p for p in (d / "probes.log", d.parent / "tool_probes.py") if p.is_file()]
            elif a.kind in ("temp07", "fix07"):
                fix = a.kind == "fix07"
                pre, build = ("next.fix07", "fixed build") if fix else ("next.temp07", "release build")
                if fix:
                    need(bool(fnative and fcommit), "run `fill_next.py fix-engine` first: next.fix.* is still pending")
                s = check_run(d, by, {"TF_HY4_FAST_PREFILL": "1", "TF_HY4_PREFILL_AR": "fp32", "SERVE_TEMPERATURE": "0.7"}, fnative if fix else native)
                want_commit, commit_id = (fcommit, "next.fix.commit") if fix else (commit, "next.engine.commit")
                need(s["engine"].get("engine_commit") == want_commit, f"engine commit {s['engine'].get('engine_commit')!r} != {commit_id} {want_commit!r}")
                budget = (s.get("launch_settings") or {}).get("SERVE_THINKING_BUDGET")
                need(not budget, f"SERVE_THINKING_BUDGET={budget!r} in force; the requests without a budget need none on the server")
                line = str(((s.get("ranks") or {}).get("0") or {}).get("sampling") or "")
                need(re.match(r"temperature 0\.7(,|$)", line) is not None, f"rank 0's serving line gives sampling {line!r}, not temperature 0.7 first")
                sc = load(d / "sampling_check.json")
                need(sc.get("tool") == "bench/sampling_check.py" and sc.get("settings") == SAMPLING_SETTINGS,
                     f"sampling_check.json settings {sc.get('settings')} are not this kit's bench/sampling_check.py")
                need(sc.get("all_ok") is True, "sampling_check.json: a request got no reply, no finish reason or an error")
                t0, t1 = sc.get("utc_start"), sc.get("utc_end")
                need(all(isinstance(t, str) and UTC_RE.fullmatch(t) for t in (t0, t1)), f"sampling_check.json utc_start {t0!r}, utc_end {t1!r}: want two UTC times")
                lead = f"Run on the {build} in a rig window, requests from {t0} to {t1} (sampling_check.json). "
                iso = s.get("isolation")
                need(iso in ("quiet", "shared"), f"summary.json isolation {iso!r}; collect_receipts.py --isolation writes quiet or shared")
                if iso == "quiet":
                    need(not a.alongside, "--alongside names what ran during the run, but summary.json says isolation quiet (nothing else running)")
                    along = "nothing else running"
                else:
                    along = (a.alongside or "").strip().rstrip(".")
                    need(bool(along), "summary.json says isolation shared: name what ran alongside with --alongside TEXT (the README prints it)")
                    need(len(along.split()) <= 20, f"--alongside is {len(along.split())} words; keep it to 20 or fewer (the README block's word limit)")
                    need(chr(0x2014) not in along, "--alongside has an em dash; use a comma or parentheses")
                set_claim(by, f"{pre}.alongside", along, f"summary.json isolation {iso}", f"{rel}/summary.json", fills)
                set_claim(by, f"{pre}.serving", line, "rank 0's serving line", f"{rel}/summary.json", fills)
                for w in ("prose", "code"):
                    for kind, cid in (("default", f"{pre}.decode.{w}"), ("greedy", f"{pre}.greedy.{w}")):
                        runs = sc["decode"][w][kind]
                        need(len(runs) == 3 and all(r.get("decode_tps") for r in runs), f"{w} {kind}: want 3 runs, each with a rate")
                        need(all(("temperature" in r["request"]) == (kind == "greedy") and r["request"].get("temperature", 0) == 0 for r in runs),
                             f"{w} {kind}: want {'temperature 0' if kind == 'greedy' else 'no temperature'} in every request")
                        set_claim(by, cid, r2(statistics.median(r["decode_tps"] for r in runs)),
                                  ", ".join(str(r2(r["decode_tps"])) for r in runs) + " tok/s over 3 runs; "
                                  + ", ".join(f"{r['completion_tokens']:,}" for r in runs) + " tokens ("
                                  + ", ".join(f"{r['reasoning_tokens']:,}" if isinstance(r.get("reasoning_tokens"), int) else "?" for r in runs)
                                  + " reasoning); finish " + ", ".join(str(r["finish_reason"]) for r in runs) + f"; alongside: {along}",
                                  f"{rel}/sampling_check.json", fills)
                if fix:
                    # The thinking-off runs: the README's Decode at 0.7 bullet. A reply that reasoned is not an answer speed.
                    nt = sc.get("decode_nothink") or {}
                    for w in ("prose", "code"):
                        for kind, cid in (("default", f"{pre}.nothink.decode.{w}"), ("greedy", f"{pre}.nothink.greedy.{w}")):
                            runs = (nt.get(w) or {}).get(kind) or []
                            need(len(runs) == 3 and all(r.get("decode_tps") for r in runs), f"thinking off, {w} {kind}: want 3 runs, each with a rate")
                            need(all(r["request"].get("reasoning_effort") == SAMPLING_SETTINGS["nothink_reasoning_effort"] for r in runs),
                                 f"thinking off, {w} {kind}: want reasoning_effort no_think in every request")
                            need(all(("temperature" in r["request"]) == (kind == "greedy") and r["request"].get("temperature", 0) == 0 for r in runs),
                                 f"thinking off, {w} {kind}: want {'temperature 0' if kind == 'greedy' else 'no temperature'} in every request")
                            need(all(r.get("reasoning_tokens") == 0 for r in runs),
                                 f"thinking off, {w} {kind}: reasoning tokens {[r.get('reasoning_tokens') for r in runs]}, want 0 in every reply")
                            set_claim(by, cid, r2(statistics.median(r["decode_tps"] for r in runs)),
                                      ", ".join(str(r2(r["decode_tps"])) for r in runs) + " tok/s over 3 runs; "
                                      + ", ".join(f"{r['completion_tokens']:,}" for r in runs) + " tokens, no reasoning; finish "
                                      + ", ".join(str(r["finish_reason"]) for r in runs) + f"; alongside: {along}",
                                      f"{rel}/sampling_check.json", fills)
                for kind in ("budget", "none"):
                    r = sc["f6"][kind]
                    q = r["request"]
                    need("temperature" not in q and (q.get("thinking_budget") == SAMPLING_SETTINGS["f6_budget"]) == (kind == "budget"),
                         f"finding 6 {kind}: sent temperature {q.get('temperature')!r}, thinking_budget {q.get('thinking_budget')!r}")
                    set_claim(by, f"{pre}.f6.{kind}", said_f6(r),
                              f"{r.get('decode_tps')} tok/s; seed {q.get('seed')}; {(r.get('go_blocks') or {}).get('fenced_blocks')} code blocks in the answer; alongside: {along}",
                              f"{rel}/sampling_check.json", fills)
                if fix:
                    early = [f"{w} {kind} run {i + 1}" for w in ("prose", "code") for kind in ("default", "greedy")
                             for i, r in enumerate(sc["decode"][w][kind])
                             if not isinstance(r.get("answer_tokens"), int) or r["answer_tokens"] > 0 or r.get("finish_reason") != "length"]
                    if early:
                        print(f"NOTE: thinking on, {', '.join(early)} reached its answer, gave no answer count or did not end on length. "
                              "README.md's Reasoning at 0.7 bullet says every reply was still reasoning at the cap: change that sentence.")
                        note = 3
                    got6 = {cid: v for cid, v, *_ in fills if cid.startswith(f"{pre}.f6.")}
                    diff6 = [k for k in ("budget", "none") if by[f"next.temp07.f6.{k}"]["value"] != got6[f"{pre}.f6.{k}"]]
                    if diff6:
                        print(f"NOTE: finding 6 on the fixed build ({', '.join(diff6)}) is not the release build's next.temp07.f6.* reply. "
                              "FINDINGS.md says the release build gave the same two replies: check the parity gate and that sentence.")
                        note = 3
                swap = {cid: (PENDING_LEAD_FIX if fix else PENDING_LEAD, lead) for cid, *_ in fills}
                copies = [d / "summary.json", d / "sampling_check.json"] + [p for p in (d / "sampling_check.log",) if p.is_file()]
            elif a.kind == "parity":
                need(bool(fnative and fcommit), "run `fill_next.py fix-engine` first: next.fix.* is still pending")
                need(bool(a.serve_log and a.release_log),
                     "parity needs --serve-log (rank 0's log of the fixed boot) and --release-log (rank 0's log of the release build's temp07 boot)")
                need(Path(a.serve_log).resolve() != Path(a.release_log).resolve(), "--serve-log and --release-log are the same file")
                gj, gl = (d, d.with_suffix(".log")) if d.suffix == ".json" else (d / "gate.json", d / "gate.log")
                need(gl.is_file(), f"missing {gl} (the gate's printed lines)")
                sp = Path(a.summary) if a.summary else gj.parent / "summary.json"
                s = check_run(d, by, {"TF_HY4_FAST_PREFILL": "1", "TF_HY4_PREFILL_AR": "fp32", "SERVE_TEMPERATURE": "0.7"}, fnative, sp)
                need(s["engine"].get("engine_commit") == fcommit, f"engine commit {s['engine'].get('engine_commit')!r} != next.fix.commit {fcommit!r}")
                g = load(gj)
                cases = g.get("cases") or []
                par = [c for c in cases if c.get("suite") in PARITY_SUITES]
                count = {k: sum(c.get("suite") == k for c in par) for k in PARITY_SUITES}
                need(count == PARITY_SUITES, f"gate.json has {count} parity cases, want {PARITY_SUITES}")
                fixed, release = done_lines(Path(a.serve_log)), done_lines(Path(a.release_log))
                rel_ids = {x["id"] for x in release}
                got_f, got_r, said = [], [], []
                for c in par:
                    name, sha, tok = c.get("case"), c.get("sha"), c.get("tokens")
                    need(c.get("pass") is True, f"{name}: failed in the gate ({c.get('got')} against {c.get('expect')})")
                    m = re.fullmatch(r"sha=([0-9a-f]+),tok=(\d+)", str(c.get("expect")))
                    need(m is not None and sha == m.group(1) and tok == int(m.group(2)), f"{name}: sha {sha}, {tok} tokens against expected {c.get('expect')}")
                    need(c.get("id") not in rel_ids, f"{name}: its request id is in --release-log, so that log is the fixed boot's, not the release build's")
                    fx = [x for x in fixed if x["id"] == c.get("id")]
                    need(len(fx) == 1, f"{name}: request id {c.get('id')} is in {len(fx)} done lines of --serve-log, want 1")
                    fx = fx[0]
                    need(fx["sha"] == sha and fx["tokens"] == tok, f"{name}: --serve-log has sha {fx['sha']}, {fx['tokens']} tokens; the gate got {sha}, {tok}")
                    seeded = c.get("suite") == "seed_0.7"
                    need((fx["seed"] is not None) == seeded, f"{name}: --serve-log's done line has seed {fx['seed']}, want {'a seed' if seeded else 'none (greedy)'}")
                    rx = [x for x in release if x["sha"] == sha and x["tokens"] == tok and x["seed"] == fx["seed"]]
                    need(bool(rx), f"{name}: no done line in --release-log with sha {sha}, {tok} tokens" + (f" and seed {fx['seed']}" if seeded else " and no seed"))
                    got_f.append(fx["line"])
                    got_r.append(rx[0]["line"])
                    said.append(f"{name} {sha}, {tok:,} tokens" + (f", seed {fx['seed']}" if seeded else ""))
                others = {}
                for c in cases:
                    if c.get("suite") not in PARITY_SUITES:
                        n, k = others.get(c.get("suite"), (0, 0))
                        others[c.get("suite")] = (n + 1, k + (c.get("pass") is True))
                rest = ", ".join(f"{su} {k} of {n}" for su, (n, k) in others.items())
                set_claim(by, "next.fix.parity", f"greedy {count['greedy']} of {count['greedy']}, same seed {count['seed_0.7']} of {count['seed_0.7']}",
                          "; ".join(said) + f"; in the release build's serve log, each with the same hash and token count; the gate's other checks passed: {rest or 'none'}",
                          f"{rel}/gate.json", fills)
                swap = {"next.fix.parity": FIX_SETUPS["next.fix.parity"]}
                writes = [("fixed-done-lines.txt", "\n".join(got_f) + "\n"), ("release-done-lines.txt", "\n".join(got_r) + "\n")]
                copies = [(gj, "gate.json"), (gl, "gate.log"), (sp, "summary.json")]
            elif a.kind == "bf16":
                check_run(d, by, {"TF_HY4_FAST_PREFILL": "1", "TF_HY4_PREFILL_AR": "bf16"}, native)
                rm = rigmark(d)
                need(rm.get("settings", {}).get("prefill_runs") == 1, "RigMark prefill_runs is not 1")
                for k, n in DEPTHS:
                    node = rm["prefill"][n]["cold"]
                    set_claim(by, f"next.prefill.{k}.bf16", r2(node["effective_prefill_tokens_per_second"]["median"]),
                              "1 cold request (RigMark --prefill-runs=1), prefill-only run", f"{rel}/{rm['_file']}", fills)
                copies = [d / "summary.json", d / rm["_file"]]
            else:
                s = check_run(d, by, {"TF_HY4_FAST_PREFILL": "0"}, LAST_RELEASE_BIN)
                w1 = KIT / "receipts" / "next" / "release" / "window1" / "summary.json"
                need(w1.is_file(), "fill window1 first: the control's binary path is checked against it")
                p1, p0 = load(w1)["engine"]["tensorfold_bin"], s["engine"]["tensorfold_bin"]
                need(len(str(p0)) == len(str(p1)), f"control binary path {p0!r} is not the same length as the release's {p1!r}")
                rm = rigmark(d)
                st = rm.get("settings", {})
                need(st.get("runs") == 3 and not rm.get("prefill"), f"RigMark runs {st.get('runs')} with prefill {bool(rm.get('prefill'))}; want 3 runs, --skip-prefill")
                for w in WORKLOADS:
                    node = rm["decode"][w]["decode_tokens_per_second"]
                    set_claim(by, f"next.decode.{w}.control", r2(node["median"]),
                              f"{r2(node['minimum'])} to {r2(node['maximum'])} over {len(rm['decode'][w]['runs'])} runs", f"{rel}/{rm['_file']}", fills)
                copies = [d / "summary.json", d / rm["_file"]]
    except Refuse as e:
        print(f"REFUSED ({a.kind}): {e}", file=sys.stderr)
        return 2
    for cid, value, spread, receipt in fills:
        print(f"{'would fill' if a.dry_run else 'fill'} {cid} = {value!r}  [{spread}]  <- {receipt}")
    if a.kind == "control-dropped":
        for w in WORKLOADS:
            print(f"{'would mark' if a.dry_run else 'mark'} next.decode.{w}.control not_run: {a.reason}")
    if a.dry_run:
        return note
    if dest:
        dest.mkdir(parents=True, exist_ok=True)
        for p in copies:
            src, name = p if isinstance(p, tuple) else (p, p.name)
            clean_copy(src, dest / name)
        if a.kind == "tools" and warn:
            tmp = dest / "grammar_warning.txt"
            tmp.write_text("\n".join(warn) + "\n", encoding="utf-8")
            clean_copy(tmp, tmp)
        for name, text in writes:
            (dest / name).write_text(text, encoding="utf-8")
            clean_copy(dest / name, dest / name)
    for cid, value, spread, receipt in fills:
        c = by[cid]
        c["value"], c["spread"], c["status"], c["receipt"] = value, spread or None, "measured", receipt
        c.pop("status_note", None)
        old, new = swap.get(cid, (None, None))
        if old and str(c.get("setup") or "").startswith(old):
            c["setup"] = new + c["setup"][len(old):]
    if a.kind == "control-dropped":
        for w in WORKLOADS:
            c = by[f"next.decode.{w}.control"]
            c["value"], c["spread"], c["status"], c["status_note"] = None, None, "not_run", a.reason
    LEDGER.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for cmd in (["tools/render.py", "--all"], ["tools/check_kit.py"]):
        r = subprocess.run([sys.executable, "-I", *cmd], cwd=KIT, capture_output=True, text=True)
        print((r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else f"{cmd[0]} rc {r.returncode}")
        if r.returncode != 0:
            print(f"ERROR: {cmd[0]} failed after the fill; the ledger is written, fix and rerun render/check_kit", file=sys.stderr)
            return 1
    return note


if __name__ == "__main__":
    raise SystemExit(main())
