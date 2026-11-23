#!/usr/bin/env python3
"""Offline test of tools/fill_next.py on a copy of the kit, with receipt folders built from the speed build's real
RigMark and needle receipts (receipts/next/prefill/) standing in for the release run's.

    python3 -I tools/test_fill_next.py

Checks: each refusal named in fill_next.py's docstring fires on its case and writes nothing; the good fills
write their claims, copy their receipts and leave check_kit passing; a second fill of the same folder is refused.
temp07 runs the kit's own bench/sampling_check.py against a fake server on 127.0.0.1, so the filler reads what that
script writes. The fixed build's kinds follow: fix-engine, fix07 on the same script's output (its default rates
raised, so the page shows which build each slot reads) and parity on a gate.json and two rank 0 logs made here in
gate.py's and the serve's formats.
"""

from __future__ import annotations

import copy
import http.server
import importlib.util
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
PRE = KIT / "receipts" / "next" / "prefill"
FAILS: list[str] = []
NATIVE = "ab" * 32
FNATIVE = "fe" * 32  # the fixed build's tensorfold-native
FIX = "f1" * 20  # the fixed build's engine commit
LAST ="4a3d5265523408f302dfeb2066cdae1f4fe9f28276310c9d3c82df1c0aa35695"


def check(cond: bool, what: str) -> None:
    print(f"  {'ok  ' if cond else 'FAIL'} {what}")
    if not cond:
        FAILS.append(what)


def summary(sha: str, settings: dict, binpath: str, commit: str = "c0ffee1") -> dict:
    return {"launch_settings": settings,
            "engine": {"engine_commit": commit, "tensorfold_bin": binpath, "bin_sha256_by_rank": {str(r): sha for r in range(4)},
                       "bin_sha256_same_on_all_ranks": True, "gpu_clock_cap_mhz": "2200", "problems": [],
                       "nccl_version_by_rank": {"0": "2.30.7+cuda13.3"}}}


class FakeServe(http.server.BaseHTTPRequestHandler):
    """A stand-in for the serve: /v1/models, and streamed chat replies shaped by what the request sends."""
    seen: list = []

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": "hy4-test"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        q = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeServe.seen.append(q)
        go = "```go\n// limiter.go\npackage limiter\n```\n\n```go\n// limiter_test.go\npackage limiter\n```"
        if "thinking_budget" in q:
            n, rt, text, fin = 6082, 4097, go, "stop"
        elif q.get("reasoning_effort") == "high":
            n, rt, text, fin = 8192, 8192, "", "length"
        elif q.get("reasoning_effort") == "no_think":
            n, rt, text, fin = (1500 if "temperature" in q else 1400), 0, "The module.", "stop"
        elif q.get("max_tokens") == 32:
            n, rt, text, fin = 32, 32, "", "length"
        else:
            n, rt, text, fin = 2048, (2048 if "temperature" in q else 1990), ("" if "temperature" in q else "An answer."), "length"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        pieces = ([{"reasoning_content": "Thinking."}, {"reasoning_content": " More."}] if rt else [{"content": "Here is "}]) + ([{"content": text}] if text else [])
        for d in pieces:
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": d}]}).encode() + b"\n\n")
            self.wfile.flush()
            time.sleep(0.01 if "temperature" in q else 0.02)
        last = {"choices": [{"delta": {}, "finish_reason": fin}],
                "usage": {"completion_tokens": n, "completion_tokens_details": {"reasoning_tokens": rt}}}
        self.wfile.write(b"data: " + json.dumps(last).encode() + b"\n\ndata: [DONE]\n\n")


def folder(root: Path, name: str, summ: dict, rig: dict | None, extra: dict) -> Path:
    d = root / name
    d.mkdir()
    (d / "summary.json").write_text(json.dumps(summ))
    if rig is not None:
        (d / "hy4-st-release.json").write_text(json.dumps(rig))
    for k, v in extra.items():
        (d / k).write_text(json.dumps(v))
    return d


def main() -> int:
    bench = json.loads((PRE / "step2-bench-ar32" / "hy4-st-step2-bench-ar32.json").read_text())
    on = json.loads((PRE / "step2-niah" / "niah_128k_thinkon.json").read_text())
    off = json.loads((PRE / "step2-niah" / "niah_128k_nothink.json").read_text())
    probe = lambda label, ok, pt=None: {"label": label, "accepted": ok, "status": 200 if ok else 400, "prompt_tokens": pt, "error": None if ok else "refused"}
    vision = {"known_pass": 3, "known_total": 3, "server_alive_after": True,
              "known_answers": [{"id": i, "pass": True, "wrong": {}, "prompt_tokens": 300, "ttft_s": t} for i, t in (("text_card", 1.5), ("color_grid", 1.0), ("bar_chart", 2.0))],
              "probes": {"formats": [probe("png", True, 300), probe("bmp", False)],
                         "sizes": [probe("1440x900", True, 1288), probe("2880x1800", True, 4000), probe("5760x3600", False), probe("8192x8192", False)],
                         "per_request": [probe("1 images", True), probe("2 images", True), probe("4 images", False)],
                         "malformed": [probe("bad base64", False), probe("truncated png", False)], "remote_url": [probe("http url", False)]}}
    with tempfile.TemporaryDirectory(prefix="hy4-fill-test-") as td:
        td = Path(td)
        kit = td / "kit"
        shutil.copytree(KIT, kit, ignore=shutil.ignore_patterns(".git", "__pycache__"))
        # Start from a ledger where the run has filled nothing yet: the engine and release rows go back to pending.
        # The agent-loop rows stay as shipped, with their receipts: fill_next.py does not fill them, and the
        # tool-call check below reads them from the page.
        # The fixed build's rows (next.fix.*, next.fix07.*) go back too, with the filler's own pending openings.
        AGENT = "receipts/next/release/agent-loop/"
        spec = importlib.util.spec_from_file_location("fill_next", kit / "tools" / "fill_next.py")
        fn = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fn)
        doc = json.loads((kit / "card" / "claims.json").read_text())
        for c in doc["claims"]:
            receipt = str(c.get("receipt") or "")
            run_row = c["id"].startswith(("next.engine.", "next.gate.merged", "next.fix.")) or receipt.startswith("receipts/next/release/")
            if run_row and not receipt.startswith(AGENT) and c["status"] in ("measured", "not_run"):
                c["value"], c["spread"], c["status"] = None, None, "pending"
                c.pop("status_note", None)
                # A filled temp07 or fix07 setup opens with the run's times; put the pending sentence back, as before the fill.
                head = str(c.get("setup") or "").split(" (sampling_check.json). ", 1)
                for pre, build, pend in (("next.temp07.", "release build", fn.PENDING_LEAD), ("next.fix07.", "fixed build", fn.PENDING_LEAD_FIX)):
                    if c["id"].startswith(pre) and head[0].startswith(f"Run on the {build} in a rig window, requests from ") and len(head) == 2:
                        c["setup"] = pend + head[1]
                # A filled next.fix.* setup lost its pending opening; put it back.
                pend, done = fn.FIX_SETUPS.get(c["id"], ("", ""))
                if done and str(c.get("setup") or "").startswith(done):
                    c["setup"] = pend + c["setup"][len(done):]
        (kit / "card" / "claims.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
        for p in sorted((kit / "receipts" / "next" / "release").glob("*")):
            if p.name != "agent-loop":
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        fill = lambda *args: subprocess.run([sys.executable, "-I", str(kit / "tools" / "fill_next.py"), *map(str, args)], capture_output=True, text=True)
        ledger = lambda: {c["id"]: c for c in json.loads((kit / "card" / "claims.json").read_text())["claims"]}
        good = {"TF_HY4_FAST_PREFILL": "1", "TF_HY4_PREFILL_AR": "fp32", "SERVE_VISION": "1"}
        b1 = copy.deepcopy(bench); b1["note"] = "/ho" + "me/someone/rigmark13 on spark-" + "267e"
        w1 = folder(td, "w1", summary(NATIVE, good, "/x/hy4/next-bin/tensorfold-native"), b1,
                    {"niah_128k_thinkon.json": on, "niah_128k_nothink.json": off, "vision_check.json": vision})

        print("== refusals before the engine fill")
        p = fill("window1", w1)
        check(p.returncode == 2 and "run `fill_next.py engine` first" in p.stderr, f"window1 before engine refused: {p.stderr.strip()[-90:]}")
        p = fill("engine", "--commit", "c0ffee1", "--native", NATIVE, "--oracle", "cd" * 32, "--boundary", "short", "--gate", "PASS")
        check(p.returncode == 2 and "full SHA-256" in p.stderr, "a short hash refused")
        p = fill("engine", "--commit", "c0ffee1", "--native", NATIVE, "--oracle", "cd" * 32, "--boundary", "ef" * 32, "--gate", "PASS (gates log x)")
        check(p.returncode == 0 and ledger()["next.engine.commit"]["value"] == "c0ffee1", f"engine filled: rc {p.returncode} {p.stderr.strip()[-120:]}")

        print("== window1 refusals")
        bad = folder(td, "w1-ar", summary(NATIVE, dict(good, TF_HY4_PREFILL_AR="bf16"), "/x/hy4/next-bin/tensorfold-native"), bench, {})
        p = fill("window1", bad)
        check(p.returncode == 2 and "TF_HY4_PREFILL_AR='bf16'" in p.stderr, "BF16 sums under the FP32 claims refused")
        bad = folder(td, "w1-sha", summary("12" * 32, good, "/x/hy4/next-bin/tensorfold-native"), bench, {})
        p = fill("window1", bad)
        check(p.returncode == 2 and "is not the expected" in p.stderr, "another binary refused")
        s = summary(NATIVE, good, "/x/hy4/next-bin/tensorfold-native"); s["engine"]["problems"] = ["rank 2: graphics clock at ready not read (read-failed)"]
        p = fill("window1", folder(td, "w1-prob", s, bench, {}))
        check(p.returncode == 2 and "engine problems" in p.stderr, "an unread clock refused")
        s = summary(NATIVE, good, "/x/hy4/next-bin/tensorfold-native"); s["engine"]["gpu_clock_cap_mhz"] = "none declared (stock clocks)"
        p = fill("window1", folder(td, "w1-cap", s, bench, {}))
        check(p.returncode == 2 and "clock cap declared" in p.stderr, "no declared cap refused")
        b2 = copy.deepcopy(bench); b2["settings"]["prefill_runs"] = 3
        p = fill("window1", folder(td, "w1-rm", summary(NATIVE, good, "/x/hy4/next-bin/tensorfold-native"), b2,
                                   {"niah_128k_thinkon.json": on, "niah_128k_nothink.json": off, "vision_check.json": vision}))
        check(p.returncode == 2 and "prefill_runs 3" in p.stderr, "other RigMark settings refused")
        check(ledger()["next.prefill.8k.fp32"]["status"] == "pending", "nothing written by the refusals")

        print("== window1 good")
        p = fill("window1", w1)
        L = ledger()
        check(p.returncode == 0, f"rc {p.returncode} {(p.stdout + p.stderr).strip()[-200:]}")
        check(L["next.prefill.8k.fp32"]["value"] == round(bench["prefill"]["8192"]["cold"]["effective_prefill_tokens_per_second"]["median"], 2)
              and L["next.niah128k.ttft.fp32"]["value"] == round(on["response"]["ttft_s"], 2), "prefill and needle values from the receipts")
        check(L["next.vision.known_answer.merged"]["value"] == 3 and L["next.vision.prefill.merged"]["value"] == 200.0
              and L["next.vision.contract.largest"]["value"] == "2880x1800" and L["next.vision.contract.per_request"]["value"] == "2 images",
              "vision rows: 3 of 3, median 200 tok/s (of 200, 300, 150), largest 2880x1800, 2 images")
        check(L["next.nccl.version"]["value"] == "2.30.7+cuda13.3" and (kit / "receipts/next/release/window1/vision_check.json").is_file(),
              "NCCL version filled and receipts copied")
        check("check_kit: PASS" in p.stdout, "check_kit passes after the fill")
        copied = (kit / "receipts/next/release/window1/hy4-st-release.json").read_text()
        check("~/rigmark13 on node1" in copied and "/ho" + "me/" not in copied, "copied RigMark JSON has ~ for the home folder and node1 for the host")
        p = fill("window1", w1)
        check(p.returncode == 2 and "not pending" in p.stderr, "a second fill refused")

        print("== codecs")
        good = {"TF_HY4_FAST_PREFILL": "1", "TF_HY4_PREFILL_AR": "fp32", "SERVE_VISION": "1"}
        cprobe = lambda label, ok, right: probe(label, ok, 300) | {"answered_right": right}
        cv = copy.deepcopy(vision)
        cv["probes"]["formats"] = [probe("png", True, 300), probe("bmp", False), cprobe("jpeg", True, True), cprobe("webp", True, True)]
        def csum(dec):
            s = summary(NATIVE, good, "/x/hy4/next-bin/tensorfold-native")
            s["engine"]["image_decoders"] = dec
            return s
        p = fill("codecs", folder(td, "cd-miss", csum({"jpeg": "missing: warning: libturbojpeg.so.0 is required", "webp": "found"}), None, {"vision_check.json": cv}))
        check(p.returncode == 2 and "both image decoders found" in p.stderr, "codecs refused when the serve log named a missing decoder")
        bad = copy.deepcopy(cv); bad["probes"]["formats"][3] = cprobe("webp", True, False)
        p = fill("codecs", folder(td, "cd-wrong", csum({"jpeg": "found", "webp": "found"}), None, {"vision_check.json": bad}))
        check(p.returncode == 2 and "webp" in p.stderr and "answered right: False" in p.stderr, "codecs refused when WebP answered wrong")
        p = fill("codecs", folder(td, "cd", csum({"jpeg": "found", "webp": "found"}), None, {"vision_check.json": cv}))
        L = ledger()
        check(p.returncode == 0 and L["next.vision.contract.formats"]["value"] == "PNG, JPEG, WebP" and "check_kit: PASS" in p.stdout
              and (kit / "receipts/next/release/codecs/vision_check.json").is_file(),
              f"codecs fills Formats as PNG, JPEG, WebP and check_kit passes: rc {p.returncode} {(p.stdout + p.stderr).strip()[-160:]}")
        print("== decode-match")
        # step2-bench-ar32 ran under the speed team's id, and its outputs are base-2200's run for run.
        base_shas = [l.split(" sha=")[1].split()[0] for l in (PRE / "base-2200" / "rank0-decode-done.txt").read_text().splitlines()]
        dm = copy.deepcopy(bench); dm.pop("prefill")
        verify = lambda shas: "MATCH\nshas=" + repr(shas) + "\n"

        def dmdir(name, rig, shas=None, settings=None, summary_here=True):
            d = td / name / "decode-match"
            d.mkdir(parents=True)
            (d / "hy4-st-release-decode-match.json").write_text(json.dumps(rig))
            if shas is not None:
                (d / "verify.txt").write_text(verify(shas))
            s = json.dumps(summary(NATIVE, settings or good, "/x/hy4/next-bin/tensorfold-native"))
            if summary_here:
                (d / "summary.json").write_text(s)
            else:
                (td / name / "codecs").mkdir()
                (td / name / "codecs" / "summary.json").write_text(s)
            return d
        other = copy.deepcopy(dm); other["run"]["comparison_id"] = "2026-11-22-hy4-release-2200mhz"
        p = fill("decode-match", dmdir("dm-id", other, base_shas))
        check(p.returncode == 2 and "another id gives other prompts" in p.stderr, "decode-match under another comparison id refused")
        p = fill("decode-match", dmdir("dm-pre", bench, base_shas))
        check(p.returncode == 2 and "--skip-prefill" in p.stderr, "decode-match with prefill refused")
        p = fill("decode-match", dmdir("dm-off", dm, base_shas, dict(good, TF_HY4_FAST_PREFILL="0")))
        check(p.returncode == 2 and "TF_HY4_FAST_PREFILL='0'" in p.stderr, "decode-match with fast prefill off refused")
        p = fill("decode-match", dmdir("dm-sha", dm, ["0" * 12] + base_shas[1:]))
        check(p.returncode == 2 and "disagree on reply [1]" in p.stderr, "serve log sha= against equal outputs refused")
        d1 = copy.deepcopy(dm); d1["decode"]["code"]["runs"][1]["output"] += " changed"
        p = fill("decode-match", dmdir("dm-diff", d1, base_shas[:1] + ["1" * 12] + base_shas[2:]), "--dry-run")
        check(p.returncode == 0 and "8 of 9 the same text; code run 2 differs, so that rate is not like for like" in p.stdout,
              f"one differing output is named and not called like for like: {p.stdout.strip()[-160:]}")
        d2 = copy.deepcopy(dm); d2["decode"]["code"]["decode_tokens_per_second"]["median"] = 30.0
        p = fill("decode-match", dmdir("dm-note", d2, base_shas), "--dry-run")
        check(p.returncode == 3 and "NOTE: code 30.000 outside" in p.stdout, "a median beyond the path effect prints a NOTE")
        check(ledger()["next.decode.same_output"]["status"] == "pending", "nothing written by the refusals and dry runs")
        dgood = dmdir("dm", dm, base_shas, summary_here=False)
        p = fill("decode-match", dgood)
        L = ledger()
        check(p.returncode == 0 and "check_kit: PASS" in p.stdout, f"decode-match filled, summary.json from ../codecs: rc {p.returncode} {(p.stdout + p.stderr).strip()[-200:]}")
        check(L["next.decode.code.match"]["value"] == round(dm["decode"]["code"]["decode_tokens_per_second"]["median"], 2)
              and "3182, 2572, 2581 tokens; the same text as the earlier build's in 3 of 3 runs" in L["next.decode.code.match"]["spread"],
              f"code median, token counts and same-text count: {L['next.decode.code.match']['spread']}")
        check(L["next.decode.same_output"]["value"] == "all 9 the same text, token for token, so the rates are like for like"
              and "serve log sha= the same in 9 of 9" in L["next.decode.same_output"]["spread"], "all 9 the same, cross-checked 9 of 9")
        page = (kit / "README.md").read_text()
        check("Outputs against the development build's run: all 9 the same text, token for token, so the rates are like for like." in page
              and f"- **{L['next.decode.code.match']['value']:.2f} tok/s** decode on code, **{L['next.decode.prose.match']['value']:.2f}** on prose" in page,
              "the page's Speed list shows the release decode rates, and Development notes the same-output sentence")
        check(all((kit / "receipts/next/release/decode-match" / n).is_file() for n in ("summary.json", "hy4-st-release-decode-match.json", "verify.txt")),
              "summary.json, the RigMark receipt and verify.txt copied")
        p = fill("decode-match", dgood)
        check(p.returncode == 2 and "not pending" in p.stderr, "a second decode-match refused")
        print("== tools")

        def tdir(name, results, settings=None):
            d = td / name / "tool-probes"
            d.mkdir(parents=True)
            (d / "summary.json").write_text(json.dumps({"model": "m", "probes": results}))
            for n, r in results.items():
                (d / f"{n}.json").write_text(json.dumps({"request": {}, "result": r}))
            (td / name / "probes").mkdir()
            (td / name / "probes" / "summary.json").write_text(json.dumps(summary(NATIVE, settings or good, "/x/hy4/next-bin/tensorfold-native")))
            return d
        refused = {"status": 400, "seconds": 0.1, "error_message": "structured output needs libtfgrammar.so"}
        called = {"status": 200, "seconds": 3.2, "finish_reason": "tool_calls", "content": None, "reasoning_chars": 300, "usage": {"completion_tokens": 90},
                  "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Austin, TX\"}"}}]}
        tres = {"tool_auto": called, "tool_required": refused, "format_json_schema": refused,
                "format_json_object": {"status": 200, "seconds": 2.0, "finish_reason": "stop", "content": "{\"city\": \"Houston\"}",
                                       "reasoning_chars": 0, "usage": {"completion_tokens": 8}}}
        p = fill("tools", tdir("tp-none", dict(tres, tool_required={"status": None, "error": "TimeoutError()", "seconds": 60})))
        check(p.returncode == 2 and "tool_required: no reply" in p.stderr, "a request with no reply refused")
        p = fill("tools", tdir("tp-off", tres, dict(good, TF_HY4_FAST_PREFILL="0")))
        check(p.returncode == 2 and "TF_HY4_FAST_PREFILL='0'" in p.stderr, "tools on the earlier prefill path refused")
        p = fill("tools", tdir("tp-nolog", tres), "--dry-run")
        check(p.returncode == 3 and "NOTE: no 'libtfgrammar.so is required' line" in p.stdout, "no serve log given: a NOTE, since the page quotes the warning")
        check(ledger()["next.api.tool_auto"]["status"] == "pending", "nothing written by the refusals and the dry run")
        log = td / "rank0.log"
        log.write_text("x\nwarning: libtfgrammar.so is required for structured output; see /ho" + "me/someone/xgrammar\n")
        p = fill("tools", tdir("tp", tres), "--serve-log", log)
        L = ledger()
        check(p.returncode == 0 and "check_kit: PASS" in p.stdout, f"tools filled: rc {p.returncode} {(p.stdout + p.stderr).strip()[-200:]}")
        check(L["next.api.tool_auto"]["value"] == 'HTTP 200, called get_weather({"city": "Austin, TX"})'
              and L["next.api.tool_required"]["value"] == "HTTP 400: structured output needs libtfgrammar.so"
              and L["next.api.json_object"]["value"] == 'HTTP 200, JSON {"city": "Houston"}', "tool call, the server's refusal and the JSON reply as values")
        tr = kit / "receipts/next/release/tool-probes"
        check(all((tr / n).is_file() for n in ("summary.json", "tool_probes.json", "tool_auto.json", "format_json_object.json"))
              and "see ~/xgrammar" in (tr / "grammar_warning.txt").read_text(), "receipts copied, the boot summary kept apart, the warning line cleaned")
        # The page's "auto" row is the five agent tasks, counted here from their receipts; the probes fill the other rows.
        page = (kit / "README.md").read_text()
        al = kit / AGENT
        tasks = [t for r in ("loop-short-nothink", "loop-short-think", "loop-long-nothink")
                 for t in json.loads((al / r / "agent_loop.json").read_text())["tasks"].values()]
        auto = (f"{sum(len(t['per_turn']) for t in tasks)} requests answered, {sum(t['tool_calls'] for t in tasks)} tool calls, "
                f"{sum(t['bad_calls'] for t in tasks)} malformed; {sum(1 for t in tasks if t['ok'])} of {len(tasks)} tasks solved")
        check(len(tasks) == 5 and L["next.agent.tool_calls"]["value"] == auto
              and f'| `tools`, `tool_choice` `"auto"`, the five tasks | {auto} |' in page,
              f"the page's auto row gives the five tasks' numbers from the receipts: {auto}")
        check('| `tools`, `tool_choice` `"required"` | HTTP 400: structured output needs libtfgrammar.so |' in page
              and '| `response_format` `json_schema` | HTTP 400: structured output needs libtfgrammar.so |' in page
              and '| `response_format` `json_object` | HTTP 200, JSON {"city": "Houston"} |' in page,
              "the page shows the probes' refusals and the JSON reply")
        print("== bf16 and control")
        p = fill("bf16", folder(td, "bf", summary(NATIVE, {"TF_HY4_FAST_PREFILL": "1", "TF_HY4_PREFILL_AR": "bf16"}, "/x/hy4/next-bin/tensorfold-native"), bench, {}))
        check(p.returncode == 0 and ledger()["next.prefill.64k.bf16"]["status"] == "measured", f"bf16 filled: rc {p.returncode} {p.stderr.strip()[-120:]}")
        dec = copy.deepcopy(bench); dec.pop("prefill")
        p = fill("control", folder(td, "c-path", summary(LAST, {"TF_HY4_FAST_PREFILL": "0"}, "/x/hy4/int/winB-bin/tensorfold-native"), dec, {}))
        check(p.returncode == 2 and "not the same length" in p.stderr, "control at a path of another length refused")
        p = fill("control", folder(td, "c-sha", summary(NATIVE, {"TF_HY4_FAST_PREFILL": "0"}, "/x/hy4/base-bin/tensorfold-native"), dec, {}))
        check(p.returncode == 2 and "is not the expected 4a3d5265" in p.stderr, "control with the release binary refused")
        p = fill("control", folder(td, "c-pre", summary(LAST, {"TF_HY4_FAST_PREFILL": "0"}, "/x/hy4/base-bin/tensorfold-native"), bench, {}))
        check(p.returncode == 2 and "--skip-prefill" in p.stderr, "control run with prefill refused")
        print("== control dropped for time, then run after all")
        p = fill("control-dropped")
        L = ledger()
        page = (kit / "README.md").read_text()
        check(p.returncode == 0 and L["next.decode.code.control"]["status"] == "not_run" and L["next.decode.code.control"]["value"] is None
              and "check_kit: PASS" in p.stdout, f"control-dropped marks not_run, check_kit passes: rc {p.returncode} {p.stderr.strip()[-120:]}")
        check("(control)" not in page and "not run (dropped for time)" not in page, "the page has no control row (decode-match is the comparison row)")
        p = fill("control-dropped")
        check(p.returncode == 2 and "not pending" in p.stderr, "a second control-dropped refused")
        p = fill("control", folder(td, "c", summary(LAST, {"TF_HY4_FAST_PREFILL": "0"}, "/x/hy4/base-bin/tensorfold-native"), dec, {}))
        check(p.returncode == 0 and ledger()["next.decode.code.control"]["status"] == "measured" and "check_kit: PASS" in p.stdout
              and "status_note" not in ledger()["next.decode.code.control"],
              f"control filled over not_run, note gone, check_kit passes: rc {p.returncode} {(p.stdout + p.stderr).strip()[-160:]}")

        print("== temp07: bench/sampling_check.py against a fake serve, then the fill")
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeServe)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        sc_dir = td / "t07"
        p = subprocess.run([sys.executable, "-I", str(kit / "bench" / "sampling_check.py"), "--url", f"http://127.0.0.1:{srv.server_address[1]}",
                            "--out", str(sc_dir)], capture_output=True, text=True, timeout=120)
        srv.shutdown()
        sc_log = p.stdout
        check(p.returncode == 0 and (sc_dir / "sampling_check.json").is_file(), f"sampling_check.py rc {p.returncode} {p.stderr.strip()[-160:]}")
        seen = FakeServe.seen
        on, off = seen[1:13], seen[13:25]
        dflt = [q for q in on if "temperature" not in q]
        check(len(seen) == 27 and all(q.get("max_tokens") == 2048 for q in on + off) and len(dflt) == 6
              and not any(k in q for q in dflt for k in ("top_p", "top_k", "seed", "reasoning_effort"))
              and sum(q.get("temperature") == 0 and "reasoning_effort" not in q for q in on) == 6,
              f"{len(seen)} requests: 6 thinking-on decode runs send nothing but the prompt and the cap, 6 send temperature 0")
        doff = [q for q in off if "temperature" not in q]
        check(all(q.get("reasoning_effort") == "no_think" for q in off) and len(doff) == 6
              and not any(k in q for q in doff for k in ("top_p", "top_k", "seed"))
              and sum(q.get("temperature") == 0 for q in off) == 6
              and [q["messages"] for q in off] == [q["messages"] for q in on],
              "then the same 12 runs with thinking off: reasoning_effort no_think and nothing else added, 6 with temperature 0")
        f6q = [q for q in seen if q.get("reasoning_effort") == "high"]
        check(len(f6q) == 2 and all("temperature" not in q and q.get("seed") == 20260905 for q in f6q) and f6q[0].get("thinking_budget") == 4096
              and "thinking_budget" not in f6q[1], "finding 6's request: no temperature, seed 20260905, with the budget then without")
        sc = json.loads((sc_dir / "sampling_check.json").read_text())
        nt = sc.get("decode_nothink") or {}
        check(sorted(nt) == ["code", "prose"] and all(len(nt[w][k]) == 3 for w in nt for k in ("greedy", "default"))
              and all(r["reasoning_tokens"] == 0 and r["answer_tokens"] == r["completion_tokens"] and r["decode_tps"] for w in nt.values() for k in w.values() for r in k)
              and sc["settings"]["nothink_reasoning_effort"] == "no_think",
              "sampling_check.json keeps the thinking-off runs apart, under decode_nothink: 3 and 3 a prompt, no reasoning, each with a rate")
        p5 = json.loads((kit / "receipts/thinking/budget_probes.json").read_text())["P5"]["payload"]["messages"]
        check(f6q[0]["messages"] == p5, "finding 6's messages are the receipt's P5, unchanged")
        t07 = {"TF_HY4_FAST_PREFILL": "1", "TF_HY4_PREFILL_AR": "fp32", "SERVE_TEMPERATURE": "0.7"}

        def tsum(settings=None, line="temperature 0.7, top_k -1, top_p 1", iso="shared", sha=NATIVE, commit="c0ffee1"):
            s = summary(sha, t07 if settings is None else settings, "/x/hy4/next-bin/tensorfold-native", commit)
            s["ranks"] = {"0": {"sampling": line}}
            if iso is not None:
                s["isolation"] = iso
            return s

        def tdir7(name, s, scj=None):
            d = td / name
            d.mkdir()
            (d / "summary.json").write_text(json.dumps(s))
            (d / "sampling_check.json").write_text(json.dumps(scj or sc))
            return d
        p = fill("temp07", tdir7("t7-empty", tsum({k: v for k, v in t07.items() if k != "SERVE_TEMPERATURE"})))
        check(p.returncode == 2 and "SERVE_TEMPERATURE=None in force, want '0.7'" in p.stderr, "a boot with SERVE_TEMPERATURE empty refused")
        p = fill("temp07", tdir7("t7-line", tsum(line="temperature 0.9, top_k -1, top_p 1")))
        check(p.returncode == 2 and "not temperature 0.7 first" in p.stderr, "a serving line at 0.9 refused")
        p = fill("temp07", tdir7("t7-budget", tsum(dict(t07, SERVE_THINKING_BUDGET="4096"))))
        check(p.returncode == 2 and "SERVE_THINKING_BUDGET='4096' in force" in p.stderr, "a server thinking budget refused")
        other = copy.deepcopy(sc); other["settings"]["decode_tokens"] = 1024
        p = fill("temp07", tdir7("t7-set", tsum(), other))
        check(p.returncode == 2 and "are not this kit's bench/sampling_check.py" in p.stderr, "other script settings refused")
        other = copy.deepcopy(sc); other["decode"]["code"]["default"][1]["request"]["temperature"] = 0.7
        p = fill("temp07", tdir7("t7-sent", tsum(), other), "--alongside", "the weights upload")
        check(p.returncode == 2 and "code default: want no temperature in every request" in p.stderr, "a run that sent a temperature refused")
        other = copy.deepcopy(sc); other["all_ok"] = False
        p = fill("temp07", tdir7("t7-ok", tsum(), other))
        check(p.returncode == 2 and "a request got no reply" in p.stderr, "an unanswered request refused")
        other = copy.deepcopy(sc); other.pop("utc_end")
        p = fill("temp07", tdir7("t7-time", tsum(), other), "--alongside", "the weights upload")
        check(p.returncode == 2 and "want two UTC times" in p.stderr, "a sampling_check.json without its end time refused")
        along = "the weights upload on node1 (one Python process, no GPU use, over Wi-Fi)"
        p = fill("temp07", tdir7("t7-shared", tsum()))
        check(p.returncode == 2 and "isolation shared: name what ran alongside with --alongside" in p.stderr, "isolation shared without --alongside refused")
        p = fill("temp07", tdir7("t7-quiet-text", tsum(iso="quiet")), "--alongside", along)
        check(p.returncode == 2 and "but summary.json says isolation quiet" in p.stderr, "--alongside with isolation quiet refused")
        p = fill("temp07", tdir7("t7-long", tsum()), "--alongside", " ".join(["word"] * 21))
        check(p.returncode == 2 and "--alongside is 21 words" in p.stderr, "an --alongside over 20 words refused")
        p = fill("temp07", tdir7("t7-noiso", tsum(iso=None)), "--alongside", along)
        check(p.returncode == 2 and "summary.json isolation None" in p.stderr, "a summary.json without isolation refused")
        check(ledger()["next.temp07.serving"]["status"] == "pending", "nothing written by the refusals")
        p = fill("temp07", tdir7("t7-quiet", tsum(iso="quiet")), "--dry-run")
        check(p.returncode == 0 and "would fill next.temp07.alongside = 'nothing else running'" in p.stdout,
              "isolation quiet gives 'nothing else running' (dry run)")
        good7 = tdir7("t7", tsum())
        (good7 / "sampling_check.log").write_text(sc_log)
        p = fill("temp07", good7, "--alongside", along + ".")
        L = ledger()
        check(p.returncode == 0 and "check_kit: PASS" in p.stdout, f"temp07 filled: rc {p.returncode} {(p.stdout + p.stderr).strip()[-200:]}")
        check(L["next.temp07.alongside"]["value"] == along and L["next.temp07.alongside"]["spread"] == "summary.json isolation shared"
              and L["next.temp07.decode.code"]["spread"].endswith(f"; alongside: {along}") and L["next.temp07.f6.none"]["spread"].endswith(f"; alongside: {along}"),
              "the upload is named in the alongside claim and in each run's spread, the trailing full stop dropped")
        lead7 = f"Run on the release build in a rig window, requests from {sc['utc_start']} to {sc['utc_end']} (sampling_check.json). "
        s7 = [c["setup"] for k, c in L.items() if k.startswith("next.temp07.")]
        check(len(s7) == 8 and all(s.startswith(lead7) and "nothing here is measured yet" not in s for s in s7),
              "each filled temp07 setup opens with the run's times, not the pending sentence")
        med = {f"{w}.{k}": round(statistics.median(r["decode_tps"] for r in sc["decode"][w][k]), 2) for w in ("prose", "code") for k in ("default", "greedy")}
        check(L["next.temp07.serving"]["value"] == "temperature 0.7, top_k -1, top_p 1"
              and L["next.temp07.decode.prose"]["value"] == med["prose.default"] and L["next.temp07.greedy.code"]["value"] == med["code.greedy"]
              and "2,048, 2,048, 2,048 tokens (1,990, 1,990, 1,990 reasoning); finish length, length, length;" in L["next.temp07.decode.prose"]["spread"],
              f"serving line and medians from the receipts: {L['next.temp07.decode.prose']['spread']}")
        b, n = L["next.temp07.f6.budget"]["value"], L["next.temp07.f6.none"]["value"]
        check(b == "stop after 6,082 tokens: 4,097 reasoning, 1,985 answer; the two Go blocks it asked for"
              and n == "length after 8,192 tokens: 8,192 reasoning, 0 answer; no answer", f"finding 6 at 0.7: {b!r}, {n!r}")
        # The release build's rows are the "before" of the README's Faster sampling paragraph; the 0.7 bullets read the fixed build.
        page = (kit / "README.md").read_text()
        check(f"With thinking on at 0.7, that build decoded {med['code.default']:.2f} tok/s on the code prompt, "
              f"against {med['code.greedy']:.2f} tok/s greedy on the same boot." in page,
              "the Faster sampling paragraph shows the release build's code rates at 0.7 and greedy")
        check(all((kit / "receipts/next/release/temp07" / f).is_file() for f in ("summary.json", "sampling_check.json", "sampling_check.log")),
              "summary.json, sampling_check.json and its log copied")
        p = fill("temp07", good7, "--alongside", along)
        check(p.returncode == 2 and "not pending" in p.stderr, "a second temp07 refused")

        print("== the fixed build: refusals before fix-engine, then fix-engine")
        p = fill("fix07", good7, "--alongside", along)
        check(p.returncode == 2 and "run `fill_next.py fix-engine` first" in p.stderr, "fix07 before fix-engine refused")
        p = fill("parity", td, "--serve-log", log, "--release-log", td / "other.log")
        check(p.returncode == 2 and "run `fill_next.py fix-engine` first" in p.stderr, "parity before fix-engine refused")
        hashes = ["--oracle", "cd" * 31 + "01", "--boundary", "ef" * 31 + "01", "--gate", "PASS (fix gates log y)"]
        p = fill("fix-engine", "--commit", "c0ffee1", "--native", FNATIVE, *hashes)
        check(p.returncode == 2 and "the release build's own commit" in p.stderr, "fix-engine with the release build's commit refused")
        p = fill("fix-engine", "--commit", FIX, "--native", NATIVE, *hashes)
        check(p.returncode == 2 and "the release build's own tensorfold-native" in p.stderr, "fix-engine with the release build's binary refused")
        p = fill("fix-engine", "--commit", FIX[:12], "--native", FNATIVE, *hashes)
        check(p.returncode == 2 and "full 40-character commit" in p.stderr, "fix-engine with a short commit refused")
        check(ledger()["next.fix.commit"]["status"] == "pending", "nothing written by the refusals")
        p = fill("fix-engine", "--commit", FIX, "--native", FNATIVE, *hashes)
        L = ledger()
        fx = {k: c for k, c in L.items() if k.startswith("next.fix.") and k != "next.fix.parity"}
        check(p.returncode == 0 and "check_kit: PASS" in p.stdout and L["next.fix.commit"]["value"] == FIX
              and L["next.fix.sha256.native"]["value"] == FNATIVE and L["next.fix.gate"]["value"] == "PASS (fix gates log y)"
              and L["next.engine.commit"]["value"] == "c0ffee1", f"fix-engine filled next.fix.*, next.engine.* untouched: rc {p.returncode} {(p.stdout + p.stderr).strip()[-160:]}")
        check(len(fx) == 5 and all(c["status"] == "measured" and not c["setup"].startswith("Pending") for c in fx.values())
              and L["next.fix.commit"]["setup"].startswith("The engine seat's head of branch hy4-sample-fast; this kit's patch is cut from it. The fix changes"),
              "the five next.fix.* setups lose their pending openings")
        page = (kit / "README.md").read_text()
        check(f"with one fix that makes sampled decode faster: {FIX}." in page and f"1.0 ships the release build with the sampling fix: {FIX}, branch" in page
              and "Its gate: PASS (fix gates log y)." in page, "the README names the fixed commit and its gate")
        p = fill("fix-engine", "--commit", FIX, "--native", FNATIVE, *hashes)
        check(p.returncode == 2 and "not pending" in p.stderr, "a second fix-engine refused")

        print("== fix07: the same script's output, its default rates raised by 10 tok/s")
        fsc = copy.deepcopy(sc)
        for w in ("prose", "code"):
            for r in fsc["decode"][w]["default"] + fsc["decode_nothink"][w]["default"]:
                r["decode_tps"] += 10
        fsum = lambda **k: tsum(sha=FNATIVE, commit=FIX, **k)
        p = fill("fix07", tdir7("f7-oldbin", tsum(commit=FIX), fsc), "--alongside", along)
        check(p.returncode == 2 and "is not the expected" in p.stderr, "fix07 on the release build's binary refused")
        p = fill("fix07", tdir7("f7-oldcommit", tsum(sha=FNATIVE), fsc), "--alongside", along)
        check(p.returncode == 2 and "engine commit 'c0ffee1' != next.fix.commit" in p.stderr, "fix07 under the release build's commit refused")
        other = copy.deepcopy(fsc); other["decode_nothink"]["code"]["default"][0]["reasoning_tokens"] = 12
        p = fill("fix07", tdir7("f7-reason", fsum(), other), "--alongside", along)
        check(p.returncode == 2 and "thinking off, code default: reasoning tokens [12, 0, 0], want 0" in p.stderr, "a thinking-off reply that reasoned refused")
        other = copy.deepcopy(fsc); other["decode_nothink"]["prose"]["greedy"][2]["request"].pop("reasoning_effort")
        p = fill("fix07", tdir7("f7-effort", fsum(), other), "--alongside", along)
        check(p.returncode == 2 and "thinking off, prose greedy: want reasoning_effort no_think" in p.stderr, "a thinking-off run without no_think refused")
        other = copy.deepcopy(fsc); other.pop("decode_nothink")
        p = fill("fix07", tdir7("f7-none", fsum(), other), "--alongside", along)
        check(p.returncode == 2 and "thinking off, prose default: want 3 runs" in p.stderr, "a sampling_check.json without the thinking-off rows refused")
        other = copy.deepcopy(fsc); other["f6"]["none"]["finish_reason"] = "stop"
        p = fill("fix07", tdir7("f7-f6", fsum(), other), "--alongside", along, "--dry-run")
        check(p.returncode == 3 and "NOTE: finding 6 on the fixed build (none) is not the release build's" in p.stdout,
              "finding 6 unlike the release build's prints a NOTE (dry run)")
        check(ledger()["next.fix07.serving"]["status"] == "pending", "nothing written by the refusals and the dry run")
        good_f7 = tdir7("f7", fsum(), fsc)
        p = fill("fix07", good_f7, "--alongside", along)
        L = ledger()
        check(p.returncode == 3 and "check_kit: PASS" in p.stdout
              and "NOTE: thinking on, prose default run 1, prose default run 2, prose default run 3, code default run 1" in p.stdout
              and "greedy run" not in p.stdout,
              f"fix07 filled; the fake's thinking-on default replies reach an answer, so a NOTE names them: rc {p.returncode} {(p.stdout + p.stderr).strip()[-240:]}")
        fmed ={f"{sec}.{w}.{k}": round(statistics.median(r["decode_tps"] for r in fsc[sec][w][k]), 2)
                for sec in ("decode", "decode_nothink") for w in ("prose", "code") for k in ("default", "greedy")}
        check(L["next.fix07.decode.code"]["value"] == fmed["decode.code.default"] and L["next.fix07.greedy.prose"]["value"] == fmed["decode.prose.greedy"]
              and L["next.fix07.nothink.decode.prose"]["value"] == fmed["decode_nothink.prose.default"]
              and L["next.fix07.nothink.greedy.code"]["value"] == fmed["decode_nothink.code.greedy"]
              and L["next.fix07.decode.code"]["value"] != L["next.temp07.decode.code"]["value"],
              "fix07 medians from its own receipt, thinking on and off, apart from temp07's")
        check("1,400, 1,400, 1,400 tokens, no reasoning; finish stop, stop, stop; alongside: " + along == L["next.fix07.nothink.decode.code"]["spread"].split(" tok/s over 3 runs; ")[1],
              f"thinking-off spread: tokens, no reasoning, finish, alongside: {L['next.fix07.nothink.decode.code']['spread']}")
        lead_f = f"Run on the fixed build in a rig window, requests from {sc['utc_start']} to {sc['utc_end']} (sampling_check.json). "
        sf = [c["setup"] for k, c in L.items() if k.startswith("next.fix07.")]
        check(len(sf) == 12 and all(s.startswith(lead_f) and "nothing here is measured yet" not in s for s in sf)
              and all(c["setup"].startswith(lead7) for k, c in L.items() if k.startswith("next.temp07.")),
              "each of the 12 fix07 setups opens with the run's times on the fixed build; temp07's are unchanged")
        page, fpage = (kit / "README.md").read_text(), (kit / "FINDINGS.md").read_text()
        check("Defaults in the serving line: temperature 0.70, top_k -1, top_p 1.\n- **Decode at 0.7.** On the fixed build" in page
              and f"median of three: {fmed['decode_nothink.prose.default']:.2f} tok/s and {fmed['decode_nothink.code.default']:.2f} tok/s on the prose and code prompts, "
                  f"against {fmed['decode_nothink.prose.greedy']:.2f} tok/s and {fmed['decode_nothink.code.greedy']:.2f} tok/s greedy on the same boot. "
                  f"The reply is the essay or the code itself. Running alongside: {along}.\n" in page
              and f"thinking on: {fmed['decode.prose.default']:.2f} tok/s and {fmed['decode.code.default']:.2f} tok/s, against "
                  f"{fmed['decode.prose.greedy']:.2f} tok/s and {fmed['decode.code.greedy']:.2f} tok/s greedy." in page,
              "the README's 0.7 bullets show the fixed build's thinking-off and thinking-on rates and what ran alongside")
        check(f"With the same settings it decoded {fmed['decode.code.default']:.2f} tok/s, against {fmed['decode.code.greedy']:.2f} tok/s greedy. "
              f"With thinking off, code written at 0.7 came at {fmed['decode_nothink.code.default']:.2f} tok/s" in page
              and f"that build decoded {med['code.default']:.2f} tok/s on the code prompt" in page,
              "the Faster sampling paragraph: the release build before, the fixed build after")
        words = {ln.split("**")[1]: len(ln.split()) for ln in page.splitlines()
                 if ln.startswith(("- **Default temperature.**", "- **Decode at 0.7.**", "- **Reasoning at 0.7.**"))}
        check(len(words) == 3 and max(words.values()) <= 70, f"the three filled bullets within the README's 70-word block limit: {words}")
        b, n = L["next.fix07.f6.budget"]["value"], L["next.fix07.f6.none"]["value"]
        check(f'With `"thinking_budget": 4096`, the finish reason was {b}. ' in fpage and f"With no budget, the finish reason was {n}. " in fpage
              and b == L["next.temp07.f6.budget"]["value"], "finding 6 shows both 0.7 requests from the fixed build")
        check(all((kit / "receipts/next/release/fix07" / f).is_file() for f in ("summary.json", "sampling_check.json")),
              "fix07's summary.json and sampling_check.json copied")
        p = fill("fix07", good_f7, "--alongside", along)
        check(p.returncode == 2 and "not pending" in p.stderr, "a second fix07 refused")

        print("== parity: gate.json and two rank 0 logs in gate.py's and the serve's formats")
        pcases = ([("greedy_prose", "greedy", None, 2048), ("greedy_code", "greedy", None, 2048)]
                  + [(f"seed_{w}_{i}", "seed_0.7", 1000 + 10 * j + i, 2048) for j, w in enumerate(("prose", "code")) for i in (1, 2, 3)]
                  + [("seed_f6_budget_4096", "seed_0.7", 20260905, 8192), ("seed_f6_no_budget", "seed_0.7", 20260905, 8192)])
        psha = {name: f"{0xa0b0c0d00000 + k:012x}" for k, (name, *_r) in enumerate(pcases)}

        def done(rid, sha, tok, seed):
            return (f"[tensorfold] done chatcmpl-{rid} prompt=57 cached=0 thinking=True effort=none tokens={tok} sha={sha} finish=length "
                    f"rounds=900 calls=600 offered=1800 accepted=900/1800 ttft=0.490 decode=80.000 elapsed=80.490 queued=0.000 tok/s=25.6"
                    + (f" seed={seed}" if seed is not None else ""))

        def plogs(name, rel_seed_shift=None, drop_fixed=None):
            d = td / name
            d.mkdir()
            rel = ["boot line", done("r0000", "1" * 12, 32, None)]
            fixed = ["boot line"]
            for k, (case, _suite, seed, tok) in enumerate(pcases):
                rel.append(done(f"r{k + 1:04d}", psha[case], tok, seed + (1 if case == rel_seed_shift else 0) if seed is not None else None))
                if case != drop_fixed:
                    fixed.append(done(f"f{k + 1:04d}", psha[case], tok, seed))
            (d / "release.log").write_text("\n".join(rel) + "\n")
            (d / "fixed.log").write_text("\n".join(fixed) + "\n")
            return d / "fixed.log", d / "release.log"

        def pgate(name, mutate=None, summ=None, as_file=False):
            d = td / name
            d.mkdir(exist_ok=True)
            cases = [{"case": case, "suite": suite, "pass": True, "expect": f"sha={psha[case]},tok={tok}", "got": f"sha={psha[case]},tok={tok}",
                      "id": f"chatcmpl-f{k + 1:04d}", "http_status": 200, "finish_reason": "length", "error": None, "tokens": tok, "sha": psha[case]}
                     for k, (case, suite, _seed, tok) in enumerate(pcases)]
            cases += [{"case": f"{su}_{i}", "suite": su, "pass": True, "tokens": 2048} for su in ("speed", "other_0.7") for i in range(4)]
            gj = {"base_url": "http://" + "192" + ".168.1.137:18080", "model": "m", "kit": "/ho" + "me/someone/kit", "log": None,
                  "reuse_enabled": False, "reuse_returncode": None, "passed": 18, "total": 18, "all_ok": True, "speed": [], "cases": cases}
            if mutate:
                mutate(gj)
            stem = "sample-gate-fix07" if as_file else "gate"
            (d / f"{stem}.json").write_text(json.dumps(gj))
            (d / f"{stem}.log").write_text("Gate target: http://" + "192" + ".168.1.137:18080  model=m  kit=/ho" + "me/someone/kit\nOVERALL: PASS (18/18 checks passed)\n")
            (d / "summary.json").write_text(json.dumps(summ or fsum()))
            return d / f"{stem}.json" if as_file else d
        flog, rlog = plogs("pl")
        p = fill("parity", pgate("pg-nolog"), "--serve-log", flog)
        check(p.returncode == 2 and "parity needs --serve-log" in p.stderr, "parity without --release-log refused")
        p = fill("parity", pgate("pg-same"), "--serve-log", flog, "--release-log", flog)
        check(p.returncode == 2 and "are the same file" in p.stderr, "the same log twice refused")
        p = fill("parity", pgate("pg-bin", summ=tsum(commit=FIX)), "--serve-log", flog, "--release-log", rlog)
        check(p.returncode == 2 and "is not the expected" in p.stderr, "a gate on the release build's binary refused")

        def fail_one(g):
            g["cases"][3]["pass"] = False
            g["cases"][3]["got"] = "sha=000000000000,tok=2048"
        p = fill("parity", pgate("pg-fail", fail_one), "--serve-log", flog, "--release-log", rlog)
        check(p.returncode == 2 and "seed_prose_2: failed in the gate" in p.stderr, "a failed parity case refused")
        p = fill("parity", pgate("pg-count", lambda g: g["cases"].pop(9)), "--serve-log", flog, "--release-log", rlog)
        check(p.returncode == 2 and "'seed_0.7': 7} parity cases" in p.stderr, "seven same-seed cases refused")
        fl2, rl2 = plogs("pl-drop", drop_fixed="greedy_code")
        p = fill("parity", pgate("pg-drop"), "--serve-log", fl2, "--release-log", rl2)
        check(p.returncode == 2 and "greedy_code: request id chatcmpl-f0002 is in 0 done lines of --serve-log" in p.stderr, "a reply missing from the fixed boot's log refused")
        f3, r3 = plogs("pl-seed", rel_seed_shift="seed_code_3")
        p = fill("parity", pgate("pg-seed"), "--serve-log", f3, "--release-log", r3)
        check(p.returncode == 2 and "seed_code_3: no done line in --release-log with sha" in p.stderr and "and seed 1013" in p.stderr,
              "a release reply under another seed refused")
        cp = td / "pl" / "fixed-copy.log"
        cp.write_text(flog.read_text())
        p = fill("parity", pgate("pg-copy"), "--serve-log", flog, "--release-log", cp)
        check(p.returncode == 2 and "that log is the fixed boot's, not the release build's" in p.stderr, "the fixed boot's log given as the release log refused")
        p = fill("parity", pgate("pg-dir"), "--serve-log", flog, "--release-log", rlog, "--dry-run")
        check(p.returncode == 0 and "would fill next.fix.parity = 'greedy 2 of 2, same seed 8 of 8'" in p.stdout, "a folder with gate.json and gate.log reads (dry run)")
        check(ledger()["next.fix.parity"]["status"] == "pending", "nothing written by the refusals and the dry run")
        gfile = pgate("pg-file", as_file=True)
        (gfile.parent / "summary.json").unlink()
        boot = td / "pg-file-boot"
        boot.mkdir()
        (boot / "summary.json").write_text(json.dumps(fsum()))
        p = fill("parity", gfile, "--serve-log", flog, "--release-log", rlog)
        check(p.returncode == 2 and "missing" in p.stderr and "summary.json" in p.stderr, "the gate's JSON alone, no summary.json beside it or --summary, refused")
        p = fill("parity", gfile, "--serve-log", flog, "--release-log", rlog, "--summary", boot / "summary.json")
        L = ledger()
        pr = kit / "receipts/next/release/fix-parity"
        check(p.returncode == 0 and "check_kit: PASS" in p.stdout and L["next.fix.parity"]["value"] == "greedy 2 of 2, same seed 8 of 8",
              f"parity filled from the gate's JSON and its .log: rc {p.returncode} {(p.stdout + p.stderr).strip()[-200:]}")
        check("seed_f6_no_budget a0b0c0d00009, 8,192 tokens, seed 20260905" in L["next.fix.parity"]["spread"]
              and L["next.fix.parity"]["spread"].endswith("the gate's other checks passed: speed 4 of 4, other_0.7 4 of 4")
              and L["next.fix.parity"]["setup"].startswith("Max ran the sampling gate on the fixed build in a rig window, kit defaults (gate.json, gate.log). Greedy:"),
              f"the spread lists each case and the other checks; the setup loses its pending opening: {L['next.fix.parity']['spread'][-120:]}")
        lines = lambda f: (pr / f).read_text().splitlines()
        check(all((pr / f).is_file() for f in ("gate.json", "gate.log", "summary.json")) and len(lines("fixed-done-lines.txt")) == 10
              and len(lines("release-done-lines.txt")) == 10 and "chatcmpl-r0010" in lines("release-done-lines.txt")[9],
              "gate.json, gate.log, summary.json and the 10 matched done lines of each log copied")
        check("http://HOST:18080" in (pr / "gate.log").read_text() and "~/kit" in (pr / "gate.json").read_text()
              and "192" + ".168.1." not in (pr / "gate.json").read_text(), "the copies have HOST for the home network address and ~ for the home folder")
        page, qpage = (kit / "README.md").read_text(), (kit / "card" / "QUALITY_CARD.md").read_text()
        top = next(ln for ln in page.splitlines() if ln.startswith("This release ships our merged engine"))
        check("gave the release build's tokens exactly: greedy 2 of 2, same seed 8 of 8 (`receipts/next/release/fix-parity/`)" in top
              and len(top.split()) <= 70 and "(greedy 2 of 2, same seed 8 of 8), so the values on this card stand for it." in qpage,
              f"the README's opening and the quality card carry the parity result ({len(top.split())} words)")
        p = fill("parity", gfile, "--serve-log", flog, "--release-log", rlog, "--summary", boot / "summary.json")
        check(p.returncode == 2 and "not pending" in p.stderr, "a second parity refused")
    print("== " + ("PASS" if not FAILS else f"FAIL ({len(FAILS)}): " + "; ".join(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
