#!/usr/bin/env python3
"""Offline test of tools/fill_next.py on a copy of the kit, with receipt folders built from the speed build's real
RigMark and needle receipts (receipts/next/prefill/) standing in for the release run's.

    python3 -I tools/test_fill_next.py

Checks: each refusal named in fill_next.py's docstring fires on its case and writes nothing; the four good fills
write their claims, copy their receipts and leave check_kit passing; a second fill of the same folder is refused.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
PRE = KIT / "receipts" / "next" / "prefill"
FAILS: list[str] = []
NATIVE = "ab" * 32
LAST = "4a3d5265523408f302dfeb2066cdae1f4fe9f28276310c9d3c82df1c0aa35695"


def check(cond: bool, what: str) -> None:
    print(f"  {'ok  ' if cond else 'FAIL'} {what}")
    if not cond:
        FAILS.append(what)


def summary(sha: str, settings: dict, binpath: str, commit: str = "c0ffee1") -> dict:
    return {"launch_settings": settings,
            "engine": {"engine_commit": commit, "tensorfold_bin": binpath, "bin_sha256_by_rank": {str(r): sha for r in range(4)},
                       "bin_sha256_same_on_all_ranks": True, "gpu_clock_cap_mhz": "2200", "problems": [],
                       "nccl_version_by_rank": {"0": "2.30.7+cuda13.3"}}}


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
        AGENT = "receipts/next/release/agent-loop/"
        doc = json.loads((kit / "card" / "claims.json").read_text())
        for c in doc["claims"]:
            receipt = str(c.get("receipt") or "")
            run_row = c["id"].startswith(("next.engine.", "next.gate.merged")) or receipt.startswith("receipts/next/release/")
            if run_row and not receipt.startswith(AGENT) and c["status"] in ("measured", "not_run"):
                c["value"], c["spread"], c["status"] = None, None, "pending"
                c.pop("status_note", None)
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
        check("Outputs against the earlier build's row: all 9 the same text, token for token, so the rates are like for like." in page
              and "| This release (default) | (pending)" not in page, "the page shows the release decode row and the same-output sentence")
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
    print("== " + ("PASS" if not FAILS else f"FAIL ({len(FAILS)}): " + "; ".join(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
