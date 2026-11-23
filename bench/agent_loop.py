#!/usr/bin/env python3
"""Agent-style tool loop against an OpenAI-compatible serve. Stdlib only. Always exits 0.

Three tasks, each a fresh conversation with tools and tool_choice "auto", streamed the way an agent
harness streams:
  trail  follow a chain of small files, one read at a time, then report the code words in order
  fix    find and fix two bugs in a small Python module until its unit tests pass
  facts  with a long document in the system prompt, write six values from it to a JSON file
Per turn it records time to first piece, wall time, prompt and completion tokens, and whether each tool
call was usable (known tool, arguments parse as a JSON object, the right argument names). A bad call is
answered with an error message, as a harness would, and counted.

usage: python3 -I agent_loop.py --url http://HOST:PORT --out DIR [--doc FILE ...] [--model NAME]
           [--tasks trail,fix,facts] [--trail-len 24] [--max-turns 60] [--reasoning no_think] [--no-stream]
The facts task checks values from the Hy4 kit README, so pass that README as the first --doc.
"""
import argparse
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--doc", action="append", default=[], help="text put in the system prompt as long context")
ap.add_argument("--model", default="")
ap.add_argument("--tasks", default="trail,fix,facts")
ap.add_argument("--trail-len", type=int, default=24)
ap.add_argument("--max-turns", type=int, default=60)
ap.add_argument("--max-tokens", type=int, default=2048)
ap.add_argument("--reasoning", default="", help='sent as reasoning_effort when set, e.g. "no_think"')
ap.add_argument("--no-stream", action="store_true")
ap.add_argument("--timeout", type=float, default=900)
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
base = a.url.rstrip("/")
log = open(os.path.join(a.out, "agent_loop.log"), "a", encoding="utf-8")


def say(line):
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()


def model_name():
    if a.model:
        return a.model
    with urllib.request.urlopen(base + "/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


MODEL = model_name()
DOC = "\n\n".join(open(p, encoding="utf-8", errors="replace").read() for p in a.doc)
SYSTEM = ("You are a coding agent working in a workspace directory through tools. Use the tools to look "
          "and to act; never guess what a file holds. Paths are relative to the workspace. When the task "
          "is finished, reply in plain text without a tool call.")
if DOC:
    SYSTEM += "\n\nReference document for the project you maintain:\n\n" + DOC


def call(messages, tools):
    """One request. Returns content, tool_calls, usage, finish, ttft (streamed only) and wall seconds."""
    b = {"model": MODEL, "messages": messages, "tools": tools, "tool_choice": "auto", "temperature": 0,
         "max_tokens": a.max_tokens, "stream": not a.no_stream}
    if a.reasoning:
        b["reasoning_effort"] = a.reasoning
    if not a.no_stream:
        b["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(b).encode(),
                                 headers={"Content-Type": "application/json"})
    out = {"content": "", "reasoning_chars": 0, "tool_calls": [], "usage": {}, "finish": None, "ttft": None}
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=a.timeout) as r:
        if a.no_stream:
            d = json.load(r)
            m = d["choices"][0]["message"]
            out.update(content=m.get("content") or "", tool_calls=m.get("tool_calls") or [],
                       usage=d.get("usage") or {}, finish=d["choices"][0].get("finish_reason"),
                       reasoning_chars=len(m.get("reasoning_content") or ""))
        else:
            calls = {}
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                d = json.loads(data)
                if d.get("usage"):
                    out["usage"] = d["usage"]
                for ch in d.get("choices") or []:
                    if ch.get("finish_reason"):
                        out["finish"] = ch["finish_reason"]
                    delta = ch.get("delta") or {}
                    got = False
                    if delta.get("content"):
                        out["content"] += delta["content"]
                        got = True
                    rc = delta.get("reasoning_content") or delta.get("reasoning")
                    if rc:
                        out["reasoning_chars"] += len(rc)
                        got = True
                    for tc in delta.get("tool_calls") or []:
                        got = True
                        c = calls.setdefault(tc.get("index", 0),
                                             {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        if tc.get("id"):
                            c["id"] = tc["id"]
                        f = tc.get("function") or {}
                        c["function"]["name"] += f.get("name") or ""
                        c["function"]["arguments"] += f.get("arguments") or ""
                    if got and out["ttft"] is None:
                        out["ttft"] = time.time() - t0
            out["tool_calls"] = [calls[k] for k in sorted(calls)]
    out["wall"] = time.time() - t0
    return out


def tool(name, description, **props):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": {k: {"type": "string", "description": v} for k, v in props.items()},
        "required": list(props)}}}


def workspace_tools(root, with_tests):
    def safe(rel):
        p = os.path.realpath(os.path.join(root, rel))
        if p != root and not p.startswith(root + os.sep):
            raise OSError("path is outside the workspace")
        return p

    def list_dir(path):
        return "\n".join(sorted(os.listdir(safe(path)))) or "(empty)"

    def read_file(path):
        return open(safe(path), encoding="utf-8").read()

    def write_file(path, content):
        p = safe(path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w", encoding="utf-8").write(content)
        return f"wrote {len(content)} characters to {path}"

    def run_tests():
        r = subprocess.run([sys.executable, "-I", "-m", "unittest", "-q"], cwd=root, capture_output=True,
                           text=True, timeout=30)
        return (r.stdout + r.stderr)[-2500:] + f"\nexit code {r.returncode}"

    impl = {"list_dir": list_dir, "read_file": read_file, "write_file": write_file}
    specs = [tool("list_dir", "List the names in a workspace directory.", path='directory, "." for the top'),
             tool("read_file", "Read a text file.", path="file path"),
             tool("write_file", "Create or replace a text file.", path="file path", content="the whole new content")]
    if with_tests:
        impl["run_tests"] = run_tests
        specs.append(tool("run_tests", "Run the workspace's unit tests and return the output."))
    return impl, specs


def run_task(name, user, impl, specs):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    turns, n_calls, bad, tool_err, status, final = [], 0, 0, 0, "max_turns", ""
    for turn in range(1, a.max_turns + 1):
        try:
            r = call(messages, specs)
        except urllib.error.HTTPError as e:
            status = f"http {e.code}: {e.read()[:400].decode('utf-8', 'replace')}"
            break
        except Exception as e:  # noqa: BLE001  the run must end with a summary
            status = f"error: {e!r}"
            break
        u = r["usage"]
        turns.append({"turn": turn, "ttft": r["ttft"], "wall": round(r["wall"], 3),
                      "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                      "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                      "finish": r["finish"], "calls": [c["function"]["name"] for c in r["tool_calls"]]})
        say(f"  {name} turn {turn}: {turns[-1]}")
        msg = {"role": "assistant", "content": r["content"]}
        if r["tool_calls"]:
            msg["tool_calls"] = r["tool_calls"]
        messages.append(msg)
        if not r["tool_calls"]:
            status, final = "answered", r["content"]
            break
        for c in r["tool_calls"]:
            n_calls += 1
            fn = c["function"]["name"]
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments are not a JSON object")
                if fn not in impl:
                    raise ValueError(f"unknown tool {fn!r}")
                result = impl[fn](**args)
            except (ValueError, TypeError) as e:  # the call itself was not usable
                bad += 1
                result = f"ERROR: bad tool call: {e}"
            except (OSError, subprocess.TimeoutExpired) as e:  # a usable call that failed
                tool_err += 1
                result = f"ERROR: {e}"
            messages.append({"role": "tool", "tool_call_id": c["id"] or f"call_{n_calls}",
                             "content": str(result)[:6000]})
    json.dump(messages, open(os.path.join(a.out, f"transcript-{name}.json"), "w", encoding="utf-8"), indent=1)
    walls = [t["wall"] for t in turns]
    ttfts = [t["ttft"] for t in turns if t["ttft"] is not None]
    return {"status": status, "final": final[-600:], "turns": len(turns), "tool_calls": n_calls,
            "bad_calls": bad, "tool_errors": tool_err, "wall_total_s": round(sum(walls), 1),
            "wall_median_s": round(statistics.median(walls), 2) if walls else None,
            "ttft_median_s": round(statistics.median(ttfts), 2) if ttfts else None,
            "ttft_max_s": round(max(ttfts), 2) if ttfts else None,
            "prompt_tokens_max": max([t["prompt_tokens"] or 0 for t in turns], default=0),
            "completion_tokens_total": sum(t["completion_tokens"] or 0 for t in turns), "per_turn": turns}


def fresh(name):
    root = os.path.realpath(os.path.join(a.out, "workspace-" + name))
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    return root


WORDS = ("maple cobalt lantern orchid granite ember willow saffron harbor quartz meadow falcon cedar "
         "indigo pebble marigold tundra copper juniper lagoon thistle onyx prairie birch velvet canyon "
         "sorrel basalt heron clover").split()


def task_trail():
    root, rng = fresh("trail"), random.Random(7)
    words = WORDS[:a.trail_len]
    names = [f"notes/{rng.getrandbits(24):06x}.txt" for _ in words]
    names[0] = "START.txt"
    os.makedirs(os.path.join(root, "notes"))
    for i, w in enumerate(words):
        nxt = f"Next file: {names[i + 1]}" if i + 1 < len(words) else "This is the last file."
        open(os.path.join(root, names[i]), "w").write(f"Code word {i + 1}: {w}\n{nxt}\n")
    for _ in range(12):  # files that are not on the chain
        open(os.path.join(root, f"notes/{rng.getrandbits(24):06x}.txt"), "w").write("Not part of the chain.\n")
    impl, specs = workspace_tools(root, False)
    res = run_task("trail", "START.txt begins a chain of files. Each file gives one code word and the path of "
                   "the next file. Follow the chain to its end, then reply with every code word in order, "
                   "comma separated, on one line that starts with ANSWER:", impl, specs)
    got = [w.strip().lower() for w in res["final"].split("ANSWER:")[-1].strip().splitlines()[0].split(",")] \
        if "ANSWER:" in res["final"] else []
    res["ok"] = got == words
    return res


PRICING = '''def line_total(qty, unit_price, discount_pct=0):
    """Total for one line. discount_pct is a percentage from 0 to 100."""
    return qty * unit_price * (1 - discount_pct)


def order_total(lines):
    """lines is a list of (qty, unit_price, discount_pct). Returns the total rounded to 2 places."""
    total = 0
    for qty, price, disc in lines:
        total += line_total(qty, price, disc)
    return round(total, 2)


def cheapest(lines):
    """Index of the line with the lowest total. A tie goes to the first such line."""
    best = 0
    for i in range(1, len(lines)):
        if line_total(*lines[i]) <= line_total(*lines[best]):
            best = i
    return best
'''
TESTS = '''import unittest

from pricing import cheapest, line_total, order_total


class PricingTest(unittest.TestCase):
    def test_no_discount(self):
        self.assertEqual(line_total(3, 2.5), 7.5)

    def test_discount_is_a_percentage(self):
        self.assertAlmostEqual(line_total(2, 10.0, 25), 15.0)

    def test_order_total(self):
        self.assertEqual(order_total([(1, 10.0, 0), (2, 5.0, 50)]), 15.0)

    def test_cheapest(self):
        self.assertEqual(cheapest([(1, 9.0, 0), (1, 4.0, 0), (1, 7.0, 0)]), 1)

    def test_cheapest_tie_goes_to_the_first(self):
        self.assertEqual(cheapest([(1, 4.0, 0), (2, 2.0, 0), (1, 9.0, 0)]), 0)


if __name__ == "__main__":
    unittest.main()
'''


def task_fix():
    root = fresh("fix")
    open(os.path.join(root, "pricing.py"), "w").write(PRICING)
    open(os.path.join(root, "test_pricing.py"), "w").write(TESTS)
    impl, specs = workspace_tools(root, True)
    res = run_task("fix", "The unit tests in this workspace fail. Find out why, fix the code without changing "
                   "the tests, run the tests until they pass, then reply DONE.", impl, specs)
    same_tests = open(os.path.join(root, "test_pricing.py")).read() == TESTS
    try:
        final = impl["run_tests"]()
    except subprocess.TimeoutExpired:
        final = "timed out"
    res["tests_unchanged"] = same_tests
    res["ok"] = same_tests and final.rstrip().endswith("exit code 0")
    return res


FACTS = {"decode_code_tok_s": 37.69, "decode_structured_tok_s": 54.58, "download_gb": 371.84,
         "per_spark_gb": 79.74, "bits_per_weight": 3.12}


def task_facts():
    root = fresh("facts")
    impl, specs = workspace_tools(root, False)
    res = run_task("facts", "From the reference document in your instructions, write summary.json in the "
                   "workspace as one JSON object with exactly these keys: decode_code_tok_s and "
                   "decode_structured_tok_s (the release build's decode medians on code and on structured "
                   "output), download_gb (the size of the whole download), per_spark_gb (what one Spark "
                   "holds), bits_per_weight (the text model), all as numbers, and slow_side_note (a string: "
                   'the full sentence group that begins "Prefill is still the slow side", copied exactly). '
                   "Then read the file back to check it and reply DONE.", impl, specs)
    try:
        d = json.load(open(os.path.join(root, "summary.json"), encoding="utf-8"))
        wrong = [k for k, v in FACTS.items() if not isinstance(d.get(k), (int, float)) or abs(d[k] - v) > 0.005]
        note = str(d.get("slow_side_note", ""))
        if "155.71" not in note or "343.59" not in note:
            wrong.append("slow_side_note")
        res["wrong_keys"], res["ok"] = wrong, not wrong
    except (OSError, ValueError, AttributeError) as e:
        res["wrong_keys"], res["ok"] = [f"summary.json unreadable: {e}"], False
    return res


TASKS = {"trail": task_trail, "fix": task_fix, "facts": task_facts}
say(f"== agent_loop {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} url={base} model={MODEL} "
    f"stream={not a.no_stream} reasoning={a.reasoning or 'server default'} doc_chars={len(DOC)}")
summary = {"url": base, "model": MODEL, "stream": not a.no_stream, "reasoning": a.reasoning or None,
           "doc_chars": len(DOC), "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tasks": {}}
for t in [x for x in a.tasks.split(",") if x]:
    res = TASKS[t]()
    summary["tasks"][t] = res
    say(f"{t}: ok={res['ok']} status={res['status'][:120]} turns={res['turns']} calls={res['tool_calls']} "
        f"bad={res['bad_calls']} tool_errors={res['tool_errors']} wall={res['wall_total_s']}s "
        f"ttft_median={res['ttft_median_s']}s ttft_max={res['ttft_max_s']}s prompt_max={res['prompt_tokens_max']}")
    json.dump(summary, open(os.path.join(a.out, "agent_loop.json"), "w", encoding="utf-8"), indent=1)
