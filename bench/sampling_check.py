#!/usr/bin/env python3
"""What a request that sends no temperature gets from a serve started by the kit's launcher (SERVE_TEMPERATURE, 0.7
by default), against greedy on the same serve. One request at a time; stdlib only.

    python3 -I bench/sampling_check.py --url http://HOST:SERVE_PORT --out DIR

Decode: bench/prompts/short-prose.txt and short-code.txt, each sent as the one user message, thinking on (no
reasoning_effort), a 2,048-token cap, three runs with no temperature and three with temperature 0, alternating,
greedy first. A run's rate is its completion tokens after the first over the client's time from the first streamed
piece (reasoning or answer) to the last. One greedy 32-token request on the prose prompt goes first as a warm-up.
Then the same decode runs with thinking off (reasoning_effort no_think), in the same order. With thinking off the
reply is the essay or the code itself, so those rates are answer speeds.
Finding 6: its coding request (the messages of receipts/thinking/budget_probes.json, P5, unchanged), reasoning_effort
high, an 8,192-token cap, seed 20260905, no temperature: once with "thinking_budget": 4096, then once without.
The no-temperature requests send no temperature, top_p, top_k or seed of their own, except finding 6's seed.

Writes DIR/sampling_check.json: every request as sent and what came back (finish reason, token counts from the
server's usage, the rate, the text), thinking on under "decode" and thinking off under "decode_nothink". Prints one
line a request. Exit 0 when every request got HTTP 200 and a finish reason, else 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
PROMPTS = (("prose", "bench/prompts/short-prose.txt"), ("code", "bench/prompts/short-code.txt"))
F6_SOURCE = "receipts/thinking/budget_probes.json"
SETTINGS = {"decode_tokens": 2048, "runs": 3, "warmup_tokens": 32, "f6_max_tokens": 8192, "f6_seed": 20260905,
            "f6_budget": 4096, "f6_reasoning_effort": "high", "prompts": [p for _, p in PROMPTS], "f6_messages": F6_SOURCE + " P5",
            "nothink_reasoning_effort": "no_think"}
FENCE = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)
GO_FILE = re.compile(r"[A-Za-z0-9_./-]+\.go\b")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def send(url: str, body: dict, timeout: float) -> dict:
    out = {"utc_start": utc(), "request": body, "http_status": None, "error": None, "finish_reason": None}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    content, reasoning, usage, server_tps = [], [], None, None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out["http_status"] = r.status
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                obj = json.loads(line[5:])
                usage = obj.get("usage") or usage
                server_tps = (obj.get("tensorfold") or {}).get("tokens_per_second", server_tps)
                for c in obj.get("choices") or []:
                    d = c.get("delta") or {}
                    rc, cc = d.get("reasoning_content") or "", d.get("content") or ""
                    if rc or cc:
                        last = time.perf_counter()
                        first = first or last
                    reasoning.append(rc)
                    content.append(cc)
                    out["finish_reason"] = c.get("finish_reason") or out["finish_reason"]
    except urllib.error.HTTPError as e:
        out["http_status"], out["error"] = e.code, e.read().decode("utf-8", "replace")[:2000]
    except Exception as e:  # noqa: BLE001 (recorded, not judged)
        out["error"] = repr(e)
    done = time.perf_counter()
    n = (usage or {}).get("completion_tokens")
    rt = ((usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
    out.update({
        "usage": usage, "completion_tokens": n, "reasoning_tokens": rt,
        "answer_tokens": n - rt if isinstance(n, int) and isinstance(rt, int) else None,
        "ttft_s": round(first - t0, 3) if first else None, "wall_s": round(done - t0, 3),
        "decode_tps": round((n - 1) / (last - first), 2) if n and n > 1 and first and last and last > first else None,
        "server_tokens_per_second": server_tps,
        "reasoning_chars": len("".join(reasoning)), "content_chars": len("".join(content)),
        "reasoning_content": "".join(reasoning), "content": "".join(content),
    })
    return out


def go_blocks(text: str) -> dict:
    files = sorted(set(GO_FILE.findall(text)))
    tests = [f for f in files if f.endswith("_test.go")]
    n = len(FENCE.findall(text))
    return {"fenced_blocks": n, "go_files": files, "two_go_blocks": n == 2 and bool(tests) and len(tests) < len(files)}


def say(label: str, r: dict) -> None:
    print(f"{r['utc_start']} {label}: HTTP {r['http_status']}, finish {r['finish_reason']}, {r['completion_tokens']} tokens "
          f"({r['reasoning_tokens']} reasoning), {r['decode_tps']} tok/s" + (f", error {r['error'][:120]}" if r["error"] else ""), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", required=True, help="the serve, http://HOST:PORT")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", help="default: the first id /v1/models lists")
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds a request may take")
    a = ap.parse_args()
    url = a.url.rstrip("/")
    model = a.model
    if not model:
        with urllib.request.urlopen(url + "/v1/models", timeout=30) as r:
            model = json.load(r)["data"][0]["id"]
    p5 = json.loads((KIT / F6_SOURCE).read_text(encoding="utf-8"))["P5"]["payload"]
    rec = {"tool": "bench/sampling_check.py", "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "url": url, "model": model, "utc_start": utc(), "settings": SETTINGS, "decode": {}, "decode_nothink": {}, "f6": {}}
    base = {"model": model, "stream": True, "stream_options": {"include_usage": True}}
    first_prompt = (KIT / PROMPTS[0][1]).read_text(encoding="utf-8")
    rec["warmup"] = send(url, base | {"messages": [{"role": "user", "content": first_prompt}], "max_tokens": SETTINGS["warmup_tokens"], "temperature": 0}, a.timeout)
    say("warm-up", rec["warmup"])
    for key, extra, tag in (("decode", {}, ""), ("decode_nothink", {"reasoning_effort": SETTINGS["nothink_reasoning_effort"]}, ", thinking off")):
        for name, path in PROMPTS:
            msgs = [{"role": "user", "content": (KIT / path).read_text(encoding="utf-8")}]
            runs = rec[key][name] = {"greedy": [], "default": []}
            for i in range(SETTINGS["runs"]):
                for kind in ("greedy", "default"):
                    body = base | {"messages": msgs, "max_tokens": SETTINGS["decode_tokens"]} | extra
                    if kind == "greedy":
                        body["temperature"] = 0
                    r = send(url, body, a.timeout)
                    runs[kind].append(r)
                    say(f"{name} {kind}{tag} run {i + 1}", r)
    for kind in ("budget", "none"):
        body = base | {"messages": p5["messages"], "reasoning_effort": SETTINGS["f6_reasoning_effort"],
                       "max_tokens": SETTINGS["f6_max_tokens"], "seed": SETTINGS["f6_seed"]}
        if kind == "budget":
            body["thinking_budget"] = SETTINGS["f6_budget"]
        r = send(url, body, a.timeout)
        r["go_blocks"] = go_blocks(r["content"])
        rec["f6"][kind] = r
        say(f"finding 6, {kind}", r)
    every = [rec["warmup"], *(r for d in ("decode", "decode_nothink") for w in rec[d].values() for k in w.values() for r in k), *rec["f6"].values()]
    rec["all_ok"] = all(r["http_status"] == 200 and r["finish_reason"] and not r["error"] for r in every)
    rec["utc_end"] = utc()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "sampling_check.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(f"{rec['utc_end']} wrote {a.out / 'sampling_check.json'}; all requests answered: {rec['all_ok']}")
    return 0 if rec["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
