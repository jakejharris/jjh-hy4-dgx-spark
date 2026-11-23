#!/usr/bin/env python3
"""Time to first token on short and mid-size cold prompts. Stdlib only. Always exits 0.

For each target size it sends REPEATS cold requests (a fresh random salt first, so no prefix is cached),
streaming, thinking off, temperature 0, 8 output tokens, and times from the request being sent to the
first streamed piece of output. The prompt's real token count is read from the server's usage: from the
stream if it carries one, otherwise from one replay of the same prompt (1 token, not timed).

usage: python3 -I ttft_sweep.py --url http://127.0.0.1:8080 --text FILE --out DIR
                                [--sizes 128,256,300,512,1024,2048,4096] [--repeats 3] [--model NAME]
Writes DIR/ttft_sweep.json and DIR/ttft_sweep.log; prints one line per request and a median per size.
"""
import argparse
import json
import os
import secrets
import statistics
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--text", required=True, help="a long plain text file to cut prompts from")
ap.add_argument("--out", required=True)
ap.add_argument("--sizes", default="128,256,300,512,1024,2048,4096")
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--model", default="")
ap.add_argument("--chars-per-token", type=float, default=4.0, help="first guess; refined from usage")
ap.add_argument("--timeout", type=float, default=300)
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
text = open(a.text, encoding="utf-8", errors="replace").read()
base = a.url.rstrip("/")
log = open(os.path.join(a.out, "ttft_sweep.log"), "a", encoding="utf-8")


def say(line):
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()


def model_name():
    if a.model:
        return a.model
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=30) as r:
            return json.load(r)["data"][0]["id"]
    except Exception as e:  # noqa: BLE001
        say(f"models lookup failed: {e!r}")
        return "default"


MODEL = model_name()
OVERHEAD_TOKENS = 40  # chat template, salt line and the instruction; refined below


def body(prompt, stream, max_tokens):
    b = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
         "max_tokens": max_tokens, "reasoning_effort": "no_think", "stream": stream}
    if stream:
        b["stream_options"] = {"include_usage": True}
    return json.dumps(b).encode()


def post(payload):
    return urllib.request.Request(base + "/v1/chat/completions", data=payload,
                                  headers={"Content-Type": "application/json"})


def one(prompt):
    """Return (ttft_s, wall_s, prompt_tokens, cached_tokens, first_piece, error)."""
    t0 = time.perf_counter()
    ttft = None
    first = ""
    usage = None
    try:
        with urllib.request.urlopen(post(body(prompt, True, 8)), timeout=a.timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except ValueError:
                    continue
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning") or ""
                    if piece and ttft is None:
                        ttft = time.perf_counter() - t0
                        first = piece
        wall = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        return None, time.perf_counter() - t0, None, None, "", repr(e)
    if usage is None:  # read the size from one replay, not timed
        try:
            with urllib.request.urlopen(post(body(prompt, False, 1)), timeout=a.timeout) as r:
                usage = json.load(r).get("usage")
        except Exception as e:  # noqa: BLE001
            say(f"  usage replay failed: {e!r}")
    pt = (usage or {}).get("prompt_tokens")
    cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is None:
        cached = (usage or {}).get("cached_prompt_tokens")
    return ttft, wall, pt, cached, first, None


rows = []
cpt = a.chars_per_token
for size in [int(s) for s in a.sizes.split(",") if s.strip()]:
    for rep in range(1, a.repeats + 1):
        salt = secrets.token_hex(8)
        n_chars = max(0, int((size - OVERHEAD_TOKENS) * cpt))
        start = secrets.randbelow(max(1, len(text) - n_chars - 1))
        prompt = (f"Run id {salt}. Read the text below, then reply with the single word: done.\n\n"
                  + text[start:start + n_chars])
        ttft, wall, pt, cached, first, err = one(prompt)
        rows.append({"target_tokens": size, "repeat": rep, "prompt_tokens": pt, "cached_prompt_tokens": cached,
                     "ttft_s": ttft, "wall_s": wall, "first_piece": first, "error": err, "salt": salt,
                     "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        say(f"target={size} rep={rep} prompt_tokens={pt} cached={cached} "
            f"ttft_s={None if ttft is None else round(ttft, 3)} wall_s={round(wall, 3)} err={err}")
        if pt and n_chars > 200:  # refine the guess so later sizes land closer to their target
            cpt = max(2.0, min(8.0, n_chars / max(1, pt - OVERHEAD_TOKENS)))

summary = []
for size in sorted({r["target_tokens"] for r in rows}):
    got = [r for r in rows if r["target_tokens"] == size and r["ttft_s"] is not None]
    if not got:
        summary.append({"target_tokens": size, "n": 0})
        continue
    summary.append({"target_tokens": size, "n": len(got),
                    "prompt_tokens": sorted(r["prompt_tokens"] for r in got if r["prompt_tokens"]),
                    "ttft_s_median": round(statistics.median(r["ttft_s"] for r in got), 3),
                    "ttft_s_min": round(min(r["ttft_s"] for r in got), 3),
                    "ttft_s_max": round(max(r["ttft_s"] for r in got), 3)})
    say(f"SIZE {size}: n={len(got)} prompt_tokens={summary[-1]['prompt_tokens']} "
        f"ttft median={summary[-1]['ttft_s_median']} s (min {summary[-1]['ttft_s_min']}, max {summary[-1]['ttft_s_max']})")

json.dump({"url": a.url, "model": MODEL, "settings": {"temperature": 0, "max_tokens": 8, "reasoning_effort": "no_think",
           "stream": True, "repeats": a.repeats, "text": a.text}, "rows": rows, "summary": summary},
          open(os.path.join(a.out, "ttft_sweep.json"), "w", encoding="utf-8"), indent=1)
