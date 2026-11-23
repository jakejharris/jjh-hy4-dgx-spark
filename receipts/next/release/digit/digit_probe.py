#!/usr/bin/env python3
"""Why did the kit's JPEG probe read a blocky 74 as 14 with drafts on, and 74 with drafts off?

Asks the kit check's own question about the same picture as JPEG and as PNG (the JPEG's decoded pixels),
with drafts on and off, then once more each way with log probabilities, and records every reply.

usage: digit_probe.py --url http://HOST:PORT --jpg FILE.jpg --png FILE.png --out DIR
Exit code is always 0: this records, it does not judge.
"""
import argparse
import base64
import json
import pathlib
import time
import urllib.error
import urllib.request

QUESTION = ("What two-digit number is written in the center, and what colors are the left half and right half? "
            "Reply in one short sentence.")


def call(url, body, timeout):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    except Exception as e:  # recorded, not raised
        return {"status": None, "error": repr(e), "seconds": round(time.time() - t0, 2)}
    out = {"status": status, "seconds": round(time.time() - t0, 2)}
    try:
        d = json.loads(raw)
    except ValueError:
        out["raw"] = raw[:2000].decode("utf-8", "replace")
        return out
    if "error" in d:
        out["error_message"] = d["error"].get("message") if isinstance(d["error"], dict) else d["error"]
        return out
    choice = (d.get("choices") or [{}])[0]
    out["answer"] = (choice.get("message") or {}).get("content")
    out["finish_reason"] = choice.get("finish_reason")
    out["usage"] = d.get("usage")
    lp = (choice.get("logprobs") or {}).get("content")
    if lp:  # keep the first 12 positions: token, its logprob, and the alternatives offered
        out["logprobs_head"] = [
            {"token": p.get("token"), "logprob": p.get("logprob"),
             "top": [(t.get("token"), t.get("logprob")) for t in (p.get("top_logprobs") or [])]}
            for p in lp[:12]
        ]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--jpg", required=True)
    ap.add_argument("--png", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=60)
    a = ap.parse_args()
    base = a.url.rstrip("/")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(base + "/v1/models", timeout=30) as r:
        model = json.load(r)["data"][0]["id"]
    images = {"jpg": ("image/jpeg", pathlib.Path(a.jpg).read_bytes()), "png": ("image/png", pathlib.Path(a.png).read_bytes())}
    rows = []
    for logprobs in (False, True):
        for fmt, (mime, raw) in images.items():
            for draft in (True, False):
                url = f"data:{mime};base64,{base64.b64encode(raw).decode()}"
                body = {"model": model, "max_tokens": 48, "temperature": 0, "top_p": 1, "reasoning_effort": "no_think",
                        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                                    {"type": "text", "text": QUESTION}]}]}
                if not draft:
                    body["draft"] = False
                if logprobs:
                    body.update(logprobs=True, top_logprobs=5)
                res = call(base + "/v1/chat/completions", body, a.timeout)
                row = {"format": fmt, "draft": draft, "logprobs": logprobs, **res}
                rows.append(row)
                print(f"{fmt} draft={'on' if draft else 'off'} logprobs={'yes' if logprobs else 'no'}:",
                      res.get("status"), res.get("answer") or res.get("error_message") or res.get("error"))
    (out / "digit_probe.json").write_text(json.dumps(
        {"model": model, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "question": QUESTION, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
