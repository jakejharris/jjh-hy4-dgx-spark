#!/usr/bin/env python3
"""Image checks for a running Hy4 server with vision on (SERVE_VISION=1). Python standard library only.

    python3 bench/vision_check.py --url http://127.0.0.1:8080 --out "$LOGDIR/vision_check.json"

1. Three known-answer images from bench/images/ (a text card, a 2x2 colour grid, a bar chart), greedy, thinking
   off. Each question asks for a fixed answer layout, and the score is strict: the text card by each field's exact
   value, the grid by POSITION (top-left must be red, and so on), the chart by DAY (MON must be 20, and so on, plus
   the highest and lowest day). A right word in the wrong place fails. Keywords appearing anywhere do not count.
2. Probes for the contract rows, each recorded with its HTTP status, the server's error text, prompt tokens and time:
   formats (PNG, BMP, and the text card as a JPEG and the bar chart as a WebP, asked and scored exactly like their
   PNG known answers; either outcome is recorded, accepted and answered right or wrong, or refused, so the same check
   serves a server with or without the two image decoders), sizes up to 8192x8192 (largest accepted, and what happens over the image-token cap),
   images per request (1, 2, 4, 8, 16), a malformed data URL (bad base64, and a truncated PNG), and an http image URL
   (which the server must refuse rather than fetch; the host is under .invalid, so nothing is fetched).
3. A text request after the probes, to show the server still answers.

It writes one JSON receipt (--out). Put it in the launcher's log folder (LOGDIR) and launch/collect_receipts.py
copies it into the receipt folder and summary.json. Exit 0 when all three known answers pass and the server still
answers, 1 otherwise, 2 when the server cannot be reached at all.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import pathlib
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib

HERE = pathlib.Path(__file__).resolve().parent
IMAGES = HERE / "images"
# JPEG and WebP format probes: label, MIME type, file, and the known answer (KNOWN id) whose question and strict rules
# judge it. The files are the text card and the bar chart again, the same pictures as the PNGs in another format.
CODEC_PROBES = (
    ("jpeg", "image/jpeg", "text_card.jpg", "text_card"),
    ("webp", "image/webp", "bar_chart.webp", "bar_chart"),
)


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def png_bytes(w: int, h: int, colour=(67, 160, 71)) -> bytes:
    """A solid-colour RGB PNG, made with zlib (no imaging library)."""
    row = b"\x00" + bytes(colour) * w
    raw = zlib.compress(row * h, 9)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", raw) + chunk(b"IEND", b""))


def bmp_bytes(w: int, h: int, colour=(67, 160, 71)) -> bytes:
    """A solid-colour 24-bit BMP."""
    pad = (4 - (w * 3) % 4) % 4
    row = bytes((colour[2], colour[1], colour[0])) * w + b"\x00" * pad
    data = row * h
    header = b"BM" + struct.pack("<IHHI", 54 + len(data), 0, 0, 54)
    info = struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, len(data), 2835, 2835, 0, 0)
    return header + info + data


def data_url(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def get_model(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=timeout) as r:
        return json.loads(r.read())["data"][0]["id"]


def ask(url: str, model: str, parts: list, max_tokens: int, timeout: float) -> dict:
    """One streamed chat request. Returns status, answer text, prompt tokens, error text and seconds; never raises."""
    payload = {"model": model, "messages": [{"role": "user", "content": parts}], "max_tokens": max_tokens,
               "temperature": 0, "top_p": 1, "stream": True, "stream_options": {"include_usage": True},
               "reasoning_effort": "no_think"}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.monotonic()
    out = {"status": None, "answer": "", "prompt_tokens": None, "error": None, "ttft_s": None}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out["status"] = resp.status
            text = []
            for line in resp:
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                obj = json.loads(body)
                if obj.get("usage"):
                    out["prompt_tokens"] = obj["usage"].get("prompt_tokens")
                if obj.get("error"):
                    out["error"] = json.dumps(obj["error"])[:400]
                for ch in obj.get("choices") or []:
                    delta = ch.get("delta") or {}
                    if out["ttft_s"] is None and (delta.get("content") or delta.get("reasoning_content")):
                        out["ttft_s"] = round(time.monotonic() - t0, 3)   # client's time to first output
                    text.append(delta.get("content") or "")
            out["answer"] = "".join(text)
    except urllib.error.HTTPError as e:
        out["status"] = e.code
        out["error"] = e.read().decode("utf-8", "replace")[:400]
    except Exception as e:  # connection refused, reset, timeout: recorded, not raised
        out["error"] = f"{type(e).__name__}: {e}"[:400]
    out["seconds"] = round(time.monotonic() - t0, 2)
    return out


def clean(v: str) -> str:
    return v.strip().strip("*`_.,;").strip()


def field(answer: str, key: str) -> str | None:
    m = re.search(r"(?im)^[\s*`_-]*" + key + r"[\s*`_]*[:=]\s*(.+?)\s*$", answer)
    return clean(m.group(1)) if m else None


KNOWN = [
    {"id": "text_card", "file": "text_card.png",
     "question": ("Read the card. Reply with exactly these three lines and nothing else:\n"
                  "CODE: <code>\nPORT: <port>\nSTATUS: <status>"),
     "expect": {"CODE": "KESTREL-4729", "PORT": "18080", "STATUS": "READY"}},
    {"id": "color_grid", "file": "color_grid.png",
     "question": ("This image is a 2x2 grid of solid coloured squares. Reply with exactly these four lines and nothing "
                  "else, one colour word each:\ntop-left: <colour>\ntop-right: <colour>\nbottom-left: <colour>\n"
                  "bottom-right: <colour>"),
     "expect": {"top-left": "red", "top-right": "blue", "bottom-left": "green", "bottom-right": "yellow"}},
    {"id": "bar_chart", "file": "bar_chart.png",
     "question": ("Read the bar chart. Reply with exactly these six lines and nothing else, numbers as digits and days "
                  "as three letters:\nMON: <value>\nTUE: <value>\nWED: <value>\nTHU: <value>\nHIGHEST: <day>\n"
                  "LOWEST: <day>"),
     "expect": {"MON": "20", "TUE": "75", "WED": "40", "THU": "90", "HIGHEST": "THU", "LOWEST": "MON"}},
]


def score(task: dict, answer: str) -> dict:
    got = {}
    for key, want in task["expect"].items():
        v = field(answer, key.replace("-", r"[\s-]*"))
        if v is not None and task["id"] == "color_grid":
            v = v.split()[0].lower() if v.split() else v
        if v is not None and key in ("HIGHEST", "LOWEST"):
            v = v[:3].upper()
        if v is not None and task["id"] == "text_card":
            v = v.upper()
        got[key] = v
    wrong = {k: {"want": w, "got": got[k]} for k, w in task["expect"].items() if got[k] != w}
    return {"parsed": got, "wrong": wrong, "pass": not wrong}


def probe(url, model, label, parts, max_tokens, timeout) -> dict:
    r = ask(url, model, parts, max_tokens, timeout)
    r["label"] = label
    r["accepted"] = r["status"] == 200 and not r["error"]
    r["answer"] = r["answer"][:200]
    print(f"  probe {label}: status {r['status']}, accepted {r['accepted']}, prompt_tokens {r['prompt_tokens']}, "
          f"{r['seconds']} s{', error ' + r['error'][:120] if r['error'] else ''}")
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", required=True, help="server base URL, e.g. http://127.0.0.1:8080")
    ap.add_argument("--out", required=True, help="receipt JSON path; put it in the launcher's LOGDIR")
    ap.add_argument("--model", default="", help="model id (default: the first one /v1/models lists)")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--skip-probes", action="store_true", help="known answers and the alive check only")
    a = ap.parse_args()

    rec = {"tool": "bench/vision_check.py", "url": a.url, "utc_start": utc()}
    try:
        model = a.model or get_model(a.url, 30)
    except Exception as e:
        print(f"ERROR: cannot reach {a.url}/v1/models: {e}", file=sys.stderr)
        return 2
    rec["model"] = model

    print("== known answers")
    known = []
    for t in KNOWN:
        raw = (IMAGES / t["file"]).read_bytes()
        r = ask(a.url, model, [{"type": "image_url", "image_url": {"url": data_url(raw, "image/png")}},
                               {"type": "text", "text": t["question"]}], 128, a.timeout)
        s = score(t, r["answer"])
        known.append({"id": t["id"], "image": f"bench/images/{t['file']}", "question": t["question"],
                      "expect": t["expect"], **r, **s})
        print(f"  {t['id']}: {'PASS' if s['pass'] else 'FAIL'} status {r['status']} prompt_tokens {r['prompt_tokens']}"
              f"{'' if s['pass'] else ' wrong ' + json.dumps(s['wrong'])}")
    rec["known_answers"] = known
    rec["known_pass"] = sum(k["pass"] for k in known)
    rec["known_total"] = len(known)

    probes = {}
    if not a.skip_probes:
        q = [{"type": "text", "text": "What colour is this image? One word."}]
        grid = (IMAGES / "color_grid.png").read_bytes()
        print("== probes: formats")
        probes["formats"] = [
            probe(a.url, model, "png", [{"type": "image_url", "image_url": {"url": data_url(grid, "image/png")}}] + q, 8, a.timeout),
            probe(a.url, model, "bmp", [{"type": "image_url", "image_url": {"url": data_url(bmp_bytes(256, 256), "image/bmp")}}] + q, 8, a.timeout),
        ]
        for label, mime, fname, tid in CODEC_PROBES:
            t = next(k for k in KNOWN if k["id"] == tid)
            r = ask(a.url, model, [{"type": "image_url", "image_url": {"url": data_url((IMAGES / fname).read_bytes(), mime)}},
                                   {"type": "text", "text": t["question"]}], 128, a.timeout)
            s = score(t, r["answer"])
            r.update(label=label, accepted=r["status"] == 200 and not r["error"], image=f"bench/images/{fname}",
                     judged_as=tid, expect=t["expect"], parsed=s["parsed"], wrong=s["wrong"])
            r["answered_right"] = r["accepted"] and s["pass"]
            print(f"  probe {label}: status {r['status']}, prompt_tokens {r['prompt_tokens']}, " +
                  (("answered right" if r["answered_right"] else "answered wrong " + json.dumps(s["wrong"]))
                   if r["accepted"] else f"refused, {str(r['error'])[:120]}"))
            probes["formats"].append(r)
        print("== probes: sizes")
        probes["sizes"] = [probe(a.url, model, f"{w}x{h}", [{"type": "image_url", "image_url": {"url": data_url(png_bytes(w, h), "image/png")}}] + q, 8, a.timeout)
                           for w, h in ((1440, 900), (2880, 1800), (5760, 3600), (8192, 8192))]
        print("== probes: images per request")
        per = []
        for n in (1, 2, 4, 8, 16):
            per.append(probe(a.url, model, f"{n} images", [{"type": "image_url", "image_url": {"url": data_url(grid, "image/png")}}] * n + q, 8, a.timeout))
            if not per[-1]["accepted"]:
                break
        probes["per_request"] = per
        print("== probes: malformed and remote")
        probes["malformed"] = [
            probe(a.url, model, "bad base64", [{"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!not-base64!!!"}}] + q, 8, a.timeout),
            probe(a.url, model, "truncated png", [{"type": "image_url", "image_url": {"url": data_url(grid[:200], "image/png")}}] + q, 8, a.timeout),
        ]
        probes["remote_url"] = [probe(a.url, model, "http url", [{"type": "image_url", "image_url": {"url": "http://example.invalid/image.png"}}] + q, 8, a.timeout)]
    rec["probes"] = probes

    alive = ask(a.url, model, [{"type": "text", "text": "Reply with the single word: ready"}], 8, 120)
    rec["alive_after"] = {"status": alive["status"], "answer": alive["answer"][:40], "error": alive["error"]}
    rec["server_alive_after"] = alive["status"] == 200 and not alive["error"]
    rec["utc_end"] = utc()
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(f"== known answers {rec['known_pass']} of {rec['known_total']}; server answering after probes: "
          f"{rec['server_alive_after']}; receipt {out}")
    return 0 if rec["known_pass"] == rec["known_total"] and rec["server_alive_after"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
