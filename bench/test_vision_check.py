#!/usr/bin/env python3
"""Offline test of bench/vision_check.py against a fake server (standard library only).

    python3 -I bench/test_vision_check.py

The fake server answers the text card and the chart right, and the colour grid with the right four colours in the
wrong places. The check must pass the first two, fail the grid by position, record every probe, write the receipt
and exit 1. Then the grid answer is fixed and the run must exit 0.
"""

from __future__ import annotations

import http.server
import json
import pathlib
import subprocess
import sys
import tempfile
import threading

HERE = pathlib.Path(__file__).resolve().parent
FAILS: list[str] = []
STATE = {"grid": "top-left: blue\ntop-right: red\nbottom-left: green\nbottom-right: yellow"}


def check(cond: bool, what: str) -> None:
    print(f"  {'ok  ' if cond else 'FAIL'} {what}")
    if not cond:
        FAILS.append(what)


class Fake(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": "hy4-test"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        parts = req["messages"][0]["content"]
        text = parts[-1]["text"] if isinstance(parts, list) else parts
        urls = [p["image_url"]["url"] for p in parts if isinstance(p, dict) and p.get("type") == "image_url"]
        if not STATE.get("codecs") and any(u.startswith(("data:image/jpeg", "data:image/webp")) for u in urls):
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"image 1: libturbojpeg.so.0 is required for JPEG inputs"}}')
            return
        if any(u.startswith("http") for u in urls) or any("!!!" in u for u in urls):
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"image refused"}}')
            return
        if len(urls) > 4:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"too many images"}}')
            return
        webp = any(u.startswith("data:image/webp") for u in urls)
        if "CODE:" in text:
            ans = "**CODE:** KESTREL-4729\nPORT: 18080\nSTATUS: READY"
        elif "top-left" in text:
            ans = STATE["grid"]
        elif "MON:" in text and webp and STATE.get("webp_wrong"):
            ans = "MON: 20\nTUE: 57\nWED: 40\nTHU: 90\nHIGHEST: THU\nLOWEST: MON"   # one digit pair swapped
        elif "MON:" in text:
            ans = "MON: 20\nTUE: 75\nWED: 40\nTHU: 90\nHIGHEST: Thursday\nLOWEST: MON"
        else:
            ans = "green"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for piece in (ans[: len(ans) // 2], ans[len(ans) // 2:]):
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}).encode() + b"\n\n")
        self.wfile.write(b"data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 100 * max(1, len(urls))}}).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")


def run(url: str, out: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(HERE / "vision_check.py"), "--url", url, "--out", str(out)],
                          capture_output=True, text=True)


def main() -> int:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    with tempfile.TemporaryDirectory(prefix="hy4-vision-test-") as td:
        out = pathlib.Path(td) / "vision_check.json"
        print("== grid answered with the right colours in the wrong places")
        p = run(url, out)
        check(p.returncode == 1, f"exit 1 (got {p.returncode}) {p.stderr.strip()[-200:]}")
        r = json.loads(out.read_text())
        k = {x["id"]: x for x in r["known_answers"]}
        check(k["text_card"]["pass"] and k["bar_chart"]["pass"], "text card and chart pass (markdown key, 'Thursday' read as THU)")
        check(not k["color_grid"]["pass"] and set(k["color_grid"]["wrong"]) == {"top-left", "top-right"},
              f"grid fails by position: wrong {sorted(k['color_grid']['wrong'])}")
        pr = r["probes"]
        check([x["accepted"] for x in pr["per_request"]] == [True, True, True, False], "per-request stops at the first refusal (8)")
        check(not pr["remote_url"][0]["accepted"] and pr["remote_url"][0]["status"] == 400 and "refused" in pr["remote_url"][0]["error"],
              "http URL refusal recorded with status and error text")
        check(not pr["malformed"][0]["accepted"] and len(pr["sizes"]) == 4 and pr["sizes"][0]["prompt_tokens"] == 100,
              "malformed refusal and the four sizes recorded, with prompt tokens")
        check(r["server_alive_after"] is True and r["known_pass"] == 2, "alive check and known_pass recorded")
        fm = {x["label"]: x for x in pr["formats"]}
        check([x["label"] for x in pr["formats"]] == ["png", "bmp", "jpeg", "webp"] and not fm["jpeg"]["accepted"]
              and fm["jpeg"]["status"] == 400 and "libturbojpeg" in fm["jpeg"]["error"] and fm["webp"]["answered_right"] is False,
              "no decoders: JPEG and WebP refused and recorded with the server's text, not counted as right")
        print("== grid answered right")
        STATE["grid"] = "Top-left: Red\nTop-right: Blue.\nBottom-left: green\nBottom-right: yellow"
        STATE["codecs"] = True
        STATE["webp_wrong"] = True
        p = run(url, out)
        check(p.returncode == 0 and json.loads(out.read_text())["known_pass"] == 3, f"exit 0 and 3 of 3 (got {p.returncode})")
        fm = {x["label"]: x for x in json.loads(out.read_text())["probes"]["formats"]}
        check(fm["jpeg"]["accepted"] and fm["jpeg"]["answered_right"] is True and fm["jpeg"]["judged_as"] == "text_card"
              and fm["jpeg"]["image"] == "bench/images/text_card.jpg",
              "with decoders: the text card as JPEG, judged by the text card's strict rules, right")
        check(fm["webp"]["accepted"] and fm["webp"]["answered_right"] is False and set(fm["webp"]["wrong"]) == {"TUE"},
              f"the bar chart as WebP with one value wrong counts as wrong, by DAY: {fm['webp'].get('wrong')}")
        STATE["webp_wrong"] = False
        p = run(url, out)
        fm = {x["label"]: x for x in json.loads(out.read_text())["probes"]["formats"]}
        check(fm["webp"]["answered_right"] is True and fm["webp"]["judged_as"] == "bar_chart", "the bar chart as WebP, read right, passes")
        p = run("http://127.0.0.1:9", out)
        check(p.returncode == 2, f"unreachable server exits 2 (got {p.returncode})")
    srv.shutdown()
    print("== " + ("PASS" if not FAILS else f"FAIL ({len(FAILS)}): " + "; ".join(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
