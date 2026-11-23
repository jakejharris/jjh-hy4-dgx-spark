#!/usr/bin/env python3
"""Four small requests that say what the served build does with tool calls and structured output.

usage: tool_probes.py --url http://HOST:PORT --out DIR

Writes DIR/<name>.json per probe (status, error message or the reply's tool calls, finish reason, usage)
and DIR/summary.json. Exit code is always 0: this records, it does not judge.
"""
import argparse
import json
import pathlib
import time
import urllib.error
import urllib.request

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City and state"}},
            "required": ["city"],
        },
    },
}
ASK_TOOL = [{"role": "user", "content": "What is the weather in Austin, TX right now? Use the tool."}]
ASK_JSON = [{"role": "user", "content": "Name one city in Texas. Answer as JSON with the key city."}]
SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "city",
        "schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
PROBES = [
    ("tool_auto", {"messages": ASK_TOOL, "tools": [TOOL], "tool_choice": "auto"}),
    ("tool_required", {"messages": ASK_TOOL, "tools": [TOOL], "tool_choice": "required"}),
    ("format_json_schema", {"messages": ASK_JSON, "response_format": SCHEMA}),
    ("format_json_object", {"messages": ASK_JSON, "response_format": {"type": "json_object"}}),
]


def call(url, body, timeout):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    except Exception as e:  # timeout, refused connection: recorded, not raised
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
    msg = choice.get("message") or {}
    out.update(
        finish_reason=choice.get("finish_reason"),
        tool_calls=msg.get("tool_calls"),
        content=msg.get("content"),
        reasoning_chars=len(msg.get("reasoning_content") or ""),
        usage=d.get("usage"),
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=180)
    a = ap.parse_args()
    base = a.url.rstrip("/")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(base + "/v1/models", timeout=30) as r:
        model = json.load(r)["data"][0]["id"]
    summary = {"model": model, "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "probes": {}}
    for name, extra in PROBES:
        body = {"model": model, "max_tokens": 1024, **extra}
        res = call(base + "/v1/chat/completions", body, a.timeout)
        (out / f"{name}.json").write_text(json.dumps({"request": body, "result": res}, indent=1))
        summary["probes"][name] = res
        calls = res.get("tool_calls") or []
        print(name, res.get("status"), res.get("finish_reason"), res.get("error_message") or [c.get("function") for c in calls] or res.get("content"))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
