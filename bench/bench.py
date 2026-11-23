#!/usr/bin/env python3
"""Unified OpenAI-compatible benchmark client for baseline (llama-server) and TensorFold."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _check_finite(name: str, val: object) -> None:
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        if math.isnan(val) or math.isinf(val):
            raise RuntimeError(f"non-finite metric {name}={val!r}")


def _stream_request(url: str, path: str, payload: dict, timeout_s: float) -> dict:
    req = urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t_start = time.perf_counter()
    t_first = None
    chunks: list[str] = []
    content_chunks: list[str] = []
    reasoning_chunks: list[str] = []
    usage = None
    timings = None
    tf_meta = None
    spec_meta = None
    finish_reason = None
    http_status = None

    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        http_status = getattr(resp, "status", resp.getcode())
        if http_status != 200:
            raise RuntimeError(f"HTTP status {http_status} != 200 on {path}")
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            obj = json.loads(body)
            if "usage" in obj and obj["usage"]:
                usage = obj["usage"]
            if "timings" in obj and obj["timings"]:
                timings = obj["timings"]
            if "tensorfold" in obj and obj["tensorfold"]:
                tf_meta = obj["tensorfold"]
            if "speculative" in obj and obj["speculative"]:
                spec_meta = obj["speculative"]
            choices = obj.get("choices") or []
            if choices:
                c0 = choices[0]
                if c0.get("finish_reason"):
                    finish_reason = c0["finish_reason"]
                delta = c0.get("delta") or {}
                rc = delta.get("reasoning_content") or ""
                cc = delta.get("content") or ""
                tx = c0.get("text") or ""
                if rc:
                    reasoning_chunks.append(rc)
                if cc:
                    content_chunks.append(cc)
                elif tx:
                    content_chunks.append(tx)
                piece = cc or rc or tx
                if piece:
                    if t_first is None:
                        t_first = time.perf_counter()
                    chunks.append(piece)

    t_end = time.perf_counter()
    ttft_s = (t_first - t_start) if t_first is not None else (t_end - t_start)
    total_s = t_end - t_start
    gen_s = (t_end - t_first) if t_first is not None else 0.0
    prompt_n = (usage or {}).get("prompt_tokens") or (timings or {}).get("prompt_n")
    pred_n = (usage or {}).get("completion_tokens") or (timings or {}).get("predicted_n")
    pp_tps = (prompt_n / ttft_s) if (prompt_n and ttft_s > 0) else None
    tg_tps = (
        (tf_meta or {}).get("tokens_per_second")
        or (((pred_n - 1) / gen_s) if (pred_n and pred_n > 1 and gen_s > 0) else None)
    )
    decode_ms_per_tok = (1000.0 / tg_tps) if (tg_tps and tg_tps > 0) else None
    token_ids = (tf_meta or {}).get("token_ids")
    token_sha256 = (tf_meta or {}).get("token_sha256")

    full_content = "".join(content_chunks)
    full_reasoning = "".join(reasoning_chunks)
    if not chunks and not (token_ids and len(token_ids) > 0):
        raise RuntimeError(f"empty completion on {path} (pred_n={pred_n}, finish_reason={finish_reason!r})")
    if pred_n is not None and pred_n <= 0:
        raise RuntimeError(f"zero completion tokens on {path} (pred_n={pred_n})")
    for k, v in (
        ("ttft_s", ttft_s),
        ("total_s", total_s),
        ("pp_tps", pp_tps),
        ("tg_tps", tg_tps),
        ("decode_ms_per_tok", decode_ms_per_tok),
    ):
        _check_finite(k, v)
    if isinstance(tf_meta, dict):
        for k in ("tokens_per_second", "seconds", "prefill_seconds", "time_to_first_token"):
            _check_finite(f"tensorfold.{k}", tf_meta.get(k))
    if isinstance(spec_meta, dict):
        for k in ("acceptance_rate", "tokens_per_round"):
            _check_finite(f"speculative.{k}", spec_meta.get(k))

    return {
        "http_status": http_status,
        "endpoint": path,
        "prompt_n": prompt_n,
        "pred_n": pred_n,
        "ttft_s": round(ttft_s, 4),
        "total_s": round(total_s, 4),
        "pp_tps": round(pp_tps, 2) if pp_tps is not None else None,
        "tg_tps": round(tg_tps, 2) if tg_tps is not None else None,
        "decode_ms_per_tok": round(decode_ms_per_tok, 3) if decode_ms_per_tok is not None else None,
        "finish_reason": finish_reason,
        "usage": usage,
        "timings": timings,
        "tensorfold": tf_meta,
        "speculative": spec_meta,
        "token_sha256": token_sha256,
        "token_ids": token_ids,
        "content": full_content,
        "reasoning_content": full_reasoning,
        "text_preview": "".join(chunks)[:200],
    }


def run_once(
    url: str,
    model: str,
    prompt_text: str,
    max_tokens: int,
    seed: int,
    endpoint: str = "auto",
    timeout_s: float = 7200.0,
    reasoning_effort: str = "omit",
    return_token_ids: bool = True,
) -> dict:
    chat_payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
        "stream": True,
        "cache_prompt": False,
        "stream_options": {"include_usage": True},
    }
    if return_token_ids:
        chat_payload["return_token_ids"] = True
    if reasoning_effort != "omit":
        chat_payload["reasoning_effort"] = reasoning_effort
    comp_payload: dict = {
        "model": model,
        "prompt": prompt_text,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
        "stream": True,
    }
    if return_token_ids:
        comp_payload["return_token_ids"] = True
    if endpoint == "completions":
        return _stream_request(url, "/v1/completions", comp_payload, timeout_s)
    if endpoint == "chat":
        return _stream_request(url, "/v1/chat/completions", chat_payload, timeout_s)
    try:
        return _stream_request(url, "/v1/chat/completions", chat_payload, timeout_s)
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        if exc.code == 400 and "chat template" in body.lower():
            return _stream_request(url, "/v1/completions", comp_payload, timeout_s)
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="hy4")
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--endpoint", choices=("auto", "chat", "completions"), default="auto")
    ap.add_argument(
        "--reasoning-effort",
        choices=("omit", "no_think", "high"),
        default="omit",
        help="Thinking switch sent on /v1/chat/completions (default: omit, Tencent thinking-on default; pass no_think or high explicitly)",
    )
    ap.add_argument(
        "--no-token-ids",
        action="store_true",
        help="Do not send return_token_ids=true in request payload",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("BENCH_TIMEOUT_S", "7200")),
        help="Per-request HTTP socket timeout in seconds (default: BENCH_TIMEOUT_S or 7200)",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    prompt_text = args.prompt.read_text(encoding="utf-8")
    ret_ids = not args.no_token_ids
    try:
        for _ in range(args.warmup):
            run_once(
                args.url, args.model, prompt_text, args.max_tokens, args.seed,
                args.endpoint, args.timeout, args.reasoning_effort, ret_ids,
            )

        results = []
        for i in range(args.runs):
            res = run_once(
                args.url, args.model, prompt_text, args.max_tokens, args.seed,
                args.endpoint, args.timeout, args.reasoning_effort, ret_ids,
            )
            res["run"] = i + 1
            results.append(res)
            print(
                f"[{args.label}] run {i + 1}/{args.runs}: "
                f"pp={res['pp_tps']} tok/s, tg={res['tg_tps']} tok/s, ttft={res['ttft_s']}s"
            )
    except Exception as exc:
        print(f"BENCH FAIL [{args.label}]: {exc}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "label": args.label,
                "prompt": str(args.prompt),
                "reasoning_effort": args.reasoning_effort,
                "timeout_s": args.timeout,
                "runs": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
