#!/usr/bin/env python3
"""Needle-in-a-haystack (NIAH) prompt generator, runner, and exact-match scorer for Hy4.

Subcommands:
  generate  Build a deterministic shippable logbook prompt sized to an exact token count
            using a model's real tokenizer.json and chat_template.jinja, planting one
            needle code sentence at a stated depth fraction.
  run       Send a NIAH prompt to an OpenAI-compatible /v1/chat/completions endpoint,
            record full reasoning/answer text and timing, and score exact match.
  score     Score a saved JSON response (from `niah.py run` or raw chat completion)
            against a .meta.json sidecar or explicit --code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TOPICS: list[tuple[str, str]] = [
    (
        "fabric-routing",
        "Direct attach copper cables link adjacent compute nodes in a physical ring topology without an intermediate switch. "
        "Each network interface maintains point-to-point subnet routes with jumbo frames enabled to reduce packet header overhead during sustained tensor transfers.",
    ),
    (
        "memory-hierarchy",
        "Unified memory architectures share a single physical LPDDR5X pool between CPU cores and GPU streaming multiprocessors. "
        "Without swap space enabled, allocating beyond available physical pages triggers immediate process termination, so active daemons sample MemAvailable at short intervals.",
    ),
    (
        "quantization-layout",
        "Low-bit ternary and mixed-precision tensor formats pack multiple quantized weights into compact byte streams alongside per-block floating-point scales. "
        "Attention projections, normalization layers, and routing gates remain in higher precision because small perturbations compound across residual streams.",
    ),
    (
        "attention-caching",
        "Multi-head latent attention compresses key and value histories into a shared low-dimensional latent vector before caching. "
        "Sparse token indexers score candidate past positions so later decoder layers attend only to the highest-scoring subset of tokens.",
    ),
    (
        "scheduler-queues",
        "Inference servers separate prompt prefill phases from autoregressive decode steps to avoid head-of-line blocking. "
        "During prefill, large matrix multiplications saturate compute units, whereas single-stream decode is bound primarily by memory bandwidth and inter-node synchronization latency.",
    ),
    (
        "verification-logs",
        "Reproducible benchmarking records the exact binary commit hash, build flags, hardware topology, background processes, and wall-clock UTC timestamps alongside every throughput sample. "
        "Reporting individual warm runs and their spread exposes thermal throttling or host contention.",
    ),
]

CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
PAD_WORDS = [
    "telemetry", "ring", "buffer", "packet", "queue", "stride", "cache", "window",
    "channel", "offset", "header", "checksum", "payload", "stream", "block", "frame",
]

RE_THINK_BLOCK = re.compile(r"<think(?::[^>]*)?>.*?</think(?::[^>]*)?>", re.DOTALL | re.IGNORECASE)
RE_OPEN_THINK = re.compile(r"<think(?::[^>]*)?>.*$", re.DOTALL | re.IGNORECASE)


def make_code(seed: int) -> str:
    rng = random.Random(seed)
    part1 = "".join(rng.choice(CODE_ALPHABET) for _ in range(4))
    part2 = "".join(rng.choice(CODE_ALPHABET) for _ in range(4))
    return f"HY4-NIAH-{part1}-{part2}"


def make_section(sec_idx: int, needle_code: str | None = None) -> str:
    topic, body = TOPICS[(sec_idx - 1) % len(TOPICS)]
    rev = ((sec_idx * 7) % 97) + 1
    bw = 180 + ((sec_idx * 13) % 45)
    qdepth = ((sec_idx * 3) % 11) + 1
    hit = 95 + ((sec_idx - 1) % 5)
    text = (
        f"Section {sec_idx} ({topic}, revision {rev}): {body} "
        f"In benchmark cycle {sec_idx}, node ring telemetry observed {bw} GB/s effective link utilization "
        f"with median queue depth {qdepth} and {hit} percent cache hit efficiency across active workers."
    )
    if needle_code is not None:
        text += f" Verification needle code recorded in Section {sec_idx}: {needle_code}."
    return text


def build_prompt_text(n_sections: int, depth: float, code: str, pad_words: int = 0) -> tuple[str, int]:
    needle_sec = max(1, min(n_sections, int(round(depth * n_sections))))
    lines = ["Below is an engineering logbook composed of numbered technical notes."]
    for i in range(1, n_sections + 1):
        sec = make_section(i, code if i == needle_sec else None)
        if i == n_sections and pad_words > 0:
            extra = " ".join(PAD_WORDS[j % len(PAD_WORDS)] for j in range(pad_words))
            sec = sec + f" Supplemental audit tags: {extra}."
        lines.append(sec)
        lines.append("")
    lines.append(
        f"Question: Based only on the logbook above, what is the exact verification needle code "
        f"recorded in Section {needle_sec}? Reply with ONLY the exact verification needle code and nothing else."
    )
    return "\n".join(lines) + "\n", needle_sec


def load_hy4_counter(tokenizer_dir: Path):
    try:
        from jinja2 import Environment
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit(
            f"ERROR: generate requires Python packages 'tokenizers' and 'jinja2': {exc}"
        )

    tok_path = tokenizer_dir / "tokenizer.json"
    tpl_path = tokenizer_dir / "chat_template.jinja"
    if not tok_path.is_file() or not tpl_path.is_file():
        raise SystemExit(
            f"ERROR: expected tokenizer.json and chat_template.jinja in {tokenizer_dir}"
        )

    tok = Tokenizer.from_file(str(tok_path))
    env = Environment()
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
    env.filters["tojson"] = lambda v, ensure_ascii=False: json.dumps(v, ensure_ascii=ensure_ascii)
    tpl = env.from_string(tpl_path.read_text(encoding="utf-8"))

    def count(text: str, reasoning_effort: str | None = None) -> tuple[int, int, str]:
        kw: dict = {
            "messages": [{"role": "user", "content": text}],
            "add_generation_prompt": True,
        }
        if reasoning_effort is not None:
            kw["reasoning_effort"] = reasoning_effort
        rendered = tpl.render(**kw)
        raw_n = len(tok.encode(text, add_special_tokens=False).ids)
        ren_n = len(tok.encode(rendered, add_special_tokens=False).ids)
        return raw_n, ren_n, rendered

    def encode_len(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False).ids)

    return count, encode_len


def cmd_generate(args: argparse.Namespace) -> int:
    if not (0.0 < args.depth < 1.0):
        raise SystemExit("ERROR: --depth must be strictly between 0.0 and 1.0")
    if args.target_tokens < 256:
        raise SystemExit("ERROR: --target-tokens must be >= 256")
    if args.target_tokens + args.max_tokens > args.context_window:
        raise SystemExit(
            f"ERROR: target_tokens ({args.target_tokens}) + max_tokens ({args.max_tokens}) "
            f"> context_window ({args.context_window})"
        )

    code = args.code or make_code(args.seed)
    counter, encode_len = load_hy4_counter(args.tokenizer_dir)

    # Estimate sections via doubling + binary search so rendered_high <= target_tokens
    lo, hi = 2, 16
    while True:
        txt, _ = build_prompt_text(hi, args.depth, code, 0)
        _, ren_n, _ = counter(txt, None)
        if ren_n >= args.target_tokens:
            break
        lo = hi
        hi *= 2

    best_n = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        txt, _ = build_prompt_text(mid, args.depth, code, 0)
        _, ren_n, _ = counter(txt, None)
        if ren_n <= args.target_tokens:
            best_n = mid
            lo = mid + 1
        else:
            hi = mid - 1

    # Fine-tune pad_words in the final section to hit target_tokens as closely as possible without exceeding it
    txt, needle_sec = build_prompt_text(best_n, args.depth, code, 0)
    _, ren_n, _ = counter(txt, None)
    best_pad = 0
    if ren_n < args.target_tokens:
        plo, phi = 0, (args.target_tokens - ren_n) * 2 + 16
        while plo <= phi:
            pmid = (plo + phi) // 2
            ptxt, _ = build_prompt_text(best_n, args.depth, code, pmid)
            _, pren_n, _ = counter(ptxt, None)
            if pren_n <= args.target_tokens:
                best_pad = pmid
                plo = pmid + 1
            else:
                phi = pmid - 1

    prompt_text, needle_sec = build_prompt_text(best_n, args.depth, code, best_pad)
    raw_tokens, ren_high, rendered_high = counter(prompt_text, None)
    _, ren_no_think, _ = counter(prompt_text, "no_think")

    needle_marker = f"Verification needle code recorded in Section {needle_sec}: {code}."
    pos_char = rendered_high.find(needle_marker)
    assert pos_char >= 0, "needle marker missing from rendered prompt"
    needle_tok_pos = encode_len(rendered_high[:pos_char])
    actual_depth = round(needle_tok_pos / ren_high, 4)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(prompt_text, encoding="utf-8")

    meta = {
        "schema": 1,
        "prompt_file": args.out.name,
        "code": code,
        "seed": args.seed,
        "needle_section": needle_sec,
        "total_sections": best_n,
        "requested_depth": args.depth,
        "actual_token_depth": actual_depth,
        "needle_token_offset": needle_tok_pos,
        "context_window": args.context_window,
        "target_tokens": args.target_tokens,
        "raw_prompt_tokens": raw_tokens,
        "rendered_tokens_thinking_on": ren_high,
        "rendered_tokens_no_think": ren_no_think,
        "chat_template_overhead_thinking_on": ren_high - raw_tokens,
        "chat_template_overhead_no_think": ren_no_think - raw_tokens,
        "recommended_max_tokens_thinking_on": args.max_tokens,
        "recommended_max_tokens_no_think": min(args.max_tokens, 256),
        "headroom_thinking_on": args.context_window - (ren_high + args.max_tokens),
        "headroom_no_think": args.context_window - (ren_no_think + min(args.max_tokens, 256)),
        "default_thinking_mode": "on (reasoning_effort omitted)",
    }
    meta_path = args.meta_out or args.out.with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(
        f"NIAH_GEN OK out={args.out} meta={meta_path} code={code} "
        f"sections={best_n} needle_sec={needle_sec} actual_depth={actual_depth} "
        f"raw={raw_tokens} rendered_thinking_on={ren_high} rendered_no_think={ren_no_think} "
        f"max_tokens={args.max_tokens} headroom={meta['headroom_thinking_on']}"
    )
    return 0


def extract_answer_text(content: str) -> str:
    cleaned = RE_THINK_BLOCK.sub("", content or "")
    cleaned = RE_OPEN_THINK.sub("", cleaned)
    return cleaned.strip()


def score_answer(content: str, code: str, reasoning: str = "") -> dict:
    answer_raw = extract_answer_text(content)
    normalized = answer_raw.strip().strip("`'\".,:;!?()[]{}").strip()
    strict_exact = answer_raw == code
    exact_match = normalized == code
    return {
        "expected_code": code,
        "answer_raw": answer_raw,
        "answer_normalized": normalized,
        "strict_exact": strict_exact,
        "exact_match": exact_match,
        "code_in_content": code in (content or ""),
        "code_in_reasoning": code in (reasoning or ""),
    }


def _stream_niah(url: str, payload: dict, timeout_s: float) -> dict:
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t_start = time.perf_counter()
    t_first = None
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
            raise RuntimeError(f"HTTP status {http_status} != 200 on /v1/chat/completions")
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            obj = json.loads(body)
            if obj.get("usage"):
                usage = obj["usage"]
            if obj.get("timings"):
                timings = obj["timings"]
            if obj.get("tensorfold"):
                tf_meta = obj["tensorfold"]
            if obj.get("speculative"):
                spec_meta = obj["speculative"]
            choices = obj.get("choices") or []
            if choices:
                c0 = choices[0]
                if c0.get("finish_reason"):
                    finish_reason = c0["finish_reason"]
                delta = c0.get("delta") or {}
                rc = delta.get("reasoning_content") or ""
                cc = delta.get("content") or ""
                if rc or cc:
                    if t_first is None:
                        t_first = time.perf_counter()
                if rc:
                    reasoning_chunks.append(rc)
                if cc:
                    content_chunks.append(cc)

    t_end = time.perf_counter()
    ttft_s = (t_first - t_start) if t_first is not None else (t_end - t_start)
    gen_s = (t_end - t_first) if t_first is not None else 0.0
    prompt_n = (usage or {}).get("prompt_tokens") or (timings or {}).get("prompt_n")
    pred_n = (usage or {}).get("completion_tokens") or (timings or {}).get("predicted_n")
    pp_tps = (prompt_n / ttft_s) if (prompt_n and ttft_s > 0) else None
    tg_tps = (
        (tf_meta or {}).get("tokens_per_second")
        or (((pred_n - 1) / gen_s) if (pred_n and pred_n > 1 and gen_s > 0) else None)
    )
    full_reasoning = "".join(reasoning_chunks)
    full_content = "".join(content_chunks)
    if not full_reasoning and not full_content:
        raise RuntimeError(f"empty completion (pred_n={pred_n}, finish_reason={finish_reason!r})")
    if pred_n is not None and pred_n <= 0:
        raise RuntimeError(f"zero completion tokens (pred_n={pred_n})")
    for k, v in (("ttft_s", ttft_s), ("total_s", t_end - t_start), ("pp_tps", pp_tps), ("tg_tps", tg_tps)):
        if isinstance(v, (int, float)) and not isinstance(v, bool) and (math.isnan(v) or math.isinf(v)):
            raise RuntimeError(f"non-finite metric {k}={v!r}")
    return {
        "http_status": http_status,
        "prompt_n": prompt_n,
        "pred_n": pred_n,
        "ttft_s": round(ttft_s, 4),
        "total_s": round(t_end - t_start, 4),
        "pp_tps": round(pp_tps, 2) if pp_tps is not None else None,
        "tg_tps": round(tg_tps, 2) if tg_tps is not None else None,
        "finish_reason": finish_reason,
        "usage": usage,
        "timings": timings,
        "tensorfold": tf_meta,
        "speculative": spec_meta,
        "reasoning_content": full_reasoning,
        "content": full_content,
    }


def _thinking_label(effort: str) -> str:
    if effort == "omit":
        return "on (default omit)"
    if effort == "high":
        return "on (high)"
    return "off (no_think)"


def cmd_run(args: argparse.Namespace) -> int:
    meta_path = args.meta or args.prompt.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    code = args.code or meta.get("code")
    if not code:
        raise SystemExit("ERROR: needle code required via --code or .meta.json sidecar")

    max_tokens = args.max_tokens
    if max_tokens is None:
        if args.reasoning_effort == "no_think":
            max_tokens = int(meta.get("recommended_max_tokens_no_think", 256))
        else:
            max_tokens = int(meta.get("recommended_max_tokens_thinking_on", 2048))

    prompt_text = args.prompt.read_text(encoding="utf-8")
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": args.seed,
        "stream": True,
        "cache_prompt": False,
        "stream_options": {"include_usage": True},
    }
    if args.reasoning_effort != "omit":
        payload["reasoning_effort"] = args.reasoning_effort

    res = _stream_niah(args.url, payload, args.timeout)
    sc = score_answer(res["content"], code, res["reasoning_content"])

    out_obj = {
        "schema": 1,
        "label": args.label,
        "prompt": str(args.prompt),
        "meta": meta,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "thinking_mode": _thinking_label(args.reasoning_effort),
        "max_tokens": max_tokens,
        "timeout_s": args.timeout,
        "response": res,
        "score": sc,
        "verdict": "PASS" if sc["exact_match"] else "FAIL",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_obj, indent=2) + "\n", encoding="utf-8")

    print(
        f"[{args.label}] verdict={out_obj['verdict']} exact_match={sc['exact_match']} "
        f"thinking={out_obj['thinking_mode']} prompt_n={res['prompt_n']} pred_n={res['pred_n']} "
        f"pp={res['pp_tps']} tok/s tg={res['tg_tps']} tok/s ttft={res['ttft_s']}s "
        f"answer={sc['answer_normalized']!r} expected={code!r}"
    )
    return 0 if sc["exact_match"] else 1


def cmd_score(args: argparse.Namespace) -> int:
    code = args.code
    if not code and args.meta:
        code = json.loads(args.meta.read_text(encoding="utf-8")).get("code")
    raw = json.loads(args.input.read_text(encoding="utf-8"))

    content = ""
    reasoning = ""
    if "response" in raw and isinstance(raw["response"], dict):
        content = raw["response"].get("content") or ""
        reasoning = raw["response"].get("reasoning_content") or ""
        if not code:
            code = (raw.get("meta") or {}).get("code") or (raw.get("score") or {}).get("expected_code")
    elif "choices" in raw and raw["choices"]:
        msg = raw["choices"][0].get("message") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
    else:
        content = str(raw.get("content") or "")
        reasoning = str(raw.get("reasoning_content") or "")

    if not code:
        raise SystemExit("ERROR: needle code required via --code, --meta, or input JSON")

    sc = score_answer(content, code, reasoning)
    verdict = "PASS" if sc["exact_match"] else "FAIL"
    print(
        f"NIAH_SCORE {verdict} exact_match={sc['exact_match']} strict_exact={sc['strict_exact']} "
        f"answer={sc['answer_normalized']!r} expected={code!r} "
        f"in_content={sc['code_in_content']} in_reasoning={sc['code_in_reasoning']}"
    )
    return 0 if sc["exact_match"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate a NIAH prompt sized with the real tokenizer")
    g.add_argument("--tokenizer-dir", type=Path, required=True, help="Directory with tokenizer.json and chat_template.jinja")
    g.add_argument("--target-tokens", type=int, required=True, help="Target rendered prompt tokens (thinking-on default)")
    g.add_argument("--context-window", type=int, default=131072, help="Server context window limit")
    g.add_argument("--max-tokens", type=int, default=2048, help="Completion budget for thinking + answer")
    g.add_argument("--depth", type=float, default=0.50, help="Needle depth fraction in (0, 1)")
    g.add_argument("--seed", type=int, default=20261120, help="Deterministic seed for needle code")
    g.add_argument("--code", default=None, help="Explicit needle code (default: generated from --seed)")
    g.add_argument("--out", type=Path, required=True, help="Output prompt .txt file")
    g.add_argument("--meta-out", type=Path, default=None, help="Output metadata .meta.json (default: <out>.meta.json)")
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("run", help="Run a NIAH prompt against an OpenAI-compatible server and score it")
    r.add_argument("--url", default="http://127.0.0.1:8080")
    r.add_argument("--model", default="hy4")
    r.add_argument("--prompt", type=Path, required=True)
    r.add_argument("--meta", type=Path, default=None)
    r.add_argument("--code", default=None)
    r.add_argument("--label", required=True)
    r.add_argument("--max-tokens", type=int, default=None)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--reasoning-effort", choices=("omit", "no_think", "high"), default="omit")
    r.add_argument("--timeout", type=float, default=float(os.environ.get("BENCH_TIMEOUT_S", "7200")))
    r.add_argument("--out", type=Path, required=True)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("score", help="Score a saved NIAH response JSON")
    s.add_argument("--input", type=Path, required=True)
    s.add_argument("--meta", type=Path, default=None)
    s.add_argument("--code", default=None)
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
