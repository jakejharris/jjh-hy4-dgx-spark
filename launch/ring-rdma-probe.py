#!/usr/bin/env python3
"""Quick 2-rank or 4-rank NCCL ring all-gather probe.

Verifies:
  1. NCCL transport selected from the actual NCCL debug log (fails closed if not IB).
  2. Bit-exact FP32 pattern across all ranks (catches swapped cable order).
  3. Median (p50), 95th percentile (p95), and worst (max) latency in ms.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def load_ring_env(env_path: Path | None) -> None:
    if not env_path or not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def make_pattern_int32(rank: int, numel: int, phase: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(numel, device=device, dtype=torch.int64)
    mix = ((idx * 2654435761 + phase * 1013904223) & 0x1FFFFF).to(torch.int32)
    return (0x3F800000 | ((rank & 0x3) << 21)) | mix


def verify_all_gather(rank: int, world: int, numel: int, phase: int, device: torch.device) -> list[str]:
    send_bits = make_pattern_int32(rank, numel, phase, device)
    recv_f32 = torch.empty(world * numel, device=device, dtype=torch.float32)
    dist.all_gather_into_tensor(recv_f32, send_bits.view(torch.float32))
    torch.cuda.synchronize(device)

    recv_bits = recv_f32.view(torch.int32).view(world, numel)
    mismatches = []
    for k in range(world):
        exp_bits = make_pattern_int32(k, numel, phase, device)
        if not torch.equal(recv_bits[k], exp_bits):
            first_bad = int((recv_bits[k] != exp_bits).nonzero(as_tuple=False)[0].item())
            got_word = int(recv_bits[k, first_bad].item())
            exp_word = int(exp_bits[first_bad].item())
            mismatches.append(
                f"slot {k}: expected 0x{exp_word:08x} got 0x{got_word:08x} at elem {first_bad}"
            )
    return mismatches


def summarize_ms(samples_ms: list[float]) -> dict[str, float | int]:
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "n": int(arr.size),
        "p50_ms": round(float(np.median(arr)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "max_ms": round(float(np.max(arr)), 3),
    }


def time_single(inp: torch.Tensor, out: torch.Tensor, warmup: int, iters: int, device: torch.device) -> dict[str, float | int]:
    for _ in range(warmup):
        dist.all_gather_into_tensor(out, inp)
    torch.cuda.synchronize(device)
    dist.barrier()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        e0.record()
        dist.all_gather_into_tensor(out, inp)
        e1.record()
        e1.synchronize()
        times.append(e0.elapsed_time(e1))
    dist.barrier()
    return summarize_ms(times)


def time_burst(inp: torch.Tensor, out: torch.Tensor, gathers: int, warmup: int, iters: int, device: torch.device) -> dict[str, float | int]:
    for _ in range(warmup):
        for _ in range(gathers):
            dist.all_gather_into_tensor(out, inp)
        torch.cuda.synchronize(device)
    dist.barrier()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        e0.record()
        for _ in range(gathers):
            dist.all_gather_into_tensor(out, inp)
        e1.record()
        e1.synchronize()
        times.append(e0.elapsed_time(e1))
    dist.barrier()
    return summarize_ms(times)


def detect_transport(nccl_log: Path | None) -> tuple[str, list[str]]:
    if not nccl_log or not nccl_log.is_file():
        return "unknown", []
    text = nccl_log.read_text(encoding="utf-8", errors="replace")
    highlights = [
        ln.strip()
        for ln in text.splitlines()
        if any(tok in ln for tok in ("Using network ", "Connected all rings", "NET/IB : Using"))
    ]
    found = re.findall(r"Using network\s+(\S+)", text)
    if found:
        return found[-1], highlights
    return "unknown", highlights


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ranks", type=int, choices=[2, 4], default=4)
    ap.add_argument("--env-file", type=Path, default=None)
    ap.add_argument("--nccl-log", type=Path, default=None)
    ap.add_argument("--decode-elems", type=int, default=6144)
    ap.add_argument("--decode-iters", type=int, default=200)
    ap.add_argument("--token-gathers", type=int, default=156)
    ap.add_argument("--token-iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timeout-s", type=int, default=60)
    args = ap.parse_args()

    load_ring_env(args.env_file)
    log_path = args.nccl_log or Path(os.environ.setdefault("NCCL_DEBUG_FILE", "/tmp/nccl_probe.log"))
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,NET")
    os.environ["NCCL_DEBUG_FILE"] = str(log_path)

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != args.ranks:
        print(f"ERROR: WORLD_SIZE={world} does not match --ranks {args.ranks}", file=sys.stderr)
        return 2

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    t0 = time.perf_counter()
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=args.timeout_s),
        device_id=device,
    )
    probe_in = torch.full((1,), float(rank), device=device, dtype=torch.float32)
    probe_out = torch.empty(world, device=device, dtype=torch.float32)
    dist.all_gather_into_tensor(probe_out, probe_in)
    torch.cuda.synchronize(device)
    init_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    bad = verify_all_gather(rank, world, args.decode_elems, 11, device)
    fail_flag = torch.tensor([1 if bad else 0], device=device, dtype=torch.int32)
    dist.all_reduce(fail_flag, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)

    if int(fail_flag.item()) > 0:
        print(f"[ring-rdma-probe] rank={rank} FAIL: {'; '.join(bad) or 'peer mismatch'}", flush=True)
        dist.destroy_process_group()
        return 4

    inp = make_pattern_int32(rank, args.decode_elems, 31, device).view(torch.float32)
    out = torch.empty(world * args.decode_elems, device=device, dtype=torch.float32)
    single_stats = time_single(inp, out, args.warmup, args.decode_iters, device)
    burst_stats = time_burst(inp, out, args.token_gathers, 5, args.token_iters, device)

    post_bad = verify_all_gather(rank, world, args.decode_elems, 59, device)
    byte_status = "FAIL" if post_bad else "PASS"

    dist.barrier()
    dist.destroy_process_group()

    transport, highlights = detect_transport(log_path)
    for ln in highlights[:6]:
        print(f"NCCL_TRANSPORT rank={rank} {ln}", flush=True)

    res = {
        "rank": rank,
        "world": world,
        "host": socket.gethostname().split(".")[0],
        "transport": transport,
        "init_ms": init_ms,
        "single_gather_ms": single_stats,
        "burst_156x_ms": burst_stats,
        "bytes_check": byte_status,
    }
    print("PROBE_JSON " + json.dumps(res, sort_keys=True), flush=True)

    if transport != "IB":
        print(
            f"[ring-rdma-probe] rank={rank} FAIL-CLOSED: expected NCCL IB transport, got {transport}",
            file=sys.stderr,
            flush=True,
        )
        return 6
    return 0 if byte_status == "PASS" else 5


if __name__ == "__main__":
    raise SystemExit(main())
