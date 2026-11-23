#!/usr/bin/env python3
"""Fast pre-launch safetensors header check on staged Hy4 EXL3 layer files.

Runs on one node (locally via --dir or over SSH on rank 0 via --env-file) before
any rank starts. Verifies that every staged sparse layer (L >= 1) has the exact
same tensor count, normalized keys, shapes, and dtypes as the reference sparse
layer (default L=1), and that every routed expert projection carries `.mul1`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import struct
import subprocess
import sys
from pathlib import Path

REMOTE_CHECKER = r'''
import json, os, re, struct, sys

stage_dir = sys.argv[1]
ref_layer = int(sys.argv[2])
expect_layers = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else 0

def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return {k: (v["dtype"], v["shape"]) for k, v in h.items() if k != "__metadata__"}

layer_files = {}
pat = re.compile(r"^model\.layers\.(\d+)\.safetensors$")
for name in os.listdir(stage_dir):
    m = pat.match(name)
    if m:
        layer_files[int(m.group(1))] = os.path.join(stage_dir, name)

if not layer_files:
    print(f"FAIL: no model.layers.*.safetensors found in {stage_dir}", file=sys.stderr)
    sys.exit(1)

layers = sorted(layer_files)
n_contig = 0
while n_contig in layer_files:
    n_contig += 1
if expect_layers and n_contig < expect_layers:
    print(f"FAIL: expected {expect_layers} contiguous layers, found {n_contig} (all={len(layers)})", file=sys.stderr)
    sys.exit(1)
if ref_layer not in layer_files:
    print(f"FAIL: reference layer {ref_layer} missing in {stage_dir}", file=sys.stderr)
    sys.exit(1)

errors = []
# Check dense layer 0 if present
if 0 in layer_files:
    h0 = read_header(layer_files[0])
    req0 = [
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.indexer.wk.weight",
    ]
    missing0 = [k for k in req0 if k not in h0]
    if missing0:
        errors.append(f"layer 0 missing required dense/attn keys: {missing0}")

# Reference sparse layer signature (strip model.layers.<L>. prefix and indexer keys which exist only on full-indexer layers)
def split_sig(hdr, l):
    pref = f"model.layers.{l}."
    base = {}
    idx_keys = {}
    mul1_projs = set()
    for k, meta in hdr.items():
        if not k.startswith(pref):
            errors.append(f"layer {l}: unexpected prefix on key {k}")
            continue
        rel = k[len(pref):]
        if rel.startswith("self_attn.indexer."):
            idx_keys[rel] = meta
        else:
            base[rel] = meta
        m = re.match(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.mul1$", rel)
        if m:
            mul1_projs.add((int(m.group(1)), m.group(2)))
    return base, idx_keys, mul1_projs

ref_hdr = read_header(layer_files[ref_layer])
ref_base, ref_idx, ref_mul1 = split_sig(ref_hdr, ref_layer)
expected_mul1 = {(e, p) for e in range(256) for p in ("gate_proj", "up_proj", "down_proj")}
if ref_mul1 != expected_mul1:
    errors.append(f"ref layer {ref_layer}: expected 768 expert .mul1 keys, found {len(ref_mul1)}")

checked_sparse = 0
full_indexer_layers = []
for l in layers:
    if l == 0:
        continue
    hdr = read_header(layer_files[l])
    base, idx_keys, mul1_projs = split_sig(hdr, l)
    checked_sparse += 1
    if len(base) != len(ref_base):
        errors.append(f"layer {l}: non-indexer tensor count {len(base)} != ref {len(ref_base)}")
    elif base != ref_base:
        diff_keys = sorted(set(base) ^ set(ref_base))
        if diff_keys:
            errors.append(f"layer {l}: key mismatch vs ref L{ref_layer}: sample={diff_keys[:5]}")
        else:
            bad_meta = [k for k in ref_base if base[k] != ref_base[k]]
            errors.append(f"layer {l}: dtype/shape mismatch vs ref L{ref_layer}: {[(k, base[k], ref_base[k]) for k in bad_meta[:5]]}")
    if mul1_projs != expected_mul1:
        missing_m = sorted(expected_mul1 - mul1_projs)
        errors.append(f"layer {l}: missing {len(missing_m)} expert .mul1 keys (sample={missing_m[:5]})")
    if idx_keys:
        full_indexer_layers.append(l)
        if ref_idx and idx_keys != ref_idx:
            errors.append(f"layer {l}: indexer key/shape mismatch vs ref L{ref_layer}")

if errors:
    for e in errors[:20]:
        print(f"FAIL: {e}", file=sys.stderr)
    print(f"HEADER_CHECK: FAIL (errors={len(errors)}, dir={stage_dir})", file=sys.stderr)
    sys.exit(1)

print(
    f"HEADER_CHECK: PASS dir={stage_dir} total_layers={len(layers)} contiguous=0..{n_contig - 1} "
    f"sparse_checked={checked_sparse} ref_layer={ref_layer} ref_non_idx_tensors={len(ref_base)} "
    f"expert_mul1_per_layer={len(ref_mul1)} full_indexer_sparse_layers={full_indexer_layers}"
)
'''


def parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        env[k] = v
    return env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir", help="Local staged model directory to inspect")
    src.add_argument("--env-file", type=Path, help="ring.env file (runs check across all RING_HOSTS over SSH)")
    ap.add_argument("--ref-layer", type=int, default=1, help="Reference sparse layer index (default: 1)")
    ap.add_argument("--expect-layers", type=int, default=0, help="Require at least N contiguous layers (0 = any)")
    ap.add_argument("--rank0-only", action="store_true", help="Check only rank 0 instead of all four RING_HOSTS")
    args = ap.parse_args()

    if args.dir:
        p = subprocess.run(
            [sys.executable, "-c", REMOTE_CHECKER, args.dir, str(args.ref_layer), str(args.expect_layers)],
            check=False,
        )
        return p.returncode

    env = parse_env_file(args.env_file)
    hosts = env.get("RING_HOSTS", "").split()
    if not hosts:
        print(f"FAIL: RING_HOSTS missing in {args.env_file}", file=sys.stderr)
        return 2
    user = env.get("STAGE_SSH_USER", "")
    stage_dir = env.get("STAGE_DIR") or env.get("MODEL_DIR", "")
    if not stage_dir:
        print(f"FAIL: STAGE_DIR / MODEL_DIR missing in {args.env_file}", file=sys.stderr)
        return 2

    check_hosts = hosts[:1] if args.rank0_only else hosts
    cmd = f"python3 - {shlex.quote(stage_dir)} {args.ref_layer} {args.expect_layers}"
    summaries: list[str] = []
    for rk, h in enumerate(check_hosts):
        target = f"{user}@{h}" if (user and "@" not in h) else h
        p = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target, cmd],
            input=REMOTE_CHECKER.encode("utf-8"),
            capture_output=True,
            check=False,
        )
        out = p.stdout.decode(errors="replace").strip()
        err = p.stderr.decode(errors="replace").strip()
        if p.returncode != 0:
            print(f"[rank {rk} {h}] FAIL:\n{err or out}", file=sys.stderr)
            return p.returncode
        print(f"[rank {rk} {h}] {out}")
        summaries.append(out)

    if len(summaries) > 1 and len(set(summaries)) != 1:
        print("FAIL: header check summaries differ across ranks!", file=sys.stderr)
        return 1
    if len(summaries) > 1:
        print(f"HEADER_CHECK_ALL_RANKS: PASS ({len(summaries)}/{len(summaries)} ranks identical)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
