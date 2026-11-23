#!/usr/bin/env python3
"""Build the vision folder (vit-bf16/) from the tencent/Hy4 source checkpoint, or check one.

The quantized pack carries no vision tower. This copies the 649 vit.* tensors unchanged, in BF16, from the
source shards that hold them into one vit.safetensors, copies the source's config.json and
processor_config.json beside it, and writes SHA256SUMS. Only rank 0 reads the folder (SERVE_VISION="1",
TF_HY4_VISION_DIR; launch/SETTINGS.md, "SERVE_VISION").

    python3 tools/vit_folder.py SOURCE_DIR OUT_DIR          # build, then check against launch/vision.sha256
    python3 tools/vit_folder.py --check OUT_DIR             # check a folder only

Building needs torch and safetensors (both are in nvcr.io/nvidia/pytorch:26.07-py3) and about 2.5 GB of
memory; --check needs neither. Ported from the run that built the published folder, which wrote the hashes in
launch/vision.sha256 (2026-11-22, on the Spark that holds the source checkpoint, 1.46 s). This copy of the
tool has not been run yet.
"""
import hashlib
import json
import os
import shutil
import sys
import time

TENSORS = 649
PAYLOAD_BYTES = 2470253568
FILES = ("vit.safetensors", "config.json", "processor_config.json")
SUMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "launch", "vision.sha256")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def check(out_dir):
    want = {}
    with open(SUMS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                digest, name = line.split()
                want[name] = digest
    bad = 0
    for name, digest in want.items():
        p = os.path.join(out_dir, name)
        got = sha256(p) if os.path.isfile(p) else "missing"
        ok = got == digest
        bad += not ok
        print(f"{name}: {'OK' if ok else 'FAILED (' + got + ')'}")
    print(f"vit_folder check: {len(want) - bad} of {len(want)} files match launch/vision.sha256")
    return bad == 0


def build(src_dir, out_dir):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(src_dir, "model.safetensors.index.json"), encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    by_shard = {}
    for name, rel in weight_map.items():
        if name.startswith("vit."):
            by_shard.setdefault(rel, []).append(name)
    tensors, total = {}, 0
    for rel in sorted(by_shard):
        with safe_open(os.path.join(src_dir, rel), framework="pt", device="cpu") as f:
            for name in by_shard[rel]:
                t = f.get_tensor(name)
                if t.dtype != torch.bfloat16:
                    sys.exit(f"vit_folder: {name} is {t.dtype}, expected bfloat16")
                tensors[name] = t
                total += t.numel() * t.element_size()
    print(f"vit_folder: {len(tensors)} vit.* tensors, {total} payload bytes, from {len(by_shard)} shards")
    if len(tensors) != TENSORS or total != PAYLOAD_BYTES:
        sys.exit(f"vit_folder: expected {TENSORS} tensors and {PAYLOAD_BYTES} bytes from tencent/Hy4 c10c0de5")
    out = os.path.join(out_dir, "vit.safetensors")
    save_file(tensors, out, metadata={"format": "pt"})
    for name in ("config.json", "processor_config.json"):
        shutil.copy2(os.path.join(src_dir, name), os.path.join(out_dir, name))
    with open(os.path.join(out_dir, "SHA256SUMS"), "w", encoding="utf-8") as f:
        for name in FILES:
            f.write(f"{sha256(os.path.join(out_dir, name))}  {name}\n")
    print(f"vit_folder: wrote {out} ({os.path.getsize(out)} bytes) and SHA256SUMS in {time.time() - t0:.2f} s")


def main(argv):
    if len(argv) == 2 and argv[0] == "--check":
        return 0 if check(argv[1]) else 1
    if len(argv) == 2 and not argv[0].startswith("-"):
        build(argv[0], argv[1])
        return 0 if check(argv[1]) else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
