#!/usr/bin/env python3
"""Size and bits per weight of the text model in each build of the pack, read from safetensors headers only.

    python3 tools/text_size.py PACK_DIR > text_size.json

PACK_DIR is the weights folder: model.safetensors.index.json, the files it names, and trunk-4b/ and trunk-6b/. The
script reads each file's 8-byte length and JSON header, never tensor data, so it runs in seconds on the pack.

The rule. The text model is every tensor the pack's index names except the DFly2 drafter's (model.dfly_layers.*);
the vision tower is not in the pack. Its weights are counted once, from the base build: a tensor kept as is counts
its elements, and a matrix packed in EXL3 counts its input times output size (the trellis has one 16x16 tile per
row of 16 by column of 16), so the scales and markers beside a packed matrix count no weights. A build's size is the
tensor data it loads for those weights, in bytes from the headers' data offsets, safetensors headers not included.
Bits per weight is that size times 8 divided by the weight count.

Builds: lossless serves the base folder as is. 4b and 6b replace each trunk matrix the trunk folder packs
(model.layers.*.safetensors there) with its packed copy; 4b6b takes o_proj and kv_a_proj_with_mqa from trunk-6b/
and the rest from trunk-4b/ (TF_HY4_TRUNK, launch/SETTINGS.md). The packed lm_head and the drafter's files in the
trunk folders are not counted: the default launch serves neither (TF_HY4_TARGET_HEAD is off, drafts are separate).
BF16 is the same weights at 2 bytes each.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import time

PACKED = (".trellis", ".suh", ".svh", ".mul1", ".mcg", ".su", ".sv")
DRAFTER = "model.dfly_layers."


def header(path: str) -> dict:
    with open(path, "rb") as h:
        n = struct.unpack("<Q", h.read(8))[0]
        d = json.loads(h.read(n))
    d.pop("__metadata__", None)
    return d


def module(name: str) -> str:
    for s in PACKED:
        if name.endswith(s):
            return name[: -len(s)]
    return name.rsplit(".", 1)[0]


def numel(shape: list) -> int:
    n = 1
    for x in shape:
        n *= x
    return n


def main() -> int:
    pack = sys.argv[1]
    idx = json.load(open(os.path.join(pack, "model.safetensors.index.json")))["weight_map"]
    heads: dict[str, dict] = {}
    base: dict[str, tuple] = {}  # tensor -> (module, bytes, weights)
    for name, f in idx.items():
        if name.startswith(DRAFTER):
            continue
        if f not in heads:
            heads[f] = header(os.path.join(pack, f))
        t = heads[f][name]
        nbytes = t["data_offsets"][1] - t["data_offsets"][0]
        if name.endswith(".trellis"):
            w = t["shape"][0] * 16 * t["shape"][1] * 16
        elif name.endswith(PACKED):
            w = 0
        else:
            w = numel(t["shape"])
        base[name] = (module(name), nbytes, w)
    weights = sum(w for _, _, w in base.values())
    vision = [n for n in base if n.startswith(("vit.", "model.vit", "visual."))]
    trunk: dict[str, dict[str, int]] = {}  # width -> module -> packed bytes
    for width in ("4b", "6b"):
        d = os.path.join(pack, f"trunk-{width}")
        mods: dict[str, int] = {}
        for f in sorted(os.listdir(d)):
            if not (f.startswith("model.layers.") and f.endswith(".safetensors")):
                continue
            for name, t in header(os.path.join(d, f)).items():
                m = module(name)
                mods[m] = mods.get(m, 0) + t["data_offsets"][1] - t["data_offsets"][0]
        trunk[width] = mods
    base_by_mod: dict[str, int] = {}
    for m, b, _ in base.values():
        base_by_mod[m] = base_by_mod.get(m, 0) + b
    missing = sorted(m for w in trunk.values() for m in w if m not in base_by_mod)
    lossless = sum(b for _, b, _ in base.values())

    def build(pick) -> int:
        total = lossless
        for m in trunk["4b"]:
            w = pick(m)
            total += trunk[w][m] - base_by_mod[m]
        return total

    sizes = {
        "lossless": lossless,
        "4b": build(lambda m: "4b"),
        "4b6b": build(lambda m: "6b" if m.endswith((".o_proj", ".kv_a_proj_with_mqa")) else "4b"),
        "6b": build(lambda m: "6b"),
        "bf16": 2 * weights,
    }
    out = {
        "tool": "tools/text_size.py",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rule": __doc__.split("\n\n")[2].replace("\n", " "),
        "text_weights": weights,
        "text_tensors": len(base),
        "drafter_tensors_left_out": sum(1 for n in idx if n.startswith(DRAFTER)),
        "vision_tensors_in_pack": len(vision),
        "trunk_matrices": {w: len(m) for w, m in trunk.items()},
        "trunk_modules_not_in_base": missing,
        "builds": {k: {"bytes": v, "gb": round(v / 1e9, 2), "gib": round(v / 2**30, 2), "bits_per_weight": round(8 * v / weights, 3)}
                   for k, v in sizes.items()},
    }
    print(json.dumps(out, indent=1))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
