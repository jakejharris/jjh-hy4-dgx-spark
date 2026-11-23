#!/usr/bin/env python3
"""Generate an ExLlamaV3 per-tensor recipe YAML for full Hy4 (HYV4VLForConditionalGeneration).

Profile:
  - Routed experts (layers with mlp_layer_types[i] == "sparse"): EXL3 mul1 at expert_bits
    (with optional down_bits override on a layer range, e.g. --down-bits 4 --down-range 20..76)
  - All other budgeted Linears (self_attn.{q_a,q_b,kv_a_proj_with_mqa,linear_gate,o_proj},
    indexer.wq_b on full layers, dense layer 0 mlp.{gate,up,down}_proj,
    shared_experts.{gate,up,down}_proj): 16 (stored lossless unquantized BF16)
  - lm_head: head_bits = 16
  - vit.* and model.dfly_layers.*: copied unquantized by compile.py (not budgeted)
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help = "Path to model config.json")
    ap.add_argument("out", help = "Output recipe.yaml path")
    ap.add_argument("--expert-bits", type = float, default = 3, help = "Bitrate for routed experts (default 3)")
    ap.add_argument("--down-bits", type = float, default = None, help = "Optional bitrate for routed down_proj")
    ap.add_argument("--down-range", type = str, default = None, help = "Inclusive layer range e.g. 20..76 for --down-bits")
    args = ap.parse_args()

    raw = json.load(open(args.config))
    cfg = raw.get("text_config", raw)
    n_layers = cfg["num_hidden_layers"]
    n_experts = cfg["n_routed_experts"]
    mlp_types = cfg.get("mlp_layer_types") or ["dense" if i < cfg.get("first_k_dense_replace", 1) else "sparse" for i in range(n_layers)]
    indexer_types = cfg["indexer_types"]

    down_lo, down_hi = None, None
    if args.down_bits is not None:
        if args.down_range:
            a, b = args.down_range.split("..")
            down_lo, down_hi = int(a), int(b)
        else:
            down_lo, down_hi = 0, n_layers - 1

    def fmt_bits(b):
        return int(b) if float(b).is_integer() else float(b)

    eb = fmt_bits(args.expert_bits)
    db = fmt_bits(args.down_bits) if args.down_bits is not None else eb

    # Use pattern rules for routed experts + explicit 16-bit entries for non-experts so the YAML stays compact
    lines = [
        f"# Hy4 EXL3 recipe: routed experts @{eb}b (down @{db}b in {down_lo}..{down_hi}), non-experts 16b lossless",
        f"target_bpw: {eb}",
        "head_bits: 16",
        "tensors:",
    ]
    n_16 = 0
    n_exp = 0
    for i in range(n_layers):
        p = f"model.layers.{i}"
        attn_keys = [
            f"{p}.self_attn.q_a_proj",
            f"{p}.self_attn.q_b_proj",
            f"{p}.self_attn.kv_a_proj_with_mqa",
            f"{p}.self_attn.linear_gate",
            f"{p}.self_attn.o_proj",
        ]
        if indexer_types[i] == "full":
            attn_keys.append(f"{p}.self_attn.indexer.wq_b")
        for k in attn_keys:
            lines.append(f'  "{k}": 16')
            n_16 += 1
        if mlp_types[i] == "dense":
            for proj in ("gate_proj", "up_proj", "down_proj"):
                lines.append(f'  "{p}.mlp.{proj}": 16')
                n_16 += 1
        else:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                lines.append(f'  "{p}.mlp.shared_experts.{proj}": 16')
                n_16 += 1
            d_bits = db if (down_lo is not None and down_lo <= i <= down_hi) else eb
            lines.append(f'  "{p}.mlp.experts.*.gate_proj": {eb}')
            lines.append(f'  "{p}.mlp.experts.*.up_proj": {eb}')
            lines.append(f'  "{p}.mlp.experts.*.down_proj": {d_bits}')
            n_exp += 3 * n_experts

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"recipe: layers={n_layers} experts_matrices={n_exp} non_expert_16={n_16} -> {args.out}")


if __name__ == "__main__":
    main()
