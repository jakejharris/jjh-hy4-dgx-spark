# Quantizing Hy4 (`tencent/Hy4`) to EXL3 (`mul1`)

This folder reproduces the EXL3 quantization of `tencent/Hy4` (`c10c0de5`) on four DGX Spark nodes (GB10, `121.7 GiB` unified memory each, `0` swap) using public ExLlamaV3 `v1.6.7` (`8c593e4e5c88da76a4aaf8f1b1838bb7c1fce4e0`) with the unified Hy4 quantizer patch (`quant/patches/0001-hy4-quant.patch`, made from quantizer commit `3b697ca1f55a53ffce9a5d2b612146f33f4b20ca`) applied on top. That commit is `c7c901f`, the head of the quantizer line the run used, plus two commits: one changes 9 lines in 4 files, all comments, docstrings or command-line help strings (6 docstring lines, 2 help strings, 1 YAML comment) that named a host, a home path, an internal spec file or a run stage, and the other renames the CPU packager to `farm/package_hy4_cpu.py` without changing its content; no code line changes. The other short ids below (`f1bb96f`, `3d92885`, `157da29`, `ae21a21`, `dbcb963`) are earlier commits on that quantizer line, between `v1.6.7` and `3b697ca`, so the patch holds their changes; the line itself is not public.

## Cluster layout and measured pace

One node coordinates (`node3`, holding all 245 BF16 shards and local NVMe scratch). The other nodes (`node1`, `node2`, and optionally `node4`) run stateless TCP encoder workers bound to the 200 Gb/s direct cable IPs. The coordinator encodes on its own GPU alongside the remote workers and shares the 768 routed-expert matrices per MoE layer through a dynamic work-stealing queue with 1-group-ahead pipelining (`f1bb96f`).

Measured module times below come from production coordinator logs (`250 x 2,048` calibration, flat `3.00 bpw` `mul1`, `--out_scales always`, single-pass capture). Checkpoint writes add about `0.5 min/layer` to wall time.

| Machines working | Setup and layer range | Module time per layer | Notes |
| --- | --- | --- | --- |
| 1 machine (dense `L0`) | `node3` alone, `L0` (no routed experts) | `6.82 min` (`409.18 s`) | Dense MLP kept lossless BF16 |
| 1 machine (MoE layer) | `node3` alone, `768` routed matrices | **Not measured** | Production MoE layers always used `>= 2` machines |
| 2 machines (`node3` + `node1`) | Pipelined, BF16 state, `L42..49` (`8` layers) | mean `16.31 min` (`15.81 to 16.80 min`) | `L39..41` fallback when `node2` dropped: `15.50 to 16.04 min` |
| 3 machines (`node3` + `node1, node2`) | Pipelined, BF16 state, `L33..38` and `L50..51` (`8` layers) | mean `12.84 min` (`12.50 to 13.34 min`) | `L11..19` pipelined: `12.85 min` (`12.06 to 14.84 min`); `L1..10` pre-pipelining: `16.03 min` (`14.62 to 19.37 min`) |
| 4 machines (`node3` + `node1, node2, node4`) | Pipelined, shared hosts, `L21..30` (`10` layers) | mean `12.04 min` (`10.85 to 13.55 min`) | `L20` startup layer took `14.56 min`; `5.00 min` serial on `node3` + `7.44 min` parallel encode (`L25..27`) |

- Early 3-machine baseline (claim `quant.time.per_layer_min`): 15.09 min/layer (component; spread: 14.73 to 15.36 across three modules).
- Final compiled pack size (claim `quant.size.total_gib`): 346.31 GiB (spread: 371,844,743,595 bytes in 251 files: base folder 338,741,532,389, trunk-4b/ 13,245,018,819, trunk-6b/ 19,858,192,387).

## Recipe table

All non-expert weights and the KV cache stay in lossless BF16 or FP32 in the base pack. Only the routed experts (`L1..77`, `256` experts per layer) are quantized to `3.00 bpw` `mul1` EXL3.

| Tensor Class | Precision / Format | Layer Scope | Notes |
| --- | --- | --- | --- |
| Routed experts (`gate_proj`, `up_proj`, `down_proj`) | EXL3 `mul1`, `3.00 bpw` (`k2=6`) | `L1..77` | Flat `3.00 bpw` (`quant/recipe-l20-76.yaml`) |
| Routed Hadamard scales (`suh`, `svh`, `.mul1`) | FP16 + `i32` marker | `L1..77` | `--out_scales always` |
| Shared experts (`shared_experts.{gate,up,down}_proj`) | BF16 (`16 bpw`, unclamped) | `L1..77` | Unquantized in base pack |
| Attention (`self_attn.*`) + `sinks` | BF16 (`16 bpw`), `sinks` FP32 | `L0..77` | Unquantized in base pack |
| Dense layer 0 (`L0.mlp.*`) | BF16 (`16 bpw`, unclamped) | `L0` | Unquantized in base pack |
| `embed_tokens`, `lm_head` | `embed` BF16; `lm_head` `16 bpw` (`FP32` on load) | Head/Tail | `head_bits=16` |
| Token indexer (`indexer.{wq_b,wk,k_norm,weights_proj}`) | `wq_b`/`wk` BF16; `k_norm`/`weights_proj` FP32 | 21 full layers | Replicated per rank |
| iHC streams, routers, norms | `iHC` + `e_score_correction_bias` FP32; router/norms BF16 | `L0..77` | Unquantized |
| `DFly2` drafter (`model.dfly_layers.*`) | Copied BF16 (`0` calib min) | Passthrough | Copied at packaging |
| Vision tower (`vit.*`) | BF16, not in the pack | Separate folder | `tools/vit_folder.py` copies the 649 tensors unchanged into `vit-bf16/` |

## Two-stage run and `--max_module 79`

`SafetensorsCollection` globs `*.safetensors` once at process start. If you start quantizing while shards for deeper layers are still downloading, run layers `0 to 19` with `--max_module 21`, then restart with `-r --override_anyway` once all 245 shards are on disk. If all 245 shards are already on disk, you can run directly to `--max_module 79`.

Always pass `--max_module 79` on the run through layer 77. Without `--max_module 79`, ExLlamaV3 runs `compile.py` right after layer 77 and writes an extra `~340 GB` compiled copy onto the coordinator disk. Passing `--max_module 79` stops cleanly after layer 77 is saved and checkpointed so you can package on a node with free disk.

## Unified-memory capture note

On unified-memory GPUs such as GB10, `torch.cuda.mem_get_info()` reads `MemFree` and ignores the Linux page cache (buffered model shards), causing `capture_passes()` to see `~1 GiB` free and split a `~2.8 GB` layer capture into 7 passes. Commit `3d92885` switches `capture_passes()` to `device_mem_info()`, which reads `MemAvailable` and restores single-pass capture. Confirm in the coordinator log during `L1` that capture prints `pass 1/1` rather than `pass 1/N` (`N > 1`).

## Patch for large values at layers 75 and 76 (`dbcb963`)

In Hy4, the shared-expert gated product `silu(gate) * up` crosses FP16 max (`65,504`) starting at layer 75 (`76,661.9` in FP32) and peaks at `1,510,548.9` at layer 76, producing `Inf`/`NaN` if computed in `torch.half`. Commit `dbcb963` (building on BF16 stream and state propagation in `157da29` and BF16 `safe_open` byte-view loading in `ae21a21`) keeps the Hy4 shared MLP SwiGLU activation (`silu(gate) * up`) in FP32 through `shared_experts.down_proj`, updates both the lossless linear path (`fp16.py`) and the EXL3 linear path (`exl3.py`) to accept FP32 inputs without overflowing FP16, keeps calibration state in BF16 across layers, and adds a `Normal-pos (|ref|<=1000)` metric line in `convert_model.py` so outlier tokens do not hide ordinary-token error.

To verify the patch took on every node running calibration (`node3` and, if running the optional trunk pass, `node4`), check the SHA-256 digests of these six files inside the ExLlamaV3 source tree and compile them with Python 3.12 inside `nvcr.io/nvidia/pytorch:26.07-py3`:

| File (under ExLlamaV3 source root) | Required SHA-256 (`dbcb963`) |
| --- | --- |
| `exllamav3/modules/hyv4.py` | `073c8510824d28e90c64e9894650649c333573d3a6d54534b1e7844c1b6fd74c` |
| `exllamav3/modules/mlp.py` | `d1ebf6dcc0bc98b6c43e4fbf40752532bfc4cd53356fec4ea1f555e6db87d825` |
| `exllamav3/modules/quant/fp16.py` | `480cf816ae81fb3281a1b0963bebacd4ff36921379f6cbf2dd84c1e702202ad1` |
| `exllamav3/modules/quant/exl3.py` | `bfe04d230bbf3178c13e1b24792d8f358399e88e8f5455c556329a4b197fdb96` |
| `exllamav3/conversion/convert_model.py` | `6c8706474feff8cf242d73fd93d61732fad556bb07427302a9fb8ec36f3ff277` |
| `exllamav3/loader/safetensors_alt.py` | `40167ccf4c26e4bb4d11ae229b3c612095a03d56cf15e26d7f6c90d71dbd9033` |

```bash
PYTHONPYCACHEPREFIX=/tmp/pyc python3 -m py_compile \
  exllamav3/modules/hyv4.py \
  exllamav3/modules/mlp.py \
  exllamav3/modules/quant/fp16.py \
  exllamav3/modules/quant/exl3.py \
  exllamav3/conversion/convert_model.py \
  exllamav3/loader/safetensors_alt.py
```

On the first layer quantized after applying `dbcb963`, also confirm the coordinator log prints `-- Normal-pos (model.layers.<N>, |ref|<=1000): ...`.

## Memory guard and co-location rule

DGX Sparks have `121.7 GiB` unified memory and `0` swap. Run `quant/memfloor.sh` on the host beside every quantization container so the container stops cleanly before Linux runs out of memory:

```bash
bash quant/memfloor.sh 8 docker hy4-exl3-coord "$LOG_DIR/coord.memfloor"
```

Use floor `18` GiB on `node3` while source weights are still downloading, and `8` GiB otherwise (coordinator, helper workers, and optional trunk pass).

**Hard rule: do not load a serving rank on a machine that is running a quantization pass.** A quantization pass holds `~25 to 40 GiB` of calibration state and workspace in unified memory with zero swap. Loading a serving rank on top pushes total demand past `121.7 GiB` and can hard-hang the node, requiring a physical power cycle and dropping any direct 200 Gb/s cable links routed through it. Pause the quantization pass at a layer boundary before loading serving ranks on that machine.

## Steps in order

In the commands below, `/farm` is the container mount pointing to your host config and recipe directory (`-v "$CONFIG_DIR:/farm"`).

1. **Check out ExLlamaV3 `v1.6.7` (`8c593e4`) and apply the Hy4 quantizer patch**:
   ```bash
   git clone https://github.com/turboderp-org/exllamav3.git "$EXL3_DIR"
   git -C "$EXL3_DIR" checkout 8c593e4e5c88da76a4aaf8f1b1838bb7c1fce4e0
   git -C "$EXL3_DIR" apply quant/patches/0001-hy4-quant.patch
   ```
   *(Maintainers regenerating `quant/patches/0001-hy4-quant.patch` from the quantizer checkout: `EXL3_REPO="$EXL3_SRC" bash quant/make_patches.sh --check "$EXL3_DIR" 3b697ca1f55a53ffce9a5d2b612146f33f4b20ca quant/patches`, where `$EXL3_DIR` is a clean checkout of `v1.6.7` `8c593e4e5c88da76a4aaf8f1b1838bb7c1fce4e0`.)*
2. **Build ExLlamaV3 on every node** inside `nvcr.io/nvidia/pytorch:26.07-py3`:
   ```bash
   unset EXLLAMA_NOCOMPILE
   export TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS=20
   pip install --no-deps --no-build-isolation -v .
   pip install marisa-trie
   ```
   Always unset `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` at runtime (`env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`).
3. **Use or regenerate the routed-expert recipe** (`quant/recipe-l20-76.yaml` is the frozen flat `3.00 bpw` recipe for all 78 layers):
   ```bash
   python3 quant/farm/make_hy4_recipe.py "$MODEL_DIR/config.json" "$CONFIG_DIR/recipe-l20-76.yaml" --expert-bits 3
   ```
4. **Start stateless TCP workers** on `node1` and `node2` (and optionally `node4`), bound to 200 Gb/s cable IPs:
   ```bash
   env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE python3 quant/farm/exl3_farm.py serve --port 5599 --device 0 --concurrency 1
   ```
   Verify from `node3` that every helper routes over a direct cable (`ip route get <worker-ip>` shows `dev enp1s0f*`).
5. **Run layers 0 to 19** on `node3` (optional first stage if shards for layers 20 to 77 are still downloading; `quant/shard-order.txt` gives layer-ordered shard pull order):
   ```bash
   EXL3_REMOTE_WORKERS="$NODE1_IP:5599,$NODE2_IP:5599" \
   EXL3_REMOTE_WORKERS_FILE="/farm/workers.txt" \
   EXL3_FARM_PREFETCH=1 \
   env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE python3 -m exllamav3.conversion.convert \
     -i "$MODEL_DIR" -o "$OUT_DIR" -w "$WORK_DIR" \
     -b 3 -hb 16 -mb 16 -cb mul1 --out_scales always \
     -cd "$CAL_DATA" -cr 250 -cc 2048 \
     -rcp /farm/recipe-l20-76.yaml --max_module 21
   ```
6. **Run layers 20 to 77** after layer 19 checkpoints and all 245 shards are on disk (`--override_anyway`, all quant flags explicit, `--max_module 79`):
   ```bash
   EXL3_REMOTE_WORKERS="$NODE1_IP:5599,$NODE2_IP:5599" \
   EXL3_REMOTE_WORKERS_FILE="/farm/workers.txt" \
   EXL3_FARM_PREFETCH=1 \
   env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE python3 -m exllamav3.conversion.convert \
     -i "$MODEL_DIR" -o "$OUT_DIR" -w "$WORK_DIR" -r --override_anyway \
     -b 3 -hb 16 -mb 16 -cb mul1 --out_scales always \
     -cd "$CAL_DATA" -cr 250 -cc 2048 \
     -rcp /farm/recipe-l20-76.yaml --max_module 79
   ```
   - **Container path rule**: `-rcp` must be the path inside the coordinator container (such as `/farm/recipe-l20-76.yaml`), not the host filesystem path.
   - **Resume flag rule**: Never run a bare `-r` resume without `--out_scales always` and `-cb mul1 -hb 16 -mb 16`; `convert.py` resets `--out_scales` to `auto` if omitted.
   - **Live worker edits**: Editing `/farm/workers.txt` (one `<ip>:5599` per line) adds or removes helper nodes at the next layer boundary without restarting the coordinator.

## Stopping and resuming cleanly at a layer boundary

1. **Wait for the current layer to checkpoint** on `node3`. Watch `$WORK_DIR/ckpt/job.json` until `next_module_idx` increments and confirm `bad_rows` is `[]` (the log prints `-- Quantized: model.layers.<L>` followed by `-- Loading unquantized module: model.layers.<L+1>`).
2. **Stop in order**:
   ```bash
   # On node3: stop watcher (if enabled), then stop and remove coordinator
   systemctl --user stop hy4-farm-watch.service 2>/dev/null || true
   docker stop -t 15 hy4-exl3-coord && docker rm -f hy4-exl3-coord
   # On each helper node (node1, node2): stop and remove worker
   docker stop hy4-exl3-worker && docker rm -f hy4-exl3-worker
   ```
   Do not delete `$WORK_DIR/qtensors/` or `$WORK_DIR/ckpt/`.
3. **Resume**: restart `hy4-exl3-worker` on each helper node, verify cable routes (`ip route get <worker-ip>` shows `dev enp1s0f*`), and re-run Step 6 with the exact same `-r --override_anyway`, quant flags, and `--max_module 79`.
4. **Check what the resume printed** before leaving it unattended:
   - Startup banner prints `-- Resuming existing job` (not a fresh start) and `-- Resuming at: model.layers.<N>` matching `next_module_idx - 2`.
   - Startup banner prints `Codebook: mul1` and `Output scales: always`, and `$WORK_DIR/args.json` shows `"apply_out_scales": true` and `"max_module": 79`.
   - On the first resumed layer, the log prints `-- Farming 768 tensors over [0, 'remote:<ip1>:5599', ...]` and ends with `-- remote <worker-ip>: <N> tensors in <T> s` (`N > 0`, zero `[local fallback]` lines) for every helper, `-- Normal-pos (model.layers.<N>, |ref|<=1000): ...`, and `bad_rows: []` in `ckpt/job.json`.

## Trunk pass (4-bit and 6-bit sidecars)

The default build (the full quant) serves the attention and shared-expert weights from packed EXL3 copies; the lossless trunk option serves them from the base pack. The trunk pass writes those copies as sidecar folders, `trunk-4b/` and `trunk-6b/`, each also holding a packed `lm_head.safetensors` and the drafter's `dfly_layers.safetensors`, without altering the base pack. `TF_HY4_TRUNK=4b6b` reads from both folders. Drafts in either build use the drafter files in `trunk-4b/`. Run `farm/quantize_trunk_pass.py` (included in `quant/patches/0001-hy4-quant.patch`) alone on `node4` in `--follow` mode after removing `node4` from the main job's helper list.

- **Recipes**: `quant/recipe-trunk-4b.yaml` and `quant/recipe-trunk-6b.yaml` (values from commit `c7c901f`; the files are the copies in `3b697ca`, where one comment line of the 4-bit recipe was reworded). Both set `codebook: mul1`, `out_scales: always`, and include two lossless overrides:
  ```yaml
  overrides:
    model.layers.75.mlp.shared_experts.down_proj: 16
    model.layers.76.mlp.shared_experts.down_proj: 16
  ```
  Those two matrices stay in lossless BF16 because `silu(gate) * up` entering `shared_experts.down_proj` exceeds FP16 range at layers 75 (`76,661.9`) and 76 (`1,510,548.9`). Keeping them lossless avoids Hessian distortion from the extreme positions and avoids passing values above `65,504` into an FP16-input EXL3 kernel, at a VRAM cost of `+4.50 MiB` per node versus 4-bit (`+3.75 MiB` per node versus 6-bit) per matrix.
- **How `--follow` trails the main job**: Mount the staged base layers read-only at `/model:ro`. For each layer `L`, `--follow` waits until `L` is both listed in verified layers in `farm_state.json` and present in `/model/model.safetensors.index.json` (written only after SHA-256 staging). Existing base files are never modified; sidecars are written under `/trunk-out/trunk-4b/` and `/trunk-out/trunk-6b/`.
- **Measured pace**: On one GB10 (`node4`, `250 x 2,048` calibration, dual-width 4-bit + 6-bit together, unshared, no main-job worker on `node4`), mean **`3.97 min/layer`**, median `3.91 min/layer`, range `3.78 to 5.08 min/layer` across layers `0 to 26` (`3.93 min/layer` steady mean across layers `2 to 26`).
- **Command** (inside `nvcr.io/nvidia/pytorch:26.07-py3` on `node4` alone, guarded by `quant/memfloor.sh 8`; add `--resume` when restarting):
  ```bash
  env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE python3 -u farm/quantize_trunk_pass.py \
    --model-dir /model --extra-model-dir /extra-model \
    --config /farm/config-full.json \
    --out-dir /trunk-out --work-dir /trunk-work \
    --recipe /farm/recipe-trunk-4b.yaml \
    --extra-recipe /farm/recipe-trunk-6b.yaml \
    --cal-data /work/farm-250x2048.safetensors --cal-rows 250 --cal-cols 2048 \
    --start-layer 0 --stop-file /trunk-work/STOP \
    --follow --poll-s 20 --farm-state /trunk-work/farm_state.json
  ```
- **Stopping and resuming**: Create `/trunk-work/STOP` to exit cleanly after the current layer, then re-run with `--resume`. `--resume` reloads recipes at startup and applies updated `overrides:` per layer without restarting from layer 0. Confirm the resume log prints `Codebook: mul1`, `Output scales: always`, and `-- Resumed checkpoint at layer <M>`. After an unplanned power cut, load-check `ckpt/job.json`, `ckpt/preserve.pt`, `ckpt/state.safetensors` (`0` non-finite values, `0` all-zero rows), and the newest `trunk-{4b,6b}/model.layers.<M-1>.safetensors` read-only before resuming.

## Packaging and source verification after the job ends

1. **Package the final model** after layer 77 checkpoints on a node with free disk space (`compile.py` or the CPU packager included in `quant/patches/0001-hy4-quant.patch`, copying `DFly2` unquantized; the pack has no `vit.*` tensors, so build the vision folder with `python3 tools/vit_folder.py "$MODEL_DIR" vit-bf16`), then fan out over the 200 Gb/s cables to all four nodes (`launch/STAGING.md`).
2. **Source weight check (run after the job ends)**: Re-hash all 294 files of `tencent/Hy4` (`c10c0de5`) on `node3` against the repository manifest (SHA-256 for LFS shards, git blob SHA-1 for metadata, with `posix_fadvise(POSIX_FADV_DONTNEED)` so page cache stays bounded). Expected terminal line: `VERIFY PASS 294 files`. Run on `2026-11-21` on `node3`: `VERIFY PASS 294 files` in `1,979 s`, exit code `0` ([log](../receipts/source/verify.log)).

## Script run status

- `quant/make_patches.sh`: verified on `2026-11-21` with `--check` against a clean public checkout of `v1.6.7` (`8c593e4e5c88da76a4aaf8f1b1838bb7c1fce4e0`, tree `6e0ee80353142ef8fb8056b0441237665e39db86`); for quantizer head `3b697ca` (`hy4-full-scrub`), `git apply` produces tree `3b8a8faee9920fee81ccd3703be5034be10157df` (`0` diff bytes against `3b697ca` with the local farm launcher and harness files excluded; patch `sha256` `8bd6382fea63acb67079d892968bd5917484a91e1fb67b93d5e3eeadfb5282eb`; unscrubbed `c7c901f` produces tree `37530ca3e8a9d0a1510221f04a456c1e929dd3c2`). No quantization run was made from the patched public tree; what stands is this tree proof.
- `quant/farm/make_hy4_recipe.py`: `--help` run on `2026-11-20`; full recipe generation was not run standalone from this kit (`quant/recipe-l20-76.yaml` was generated by the same script inside the `exllamav3` checkout).
- `py_compile` verification command for the six `dbcb963` files: run as written on `2026-11-20` inside `nvcr.io/nvidia/pytorch:26.07-py3` (`6` files compiled, `6/6` SHA-256 digests matched across coordinator and trunk nodes).
- `pip install` ExLlamaV3 build commands: not run as written from this kit (wheels were built inside `nvcr.io/nvidia/pytorch:26.07-py3` from the patched `exllamav3` checkout with `TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS=20`).
- `quant/farm/exl3_farm.py`: not run as written from this kit (ran via `farm/launch_hy4_farm.sh` from the patched `exllamav3` checkout inside Docker on `node1`, `node2`, `node4`).
- Main quantization commands (layers `0 to 19`, layers `20 to 77`, stop and resume): not run as bare commands from this kit (ran wrapped by `farm/launch_hy4_farm.sh` inside Docker on `node3` with container mounts, `hy4-farm-watch.service`, and the memory guard).
- `quant/memfloor.sh`: not run as written from this kit (the same script ran from the host oracle directory on all four nodes during quantization on `2026-11-20`).
- Trunk pass (`farm/quantize_trunk_pass.py` with `quant/recipe-trunk-4b.yaml` and `quant/recipe-trunk-6b.yaml`): not run as bare command from this kit (ran wrapped by `farm/launch_hy4_trunk.sh` on `node4` with identical recipe files; it completed layers `0 to 77` for both widths, and `launch/stage_trunk.py --verify-only` passed for all 81 files of each folder on all four nodes at `2026-11-21T07:38Z`).
- Final packaging (step 1 of "Packaging and source verification"): not run from this kit.
- Source weight check (step 2): run on `2026-11-21` on `node3` (`17:08:12` to `17:41:11` UTC), ending `VERIFY PASS 294 files` with exit code `0` (`receipts/source/verify.log`). The check script that ran is not in this kit.
