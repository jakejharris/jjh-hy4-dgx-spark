# Building the TensorFold `1.3` Hy4 Server

The serving binary set is built on public TensorFold `1.3` (`main` commit `2cee7d938a5f72f89ce2062c69e705b4dd74c7ce`), with the Hy4 integration patch (`serve/patches/0001-hy4-port.patch`, cut from our merged engine commit `ef8ebb7c8` (branch `hy4-next`), which is not public; it carries the Hy4 vision tower, the fast prefill (attention and token index on tensor cores, one ring all-reduce), the Apache-2.0 `hy4_flash` and shared-slab additions from upstream PR `#4004`, the full Hy4 CUDA kernels and a second pass of kernel tuning, packed EXL3 trunk, four-rank TP boundary, DFly2 drafter, opt-in repeat-stable adaptive drafting, and `-Dstrip=true` executable stripping) applied directly on top.

## Source layout

- **Base**: public TensorFold `main` commit `2cee7d938a5f72f89ce2062c69e705b4dd74c7ce` (tree `2d7f95316e79f897f3b25de7f14011334eb00b97`); `serve/patches/0001-hy4-port.patch` stands on `2cee7d9` alone and omits any local run notices.
- **Kernels**: `zig/kernels/cuda/hy4/` (`ihc`, `ihc_decode`, `head_bf16`, `mla_project`, `mla_attention`, `attention_sink`, `token_indexer`, `router`, `routed`, `routed_wide`, `trunk`, `trunk_range`), `zig/src/cuda/hy4_{kernels,ops,routed,trunk,trunk_range,column_plan,tp*}.zig`, and `zig/kernels/cuda/contract_hy4*.zig`.
- **Family and serving**: `zig/src/families/hy4/` (`hy4`, `config`, `weights`, `exl3_weights`, `trunk_exl3*`, `pack_parts`, `pack_export`, `ranks`, `boundary`, `forward`, `moe_forward`, `head_bf16`, `greedy`, `cache`, `kv_pool`, `direct_load`, `resident`, `drafter`, `draft_policy`, `repeat_stable`, `speculative`, `window_bench`, `cuda_backend`, `cuda_native`, `dump_oracle`), `zig/src/fabric/tp/`, `zig/src/native/{cuda_families,cuda}.zig`, `zig/build/{cuda,cuda_engines,hy4_cuda,kernel_contract}.zig`.
- **Vision**: `zig/kernels/cuda/hy4/vision.cu`, `vision_pos.h`, `zig/src/cuda/hy4_vision.zig`, `zig/src/families/hy4/vision_*.zig`, `zig/src/core/engine_vision.zig` and `zig/src/core/picture/png.zig`. The fast prefill changes files listed above and adds none.

## Toolchain (GB10, `sm_121`)

- Zig `0.17.0` (matching `.zig-version`)
- CUDA `13.0` (`nvcc` `V13.0.88`, `libcudart.so.13.0.96`)
- Target: `-Dsm=121` (arch-specific kernels compile to `sm_121a`)

Ensure `zig` and `nvcc` are on `PATH`:

```bash
export PATH="$ZIG_DIR:/usr/local/cuda-13.0/bin:$PATH"
```

## Pin the build: container, cores and a clean cache

Zig builds for the CPU core it starts on. A DGX Spark has two kinds: cores 0 to 4 and 10 to 14 are Cortex-A725, cores 5 to 9 and 15 to 19 are Cortex-X925, and each gives different machine code. Zig also keeps the first core's choice in `.zig-cache`. So pin the build to cores 0 to 4, clear the cache first, and build inside the same container the server runs in:

```bash
docker run --rm --pull=never --network none --memory=18g --ipc host \
  --user "$(id -u):$(id -g)" \
  -v "$TF_DIR:$TF_DIR" -v "$OUT_DIR:$OUT_DIR" -v "$ZIG_DIR:/opt/zig:ro" \
  -v /usr/local/cuda-13.0:/usr/local/cuda-13.0:ro \
  -w "$TF_DIR" -e ZIG_LOCAL_CACHE_DIR="$TF_DIR/.zig-cache" \
  --entrypoint /bin/bash nvcr.io/nvidia/pytorch:26.07-py3 -ec "
    rm -rf '$TF_DIR/.zig-cache' zig-out
    taskset -c 0-4 /opt/zig/zig build fatbins hy4-oracle native hy4-tp-boundary-bin \
      -Dstrip=true -Doptimize=fast -Dnvcc=/usr/local/cuda-13.0/bin/nvcc -Dsm=121 \
      -Dtp-fatbin='$OUT_DIR/fatbin/tp_mailbox.fatbin' -p '$OUT_DIR' -j2 --summary all"
objdump -d "$OUT_DIR/native/bin/tensorfold-native" | grep -c ldapur   # Cortex-A725 code: more than 0
```

Built this way on 2026-11-22, the earlier build's source gave its `tensorfold-native` byte for byte (56,463,624 bytes, SHA-256 `4a3d5265523408f302dfeb2066cdae1f4fe9f28276310c9d3c82df1c0aa35695`). Builds on unpinned cores, or over an old `.zig-cache`, gave other bytes from the same source.

The binary's file path also moved decode speed. Those same bytes, run from a different path, decoded RigMark's structured workload at 54.5 tok/s against 55.5 from the release path (median of 3 runs each, GPU clocks capped at 2,200 MHz; `receipts/next/prefill/base6e-exact-winb/` and `base-2200/`). So compare two builds from the same path. This release's speed steps were built with the block above and compared that way.

## Build steps

`bash serve/build_release.sh --tf-dir DIR --out-dir DIR --zig-dir DIR` runs steps 1 to 3 in one go, pinned as above, on the Spark that builds: it checks the base checkout is `2cee7d9` with no local changes, applies the patch, builds, checks for Cortex-A725 code and writes `OUT_DIR/bin/SHA256SUMS`. Then, unless `--no-codecs`, it runs `serve/build_codecs.sh` into `OUT_DIR/lib`: the JPEG and WebP decoders (`libturbojpeg.so.0` from libjpeg-turbo 3.1.4.1, `libwebpdecoder.so.3` from libwebp 1.6.0), built with `zig cc` from the upstream tarballs, each checked against the SHA-256 pinned in the script, in about 30 s. `--codec-src DIR` takes the tarballs from a local folder, and the script prints which source it used. Without the decoders, JPEG and WebP get HTTP 400. `launch/stage_binary.sh` then copies the four files, and the two decoders when the build has them, to the same folder on every Spark and checks each hash there. The steps below are what the script does.

1. **Check out `2cee7d9` from public TensorFold and apply the port patch**:
   ```bash
   git clone https://github.com/ashhart/TensorFold.git "$TF_DIR"
   git -C "$TF_DIR" checkout 2cee7d938a5f72f89ce2062c69e705b4dd74c7ce
   git -C "$TF_DIR" apply serve/patches/0001-hy4-port.patch
   ```
   *(Maintainers regenerating `serve/patches/0001-hy4-port.patch` from the port checkout: `TF_REPO="$PORT_REPO" bash serve/make_patches.sh --check "$TF_DIR" ef8ebb7c87aa81ff4afe4bb43bb38d05f5dd4e4e serve/patches`, where `$TF_DIR` is a clean checkout of `2cee7d938a5f72f89ce2062c69e705b4dd74c7ce`.)*

   The patch in `serve/patches/` is this release's: cut from the merged engine at `ef8ebb7c87aa81ff4afe4bb43bb38d05f5dd4e4e` (branch `hy4-next`, `next.engine.commit` in `card/claims.json`) with `--check`. The earlier build's patch was cut the same way from `6e0fcf1ec`.
2. **Build the four-rank RoCE/TP mailbox fatbin (`tp_mailbox.fatbin`, `-O3`)**:
   ```bash
   mkdir -p "$OUT_DIR/bin" "$OUT_DIR/fatbin"
   /usr/local/cuda-13.0/bin/nvcc -fatbin -O3 -std=c++17 \
     -gencode=arch=compute_120,code=sm_120 \
     -gencode=arch=compute_121,code=sm_121 \
     -gencode=arch=compute_90,code=compute_90 \
     -o "$OUT_DIR/fatbin/tp_mailbox.fatbin" "$TF_DIR/zig/src/fabric/tp/mailbox.cu"
   ```
3. **Clean release build of CUDA kernel fatbins and stripped Hy4 executables** (`-Dstrip=true -Doptimize=fast` strips all DWARF debug sections and build-machine paths from all three ELF binaries; `-j4` / `-j2` keeps parallel `nvcc` + Zig memory under 16 GiB):
   ```bash
   cd "$TF_DIR"
   zig build fatbins \
     -Dstrip=true -Doptimize=fast -Dnvcc=/usr/local/cuda-13.0/bin/nvcc -Dsm=121 \
     -p "$OUT_DIR" -j4
   zig build hy4-oracle native hy4-tp-boundary-bin \
     -Dstrip=true -Doptimize=fast -Dnvcc=/usr/local/cuda-13.0/bin/nvcc -Dsm=121 \
     -Dtp-fatbin="$OUT_DIR/fatbin/tp_mailbox.fatbin" \
     -p "$OUT_DIR" -j2
   mv -f "$OUT_DIR/native/bin/tensorfold-native" "$OUT_DIR/bin/tensorfold-native"
   rmdir "$OUT_DIR/native/bin" "$OUT_DIR/native"
   mv -f "$OUT_DIR/hy4-boundary" "$OUT_DIR/bin/hy4-boundary"
   chmod +x "$OUT_DIR/bin/tf-hy4-oracle" "$OUT_DIR/bin/tensorfold-native" "$OUT_DIR/bin/hy4-boundary"
   ```
4. **Fast inner-loop rebuilds** (after `$OUT_DIR/fatbin/` is populated, reuse prebuilt fatbins without re-running `nvcc`):
   ```bash
   cd "$TF_DIR"
   zig build hy4-oracle native hy4-tp-boundary-bin \
     -Dstrip=true -Doptimize=fast -Dfatbins="$OUT_DIR/fatbin" \
     -Dtp-fatbin="$OUT_DIR/fatbin/tp_mailbox.fatbin" \
     -p "$OUT_DIR" -j2 --summary all
   ```
5. **Host and GPU verification**:
   ```bash
   cd "$TF_DIR"
   zig build test-gradient-serving test-hy4-host test-hy4-ops-host test-draft-cuda-host test-hy4-flash-host \
     -Dfatbins="$OUT_DIR/fatbin" --summary all
   "$OUT_DIR/bin/tf-hy4-oracle" --self-test
   ```

## Shipped release files, stripping ruling, and sizes (`aarch64-linux-gnu`, `sm_121`)

All three host ELF executables ship **stripped** (`-Dstrip=true -Doptimize=fast`, removing `.debug_*`, `.symtab`, `.strtab`, and all build-host paths); CUDA device fatbins ship **not host-stripped** (`nvcc -fatbin -O3` without `-G` or `-lineinfo`, which already omits host paths and debug sections):

- `$OUT_DIR/bin/tensorfold-native`: **stripped** (`-Dstrip=true -Doptimize=fast`, embeds `tp_mailbox.fatbin` and all 140 engine fatbins; logs `repeat_stable=on|off` at startup, defaulting to `off` unless `TF_HY4_REPEAT_STABLE=1` or `,untimed:1` in `TF_HY4_ROW_COST_TABLE`; the kit's launch files pass `TF_HY4_REPEAT_STABLE=1`, see the README launch settings) - **`57,027,720` bytes** (`54.39 MiB`, this release's build; the earlier build's was 56,463,624).
- `$OUT_DIR/bin/tf-hy4-oracle`: **stripped** (`-Dstrip=true -Doptimize=fast`, layer-streamed CUDA forward, `--self-test`, `--check-rank-pack`, and parity oracle dump runner) - **`12,776,824` bytes** (`12.19 MiB`; `25,472,280` bytes unstripped).
- `$OUT_DIR/bin/hy4-boundary`: **stripped** (`-Dstrip=true -Doptimize=fast`, four-process TP gather and fold boundary contract binary) - **`4,191,408` bytes** (`4.00 MiB`; `10,592,336` bytes unstripped).
- `$OUT_DIR/fatbin/tp_mailbox.fatbin`: **not stripped** (`nvcc -fatbin -O3` release device binary carrying `sm_120` cubin, `sm_121` cubin, and `compute_90` PTX) - **`428,544` bytes** (`418.50 KiB`), alongside the 140 `-O3` `sm_121a` engine fatbins in `$OUT_DIR/fatbin/*.fatbin`.

The sizes of `tf-hy4-oracle`, `hy4-boundary` and `tp_mailbox.fatbin` above are the earlier build's. This release's build printed only `tensorfold-native`'s size; `OUT_DIR/bin/SHA256SUMS` holds all four hashes.

## Script and build run status

- `serve/make_patches.sh`: run on `2026-11-22` with `--check` against a clean checkout of `2cee7d938a5f72f89ce2062c69e705b4dd74c7ce` (tree `2d7f95316e79f897f3b25de7f14011334eb00b97`); for this release's head `ef8ebb7c87aa81ff4afe4bb43bb38d05f5dd4e4e` (`hy4-next`), `git apply` produces tree `78760a9bf5ee769524d6b792f7e81aaf265625ee`, the head's tree with upstream's notice files restored (`check_patch: PASS`; patch 2,225,825 bytes, `sha256` `871bbf61298c7b8b8947aed117a06f2f160ca7bc8e2978b8488ed2ab78f77b6f`). The earlier build's patch, from `6e0fcf1ec`, was checked the same way on `2026-11-21`.
- `zig build` and GPU verification: for the earlier build's patch, executed and verified on `2026-11-21` on GB10 (`aarch64-linux-gnu`, `sm_121`, Zig `0.17.0`, CUDA `13.0.88`) from both the release checkout and a clean `2cee7d9` + patch checkout. This release's patch is built by `serve/build_release.sh` in the release run.
