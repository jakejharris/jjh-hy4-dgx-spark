<p align="center">
  <img src="assets/hero.jpg" width="100%" alt="Four DGX Sparks carrying a giant penguin in a gold scarf. JJH HY4 1.0 for 4x DGX Sparks. 774B parameters. Four small boxes.">
</p>

# JJH HY4 1.0 for 4x DGX Sparks

A recipe for running Tencent's Hy4 (774B parameters, checkpoint `c10c0de5`) on four DGX Sparks cabled in a ring: quantized to {{bare:quant.text.bpw.4b6b}} bits per weight, served by TensorFold four-way tensor parallel over RoCE.

The claims ledger, `card/claims.json`, ties each measured value to the run that produced it.

[Weights](https://huggingface.co/jakejharris/jjh-hy4-exl3) · [What you need](#what-you-need) · [Steps](#steps) · [Limits](#limits) · [Findings](FINDINGS.md) · [Claims ledger](card/claims.json)

## Speed

This release, one stream, the full quant:

- **{{bare:next.decode.code.match}} tok/s** decode on code, **{{bare:next.decode.prose.match}}** on prose and **{{bare:next.decode.structured.match}}** on RigMark's structured task, greedy.
- **{{bare:next.prefill.8k.fp32}} / {{bare:next.prefill.32k.fp32}} / {{bare:next.prefill.64k.fp32}} tok/s** cold prefill at 8K / 32K / 64K.
- **{{value:next.ttft.cold.p128}}** to the first token at about 128 prompt tokens, **{{value:next.ttft.cold.p1024}}** at 1K and **{{value:next.ttft.cold.p4096}}** at 4K.
- **{{bare:next.fix07.nothink.decode.code}} tok/s** on code and **{{bare:next.fix07.nothink.decode.prose}}** on prose, sampled at temperature 0.7 with thinking off.
- **131,072-token** context. At 128K the needle test found its code, with thinking on and off.

Drafts on, thinking off unless marked, each Spark's graphics clock capped at 2,200 MHz. Decode and first-token times are medians of 3 runs; prefill is one cold request per depth.

Prefill is the slow side: a cold 64K prompt waits **{{value:next.prefill.64k.ttft.fp32}}** for its first token, and the 128K needle prompt **{{value:next.niah128k.ttft.fp32}}**. Later releases work on it.

### Reading the numbers

<details>
<summary>How each number was measured, and on which build</summary>

This release ships our merged engine (`ef8ebb7c`, the release build) with one fix that makes sampled decode faster: {{value:next.fix.commit}}. On the fixed build, greedy and same-seed sampled requests gave the release build's tokens exactly: {{value:next.fix.parity}} (`receipts/next/release/fix-parity/`). So the release build's values stand for 1.0, except the speed of sampled decode.

**Decode and prefill.** [RigMark](#rigmark) 1.3.0 with custom settings, on the release build booted through `launch/launch.sh`. Prefill is prompt tokens over the client's time to first output, for a cold prompt on an already running serve: one cold request and one immediate replay per depth, 8 output tokens. Decode is the median of three runs per workload, from a decode-only run. Setups: `card/claims.json` (`next.prefill.*`, `next.decode.*`).

**BF16 sums.** Prefill adds the four Sparks' partial results in FP32 by default (`TF_HY4_PREFILL_AR=fp32`). BF16 sums are faster, {{bare:next.prefill.8k.bf16}} / {{bare:next.prefill.32k.bf16}} / {{bare:next.prefill.64k.bf16}} tok/s at 8K / 32K / 64K in a prefill-only run, with lower agreement ([Accuracy](#accuracy)). Decode does not depend on how prefill sums.

**First token on short prompts.** Cold prompts on the release build, kit defaults, vision on, thinking off, greedy, 8 output tokens, one request at a time. The time is the client's, from sending the request to the first streamed piece of output. Each prompt starts with a fresh random line, so no prefix is cached, and the server reported 0 cached tokens on all 21 requests.

| Prompt tokens (the three requests) | Time to first token, median of 3 (min to max) |
| --- | --- |
| 113, 128, 129 | {{value:next.ttft.cold.p128}} ({{spread:next.ttft.cold.p128}}) |
| 245, 247, 253 | {{value:next.ttft.cold.p256}} ({{spread:next.ttft.cold.p256}}) |
| 302, 303, 308 | {{value:next.ttft.cold.p300}} ({{spread:next.ttft.cold.p300}}) |
| 488, 509, 517 | {{value:next.ttft.cold.p512}} ({{spread:next.ttft.cold.p512}}) |
| 1,010, 1,017, 1,022 | {{value:next.ttft.cold.p1024}} ({{spread:next.ttft.cold.p1024}}) |
| 2,043, 2,043, 2,047 | {{value:next.ttft.cold.p2048}} ({{spread:next.ttft.cold.p2048}}) |
| 4,090, 4,101, 4,109 | {{value:next.ttft.cold.p4096}} ({{spread:next.ttft.cold.p4096}}) |

The row near 250 tokens is slower than the row near 300. The fast prefill path starts at 256 prompt rows (`TF_HY4_FAST_PREFILL_MIN_ROWS`); that is the likely reason, and it was not isolated by a run. The script is the kit's own `bench/ttft_sweep.py`, not RigMark (`receipts/next/release/ttft/`).

**Faster sampling.** On the release build, a sampled request (any `temperature` above 0) moved every sampled row's full logits between the four Sparks over TCP, in 16 KiB pieces: about 90 four-rank exchanges in a round of five drafts, read from the code. Greedy moves 8 bytes. With thinking on at 0.7, that build decoded {{value:next.temp07.decode.code}} on the code prompt, against {{value:next.temp07.greedy.code}} greedy on the same boot.

The fixed build sends the same bytes over the RoCE link in one or two calls a round and leaves greedy alone, and the same seed gives the same tokens. With the same settings it decoded {{value:next.fix07.decode.code}}, against {{value:next.fix07.greedy.code}} greedy. With thinking off, code written at 0.7 came at {{value:next.fix07.nothink.decode.code}}.

- **Default temperature.** A request that sends no `temperature` gets 0.7 from the launcher, not Tencent's 0.9 ([`SERVE_TEMPERATURE`](launch/SETTINGS.md#serve_temperature)). Defaults in the serving line: {{value:next.fix07.serving}}.
- **Decode at 0.7.** On the fixed build, thinking off, a 2,048-token cap, median of three: {{value:next.fix07.nothink.decode.prose}} and {{value:next.fix07.nothink.decode.code}} on the prose and code prompts, against {{value:next.fix07.nothink.greedy.prose}} and {{value:next.fix07.nothink.greedy.code}} greedy on the same boot. The reply is the essay or the code itself. Running alongside: {{value:next.fix07.alongside}}.
- **Reasoning at 0.7.** The same boot and prompts, thinking on: {{value:next.fix07.decode.prose}} and {{value:next.fix07.decode.code}}, against {{value:next.fix07.greedy.prose}} and {{value:next.fix07.greedy.code}} greedy. Every reply was still reasoning at the cap, so these are reasoning speeds.

**Drafts.** With drafts on, Hy4's small drafting model (DFly2) proposes up to six tokens a pass and the full model keeps the ones it would pick itself, which is faster. The repeat option (`TF_HY4_REPEAT_STABLE=1`, on by default) makes the same prompt give the same tokens with drafts on ([measured](card/QUALITY_CARD.md#repeat-option)). `"draft": false` on a request gives plain output.

**Clocks.** Each run holds one graphics-clock snapshot per Spark, taken once early in the run (2,184 MHz in the four bench runs here, 2,171 to 2,190 MHz across the round's runs), not a memory clock and not a trace over the whole run. RigMark's own card in those receipts says stock clocks; for these runs that line is wrong.

**Engine.** Agreement was measured on speed step 2 and the vision tower's closeness on a vision build, two builds of our engine that the release build merges. This release's prefill, decode, needle and image-request values come from the release build. Its commit is {{value:next.engine.commit}}.

Its gate: {{value:next.gate.merged}}.

1.0 ships the release build with the sampling fix: {{value:next.fix.commit}}, branch `hy4-sample-fast`.

Its gate: {{value:next.fix.gate}}.

`serve/patches/0001-hy4-port.patch` is cut from the fixed commit ({{value:next.fix.commit}}) by `serve/make_patches.sh --check`, so building from this kit gives the fixed build ([serve/BUILD.md](serve/BUILD.md)).

</details>

### RigMark

[RigMark](THIRD_PARTY.md#rigmark) is a third-party benchmark, credited with its license. It puts a nonce made from `--comparison-id` into every prompt, so decode rates compare only under one id. This release's decode rows ran under `2026-11-22-hy4-speed-team-2200mhz`; an earlier decode run of the release build used the kit's own id, so other prompts (`next.decode.*.fp32` in `card/claims.json`).

Its structured task is a predictable JSON reply, a ceiling. Our one run of its full standard suite was cut short by a power loss ([Development notes](#development-notes)).

## What you get

**Vision.** The model's own vision tower runs on rank 0 for still images, sent as OpenAI `image_url` parts with a data URL. A 1440x900 screenshot is {{value:next.vision.tokens_1440x900}}. It answered {{value:next.vision.known_answer.merged}} known-answer images right. That is a spot check on three images, not a benchmark.

Vision is off unless `SERVE_VISION=1` ([settings](launch/SETTINGS.md#serve_vision)). The tower is a separate folder, `vit-bf16/` (2.47 GB), on rank 0 only: in the weights repository, or built from Tencent's checkpoint ([step 6](#steps)).

<details>
<summary>Vision in detail: the tower, how close it is, speed, formats and request limits</summary>

**The tower.** `vit-bf16/` holds `vit.safetensors` (2,470,326,144 bytes, the 649 `vit.*` tensors copied unchanged in BF16 from `tencent/Hy4` `c10c0de5`) with the source's `config.json` and `processor_config.json`. `python3 tools/vit_folder.py SOURCE_DIR OUT_DIR` builds it and `launch/vision.sha256` holds its hashes.

Point `TF_HY4_VISION_DIR` at that folder on rank 0. `launch/preflight.sh` refuses the launch if a file is missing or a hash differs. Rank 0 holds {{value:next.vision.rank0_extra_gib}} more, and its lowest free memory at 128K was {{value:next.vision.min_free128k}}. On the vision build, {{value:next.vision.text_same}} text requests gave the same tokens with vision on and off.

**How close it is.** Every 4-layer stretch of the tower matches PyTorch BF16 at a cosine of {{bare:next.vision.segment_cos_min}} or better. End to end it is as close to a full-precision PyTorch run as PyTorch's own BF16 is ({{spread:next.vision.e2e_cos_fp32}}), because the tower amplifies BF16 rounding.

**Speed.** On the vision build, without the fast path, image requests prefilled at {{spread:next.vision.prefill.vision_build}}. On the release build, the median of the three known-answer images: {{value:next.vision.prefill.merged}}. Gates and answers: `receipts/next/vision/`.

On the release build, three screenshots in one request (3,951 prompt tokens) took 24.4 s to the first token, and two images filling the 8,192-token cap (8,255 prompt tokens) took 70.5 s, with rank 0's lowest free memory at 21.61 GiB (2026-11-22, `receipts/next/release/engine-checks/gate_b_summary.json`).

| Image requests | |
| --- | --- |
| Known-answer images, scored strictly | {{value:next.vision.known_answer.merged}} |
| Formats | {{value:next.vision.contract.formats}} |
| Largest image | {{value:next.vision.contract.largest}} |
| Images per request | {{value:next.vision.contract.per_request}} |
| Over the size limits (8192x8192) | {{value:next.vision.contract.over_cap}} |
| Malformed image | {{value:next.vision.contract.malformed}} |

**JPEG and WebP.** The server decodes PNG on its own. JPEG and WebP need `libturbojpeg.so.0` (libjpeg-turbo 3.1.4.1) and `libwebpdecoder.so.3` (libwebp 1.6.0), exactly those versions, and the released container does not carry them. `serve/build_release.sh` builds both from upstream source, `launch/stage_binary.sh` copies them beside the binary, and `launch/launch.sh` mounts them. Without them, PNG still decodes and JPEG and WebP get HTTP 400 naming the missing library.

The Formats row comes from one boot through the kit's launcher with both decoders staged: `bench/vision_check.py` sends a PNG, a BMP, the text card as a JPEG and the bar chart as a WebP, the last two judged by the same strict rules as their PNGs. The receipt records whether rank 0's serve log named a missing decoder.

**Request limits.** These are read from the release build's code; the probes in the table above ran on the Sparks. Per image: 10 MiB encoded, 8,192 pixels a side, 16,777,216 pixels decoded. Per request: 20 MiB encoded, at most 4 images and 8,192 image tokens, the engine's defaults (`SERVE_VISION_TOKENS` sets the second). Animated images are refused.

**Data URLs only.** With the kit's launch the server never fetches an image URL. An `http` URL is refused with a message that names HTTPS URLs. An `https` URL is refused too, since the launch does not pass the engine's `--vision-urls`. On the release build it got HTTP 400: "image URLs are off on this server; send the image as a data URL, or start the server with --vision-urls" (`receipts/next/release/https-url/`).

Each image is resized by the checkpoint's own rule: at most 2,048x2,048 pixels of area, the long side at most 200 times the short. The image tokens are split equally among a request's images, and an image over its share is shrunk, not refused. Beside a 1x1 image, a 2,048x2,048 one has a 4,096-token share, so it shrinks to 2,016x2,016. `"detail": "low"` caps an image at 256 tokens.

</details>

**Tool calls.** `tools` with `tool_choice` `"auto"`, streamed, each result sent back as a `tool` message. Five multi-turn agent tasks on the release build: {{value:next.agent.tool_calls}}. No real agent client was run.

<details>
<summary>Tool calls in detail: the five tasks and the requests the server refuses</summary>

On the release build, `bench/agent_loop.py` ran five multi-turn tasks, each a fresh conversation with three or four tools, `tool_choice` `"auto"`, streamed, greedy, one request at a time. Twice it followed a 24-file trail, and twice it fixed two bugs until the unit tests passed. Once, with the kit's README and FINDINGS in the system prompt, it wrote six values from the README to a JSON file.

The trail and the fix ran once with thinking off and once with no `reasoning_effort`, the server's default; whether those two runs reasoned was not recorded. The README task ran with thinking off.

| Request | What the server did |
| --- | --- |
| `tools`, `tool_choice` `"auto"`, the five tasks | {{value:next.agent.tool_calls}} |
| `tools`, `tool_choice` `"required"` | {{value:next.api.tool_required}} |
| `response_format` `json_schema` | {{value:next.api.json_schema}} |
| `response_format` `json_object` | {{value:next.api.json_object}} |

In that row, malformed means an unknown tool, arguments that are not a JSON object, or wrong argument names. In 3 turns the model made two calls at once, and the server took both results back. Solved is the script's own check of each task's result (`receipts/next/release/agent-loop/`).

The other three rows are one request each, with the server's defaults then (sampled at Tencent's own 0.9, thinking on) and a 1,024-token limit (`receipts/next/release/tool-probes/`). Whether the library below would change them is not tested: the structured-output refusal says the engine does not support it yet.

At boot, rank 0's serve log warns that "libtfgrammar.so is required for structured output and required tool calls (HTTP 400 when missing)". The kit's build does not make that library. The warning names the engine's own build of it, `zig build grammar` with xgrammar `v0.2.8`, which the kit has not run.

</details>

### Two builds

Both builds share the same routed experts. They differ only in the attention and shared-expert weights (the trunk), which is one setting at launch. Every speed number above is the full quant's.

| | **Full quant** (default) | **Lossless trunk** (option) |
| --- | --- | --- |
| Routed experts | EXL3, 3.00 bits per weight | EXL3, 3.00 bits per weight |
| Attention and shared experts | EXL3, `{{bare:build.full.trunk}}` | BF16, as Tencent ships them |
| Text model | {{value:quant.text.size.4b6b}}, {{bare:quant.text.bpw.4b6b}} bits per weight | {{value:quant.text.size.lossless}}, {{bare:quant.text.bpw.lossless}} bits per weight |

`4b6b` packs the trunk at 4 bits with `o_proj` and `kv_a` at 6 bits; the [quality card](card/QUALITY_CARD.md#why-4b6b-the-four-build-table) says why and compares the two builds. The lossless trunk is not booted at 128K, where the memory gate does not let it run.

<details>
<summary>How the sizes were read, and what the download holds</summary>

Read from the weight files' headers on a Spark that holds the full pack. Every weight in BF16 would be {{value:quant.text.size.bf16}} ({{bare:quant.text.bpw.bf16}} bits per weight).

The text model is every tensor in the weight index except the DFly2 drafter's. Its size counts tensor data, not file headers, and bits per weight is that size in bits over its weights, each packed matrix counted at its full shape. The BF16 figure is 2 bytes a weight, not the size of Tencent's files. GB is 10^9 bytes. Script and output: [`tools/text_size.py`](tools/text_size.py) and `receipts/quant/text_size.json`.

Served, the full quant reads {{value:quant.text.size.4b6b}} of text weights, plus its drafter and, with vision on, `vit-bf16/` (2.47 GB) on rank 0. The download is 371.84 GB because one folder holds both builds: the source trunk and both packed trunks ([Get the weights](#get-the-weights)). Cut into parts, each Spark holds 79.74 GB: its own 75.69 GB and 4.05 GB that all four keep ([Per-Spark weight files](#per-spark-weight-files)).

</details>

## What you need

- **Hardware**: four DGX Sparks (GB10, 121.7 GiB unified memory each, no swap), cabled directly in a 200 Gb/s ring with no switch.
- **Software**: `nvcr.io/nvidia/pytorch:26.07-py3`, whose NCCL 2.30.7 was built with CUDA 13.3 (the serve log prints `NCCL version 2.30.7+cuda13.3`); Zig 0.17.0; CUDA 13.0 at `/usr/local/cuda-13.0`, which `serve/build_release.sh` compiles with; and `tencent/Hy4` at `c10c0de5`.
- **For images**: the vision folder `vit-bf16/` (2.47 GB) on rank 0 only.

## Get the weights

The weights are at [jakejharris/jjh-hy4-exl3](https://huggingface.co/jakejharris/jjh-hy4-exl3): 371.84 GB for both builds (base folder 338.74 GB, `trunk-4b/` 13.25 GB, `trunk-6b/` 19.86 GB). Each Spark serving from the whole pack needs it at `MODEL_DIR`, the same path on all four. Download it on one Spark, check it, then copy it to the other three.

```bash
hf download jakejharris/jjh-hy4-exl3 --local-dir /data/models/hy4-exl3
cd /data/models/hy4-exl3 && sha256sum -c SHA256SUMS
```

Every line must end in `OK`. `launch/weights.sha256` in this repository is the same list. Per-Spark parts are cut from the pack with `tools/split_ranks.sh` (79.74 GB per Spark for the full quant); from parts, the full quant has run only at a 4,096-token window, not at 131,072 ([Per-Spark weight files](#per-spark-weight-files)).

The same repository holds the vision folder `vit-bf16/` (2.47 GB), which only rank 0 needs. Check it with `cd vit-bf16 && sha256sum -c SHA256SUMS`, the same list as `launch/vision.sha256`. `--exclude 'vit-bf16/*'` leaves it out of the download.

## Steps

Run these in order. Each step's guide has the details and the expected output.

1. **Quantize the routed experts** ([quant/README.md](quant/README.md); two phases at 3.00 bits per weight). The module pass took {{value:quant.time.per_layer_min}}; "component" means it times one step of the run, here the mean of three production layers on three GB10s without checkpoint writes. After our run, a re-hash of the 294 source files against the `tencent/Hy4` repository manifest ended `VERIFY PASS 294 files` ([log](receipts/source/verify.log)).
   ```bash
   python3 -m exllamav3.conversion.convert -i "$MODEL_DIR" -o "$OUT_DIR" -w "$WORK_DIR" -b 3 -hb 16 -mb 16 -cb mul1 --out_scales always -cd "$CAL_DATA" -cr 250 -cc 2048 -rcp /farm/recipe-l20-76.yaml
   ```
2. **Quantize the trunk** ([quant/README.md](quant/README.md), "Trunk pass"). Writes `trunk-4b/` and `trunk-6b/` beside the base pack without changing it, plus the drafter's 4-bit files. The full quant needs it, and so do drafts in either build. It can trail step 1 layer by layer on a fourth machine.
3. **Configure the ring.** Copy `launch/ring.env.example` to `launch/ring.env` and set your hosts, cable addresses and paths. Rank order must follow the physical cables, not hostnames ([launch/RING-RDMA.md](launch/RING-RDMA.md)). The stagers, preflight and launcher all read this file.
4. **Stage the weights** ([launch/STAGING.md](launch/STAGING.md)). Copy finished layers to all four Sparks over the ring cables as they land (with the drafter, since `STAGE_DRAFTER="1"`), then the trunk folders once the trunk pass ends:
   ```bash
   python3 -u launch/stage_layers.py --env-file launch/ring.env
   python3 launch/stage_trunk.py --env-file launch/ring.env
   ```
5. **Check the ring:**
   ```bash
   bash launch/preflight.sh launch/ring.env --probe --ranks 4
   ```
6. **Build, measure the row-cost table, and launch.** On one Spark, from a copy of this kit, build the engine from public TensorFold `2cee7d9` and `serve/patches/0001-hy4-port.patch`, pinned to cores 0 to 4 in the server's container ([serve/BUILD.md](serve/BUILD.md#pin-the-build-container-cores-and-a-clean-cache)). Then, from the machine that launches, copy it to the same folder on all four Sparks; the copy checks every file's SHA-256 on every Spark:
   ```bash
   bash serve/build_release.sh --tf-dir "$SPARK_HOME/hy4/tf" --out-dir "$SPARK_HOME/hy4/next-out" --zig-dir "$ZIG_DIR"
   bash launch/stage_binary.sh --env-file launch/ring.env --from "BUILD_SPARK:$SPARK_HOME/hy4/next-out/bin" --to "$SPARK_HOME/hy4/next-bin"
   ```
   The build also makes the JPEG and WebP decoders in `next-out/lib/`, from upstream tarballs it checks by SHA-256, in about 30 s. On a Spark with no route to the internet, add `--codec-src DIR` with the two tarballs in DIR; the build says which source it used ([serve/BUILD.md](serve/BUILD.md)).

   The build Spark needs this kit and a TensorFold checkout at `2cee7d9`. With no route to GitHub, clone TensorFold and check out `2cee7d9` on a machine that has one, then copy both folders over (`rsync -a` or `scp -r`). `serve/build_release.sh` checks that the checkout is `2cee7d9` with no local changes.

   `SPARK_HOME` is your home folder on the Sparks, as a full path. Set `TENSORFOLD_BIN` to `next-bin/tensorfold-native` and `TF_TP_KERNEL_IMAGE` to `next-bin/tp_mailbox.fatbin` in `launch/ring.env`, full paths. The launcher reads each Spark's binary hash first and stops if one cannot be read or the four differ. The default launch is the full quant at `4b6b` with drafts on, over the direct RoCE relay:
   ```bash
   bash tools/row_cost_table.sh --env-file launch/ring.env --env-file launch/ring.relay.env.example --oracle "$ORACLE_BIN"
   bash launch/launch.sh --env-file launch/ring.env --env-file launch/ring.relay.env.example
   ```
   Put the printed `TF_HY4_ROW_COST_TABLE` line in `launch/ring.env` ([launch/SETTINGS.md](launch/SETTINGS.md#tf_hy4_row_cost_table)). We measured ours with `tools/row_cost_table.sh` on 2026-11-21 (four Sparks, relay, `4b6b` trunk), and every boot of this release ran with it (`launch_settings` in the `summary.json` files under `receipts/next/release/`):
   ```bash
   TF_HY4_ROW_COST_TABLE="v1:56.74,v2:68.72,v3:78.15,v4:86.81,v5:95.48,v6:104.15,v7:112.81,d:6.44,e:0.41"
   ```
   The [quality card](card/QUALITY_CARD.md#row-cost-table) gives how the table is made, an earlier table and its measured effect.

   **Lossless trunk option.** Add one overlay; nothing else changes. Measure its own row-cost table first, with the same overlay passed to `tools/row_cost_table.sh`, since the verify cost depends on the trunk:
   ```bash
   bash launch/launch.sh --env-file launch/ring.env --env-file launch/ring.relay.env.example --env-file launch/ring.lossless.env.example
   ```
   **Vision option.** Take the vision folder `vit-bf16/` from the weights download, or build it once from Tencent's checkpoint and check it, as below; it goes on rank 0 only. Then copy `launch/ring.vision.env.example` to `launch/ring.vision.env`, set `TF_HY4_VISION_DIR` in it to that folder's full path on rank 0, and add it as one more overlay:
   ```bash
   python3 tools/vit_folder.py "$ORIG_MODEL_DIR" vit-bf16
   python3 tools/vit_folder.py --check vit-bf16
   bash launch/launch.sh --env-file launch/ring.env --env-file launch/ring.relay.env.example --env-file launch/ring.vision.env
   ```
   The launcher runs preflight with every overlay merged, so it refuses the launch if the folder on rank 0 is missing a file or a hash differs from `launch/vision.sha256`. Send images as OpenAI `image_url` parts with a data URL.

   **Receipts of the build.** For a measured run, set `GPU_CLOCK_CAP_MHZ` (only if you capped the clocks yourself) and `ENGINE_COMMIT` in `launch/ring.env`. The launcher writes them to `launch_meta.json` in the log folder with each rank's `tensorfold-native` SHA-256, the switches in force and a graphics-clock snapshot per rank at launch and at ready. `launch/collect_receipts.py` copies them into `summary.json` under `engine`.

   **Stopping it.** Press Ctrl-C in the launcher's terminal, or from a script send `kill -TERM` to the launcher's process. A background job in a script ignores SIGINT, so `kill -INT` does nothing there. Either way the launcher writes `launch_meta.json` and removes the four rank containers.

   If the launcher does not exit, a rank container outlived the stop (while rank 0's is up, `/health` still answers): find it with `docker ps --filter name=hy4-tp4-r` on each Spark, rank 0 being the first in `RING_HOSTS`, and remove it there with `docker rm -f` and its name.

   Load to ready on this release: 81 s cold, full quant, 131,072-token window, vision on (`receipts/next/release/window1/systems.load.cold.exl3.json`). Each `next.*` claim names the build it was measured on.
7. **Benchmark** ([bench/README.md](bench/README.md)). Decode at 4K and the 128K needle on the running build:
   ```bash
   python3 bench/bench.py --url http://127.0.0.1:8080 --prompt bench/prompts/short-prose.txt --label prose --runs 3 --max-tokens 128 --out results/prose.json
   python3 bench/niah.py run --url http://127.0.0.1:8080 --model hy4-exl3 --label niah-128k --prompt bench/prompts/niah-128k.txt --meta bench/prompts/niah-128k.meta.json --max-tokens 2048 --out results/niah-128k.json
   python3 bench/niah.py score --input results/niah-128k.json --meta bench/prompts/niah-128k.meta.json
   ```
   `--model` is the name `/v1/models` lists: the last part of `MODEL_DIR` (`hy4-exl3` for `/data/models/hy4-exl3`), unless `SERVE_MODEL_NAME` sets one.
   With vision on, check images on the running build. It scores three known-answer images strictly and probes the image contract; exit 0 means 3 of 3 and the server still answering:
   ```bash
   python3 bench/vision_check.py --url http://127.0.0.1:8080 --out "$LOGDIR/vision_check.json"
   python3 launch/collect_receipts.py --logdir "$LOGDIR" --out-dir receipts/run --label run
   ```
   The needle is one code at half depth, run with thinking on and again with thinking off (`--reasoning-effort no_think --max-tokens 256`). Time to first token on the 128,000-token prompt, full quant: {{value:next.niah128k.ttft.fp32}}. The lossless trunk is not booted at 128K. For prefill at 32K, use `bench/prompts/pp-32k.txt` with `--runs 3 --max-tokens 64` on a 32,768-token boot.
8. **Evaluate quality** ([eval/README.md](eval/README.md)). Top-1 agreement with the BF16 reference on the familiar public prefix set (the fresh set runs the same way from `eval/prefixes/fresh/`, with its own output names). Run it on the lossless trunk first, which sets the positions, then on the full quant at the same positions:
   ```bash
   python3 eval/agreement.py collect --url http://127.0.0.1:8080 --model hy4-exl3 --prefixes eval/prefixes/familiar/prefixes.safetensors --cuts 384,192,288,96 --max-seconds 1800 --label lossless --out results/lossless_picks.json
   python3 eval/agreement.py collect --url http://127.0.0.1:8080 --model hy4-exl3 --prefixes eval/prefixes/familiar/prefixes.safetensors --match-grid results/lossless_picks.json --max-seconds 1800 --label 4b6b --out results/4b6b_picks.json
   python3 eval/agreement.py score --teacher eval/prefixes/familiar/reference_picks.safetensors --cand results/4b6b_picks.json --lossless results/lossless_picks.json --json-out results/4b6b_score.json
   ```
   The paired counts behind the change against the lossless trunk are on the [quality card](card/QUALITY_CARD.md#agreement-with-the-bf16-reference).

   `collect` stops on a whole cut after `--max-seconds` (270 by default), so pass a cap that lets every cut finish: four cuts of the 84 prefixes give 336 positions, and the ledger fills an agreement cell only from 256 positions or more.

To fill the ledger from one boot's outputs, lay them out as `card/fill-map.json` describes and run `python3 tools/fill_claims.py --outputs DIR --dry-run`, then again without `--dry-run`. It checks each file's build, trunk width, context, transport and draft labels before writing a value, copies the receipt into `receipts/`, and refuses anything that does not match.

## Accuracy

Scored against Tencent's BF16 checkpoint. A KL panel compares the whole next-token distribution, not only the top pick: `KL(BF16 || row)` in nats over the full 120,832-token vocabulary, and top-1 agreement. Every row is scored on the same positions.

| Row | Text model | Bits per weight | Up to 2,048 tokens: 512 positions, 8 prompts | Past 2,048 tokens: 768 positions, 4 prompts |
| --- | --- | --- | --- | --- |
| Tencent's BF16, run again at another batch width | {{value:quant.text.size.bf16}} | {{bare:quant.text.bpw.bf16}} | {{spread:quality.kl.short.bf16}} | {{spread:quality.kl.long.bf16}} |
| Lossless trunk (option) | {{value:quant.text.size.lossless}} | {{bare:quant.text.bpw.lossless}} | {{spread:quality.kl.short.lossless}} | {{spread:quality.kl.long.lossless}} |
| Full quant (default) | {{value:quant.text.size.4b6b}} | {{bare:quant.text.bpw.4b6b}} | {{spread:quality.kl.short.exl3}} | {{spread:quality.kl.long.exl3}} |

Each cell gives top-1 agreement, then the median and mean KL. The first row is BF16 against itself at another batch width. It is the floor for this panel: a build cannot be expected to agree with BF16 more than BF16 agrees with itself.

The full quant is under the floor: by 13 of the 512 positions up to 2,048 tokens (2.54 points) and by 24 of the 768 past them (3.13 points).

Top-1 agreement on the public prefix sets, FP32 sums (the default): **{{value:next.quality.familiar.fp32}}** on the familiar set and **{{value:next.quality.fresh.fp32}}** on the fresh set, measured on speed step 2, one of the two builds the release build merges. Read the fresh set: it was first published in November 2026.

<details>
<summary>How accuracy was scored</summary>

**The KL panel.** A panel tool on our engine branch `hy4-kl-panel` (`b51568b41`, one commit off the release commit), which is not in this kit yet ([open items](FINDINGS.md#open-items)). Its picks were not cross-checked against the served build's picks (`served_pick_checks` is empty in the receipts). The logit files, about 990 MB each, are not in the kit either. Receipts: `receipts/quant/kl-panel/`.

What causes the floor was not measured. A few positions with large KL pull the mean up, so the median and the top-1 counts sit beside it. This panel's prompts are not those of the 720 public positions (680 agree, 94.44%), so {{value:quality.kl.short.exl3}} on this panel and 94.44% on those positions do not contradict each other.

**Agreement.** The served build's greedy next token against that of Tencent's BF16 checkpoint, on the public sets in `eval/prefixes/`. The familiar set is widely copied, so builds look alike there; the fresh set was first published in November 2026. There the reference predicts 65.44% of next tokens (91.04% familiar), with 2.7 times the close calls, so read the fresh rows.

**Why FP32 sums are the default.** On speed step 2, switching the sums from FP32 to BF16 lost four fresh positions of agreement with Tencent's BF16 reference. With FP32 sums: {{value:next.quality.familiar.fp32}} on the familiar set, {{value:next.quality.fresh.fp32}} on the fresh set. With BF16 sums: {{value:next.quality.familiar.bf16}} and {{value:next.quality.fresh.bf16}}. BF16 sums prefill faster ([Reading the numbers](#reading-the-numbers)).

**Fast path on against off.** On the release build (greedy, drafts and thinking off), 13 long prompts were sent both ways. Ten of 8,185 to 32,779 tokens: 9 gave the same tokens, one differed from its first token. Three that check facts in the prompt (2,120 to 20,000 tokens): two parted after 23 and 21 equal tokens, and all three kept every fact (`receipts/next/release/longcheck/`).

</details>

## Limits

Know these before you start:

- **One request at a time.** Concurrent streams queue.
- **No prefix reuse.** Every request is read from its first token, so each turn of a conversation waits for its whole prompt: 20,414 prompt tokens waited {{value:next.agent.long.turn1}} for the first streamed piece, and the next turn, 20,455 tokens, {{value:next.agent.long.turn2}}, with 0 cached tokens on both ([open items](FINDINGS.md#open-items)).
- **Tool calls: `tool_choice` `"auto"` only.** `"required"` and a named function get HTTP 400.
- **No structured output.** `response_format`, `json_schema` and `json_object` alike, gets HTTP 400: the engine does not support it yet.
- **Drafts can flip a close call.** Each drafted token is checked in a wider batch, where BF16 rounds differently, so drafted text can differ from plain text ([counts](card/QUALITY_CARD.md#drafted-against-plain)). A two-digit number in a picture was read as 14 instead of 74 with drafts on ([finding 7](FINDINGS.md#7-a-two-digit-number-in-a-picture-read-with-drafts-on-and-off)). For exact reads, send `"draft": false`.
- **Thinking on images.** With thinking on and no thinking budget, the kit default, two image requests spent their whole token limit thinking and gave no answer. Ask about images with thinking off, as `bench/vision_check.py` does, or set `SERVE_THINKING_BUDGET` ([settings](launch/SETTINGS.md#serve_thinking_budget)). A long coding reply can do the same ([finding 6](FINDINGS.md#6-with-thinking-on-a-long-coding-reply-can-end-before-the-answer-starts)).
- **A failed image request.** If an image request fails with a memory error, restart the serve: the other three Sparks can keep the earlier image and spoil the next text request ([open items](FINDINGS.md#open-items)).
- **Clocks.** Every speed number ran with each Spark's graphics clock capped at 2,200 MHz. At stock clocks, one of our Sparks lost power three times under four-Spark load ([open items](FINDINGS.md#open-items)).

## Launch settings

`launch/ring.env.example` carries the defaults. [launch/SETTINGS.md](launch/SETTINGS.md) has the detail and the measurements for each setting.

<details>
<summary>The settings that change what runs</summary>

| Setting | Default | What it does |
| --- | --- | --- |
| `TF_HY4_TRUNK` | `4b6b` | Packed trunk width (`4b`, `4b6b`, `6b`); empty serves the trunk lossless. Set by `ring.lossless.env.example`. |
| `SERVE_DRAFTS` | `1` | DFly2 drafts on: faster decode, but drafted text can differ from plain text. Empty or missing also means on; `0` turns them off ([more](launch/SETTINGS.md#serve_drafts)). |
| `SERVE_TEMPERATURE` | `0.7` | The temperature a request gets when it sends none. Empty passes no flag (Tencent's 0.9); `0` makes greedy the default ([more](launch/SETTINGS.md#serve_temperature)). |
| `SERVE_THINKING_BUDGET` | empty (no budget) | A default thinking budget, in tokens, for every request that reasons. Run once: greedy finished, sampled at 0.9 did not ([more](launch/SETTINGS.md#serve_thinking_budget)). |
| `SERVE_MODEL_NAME` | empty (the model folder's name) | The name `/v1/models` lists; run once on the Sparks ([more](launch/SETTINGS.md#serve_model_name)). |
| `PARTS_ROOT` | empty (the full pack) | Serve from per-Spark parts; set by `ring.parts.env.example` ([Per-Spark weight files](#per-spark-weight-files)). Run on our four Sparks at a 4,096-token window. |
| `TF_HY4_DRAFTER_EXL3`, `TF_HY4_DRAFTER_HEAD` | `4b` | The drafter's 4-bit files from the trunk pass. Use `6b` with a `6b` trunk. |
| `TF_HY4_TARGET_HEAD` | off | Packs the output head too. Changes output. |
| `TF_HY4_HEAD_BF16` | off | Reads the output head from its BF16 weights instead of an FP32 copy. Faster head, but the last bits of the logits change. |
| `ring.relay.env.example` | used in the launch example | Overlay: a direct RoCE relay in place of NCCL for the per-layer sums. Lossless trunk, no drafts: same tokens, 2.0% faster ([more](launch/SETTINGS.md#relay-overlay)). |
| `SWITCH_FABRIC`, `SWITCH_RAILS` | `0` (cabled ring), empty | `1` is for four Sparks on one switch: preflight checks two rails instead of the cable order, and refuses the relay. Set by `ring.switch.env.example`; run on one such cluster ([launch/SWITCH.md](launch/SWITCH.md)). |
| `TF_HY4_ROW_COST_TABLE` | empty (the engine's built-in table) | Verify cost per batch width; sets how many tokens the drafter proposes. Measure yours (step 6). Changes speed, and can flip close calls ([more](launch/SETTINGS.md#tf_hy4_row_cost_table)). |
| `TF_HY4_DRAFT_MIN_MARGIN` | off | Cuts a round's drafts where the drafter's top-two gap falls under this value, for example `0.85`. Ships off ([measured](card/QUALITY_CARD.md#draft-cutoff), [more](launch/SETTINGS.md#tf_hy4_draft_min_margin)). |
| `TF_HY4_REPEAT_STABLE` | `1` (on) | With drafts on, the same prompt gives the same tokens. `"0"` turns it off; any value but `0` or `1` is refused ([more](launch/SETTINGS.md#tf_hy4_repeat_stable)). |
| `TF_HY4_FAST_PREFILL` | `1` (on) | The faster prefill path for text requests. `"0"` turns it off: the slower path, with other rounding ([more](launch/SETTINGS.md#tf_hy4_fast_prefill)). |
| `TF_HY4_PREFILL_AR` | `fp32` | How prefill adds the four Sparks' partial results: `fp32`, or `bf16`, faster with lower agreement ([more](launch/SETTINGS.md#tf_hy4_prefill_ar)). |
| `TF_HY4_FAST_PREFILL_MIN_ROWS`, `TF_HY4_PREFILL_TIMERS` | empty (256 rows; timers off) | Shortest prompt, in tokens, that takes the fast path; per-stage prefill times in rank 0's log ([more](launch/SETTINGS.md#tf_hy4_fast_prefill)). |
| `SERVE_VISION`, `TF_HY4_VISION_DIR` | `0` (off), empty | `1` serves still images; rank 0 reads the vision folder the second names, checked by preflight. Set in your copy of `ring.vision.env.example` ([more](launch/SETTINGS.md#serve_vision)). |
| `GPU_CLOCK_CAP_MHZ`, `ENGINE_COMMIT` | empty | Receipts only: the clock cap you set and the engine commit you built, written to `launch_meta.json`. Changes nothing that runs. |

</details>

## Per-Spark weight files

Each Spark only needs the bytes its own rank reads. `bash tools/split_ranks.sh FULL_PACK OUT --check` cuts a full pack into `OUT/common/` (tensors every rank reads whole, such as the embedding table, norms, routers and scales) and `OUT/rank0/` to `OUT/rank3/` (that rank's slices of the experts, attention and output head), using the engine's own slicing code.

`--check` re-reads every file against the SHA-256 recorded at export and every tensor against the source bytes it came from. A Spark downloads `common/` plus its own `rankN/` and points `--model` at `common/`. The loader refuses another rank's part, a part cut from a different pack, and a mix of full-pack and part files.

For the full quant, pass the same width as `TF_HY4_TRUNK` (`bash tools/split_ranks.sh FULL_PACK OUT --trunk 4b6b --check`). `tools/split_ranks.sh` hands `--trunk` and `--target-head` to the engine's exporter, which records them in the parts, and loading refuses a different width. The exporter must be a build that accepts both flags (`tf-hy4-oracle --help` lists them).

**Launching from parts.** Each Spark needs `common/` and its own `rankN/` at the same `PARTS_ROOT`. The kit's 4-bit drafter is not in the parts: it needs a drafter folder of five files, 8.57 GB, which `bash tools/drafter_folder.sh FULL_PACK DIR` builds and checks. That makes 88.31 GB per Spark against 371.84 GB for the full pack. One overlay sets the paths ([launch/SETTINGS.md](launch/SETTINGS.md#parts_root)):

```bash
bash launch/launch.sh --env-file launch/ring.env --env-file launch/ring.relay.env.example --env-file launch/ring.parts.env.example
```

<details>
<summary>Parts as measured on our Sparks: the split, the hand-out, the drafter folder and a launch</summary>

On a 38-layer test pack of the lossless trunk build (layers 0 to 37, one DGX Spark), `common/` was 3.47 GB and `rank0/` 39.55 GB, 43.01 GB per Spark against 162.02 GB for the full pack.

On the full quant (78 layers, `--trunk 4b6b`, the release `tf-hy4-oracle`, one DGX Spark, 2026-11-21), `common/` was 4.05 GB and each `rankN/` 75.69 GB: 79.74 GB per Spark against 371.84 GB for the full pack of both builds. The export took 950 s. With `--check`, which compared 598,517 tensors in 1,181 files and found 0 failures, the whole job took 4,103 s ([log](receipts/parts/split-78l-4b6b.log)).

With both drafter settings set to `""` and no drafter folder, the server drafts with the BF16 drafter inside the parts instead. Every token is still the full model's pick, but the drafts are not the measured 4-bit drafter's, so that launch's speed is not known.

**Handing the parts out, as measured.** On 2026-11-21 the full quant's parts went from the Spark that cut them to the other three over the direct ring cables only: `common/` to all three, each `rankN/` only to its own Spark. The Spark two hops away got its part through its neighbour, which wrote nothing to disk.

Each file was hashed at the source and checked on arrival, then every destination re-read and re-hashed all its files. That took 2,928.09 s in all, of which 404.06 s went to hashing the source, 2,404.67 s to copying and 108.59 s to re-reading. Then, on every Spark, this check re-read each file against the SHA-256 recorded in its part ([how to copy and check](launch/STAGING.md#per-spark-parts)):

```bash
"$ORACLE_BIN" --check-rank-pack /data/models/hy4-parts
```

Each Spark printed `files=473 bytes=79706444762 tensors_compared=0 failures=0`, in 78.86 s with all four in parallel ([hand-out log](receipts/parts/handout-parts.log), [receipt](receipts/parts/handout-receipt.json)).

**The drafter folder, as measured.** Later that day, `tools/drafter_folder.sh` built it from the full pack on the Spark that cut the parts: 5 hard links, 8,566,396,282 bytes, in 5.5 s. Copied over one cable to two more Sparks, it passed its in-place check on all three, in 5.3 to 9.5 s, and on the fourth in 5.4 s the next day, after a power cut ([logs](receipts/parts/drafter-folder.log), [day two](receipts/parts/step0-node4.log), [copy times](launch/STAGING.md#per-spark-parts)).

**Launching from parts, as measured.** On 2026-11-22 the command above, with one more overlay for our paths and a 4,096-token window, served the full quant from parts on our four Sparks with the 4-bit drafter ([receipts](receipts/parts/launch-4k/)). Plain output equalled the full pack's on 416 of 416 tokens over four prompts, and drafted output on 384 of 384 over three. Three repeats of each of three prompts were exact.

It was ready in 48 s. Against the full-pack boot with the same settings, each Spark loaded in 27.5 to 43.1 s instead of 47.4 to 76.3 s, and the lowest `MemAvailable` was 37.23 GiB instead of 32.49 ([that boot](receipts/speed/exl3-4k-rs/boot-dfly.json)). No launch from parts has run at the 131,072-token window.

</details>

## Repository contents

<details>
<summary>What each folder holds</summary>

- `assets/`: the picture at the top of this page.
- `quant/`: quantization scripts, the frozen 3.00-bit expert recipe, the 4-bit and 6-bit trunk recipes, and the quantizer patch (`quant/patches/0001-hy4-quant.patch`).
- `serve/`: TensorFold build guide, patch generator and the port patch (`serve/patches/0001-hy4-port.patch`).
- `launch/`: ring stagers, preflight, RDMA probe, launcher, settings files, the settings guide ([launch/SETTINGS.md](launch/SETTINGS.md)) and the weight manifest (`launch/weights.sha256`, for `sha256sum -c` in the weights folder).
- `bench/`: benchmark client, needle test, prompts, the short-prompt first-token timer (`bench/ttft_sweep.py`), the agent-style tool loop (`bench/agent_loop.py`), and the image check (`bench/vision_check.py`, with its three images in `bench/images/`, two of them also as a JPEG and a WebP).
- `eval/`: the top-1 agreement scorer and its public prefix set; the full-vocabulary BF16 reference panel and router audit, registered for a later run.
- `card/`: model card, quality card, the claims ledger and the fill map. The pages that quote the ledger (this README, both cards, `FINDINGS.md`, `launch/RING-RDMA.md`, `launch/SETTINGS.md` and `quant/README.md`) are written from their templates in `card/templates/` by `python3 tools/render.py --all`, and `tools/fill_claims.py` rewrites them after each fill. A value not measured yet reads `(pending)`.
  - The ledger also registers 135 measurements this release did not run: 98 in the `later` tier and 37 `stretch` claims. They read `pending` in `card/claims.json` and appear on no page.
  - `next.speedbuild.*` holds the speed build's own measured rates, kept for comparison and shown on no page; the pages show the release build's values instead.
- `receipts/`: the receipt behind each measured value in the ledger (the two GGUF baseline sizes cite our run notes instead); `tools/fill_claims.py` copies in the receipts of each fill.
- `tools/`: kit checks, rendering, filling the ledger from boot outputs (`tools/fill_next.py` for this release's run), per-Spark splitting, the drafter folder for a launch from parts, and the vision folder (`tools/vit_folder.py`).
- [FINDINGS.md](FINDINGS.md): measured results that hold beyond this pack.

Rank order must match your physical ring cables, which `launch/preflight.sh` checks (`enp1s0f1*` to `enp1s0f0*` on separate `/24` subnets with one-hop routes); it refuses timed runs on `NET/Socket`. The pack carries DFly2 at source precision beside its 4-bit copy. It does not carry the vision tower, which is the separate `vit-bf16/` folder ([What you get](#what-you-get)).

</details>

## What was not run

<details>
<summary>Runs this release did not make, and parts not yet run on the Sparks</summary>

- The lossless trunk's 78-layer split has not been run; the full quant's has, and its check was clean ([Per-Spark weight files](#per-spark-weight-files)).
- From parts, the full quant has run only at a 4,096-token window, not at 131,072.
- The hand-out lines in `launch/STAGING.md` are a plain form of the copy we ran with our own tool. All three have run as written for the drafter folder alone; the parts went out with our own tool.
- The lossless trunk is not booted at 128K, where the memory gate does not let it run.
- The full-vocabulary BF16 reference panel and router audit are registered for a later run.
- The GGUF comparison (llama.cpp with Tencent's `AngelSlim/Hy4-GGUF`) and its matched panel in `card/claims.json` were not run.
- The full RigMark standard suite has not completed on this rig ([Development notes](#development-notes)).
- This release's numbers were not measured at stock GPU clocks, only with the clocks capped at 2,200 MHz.
- On the fixed build, only the 0.7 rows and the parity gate ran. Every other release-build value was measured before the sampling fix, on `ef8ebb7c`.
- Agreement was measured on the speed build and the vision tower's closeness to PyTorch on the vision build, not on the release build. The image check and the image-request limits ran on the release build.
- No independent review of the release build's vision code has run. A review of the vision build named four issues to fix first; the release build carries a change for each, not reviewed again.
- Vision: still images only, checked on three images and three screenshots; no image benchmark and no video.
- `serve/build_release.sh`, `launch/stage_binary.sh` and `bench/vision_check.py` are new in this release. Their host tests are `bench/test_vision_check.py`, `launch/test_receipt_reads.sh` and `tools/test_fill_next.py`, and their first run on the Sparks was the release run.
- `serve/build_codecs.sh` is new. On 2026-11-22 it built both decoders on a Spark, each source matching its pinned SHA-256, and the release build's serve found both at boot (`receipts/next/release/codecs/`).
- `tools/vit_folder.py` is new. On 2026-11-22 it built `vit-bf16/` from `tencent/Hy4` `c10c0de5` on a Spark, CPU only, with no network and the source read-only. All three files matched `launch/vision.sha256`, by its own `--check` and by a plain `sha256sum` (`receipts/next/release/vit-folder/`).
- `bench/ttft_sweep.py` is new. Its one run is the first-token table's, on the release build, and it has no host test.
- `bench/agent_loop.py` is new and has no host test. Its three runs give the `"auto"` tool-call row and the two turns without prefix reuse, on the release build. Not covered: a real agent client, and a recorded thinking-on run.

</details>

## Development notes

<details>
<summary>Our own unpublished development build, and what was measured against it</summary>

Before 1.0 we ran a development build of our own (engine `6e0fcf1ec`, kit `b5559f6`) and never published it. You do not need it for anything above. These facts compare against it or were measured on it. Its receipts stay in this repository (`receipts/next/prefill/base-2200/`), and `card/claims.json` names the receipt of every number.

**Prefill.** At the same 2,200 MHz cap it prefilled {{bare:next.prefill.8k.before}} / {{bare:next.prefill.32k.before}} / {{bare:next.prefill.64k.before}} tok/s at 8K / 32K / 64K. Its 128K needle waited {{value:long.niah128k.ttft.exl3}} for the first token, at stock clocks. `TF_HY4_FAST_PREFILL=0` gives its prefill path.

**Decode.** At the same cap it decoded {{bare:next.decode.code.last}} / {{bare:next.decode.prose.last}} / {{bare:next.decode.structured.last}} tok/s on code / prose / structured. Both decode runs used RigMark's `--comparison-id` `2026-11-22-hy4-speed-team-2200mhz`, so the same prompts. Outputs against the development build's run: {{value:next.decode.same_output}}.

The release does not change the greedy decode path, and these rows are greedy, so we make no faster or slower claim on decode. The binary's path alone moves decode by about 2%: the development build's own bytes, run from another folder, read {{bare:next.decode.code.before}} / {{bare:next.decode.prose.before}} / {{bare:next.decode.structured.before}} tok/s ([serve/BUILD.md](serve/BUILD.md#pin-the-build-container-cores-and-a-clean-cache)). A control run at the release binary's path length was dropped for time.

**Output.** The fast prefill path rounds and sums differently, so its output is not bit-identical to the development build's. It rounds to BF16 in places the older path does not ([which](launch/SETTINGS.md#tf_hy4_fast_prefill)). The same prompt and seed, greedy included, can give a different answer than the development build gave.

**Agreement.** At the positions behind the release's agreement rows, the development build agreed on {{value:quality.teacher.top1.exl3}} of the familiar set and {{value:quality.teacher.fresh.top1.exl3}} of the fresh set.

**Measured on it.** The quality card's two-build table, at stock clocks, and the row-cost table the release ran with (2026-11-21). Load to ready on it: {{value:systems.load.cold.exl3}} (full quant, 131,072-token window), {{value:systems.load.cold.lossless}} (lossless trunk, 32,768-token window).

**A RigMark run cut short.** The full standard suite ran on it on 2026-11-21 from 20:57 UTC, with the kit's launcher, drafts on, thinking off and stock clocks. At 22:25 UTC one Spark lost power ([open items](FINDINGS.md#open-items)), during the last 64K replay. RigMark wrote no receipt and no card, and the concurrency part never ran. So there is no RigMark result; the lines below are from the logs.

| Cut-short run, one stream, decode | Code | Prose | Structured |
| --- | --- | --- | --- |
| tok/s, median of 5 runs | 38.4 | 19.1 | 55.3 |

| Cut-short run, cold prefill | 8K | 32K | 64K |
| --- | --- | --- | --- |
| tok/s, median of 3 requests | 165.0 | 146.4 | 133.8 |

These lines come from the benchmark client's own log and the server's log. RigMark's completion gates were not recorded for this run, and neither was its revision. All 15 decode replies ended by themselves. The 8 replays that finished gave the same output as their cold requests, within 0.2 tok/s.

The 5 structured replies are the same 422 tokens as those of an earlier quick run, which passed its gates. The client prints a 64K pair only after its replay, so the third 64K cold rate is from the server's log: 65,536 tokens in 489.745 s. Logs: `receipts/rigmark-cut-short/`.

</details>

## License

Our scripts, documents and patches are under Apache-2.0 ([LICENSE](LICENSE)). The weights stay under the Tencent Hy License 1.0, Section 10 included, and third-party parts keep their own licenses ([THIRD_PARTY.md](THIRD_PARTY.md), [NOTICE](NOTICE)).
