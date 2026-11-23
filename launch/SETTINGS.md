# Launch settings

`launch/ring.env.example` carries the defaults; copy it to `launch/ring.env` and pass overlays after it. The [README](../README.md#launch-settings) gives each setting in one line. This page gives the detail, one section per setting. The measured records behind the numbers are on the [quality card](../card/QUALITY_CARD.md) and in `card/claims.json`.

## `TF_HY4_TRUNK`

Default `4b6b`. The packed trunk width: `4b` packs every trunk matrix at 4 bits, `4b6b` keeps `o_proj` and `kv_a` at 6 bits, `6b` packs everything at 6 bits. Empty serves the trunk lossless, which is what `ring.lossless.env.example` sets. Why `4b6b`: the [four-build table](../card/QUALITY_CARD.md#why-4b6b-the-four-build-table).

## `SERVE_DRAFTS`

Default `1`. DFly2 drafts on: faster decode. Every token is still the full model's own greedy pick, but checked in a wider batch, so at a close call the text can differ from plain decoding ([drafted against plain](../card/QUALITY_CARD.md#drafted-against-plain)). With the repeat option on (`TF_HY4_REPEAT_STABLE` below), a repeat of the same prompt gives the same tokens.

`0` turns drafts off for the whole server, as does `"draft": false` for one request; plain decoding is bit-exact on repeat. `launch/launch.sh` also starts with drafts on when the line is missing or empty, refuses any value but `0` or `1`, and stops before launch if drafts are on and no drafter is staged (`launch/stage_layers.py` stages it unless `STAGE_DRAFTER="0"`).

## `SERVE_TEMPERATURE`

Default `0.7`: `--temperature 0.7` on every rank, so a request that sends no `temperature` is sampled at 0.7. The flag sits over Tencent's `generation_config.json`, which is not changed and still says 0.9; `top_p` 1 and `top_k` -1 still come from that file. A request's own `temperature` wins, so `temperature` 0 is still greedy. The serving line prints the defaults in force.

The launcher passes 0.7 when the line is missing or commented out. Set it empty (`SERVE_TEMPERATURE=""`) to pass no flag, so Tencent's file decides: 0.9. `"0"` makes greedy the default. `launch_meta.json` records the value, and the launcher refuses anything but empty or a number of 0 or more. The kit's decode numbers send `temperature` 0, so this setting does not change them. Speed and answers at 0.7 on the fixed build: [README](../README.md#reading-the-numbers) and [finding 6](../FINDINGS.md#6-with-thinking-on-a-long-coding-reply-can-end-before-the-answer-starts).

## `SERVE_THINKING_BUDGET`

Default empty: no budget, the launch as measured. A whole number N passes `--thinking-budget N` to every rank. A request that reasons and sends no `thinking_budget` of its own then has its reasoning closed after N tokens, and the answer follows. A request's own non-zero `thinking_budget` wins; sending `0` does not turn the server's budget off. `max_tokens` still counts reasoning and answer together, so keep N well under the `max_tokens` your clients send. The engine's `--thinking-reserve` (room for reasoning beyond `max_tokens`) is a separate flag; the budget does not need it and the launcher does not pass it. The launcher refuses anything but empty or a whole number above 0.

Run once on the four Sparks (2026-11-21, the kit's launcher at `9cf80ed`, full quant, `SERVE_THINKING_BUDGET="4096"`): `--thinking-budget 4096` was on all four ranks. On one coding request with no `thinking_budget` field, greedy gave the same 6,082 tokens as the same request sending `"thinking_budget": 4096`, and the answer it asked for. The same request with thinking off was left alone. With Tencent's own default, 0.9, the reasoning closed at 4,097 tokens, but the reply went on reasoning in the visible answer and ran to the 8,192-token cap. What holds at other temperatures and on other tasks is not measured yet ([FINDINGS.md](../FINDINGS.md#6-with-thinking-on-a-long-coding-reply-can-end-before-the-answer-starts), finding 6).

## Sampling and thinking

These are request fields, not launcher settings. What the release engine does with each is read from its code; the measured lines come from one coding request on the full quant ([finding 6](../FINDINGS.md#6-with-thinking-on-a-long-coding-reply-can-end-before-the-answer-starts)).

- **No `temperature`**: sampled at 0.7, the launcher's `--temperature` (`SERVE_TEMPERATURE` above), with `top_p` 1 and `top_k` -1 from Tencent's `generation_config.json`, which the server loads at startup. Without that flag the file's own 0.9 applies; the measured lines below are at 0.9 (serving line then: `sampling: temperature 0.9, top_k -1, top_p 1`). A request's own `temperature`, `top_p`, `top_k` or `seed` replaces the default.
- **`temperature` 0**: greedy, each token the full model's top pick. The kit's decode numbers are greedy. With sampling and drafts on, a draft is kept only when the full model samples the same token. Sampled at 0.9, before this release's sampling fix, that request decoded at 13.27 and 12.96 tok/s against 32.72 greedy.
- **`reasoning_effort`**: `high` (thinking on, the default in Tencent's model card) or `no_think` (thinking off), the two values this kit measured. With thinking on, `max_tokens` counts the reasoning and the answer together.
- **`thinking_budget`**: closes the reasoning after that many tokens, then the answer follows. With `4096` the greedy request gave 4,097 reasoning and 1,985 answer tokens, the code it asked for, at 28.42 tok/s; without it the reasoning ran to the 8,192-token cap. Sampled at Tencent's own default, 0.9, with the same budget, the reasoning closed at 4,097 tokens, but the answer went on reasoning and ran to the cap. `SERVE_THINKING_BUDGET` above sets a default for every request.
- **Penalties**: `repetition_penalty` is accepted and ignored. `frequency_penalty` and `presence_penalty` are refused (HTTP 400) unless they are 0.
- **Loop guard**: the engine has `--loop-guard`, which closes the reasoning once it has repeated a passage exactly for 1,024 tokens (`--loop-guard-stop` ends the reply instead). The launcher does not pass it, and we did not run it.

## `SERVE_MODEL_NAME`

Default empty: no flag, and the server answers to the last part of the model folder's path (`/data/models/hy4-exl3` gives `hy4-exl3`). A name passes `--name NAME` to every rank, and `/v1/models` lists that name instead. The launcher accepts letters, digits, dot, dash and underscore, starting with a letter or digit, and refuses anything else. With per-Spark parts the model folder is `common`, so `launch/ring.parts.env.example` sets `hy4-exl3`, the full pack's name. On the four Sparks (2026-11-21, the kit's launcher at `af927a4`, full quant), `SERVE_MODEL_NAME="hy4-exl3-3.0bpw-4b6b"` made `/v1/models` list `hy4-exl3-3.0bpw-4b6b` ([its answer](../receipts/launch/served-name-v1-models.json)).

## `PARTS_ROOT`

Default empty: the full pack in `MODEL_DIR`, as measured. A folder serves from per-Spark parts instead (`launch/ring.parts.env.example`). Each Spark holds `PARTS_ROOT/common/` and its own `PARTS_ROOT/rankN/`, where N is its place in `RING_HOSTS` counting from 0, as `tools/split_ranks.sh` cuts them. `MODEL_DIR` must be `PARTS_ROOT/common`. The launcher then:

- mounts all of `PARTS_ROOT` in the container, so the engine finds `rankN/` beside `common/`;
- skips the trunk folder check, since the parts record their trunk width and the engine refuses another `TF_HY4_TRUNK`;
- refuses an empty `TF_HY4_TRUNK`, which would serve whatever width the parts were cut with;
- checks that each Spark holds `common/` and its own `rankN/`, and no other rank's folder.

The parts carry the BF16 drafter only. For the kit's 4-bit drafter, every Spark needs a drafter folder at the same path: `bash tools/drafter_folder.sh FULL_PACK DIR` builds it, 5 files, 8.57 GB, each checked against `launch/weights.sha256`, and [STAGING.md](STAGING.md#per-spark-parts) gives how to copy and check it. Set `DRAFTER_DIR` to it; the launcher passes `--drafter DIR` and checks the five files on every Spark. Packed drafter settings with no `DRAFTER_DIR` are refused, as the engine would refuse them.

With `TF_HY4_DRAFTER_EXL3` and `TF_HY4_DRAFTER_HEAD` set to `""` and no `DRAFTER_DIR`, the server drafts with the BF16 drafter inside the parts. Every token is still the full model's pick, but the drafts differ from the measured 4-bit drafter's, so speed can differ, and so can which close calls resolve which way. Not measured.

Run on our four Sparks with this launcher at a 4,096-token window, with the 4-bit drafter: the full pack's tokens on every prompt compared (2026-11-22, [README](../README.md#per-spark-weight-files), `receipts/parts/launch-4k/`). Not run at 131,072 tokens.

## `TF_HY4_DRAFTER_EXL3` and `TF_HY4_DRAFTER_HEAD`

Default `4b`: the drafter's 4-bit files from the trunk pass. Use `6b` with a `6b` trunk.

## `TF_HY4_TARGET_HEAD`

Default off. Packs the output head too. Changes output.

## `TF_HY4_HEAD_BF16`

Default off. Reads the output head from its BF16 weights instead of an FP32 copy. Faster head, but the last bits of the logits change.

## Relay overlay

`launch/ring.relay.env.example`, used in the launch example. An overlay for a direct RoCE relay in place of NCCL for the per-layer sums. On the lossless trunk without drafts it gave the same tokens as stock NCCL IB (0 of 384 differ over three prompts) and 9.37 against 9.19 tok/s, 2.0% faster (2026-11-21).

The relay reads its GPU kernel from the file `TF_TP_KERNEL_IMAGE` names, on every Spark: `tp_mailbox.fatbin` from the engine build ([serve/BUILD.md](../serve/BUILD.md) step 2), by default at `/opt/tensorfold/images/tp_mailbox.fatbin`. A path in the caller's environment wins over the settings files, then a path in `launch/ring.env`; `launch/launch.sh` and `tools/row_cost_table.sh` use the same one, and the launcher stops before launch if the file is missing on any Spark.

## `TF_HY4_ROW_COST_TABLE`

Default empty: the engine's built-in table. What one verify step costs at each batch width, which sets how many tokens the drafter proposes.

Measure it on your own four Sparks, with the settings files you serve with: `tools/row_cost_table.sh` (README step 6) prints the line to put in `launch/ring.env`. The lossless trunk needs its own table, measured with its overlay, since the verify cost depends on the trunk.

Changes speed. With drafts on it can also change which close calls resolve which way, as any change of verify width can. The release table, an earlier one and the measured effect are on the [quality card](../card/QUALITY_CARD.md#row-cost-table).

## `TF_HY4_DRAFT_MIN_MARGIN`

Default off. Truncates a round's drafted proposals where the drafter's own top-two logit gap falls under this value (for example `0.85`), so fewer rows go to the full model's verify. The shipped engine has it ([serve/BUILD.md](../serve/BUILD.md)).

It ships off: at `0.85` it helped code and cost prose with thinking on. Like any change of verify width, it also changes which close calls resolve which way. Both measured comparisons, at 0.85 and 0.5, are on the [quality card](../card/QUALITY_CARD.md#draft-cutoff).

## `TF_HY4_REPEAT_STABLE`

Default `1` (on). The repeat option. The kit default is on: `ring.env.example` sets `"1"`, and `launch/launch.sh` passes `1` when the line is empty or missing (the engine's own default is off).

On, the engine prices each draft depth from the row-cost table alone, with no wall clock, so a request drafts the same whatever came before it and a repeat of the same prompt gives the same tokens. Off, it prices each depth from live round times, so timing jitter can pick another verify width and a repeat can differ.

To turn it off, set `TF_HY4_REPEAT_STABLE="0"` in `launch/ring.env`; any other value than `0` or `1` is refused. `,untimed:1` in `TF_HY4_ROW_COST_TABLE` also turns it on, so the launcher refuses `0` with that entry.

With either setting, drafted text can differ from plain text. The measurements, per prompt, with the option on and off, are on the [quality card](../card/QUALITY_CARD.md#repeat-option).

## `TF_HY4_FAST_PREFILL`

Default `1` (on), new in this release. Prefill runs the prompt's attention and token index on tensor cores and adds the four Sparks' partial results in one ring all-reduce. `launch/launch.sh` is the one place that sets the kit default: it passes `1` when the line is empty or missing (the engine's own default is off). `"0"` gives the prefill path of [our development build](../README.md#development-notes); any other value is refused.

On the vision build, image requests did not take it. On the release build, image requests prefilled at 205.50 tok/s, the median of the three known-answer images (`next.vision.prefill.merged`). Its output is not bit-identical to our development build's, so the same prompt and seed can give a different answer than that build gave. Speed, agreement and the needle: [README](../README.md#speed), `card/claims.json` (`next.*`), all with graphics clocks capped at 2,200 MHz.

It also sets the prompt chunk at boot to 2,048 rows instead of 1,024, whatever a prompt's length. The startup log still prints 1,024.

Where it rounds to BF16: in four places our development build's prefill keeps wider. The trunk's codebook values (FP16 there). Each expert's output, before the weighted sum of the top eight. The attention context, before the KV projection. And only with `TF_HY4_PREFILL_AR=bf16`, each Spark's partial result before the sum across the four. On the speed build, out of 720 familiar and fresh positions, agreement with the BF16 reference was 680 with FP32 sums, 674 with BF16 sums, and 676 for our development build.

`TF_HY4_FAST_PREFILL_MIN_ROWS` (empty: the engine's 256) is the shortest prefill, in prompt tokens, that takes the fast path. The agreement rows were scored with `64`, so that the fast path covered their 384-token prefixes. `TF_HY4_PREFILL_TIMERS="1"` writes per-stage prefill times to rank 0's log; leave it off for timed runs.

## `TF_HY4_PREFILL_AR`

Default `fp32`, set in `launch/launch.sh`. How the fast prefill adds the four Sparks' partial results: `fp32` keeps the sums in FP32, `bf16` rounds each partial result to BF16 first, and `off` keeps our development build's gather. `bf16` prefills faster (the README's [Reading the numbers](../README.md#reading-the-numbers), from the release build) and agrees less with the BF16 reference: on the speed build, 347 of 384 fresh positions against 351 with `fp32` (`next.quality.*`). Any other value is refused.

## `SERVE_VISION`

Default `0` (off), new in this release. `"1"` passes `--vision` to every rank, and rank 0 reads the vision tower from `TF_HY4_VISION_DIR`: the full path of the `vit-bf16/` folder on rank 0. The other Sparks need nothing. `python3 tools/vit_folder.py SOURCE_DIR OUT_DIR` builds the folder from `tencent/Hy4` `c10c0de5`.

JPEG and WebP also need the two image decoders beside the binary: `serve/build_release.sh` builds them and `launch/stage_binary.sh` copies them. The launcher mounts them read-only when they are there; without them PNG still works and JPEG and WebP get HTTP 400. `launch/collect_receipts.py` records whether rank 0's serve log named a missing decoder (`image_decoders` in `summary.json`).

Before anything starts, `launch/preflight.sh` checks that folder on rank 0 against `launch/vision.sha256` and refuses a missing file or a wrong hash. `SERVE_VISION_TOKENS` (empty: the engine's default) sets the image-token cap a request may use. Measured: [README](../README.md#what-you-get), `next.vision.*`. The overlay `launch/ring.vision.env.example` sets both lines; the launcher hands preflight its overlays merged, so the check sees them.

## `GPU_CLOCK_CAP_MHZ` and `ENGINE_COMMIT`

Empty by default, new in this release, receipts only. The launcher sets no clocks and passes neither to the engine. It writes both to `launch_meta.json` in the log folder, with each rank's `tensorfold-native` SHA-256 (read on that Spark before launch), the prefill and vision switches in force, and one graphics-clock snapshot per rank at launch and one at ready. A snapshot is not a memory clock and not a trace over the run. `launch/collect_receipts.py` copies all of it into `summary.json` under `engine`, says "not declared" for a missing commit, and flags ranks whose hashes differ.

## `TF_HY4_TP_TIMEOUT_S`

The seconds each rank waits on the others at startup (`ring.env.example` sets `900`; the wait after the weight load has its own `TF_HY4_LOAD_TIMEOUT_S`). It also bounds the full-logit head gather.

It does not bound the gathers inside a prefill. If a Spark dies during a long prefill, the request hangs with no error and `/health` keeps answering 200. `launch/launch.sh` ends the serve only when its ssh session to that Spark gives up: 15 min 20 s on 2026-11-21. Set a client timeout above your longest prefill. ssh keepalive for your Spark hosts in `~/.ssh/config` (`ServerAliveInterval`, `ServerAliveCountMax`) is OpenSSH's documented way to notice a dead peer sooner; we did not measure it with this launcher ([FINDINGS.md](../FINDINGS.md#open-items)).
