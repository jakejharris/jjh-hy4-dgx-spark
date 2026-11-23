# Hy4 EXL3 quality and serving card

Version 1.0 of the recipe. The earlier build it compares with was not published ([README](../README.md)).

The EXL3 pack serves in two configurations. The **full quant** (default) packs the routed experts and the trunk (attention and shared experts); its output differs from the lossless trunk configuration. The **lossless trunk** keeps the same routed experts and leaves the trunk and the KV cache at source precision. Both run on TensorFold over four DGX Sparks.

The ledger (`card/claims.json`) records each figure's setup, command and receipt. Each table gives a figure's spread in the column beside it: read each timing with its min to max range. The quality number is top-1 agreement with Tencent's BF16 checkpoint on the same positions for both configurations.

Sections: [engine](#engine), [agreement](#agreement-with-the-bf16-reference), [decode at 4K](#decode-at-4k), [drafts](#drafts), [drafted against plain](#drafted-against-plain), [repeat option](#repeat-option), [draft cutoff](#draft-cutoff), [row-cost table](#row-cost-table), [why `4b6b`](#why-4b6b-the-four-build-table), [32K and 128K](#32k-and-128k), [systems](#systems-and-reproduction), [component checks](#component-checks).

## Engine

Every speed and quality number the ledger fills from the release window (2026-11-21) was measured on public TensorFold `2cee7d9` with `serve/patches/0001-hy4-port.patch` applied, stripped build ([serve/BUILD.md](../serve/BUILD.md)), with these binaries on all four Sparks:

| Binary | SHA-256 |
| --- | --- |
| `tf-hy4-oracle` | `3c6f9366fe7a98b3c4a39edffb8acfc4fddf182f260b4a0decfc5c49ea2aa14e` |
| `tensorfold-native` | `4a3d5265523408f302dfeb2066cdae1f4fe9f28276310c9d3c82df1c0aa35695` |
| `hy4-boundary` | `d115fca99c8acafb2f629908e3ff7b4d2584705a5cfc6aea95ba84d61ac0059a` |
| `tp_mailbox.fatbin` | `0c0e514571ae80946cb02667849752bdd47954d2fa1274761e5f030e1adf27c9` |

A number from an earlier engine names that engine where it appears.

This release's engine merges two builds measured separately: speed step 2 (`tensorfold-native` SHA-256 `40188e7f905724278693a05dfa25f257ab8ba88a1db4ccf6eaaab894dc13570c`) for agreement, and the vision build (`ffa371fa401257ab17ba6ebc8e96d04efd9ea29b02c9d1e9b79f5cfbbefb0082`) for vision. Their numbers are the `next.*` claims ([README](../README.md#what-is-new-against-the-earlier-build)), with graphics clocks capped at 2,200 MHz. The release's prefill and decode rates come from the merged build (`ef8ebb7c87aa81ff4afe4bb43bb38d05f5dd4e4e`, branch `hy4-next`), whose binaries are:

| Binary | SHA-256 |
| --- | --- |
| `tensorfold-native` | {{value:next.engine.sha256.native}} |
| `tf-hy4-oracle` | {{value:next.engine.sha256.oracle}} |
| `hy4-boundary` | {{value:next.engine.sha256.boundary}} |

**Where this release's prefill rounds to BF16.** The fast prefill rounds to BF16 in four places the earlier build's prefill keeps wider. The trunk's codebook values (FP16 there). Each expert's output, before the weighted sum of the top eight. The attention context, before the KV projection. And only with BF16 sums, each Spark's partial result before the sum across the four. Decode keeps the earlier build's precision.

Effect on the speed build, out of 720 familiar and fresh positions agreeing with the BF16 reference: 680 with FP32 sums (the default), 674 with BF16 sums, 676 for the earlier build (`next.quality.*` in `card/claims.json`). Speed step 2's code review found these four; settings: [launch/SETTINGS.md](../launch/SETTINGS.md#tf_hy4_fast_prefill).

## Agreement with the BF16 reference

| Metric | Full quant (default) | Spread | Lossless trunk | Spread |
| --- | --- | --- | --- | --- |
| Trunk | EXL3 `{{bare:build.full.trunk}}` | | BF16 | |
| Top-1 agreement, familiar set | {{value:quality.teacher.top1.exl3}} | {{spread:quality.teacher.top1.exl3}} | {{value:quality.teacher.top1.lossless}} | {{spread:quality.teacher.top1.lossless}} |
| Top-1, full quant minus lossless trunk, familiar set | {{value:quality.teacher.delta.exl3}} | {{spread:quality.teacher.delta.exl3}} | Not applicable | |
| Top-1 agreement, fresh set | {{value:quality.teacher.fresh.top1.exl3}} | {{spread:quality.teacher.fresh.top1.exl3}} | {{value:quality.teacher.fresh.top1.lossless}} | {{spread:quality.teacher.fresh.top1.lossless}} |
| Top-1, full quant minus lossless trunk, fresh set | {{value:quality.teacher.fresh.delta.exl3}} | {{spread:quality.teacher.fresh.delta.exl3}} | Not applicable | |

A split of a few positions in the paired counts (the positions only one configuration gets right), such as 3 against 8, does not prove that either configuration agrees with the reference better.

The same sets, with the reference itself scored against the published text:

| Set | Next-token accuracy | Prose | Code | Close calls |
| --- | ---: | ---: | ---: | ---: |
| familiar | 91.04% | 92.15% | 90.21% | 6.38% of positions |
| fresh | 65.44% | 54.94% | 75.94% | 17.23% of positions |

At the same positions for both configurations, the served greedy next token is compared with the reference's (Tencent's checkpoint, BF16 weights, FP32 head), on the public prefix sets in `eval/prefixes/`: 384-token excerpts of published prose and source code, each listed with its source and license in its set's manifest, so anyone can rerun the row.

Scored by `eval/agreement.py` (README step 8). The top-1 change is full quant minus lossless trunk on those positions, with the paired counts of positions only one configuration gets right (`eval/README.md`). It is computed from the counts, so it can differ by 0.01 from the difference of the two rounded percentages.

Beside each set, the reference's own accuracy on the published text and the share of close calls (positions where the reference's top two logits are less than 0.5 apart).

A set the model has memorized makes every build look alike; the `familiar` set (widely copied documentation and code) is near that end, and the `fresh` set (text first published in November 2026) is there to show the comparison on text the model cannot have seen ([eval/prefixes/README.md](../eval/prefixes/README.md)).

The fresh set reads as new text, near the internal set behind the four-build table (46.53% prose, 72.45% code), and has 2.7 times the close calls, where a small rounding change can flip a pick. So its rows are the ones that tell builds apart (`eval/prefixes/familiar/stats.json`, `eval/prefixes/fresh/stats.json`).

## Decode at 4K

| Metric | Full quant (default) | Spread | Lossless trunk | Spread |
| --- | --- | --- | --- | --- |
| Prose, thinking on, plain | {{value:speed.4k.prose.plain.exl3}} | {{spread:speed.4k.prose.plain.exl3}} | {{value:speed.4k.prose.plain.lossless}} | {{spread:speed.4k.prose.plain.lossless}} |
| Prose, thinking on, drafts | {{value:speed.4k.prose.dfly.exl3}} | {{spread:speed.4k.prose.dfly.exl3}} | {{value:speed.4k.prose.dfly.lossless}} | {{spread:speed.4k.prose.dfly.lossless}} |
| Prose, thinking off, plain | {{value:speed.4k.prose.off.plain.exl3}} | {{spread:speed.4k.prose.off.plain.exl3}} | {{value:speed.4k.prose.off.plain.lossless}} | {{spread:speed.4k.prose.off.plain.lossless}} |
| Code, thinking on, plain | {{value:speed.4k.code.plain.exl3}} | {{spread:speed.4k.code.plain.exl3}} | {{value:speed.4k.code.plain.lossless}} | {{spread:speed.4k.code.plain.lossless}} |
| Code, thinking on, drafts | {{value:speed.4k.code.dfly.exl3}} | {{spread:speed.4k.code.dfly.exl3}} | {{value:speed.4k.code.dfly.lossless}} | {{spread:speed.4k.code.dfly.lossless}} |
| Drafts kept per pass, prose | {{value:speed.4k.kept.prose.exl3}} | {{spread:speed.4k.kept.prose.exl3}} | {{value:speed.4k.kept.prose.lossless}} | {{spread:speed.4k.kept.prose.lossless}} |

Setup. Four Sparks on the cable ring, the direct RoCE relay for the per-layer sums (`ring.relay.env.example`), nothing else running, a 4,096-token context. One stream, greedy, `short-prose.txt` or `short-code.txt`, 128 generated tokens, the median of three runs, timed from the first generated token to the last: rate is tokens after the first, divided by that time.

Drafts are DFly2 from its 4-bit files, up to six proposals a pass, with the row-cost table measured on the same ring and the draft cutoff off unless the row says otherwise; kept counts the proposals the full model accepted.

Drafted rows use the kit default, the repeat option on (`TF_HY4_REPEAT_STABLE=1`): the engine prices each draft depth from the row-cost table alone, so a repeat of the same prompt gives the same tokens.

## Drafts

Hy4 ships a small drafting model (DFly2) that proposes up to six tokens a pass; the full model checks them all in one wider batch and keeps the ones it would have picked itself. Every emitted token is still the full model's own greedy pick, but computed at a different batch width, and BF16 rounds differently at another width.

Through 78 layers that moves the gap between the top two tokens: by 0.26 logit on the one pair measured (2026-11-21, lossless trunk, one row against seven). When the top two are that close, the pick can go the other way, and from there the text is a different, equally valid continuation.

So drafted text can differ from plain text, with the repeat option on or off; never read drafted output as plain output.

What holds is accuracy, not identity. A seven-row verify against one-row steps on real text (an earlier run on 2026-11-21, stock NCCL IB, lossless trunk and `4b6b`) gave the one-row pick at 28 of 28 positions where the reference's top-two gap is 1.1 or more, and gave the reference's pick on the one close call measured (gap 0.50).

The kit launches with the repeat option on ([launch/SETTINGS.md](../launch/SETTINGS.md#tf_hy4_repeat_stable)), and the drafted rows are for that mode unless they say otherwise. How many drafts the full model keeps on Tencent's checkpoint at full precision, and how often TensorFold's drafter proposes the reference's tokens, are under [component checks](#component-checks).

### Drafted against plain

Tokens that differ, drafts on against plain decoding, out of 128 generated, with the first position that differs, per prompt. Each count is against the plain boot's first run of the same prompt.

**Full quant, repeat option on.** Its three repeats gave the same tokens.

{{rows:quality.drafts.same.exl3}}

Setup: {{spread:quality.drafts.same.exl3}}.

**Full quant, repeat option off.** The first run; a later run of the same prompt can differ from it.

{{rows:quality.drafts.same.rsoff.exl3}}

Setup: {{spread:quality.drafts.same.rsoff.exl3}}.

**Lossless trunk, repeat option on.**

{{rows:quality.drafts.same.lossless}}

Setup: {{spread:quality.drafts.same.lossless}}.

**`"draft": false` on one request** gives plain output. Tokens that differ from the plain boot:

{{rows:quality.drafts.off.request.exl3}}

Setup: {{spread:quality.drafts.off.request.exl3}}.

`SERVE_DRAFTS=0` turns drafts off for the whole server. Plain decoding repeats exactly: on the release engine, three runs of each of the four bench prompts gave bit-equal tokens.

### Drafted path against the reference

The drafted path scores the full model's pick from a five-row verify (repeat option on) against the reference at 192 positions of an internal grid, with the plain path's score on the same positions, how often the two pick the same token, and the largest reference top-two gap where they differ.

{{rows:quality.drafts.top1.exl3}}

### Repeat option

With drafts on, the repeat option decides whether a repeat gives the same tokens ([launch/SETTINGS.md](../launch/SETTINGS.md#tf_hy4_repeat_stable)). On, the engine prices each draft depth from the row-cost table alone. Off, it prices each depth from live round times; two depths can cost within 0.5% of each other, so timing jitter picks the verify width, and another width can round a close call the other way, so a repeat can differ.

Prompts whose three repeats gave the same tokens, repeat option on: {{value:speed.4k.repeatstable.same.exl3}}.

{{rows:speed.4k.repeatstable.same.exl3}}

Repeat option off: {{value:speed.4k.repeatstable.same.rsoff.exl3}}.

{{rows:speed.4k.repeatstable.same.rsoff.exl3}}

Drafted decode, repeat option off against on:

{{rows:speed.4k.repeatstable.speed.exl3}}

Setup: {{spread:speed.4k.repeatstable.speed.exl3}}.

With the option on, the three short prompts also gave the same tokens and the same number of drafted proposals over five runs, two of them sent after other prompts (release engine, 2026-11-21, `4b6b` full quant, relay, 4K, row-cost table measured the same day).

### Draft cutoff

`TF_HY4_DRAFT_MIN_MARGIN` truncates a round's drafted proposals where the drafter's own top-two logit gap falls under this value, so fewer rows go to the full model's verify ([launch/SETTINGS.md](../launch/SETTINGS.md#tf_hy4_draft_min_margin)). It ships off. Like any change of verify width, it also changes which close calls resolve which way, so the tokens differ from the cutoff-off run as listed.

**At 0.85**, on the `4b6b` full quant with drafts, over the relay with the measured row-cost table (2026-11-21, an earlier internal engine build, `d13f7b4f4`, which has no repeat option):

{{rows:speed.4k.cutoff.exl3}}

Setup: {{spread:speed.4k.cutoff.exl3}}.

At 0.85 it helped code and cost prose with thinking on; the change on prose with thinking off is inside the run-to-run spread.

**At 0.5**, on the release engine with the repeat option on:

{{rows:speed.4k.cutoff05.exl3}}

Setup: {{spread:speed.4k.cutoff05.exl3}}.

## Row-cost table

The row-cost table tells the drafter what one verify step costs at each batch width, which sets how many tokens it proposes ([launch/SETTINGS.md](../launch/SETTINGS.md#tf_hy4_row_cost_table)). `tools/row_cost_table.sh` runs `tf-hy4-oracle --bench-window` on all four ranks at widths 1, 2, 3 and 7 and turns rank 0's log into the table (`tools/make_row_cost_table.py`): widths 4 to 6 are interpolated, and `d` and `e` are the parser's relay defaults because the log carries neither.

**Release table.** The release run ran `tools/row_cost_table.sh` from the kit on the same four Sparks, on the relay with the `4b6b` trunk (2026-11-21), and it passed. The full quant's 4K, 32K and 128K rows use this table; the lossless trunk needs its own, measured with its overlay (README step 6):

```bash
TF_HY4_ROW_COST_TABLE="v1:56.74,v2:68.72,v3:78.15,v4:86.81,v5:95.48,v6:104.15,v7:112.81,d:6.44,e:0.41"
```

**Earlier table.** Before that run, the same oracle command and parser, not the script, made a table for four Sparks on the relay with the `4b6b` trunk (2026-11-21 08:52 UTC):

```bash
TF_HY4_ROW_COST_TABLE="v1:56.14,v2:69.85,v3:78.43,v4:86.84,v5:95.25,v6:103.66,v7:112.07,d:6.44,e:0.41"
```

Its effect, as measured on an earlier internal engine build (`b1ee17414`, repeat option absent): the picked build's 4K bench ran with this table and the relay together, and drafted prose went from 29.93 tok/s (stock NCCL IB, the engine's default table, median of two runs) to 32.66 (median of three). The relay and the table were not measured apart on that bench, so the gain belongs to the pair.

Measured alone on the lossless trunk without drafts, the relay was 2.0% faster than stock NCCL IB (9.37 against 9.19 tok/s) with the same tokens.

## Why `4b6b`: the four-build table

`4b` packs every trunk matrix at 4 bits, `4b6b` keeps `o_proj` and `kv_a` at 6 bits, `6b` packs everything at 6 bits. The full quant serves `4b6b`, with the drafter from its 4-bit files: of the widths whose top-1 agreement stays within one point of the lossless build on the same positions, it is the fastest with drafts on.

On 192 positions the gap between `4b` and `4b6b` is not proven.

In both builds the shared-expert `down_proj` at layers 75 and 76 stays BF16, because its input overflows FP16 there ([FINDINGS.md](../FINDINGS.md), finding 5). The output head is FP32 in both.

Setup. Measured 2026-11-21 08:08 to 08:36 UTC on four Sparks, 78 layers, 4,096-token context, stock NCCL IB with 4 channels (not the relay), default row-cost table, median of two runs, prose with thinking on. Drafted rows use DFly2 from its 4-bit files (6-bit for `6b`).

Agreement is on 192 positions of an internal 64-prefix set (32 prose, 32 code, cuts at 384, 192 and 288 tokens) that cannot ship, so this row cannot be rerun from the kit. On that set the reference predicts the next token of the text at 46.53% (prose) and 72.45% (code) of positions, and 38 of the 192 scored positions (19.8%) are close calls.

| Trunk | Top-1 agreement | Positions only lossless gets right / only this build | Plain decode | Drafted decode | Lowest free memory |
| --- | ---: | --- | ---: | ---: | ---: |
| lossless | 93.75% | baseline | 9.18 tok/s | 21.95 tok/s | 23.93 GiB |
| `4b` | 92.19% | 6 / 3 (-1.56 points) | 18.30 tok/s | 28.81 tok/s | 33.22 GiB |
| `4b6b` | 95.83% | 1 / 5 (+2.08 points) | 16.97 tok/s | 29.93 tok/s | 32.80 GiB |
| `6b` | 94.79% | 2 / 4 (+1.04 points) | 15.36 tok/s | 25.97 tok/s | 32.05 GiB |

`4b` is the only width outside one point. A discordant count of 6 against 3, or 1 against 5, does not prove a difference on 192 positions; a larger score may revisit `4b` against `4b6b`. The picked build's own 4K numbers, on the relay with a measured row-cost table, are the ones under [decode at 4K](#decode-at-4k).

## 32K and 128K

| Metric | Full quant (default) | Spread | Lossless trunk | Spread |
| --- | --- | --- | --- | --- |
| 32K prompt, prefill rate | {{value:long.pp32k.prefill.exl3}} | {{spread:long.pp32k.prefill.exl3}} | {{value:long.pp32k.prefill.lossless}} | {{spread:long.pp32k.prefill.lossless}} |
| 32K prompt, time to first token | {{value:long.pp32k.ttft.exl3}} | {{spread:long.pp32k.ttft.exl3}} | {{value:long.pp32k.ttft.lossless}} | {{spread:long.pp32k.ttft.lossless}} |
| 32K prompt, decode after prefill | {{value:long.pp32k.decode.exl3}} | {{spread:long.pp32k.decode.exl3}} | {{value:long.pp32k.decode.lossless}} | {{spread:long.pp32k.decode.lossless}} |
| 128K prompt, prefill rate | {{value:long.niah128k.prefill.exl3}} | {{spread:long.niah128k.prefill.exl3}} | Not measured: the lossless trunk is not booted at 128K | |
| 128K prompt, time to first token | {{value:long.niah128k.ttft.exl3}} | {{spread:long.niah128k.ttft.exl3}} | Not measured | |
| 128K prompt, decode after prefill | {{value:long.niah128k.decode.exl3}} | {{spread:long.niah128k.decode.exl3}} | Not measured | |
| 128K needle, thinking on | {{value:long.niah128k.on.exl3}} | {{spread:long.niah128k.on.exl3}} | Not measured | |
| 128K needle, thinking off | {{value:long.niah128k.off.exl3}} | {{spread:long.niah128k.off.exl3}} | Not measured | |
| Weights per Spark, largest rank | {{value:systems.weights.max.exl3}} | {{spread:systems.weights.max.exl3}} | {{value:systems.weights.max.lossless}} | {{spread:systems.weights.max.lossless}} |
| Lowest MemAvailable, any Spark, 128K | {{value:systems.memory.min128k.exl3}} | {{spread:systems.memory.min128k.exl3}} | Not measured: the lossless trunk is not booted at 128K | |

Setup. Same Sparks and transport, the default settings for both configurations (drafter loaded, drafts on, repeat option on), cold prompt state, one request at a time. The 32K prompt is `pp-32k.txt` (32,010 tokens) in a 32,768-token context, thinking on, 64 generated tokens, the median of three runs.

The 128K prompt is `niah-128k.txt` (128,000 rendered tokens with thinking on, 128,003 with thinking off) in a 131,072-token context, run once each way; one code sits at half depth and passes if the answer, with any reasoning removed, is exactly that code.

The thinking-off request ran on a second boot of the same build and settings that day, while a passive telemetry capture read all four Sparks. On the first boot, one Spark lost power during that request ([FINDINGS.md](../FINDINGS.md#open-items)). Every other 128K row comes from the first boot.

Weights are what each rank reports at load, without KV cache or scratch. Free memory is the lowest `MemAvailable` any Spark reached from launch through a 128,000-token request, with a 131,072-token context; it is sampled on every Spark from launch through the 128K requests of the first boot, and the receipt records whether the drafter was loaded.

## Systems and reproduction

| Metric | Full quant (default) | Spread | Lossless trunk | Spread |
| --- | --- | --- | --- | --- |
| Rank 0 resident weight payload | {{value:systems.weights.r0.exl3}} | {{spread:systems.weights.r0.exl3}} | {{value:systems.weights.r0.lossless}} | {{spread:systems.weights.r0.lossless}} |
| Rank 0 minimum MemAvailable | {{value:systems.memory.r0.exl3}} | {{spread:systems.memory.r0.exl3}} | Not measured (no 128K boot) | |
| Rank 1 resident weight payload | {{value:systems.weights.r1.exl3}} | {{spread:systems.weights.r1.exl3}} | {{value:systems.weights.r1.lossless}} | {{spread:systems.weights.r1.lossless}} |
| Rank 1 minimum MemAvailable | {{value:systems.memory.r1.exl3}} | {{spread:systems.memory.r1.exl3}} | Not measured (no 128K boot) | |
| Rank 2 resident weight payload | {{value:systems.weights.r2.exl3}} | {{spread:systems.weights.r2.exl3}} | {{value:systems.weights.r2.lossless}} | {{spread:systems.weights.r2.lossless}} |
| Rank 2 minimum MemAvailable | {{value:systems.memory.r2.exl3}} | {{spread:systems.memory.r2.exl3}} | Not measured (no 128K boot) | |
| Rank 3 resident weight payload | {{value:systems.weights.r3.exl3}} | {{spread:systems.weights.r3.exl3}} | {{value:systems.weights.r3.lossless}} | {{spread:systems.weights.r3.lossless}} |
| Rank 3 minimum MemAvailable | {{value:systems.memory.r3.exl3}} | {{spread:systems.memory.r3.exl3}} | Not measured (no 128K boot) | |
| KV pool token capacity across admitted slots | {{value:systems.kv.pool.exl3}} | {{spread:systems.kv.pool.exl3}} | {{value:systems.kv.pool.lossless}} | {{spread:systems.kv.pool.lossless}} |
| Load to ready | {{value:systems.load.cold.exl3}} | {{spread:systems.load.cold.exl3}} | {{value:systems.load.cold.lossless}} | {{spread:systems.load.cold.lossless}} |
| Shipped one-command startup check | {{value:systems.launch.exl3}} | {{spread:systems.launch.exl3}} | {{value:systems.launch.lossless}} | {{spread:systems.launch.lossless}} |

The full quant column comes from one 131,072-token boot and the lossless column from one 32,768-token boot (the lossless trunk is not booted at 128K, where the memory gate does not let it run), each with the default settings (drafter loaded), written by `launch/collect_receipts.py`. Rank labels follow the physical cable ring. Weight payload excludes KV and scratch. The memory guard samples available RAM from launch through the 128K requests.

Load to ready runs from the launcher start to the server answering on all four ranks, counted in whole seconds; the file cache is not cleared first. The startup check passes when all four ranks report NCCL IB and the server is ready; its receipt keeps the known-answer reply.

An apparently low memory or startup number can come from dropping a rank, warming the caches, or finishing before all workers are ready. The rank logs, measured memory minima and launch receipt make those cases visible. Follow the one-command launcher in [jakejharris/jjh-hy4-dgx-spark](https://github.com/jakejharris/jjh-hy4-dgx-spark).

Four machines and one machine do not give bit-equal output on real weights, even with the lossless trunk ([FINDINGS.md](../FINDINGS.md), finding 3). Splitting the matrix products changes the order of floating-point arithmetic. Four ranks sum their partial results in FP32 before rounding to BF16. Small changes can move an expert, attention key or next token across a selection boundary.

The tests checked exact weight reassembly, repeatability across fresh four-rank runs, first-token agreement, greedy prefixes and logit drift on real-weight slices. On the 16-layer slice at 32K, the first token matched, and the one-rank and four-rank runs gave the same greedy tokens until the sequences diverged. Tokens matched before that: {{value:component.parity.tp4.16l.32k.prefix}}. Full-model fidelity is the top-1 agreement with the BF16 reference in the [agreement](#agreement-with-the-bf16-reference) table.

## Component checks

| Measurement | Result | Spread |
| --- | --- | --- |
| Synthetic rank compute at 32K | {{value:component.kernels.r3.decode}} | {{spread:component.kernels.r3.decode}} |
| Synthetic four-local-rank vs one-rank maximum logit drift | {{value:component.kernels.r3.logits}} | {{spread:component.kernels.r3.logits}} |
| Rank compute at 32K, paired replay | {{value:component.kernels.r4.decode}} | {{spread:component.kernels.r4.decode}} |
| Rank chunk compute at 32K, paired replay | {{value:component.kernels.r4.prefill}} | {{spread:component.kernels.r4.prefill}} |
| 156 decode gather-and-fold calls, SOCKET | {{value:component.fabric.socket.burst}} | {{spread:component.fabric.socket.burst}} |
| One 2,048-row gather-and-fold call, SOCKET | {{value:component.fabric.socket.chunk}} | {{spread:component.fabric.socket.chunk}} |
| 156 decode gather-and-fold calls, IB | {{value:component.fabric.ib.burst}} | {{spread:component.fabric.ib.burst}} |
| One 2,048-row gather-and-fold call, IB | {{value:component.fabric.ib.chunk}} | {{spread:component.fabric.ib.chunk}} |
| Live decode on four Sparks, 16 real layers, shared with quantization | {{value:component.serve.16l.decode}} | {{spread:component.serve.16l.decode}} |
| Four ranks versus one at 8K, four real layers, active prefill | {{value:component.prefill.tp4.8k}} | {{spread:component.prefill.tp4.8k}} |
| Four ranks versus one at 32K, four real layers, active prefill | {{value:component.prefill.tp4.32k}} | {{spread:component.prefill.tp4.32k}} |
| Two-rank stock NCCL IB decode burst | {{value:component.fabric.ib.burst.ranks2}} | {{spread:component.fabric.ib.burst.ranks2}} |
| Two-rank stock NCCL IB single gather | {{value:component.fabric.ib.single.ranks2}} | {{spread:component.fabric.ib.single.ranks2}} |
| BF16 KV and index pool layout | {{value:component.kv.bf16.bytes}} | {{spread:component.kv.bf16.bytes}} |
| 128K fixture cached replay greedy matches | {{value:component.context.fixture.greedy}} | {{spread:component.context.fixture.greedy}} |
| Farm layer 1 packed shard | {{value:component.quant.l1.size}} | {{spread:component.quant.l1.size}} |
| Farm layer 1 routed weight bits | {{value:component.quant.l1.bpw}} | {{spread:component.quant.l1.bpw}} |
| Farm mean MoE module time in phase 1 | {{value:quant.time.per_layer_min}} | {{spread:quant.time.per_layer_min}} |
| Farm wall time per layer including checkpoint | {{value:component.quant.checkpoint.time}} | {{spread:component.quant.checkpoint.time}} |
| Farm layer 1 cosine loss on prompt16 | {{value:component.quant.l1.cos}} | {{spread:component.quant.l1.cos}} |
| DFly2 drafts kept per pass on the full-precision reference, prose | {{value:component.dfly.keep.prose}} | {{spread:component.dfly.keep.prose}} |
| DFly2 drafts kept per pass on the full-precision reference, code | {{value:component.dfly.keep.code}} | {{spread:component.dfly.keep.code}} |
| TensorFold drafter proposals equal to the reference's, 78-layer states | {{value:component.dfly.parity}} | {{spread:component.dfly.parity}} |

Setup. These tests cover components or partial models. The drafter rows ran Tencent's checkpoint at full precision on 64 held-out 384-token prefixes (32 prose, 32 code), greedy, six proposals a pass. The four-layer prefill timings cover active compute with the output head skipped and jobs sharing the machines. The two-rank transport probe used one direct cable. The ledger gives each setup; the cited run notes are not shipped.
