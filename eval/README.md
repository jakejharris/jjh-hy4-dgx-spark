# Quality Evaluation (`eval/`)

The release's quality number is top-1 agreement with the BF16 reference, measured on the running server by `eval/agreement.py` over the public prefix sets in `eval/prefixes/` (`familiar/` and `fresh/`; run the fresh set by passing its folder in place of `familiar/`, with its own output names; pass `--cuts 384,192,288,96 --max-seconds 1800` to `collect` so every cut finishes, since the default stops after 270 s). Each set folder carries `stats.json`: the reference's own next-token accuracy on the published text and the share of close calls (top-two gap under 0.5), which the cards print beside every agreement number, because a set the model has memorized makes every build look alike. The rest of this folder is the full-vocabulary panel: a 78-layer BF16/FP32 reference streamed one layer at a time over HTTP (`eval/teacher.py`), router replacement audits (`eval/router_audit.py`) and thinking-mode probes (`eval/thinking-probes.json`). That panel is registered for a later run and is not part of this release's figures.

## Top-1 agreement (`agreement.py`)

**What it sends.** For each prefix and cut length `c`, one `POST /v1/completions` to the running server with the first `c` token IDs of the prefix as a raw token list (no chat template, nothing added), `max_tokens: 1`, `temperature: 0`, `ignore_eos: true` and `return_token_ids: true`. It keeps the one token the server returns (`tensorfold.token_ids[0]`), the server's greedy next token after exactly those `c` tokens.

**The grid.** Each prefix is 384 tokens long. Cuts run in the order 384, 192, 288, 96, 336, 144, 240, 48, every prefix at one cut before the next cut starts. With `--max-seconds` (default 270) the collection stops at the end of a cut once another cut would pass the limit, so the number of positions is the number of prefixes times the number of completed cuts. The first build collected (the lossless trunk) sets the grid; every other build passes `--match-grid` with the lossless picks file and answers exactly the same (prefix, cut) positions, with no time limit.

**The reference.** The reference picks file holds, for every prefix and every length 1 to 384, the greedy next token of Tencent's checkpoint at full precision (BF16 weights, FP32 head), and the margin between its top two logits. A position counts as agreeing when the served pick equals the reference pick.

**Reading the score.** `score` prints and writes `overall_pct` (agreeing positions over positions scored), `prose_pct` and `code_pct` (split by the prefix's `is_code` flag) and, when margins are present, the same split by reference margin below 1 and at least 1 (low-margin positions are where any rounding change can flip the pick). With `--lossless`, it scores only positions present in all three files and adds the paired counts:

| Field | Meaning |
| --- | --- |
| `both_match` | both builds pick the reference token |
| `lossless_only` | only the lossless trunk picks it |
| `cand_only` | only the candidate picks it |
| `neither` | neither does |
| `delta_pts_vs_lossless` | candidate minus lossless agreement, in points: `100 * (cand_only - lossless_only) / positions` |
| `within_1pt` | `delta_pts_vs_lossless >= -1.0` |

Only the two discordant counts move the difference; positions both builds get right or both get wrong cancel. Read `lossless_only` against `cand_only` as a sign test on the discordant positions: the self-test's 10 against 6 gives a delta of -0.78 points, and a two-sided binomial probability of about 0.45 that a split at least that uneven arises by chance, so it does not show the candidate is worse. The width rule uses `within_1pt`. The ledger takes `overall_pct` (spread: `prose_pct`, `code_pct`) for `quality.teacher.top1.*` and `delta_pts_vs_lossless` (spread: `lossless_only`, `cand_only`) for `quality.teacher.delta.exl3`; `tools/fill_claims.py` reads the score files directly.

```bash
python3 eval/agreement.py selftest
# lossless trunk running:
python3 eval/agreement.py collect --url http://127.0.0.1:8080 --model MODEL \
  --prefixes eval/prefixes/familiar/prefixes.safetensors --label lossless --out OUT/lossless_picks.json
python3 eval/agreement.py score --teacher eval/prefixes/familiar/reference_picks.safetensors \
  --cand OUT/lossless_picks.json --json-out OUT/lossless_score.json
# full quant running (label = its trunk width):
python3 eval/agreement.py collect --url http://127.0.0.1:8080 --model MODEL \
  --prefixes eval/prefixes/familiar/prefixes.safetensors --match-grid OUT/lossless_picks.json \
  --label 4b6b --out OUT/4b6b_picks.json
python3 eval/agreement.py score --teacher eval/prefixes/familiar/reference_picks.safetensors \
  --cand OUT/4b6b_picks.json --lossless OUT/lossless_picks.json --json-out OUT/4b6b_score.json
```

The prefix set and reference picks are described in [prefixes/README.md](prefixes/README.md). `agreement.py` needs only the Python standard library.

## Full-vocabulary panel (later)

## What each step proves

1. **Control (`teacher.py control`)**:
   - Repeats real `L0 + head` (`prompt16`) and real `L1`/`L2` against saved CUDA oracle dumps, checking exact discrete IDs, L2 shared indices, and float tolerances.
   - Runs a negative control (cycling one-based score positions left by one) and requires the comparator to return **MISS** before trusting any teacher logits.
2. **Teacher & reduction-order null (`teacher.py teacher`)**:
   - Streams all 78 blocks of `tencent/Hy4` (`c10c0de5`) in BF16/FP32 over a 2,048-position frozen panel (`16 x 1,024` unpruned + `4 x 4,096` active DSA pruning at `K=2,048`), persisting actual DSA selections across shared layers.
   - Replays 8 paired prompts (`4 x 1,024` + `4 x 4,096`, `1,280` positions) from embedding to head under altered batch shapes (`1K: 4->1`, `4K: 1->2`) to measure end-to-end numerical drift `N_KL` (1,000 whole-prompt bootstrap samples, seed `20261120`).
   - Emits `artifacts/bf16_ref_logits.safetensors` (`FP32 [2048, 120832]`) and `null.json`.
   - Ledger ids (tier `later`): `quality.kl.mean.le2048`, `quality.kl.mean.gt2048`, `quality.top1.le2048`, `quality.top1.gt2048`.
3. **Thinking-mode probe (`thinking-probes.json`)**:
   - Uses `chat_template.jinja` directly from the model repository (`$MODEL_DIR/chat_template.jinja`, not copied here).
   - Compares `reasoning_effort: "high"` (position 35, ends with `<think:opensource>`) against `reasoning_effort: "no_think"` (position 38, ends with `<think:opensource></think:opensource>`).
4. **Router audit (`router_audit.py`)**:
   - Measures unordered expert replacements (`8 - |ref_ids & cand_ids|`) against both the BF16 reference and an unquantized same-engine control, binned by FP32 choice-score margin (`rank8 - rank9`).

## Reference self-checks

Run both self-checks before trusting any quantized KL or Top-1 score:

1. **Planted one-position shift (negative control)**:
   ```bash
   python3 eval/teacher.py control --source "$SOURCE_URL" --out artifacts
   ```
   Repeats real `L0 + head`, `L1`, and `L2` against saved CUDA oracle dumps, then cycles the one-based score positions left by one with wraparound. **Pass condition**: `L0..L2` and `head` pass all discrete ID and float bands, and the comparator returns **MISS** on the shifted logits before `control.json` is written.
2. **Full-precision repeat (reduction-order null `N_KL`)**:
   ```bash
   python3 eval/teacher.py teacher --source "$SOURCE_URL" --out artifacts
   ```
   Replays 8 paired prompts (4 x `1,024` and 4 x `4,096`, `1,280` positions) from embedding to head under altered batch shapes (`1K: 4 -> 1`, `4K: 1 -> 2`). **Pass condition**: `null.json` and `complete.json` are written with all output SHA-256 digests verified, and `N_KL` (largest of all, unpruned `<=2048`, and pruned `>2048` 95% bootstrap upper endpoints and observed means, 1,000 draws, seed `20261120`) is recorded (`quality.null.floor`) to calibrate the `KL` gate (`max(0.085, N_KL + 0.04)`).

## Commands

```bash
SOURCE_URL="http://<node3-cable-ip>:18765" SITE_DIR="/path/to/transformers-site" bash eval/run.sh
```

Router audit on retained layer dumps (`L21`, `L39`, `L57`):

```bash
python3 eval/router_audit.py --layer 21 --reference L21/BF16.safetensors --candidate L21/cand.safetensors --unquant L21/EXL3_UNQUANT.safetensors
```

## Script run status

- `eval/agreement.py`: the scorer the release boot runs (only its help text differs); its self-test passes with the standard library alone.
- `eval/teacher.py`, `eval/router_audit.py`, `eval/run.sh`: not yet run from this kit (requires PyTorch container with `transformers` and the HTTP BF16 shard server on a GPU node).
