# Agreement prefix sets (`eval/prefixes/`)

The prompts for the top-1 agreement figures (`eval/agreement.py`, described in [../README.md](../README.md)). Every prefix is an excerpt of a published document or source file with a license that allows redistribution, so anyone can rerun the agreement row on their own four Sparks and compare.

Two sets, one folder each:

| Set | What it is | State |
| --- | --- | --- |
| `familiar/` | 84 prefixes (36 prose, 48 code) from widely copied documentation and source: PEP 8, the Rust book, the Python tutorial and library docs, Flask and Click docs, CPython, serde, Go, nlohmann/json and jq. | Imported 2026-11-21 with reference picks. 12 IETF RFC excerpts in the source set are left out, because the IETF Trust terms allow unmodified full copies, not cut passages. |
| `fresh/` | 96 prefixes (48 prose, 48 code), every source first published on or after 2026-11-01: US Government prose (EIA, NASA, NIST, Federal Register) and files newly added to uv, Go, DuckDB and curl. | Imported 2026-11-21 from its final freeze after the license audit, with the byline and license basis of every source. Its reference picks were added the same day from the reference pass that finished at 10:48 UTC, with `stats.json`; its agreement cells stay pending until a boot collects the served picks. |

Why two: a model that has memorized a text predicts it well whatever the quantization, so on a familiar set every build looks alike. Beside every agreement number the cards print, for the same set, the reference's own next-token accuracy on the published text and the share of close calls. On `familiar/` the reference predicts the published next token at 91.04% of positions (prose 92.15%, code 90.21%) and 6.38% of positions are close calls (2,057 of 32,256), from `familiar/stats.json`. A high accuracy with few close calls means there is little room for a build to differ; the fresh set is there to show the same comparison on text the model cannot have seen. On `fresh/` the reference predicts the published next token at 65.44% of positions (prose 54.94%, code 75.94%) and 17.23% of positions are close calls (6,353 of 36,864), from `fresh/stats.json`.

Each set folder holds:

| File | Contents |
| --- | --- |
| `manifest.json` | One entry per prefix, in order: index (and `source_idx`, its index in the set as built), genre (`prose` or `code`), source id and title, version, URL fetched, SHA-256 of the fetched file, character range taken, token length, license. At the top: counts, the SHA-256 of every source file read (`imported_from`) and of every file written (`files`), and the source ids left out as not shipped. |
| `prefixes.safetensors` | `input_ids` (I64, one row of 384 Hy4 token IDs per prefix) and `is_code` (U8, 1 for code). Row order matches the manifest. |
| `reference_picks.safetensors` | The reference's greedy next token after every length 1 to 384 of every prefix (`picks`, I64), its second choice (`top2`), the margin between its top two logits (`margin`, F32), and the input IDs it saw (`input_ids`). Produced by Tencent's checkpoint at full precision (BF16 weights, FP32 head). |
| `stats.json` | The reference's own next-token accuracy (`picks[:, :-1] == input_ids[:, 1:]`) and the share of close calls (margin under 0.5), overall and per genre. |

The license of every source is listed in [../../THIRD_PARTY.md](../../THIRD_PARTY.md), one row per source, written from the manifest. When a manifest carries `byline`, `license_basis_url` and `license_basis_sentence` (the fresh set does), the rows also give the byline and the sentence the license rests on, with the page it is quoted from. The full text of each license is in [../../LICENSES/](../../LICENSES/), taken from each project's own license file, with the file each came from and its sha256 in `THIRD_PARTY.md`; the import does not fetch them.

## Adding a set

```bash
python3 eval/prefixes/import_set.py NAME SRC_DIR --expect prefixes.safetensors=SHA256 --expect manifest.json=SHA256
```

`SRC_DIR` holds `manifest.json`, `prefixes.safetensors` and, when they exist, the reference picks (`reference_picks.safetensors` or `teacher_*.safetensors`, or name it with `--picks`). The script checks every `--expect` hash, refuses the set if any manifest entry has no license (or, in a set that uses them, lacks a byline or license basis, or disagrees with another entry of the same source on them), checks the manifest against the tensors and the picks against the prefixes, leaves out entries marked `"shipped": false`, then writes `eval/prefixes/NAME/` and this set's rows in `THIRD_PARTY.md`. It writes nothing when a check fails (exit 2). `--dry-run` reports what it would write.

The familiar set was imported with:

```bash
python3 eval/prefixes/import_set.py familiar SRC_DIR --picks teacher_public.safetensors \
  --expect prefixes.safetensors=95b52477e333076caa0205c8716fcf4a44dd448de82d3d886770800619167900 \
  --expect teacher_public.safetensors=0426580f67176543cca39baca8e6b53d16aea7fe82d2665a134f0838ba095abe \
  --expect manifest.json=127c0ef4d5817e1dce4e67ee311fbbc69006352906aae6d63eaf24fff373a6b1
```

The fresh set was imported with the hashes of its final freeze (2026-11-21 license audit), first without picks (`--no-picks`), then again with its reference picks and the reference pass's own stats file once that pass had run:

```bash
python3 eval/prefixes/import_set.py fresh SRC_DIR --manifest fresh_manifest.json --prefixes fresh_prefixes.safetensors --picks teacher_fresh.safetensors \
  --expect fresh_prefixes.safetensors=e5e6f8948d23e2552623f5adb33fd1395089e94330b06ba98b00f4dd99cd501b \
  --expect fresh_manifest.json=877288f26fed8f9a58af0c9d769ccba03b081fcc2e01497373cb4c0c11bec269 \
  --expect teacher_fresh.safetensors=bdabc45338b130206880bc089114eab6c49e21ca1b8eb439ac22a51db1f1eade \
  --expect teacher_fresh_stats.json=3adfc83b53fff863ea62517493a6423409d249e8c05099027c6a90682d22422c
```

A picks file whose `input_ids` are not this set's prefixes is refused. The `stats.json` the import computed from the picks gives the same figures as the reference pass's stats file: prose 54.94%, code 75.94%, 6,353 close calls of 36,864 positions.

## Checking a set

Fetch each URL in the manifest, compare its SHA-256, cut the character range, tokenize it with the Hy4 tokenizer and compare the first 384 IDs with the matching row of `prefixes.safetensors`. `eval/agreement.py score` scores only the positions present in `reference_picks.safetensors` and in every picks file it is given.

The sets exist so the agreement row can be rerun by anyone: every prefix comes from a public source named in the manifest, and none from this repository.
