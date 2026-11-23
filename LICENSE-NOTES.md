# License Notes

This file summarizes the license obligations for each part of the release and the license of our own scripts and patches.

## 1. Model weights (`tencent/Hy4`, Tencent Hy License, Version 1.0)

Under Section 1 of Tencent Hy License 1.0, quantized weights are **Derivative Works**. Redistribution is permitted in any medium provided we satisfy Section 4:
- Ship a full copy of Tencent Hy License, Version 1.0 (`LICENSE`) with the weights.
- Retain all upstream notices in Source form (no upstream `NOTICE` exists in `c10c0de5`).
- Carry prominent change notices on modified files (Section 4(b)). The weights repository's `NOTICE` lists every file we changed or added, by name or exact pattern, with counts and what changed: 238 `.safetensors` files and 4 others. The model card sums it up. The `.safetensors` files are not altered, so their SHA-256 sums stay those in `launch/weights.sha256`, which all four Sparks verified for the 238. A notice inside each file's header is a possible later version.
- The vision folder `vit-bf16/` is published beside the pack under the same license, and the same notice must cover it: `vit.safetensors` is new (the 649 `vit.*` tensors copied unchanged in BF16 from eight `c10c0de5` shards into one file), and `config.json` and `processor_config.json` are Tencent's files unchanged. Its `SHA256SUMS` matches `launch/vision.sha256`. The model card lists the three files.
- Pass Section 10 through unchanged: never apply plain Apache-2.0 or MIT to the quantized weights. Section 10.2 requires a separate written license from Tencent for MaaS operators exceeding US$50M aggregate worldwide gross revenue (12-month) or 100M monthly active users. Section 10.3 exempts solely internal use when neither weights nor outputs are shared with third parties.
- Section 6 grants no trademark use beyond reasonable and customary origin description (the repository names `jakejharris/jjh-hy4-dgx-spark` and `jakejharris/jjh-hy4-exl3` use `Hy4` only to describe the model's origin).

## 2. Serving port (`TensorFold`, Apache-2.0)

From `LICENSE`, `NOTICE`, and `CONTRIBUTING.md` in `tf-base`:
- **`LICENSE` (Apache 2.0)**: distributing a modified port requires shipping a copy of Apache 2.0, retaining `NOTICE` (`Copyright 2026 TensorFold contributors`, noting pre-0.6.0 MIT files and `THIRD_PARTY_NOTICES.md`), and adding prominent change notices to every modified file.
- **`CONTRIBUTING.md`**: describes upstream's PR workflow (`main`, opening an issue before a new model family, ATC contributor coordination, Zig `0.17.0`, host/device tests, kernel contracts, and exactness contracts). Because this kit ships as a standalone local port / patch series (`serve/make_patches.sh`), do not run `atc` or open upstream PRs unless explicitly publishing upstream.

## 3. Quantizer (`ExLlamaV3 v1.6.7`, MIT) and Baseline (`llama.cpp`, MIT)

Both are MIT-licensed: retain their copyright and permission notices when distributing patches or builds.

## 4. License for our own scripts and patches

Our original scripts, documents and patch series in this repository are licensed under Apache-2.0 (`LICENSE`), Copyright 2026 JJH Digital. The quantized weights stay under Tencent Hy License 1.0 with Section 10 intact, never under this license; third-party files keep their own licenses (`THIRD_PARTY.md`, `LICENSES/`).
