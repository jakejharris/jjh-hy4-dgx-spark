# Third-Party Software and Model Assets

| Component | Upstream / Pin | License | Role in Kit |
|---|---|---|---|
| `tencent/Hy4` | `c10c0de5` (https://huggingface.co/tencent/Hy4/tree/c10c0de5) | Tencent Hy License, Version 1.0 | Source BF16 model weights (774.4B) |
| `TensorFold` | `2cee7d9` at https://github.com/ashhart/TensorFold (patch includes upstream PR `#4004` `9ef309f39..1da00aa2f` by `ashhart` and `Shantanu Goel`, landed on `main` in `a3e21b3c5..1ae2cc7bf`) | Apache-2.0 (pre-0.6.0 MIT) | Serving engine base (`serve/`) |
| `gr8` | `kindlingai/gr8` at `4964aab` | MIT | Kernel and draft-length designs adapted into the TensorFold patch series (`serve/`), listed below |
| `ExLlamaV3` | `v1.6.7` at https://github.com/turboderp-org/exllamav3 (`8c593e4`; patch includes `hy4-flash` commit `1af65c4` by `Moshe Malawach` from `moshemalawach/exllamav3`) | MIT | EXL3 quantization base (`quant/`) |
| `llama.cpp` | `f8d4cc414` (PR `#32517`) + `stq1_0.patch` | MIT | Engine for the GGUF comparison, registered for a later run (`bench/`) |
| `AngelSlim/Hy4-GGUF` | `Hy4-STQ1_0.gguf` | Tencent Hy License, Version 1.0 | GGUF comparison weights, registered for a later run (`bench/`) |
| `NVIDIA PyTorch Container` | `nvcr.io/nvidia/pytorch:26.07-py3` | NVIDIA Deep Learning Container License | Runtime container with NCCL 2.30.7 |
| Agreement prefix sets | `eval/prefixes/familiar/manifest.json` and `eval/prefixes/fresh/manifest.json` (one entry per prefix) | Each excerpt keeps its source's license, named in the manifest | Prompts for the top-1 agreement figures (`eval/prefixes/`) |
| RigMark | `68800bf` at https://github.com/alexellis/rigmark (protocol 1.3.0) | MIT | Third-party benchmark behind the README's RigMark rows; not shipped ([RigMark](#rigmark)) |
| Hy4 vision reference files | `tencent/Hy4` repository files, Hugging Face `transformers` and vLLM's Hy4 model code ([listed below](#files-the-vision-code-follows)) | Apache-2.0 or Tencent Hy License, Version 1.0, per file | Followed by the vision code in the TensorFold patch (`serve/`); not in the kit |
| `libjpeg-turbo` | `3.1.4.1` (https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.4.1/libjpeg-turbo-3.1.4.1.tar.gz, sha256 `ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022`) | IJG License and BSD-3-Clause | Source of the JPEG decoder `libturbojpeg.so.0`, built unmodified by `serve/build_codecs.sh`; not shipped ([Image decoders](#image-decoders)) |
| `libwebp` | `1.6.0` (https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.6.0.tar.gz, sha256 `e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564`) | BSD-3-Clause, with the WebM Additional IP Rights Grant | Source of the WebP decoder `libwebpdecoder.so.3`, built unmodified by `serve/build_codecs.sh`; not shipped ([Image decoders](#image-decoders)) |

## RigMark

[RigMark](https://github.com/alexellis/rigmark) is a third-party benchmark by Alex Ellis, OpenFaaS Ltd. The README's RigMark rows are for protocol 1.3.0 at revision `68800bf29cf1f85f834affdf81fe3e8703327342`, run unmodified against our server. Its license is MIT ("Copyright (c) 2026 Alex Ellis, OpenFaaS Ltd"), as read from `LICENSE` at that revision (sha256 `19058b602fc9af61d619420a322be1deeb52c4a2da78a6ace1120c87a5e0b044`). No RigMark code is in this repository. The rows stay pending until a whole standard suite writes its JSON receipt, which goes in `receipts/rigmark/`. The one full run so far was cut short by a power cut and wrote no receipt; the README gives its rates from the client's and the server's logs, in `receipts/rigmark-cut-short/`.

<!-- licenses begin (LICENSES/: each project's own license file; sha256 of the file as committed) -->
## License texts

`LICENSES/` holds the text of every license named above, taken from each project's own license file at the version the kit uses (read 2026-11-21). The sha256 is of the file in `LICENSES/`, which is the fetched file unchanged except where the row says otherwise.

| File in `LICENSES/` | License | Project | Taken from | sha256 |
| --- | --- | --- | --- | --- |
| `Tencent-Hy-License-1.0.Hy4.txt` | Tencent Hy License, Version 1.0 | `tencent/Hy4` | https://huggingface.co/tencent/Hy4/raw/c10c0de5/LICENSE | `52755a6786f2f4b83b48cd7246aa9e34f7aac98563e4e60808ab02390799fdbd` |
| `Tencent-Hy-License-1.0.Hy4-GGUF.txt` | Tencent Hy License, Version 1.0 | `AngelSlim/Hy4-GGUF` | https://huggingface.co/AngelSlim/Hy4-GGUF/raw/main/LICENSE | `eeeeda6277e82258c47d4cb87a2574d1b67ce2c6b965112f0e3ecf779ef27d87` |
| `TensorFold.Apache-2.0.txt` | Apache-2.0 | `TensorFold` at `2cee7d9` | https://raw.githubusercontent.com/ashhart/TensorFold/2cee7d9/LICENSE | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `TensorFold.MIT-before-0.6.0.txt` | MIT (TensorFold before 0.6.0) | `TensorFold` at `094f5e4d74b5` (the LICENSE before commit `e3ac0eaad8d3`, "Apache-2.0 from 0.6.0") | https://raw.githubusercontent.com/ashhart/TensorFold/094f5e4d74b5/LICENSE | `be6a9ee4462784a762b21790954b8e76c4dcacc9fb6da2a40d03a8a9bddec93e` |
| `gr8.MIT.txt` | MIT | `kindlingai/gr8` at `4964aab` | https://raw.githubusercontent.com/kindlingai/gr8/4964aab/LICENSE | `70d01af48be4eca4e9d91c1199d2d7dddce0bc6846b9e99fdc68d35f437adef4` |
| `ExLlamaV3.MIT.txt` | MIT | `turboderp-org/exllamav3` at `v1.6.7` | https://raw.githubusercontent.com/turboderp-org/exllamav3/v1.6.7/LICENSE | `27a32b6263fcd96c79d3beeecf221c4366780bdf15ad51986f48650bd7369bff` |
| `llama.cpp.MIT.txt` | MIT | `ggml-org/llama.cpp` at `f8d4cc414` | https://raw.githubusercontent.com/ggml-org/llama.cpp/f8d4cc414/LICENSE | `94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d` |
| `NVIDIA-Deep-Learning-Container-License.pdf` | NVIDIA Deep Learning Container License | NVIDIA PyTorch Container | https://developer.download.nvidia.com/licenses/NVIDIA_Deep_Learning_Container_License.pdf | `7751feaf67e7f9d9003e829ab00b35cbdbc2f040513a2f8e0a066ae64850420c` |
| `Click.BSD-3-Clause.txt` | BSD-3-Clause | `pallets/click` at `8.1.7` (prefix set `familiar`) | https://raw.githubusercontent.com/pallets/click/8.1.7/LICENSE.rst | `9a8ad106a394e853bfe21f42f4e72d592819a22805d991b5f3275029292b658d` |
| `Flask.BSD-3-Clause.txt` | BSD-3-Clause | `pallets/flask` at `3.0.3` (prefix set `familiar`) | https://raw.githubusercontent.com/pallets/flask/3.0.3/LICENSE.txt | `489a8e1108509ed98a37bb983e11e0f7e1d31f0bd8f99a79c8448e7ff37d07ea` |
| `CPython.PSF-2.0.txt` | PSF-2.0 | `python/cpython` at `v3.12.7`, code and documentation (prefix set `familiar`) | https://raw.githubusercontent.com/python/cpython/v3.12.7/LICENSE | `3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf` |
| `Go.BSD-3-Clause.go1.23.2.txt` | BSD-3-Clause | `golang/go` at `go1.23.2` (prefix set `familiar`) | https://raw.githubusercontent.com/golang/go/go1.23.2/LICENSE | `2d36597f7117c38b006835ae7f537487207d8ec407aa9d9980794b2030cbc067` |
| `Go.BSD-3-Clause.ec1f7330f9.txt` | BSD-3-Clause | `golang/go` at `ec1f7330f9` (prefix set `fresh`; its copyright line differs from go1.23.2's) | https://github.com/golang/go/raw/ec1f7330f9/LICENSE | `911f8f5782931320f5b8d1160a76365b83aea6447ee6c04fa6d5591467db9dad` |
| `jq.MIT.txt` | MIT | `jqlang/jq` at `jq-1.7.1` (prefix set `familiar`) | https://raw.githubusercontent.com/jqlang/jq/jq-1.7.1/COPYING | `10e974638a41fadfd72357f2f3a4325e20b856c563365128f72feaa406f8c92d` |
| `nlohmann-json.MIT.txt` | MIT | `nlohmann/json` at `v3.11.3` (prefix set `familiar`) | https://raw.githubusercontent.com/nlohmann/json/v3.11.3/LICENSE.MIT | `86b998c792894ccb911a1cb7994f7a9652894e7a094c0b5e45be2f553f45cf14` |
| `Rust-book.MIT.txt` | MIT (of MIT OR Apache-2.0) | `rust-lang/book` at `main` (prefix set `familiar`) | https://raw.githubusercontent.com/rust-lang/book/main/LICENSE-MIT | `0621878e61f0d0fda054bcbe02df75192c28bde1ecc8289cbd86aeba2dd72720` |
| `Rust-book.Apache-2.0.txt` | Apache-2.0 (of MIT OR Apache-2.0) | `rust-lang/book` at `main` (prefix set `familiar`) | https://raw.githubusercontent.com/rust-lang/book/main/LICENSE-APACHE | `0f2763086f981043fb18879abfa15e75ecfc188219ef9eba4483f45ce7438e1c` |
| `serde.MIT.txt` | MIT (of MIT OR Apache-2.0) | `serde-rs/serde` at `v1.0.210` (prefix set `familiar`) | https://raw.githubusercontent.com/serde-rs/serde/v1.0.210/LICENSE-MIT | `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3` |
| `serde.Apache-2.0.txt` | Apache-2.0 (of MIT OR Apache-2.0) | `serde-rs/serde` at `v1.0.210` (prefix set `familiar`) | https://raw.githubusercontent.com/serde-rs/serde/v1.0.210/LICENSE-APACHE | `62c7a1e35f56406896d7aa7ca52d0cc0d272ac022b5d2796e7d6905db8a3636a` |
| `DuckDB.MIT.txt` | MIT | `duckdb/duckdb` at `86ed40fa28`, `6576e7f8fd` and `a8d5de33ea` (byte-equal at all three; prefix set `fresh`) | https://github.com/duckdb/duckdb/raw/86ed40fa28/LICENSE | `075c33400ffcb0c586dd106a029d3e733e9da3693f8fcb9cebfc651c8a2f14e3` |
| `uv.MIT.txt` | MIT (of MIT OR Apache-2.0) | `astral-sh/uv` at `9ac2aab082`, `d92359dd3a` and `7424870ae4` (byte-equal; prefix set `fresh`) | https://github.com/astral-sh/uv/raw/9ac2aab082/LICENSE-MIT | `860e3d7a86b84e6a7012c7a635fc64df475cebc6cce34dfeb73a5982ec58176c` |
| `uv.Apache-2.0.txt` | Apache-2.0 (of MIT OR Apache-2.0) | `astral-sh/uv` at the same three commits (byte-equal; prefix set `fresh`) | https://github.com/astral-sh/uv/raw/9ac2aab082/LICENSE-APACHE | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
| `curl.txt` | curl (MIT-style) | `curl/curl` at `1c08e37a61` (prefix set `fresh`) | https://github.com/curl/curl/raw/1c08e37a61/COPYING | `82f2f4427d6545ee5aaac4f0b80428da6cc8ba41c2cf5da3a03680ec327b9681` |
| `PEP-8.public-domain.txt` | Public domain | PEP 8, `python/peps` at `main` (prefix set `familiar`); the repository has no license file, so this is PEP 8's own Copyright section, cut verbatim | https://raw.githubusercontent.com/python/peps/main/peps/pep-0008.rst | `fb8a44757e5fc51d6dda51b95739f69af22469884676d6b5647513c8d67bf986` |
| `US-Government-works.txt` | Public domain (US Government, 17 U.S.C. 105) | EIA, NASA, NIST and Federal Register sources (prefix set `fresh`); no license file exists, so this lists the basis sentence and page per agency from the fresh manifest | the pages listed in the file | `fb7d0e76888308683f490b41bf40de6ba7b43c946d46af08d8ad7eb40e1c7947` |
| `libjpeg-turbo.BSD-3-Clause-IJG.txt` | BSD-3-Clause and IJG (libjpeg-turbo LICENSE.md) | `libjpeg-turbo` at `3.1.4.1` | LICENSE.md in https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.4.1/libjpeg-turbo-3.1.4.1.tar.gz | `e10114e6e40f3d0311c401ca25245ac5ef459a43c20f976fd63f03e816f5741f` |
| `libjpeg-turbo.README-IJG.txt` | IJG License (README.ijg) | `libjpeg-turbo` at `3.1.4.1` | README.ijg in https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.4.1/libjpeg-turbo-3.1.4.1.tar.gz | `75815e3bf6484201a3c3d17a1bbf10f2e8e3237f84df10a2357ea896db2a81d6` |
| `libwebp.BSD-3-Clause.txt` | BSD-3-Clause | `libwebp` at `1.6.0` | COPYING in https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.6.0.tar.gz | `5aec868f669e384a22372a4e8a1a6cd7d44c64cd451f960ca69cc170d1e13acf` |
| `libwebp.PATENTS.txt` | WebM Additional IP Rights Grant | `libwebp` at `1.6.0` | PATENTS in https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.6.0.tar.gz | `cc3273e0694ea5896145e0677699b53471b03ea43021ddc50e7923fbb9f5023c` |
<!-- licenses end -->

## Agreement prefixes

`eval/prefixes/familiar/` and `eval/prefixes/fresh/` hold short excerpts (384 tokens each) of published documents and source code, tokenized for Hy4, used only as prompts for the top-1 agreement figures. They are not part of the model or the engine. Each set's `manifest.json` is the license record, and `eval/prefixes/import_set.py` writes one table per set at the end of this file from it: for every prefix it names the source, its version, where it was fetched, the SHA-256 of the fetched file, the character range taken and the license. Each excerpt stays under its source's license; the text of every license named in the manifests is in `LICENSES/` (see "License texts" above), with each source's copyright line where its license file carries one. The reference picks file in each set folder holds token IDs produced by Tencent's Hy4 checkpoint on those prefixes (Tencent Hy License, Version 1.0).

## Files adapted from gr8

These files in the TensorFold patch series carry code or designs adapted from gr8 (`4964aab`). The MIT notice below applies to each of them.

| File in the patch series | gr8 source |
|---|---|
| `zig/kernels/cuda/hy4/trunk.cu` (4-bit trunk tiles) | `engine/kernels/tr3.cuh`, `engine/kernels/dtr.cuh`, `engine/kernels/dtr_exact_opt.cuh` |
| `zig/kernels/cuda/hy4/routed.cu` (routed expert scheduling) | `engine/kernels/k1.cuh` |
| `zig/kernels/cuda/hy4/routed_wide.cuh` (wide trellis decode) | `engine/kernels/tr3_wide.cuh` |
| `zig/kernels/cuda/hy4/mla_project.cu` (multi-row attention front) | register-reuse design from the gr8 kernels |
| `zig/src/families/hy4/draft_policy.zig` (draft length policy and cost table) | `engine/runtime/spec_len.h`, `engine/scheduler-rs/src/spec_len.rs`, `engine/runtime/s2_batch.h` |

```
MIT License

Copyright (c) 2026 gr8 engine contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Image decoders

This software is based in part on the work of the Independent JPEG Group.

The kit ships no decoder. `serve/build_codecs.sh` builds them, unmodified, from the upstream tarballs named in the table above, and whoever passes the built files on carries these notices:

- `libturbojpeg.so.0`, from libjpeg-turbo 3.1.4.1: Copyright (C) 2009-2026 D. R. Commander, Copyright (C) 2015 Viktor Szathmary, and Copyright (C) 1991-2020 Thomas G. Lane, Guido Vollbeding, under the IJG License and the Modified (3-Clause) BSD License (`LICENSES/libjpeg-turbo.BSD-3-Clause-IJG.txt`, `LICENSES/libjpeg-turbo.README-IJG.txt`).
- `libwebpdecoder.so.3`, from libwebp 1.6.0: Copyright (c) 2010, Google Inc. All rights reserved., under the 3-Clause BSD License and the WebM Additional IP Rights Grant (`LICENSES/libwebp.BSD-3-Clause.txt`, `LICENSES/libwebp.PATENTS.txt`).

## Files the vision code follows

The vision code in the TensorFold patch is Zig and CUDA. It follows these Python files for its algorithms, constants and layout. None of them is in the kit. The `transformers` and vLLM revisions were not recorded.

| File | Source and license | What the vision code takes from it |
|---|---|---|
| `modeling_hy_v4_vl.py` (vision patch embed, vision model, patch merger, `get_vision_position_ids`, `get_rope_index`, indexer, attention) | Hugging Face `transformers` reference for Hy4, "Tencent HunYuan Team and The HuggingFace Inc. team", Apache-2.0 (file header) | The 40-layer tower and its merger (2x2 stride-2 convolution, GELU, 1x1 convolution, a learned newline at each row end); the position-embedding grid and its (size + 0.1) / 128 scale; 3D positions and their deltas; the indexer's interleaved RoPE on dims 64 to 127, against MLA's split-half RoPE |
| `configuration_hy_v4_vl.py` | the same, Apache-2.0 (file header) | Default geometry and the image marker token ids (120118 to 120121) |
| `config.json`, `processor_config.json` | `tencent/Hy4` at `c10c0de5` (SHA-256 in `launch/vision.sha256`); no header, so the repository's Tencent Hy License, Version 1.0 | Geometry (hidden size 1536, 40 layers, 16 heads, patch 16, merge 2, `mrope_section` [12, 10, 10]), image mean and std 0.5, pixel floor 4,096 and ceiling 4,194,304 |
| `image_processing_pil_hy_v4_vl.py` (`smart_resize`) and `transformers`' `image_transforms.py` | `tencent/Hy4` repository, Apache-2.0 (file header); `transformers`, Apache-2.0 | The resize rule (grid 32, aspect limit 200), bicubic resampling, the 1/255 rescale and (x - 0.5) / 0.5 normalization, and the patch order |
| `processing_hy_v4_vl.py`, `hy_v4_vl_swift_patches.py`, `chat_template.jinja` | `tencent/Hy4` repository; the first Apache-2.0 (file header), the other two no license header, so the repository's Tencent Hy License, Version 1.0 | The placeholder count per image, (gh / 2) x (gw / 2 + 1) `<\|image_pad\|>` tokens, and the template's vision start, pad and end tokens |
| `model.py`, `hc.py`, `attention.py`, `dfly.py` | vLLM's Hy4 model code, "Copyright contributors to the vLLM project", Apache-2.0 (SPDX header) | The 4-stream residual channels, the MLA and indexer cache layout, and the drafter's anchor and M-RoPE position offset |

<!-- prefix-set:familiar begin (written by eval/prefixes/import_set.py; do not edit by hand) -->
| Prefix set `familiar`: source | URL | License | Prefixes |
| --- | --- | --- | ---: |
| Click Documentation: Arguments | https://raw.githubusercontent.com/pallets/click/8.1.7/docs/arguments.rst | BSD-3-Clause | 5 |
| CPython Lib/argparse.py (Python) | https://raw.githubusercontent.com/python/cpython/v3.12.7/Lib/argparse.py | PSF-2.0 | 6 |
| Python 3.12 Library: asyncio-task | https://raw.githubusercontent.com/python/cpython/v3.12.7/Doc/library/asyncio-task.rst | PSF-2.0 | 5 |
| Python 3.12 Tutorial: Classes | https://raw.githubusercontent.com/python/cpython/v3.12.7/Doc/tutorial/classes.rst | PSF-2.0 | 5 |
| CPython Lib/pathlib.py (Python) | https://raw.githubusercontent.com/python/cpython/v3.12.7/Lib/pathlib.py | PSF-2.0 | 6 |
| Flask Documentation: Quickstart | https://raw.githubusercontent.com/pallets/flask/3.0.3/docs/quickstart.rst | BSD-3-Clause | 5 |
| Go net/http/server.go (Go) | https://raw.githubusercontent.com/golang/go/go1.23.2/src/net/http/server.go | BSD-3-Clause | 6 |
| Go encoding/json/decode.go (Go) | https://raw.githubusercontent.com/golang/go/go1.23.2/src/encoding/json/decode.go | BSD-3-Clause | 6 |
| jq src/execute.c (C) | https://raw.githubusercontent.com/jqlang/jq/jq-1.7.1/src/execute.c | MIT | 6 |
| nlohmann/json.hpp (C++) | https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp | MIT | 6 |
| PEP 8: Style Guide for Python Code | https://raw.githubusercontent.com/python/peps/main/peps/pep-0008.rst | Public Domain / CC0 | 6 |
| The Rust Programming Language: Generics & Traits | https://raw.githubusercontent.com/rust-lang/book/main/src/ch10-02-traits.md | MIT OR Apache-2.0 | 5 |
| The Rust Programming Language: Ownership | https://raw.githubusercontent.com/rust-lang/book/main/src/ch04-01-what-is-ownership.md | MIT OR Apache-2.0 | 5 |
| serde_derive/src/de.rs (Rust) | https://raw.githubusercontent.com/serde-rs/serde/v1.0.210/serde_derive/src/de.rs | MIT OR Apache-2.0 | 6 |
| serde_derive/src/ser.rs (Rust) | https://raw.githubusercontent.com/serde-rs/serde/v1.0.210/serde_derive/src/ser.rs | MIT OR Apache-2.0 | 6 |
<!-- prefix-set:familiar end -->

<!-- prefix-set:fresh begin (written by eval/prefixes/import_set.py; do not edit by hand) -->
| Prefix set `fresh`: source | URL | License | Byline | License basis | Prefixes |
| --- | --- | --- | --- | --- | ---: |
| How does EIA account for small differences between energy supply and demand data? | https://www.eia.gov/todayinenergy/detail.php?id=68464 | Public Domain (US Gov, 17 USC 105) | EIA staff (no external contributor credit) | U.S. government publications are in the public domain and are not subject to copyright protection. (https://www.eia.gov/about/copyrights_reuse.php) | 1 |
| Hydropower generation in the Southwest fell to record lows this summer | https://www.eia.gov/todayinenergy/detail.php?id=68445 | Public Domain (US Gov, 17 USC 105) | EIA staff (no external contributor credit) | U.S. government publications are in the public domain and are not subject to copyright protection. (https://www.eia.gov/about/copyrights_reuse.php) | 1 |
| Refineries have shifted production away from gasoline to middle distillates | https://www.eia.gov/todayinenergy/detail.php?id=68424 | Public Domain (US Gov, 17 USC 105) | EIA staff (no external contributor credit) | U.S. government publications are in the public domain and are not subject to copyright protection. (https://www.eia.gov/about/copyrights_reuse.php) | 1 |
| Global refinery inputs fell sharply in the second quarter of 2026 before modest recovery | https://www.eia.gov/todayinenergy/detail.php?id=68404 | Public Domain (US Gov, 17 USC 105) | EIA staff (no external contributor credit) | U.S. government publications are in the public domain and are not subject to copyright protection. (https://www.eia.gov/about/copyrights_reuse.php) | 1 |
| U.S. crude oil imports from Venezuela in July were highest since 2019 | https://www.eia.gov/todayinenergy/detail.php?id=68425 | Public Domain (US Gov, 17 USC 105) | EIA staff (no external contributor credit) | U.S. government publications are in the public domain and are not subject to copyright protection. (https://www.eia.gov/about/copyrights_reuse.php) | 1 |
| EIA forecasts Alaskan crude oil production to grow 13% in 2026 | https://www.eia.gov/todayinenergy/detail.php?id=68444 | Public Domain (US Gov, 17 USC 105) | EIA staff (no external contributor credit) | U.S. government publications are in the public domain and are not subject to copyright protection. (https://www.eia.gov/about/copyrights_reuse.php) | 1 |
| U.S. natural gas inventories end injection season above normal | https://www.eia.gov/todayinenergy/detail.php?id=68484 | Public Domain (US Gov, 17 USC 105) | EIA staff (no external contributor credit) | U.S. government publications are in the public domain and are not subject to copyright protection. (https://www.eia.gov/about/copyrights_reuse.php) | 1 |
| Custodial Deposits: Implementing the 21st Century ROAD to Housing Act (Federal Deposit Insurance Corporation) | https://www.federalregister.gov/documents/2026/11/20/2026-23669/custodial-deposits-implementing-the-21st-century-road-to-housing-act | Public Domain (US Gov, 17 USC 105) | Federal Deposit Insurance Corporation | Works of the U.S. Government (17 U.S.C. 105) published in the Federal Register are in the public domain. (https://www.federalregister.gov/reader-aids/office-of-the-federal-register-announcements/copyright-and-public-domain) | 2 |
| Endangered and Threatened Wildlife and Plants; Endangered Species Status for Mazama Newt (Interior Department) | https://www.federalregister.gov/documents/2026/11/20/2026-23706/endangered-and-threatened-wildlife-and-plants-endangered-species-status-for-mazama-newt | Public Domain (US Gov, 17 USC 105) | Interior Department | Works of the U.S. Government (17 U.S.C. 105) published in the Federal Register are in the public domain. (https://www.federalregister.gov/reader-aids/office-of-the-federal-register-announcements/copyright-and-public-domain) | 2 |
| Endangered and Threatened Wildlife and Plants; Threatened Species Status With Section 4(d) Rule for White Sands Pupfish and Critical Habitat Determination (Interior Department) | https://www.federalregister.gov/documents/2026/11/20/2026-23711/endangered-and-threatened-wildlife-and-plants-threatened-species-status-with-section-4d-rule-for | Public Domain (US Gov, 17 USC 105) | Interior Department | Works of the U.S. Government (17 U.S.C. 105) published in the Federal Register are in the public domain. (https://www.federalregister.gov/reader-aids/office-of-the-federal-register-announcements/copyright-and-public-domain) | 2 |
| Taking and Importing Marine Mammals; Taking Marine Mammals Incidental to Military Readiness Activities in the Northwest Training and Testing Study Area (Commerce Department) | https://www.federalregister.gov/documents/2026/11/20/2026-23728/taking-and-importing-marine-mammals-taking-marine-mammals-incidental-to-military-readiness | Public Domain (US Gov, 17 USC 105) | Commerce Department | Works of the U.S. Government (17 U.S.C. 105) published in the Federal Register are in the public domain. (https://www.federalregister.gov/reader-aids/office-of-the-federal-register-announcements/copyright-and-public-domain) | 2 |
| curl/curl: lib/vquic/ngtcp2_shim.c | https://github.com/curl/curl/blob/1c08e37a61/lib/vquic/ngtcp2_shim.c | MIT/curl | Copyright (C) Daniel Stenberg, et al. | Permission to use, copy, modify, and distribute this software for any purpose with or without fee is hereby granted... (https://github.com/curl/curl/blob/master/COPYING) | 8 |
| curl/curl: lib/vquic/ngtcp2_shim.h | https://github.com/curl/curl/blob/1c08e37a61/lib/vquic/ngtcp2_shim.h | MIT/curl | Copyright (C) Daniel Stenberg, et al. | Permission to use, copy, modify, and distribute this software for any purpose with or without fee is hereby granted... (https://github.com/curl/curl/blob/master/COPYING) | 3 |
| duckdb/duckdb: src/include/duckdb/common/buffer_ptr.hpp | https://github.com/duckdb/duckdb/blob/86ed40fa28/src/include/duckdb/common/buffer_ptr.hpp | MIT | DuckDB contributors (DuckDB MIT header) | Permission is hereby granted, free of charge, to any person obtaining a copy of this software... (https://github.com/duckdb/duckdb/blob/main/LICENSE) | 4 |
| duckdb/duckdb: src/execution/operator/join/join_filter_pushdown.cpp | https://github.com/duckdb/duckdb/blob/6576e7f8fd/src/execution/operator/join/join_filter_pushdown.cpp | MIT | DuckDB contributors (DuckDB MIT header) | Permission is hereby granted, free of charge, to any person obtaining a copy of this software... (https://github.com/duckdb/duckdb/blob/main/LICENSE) | 11 |
| duckdb/duckdb: src/main/capi/v2/capi_v2_object_cache.cpp | https://github.com/duckdb/duckdb/blob/a8d5de33ea/src/main/capi/v2/capi_v2_object_cache.cpp | MIT | DuckDB contributors (DuckDB MIT header) | Permission is hereby granted, free of charge, to any person obtaining a copy of this software... (https://github.com/duckdb/duckdb/blob/main/LICENSE) | 3 |
| golang/go: src/crypto/tls/upki.go | https://github.com/golang/go/blob/ec1f7330f9/src/crypto/tls/upki.go | BSD-3-Clause | Copyright 2026 The Go Authors | Redistribution and use in source and binary forms, with or without modification, are permitted... (https://github.com/golang/go/blob/master/LICENSE) | 3 |
| astral-sh/uv: crates/uv-normalize/src/variant.rs | https://github.com/astral-sh/uv/blob/9ac2aab082/crates/uv-normalize/src/variant.rs | MIT OR Apache-2.0 | astral-sh/uv contributors (no third-party header) | Permission is hereby granted, free of charge, to any person obtaining a copy of this software... (https://github.com/astral-sh/uv/blob/main/LICENSE-MIT) | 1 |
| astral-sh/uv: crates/uv-distribution-types/src/proxy_index.rs | https://github.com/astral-sh/uv/blob/d92359dd3a/crates/uv-distribution-types/src/proxy_index.rs | MIT OR Apache-2.0 | astral-sh/uv contributors (no third-party header) | Permission is hereby granted, free of charge, to any person obtaining a copy of this software... (https://github.com/astral-sh/uv/blob/main/LICENSE-MIT) | 11 |
| astral-sh/uv: crates/uv-pep508/src/marker/algebra/platform.rs | https://github.com/astral-sh/uv/blob/7424870ae4/crates/uv-pep508/src/marker/algebra/platform.rs | MIT OR Apache-2.0 | astral-sh/uv contributors (no third-party header) | Permission is hereby granted, free of charge, to any person obtaining a copy of this software... (https://github.com/astral-sh/uv/blob/main/LICENSE-MIT) | 4 |
| New NASA Seed Funding Primes Small Businesses for Innovation | https://www.nasa.gov/directorates/rtmd/new-nasa-seed-funding-primes-small-businesses-for-innovation/ | Public Domain (US Gov, 17 USC 105) | Abby Tabor | NASA content – images, audio, video, and media files ... – generally are not subject to copyright in the United States. (https://www.nasa.gov/nasa-brand-center/images-and-media/) | 1 |
| NASA Honor Awards Ceremony Recognizes Employees at Stennis Space Center | https://www.nasa.gov/centers-and-facilities/stennis/honor-awards-ceremony-at-stennis/ | Public Domain (US Gov, 17 USC 105) | NASA Stennis Communications | NASA content – images, audio, video, and media files ... – generally are not subject to copyright in the United States. (https://www.nasa.gov/nasa-brand-center/images-and-media/) | 1 |
| The Stormy Southwest Pacific | https://science.nasa.gov/earth/earth-observatory/the-stormy-southwest-pacific/ | Public Domain (US Gov, 17 USC 105) | Story by Adam Voiland (NASA Earth Observatory) | NASA Earth Observatory (under NASA Science): NASA content is generally not subject to copyright in the United States unless noted. (https://science.nasa.gov/earth/earth-observatory/) | 1 |
| NASA Science Data Powers New Method to Discover Hard-to-Find Pulsars | https://science.nasa.gov/open-science/fermi-data-pulsar-discovery/ | Public Domain (US Gov, 17 USC 105) | By Amanda Moon Adams | NASA content – images, audio, video, and media files ... – generally are not subject to copyright in the United States. (https://www.nasa.gov/nasa-brand-center/images-and-media/) | 2 |
| MAF EAP – Holidays – How to manage the BAH HUMBUG that tries to sneak in | https://www.nasa.gov/centers-and-facilities/michoud/maf-eap/maf-eap-holidays-how-to-manage-the-bah-humbug-that-tries-to-sneak-in/ | Public Domain (US Gov, 17 USC 105) | Justin M. Robert (Web Editor) | NASA content – images, audio, video, and media files ... – generally are not subject to copyright in the United States. (https://www.nasa.gov/nasa-brand-center/images-and-media/) | 2 |
| NASA Announces Mid-Power Electric Rocket Challenge for US Companies | https://www.nasa.gov/directorates/stmd/prizes-challenges-crowdsourcing-program/nasa-announces-mid-power-electric-rocket-challenge-for-us-companies/ | Public Domain (US Gov, 17 USC 105) | Bailey Light | NASA content – images, audio, video, and media files ... – generally are not subject to copyright in the United States. (https://www.nasa.gov/nasa-brand-center/images-and-media/) | 1 |
| What Is the Moon Base? | https://www.nasa.gov/learning-resources/for-kids-and-students/what-is-the-moon-base/ | Public Domain (US Gov, 17 USC 105) | NASA STEM Team | NASA content – images, audio, video, and media files ... – generally are not subject to copyright in the United States. (https://www.nasa.gov/nasa-brand-center/images-and-media/) | 2 |
| BPS Digital AVATAR Challenge | https://www.nasa.gov/directorates/stmd/prizes-challenges-crowdsourcing-program/center-of-excellence-for-collaborative-innovation-coeci/bps-digital-avatar-challenge/ | Public Domain (US Gov, 17 USC 105) | Bailey Light | NASA content – images, audio, video, and media files ... – generally are not subject to copyright in the United States. (https://www.nasa.gov/nasa-brand-center/images-and-media/) | 1 |
| NICER Status Updates | https://www.nasa.gov/missions/station/nicer-status-updates/ | Public Domain (US Gov, 17 USC 105) | Jeanette Kazmierczak, Francis Reddy | NASA content – images, audio, video, and media files ... – generally are not subject to copyright in the United States. (https://www.nasa.gov/nasa-brand-center/images-and-media/) | 3 |
| Contractor to Civil Servant: NASA Welcomes Jim Mirandy | https://www.nasa.gov/centers-and-facilities/stennis/contractor-to-civil-servant-jim-mirandy/ | Public Domain (US Gov, 17 USC 105) | Bo Black (Communications Strategist) | NASA content – images, audio, video, and media files ... – generally are not subject to copyright in the United States. (https://www.nasa.gov/nasa-brand-center/images-and-media/) | 1 |
| I Am Artemis: Katie Spicer | https://www.nasa.gov/missions/artemis/i-am-artemis-katie-spicer/ | Public Domain (US Gov, 17 USC 105) | Beverly Perry (Communications Strategist) | NASA content – images, audio, video, and media files ... – generally are not subject to copyright in the United States. (https://www.nasa.gov/nasa-brand-center/images-and-media/) | 2 |
| El Niño Rains Paint the Atacama | https://science.nasa.gov/earth/earth-observatory/el-nino-rains-paint-the-atacama/ | Public Domain (US Gov, 17 USC 105) | NASA Earth Observatory | NASA Earth Observatory (under NASA Science): NASA content is generally not subject to copyright in the United States unless noted. (https://science.nasa.gov/earth/earth-observatory/) | 2 |
| Thank You for Doing NASA Science! | https://science.nasa.gov/get-involved/citizen-science/thank-you-for-doing-nasa-science/ | Public Domain (US Gov, 17 USC 105) | NASA Science (science.nasa.gov) | NASA content – images, audio, video, and media files ... – generally are not subject to copyright in the United States. (https://www.nasa.gov/nasa-brand-center/images-and-media/) | 2 |
| NASA Data Clinches First-of-Its-Kind Stellar Merger | https://science.nasa.gov/missions/webb/nasa-data-clinches-first-of-its-kind-stellar-merger/ | Public Domain (US Gov, 17 USC 105) | NASA Science (science.nasa.gov) | NASA content – images, audio, video, and media files ... – generally are not subject to copyright in the United States. (https://www.nasa.gov/nasa-brand-center/images-and-media/) | 3 |
| Sierra Snow Season Spins Up | https://science.nasa.gov/earth/earth-observatory/sierra-snow-season-spins-up/ | Public Domain (US Gov, 17 USC 105) | Story by Lindsey Doermann (NASA Earth Observatory) | NASA Earth Observatory (under NASA Science): NASA content is generally not subject to copyright in the United States unless noted. (https://science.nasa.gov/earth/earth-observatory/) | 1 |
| NASA Launches Nuclear Power Systems Student Contest | https://science.nasa.gov/centers-and-facilities/glenn/nasa-launches-nuclear-power-systems-student-contest/ | Public Domain (US Gov, 17 USC 105) | NASA Glenn / Power to Explore (www.nasa.gov) | NASA content – images, audio, video, and media files ... – generally are not subject to copyright in the United States. (https://www.nasa.gov/nasa-brand-center/images-and-media/) | 1 |
| NASA Space Apps Challenge Breaks Global Hackathon Records | https://science.nasa.gov/open-science/nasa-space-apps-challenge-breaks-global-hackathon-records/ | Public Domain (US Gov, 17 USC 105) | NASA Space Apps (www.nasa.gov) | NASA content – images, audio, video, and media files ... – generally are not subject to copyright in the United States. (https://www.nasa.gov/nasa-brand-center/images-and-media/) | 1 |
| NIST Project Aims to Accelerate Mobile Driver’s License Adoption | https://www.nist.gov/news-events/news/2026/11/nist-project-aims-accelerate-mobile-drivers-license-adoption | Public Domain (US Gov, 17 USC 105) | NIST staff (no external byline) | With the exception of material marked as copyrighted, information presented on NIST sites are considered public information and may be distributed or copied. (https://www.nist.gov/copyrights-disclaimers) | 2 |
| Quantum Physics Can Verify a Person’s Location, NIST Researchers Show | https://www.nist.gov/news-events/news/2026/11/quantum-physics-can-verify-persons-location-nist-researchers-show | Public Domain (US Gov, 17 USC 105) | NIST staff (no external byline) | With the exception of material marked as copyrighted, information presented on NIST sites are considered public information and may be distributed or copied. (https://www.nist.gov/copyrights-disclaimers) | 2 |
| Self-Employed? Solopreneur? NIST Has a Cybersecurity Guide Just for You | https://www.nist.gov/news-events/news/2026/11/self-employed-solopreneur-nist-has-cybersecurity-guide-just-you | Public Domain (US Gov, 17 USC 105) | NIST staff (no external byline) | With the exception of material marked as copyrighted, information presented on NIST sites are considered public information and may be distributed or copied. (https://www.nist.gov/copyrights-disclaimers) | 2 |
<!-- prefix-set:fresh end -->
