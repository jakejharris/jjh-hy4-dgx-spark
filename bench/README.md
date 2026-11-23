# Benchmarking our build

The same prompt files under `bench/prompts/` and the same counting harness (`bench/bench.py`) measure the TensorFold `TP=4` server on four DGX Sparks. The commands below are the ones the release boot ran (2026-11-21, `4b6b` full quant, four Sparks, 78 layers), with paths made relative to the kit. A `GGUF` comparison on `llama.cpp` RPC is kept at the end, registered for a later run; it is not part of the release numbers.

## Prompts

- `bench/prompts/known-answer.txt`: short correctness check (`17 * 23`, capital of Japan, symbol for gold).
- `bench/prompts/short-prose.txt`: `57`-token prompt; the release decode rows use `max_tokens=128`, 3 runs.
- `bench/prompts/short-code.txt`: `63`-token prompt; `max_tokens=128`, 3 runs.
- `bench/prompts/pp-4k.txt`: `4,040`-token prefill, `max_tokens=32` (3 runs).
- `bench/prompts/pp-32k.txt`: `32,010`-token prefill; the 32K rows use `max_tokens=64` (3 runs) on a 32,768-token boot.
- `bench/prompts/niah-4k.txt` (+ `niah-4k.meta.json`): `3,072`-token rendered prompt (`3,051` raw + `21` thinking-on chat template; `3,075` `no_think`), needle code planted at depth `0.4867`, `max_tokens=768` inside `4,096`.
- `bench/prompts/niah-128k.txt` (+ `niah-128k.meta.json`): `128,000`-token rendered prompt (`127,979` raw + `21` thinking-on chat template; `128,003` `no_think`), needle code planted at depth `0.4969`, `max_tokens=2048` inside `131,072` (`1,024` tokens headroom).

## 1. Our build (TensorFold `1.3`, `TP=4`, four Sparks)

Boot the build you want to measure, one at a time, with nothing else running on the Sparks. The release numbers ran over the relay with a row-cost table measured on the same ring (README, step 6):

```bash
# plain boot (drafts off): settings files are sourced in order, so set it in one more overlay
printf 'SERVE_DRAFTS="0"\n' > launch/ring.plain.env
bash launch/launch.sh --timed --env-file launch/ring.env --env-file launch/ring.relay.env.example --env-file launch/ring.plain.env
# drafted boot (the default)
bash launch/launch.sh --timed --env-file launch/ring.env --env-file launch/ring.relay.env.example
```

On each boot, once the server answers, run the 4K set (`URL` is the server, `MODEL` the name it serves; `--reasoning-effort no_think` is the thinking-off row, omitted is thinking on):

```bash
python3 -u bench/bench.py --url URL --model MODEL --prompt bench/prompts/known-answer.txt --timeout 7200 --runs 1 --max-tokens 128 --label ka --out results/known_answer.json
python3 -u bench/bench.py --url URL --model MODEL --prompt bench/prompts/short-prose.txt --timeout 7200 --runs 3 --max-tokens 128 --label prose --out results/MODE_prose_default.json
python3 -u bench/bench.py --url URL --model MODEL --prompt bench/prompts/short-prose.txt --reasoning-effort no_think --timeout 7200 --runs 3 --max-tokens 128 --label prose-nothink --out results/MODE_prose_nothink.json
python3 -u bench/bench.py --url URL --model MODEL --prompt bench/prompts/short-code.txt --timeout 7200 --runs 3 --max-tokens 128 --label code --out results/MODE_code_default.json
python3 -u bench/bench.py --url URL --model MODEL --prompt bench/prompts/pp-4k.txt --timeout 7200 --runs 3 --max-tokens 32 --label pp4k --out results/MODE_pp4k_default.json
```

`MODE` is `plain` or `dfly`. Then compare the token IDs of the two boots, prompt by prompt (first run of each). Drafted tokens are the full model's own greedy picks, checked in a wider batch, so at a close call they can differ from plain decoding; this records how many differ and where they start:

```bash
python3 bench/compare_tokens.py \
  --pair prose_default:results/plain_prose_default.json:results/dfly_prose_default.json \
  --pair prose_nothink:results/plain_prose_nothink.json:results/dfly_prose_nothink.json \
  --pair code_default:results/plain_code_default.json:results/dfly_code_default.json \
  --out results/draft_diff.json
```

Write the systems receipts from the boot's log folder (host names become `node1` to `node4` as they are written):

```bash
python3 launch/collect_receipts.py --logdir LOGDIR --env-file launch/ring.env --launch-log results/dfly-launch.log --known-answer results/known_answer.json --label 78l-4b6b-4k-dfly --isolation quiet --out-dir receipts/systems/exl3-4k
```

Needle at 128K, on a 131,072-token boot (exact-match scoring on the final answer, recording thinking mode separately):

```bash
python3 bench/niah.py run --url URL --model MODEL --label niah-128k --prompt bench/prompts/niah-128k.txt --meta bench/prompts/niah-128k.meta.json --max-tokens 2048 --out results/niah-128k.json
python3 bench/niah.py score --input results/niah-128k.json --meta bench/prompts/niah-128k.meta.json
```

For the thinking-off row, add `--reasoning-effort no_think` to the `run` line and use `--max-tokens 256` (the prompt renders to 128,003 tokens that way), with its own `--label` and `--out` names.

What these gave on the `4b6b` full quant over the relay at 4K (2026-11-21, three runs, medians; the cards carry them with their receipts once filled):

| Prompt | Plain | With drafts | Drafts kept per pass | Tokens that differ, drafts on against off |
| --- | ---: | ---: | ---: | --- |
| Prose, thinking on | 17.56 tok/s | 32.66 tok/s | 2.34 | 10 of 128, first at 118 |
| Prose, thinking off | 17.53 tok/s | 24.76 tok/s | 1.23 | 27 of 128, first at 100 |
| Code, thinking on | 17.44 tok/s | 36.74 tok/s | 2.85 | 0 of 128 |
| 4K prompt, 32 tokens | 14.90 tok/s | 43.29 tok/s | 4.17 | not compared |

Plain runs repeated exactly (three of three, every prompt). Drafted runs did not always: one run in three differed from the others on prose thinking off and on code, because the engine picks each round's verify width from live round times ([../README.md](../README.md), Drafts). The 32K runs go on the cards when they are filled; the 128K and needle runs have not run yet.

## Script run status

- `bench/bench.py`: ran every 4K measurement above on four Sparks on 2026-11-21 (TensorFold `TP=4`, 78 layers).
- `bench/compare_tokens.py`: wrote the drafted-against-plain comparison of the same boots.
- `bench/niah.py`: `generate` run as written on `2026-11-21` using the real Hy4 tokenizer and chat template to build `bench/prompts/niah-128k.txt` (`128,000` rendered tokens, `max_tokens=2048`) and `bench/prompts/niah-4k.txt` (`3,072` rendered tokens, `max_tokens=768`); `score` unit-tested on `7` synthetic pass/fail fixtures; `run` on the 128K boot not reported yet.

## 2. Registered for a later run: GGUF baseline (`AngelSlim/Hy4-GGUF`, `Hy4-STQ1_0.gguf`)

Not part of the release numbers. Build `Little0o0/llama.cpp` (`f8d4cc414`, PR #32517) + `stq1_0.patch` (`GGML_TYPE_STQ1_0 = 44`) with CUDA and RPC:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DGGML_RPC=ON -DLLAMA_CURL=OFF -DGGML_NATIVE=ON \
  -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF
cmake --build build --config Release -j 16 --target llama-server llama-cli ggml-rpc-server llama-bench
```

Start `ggml-rpc-server` on `node1`, `node2`, `node3` bound to each node's 200 Gb/s cable IP (set `GGML_RPC_NO_RDMA=1` on the 2-hop opposite peer to avoid RoCEv2 retry timeouts):

```bash
ggml-rpc-server -H "$CABLE_IP" -p 50052 -c
```

Start `llama-server` on `node4` (`--tensor-split` maps to `RPC0,RPC1,RPC2,CUDA0`; use `25,25,25,25` for an even split):

```bash
llama-server \
  -m "$GGUF_DIR/Hy4-STQ1_0.gguf" \
  --rpc "$NODE2_IP:50052,$NODE3_IP:50052,$NODE1_IP:50052" \
  -ts 25,25,25,25 -ngl 99 -c 32768 -np 1 --jinja --metrics --host 0.0.0.0 --port 8080
```

Run the baseline suite (`cache_prompt: false`, `temperature: 0.0`, `seed: 42`):

```bash
python3 bench/bench.py --url http://127.0.0.1:8080 --prompt bench/prompts/short-prose.txt --label gguf-prose --runs 3 --max-tokens 256 --out results/gguf-prose.json
python3 bench/bench.py --url http://127.0.0.1:8080 --prompt bench/prompts/short-code.txt  --label gguf-code  --runs 3 --max-tokens 256 --out results/gguf-code.json
python3 bench/bench.py --url http://127.0.0.1:8080 --prompt bench/prompts/pp-4k.txt       --label gguf-pp4k  --runs 3 --max-tokens 32  --out results/gguf-pp4k.json
python3 bench/bench.py --url http://127.0.0.1:8080 --prompt bench/prompts/pp-32k.txt      --label gguf-pp32k --runs 3 --max-tokens 16  --out results/gguf-pp32k.json
```
