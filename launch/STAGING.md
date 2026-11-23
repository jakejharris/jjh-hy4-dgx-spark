# Continuous Cable-Only Layer Staging

`launch/stage_layers.py` keeps a staged `TP=4` model directory synchronized across all four nodes while quantization runs on the fourth node (`RING_HOSTS` rank 3).

## How it works

1. Verifies that every transfer hop is a direct 1-hop cable route (`STAGE_CABLE_IFACE_PREFIX`, no `via`). Fails closed before copying if a hop would fall back to Wi-Fi or management Ethernet.
2. Reads each finished layer (`layers_done` in `QUANT_STATE_PATH`) **once** on the quantizing node (`rank 3`), computing source SHA-256 inline while streaming to `rank 0`.
3. Relays `rank 0 -> rank 1 -> rank 2` over direct copper links (`ssh -T -o Compression=no -c aes128-gcm@openssh.com`), writing to a hidden `.tmp` file, verifying SHA-256 and byte count against the source, and atomically renaming.
4. Symlinks the finished layer on `rank 3` (zero extra reads and zero duplicate disk space on the quantizing node) and atomically updates `config.json` and `model.safetensors.index.json` across all four nodes so `--max-layers N` works at any point.

## Configure `launch/ring.env`

Set the staging keys from `launch/ring.env.example`:
- `STAGE_DIR`: destination model directory present at the same path on all four nodes.
- `STAGE_CABLE_PULL_IP`, `STAGE_CABLE_HOP1_IP`, `STAGE_CABLE_HOP2_IP`: direct 1-hop cable IPs for `rank 0 -> rank 3`, `rank 0 -> rank 1`, and `rank 1 -> rank 2`.
- `QUANT_STATE_PATH`, `QUANT_LAYERS_DIR`, `ORIG_MODEL_DIR`: paths on the quantizing node (`rank 3`).
- `STAGE_STATE_DIR`: local directory on the controller for `verified.json` and the `STOP` file.

## Start, Verify, and Stop

Start unattended in a detached `tmux` session (runs until layer 77 is staged):

```bash
tmux new-session -d -s hy4-stage-layers \
  "python3 -u launch/stage_layers.py --env-file launch/ring.env >> /tmp/hy4-stage.log 2>&1"
```

Verify already-staged layers across all four nodes without modifying any files:

```bash
python3 -u launch/stage_layers.py --env-file launch/ring.env --verify-only --state-dir /tmp/hy4-stage-verify
```

## Pre-launch integrity checks (run before any rank starts)

1. **Header check on one node** (takes seconds before a four-node load): confirms every staged sparse layer matches layer 1's exact tensor count, keys, shapes, and dtypes, and that every routed expert projection carries `.mul1`:

```bash
python3 -u launch/check_staged_headers.py --env-file launch/ring.env
```

2. **Full-file four-node SHA-256 check**: TensorFold's startup pack check hashes only each file's safetensors header and its first and last 64 KiB, so a corruption in the middle of a shard on one node would pass startup and produce a silent wrong result. Always run `--verify-only` on the full layer set right before launch:

```bash
python3 -u launch/stage_layers.py --env-file launch/ring.env --verify-only --state-dir /tmp/hy4-stage-verify
```

## Drafter staging (on by default)

Drafts are on in the default launch, so the stager stages the drafter unless `STAGE_DRAFTER="0"` (`launch/ring.env.example` sets `"1"`, and a missing or empty line also means `"1"`): it copies the 6.58 GB drafter shard (`model-00245-of-00245.safetensors`) over the direct cables with the same four-node SHA-256 verification, adds its 82 tensors to `model.safetensors.index.json`, and keeps `dfly_num_hidden_layers` and the target layer lists in `config.json`. `--include-drafter` forces the same for one run:

```bash
python3 -u launch/stage_layers.py --env-file launch/ring.env --include-drafter
```

With `STAGE_DRAFTER="0"` the staged folder leaves `dfly_num_hidden_layers=0` and omits the shard; serve that folder with `SERVE_DRAFTS="0"`. Adding the drafter changes the pack hash, so never do it against a live staged folder while serving, and switch all four nodes together.

## Trunk staging (the full quant)

The trunk pass writes `trunk-4b/` and `trunk-6b/` on the node that ran it. `launch/stage_trunk.py` copies each finished layer file to the other three nodes over the direct cables (hidden temporary file, SHA-256 and byte length checked against the source, then an atomic rename), and copies `lm_head.safetensors`, `dfly_layers.safetensors` and `trunk.index.json` only once all 78 layers are done on the source, so a half-staged folder never looks complete. Re-runs skip files that already verify.

```bash
python3 launch/stage_trunk.py --env-file launch/ring.env
python3 launch/stage_trunk.py --env-file launch/ring.env --verify-only
```

`--source-rank` names the rank in `RING_HOSTS` that ran the trunk pass (default 2), and `--folders` limits the copy (default `trunk-4b,trunk-6b`). The lossless trunk option with drafts still needs `trunk-4b/` for the drafter's 4-bit files.

**Serving folder.** The launcher serves `MODEL_DIR` and expects `MODEL_DIR/trunk-<width>/` inside it. With `STAGE_DIR` unset, the stagers write straight into `MODEL_DIR`, and nothing copies the 78 layers a second time. If you stage into a separate `STAGE_DIR`, `stage_trunk.py` links each folder into `MODEL_DIR` once its stage passes (or into `--link-into DIR`): hard links first, which take no extra disk and no copy time. Where a node refuses hard links, it uses one symlink per file instead and logs `mode=symlink` for that node; `launch/launch.sh` mounts the symlink targets into the container. We hit this on our ring: on one Spark the quantizer had written the trunk files as root, and `fs.protected_hardlinks=1` refused hard links to them, so that node served from per-file symlinks. Do not freeze a serving folder by copying the layers: on our ring that copy of the 78 layers took 48 minutes.

## Per-Spark parts

To serve from per-Spark parts ([README](../README.md#per-spark-weight-files), `launch/ring.parts.env.example`), each Spark needs `common/` and its own `rankN/` at the same `PARTS_ROOT`, where N is its place in `RING_HOSTS`. For the 4-bit drafter, each also needs the drafter folder at the same `DRAFTER_DIR`.

**Cut and build on rank 0**, which holds the full pack (`FULL_PACK`). The drafter folder is hard links into the full pack, so it takes no extra disk there:

```bash
ORACLE="$ORACLE_BIN" bash tools/split_ranks.sh FULL_PACK /data/models/hy4-parts --trunk 4b6b --check
bash tools/drafter_folder.sh FULL_PACK /data/models/hy4-drafter-4b
```

On our rank 0 (2026-11-21, 23:10:47 UTC) the second line made 5 hard links, 8,566,396,282 bytes, and checked them, in 5.544 s. The disk in use grew by 8,192 bytes, the two new folders (`receipts/parts/drafter-folder.log`).

**Copy over the cables.** The routes are the stager's: from rank 0, `STAGE_CABLE_HOP1_IP` reaches rank 1 and `STAGE_CABLE_PULL_IP` reaches rank 3; from rank 1, `STAGE_CABLE_HOP2_IP` reaches rank 2. Run this on rank 0, with those three values from `launch/ring.env` in the shell. The third line passes through rank 1 without writing there, and the last leaves rank 0 with only its own part:

```bash
cd /data/models
SSH="ssh -T -o Compression=no -c aes128-gcm@openssh.com"
tar -cf - hy4-parts/common hy4-parts/rank1 hy4-drafter-4b | $SSH "$STAGE_CABLE_HOP1_IP" "tar -C /data/models -xf -"
tar -cf - hy4-parts/common hy4-parts/rank3 hy4-drafter-4b | $SSH "$STAGE_CABLE_PULL_IP" "tar -C /data/models -xf -"
tar -cf - hy4-parts/common hy4-parts/rank2 hy4-drafter-4b | $SSH "$STAGE_CABLE_HOP1_IP" "$SSH $STAGE_CABLE_HOP2_IP 'tar -C /data/models -xf -'"
rm -r hy4-parts/rank1 hy4-parts/rank2 hy4-parts/rank3
```

These lines are a plain form of the hand-out we ran. Ours used a copy tool of our own over the same three routes, which also hashed each file at the source and checked it on arrival. That tool is not in the kit: the check below finds a bad copy whatever made it, and the tool names our own hosts. With it, the drafter folder took 21.3 s to rank 1 and 26.3 s to rank 3, one cable each, and the next day 26.7 s from rank 1 to rank 2.

All three lines have since run as written, for the drafter folder only, into a scratch folder: 8,566,396,282 bytes in 7.8 s to rank 1 and 8.0 s to rank 3 (2026-11-21, `receipts/parts/staging-tar.log`), and in 14.5 s to rank 2 through rank 1, 0.59 GB/s over the two hops (2026-11-22, `receipts/parts/step0-node4.log`). The check below passed on all three copies. They have not run for the parts.

**Check on every Spark** before the first launch. The loader checks that the parts were cut from the same pack, but it does not hash every file, so this is the full-file check:

```bash
"$ORACLE_BIN" --check-rank-pack /data/models/hy4-parts
bash tools/drafter_folder.sh /data/models/hy4-drafter-4b /data/models/hy4-drafter-4b
```

The first line re-reads every file in `common/` and that Spark's `rankN/` against the SHA-256 recorded in its part; `bash tools/split_ranks.sh --check DIR` runs the same. On our four Sparks it printed `files=473 bytes=79706444762 tensors_compared=0 failures=0` on each, in 78.86 s with all four in parallel (2026-11-21, `receipts/parts/handout-parts.log`). The second line checks the five drafter files against `launch/weights.sha256` and needs the kit on that Spark. On all four of our Sparks it printed five `OK` lines and `5 files, 8566396282 bytes`: on three in 5.3 to 9.5 s each (2026-11-21, `receipts/parts/drafter-folder.log`), and on the fourth in 5.4 s the next day, after a power cut had kept it off ([FINDINGS.md](../FINDINGS.md#open-items), `receipts/parts/step0-node4.log`). A Spark can also build its drafter folder from those five files downloaded from the weights repository: `bash tools/drafter_folder.sh DOWNLOADED DIR`.

At every launch from parts, `launch/launch.sh` also checks that each Spark holds `common/` and its own `rankN/` and no other rank's folder, and that the drafter folder holds its five files.

## Systems receipts (after any four-rank boot)

`launch/launch.sh` writes `launch_meta.json`, `rank{0..3}_<host>.log`, and continuous `memfloor_r{0..3}.log` into `LOGDIR`. Generate structured systems receipts (`summary.json` plus per-claim JSON files matching `card/claims.json`, named `systems.*.exl3.json` or `systems.*.lossless.json` for the build the rank load lines report; `--build` states it and must agree) with the output folder named for the build and context, `receipts/systems/exl3/` for the full quant at 128K and `receipts/systems/lossless-128k/` for the lossless trunk:

```bash
python3 -u launch/collect_receipts.py \
  --logdir /tmp/hy4-tp4-logs \
  --env-file launch/ring.env \
  --label full-quant-128k \
  --isolation quiet \
  --out-dir receipts/systems/exl3
```

`python3 -I launch/test_collect_receipts.py` checks the collector offline against excerpts of real rank logs (`launch/fixtures/`).

## Stop cleanly

Stop cleanly after the current layer finishes:

```bash
touch "${STAGE_STATE_DIR}/STOP"
```

## Retention rule (do not clean the quant work folder)

Because `rank 3` symlinks staged layer files into `QUANT_LAYERS_DIR` (for example `/data/quant/qtensors`), **never delete, move, or clean `QUANT_LAYERS_DIR` while a server may be up or before the release is cut**, and do not restart or overwrite finished quant layers while a server is running. Mount `QUANT_LAYERS_DIR` read-only alongside `MODEL_DIR` on `rank 3` when running Docker containers (handled automatically by `launch/launch.sh`).
