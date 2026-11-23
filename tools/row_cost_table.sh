#!/usr/bin/env bash
# Measure TF_HY4_ROW_COST_TABLE on your own ring.
#
# The drafter chooses how many tokens to propose from a table of what one verify step costs at each
# batch width. This runs one whole-model verify bench on all four Sparks (widths 1, 2, 3 and 7, 15
# iterations, a 512-token synthetic prefix) through launch/launch.sh in command mode, then turns rank
# 0's log into the table with tools/make_row_cost_table.py. Run it with the same settings files you
# serve with, on a quiet ring, after staging and before launch; paste the printed line into ring.env.
#
# Usage:
#   bash tools/row_cost_table.sh --env-file launch/ring.env --env-file launch/ring.relay.env.example \
#     --oracle /path/to/tf-hy4-oracle [--trunk 4b6b] [--out row-cost-table.txt]
#
# --oracle   the tf-hy4-oracle binary from the same engine build as the server (serve/BUILD.md)
# --trunk    trunk width to bench (default: TF_HY4_TRUNK from the settings files; empty means lossless)
# --out      where the table line is written (default: row-cost-table.txt)
# The transport is relay when TF_TP_ROCE_RELAY=1 in the settings files, otherwise stock.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# launch/launch.sh lets a TF_TP_KERNEL_IMAGE from the caller's environment win over the settings files.
# Sourcing the files below overwrites it here, so keep the caller's value and hand it to launch.sh.
CALLER_TF_TP_KERNEL_IMAGE="${TF_TP_KERNEL_IMAGE:-}"
ENV_ARGS=()
ENV_FILES=()
ORACLE=""
TRUNK=""
OUT="row-cost-table.txt"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file) ENV_ARGS+=(--env-file "$2"); ENV_FILES+=("$2"); shift 2 ;;
    --oracle) ORACLE="$2"; shift 2 ;;
    --trunk) TRUNK="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ ${#ENV_FILES[@]} -eq 0 ]]; then
  echo "ERROR: pass --env-file launch/ring.env (and the relay overlay if you serve on the relay)" >&2
  exit 2
fi
if [[ -z "$ORACLE" ]]; then
  echo "ERROR: pass --oracle /path/to/tf-hy4-oracle (the same engine build as the server)" >&2
  exit 2
fi

for ef in "${ENV_FILES[@]}"; do
  # shellcheck disable=SC1090
  source "$ef"
done
TRUNK="${TRUNK:-${TF_HY4_TRUNK:-}}"
if [[ "$TRUNK" == "PICK" ]]; then
  echo "ERROR: TF_HY4_TRUNK is still PICK; set the width (the table is measured per trunk)" >&2
  exit 2
fi
TRUNK_ARGS=()
if [[ -n "$TRUNK" && "$TRUNK" != "bf16" ]]; then
  TRUNK_ARGS=(--trunk "$TRUNK")
fi
TRANSPORT="stock"
if [[ "${TF_TP_ROCE_RELAY:-}" == "1" ]]; then
  TRANSPORT="relay"
fi
read -r -a HOSTS <<< "$RING_HOSTS"

LOGDIR="$(mktemp -d)/rowcost"
export LOGDIR
export ORACLE_BIN="$ORACLE"
export PREFIX="hy4-rowcost"
export TIMEOUT_S="${TIMEOUT_S:-900}"
echo "row cost: trunk ${TRUNK:-lossless}, transport ${TRANSPORT}, logs ${LOGDIR}"

# --output: the 2026-11-21 bench run was refused (BadArgs) without --trunk and --output, and passed with
# both. The file lands in the container's own /tmp and goes away with it; only rank 0's log is read.
TF_TP_KERNEL_IMAGE="$CALLER_TF_TP_KERNEL_IMAGE" bash "${ROOT}/launch/launch.sh" "${ENV_ARGS[@]}" -- \
  "$ORACLE" --tp 4 --require-ib --model "$MODEL_DIR" "${TRUNK_ARGS[@]}" \
  --synthetic-len 512 --seed 20261120 --chunk-size 1024 \
  --bench-window 15 --bench-widths 1,2,3,7 --mem-floor-gib 8 \
  --output /tmp/hy4-rowcost.safetensors

LOG0="${LOGDIR}/rank0_${HOSTS[0]}.log"
if [[ ! -f "$LOG0" ]]; then
  echo "ERROR: rank 0 log not found: $LOG0" >&2
  exit 1
fi
python3 "${ROOT}/tools/make_row_cost_table.py" --log "$LOG0" --transport "$TRANSPORT" --out "$OUT"
echo "Add this line to launch/ring.env (or an overlay) and launch:"
echo "TF_HY4_ROW_COST_TABLE=\"$(cat "$OUT")\""
