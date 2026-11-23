#!/usr/bin/env bash
# Split a full Hy4 pack into per-Spark weight files, using the engine's own slicing (tf-hy4-oracle):
#   OUT/common/   tensors all four ranks read whole (copy to every Spark)
#   OUT/rank<R>/  rank R's slices of every tensor the TP=4 loader reads sliced (copy only to the Spark running rank R)
# A Spark then boots with --model OUT/common and its rank<R>/ folder beside it; the loader refuses another rank's
# part, a part cut from a different source pack, and a mix of full-pack and part files.
# Usage:
#   bash tools/split_ranks.sh SRC_DIR OUT_DIR [--trunk 4b|4b6b|6b] [--target-head 4b|6b] [--ranks 0,1,2,3] [--layers L,L,...] [--no-common] [--check]
#   bash tools/split_ranks.sh --check OUT_DIR [SRC_DIR]
# --trunk and --target-head go to the exporter unchanged (the same values as TF_HY4_TRUNK and TF_HY4_TARGET_HEAD);
# it records them in the parts, and the loader refuses a part cut at another width. Leave both off for the
# lossless trunk. They need an exporter that accepts them (tf-hy4-oracle --help lists them).
# --check re-reads what was written: every file against the SHA-256 in its part's index and, with the source,
# every tensor against the exact source bytes it was cut from. ORACLE=/path/to/tf-hy4-oracle picks the binary
# (built per serve/BUILD.md); the split and the check need no GPU.
set -euo pipefail

ORACLE="${ORACLE:-tf-hy4-oracle}"

if [[ "${1:-}" == "--check" ]]; then
  [[ $# -ge 2 ]] || { sed -n '7,9p' "$0"; exit 2; }
  out="$2"
  src="${3:-}"
  exec "$ORACLE" --check-rank-pack "$out" ${src:+--source "$src"}
fi

[[ $# -ge 2 ]] || { sed -n '7,9p' "$0"; exit 2; }
src="$1"
out="$2"
shift 2
ranks="0,1,2,3"
layers=""
check=0
common_flag=()
export_flags=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ranks) ranks="$2"; shift 2 ;;
    --layers) layers="$2"; shift 2 ;;
    --no-common) common_flag=(--no-common); shift ;;
    --trunk) export_flags+=(--trunk "$2"); shift 2 ;;
    --target-head) export_flags+=(--target-head "$2"); shift 2 ;;
    --check) check=1; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -f "$src/config.json" ]] || { echo "ERROR: $src has no config.json (want a full pack folder)" >&2; exit 2; }
mkdir -p "$out"
"$ORACLE" --export-rank-pack "$src" --out "$out" --ranks "$ranks" ${layers:+--layers "$layers"} "${export_flags[@]}" "${common_flag[@]}"
if [[ "$check" == "1" ]]; then
  "$ORACLE" --check-rank-pack "$out" --source "$src"
fi
for part in "$out"/common "$out"/rank*; do
  [[ -d "$part" ]] && du -sb --apparent-size "$part"
done
