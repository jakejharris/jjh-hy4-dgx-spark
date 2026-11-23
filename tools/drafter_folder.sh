#!/usr/bin/env bash
# Build the drafter folder that a launch from per-Spark parts needs for the packed (4-bit or 6-bit) drafter.
# The parts carry the BF16 drafter only. This folder holds five files, 8,566,396,282 bytes at 4b:
#   OUT/config.json
#   OUT/model-00245-of-00245.safetensors      the drafter's shard
#   OUT/trunk-W/dfly_layers.safetensors       the drafter layers, packed
#   OUT/trunk-W/lm_head.safetensors           the drafter head, packed
#   OUT/trunk-W/trunk.index.json
# Usage:
#   bash tools/drafter_folder.sh SRC OUT [--width 4b|6b]     (default 4b; DRAFTER_DIR is then OUT)
#   bash tools/drafter_folder.sh DIR DIR [--width 4b|6b]     (checks a folder built before; changes nothing)
# SRC is a full pack staged with the drafter and trunk-W/ (README steps 3 and 4), or a folder that holds these
# five files, for example from the weights repository. Each file is hard-linked when SRC and OUT share a file
# system (no extra disk), else copied. Then every file is checked against its SHA-256 in launch/weights.sha256.
# OUT must be new or empty. Every Spark needs the folder at the same path: build it where the full pack is and
# copy it, then check it on each Spark with DIR DIR (launch/STAGING.md, "Per-Spark parts").
set -euo pipefail

usage() { sed -n '9,11p' "$0" >&2; exit 2; }
[[ $# -ge 2 ]] || usage
src="${1%/}"
out="${2%/}"
shift 2
width="4b"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --width) [[ $# -ge 2 ]] || usage; width="$2"; shift 2 ;;
    *) usage ;;
  esac
done
case "$width" in
  4b|6b) ;;
  *) echo "ERROR: --width is 4b or 6b, not \"${width}\"." >&2; exit 2 ;;
esac

manifest="$(cd "$(dirname "$0")/.." && pwd)/launch/weights.sha256"
[[ -f "$manifest" ]] || { echo "ERROR: ${manifest} is not there; run this from the kit." >&2; exit 2; }
files=(config.json model-00245-of-00245.safetensors "trunk-${width}/dfly_layers.safetensors"
       "trunk-${width}/lm_head.safetensors" "trunk-${width}/trunk.index.json")
for f in "${files[@]}"; do
  if [[ ! -f "${src}/${f}" ]]; then
    echo "ERROR: ${src}/${f} is not there. SRC is a full pack staged with the drafter and trunk-${width}/," >&2
    echo "       or a folder that holds the five files." >&2
    exit 2
  fi
done

if [[ -d "$out" && "$(cd "$src" && pwd -P)" == "$(cd "$out" && pwd -P)" ]]; then
  echo "== checking ${out} in place"
else
  if [[ -e "$out" && -n "$(ls -A "$out")" ]]; then
    echo "ERROR: ${out} is not empty. Give a new or empty folder (or the same folder twice to check it)." >&2
    exit 2
  fi
  mkdir -p "${out}/trunk-${width}"
  for f in "${files[@]}"; do
    if ln -L "${src}/${f}" "${out}/${f}" 2>/dev/null; then
      echo "linked  ${f}"
    else
      cp -L "${src}/${f}" "${out}/${f}"
      echo "copied  ${f}"
    fi
  done
fi

fail=0
bytes=0
for f in "${files[@]}"; do
  want="$(awk -v f="$f" '$2 == f { print $1 }' "$manifest")"
  got="$(sha256sum < "${out}/${f}")"
  got="${got%% *}"
  bytes=$(( bytes + $(stat -L -c %s "${out}/${f}") ))
  if [[ -n "$want" && "$got" == "$want" ]]; then
    echo "OK      ${f}"
  else
    echo "FAIL    ${f}: sha256 ${got}, launch/weights.sha256 has ${want:-no line}"
    fail=1
  fi
done
if [[ "$fail" != 0 ]]; then
  echo "drafter folder ${out}: FAIL (a file differs from launch/weights.sha256). Remove ${out} and build it again from a matching SRC." >&2
  exit 1
fi
echo "drafter folder ${out}: ${#files[@]} files, ${bytes} bytes, each equal to launch/weights.sha256; set DRAFTER_DIR=\"${out}\""
