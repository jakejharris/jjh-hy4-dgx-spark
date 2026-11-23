#!/usr/bin/env bash
# Copy one engine build to the same folder on every Spark in RING_HOSTS, through this machine, and check it.
#
#   bash launch/stage_binary.sh --env-file launch/ring.env --from HOST:DIR --to DIR
#
# Copies tensorfold-native, tf-hy4-oracle, hy4-boundary and tp_mailbox.fatbin (whichever of them DIR holds)
# from HOST:DIR to DIR on all four Sparks, then reads each file's SHA-256 on every Spark and refuses (exit 1)
# if any Spark differs from the source. The image decoders libturbojpeg.so.0 and libwebpdecoder.so.3, when the build
# has them in DIR/../lib (serve/build_release.sh puts them there), are copied and checked beside the binary, where
# the engine looks for them and launch/launch.sh mounts them. Then set TENSORFOLD_BIN="DIR/tensorfold-native" (and, with the relay,
# TF_TP_KERNEL_IMAGE="DIR/tp_mailbox.fatbin") in launch/ring.env. Keep DIR the same length when comparing two
# builds: the binary's path moved decode speed (serve/BUILD.md).
set -euo pipefail
ENV_FILE=""; FROM=""; TO=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    --to) TO="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "ERROR: unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -f "$ENV_FILE" && "$FROM" == *:/* && "$TO" == /* ]] || { echo "Usage: $0 --env-file launch/ring.env --from HOST:/dir --to /dir" >&2; exit 2; }
# shellcheck disable=SC1090
source "$ENV_FILE"
read -r -a HOSTS <<< "$RING_HOSTS"
SRC_HOST="${FROM%%:*}"; SRC_DIR="${FROM#*:}"; SRC_DIR="${SRC_DIR%/}"; TO="${TO%/}"
LOCAL="$(mktemp -d "${TMPDIR:-/tmp}/hy4-stage-bin.XXXXXX")"
trap 'rm -rf "${LOCAL:?}"' EXIT
FILES=(); SRCS=()
for f in tensorfold-native tf-hy4-oracle hy4-boundary tp_mailbox.fatbin; do
  if ssh -n -o BatchMode=yes "$SRC_HOST" "test -f '${SRC_DIR}/${f}'"; then FILES+=("$f"); SRCS+=("${SRC_DIR}/${f}"); fi
done
[[ " ${FILES[*]} " == *" tensorfold-native "* ]] || { echo "ERROR: ${FROM}/tensorfold-native not found" >&2; exit 2; }
NDEC=0
for f in libturbojpeg.so.0 libwebpdecoder.so.3; do
  if ssh -n -o BatchMode=yes "$SRC_HOST" "test -f '${SRC_DIR}/../lib/${f}'"; then
    FILES+=("$f"); SRCS+=("${SRC_DIR}/../lib/${f}"); NDEC=$((NDEC + 1))
  fi
done
echo "image decoders in ${FROM}/../lib: ${NDEC} of 2 (without both, JPEG and WebP get HTTP 400)"
for i in "${!FILES[@]}"; do scp -q "${SRC_HOST}:${SRCS[$i]}" "${LOCAL}/${FILES[$i]}"; done
(cd "$LOCAL" && sha256sum "${FILES[@]}") > "${LOCAL}/SHA256SUMS.src"
cat "${LOCAL}/SHA256SUMS.src"
FAIL=0
for h in "${HOSTS[@]}"; do
  ssh -n -o BatchMode=yes "$h" "mkdir -p '${TO}' && rm -f '${TO}/libturbojpeg.so.0' '${TO}/libwebpdecoder.so.3'"
  for f in "${FILES[@]}"; do scp -q "${LOCAL}/${f}" "${h}:${TO}/${f}"; done
  ssh -n -o BatchMode=yes "$h" "chmod +x '${TO}/tensorfold-native' '${TO}/tf-hy4-oracle' '${TO}/hy4-boundary' 2>/dev/null; true"
  if ssh -o BatchMode=yes "$h" "cd '${TO}' && sha256sum -c --strict --quiet -" < "${LOCAL}/SHA256SUMS.src"; then
    echo "OK: ${h}:${TO} matches the source (${#FILES[@]} files)"
  else
    echo "ERROR: ${h}:${TO} does not match the source" >&2; FAIL=1
  fi
done
exit "$FAIL"
