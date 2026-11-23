#!/usr/bin/env bash
# Build the release engine from this kit's patch, on one DGX Spark, the way serve/BUILD.md pins it.
#
#   bash serve/build_release.sh --tf-dir DIR --out-dir DIR --zig-dir DIR [--self-test] [--codec-src DIR] [--no-codecs]
#
# Run it on the Spark that builds (from a copy of the kit on that Spark). It:
#   1. clones public TensorFold into --tf-dir if that folder has no checkout (TF_GIT, default the public repo),
#      checks it is commit 2cee7d9 with tree 2d7f953 and no local changes, and refuses anything else;
#   2. applies serve/patches/0001-hy4-port.patch (git apply --check first);
#   3. builds tp_mailbox.fatbin with nvcc, then the fatbins and the three binaries in the server's container,
#      pinned to Cortex-A725 cores 0 to 4 after clearing .zig-cache (serve/BUILD.md, "Pin the build");
#   4. checks the binary carries Cortex-A725 code (objdump ldapur count above 0);
#   5. writes OUT_DIR/bin/{tensorfold-native,tf-hy4-oracle,hy4-boundary,tp_mailbox.fatbin} and OUT_DIR/bin/SHA256SUMS;
#   6. builds the image decoders for JPEG and WebP into OUT_DIR/lib with serve/build_codecs.sh, from upstream
#      tarballs checked by SHA-256 (--codec-src DIR takes them from a local folder; --no-codecs skips this step).
# --self-test also runs tf-hy4-oracle --self-test, which uses the GPU. No step touches a running server.
# Stage the result on all four Sparks with launch/stage_binary.sh.
set -euo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TF_GIT="${TF_GIT:-https://github.com/ashhart/TensorFold.git}"
BASE="2cee7d938a5f72f89ce2062c69e705b4dd74c7ce"
BASE_TREE="2d7f95316e79f897f3b25de7f14011334eb00b97"
IMAGE="${BUILD_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
CUDA="${CUDA_DIR:-/usr/local/cuda-13.0}"
PATCH="${KIT}/serve/patches/0001-hy4-port.patch"
TF_DIR=""; OUT_DIR=""; ZIG_DIR=""; SELF_TEST=0; CODECS=1; CODEC_SRC=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tf-dir) TF_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --zig-dir) ZIG_DIR="$2"; shift 2 ;;
    --patch) PATCH="$2"; shift 2 ;;
    --self-test) SELF_TEST=1; shift ;;
    --codec-src) CODEC_SRC="$2"; shift 2 ;;
    --no-codecs) CODECS=0; shift ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "ERROR: unknown argument $1" >&2; exit 2 ;;
  esac
done
for v in TF_DIR OUT_DIR ZIG_DIR; do
  [[ -n "${!v}" && "${!v}" == /* ]] || { echo "ERROR: --${v,,} must be a full path" >&2; exit 2; }
done
TF_DIR="${TF_DIR%/}"; OUT_DIR="${OUT_DIR%/}"; ZIG_DIR="${ZIG_DIR%/}"
[[ "$OUT_DIR" == /*/*/* && "$TF_DIR" == /*/*/* ]] || { echo "ERROR: --out-dir and --tf-dir must be at least three folders deep (this script clears them)" >&2; exit 2; }
[[ -x "${ZIG_DIR}/zig" ]] || { echo "ERROR: no zig at ${ZIG_DIR}/zig (Zig 0.17.0, serve/BUILD.md)" >&2; exit 2; }
[[ -x "${CUDA}/bin/nvcc" ]] || { echo "ERROR: no nvcc at ${CUDA}/bin/nvcc (CUDA 13.0)" >&2; exit 2; }
[[ -f "$PATCH" ]] || { echo "ERROR: patch not found: $PATCH" >&2; exit 2; }
docker image inspect "$IMAGE" > /dev/null || { echo "ERROR: container image $IMAGE is not on this machine (docker pull it first)" >&2; exit 2; }

echo "== $(date -u +%FT%TZ) source: TensorFold ${BASE:0:7} + $(basename "$PATCH") (sha256 $(sha256sum "$PATCH" | cut -c1-12))"
if [[ ! -d "${TF_DIR}/.git" ]]; then
  git clone "$TF_GIT" "$TF_DIR"
fi
git -C "$TF_DIR" checkout -q "$BASE"
[[ "$(git -C "$TF_DIR" rev-parse 'HEAD^{tree}')" == "$BASE_TREE" ]] || { echo "ERROR: ${TF_DIR} is not tree ${BASE_TREE}" >&2; exit 2; }
if [[ -n "$(git -C "$TF_DIR" status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: ${TF_DIR} has local changes; use a clean checkout of ${BASE}" >&2; exit 2
fi
git -C "$TF_DIR" apply --check "$PATCH"
git -C "$TF_DIR" apply "$PATCH"
echo "   patch applied: $(git -C "$TF_DIR" status --porcelain | wc -l) paths changed"

rm -rf "${OUT_DIR:?}" "${TF_DIR:?}/zig-out" "${TF_DIR:?}/.zig-cache"
mkdir -p "${OUT_DIR}/bin" "${OUT_DIR}/fatbin"
T0=$(date +%s)
echo "== $(date -u +%FT%TZ) build in ${IMAGE}, cores 0 to 4, clean .zig-cache"
docker run --rm --pull=never --network none --memory=18g --ipc host --ulimit nofile=1048576 \
  --user "$(id -u):$(id -g)" \
  -v "${TF_DIR}:${TF_DIR}" -v "${OUT_DIR}:${OUT_DIR}" -v "${ZIG_DIR}:/opt/zig:ro" -v "${CUDA}:${CUDA}:ro" \
  -w "${TF_DIR}" -e ZIG_LOCAL_CACHE_DIR="${TF_DIR}/.zig-cache" -e ZIG_GLOBAL_CACHE_DIR="${OUT_DIR}/zig-global-cache" \
  --entrypoint /bin/bash "$IMAGE" -ec "
    '${CUDA}/bin/nvcc' -fatbin -O3 -std=c++17 \
      -gencode=arch=compute_120,code=sm_120 -gencode=arch=compute_121,code=sm_121 \
      -gencode=arch=compute_90,code=compute_90 \
      -o '${OUT_DIR}/fatbin/tp_mailbox.fatbin' '${TF_DIR}/zig/src/fabric/tp/mailbox.cu'
    taskset -c 0-4 /opt/zig/zig build fatbins hy4-oracle native hy4-tp-boundary-bin \
      -Dstrip=true -Doptimize=fast -Dnvcc='${CUDA}/bin/nvcc' -Dsm=121 \
      -Dtp-fatbin='${OUT_DIR}/fatbin/tp_mailbox.fatbin' -p '${OUT_DIR}' -j2 --summary all"
mv -f "${OUT_DIR}/native/bin/tensorfold-native" "${OUT_DIR}/bin/tensorfold-native"
rmdir "${OUT_DIR}/native/bin" "${OUT_DIR}/native" 2>/dev/null || true
[[ -f "${OUT_DIR}/hy4-boundary" ]] && mv -f "${OUT_DIR}/hy4-boundary" "${OUT_DIR}/bin/hy4-boundary"
cp -f "${OUT_DIR}/fatbin/tp_mailbox.fatbin" "${OUT_DIR}/bin/tp_mailbox.fatbin"
chmod +x "${OUT_DIR}/bin/tensorfold-native" "${OUT_DIR}/bin/tf-hy4-oracle" "${OUT_DIR}/bin/hy4-boundary"
echo "   build took $(( $(date +%s) - T0 )) s"

N="$(objdump -d "${OUT_DIR}/bin/tensorfold-native" | grep -c ldapur || true)"
if [[ "${N:-0}" -le 0 ]]; then
  echo "ERROR: tensorfold-native has no Cortex-A725 code (ldapur count ${N:-0}); rebuild pinned to cores 0 to 4" >&2; exit 1
fi
echo "   Cortex-A725 code present (ldapur count ${N})"
if [[ "$SELF_TEST" -eq 1 ]]; then
  "${OUT_DIR}/bin/tf-hy4-oracle" --self-test
fi
(cd "${OUT_DIR}/bin" && sha256sum tensorfold-native tf-hy4-oracle hy4-boundary tp_mailbox.fatbin > SHA256SUMS && cat SHA256SUMS)
if [[ "$CODECS" -eq 1 ]]; then
  echo "== $(date -u +%FT%TZ) image decoders into ${OUT_DIR}/lib (serve/build_codecs.sh)"
  ZIG="${ZIG_DIR}/zig" ZIG_GLOBAL_CACHE_DIR="${OUT_DIR}/zig-global-cache" \
    bash "${KIT}/serve/build_codecs.sh" "${OUT_DIR}/lib" ${CODEC_SRC:+--src-cache "$CODEC_SRC"}
else
  echo "   image decoders skipped (--no-codecs): this build serves PNG; JPEG and WebP get HTTP 400"
fi
stat -c '%s %n' "${OUT_DIR}/bin/tensorfold-native"
echo "== $(date -u +%FT%TZ) built: ${OUT_DIR}/bin"
