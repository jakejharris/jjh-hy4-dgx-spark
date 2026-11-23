#!/usr/bin/env bash
# Build the two image decoders the engine looks for beside itself, at the exact versions it accepts:
#   libturbojpeg.so.0    libjpeg-turbo 3.1.4.1 (the engine checks the version string inside the file)
#   libwebpdecoder.so.3  libwebp 1.6.0 (the engine checks WebPGetDecoderVersion)
#
#   bash serve/build_codecs.sh OUT_DIR [--src-cache DIR] [--target TRIPLE]
#
# Needs zig 0.17.0 (`zig cc`; or ZIG=/path/to/zig), tar and sha256sum, and curl unless --src-cache DIR holds both
# tarballs. Each tarball is checked against its pinned SHA-256, and the script says whether it came from DIR or the
# network. Writes OUT_DIR/libturbojpeg.so.0, OUT_DIR/libwebpdecoder.so.3 and OUT_DIR/SHA256SUMS. About 30 s.
# serve/build_release.sh runs it into OUT_DIR/lib. Nothing it builds is in this kit.
set -euo pipefail

OUT_DIR="./lib"
TARGET=""
SRC_CACHE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    --src-cache) SRC_CACHE="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,11p' "$0"
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
    *)
      OUT_DIR="$1"; shift
      ;;
  esac
done

ZIG_BIN="${ZIG:-zig}"
if ! command -v "$ZIG_BIN" >/dev/null 2>&1; then
  echo "ERROR: zig not found (set ZIG=/path/to/zig or add zig 0.17.0 to PATH)" >&2
  exit 2
fi

if [[ -z "$TARGET" ]]; then
  case "$(uname -m)" in
    aarch64|arm64) TARGET="aarch64-linux-gnu.2.28" ;;
    x86_64|amd64)  TARGET="x86_64-linux-gnu.2.28" ;;
    *) echo "ERROR: unsupported host arch $(uname -m); pass --target explicitly" >&2; exit 2 ;;
  esac
fi

LJT_VER="3.1.4.1"
LJT_TAR="libjpeg-turbo-${LJT_VER}.tar.gz"
LJT_URL="https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/${LJT_VER}/${LJT_TAR}"
LJT_SHA256="ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022"

WEBP_VER="1.6.0"
WEBP_TAR="libwebp-${WEBP_VER}.tar.gz"
WEBP_URL="https://storage.googleapis.com/downloads.webmproject.org/releases/webp/${WEBP_TAR}"
WEBP_SHA256="e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/tf-codecs-build.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fetch_verified() {
  local name="$1" url="$2" want="$3" dst="$WORK/$1"
  if [[ -n "$SRC_CACHE" && -f "$SRC_CACHE/$name" ]]; then
    cp "$SRC_CACHE/$name" "$dst"
    echo "   source: $name from the local folder $SRC_CACHE"
  else
    if [[ -n "$SRC_CACHE" ]]; then echo "   $name is not in $SRC_CACHE; fetching it"; fi
    curl -fsSL -o "$dst" "$url"
    echo "   source: $name from the network ($url)"
  fi
  local got
  got="$(sha256sum "$dst" | awk '{print $1}')"
  if [[ "$got" != "$want" ]]; then
    echo "ERROR: sha256 mismatch for $name: expected $want, got $got" >&2
    exit 1
  fi
  echo "   sha256 ${got} matches the pin"
}

# Compile jobs run eight at a time; a failed one stops the build (a plain `wait` would hide it).
pids=()
reap() {
  local p
  for p in "${pids[@]}"; do wait "$p" || { echo "ERROR: a compile step failed" >&2; exit 1; }; done
  pids=()
}

fetch_verified "$LJT_TAR" "$LJT_URL" "$LJT_SHA256"
fetch_verified "$WEBP_TAR" "$WEBP_URL" "$WEBP_SHA256"

tar -xzf "$WORK/$LJT_TAR" -C "$WORK"
tar -xzf "$WORK/$WEBP_TAR" -C "$WORK"

mkdir -p "$OUT_DIR"
OUT_ABS="$(cd "$OUT_DIR" && pwd)"

IS_AARCH64=0
if [[ "$TARGET" == aarch64-* ]]; then
  IS_AARCH64=1
fi

COMMON_CFLAGS=(
  -target "$TARGET"
  -O3 -DNDEBUG -fPIC
  "-ffile-prefix-map=${WORK}=/src"
  "-fdebug-prefix-map=${WORK}=/src"
)

# ---------------------------------------------------------------------------
# 1. libjpeg-turbo 3.1.4.1 -> libturbojpeg.so.0
# ---------------------------------------------------------------------------
LJT_SRC="$WORK/libjpeg-turbo-${LJT_VER}"
LJT_CFG="$WORK/ljt-cfg"
mkdir -p "$LJT_CFG/arm"

if [[ "$IS_AARCH64" -eq 1 ]]; then
  SIMD_DEF="#define WITH_SIMD 1"
else
  SIMD_DEF="/* scalar C on x86_64 when built without NASM; integer IDCT is bit-exact */"
fi

cat > "$LJT_CFG/jconfig.h" <<EOF
#define JPEG_LIB_VERSION 62
#define LIBJPEG_TURBO_VERSION 3.1.4.1
#define LIBJPEG_TURBO_VERSION_NUMBER 3001004
#define C_ARITH_CODING_SUPPORTED 1
#define D_ARITH_CODING_SUPPORTED 1
#define MEM_SRCDST_SUPPORTED 1
${SIMD_DEF}
#ifndef BITS_IN_JSAMPLE
#define BITS_IN_JSAMPLE 8
#endif
EOF

cat > "$LJT_CFG/jconfigint.h" <<EOF
#define BUILD "20260327"
#define HIDDEN __attribute__((visibility("hidden")))
#undef inline
#define INLINE __inline__ __attribute__((always_inline))
#define THREAD_LOCAL __thread
#define PACKAGE_NAME "libjpeg-turbo"
#define VERSION "3.1.4.1"
#define SIZEOF_SIZE_T 8
#define HAVE_BUILTIN_CTZL 1
#if defined(__has_attribute)
#if __has_attribute(fallthrough)
#define FALLTHROUGH __attribute__((fallthrough));
#else
#define FALLTHROUGH
#endif
#else
#define FALLTHROUGH
#endif
#ifndef BITS_IN_JSAMPLE
#define BITS_IN_JSAMPLE 8
#endif
#undef C_ARITH_CODING_SUPPORTED
#undef D_ARITH_CODING_SUPPORTED
#undef WITH_SIMD
#if BITS_IN_JSAMPLE == 8
#define C_ARITH_CODING_SUPPORTED 1
#define D_ARITH_CODING_SUPPORTED 1
${SIMD_DEF}
#endif
EOF

sed 's/@COPYRIGHT_YEAR@/1991-2026/g' "$LJT_SRC/src/jversion.h.in" > "$LJT_CFG/jversion.h"

LJT_FILES=(
  src/jcapimin.c
  src/wrapper/jcapistd-8.c src/wrapper/jcapistd-12.c src/wrapper/jcapistd-16.c
  src/wrapper/jccoefct-8.c src/wrapper/jccoefct-12.c
  src/wrapper/jccolor-8.c src/wrapper/jccolor-12.c src/wrapper/jccolor-16.c
  src/wrapper/jcdctmgr-8.c src/wrapper/jcdctmgr-12.c
  src/wrapper/jcdiffct-8.c src/wrapper/jcdiffct-12.c src/wrapper/jcdiffct-16.c
  src/jchuff.c src/jcicc.c src/jcinit.c src/jclhuff.c
  src/wrapper/jclossls-8.c src/wrapper/jclossls-12.c src/wrapper/jclossls-16.c
  src/wrapper/jcmainct-8.c src/wrapper/jcmainct-12.c src/wrapper/jcmainct-16.c
  src/jcmarker.c src/jcmaster.c src/jcomapi.c src/jcparam.c src/jcphuff.c
  src/wrapper/jcprepct-8.c src/wrapper/jcprepct-12.c src/wrapper/jcprepct-16.c
  src/wrapper/jcsample-8.c src/wrapper/jcsample-12.c src/wrapper/jcsample-16.c
  src/jctrans.c src/jdapimin.c
  src/wrapper/jdapistd-8.c src/wrapper/jdapistd-12.c src/wrapper/jdapistd-16.c
  src/jdatadst.c src/jdatasrc.c
  src/wrapper/jdcoefct-8.c src/wrapper/jdcoefct-12.c
  src/wrapper/jdcolor-8.c src/wrapper/jdcolor-12.c src/wrapper/jdcolor-16.c
  src/wrapper/jddctmgr-8.c src/wrapper/jddctmgr-12.c
  src/wrapper/jddiffct-8.c src/wrapper/jddiffct-12.c src/wrapper/jddiffct-16.c
  src/jdhuff.c src/jdicc.c src/jdinput.c src/jdlhuff.c
  src/wrapper/jdlossls-8.c src/wrapper/jdlossls-12.c src/wrapper/jdlossls-16.c
  src/wrapper/jdmainct-8.c src/wrapper/jdmainct-12.c src/wrapper/jdmainct-16.c
  src/jdmarker.c src/jdmaster.c
  src/wrapper/jdmerge-8.c src/wrapper/jdmerge-12.c
  src/jdphuff.c
  src/wrapper/jdpostct-8.c src/wrapper/jdpostct-12.c src/wrapper/jdpostct-16.c
  src/wrapper/jdsample-8.c src/wrapper/jdsample-12.c src/wrapper/jdsample-16.c
  src/jdtrans.c src/jerror.c src/jfdctflt.c
  src/wrapper/jfdctfst-8.c src/wrapper/jfdctfst-12.c
  src/wrapper/jfdctint-8.c src/wrapper/jfdctint-12.c
  src/wrapper/jidctflt-8.c src/wrapper/jidctflt-12.c
  src/wrapper/jidctfst-8.c src/wrapper/jidctfst-12.c
  src/wrapper/jidctint-8.c src/wrapper/jidctint-12.c
  src/wrapper/jidctred-8.c src/wrapper/jidctred-12.c
  src/jmemmgr.c src/jmemnobs.c src/jpeg_nbits.c
  src/wrapper/jquant1-8.c src/wrapper/jquant1-12.c
  src/wrapper/jquant2-8.c src/wrapper/jquant2-12.c
  src/wrapper/jutils-8.c src/wrapper/jutils-12.c src/wrapper/jutils-16.c
  src/jaricom.c src/jcarith.c src/jdarith.c
  src/turbojpeg.c src/transupp.c src/jdatadst-tj.c src/jdatasrc-tj.c
  src/rdbmp.c
  src/wrapper/rdppm-8.c src/wrapper/rdppm-12.c src/wrapper/rdppm-16.c
  src/wrbmp.c
  src/wrapper/wrppm-8.c src/wrapper/wrppm-12.c src/wrapper/wrppm-16.c
)

LJT_EXTRA_CFLAGS=(-I"$LJT_CFG" -I"$LJT_SRC/src" -DBMP_SUPPORTED -DPPM_SUPPORTED)
if [[ "$IS_AARCH64" -eq 1 ]]; then
  cat > "$LJT_CFG/arm/neon-compat.h" <<'EOF'
#define HAVE_VLD1_S16_X3 1
#define HAVE_VLD1_U16_X2 1
#define HAVE_VLD1Q_U8_X4 1
#define BUILTIN_CLZ(x) __builtin_clz(x)
#define BUILTIN_CLZLL(x) __builtin_clzll(x)
#define BUILTIN_BSWAP64(x) __builtin_bswap64(x)
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wdeclaration-after-statement"
#pragma clang diagnostic ignored "-Wc99-extensions"
#endif
EOF
  LJT_EXTRA_CFLAGS+=(-I"$LJT_CFG/arm" -I"$LJT_SRC/simd/arm" -I"$LJT_SRC/simd" -DNEON_INTRINSICS=1)
  LJT_FILES+=(
    simd/arm/jcgray-neon.c simd/arm/jcphuff-neon.c simd/arm/jcsample-neon.c
    simd/arm/jdmerge-neon.c simd/arm/jdsample-neon.c simd/arm/jfdctfst-neon.c
    simd/arm/jidctred-neon.c simd/arm/jquanti-neon.c
    simd/arm/jccolor-neon.c simd/arm/jidctint-neon.c simd/arm/jidctfst-neon.c
    simd/arm/aarch64/jchuff-neon.c simd/arm/jdcolor-neon.c simd/arm/jfdctint-neon.c
    simd/arm/aarch64/jsimd.c
  )
fi

LJT_OBJ="$WORK/ljt-obj"
mkdir -p "$LJT_OBJ"
(
  cd "$LJT_SRC"
  idx=0
  for src in "${LJT_FILES[@]}"; do
    idx=$((idx + 1))
    "$ZIG_BIN" cc "${COMMON_CFLAGS[@]}" "${LJT_EXTRA_CFLAGS[@]}" \
      -c "$src" -o "$LJT_OBJ/$(printf '%03d' "$idx")_$(basename "$src" .c).o" &
    pids+=("$!")
    if (( ${#pids[@]} == 8 )); then reap; fi
  done
  reap
  "$ZIG_BIN" cc "${COMMON_CFLAGS[@]}" -shared \
    -Wl,-soname,libturbojpeg.so.0 \
    "-Wl,--version-script,$LJT_SRC/src/turbojpeg-mapfile" \
    -Wl,-s \
    "$LJT_OBJ"/*.o -lc \
    -o "$OUT_ABS/libturbojpeg.so.0"
)

# ---------------------------------------------------------------------------
# 2. libwebp 1.6.0 -> libwebpdecoder.so.3
# ---------------------------------------------------------------------------
WEBP_SRC="$WORK/libwebp-${WEBP_VER}"
WEBP_OBJ="$WORK/webp-obj"
mkdir -p "$WEBP_OBJ"

WEBP_FILES=(
  src/dec/alpha_dec.c src/dec/buffer_dec.c src/dec/frame_dec.c
  src/dec/idec_dec.c src/dec/io_dec.c src/dec/quant_dec.c
  src/dec/tree_dec.c src/dec/vp8_dec.c src/dec/vp8l_dec.c src/dec/webp_dec.c
  src/dsp/alpha_processing.c src/dsp/alpha_processing_mips_dsp_r2.c
  src/dsp/alpha_processing_neon.c src/dsp/alpha_processing_sse2.c
  src/dsp/alpha_processing_sse41.c
  src/dsp/cpu.c src/dsp/dec.c src/dsp/dec_clip_tables.c
  src/dsp/dec_mips32.c src/dsp/dec_mips_dsp_r2.c src/dsp/dec_msa.c
  src/dsp/dec_neon.c src/dsp/dec_sse2.c src/dsp/dec_sse41.c
  src/dsp/filters.c src/dsp/filters_mips_dsp_r2.c src/dsp/filters_msa.c
  src/dsp/filters_neon.c src/dsp/filters_sse2.c
  src/dsp/lossless.c src/dsp/lossless_avx2.c src/dsp/lossless_mips_dsp_r2.c
  src/dsp/lossless_msa.c src/dsp/lossless_neon.c src/dsp/lossless_sse2.c
  src/dsp/lossless_sse41.c
  src/dsp/rescaler.c src/dsp/rescaler_mips32.c src/dsp/rescaler_mips_dsp_r2.c
  src/dsp/rescaler_msa.c src/dsp/rescaler_neon.c src/dsp/rescaler_sse2.c
  src/dsp/upsampling.c src/dsp/upsampling_mips_dsp_r2.c
  src/dsp/upsampling_msa.c src/dsp/upsampling_neon.c
  src/dsp/upsampling_sse2.c src/dsp/upsampling_sse41.c
  src/dsp/yuv.c src/dsp/yuv_mips32.c src/dsp/yuv_mips_dsp_r2.c
  src/dsp/yuv_neon.c src/dsp/yuv_sse2.c src/dsp/yuv_sse41.c
  src/utils/bit_reader_utils.c src/utils/color_cache_utils.c
  src/utils/filters_utils.c src/utils/huffman_utils.c src/utils/palette.c
  src/utils/quant_levels_dec_utils.c src/utils/random_utils.c
  src/utils/rescaler_utils.c src/utils/thread_utils.c src/utils/utils.c
)

WEBP_CFLAGS=(-I"$WEBP_SRC" -I"$WEBP_SRC/src" -DWEBP_USE_THREAD -fvisibility=hidden)
if [[ "$TARGET" == x86_64-* ]]; then
  WEBP_CFLAGS+=(-DWEBP_HAVE_SSE41 -DWEBP_HAVE_AVX2)
fi

(
  cd "$WEBP_SRC"
  idx=0
  for src in "${WEBP_FILES[@]}"; do
    idx=$((idx + 1))
    extra=()
    if [[ "$TARGET" == x86_64-* ]]; then
      case "$src" in
        *_sse41.c) extra=(-msse4.1) ;;
        *_avx2.c)  extra=(-mavx2) ;;
      esac
    fi
    "$ZIG_BIN" cc "${COMMON_CFLAGS[@]}" "${WEBP_CFLAGS[@]}" "${extra[@]}" \
      -c "$src" -o "$WEBP_OBJ/$(printf '%03d' "$idx")_$(basename "$src" .c).o" &
    pids+=("$!")
    if (( ${#pids[@]} == 8 )); then reap; fi
  done
  reap
  "$ZIG_BIN" cc "${COMMON_CFLAGS[@]}" -shared \
    -Wl,-soname,libwebpdecoder.so.3 \
    -Wl,-s \
    "$WEBP_OBJ"/*.o -lpthread -lm -lc \
    -o "$OUT_ABS/libwebpdecoder.so.3"
)

chmod 755 "$OUT_ABS/libturbojpeg.so.0" "$OUT_ABS/libwebpdecoder.so.3"
(cd "$OUT_ABS" && sha256sum libturbojpeg.so.0 libwebpdecoder.so.3 > SHA256SUMS && cat SHA256SUMS)
echo "   built for ${TARGET} into ${OUT_ABS}"
