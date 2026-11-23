#!/usr/bin/env bash
# Generate the public unified patch for our ExLlamaV3 Hy4 quantizer from the
# single immutable upstream release commit (v1.6.7 = 8c593e4) to <head-ref>,
# excluding local cluster farm launchers, watchers, and experiment harnesses.
#
# Usage:
#   EXL3_REPO=/path/to/exllamav3 bash quant/make_patches.sh [--check <clean-v1.6.7-dir>] [head-ref] [out-dir] [base-ref]
set -euo pipefail

UPSTREAM_TAG_COMMIT="8c593e4e5c88da76a4aaf8f1b1838bb7c1fce4e0"
UPSTREAM_TAG_TREE="6e0ee80353142ef8fb8056b0441237665e39db86"
DEFAULT_HEAD="3b697ca1f55a53ffce9a5d2b612146f33f4b20ca"
FARM_REMOVE_PATHS=(
  "farm/farm_state_watch.py"
  "farm/launch_hy4_farm.sh"
  "farm/launch_hy4_trunk.sh"
  "farm/patch_farm.py"
  "farm/recipe-trunk-mixed.example.yaml"
  "farm/status.sh"
  "farm/test_phase2_resume.py"
  "scripts/quantize_single_layer.py"
)

CHECK_DIR=""
POS_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)
      [[ $# -ge 2 ]] || { echo "error: --check requires a clean v1.6.7 checkout directory" >&2; exit 2; }
      CHECK_DIR="$2"
      shift 2
      ;;
    --check=*)
      CHECK_DIR="${1#--check=}"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--check <clean-v1.6.7-dir>] [head-ref] [out-dir] [base-ref]" >&2
      exit 0
      ;;
    *)
      POS_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POS_ARGS[@]} -gt 3 ]]; then
  echo "Usage: $0 [--check <clean-v1.6.7-dir>] [head-ref] [out-dir] [base-ref]" >&2
  exit 2
fi

HEAD_REF="${POS_ARGS[0]:-$DEFAULT_HEAD}"
OUT_DIR="${POS_ARGS[1]:-quant/patches}"
BASE_REF="${POS_ARGS[2]:-$UPSTREAM_TAG_COMMIT}"
REPO_DIR="${EXL3_REPO:-.}"

TMP_WORK="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_WORK"
}
trap cleanup EXIT

BASE_COMMIT="$(git -C "$REPO_DIR" rev-parse --verify "${BASE_REF}^{commit}")"
BASE_TREE="$(git -C "$REPO_DIR" rev-parse --verify "${BASE_COMMIT}^{tree}")"
BASE_DESC="${BASE_COMMIT:0:9} (tree ${BASE_TREE})"

TMP_IDX="$TMP_WORK/target.index"
GIT_INDEX_FILE="$TMP_IDX" git -C "$REPO_DIR" read-tree "$HEAD_REF"
for rel in "${FARM_REMOVE_PATHS[@]}"; do
  GIT_INDEX_FILE="$TMP_IDX" git -C "$REPO_DIR" update-index --force-remove "$rel"
done
TARGET_TREE="$(GIT_INDEX_FILE="$TMP_IDX" git -C "$REPO_DIR" write-tree)"

mkdir -p "$OUT_DIR"
rm -f "${OUT_DIR}"/*.patch
PATCH_PATH="${OUT_DIR}/0001-hy4-quant.patch"
git -C "$REPO_DIR" diff --binary --full-index "$BASE_TREE" "$TARGET_TREE" > "$PATCH_PATH"

echo "$PATCH_PATH"
echo "Wrote patch for ${BASE_DESC} -> ${HEAD_REF} (clean target tree ${TARGET_TREE})"

if [[ -n "$CHECK_DIR" ]]; then
  if ! git -C "$CHECK_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "error: --check path is not a git worktree: $CHECK_DIR" >&2
    exit 1
  fi
  if [[ -n "$(git -C "$CHECK_DIR" status --porcelain)" ]]; then
    echo "error: --check checkout is not clean (uncommitted or untracked files present): $CHECK_DIR" >&2
    exit 1
  fi
  CHECK_HEAD_COMMIT="$(git -C "$CHECK_DIR" rev-parse --verify HEAD)"
  CHECK_BASE_TREE="$(git -C "$CHECK_DIR" rev-parse --verify 'HEAD^{tree}')"
  if [[ "$CHECK_HEAD_COMMIT" != "$UPSTREAM_TAG_COMMIT" || "$CHECK_BASE_TREE" != "$UPSTREAM_TAG_TREE" ]]; then
    echo "error: --check requires a clean checkout of v1.6.7 ($UPSTREAM_TAG_COMMIT, tree $UPSTREAM_TAG_TREE), got commit $CHECK_HEAD_COMMIT (tree $CHECK_BASE_TREE)" >&2
    exit 1
  fi
  if [[ "$BASE_COMMIT" != "$UPSTREAM_TAG_COMMIT" || "$BASE_TREE" != "$UPSTREAM_TAG_TREE" ]]; then
    echo "error: --check refuses non-default base $BASE_COMMIT (expected $UPSTREAM_TAG_COMMIT)" >&2
    exit 1
  fi
  PATCH_ABS="$(cd "$(dirname "$PATCH_PATH")" && pwd)/$(basename "$PATCH_PATH")"
  git clone -q --shared --no-checkout "$CHECK_DIR" "$TMP_WORK/verify"
  git -C "$TMP_WORK/verify" checkout -q --detach "$UPSTREAM_TAG_COMMIT"
  git -C "$TMP_WORK/verify" apply --index "$PATCH_ABS"
  APPLIED_TREE="$(git -C "$TMP_WORK/verify" write-tree)"
  if [[ "$APPLIED_TREE" != "$TARGET_TREE" ]]; then
    echo "error: patched tree ($APPLIED_TREE) != expected clean target tree ($TARGET_TREE)" >&2
    exit 1
  fi
  echo "check_patch: PASS (base_commit=${CHECK_HEAD_COMMIT} base_tree=${CHECK_BASE_TREE} -> patched_tree=${APPLIED_TREE})"
fi
