#!/usr/bin/env bash
# Generate the public unified patch for our TensorFold Hy4 port from the single
# immutable upstream base commit (main 2cee7d9) to <head-ref>, excluding the five
# local-only run-notice files so the patched tree matches our engine tree with
# upstream's docs intact.
#
# Usage:
#   TF_REPO=/path/to/tensorfold bash serve/make_patches.sh [--check <clean-2cee7d9-dir>] <head-ref> [out-dir] [base-ref]
set -euo pipefail

UPSTREAM_MAIN="2cee7d938a5f72f89ce2062c69e705b4dd74c7ce"
UPSTREAM_MAIN_TREE="2d7f95316e79f897f3b25de7f14011334eb00b97"
NOTICE_RESTORE_PATHS=(
  "AGENTS.md"
  "docs/atc/SKILL.md"
  "docs/pr/SKILL.md"
  "docs/recipes/SKILL.md"
)
NOTICE_REMOVE_PATHS=(
  "docs/upstream-AGENTS.md"
)

CHECK_DIR=""
POS_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)
      [[ $# -ge 2 ]] || { echo "error: --check requires a clean 2cee7d9 checkout directory" >&2; exit 2; }
      CHECK_DIR="$2"
      shift 2
      ;;
    --check=*)
      CHECK_DIR="${1#--check=}"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--check <clean-2cee7d9-dir>] <head-ref> [out-dir] [base-ref]" >&2
      exit 0
      ;;
    *)
      POS_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POS_ARGS[@]} -lt 1 || ${#POS_ARGS[@]} -gt 3 ]]; then
  echo "Usage: $0 [--check <clean-2cee7d9-dir>] <head-ref> [out-dir] [base-ref]" >&2
  exit 2
fi

HEAD_REF="${POS_ARGS[0]}"
OUT_DIR="${POS_ARGS[1]:-serve/patches}"
BASE_REF="${POS_ARGS[2]:-$UPSTREAM_MAIN}"
REPO_DIR="${TF_REPO:-.}"

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
for rel in "${NOTICE_RESTORE_PATHS[@]}"; do
  entry="$(git -C "$REPO_DIR" ls-tree "$UPSTREAM_MAIN" -- "$rel")"
  if [[ -n "$entry" ]]; then
    mode="$(awk '{print $1}' <<<"$entry")"
    blob="$(awk '{print $3}' <<<"$entry")"
    GIT_INDEX_FILE="$TMP_IDX" git -C "$REPO_DIR" update-index --cacheinfo "${mode},${blob},${rel}"
  fi
done
for rel in "${NOTICE_REMOVE_PATHS[@]}"; do
  GIT_INDEX_FILE="$TMP_IDX" git -C "$REPO_DIR" update-index --force-remove "$rel"
done
TARGET_TREE="$(GIT_INDEX_FILE="$TMP_IDX" git -C "$REPO_DIR" write-tree)"

mkdir -p "$OUT_DIR"
rm -f "${OUT_DIR}"/*.patch
PATCH_PATH="${OUT_DIR}/0001-hy4-port.patch"
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
  if [[ "$CHECK_HEAD_COMMIT" != "$UPSTREAM_MAIN" || "$CHECK_BASE_TREE" != "$UPSTREAM_MAIN_TREE" ]]; then
    echo "error: --check requires a clean checkout of $UPSTREAM_MAIN (tree $UPSTREAM_MAIN_TREE), got commit $CHECK_HEAD_COMMIT (tree $CHECK_BASE_TREE)" >&2
    exit 1
  fi
  if [[ "$BASE_COMMIT" != "$UPSTREAM_MAIN" || "$BASE_TREE" != "$UPSTREAM_MAIN_TREE" ]]; then
    echo "error: --check refuses non-default base $BASE_COMMIT (expected $UPSTREAM_MAIN)" >&2
    exit 1
  fi
  PATCH_ABS="$(cd "$(dirname "$PATCH_PATH")" && pwd)/$(basename "$PATCH_PATH")"
  git clone -q --shared --no-checkout "$CHECK_DIR" "$TMP_WORK/verify"
  git -C "$TMP_WORK/verify" checkout -q --detach "$UPSTREAM_MAIN"
  git -C "$TMP_WORK/verify" apply --index "$PATCH_ABS"
  APPLIED_TREE="$(git -C "$TMP_WORK/verify" write-tree)"
  if [[ "$APPLIED_TREE" != "$TARGET_TREE" ]]; then
    echo "error: patched tree ($APPLIED_TREE) != expected clean target tree ($TARGET_TREE)" >&2
    exit 1
  fi
  echo "check_patch: PASS (base_commit=${CHECK_HEAD_COMMIT} base_tree=${CHECK_BASE_TREE} -> patched_tree=${APPLIED_TREE})"
fi
