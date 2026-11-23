#!/usr/bin/env bash
# Launch 4-rank TP=4 TensorFold on the 4-node ring.
# Usage:
#   bash launch/launch.sh [--env-file launch/ring.env [--env-file launch/ring.relay.env]] [--timed] [-- command...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILES=()
TIMED_FLAG=""
CMD=()
_CALLER_TF_TP_KERNEL_IMAGE="${TF_TP_KERNEL_IMAGE:-}"
_CALLER_LOGDIR="${LOGDIR:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file) ENV_FILES+=("$2"); shift 2 ;;
    --timed) TIMED_FLAG="1"; shift ;;
    --) shift; CMD=("$@"); break ;;
    -h|--help)
      sed -n '2,5p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ ${#ENV_FILES[@]} -eq 0 ]]; then
  ENV_FILES=("${SCRIPT_DIR}/ring.env")
fi

for ef in "${ENV_FILES[@]}"; do
  if [[ ! -f "$ef" ]]; then
    echo "ERROR: $ef not found (copy launch/ring.env.example to launch/ring.env)" >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "$ef"
done
if [[ -n "$_CALLER_TF_TP_KERNEL_IMAGE" ]]; then
  TF_TP_KERNEL_IMAGE="$_CALLER_TF_TP_KERNEL_IMAGE"
fi
if [[ -n "$_CALLER_LOGDIR" ]]; then
  LOGDIR="$_CALLER_LOGDIR"
fi
TIMED_RUN="${TIMED_FLAG:-${TIMED_RUN:-0}}"

# Repeat option: the kit launches with it on. Empty or unset means "1"; "0" turns it off. The engine's own
# default is off, so the launcher always passes the value. "untimed:1" (or "repeat_stable:1") inside
# TF_HY4_ROW_COST_TABLE turns it on in the engine whatever this says, so "0" with that entry is refused.
TF_HY4_REPEAT_STABLE="${TF_HY4_REPEAT_STABLE:-1}"
case "$TF_HY4_REPEAT_STABLE" in
  0|1) ;;
  *) echo "ERROR: TF_HY4_REPEAT_STABLE=\"${TF_HY4_REPEAT_STABLE}\"; use \"1\" (on, the kit default) or \"0\" (off)." >&2
     exit 2 ;;
esac
if [[ "$TF_HY4_REPEAT_STABLE" == "0" ]]; then
  IFS=',' read -r -a _RCT_ENTRIES <<< "${TF_HY4_ROW_COST_TABLE:-}"
  for _e in ${_RCT_ENTRIES[@]+"${_RCT_ENTRIES[@]}"}; do
    [[ "$_e" == *:* ]] || continue
    _k="${_e%%:*}"; _k="${_k//[[:space:]]/}"
    _v="${_e#*:}"; _v="${_v//[[:space:]]/}"; _v="${_v,,}"
    if [[ ( "$_k" == "untimed" || "$_k" == "repeat_stable" ) && ( "$_v" == "1" || "$_v" == "true" || "$_v" == "on" ) ]]; then
      echo "ERROR: TF_HY4_REPEAT_STABLE=\"0\" but TF_HY4_ROW_COST_TABLE has \"${_e}\", which turns the option on." >&2
      echo "       Remove that entry from the table, or set TF_HY4_REPEAT_STABLE=\"1\"." >&2
      exit 2
    fi
  done
fi

# Drafts: on in this kit. Empty or unset means "1"; "0" turns them off (--no-drafts on every rank).
SERVE_DRAFTS="${SERVE_DRAFTS:-1}"
case "$SERVE_DRAFTS" in
  0|1) ;;
  *) echo "ERROR: SERVE_DRAFTS=\"${SERVE_DRAFTS}\"; use \"1\" (drafts on, the kit default) or \"0\" (off)." >&2
     exit 2 ;;
esac

# Thinking budget: empty in this kit, so no flag is passed and the command line is the one measured. A whole
# number N passes --thinking-budget N on every rank: a request that reasons and sends no non-zero
# "thinking_budget" of its own has its reasoning closed after N tokens. max_tokens still counts both.
SERVE_THINKING_BUDGET="${SERVE_THINKING_BUDGET:-}"
if [[ -n "$SERVE_THINKING_BUDGET" && ! "$SERVE_THINKING_BUDGET" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: SERVE_THINKING_BUDGET=\"${SERVE_THINKING_BUDGET}\"; leave it empty (no budget, the kit default) or set a whole number of tokens above 0, for example \"4096\"." >&2
  exit 2
fi

# Temperature: 0.7 in this kit. With the line missing or commented out, every rank gets --temperature 0.7, so a
# request that sends no "temperature" is sampled at 0.7; top_p 1 and top_k -1 still come from Tencent's
# generation_config.json, which is not changed. A request's own "temperature" wins, and 0 is greedy. Set it
# empty (SERVE_TEMPERATURE="") to pass no flag, so that file decides (0.9); "0" makes greedy the default.
SERVE_TEMPERATURE="${SERVE_TEMPERATURE-0.7}"
if [[ -n "$SERVE_TEMPERATURE" && ! "$SERVE_TEMPERATURE" =~ ^([0-9]+|[0-9]*\.[0-9]+)$ ]]; then
  echo "ERROR: SERVE_TEMPERATURE=\"${SERVE_TEMPERATURE}\"; use a number of 0 or more, for example \"0.7\" (the kit default) or \"0\" (greedy by default), or set it empty to pass no flag (Tencent's 0.9)." >&2
  exit 2
fi

# Served model name: empty in this kit, so no flag is passed and the server answers to the model folder's last
# part. A name passes --name NAME on every rank: letters, digits, dot, dash and underscore, starting with a
# letter or digit.
SERVE_MODEL_NAME="${SERVE_MODEL_NAME:-}"
if [[ -n "$SERVE_MODEL_NAME" && ! "$SERVE_MODEL_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
  echo "ERROR: SERVE_MODEL_NAME=\"${SERVE_MODEL_NAME}\"; leave it empty (the model folder's name, the kit default) or use letters, digits, dot, dash and underscore, starting with a letter or digit, at most 128 characters." >&2
  exit 2
fi

# Fast prefill: on, with the four Sparks' partial sums added in FP32. This is the one place the kit sets that
# default. TF_HY4_FAST_PREFILL="0" gives the last release's prefill path; TF_HY4_PREFILL_AR="bf16" is the
# opt-in, faster with lower agreement (launch/SETTINGS.md, "TF_HY4_FAST_PREFILL" and "TF_HY4_PREFILL_AR").
TF_HY4_FAST_PREFILL="${TF_HY4_FAST_PREFILL:-1}"
TF_HY4_PREFILL_AR="${TF_HY4_PREFILL_AR:-fp32}"
TF_HY4_FAST_PREFILL_MIN_ROWS="${TF_HY4_FAST_PREFILL_MIN_ROWS:-}"
TF_HY4_PREFILL_TIMERS="${TF_HY4_PREFILL_TIMERS:-}"
case "$TF_HY4_FAST_PREFILL" in
  0|1) ;;
  *) echo "ERROR: TF_HY4_FAST_PREFILL=\"${TF_HY4_FAST_PREFILL}\"; use \"1\" (on, the kit default) or \"0\" (the last release's prefill path)." >&2
     exit 2 ;;
esac
case "$TF_HY4_PREFILL_AR" in
  fp32|bf16|off) ;;
  *) echo "ERROR: TF_HY4_PREFILL_AR=\"${TF_HY4_PREFILL_AR}\"; use \"fp32\" (the kit default), \"bf16\" (faster, lower agreement) or \"off\"." >&2
     exit 2 ;;
esac
if [[ -n "$TF_HY4_FAST_PREFILL_MIN_ROWS" && ! "$TF_HY4_FAST_PREFILL_MIN_ROWS" =~ ^[1-9][0-9]{0,6}$ ]]; then
  echo "ERROR: TF_HY4_FAST_PREFILL_MIN_ROWS=\"${TF_HY4_FAST_PREFILL_MIN_ROWS}\"; leave it empty (the engine's 256) or give a whole number of prompt rows." >&2
  exit 2
fi

# Vision: off unless SERVE_VISION="1". Then every rank gets --vision, and rank 0 alone reads the vision folder
# TF_HY4_VISION_DIR (vit-bf16/, built by tools/vit_folder.py). launch/preflight.sh checks that folder against
# launch/vision.sha256 before anything starts.
SERVE_VISION="${SERVE_VISION:-0}"
SERVE_VISION_TOKENS="${SERVE_VISION_TOKENS:-}"
TF_HY4_VISION_DIR="${TF_HY4_VISION_DIR:-}"
TF_HY4_VISION_DIR="${TF_HY4_VISION_DIR%/}"
case "$SERVE_VISION" in
  0|1) ;;
  *) echo "ERROR: SERVE_VISION=\"${SERVE_VISION}\"; use \"0\" (off, the kit default) or \"1\" (serve images)." >&2
     exit 2 ;;
esac
if [[ "$SERVE_VISION" == "1" && "$TF_HY4_VISION_DIR" != /* ]]; then
  echo "ERROR: SERVE_VISION=\"1\" but TF_HY4_VISION_DIR=\"${TF_HY4_VISION_DIR}\". Set it to the full path of the vision folder (vit-bf16/) on rank 0." >&2
  exit 2
fi
if [[ -n "$SERVE_VISION_TOKENS" && ! "$SERVE_VISION_TOKENS" =~ ^[1-9][0-9]{0,6}$ ]]; then
  echo "ERROR: SERVE_VISION_TOKENS=\"${SERVE_VISION_TOKENS}\"; leave it empty (the engine's default) or give a whole number of image tokens." >&2
  exit 2
fi

# Per-Spark parts: empty in this kit, so the full pack is served. A folder that holds, on each Spark, common/ and
# that Spark's own rankN/ (N is its place in RING_HOSTS, from 0), as tools/split_ranks.sh cuts them. The server
# then reads ${PARTS_ROOT}/common, so MODEL_DIR must name it (launch/ring.parts.env.example sets both).
PARTS_ROOT="${PARTS_ROOT:-}"
PARTS_ROOT="${PARTS_ROOT%/}"
if [[ -n "$PARTS_ROOT" ]]; then
  if [[ "$PARTS_ROOT" != /* ]]; then
    echo "ERROR: PARTS_ROOT=\"${PARTS_ROOT}\"; give the full path from /, the same on every Spark." >&2
    exit 2
  fi
  _md="${MODEL_DIR:-}"; _md="${_md//\/\//\/}"; _md="${_md%/}"
  if [[ "$_md" != "${PARTS_ROOT}/common" ]]; then
    echo "ERROR: PARTS_ROOT=\"${PARTS_ROOT}\" but MODEL_DIR=\"${MODEL_DIR:-}\". With per-Spark parts the server reads ${PARTS_ROOT}/common:" >&2
    echo "       set MODEL_DIR=\"\${PARTS_ROOT}/common\" (launch/ring.parts.env.example does)." >&2
    exit 2
  fi
  if [[ -z "${TF_HY4_TRUNK:-}" ]]; then
    echo "ERROR: PARTS_ROOT is set and TF_HY4_TRUNK is empty. The engine would then serve whatever trunk the parts were cut with," >&2
    echo "       and the launch record would say none. Set TF_HY4_TRUNK to the width the parts were cut with (tools/split_ranks.sh --trunk;" >&2
    echo "       4b6b for the kit's default), or \"bf16\" for parts cut lossless. The engine refuses a width that differs from the parts." >&2
    exit 2
  fi
  if [[ "$SERVE_DRAFTS" == "1" && -z "${DRAFTER_DIR:-}" && ( -n "${TF_HY4_DRAFTER_EXL3:-}" || -n "${TF_HY4_DRAFTER_HEAD:-}" ) ]]; then
    echo "ERROR: the parts carry the BF16 drafter only, and TF_HY4_DRAFTER_EXL3=\"${TF_HY4_DRAFTER_EXL3:-}\" TF_HY4_DRAFTER_HEAD=\"${TF_HY4_DRAFTER_HEAD:-}\" ask for packed drafter files." >&2
    echo "       Put a drafter folder on every Spark (tools/drafter_folder.sh; launch/STAGING.md, \"Per-Spark parts\") and set DRAFTER_DIR to it," >&2
    echo "       or set both drafter settings to \"\" to draft with the BF16 drafter from the parts." >&2
    exit 2
  fi
fi

# Preflight reads one settings file. With overlays, give it all of them merged in the order they were
# sourced here, so it checks the settings that will actually run.
ENV_FILE="${ENV_FILES[0]}"
if [[ ${#ENV_FILES[@]} -gt 1 ]]; then
  MERGED_ENV="$(mktemp "${TMPDIR:-/tmp}/hy4-ring-merged.XXXXXX")"
  trap 'rm -f "$MERGED_ENV"' EXIT
  for ef in "${ENV_FILES[@]}"; do
    printf '# from %s\n' "$ef"
    cat "$ef"
    printf '\n'
  done > "$MERGED_ENV"
  ENV_FILE="$MERGED_ENV"
fi

# The trunk width must be set, and its folders staged on every Spark.
if [[ "${TF_HY4_TRUNK:-}" == "PICK" ]]; then
  echo "ERROR: TF_HY4_TRUNK is \"PICK\". Set it to the full quant's width (4b, 4b6b or 6b; card claim build.full.trunk)," >&2
  echo "       or add --env-file launch/ring.lossless.env.example to serve the lossless trunk." >&2
  exit 2
fi
NEED_DIRS=()
NEED_DRAFTER_DIRS=()  # parts only: the packed drafter's folders, under DRAFTER_DIR
if [[ -z "$PARTS_ROOT" ]]; then
  case "${TF_HY4_TRUNK:-}" in
    "") ;;
    4b) NEED_DIRS+=(trunk-4b) ;;
    6b) NEED_DIRS+=(trunk-6b) ;;
    4b6b) NEED_DIRS+=(trunk-4b trunk-6b) ;;
    *) echo "NOTE: TF_HY4_TRUNK=\"${TF_HY4_TRUNK}\" is not 4b, 4b6b or 6b; its folders are not checked" >&2 ;;
  esac
fi
# With parts there is no trunk folder to check: the parts record the width they were cut with, and the engine
# refuses a TF_HY4_TRUNK that names another one.
if [[ "$SERVE_DRAFTS" == "1" ]]; then
  for w in "${TF_HY4_DRAFTER_EXL3:-}" "${TF_HY4_DRAFTER_HEAD:-}"; do
    case "$w" in
      4b|6b)
        if [[ -n "$PARTS_ROOT" ]]; then
          [[ " ${NEED_DRAFTER_DIRS[*]} " == *" trunk-${w} "* ]] || NEED_DRAFTER_DIRS+=("trunk-${w}")
        else
          [[ " ${NEED_DIRS[*]} " == *" trunk-${w} "* ]] || NEED_DIRS+=("trunk-${w}")
        fi ;;
    esac
  done
fi
read -r -a GUARD_HOSTS <<< "$RING_HOSTS"
GUARD_FAIL=0
for h in "${GUARD_HOSTS[@]}"; do
  for d in "${NEED_DIRS[@]}"; do
    if ! ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$h" "test -f '${MODEL_DIR}/${d}/trunk.index.json'"; then
      echo "ERROR: ${h}: ${MODEL_DIR}/${d}/trunk.index.json is missing (stage it: python3 launch/stage_trunk.py --env-file ...)" >&2
      GUARD_FAIL=1
    fi
  done
  for d in "${NEED_DRAFTER_DIRS[@]}"; do
    if ! ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$h" "cd '${DRAFTER_DIR}' && test -f config.json && test -f model-00245-of-00245.safetensors && test -f ${d}/dfly_layers.safetensors && test -f ${d}/lm_head.safetensors && test -f ${d}/trunk.index.json"; then
      echo "ERROR: ${h}: DRAFTER_DIR ${DRAFTER_DIR} lacks one of config.json, model-00245-of-00245.safetensors, ${d}/dfly_layers.safetensors," >&2
      echo "       ${d}/lm_head.safetensors, ${d}/trunk.index.json. Build it: bash tools/drafter_folder.sh FULL_PACK ${DRAFTER_DIR} --width ${d#trunk-}" >&2
      GUARD_FAIL=1
    fi
  done
done

# Parts: each Spark must hold common/ and its own rankN/, and no other rank's folder. The engine reads every
# rankN/ beside common/ and stops on one cut from another source pack, and a Spark needs only its own.
if [[ -n "$PARTS_ROOT" ]]; then
  for r in "${!GUARD_HOSTS[@]}"; do
    h="${GUARD_HOSTS[$r]}"
    others=""
    for o in 0 1 2 3; do
      [[ "$o" == "$r" ]] || others+=" rank${o}"
    done
    if ! extra="$(ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$h" \
      "cd '${PARTS_ROOT}' && test -f common/model.safetensors.index.json && test -f rank${r}/index.json && for d in${others}; do if test -e \"\$d\"; then echo \"\$d\"; fi; done")"; then
      echo "ERROR: ${h} (rank ${r}): ${PARTS_ROOT}/common/model.safetensors.index.json or ${PARTS_ROOT}/rank${r}/index.json is missing." >&2
      echo "       Copy common/ and rank${r}/ there (launch/STAGING.md, \"Per-Spark parts\"); rank ${r} is ${h}'s place in RING_HOSTS." >&2
      GUARD_FAIL=1
    elif [[ -n "$extra" ]]; then
      echo "ERROR: ${h} (rank ${r}): ${PARTS_ROOT} also holds $(echo $extra). The engine reads every rankN/ beside common/;" >&2
      echo "       keep only common/ and rank${r}/ on this Spark." >&2
      GUARD_FAIL=1
    fi
  done
fi

# Drafts on needs a drafter: a DRAFTER_DIR on every Spark, or the inline drafter staged with the layers
# (STAGE_DRAFTER="1" leaves dfly_num_hidden_layers above 0 in the staged config.json). Without one the
# engine would fall back to a different draft source than the one this recipe measured.
DRAFTS_HINT="Drafts are on by default (SERVE_DRAFTS empty or missing means \"1\"). Stage the drafter with STAGE_DRAFTER=\"1\" (launch/stage_layers.py, the default), or point DRAFTER_DIR at one, or set SERVE_DRAFTS=\"0\" to serve without drafts."
if [[ "$SERVE_DRAFTS" == "1" && ${#CMD[@]} -eq 0 ]]; then
  for h in "${GUARD_HOSTS[@]}"; do
    if [[ -n "${DRAFTER_DIR:-}" ]]; then
      if ! ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$h" "test -d '${DRAFTER_DIR}'"; then
        echo "ERROR: ${h}: drafts are on but DRAFTER_DIR ${DRAFTER_DIR} is not found." >&2
        echo "       ${DRAFTS_HINT}" >&2
        GUARD_FAIL=1
      fi
    else
      n_dfly="$(ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$h" \
        "grep -o '\"dfly_num_hidden_layers\"[[:space:]]*:[[:space:]]*[0-9]*' '${MODEL_DIR}/config.json' | grep -o '[0-9]*\$' | sort -n | tail -n 1" || true)"
      if [[ -z "$n_dfly" || "$n_dfly" == "0" ]]; then
        echo "ERROR: ${h}: drafts are on but no drafter is staged (dfly_num_hidden_layers is ${n_dfly:-absent} in ${MODEL_DIR}/config.json)." >&2
        echo "       ${DRAFTS_HINT}" >&2
        GUARD_FAIL=1
      fi
    fi
  done
fi

# The relay reads its GPU mailbox kernel from the file TF_TP_KERNEL_IMAGE names, on every Spark (serve and
# command mode alike). serve/BUILD.md step 2 builds it; launch/ring.relay.env.example defaults to
# /opt/tensorfold/images/tp_mailbox.fatbin.
if [[ "${TF_TP_ROCE_RELAY:-}" == "1" ]]; then
  for h in "${GUARD_HOSTS[@]}"; do
    if [[ -z "${TF_TP_KERNEL_IMAGE:-}" ]] || ! ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$h" "test -f '${TF_TP_KERNEL_IMAGE}'"; then
      echo "ERROR: ${h}: the relay is on but its kernel image TF_TP_KERNEL_IMAGE=\"${TF_TP_KERNEL_IMAGE:-}\" is not found." >&2
      echo "       Copy tp_mailbox.fatbin from the engine build (serve/BUILD.md step 2) to that path on every Spark, or set TF_TP_KERNEL_IMAGE in launch/ring.env." >&2
      GUARD_FAIL=1
    fi
  done
fi
[[ "$GUARD_FAIL" -eq 0 ]] || exit 2

if [[ "$TIMED_RUN" == "1" ]]; then
  bash "${SCRIPT_DIR}/preflight.sh" "$ENV_FILE" --timed
else
  bash "${SCRIPT_DIR}/preflight.sh" "$ENV_FILE"
fi

SERVE_MODE=0
if [[ ${#CMD[@]} -eq 0 ]]; then
  SERVE_MODE=1
fi

read -r -a HOSTS <<< "$RING_HOSTS"
LOGDIR="${LOGDIR:-/tmp/hy4-tp4-logs}"
PREFIX="${PREFIX:-hy4-tp4}"
TIMEOUT_S="${TIMEOUT_S:-900}"
TF_HY4_TP_TIMEOUT_S="${TF_HY4_TP_TIMEOUT_S:-900}"
TF_HY4_LOAD_TIMEOUT_S="${TF_HY4_LOAD_TIMEOUT_S:-900}"
TENSORFOLD_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-}"
TENSORFOLD_CUDA_MEMORY_LIMIT_GB="${TENSORFOLD_CUDA_MEMORY_LIMIT_GB:-}"
TF_HY4_MEM_FLOOR_GIB="${TF_HY4_MEM_FLOOR_GIB:-}"
# Trunk, drafter, relay, graph, and A/B bench switches (real names from wt-trunk, wt-kernels, wt-family-dfly)
TF_HY4_TRUNK="${TF_HY4_TRUNK:-}"
HY4_TRUNK_FAST="${HY4_TRUNK_FAST:-}"
DRAFTER_DIR="${DRAFTER_DIR:-}"
TF_HY4_DRAFTER_EXL3="${TF_HY4_DRAFTER_EXL3:-}"
TF_HY4_DRAFTER_HEAD="${TF_HY4_DRAFTER_HEAD:-}"
TF_HY4_TARGET_HEAD="${TF_HY4_TARGET_HEAD:-}"
TF_HY4_IHC_SPLIT="${TF_HY4_IHC_SPLIT:-}"
TF_HY4_HEAD_BF16="${TF_HY4_HEAD_BF16:-}"
TF_HY4_DRAFT_MIN_MARGIN="${TF_HY4_DRAFT_MIN_MARGIN:-}"
TF_HY4_ROW_COST_TABLE="${TF_HY4_ROW_COST_TABLE:-}"
TF_HY4_REPEAT_STABLE="${TF_HY4_REPEAT_STABLE:-1}"  # set and checked above; on unless "0"
TF_TP_ROCE_RELAY="${TF_TP_ROCE_RELAY:-}"
TF_TP_ROCE_RELAY_MAX_KB="${TF_TP_ROCE_RELAY_MAX_KB:-}"
TF_TP_KERNEL_IMAGE="${TF_TP_KERNEL_IMAGE:-}"
HY4_TP_DIRECT_WRITE="${HY4_TP_DIRECT_WRITE:-}"
TF_TP_ROCE_FAST="${TF_TP_ROCE_FAST:-}"
TF_TP_ROCE_BLOCKS="${TF_TP_ROCE_BLOCKS:-}"
HY4_DECODE_GRAPH="${HY4_DECODE_GRAPH:-}"
HY4_ROUTED_FAST="${HY4_ROUTED_FAST:-}"
TF_HY4_INDEX_PAGE_2K="${TF_HY4_INDEX_PAGE_2K:-}"
TF_HY4_GREEDY_FULL_GATHER="${TF_HY4_GREEDY_FULL_GATHER:-}"
NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-}"
SERVE_PARALLEL="${SERVE_PARALLEL:-1}"
SERVE_DRAFTS="${SERVE_DRAFTS:-1}"  # set and checked above; on unless "0"
SERVE_THINKING_BUDGET="${SERVE_THINKING_BUDGET:-}"  # set and checked above; empty passes no flag
SERVE_TEMPERATURE="${SERVE_TEMPERATURE-0.7}"  # set and checked above; 0.7 when unset, empty passes no flag
SERVE_MODEL_NAME="${SERVE_MODEL_NAME:-}"  # set and checked above; empty passes no flag
# The container sees the model folder; with parts, the whole PARTS_ROOT, so rankN/ beside common/ is visible.
MODEL_MOUNT="${MODEL_DIR}"
PARTS_LINKS=""
if [[ -n "$PARTS_ROOT" ]]; then
  MODEL_MOUNT="${PARTS_ROOT}"
  # A weight file under rankN/ (or the drafter folder's trunk-*/) that is a symlink gets its target folder mounted.
  PARTS_LINKS=" '${PARTS_ROOT}'/rank*/*.safetensors '${PARTS_ROOT}'/rank*/trunk-*/*.safetensors"
  [[ -z "${DRAFTER_DIR:-}" ]] || PARTS_LINKS+=" '${DRAFTER_DIR}'/trunk-*/*.safetensors"
fi
MEMFLOOR_GIB="${MEMFLOOR_GIB:-8}"
REQUIRE_IB_VAL="0"
if [[ "${NCCL_NET:-}" == "IB" ]]; then
  REQUIRE_IB_VAL="1"
fi
mkdir -p "$LOGDIR"
rm -f "$LOGDIR"/rank*.log "$LOGDIR"/rank*.rc "$LOGDIR"/memfloor_r*.log "$LOGDIR"/launch_meta.json

cleanup_all() {
  for r in 0 1 2 3; do
    ssh -n -o BatchMode=yes -o ConnectTimeout=5 "${HOSTS[$r]}" \
      "docker rm -f '${PREFIX}-r${r}' >/dev/null 2>&1 || true" >/dev/null 2>&1 &
  done
  wait || true
}
trap 'cleanup_all; exit 0' INT TERM
cleanup_all

CMD_QUOTED=""
for arg in "${CMD[@]}"; do
  CMD_QUOTED+=" $(printf '%q' "$arg")"
done

# Receipt of what runs: the SHA-256 of the tensorfold-native each rank's container mounts, read on that Spark,
# and one graphics-clock snapshot per rank (nvidia-smi clocks.gr) now and again when the server is ready.
# These are single readings, not a trace. GPU_CLOCK_CAP_MHZ (empty: stock clocks) and ENGINE_COMMIT are what
# the operator declares; launch_meta.json and launch/collect_receipts.py keep them beside the measured values.
GPU_CLOCK_CAP_MHZ="${GPU_CLOCK_CAP_MHZ:-}"
ENGINE_COMMIT="${ENGINE_COMMIT:-}"
# The reads and their failure rules live in launch/receipt_reads.sh (host test: launch/test_receipt_reads.sh).
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/receipt_reads.sh"
CLK_READY=()
if ! read_receipts_at_launch; then
  echo "ERROR: receipt reads failed (above); not launching." >&2
  exit 2
fi

PIDS=()
for r in 0 1 2 3; do
  h="${HOSTS[$r]}"
  cname="${PREFIX}-r${r}"
  logf="${LOGDIR}/rank${r}_${h}.log"

  if [[ "$SERVE_MODE" -eq 1 ]]; then
    DRAFT_ARGS="--no-drafts"
    if [[ "$SERVE_DRAFTS" == "1" ]]; then
      DRAFT_ARGS=""
      if [[ -n "$DRAFTER_DIR" ]]; then
        DRAFT_ARGS=$(printf -- '--drafter %q' "${DRAFTER_DIR}")
      fi
    fi
    # Empty budget: nothing is added.
    if [[ -n "$SERVE_THINKING_BUDGET" ]]; then
      DRAFT_ARGS+="${DRAFT_ARGS:+ }$(printf -- '--thinking-budget %q' "${SERVE_THINKING_BUDGET}")"
    fi
    # Temperature: --temperature on every rank (0.7 unless set); empty adds nothing.
    if [[ -n "$SERVE_TEMPERATURE" ]]; then
      DRAFT_ARGS+="${DRAFT_ARGS:+ }$(printf -- '--temperature %q' "${SERVE_TEMPERATURE}")"
    fi
    if [[ -n "$SERVE_MODEL_NAME" ]]; then
      DRAFT_ARGS+="${DRAFT_ARGS:+ }$(printf -- '--name %q' "${SERVE_MODEL_NAME}")"
    fi
    # Vision off: nothing is added. On: --vision on every rank, the image-token cap only if one is set.
    if [[ "$SERVE_VISION" == "1" ]]; then
      DRAFT_ARGS+="${DRAFT_ARGS:+ }--vision"
      if [[ -n "$SERVE_VISION_TOKENS" ]]; then
        DRAFT_ARGS+=" $(printf -- '--vision-image-tokens %q' "${SERVE_VISION_TOKENS}")"
      fi
    fi
    RANK_CMD=$(printf '%q serve %q --backend cuda --tp 4 --rank %d --master %q --master-port %q --host %q --port %q --context %q --parallel %q %s --no-update-check' \
      "${TENSORFOLD_BIN}" "${MODEL_DIR}" "${r}" "${MASTER_ADDR}" "${MASTER_PORT}" \
      "${SERVE_HOST}" "${SERVE_PORT}" "${CONTEXT_LEN}" "${SERVE_PARALLEL}" "${DRAFT_ARGS}")
  else
    RANK_CMD="${CMD_QUOTED}"
  fi
  # Only rank 0 sees the vision folder.
  R0_VIS_DIR=""
  if [[ "$r" -eq 0 && "$SERVE_VISION" == "1" ]]; then
    R0_VIS_DIR="$TF_HY4_VISION_DIR"
  fi
  # The image decoders sit beside the binary (launch/stage_binary.sh copies them there), where the engine looks.
  TF_BIN_DIR="$(dirname "${TENSORFOLD_BIN:-/none/tensorfold-native}")"

  remote_cmd=$(cat <<EOF
set -o pipefail
docker rm -f '${cname}' >/dev/null 2>&1 || true
EXTRA_MOUNTS=()
if [[ -f '${TENSORFOLD_BIN}' ]]; then
  EXTRA_MOUNTS+=(-v '${TENSORFOLD_BIN}:${TENSORFOLD_BIN}:ro')
fi
for codec_lib in '${TF_BIN_DIR}/libturbojpeg.so.0' '${TF_BIN_DIR}/libwebpdecoder.so.3'; do
  if [[ -f '${TENSORFOLD_BIN}' && -f "\${codec_lib}" ]]; then EXTRA_MOUNTS+=(-v "\${codec_lib}:\${codec_lib}:ro"); fi
done
if [[ -n '${ORACLE_BIN:-}' && -f '${ORACLE_BIN:-}' ]]; then
  EXTRA_MOUNTS+=(-v '${ORACLE_BIN:-}:${ORACLE_BIN:-}:ro')
fi
if [[ -n '${TF_TP_KERNEL_IMAGE}' && -f '${TF_TP_KERNEL_IMAGE}' ]]; then
  EXTRA_MOUNTS+=(-v '${TF_TP_KERNEL_IMAGE}:${TF_TP_KERNEL_IMAGE}:ro')
fi
if [[ -n '${DRAFTER_DIR}' && -d '${DRAFTER_DIR}' && '${DRAFTER_DIR}' != '${MODEL_DIR}' ]]; then
  EXTRA_MOUNTS+=(-v '${DRAFTER_DIR}:${DRAFTER_DIR}:ro')
fi
if [[ -n '${R0_VIS_DIR}' && -d '${R0_VIS_DIR}' ]]; then
  EXTRA_MOUNTS+=(-v '${R0_VIS_DIR}:${R0_VIS_DIR}:ro')
fi
SEEN_DIRS=":${MODEL_DIR}:${DRAFTER_DIR}:${R0_VIS_DIR}:"
for ext_val in '${TF_HY4_DRAFTER_EXL3}' '${TF_HY4_DRAFTER_HEAD}' '${TF_HY4_TARGET_HEAD}'; do
  [[ -z "\$ext_val" ]] && continue
  ext_path="\${ext_val#*=}"
  if [[ -d "\$ext_path" ]]; then
    for cand in "\$ext_path" "\$(readlink -f "\$ext_path")"; do
      if [[ -d "\$cand" && "\$SEEN_DIRS" != *":\${cand}:"* ]]; then
        SEEN_DIRS="\${SEEN_DIRS}\${cand}:"
        EXTRA_MOUNTS+=(-v "\${cand}:\${cand}:ro")
      fi
    done
    for sf in "\$ext_path"/*.safetensors; do
      if [[ -L "\$sf" ]]; then
        tdir="\$(dirname "\$(readlink -f "\$sf")")"
        if [[ "\$SEEN_DIRS" != *":\${tdir}:"* ]]; then
          SEEN_DIRS="\${SEEN_DIRS}\${tdir}:"
          EXTRA_MOUNTS+=(-v "\${tdir}:\${tdir}:ro")
        fi
      fi
    done
  elif [[ -f "\$ext_path" ]]; then
    for cand in "\$(dirname "\$ext_path")" "\$(dirname "\$(readlink -f "\$ext_path")")"; do
      if [[ -d "\$cand" && "\$SEEN_DIRS" != *":\${cand}:"* ]]; then
        SEEN_DIRS="\${SEEN_DIRS}\${cand}:"
        EXTRA_MOUNTS+=(-v "\${cand}:\${cand}:ro")
      fi
    done
  fi
done
for f in '${MODEL_DIR}'/*.safetensors '${MODEL_DIR}'/trunk-*/*.safetensors '${DRAFTER_DIR}'/*.safetensors${PARTS_LINKS}; do
  if [[ -L "\$f" ]]; then
    tdir="\$(dirname "\$(readlink -f "\$f")")"
    if [[ "\$SEEN_DIRS" != *":\${tdir}:"* ]]; then
      SEEN_DIRS="\${SEEN_DIRS}\${tdir}:"
      EXTRA_MOUNTS+=(-v "\${tdir}:\${tdir}:ro")
    fi
  fi
done
docker run --rm --name '${cname}' \
  --gpus all --network host --ipc=host \
  --device=/dev/infiniband --cap-add=IPC_LOCK \
  --ulimit memlock=-1 --ulimit nofile=1048576 \
  -v '${MODEL_MOUNT}:${MODEL_MOUNT}:ro' \
  "\${EXTRA_MOUNTS[@]}" \
  -e RANK=${r} -e WORLD_SIZE=4 -e LOCAL_RANK=0 \
  -e MASTER_ADDR='${MASTER_ADDR}' -e MASTER_PORT='${MASTER_PORT}' \
  -e TF_HY4_REQUIRE_IB='${REQUIRE_IB_VAL}' \
  -e TF_HY4_TP_TIMEOUT_S='${TF_HY4_TP_TIMEOUT_S}' \
  -e TF_HY4_LOAD_TIMEOUT_S='${TF_HY4_LOAD_TIMEOUT_S}' \
  ${TENSORFOLD_MEMORY_RESERVE_GIB:+-e "TENSORFOLD_MEMORY_RESERVE_GIB=${TENSORFOLD_MEMORY_RESERVE_GIB}"} \
  ${TENSORFOLD_CUDA_MEMORY_LIMIT_GB:+-e "TENSORFOLD_CUDA_MEMORY_LIMIT_GB=${TENSORFOLD_CUDA_MEMORY_LIMIT_GB}"} \
  ${TF_HY4_MEM_FLOOR_GIB:+-e "TF_HY4_MEM_FLOOR_GIB=${TF_HY4_MEM_FLOOR_GIB}"} \
  ${TF_HY4_TRUNK:+-e "TF_HY4_TRUNK=${TF_HY4_TRUNK}"} \
  ${HY4_TRUNK_FAST:+-e "HY4_TRUNK_FAST=${HY4_TRUNK_FAST}"} \
  ${TF_HY4_DRAFTER_EXL3:+-e "TF_HY4_DRAFTER_EXL3=${TF_HY4_DRAFTER_EXL3}"} \
  ${TF_HY4_DRAFTER_HEAD:+-e "TF_HY4_DRAFTER_HEAD=${TF_HY4_DRAFTER_HEAD}"} \
  ${TF_HY4_TARGET_HEAD:+-e "TF_HY4_TARGET_HEAD=${TF_HY4_TARGET_HEAD}"} \
  ${TF_HY4_IHC_SPLIT:+-e "TF_HY4_IHC_SPLIT=${TF_HY4_IHC_SPLIT}"} \
  ${TF_HY4_HEAD_BF16:+-e "TF_HY4_HEAD_BF16=${TF_HY4_HEAD_BF16}"} \
  ${TF_HY4_DRAFT_MIN_MARGIN:+-e "TF_HY4_DRAFT_MIN_MARGIN=${TF_HY4_DRAFT_MIN_MARGIN}"} \
  ${TF_HY4_ROW_COST_TABLE:+-e "TF_HY4_ROW_COST_TABLE=${TF_HY4_ROW_COST_TABLE}"} \
  ${TF_HY4_REPEAT_STABLE:+-e "TF_HY4_REPEAT_STABLE=${TF_HY4_REPEAT_STABLE}"} \
  ${TF_TP_ROCE_RELAY:+-e "TF_TP_ROCE_RELAY=${TF_TP_ROCE_RELAY}"} \
  ${TF_TP_ROCE_RELAY_MAX_KB:+-e "TF_TP_ROCE_RELAY_MAX_KB=${TF_TP_ROCE_RELAY_MAX_KB}"} \
  ${TF_TP_KERNEL_IMAGE:+-e "TF_TP_KERNEL_IMAGE=${TF_TP_KERNEL_IMAGE}"} \
  ${HY4_TP_DIRECT_WRITE:+-e "HY4_TP_DIRECT_WRITE=${HY4_TP_DIRECT_WRITE}"} \
  ${TF_TP_ROCE_FAST:+-e "TF_TP_ROCE_FAST=${TF_TP_ROCE_FAST}"} \
  ${TF_TP_ROCE_BLOCKS:+-e "TF_TP_ROCE_BLOCKS=${TF_TP_ROCE_BLOCKS}"} \
  ${HY4_DECODE_GRAPH:+-e "HY4_DECODE_GRAPH=${HY4_DECODE_GRAPH}"} \
  ${HY4_ROUTED_FAST:+-e "HY4_ROUTED_FAST=${HY4_ROUTED_FAST}"} \
  ${TF_HY4_INDEX_PAGE_2K:+-e "TF_HY4_INDEX_PAGE_2K=${TF_HY4_INDEX_PAGE_2K}"} \
  ${TF_HY4_GREEDY_FULL_GATHER:+-e "TF_HY4_GREEDY_FULL_GATHER=${TF_HY4_GREEDY_FULL_GATHER}"} \
  -e TF_HY4_FAST_PREFILL='${TF_HY4_FAST_PREFILL}' \
  -e TF_HY4_PREFILL_AR='${TF_HY4_PREFILL_AR}' \
  ${TF_HY4_FAST_PREFILL_MIN_ROWS:+-e "TF_HY4_FAST_PREFILL_MIN_ROWS=${TF_HY4_FAST_PREFILL_MIN_ROWS}"} \
  ${TF_HY4_PREFILL_TIMERS:+-e "TF_HY4_PREFILL_TIMERS=${TF_HY4_PREFILL_TIMERS}"} \
  ${R0_VIS_DIR:+-e "TF_HY4_VISION_DIR=${R0_VIS_DIR}"} \
  -e NCCL_NET='${NCCL_NET}' \
  -e NCCL_IB_DISABLE='${NCCL_IB_DISABLE}' \
  -e NCCL_IB_HCA='${NCCL_IB_HCA}' \
  -e NCCL_IB_GID_INDEX='${NCCL_IB_GID_INDEX}' \
  -e NCCL_IB_SUBNET_AWARE_ROUTING='${NCCL_IB_SUBNET_AWARE_ROUTING}' \
  -e NCCL_IB_SUBNET_PREFIX_LEN='${NCCL_IB_SUBNET_PREFIX_LEN}' \
  -e NCCL_IB_MERGE_NICS='${NCCL_IB_MERGE_NICS}' \
  -e NCCL_CROSS_NIC='${NCCL_CROSS_NIC}' \
  -e NCCL_ALGO='${NCCL_ALGO}' \
  -e NCCL_PROTO='${NCCL_PROTO}' \
  -e NCCL_SHM_DISABLE='${NCCL_SHM_DISABLE}' \
  -e NCCL_CUMEM_ENABLE='${NCCL_CUMEM_ENABLE}' \
  -e NCCL_RUNTIME_CONNECT='${NCCL_RUNTIME_CONNECT}' \
  ${NCCL_MIN_NCHANNELS:+-e "NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS}"} \
  -e NCCL_MAX_NCHANNELS='${NCCL_MAX_NCHANNELS}' \
  -e NCCL_BUFFSIZE='${NCCL_BUFFSIZE}' \
  -e NCCL_SOCKET_IFNAME='${NCCL_SOCKET_IFNAME}' \
  -e GLOO_SOCKET_IFNAME='${GLOO_SOCKET_IFNAME}' \
  -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET \
  '${IMAGE}' ${RANK_CMD}
EOF
  )

  (
    rc=0
    ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$h" "$remote_cmd" >"$logf" 2>&1 || rc=$?
    echo "$rc" > "${LOGDIR}/rank${r}.rc"
    exit "$rc"
  ) &
  PIDS+=("$!")
done

LAUNCH_START_S="$(date +%s)"
LAUNCH_START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
DEADLINE=$(( LAUNCH_START_S + TIMEOUT_S ))
SERVER_READY=0
COLD_READY_S=""
LAST_READY_PROBE=0
LAST_MEM_PROBE=0
MIN_AVAIL=("9999.9" "9999.9" "9999.9" "9999.9")
MIN_UTC=("" "" "" "")

write_launch_meta() {
  python3 - "$LOGDIR/launch_meta.json" "$LAUNCH_START_UTC" "${COLD_READY_S:-}" \
    "$RING_HOSTS" "$CONTEXT_LEN" "$SERVE_PARALLEL" "$SERVE_DRAFTS" \
    "$TIMEOUT_S" "$TF_HY4_TP_TIMEOUT_S" "$TF_HY4_LOAD_TIMEOUT_S" \
    "${TENSORFOLD_MEMORY_RESERVE_GIB:-}" "${TENSORFOLD_CUDA_MEMORY_LIMIT_GB:-}" \
    "${MIN_AVAIL_GIB:-}" "$MEMFLOOR_GIB" \
    "${TF_HY4_TRUNK:-}" "${HY4_TRUNK_FAST:-}" "${DRAFTER_DIR:-}" \
    "${TF_HY4_DRAFTER_EXL3:-}" "${TF_HY4_DRAFTER_HEAD:-}" "${TF_HY4_TARGET_HEAD:-}" \
    "${TF_HY4_IHC_SPLIT:-}" "${TF_HY4_HEAD_BF16:-}" "${TF_HY4_DRAFT_MIN_MARGIN:-}" "${TF_HY4_ROW_COST_TABLE:-}" \
    "${TF_HY4_REPEAT_STABLE:-}" \
    "${TF_TP_ROCE_RELAY:-}" "${TF_TP_ROCE_RELAY_MAX_KB:-}" "${TF_TP_KERNEL_IMAGE:-}" \
    "${HY4_TP_DIRECT_WRITE:-}" "${TF_TP_ROCE_FAST:-}" "${TF_TP_ROCE_BLOCKS:-}" \
    "${HY4_DECODE_GRAPH:-}" "${HY4_ROUTED_FAST:-}" \
    "${TF_HY4_INDEX_PAGE_2K:-}" "${TF_HY4_GREEDY_FULL_GATHER:-}" \
    "$NCCL_NET" "$NCCL_IB_HCA" "$NCCL_IB_GID_INDEX" \
    "$NCCL_IB_SUBNET_AWARE_ROUTING" "$NCCL_ALGO" "$NCCL_PROTO" \
    "${NCCL_MIN_NCHANNELS:-}" "${NCCL_MAX_NCHANNELS:-}" \
    "$NCCL_CUMEM_ENABLE" "$NCCL_RUNTIME_CONNECT" "$TIMED_RUN" "${SERVE_THINKING_BUDGET:-}" "${SERVE_TEMPERATURE:-}" \
    "${SERVE_MODEL_NAME:-}" "${PARTS_ROOT:-}" \
    "${TENSORFOLD_BIN:-}" "$TF_HY4_FAST_PREFILL" "$TF_HY4_PREFILL_AR" "${TF_HY4_FAST_PREFILL_MIN_ROWS:-}" \
    "${TF_HY4_PREFILL_TIMERS:-}" "$SERVE_VISION" "${SERVE_VISION_TOKENS:-}" "${TF_HY4_VISION_DIR:-}" \
    "${GPU_CLOCK_CAP_MHZ:-}" "${ENGINE_COMMIT:-}" \
    "${BIN_SHA[*]:-}" "${CLK_LAUNCH[*]:-}" "${CLK_READY[*]:-}" <<'PY' 2>/dev/null || true
import json, sys
out = sys.argv[1]
keys = [
    "launch_start_utc", "cold_ready_s", "RING_HOSTS", "CONTEXT_LEN",
    "SERVE_PARALLEL", "SERVE_DRAFTS", "TIMEOUT_S", "TF_HY4_TP_TIMEOUT_S",
    "TF_HY4_LOAD_TIMEOUT_S", "TENSORFOLD_MEMORY_RESERVE_GIB", "TENSORFOLD_CUDA_MEMORY_LIMIT_GB",
    "MIN_AVAIL_GIB", "MEMFLOOR_GIB",
    "TF_HY4_TRUNK", "HY4_TRUNK_FAST", "DRAFTER_DIR",
    "TF_HY4_DRAFTER_EXL3", "TF_HY4_DRAFTER_HEAD", "TF_HY4_TARGET_HEAD",
    "TF_HY4_IHC_SPLIT", "TF_HY4_HEAD_BF16", "TF_HY4_DRAFT_MIN_MARGIN", "TF_HY4_ROW_COST_TABLE",
    "TF_HY4_REPEAT_STABLE",
    "TF_TP_ROCE_RELAY", "TF_TP_ROCE_RELAY_MAX_KB", "TF_TP_KERNEL_IMAGE",
    "HY4_TP_DIRECT_WRITE", "TF_TP_ROCE_FAST", "TF_TP_ROCE_BLOCKS",
    "HY4_DECODE_GRAPH", "HY4_ROUTED_FAST",
    "TF_HY4_INDEX_PAGE_2K", "TF_HY4_GREEDY_FULL_GATHER",
    "NCCL_NET", "NCCL_IB_HCA", "NCCL_IB_GID_INDEX",
    "NCCL_IB_SUBNET_AWARE_ROUTING", "NCCL_ALGO", "NCCL_PROTO",
    "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS",
    "NCCL_CUMEM_ENABLE", "NCCL_RUNTIME_CONNECT", "TIMED_RUN", "SERVE_THINKING_BUDGET", "SERVE_TEMPERATURE",
    "SERVE_MODEL_NAME", "PARTS_ROOT",
    "TENSORFOLD_BIN", "TF_HY4_FAST_PREFILL", "TF_HY4_PREFILL_AR", "TF_HY4_FAST_PREFILL_MIN_ROWS",
    "TF_HY4_PREFILL_TIMERS", "SERVE_VISION", "SERVE_VISION_TOKENS", "TF_HY4_VISION_DIR",
    "GPU_CLOCK_CAP_MHZ", "ENGINE_COMMIT",
    "bin_sha256_by_rank", "gpu_clock_mhz_at_launch_by_rank", "gpu_clock_mhz_at_ready_by_rank",
]
obj = {}
for k, v in zip(keys, sys.argv[2:]):
    if k == "cold_ready_s":
        obj[k] = float(v) if v else None
    elif k.endswith("_by_rank"):
        vals = v.split()
        if vals:
            obj[k] = {str(r): x for r, x in enumerate(vals)}
    elif v != "":
        obj[k] = v
with open(out, "w", encoding="utf-8") as f:
    json.dump(obj, f, indent=2)
    f.write("\n")
PY
  for r in 0 1 2 3; do
    if [[ "${MIN_AVAIL[$r]}" != "9999.9" ]]; then
      printf 'min_avail_gib=%s floor_gib=%s stopped=0 min_utc=%s\n' \
        "${MIN_AVAIL[$r]}" "$MEMFLOOR_GIB" "${MIN_UTC[$r]}" >> "${LOGDIR}/memfloor_r${r}.log"
    fi
  done
}
trap 'write_launch_meta; cleanup_all; exit 0' INT TERM

while true; do
  ALIVE=0
  for r in 0 1 2 3; do
    pid="${PIDS[$r]}"
    if kill -0 "$pid" 2>/dev/null; then
      ALIVE=$((ALIVE + 1))
    elif [[ -f "${LOGDIR}/rank${r}.rc" ]] && [[ "$(cat "${LOGDIR}/rank${r}.rc")" != "0" ]]; then
      echo "FAIL-CLOSED: rank $r (${HOSTS[$r]}) exited non-zero" >&2
      write_launch_meta
      cleanup_all
      exit 1
    fi
  done
  [[ "$ALIVE" -eq 0 ]] && break

  NOW="$(date +%s)"
  if [[ $((NOW - LAST_MEM_PROBE)) -ge 2 ]]; then
    LAST_MEM_PROBE="$NOW"
    TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    for r in 0 1 2 3; do
      avail="$(ssh -n -o BatchMode=yes -o ConnectTimeout=3 "${HOSTS[$r]}" \
        'awk "/^MemAvailable:/ {printf \"%.2f\", \$2 / 1048576}" /proc/meminfo' 2>/dev/null || echo "")"
      if [[ -n "$avail" ]]; then
        printf '%s avail_gib=%s\n' "$TS" "$avail" >> "${LOGDIR}/memfloor_r${r}.log"
        lower="$(awk -v a="$avail" -v m="${MIN_AVAIL[$r]}" 'BEGIN {print (a < m) ? 1 : 0}')"
        if [[ "$lower" == "1" ]]; then
          MIN_AVAIL[$r]="$avail"
          MIN_UTC[$r]="$TS"
        fi
        breach="$(awk -v a="$avail" -v f="$MEMFLOOR_GIB" 'BEGIN {print (a < f) ? 1 : 0}')"
        if [[ "$breach" == "1" ]]; then
          printf 'min_avail_gib=%s floor_gib=%s stopped=1 min_utc=%s\n' "$avail" "$MEMFLOOR_GIB" "$TS" >> "${LOGDIR}/memfloor_r${r}.log"
          echo "FAIL-CLOSED: memfloor breach on rank $r (${HOSTS[$r]}): ${avail} GiB < ${MEMFLOOR_GIB} GiB" >&2
          write_launch_meta
          cleanup_all
          exit 9
        fi
      fi
    done
  fi
  if [[ "$SERVE_MODE" -eq 1 && "$SERVER_READY" -eq 0 ]]; then
    if [[ $((NOW - LAST_READY_PROBE)) -ge 2 ]]; then
      LAST_READY_PROBE="$NOW"
      if ssh -n -o BatchMode=yes -o ConnectTimeout=4 "${HOSTS[0]}" \
        "curl -sf http://127.0.0.1:${SERVE_PORT}/v1/models >/dev/null 2>&1" 2>/dev/null; then
        SERVER_READY=1
        COLD_READY_S=$(( NOW - LAUNCH_START_S ))
        for rr in 0 1 2 3; do
          CLK_READY[$rr]="$(gpu_clock_snapshot "${HOSTS[$rr]}" "$rr" ready)"
        done
        write_launch_meta
        echo "Server ready on ${HOSTS[0]}:${SERVE_PORT} in ${COLD_READY_S}s; serving indefinitely (send INT/TERM to stop). Logs in ${LOGDIR}"
      fi
    fi
    if [[ "$SERVER_READY" -eq 0 && "$TIMEOUT_S" -gt 0 && "$NOW" -ge "$DEADLINE" ]]; then
      echo "FAIL-CLOSED: server not ready after ${TIMEOUT_S}s" >&2
      write_launch_meta
      cleanup_all
      exit 124
    fi
  elif [[ "$SERVE_MODE" -eq 0 && "$TIMEOUT_S" -gt 0 && "$NOW" -ge "$DEADLINE" ]]; then
    echo "FAIL-CLOSED: timeout after ${TIMEOUT_S}s" >&2
    write_launch_meta
    cleanup_all
    exit 124
  fi
  sleep 0.5
done
write_launch_meta

if [[ "$TIMED_RUN" == "1" ]]; then
  for r in 0 1 2 3; do
    logf="${LOGDIR}/rank${r}_${HOSTS[$r]}.log"
    if grep -q 'Using network Socket' "$logf" 2>/dev/null || ! grep -q 'Using network IB' "$logf" 2>/dev/null; then
      echo "FAIL-CLOSED: rank $r did not use NCCL IB transport during a timed run" >&2
      exit 3
    fi
  done
fi

echo "Launch completed cleanly. Logs in ${LOGDIR}"
