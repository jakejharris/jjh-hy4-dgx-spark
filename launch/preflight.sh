#!/usr/bin/env bash
# Preflight checks (and optional 2-rank or 4-rank RDMA ring probe) for switchless RoCE launch.
# Usage: bash launch/preflight.sh [path/to/ring.env] [--env-file PATH] [--timed] [--probe] [--ranks 2|4]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/ring.env"
TIMED_FLAG=""
RUN_PROBE=0
PROBE_RANKS=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --timed) TIMED_FLAG="1"; shift ;;
    --probe) RUN_PROBE=1; shift ;;
    --ranks) PROBE_RANKS="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [path/to/ring.env] [--env-file PATH] [--timed] [--probe] [--ranks 2|4]"
      exit 0
      ;;
    *) ENV_FILE="$1"; shift ;;
  esac
done

if [[ ! -f "$ENV_FILE" ]]; then
  echo "FAIL: env file not found: $ENV_FILE (copy launch/ring.env.example to launch/ring.env)" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$ENV_FILE"
TIMED_RUN="${TIMED_FLAG:-${TIMED_RUN:-0}}"

if [[ "$RUN_PROBE" -eq 1 && "$PROBE_RANKS" != "2" && "$PROBE_RANKS" != "4" ]]; then
  echo "FAIL: --ranks must be 2 or 4 (got $PROBE_RANKS)" >&2
  exit 2
fi

if [[ "$TIMED_RUN" == "1" || "$RUN_PROBE" -eq 1 ]]; then
  if [[ "${NCCL_NET:-}" == "Socket" || "${NCCL_IB_DISABLE:-0}" == "1" ]]; then
    echo "FAIL-CLOSED: refusing RDMA probe or timed run on NCCL Socket transport (NCCL_NET=${NCCL_NET:-}, NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0})" >&2
    exit 3
  fi
fi

read -r -a HOSTS <<< "${RING_HOSTS:-}"
CHECK_RANKS=4
if [[ "$RUN_PROBE" -eq 1 ]]; then
  CHECK_RANKS="$PROBE_RANKS"
fi
if [[ ${#HOSTS[@]} -lt "$CHECK_RANKS" ]]; then
  echo "FAIL: RING_HOSTS must list at least ${CHECK_RANKS} nodes in physical cable order, got ${#HOSTS[@]}" >&2
  exit 2
fi
if [[ "$RUN_PROBE" -eq 0 && ${#HOSTS[@]} -ne 4 ]]; then
  echo "FAIL: RING_HOSTS must list exactly 4 nodes in physical cable order, got ${#HOSTS[@]}" >&2
  exit 2
fi

MIN_GIB="${MIN_AVAIL_GIB:-95}"
MEMFLOOR="${MEMFLOOR_GIB:-8}"
MODEL_PATH="${MODEL_DIR:-}"
if [[ "$RUN_PROBE" -eq 0 && -z "$MODEL_PATH" ]]; then
  echo "FAIL: MODEL_DIR must be set in $ENV_FILE" >&2
  exit 2
fi
IMG="${IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
CHECK_MIN_GIB="$MIN_GIB"
if [[ "$RUN_PROBE" -eq 1 ]]; then
  # Probe uses < 1 GiB per rank; only enforce MEMFLOOR_GIB + 2 GiB headroom if MIN_AVAIL_GIB is higher.
  CHECK_MIN_GIB="$(awk -v m="$MIN_GIB" -v f="$MEMFLOOR" 'BEGIN { need = f + 2; print (m < need) ? need : m }')"
fi

FAIL=0
RANK_F0_IP=()
RANK_F0_SUB=()
RANK_F1_IP=()
RANK_F1_SUB=()
RANK_AVAIL_GIB=()
for ((r = 0; r < CHECK_RANKS; r++)); do
  h="${HOSTS[$r]}"
  echo "=== Preflight rank $r ($h) ==="

  if ! ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$h" 'true' 2>/dev/null; then
    echo "  FAIL: node $h unreachable over SSH" >&2
    FAIL=1
    continue
  fi

  # Do NOT pass -n here: ssh must feed the heredoc on stdin to remote bash -s.
  out="$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" bash -s -- "$MODEL_PATH" "$CHECK_MIN_GIB" "$IMG" "$RUN_PROBE" <<'EOF'
set -euo pipefail
MODEL_PATH="$1"
MIN_GIB="$2"
IMG="$3"
IS_PROBE="$4"

# 1. RDMA devices present
if [[ ! -d /sys/class/infiniband ]] || [[ -z "$(ls -A /sys/class/infiniband 2>/dev/null)" ]]; then
  echo "ERR:no_rdma_devices"
  exit 1
fi
devs="$(ls /sys/class/infiniband | tr '\n' ',')"

# 2. Cable interfaces with IPv4 addresses (enp1s0f*)
subnets="$(ip -o -4 addr show | awk '$2 ~ /^enp1s0f/ {print $2 "=" $4}' | tr '\n' ' ')"
if [[ -z "$subnets" ]]; then
  echo "ERR:no_cable_subnets"
  exit 1
fi
f0_cidr="$(ip -o -4 addr show | awk '$2 ~ /^enp1s0f0/ {print $4; exit}')"
f1_cidr="$(ip -o -4 addr show | awk '$2 ~ /^enp1s0f1/ {print $4; exit}')"
if [[ -z "$f0_cidr" || -z "$f1_cidr" ]]; then
  echo "ERR:missing_cable_ports:f0=${f0_cidr:-none},f1=${f1_cidr:-none}"
  exit 1
fi

# 3. memlock limit
mlock="$(ulimit -l)"

# 4. Model folder present (required for serve preflight; optional when running --probe before quant/download)
model_note="model=ok"
if [[ -z "$MODEL_PATH" || ! -d "$MODEL_PATH" ]]; then
  if [[ "$IS_PROBE" == "1" ]]; then
    model_note="model=skipped_for_probe"
  else
    echo "ERR:missing_model_dir:${MODEL_PATH:-unset}"
    exit 1
  fi
fi

# 5. Free memory check (MemAvailable in GiB)
avail_gib="$(awk '/^MemAvailable:/ {printf "%.1f", $2 / 1048576}' /proc/meminfo)"
ok_mem="$(awk -v a="$avail_gib" -v m="$MIN_GIB" 'BEGIN {print (a >= m) ? 1 : 0}')"
if [[ "$ok_mem" != "1" ]]; then
  echo "ERR:low_mem:${avail_gib}GiB<${MIN_GIB}GiB"
  exit 1
fi

# 6. Container image present
if ! docker image inspect "$IMG" >/dev/null 2>&1; then
  echo "ERR:missing_image:$IMG"
  exit 1
fi

echo "OK rdma=${devs%,} f0=${f0_cidr} f1=${f1_cidr} subnets=[${subnets% }] memlock=${mlock} avail=${avail_gib}GiB ${model_note}"
EOF
  )" || {
    echo "  FAIL on $h: $out" >&2
    FAIL=1
    continue
  }
  echo "  $out"
  f0_cidr="$(printf '%s\n' "$out" | sed -n 's/.* f0=\([^ ]*\).*/\1/p')"
  f1_cidr="$(printf '%s\n' "$out" | sed -n 's/.* f1=\([^ ]*\).*/\1/p')"
  f0_ip="${f0_cidr%/*}"
  f1_ip="${f1_cidr%/*}"
  RANK_F0_IP+=("$f0_ip")
  RANK_F0_SUB+=("${f0_ip%.*}")
  RANK_F1_IP+=("$f1_ip")
  RANK_F1_SUB+=("${f1_ip%.*}")
  avail_val="$(printf '%s\n' "$out" | sed -n 's/.* avail=\([0-9.]*\)GiB.*/\1/p')"
  RANK_AVAIL_GIB+=("${avail_val:-0}")
done

if [[ "$FAIL" -ne 0 ]]; then
  echo "Preflight: FAILED" >&2
  exit 1
fi

# Vision (off unless SERVE_VISION="1"): rank 0 must hold the vision folder TF_HY4_VISION_DIR, and every file
# launch/vision.sha256 names must match it. A missing folder, a missing file or a wrong hash is refused.
if [[ "${SERVE_VISION:-0}" == "1" && "$RUN_PROBE" -eq 0 ]]; then
  VIS_SUMS="${SCRIPT_DIR}/vision.sha256"
  VIS_DIR="${TF_HY4_VISION_DIR:-}"
  if [[ -z "$VIS_DIR" || "$VIS_DIR" != /* ]]; then
    echo "FAIL: SERVE_VISION=\"1\" needs TF_HY4_VISION_DIR, the full path of the vision folder (vit-bf16/) on rank 0 (${HOSTS[0]})." >&2
    exit 2
  fi
  echo "Checking the vision folder ${VIS_DIR} on rank 0 (${HOSTS[0]}) against launch/vision.sha256..."
  # Do NOT pass -n here: ssh feeds the hash list on stdin to sha256sum -c.
  if ! vis_out="$(ssh -o BatchMode=yes -o ConnectTimeout=8 "${HOSTS[0]}" "cd '${VIS_DIR}' && sha256sum -c --strict -" < "$VIS_SUMS" 2>&1)"; then
    echo "FAIL: the vision folder on ${HOSTS[0]} is missing or does not match launch/vision.sha256:" >&2
    printf '%s\n' "$vis_out" | sed 's/^/  /' >&2
    echo "  Build it with tools/vit_folder.py, or copy the published vit-bf16/ folder, and point TF_HY4_VISION_DIR at it." >&2
    exit 2
  fi
  echo "  OK: $(printf '%s\n' "$vis_out" | grep -c ': OK$') vision files match launch/vision.sha256"
fi

# 7. Cable-order check: rank r's second port (f1) subnet must match next rank's first port (f0) subnet,
#    and the route from rank r to next rank's f0 IP must be a direct 1-hop cable route (dev enp1s0f*, no via).
CABLE_FAIL=0
LINK_COUNT="$CHECK_RANKS"
if [[ "$CHECK_RANKS" -eq 2 ]]; then
  LINK_COUNT=1
fi
for ((r = 0; r < LINK_COUNT; r++)); do
  next=$(( (r + 1) % CHECK_RANKS ))
  if [[ "${RANK_F1_SUB[$r]}" != "${RANK_F0_SUB[$next]}" ]]; then
    echo "FAIL-CLOSED: cable-order mismatch between rank ${r} (${HOSTS[$r]} f1=${RANK_F1_SUB[$r]}.0/24) and rank ${next} (${HOSTS[$next]} f0=${RANK_F0_SUB[$next]}.0/24)" >&2
    CABLE_FAIL=1
  fi
  route_line="$(ssh -n -o BatchMode=yes -o ConnectTimeout=8 "${HOSTS[$r]}" "ip route get ${RANK_F0_IP[$next]} | head -1" 2>/dev/null || echo "route_error")"
  if [[ "$route_line" != *"dev enp1s0f"* || "$route_line" == *" via "* ]]; then
    echo "FAIL-CLOSED: rank ${r} (${HOSTS[$r]}) -> rank ${next} (${HOSTS[$next]} ${RANK_F0_IP[$next]}) is not a direct 1-hop cable route: ${route_line}" >&2
    CABLE_FAIL=1
  fi
done
if [[ "$CABLE_FAIL" -ne 0 ]]; then
  # Try to suggest the clockwise order by matching f1_sub -> f0_sub.
  SUGGEST=("${HOSTS[0]}")
  cur=0
  for ((step = 1; step < CHECK_RANKS; step++)); do
    found=""
    for ((k = 0; k < CHECK_RANKS; k++)); do
      if [[ "${RANK_F1_SUB[$cur]}" == "${RANK_F0_SUB[$k]}" ]]; then
        found="$k"
        break
      fi
    done
    if [[ -z "$found" ]]; then
      SUGGEST=()
      break
    fi
    SUGGEST+=("${HOSTS[$found]}")
    cur="$found"
  done
  if [[ ${#SUGGEST[@]} -eq "$CHECK_RANKS" ]]; then
    echo "HINT: suggested cable-order RING_HOSTS=\"${SUGGEST[*]}\"" >&2
  fi
  exit 4
fi

echo "Preflight: PASS (${CHECK_RANKS} nodes ready, RDMA devices and subnets verified, mem >= ${CHECK_MIN_GIB} GiB)"

if [[ "$RUN_PROBE" -eq 0 ]]; then
  echo "=== MemAvailable summary vs 78-layer arithmetic thresholds (warning only; unverified until first boot) ==="
  for ((r = 0; r < CHECK_RANKS; r++)); do
    h="${HOSTS[$r]}"
    a="${RANK_AVAIL_GIB[$r]}"
    w4k="$(awk -v a="$a" 'BEGIN {print (a < 97.0) ? 1 : 0}')"
    w128k="$(awk -v a="$a" 'BEGIN {print (a < 111.0) ? 1 : 0}')"
    if [[ "$w4k" == "1" ]]; then
      echo "  WARN rank ${r} (${h}): MemAvailable=${a} GiB (< 97 GiB arithmetic threshold for 78L 4,096 boot; < 111 GiB for 78L 128K boot)"
    elif [[ "$w128k" == "1" ]]; then
      echo "  WARN rank ${r} (${h}): MemAvailable=${a} GiB (>= 97 GiB for 78L 4,096 boot; < 111 GiB arithmetic threshold for 78L 128K boot)"
    else
      echo "  OK   rank ${r} (${h}): MemAvailable=${a} GiB (>= 97 GiB for 78L 4,096 boot; >= 111 GiB for 78L 128K boot)"
    fi
  done
fi

if [[ "$RUN_PROBE" -eq 1 ]]; then
  PROBE_DIR="${LOGDIR:-/tmp/hy4-tp4-logs}/probe-r${PROBE_RANKS}"
  PROBE_PREFIX="${PREFIX:-hy4-tp4}-probe"
  PROBE_TIMEOUT_S="${PROBE_TIMEOUT_S:-${TIMEOUT_S:-60}}"
  if [[ "$PROBE_TIMEOUT_S" -gt 120 ]]; then
    PROBE_TIMEOUT_S=90
  fi
  mkdir -p "$PROBE_DIR"
  rm -f "${PROBE_DIR}"/rank*.log "${PROBE_DIR}"/rank*.rc

  cleanup_probe() {
    for ((r = 0; r < PROBE_RANKS; r++)); do
      local h="${HOSTS[$r]}"
      local cname="${PROBE_PREFIX}-r${r}"
      ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$h" \
        "docker rm -f '${cname}' >/dev/null 2>&1 || true; rm -f '/tmp/${cname}.py'" >/dev/null 2>&1 &
    done
    wait || true
  }
  trap cleanup_probe INT TERM EXIT
  cleanup_probe

  # Stage probe script to /tmp on each participating node so remote docker does not depend on controller paths.
  for ((r = 0; r < PROBE_RANKS; r++)); do
    h="${HOSTS[$r]}"
    cname="${PROBE_PREFIX}-r${r}"
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "cat > '/tmp/${cname}.py'" < "${SCRIPT_DIR}/ring-rdma-probe.py"
  done

  PIDS=()
  for ((r = 0; r < PROBE_RANKS; r++)); do
    h="${HOSTS[$r]}"
    cname="${PROBE_PREFIX}-r${r}"
    logf="${PROBE_DIR}/rank${r}_${h}.log"
    rcf="${PROBE_DIR}/rank${r}.rc"

    remote_cmd=$(cat <<EOF
set -o pipefail
docker rm -f '${cname}' >/dev/null 2>&1 || true
docker run --rm --name '${cname}' \
  --gpus all --network host --ipc=host \
  --device=/dev/infiniband --cap-add=IPC_LOCK \
  --ulimit memlock=-1 --ulimit nofile=1048576 \
  -v '/tmp/${cname}.py:/probe/ring-rdma-probe.py:ro' \
  -e RANK=${r} -e WORLD_SIZE=${PROBE_RANKS} -e LOCAL_RANK=0 \
  -e MASTER_ADDR='${MASTER_ADDR}' -e MASTER_PORT='${MASTER_PORT}' \
  -e NCCL_NET='${NCCL_NET}' -e NCCL_IB_DISABLE='${NCCL_IB_DISABLE}' \
  -e NCCL_IB_HCA='${NCCL_IB_HCA}' -e NCCL_IB_GID_INDEX='${NCCL_IB_GID_INDEX}' \
  -e NCCL_IB_SUBNET_AWARE_ROUTING='${NCCL_IB_SUBNET_AWARE_ROUTING}' \
  -e NCCL_IB_SUBNET_PREFIX_LEN='${NCCL_IB_SUBNET_PREFIX_LEN}' \
  -e NCCL_IB_MERGE_NICS='${NCCL_IB_MERGE_NICS}' -e NCCL_CROSS_NIC='${NCCL_CROSS_NIC}' \
  -e NCCL_ALGO='${NCCL_ALGO}' -e NCCL_PROTO='${NCCL_PROTO}' \
  -e NCCL_SHM_DISABLE='${NCCL_SHM_DISABLE}' -e NCCL_CUMEM_ENABLE='${NCCL_CUMEM_ENABLE}' \
  -e NCCL_RUNTIME_CONNECT='${NCCL_RUNTIME_CONNECT}' \
  ${NCCL_MIN_NCHANNELS:+-e "NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS}"} \
  -e NCCL_MAX_NCHANNELS='${NCCL_MAX_NCHANNELS}' -e NCCL_BUFFSIZE='${NCCL_BUFFSIZE}' \
  -e NCCL_SOCKET_IFNAME='${NCCL_SOCKET_IFNAME}' -e GLOO_SOCKET_IFNAME='${GLOO_SOCKET_IFNAME}' \
  -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET \
  -e NCCL_DEBUG_FILE=/tmp/nccl_probe.log \
  '${IMG}' python3 /probe/ring-rdma-probe.py --ranks '${PROBE_RANKS}' --nccl-log /tmp/nccl_probe.log --timeout-s '${PROBE_TIMEOUT_S}'
EOF
    )

    (
      rc=0
      ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$h" "$remote_cmd" >"$logf" 2>&1 || rc=$?
      echo "$rc" > "$rcf"
      exit "$rc"
    ) &
    PIDS+=("$!")
  done

  MIN_AVAIL=()
  for ((r = 0; r < PROBE_RANKS; r++)); do MIN_AVAIL+=("9999.9"); done
  DEADLINE=$(( $(date +%s) + PROBE_TIMEOUT_S ))
  LAST_MEM_CHECK=0

  while true; do
    ALIVE=0
    for ((r = 0; r < PROBE_RANKS; r++)); do
      pid="${PIDS[$r]}"
      if kill -0 "$pid" 2>/dev/null; then
        ALIVE=$((ALIVE + 1))
      elif [[ -f "${PROBE_DIR}/rank${r}.rc" ]] && [[ "$(cat "${PROBE_DIR}/rank${r}.rc")" != "0" ]]; then
        echo "FAIL-CLOSED: probe rank $r (${HOSTS[$r]}) exited $(cat "${PROBE_DIR}/rank${r}.rc") (see ${PROBE_DIR})" >&2
        cleanup_probe
        trap - INT TERM EXIT
        exit 1
      fi
    done
    [[ "$ALIVE" -eq 0 ]] && break

    NOW="$(date +%s)"
    if [[ "$NOW" -ge "$DEADLINE" ]]; then
      echo "FAIL-CLOSED: probe timeout after ${PROBE_TIMEOUT_S}s (see ${PROBE_DIR})" >&2
      cleanup_probe
      trap - INT TERM EXIT
      exit 124
    fi
    if [[ $((NOW - LAST_MEM_CHECK)) -ge 2 ]]; then
      LAST_MEM_CHECK="$NOW"
      for ((r = 0; r < PROBE_RANKS; r++)); do
        avail="$(ssh -n -o BatchMode=yes -o ConnectTimeout=4 "${HOSTS[$r]}" \
          'awk "/^MemAvailable:/ {printf \"%.1f\", \$2 / 1048576}" /proc/meminfo' 2>/dev/null || echo "")"
        if [[ -n "$avail" ]]; then
          MIN_AVAIL[$r]="$(awk -v a="$avail" -v m="${MIN_AVAIL[$r]}" 'BEGIN {printf "%.1f", (a < m) ? a : m}')"
          breach="$(awk -v a="$avail" -v f="$MEMFLOOR" 'BEGIN {print (a < f) ? 1 : 0}')"
          if [[ "$breach" == "1" ]]; then
            echo "FAIL-CLOSED: memfloor breach on ${HOSTS[$r]} (${avail} GiB < ${MEMFLOOR} GiB)" >&2
            cleanup_probe
            trap - INT TERM EXIT
            exit 9
          fi
        fi
      done
    fi
    sleep 0.5
  done

  for ((r = 0; r < PROBE_RANKS; r++)); do
    wait "${PIDS[$r]}" || { echo "Preflight probe: FAILED (see $PROBE_DIR)" >&2; exit 1; }
  done

  for ((r = 0; r < PROBE_RANKS; r++)); do
    logf="${PROBE_DIR}/rank${r}_${HOSTS[$r]}.log"
    if grep -q 'Using network Socket' "$logf" || ! grep -q 'Using network IB' "$logf"; then
      echo "FAIL-CLOSED: rank $r (${HOSTS[$r]}) did not use NCCL IB transport (see $logf)" >&2
      exit 3
    fi
    grep -E 'NCCL_TRANSPORT|PROBE_JSON' "$logf" || true
  done
  echo "PROBE_MEMFLOOR min_avail_gib=(${MIN_AVAIL[*]}) floor=${MEMFLOOR} stopped=0"
  echo "Preflight probe (${PROBE_RANKS} ranks): PASS"
fi
