# Sourced by launch/launch.sh (and launch/test_receipt_reads.sh). Reads what a receipt must name, on each Spark:
# the SHA-256 of the tensorfold-native file the launcher mounts, and one graphics-clock snapshot (nvidia-smi
# clocks.gr). A read that fails says so by rank and host on stderr and is recorded as "read-failed", never a blank.

gpu_clock_snapshot() {   # $1 host, $2 rank, $3 when; prints MHz or read-failed
  local out rc v
  out="$(ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$1" \
    "nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits" 2>&1)" && rc=0 || rc=$?
  v="$(printf '%s\n' "$out" | head -n 1 | tr -d ' \r')"
  if [[ "$rc" -ne 0 || ! "$v" =~ ^[0-9]+$ ]]; then
    echo "ERROR: rank $2 ($1): graphics-clock read at $3 failed (rc ${rc}): $(printf '%s' "$out" | head -c 200)" >&2
    echo "read-failed"
    return 0
  fi
  echo "$v"
}

# read_receipts_at_launch: fills BIN_SHA[] and CLK_LAUNCH[] from HOSTS[0..3], TENSORFOLD_BIN and GPU_CLOCK_CAP_MHZ.
# Returns 1 when a binary hash cannot be read, when the four hashes differ, or when a clock cannot be read while
# GPU_CLOCK_CAP_MHZ is declared (the receipt then claims a cap). The caller stops the launch on 1.
read_receipts_at_launch() {
  local r out rc s fail=0
  BIN_SHA=(); CLK_LAUNCH=()
  for r in 0 1 2 3; do
    out="$(ssh -n -o BatchMode=yes -o ConnectTimeout=5 "${HOSTS[$r]}" "sha256sum '${TENSORFOLD_BIN}'" 2>&1)" && rc=0 || rc=$?
    s="${out%% *}"
    if [[ "$rc" -ne 0 || ! "$s" =~ ^[0-9a-f]{64}$ ]]; then
      echo "ERROR: rank ${r} (${HOSTS[$r]}): cannot hash TENSORFOLD_BIN=\"${TENSORFOLD_BIN}\" (rc ${rc}): $(printf '%s' "$out" | head -c 200)" >&2
      echo "       Stage the release binary at that path on every Spark; the receipt must name the bytes that run." >&2
      s="read-failed"; fail=1
    fi
    BIN_SHA[$r]="$s"
    CLK_LAUNCH[$r]="$(gpu_clock_snapshot "${HOSTS[$r]}" "$r" launch)"
    if [[ "${CLK_LAUNCH[$r]}" == "read-failed" && -n "${GPU_CLOCK_CAP_MHZ:-}" ]]; then
      echo "ERROR: rank ${r} (${HOSTS[$r]}): GPU_CLOCK_CAP_MHZ=${GPU_CLOCK_CAP_MHZ} is declared but the clock could not be read." >&2
      fail=1
    fi
    echo "rank ${r} (${HOSTS[$r]}): tensorfold-native sha256 ${BIN_SHA[$r]}, graphics clock ${CLK_LAUNCH[$r]} MHz"
  done
  if [[ "$fail" -eq 0 && "$(printf '%s\n' "${BIN_SHA[@]}" | sort -u | wc -l)" -ne 1 ]]; then
    echo "ERROR: the four ranks hold different tensorfold-native files (hashes above). Stage the same binary on all four." >&2
    fail=1
  fi
  return "$fail"
}
