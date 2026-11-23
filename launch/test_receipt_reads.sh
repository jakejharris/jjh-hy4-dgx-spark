#!/usr/bin/env bash
# Offline test of launch/receipt_reads.sh with a fake ssh on PATH (no Spark is contacted).
#   bash launch/test_receipt_reads.sh
# Cases: all four read; one binary missing; one rank holding other bytes; a clock read failing with and
# without a declared cap. Each failure must name its rank and host and print read-failed, never a blank.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/hy4-reads-test.XXXXXX")"
trap 'rm -rf "${TMP:?}"' EXIT
mkdir -p "$TMP/bin"
cat > "$TMP/bin/ssh" <<'EOF'
#!/usr/bin/env bash
# fake ssh: the host is the argument before the remote command, which is last
host="${@: -2:1}"; cmd="${@: -1}"
case "$cmd" in
  sha256sum*)
    if [[ "$FAKE_MODE" == "missing" && "$host" == "node3" ]]; then echo "sha256sum: /x/tensorfold-native: No such file or directory" >&2; exit 1; fi
    if [[ "$FAKE_MODE" == "diff" && "$host" == "node4" ]]; then echo "1111111111111111111111111111111111111111111111111111111111111111  /x/tensorfold-native"; exit 0; fi
    echo "40188e7f905724278693a05dfa25f257ab8ba88a1db4ccf6eaaab894dc13570c  /x/tensorfold-native" ;;
  nvidia-smi*)
    if [[ "$FAKE_MODE" == "clock" && "$host" == "node2" ]]; then echo "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver." ; exit 9; fi
    echo "2184" ;;
esac
EOF
chmod +x "$TMP/bin/ssh"
export PATH="$TMP/bin:$PATH"
# shellcheck disable=SC1091
source "$HERE/receipt_reads.sh"
HOSTS=(node1 node2 node4 node3)
TENSORFOLD_BIN=/x/tensorfold-native
FAILS=0
check() { if eval "$1"; then echo "  ok   $2"; else echo "  FAIL $2"; FAILS=$((FAILS + 1)); fi; }

run_case() {  # $1 mode, $2 cap
  export FAKE_MODE="$1"; GPU_CLOCK_CAP_MHZ="$2"
  read_receipts_at_launch > "$TMP/out" 2> "$TMP/err"; RC=$?
}

echo "== all four read"
run_case ok 2200
check '[[ $RC -eq 0 && "${BIN_SHA[*]}" == "$(printf "40188e7f905724278693a05dfa25f257ab8ba88a1db4ccf6eaaab894dc13570c %.0s" 1 2 3 4 | sed "s/ $//")" && "${CLK_LAUNCH[*]}" == "2184 2184 2184 2184" ]]' "rc 0, four hashes, four clocks"
echo "== rank 3's binary missing"
run_case missing ""
check '[[ $RC -eq 1 && "${BIN_SHA[3]}" == "read-failed" ]] && grep -q "rank 3 (node3): cannot hash" "$TMP/err"' "rc 1, read-failed, error names rank 3 (node3)"
echo "== rank 2 holds other bytes"
run_case diff ""
check '[[ $RC -eq 1 ]] && grep -q "hold different tensorfold-native" "$TMP/err"' "rc 1, different binaries refused"
echo "== rank 1's clock read fails, cap declared"
run_case clock 2200
check '[[ $RC -eq 1 && "${CLK_LAUNCH[1]}" == "read-failed" ]] && grep -q "rank 1 (node2): graphics-clock read at launch failed (rc 9)" "$TMP/err"' "rc 1, read-failed, error names rank 1 (node2) and rc"
echo "== rank 1's clock read fails, no cap declared"
run_case clock ""
check '[[ $RC -eq 0 && "${CLK_LAUNCH[1]}" == "read-failed" ]] && grep -q "rank 1 (node2)" "$TMP/err"' "rc 0 (no cap claimed) but still named and recorded read-failed"
echo "== at ready"
FAKE_MODE=clock
v="$(gpu_clock_snapshot node2 1 ready 2> "$TMP/err")"
check '[[ "$v" == "read-failed" ]] && grep -q "at ready failed" "$TMP/err"' "ready snapshot failure recorded and named"
if [[ "$FAILS" -eq 0 ]]; then echo "== PASS"; else echo "== FAIL ($FAILS)"; exit 1; fi
