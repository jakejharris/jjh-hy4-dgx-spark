#!/usr/bin/env bash
# Offline test of SERVE_TEMPERATURE in launch/launch.sh (no Spark is contacted).
#   bash launch/test_launch_temperature.sh
# The real launch.sh and receipt_reads.sh are copied into a temp folder beside a preflight.sh that only exits 0,
# and a fake ssh on PATH answers the launcher's reads and prints each rank's command into that rank's log.
# Cases: the line missing (--temperature 0.7 on all four ranks), empty (no flag), "0" (--temperature 0), and bad
# values (refused with exit 2 before any ssh). launch_meta.json must record the value in force.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/hy4-temp-test.XXXXXX")"
trap 'rm -rf "${TMP:?}"' EXIT
mkdir -p "$TMP/bin" "$TMP/launch"
cp "$HERE/launch.sh" "$HERE/receipt_reads.sh" "$HERE/ring.env.example" "$TMP/launch/"
printf '#!/usr/bin/env bash\nexit 0\n' > "$TMP/launch/preflight.sh"
cat > "$TMP/bin/ssh" <<'EOF'
#!/usr/bin/env bash
# fake ssh: the host is the argument before the remote command, which is last
host="${@: -2:1}"; cmd="${@: -1}"
printf '%s %s\n' "$host" "${cmd:0:60}" >> "$FAKE_SSH_LOG"
case "$cmd" in
  *"docker run --rm --name"*) printf '%s\n' "$cmd" ;;
  sha256sum*) echo "40188e7f905724278693a05dfa25f257ab8ba88a1db4ccf6eaaab894dc13570c  /x/tensorfold-native" ;;
  nvidia-smi*) echo "2184" ;;
  *dfly_num_hidden_layers*) echo "4" ;;
  *MemAvailable*) echo "100.00" ;;
  *curl*) exit 1 ;;
esac
exit 0
EOF
chmod +x "$TMP/bin/ssh"
export PATH="$TMP/bin:$PATH"
export FAKE_SSH_LOG="$TMP/ssh.log"
unset SERVE_TEMPERATURE
FAILS=0
check() { if eval "$1"; then echo "  ok   $2"; else echo "  FAIL $2"; FAILS=$((FAILS + 1)); fi; }

run_case() {  # $1 case name, $2 overlay line (or "none")
  export LOGDIR="$TMP/logs-$1"
  : > "$FAKE_SSH_LOG"
  local envs=(--env-file "$TMP/launch/ring.env.example")
  if [[ "$2" != "none" ]]; then
    printf '%s\n' "$2" > "$TMP/overlay-$1.env"
    envs+=(--env-file "$TMP/overlay-$1.env")
  fi
  bash "$TMP/launch/launch.sh" "${envs[@]}" > "$TMP/out-$1" 2> "$TMP/err-$1"; RC=$?
  RANKS_WITH=$(cat "$LOGDIR"/rank?_*.log 2>/dev/null | grep -c -F -- "$3")
  RANKS_ANY=$(cat "$LOGDIR"/rank?_*.log 2>/dev/null | grep -c -F -- "--temperature")
  META=$(python3 -I - "$LOGDIR/launch_meta.json" <<'PY' 2>/dev/null
import json, sys
print(json.load(open(sys.argv[1])).get("SERVE_TEMPERATURE", "absent"))
PY
)
}

echo "== line missing (the example has it commented out)"
check '! grep -q "^SERVE_TEMPERATURE=" "$TMP/launch/ring.env.example"' "ring.env.example leaves SERVE_TEMPERATURE unset"
run_case default none "--temperature 0.7 "
check '[[ $RC -eq 0 && $RANKS_WITH -eq 4 && $RANKS_ANY -eq 4 ]]' "rc $RC, --temperature 0.7 on $RANKS_WITH of 4 ranks"
check '[[ "$META" == "0.7" ]]' "launch_meta.json SERVE_TEMPERATURE $META"

echo "== empty"
run_case empty 'SERVE_TEMPERATURE=""' "--temperature"
check '[[ $RC -eq 0 && $RANKS_ANY -eq 0 ]] && grep -q -F -- "--no-update-check" "$LOGDIR"/rank0_*.log' "rc $RC, no --temperature on any rank ($RANKS_ANY), rank 0 still launched"
check '[[ "$META" == "absent" ]]' "launch_meta.json has no SERVE_TEMPERATURE ($META)"

echo "== \"0\" (greedy by default)"
run_case zero 'SERVE_TEMPERATURE="0"' "--temperature 0 "
check '[[ $RC -eq 0 && $RANKS_WITH -eq 4 && $RANKS_ANY -eq 4 ]]' "rc $RC, --temperature 0 on $RANKS_WITH of 4 ranks"
check '[[ "$META" == "0" ]]' "launch_meta.json SERVE_TEMPERATURE $META"

for bad in warm -1 0.7.1 "0,7"; do
  echo "== bad value \"$bad\""
  run_case "bad" "SERVE_TEMPERATURE=\"$bad\"" "--temperature"
  check '[[ $RC -eq 2 ]] && grep -q -F "ERROR: SERVE_TEMPERATURE=\"$bad\"; use a number of 0 or more" "$TMP/err-bad"' "rc $RC, refused with a plain error"
  check '[[ ! -s "$FAKE_SSH_LOG" && $RANKS_ANY -eq 0 ]]' "nothing contacted, nothing launched"
done
if [[ "$FAILS" -eq 0 ]]; then echo "== PASS"; else echo "== FAIL ($FAILS)"; exit 1; fi
