#!/usr/bin/env bash
# Offline test of SWITCH_FABRIC in launch/preflight.sh (no Spark is contacted).
#   bash launch/test_preflight_switch.sh
# A fake ssh on PATH answers the per-Spark check with canned addresses and "ip route get" with canned routes.
# Cases: a good switch, the ring default unchanged (cable order still checked), and every refusal.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/hy4-switch-test.XXXXXX")"
trap 'rm -rf "${TMP:?}"' EXIT
mkdir -p "$TMP/bin"
cat > "$TMP/bin/ssh" <<'EOS'
#!/usr/bin/env bash
host=""; prev=""; remote=0
for a in "$@"; do
  if [[ "$a" == node[1-4] && -z "$host" ]]; then host="$a"; fi
  if [[ "$prev" == "bash" && "$a" == "-s" ]]; then remote=1; fi
  prev="$a"
done
n="${host#node}"; cmd="${@: -1}"
printf '%s\n' "$*" >> "$FAKE_SSH_LOG"
if [[ "$remote" -eq 1 ]]; then
  cat > "$FAKE_SSH_LOG.script"
  if [[ "$FAKE_CASE" == ring* ]]; then
    f0="10.20.${n}.2"; f1="10.20.$(( n % 4 + 1 )).1"
    if [[ "$FAKE_CASE" == ring_swapped && "$n" == 2 ]]; then f1="10.20.1.1"; fi
  else
    f0="10.50.1.${n}"; f1="10.50.2.${n}"
    if [[ "$FAKE_CASE" == subnet && "$n" == 3 ]]; then f1="10.50.9.${n}"; fi
    if [[ "$FAKE_CASE" == samenet ]]; then f1="10.50.1.1${n}"; fi
    if [[ "$FAKE_CASE" == norail && "$n" == 4 ]]; then echo "ERR:missing_switch_rails:enp1s0f0np0=10.50.1.4/24,enP2p1s0f0np0=none"; exit 1; fi
  fi
  echo "OK rdma=a,b f0=${f0}/24 f1=${f1}/24 subnets=[x] memlock=unlimited avail=117.0GiB model=ok"
  exit 0
fi
case "$cmd" in
  "ip route get "*)
    ip="${cmd#ip route get }"; ip="${ip%% *}"
    dev="enp1s0f0np0"; via=""
    case "$ip" in 10.50.2.*) dev="enP2p1s0f0np0" ;; esac
    if [[ "$FAKE_CASE" == ring* ]]; then dev="enp1s0f1np1"; fi
    if [[ "$FAKE_CASE" == gateway && "$n" == 1 && "$ip" == 10.50.2.3 ]]; then via="via 10.0.0.1 "; fi
    if [[ "$FAKE_CASE" == wrongdev && "$n" == 2 && "$ip" == 10.50.2.4 ]]; then dev="enp1s0f0np0"; fi
    if [[ "$FAKE_CASE" == prefixdev && "$ip" == 10.50.1.* ]]; then dev="enp1s0f0np01"; fi
    echo "${ip} ${via}dev ${dev} src 10.0.0.9 uid 1000" ;;
esac
exit 0
EOS
chmod +x "$TMP/bin/ssh"
export PATH="$TMP/bin:$PATH" FAKE_SSH_LOG="$TMP/ssh.log"
FAILS=0
check() { if eval "$1"; then echo "  ok   $2"; else echo "  FAIL $2"; FAILS=$((FAILS + 1)); fi; }

run_case() {  # $1 fake case, then settings lines
  export FAKE_CASE="$1"; shift
  : > "$FAKE_SSH_LOG"
  printf '%s\n' 'RING_HOSTS="node1 node2 node3 node4"' 'MODEL_DIR="/m"' "$@" > "$TMP/ring.env"
  env -u SWITCH_FABRIC -u SWITCH_RAILS -u TF_TP_ROCE_RELAY bash "$HERE/preflight.sh" "$TMP/ring.env" > "$TMP/out" 2> "$TMP/err"; RC=$?
}
ON='SWITCH_FABRIC="1"'

echo "== a good switch"
run_case good "$ON"
check '[[ $RC -eq 0 ]] && grep -q "^Preflight: PASS" "$TMP/out"' "rc $RC, pass"
check '[[ $(grep -c "ip route get" "$FAKE_SSH_LOG") -eq 24 ]]' "24 routes checked (4 ranks, 3 others, 2 rails)"
check '[[ $(grep -c -- "0 1 enp1s0f0np0 enP2p1s0f0np0$" "$FAKE_SSH_LOG") -eq 4 ]]' "the default rails reach all four Sparks"

echo "== other rail names"
run_case good "$ON" 'SWITCH_RAILS="railA railB"'
check '[[ $RC -eq 4 ]] && grep -q "does not leave directly through railA" "$TMP/err"' "rc $RC, the named rails are the ones checked"

echo "== the shipped overlay"
run_case good "$(cat "$HERE/ring.env.example")" "$(cat "$HERE/ring.switch.env.example")" 'RING_HOSTS="node1 node2 node3 node4"'
check '[[ $RC -eq 0 ]] && grep -q "^Preflight: PASS" "$TMP/out"' "rc $RC, ring.env.example plus ring.switch.env.example passes"

echo "== ring default"
run_case ring
check '[[ $RC -eq 0 ]] && grep -q "^Preflight: PASS" "$TMP/out"' "rc $RC, ring passes with SWITCH_FABRIC missing"
check '[[ $(grep -c "ip route get" "$FAKE_SSH_LOG") -eq 4 ]]' "4 cable routes checked, as before"
check '[[ $(grep -c -- "0 0 - -$" "$FAKE_SSH_LOG") -eq 4 ]]' "no rails passed"
run_case ring 'SWITCH_FABRIC="0"' 'TF_TP_ROCE_RELAY="1"'
check '[[ $RC -eq 0 ]]' "rc $RC, ring with the relay is not refused"
run_case ring_swapped
check '[[ $RC -eq 4 ]] && grep -q "cable-order mismatch" "$TMP/err"' "rc $RC, a wrong cable order is still refused"
run_case good
check '[[ $RC -eq 4 ]]' "rc $RC, switch addresses without SWITCH_FABRIC are refused as before"

echo "== refused before any Spark is contacted"
for line in 'SWITCH_FABRIC="yes"' 'SWITCH_FABRIC="2"'; do
  run_case good "$line"
  check '[[ $RC -eq 2 && ! -s "$FAKE_SSH_LOG" ]] && grep -q "FAIL: SWITCH_FABRIC" "$TMP/err"' "rc $RC, $line"
done
for rails in "one" "a b c" "a a" "a b;c" "a -b"; do
  run_case good "$ON" "SWITCH_RAILS=\"$rails\""
  check '[[ $RC -eq 2 && ! -s "$FAKE_SSH_LOG" ]] && grep -q "SWITCH_RAILS" "$TMP/err"' "rc $RC, SWITCH_RAILS=\"$rails\""
done
run_case good "$ON" 'TF_TP_ROCE_RELAY="1"'
check '[[ $RC -eq 2 && ! -s "$FAKE_SSH_LOG" ]] && grep -q "with the relay on" "$TMP/err"' "rc $RC, the relay on a switch"

echo "== refused after looking"
run_case subnet "$ON"
check '[[ $RC -eq 4 ]] && grep -q "are in different subnets" "$TMP/err"' "rc $RC, one rail in another subnet"
run_case samenet "$ON"
check '[[ $RC -eq 4 ]] && grep -q "has both rails in" "$TMP/err"' "rc $RC, both rails in one subnet"
run_case gateway "$ON"
check '[[ $RC -eq 4 ]] && grep -q "does not leave directly through enP2p1s0f0np0" "$TMP/err"' "rc $RC, a route through a gateway"
run_case wrongdev "$ON"
check '[[ $RC -eq 4 && $(grep -c "FAIL-CLOSED" "$TMP/err") -eq 1 ]]' "rc $RC, a route out of the other rail (one line)"
run_case prefixdev "$ON"
check '[[ $RC -eq 4 ]]' "rc $RC, an interface whose name only starts like the rail"
run_case norail "$ON"
check '[[ $RC -eq 1 ]] && grep -q "missing_switch_rails" "$TMP/err"' "rc $RC, a Spark without the second rail"
if [[ "$FAILS" -eq 0 ]]; then echo "== PASS"; else echo "== FAIL ($FAILS)"; exit 1; fi
