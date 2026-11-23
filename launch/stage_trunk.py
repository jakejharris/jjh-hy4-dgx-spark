#!/usr/bin/env python3
"""Cable-only stager for packed-trunk sidecar folders (trunk-4b, trunk-6b) on a 4-node ring.

Reads node names, direct 1-hop cable IPs, and STAGE_DIR from a ring.env file (--env-file).
Copies finished layer files listed in metadata.layers_done of trunk.index.json from the
trunk-pass node (default --source-rank 2) to the other three nodes over the verified direct
1-hop cables only (no Wi-Fi, no StrictHostKeyChecking=no).

Each hop writes to a hidden temporary file, verifies SHA-256 and byte length against the
source before atomic rename, and skips files that are already hash-verified on re-run.
Final metadata and extra shards (lm_head.safetensors, dfly_layers.safetensors, and last
trunk.index.json) are copied only once all target layers (default 78) and both extras are
finished on the source, so a half-staged folder can never appear complete.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SSH_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=4",
]

FAST_CABLE_SSH_OPTS = (
    "-o BatchMode=yes -o ConnectTimeout=10 -o Compression=no "
    "-c aes128-gcm@openssh.com"
)

SENDER_SCRIPT = """#!/usr/bin/env python3
import hashlib, sys
p = sys.argv[1]
h = hashlib.sha256()
n = 0
buf = bytearray(4 * 1024 * 1024)
mv = memoryview(buf)
with open(p, "rb", buffering=0) as f:
    out = sys.stdout.buffer
    while True:
        r = f.readinto(buf)
        if not r:
            break
        h.update(mv[:r])
        out.write(mv[:r])
        n += r
    out.flush()
sys.stderr.write(f"SRC_SHA256={h.hexdigest()} SRC_BYTES={n}\\n")
sys.stderr.flush()
"""

RECEIVER_SCRIPT = """#!/usr/bin/env python3
import hashlib, os, sys
tmp = sys.argv[1]
h = hashlib.sha256()
n = 0
buf = bytearray(4 * 1024 * 1024)
mv = memoryview(buf)
inp = sys.stdin.buffer
with open(tmp, "wb", buffering=0) as f:
    while True:
        r = inp.readinto(buf)
        if not r:
            break
        h.update(mv[:r])
        f.write(mv[:r])
        n += r
    f.flush()
    os.fsync(f.fileno())
print(f"DST_SHA256={h.hexdigest()} DST_BYTES={n}", flush=True)
"""

BATCH_HASH_PY = """
import hashlib, json, os, sys
base = sys.argv[1]
names = sys.argv[2:]
out = {}
buf = bytearray(4 * 1024 * 1024)
mv = memoryview(buf)
for name in names:
    p = os.path.join(base, name)
    if not os.path.isfile(p):
        continue
    h = hashlib.sha256()
    n = 0
    with open(p, "rb", buffering=0) as f:
        while True:
            r = f.readinto(buf)
            if not r:
                break
            h.update(mv[:r])
            n += r
    out[name] = [h.hexdigest(), n]
print(json.dumps(out))
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_line(msg: str) -> None:
    print(f"{utc_now()} {msg}", flush=True)


def parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        env[k] = v
    return env


def ssh_target(host: str, user: str) -> str:
    if user and "@" not in host:
        return f"{user}@{host}"
    return host


def run_ssh(target: str, cmd: str, *, input_bytes: bytes | None = None, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", *SSH_OPTS, target, cmd],
        input=input_bytes,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


class TrunkRingConfig:
    def __init__(self, env: dict[str, str], source_rank: int, state_dir_override: str | None) -> None:
        hosts = env.get("RING_HOSTS", "").split()
        if len(hosts) != 4:
            raise ValueError(f"RING_HOSTS must list 4 nodes in physical cable order, got {len(hosts)}")
        if source_rank not in (2, 3):
            raise ValueError(f"--source-rank must be 2 or 3 (got {source_rank})")
        self.user = env.get("STAGE_SSH_USER", "")
        self.hosts = [ssh_target(h, self.user) for h in hosts]
        self.raw_hosts = hosts
        self.r0, self.r1, self.r2, self.r3 = self.hosts
        self.source_rank = source_rank

        self.pull_ip = env.get("STAGE_CABLE_PULL_IP", "")
        self.hop1_ip = env.get("STAGE_CABLE_HOP1_IP", "")
        self.hop2_ip = env.get("STAGE_CABLE_HOP2_IP", "")
        if not (self.pull_ip and self.hop1_ip and self.hop2_ip):
            raise ValueError("STAGE_CABLE_PULL_IP, STAGE_CABLE_HOP1_IP, and STAGE_CABLE_HOP2_IP must be set in ring.env")

        self.iface_prefix = env.get("STAGE_CABLE_IFACE_PREFIX", "enp1s0f")
        self.stage_dir = env.get("STAGE_DIR") or env.get("MODEL_DIR", "")
        if not self.stage_dir:
            raise ValueError("STAGE_DIR or MODEL_DIR must be set in ring.env")

        self.helper_dir = str(Path(self.stage_dir).parent)
        self.send_helper = f"{self.helper_dir}/.hy4_trunk_send.py"
        self.recv_helper = f"{self.helper_dir}/.hy4_trunk_recv.py"
        self.state_dir = Path(state_dir_override or env.get("STAGE_TRUNK_STATE_DIR", "/tmp/hy4-stage-trunk"))

    @property
    def src_host(self) -> str:
        return self.hosts[self.source_rank]

    @property
    def dst_indices(self) -> list[int]:
        return [1, 0, 3] if self.source_rank == 2 else [0, 1, 2]


def check_cable_routes(cfg: TrunkRingConfig) -> None:
    checks = [
        (cfg.r0, cfg.pull_ip, f"{cfg.raw_hosts[0]}->{cfg.raw_hosts[3]}"),
        (cfg.r0, cfg.hop1_ip, f"{cfg.raw_hosts[0]}->{cfg.raw_hosts[1]}"),
        (cfg.r1, cfg.hop2_ip, f"{cfg.raw_hosts[1]}->{cfg.raw_hosts[2]}"),
    ]
    summary = []
    for src_host, dst_ip, label in checks:
        p = run_ssh(src_host, f"ip route get {shlex.quote(dst_ip)} | head -1", timeout=15)
        if p.returncode != 0:
            raise RuntimeError(f"route check failed for {label} ({dst_ip}): {p.stderr.decode(errors='replace').strip()}")
        line = p.stdout.decode(errors="replace").strip()
        if f"dev {cfg.iface_prefix}" not in line or " via " in line:
            raise RuntimeError(f"FAIL-CLOSED non-cable route for {label} ({dst_ip}): {line}")
        dev = line.split("dev ", 1)[1].split()[0]
        summary.append(f"{label}={dst_ip}({dev})")
    log_line("routes_verified " + " ".join(summary))


def install_helpers(cfg: TrunkRingConfig) -> None:
    for h in cfg.hosts:
        for path, body in ((cfg.send_helper, SENDER_SCRIPT), (cfg.recv_helper, RECEIVER_SCRIPT)):
            p = run_ssh(
                h,
                f"mkdir -p {shlex.quote(cfg.helper_dir)} && cat > {shlex.quote(path)} && chmod +x {shlex.quote(path)}",
                input_bytes=body.encode("utf-8"),
                timeout=20,
            )
            if p.returncode != 0:
                raise RuntimeError(f"helper install failed on {h}: {p.stderr.decode(errors='replace')}")


def read_trunk_index(cfg: TrunkRingConfig, folder: str) -> dict:
    idx_path = f"{cfg.stage_dir}/{folder}/trunk.index.json"
    p = run_ssh(cfg.src_host, f"cat {shlex.quote(idx_path)}", timeout=20)
    if p.returncode != 0:
        raise RuntimeError(f"cannot read {idx_path} on {cfg.src_host}: {p.stderr.decode(errors='replace').strip()}")
    return json.loads(p.stdout.decode("utf-8"))


def batch_hash_on_host(host: str, folder_path: str, names: list[str]) -> dict[str, tuple[str, int]]:
    if not names:
        return {}
    quoted = " ".join(shlex.quote(n) for n in names)
    cmd = f"python3 -c {shlex.quote(BATCH_HASH_PY)} {shlex.quote(folder_path)} {quoted}"
    p = run_ssh(host, cmd, timeout=600)
    if p.returncode != 0:
        raise RuntimeError(f"batch hash failed on {host} ({folder_path}): {p.stderr.decode(errors='replace').strip()}")
    raw = json.loads(p.stdout.decode("utf-8"))
    return {k: (v[0], int(v[1])) for k, v in raw.items()}


def batch_hash_all_ranks(cfg: TrunkRingConfig, folder: str, names: list[str]) -> dict[int, dict[str, tuple[str, int]]]:
    folder_path = f"{cfg.stage_dir}/{folder}"
    out: dict[int, dict[str, tuple[str, int]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(batch_hash_on_host, cfg.hosts[i], folder_path, names): i for i in range(4)}
        for fut, idx in futs.items():
            out[idx] = fut.result()
    return out


def _parse_tokens(text: str) -> dict[str, str]:
    vals: dict[str, str] = {}
    for tok in text.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            if k in ("SRC_SHA256", "SRC_BYTES", "DST_SHA256", "DST_BYTES"):
                vals[k] = v
    return vals


def pull_hop(
    dst_host: str,
    src_cable_ip: str,
    user: str,
    send_helper: str,
    recv_helper: str,
    src_path: str,
    dst_final: str,
    dst_tmp: str,
    exp_sha: str,
    exp_bytes: int,
) -> float:
    src_target = ssh_target(src_cable_ip, user)
    t0 = time.monotonic()
    cmd = (
        f"set -euo pipefail; "
        f"rm -f {shlex.quote(dst_tmp)} {shlex.quote(dst_tmp + '.err')}; "
        f"ssh -T {FAST_CABLE_SSH_OPTS} {shlex.quote(src_target)} "
        f"python3 -u {shlex.quote(send_helper)} {shlex.quote(src_path)} "
        f"2>{shlex.quote(dst_tmp + '.err')} "
        f"| python3 -u {shlex.quote(recv_helper)} {shlex.quote(dst_tmp)}; "
        f"cat {shlex.quote(dst_tmp + '.err')}; rm -f {shlex.quote(dst_tmp + '.err')}"
    )
    p = run_ssh(dst_host, cmd, timeout=600)
    dt = time.monotonic() - t0
    out = p.stdout.decode(errors="replace") + "\n" + p.stderr.decode(errors="replace")
    toks = _parse_tokens(out)
    src_sha = toks.get("SRC_SHA256", "")
    dst_sha = toks.get("DST_SHA256", "")
    src_b = int(toks.get("SRC_BYTES", "-1"))
    dst_b = int(toks.get("DST_BYTES", "-1"))
    if (
        p.returncode != 0
        or not dst_sha
        or src_sha != exp_sha
        or dst_sha != exp_sha
        or src_b != exp_bytes
        or dst_b != exp_bytes
    ):
        run_ssh(dst_host, f"rm -f {shlex.quote(dst_tmp)}", timeout=15)
        raise RuntimeError(
            f"pull_hop failed ({src_cable_ip} -> {dst_host} {dst_final}): "
            f"rc={p.returncode} src={src_sha}({src_b}) dst={dst_sha}({dst_b}) exp={exp_sha}({exp_bytes})"
        )
    mv = run_ssh(dst_host, f"mv -f {shlex.quote(dst_tmp)} {shlex.quote(dst_final)}", timeout=15)
    if mv.returncode != 0:
        raise RuntimeError(f"atomic mv failed on {dst_host} for {dst_final}")
    return dt


def push_hop(
    src_host: str,
    dst_host: str,
    dst_cable_ip: str,
    user: str,
    recv_helper: str,
    src_path: str,
    dst_final: str,
    dst_tmp: str,
    exp_sha: str,
    exp_bytes: int,
) -> float:
    dst_target = ssh_target(dst_cable_ip, user)
    t0 = time.monotonic()
    cmd = (
        f"set -euo pipefail; "
        f"ssh -T {FAST_CABLE_SSH_OPTS} {shlex.quote(dst_target)} "
        f"\"rm -f {shlex.quote(dst_tmp)} && python3 -u {shlex.quote(recv_helper)} {shlex.quote(dst_tmp)}\" "
        f"< {shlex.quote(src_path)}"
    )
    p = run_ssh(src_host, cmd, timeout=600)
    dt = time.monotonic() - t0
    out = p.stdout.decode(errors="replace") + "\n" + p.stderr.decode(errors="replace")
    toks = _parse_tokens(out)
    dst_sha = toks.get("DST_SHA256", "")
    dst_b = int(toks.get("DST_BYTES", "-1"))
    if p.returncode != 0 or dst_sha != exp_sha or dst_b != exp_bytes:
        run_ssh(dst_host, f"rm -f {shlex.quote(dst_tmp)}", timeout=15)
        raise RuntimeError(
            f"push_hop failed ({src_host} -> {dst_cable_ip} {dst_final}): "
            f"rc={p.returncode} dst={dst_sha}({dst_b}) exp={exp_sha}({exp_bytes})"
        )
    mv = run_ssh(dst_host, f"mv -f {shlex.quote(dst_tmp)} {shlex.quote(dst_final)}", timeout=15)
    if mv.returncode != 0:
        raise RuntimeError(f"atomic mv failed on {dst_host} for {dst_final}")
    return dt


def stage_one_file(
    cfg: TrunkRingConfig,
    folder: str,
    fname: str,
    exp_sha: str,
    exp_bytes: int,
    have: dict[int, dict[str, tuple[str, int]]],
) -> tuple[bool, list[str]]:
    folder_path = f"{cfg.stage_dir}/{folder}"
    final_path = f"{folder_path}/{fname}"
    tmp_path = f"{folder_path}/.{fname}.tmp"
    hop_times: list[str] = []
    did_copy = False

    def ok(idx: int) -> bool:
        return have[idx].get(fname) == (exp_sha, exp_bytes)

    if cfg.source_rank == 2:
        # Spanning tree over the 3 verified direct 1-hop cables: r2 -> r1 -> r0 -> r3
        if not ok(1):
            dt = pull_hop(cfg.r1, cfg.hop2_ip, cfg.user, cfg.send_helper, cfg.recv_helper, final_path, final_path, tmp_path, exp_sha, exp_bytes)
            have[1][fname] = (exp_sha, exp_bytes)
            hop_times.append(f"r1={dt:.2f}s")
            did_copy = True
        if not ok(0):
            dt = pull_hop(cfg.r0, cfg.hop1_ip, cfg.user, cfg.send_helper, cfg.recv_helper, final_path, final_path, tmp_path, exp_sha, exp_bytes)
            have[0][fname] = (exp_sha, exp_bytes)
            hop_times.append(f"r0={dt:.2f}s")
            did_copy = True
        if not ok(3):
            dt = push_hop(cfg.r0, cfg.r3, cfg.pull_ip, cfg.user, cfg.recv_helper, final_path, final_path, tmp_path, exp_sha, exp_bytes)
            have[3][fname] = (exp_sha, exp_bytes)
            hop_times.append(f"r3={dt:.2f}s")
            did_copy = True
    else:
        # source_rank == 3: r3 -> r0 -> r1 -> r2
        if not ok(0):
            dt = pull_hop(cfg.r0, cfg.pull_ip, cfg.user, cfg.send_helper, cfg.recv_helper, final_path, final_path, tmp_path, exp_sha, exp_bytes)
            have[0][fname] = (exp_sha, exp_bytes)
            hop_times.append(f"r0={dt:.2f}s")
            did_copy = True
        if not ok(1):
            dt = push_hop(cfg.r0, cfg.r1, cfg.hop1_ip, cfg.user, cfg.recv_helper, final_path, final_path, tmp_path, exp_sha, exp_bytes)
            have[1][fname] = (exp_sha, exp_bytes)
            hop_times.append(f"r1={dt:.2f}s")
            did_copy = True
        if not ok(2):
            dt = push_hop(cfg.r1, cfg.r2, cfg.hop2_ip, cfg.user, cfg.recv_helper, final_path, final_path, tmp_path, exp_sha, exp_bytes)
            have[2][fname] = (exp_sha, exp_bytes)
            hop_times.append(f"r2={dt:.2f}s")
            did_copy = True

    return did_copy, hop_times


def process_folder(
    cfg: TrunkRingConfig,
    folder: str,
    target_layers: int,
    batch_size: int,
    pause_s: float,
    verify_only: bool,
) -> tuple[bool, dict]:
    idx_obj = read_trunk_index(cfg, folder)
    meta = idx_obj.get("metadata", {})
    layers_done = sorted(set(int(x) for x in meta.get("layers_done", [])))
    wm_files = set(idx_obj.get("weight_map", {}).values())
    all_layers_done = all(l in set(layers_done) for l in range(target_layers))
    extras_ready = (
        all_layers_done
        and "lm_head.safetensors" in wm_files
        and "dfly_layers.safetensors" in wm_files
    )

    layer_files = [f"model.layers.{l}.safetensors" for l in layers_done]
    final_files = ["lm_head.safetensors", "dfly_layers.safetensors", "trunk.index.json"] if extras_ready else []
    ordered_files = layer_files + final_files

    log_line(
        f"folder_plan folder={folder} src={cfg.raw_hosts[cfg.source_rank]} "
        f"layers_done={len(layers_done)}/{target_layers} extras_ready={int(extras_ready)} "
        f"files={len(ordered_files)} verify_only={int(verify_only)}"
    )

    if not verify_only:
        folder_path = f"{cfg.stage_dir}/{folder}"
        for i in cfg.dst_indices:
            p = run_ssh(cfg.hosts[i], f"mkdir -p {shlex.quote(folder_path)}", timeout=20)
            if p.returncode != 0:
                raise RuntimeError(f"mkdir failed on {cfg.hosts[i]}: {p.stderr.decode(errors='replace')}")

    have = batch_hash_all_ranks(cfg, folder, ordered_files)
    src_hashes = have[cfg.source_rank]
    for fname in ordered_files:
        if fname not in src_hashes:
            raise RuntimeError(f"source file missing on {cfg.src_host}: {cfg.stage_dir}/{folder}/{fname}")

    copied = 0
    skipped = 0
    copied_bytes = 0
    t_folder0 = time.monotonic()

    if not verify_only:
        batch_count = 0
        for fname in ordered_files:
            exp_sha, exp_bytes = src_hashes[fname]
            did_copy, hops = stage_one_file(cfg, folder, fname, exp_sha, exp_bytes, have)
            if did_copy:
                copied += 1
                copied_bytes += exp_bytes
                batch_count += 1
                log_line(f"staged folder={folder} file={fname} bytes={exp_bytes} sha256={exp_sha[:16]} {' '.join(hops)} status=OK")
                if batch_size > 0 and batch_count >= batch_size and pause_s > 0:
                    batch_count = 0
                    time.sleep(pause_s)
            else:
                skipped += 1

    # Full 4-node independent post-check hash of all eligible files
    final_hashes = batch_hash_all_ranks(cfg, folder, ordered_files)
    ref = final_hashes[cfg.source_rank]
    per_host_ok: dict[str, int] = {}
    all_ok = True
    mismatches: list[str] = []

    for idx in range(4):
        matched = 0
        for fname in ordered_files:
            if final_hashes[idx].get(fname) == ref.get(fname) and fname in ref:
                matched += 1
            else:
                all_ok = False
                mismatches.append(f"{cfg.raw_hosts[idx]}:{fname}")
        per_host_ok[cfg.raw_hosts[idx]] = matched

    # Guardrail: when incomplete, verify final metadata/extras were NOT staged on destination nodes
    if not extras_ready:
        guard = batch_hash_all_ranks(cfg, folder, ["lm_head.safetensors", "dfly_layers.safetensors", "trunk.index.json"])
        for idx in cfg.dst_indices:
            if guard[idx]:
                all_ok = False
                mismatches.append(f"{cfg.raw_hosts[idx]}:unexpected_final_files={sorted(guard[idx])}")

    dt_folder = time.monotonic() - t_folder0
    counts_str = " ".join(f"{h}={per_host_ok[h]}/{len(ordered_files)}" for h in cfg.raw_hosts)
    log_line(
        f"folder_done folder={folder} layers_done={len(layers_done)}/{target_layers} "
        f"extras_ready={int(extras_ready)} copied={copied} skipped={skipped} "
        f"copied_bytes={copied_bytes} wall_s={dt_folder:.2f} {counts_str} "
        f"status={'PASS' if all_ok else 'FAIL'}"
    )
    if mismatches:
        log_line(f"folder_mismatches folder={folder} items={mismatches[:10]}")

    summary = {
        "folder": folder,
        "source_host": cfg.raw_hosts[cfg.source_rank],
        "layers_done": layers_done,
        "layers_done_count": len(layers_done),
        "target_layers": target_layers,
        "extras_ready": extras_ready,
        "files_expected": len(ordered_files),
        "copied": copied,
        "skipped": skipped,
        "copied_bytes": copied_bytes,
        "wall_s": round(dt_folder, 2),
        "per_host_verified": per_host_ok,
        "status": "PASS" if all_ok else "FAIL",
        "files": {f: {"sha256": ref[f][0], "bytes": ref[f][1]} for f in ordered_files},
    }
    return all_ok, summary


# Link a staged trunk folder into the serving folder on one node. Hard links first (no extra disk, no
# copy); where the kernel refuses them (files owned by another user under fs.protected_hardlinks=1, or
# a different filesystem), per-file symlinks instead, which launch/launch.sh mounts. trunk.index.json
# goes last, so a half-linked folder never looks complete. Prints "LINKED <folder> hard|symlink".
LINK_SH = r'''set -eu
src="$1"; dst="$2"
[ -f "$src/trunk.index.json" ] || { echo "MISSING $src/trunk.index.json"; exit 3; }
mkdir -p "$dst"; mode=hard
for f in "$src"/*.safetensors "$src/trunk.index.json"; do
  b="${f##*/}"
  if [ -e "$dst/$b" ] && [ "$f" -ef "$dst/$b" ]; then continue; fi
  rm -f "$dst/$b"
  if ln "$f" "$dst/$b" 2>/dev/null; then :; else ln -sfn "$f" "$dst/$b"; fi
done
for f in "$dst"/*; do if [ -L "$f" ]; then mode=symlink; fi; done
echo "LINKED ${dst##*/} $mode"'''


def link_into(cfg: TrunkRingConfig, folders: list[str], dest_root: str) -> bool:
    ok = True
    for idx, host in enumerate(cfg.hosts):
        for folder in folders:
            src, dst = f"{cfg.stage_dir}/{folder}", f"{dest_root}/{folder}"
            p = run_ssh(host, f"bash -c {shlex.quote(LINK_SH)} _ {shlex.quote(src)} {shlex.quote(dst)}", timeout=120)
            out = (p.stdout.decode(errors="replace").strip() or p.stderr.decode(errors="replace").strip())
            if p.returncode != 0 or not out.startswith("LINKED"):
                ok = False
                log_line(f"link_fail host={cfg.raw_hosts[idx]} folder={folder} rc={p.returncode} out={out[:200]}")
                continue
            mode = out.split()[-1]
            note = " (hard links refused here; per-file symlinks used)" if mode == "symlink" else ""
            log_line(f"linked host={cfg.raw_hosts[idx]} {src} -> {dst} mode={mode}{note}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env-file", required=True, type=Path, help="Path to ring.env")
    ap.add_argument("--link-into", default=None, help="After a passing stage, link each folder into this serving folder on every node "
                    "(default: MODEL_DIR when STAGE_DIR is set and differs). Hard links, else per-file symlinks.")
    ap.add_argument("--folders", default="trunk-4b,trunk-6b", help="Comma-separated trunk sidecar folders under STAGE_DIR")
    ap.add_argument("--source-rank", type=int, default=2, choices=(2, 3), help="Rank index in RING_HOSTS that ran the trunk pass (default 2)")
    ap.add_argument("--target-layers", type=int, default=78, help="Total layer count required before copying extras and trunk.index.json")
    ap.add_argument("--batch-size", type=int, default=8, help="Files to copy before pausing (--pause-s) so cables yield to active jobs")
    ap.add_argument("--pause-s", type=float, default=2.0, help="Seconds to pause between batches (0 disables)")
    ap.add_argument("--verify-only", action="store_true", help="Hash already-staged files across all 4 nodes without copying")
    ap.add_argument("--state-dir", default=None, help="Local directory to write trunk_verified.json receipt")
    args = ap.parse_args()

    cfg = TrunkRingConfig(parse_env_file(args.env_file), args.source_rank, args.state_dir)
    folders = [x.strip() for x in args.folders.split(",") if x.strip()]
    if not folders:
        raise ValueError("--folders must name at least one folder")

    check_cable_routes(cfg)
    if not args.verify_only:
        install_helpers(cfg)

    all_ok = True
    report: dict[str, dict] = {}
    for folder in folders:
        ok, summary = process_folder(cfg, folder, args.target_layers, args.batch_size, args.pause_s, args.verify_only)
        report[folder] = summary
        if not ok:
            all_ok = False

    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.state_dir / ("trunk_verify_only.json" if args.verify_only else "trunk_verified.json")
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"updated_utc": utc_now(), "folders": report}, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, out_path)
    log_line(f"stage_trunk_done status={'PASS' if all_ok else 'FAIL'} receipt={out_path}")
    env = parse_env_file(args.env_file)
    dest = args.link_into or (env.get("MODEL_DIR") if env.get("STAGE_DIR") and env.get("MODEL_DIR") != env.get("STAGE_DIR") else None)
    if all_ok and dest and dest != cfg.stage_dir:
        all_ok = link_into(cfg, folders, dest)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
