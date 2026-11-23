#!/usr/bin/env python3
"""Continuous cable-only layer stager for a 4-node direct-cabled ring.

Reads all node names, cable IPs, and paths from a ring.env file (--env-file).
Polls the quantizing node (rank 3 in RING_HOSTS) for finished layers, pulls each
finished layer ONCE over the direct cable to rank 0 while computing source
SHA-256 in the same sequential read, relays rank 0 -> rank 1 -> rank 2 over
direct 1-hop cables with temporary files and SHA-256 verification before atomic
rename, symlinks on rank 3, and keeps config.json + model.safetensors.index.json
current across all four nodes.

Use --verify-only to check cable routes and hash every already-staged layer
across all four nodes against the source without modifying any remote files.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TRUNK_FILES = [
    "model-00053-of-00245.safetensors",
    "model-00061-of-00245.safetensors",
    "model-00124-of-00245.safetensors",
    "chat_template.jinja",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
]

DEFAULT_DRAFTER_FILE = "model-00245-of-00245.safetensors"

TRUNK_WEIGHT_MAP = {
    "model.embed_tokens.weight": "model-00053-of-00245.safetensors",
    "lm_head.weight": "model-00061-of-00245.safetensors",
    "model.hc_head.hc_head_base": "model-00124-of-00245.safetensors",
    "model.hc_head.hc_head_fn": "model-00124-of-00245.safetensors",
    "model.hc_head.hc_head_scale": "model-00124-of-00245.safetensors",
    "model.norm.weight": "model-00124-of-00245.safetensors",
}

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

STOP_REQUESTED = False


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_line(msg: str) -> None:
    print(f"{utc_now()} {msg}", flush=True)


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse KEY=VAL lines from a shell ring.env file without executing it."""
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


class RingConfig:
    def __init__(self, env: dict[str, str], state_dir_override: str | None) -> None:
        hosts = env.get("RING_HOSTS", "").split()
        if len(hosts) != 4:
            raise ValueError(f"RING_HOSTS must list 4 nodes in physical cable order, got {len(hosts)}")
        self.user = env.get("STAGE_SSH_USER", "")
        self.hosts = [ssh_target(h, self.user) for h in hosts]
        self.r0, self.r1, self.r2, self.r3 = self.hosts
        self.raw_hosts = hosts

        self.pull_ip = env.get("STAGE_CABLE_PULL_IP", "")
        self.hop1_ip = env.get("STAGE_CABLE_HOP1_IP", "")
        self.hop2_ip = env.get("STAGE_CABLE_HOP2_IP", "")
        if not (self.pull_ip and self.hop1_ip and self.hop2_ip):
            raise ValueError("STAGE_CABLE_PULL_IP, STAGE_CABLE_HOP1_IP, and STAGE_CABLE_HOP2_IP must be set in ring.env")

        self.iface_prefix = env.get("STAGE_CABLE_IFACE_PREFIX", "enp1s0f")
        self.stage_dir = env.get("STAGE_DIR") or env.get("MODEL_DIR", "")
        self.quant_state = env.get("QUANT_STATE_PATH", "")
        self.quant_layers = env.get("QUANT_LAYERS_DIR", "")
        self.orig_model = env.get("ORIG_MODEL_DIR", "")
        self.seed_dir = env.get("STAGE_SEED_DIR", "")
        self.drafter_file = env.get("DRAFTER_FILE", DEFAULT_DRAFTER_FILE)
        # Drafts are on by default (launch/launch.sh), so the drafter is staged unless STAGE_DRAFTER="0".
        stage_drafter = env.get("STAGE_DRAFTER") or "1"
        if stage_drafter not in ("0", "1"):
            raise ValueError(f'STAGE_DRAFTER="{stage_drafter}"; use "1" (stage the drafter, the default) or "0"')
        self.include_drafter = stage_drafter == "1"
        if not (self.stage_dir and self.quant_state and self.quant_layers and self.orig_model):
            raise ValueError("STAGE_DIR, QUANT_STATE_PATH, QUANT_LAYERS_DIR, and ORIG_MODEL_DIR must be set in ring.env")

        self.helper_dir = str(Path(self.stage_dir).parent)
        self.send_helper = f"{self.helper_dir}/.hy4_send.py"
        self.recv_helper = f"{self.helper_dir}/.hy4_recv.py"
        self.state_dir = Path(state_dir_override or env.get("STAGE_STATE_DIR", "/tmp/hy4-stage-state"))


def check_cable_routes(cfg: RingConfig) -> None:
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


def install_helpers(cfg: RingConfig) -> None:
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


def copy_file_across_ring(cfg: RingConfig, src_path_r3: str, rel_name: str) -> tuple[str, int, float, float, float]:
    dst_final = f"{cfg.stage_dir}/{rel_name}"
    dst_tmp = f"{cfg.stage_dir}/.{rel_name}.tmp"
    pull_target = ssh_target(cfg.pull_ip, cfg.user)
    hop1_target = ssh_target(cfg.hop1_ip, cfg.user)
    hop2_target = ssh_target(cfg.hop2_ip, cfg.user)

    t0 = time.monotonic()
    r0_pull = (
        f"set -euo pipefail; "
        f"rm -f {shlex.quote(dst_tmp)} {shlex.quote(dst_tmp + '.err')}; "
        f"ssh -T {FAST_CABLE_SSH_OPTS} {shlex.quote(pull_target)} "
        f"python3 -u {shlex.quote(cfg.send_helper)} {shlex.quote(src_path_r3)} "
        f"2>{shlex.quote(dst_tmp + '.err')} "
        f"| python3 -u {shlex.quote(cfg.recv_helper)} {shlex.quote(dst_tmp)}; "
        f"cat {shlex.quote(dst_tmp + '.err')}; rm -f {shlex.quote(dst_tmp + '.err')}"
    )
    p0 = run_ssh(cfg.r0, r0_pull, timeout=600)
    t_pull = time.monotonic() - t0
    out0 = p0.stdout.decode(errors="replace") + "\n" + p0.stderr.decode(errors="replace")
    if p0.returncode != 0:
        run_ssh(cfg.r0, f"rm -f {shlex.quote(dst_tmp)}", timeout=15)
        raise RuntimeError(f"pull r3->r0 failed for {rel_name}: {out0.strip()}")

    src_sha = dst_sha = ""
    src_bytes = dst_bytes = -1
    for tok in out0.split():
        if tok.startswith("SRC_SHA256="):
            src_sha = tok.split("=", 1)[1]
        elif tok.startswith("SRC_BYTES="):
            src_bytes = int(tok.split("=", 1)[1])
        elif tok.startswith("DST_SHA256="):
            dst_sha = tok.split("=", 1)[1]
        elif tok.startswith("DST_BYTES="):
            dst_bytes = int(tok.split("=", 1)[1])

    if not src_sha or src_sha != dst_sha or src_bytes <= 0 or src_bytes != dst_bytes:
        run_ssh(cfg.r0, f"rm -f {shlex.quote(dst_tmp)}", timeout=15)
        raise RuntimeError(f"SHA/size mismatch r3->r0 for {rel_name}: src={src_sha}({src_bytes}) dst={dst_sha}({dst_bytes})")
    if run_ssh(cfg.r0, f"mv -f {shlex.quote(dst_tmp)} {shlex.quote(dst_final)}", timeout=15).returncode != 0:
        raise RuntimeError(f"atomic mv failed on r0 for {rel_name}")

    t1 = time.monotonic()
    r0_to_r1 = (
        f"set -euo pipefail; "
        f"ssh -T {FAST_CABLE_SSH_OPTS} {shlex.quote(hop1_target)} "
        f"\"rm -f {shlex.quote(dst_tmp)} && python3 -u {shlex.quote(cfg.recv_helper)} {shlex.quote(dst_tmp)}\" "
        f"< {shlex.quote(dst_final)}"
    )
    p1 = run_ssh(cfg.r0, r0_to_r1, timeout=600)
    t_r1 = time.monotonic() - t1
    out1 = p1.stdout.decode(errors="replace") + "\n" + p1.stderr.decode(errors="replace")
    r1_sha, r1_bytes = "", -1
    for tok in out1.split():
        if tok.startswith("DST_SHA256="):
            r1_sha = tok.split("=", 1)[1]
        elif tok.startswith("DST_BYTES="):
            r1_bytes = int(tok.split("=", 1)[1])
    if p1.returncode != 0 or r1_sha != src_sha or r1_bytes != src_bytes:
        run_ssh(cfg.r1, f"rm -f {shlex.quote(dst_tmp)}", timeout=15)
        raise RuntimeError(f"relay r0->r1 failed for {rel_name}: sha={r1_sha}!={src_sha}")
    if run_ssh(cfg.r1, f"mv -f {shlex.quote(dst_tmp)} {shlex.quote(dst_final)}", timeout=15).returncode != 0:
        raise RuntimeError(f"atomic mv failed on r1 for {rel_name}")

    t2 = time.monotonic()
    r1_to_r2 = (
        f"set -euo pipefail; "
        f"ssh -T {FAST_CABLE_SSH_OPTS} {shlex.quote(hop2_target)} "
        f"\"rm -f {shlex.quote(dst_tmp)} && python3 -u {shlex.quote(cfg.recv_helper)} {shlex.quote(dst_tmp)}\" "
        f"< {shlex.quote(dst_final)}"
    )
    p2 = run_ssh(cfg.r1, r1_to_r2, timeout=600)
    t_r2 = time.monotonic() - t2
    out2 = p2.stdout.decode(errors="replace") + "\n" + p2.stderr.decode(errors="replace")
    r2_sha, r2_bytes = "", -1
    for tok in out2.split():
        if tok.startswith("DST_SHA256="):
            r2_sha = tok.split("=", 1)[1]
        elif tok.startswith("DST_BYTES="):
            r2_bytes = int(tok.split("=", 1)[1])
    if p2.returncode != 0 or r2_sha != src_sha or r2_bytes != src_bytes:
        run_ssh(cfg.r2, f"rm -f {shlex.quote(dst_tmp)}", timeout=15)
        raise RuntimeError(f"relay r1->r2 failed for {rel_name}: sha={r2_sha}!={src_sha}")
    if run_ssh(cfg.r2, f"mv -f {shlex.quote(dst_tmp)} {shlex.quote(dst_final)}", timeout=15).returncode != 0:
        raise RuntimeError(f"atomic mv failed on r2 for {rel_name}")

    return src_sha, src_bytes, t_pull, t_r1, t_r2


def ensure_trunk_and_dirs(cfg: RingConfig) -> None:
    for h in cfg.hosts:
        p = run_ssh(h, f"mkdir -p {shlex.quote(cfg.stage_dir)}", timeout=20)
        if p.returncode != 0:
            raise RuntimeError(f"mkdir failed on {h}: {p.stderr.decode(errors='replace')}")

    for h in cfg.hosts:
        src_dir = cfg.orig_model if h == cfg.r3 else cfg.seed_dir
        if not src_dir:
            continue
        script = "\n".join([
            "set -euo pipefail",
            f"for f in {' '.join(shlex.quote(x) for x in TRUNK_FILES)}; do",
            f"  dst={shlex.quote(cfg.stage_dir)}/\"$f\"",
            f"  src={shlex.quote(src_dir)}/\"$f\"",
            "  if [ ! -f \"$dst\" ] && [ -f \"$src\" ]; then",
            "    ln \"$src\" \"$dst\" 2>/dev/null || ln -sfn \"$src\" \"$dst\"",
            "  fi",
            "done",
        ])
        run_ssh(h, script, timeout=60)

    p3 = run_ssh(
        cfg.r3,
        f"cd {shlex.quote(cfg.stage_dir)} && sha256sum {' '.join(shlex.quote(x) for x in TRUNK_FILES)}",
        timeout=180,
    )
    if p3.returncode != 0:
        raise RuntimeError(f"r3 trunk sha256 failed: {p3.stderr.decode(errors='replace')}")
    ref_shas = {ln.split()[1]: ln.split()[0] for ln in p3.stdout.decode().splitlines() if len(ln.split()) >= 2}

    for fname in TRUNK_FILES:
        for h in (cfg.r0, cfg.r1, cfg.r2):
            if run_ssh(h, f"test -f {shlex.quote(cfg.stage_dir + '/' + fname)}", timeout=15).returncode != 0:
                copy_file_across_ring(cfg, f"{cfg.orig_model}/{fname}", fname)
                break

    for h in (cfg.r0, cfg.r1, cfg.r2):
        ph = run_ssh(
            h,
            f"cd {shlex.quote(cfg.stage_dir)} && sha256sum {' '.join(shlex.quote(x) for x in TRUNK_FILES)}",
            timeout=180,
        )
        if ph.returncode != 0:
            raise RuntimeError(f"trunk sha256 failed on {h}")
        got = {ln.split()[1]: ln.split()[0] for ln in ph.stdout.decode().splitlines() if len(ln.split()) >= 2}
        for fname in TRUNK_FILES:
            if got.get(fname) != ref_shas.get(fname):
                raise RuntimeError(f"trunk SHA mismatch on {h} for {fname}")
    log_line(f"trunk_verified files={len(TRUNK_FILES)} dir={cfg.stage_dir}")


def fetch_base_config(cfg: RingConfig) -> dict:
    p = run_ssh(cfg.r3, f"cat {shlex.quote(cfg.orig_model + '/config.json')}", timeout=20)
    if p.returncode != 0:
        raise RuntimeError(f"cannot read config.json on {cfg.r3}")
    return json.loads(p.stdout.decode())


def read_safetensors_header_keys(cfg: RingConfig, remote_path: str) -> list[str]:
    py = (
        "import json,struct,sys;"
        "f=open(sys.argv[1],'rb');"
        "n=struct.unpack('<Q',f.read(8))[0];"
        "h=json.loads(f.read(n));"
        "print(json.dumps(sorted(k for k in h if k!='__metadata__')))"
    )
    p = run_ssh(cfg.r0, f"python3 -c {shlex.quote(py)} {shlex.quote(remote_path)}", timeout=30)
    if p.returncode != 0:
        raise RuntimeError(f"header read failed on {cfg.r0} for {remote_path}")
    return json.loads(p.stdout.decode())


def read_layer_header_keys(cfg: RingConfig, layer_idx: int) -> list[str]:
    return read_safetensors_header_keys(cfg, f"{cfg.stage_dir}/model.layers.{layer_idx}.safetensors")


def ensure_drafter_staged(cfg: RingConfig) -> tuple[str, int, list[str]]:
    """Stage the drafter shard across all four nodes with full SHA-256 verification."""
    rel = cfg.drafter_file
    src_r3 = f"{cfg.orig_model}/{rel}"
    dst_r3 = f"{cfg.stage_dir}/{rel}"
    src_sha, src_bytes = hash_layer_on_host(cfg.r3, src_r3)
    if not src_sha or src_bytes <= 0:
        raise RuntimeError(f"drafter shard missing on {cfg.r3}: {src_r3}")

    # Check if already present with matching byte count on all 4 nodes
    all_present = True
    for h in cfg.hosts:
        ps = run_ssh(h, f"stat -L -c %s {shlex.quote(dst_r3)} 2>/dev/null || echo -1", timeout=15)
        if ps.stdout.decode().strip() != str(src_bytes):
            all_present = False
            break

    if not all_present:
        sha, nbytes, t_pull, t_r1, t_r2 = copy_file_across_ring(cfg, src_r3, rel)
        if sha != src_sha or nbytes != src_bytes:
            raise RuntimeError(f"drafter copy mismatch: got {sha}({nbytes}) != {src_sha}({src_bytes})")
        ln3 = run_ssh(cfg.r3, f"ln -sfn {shlex.quote(src_r3)} {shlex.quote(dst_r3)}", timeout=15)
        if ln3.returncode != 0:
            raise RuntimeError(f"drafter r3 symlink failed: {ln3.stderr.decode(errors='replace')}")
        log_line(
            f"drafter file={rel} bytes={nbytes} sha256={sha} "
            f"pull_s={t_pull:.2f} hop1_s={t_r1:.2f} hop2_s={t_r2:.2f} status=STAGED"
        )

    # Four-node full SHA-256 check (same bar as layers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(hash_layer_on_host, h, dst_r3) for h in cfg.hosts]
        results = [f.result() for f in futs]
    matched = sum(1 for sha, sz in results if sha == src_sha and sz == src_bytes)
    if matched != 4:
        raise RuntimeError(f"drafter 4-node SHA-256 verification failed ({matched}/4 matched {src_sha})")
    keys = read_safetensors_header_keys(cfg, dst_r3)
    log_line(f"drafter_verified file={rel} bytes={src_bytes} tensors={len(keys)} sha256={src_sha} nodes=4/4 status=OK")
    return src_sha, src_bytes, keys


def update_index_and_config(
    cfg: RingConfig,
    base_cfg: dict,
    n_contiguous: int,
    layer_keys_cache: dict[int, list[str]],
    layer_bytes_cache: dict[int, int],
    drafter_keys: list[str] | None = None,
    drafter_bytes: int = 0,
) -> None:
    if n_contiguous <= 0:
        return
    c = json.loads(json.dumps(base_cfg))
    for d in (c, c.get("text_config", {})):
        if isinstance(d, dict):
            if "num_hidden_layers" in d:
                d["num_hidden_layers"] = n_contiguous
            if "indexer_types" in d and isinstance(d["indexer_types"], list):
                d["indexer_types"] = d["indexer_types"][:n_contiguous]
            if "mlp_layer_types" in d and isinstance(d["mlp_layer_types"], list):
                d["mlp_layer_types"] = d["mlp_layer_types"][:n_contiguous]
            if not drafter_keys:
                if "dfly_num_hidden_layers" in d:
                    d["dfly_num_hidden_layers"] = 0
                if "dfly_target_index_layer" in d:
                    d["dfly_target_index_layer"] = []
                if "dfly_target_kv_layers" in d:
                    d["dfly_target_kv_layers"] = []

    weight_map = dict(TRUNK_WEIGHT_MAP)
    total_size = 0
    for l in range(n_contiguous):
        if l not in layer_keys_cache:
            layer_keys_cache[l] = read_layer_header_keys(cfg, l)
        fname = f"model.layers.{l}.safetensors"
        for k in layer_keys_cache[l]:
            weight_map[k] = fname
        total_size += layer_bytes_cache.get(l, 0)

    if drafter_keys:
        for k in drafter_keys:
            weight_map[k] = cfg.drafter_file
        total_size += drafter_bytes

    idx_obj = {
        "metadata": {
            "total_size": total_size,
            "staged_layers": n_contiguous,
            "drafter_staged": bool(drafter_keys),
        },
        "weight_map": weight_map,
    }
    cfg_bytes = (json.dumps(c, indent=2) + "\n").encode("utf-8")
    idx_bytes = (json.dumps(idx_obj, indent=2) + "\n").encode("utf-8")

    for h in cfg.hosts:
        for name, data in (("config.json", cfg_bytes), ("model.safetensors.index.json", idx_bytes)):
            dst = f"{cfg.stage_dir}/{name}"
            tmp = f"{cfg.stage_dir}/.{name}.tmp"
            p = run_ssh(h, f"cat > {shlex.quote(tmp)} && mv -f {shlex.quote(tmp)} {shlex.quote(dst)}", input_bytes=data, timeout=30)
            if p.returncode != 0:
                raise RuntimeError(f"failed installing {name} on {h}")


def read_layers_done(cfg: RingConfig) -> list[int]:
    p = run_ssh(cfg.r3, f"cat {shlex.quote(cfg.quant_state)}", timeout=20)
    if p.returncode != 0:
        raise RuntimeError(f"cannot read {cfg.quant_state} on {cfg.r3}")
    return sorted(set(int(x) for x in json.loads(p.stdout.decode()).get("layers_done", [])))


def hash_layer_on_host(host: str, path: str) -> tuple[str, int]:
    cmd = f"stat -L -c %s {shlex.quote(path)} && sha256sum {shlex.quote(path)}"
    p = run_ssh(host, cmd, timeout=300)
    if p.returncode != 0:
        return "", -1
    lines = p.stdout.decode(errors="replace").strip().splitlines()
    if len(lines) < 2:
        return "", -1
    return lines[1].split()[0], int(lines[0].strip())


def run_verify_only(cfg: RingConfig, state_dir: Path) -> int:
    """Read-only verification of all already-staged layers against the source on rank 3."""
    check_cable_routes(cfg)
    layers: list[int] = []
    p_idx = run_ssh(cfg.r0, f"cat {shlex.quote(cfg.stage_dir + '/model.safetensors.index.json')} 2>/dev/null || true", timeout=20)
    if p_idx.returncode == 0 and p_idx.stdout.strip():
        try:
            n_staged = int(json.loads(p_idx.stdout.decode()).get("metadata", {}).get("staged_layers", 0))
            if n_staged > 0:
                layers = list(range(n_staged))
        except Exception:
            layers = []
    if not layers:
        p = run_ssh(cfg.r0, f"ls {shlex.quote(cfg.stage_dir)}/model.layers.*.safetensors 2>/dev/null || true", timeout=20)
        for line in p.stdout.decode().splitlines():
            base = os.path.basename(line.strip())
            if base.startswith("model.layers.") and base.endswith(".safetensors"):
                try:
                    layers.append(int(base.split(".")[2]))
                except (IndexError, ValueError):
                    pass
        layers = sorted(set(layers))
    if not layers:
        log_line(f"verify_only staged_layers=0 dir={cfg.stage_dir} status=EMPTY")
        return 1

    state_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, dict] = {}
    all_ok = True

    for l in layers:
        rel = f"model.layers.{l}.safetensors"
        paths = [
            (cfg.r3, f"{cfg.quant_layers}/{rel}"),
            (cfg.r0, f"{cfg.stage_dir}/{rel}"),
            (cfg.r1, f"{cfg.stage_dir}/{rel}"),
            (cfg.r2, f"{cfg.stage_dir}/{rel}"),
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(hash_layer_on_host, h, pth) for h, pth in paths]
            results = [f.result() for f in futs]

        src_sha, src_bytes = results[0]
        matched = sum(1 for sha, sz in results if sha and sha == src_sha and sz == src_bytes)
        ok = (matched == 4 and src_bytes > 0)
        if not ok:
            all_ok = False
        status = "OK" if ok else "MISMATCH"
        log_line(f"verify layer={l:02d} bytes={src_bytes} sha256={src_sha} nodes={matched}/4 status={status}")
        record[str(l)] = {"sha256": src_sha, "bytes": src_bytes, "nodes_matched": matched, "status": status}

    if cfg.include_drafter:
        rel = cfg.drafter_file
        paths = [
            (cfg.r3, f"{cfg.orig_model}/{rel}"),
            (cfg.r0, f"{cfg.stage_dir}/{rel}"),
            (cfg.r1, f"{cfg.stage_dir}/{rel}"),
            (cfg.r2, f"{cfg.stage_dir}/{rel}"),
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            results = [f.result() for f in [ex.submit(hash_layer_on_host, h, pth) for h, pth in paths]]
        src_sha, src_bytes = results[0]
        matched = sum(1 for sha, sz in results if sha and sha == src_sha and sz == src_bytes)
        ok = (matched == 4 and src_bytes > 0)
        if not ok:
            all_ok = False
        status = "OK" if ok else "MISMATCH"
        log_line(f"verify drafter={rel} bytes={src_bytes} sha256={src_sha} nodes={matched}/4 status={status}")
        record["drafter"] = {"file": rel, "sha256": src_sha, "bytes": src_bytes, "nodes_matched": matched, "status": status}

    out_path = state_dir / "verify_only.json"
    tmp = state_dir / "verify_only.json.tmp"
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, out_path)
    log_line(f"verify_only_done layers={len(layers)} status={'PASS' if all_ok else 'FAIL'} state={out_path}")
    return 0 if all_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env-file", required=True, type=Path, help="Path to ring.env")
    ap.add_argument("--state-dir", default=None, help="Override local state directory (STOP file and verified.json)")
    ap.add_argument("--poll-s", type=int, default=30, help="Poll interval seconds")
    ap.add_argument("--target-layers", type=int, default=78, help="Stop when 0..target-1 are staged")
    ap.add_argument("--verify-only", action="store_true", help="Hash already-staged layers across all 4 nodes; change nothing")
    ap.add_argument(
        "--include-drafter",
        action="store_true",
        help="Stage the drafter shard on all 4 nodes and enable dfly_* in config.json + index (already the default unless STAGE_DRAFTER=\"0\"; changes pack hash)",
    )
    args = ap.parse_args()

    cfg = RingConfig(parse_env_file(args.env_file), args.state_dir)
    if args.include_drafter:
        cfg.include_drafter = True

    if args.verify_only:
        return run_verify_only(cfg, cfg.state_dir)

    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    stop_file = cfg.state_dir / "STOP"
    verified_file = cfg.state_dir / "verified.json"

    def handle_sig(signum, _frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
        log_line(f"signal={signum} stopping cleanly after current step")

    signal.signal(signal.SIGTERM, handle_sig)
    signal.signal(signal.SIGINT, handle_sig)

    log_line(f"stager_start pid={os.getpid()} stage_dir={cfg.stage_dir} stop_file={stop_file} target={args.target_layers} drafter={int(cfg.include_drafter)}")
    check_cable_routes(cfg)
    install_helpers(cfg)
    ensure_trunk_and_dirs(cfg)
    base_cfg = fetch_base_config(cfg)
    drafter_keys: list[str] | None = None
    drafter_bytes = 0
    if cfg.include_drafter:
        _, drafter_bytes, drafter_keys = ensure_drafter_staged(cfg)

    verified: dict[int, dict] = {}
    if verified_file.is_file():
        verified = {int(k): v for k, v in json.loads(verified_file.read_text(encoding="utf-8")).items()}
    layer_keys_cache: dict[int, list[str]] = {}
    layer_bytes_cache: dict[int, int] = {k: int(v["bytes"]) for k, v in verified.items() if "bytes" in v}
    last_published_n = 0

    def contig() -> int:
        n = 0
        while n in verified:
            n += 1
        return n

    while not STOP_REQUESTED:
        if stop_file.exists():
            log_line(f"stop_file_detected path={stop_file} exiting")
            break
        try:
            done_layers = read_layers_done(cfg)
        except Exception as exc:
            log_line(f"poll_error err={exc}")
            time.sleep(args.poll_s)
            continue

        did_work = False
        for l in done_layers:
            if STOP_REQUESTED or stop_file.exists():
                break
            if l in verified:
                fname = f"{cfg.stage_dir}/model.layers.{l}.safetensors"
                exp_b = int(verified[l]["bytes"])
                present = True
                for h in cfg.hosts:
                    ps = run_ssh(h, f"stat -L -c %s {shlex.quote(fname)} 2>/dev/null || echo -1", timeout=15)
                    if ps.stdout.decode().strip() != str(exp_b):
                        present = False
                        break
                if present:
                    continue

            rel = f"model.layers.{l}.safetensors"
            src_r3 = f"{cfg.quant_layers}/{rel}"
            try:
                sha, nbytes, t_pull, t_r1, t_r2 = copy_file_across_ring(cfg, src_r3, rel)
                ln3 = run_ssh(cfg.r3, f"ln -sfn {shlex.quote(src_r3)} {shlex.quote(cfg.stage_dir + '/' + rel)}", timeout=15)
                if ln3.returncode != 0:
                    raise RuntimeError(f"r3 symlink failed: {ln3.stderr.decode(errors='replace')}")
                verified[l] = {"sha256": sha, "bytes": nbytes, "staged_utc": utc_now()}
                layer_bytes_cache[l] = nbytes
                tmp = verified_file.with_suffix(".json.tmp")
                tmp.write_text(json.dumps({str(k): v for k, v in sorted(verified.items())}, indent=2) + "\n", encoding="utf-8")
                os.replace(tmp, verified_file)

                n_contig = contig()
                if n_contig > last_published_n:
                    update_index_and_config(cfg, base_cfg, n_contig, layer_keys_cache, layer_bytes_cache, drafter_keys, drafter_bytes)
                    last_published_n = n_contig

                log_line(
                    f"layer={l:02d} bytes={nbytes} sha256={sha} "
                    f"pull_s={t_pull:.2f} hop1_s={t_r1:.2f} hop2_s={t_r2:.2f} "
                    f"staged_contiguous={n_contig} status=OK"
                )
                did_work = True
            except Exception as exc:
                log_line(f"layer={l:02d} status=ERROR err={exc}")
                time.sleep(10)

        n_contig = contig()
        if n_contig > last_published_n:
            update_index_and_config(cfg, base_cfg, n_contig, layer_keys_cache, layer_bytes_cache, drafter_keys, drafter_bytes)
            last_published_n = n_contig
        if n_contig >= args.target_layers:
            log_line(f"all_layers_staged contiguous={n_contig} target={args.target_layers} exiting")
            break
        if not did_work:
            for _ in range(args.poll_s):
                if STOP_REQUESTED or stop_file.exists():
                    break
                time.sleep(1)

    log_line(f"stager_exit staged_contiguous={contig()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
