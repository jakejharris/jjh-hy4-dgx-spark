#!/usr/bin/env python3
"""Collect systems receipts for TensorFold TP=4 boots from launch/rank logs.

Reads a launch log directory (rank0..rank3 logs, optional launch_meta.json and
memfloor_r0..r3.log written by launch/launch.sh, plus optional --launch-log,
--check-load-dir, and --known-answer) and writes:
  - <out-dir>/summary.json
  - <out-dir>/systems.weights.r{0..3}.<build>.json
  - <out-dir>/systems.memory.r{0..3}.<build>.json
  - <out-dir>/systems.kv.pool.<build>.json
  - <out-dir>/systems.kv.bytes.<build>.json
  - <out-dir>/systems.load.cold.<build>.json
  - <out-dir>/systems.launch.<build>.json

<build> is "exl3" for the full quant and "lossless" for the lossless trunk, and each file's claim_id carries the
same suffix. The build comes from trunk= on the rank load lines (trunk=lossless is the lossless trunk), else from
TF_HY4_TRUNK in the boot's settings (empty is the lossless trunk); --build sets it and must agree with both.

Receipts are meant to ship, so every file is redacted as it is written: the
machine host names (RING_HOSTS and the host in each rank<r>_<host>.log name)
become node1..node4 in rank order, and home folders become ~. Values are not
touched. To redact receipts written by an older copy of this script, in place:

    python3 launch/collect_receipts.py --redact DIR [--hosts "h1 h2 h3 h4"]

(--hosts defaults to RING_HOSTS from DIR/summary.json.)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

GIB = float(1 << 30)

# Host name -> node<k>, filled in rank order before anything is written.
HOST_MAP: dict[str, str] = {}
# Assembled so this file does not contain the string it removes.
HOME_RE = re.compile("/ho" + r"me/[^/\s\"']+")


def set_host_map(hosts: list[str]) -> None:
    for i, h in enumerate(hosts):
        h = h.strip()
        if h and h not in HOST_MAP and not re.fullmatch(r"node[1-4]", h):
            HOST_MAP[h] = f"node{i + 1}"


def redact(text: str) -> tuple[str, int]:
    """Replace mapped host names and home folders; return the text and the count replaced."""
    total = 0
    for h in sorted(HOST_MAP, key=len, reverse=True):
        pat = re.compile(r"(?<![A-Za-z0-9.-])" + re.escape(h) + r"(?![A-Za-z0-9-])")
        text, n = pat.subn(HOST_MAP[h], text)
        total += n
    text, n = HOME_RE.subn("~", text)
    return text, total + n

# The load line every rank prints. Its weights= is the device memory the model and drafter took at load,
# before the KV pool and scratch: the kit's "weights per rank".
RE_HY4_LOADED = re.compile(
    r"info:\s+Hy4\s+\((?P<model_type>[^,]+),\s*tp=(?P<tp>\d+)\s+rank=(?P<rank>\d+)\)\s+"
    r"loaded in\s+(?P<load_s>[0-9.]+)\s*s:\s*(?P<layers>\d+)\s+layers,\s*"
    r"(?P<streams>\d+)\s+streams?,\s*(?P<kv_gib>[0-9.]+)\s+GiB lossless BF16 KV/stream at\s+(?P<window>\d+)\s+window"
    r"(?P<fields>[^\n]*)"
)
# The ", key=value" fields after "window". A value runs to the next ", key=" or the end of the line, so a
# field the engine adds (the release engine ends with repeat_stable=on|off) never cuts the one before it.
RE_LOAD_FIELD = re.compile(r",\s*(?P<key>[a-z_]+)=(?P<value>.*?)(?=,\s*[a-z_]+=|$)")
RE_LEADING_GIB = re.compile(r"(?P<gib>[0-9.]+)\s*GiB")
RE_WEIGHT_BYTES = re.compile(r"weight_bytes=(?P<n>\d+)")

# The serve banner, rank 0 only. Its "model" figure is what that rank holds once serving: on the 128K boot it was
# the load line's weights plus the KV pool plus 0.44 GiB (87.29 = 75.22 + 11.63 + 0.44 at 128K), so it grows
# with the context window. It is kept as serve_resident_gib, never as weights.
RE_CUDA_STARTUP = re.compile(
    r"\[tensorfold\]\s+CUDA\s+sm_(?P<sm>\d+)\s+device\s+(?P<dev>\d+)\s+\((?P<device_name>[^)]+)\):\s+"
    r"model\s+(?P<model_gib>[0-9.]+)\s+GiB[^;]*;\s+"
    r"(?P<streams>\d+)\s+streams?\s+at once,\s+(?P<kv_gib>[0-9.]+)\s+GiB each at a\s+(?P<window>\d+)-token window,\s+"
    r"of\s+(?P<free_after_reserve_gib>[0-9.]+)\s+GiB left after a\s+(?P<reserve_gib>[0-9.]+)\s+GiB reserve"
)

# Its "sampling:" part is the server's defaults for a request that sends none ("greedy", or name-value pairs).
RE_SERVING = re.compile(
    r"\[tensorfold\]\s+serving\s+(?P<model_id>\S+)\s+at\s+(?P<url>\S+)\s+\("
    r"(?:sampling:\s*(?P<sampling>[^;)]*);\s*)?"
    r"[^)]*drafts:\s*(?P<drafts>\w+);\s*context:\s*(?P<context>\d+);\s*loaded in\s+(?P<serve_load_s>[0-9.]+)s\)"
)

RE_CHECK_LOAD = re.compile(
    r"hy4 check-load PASS\s+\(tp=(?P<tp>\d+)\s+rank=(?P<rank>\d+)\):.*?"
    r"placement=(?P<placement_gib>[0-9.]+)\s+GiB,\s*MemAvailable\s+(?P<mem_avail_gib>[0-9.]+)\s+GiB"
)

RE_PREFLIGHT_RANK = re.compile(
    r"=== Preflight rank (?P<rank>\d+) \((?P<host>[^)]+)\) ==="
)
RE_PREFLIGHT_OK = re.compile(
    r"OK\s+rdma=(?P<rdma>\S+).*?memlock=(?P<memlock>\S+)\s+avail=(?P<avail_gib>[0-9.]+)GiB"
)
RE_MEM_SAMPLE = re.compile(r"avail_gib=(?P<avail>[0-9.]+)")
RE_MEM_SUMMARY = re.compile(
    r"min_avail_gib=(?P<min_avail>[0-9.]+)\s+floor_gib=(?P<floor>[0-9.]+)\s+stopped=(?P<stopped>\d+)"
    r"(?:\s+min_utc=(?P<min_utc>\S+))?"
)


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
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        env[k] = v
    return env


def parse_rank_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    info: dict = {
        "log_file": path.name,
        "nccl_ib_used": "Using network IB" in text,
        "nccl_socket_used": "Using network Socket" in text,
    }
    nv = re.search(r"NCCL INFO NCCL version (\S+)", text)
    if nv:
        info["nccl_version"] = nv.group(1)
    warns = {}
    for key, names in (("jpeg", ("libturbojpeg.so.0", "libjpeg-turbo")), ("webp", ("libwebpdecoder.so.3", "libwebp"))):
        hit = next((ln.strip() for ln in text.splitlines() if "warning" in ln.lower() and any(n in ln for n in names)), None)
        if hit:
            warns[key] = hit[:200]
    info["codec_warnings"] = warns
    m = RE_HY4_LOADED.search(text)
    if m:
        info["model_type"] = m.group("model_type")
        info["tp"] = int(m.group("tp"))
        info["rank"] = int(m.group("rank"))
        info["load_s"] = float(m.group("load_s"))
        info["layers"] = int(m.group("layers"))
        info["streams"] = int(m.group("streams"))
        info["kv_gib_per_stream"] = float(m.group("kv_gib"))
        info["context_window"] = int(m.group("window"))
        fields = {f.group("key"): f.group("value").strip() for f in RE_LOAD_FIELD.finditer(m.group("fields"))}
        g = RE_LEADING_GIB.match(fields.get("weights", ""))
        if g:
            info["weights_gib"] = float(g.group("gib"))
        b = RE_WEIGHT_BYTES.search(fields.get("weights", ""))
        if b:
            info["weight_bytes"] = int(b.group("n"))
        g = RE_LEADING_GIB.match(fields.get("room_after_reserve", ""))
        if g:
            info["free_after_reserve_gib"] = float(g.group("gib"))
        for key, name in (("trunk", "trunk_loaded"), ("drafter", "drafter_loaded"), ("head", "head_loaded"),
                          ("repeat_stable", "repeat_stable_loaded")):
            if fields.get(key):
                info[name] = fields[key]

    mc = RE_CUDA_STARTUP.search(text)
    if mc:
        info["cuda_sm"] = int(mc.group("sm"))
        info["device_name"] = mc.group("device_name")
        info["serve_resident_gib"] = float(mc.group("model_gib"))
        info["serve_free_after_reserve_gib"] = float(mc.group("free_after_reserve_gib"))
        info["reserve_gib"] = float(mc.group("reserve_gib"))
        info.setdefault("streams", int(mc.group("streams")))
        info.setdefault("kv_gib_per_stream", float(mc.group("kv_gib")))
        info.setdefault("context_window", int(mc.group("window")))

    ms = RE_SERVING.search(text)
    if ms:
        info["model_id"] = ms.group("model_id")
        if ms.group("sampling"):
            info["sampling"] = ms.group("sampling").strip()
        info["drafts"] = ms.group("drafts")
        info["serve_context"] = int(ms.group("context"))
        info["serve_load_s"] = float(ms.group("serve_load_s"))

    return info


def parse_preflight_log(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    cur_rank: int | None = None
    cur_host = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        mr = RE_PREFLIGHT_RANK.search(line)
        if mr:
            cur_rank = int(mr.group("rank"))
            cur_host = mr.group("host")
            continue
        mo = RE_PREFLIGHT_OK.search(line)
        if mo and cur_rank is not None:
            out[cur_rank] = {
                "host": cur_host,
                "rdma": mo.group("rdma"),
                "memlock": mo.group("memlock"),
                "preflight_avail_gib": float(mo.group("avail_gib")),
            }
            cur_rank = None
    return out


def parse_memfloor_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = list(RE_MEM_SUMMARY.finditer(text))
    if matches:
        ms = matches[-1]
        return {
            "source": "continuous_memfloor",
            "min_avail_gib": float(ms.group("min_avail")),
            "floor_gib": float(ms.group("floor")),
            "stopped": int(ms.group("stopped")),
            "min_utc": ms.group("min_utc"),
        }
    vals = [float(m.group("avail")) for m in RE_MEM_SAMPLE.finditer(text)]
    if vals:
        return {
            "source": "continuous_memfloor_samples",
            "min_avail_gib": min(vals),
            "samples": len(vals),
            "stopped": 0,
        }
    return {}


def parse_check_load_dir(d: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for p in sorted(d.glob("*.log")):
        text = p.read_text(encoding="utf-8", errors="replace")
        for m in RE_CHECK_LOAD.finditer(text):
            rk = int(m.group("rank"))
            out[rk] = {
                "placement_gib": float(m.group("placement_gib")),
                "mem_avail_gib": float(m.group("mem_avail_gib")),
                "log_file": p.name,
            }
    return out


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text, _ = redact(json.dumps(obj, indent=2) + "\n")
    path.write_text(text, encoding="utf-8")


def redact_dir(d: Path, hosts: str | None) -> int:
    """Redact receipt JSON files already on disk, in place."""
    if hosts is None:
        summary = d / "summary.json"
        if not summary.is_file():
            raise SystemExit(f"ERROR: {summary} not found; pass --hosts \"h1 h2 h3 h4\" in rank order")
        hosts = (json.loads(summary.read_text(encoding="utf-8")).get("launch_settings") or {}).get("RING_HOSTS")
        if not hosts:
            raise SystemExit("ERROR: no RING_HOSTS in summary.json; pass --hosts \"h1 h2 h3 h4\" in rank order")
    set_host_map(hosts.split())
    total = 0
    for p in sorted(d.glob("*.json")):
        text, n = redact(p.read_text(encoding="utf-8"))
        if n:
            json.loads(text)  # still valid JSON
            p.write_text(text, encoding="utf-8")
            print(f"redacted {p.name}: {n}")
            total += n
    print(f"redacted {total} occurrences in {d} (hosts -> {', '.join(f'{k}={v}' for k, v in HOST_MAP.items())})"
          if total else f"nothing to redact in {d}")
    return 0


BUILDS = ("exl3", "lossless")


def engine_receipt(launch_meta: dict, env: dict) -> dict:
    """What ran: per-rank SHA-256 of the mounted tensorfold-native and graphics-clock snapshots (launch/launch.sh
    reads both on each Spark), with the declared clock cap and engine commit. A missing value says so."""
    shas = launch_meta.get("bin_sha256_by_rank") or {}
    known = [v for v in shas.values() if v and v != "unknown"]
    cap = launch_meta.get("GPU_CLOCK_CAP_MHZ", env.get("GPU_CLOCK_CAP_MHZ"))
    return {
        "engine_commit": launch_meta.get("ENGINE_COMMIT", env.get("ENGINE_COMMIT")) or "not declared",
        "tensorfold_bin": launch_meta.get("TENSORFOLD_BIN", env.get("TENSORFOLD_BIN")),
        "bin_sha256_by_rank": shas or "not recorded (launcher older than this kit)",
        "bin_sha256_same_on_all_ranks": bool(shas) and len(known) == len(shas) == 4 and len(set(known)) == 1,
        "gpu_clock_cap_mhz": cap if cap else "none declared (stock clocks)",
        "gpu_clock_mhz_at_launch_by_rank": launch_meta.get("gpu_clock_mhz_at_launch_by_rank") or "not recorded",
        "gpu_clock_mhz_at_ready_by_rank": launch_meta.get("gpu_clock_mhz_at_ready_by_rank") or "not recorded",
        "clock_note": "single nvidia-smi clocks.gr readings per rank, before launch and when the server was ready; not a trace",
        "problems": engine_problems(launch_meta),
    }


def image_decoders(r0: dict, launch_settings: dict) -> dict:
    """Whether rank 0, the rank that decodes images, found libturbojpeg.so.0 and libwebpdecoder.so.3 at boot. Started
    with --vision, the engine names each missing or unusable decoder in a warning; no warning means it was found."""
    if str(launch_settings.get("SERVE_VISION")) != "1":
        return {"jpeg": "not checked (vision off)", "webp": "not checked (vision off)"}
    if not r0:
        return {"jpeg": "not checked (no rank 0 log)", "webp": "not checked (no rank 0 log)"}
    w = r0.get("codec_warnings") or {}
    return {k: ("missing: " + w[k]) if w.get(k) else "found" for k in ("jpeg", "webp")}


def engine_problems(launch_meta: dict) -> list[str]:
    """Each read that did not give a value, by rank: a missing or malformed hash, a clock that is not a number."""
    out = []
    shas = launch_meta.get("bin_sha256_by_rank") or {}
    if not shas:
        out.append("no per-rank binary hashes in launch_meta.json")
    for r, v in sorted(shas.items()):
        if not re.fullmatch(r"[0-9a-f]{64}", str(v)):
            out.append(f"rank {r}: binary hash not read ({v})")
    if len({str(v) for v in shas.values()}) > 1:
        out.append("ranks hold different binaries")
    for key, when in (("gpu_clock_mhz_at_launch_by_rank", "launch"), ("gpu_clock_mhz_at_ready_by_rank", "ready")):
        clk = launch_meta.get(key) or {}
        if not clk:
            out.append(f"no graphics-clock snapshots at {when}")
        for r, v in sorted(clk.items()):
            if not str(v).isdigit():
                out.append(f"rank {r}: graphics clock at {when} not read ({v})")
    return out


def boot_build(rank_logs: dict, launch_settings: dict, override: str | None) -> str:
    """exl3 or lossless: from trunk= on the load lines, else TF_HY4_TRUNK (empty is lossless); --build must agree."""
    trunks = {rl.get("trunk_loaded") for rl in rank_logs.values()} - {None}
    if len(trunks) > 1:
        raise SystemExit(f"ERROR: the ranks loaded different trunks: {sorted(trunks)}")
    found = None
    if trunks:
        found = "lossless" if trunks.pop() == "lossless" else "exl3"
    elif launch_settings.get("TF_HY4_TRUNK") is not None:
        found = "exl3" if str(launch_settings["TF_HY4_TRUNK"]).strip() else "lossless"
    if override and found and override != found:
        raise SystemExit(f"ERROR: --build {override}, but the boot is {found} (trunk= on the load lines, else TF_HY4_TRUNK)")
    if not (override or found):
        raise SystemExit("ERROR: no trunk= on the load lines and no TF_HY4_TRUNK; pass --build exl3 or --build lossless")
    return override or found


def main() -> int:
    if "--redact" in sys.argv:
        ap = argparse.ArgumentParser(description="Redact existing receipt files in place.")
        ap.add_argument("--redact", type=Path, required=True)
        ap.add_argument("--hosts", default=None)
        a = ap.parse_args()
        return redact_dir(a.redact, a.hosts)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logdir", type=Path, required=True, help="Directory with rank0..rank3 logs from launch.sh")
    ap.add_argument("--env-file", type=Path, default=None, help="ring.env used for the boot")
    ap.add_argument("--launch-log", type=Path, default=None, help="Optional stdout/stderr log from launch.sh")
    ap.add_argument("--check-load-dir", type=Path, default=None, help="Optional dir with tf-hy4-oracle --check-load logs")
    ap.add_argument("--known-answer", type=Path, default=None, help="Optional JSON response from known-answer request")
    ap.add_argument("--label", default="boot", help="Run label (e.g. 16l-rehearsal, 78l-boot1-4k)")
    ap.add_argument("--isolation", choices=("quiet", "shared"), default="shared", help="Cluster isolation state")
    ap.add_argument("--cold-ready-s", type=float, default=None, help="Override elapsed launch-to-ready wall seconds")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory for receipt JSON files")
    ap.add_argument("--build", choices=BUILDS, default=None,
                    help="Build suffix for the per-claim files (default: from trunk= on the load lines)")
    args = ap.parse_args()

    env = parse_env_file(args.env_file) if args.env_file else {}
    meta_path = args.logdir / "launch_meta.json"
    launch_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    preflight = parse_preflight_log(args.launch_log) if (args.launch_log and args.launch_log.is_file()) else {}
    check_load = parse_check_load_dir(args.check_load_dir) if (args.check_load_dir and args.check_load_dir.is_dir()) else {}

    rank_logs: dict[int, dict] = {}
    for r in range(4):
        matches = sorted(args.logdir.glob(f"rank{r}_*.log"))
        if matches:
            rank_logs[r] = parse_rank_log(matches[0])
            rank_logs[r].setdefault("rank", r)
            # Extract host from rank<r>_<host>.log
            stem = matches[0].stem
            parts = stem.split("_", 1)
            rank_logs[r]["host"] = parts[1] if len(parts) == 2 else f"rank{r}"

    if not rank_logs:
        raise SystemExit(f"ERROR: no rank*_*.log files found in {args.logdir}")

    # Host names never reach a receipt: map RING_HOSTS (else the rank-log hosts) to node1..node4.
    ring = str(launch_meta.get("RING_HOSTS") or env.get("RING_HOSTS") or "").split()
    set_host_map(ring if len(ring) == 4 else [rank_logs[r]["host"] for r in sorted(rank_logs)])
    set_host_map([rank_logs[r]["host"] for r in sorted(rank_logs)])

    # Merge check-load placement GiB when follower serve logs do not print resident weight GiB
    for r, rl in rank_logs.items():
        if "weights_gib" not in rl and r in check_load:
            rl["weights_gib"] = check_load[r]["placement_gib"]
            rl["weights_source"] = "check_load"
        elif "weights_gib" in rl:
            rl["weights_source"] = "rank_log"
        if r in preflight:
            rl["preflight_avail_gib"] = preflight[r]["preflight_avail_gib"]
        mf_path = args.logdir / f"memfloor_r{r}.log"
        if mf_path.is_file():
            mf = parse_memfloor_log(mf_path)
            if mf:
                rl["memfloor"] = mf

    ib_ranks = sum(1 for rl in rank_logs.values() if rl.get("nccl_ib_used") and not rl.get("nccl_socket_used"))
    load_times = [rl["load_s"] for rl in rank_logs.values() if "load_s" in rl]
    max_rank_load_s = max(load_times) if load_times else None
    r0 = rank_logs.get(0, {})
    serve_load_s = r0.get("serve_load_s")
    cold_ready_s = args.cold_ready_s
    if cold_ready_s is None and launch_meta.get("cold_ready_s") is not None:
        cold_ready_s = float(launch_meta["cold_ready_s"])
    if cold_ready_s is None:
        cold_ready_s = serve_load_s if serve_load_s is not None else max_rank_load_s

    streams = r0.get("streams")
    ctx_win = r0.get("context_window") or (int(env["CONTEXT_LEN"]) if "CONTEXT_LEN" in env else None)
    kv_gib_stream = r0.get("kv_gib_per_stream")
    kv_pool_tokens = (streams * ctx_win) if (streams and ctx_win) else None
    kv_bytes_per_tok = round((kv_gib_stream * GIB) / ctx_win, 1) if (kv_gib_stream and ctx_win) else None

    known_answer_info = None
    if args.known_answer and args.known_answer.is_file():
        raw_ka = args.known_answer.read_text(encoding="utf-8", errors="replace")
        ka_json = None
        try:
            ka_json = json.loads(raw_ka)
        except json.JSONDecodeError:
            for line in raw_ka.splitlines():
                s = line.strip()
                if s.startswith("{") and s.endswith("}"):
                    try:
                        ka_json = json.loads(s)
                        break
                    except json.JSONDecodeError:
                        pass
        if isinstance(ka_json, dict):
            choices = ka_json.get("choices") or []
            msg = (choices[0].get("message") or {}) if choices else {}
            known_answer_info = {
                "ok": True,
                "model": ka_json.get("model"),
                "usage": ka_json.get("usage"),
                "tensorfold": ka_json.get("tensorfold"),
                "content_preview": (msg.get("content") or "")[:120],
            }
        else:
            known_answer_info = {"ok": "HTTP_CODE=200" in raw_ka, "raw_bytes": len(raw_ka)}

    launch_settings = {
        k: launch_meta.get(k, env.get(k))
        for k in (
            "RING_HOSTS", "CONTEXT_LEN", "SERVE_PARALLEL", "SERVE_DRAFTS", "DRAFTER_DIR",
            "TIMEOUT_S", "TF_HY4_TP_TIMEOUT_S", "TF_HY4_LOAD_TIMEOUT_S",
            "TENSORFOLD_MEMORY_RESERVE_GIB", "TENSORFOLD_CUDA_MEMORY_LIMIT_GB",
            "MIN_AVAIL_GIB", "MEMFLOOR_GIB",
            "TF_HY4_TRUNK", "HY4_TRUNK_FAST",
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
        )
        if launch_meta.get(k, env.get(k)) is not None
    }
    engine = engine_receipt(launch_meta, env)
    engine["nccl_version_by_rank"] = {str(r): rank_logs[r].get("nccl_version", "not printed") for r in sorted(rank_logs)}
    engine["image_decoders"] = image_decoders(rank_logs.get(0) or {}, launch_settings)
    for p in engine["problems"]:
        print(f"WARNING: engine receipt: {p}", file=sys.stderr)
    vision_info = None
    vpath = args.logdir / "vision_check.json"
    if vpath.is_file():
        v = json.loads(vpath.read_text(encoding="utf-8"))
        vision_info = {"file": "vision_check.json", "known_pass": v.get("known_pass"), "known_total": v.get("known_total"),
                       "server_alive_after": v.get("server_alive_after"), "utc_start": v.get("utc_start"),
                       "utc_end": v.get("utc_end")}

    build = boot_build(rank_logs, launch_settings, args.build)
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary = {
        "schema": 1,
        "generated_utc": now_utc,
        "build": build,
        "label": args.label,
        "isolation": args.isolation,
        "ranks_present": sorted(rank_logs),
        "nccl_ib_verified_ranks": ib_ranks,
        "cold_ready_s": cold_ready_s,
        "serve_load_s": serve_load_s,
        "max_rank_load_s": max_rank_load_s,
        "layers": r0.get("layers"),
        "streams": streams,
        "context_window": ctx_win,
        "kv_pool_tokens": kv_pool_tokens,
        "kv_gib_per_stream": kv_gib_stream,
        "kv_bytes_per_token_est": kv_bytes_per_tok,
        "launch_settings": launch_settings,
        "engine": engine,
        "known_answer": known_answer_info,
        "vision_check": vision_info,
        "ranks": {str(r): rank_logs[r] for r in sorted(rank_logs)},
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "summary.json", summary)
    if vision_info:
        vtext, _ = redact(vpath.read_text(encoding="utf-8"))
        (args.out_dir / "vision_check.json").write_text(vtext, encoding="utf-8")

    # Emit per-claim JSON files matching K/card/claims.json receipt paths
    for r in range(4):
        rl = rank_logs.get(r, {})
        w_val = rl.get("weights_gib")
        write_json(
            args.out_dir / f"systems.weights.r{r}.{build}.json",
            {
                "claim_id": f"systems.weights.r{r}.{build}",
                "generated_utc": now_utc,
                "label": args.label,
                "isolation": args.isolation,
                "rank": r,
                "unit": "GiB",
                "value": w_val,
                "weight_bytes": rl.get("weight_bytes"),
                "source": rl.get("weights_source", "missing"),
                "status": "measured" if w_val is not None else "pending",
                "note": None if w_val is not None else "no weights= on this rank's load line; supply --check-load-dir or per-rank weight log",
            },
        )
        mf = rl.get("memfloor") or {}
        mem_val = mf.get("min_avail_gib")
        mem_src = mf.get("source")
        if mem_val is None and rl.get("preflight_avail_gib") is not None:
            mem_val = rl["preflight_avail_gib"]
            mem_src = "preflight_only_not_continuous"
        write_json(
            args.out_dir / f"systems.memory.r{r}.{build}.json",
            {
                "claim_id": f"systems.memory.r{r}.{build}",
                "generated_utc": now_utc,
                "label": args.label,
                "isolation": args.isolation,
                "rank": r,
                "unit": "GiB",
                "value": mem_val,
                "source": mem_src or "missing",
                "floor_gib": mf.get("floor_gib"),
                "stopped": mf.get("stopped"),
                "min_utc": mf.get("min_utc"),
                "free_after_reserve_gib": rl.get("free_after_reserve_gib"),
                "reserve_gib": rl.get("reserve_gib"),
                "status": "measured" if (mem_src and mem_src.startswith("continuous_memfloor")) else "provisional",
            },
        )

    write_json(
        args.out_dir / f"systems.kv.pool.{build}.json",
        {
            "claim_id": f"systems.kv.pool.{build}",
            "generated_utc": now_utc,
            "label": args.label,
            "isolation": args.isolation,
            "unit": "tokens",
            "value": kv_pool_tokens,
            "streams": streams,
            "context_window": ctx_win,
            "kv_gib_per_stream": kv_gib_stream,
            "status": "measured" if kv_pool_tokens is not None else "pending",
        },
    )
    write_json(
        args.out_dir / f"systems.kv.bytes.{build}.json",
        {
            "claim_id": f"systems.kv.bytes.{build}",
            "generated_utc": now_utc,
            "label": args.label,
            "isolation": args.isolation,
            "unit": "B/token",
            "value": kv_bytes_per_tok,
            "kv_gib_per_stream": kv_gib_stream,
            "context_window": ctx_win,
            "layers": r0.get("layers"),
            "status": "measured" if kv_bytes_per_tok is not None else "pending",
        },
    )
    write_json(
        args.out_dir / f"systems.load.cold.{build}.json",
        {
            "claim_id": f"systems.load.cold.{build}",
            "generated_utc": now_utc,
            "label": args.label,
            "isolation": args.isolation,
            "unit": "s",
            "value": cold_ready_s,
            "serve_load_s": serve_load_s,
            "max_rank_load_s": max_rank_load_s,
            "per_rank_load_s": {str(r): rank_logs[r].get("load_s") for r in sorted(rank_logs)},
            "status": "measured" if cold_ready_s is not None else "pending",
        },
    )
    write_json(
        args.out_dir / f"systems.launch.{build}.json",
        {
            "claim_id": f"systems.launch.{build}",
            "generated_utc": now_utc,
            "label": args.label,
            "isolation": args.isolation,
            "nccl_ib_verified_ranks": ib_ranks,
            "launch_settings": launch_settings,
            "known_answer": known_answer_info,
            "verdict": "PASS" if (ib_ranks == 4 and cold_ready_s is not None) else "INCOMPLETE",
            "status": "measured" if (ib_ranks == 4 and cold_ready_s is not None) else "pending",
        },
    )

    print(
        f"RECEIPTS_OK out={args.out_dir} build={build} label={args.label} isolation={args.isolation} "
        f"layers={r0.get('layers')} ib={ib_ranks}/4 cold_ready_s={cold_ready_s} "
        f"r0_weights_gib={r0.get('weights_gib')} kv_pool_tokens={kv_pool_tokens}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
