#!/usr/bin/env python3
"""EXL3 encode farm: run exllamav3's trellis quantization of individual linears on remote GPUs.

Updated for ExLlamaV3 v1.6.7 (handles packed/n Hessian metadata and tensor quant_args such as
out_sensitivity/out_hessian_diag). Supports 1-group-ahead receive/send pipelining around the
GPU semaphore so TCP recv + torch.load of group k+1 overlaps with GPU quantization of group k.

Worker:  python3 farm/exl3_farm.py serve --port 5599 --device 0 --concurrency 1
Client:  quantize_remote(host, port, weights, H_datas, quant_args_list) -> (results, quant_args_list)

Wire format: 8-byte little-endian length + torch.save payload (pickled dict of CPU tensors).
"""
import argparse
import io
import socket
import struct
import sys
import threading
import time
import traceback

import torch


def _send(sock: socket.socket, obj) -> None:
    buf = io.BytesIO()
    torch.save(obj, buf)
    data = buf.getvalue()
    sock.sendall(struct.pack("<Q", len(data)))
    sock.sendall(data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    got = 0
    while got < n:
        c = sock.recv(min(n - got, 64 << 20))
        if not c:
            raise ConnectionError("socket closed")
        chunks.append(c)
        got += len(c)
    return b"".join(chunks)


def _recv(sock: socket.socket, map_location = "cpu"):
    n = struct.unpack("<Q", _recv_exact(sock, 8))[0]
    data = _recv_exact(sock, n)
    return torch.load(io.BytesIO(data), map_location = map_location, weights_only = False)


# ------------------------------------------------------------------ client (master side)
_H_KEYS = (
    "H", "packed", "n", "first_key", "count", "finalized",
    "num_total", "inf_nan", "su", "L", "diag", "q_fallback",
)


def _export_H(hd: dict) -> dict:
    out = {}
    for k in _H_KEYS:
        if k in hd:
            v = hd[k]
            if isinstance(v, torch.Tensor):
                v = v.detach().to("cpu")
            out[k] = v
    return out


def _export_qa(qa: dict) -> dict:
    out = {}
    for k, v in qa.items():
        if k == "devices":
            continue
        if isinstance(v, torch.Tensor):
            v = v.detach().to("cpu")
        out[k] = v
    return out


def quantize_remote(host: str, port: int, weights, H_datas, quant_args_list, timeout = 3600):
    """Quantize a group of linears on a remote worker.

    weights: list of (in, out) fp16/fp32 tensors (CPU ok)
    H_datas: list of FINALIZED H_data dicts (may repeat the same object)
    returns: ([(proxy_err, out_tensors_cpu), ...], updated quant_args_list)
    """
    pool, idx = [], []
    for hd in H_datas:
        for i, p in enumerate(pool):
            if p is hd:
                idx.append(i)
                break
        else:
            pool.append(hd)
            idx.append(len(pool) - 1)
    req = {
        "weights": [w.detach().to("cpu") for w in weights],
        "H_pool": [_export_H(hd) for hd in pool],
        "H_idx": idx,
        "quant_args_list": [_export_qa(qa) for qa in quant_args_list],
    }
    sock = socket.create_connection((host, port), timeout = timeout)
    try:
        sock.settimeout(timeout)
        _send(sock, req)
        rep = _recv(sock, map_location = "cpu")
    finally:
        sock.close()
    if rep.get("status") != "ok":
        raise RuntimeError(f"remote {host}:{port} failed: {rep.get('error')}")
    for qa, nqa in zip(quant_args_list, rep["quant_args_list"]):
        for k, v in nqa.items():
            if k != "devices":
                qa[k] = v
    return rep["results"], quant_args_list


# ------------------------------------------------------------------ server (worker side)
def _handle(conn: socket.socket, device: torch.device, log, gpu_lock: threading.Semaphore):
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3, quantize_exl3_batch

    t0 = time.time()
    req = _recv(conn, map_location = "cpu")
    t1 = time.time()
    with gpu_lock:
        tq0 = time.time()
        try:
            with torch.inference_mode():
                pool = req["H_pool"]
                for hd in pool:
                    for k, v in list(hd.items()):
                        if isinstance(v, torch.Tensor):
                            hd[k] = v.to(device)
                    hd["device"] = device
                H_datas = [pool[i] for i in req["H_idx"]]
                weights = req["weights"]
                qal = req["quant_args_list"]
                for qa in qal:
                    qa["devices"] = [device.index or 0]
                    qa["device_ratios"] = None
                    qa["debug_dir"] = "/tmp/exl3-farm-debug"
                if len(weights) > 1:
                    results = quantize_exl3_batch(weights, H_datas, qal)
                else:
                    _, pe, out = quantize_exl3(weights[0].to(device).float(), H_datas[0], qal[0], False)
                    results = [(pe, out)]
                cpu_results = []
                for pe, out in results:
                    cpu_results.append((float(pe), {k: v.detach().to("cpu") for k, v in out.items() if v is not None}))
                rep = {
                    "status": "ok",
                    "results": cpu_results,
                    "quant_args_list": [_export_qa(qa) for qa in qal],
                }
            torch.cuda.synchronize(device)
        except Exception as e:  # noqa: BLE001
            rep = {"status": "error", "error": f"{e!r}\n{traceback.format_exc()}"}
            log(f"ERROR: {e!r}")
        tq1 = time.time()
    _send(conn, rep)
    t3 = time.time()
    n = len(req["weights"])
    log(f"group of {n}: recv {t1 - t0:.2f}s wait {tq0 - t1:.2f}s quant {tq1 - tq0:.2f}s send {t3 - tq1:.2f}s")


def serve(port: int, device_index: int, concurrency: int):
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    gpu_lock = threading.Semaphore(concurrency)

    def log(msg):
        print(f"[farm {time.strftime('%H:%M:%S')}] {msg}", flush = True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(16)
    log(f"listening on :{port} device={device} concurrency={concurrency} (pipelined recv/send)")

    def worker(conn):
        try:
            _handle(conn, device, log, gpu_lock)
        except Exception as e:  # noqa: BLE001
            log(f"connection failed: {e!r}")
        finally:
            conn.close()

    while True:
        conn, _addr = srv.accept()
        conn.settimeout(3600)
        threading.Thread(target = worker, args = (conn,), daemon = True).start()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest = "cmd")
    s = sub.add_parser("serve")
    s.add_argument("--port", type = int, default = 5599)
    s.add_argument("--device", type = int, default = 0)
    s.add_argument("--concurrency", type = int, default = 1)
    a = ap.parse_args()
    if a.cmd == "serve":
        serve(a.port, a.device, a.concurrency)
    else:
        ap.print_help()
        sys.exit(1)
