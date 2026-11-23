#!/usr/bin/env python3
"""Patch exllamav3/conversion/convert_model.py (v1.6.7) so expert encoding can be farmed to remote GPUs.

EXL3_REMOTE_WORKERS="host:port,host:port" adds one work thread per remote worker to the
per-layer parallel quantization; each thread ships its groups (weights + finalized Hessians)
via exl3_farm.quantize_remote and rebuilds LinearEXL3 from the returned tensors. Falls back to
local encoding for a group if the remote call fails. Idempotent.
"""
import sys

p = sys.argv[1]
s = open(p).read()
if "EXL3_REMOTE_WORKERS" in s:
    print("already patched")
    sys.exit(0)

# 1. remote work path inside quantize_linears_parallel
old_thread = "    def work_thread_(device_idx, dev_groups):\n        global curr_progress\n\n        with torch.inference_mode():\n"
assert old_thread in s, "work_thread_ anchor not found"
new_thread = '''    def work_thread_(device_idx, dev_groups):
        global curr_progress

        if isinstance(device_idx, str) and device_idx.startswith("remote:"):
            _remote_work_thread(device_idx, dev_groups)
            return

        with torch.inference_mode():
'''
s = s.replace(old_thread, new_thread, 1)

remote_fn = '''
    def _remote_work_thread(device_idx, dev_groups):
        global curr_progress
        try:
            from farm.exl3_farm import quantize_remote
        except ImportError:
            from exl3_farm import quantize_remote
        from exllamav3.modules.quant.exl3_lib.quantize import finalize_capture_H
        host, port = device_idx[len("remote:"):].rsplit(":", 1)
        port = int(port)
        local_dev = torch.device([d for d in devices if not isinstance(d, str)][0])
        t0 = time.time()
        with torch.inference_mode():
            for group in dev_groups:
                quant_args_list = [
                    make_quant_args(
                        args, idx, strategy[l.key], [0],
                        out_sensitivity = out_sens.get(l.key), out_hessian_diag = out_hess.get(l.key))
                    for l in group
                ]
                H_datas = [capture_H[l.qmap] if state else l.init_H_data(False) for l in group]
                for hd, qa in zip(H_datas, quant_args_list):
                    torch.manual_seed(qa["seed"])
                    finalize_capture_H(hd, qa, False)
                weights = [l.inner.get_weight_tensor() for l in group]
                biases = [l.inner.get_bias_tensor() for l in group]
                try:
                    tq = time.time()
                    results, quant_args_list = quantize_remote(host, port, weights, H_datas, quant_args_list)
                    tstr = f"  [remote {host} {(time.time() - tq) / len(group):4.2f} s]"
                except Exception as e:
                    print(f" !! remote {host}:{port} failed ({e!r}); encoding group locally", flush = True)
                    for l in group:
                        if not exl3_quantize.NO_SWAP: l.inner.swap_cpu()
                    if len(group) > 1:
                        quant_args_list = [
                            make_quant_args(
                                args, idx, strategy[l.key], [local_dev.index or 0],
                                out_sensitivity = out_sens.get(l.key), out_hessian_diag = out_hess.get(l.key))
                            for l in group
                        ]
                        proxy_errs = convert_exl3_group(group, H_datas, quant_args_list)
                        for l, qa, pe in zip(group, quant_args_list, proxy_errs):
                            if not exl3_quantize.NO_SWAP: l.inner.swap_cpu()
                            print_quantized_linear(config, l, qa, pe, "  [local fallback]")
                            with progress_lock:
                                curr_progress += 1
                    else:
                        l = group[0]
                        qa = make_quant_args(
                            args, idx, strategy[l.key], [local_dev.index or 0],
                            out_sensitivity = out_sens.get(l.key), out_hessian_diag = out_hess.get(l.key))
                        pe = l.convert_exl3(H_datas[0], quant_args = qa, verbose = False, save_reg = False,
                                            override_swap_device = local_dev)
                        if not exl3_quantize.NO_SWAP: l.inner.swap_cpu()
                        print_quantized_linear(config, l, qa, pe, "  [local fallback]")
                        with progress_lock:
                            curr_progress += 1
                    continue
                for l, bias, qa, (pe, out) in zip(group, biases, quant_args_list, results):
                    out = {k: v.to(local_dev) for k, v in out.items()}
                    l.inner = LinearEXL3(
                        l.config, l.in_features, l.out_features,
                        out.get("scale"), out.get("su"), out.get("sv"), out.get("suh"), out.get("svh"),
                        out.get("trellis"), out.get("mcg"), out.get("mul1"),
                        bias, l.out_dtype, key = l.key
                    )
                    if not exl3_quantize.NO_SWAP: l.inner.swap_cpu()
                    print_quantized_linear(config, l, qa, pe, tstr)
                    with progress_lock:
                        curr_progress += 1
                malloc_trim()
        print(f" -- remote {host}: {sum(len(g) for g in dev_groups)} tensors in {time.time() - t0:.0f} s", flush = True)

'''
anchor = "    # Launch\n    threads = []\n"
assert anchor in s, "Launch anchor not found"
s = s.replace(anchor, remote_fn + anchor, 1)

# 2. dispatch inside quantize_list (v1.6.7): farm the <=8-bit linears, keep 16-bit ones on the single path
old_dispatch = '''            def quantize_list(linears_, capture_H_):
                if (
                    len(linears_) >= len(devices) and
                    all(strategy[l.key] <= 8 for l in linears_)
                ):
                    quantize_linears_parallel(
                        args, linears_, config, strategy, idx, devices, eff_ratios("quant_thread"), capture_H_, state, out_sens)
                else:
                    quantize_linears_single(
                        args, linears_, config, strategy, idx, devices, eff_ratios("quant_tiles"), capture_H_, state, out_sens)
'''
assert old_dispatch in s, "quantize_list anchor not found"
new_dispatch = '''            def quantize_list(linears_, capture_H_):
                _remotes = [f"remote:{w.strip()}" for w in os.environ.get("EXL3_REMOTE_WORKERS", "").split(",") if w.strip()]
                _q_linears = [l for l in linears_ if strategy[l.key] <= 8]
                _u_linears = [l for l in linears_ if strategy[l.key] > 8]
                if _remotes and len(_q_linears) >= 2 * (len(devices) + len(_remotes)):
                    if _u_linears:
                        quantize_linears_single(
                            args, _u_linears, config, strategy, idx, devices, eff_ratios("quant_tiles"), capture_H_, state, out_sens)
                    print(f" -- Farming {len(_q_linears)} tensors over {devices + _remotes}", flush = True)
                    quantize_linears_parallel(
                        args, _q_linears, config, strategy, idx, devices + _remotes, None, capture_H_, state, out_sens)
                elif (
                    len(linears_) >= len(devices) and
                    all(strategy[l.key] <= 8 for l in linears_)
                ):
                    quantize_linears_parallel(
                        args, linears_, config, strategy, idx, devices, eff_ratios("quant_thread"), capture_H_, state, out_sens)
                else:
                    quantize_linears_single(
                        args, linears_, config, strategy, idx, devices, eff_ratios("quant_tiles"), capture_H_, state, out_sens)
'''
s = s.replace(old_dispatch, new_dispatch, 1)
open(p, "w").write(s)
print("patched", p)
