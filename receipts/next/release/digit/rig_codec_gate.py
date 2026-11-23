#!/usr/bin/env python3
"""Known-answer JPEG and WebP end-to-end gate for Hy4-VL on TensorFold."""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import json
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request


SEGMENTS = {
    "0": (1, 1, 1, 0, 1, 1, 1),
    "1": (0, 0, 1, 0, 0, 1, 0),
    "2": (1, 0, 1, 1, 1, 0, 1),
    "3": (1, 0, 1, 1, 0, 1, 1),
    "4": (0, 1, 1, 1, 0, 1, 0),
    "5": (1, 1, 0, 1, 0, 1, 1),
    "6": (1, 1, 0, 1, 1, 1, 1),
    "7": (1, 0, 1, 0, 0, 1, 0),
    "8": (1, 1, 1, 1, 1, 1, 1),
    "9": (1, 1, 1, 1, 0, 1, 1),
}


def fill_rect(buf: bytearray, w: int, h: int, x0: int, y0: int, x1: int, y1: int, rgb: tuple[int, int, int]) -> None:
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return
    row_bytes = bytes(rgb) * (x1 - x0)
    stride = w * 3
    for y in range(y0, y1):
        off = y * stride + x0 * 3
        buf[off : off + len(row_bytes)] = row_bytes


def draw_digit(
    buf: bytearray, w: int, h: int, digit: str, x: int, y: int, dw: int, dh: int, t: int, rgb: tuple[int, int, int]
) -> None:
    a, b, c, d, e, f, g = SEGMENTS[digit]
    mid = y + (dh - t) // 2
    if a:
        fill_rect(buf, w, h, x, y, x + dw, y + t, rgb)
    if b:
        fill_rect(buf, w, h, x, y, x + t, mid + t, rgb)
    if c:
        fill_rect(buf, w, h, x + dw - t, y, x + dw, mid + t, rgb)
    if d:
        fill_rect(buf, w, h, x, mid, x + dw, mid + t, rgb)
    if e:
        fill_rect(buf, w, h, x, mid, x + t, y + dh, rgb)
    if f:
        fill_rect(buf, w, h, x + dw - t, mid, x + dw, y + dh, rgb)
    if g:
        fill_rect(buf, w, h, x, y + dh - t, x + dw, y + dh, rgb)


def make_jpeg_card_rgb(w: int = 512, h: int = 512) -> bytes:
    """Left half solid red, right half solid blue, dark center plaque with white '74'."""
    buf = bytearray(w * h * 3)
    fill_rect(buf, w, h, 0, 0, w // 2, h, (225, 30, 30))
    fill_rect(buf, w, h, w // 2, 0, w, h, (30, 75, 225))
    fill_rect(buf, w, h, 106, 136, 406, 376, (20, 20, 24))
    draw_digit(buf, w, h, "7", 140, 166, 100, 180, 26, (255, 255, 255))
    draw_digit(buf, w, h, "4", 272, 166, 100, 180, 26, (255, 255, 255))
    return bytes(buf)


def make_webp_card_rgb(w: int = 512, h: int = 512) -> bytes:
    """Top half solid green, bottom half solid yellow, dark center plaque with white '39'."""
    buf = bytearray(w * h * 3)
    fill_rect(buf, w, h, 0, 0, w, h // 2, (25, 175, 55))
    fill_rect(buf, w, h, 0, h // 2, w, h, (245, 215, 25))
    fill_rect(buf, w, h, 106, 136, 406, 376, (20, 20, 24))
    draw_digit(buf, w, h, "3", 140, 166, 100, 180, 26, (255, 255, 255))
    draw_digit(buf, w, h, "9", 272, 166, 100, 180, 26, (255, 255, 255))
    return bytes(buf)


def encode_jpeg_rgb(ljt_so: pathlib.Path, rgb: bytes, w: int, h: int, quality: int = 95) -> bytes:
    ljt = ctypes.CDLL(str(ljt_so))
    ljt.tj3Init.argtypes = [ctypes.c_int]
    ljt.tj3Init.restype = ctypes.c_void_p
    ljt.tj3Destroy.argtypes = [ctypes.c_void_p]
    ljt.tj3Set.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    ljt.tj3Set.restype = ctypes.c_int
    ljt.tj3Compress8.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    ljt.tj3Compress8.restype = ctypes.c_int
    ljt.tj3Free.argtypes = [ctypes.c_void_p]

    handle = ljt.tj3Init(0)
    assert handle
    try:
        assert ljt.tj3Set(handle, 4, 2) == 0  # TJPARAM_SUBSAMP = TJSAMP_420
        assert ljt.tj3Set(handle, 3, quality) == 0  # TJPARAM_QUALITY
        dst_ptr = ctypes.c_void_p()
        dst_len = ctypes.c_size_t(0)
        rc = ljt.tj3Compress8(handle, rgb, w, 0, h, 0, ctypes.byref(dst_ptr), ctypes.byref(dst_len))
        assert rc == 0, f"tj3Compress8 failed rc={rc}"
        try:
            return ctypes.string_at(dst_ptr.value, dst_len.value)
        finally:
            ljt.tj3Free(dst_ptr)
    finally:
        ljt.tj3Destroy(handle)


def check_local_codecs(lib_dir: pathlib.Path) -> dict:
    ljt_path = lib_dir / "libturbojpeg.so.0"
    webp_path = lib_dir / "libwebpdecoder.so.3"
    ljt_bytes = ljt_path.read_bytes()
    key = b"libjpeg-turbo version "
    idx = ljt_bytes.find(key)
    if idx < 0:
        raise RuntimeError(f"missing '{key.decode()}' in {ljt_path}")
    rest = ljt_bytes[idx + len(key) :]
    end = min(i for i in (rest.find(b" "), rest.find(b"\x00")) if i >= 0)
    ljt_ver = rest[:end].decode("ascii")
    if ljt_ver != "3.1.4.1":
        raise RuntimeError(f"libturbojpeg version pin mismatch: {ljt_ver!r} != '3.1.4.1'")

    ljt = ctypes.CDLL(str(ljt_path))
    for sym in (
        "tj3Init",
        "tj3Destroy",
        "tj3Set",
        "tj3Get",
        "tj3DecompressHeader",
        "tj3Decompress8",
        "tj3GetErrorCode",
    ):
        getattr(ljt, sym)

    webp = ctypes.CDLL(str(webp_path))
    for sym in ("WebPGetDecoderVersion", "WebPGetInfo", "WebPDecodeRGBAInto"):
        getattr(webp, sym)
    webp.WebPGetDecoderVersion.restype = ctypes.c_int
    webp_ver = webp.WebPGetDecoderVersion()
    if webp_ver != 0x010600:
        raise RuntimeError(f"libwebpdecoder version pin mismatch: {hex(webp_ver)} != 0x10600")

    return {
        "libturbojpeg_version": ljt_ver,
        "libturbojpeg_sha256": hashlib.sha256(ljt_bytes).hexdigest(),
        "libwebpdecoder_version_hex": hex(webp_ver),
        "libwebpdecoder_sha256": hashlib.sha256(webp_path.read_bytes()).hexdigest(),
    }


def verify_webp_decode(webp_so: pathlib.Path, webp_bytes: bytes, want_w: int, want_h: int) -> None:
    assert webp_bytes[:4] == b"RIFF" and webp_bytes[8:12] == b"WEBP"
    webp = ctypes.CDLL(str(webp_so))
    webp.WebPGetInfo.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    webp.WebPGetInfo.restype = ctypes.c_int
    webp.WebPDecodeRGBAInto.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    webp.WebPDecodeRGBAInto.restype = ctypes.c_void_p
    ww = ctypes.c_int(0)
    wh = ctypes.c_int(0)
    assert webp.WebPGetInfo(webp_bytes, len(webp_bytes), ctypes.byref(ww), ctypes.byref(wh)) == 1
    assert (ww.value, wh.value) == (want_w, want_h), f"got {(ww.value, wh.value)}, want {(want_w, want_h)}"
    rgba = ctypes.create_string_buffer(want_w * want_h * 4)
    assert webp.WebPDecodeRGBAInto(webp_bytes, len(webp_bytes), rgba, len(rgba), want_w * 4) is not None


def generate_test_images(
    out_dir: pathlib.Path, lib_dir: pathlib.Path, ffmpeg_bin: str = "ffmpeg", gate2b_png_dir: pathlib.Path | None = None
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ljt_so = lib_dir / "libturbojpeg.so.0"
    webp_so = lib_dir / "libwebpdecoder.so.3"

    # 1. Synthetic JPEG card (74, red/blue)
    jpg_bytes = encode_jpeg_rgb(ljt_so, make_jpeg_card_rgb(512, 512), 512, 512, quality=95)
    jpg_path = out_dir / "known_74_red_blue.jpg"
    jpg_path.write_bytes(jpg_bytes)

    # 2. Synthetic WebP card (39, green/yellow)
    webp_path = out_dir / "known_39_green_yellow.webp"
    subprocess.run(
        [
            ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "512x512",
            "-i",
            "pipe:0",
            "-frames:v",
            "1",
            "-c:v",
            "libwebp",
            "-lossless",
            "1",
            str(webp_path),
        ],
        input=make_webp_card_rgb(512, 512),
        check=True,
    )
    webp_bytes = webp_path.read_bytes()
    verify_webp_decode(webp_so, webp_bytes, 512, 512)

    out = {
        "known_74_red_blue.jpg": {"bytes": len(jpg_bytes), "sha256": hashlib.sha256(jpg_bytes).hexdigest()},
        "known_39_green_yellow.webp": {"bytes": len(webp_bytes), "sha256": hashlib.sha256(webp_bytes).hexdigest()},
    }

    # 3. Also transcode proven Round-2 Gate 2b PNGs to baseline JPEG (KESTREL-4729) and lossy WebP (bar chart).
    if gate2b_png_dir and gate2b_png_dir.is_dir():
        tc_png = gate2b_png_dir / "gate2b_text_card.png"
        bc_png = gate2b_png_dir / "gate2b_bar_chart.png"
        if tc_png.is_file():
            raw_rgb = subprocess.check_output(
                [ffmpeg_bin, "-loglevel", "error", "-i", str(tc_png), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
            )
            tc_jpg_bytes = encode_jpeg_rgb(ljt_so, raw_rgb, 896, 448, quality=94)
            tc_jpg = out_dir / "gate2b_text_card.jpg"
            tc_jpg.write_bytes(tc_jpg_bytes)
            out["gate2b_text_card.jpg"] = {"bytes": len(tc_jpg_bytes), "sha256": hashlib.sha256(tc_jpg_bytes).hexdigest()}
        if bc_png.is_file():
            bc_webp = out_dir / "gate2b_bar_chart.webp"
            subprocess.run(
                [ffmpeg_bin, "-y", "-loglevel", "error", "-i", str(bc_png), "-c:v", "libwebp", "-quality", "92", str(bc_webp)],
                check=True,
            )
            bc_webp_bytes = bc_webp.read_bytes()
            verify_webp_decode(webp_so, bc_webp_bytes, 896, 576)
            out["gate2b_bar_chart.webp"] = {"bytes": len(bc_webp_bytes), "sha256": hashlib.sha256(bc_webp_bytes).hexdigest()}

    return out


def post_chat(base_url: str, model: str, messages: list[dict], max_tokens: int = 128) -> tuple[int, dict, float]:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "draft": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw), time.monotonic() - t0
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"raw": raw}
        return e.code, parsed, time.monotonic() - t0


def run_http_gate(base_url: str, model: str, images_dir: pathlib.Path, out_json: pathlib.Path) -> int:
    if not model:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/v1/models", timeout=15) as resp:
            models = json.loads(resp.read().decode("utf-8"))
        model = models["data"][0]["id"]

    checks: list[tuple[str, str, pathlib.Path, str, list[str]]] = []
    if (images_dir / "gate2b_text_card.jpg").is_file():
        checks.append(
            (
                "jpeg_kestrel_card",
                "image/jpeg",
                images_dir / "gate2b_text_card.jpg",
                "Read the exact code, port number, and status shown in this image.",
                ["kestrel-4729", "18080", "ready"],
            )
        )
    checks.append(
        (
            "jpeg_74_red_blue",
            "image/jpeg",
            images_dir / "known_74_red_blue.jpg",
            "What two-digit number is written in the center, and what colors are the left half and right half? Reply in one short sentence.",
            ["74", "red", "blue"],
        )
    )
    if (images_dir / "gate2b_bar_chart.webp").is_file():
        checks.append(
            (
                "webp_bar_chart",
                "image/webp",
                images_dir / "gate2b_bar_chart.webp",
                "Look at the bar chart. Which day has the highest bar, which day has the lowest bar, and what are the four values from Monday to Thursday?",
                ["thursday", "90", "monday", "20", "75", "40"],
            )
        )
    checks.append(
        (
            "webp_39_green_yellow",
            "image/webp",
            images_dir / "known_39_green_yellow.webp",
            "What two-digit number is written in the center, and what colors are the top half and bottom half? Reply in one short sentence.",
            ["39", "green", "yellow"],
        )
    )

    results = {}
    overall = True
    for name, mime, path, prompt, tokens_needed in checks:
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        st, resp, dt = post_chat(
            base_url,
            model,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        ans = resp.get("choices", [{}])[0].get("message", {}).get("content", "") if st == 200 else ""
        low = ans.lower()
        ok = st == 200 and all(tok in low for tok in tokens_needed)
        overall = overall and ok
        results[name] = {"status": st, "latency_s": round(dt, 3), "answer": ans, "pass": ok}

    st_txt, r_txt, dt_txt = post_chat(
        base_url,
        model,
        [{"role": "user", "content": "Reply with the single digit equal to 2 + 2 and nothing else."}],
        max_tokens=16,
    )
    txt_ctrl = r_txt.get("choices", [{}])[0].get("message", {}).get("content", "") if st_txt == 200 else ""
    pass_txt = st_txt == 200 and ("4" in txt_ctrl)
    overall = overall and pass_txt
    results["text_after_vision"] = {"status": st_txt, "latency_s": round(dt_txt, 3), "answer": txt_ctrl, "pass": pass_txt}

    summary = {"model": model, "overall_pass": overall, "checks": results}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if overall else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-local-codecs", type=pathlib.Path, help="Verify built .so symbols and exact version pins")
    ap.add_argument("--generate-images", type=pathlib.Path, help="Generate deterministic JPEG + WebP known-answer cards")
    ap.add_argument("--gate2b-png-dir", type=pathlib.Path, default=None)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--base-url", help="Run live HTTP JPEG + WebP known-answer gate against served endpoint")
    ap.add_argument("--model", default="")
    ap.add_argument("--images-dir", type=pathlib.Path)
    ap.add_argument("--out-json", type=pathlib.Path, default=pathlib.Path("/tmp/rig_codec_gate.json"))
    args = ap.parse_args()

    if args.check_local_codecs:
        info = check_local_codecs(args.check_local_codecs)
        print(json.dumps(info, indent=2))
    if args.generate_images:
        if not args.check_local_codecs:
            raise SystemExit("--generate-images requires --check-local-codecs <lib_dir>")
        img_info = generate_test_images(args.generate_images, args.check_local_codecs, args.ffmpeg, args.gate2b_png_dir)
        print(json.dumps(img_info, indent=2))
    if args.base_url:
        if not args.images_dir:
            raise SystemExit("--base-url requires --images-dir")
        return run_http_gate(args.base_url, args.model, args.images_dir, args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
