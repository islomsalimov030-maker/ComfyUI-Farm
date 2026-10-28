#!/usr/bin/env python3
"""Проверки приёмки FlashVSR+facesafe по критериям задания.

Скрипт работает на ПОДЕ (ffmpeg/numpy доступны там). Пути — на поде.

    python vsr_check.py --input  /workspace/input/vsr_test/clip_x.mp4 \
                        --output /workspace/output/vsr_test/out_x.mp4 \
                        --mask   /workspace/output/vsr_test/mask_x.mp4  (необязательно)

Критерии:
 1. ffprobe: 1920x1080, h264, yuv420p, fps = fps входа, nb_frames = кадрам входа, звука нет.
 2. Лицо не тронуто: в зоне маски >= 0.99 доля кадров со средней |выход-вход| < 1/255.
 3. Снаружи маски VSR работает: заметная разница с входом + рост вариации Лапласиана.
 4. Шов петли: d(i)=средняя |кадр i+1 - кадр i|; d(last->first) <= 1.5 * медианы d.
    То же внутри маски и снаружи. Сравнение с входом.
 5. Шов вокруг лица: кроп вокруг овала; межкадровая разница маски на краю.
 6. Кадры без лица: без ошибок, кадр целиком из VSR.
 7. Швов тайлов нет (тайл 384, перекрытие 32 на входе 960x540).
 8. Пики памяти/время — из лога прогона.

Вывод: JSON + человекочитаемая таблица.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=900).stdout


def probe(path: str) -> dict:
    out = sh(["ffprobe", "-v", "error", "-show_streams", "-show_format",
              "-of", "json", path])
    return json.loads(out or "{}")


def video_info(path: str) -> dict:
    d = probe(path)
    vs = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), {})
    has_audio = any(s.get("codec_type") == "audio" for s in d.get("streams", []))
    num, den = (vs.get("r_frame_rate", "0/1").split("/") + ["1"])[:2]
    fps = float(num) / float(den or 1)
    nb = vs.get("nb_frames")
    if nb in (None, "N/A"):
        nb = int(round(float(d.get("format", {}).get("duration", 0)) * fps))
    return {
        "width": vs.get("width"), "height": vs.get("height"),
        "codec": vs.get("codec_name"), "pix_fmt": vs.get("pix_fmt"),
        "fps": fps, "nb_frames": int(nb), "has_audio": has_audio,
        "duration": float(d.get("format", {}).get("duration", 0)),
    }


def read_frames(path: str, max_frames: int | None = None) -> np.ndarray:
    """Кадры как float32 [N,H,W,3] в 0..255."""
    cmd = ["ffmpeg", "-v", "error", "-i", path]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, timeout=1800).stdout
    if not raw:
        return np.zeros((0, 0, 0, 3), dtype=np.float32)
    meta = video_info(path)
    w, h = meta["width"], meta["height"]
    n = len(raw) // (w * h * 3)
    a = np.frombuffer(raw[: n * w * h * 3], dtype=np.uint8)
    return a.reshape(n, h, w, 3).astype(np.float32)


def read_gray(path: str, max_frames: int | None = None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", path]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "gray", "-"]
    raw = subprocess.run(cmd, capture_output=True, timeout=1800).stdout
    meta = video_info(path)
    w, h = meta["width"], meta["height"]
    n = len(raw) // (w * h)
    return np.frombuffer(raw[: n * w * h], dtype=np.uint8).reshape(n, h, w).astype(np.float32)


def laplacian_var(frames_gray: np.ndarray) -> np.ndarray:
    """Вариация Лапласиана по каждому кадру (мера резкости)."""
    out = []
    for f in frames_gray:
        k = (f[:-2, 1:-1] + f[2:, 1:-1] + f[1:-1, :-2] + f[1:-1, 2:]
             - 4 * f[1:-1, 1:-1])
        out.append(float(k.var()))
    return np.array(out)


def frame_diffs(frames: np.ndarray) -> np.ndarray:
    """d(i) = средняя |кадр i+1 - кадр i| по всем каналам; d[-1] = |последний - первый|."""
    n = len(frames)
    if n < 2:
        return np.zeros(0)
    d = np.array([np.abs(frames[i + 1] - frames[i]).mean() for i in range(n - 1)])
    return np.concatenate([d, [np.abs(frames[-1] - frames[0]).mean()]])


def loop_metrics(frames: np.ndarray, mask: np.ndarray | None) -> dict:
    """Метрики шва петли. mask — [N,H,W] float 0..1 (None = вся рамка)."""
    d_all = frame_diffs(frames)
    res: dict = {}
    if len(d_all) == 0:
        return res
    med = float(np.median(d_all[:-1]))
    seam = float(d_all[-1])
    res["all"] = {"median_d": med, "seam_d": seam, "ratio": seam / med if med > 0 else float("inf")}

    if mask is not None and len(mask) == len(frames):
        per_frame_mask = mask.mean(axis=(1, 2)) > 0.001
        # внутри маски: только кадры, где маска есть
        inside = []
        for i in range(len(frames) - 1):
            m = mask[i + 1] * mask[i]
            inside.append(np.abs(frames[i + 1] - frames[i])[m > 0.5].mean() if (m > 0.5).any() else np.nan)
        inside = np.array(inside, dtype=float)
        seam_in = (np.abs(frames[-1] - frames[0]) * (mask[-1] * mask[0] > 0.5)).mean() \
            if ((mask[-1] * mask[0]) > 0.5).any() else float("nan")
        inside = inside[~np.isnan(inside)]
        if len(inside):
            res["inside_mask"] = {
                "median_d": float(np.median(inside)),
                "seam_d": float(seam_in),
                "ratio": float(seam_in) / float(np.median(inside)) if np.median(inside) > 0 else float("inf"),
            }
        # снаружи маски
        outside = []
        for i in range(len(frames) - 1):
            m = 1.0 - (mask[i + 1] * mask[i])
            outside.append(np.abs(frames[i + 1] - frames[i])[m > 0.5].mean() if (m > 0.5).any() else np.nan)
        outside = np.array(outside, dtype=float)
        outside = outside[~np.isnan(outside)]
        seam_out = (np.abs(frames[-1] - frames[0]) * (1.0 - (mask[-1] * mask[0]) > 0.5)).mean() \
            if ((1.0 - (mask[-1] * mask[0])) > 0.5).any() else float("nan")
        if len(outside):
            res["outside_mask"] = {
                "median_d": float(np.median(outside)),
                "seam_d": float(seam_out),
                "ratio": float(seam_out) / float(np.median(outside)) if np.median(outside) > 0 else float("inf"),
            }
    return res


def mask_from_file(path: str) -> np.ndarray:
    """Маска из debug-видео (белая маска на чёрном) -> [N,H,W] float 0..1."""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    raw = subprocess.run(cmd, capture_output=True, timeout=900).stdout
    meta = video_info(path)
    w, h = meta["width"], meta["height"]
    n = len(raw) // (w * h)
    a = np.frombuffer(raw[: n * w * h], dtype=np.uint8).reshape(n, h, w)
    return a.astype(np.float32) / 255.0


def tile_seam_check(frames_in: np.ndarray, frames_out: np.ndarray, tile: int = 384,
                    overlap: int = 32) -> dict:
    """Ищем вертикальные/горизонтальные полосы на границах тайлов.

    Сравниваем среднюю |разницу| в колонках, прилегающих к границам тайлов,
    с фоновым уровнем.
    """
    h, w = frames_in.shape[1], frames_in.shape[2]
    step = tile - overlap
    cols = list(range(step, w - 1, step))
    rows = list(range(step, h - 1, step))
    res = {"tile": tile, "overlap": overlap, "n_col_seams": len(cols), "n_row_seams": len(rows)}

    if cols:
        seam_cols = []
        for c in cols:
            # полоса шириной 2 px вокруг границы тайла
            band = np.abs(frames_out[:, :, max(c - 1, 0):c + 2, :] - frames_in[:, :, max(c - 1, 0):c + 2, :]).mean(axis=(1, 2, 3))
            seam_cols.append(float(band.mean()))
        res["col_seam_mean"] = float(np.mean(seam_cols))
        res["col_seam_max"] = float(np.max(seam_cols))
    if rows:
        seam_rows = []
        for r in rows:
            band = np.abs(frames_out[:, max(r - 1, 0):r + 2, :, :] - frames_in[:, max(r - 1, 0):r + 2, :, :]).mean(axis=(1, 2, 3))
            seam_rows.append(float(band.mean()))
        res["row_seam_mean"] = float(np.mean(seam_rows))
        res["row_seam_max"] = float(np.max(seam_rows))

    # фоновая разница (вне швов)
    res["global_diff_mean"] = float(np.abs(frames_out - frames_in).mean())
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mask", help="debug-видео с маской (белая на чёрном)")
    ap.add_argument("--json", help="сохранить результат в JSON")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-frame-check", action="store_true",
                    help="не читать кадры (быстрая проверка формата)")
    args = ap.parse_args()

    mf = args.max_frames or None
    out: dict = {"input": args.input, "output": args.output}

    vi_in = video_info(args.input)
    vi_out = video_info(args.output)
    out["probe_in"] = vi_in
    out["probe_out"] = vi_out

    c1 = {
        "resolution_ok": (vi_out["width"], vi_out["height"]) == (1920, 1080),
        "codec_h264": vi_out["codec"] == "h264",
        "pix_fmt_ok": vi_out["pix_fmt"] == "yuv420p",
        "fps_matches": abs(vi_out["fps"] - vi_in["fps"]) < 0.01,
        "frames_match": vi_out["nb_frames"] == vi_in["nb_frames"],
        "no_audio": not vi_out["has_audio"],
    }
    out["crit1_format"] = c1

    if args.no_frame_check:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    fi = read_frames(args.input, mf)
    fo = read_frames(args.output, mf)
    n = min(len(fi), len(fo))
    fi, fo = fi[:n], fo[:n]
    out["n_frames_compared"] = n

    mask = None
    if args.mask and os.path.exists(args.mask):
        mask = mask_from_file(args.mask)
        if len(mask) > n:
            mask = mask[:n]
        elif len(mask) < n:
            mask = np.pad(mask, ((0, n - len(mask)), (0, 0), (0, 0)))
    out["mask_available"] = mask is not None

    # ── критерий 2: лицо не тронуто
    if mask is not None:
        m = mask > 0.5
        if m.any():
            per_frame = []
            for i in range(n):
                if m[i].any():
                    per_frame.append(float(np.abs(fo[i] - fi[i])[m[i]].mean()))
            if per_frame:
                arr = np.array(per_frame)
                frac_ok = float((arr < (1.0 / 255.0)).mean())
                out["crit2_face_untouched"] = {
                    "frac_frames_under_1_255": frac_ok,
                    "pass": frac_ok >= 0.99,
                    "mean_abs_diff_in_mask": float(arr.mean()),
                    "max_abs_diff_in_mask": float(arr.max()),
                }
        # ── критерий 3: снаружи VSR работает
        om = ~m
        if om.any():
            diffs = [float(np.abs(fo[i] - fi[i])[om[i]].mean()) for i in range(n) if om[i].any()]
            gi = read_gray(args.input, mf)[:n]
            go = read_gray(args.output, mf)[:n]
            li = laplacian_var(gi)
            lo = laplacian_var(go)
            if len(li) == len(lo) and len(li) > 1:
                # нарезка по кадрам, где маска снаружи
                idx = [i for i in range(n) if om[i].any()]
                li_m = li[idx].mean() if idx else 0.0
                lo_m = lo[idx].mean() if idx else 0.0
            else:
                li_m, lo_m = float(li.mean()), float(lo.mean())
            out["crit3_vsr_outside"] = {
                "mean_abs_diff_outside": float(np.mean(diffs)) if diffs else 0.0,
                "laplacian_var_in": li_m,
                "laplacian_var_out": lo_m,
                "laplacian_gain": (lo_m / li_m) if li_m > 0 else float("inf"),
                "pass": (bool(diffs) and float(np.mean(diffs)) > 0.05
                         and lo_m > li_m),
            }

    # ── критерий 4: шов петли
    loop_out = loop_metrics(fo, mask)
    loop_in = loop_metrics(fi, mask)
    out["crit4_loop_in"] = loop_in
    out["crit4_loop_out"] = loop_out
    if loop_out:
        r = loop_out["all"]["ratio"]
        ri = loop_in.get("all", {}).get("ratio", float("inf"))
        out["crit4_loop_pass"] = {
            "ratio_out": r, "ratio_in": ri,
            "pass": (r <= 1.5) and (r <= ri + 1e-9),
        }

    # ── критерий 7: швы тайлов
    out["crit7_tiles"] = tile_seam_check(fi, fo)

    # ── критерий 5: межкадровая разница маски на краю
    if mask is not None and len(mask) > 1:
        edge_diff = []
        for i in range(len(mask) - 1):
            e = np.abs(mask[i + 1] - mask[i])
            edge_diff.append(float(e.mean()))
        out["crit5_mask_stability"] = {
            "mean_frame_to_frame_mask_diff": float(np.mean(edge_diff)),
            "max_frame_to_frame_mask_diff": float(np.max(edge_diff)),
        }

    if args.json:
        json.dump(out, open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
