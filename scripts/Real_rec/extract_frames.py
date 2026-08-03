#!/usr/bin/env python3
"""从视频中按间隔抽帧，带模糊过滤。

用法:
    python extract_frames.py -i video.mp4 -t 1              # 每1秒抽1帧
    python extract_frames.py -i video.mp4 -s 30             # 每30帧抽1帧
    python extract_frames.py -i video.mp4 -t 0.5 -m 200     # 每0.5秒抽，最多200张
    python extract_frames.py -i ./videos/ -t 1 -o ./frames/ # 批量处理整个目录
    python extract_frames.py -i video.mp4 -t 1 -n           # 预览模式（不保存）

依赖: opencv-python, numpy
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# ═══════════════════════════════════════════════════
# 配置（可直接修改）
# ═══════════════════════════════════════════════════

BLUR_THRESHOLD = 100.0   # Laplacian 方差阈值，越小越严格


def is_blurry(bgr_img):
    """True = 画面模糊，建议跳过。"""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD


def find_videos(path):
    """返回目录下所有视频文件，或单个文件。"""
    p = Path(path)
    exts = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}

    if p.is_file():
        if p.suffix.lower() in exts:
            return [str(p)]
        sys.exit(f"❌ 不支持: {p.suffix}，支持的格式: {exts}")

    if p.is_dir():
        vids = sorted(str(f) for f in p.iterdir()
                       if f.is_file() and f.suffix.lower() in exts)
        if not vids:
            sys.exit(f"❌ {path} 下无视频文件")
        return vids

    sys.exit(f"❌ 路径不存在: {path}")


def process(video_path, args):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ 无法打开: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = total / fps if fps > 0 else 0

    # 抽帧间隔
    if args.step:
        step = args.step
        strategy = f"每 {step} 帧一张"
    else:
        step = max(1, int(fps * args.interval))
        strategy = f"每 {args.interval}s 一张 (~{step} 帧)"

    print(f"\n  {os.path.basename(video_path)}")
    print(f"  {w}x{h}  |  {fps:.1f}fps  |  {total}帧  |  {dur:.1f}s  |  {strategy}")

    # 输出目录
    out = Path(args.output) if args.output else \
          Path(video_path).parent / f"frames_{Path(video_path).stem}"
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step == 0:
            if not args.keep_blur and is_blurry(frame):
                skipped += 1
                if args.dry_run:
                    print(f"    [SKIP 模糊] 帧 {idx}  t={idx/fps:.1f}s")
            else:
                if args.dry_run:
                    print(f"    [SAVE] 帧 {idx}  t={idx/fps:.1f}s")
                else:
                    fname = f"{args.prefix}_{idx:06d}.{args.fmt}"
                    fpath = out / fname
                    if args.fmt == "jpg":
                        cv2.imwrite(str(fpath), frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                    else:
                        cv2.imwrite(str(fpath), frame)
                saved += 1
                if args.max and saved >= args.max:
                    print(f"  → 已达上限 {args.max} 张，停止")
                    break

        idx += 1

    cap.release()

    if args.dry_run:
        print(f"\n  [预览] 将保存 {saved} 张，跳过模糊 {skipped} 张 → {out}/")
    else:
        print(f"  ✓ {saved} 张 → {out}/")
        if skipped:
            print(f"    跳过模糊帧: {skipped} 张")


def main():
    p = argparse.ArgumentParser(description="从视频中按间隔抽帧")
    p.add_argument("-i", "--input", required=True, help="视频文件或目录")
    p.add_argument("-o", "--output", default=None, help="输出目录")
    p.add_argument("-t", "--interval", type=float, default=1.0, help="秒间隔，默认1")
    p.add_argument("-s", "--step", type=int, default=None, help="帧间隔（覆盖-t）")
    p.add_argument("-m", "--max", type=int, default=None, help="最多抽取张数")
    p.add_argument("--fmt", choices=["jpg", "png"], default="jpg", help="格式")
    p.add_argument("-q", "--quality", type=int, default=95, help="JPG质量 1-100")
    p.add_argument("--prefix", default="frame", help="文件名前缀")
    p.add_argument("-n", "--dry-run", action="store_true", help="预览，不保存")
    p.add_argument("--keep-blur", action="store_true", help="保留模糊帧")
    args = p.parse_args()

    if args.step is None and args.interval <= 0:
        sys.exit("❌ --interval 必须 > 0")

    videos = find_videos(args.input)
    print(f"找到 {len(videos)} 个视频")

    for v in videos:
        process(v, args)

    print("\n全部完成。")


if __name__ == "__main__":
    main()