#!/usr/bin/env python3
"""从 CEPB 数据集按类别均匀采样 N 张，复制到真实数据集文件夹。

用法:
    python sample_cepb_to_real.py \
        --cepb /root/dataset_train \
        --real /root/yolo/dataset_real_remapped \
        --total 150

输出: 将 CEPB 采样的图片和标签复制到真实数据集的 train 目录下，
      图片加 cepb_ 前缀避免重名。
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path


def group_by_class(labels_dir):
    """遍历 labels 目录，按第一个 class id 分组。

    返回: {class_id: [stem1, stem2, ...]}
    """
    by_class = {}
    for lbl in Path(labels_dir).glob("*.txt"):
        with open(lbl) as f:
            first = f.readline().strip()
            if not first:
                continue
            cid = int(first.split()[0])
        by_class.setdefault(cid, []).append(lbl.stem)
    return by_class


def main():
    parser = argparse.ArgumentParser(
        description="从 CEPB 按类别均匀采样并复制到真实数据集"
    )
    parser.add_argument("--cepb", required=True,
                        help="CEPB 数据集根目录（含 images/train, labels/train）")
    parser.add_argument("--real", required=True,
                        help="真实数据集根目录（图片和标签复制到其 images/train, labels/train）")
    parser.add_argument("--total", type=int, default=150,
                        help="总共采样张数，默认 150")
    parser.add_argument("--prefix", default="cepb_",
                        help="文件名前缀，防止和真实图片重名，默认 cepb_")
    parser.add_argument("--seed", type=int, default=41,
                        help="随机种子，默认 42")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览，不实际复制")
    args = parser.parse_args()

    random.seed(args.seed)

    cepb_img = Path(args.cepb) / "images" / "train"
    cepb_lbl = Path(args.cepb) / "labels" / "train"
    real_img = Path(args.real) / "images" / "train"
    real_lbl = Path(args.real) / "labels" / "train"

    for d in [cepb_img, cepb_lbl]:
        if not d.is_dir():
            sys.exit(f"❌ 目录不存在: {d}")

    # 按类别分组
    by_class = group_by_class(cepb_lbl)
    if not by_class:
        sys.exit(f"❌ {cepb_lbl} 下未找到标签文件")

    num_classes = len(by_class)
    per_class = args.total // num_classes
    remainder = args.total % num_classes

    print(f"CEPB: {cepb_lbl}")
    print(f"  总标签文件: {sum(len(v) for v in by_class.values())}")
    print(f"  类别数: {num_classes}")
    print(f"  每类采样: {per_class} + 前 {remainder} 类多 1 张 = {args.total} 总\n")

    # 确保目标目录存在
    if not args.dry_run:
        real_img.mkdir(parents=True, exist_ok=True)
        real_lbl.mkdir(parents=True, exist_ok=True)

    copied = 0
    for cid in sorted(by_class.keys()):
        stems = by_class[cid]
        n = per_class + (1 if cid < remainder else 0)
        sampled = random.sample(stems, min(n, len(stems)))

        for stem in sampled:
            # 找图片（支持多种扩展名）
            img_path = None
            for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
                p = cepb_img / f"{stem}{ext}"
                if p.exists():
                    img_path = p
                    break

            if img_path is None:
                print(f"  ⚠ 图片不存在: {stem}.* ，跳过")
                continue

            new_stem = f"{args.prefix}{stem}"
            dst_img = real_img / f"{new_stem}{img_path.suffix}"
            dst_lbl = real_lbl / f"{new_stem}.txt"

            if args.dry_run:
                print(f"  [{cid:2d}] {img_path.name}  →  {dst_img.name}")
            else:
                shutil.copy2(img_path, dst_img)
                shutil.copy2(cepb_lbl / f"{stem}.txt", dst_lbl)

            copied += 1

    print(f"\n{'[预览] ' if args.dry_run else ''}✓ 采样 {copied} 张 → {args.real}/images/train/")
    if not args.dry_run:
        real_count = len(list(real_img.iterdir()))
        print(f"  真实数据集 train 目录现有 {real_count} 个文件")


if __name__ == "__main__":
    main()