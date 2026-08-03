#!/usr/bin/env python3
"""
debug_centers.py — 独立脚本：可视化 K-Means 聚类中心颜色 vs 量化后 seg 图像。

用法:
    python debug_centers.py <scene_index> <input_dir> [--output out.png]

示例:
    python debug_centers.py 1000 /root/dataset/ --output centers_check.png
"""

import argparse
import os
import re
import sys

import numpy as np
import yaml
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ========== 配置 ==========
STEP = 10
IMG_W = 2048
IMG_H = 1536
VIS_THRESH = 0.2


def load_yaml_objects(yaml_path: str) -> list[dict]:
    """读取 yaml，返回可见物体的列表 [{class_name, cx, cy, vis}, ...]"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    objects = []
    for key, info in data['objects'].items():
        class_name = re.sub(r'^\d+-', '', key)
        x, y = info['2D_centroid']
        vis = info['visibility'][0]
        objects.append({'class_name': class_name, 'cx': x, 'cy': y, 'vis': vis})
    return objects


def build_centers(quantized, objects):
    """
    完全复刻 seg_to_cluster 中的逻辑，返回 (centers, labels, marker_positions)。

    centers:          [(R,G,B), ...]  聚类中心颜色
    labels:           [str, ...]      每个中心的类名
    marker_positions: [(x, y), ...]   该中心在图像上的取样位置（没有位置的填 None）

    聚类中心顺序:
      0: 黑色背景 (四角平均)
      1: 白色桌面 (250, 250, 250)
      2+: 各物体质心色
    """
    # ---- 黑色背景 ----
    corners = np.array([
        quantized[0, 0, :3],
        quantized[0, IMG_W - 1, :3],
        quantized[IMG_H - 1, 0, :3],
        quantized[IMG_H - 1, IMG_W - 1, :3],
    ])
    bg_black = tuple(np.mean(corners, axis=0).astype(np.uint8).tolist())

    # ---- 白色桌面 ----
    white_val = (255 // STEP) * STEP
    bg_white = (white_val, white_val, white_val)

    centers = [bg_black, bg_white]
    labels = ['__background__', '__table__']
    markers = [None, None]          # 这两个没有单一取样点

    # ---- 各物体 ----
    for obj in objects:
        if obj['vis'] < VIS_THRESH:
            continue
        cx, cy = int(obj['cx']), int(obj['cy'])
        color = tuple(quantized[cy, cx].tolist())
        centers.append(color)
        labels.append(obj['class_name'])
        markers.append((cx, cy))

    return centers, labels, markers


def draw_centers_debug(quantized_img, centers, labels, markers, save_path):
    """
    生成调试图：
      左图：量化后 seg 图像 + 取样位置标记
      右图：聚类中心色块列表

    quantized_img: (H, W, 3) uint8 — 量化后的 seg 图像
    centers:       [(R, G, B), ...] — 聚类中心颜色
    labels:        [str, ...] — 对应类名
    markers:       [None | (x, y), ...] — 取样位置（像素坐标）
    """
    n = len(centers)

    fig = plt.figure(figsize=(20, 10))

    # ========================
    # 左图：量化图像 + 取样标记
    # ========================
    ax_img = fig.add_axes([0.02, 0.05, 0.55, 0.90])
    ax_img.imshow(quantized_img)
    ax_img.set_title(f"量化图像  step={STEP}  |  聚类中心取样位置",
                     fontsize=13, fontweight='bold')

    for i, (color, label, pos) in enumerate(zip(centers, labels, markers)):
        r, g, b = color
        hex_color = f'#{r:02x}{g:02x}{b:02x}'

        if pos is not None:
            x, y = pos
            # 十字标记
            ax_img.plot(x, y, marker='o', markersize=10,
                        markerfacecolor=np.array(color) / 255,
                        markeredgecolor='white' if sum(color) < 384 else 'black',
                        markeredgewidth=2)
            # 编号
            offset = 30
            ax_img.annotate(
                f"{i}", (x, y),
                textcoords="offset points", xytext=(offset, -offset),
                fontsize=9, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=np.array(color)/255,
                          edgecolor='white', alpha=0.9),
            )
        else:
            # 背景/桌面没有具体取样点，在图上用一个图例色块代替
            # 放在左下角，ID 作为标记
            pass

    ax_img.axis('off')

    # 在图像下方加一行注释：哪些没有取样点
    no_marker = [f"#{i} {l}" for i, (l, m) in enumerate(zip(labels, markers)) if m is None]
    if no_marker:
        ax_img.text(0.5, -0.03,
                    f"无固定取样点: {', '.join(no_marker)}",
                    transform=ax_img.transAxes, fontsize=9, color='gray',
                    ha='center')

    # ========================
    # 右图：色块列表
    # ========================
    ax_swatch = fig.add_axes([0.62, 0.05, 0.35, 0.90])
    ax_swatch.set_xlim(0, 100)
    ax_swatch.set_ylim(-n * 12 - 2, 2)
    ax_swatch.axis('off')
    ax_swatch.set_title("聚类中心颜色列表  |  编号 = K-Means init 顺序",
                        fontsize=13, fontweight='bold')

    row_h = 11
    for i, (color, label, pos) in enumerate(zip(centers, labels, markers)):
        y_base = -(i + 1) * row_h
        r, g, b = color
        hex_str = f'#{r:02x}{g:02x}{b:02x}'
        norm = np.array(color) / 255

        # 色块
        rect = Rectangle((2, y_base), 8, 8, linewidth=1.5,
                         edgecolor='black', facecolor=norm)
        ax_swatch.add_patch(rect)

        white_val = (255 // STEP) * STEP

        # 文字
        if pos is not None:
            pos_str = f"取样 @ ({pos[0]}, {pos[1]})"
        elif label == '__background__':
            pos_str = "四角平均"
        elif label == '__table__':
            pos_str = f"固定: ({white_val}, {white_val}, {white_val})"
        else:
            pos_str = "(无)"

        ax_swatch.text(12, y_base + 4,
                       f"[{i}] {label}",
                       fontsize=10, fontweight='bold', va='center')
        ax_swatch.text(12, y_base + 0,
                       f"RGB({r:3d}, {g:3d}, {b:3d})  {hex_str}  |  {pos_str}",
                       fontsize=8, va='center', color='#444444')

    # 保存
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='可视化 K-Means 聚类中心颜色'
    )
    parser.add_argument('scene_index', type=int, help='场景编号，如 1000')
    parser.add_argument('input_dir', help='数据集目录，如 /root/dataset/')
    parser.add_argument('--output', '-o', default=None,
                        help='输出 PNG 路径（默认: ./centers_scene_N.png）')
    parser.add_argument('--view', '-v', type=int, default=0,
                        choices=[0, 1, 2], help='视角: 0=left, 1=middle, 2=right (默认 0)')
    args = parser.parse_args()

    # ---- 构建路径 ----
    view_names = ['left', 'middle', 'right']
    vn = view_names[args.view]
    seg_path = os.path.join(args.input_dir, f'{vn}_camera_scene_{args.scene_index}_segmentation.png')
    yaml_path = os.path.join(args.input_dir, f'GT_{vn}_camera_{args.scene_index}.yaml')

    for p, name in [(seg_path, 'seg'), (yaml_path, 'yaml')]:
        if not os.path.isfile(p):
            print(f"[ERROR] {name} 文件不存在: {p}")
            sys.exit(1)

    output = args.output or f'centers_scene_{args.scene_index}_view_{args.view}.png'

    # ---- 读取 ----
    print(f"场景 {args.scene_index}, 视角 {args.view} ({vn})")
    print(f"  seg:  {seg_path}")
    print(f"  yaml: {yaml_path}")

    img = np.array(Image.open(seg_path))
    rgb = img[:, :, :3]
    quantized = (rgb // STEP) * STEP

    objects = load_yaml_objects(yaml_path)
    visible = [o for o in objects if o['vis'] >= VIS_THRESH]
    print(f"  物体: {len(objects)} 个 (可见度≥{VIS_THRESH}: {len(visible)} 个)")

    # ---- 复刻聚类中心 ----
    centers, labels, markers = build_centers(quantized, objects)
    print(f"  聚类中心: {len(centers)} 个")

    for i, (c, l, m) in enumerate(zip(centers, labels, markers)):
        pos_info = f"@ ({m[0]}, {m[1]})" if m else "(无固定点)"
        print(f"    [{i}] {l:20s}  RGB({c[0]:3d},{c[1]:3d},{c[2]:3d})  {pos_info}")

    # ---- 画图 ----
    draw_centers_debug(quantized, centers, labels, markers, output)


if __name__ == '__main__':
    main()
