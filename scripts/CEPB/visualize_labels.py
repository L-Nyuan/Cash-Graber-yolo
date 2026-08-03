#!/usr/bin/env python3
"""
visualize_labels.py — 独立脚本：将 YOLO polygon 标签画在对应 RGB 图像上。

用法:
    python visualize_labels.py <image.jpg> <label.txt> [--output out.png]

示例:
    python visualize_labels.py \
        /root/yolo/test_output/images/1_view_0_dir.jpg \
        /root/yolo/test_output/labels/1_view_0_dir.txt \
        --output debug_polygon.png
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

# ============================================================
# 与 CEPBProcessor.CLASS_NAMES 保持严格一致（sorted 后的顺序）
# ============================================================
CLASS_NAMES = sorted([
    'Cheez-it', 'Starkist_Tuna', 'Scissors', 'Frenchs_Mustard', 'Tomato_Soup',
    'Foam_Brick', 'Clamp', 'Plastic_Banana', 'Mug', 'meat_can',
])

# ============================================================
# 每个 class_id 的绘制颜色 (R, G, B)，0-255
# ============================================================
CLASS_COLORS_RGB = {
    0:  (227,  26,  28),   # 红     — Cheez-it
    1:  (56,  125, 184),   # 蓝     — Starkist_Tuna
    2:  (77,  176,  74),   # 绿     — Scissors
    3:  (255, 204,   0),   # 金黄   — Frenchs_Mustard
    4:  (230, 102,   0),   # 橙     — Tomato_Soup
    5:  (140,  61, 153),   # 紫     — Foam_Brick
    6:  (0,   191, 191),   # 青     — Clamp
    7:  (245, 222,  77),   # 淡黄   — Plastic_Banana
    8:  (128, 128, 140),   # 灰蓝   — Mug
    9:  (237,  71, 130),   # 粉红   — meat_can
}

IMG_W = 2048   # CEPB 图像宽度
IMG_H = 1536   # CEPB 图像高度


def load_label(txt_path: str) -> list[tuple[int, np.ndarray]]:
    """
    读取 YOLO polygon 标签文件。

    格式: class_id x1 y1 x2 y2 ... xn yn
    坐标是归一化的 [0, 1]，这里反归一化回像素坐标。

    返回: [(class_id, points_xy), ...]，其中 points_xy 形状 (N, 2)，int 像素坐标
    """
    instances = []

    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 7:           # 至少: class_id + 3 个点 (x,y) = 7 个数
                print(f"  [WARN] 跳过行（点数不足 3）: {line[:80]}...")
                continue

            class_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]

            if len(coords) % 2 != 0:
                print(f"  [WARN] 跳过行（坐标数不是偶数）: {line[:80]}...")
                continue

            # 每两个数组成一个 (x, y)，反归一化
            pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= IMG_W
            pts[:, 1] *= IMG_H
            pts = pts.astype(np.int32)

            instances.append((class_id, pts))

    return instances


def draw_labels(image: Image.Image, instances: list[tuple[int, np.ndarray]],
                alpha: float = 0.35, line_width: int = 3) -> Image.Image:
    """
    在原图上绘制所有 polygon 实例。

    - 先画半透明填充（alpha 控制透明度）
    - 再画不透明轮廓线 + 顶点圆点
    - 在质心位置标注类别名

    返回带标注的 PIL Image（RGB）。
    """
    # 转为 RGBA 以支持透明度，绘制完再转回 RGB
    base = image.convert('RGBA')

    # --- 第一步：为每个物体创建独立的半透明填充层 ---
    overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)

    for class_id, pts in instances:
        color_rgb = CLASS_COLORS_RGB.get(class_id, (200, 200, 200))
        fill_color = (*color_rgb, int(255 * alpha))          # (R, G, B, A)

        # polygon 填充
        draw_ov.polygon([(x, y) for x, y in pts], fill=fill_color)

    base = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(base)

    # --- 第二步：画轮廓线、顶点、类别名 ---
    for class_id, pts in instances:
        color_rgb = CLASS_COLORS_RGB.get(class_id, (200, 200, 200))

        # 轮廓线（不透明）
        draw.line([(x, y) for x, y in pts] + [(pts[0][0], pts[0][1])],
                  fill=color_rgb, width=line_width)

        # 每个顶点画一个小圆点
        r = max(line_width + 1, 4)
        for x, y in pts:
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color_rgb)

        # 类别名标注在质心
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        class_name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else f"cls_{class_id}"

        # 文字背后画一个半透明底框，确保可读
        bbox = draw.textbbox((cx, cy), class_name)          # 返回 (left, top, right, bottom)
        draw.rectangle(
            (bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2),
            fill=(*color_rgb, 200),
        )
        draw.text((cx, cy), class_name, fill='white')

    return base.convert('RGB')


def main():
    parser = argparse.ArgumentParser(
        description='将 YOLO polygon 标签画在对应图像上'
    )
    parser.add_argument('image',   help='输入 jpg 图像路径')
    parser.add_argument('label',   help='输入 txt 标签路径')
    parser.add_argument('--output', '-o', default=None,
                        help='输出 png 路径（默认: <label>_vis.png）')
    parser.add_argument('--alpha', '-a', type=float, default=0.35,
                        help='填充透明度 0-1 (默认 0.35)')
    parser.add_argument('--line-width', '-l', type=int, default=3,
                        help='轮廓线宽 (默认 3)')
    args = parser.parse_args()

    # 校验输入
    for p, name in [(args.image, '图像'), (args.label, '标签')]:
        if not os.path.isfile(p):
            print(f"[ERROR] {name}文件不存在: {p}")
            sys.exit(1)

    output = args.output or os.path.splitext(args.label)[0] + '_vis.png'

    # ---- 加载 ----
    print(f"读取图像: {args.image}")
    img = Image.open(args.image).convert('RGB')

    print(f"读取标签: {args.label}")
    instances = load_label(args.label)

    if not instances:
        print("[ERROR] 标签文件为空或格式无效，无任何 polygon 可绘制。")
        sys.exit(1)

    print(f"  找到 {len(instances)} 个实例:")
    for cid, pts in instances:
        name = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else f"cls_{cid}"
        print(f"    class_id={cid} ({name}) — {len(pts)} 顶点")

    # ---- 绘制 ----
    print(f"绘制 polygon (alpha={args.alpha}, line_width={args.line_width}) ...")
    result = draw_labels(img, instances, alpha=args.alpha, line_width=args.line_width)

    # ---- 保存 ----
    result.save(output)
    print(f"已保存: {output}")


if __name__ == '__main__':
    main()
