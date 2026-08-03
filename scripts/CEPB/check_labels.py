#!/usr/bin/env python3
# check_labels.py —— 从训练集中每类随机抽取 N 张，绘制 YOLO polygon 标签
"""
用途：检查训练标签是否有错位（标注框/多边形与物体不对齐）。

在容器内直接运行：
    python check_labels.py

输出保存到 OUTPUT_DIR，每类 SAMPLES_PER_CLASS 张图。
"""

import os
import random
import cv2
import numpy as np

# ═══════════════════════════════════════════════════════════
# 配置区（直接在下面改）
# ═══════════════════════════════════════════════════════════

IMAGE_DIR = os.path.expanduser("~/dataset_yolo/images/train")
LABEL_DIR = os.path.expanduser("~/dataset_yolo/labels/train")
OUTPUT_DIR = "/root/yolo"
SAMPLES_PER_CLASS = 10   # 每类抽取多少张

# 类别名（与 data.yaml 一致，class_id → 名称）
CLASS_NAMES = {
    0: "Cheez-it",
    1: "Starkist_Tuna",
    2: "Scissors",
    3: "Frenchs_Mustard",
    4: "Tomato_Soup",
    5: "Foam_Brick",
    6: "Clamp",
    7: "Plastic_Banana",
    8: "Mug",
    9: "meat_can",
}

# 调色板（每类一种颜色，BGR）
COLORS = [
    (0, 255, 0),      # 0 绿
    (255, 0, 0),      # 1 蓝
    (0, 0, 255),      # 2 红
    (255, 255, 0),    # 3 青
    (255, 0, 255),    # 4 品红
    (0, 255, 255),    # 5 黄
    (128, 255, 0),    # 6 草绿
    (255, 128, 0),    # 7 橙
    (128, 0, 255),    # 8 紫
    (0, 128, 255),    # 9 金
]

# ═══════════════════════════════════════════════════════════


def parse_yolo_polygon(line: str):
    parts = line.strip().split()
    if len(parts) < 7:
        return None
    class_id = int(parts[0])
    coords = [(float(parts[i]), float(parts[i + 1])) for i in range(1, len(parts), 2)]
    return class_id, coords


def draw_polygon_on_image(image_bgr, polygons, class_names, colors):
    h, w = image_bgr.shape[:2]
    overlay = image_bgr.copy()

    for class_id, norm_coords in polygons:
        color = colors[class_id % len(colors)]
        pts = np.array(
            [[int(nx * w), int(ny * h)] for nx, ny in norm_coords],
            dtype=np.int32,
        ).reshape((-1, 1, 2))

        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(image_bgr, [pts], isClosed=True, color=color, thickness=2)

        name = class_names.get(class_id, f"cls_{class_id}")
        M = cv2.moments(pts)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                image_bgr,
                (cx - tw // 2 - 4, cy - th - 8),
                (cx + tw // 2 + 4, cy),
                color,
                -1,
            )
            text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
            cv2.putText(
                image_bgr, name,
                (cx - tw // 2, cy - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2,
            )

    cv2.addWeighted(overlay, 0.35, image_bgr, 0.65, 0, image_bgr)


def main():
    random.seed(42)  # 可复现
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 扫描所有 label，收集  class_id → [(label_path, image_path, stem), ...]
    class_pool = {cid: [] for cid in CLASS_NAMES}
    label_files = sorted(os.listdir(LABEL_DIR))

    for fname in label_files:
        if not fname.endswith(".txt"):
            continue

        label_path = os.path.join(LABEL_DIR, fname)
        stem = fname[:-4]

        # 找对应图片
        image_path = os.path.join(IMAGE_DIR, fname.replace(".txt", ".jpg"))
        if not os.path.exists(image_path):
            alt = os.path.join(IMAGE_DIR, fname.replace(".txt", ".png"))
            if os.path.exists(alt):
                image_path = alt
            else:
                continue

        # 读 label，按 class_id 归类
        classes_in_file = set()
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(parts[0])
                classes_in_file.add(cid)

        for cid in classes_in_file:
            if cid in class_pool:
                class_pool[cid].append((label_path, image_path, stem))

    # 打印每类可用数量
    print(f"找到 {len(CLASS_NAMES)} 个类别：")
    for cid in sorted(CLASS_NAMES.keys()):
        total = len(class_pool[cid])
        print(f"  [{cid}] {CLASS_NAMES[cid]:20s}  共 {total} 个样本")

    print(f"\n每类随机抽取 {SAMPLES_PER_CLASS} 张：")

    # 2. 每类随机抽 N 张
    for class_id in sorted(CLASS_NAMES.keys()):
        name = CLASS_NAMES[class_id]
        pool = class_pool[class_id]

        if not pool:
            print(f"  [{class_id}] {name}: ❌ 没有样本")
            continue

        samples = random.sample(pool, min(SAMPLES_PER_CLASS, len(pool)))

        for idx, (label_path, image_path, stem) in enumerate(samples):
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                print(f"    [{idx}] {stem}: ❌ 无法读取")
                continue

            polygons = []
            with open(label_path, "r") as f:
                for line in f:
                    result = parse_yolo_polygon(line)
                    if result is not None:
                        polygons.append(result)

            draw_polygon_on_image(image_bgr, polygons, CLASS_NAMES, COLORS)

            out_path = os.path.join(
                OUTPUT_DIR, f"{class_id:02d}_{name}_{idx:02d}_{stem}.jpg"
            )
            cv2.imwrite(out_path, image_bgr)
            print(f"    [{idx}] {stem}: ✓")

        print(f"  [{class_id}] {name}: 完成 {len(samples)} 张 → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()