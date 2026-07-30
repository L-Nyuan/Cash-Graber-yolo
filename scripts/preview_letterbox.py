import cv2
import numpy as np
import argparse
from pathlib import Path


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    """
    复现 YOLO 的 letterbox 逻辑：
    1. 保持宽高比缩放到 new_shape 以内
    2. 不足的部分用灰色填充
    """
    h, w = img.shape[:2]
    r = new_shape / max(h, w)            # 缩放比例
    if r != 1:
        new_h, new_w = int(h * r), int(w * r)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 填充到 new_shape × new_shape
    h, w = img.shape[:2]
    dh = new_shape - h
    dw = new_shape - w
    top, bottom = dh // 2, dh - dh // 2
    left, right  = dw // 2, dw - dw // 2
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)
    return img, (top, left, r)  # 返回填充量和缩放比


def simulate_yolo_transform(img_path, output_path=None, label_path=None):
    """
    读取原图 → letterbox → 640×640 → 并排对比显示

    label_path: 如果提供标签文件（YOLO polygon 格式），会在变换后图像上画出 mask
    """
    orig = cv2.imread(str(img_path))
    if orig is None:
        print(f"无法读取: {img_path}")
        return

    print(f"原图尺寸: {orig.shape[1]}×{orig.shape[0]}")
    # 模拟 YOLO 变换
    transformed, (pad_top, pad_left, scale) = letterbox(orig, 640)

    h_orig, w_orig = orig.shape[:2]

    # ========== 画标签（如果有） ==========
    if label_path and Path(label_path).exists():
        _draw_labels(orig, label_path, w_orig, h_orig, scale=1.0, pad=(0, 0))
        _draw_labels(transformed, label_path, w_orig, h_orig,
                     scale=scale, pad=(pad_left, pad_top))

    # ========== 并排对比 ==========
    # 原图可能不是 640×640，也 pad 成同样大小方便对比
    orig_square = letterbox(orig, 640, color=(0, 0, 0))[0]

    # 拼接: 原图 | 变换后
    combined = np.hstack([orig_square, transformed])

    # 加文字标注
    cv2.putText(combined, f"Original ({orig.shape[1]}x{orig.shape[0]})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    x_offset = 640 + 10
    cv2.putText(combined, "YOLO input (640x640 letterbox)",
                (x_offset, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 缩放显示（屏幕放不下 1280×640 就缩放）
    disp = cv2.resize(combined, None, fx=0.7, fy=0.7)

    cv2.imshow("YOLO Letterbox — Original vs 640x640 | ESC to quit", disp)

    if output_path:
        cv2.imwrite(str(output_path), combined)
        print(f"已保存: {output_path}")

    print("按 ESC 关闭，按空格切换下一张...")
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    return key  # 返回按键，方便批量遍历时判断
def _draw_labels(img, label_path, orig_w, orig_h, scale, pad):
    """在图像上画出 YOLO polygon mask"""
    colors = [
        (0, 0, 255), (0, 224, 0), (224, 0, 0), (0, 224, 96),
        (96, 0, 224), (224, 96, 0), (96, 224, 224), (224, 96, 224),
        (224, 224, 96), (178, 178, 178)
    ]
    pad_x, pad_y = pad

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            cls_id = int(parts[0])
            points = list(map(float, parts[1:]))
            color = colors[cls_id % len(colors)]

            # 归一化坐标 → 变换后像素坐标
            pts = []
            for i in range(0, len(points), 2):
                x = int(points[i] * orig_w * scale + pad_x)
                y = int(points[i+1] * orig_h * scale + pad_y)
                pts.append([x, y])

            if len(pts) >= 3:
                pts = np.array(pts, dtype=np.int32)
                cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)


def batch_preview(image_dir, label_dir=None, img_pattern="*.png"):
    """批量浏览整个数据集"""
    import glob
    images = sorted(glob.glob(str(Path(image_dir) / img_pattern)))
    print(f"找到 {len(images)} 张图片")

    for i, img_path in enumerate(images):
        label_path = None
        if label_dir:
            lp = Path(label_dir) / (Path(img_path).stem + ".txt")
            if lp.exists():
                label_path = str(lp)
        print(f"\n[{i+1}/{len(images)}] {Path(img_path).name}")
        key = simulate_yolo_transform(img_path, label_path=label_path)
        if key == 27:  # ESC
            print("退出浏览")
            break


# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="模拟 YOLO letterbox 变换，直观查看压缩效果")
    ap.add_argument("image", nargs="?", help="单张图片路径")
    ap.add_argument("-l", "--label", help="对应的 YOLO 标签文件")
    ap.add_argument("-d", "--dir", help="批量模式：图片目录")
    ap.add_argument("--label-dir", help="批量模式：标签目录")
    ap.add_argument("-o", "--output", help="保存对比图到文件")
    ap.add_argument("--pattern", default="*.png", help="图片后缀 (默认 *.png)")
    args = ap.parse_args()

    if args.dir:
        batch_preview(args.dir, args.label_dir, args.pattern)
    elif args.image:
        simulate_yolo_transform(args.image, args.output, args.label)
    else:
        ap.print_help()