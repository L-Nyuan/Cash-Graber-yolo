# visualization_utils.py —— YOLO 分割结果可视化工具
"""
在图像上绘制分割 mask，独立模块，不影响推理节点结构。
"""

import numpy as np
import cv2


# 调色板：为不同类别预分配颜色（20 种，超出循环使用）
_PALETTE = [
    (0, 255, 0),      # 绿
    (255, 0, 0),      # 蓝
    (0, 0, 255),      # 红
    (255, 255, 0),    # 青
    (255, 0, 255),    # 品红
    (0, 255, 255),    # 黄
    (128, 255, 0),    # 草绿
    (255, 128, 0),    # 橙
    (128, 0, 255),    # 紫
    (255, 128, 128),  # 粉
    (0, 128, 255),    # 金
    (128, 255, 128),  # 浅绿
    (255, 0, 128),    # 玫红
    (0, 255, 128),    # 春绿
    (128, 0, 128),    # 紫红
    (255, 255, 128),  # 淡黄
    (128, 128, 255),  # 淡紫
    (0, 128, 128),    # 深青
    (128, 128, 0),    # 橄榄
    (128, 0, 0),      # 深蓝
]


def draw_segmentation(
    image_rgb: np.ndarray,
    objects: list,
    show_labels: bool = True,
    show_confidence: bool = True,
    mask_alpha: float = 0.35,
    line_thickness: int = 2,
) -> np.ndarray:
    """在 RGB 图像上绘制每个物体的 mask 轮廓和标签。

    Args:
        image_rgb:  (H, W, 3) uint8, RGB 顺序
        objects:    YOLO 推理结果列表，每项含 class_name, confidence, mask
        show_labels:      是否在 mask 中心绘制类别名
        show_confidence:  是否在标签中显示置信度
        mask_alpha:       半透明填充透明度（0=完全透明, 1=完全不透明）
        line_thickness:   轮廓线宽

    Returns:
        (H, W, 3) uint8, BGR 顺序（可直接 cv2.imwrite / cv2.imshow）

    Examples:
        # 基本用法
        vis = draw_segmentation(rgb_image, objects)
        cv2.imwrite("/tmp/debug.png", vis)

        # 不显示标签（只看 mask 区域）
        vis = draw_segmentation(rgb_image, objects, show_labels=False)
    """
    # RGB → BGR（OpenCV 用 BGR）
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()

    # 半透明填充层
    overlay = canvas.copy()

    for idx, obj in enumerate(objects):
        color = _PALETTE[idx % len(_PALETTE)]
        mask = obj["mask"]        # (H, W) bool

        # 1) 半透明填充 mask 区域
        overlay[mask] = color

        # 2) 提取轮廓并绘制边界
        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, color, line_thickness)

        # 3) 标签文字（在 mask 像素的中心）
        if show_labels and contours:
            # 用 mask 非零像素的均值做中心（比最大轮廓的矩更鲁棒）
            ys, xs = mask.nonzero()
            if len(xs) > 0:
                cx, cy = int(np.mean(xs)), int(np.mean(ys))

                label = obj["class_name"]
                if show_confidence:
                    label += f" {obj['confidence']:.2f}"

                # 文字背景
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    canvas,
                    (cx - tw // 2 - 3, cy - th - 6),
                    (cx + tw // 2 + 3, cy),
                    color,
                    -1,
                )
                # 文字（黑或白，看背景亮度）
                text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
                cv2.putText(
                    canvas,
                    label,
                    (cx - tw // 2, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    text_color,
                    1,
                )

    # 混合半透明填充
    cv2.addWeighted(overlay, mask_alpha, canvas, 1 - mask_alpha, 0, canvas)

    return canvas


def save_debug_image(
    image_rgb: np.ndarray,
    objects: list,
    output_path: str,
    **kwargs,
):
    """便捷函数：绘制分割结果并直接保存到文件。

    Args:
        image_rgb:   RGB 图像
        objects:     检测结果列表
        output_path: 保存路径，如 "/tmp/debug_0001.png"
        **kwargs:    透传给 draw_segmentation 的参数
    """
    vis = draw_segmentation(image_rgb, objects, **kwargs)
    cv2.imwrite(output_path, vis)