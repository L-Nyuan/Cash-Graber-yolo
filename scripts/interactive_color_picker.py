"""
交互式分割图取色工具
用法: python interactive_color_picker.py <scene_id>
示例: python interactive_color_picker.py 1

操作:
  鼠标左键点击  → 终端打印 (x, y) 坐标 + RGB 值 + 量化后 RGB
  鼠标滚轮      → 缩放
  按住中键拖动  → 平移
  按 'q' 键    → 退出
  按 's' 键    → 保存当前视图到 picker_screenshot.png
  按 't' 键    → 切换显示：原始 / 量化后 (step=32)

依赖 (都在你的 yolo 环境里): numpy, pillow, matplotlib, yaml
"""

import sys
import numpy as np
from PIL import Image
import yaml
import re
import matplotlib
matplotlib.use('TkAgg')  # 交互式后端
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


STEP = 32
DATA_DIR = '/root/dataset/'


def load_data(scene_id):
    """读取分割图 + YAML 标注"""
    seg_path = f'{DATA_DIR}left_camera_scene_{scene_id}_segmentation.png'
    yaml_path = f'{DATA_DIR}GT_left_camera_{scene_id}.yaml'

    seg = np.array(Image.open(seg_path))
    rgb = seg[:, :, :3]
    quantized = (rgb // STEP) * STEP

    with open(yaml_path) as f:
        gt = yaml.safe_load(f)

    return rgb, quantized, gt


class ColorPicker:
    def __init__(self, rgb, quantized, gt, scene_id):
        self.rgb = rgb
        self.quantized = quantized
        self.gt = gt
        self.scene_id = scene_id
        self.show_quantized = False  # 当前显示模式
        self.points = []  # 已标记的点

        self.fig, self.ax = plt.subplots(figsize=(14, 10))



        self._update_image()
        self._draw_gt_centroids()

        # 标题 + 操作提示
        title = (
            f"Scene {scene_id}  |  "
            f"左键:取色 | 滚轮:缩放 | 中键拖拽:平移 | t:切换量化 | s:截图 | q:退出"
        )
        self.ax.set_title(title, fontsize=11, fontweight='bold')
        self.ax.axis('off')

        # 绑定事件
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # 状态栏
        self.status_text = self.ax.text(
            0.5, -0.02, '', transform=self.ax.transAxes,
            ha='center', fontsize=10, color='gray'
        )

        # 显示 GT 物体信息
        self._print_gt_summary()

    def _update_image(self):
        """切换显示模式"""
        img = self.quantized if self.show_quantized else self.rgb
        if hasattr(self, 'im'):
            self.im.set_data(img)
        else:
            self.im = self.ax.imshow(img)
        mode = "量化后 (step=32)" if self.show_quantized else "原始"
        self.ax.set_xlabel(f"当前显示: {mode}  |  {self.rgb.shape[1]}×{self.rgb.shape[0]}",
                           fontsize=9)
        self.fig.canvas.draw_idle()

    def _draw_gt_centroids(self):
        """在图上标出 YAML 中每个物体的 2D_centroid"""
        for key, info in self.gt['objects'].items():
            class_name = re.sub(r'^\d+-', '', key)
            x, y = info['2D_centroid']
            vis = info['visibility'][0]
            color = 'lime' if vis > 0.5 else 'orange' if vis > 0.2 else 'red'
            self.ax.plot(x, y, 'o', color=color, markersize=8,
                         markeredgecolor='black', markeredgewidth=0.5)
            self.ax.annotate(
                f"{class_name}\nvis={vis:.2f}",
                (x + 15, y), fontsize=6.5, color=color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6),
            )

    def _print_gt_summary(self):
        print(f"\n{'='*60}")
        print(f"Scene {self.scene_id}  —  GT 物体列表")
        print(f"{'='*60}")
        print(f"{'物体名':<20} {'2D_centroid':<16} {'visibility':<10}")
        print(f"{'-'*46}")
        for key, info in self.gt['objects'].items():
            class_name = re.sub(r'^\d+-', '', key)
            x, y = info['2D_centroid']
            vis = info['visibility'][0]
            print(f"{class_name:<20} ({x:4d}, {y:4d})        {vis:.3f}")
        print(f"{'='*60}\n")

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button != 1:  # 只响应左键
            return

        x, y = int(round(event.xdata)), int(round(event.ydata))

        if x < 0 or x >= self.rgb.shape[1] or y < 0 or y >= self.rgb.shape[0]:
            return

        # 取色
        raw_color = self.rgb[y, x]
        q_color = self.quantized[y, x]

        # 在图上标记
        dot = Circle((x, y), radius=6, color='red', fill=False, linewidth=2)
        self.ax.add_patch(dot)
        self.points.append(dot)

        # 显示十字线标记
        cross_h = self.ax.axhline(y, color='red', alpha=0.4, linewidth=0.5)
        cross_v = self.ax.axvline(x, color='red', alpha=0.4, linewidth=0.5)
        self.points.append(cross_h)
        self.points.append(cross_v)

        # 终端输出
        print(f"🖱️  点击: 坐标=({x}, {y})  |  "
              f"原始RGB=({raw_color[0]:3d}, {raw_color[1]:3d}, {raw_color[2]:3d})  |  "
              f"量化RGB=({q_color[0]:3d}, {q_color[1]:3d}, {q_color[2]:3d})")

        # 最多保留 20 个标记点
        while len(self.points) > 60:
            p = self.points.pop(0)
            p.remove()

        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == 'q':
            print("\n退出取色工具。")
            plt.close(self.fig)
        elif event.key == 't':
            self.show_quantized = not self.show_quantized
            self._update_image()
            mode = "量化后 (step=32)" if self.show_quantized else "原始"
            print(f"  → 切换到: {mode}")
        elif event.key == 's':
            path = f'picker_scene_{self.scene_id}.png'
            self.fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  → 截图已保存到: {path}")
        elif event.key == 'r':
            # 清除所有标记
            for p in self.points:
                p.remove()
            self.points.clear()
            self.fig.canvas.draw_idle()
            print("  → 已清除所有标记")

    def run(self):
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python interactive_color_picker.py <scene_id>")
        print("示例: python interactive_color_picker.py 1")
        sys.exit(1)

    scene_id = int(sys.argv[1])
    rgb, quantized, gt = load_data(scene_id)

    picker = ColorPicker(rgb, quantized, gt, scene_id)
    picker.run()
