#!/usr/bin/env python3
# ros2_inference_node.py —— YOLO11-seg ROS2 推理节点
"""
从 color 话题取流 → YOLO11-seg 推理 → 输出分割结果。

参数（可通过 --ros-args -p 覆盖）:
  model_path:  权重路径，默认 /root/yolo/result/exp_1/best.pt
  imgsz:       推理尺寸，默认 640
  conf:        置信度阈值，默认 0.5
  iou:         NMS IoU 阈值，默认 0.7
  debug:       是否保存可视化结果，默认 False
  debug_dir:   可视化输出目录，默认 /tmp/yolo_debug
"""

import sys
import time
import os
import numpy as np
import rclpy
import cv2
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo

from image_utils import ros_image_to_numpy
from yolo_inference import YOLOSegInference
from visualization_utils import save_debug_image


class YOLOInferenceNode(Node):
    """订阅 color 话题 → YOLO 推理 → 输出分割结果。"""

    def __init__(self):
        super().__init__("yolo_inference_node")

        # ── 可配置参数 ──────────────────────────────────────────
        self.declare_parameter("model_path", "/root/yolo/result/exp_3/last.pt")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.8)
        self.declare_parameter("iou", 0.7)
        self.declare_parameter("debug", True)
        self.declare_parameter("debug_dir", "/root/yolo/yolo_debug")

        self.yolo = YOLOSegInference(
            model_path=self.get_parameter("model_path").value,
            imgsz=self.get_parameter("imgsz").value,
            conf=self.get_parameter("conf").value,
            iou=self.get_parameter("iou").value,
        )

        self._debug = self.get_parameter("debug").value
        self._debug_dir = self.get_parameter("debug_dir").value
        if self._debug:
            os.makedirs(self._debug_dir, exist_ok=True)
            self.get_logger().info(f"[DEBUG] 可视化保存到: {self._debug_dir}")

        # QoS
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)

        # 订阅
        self.color_sub = self.create_subscription(
            Image, "/Wrist_Camera/d435i/color/image_raw", self._color_cb, qos)
        self.info_sub = self.create_subscription(
            CameraInfo, "/Wrist_Camera/d435i/color/camera_info", self._info_cb, qos)

        self._latest_color = None
        self._latest_info = None
        self._frame_count = 0
        self._total_inference_ms = 0.0

        # 推理定时器
        self._timer = self.create_timer(0.05, self._inference_loop)
        # 状态打印
        self._status_timer = self.create_timer(5.0, self._print_status)

        self.get_logger().info(
            f"YOLOInferenceNode 启动 | model={self.get_parameter('model_path').value} | "
            f"imgsz={self.get_parameter('imgsz').value} | debug={self._debug}"
        )

    # ── 回调 ────────────────────────────────────────────────────

    def _color_cb(self, msg: Image):
        try:
            self._latest_color = ros_image_to_numpy(msg)
        except Exception as e:
            self.get_logger().error(f"解码失败: {e}")

    def _info_cb(self, msg: CameraInfo):
        self._latest_info = msg

    # ── 推理循环 ────────────────────────────────────────────────

    def _inference_loop(self):
        image = self._latest_color
        if image is None:
            return
        self._latest_color = None
        self._frame_count += 1

        # 查看rgb图像是否正确
        cv2.imwrite(
        "/root/yolo/rgb_test/test_ros.png",
        image
        )

        t0 = time.perf_counter()
        result = self.yolo.predict(image)
        total_ms = (time.perf_counter() - t0) * 1000
        self._total_inference_ms += result["inference_time_ms"]

        n = len(result["objects"])
        names = ", ".join(o["class_name"] for o in result["objects"][:5])
        self.get_logger().info(
            f"[#{self._frame_count}] {n} 物体: {names} | "
            f"模型={result['inference_time_ms']:.0f}ms | "
            f"总={total_ms:.0f}ms"
        )

        # ── 调试可视化 ─────────────────────────────────────────
        if self._debug and n > 0:
            debug_path = os.path.join(
                self._debug_dir, f"frame_{self._frame_count:06d}.png"
            )
            save_debug_image(image, result["objects"], debug_path)
            self.get_logger().debug(f"  可视化已保存: {debug_path}")

    def _print_status(self):
        if self._frame_count == 0:
            self.get_logger().info("等待图像...")
            return
        avg = self._total_inference_ms / self._frame_count
        self.get_logger().info(
            f"累计 {self._frame_count} 帧 | 平均推理 {avg:.0f}ms | "
            f"等效 {1000/avg:.0f} FPS"
        )


def main():
    rclpy.init(args=sys.argv)
    try:
        rclpy.spin(YOLOInferenceNode())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()