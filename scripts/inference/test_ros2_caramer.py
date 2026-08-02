#!/usr/bin/env python3
"""
ROS2 相机通信测试脚本（无 cv_bridge 版本）
==========================================
绕过 cv_bridge 的 NumPy 1.x/2.x 不兼容问题，直接手动解码
ROS sensor_msgs/Image 和 sensor_msgs/PointCloud2 消息。

使用方法：
  1. conda activate yolo
  2. source /opt/ros/humble/setup.bash
  3. python test_ros2_camera.py

前提：
  - realsense2_camera 节点正在运行
  - yolo conda 环境（NumPy 2.x + torch 2.5 不受影响）
"""

import sys
import time
import os
import struct

# ── ROS2 ────────────────────────────────────────────────────────────────
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# ── 消息类型 ────────────────────────────────────────────────────────────
from sensor_msgs.msg import Image, PointCloud2, CameraInfo

# ── 数学库 ──────────────────────────────────────────────────────────────
import numpy as np
import cv2  # 仅用于 imwrite 存图和 color map，不涉及 cv_bridge

# ── 输出目录 ────────────────────────────────────────────────────────────
OUTPUT_DIR = "/tmp/camera_test_output"


# ═══════════════════════════════════════════════════════════════════════════
# 手动 Image 解码（替代 cv_bridge）
# ═══════════════════════════════════════════════════════════════════════════

# ROS sensor_msgs/Image encoding → numpy dtype 映射
_ENCODING_TO_DTYPE = {
    "rgb8":   (np.uint8,  3),
    "rgba8":  (np.uint8,  4),
    "bgr8":   (np.uint8,  3),
    "bgra8":  (np.uint8,  4),
    "mono8":  (np.uint8,  1),
    "8UC1":   (np.uint8,  1),
    "16UC1":  (np.uint16, 1),
    "32FC1":  (np.float32, 1),
    "16SC1":  (np.int16,  1),
    "32SC1":  (np.int32,  1),
}


def ros_image_to_numpy(msg: Image) -> np.ndarray:
    """
    把 ROS sensor_msgs/Image 直接解码成 numpy 数组，不经过 cv_bridge。

    Args:
        msg: sensor_msgs/Image 消息

    Returns:
        numpy 数组，shape 为 (height, width) 或 (height, width, channels)
    """
    if msg.encoding not in _ENCODING_TO_DTYPE:
        # fallback: 尝试用 cv2 直接解码裸数据（JPEG/PNG 压缩格式）
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        return cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)

    dtype, channels = _ENCODING_TO_DTYPE[msg.encoding]
    raw = np.frombuffer(msg.data, dtype=dtype)

    if channels == 1:
        raw = raw.reshape((msg.height, msg.width))
    else:
        raw = raw.reshape((msg.height, msg.width, channels))

    return raw


_DATATYPE_SIZES = {
    1: (1, np.int8),       # INT8
    2: (1, np.uint8),      # UINT8
    3: (2, np.int16),      # INT16
    4: (2, np.uint16),     # UINT16
    5: (4, np.uint32),     # UINT32
    6: (8, np.float64),    # FLOAT64
    7: (4, np.float32),    # FLOAT32
    8: (4, np.int32),      # INT32
}


def ros_pointcloud_to_numpy(msg: PointCloud2) -> np.ndarray:
    """
    把 ROS sensor_msgs/PointCloud2 解码成 numpy 结构化数组。

    正确处理 point_step 中的 padding 字节（RealSense 点云通常有 padding）。
    """
    names = []
    formats = []
    offsets = []

    for field in msg.fields:
        if field.datatype not in _DATATYPE_SIZES:
            continue
        byte_size, np_type = _DATATYPE_SIZES[field.datatype]
        names.append(field.name)
        formats.append(np_type)
        offsets.append(field.offset)

    dtype = np.dtype({
        "names": names,
        "formats": formats,
        "offsets": offsets,
        "itemsize": msg.point_step,  # 关键：point_step 含 padding
    })

    n_points = len(msg.data) // msg.point_step
    return np.frombuffer(msg.data, dtype=dtype, count=n_points)


# ═══════════════════════════════════════════════════════════════════════════
# 测试节点
# ═══════════════════════════════════════════════════════════════════════════

class CameraTestNode(Node):
    """订阅 RealSense 话题，验证数据是否正常到达。"""

    def __init__(self):
        super().__init__("camera_test_node")

        self.start_time = time.time()

        # QoS —— 匹配 RealSense ROS2 driver 的默认策略
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── 话题订阅 ──────────────────────────────────────────────────
        self.color_sub = self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.color_callback,
            sensor_qos,
        )

        self.depth_sub = self.create_subscription(
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",
            self.depth_callback,
            sensor_qos,
        )

        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            "/camera/camera/depth/color/points",
            self.pointcloud_callback,
            sensor_qos,
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self.camera_info_callback,
            sensor_qos,
        )

        # ── 计数器 ────────────────────────────────────────────────────
        self.color_count = 0
        self.depth_count = 0
        self.pointcloud_count = 0
        self.camera_info_received = False

        self.last_color_ts = None
        self.last_depth_ts = None
        self.last_pointcloud_ts = None

        # 保存前 3 帧到磁盘
        self.max_saved = 3
        self.saved_color = 0
        self.saved_depth = 0

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 30 秒后自动停止
        self.timer = self.create_timer(30.0, self.shutdown_callback)
        # 每 2 秒打印状态
        self.status_timer = self.create_timer(2.0, self.print_status)

        self.get_logger().info("=" * 60)
        self.get_logger().info("CameraTestNode 启动（无 cv_bridge 版本）")
        self.get_logger().info(f"输出目录: {OUTPUT_DIR}")
        self.get_logger().info("=" * 60)

    # ── 回调 ────────────────────────────────────────────────────────────

    def color_callback(self, msg: Image):
        self.color_count += 1

        try:
            # 手动解码，不经过 cv_bridge
            cv_img = ros_image_to_numpy(msg)
            h, w = cv_img.shape[:2]

            if self.color_count == 1:
                self.get_logger().info(
                    f"[COLOR] 首帧到达 ✅  "
                    f"分辨率={w}x{h}, "
                    f"encoding={msg.encoding}"
                )

            # 保存前几帧
            if self.saved_color < self.max_saved:
                # 如果是 RGB 转 BGR 给 OpenCV 存
                if msg.encoding == "rgb8":
                    save_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
                else:
                    save_img = cv_img
                fname = os.path.join(OUTPUT_DIR, f"color_{self.saved_color:04d}.png")
                cv2.imwrite(fname, save_img)
                self.get_logger().info(f"[COLOR] 已保存: {fname}")
                self.saved_color += 1

        except Exception as e:
            self.get_logger().error(f"[COLOR] 解码失败: {e}")

    def depth_callback(self, msg: Image):
        self.depth_count += 1

        try:
            cv_img = ros_image_to_numpy(msg)  # shape: (480, 640), dtype=uint16
            h, w = cv_img.shape[:2]

            # 有效深度范围
            valid = cv_img[cv_img > 0]
            min_val = float(np.min(valid)) if len(valid) > 0 else 0.0
            max_val = float(np.max(valid)) if len(valid) > 0 else 0.0

            if self.depth_count == 1:
                self.get_logger().info(
                    f"[DEPTH] 首帧到达 ✅  "
                    f"分辨率={w}x{h}, "
                    f"encoding={msg.encoding}, "
                    f"深度范围=[{min_val:.0f}, {max_val:.0f}] mm"
                )

            # 保存伪彩色深度图
            if self.saved_depth < self.max_saved:
                vis = cv_img.astype(np.float32) / 1000.0  # mm → m
                mask = vis > 0
                if np.any(mask):
                    vis[mask] = (vis[mask] - vis[mask].min()) / (
                        vis[mask].max() - vis[mask].min() + 1e-8
                    )
                vis = (vis * 255).astype(np.uint8)
                vis_colored = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                fname = os.path.join(OUTPUT_DIR, f"depth_{self.saved_depth:04d}.png")
                cv2.imwrite(fname, vis_colored)
                self.get_logger().info(f"[DEPTH] 已保存: {fname}")
                self.saved_depth += 1

        except Exception as e:
            self.get_logger().error(f"[DEPTH] 解码失败: {e}")

    def pointcloud_callback(self, msg: PointCloud2):
        self.pointcloud_count += 1

        if self.pointcloud_count == 1:
            fields = [(f.name, f.datatype) for f in msg.fields]
            dt_map = {7: "FLOAT32", 6: "FLOAT64", 4: "UINT16"}
            self.get_logger().info(
                f"[POINTCLOUD] 首帧到达 ✅  "
                f"height={msg.height}, "
                f"width={msg.width}, "
                f"fields={[(n, dt_map.get(t, str(t))) for n, t in fields]}"
            )

            # 采样前几个点
            pc_data = ros_pointcloud_to_numpy(msg)
            n_sample = min(5, len(pc_data))
            for i in range(n_sample):
                p = pc_data[i]
                self.get_logger().info(
                    f"  sample[{i}]: x={p['x']:.4f}, y={p['y']:.4f}, z={p['z']:.4f}"
                )

    def camera_info_callback(self, msg: CameraInfo):
        if not self.camera_info_received:
            self.camera_info_received = True
            self.get_logger().info(
                f"[CAMERA_INFO] ✅  "
                f"分辨率={msg.width}x{msg.height}, "
                f"K=[{msg.k[0]:.1f}, 0, {msg.k[2]:.1f}; "
                f"0, {msg.k[4]:.1f}, {msg.k[5]:.1f}; 0, 0, 1]"
            )

    # ── 状态打印 ────────────────────────────────────────────────────────

    def print_status(self):
        elapsed = time.time() - self.start_time
        parts = []
        if self.color_count > 0:
            parts.append(f"Color={self.color_count}")
        if self.depth_count > 0:
            parts.append(f"Depth={self.depth_count}")
        if self.pointcloud_count > 0:
            parts.append(f"PC={self.pointcloud_count}")
        if self.camera_info_received:
            parts.append("CamInfo✓")

        status = "  ".join(parts) if parts else "等待数据..."
        self.get_logger().info(f"[{elapsed:5.1f}s] {status}")

    def shutdown_callback(self):
        elapsed = time.time() - self.start_time

        self.get_logger().info("=" * 60)
        self.get_logger().info(f"测试完成（{elapsed:.0f} 秒）")
        self.get_logger().info(f"  Color:      {self.color_count:4d} 帧  {'✅' if self.color_count > 0 else '❌'}")
        self.get_logger().info(f"  Depth:      {self.depth_count:4d} 帧  {'✅' if self.depth_count > 0 else '❌'}")
        self.get_logger().info(f"  PointCloud: {self.pointcloud_count:4d} 帧  {'✅' if self.pointcloud_count > 0 else '❌'}")
        self.get_logger().info(f"  CameraInfo: {'✅ 正常' if self.camera_info_received else '❌ 未收到'}")
        self.get_logger().info("=" * 60)

        all_ok = (
            self.color_count > 0
            and self.depth_count > 0
            and self.pointcloud_count > 0
            and self.camera_info_received
        )
        if all_ok:
            self.get_logger().info("✅ 全部话题正常，可以开始编写 YOLO 推理节点")
        else:
            self.get_logger().warn("⚠️  部分话题未收到数据")

        self.get_logger().info(f"采样帧: {OUTPUT_DIR}")
        self.destroy_node()
        rclpy.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    rclpy.init(args=sys.argv)
    node = CameraTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print(f"\n输出文件: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()