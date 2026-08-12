#!/usr/bin/env python3
"""
按需点云请求调试脚本（带颜色保存）
==================================

向 /yolo/request_object_cloud 发送 Int32(track_id)，
接收 /yolo/object_cloud 返回的彩色点云，可选保存。

用法:
  # 一次性请求（不保存）
  source /opt/ros/humble/setup.bash && python request_pointcloud_debug.py --ros-args -p track_id:=3

  # 请求并保存为 .ply（带 RGB，GraspNet 可用）
  ... -p track_id:=1 -p save_cloud:=true -p save_dir:=/root/yolo/debug_clouds

  # 循环请求，每秒一次
  ... -p track_id:=2 -p loop:=true -p save_cloud:=true

  # 保存格式：ply（推荐，Open3D 原生支持）或 npy
  ... -p track_id:=1 -p save_cloud:=true -p save_format:=ply

参数:
  track_id       要请求的目标 track_id（整数，必填）
  save_cloud     是否保存点云到文件（默认 false）
  save_dir       点云保存目录（默认 /root/yolo/debug_clouds）
  save_format    保存格式: ply 或 npy（默认 ply）
  loop           循环请求模式，每秒请求一次（默认 false）
  wait_timeout   等待点云响应的超时秒数（默认 3.0）
"""

import sys
import os
import time
import threading
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32
from sensor_msgs.msg import PointCloud2


# ============================================================
# PointCloud2 解析
# ============================================================

def pointcloud2_to_xyz_rgb(msg: PointCloud2) -> np.ndarray:
    """解析 PointCloud2 → (N, 6) [x, y, z, r, g, b]。

    自动识别新格式（rgb 字段，PCL 打包）或旧格式（intensity 字段）。
    """
    field_names = {f.name for f in msg.fields}
    n = msg.width * msg.height
    if n == 0:
        return np.empty((0, 6), dtype=np.float32)

    if "rgb" in field_names:
        # ── 新格式：x, y, z, rgb ──────────────────────
        dtype = np.dtype([
            ("x",   np.float32),
            ("y",   np.float32),
            ("z",   np.float32),
            ("rgb", np.float32),
        ])
        arr = np.frombuffer(msg.data, dtype=dtype)
        rgb_uint32 = arr["rgb"].view(np.uint32)

        out = np.empty((n, 6), dtype=np.float32)
        out[:, 0] = arr["x"]
        out[:, 1] = arr["y"]
        out[:, 2] = arr["z"]
        out[:, 3] = ((rgb_uint32 >> 16) & 0xFF).astype(np.float32)   # R
        out[:, 4] = ((rgb_uint32 >> 8)  & 0xFF).astype(np.float32)   # G
        out[:, 5] = ( rgb_uint32        & 0xFF).astype(np.float32)   # B
        return out

    elif "intensity" in field_names:
        # ── 旧格式：x, y, z, intensity ─────────────────
        dtype = np.dtype([
            ("x",         np.float32),
            ("y",         np.float32),
            ("z",         np.float32),
            ("intensity", np.float32),
        ])
        arr = np.frombuffer(msg.data, dtype=dtype)
        out = np.zeros((n, 6), dtype=np.float32)
        out[:, 0] = arr["x"]
        out[:, 1] = arr["y"]
        out[:, 2] = arr["z"]
        return out

    else:
        # 未知格式，尝试只用 xyz
        offsets = {f.name: f.offset for f in msg.fields}
        out = np.zeros((n, 6), dtype=np.float32)
        if {"x", "y", "z"}.issubset(offsets):
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
            for i, name in enumerate(["x", "y", "z"]):
                off = offsets[name]
                out[:, i] = np.frombuffer(
                    np.ascontiguousarray(raw[:, off:off+4]).tobytes(),
                    dtype=np.float32)
        return out


# ============================================================
# 保存
# ============================================================

def save_as_ply(xyz_rgb: np.ndarray, filepath: str):
    """保存为 ASCII PLY 文件（带 RGB 颜色，Open3D / CloudCompare 可读）。

    xyz_rgb: (N, 6) float32 — [x, y, z, r, g, b]，rgb 范围 0–255
    """
    n = xyz_rgb.shape[0]
    with open(filepath, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            f.write(
                f"{xyz_rgb[i, 0]:.6f} {xyz_rgb[i, 1]:.6f} {xyz_rgb[i, 2]:.6f} "
                f"{int(xyz_rgb[i, 3])} {int(xyz_rgb[i, 4])} {int(xyz_rgb[i, 5])}\n"
            )
    print(f"  已保存: {filepath}  shape=({n}, 6) 含 RGB")


def save_as_npy(xyz_rgb: np.ndarray, filepath: str):
    """保存为 .npy 文件（包含 xyz+rgb 全部数据）。"""
    np.save(filepath, xyz_rgb)
    print(f"  已保存: {filepath}  shape={xyz_rgb.shape}")


# ============================================================
# 节点
# ============================================================

class PointCloudRequester(Node):
    """请求点云并接收响应。"""

    def __init__(self):
        super().__init__("pointcloud_requester")

        # ── 参数 ──────────────────────────────────────────
        self.declare_parameter("track_id", -1)
        self.declare_parameter("save_cloud", False)
        self.declare_parameter("save_dir", "/root/yolo/debug_clouds")
        self.declare_parameter("save_format", "ply")
        self.declare_parameter("loop", False)
        self.declare_parameter("wait_timeout", 3.0)

        self._track_id = self.get_parameter("track_id").value
        self._save_cloud = self.get_parameter("save_cloud").value
        self._save_dir = self.get_parameter("save_dir").value
        self._save_format = self.get_parameter("save_format").value
        self._loop = self.get_parameter("loop").value
        self._wait_timeout = self.get_parameter("wait_timeout").value

        if self._track_id < 0:
            self.get_logger().error("track_id 未设置！使用 -p track_id:=N 指定")
            sys.exit(1)

        # ── QoS ────────────────────────────────────────────
        qos_reliable = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # ── 发布请求 ───────────────────────────────────────
        self.request_pub = self.create_publisher(
            Int32, "/yolo/request_object_cloud", qos_reliable)

        # ── 订阅响应 ───────────────────────────────────────
        self.cloud_sub = self.create_subscription(
            PointCloud2, "/yolo/object_cloud",
            self._cloud_cb, qos_reliable)

        # ── 同步机制 ───────────────────────────────────────
        self._response: Optional[np.ndarray] = None
        self._response_event = threading.Event()

        # ── 保存准备 ───────────────────────────────────────
        if self._save_cloud:
            os.makedirs(self._save_dir, exist_ok=True)
            self._save_counter = 0

        self.get_logger().info(
            f"点云请求节点启动 | "
            f"track_id={self._track_id} | "
            f"save={self._save_cloud} | "
            f"format={self._save_format} | "
            f"loop={self._loop} | "
            f"timeout={self._wait_timeout}s")

    # ── 响应回调 ───────────────────────────────────────────

    def _cloud_cb(self, msg: PointCloud2):
        points = pointcloud2_to_xyz_rgb(msg)
        if points.shape[0] > 0:
            self._response = points
        else:
            self.get_logger().warn("收到空点云（track_id 无效或无深度数据）")
        self._response_event.set()

    # ── 发送请求并等待 ─────────────────────────────────────

    def request_once(self):
        """发送一次请求，用 spin_once 等待响应，可选保存。"""
        self._response = None
        self._response_event.clear()

        msg = Int32(data=self._track_id)
        self.request_pub.publish(msg)
        self.get_logger().info(
            f"→ 请求物体 #{self._track_id} 点云 ...")

        deadline = time.time() + self._wait_timeout
        got_response = False
        while time.time() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._response_event.is_set():
                got_response = True
                break

        if got_response:
            points = self._response
            if points is None or points.shape[0] == 0:
                self.get_logger().error(
                    f"✗ 物体 #{self._track_id} 无点云！"
                    f"检查 track_id 是否活跃，或深度图是否到达")
                return None, False

            n_points = points.shape[0]
            mean_z = float(points[:, 2].mean())
            has_rgb = points.shape[1] >= 6 and np.any(points[:, 3:6] > 0)
            self.get_logger().info(
                f"✓ 收到 #{self._track_id} 点云: "
                f"{n_points} 点, 平均深度={mean_z:.3f}m, "
                f"颜色={'有' if has_rgb else '无'}")

            if self._save_cloud:
                self._save_counter += 1
                filename = (
                    f"obj_{self._track_id}_"
                    f"{self._save_counter:04d}.{self._save_format}"
                )
                filepath = os.path.join(self._save_dir, filename)

                if self._save_format == "ply":
                    save_as_ply(points, filepath)
                else:
                    save_as_npy(points, filepath)

            return points, True
        else:
            self.get_logger().error(
                f"✗ 超时：{self._wait_timeout}s 内未收到物体 "
                f"#{self._track_id} 的点云响应")
            return None, False


def main():
    rclpy.init(args=sys.argv)
    node = PointCloudRequester()

    if node._loop:
        node.get_logger().info("循环请求模式，Ctrl+C 退出")
        try:
            while rclpy.ok():
                node.request_once()
                deadline = time.time() + 1.0
                while time.time() < deadline and rclpy.ok():
                    rclpy.spin_once(node, timeout_sec=0.05)
        except KeyboardInterrupt:
            print("\n用户中断")
    else:
        node.request_once()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()