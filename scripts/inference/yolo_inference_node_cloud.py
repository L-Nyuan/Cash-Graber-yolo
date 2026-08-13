#!/usr/bin/env python3
"""
YOLO11-seg ROS2 推理节点（RealSense 彩色点云直取版，带同步诊断）
============================================================

相比原版主要改动：
  1. 点云选择增加 sync_tolerance；找不到容差内的同步点云时，默认跳过本帧裁剪，
     避免旧 segmentation 套到新点云上。
  2. 清空点云缓存，防止决策系统请求到上一帧/上一姿态的旧点云。
  3. 增加 sync/crop 调试日志，可直接用 ROS2 logger debug 级别输出。

运行示例：
  # 默认模式
  python yolo_inference_node_rs_cloud_debug.py --ros-args -p mode:=production

  # 详细 debug 日志
  python yolo_inference_node_rs_cloud_debug.py --ros-args \
    -p mode:=debug \
    -p publish_debug_cloud:=true \
    -p sync_debug:=true \
    --log-level yolo_inference_node:=debug

  # 手腕相机移动快时，可以收紧同步容差：
  python yolo_inference_node_rs_cloud_debug.py --ros-args \
    -p sync_tolerance:=0.03
"""

import json
import sys
import time
import os
import threading
from typing import Dict, List, Tuple, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header, String, Int32

from image_utils import ros_image_to_numpy
from yolo_inference import YOLOSegInference
from object_tracker import ObjectTracker

try:
    from visualization_utils import save_debug_image
    _HAS_VIZ = True
except ImportError:
    _HAS_VIZ = False

try:
    from marker_rviz import _publish_markers
    _HAS_MARKER = True
except ImportError:
    _HAS_MARKER = False


# ============================================================
# 工具函数
# ============================================================

def realsense_cloud_to_xyzrgb(msg: PointCloud2) -> Tuple[np.ndarray, np.ndarray]:
    """解析 RealSense 彩色点云 → (N,3) xyz + (N,3) rgb [0-255]。

    兼容三种字段布局：
      1. rgb  字段（PCL 打包 float32：R<<16 | G<<8 | B）
      2. rgba 字段（同 rgb，忽略 alpha）
      3. 分离的 r/g/b 字段
    无颜色字段时 rgb 返回全 0。

    与 RViz 相同的解码约定：R = (packed >> 16) & 0xFF。
    """
    n = msg.width * msg.height
    if n == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    names = {f.name for f in msg.fields}

    if "rgb" in names or "rgba" in names:
        cname = "rgb" if "rgb" in names else "rgba"
        dt = np.dtype([
            ("x", np.float32), ("y", np.float32), ("z", np.float32),
            (cname, np.float32),
        ])
        if msg.point_step == dt.itemsize:
            arr = np.frombuffer(msg.data, dtype=dt)
            xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=-1).astype(np.float32)
            packed = arr[cname].view(np.uint32)
            rgb = np.empty((n, 3), dtype=np.float32)
            rgb[:, 0] = ((packed >> 16) & 0xFF).astype(np.float32)
            rgb[:, 1] = ((packed >> 8) & 0xFF).astype(np.float32)
            rgb[:, 2] = (packed & 0xFF).astype(np.float32)
            return xyz, rgb

        offs = {f.name: f.offset for f in msg.fields}
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)
        xyz = np.zeros((n, 3), dtype=np.float32)
        for i, nm in enumerate(["x", "y", "z"]):
            seg = np.ascontiguousarray(raw[:, offs[nm]:offs[nm] + 4])
            xyz[:, i] = np.frombuffer(seg.tobytes(), dtype=np.float32)
        seg = np.ascontiguousarray(raw[:, offs[cname]:offs[cname] + 4])
        packed = np.frombuffer(seg.tobytes(), dtype=np.float32).view(np.uint32)
        rgb = np.empty((n, 3), dtype=np.float32)
        rgb[:, 0] = ((packed >> 16) & 0xFF).astype(np.float32)
        rgb[:, 1] = ((packed >> 8) & 0xFF).astype(np.float32)
        rgb[:, 2] = (packed & 0xFF).astype(np.float32)
        return xyz, rgb

    if {"r", "g", "b"}.issubset(names):
        offs = {f.name: f.offset for f in msg.fields}
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)
        xyz = np.zeros((n, 3), dtype=np.float32)
        for i, nm in enumerate(["x", "y", "z"]):
            seg = np.ascontiguousarray(raw[:, offs[nm]:offs[nm] + 4])
            xyz[:, i] = np.frombuffer(seg.tobytes(), dtype=np.float32)
        rgb = np.zeros((n, 3), dtype=np.float32)
        for i, nm in enumerate(["r", "g", "b"]):
            rgb[:, i] = raw[:, offs[nm]].astype(np.float32)
        return xyz, rgb

    offs = {f.name: f.offset for f in msg.fields}
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)
    xyz = np.zeros((n, 3), dtype=np.float32)
    for i, nm in enumerate(["x", "y", "z"]):
        seg = np.ascontiguousarray(raw[:, offs[nm]:offs[nm] + 4])
        xyz[:, i] = np.frombuffer(seg.tobytes(), dtype=np.float32)
    return xyz, np.zeros((n, 3), dtype=np.float32)


def ensure_mask_resolution(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """把 mask 缩放到 (H, W) 目标分辨率（最近邻，保持边界）。"""
    h, w = target_hw
    mask = np.asarray(mask, dtype=bool)
    if mask.shape == (h, w):
        return mask
    import cv2
    m = mask.astype(np.uint8) * 255
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return m > 0


def crop_cloud_by_mask(xyz: np.ndarray, rgb: np.ndarray,
                       mask: np.ndarray,
                       camera_info: CameraInfo) -> np.ndarray:
    """用 2D mask 裁剪 RealSense 彩色点云。

    返回: (N,6) float32 [x, y, z, r, g, b]
    """
    if len(xyz) == 0:
        return np.empty((0, 6), dtype=np.float32)

    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0.01)
    xyz = xyz[valid]
    rgb = rgb[valid]
    if len(xyz) == 0:
        return np.empty((0, 6), dtype=np.float32)

    fx, fy = camera_info.k[0], camera_info.k[4]
    cx, cy = camera_info.k[2], camera_info.k[5]

    z = xyz[:, 2]
    u = np.round(xyz[:, 0] * fx / z + cx).astype(np.int32)
    v = np.round(xyz[:, 1] * fy / z + cy).astype(np.int32)

    h, w = mask.shape
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    keep = mask[v[inside], u[inside]]

    idx = np.where(inside)[0][keep]
    out = np.concatenate([xyz[idx], rgb[idx]], axis=-1)
    return out.astype(np.float32)


def build_pointcloud2(xyz_rgb: np.ndarray, header: Header) -> PointCloud2:
    """numpy (N,6) float32 [x,y,z,r,g,b] → sensor_msgs/PointCloud2。"""
    n = xyz_rgb.shape[0]
    if n == 0:
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = 0
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 0
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.data = b""
        return msg

    r = xyz_rgb[:, 3].astype(np.uint32)
    g = xyz_rgb[:, 4].astype(np.uint32)
    b = xyz_rgb[:, 5].astype(np.uint32)
    rgb_int = (r << 16) | (g << 8) | b
    rgb_packed = rgb_int.astype(np.uint32).view(np.float32)

    dtype = np.dtype([
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("rgb", np.float32),
    ])
    buf = np.empty(n, dtype=dtype)
    buf["x"] = xyz_rgb[:, 0]
    buf["y"] = xyz_rgb[:, 1]
    buf["z"] = xyz_rgb[:, 2]
    buf["rgb"] = rgb_packed

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = n
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * n
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = buf.tobytes()
    return msg


def _get_bbox(obj: dict) -> np.ndarray:
    """从 YOLO 结果 dict 中提取 bbox [x1,y1,x2,y2]。"""
    if "bbox" in obj:
        return np.array(obj["bbox"], dtype=np.float32)
    if "box" in obj:
        return np.array(obj["box"], dtype=np.float32)
    if "xyxy" in obj:
        return np.array(obj["xyxy"], dtype=np.float32)
    mask = obj.get("mask")
    if mask is not None and mask.any():
        ys, xs = np.where(mask)
        return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
    return np.zeros(4, dtype=np.float32)


def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """两矩形框 IoU。"""
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _stamp_ns(s) -> int:
    """ROS 时间戳 → 纳秒整数。"""
    return int(s.sec) * 1_000_000_000 + int(s.nanosec)


def _stamp_text(s) -> str:
    """把 ROS 时间转成 sec.nanosec 字符串，方便读日志。"""
    if s is None:
        return "None"
    return f"{int(s.sec)}.{int(s.nanosec):09d}"


# ============================================================
# 主节点
# ============================================================

class YOLOInferenceNode(Node):
    """YOLO11-seg 推理 + IoU 追踪 + RealSense 彩色点云裁剪。"""

    def __init__(self):
        super().__init__("yolo_inference_node")

        # ── 参数 ─────────────────────────────────────────
        self.declare_parameter("model_path", "/root/yolo/result/final/best.pt")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.8)
        self.declare_parameter("iou", 0.7)
        self.declare_parameter("mode", "production")
        self.declare_parameter("publish_debug_cloud", False)
        self.declare_parameter("publish_markers", False)
        self.declare_parameter("debug", False)
        self.declare_parameter("debug_cloud_hz", 15.0)
        self.declare_parameter("debug_dir", "/root/yolo/yolo_debug")
        self.declare_parameter("tracker_max_age", 30)
        self.declare_parameter("tracker_min_hits", 3)
        self.declare_parameter("tracker_iou_threshold", 0.3)
        self.declare_parameter("mask_erode", 1)
        self.declare_parameter("cloud_sor", True)

        # 时间同步参数
        self.declare_parameter("sync_tolerance", 0.05)
        self.declare_parameter("require_synced_cloud", True)
        self.declare_parameter("sync_debug", True)
        self.declare_parameter("cloud_buffer_size", 30)
        self.declare_parameter("pending_max_wait", 0.12)

        # ── 话题名（如需改命名空间可覆盖）────────────────
        self.declare_parameter("cloud_topic", "/Wrist_Camera/d435i/depth/color/points")
        self.declare_parameter("color_topic", "/Wrist_Camera/d435i/color/image_raw")
        self.declare_parameter("info_topic", "/Wrist_Camera/d435i/color/camera_info")

        self._cloud_topic = self.get_parameter("cloud_topic").value
        self._color_topic = self.get_parameter("color_topic").value
        self._info_topic = self.get_parameter("info_topic").value

        self._sync_tolerance_ns = int(
            self.get_parameter("sync_tolerance").value * 1_000_000_000)
        self._require_synced_cloud = bool(
            self.get_parameter("require_synced_cloud").value)
        self._sync_debug = bool(self.get_parameter("sync_debug").value)
        self._cloud_buffer_size = int(
            self.get_parameter("cloud_buffer_size").value)
        self._pending_max_wait = float(
            self.get_parameter("pending_max_wait").value)

        # ── YOLO 模型 ────────────────────────────────────
        self.yolo = YOLOSegInference(
            model_path=self.get_parameter("model_path").value,
            imgsz=self.get_parameter("imgsz").value,
            conf=self.get_parameter("conf").value,
            iou=self.get_parameter("iou").value,
        )

        self._mode = self.get_parameter("mode").value
        self._publish_debug_cloud = self.get_parameter("publish_debug_cloud").value
        self._publish_markers = self.get_parameter("publish_markers").value
        self._debug = self.get_parameter("debug").value
        self._debug_cloud_interval = (
            1.0 / max(float(self.get_parameter("debug_cloud_hz").value), 1.0))
        self._debug_dir = self.get_parameter("debug_dir").value
        if self._debug and _HAS_VIZ:
            os.makedirs(self._debug_dir, exist_ok=True)

        # ── 点云后处理 ─────────────────────────────────────
        self._mask_erode = int(self.get_parameter("mask_erode").value)
        self._cloud_sor = bool(self.get_parameter("cloud_sor").value)
        self._o3d = None
        if self._cloud_sor:
            try:
                import open3d as o3d
                self._o3d = o3d
            except ImportError:
                self.get_logger().warn(
                    "cloud_sor=true 但环境无 open3d，跳过离群点移除（不影响主流程）")

        # ── 追踪器 ───────────────────────────────────────
        self._tracker = ObjectTracker(
            max_age=self.get_parameter("tracker_max_age").value,
            min_hits=self.get_parameter("tracker_min_hits").value,
            iou_threshold=self.get_parameter("tracker_iou_threshold").value,
        )

        # ── 点云缓存：track_id → (N,6) xyz+rgb ─────────
        self._cloud_cache: Dict[int, np.ndarray] = {}
        self._cloud_lock = threading.Lock()

        # ── QoS ──────────────────────────────────────────
        qos_sensor = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE)
        qos_cloud = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE)
        qos_reliable = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # ── 订阅 ─────────────────────────────────────────
        self.color_sub = self.create_subscription(
            Image, self._color_topic, self._color_cb, qos_sensor)
        self.info_sub = self.create_subscription(
            CameraInfo, self._info_topic, self._info_cb, qos_sensor)
        self.cloud_sub = self.create_subscription(
            PointCloud2, self._cloud_topic, self._cloud_cb, qos_cloud)

        # ── 发布 ─────────────────────────────────────────
        self.det_pub = self.create_publisher(
            String, "/yolo/detections", qos_reliable)
        self.cloud_pub = self.create_publisher(
            PointCloud2, "/yolo/object_cloud", qos_reliable)
        self.request_sub = self.create_subscription(
            Int32, "/yolo/request_object_cloud",
            self._on_request_cloud, qos_reliable)

        if self._publish_debug_cloud or self._mode == "debug":
            self.debug_cloud_pub = self.create_publisher(
                PointCloud2, "/yolo/debug_cloud", qos_reliable)
        if self._publish_markers or self._mode == "debug":
            if _HAS_MARKER:
                from visualization_msgs.msg import MarkerArray
                self.marker_pub = self.create_publisher(
                    MarkerArray, "/yolo/markers", qos_reliable)

        # ── 状态 ─────────────────────────────────────────
        self._latest_color: Optional[np.ndarray] = None
        self._latest_color_stamp = None
        self._latest_info: Optional[CameraInfo] = None
        self._latest_cloud_xyz: Optional[np.ndarray] = None
        self._latest_cloud_rgb: Optional[np.ndarray] = None
        self._latest_cloud_stamp = None
        self._latest_cloud_frame: str = ""
        self._cloud_buffer: List = []
        self._cloud_msg_count = 0
        self._color_frame_id: str = "d435i_color_optical_frame"
        self._last_crop_stamp = None
        self._last_crop_frame: str = "d435i_color_optical_frame"

        self._frame_count = 0
        self._total_inference_ms = 0.0
        self._color_msg_count = 0
        self._color_encoding: str = ""
        self._shape_warned = False
        self._last_debug_pub = 0.0

        self._cloud_sync_ok_count = 0
        self._cloud_sync_skip_count = 0
        self._sync_hold_count = 0
        self._last_sync_log_time = 0.0

        self._pending_color: Optional[np.ndarray] = None
        self._pending_color_stamp = None
        self._pending_color_deadline = 0.0

        # ── 定时器 ───────────────────────────────────────
        self._timer = self.create_timer(0.05, self._inference_loop)
        self._status_timer = self.create_timer(5.0, self._print_status)

        self.get_logger().info(
            f"YOLOInferenceNode(RealSense云, synced) 启动 | mode={self._mode} | "
            f"model={self.get_parameter('model_path').value} | "
            f"imgsz={self.get_parameter('imgsz').value} | "
            f"点云来源: {self._cloud_topic} | "
            f"sync_tolerance={self.get_parameter('sync_tolerance').value:.3f}s | "
            f"require_synced_cloud={self._require_synced_cloud}"
        )

    # ============================================================
    # 回调
    # ============================================================

    def _color_cb(self, msg: Image):
        try:
            arr, _ = ros_image_to_numpy(msg)
            arr = arr.copy()
            self._color_encoding = msg.encoding
            if "bgr" in msg.encoding.lower():
                arr = arr[:, :, ::-1]
            self._latest_color_stamp = msg.header.stamp
            self._latest_color = arr
            self._color_msg_count += 1
        except Exception as e:
            self.get_logger().error(f"color 解码失败: {e}")

    def _info_cb(self, msg: CameraInfo):
        self._latest_info = msg
        if msg.header.frame_id:
            self._color_frame_id = msg.header.frame_id

    def _cloud_cb(self, msg: PointCloud2):
        try:
            xyz, rgb = realsense_cloud_to_xyzrgb(msg)
            self._latest_cloud_xyz = xyz
            self._latest_cloud_rgb = rgb
            self._latest_cloud_stamp = msg.header.stamp
            if msg.header.frame_id:
                self._latest_cloud_frame = msg.header.frame_id

            self._cloud_buffer.append(
                (msg.header.stamp, msg.header.frame_id, xyz, rgb))
            if len(self._cloud_buffer) > self._cloud_buffer_size:
                self._cloud_buffer.pop(0)
            self._cloud_msg_count += 1

            if self._sync_debug and self._cloud_msg_count <= 3:
                self.get_logger().info(
                    f"[cloud_cb] stamp={_stamp_text(msg.header.stamp)} "
                    f"frame={msg.header.frame_id or '(empty)'} "
                    f"points={len(xyz)}"
                )
        except Exception as e:
            self.get_logger().error(f"点云解码失败: {e}")

    # ============================================================
    # 时间同步匹配
    # ============================================================

    def _match_cloud_for_color(self, color_stamp):
        """为彩色图匹配点云。

        Returns:
            (xyz, rgb, cloud_stamp, cloud_frame, delta_ns, matched)
        """
        if color_stamp is None or not self._cloud_buffer:
            if color_stamp is not None and not self._cloud_buffer:
                self.get_logger().warn(
                    "[sync] color 已到但 cloud_buffer 为空，无法匹配",
                    throttle_duration_sec=1.0)
            return (None, None, None, self._color_frame_id, None, False)

        t_img = _stamp_ns(color_stamp)
        best_idx = -1
        best_d = float("inf")
        for i, (st, _fr, _xyz, _rgb) in enumerate(self._cloud_buffer):
            d = abs(_stamp_ns(st) - t_img)
            if d < best_d:
                best_d = d
                best_idx = i

        if best_idx < 0:
            return (None, None, None, self._color_frame_id, None, False)

        cloud_stamp, cloud_frame, xyz, rgb = self._cloud_buffer[best_idx]
        matched = best_d <= self._sync_tolerance_ns
        return (xyz, rgb, cloud_stamp,
                cloud_frame or self._color_frame_id,
                best_d, matched)

    def _log_sync(self, color_stamp, cloud_stamp, delta_ns, matched, cloud_n):
        if not self._sync_debug:
            return
        now = time.time()
        if now - self._last_sync_log_time < 0.2:
            return
        self._last_sync_log_time = now

        if delta_ns is None:
            self.get_logger().warn(
                f"[sync] color={_stamp_text(color_stamp)} cloud={_stamp_text(cloud_stamp)} "
                f"delta=None buffer={cloud_n} match={matched}")
            return

        delta_ms = delta_ns / 1_000_000.0
        self.get_logger().info(
            f"[sync] color={_stamp_text(color_stamp)} "
            f"cloud={_stamp_text(cloud_stamp)} "
            f"delta={delta_ms:.2f}ms buffer={cloud_n} match={matched}"
        )

    # ============================================================
    # 按需点云请求（决策系统 → 本节点）
    # ============================================================

    def _on_request_cloud(self, msg: Int32):
        object_id = msg.data
        with self._cloud_lock:
            cloud = self._cloud_cache.get(object_id)

        if cloud is None or cloud.shape[0] < 10:
            self.get_logger().warn(
                f"[Request] ID={object_id} 无有效点云 "
                f"(缓存: {list(self._cloud_cache.keys())})")
            stamp = self._last_crop_stamp or self.get_clock().now().to_msg()
            header = Header(stamp=stamp, frame_id=self._last_crop_frame)
            empty = build_pointcloud2(np.empty((0, 6), dtype=np.float32), header)
            self.cloud_pub.publish(empty)
            return

        stamp = self._last_crop_stamp or self.get_clock().now().to_msg()
        header = Header(stamp=stamp, frame_id=self._last_crop_frame)
        pc2_msg = build_pointcloud2(cloud, header)
        self.cloud_pub.publish(pc2_msg)
        self.get_logger().info(
            f"[Request] ID={object_id} → {cloud.shape[0]} 点",
            throttle_duration_sec=1.0)

    # ============================================================
    # 推理循环
    # ============================================================

    def _inference_loop(self):
        now_mono = time.perf_counter()
        camera_info = self._latest_info
        if camera_info is None:
            return

        # 如果上一帧暂时没等到同步点云，先让 pending 帧继续匹配，
        # 而不是把这一帧直接丢掉，从而减少 RViz 中偶发的一拍冻结。
        if self._pending_color is not None and now_mono > self._pending_color_deadline:
            self.get_logger().debug(
                f"[sync] pending color {_stamp_text(self._pending_color_stamp)} "
                "超时，丢弃该帧")
            self._pending_color = None
            self._pending_color_stamp = None

        image = self._latest_color
        color_stamp = self._latest_color_stamp

        use_pending = False
        if self._pending_color is not None:
            if image is None:
                use_pending = True
            else:
                use_pending = (
                    _stamp_ns(self._pending_color_stamp) <= _stamp_ns(color_stamp))

        if use_pending:
            image = self._pending_color
            color_stamp = self._pending_color_stamp
        elif image is not None:
            # 有更新的彩色图，且没有更早的 pending 帧，直接处理新帧。
            self._pending_color = None
            self._pending_color_stamp = None
            self._latest_color = None
        else:
            return

        # 点云时间同步：没有容差内匹配时，宁可本帧不裁点云，
        # 也不要把旧 mask 套到新点云上。
        cloud_xyz, cloud_rgb, cloud_stamp, cloud_frame, delta_ns, matched = \
            self._match_cloud_for_color(color_stamp)

        self._log_sync(color_stamp, cloud_stamp, delta_ns, matched,
                       len(self._cloud_buffer))

        if matched:
            self._cloud_sync_ok_count += 1
        else:
            self._cloud_sync_skip_count += 1

        delta_text = "None" if delta_ns is None else f"{delta_ns / 1e6:.2f}ms"

        # 没有同步点云时默认等待一小段时间，等待对应点云回调到达。
        # 超过 pending_max_wait 后仍不匹配，再丢弃该帧，不无限等待。
        if not matched and self._require_synced_cloud:
            if self._pending_max_wait > 0.0:
                if not use_pending:
                    self._pending_color = image
                    self._pending_color_stamp = color_stamp
                    self._pending_color_deadline = now_mono + self._pending_max_wait
                self._sync_hold_count += 1
                if not self._cloud_buffer:
                    self.get_logger().debug(
                        f"[sync] WAIT_CLOUD color={_stamp_text(color_stamp)} "
                        f"cloud_buffer=0 cloud_msgs={self._cloud_msg_count} "
                        f"wait={self._pending_max_wait:.3f}s")
                else:
                    self.get_logger().debug(
                        f"[sync] HOLD color={_stamp_text(color_stamp)} "
                        f"cloud={_stamp_text(cloud_stamp)} "
                        f"delta={delta_text} buffer={len(self._cloud_buffer)} "
                        f"cloud_msgs={self._cloud_msg_count}")
                return

            self.get_logger().debug(
                f"[sync] DROP color={_stamp_text(color_stamp)} "
                f"cloud={_stamp_text(cloud_stamp)} "
                f"delta={delta_text} buffer={len(self._cloud_buffer)} "
                f"cloud_msgs={self._cloud_msg_count}")

        # 只有真正采用匹配点云时，才更新“最后一次裁剪使用的点云时间戳”。
        if matched and cloud_stamp is not None:
            self._last_crop_stamp = cloud_stamp
            self._last_crop_frame = cloud_frame
        elif not matched:
            self._last_crop_stamp = None
            self._last_crop_frame = self._color_frame_id

        # 成功进入推理时，清除 pending 帧，避免下一轮重复处理。
        self._pending_color = None
        self._pending_color_stamp = None

        if not matched and self._require_synced_cloud:
            cloud_xyz = None
            cloud_rgb = None

        # 首帧诊断
        if not self._shape_warned:
            self._shape_warned = True
            self.get_logger().info(
                f"[诊断] color={image.shape} dtype={image.dtype} "
                f"encoding={self._color_encoding} | "
                f"RealSense云={None if cloud_xyz is None else cloud_xyz.shape} "
                f"(已收 {self._cloud_msg_count} 帧) | "
                f"camera_info={camera_info.width}x{camera_info.height} | "
                f"color_frame={self._color_frame_id} | "
                f"cloud_frame={self._latest_cloud_frame or '(未收到点云)'} | "
                f"sync_tolerance={self._sync_tolerance_ns / 1e6:.2f}ms")
            if cloud_xyz is None:
                self.get_logger().warn(
                    "[诊断] 尚未收到/未匹配 RealSense 点云！请确认相机已启动，"
                    "且命令行带 pointcloud.enable:=true；若已启动，查看 sync 日志。")

        color_for_cloud = image.copy()
        self._frame_count += 1

        # ── 1. YOLO 推理 ──────────────────────────────
        t0 = time.perf_counter()
        result = self.yolo.predict(image)
        inference_ms = result["inference_time_ms"]
        self._total_inference_ms += inference_ms
        objects = result.get("objects", [])

        target_hw = (int(camera_info.height), int(camera_info.width))

        if objects and not hasattr(self, '_mask_shape_warned'):
            self._mask_shape_warned = True
            m0 = objects[0].get("mask")
            if m0 is not None:
                self.get_logger().info(
                    f"[诊断] YOLO mask={m0.shape} → 缩放到 {target_hw} 再投影裁剪")

        # ── 2. IoU 追踪 ──────────────────────────────
        dets = [{"bbox": _get_bbox(o), "class_name": o["class_name"],
                 "confidence": o["confidence"]} for o in objects]
        tracks = self._tracker.update(dets)

        # ── 3. 裁剪点云 ──────────────────────────────
        tracked_objects: List[Tuple[int, dict]] = []
        new_clouds: Dict[int, np.ndarray] = {}

        if cloud_xyz is None and matched:
            self.get_logger().warn(
                "[sync] matched=true 但 cloud_xyz=None，裁剪将跳过",
                throttle_duration_sec=1.0)

        if cloud_xyz is None and not matched:
            # 清掉旧缓存，避免决策系统请求到上一姿态的点云。
            with self._cloud_lock:
                self._cloud_cache.clear()

        for t in tracks:
            if t.det_idx < 0:
                continue
            obj = objects[t.det_idx]
            obj["track_id"] = t.id

            mask = obj.get("mask")
            cloud = np.empty((0, 6), dtype=np.float32)

            if mask is not None and cloud_xyz is not None:
                try:
                    mask_shape_before = mask.shape
                    mask_count_before = int(np.count_nonzero(mask))

                    mask = ensure_mask_resolution(mask, target_hw)
                    if self._mask_erode > 0:
                        import cv2
                        mask = cv2.erode(
                            mask.astype(np.uint8),
                            np.ones((3, 3), np.uint8),
                            iterations=self._mask_erode) > 0

                    mask_count_after = int(np.count_nonzero(mask))
                    cloud = crop_cloud_by_mask(
                        cloud_xyz, cloud_rgb, mask, camera_info)
                    crop_count = cloud.shape[0]

                    if self._o3d is not None and cloud.shape[0] > 50:
                        pcd = self._o3d.geometry.PointCloud()
                        pcd.points = self._o3d.utility.Vector3dVector(cloud[:, :3])
                        _, idx = pcd.remove_statistical_outlier(
                            nb_neighbors=20, std_ratio=1.0)
                        cloud = cloud[np.asarray(idx)]

                    if self._sync_debug:
                        z_mean = float(np.mean(cloud[:, 2])) if cloud.shape[0] else 0.0
                        rgb_mean = tuple(
                            float(np.mean(cloud[:, 3 + i])) if cloud.shape[0] else 0.0
                            for i in range(3))
                        self.get_logger().debug(
                            f"[crop] tid={t.id} {obj['class_name']} "
                            f"mask={mask_shape_before}->{mask.shape} "
                            f"px={mask_count_before}->{mask_count_after} "
                            f"src_cloud={len(cloud_xyz)} crop={crop_count} "
                            f"after_sor={cloud.shape[0]} "
                            f"z_mean={z_mean:.3f}m rgb_mean=({rgb_mean[0]:.0f},"
                            f"{rgb_mean[1]:.0f},{rgb_mean[2]:.0f})")
                except Exception as e:
                    self.get_logger().error(f"裁剪点云失败: {e}")
                    cloud = np.empty((0, 6), dtype=np.float32)

            obj["cloud"] = cloud
            tracked_objects.append((t.id, obj))
            if cloud.shape[0] >= 10:
                new_clouds[t.id] = cloud

        # 清理过期 + 写入新点云。
        # 只有本帧实际产生过同步裁剪时才更新缓存，避免旧点云跨帧残留。
        if cloud_xyz is not None:
            active_ids = {t.id for t in tracks}
            with self._cloud_lock:
                for k in list(self._cloud_cache):
                    if k not in active_ids:
                        del self._cloud_cache[k]
                self._cloud_cache.update(new_clouds)

        # ── 4. 发布检测元数据 ─────────────────────────
        self._publish_detections(tracked_objects)

        # ── 5. 调试点云 ───────────────────────────────
        if self._publish_debug_cloud or self._mode == "debug":
            clouds = [obj["cloud"] for _, obj in tracked_objects
                      if obj.get("cloud") is not None and obj["cloud"].shape[0] >= 10]
            if clouds and time.time() - self._last_debug_pub > self._debug_cloud_interval:
                self._last_debug_pub = time.time()
                stamp = self._last_crop_stamp
                if stamp is None:
                    stamp = self.get_clock().now().to_msg()
                header = Header(stamp=stamp, frame_id=self._last_crop_frame)
                self.debug_cloud_pub.publish(
                    build_pointcloud2(np.vstack(clouds), header))

        if (self._publish_markers or self._mode == "debug") and _HAS_MARKER:
            if hasattr(self, "marker_pub"):
                now = self.get_clock().now().to_msg()
                header = Header(stamp=now, frame_id=self._color_frame_id)
                _publish_markers([obj for _, obj in tracked_objects],
                                 header, publisher=self.marker_pub)

        # ── 日志 ─────────────────────────────────────
        total_ms = (time.perf_counter() - t0) * 1000
        names = ", ".join(
            f"#{tid} {obj['class_name']}({obj['cloud'].shape[0]}点)"
            for tid, obj in tracked_objects[:5])
        self.get_logger().info(
            f"[#{self._frame_count}] {len(tracked_objects)} 目标: {names} | "
            f"推理={inference_ms:.0f}ms 总={total_ms:.0f}ms | "
            f"sync={'OK' if matched else 'SKIP'}",
            throttle_duration_sec=2.0)

    # ============================================================
    # 检测发布 (JSON)
    # ============================================================

    def _publish_detections(self, tracked_objects: List[Tuple[int, dict]]):
        now = self.get_clock().now().to_msg()
        payload = {
            "header": {
                "stamp": {"sec": now.sec, "nanosec": now.nanosec},
                "frame_id": self._color_frame_id,
            },
            "detections": [],
        }
        for tid, obj in tracked_objects:
            bbox = obj["bbox"]
            payload["detections"].append({
                "id": tid,
                "class_name": obj["class_name"],
                "confidence": round(float(obj["confidence"]), 4),
                "center_x": round((bbox[0] + bbox[2]) / 2.0, 1),
                "center_y": round((bbox[1] + bbox[3]) / 2.0, 1),
                "bbox": [round(float(v), 1) for v in bbox],
            })

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.det_pub.publish(msg)

    # ============================================================
    # 状态打印
    # ============================================================

    def _print_status(self):
        if self._frame_count == 0:
            self.get_logger().info(
                f"等待首帧... color={self._color_msg_count} "
                f"cloud={self._cloud_msg_count} "
                f"info={self._latest_info is not None}",
                throttle_duration_sec=5.0)
            return
        avg = self._total_inference_ms / max(self._frame_count, 1)
        self.get_logger().info(
            f"[状态] {self._frame_count} 帧 | color={self._color_msg_count} "
            f"cloud={self._cloud_msg_count} | "
            f"sync_ok={self._cloud_sync_ok_count} "
            f"sync_skip={self._cloud_sync_skip_count} "
            f"sync_hold={self._sync_hold_count} | "
            f"avg {avg:.0f}ms ({1000/avg:.0f} FPS) | "
            f"活跃目标 {len(self._cloud_cache)} | mode={self._mode}",
            throttle_duration_sec=5.0)


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
