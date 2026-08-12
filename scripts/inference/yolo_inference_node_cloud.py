#!/usr/bin/env python3
"""
YOLO11-seg ROS2 推理节点（RealSense 彩色点云直取版 ）
============================================================

前提（相机启动参数，必须与下面一致）：
  ros2 launch realsense2_camera rs_launch.py \
    pointcloud.enable:=true \
    align_depth.enable:=true \
    enable_color:=true \
    enable_depth:=true \
    rgb_camera.color_profile:=640x480x30 \
    depth_module.depth_profile:=640x480x30 \
    camera_name:=d435i \
    camera_namespace:="Wrist_Camera"

  align_depth.enable:=true 是关键：
  深度对齐到彩色 → 点云坐标系 = 彩色光轴系 = color/camera_info 内参坐标系，
  3D 点投影回彩色图像素无需任何坐标变换，一一对应。

话题:
  /yolo/detections              std_msgs/String  每帧检测元数据 (JSON)
  /yolo/request_object_cloud    std_msgs/Int32   决策系统请求要抓的 track_id
  /yolo/object_cloud            PointCloud2      对应物体的彩色点云 (xyz+rgb)
  /yolo/debug_cloud             PointCloud2      调试用，全部物体合并点云 (xyz+rgb)
  /yolo/markers                 MarkerArray      调试用 RViz 标记

用法:
  python yolo_inference_node_rs_cloud.py --ros-args -p mode:=production
  python yolo_inference_node_rs_cloud.py --ros-args -p mode:=debug \
      -p publish_debug_cloud:=true
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
    若发现 R/G 对调，把下方三行的通道顺序改一下即可。
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
            # 紧排结构，直接结构化解析（最快）
            arr = np.frombuffer(msg.data, dtype=dt)
            xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=-1).astype(np.float32)
            packed = arr[cname].view(np.uint32)
            rgb = np.empty((n, 3), dtype=np.float32)
            rgb[:, 0] = ((packed >> 16) & 0xFF).astype(np.float32)   # R
            rgb[:, 1] = ((packed >> 8)  & 0xFF).astype(np.float32)   # G
            rgb[:, 2] = ( packed        & 0xFF).astype(np.float32)   # B
            return xyz, rgb
        # 兜底：按 offset 逐字节解析
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
        rgb[:, 1] = ((packed >> 8)  & 0xFF).astype(np.float32)
        rgb[:, 2] = ( packed        & 0xFF).astype(np.float32)
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

    # 无颜色：只解析 xyz
    offs = {f.name: f.offset for f in msg.fields}
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)
    xyz = np.zeros((n, 3), dtype=np.float32)
    for i, nm in enumerate(["x", "y", "z"]):
        seg = np.ascontiguousarray(raw[:, offs[nm]:offs[nm] + 4])
        xyz[:, i] = np.frombuffer(seg.tobytes(), dtype=np.float32)
    return xyz, np.zeros((n, 3), dtype=np.float32)


def ensure_mask_resolution(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """把 mask 缩放到 (H, W) 目标分辨率（最近邻，保持边界）。

    YOLO 返回的 mask 分辨率可能与彩色图不同（例如 160x160），
    而 RealSense 点云反投影出的像素坐标是彩色图像素，必须对齐。
    """
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

    xyz: (M,3) float32 米（彩色光轴系）
    rgb: (M,3) float32 [0-255]
    mask: (H,W) bool —— 必须已缩放为彩色图分辨率（先 ensure_mask_resolution）
    camera_info: 彩色相机内参

    返回: (N,6) float32 [x, y, z, r, g, b]
    """
    if len(xyz) == 0:
        return np.empty((0, 6), dtype=np.float32)

    # 只保留有效点（RealSense 无效深度是 NaN）
    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0.01)
    xyz = xyz[valid]
    rgb = rgb[valid]
    if len(xyz) == 0:
        return np.empty((0, 6), dtype=np.float32)

    fx, fy = camera_info.k[0], camera_info.k[4]
    cx, cy = camera_info.k[2], camera_info.k[5]

    # 3D 点反投影到彩色图像素（aligned 深度保证一一对应）
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
    """numpy (N,6) float32 [x,y,z,r,g,b] → sensor_msgs/PointCloud2。

    输出包含 PCL 兼容的打包 rgb 字段（float32 重解释 uint32 RGB）。
    """
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
            PointField(name="x",   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y",   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z",   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.data = b""
        return msg

    # 打包 RGB: (R << 16) | (G << 8) | B，按【位模式】重解释为 float32（PCL 约定）。
    # 关键：必须 .astype(np.uint32).view(np.float32)——把 RGB 整数的位模式原样当
    # float 存。绝不能 .astype(np.float32)：那是数值转换，位模式面目全非，
    # 下游 view(np.uint32) 解回时颜色全错（白→(127,127,254)、黄→(127,255,192)）。
    # 这正是之前"颜色错乱 + 相近色差异被放大 + 彩色噪声"的总根源。
    r = xyz_rgb[:, 3].astype(np.uint32)
    g = xyz_rgb[:, 4].astype(np.uint32)
    b = xyz_rgb[:, 5].astype(np.uint32)
    rgb_int = (r << 16) | (g << 8) | b          # uint32，如黄色 0x00FFFF00
    rgb_packed = rgb_int.astype(np.uint32).view(np.float32)

    dtype = np.dtype([
        ("x",   np.float32),
        ("y",   np.float32),
        ("z",   np.float32),
        ("rgb", np.float32),
    ])
    buf = np.empty(n, dtype=dtype)
    buf["x"]   = xyz_rgb[:, 0]
    buf["y"]   = xyz_rgb[:, 1]
    buf["z"]   = xyz_rgb[:, 2]
    buf["rgb"] = rgb_packed

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = n
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * n
    msg.fields = [
        PointField(name="x",   offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name="y",   offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name="z",   offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = buf.tobytes()
    return msg


def _get_bbox(obj: dict) -> np.ndarray:
    """从 YOLO 结果 dict 中提取 bbox [x1,y1,x2,y2]，兼容多种 key 名。

    优先级: bbox > box > xyxy > 从 mask 自动计算
    """
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
    """ROS 时间戳 → 纳秒整数（时间对齐比较用）。"""
    return int(s.sec) * 1_000_000_000 + int(s.nanosec)


# ============================================================
# 主节点
# ============================================================

class YOLOInferenceNode(Node):
    """YOLO11-seg 推理 + IoU 追踪 + RealSense 彩色点云裁剪。零自定义接口。"""

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
        self.declare_parameter("debug_dir", "/root/yolo/yolo_debug")
        self.declare_parameter("tracker_max_age", 30)
        self.declare_parameter("tracker_min_hits", 3)
        self.declare_parameter("tracker_iou_threshold", 0.3)

        # ── 话题名（如需改命名空间可覆盖）────────────────
        self.declare_parameter("cloud_topic", "/Wrist_Camera/d435i/depth/color/points")
        self.declare_parameter("color_topic", "/Wrist_Camera/d435i/color/image_raw")
        self.declare_parameter("info_topic", "/Wrist_Camera/d435i/color/camera_info")

        self._cloud_topic = self.get_parameter("cloud_topic").value
        self._color_topic = self.get_parameter("color_topic").value
        self._info_topic = self.get_parameter("info_topic").value

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
        self._debug_dir = self.get_parameter("debug_dir").value
        if self._debug and _HAS_VIZ:
            os.makedirs(self._debug_dir, exist_ok=True)

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
        # 点云用 BEST_EFFORT：reliable 发布方也能兼容，best_effort 发布方也可
        qos_cloud = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE)
        qos_reliable = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # ── 订阅 ─────────────────────────────────────────
        self.color_sub = self.create_subscription(
            Image, self._color_topic, self._color_cb, qos_sensor)
        self.info_sub = self.create_subscription(
            CameraInfo, self._info_topic, self._info_cb, qos_sensor)
        # ★ 核心变化：直接订阅 RealSense 彩色点云，不再订阅深度图
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
        self._latest_color_stamp = None          # 最新彩色图时间戳（时间对齐用）
        self._latest_info: Optional[CameraInfo] = None
        self._latest_cloud_xyz: Optional[np.ndarray] = None
        self._latest_cloud_rgb: Optional[np.ndarray] = None
        self._cloud_buffer: List = []            # [(stamp, xyz, rgb), ...] 时间对齐缓冲
        self._cloud_msg_count = 0
        self._color_frame_id: str = "d435i_color_optical_frame"

        self._frame_count = 0
        self._total_inference_ms = 0.0
        self._color_msg_count = 0
        self._color_encoding: str = ""
        self._shape_warned = False
        self._last_debug_pub = 0.0          # 调试点云限频

        # ── 定时器 ───────────────────────────────────────
        self._timer = self.create_timer(0.05, self._inference_loop)
        self._status_timer = self.create_timer(5.0, self._print_status)

        self.get_logger().info(
            f"YOLOInferenceNode(RealSense云) 启动 | mode={self._mode} | "
            f"model={self.get_parameter('model_path').value} | "
            f"imgsz={self.get_parameter('imgsz').value} | "
            f"点云来源: {self._cloud_topic}"
        )

    # ============================================================
    # 回调
    # ============================================================

    def _color_cb(self, msg: Image):
        try:
            arr, _ = ros_image_to_numpy(msg)
            arr = arr.copy()                          # 防 ROS buffer 复用
            self._color_encoding = msg.encoding
            if "bgr" in msg.encoding.lower():        # 自动 BGR→RGB
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
            self._cloud_buffer.append((msg.header.stamp, xyz, rgb))
            if len(self._cloud_buffer) > 15:
                self._cloud_buffer.pop(0)      # 只保留最近 ~0.5s
            self._cloud_msg_count += 1
            if msg.header.frame_id:
                self._color_frame_id = msg.header.frame_id
        except Exception as e:
            self.get_logger().error(f"点云解码失败: {e}")

    # ============================================================
    # 按需点云请求（决策系统 → 本节点）
    # ============================================================

    def _on_request_cloud(self, msg: Int32):
        """收到决策系统的点云请求 → 从缓存取出 → 发布到 /yolo/object_cloud。"""
        object_id = msg.data
        with self._cloud_lock:
            cloud = self._cloud_cache.get(object_id)

        if cloud is None or cloud.shape[0] < 10:
            self.get_logger().warn(
                f"[Request] ID={object_id} 无有效点云 "
                f"(缓存: {list(self._cloud_cache.keys())})")
            now = self.get_clock().now().to_msg()
            header = Header(stamp=now, frame_id=self._color_frame_id)
            empty = build_pointcloud2(np.empty((0, 6), dtype=np.float32), header)
            self.cloud_pub.publish(empty)
            return

        now = self.get_clock().now().to_msg()
        header = Header(stamp=now, frame_id=self._color_frame_id)
        pc2_msg = build_pointcloud2(cloud, header)
        self.cloud_pub.publish(pc2_msg)
        self.get_logger().info(
            f"[Request] ID={object_id} → {cloud.shape[0]} 点",
            throttle_duration_sec=1.0)

    # ============================================================
    # 推理循环
    # ============================================================

    def _inference_loop(self):
        image = self._latest_color
        color_stamp = self._latest_color_stamp
        camera_info = self._latest_info
        if image is None or camera_info is None:
            return
        self._latest_color = None

        # 点云时间对齐：取与当前彩色图像时间戳最接近的点云帧。
        # 腕部相机在动时，latest 点云可能比图像晚几十 ms，mask 会裁剪到
        # 错位的点云 → 边界抖动。按时间戳找最近一帧点云可消除。
        cloud_xyz = self._latest_cloud_xyz
        cloud_rgb = self._latest_cloud_rgb
        if color_stamp is not None and self._cloud_buffer:
            t_img = _stamp_ns(color_stamp)
            best_d = float("inf")
            for st, xyz, rgb in self._cloud_buffer:
                d = abs(_stamp_ns(st) - t_img)
                if d < best_d:
                    best_d, cloud_xyz, cloud_rgb = d, xyz, rgb

        # 首帧诊断
        if not self._shape_warned:
            self._shape_warned = True
            self.get_logger().info(
                f"[诊断] color={image.shape} dtype={image.dtype} "
                f"encoding={self._color_encoding} | "
                f"RealSense云={None if cloud_xyz is None else cloud_xyz.shape} "
                f"(已收 {self._cloud_msg_count} 帧) | "
                f"camera_info={camera_info.width}x{camera_info.height}")
            if cloud_xyz is None:
                self.get_logger().warn(
                    "[诊断] 尚未收到 RealSense 点云！请确认相机已启动，"
                    "且命令行带 pointcloud.enable:=true")

        color_for_cloud = image.copy()   # YOLO 推理用，不再用于取色
        self._frame_count += 1

        # ── 1. YOLO 推理 ──────────────────────────────
        t0 = time.perf_counter()
        result = self.yolo.predict(image)
        inference_ms = result["inference_time_ms"]
        self._total_inference_ms += inference_ms
        objects = result.get("objects", [])

        # 目标 mask 分辨率 = 彩色图分辨率（以 camera_info 为准，最可靠）
        target_hw = (int(camera_info.height), int(camera_info.width))

        # mask 分辨率诊断（首帧）
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

        # ── 3. 用 mask 裁剪 RealSense 彩色点云 ────────
        tracked_objects: List[Tuple[int, dict]] = []
        new_clouds: Dict[int, np.ndarray] = {}

        for t in tracks:
            if t.det_idx < 0:
                continue
            obj = objects[t.det_idx]
            obj["track_id"] = t.id

            mask = obj.get("mask")
            if mask is None or cloud_xyz is None:
                cloud = np.empty((0, 6), dtype=np.float32)
            else:
                try:
                    mask = ensure_mask_resolution(mask, target_hw)
                    cloud = crop_cloud_by_mask(
                        cloud_xyz, cloud_rgb, mask, camera_info)
                except Exception as e:
                    self.get_logger().error(f"裁剪点云失败: {e}")
                    cloud = np.empty((0, 6), dtype=np.float32)

            obj["cloud"] = cloud
            tracked_objects.append((t.id, obj))
            if cloud.shape[0] >= 10:
                new_clouds[t.id] = cloud

        # 清理过期 + 写入新点云
        active_ids = {t.id for t in tracks}
        with self._cloud_lock:
            for k in list(self._cloud_cache):
                if k not in active_ids:
                    del self._cloud_cache[k]
            self._cloud_cache.update(new_clouds)

        # ── 4. 发布检测元数据 ─────────────────────────
        self._publish_detections(tracked_objects)

        # ── 5. 调试点云（限频 10Hz，避免 RViz 高频刷新大点云而卡顿）──
        if self._publish_debug_cloud or self._mode == "debug":
            clouds = [obj["cloud"] for _, obj in tracked_objects
                      if obj.get("cloud") is not None and obj["cloud"].shape[0] >= 10]
            if clouds and time.time() - self._last_debug_pub > 0.1:
                self._last_debug_pub = time.time()
                now = self.get_clock().now().to_msg()
                header = Header(stamp=now, frame_id=self._color_frame_id)
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
            f"推理={inference_ms:.0f}ms 总={total_ms:.0f}ms",
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