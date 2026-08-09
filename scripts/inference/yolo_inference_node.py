#!/usr/bin/env python3
"""
YOLO11-seg ROS2 推理节点（正式版 · 零编译）
==========================================

纯标准消息类型，无需编译自定义接口，conda 环境直接用。

架构:
  RealSense ─┬─ color/image_raw ─────→ YOLO 推理
             ├─ aligned_depth ────────→ mask → 点云投影
             └─ color/camera_info ────→ 内参

  YOLO Node ─┬─ /yolo/detections (String/JSON) ──→ 决策系统
             ├─ /yolo/request_object_cloud (Int32) ← 决策系统请求
             └─ /yolo/object_cloud (PointCloud2)  ──→ GraspNet

话题:
  /yolo/detections              std_msgs/String  每帧检测元数据 (JSON)
  /yolo/request_object_cloud    std_msgs/Int32   决策系统发布要抓的 track_id
  /yolo/object_cloud            PointCloud2      对应物体的 3D 点云
  /yolo/debug_cloud             PointCloud2      调试用，全部物体合并点云
  /yolo/markers                 MarkerArray      调试用 RViz 标记

用法:
  python yolo_inference_node.py --ros-args -p mode:=production
  python yolo_inference_node.py --ros-args -p mode:=debug
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

# ── 可选导入 ──────────────────────────────────────────────
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

def build_pointcloud2(points: np.ndarray, header: Header) -> PointCloud2:
    """numpy (N,3) float32 → sensor_msgs/PointCloud2。"""
    n = points.shape[0]
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = n
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * n
    msg.fields = [
        PointField(name="x",         offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name="y",         offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name="z",         offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    buf = np.zeros(n, dtype=[
        ("x", np.float32), ("y", np.float32),
        ("z", np.float32), ("intensity", np.float32),
    ])
    buf["x"] = points[:, 0]
    buf["y"] = points[:, 1]
    buf["z"] = points[:, 2]
    msg.data = buf.tobytes()
    return msg


def mask_to_pointcloud(mask: np.ndarray, depth: np.ndarray,
                       camera_info: CameraInfo,
                       scale: float = 0.001) -> np.ndarray:
    """mask + 深度图 + 内参 → 物体 3D 点云 (N,3) 米。"""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.empty((0, 3), dtype=np.float32)

    Z = depth[ys, xs].astype(np.float32) * scale
    valid = (Z > 0.05) & (Z < 3.0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)
    ys, xs, Z = ys[valid], xs[valid], Z[valid]

    fx, fy = camera_info.k[0], camera_info.k[4]
    cx, cy = camera_info.k[2], camera_info.k[5]
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy
    return np.stack([X, Y, Z], axis=-1).astype(np.float32)


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
    # 兜底：从 mask 算 bbox
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


# ============================================================
# 主节点
# ============================================================

class YOLOInferenceNode(Node):
    """YOLO11-seg 推理 + IoU 追踪 + 按需点云。零自定义接口依赖。"""

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

        # ── 点云缓存 ─────────────────────────────────────
        self._cloud_cache: Dict[int, np.ndarray] = {}   # track_id → (N,3)
        self._cloud_lock = threading.Lock()

        # ── QoS ──────────────────────────────────────────
        # 这给谁都不许改，否则会打断腿
        qos_sensor = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE)
        qos_reliable = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # ── 订阅 ─────────────────────────────────────────
        self.color_sub = self.create_subscription(
            Image, "/Wrist_Camera/d435i/color/image_raw",
            self._color_cb, qos_sensor)
        self.info_sub = self.create_subscription(
            CameraInfo, "/Wrist_Camera/d435i/color/camera_info",
            self._info_cb, qos_sensor)
        self.depth_sub = self.create_subscription(
            Image, "/Wrist_Camera/d435i/aligned_depth_to_color/image_raw",
            self._depth_cb, qos_sensor)

        # ── 发布 ─────────────────────────────────────────
        # 检测元数据（JSON）
        self.det_pub = self.create_publisher(
            String, "/yolo/detections", qos_reliable)

        # 按需点云响应
        self.cloud_pub = self.create_publisher(
            PointCloud2, "/yolo/object_cloud", qos_reliable)

        # 点云请求订阅：决策系统发 Int32(track_id) → 本节点发布对应 PointCloud2
        self.request_sub = self.create_subscription(
            Int32, "/yolo/request_object_cloud",
            self._on_request_cloud, qos_reliable)

        # 调试
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
        self._latest_depth: Optional[np.ndarray] = None
        self._depth_scale: float = 0.001
        self._latest_info: Optional[CameraInfo] = None
        self._color_frame_id: str = "d435i_color_optical_frame"

        self._frame_count = 0
        self._total_inference_ms = 0.0
        self._color_msg_count = 0      # 诊断：收到的 color 消息总数
        self._depth_msg_count = 0      # 诊断：收到的 depth 消息总数

        # ── 定时器 ───────────────────────────────────────
        self._timer = self.create_timer(0.05, self._inference_loop)
        self._status_timer = self.create_timer(5.0, self._print_status)

        self.get_logger().info(
            f"YOLOInferenceNode 启动 | mode={self._mode} | "
            f"model={self.get_parameter('model_path').value} | "
            f"imgsz={self.get_parameter('imgsz').value} | "
            f"conf={self.get_parameter('conf').value} | "
            f"零编译，纯标准消息类型"
        )

    # ============================================================
    # 回调
    # ============================================================

    def _color_cb(self, msg: Image):
        try:
            self._latest_color, _ = ros_image_to_numpy(msg)
            self._color_msg_count += 1
        except Exception as e:
            self.get_logger().error(f"color 解码失败: {e}")


    def _info_cb(self, msg: CameraInfo):
        self._latest_info = msg
        if msg.header.frame_id:
            self._color_frame_id = msg.header.frame_id

    def _depth_cb(self, msg: Image):
        try:
            self._latest_depth, self._depth_scale = ros_image_to_numpy(msg)
            self._depth_msg_count += 1
        except Exception as e:
            self.get_logger().error(f"depth 解码失败: {e}")

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
            # 发布空点云表示失败
            now = self.get_clock().now().to_msg()
            header = Header(stamp=now, frame_id=self._color_frame_id)
            empty = build_pointcloud2(np.empty((0, 3), dtype=np.float32), header)
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
        depth = self._latest_depth
        camera_info = self._latest_info

        if image is None:
            return
        self._latest_color = None
        self._frame_count += 1

        # ── 1. YOLO 推理 ──────────────────────────────
        t0 = time.perf_counter()
        result = self.yolo.predict(image)
        inference_ms = result["inference_time_ms"]
        self._total_inference_ms += inference_ms

        objects = result.get("objects", [])

        # ── 2. IoU 追踪（始终调用，让 track 自然老化）──
        dets = [{"bbox": _get_bbox(o), "class_name": o["class_name"],
                 "confidence": o["confidence"]} for o in objects]
        tracks = self._tracker.update(dets)

        # ── 3. 用 tracker 的 det_idx 直接映射，不再二次匹配 ──
        tracked_objects: List[Tuple[int, dict]] = []
        new_clouds: Dict[int, np.ndarray] = {}

        for t in tracks:
            if t.det_idx < 0:               # 未匹配到当前帧 detection
                continue
            obj = objects[t.det_idx]
            obj["track_id"] = t.id

            if depth is not None and camera_info is not None:
                cloud = mask_to_pointcloud(
                    obj["mask"], depth, camera_info, scale=self._depth_scale)
            else:
                cloud = np.empty((0, 3), dtype=np.float32)

            obj["cloud"] = cloud
            tracked_objects.append((t.id, obj))
            if cloud.shape[0] >= 10:
                new_clouds[t.id] = cloud

        # 清理已删除 track 的旧点云缓存，写入新点云
        active_ids = {t.id for t in tracks}
        with self._cloud_lock:
            for k in list(self._cloud_cache):
                if k not in active_ids:
                    del self._cloud_cache[k]
            self._cloud_cache.update(new_clouds)

        # ── 4. 发布检测元数据 (JSON) ──────────────────
        self._publish_detections(tracked_objects)

        # ── 5. 调试 ───────────────────────────────────
        if self._publish_debug_cloud or self._mode == "debug":
            clouds = [obj["cloud"] for _, obj in tracked_objects
                      if obj.get("cloud") is not None and obj["cloud"].shape[0] >= 10]
            if clouds:
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
        # 没必要，还会降低帧率
        # if self._debug and _HAS_VIZ and objects:
        #     save_debug_image(image, objects, os.path.join(
        #         self._debug_dir, f"frame_{self._frame_count:06d}.png"))

        # ── 日志 ─────────────────────────────────────
        total_ms = (time.perf_counter() - t0) * 1000
        names = ", ".join(
            f"#{tid} {obj['class_name']}" for tid, obj in tracked_objects[:5])
        self.get_logger().info(
            f"[#{self._frame_count}] {len(tracked_objects)} 目标: {names} | "
            f"推理={inference_ms:.0f}ms 总={total_ms:.0f}ms",
            throttle_duration_sec=2.0)

    # ============================================================
    # 检测发布 (JSON)
    # ============================================================

    def _publish_detections(self, tracked_objects: List[Tuple[int, dict]]):
        """将追踪结果序列化为 JSON → /yolo/detections (std_msgs/String)。"""
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
                f"depth={self._depth_msg_count} "
                f"latest_color={self._latest_color is not None}",
                throttle_duration_sec=5.0)
            return
        avg = self._total_inference_ms / max(self._frame_count, 1)
        self.get_logger().info(
            f"[状态] {self._frame_count} 帧 | color={self._color_msg_count} depth={self._depth_msg_count} | "
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