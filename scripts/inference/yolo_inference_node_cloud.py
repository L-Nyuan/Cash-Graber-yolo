#!/usr/bin/env python3
"""
YOLO11-seg ROS2 推理节点（RealSense 彩色点云直取版，带同步诊断）
============================================================

相比原版主要改动：
  1. 点云选择增加 sync_tolerance；找不到容差内的同步点云时，默认跳过本帧裁剪，
     避免旧 segmentation 套到新点云上。
  2. 清空点云缓存，防止决策系统请求到上一帧/上一姿态的旧点云。
  3. 增加 sync/crop 调试日志，可直接用 ROS2 logger debug 级别输出。

参数按前缀分组，命令行更规整：
  model.   模型（权重、尺寸、阈值）
  tracker. IoU 追踪器
  cloud.   点云后处理（mask 腐蚀、离群点移除）
  sync.    彩色图/点云时间同步
  topic.   话题名
  debug.   调试输出（仅 mode=debug 生效）

运行示例：
  # 生产模式（默认）：只输出识别标签 /yolo/detections + 按需点云 /yolo/object_cloud
  python yolo_inference_node_cloud.py --ros-args

  # 调试模式：额外发布所有识别物品的点云 /yolo/debug_cloud 与 RViz 标记 /yolo/markers
  python yolo_inference_node_cloud.py --ros-args -p mode:=debug \
    --log-level yolo_inference_node:=debug

  # 手腕相机移动快时，可以收紧同步容差：
  python yolo_inference_node_cloud.py --ros-args -p sync.tolerance:=0.03
"""

import json
import sys
import time
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Header, String, Int32

from cloud_utils import (
    build_pointcloud2,
    crop_cloud_by_mask,
    ensure_mask_resolution,
    erode_mask,
    remove_outliers_sor,
    realsense_cloud_to_xyzrgb,
)
from node_config import NodeConfig
from cloud_state import CloudCache, CloudSyncMatcher
from image_utils import ros_image_to_numpy
from yolo_inference import YOLOSegInference
from object_tracker import ObjectTracker
from rviz_launcher import RvizLauncher

try:
    from marker_rviz import _publish_markers
    _HAS_MARKER = True
except ImportError:
    _HAS_MARKER = False


# ============================================================
# 工具函数
# ============================================================

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


@dataclass
class FrameData:
    """_acquire_frame 返回的一帧输入（含点云时间同步结果）。"""

    image: np.ndarray
    color_stamp: object
    cloud_xyz: Optional[np.ndarray]
    cloud_rgb: Optional[np.ndarray]
    cloud_stamp: object
    cloud_frame: str
    matched: bool


# ============================================================
# 主节点
# ============================================================

class YOLOInferenceNode(Node):
    """YOLO11-seg 推理 + IoU 追踪 + RealSense 彩色点云裁剪。"""

    def __init__(self):
        super().__init__("yolo_inference_node")

        # ── 参数：全部声明与解析集中在 node_config.NodeConfig ──
        cfg = NodeConfig.load(self)

        # ── 模式与话题 ────────────────────────────────────
        self._mode = cfg.mode
        self._debug_mode = cfg.debug_mode
        self._cloud_topic = cfg.cloud_topic
        self._color_topic = cfg.color_topic
        self._info_topic = cfg.info_topic

        # ── 同步参数 ──────────────────────────────────────
        self._sync_tolerance_ns = cfg.sync_tolerance_ns
        self._sync_debug = cfg.sync_debug
        self._cloud_buffer_size = cfg.cloud_buffer_size

        # ── YOLO 模型 ────────────────────────────────────
        self.yolo = YOLOSegInference(
            model_path=cfg.model_path,
            imgsz=cfg.model_imgsz,
            conf=cfg.model_conf,
            iou=cfg.model_iou,
        )

        self._debug_cloud_interval = cfg.debug_cloud_interval
        self._debug_dir = cfg.debug_dir
        if self._debug_mode:
            os.makedirs(self._debug_dir, exist_ok=True)
        self._launch_rviz = cfg.launch_rviz
        self._rviz_config = cfg.rviz_config
        self._rviz = RvizLauncher(
            cfg.rviz_config, cfg.debug_dir, self.get_logger())
        if self._debug_mode and self._launch_rviz:
            self._rviz.start()

        # ── 点云后处理 ─────────────────────────────────────
        self._mask_erode = cfg.mask_erode
        self._cloud_sor = cfg.cloud_sor
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
            max_age=cfg.tracker_max_age,
            min_hits=cfg.tracker_min_hits,
            iou_threshold=cfg.tracker_iou,
        )

        # ── 点云缓存：track_id → (N,6) xyz+rgb ─────────
        self._cloud_cache = CloudCache()

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

        if self._debug_mode:
            self.debug_cloud_pub = self.create_publisher(
                PointCloud2, "/yolo/debug_cloud", qos_reliable)
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
        self._color_frame_id: str = "d435i_color_optical_frame"

        self._frame_count = 0
        self._total_inference_ms = 0.0
        self._color_msg_count = 0
        self._color_encoding: str = ""
        self._shape_warned = False
        self._last_debug_pub = 0.0

        # ── 时间同步：点云缓冲与最近邻匹配 ──────────────
        self._sync = CloudSyncMatcher(
            tolerance_ns=self._sync_tolerance_ns,
            buffer_size=self._cloud_buffer_size,
            sync_debug=self._sync_debug,
            logger=self.get_logger(),
            require_synced_cloud=cfg.require_synced_cloud,
            pending_max_wait=cfg.pending_max_wait,
        )

        # ── 定时器 ───────────────────────────────────────
        self._timer = self.create_timer(0.05, self._inference_loop)
        self._status_timer = self.create_timer(5.0, self._print_status)

        self.get_logger().info(
            f"YOLOInferenceNode(RealSense云, synced) 启动 | mode={self._mode} | "
            f"model={self.get_parameter('model.path').value} | "
            f"imgsz={self.get_parameter('model.imgsz').value} | "
            f"点云来源: {self._cloud_topic} | "
            f"sync.tolerance={self.get_parameter('sync.tolerance').value:.3f}s | "
            f"sync.require_cloud={cfg.require_synced_cloud} | "
            f"输出: /yolo/detections + /yolo/object_cloud"
            + (" + /yolo/debug_cloud + /yolo/markers" if self._debug_mode else "")
            + (f" | rviz2={self._rviz_config}" if self._rviz.running else "")
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
            self._sync.add_cloud(
                msg.header.stamp, msg.header.frame_id, xyz, rgb)
        except Exception as e:
            self.get_logger().error(f"点云解码失败: {e}")

    # ============================================================
    # 按需点云请求（决策系统 → 本节点）
    # ============================================================

    def _on_request_cloud(self, msg: Int32):
        object_id = msg.data
        cloud = self._cloud_cache.get(object_id)

        if cloud is None or cloud.shape[0] < 10:
            self.get_logger().warn(
                f"[Request] ID={object_id} 无有效点云 "
                f"(缓存: {self._cloud_cache.keys()})")
            stamp = self._sync.last_crop_stamp or self.get_clock().now().to_msg()
            header = Header(stamp=stamp, frame_id=self._sync.last_crop_frame)
            empty = build_pointcloud2(np.empty((0, 6), dtype=np.float32), header)
            self.cloud_pub.publish(empty)
            return

        stamp = self._sync.last_crop_stamp or self.get_clock().now().to_msg()
        header = Header(stamp=stamp, frame_id=self._sync.last_crop_frame)
        pc2_msg = build_pointcloud2(cloud, header)
        self.cloud_pub.publish(pc2_msg)
        self.get_logger().info(
            f"[Request] ID={object_id} → {cloud.shape[0]} 点",
            throttle_duration_sec=1.0)

    # ============================================================
    # 推理循环
    # ============================================================

    def _acquire_frame(self) -> Optional[FrameData]:
        """取一帧可处理的彩色图并完成点云时间同步决策（含 pending 等待逻辑）。"""
        camera_info = self._latest_info
        if camera_info is None:
            return None

        sync = self._sync.acquire(
            self._latest_color, self._latest_color_stamp,
            self._color_frame_id)
        if sync is None:
            return None
        if sync.consumed_latest:
            self._latest_color = None

        # 首帧诊断
        if not self._shape_warned:
            self._shape_warned = True
            self.get_logger().info(
                f"[诊断] color={sync.image.shape} dtype={sync.image.dtype} "
                f"encoding={self._color_encoding} | "
                f"RealSense云={None if sync.xyz is None else sync.xyz.shape} "
                f"(已收 {self._sync.msgs} 帧) | "
                f"camera_info={camera_info.width}x{camera_info.height} | "
                f"color_frame={self._color_frame_id} | "
                f"cloud_frame={self._latest_cloud_frame or '(未收到点云)'} | "
                f"sync_tolerance={self._sync_tolerance_ns / 1e6:.2f}ms")
            if sync.xyz is None:
                self.get_logger().warn(
                    "[诊断] 尚未收到/未匹配 RealSense 点云！请确认相机已启动，"
                    "且命令行带 pointcloud.enable:=true；若已启动，查看 sync 日志。")

        return FrameData(
            image=sync.image, color_stamp=sync.color_stamp,
            cloud_xyz=sync.xyz, cloud_rgb=sync.rgb,
            cloud_stamp=sync.cloud_stamp, cloud_frame=sync.cloud_frame,
            matched=sync.matched)

    def _run_inference(self, image, camera_info):
        """YOLO 推理 + IoU 追踪，组装带 track_id 与空点云占位的目标列表。

        Returns:
            (tracked_objects, tracks, inference_ms)
        """
        target_hw = (int(camera_info.height), int(camera_info.width))
        result = self.yolo.predict(image)
        inference_ms = result["inference_time_ms"]
        self._total_inference_ms += inference_ms
        objects = result.get("objects", [])

        if objects and not hasattr(self, '_mask_shape_warned'):
            self._mask_shape_warned = True
            m0 = objects[0].get("mask")
            if m0 is not None:
                self.get_logger().info(
                    f"[诊断] YOLO mask={m0.shape} → 缩放到 {target_hw} 再投影裁剪")

        # ── IoU 追踪 ─────────────────────────────────
        dets = [{"bbox": _get_bbox(o), "class_name": o["class_name"],
                 "confidence": o["confidence"]} for o in objects]
        tracks = self._tracker.update(dets)

        tracked_objects: List[Tuple[int, dict]] = []
        for t in tracks:
            if t.det_idx < 0:
                continue
            obj = objects[t.det_idx]
            obj["track_id"] = t.id
            obj["cloud"] = np.empty((0, 6), dtype=np.float32)
            tracked_objects.append((t.id, obj))

        return tracked_objects, tracks, inference_ms

    def _crop_tracked_clouds(self, tracked_objects, frame,
                             camera_info) -> Dict[int, np.ndarray]:
        """逐目标裁剪点云（mask 缩放/腐蚀/裁剪/SOR），返回本帧新点云。"""
        cloud_xyz = frame.cloud_xyz
        cloud_rgb = frame.cloud_rgb
        target_hw = (int(camera_info.height), int(camera_info.width))
        new_clouds: Dict[int, np.ndarray] = {}

        if cloud_xyz is None and frame.matched:
            self.get_logger().warn(
                "[sync] matched=true 但 cloud_xyz=None，裁剪将跳过",
                throttle_duration_sec=1.0)

        for tid, obj in tracked_objects:
            mask = obj.get("mask")
            cloud = np.empty((0, 6), dtype=np.float32)

            if mask is not None and cloud_xyz is not None:
                try:
                    mask_shape_before = mask.shape
                    mask_count_before = int(np.count_nonzero(mask))

                    mask = ensure_mask_resolution(mask, target_hw)
                    if self._mask_erode > 0:
                        mask = erode_mask(mask, self._mask_erode)

                    mask_count_after = int(np.count_nonzero(mask))
                    cloud = crop_cloud_by_mask(
                        cloud_xyz, cloud_rgb, mask, camera_info)
                    crop_count = cloud.shape[0]

                    if self._o3d is not None and cloud.shape[0] > 50:
                        cloud = remove_outliers_sor(
                            cloud, self._o3d, nb_neighbors=20, std_ratio=1.0)

                    if self._sync_debug:
                        z_mean = float(np.mean(cloud[:, 2])) if cloud.shape[0] else 0.0
                        rgb_mean = tuple(
                            float(np.mean(cloud[:, 3 + i])) if cloud.shape[0] else 0.0
                            for i in range(3))
                        self.get_logger().debug(
                            f"[crop] tid={tid} {obj['class_name']} "
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
            if cloud.shape[0] >= 10:
                new_clouds[tid] = cloud

        return new_clouds

    def _update_cloud_cache(self, tracks, new_clouds, cloud_xyz, matched):
        """按同步结果更新点云缓存：无匹配时清空，有匹配时清理过期并写入。"""
        if cloud_xyz is None and not matched:
            # 清掉旧缓存，避免决策系统请求到上一姿态的点云。
            self._cloud_cache.clear()
        elif cloud_xyz is not None:
            # 只有本帧实际产生过同步裁剪时才更新缓存，避免旧点云跨帧残留。
            active_ids = {t.id for t in tracks}
            self._cloud_cache.update(active_ids, new_clouds)

    def _publish_debug_outputs(self, tracked_objects):
        """debug 模式：发布合并点云与 RViz markers。"""
        if not self._debug_mode:
            return
        clouds = [obj["cloud"] for _, obj in tracked_objects
                  if obj.get("cloud") is not None and obj["cloud"].shape[0] >= 10]
        if clouds and time.time() - self._last_debug_pub > self._debug_cloud_interval:
            self._last_debug_pub = time.time()
            stamp = self._sync.last_crop_stamp
            if stamp is None:
                stamp = self.get_clock().now().to_msg()
            header = Header(stamp=stamp, frame_id=self._sync.last_crop_frame)
            self.debug_cloud_pub.publish(
                build_pointcloud2(np.vstack(clouds), header))

        if _HAS_MARKER:
            if hasattr(self, "marker_pub"):
                now = self.get_clock().now().to_msg()
                header = Header(stamp=now, frame_id=self._color_frame_id)
                _publish_markers([obj for _, obj in tracked_objects],
                                 header, publisher=self.marker_pub)

    def _inference_loop(self):
        frame = self._acquire_frame()
        if frame is None:
            return

        image = frame.image
        cloud_xyz = frame.cloud_xyz
        cloud_rgb = frame.cloud_rgb
        cloud_stamp = frame.cloud_stamp
        cloud_frame = frame.cloud_frame
        matched = frame.matched
        camera_info = self._latest_info

        self._frame_count += 1

        # ── 1+2. YOLO 推理 + IoU 追踪 ─────────────────
        t0 = time.perf_counter()
        tracked_objects, tracks, inference_ms = self._run_inference(
            image, camera_info)

        # ── 3+4. 裁剪点云 + 更新缓存 ─────────────────
        new_clouds = self._crop_tracked_clouds(
            tracked_objects, frame, camera_info)
        self._update_cloud_cache(tracks, new_clouds, cloud_xyz, matched)

        # ── 5. 发布检测元数据 ─────────────────────────
        self._publish_detections(tracked_objects)

        # ── 6. debug 输出（合并点云 + RViz markers）───
        self._publish_debug_outputs(tracked_objects)

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
            det = {
                "id": tid,
                "class_name": obj["class_name"],
                "confidence": round(float(obj["confidence"]), 4),
                "center_x": round(float(bbox[0] + bbox[2]) / 2.0, 1),
                "center_y": round(float(bbox[1] + bbox[3]) / 2.0, 1),
                "bbox": [round(float(v), 1) for v in bbox],
            }
            # debug 模式附加每个物品的点云信息（点数 + 质心）
            if self._debug_mode:
                cloud = obj.get("cloud")
                det["cloud_points"] = int(cloud.shape[0]) if cloud is not None else 0
                if cloud is not None and cloud.shape[0] > 0:
                    det["cloud_centroid"] = [
                        round(float(v), 4)
                        for v in np.mean(cloud[:, :3], axis=0)]
            payload["detections"].append(det)

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
                f"cloud={self._sync.msgs} "
                f"info={self._latest_info is not None}",
                throttle_duration_sec=5.0)
            return
        avg = self._total_inference_ms / max(self._frame_count, 1)
        self.get_logger().info(
            f"[状态] {self._frame_count} 帧 | color={self._color_msg_count} "
            f"cloud={self._sync.msgs} | "
            f"sync_ok={self._sync.sync_ok_count} "
            f"sync_skip={self._sync.sync_skip_count} "
            f"sync_hold={self._sync.hold_count} | "
            f"avg {avg:.0f}ms ({1000/avg:.0f} FPS) | "
            f"活跃目标 {len(self._cloud_cache)} | mode={self._mode}",
            throttle_duration_sec=5.0)


def main():
    rclpy.init(args=sys.argv)
    node = None
    try:
        node = YOLOInferenceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node._rviz.stop()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
