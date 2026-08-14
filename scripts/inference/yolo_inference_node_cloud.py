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
import subprocess
import sys
import time
import os
import threading
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
    stamp_ns,
    stamp_text,
)
from node_config import NodeConfig
from image_utils import ros_image_to_numpy
from yolo_inference import YOLOSegInference
from object_tracker import ObjectTracker

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
        self._require_synced_cloud = cfg.require_synced_cloud
        self._sync_debug = cfg.sync_debug
        self._cloud_buffer_size = cfg.cloud_buffer_size
        self._pending_max_wait = cfg.pending_max_wait

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
        self._rviz_proc: Optional[subprocess.Popen] = None
        self._rviz_stopping = False
        if self._debug_mode and self._launch_rviz:
            self._launch_rviz_window()

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
            f"model={self.get_parameter('model.path').value} | "
            f"imgsz={self.get_parameter('model.imgsz').value} | "
            f"点云来源: {self._cloud_topic} | "
            f"sync.tolerance={self.get_parameter('sync.tolerance').value:.3f}s | "
            f"sync.require_cloud={self._require_synced_cloud} | "
            f"输出: /yolo/detections + /yolo/object_cloud"
            + (" + /yolo/debug_cloud + /yolo/markers" if self._debug_mode else "")
            + (f" | rviz2={self._rviz_config}" if self._rviz_proc is not None else "")
        )

    # ============================================================
    # rviz2（debug 模式自动打开）
    # ============================================================

    def _launch_rviz_window(self):
        """debug 模式下自动启动 rviz2 并加载调试配置。"""
        cfg = self._rviz_config
        if not os.path.isfile(cfg):
            self.get_logger().warn(
                f"debug.launch_rviz=true 但配置不存在: {cfg}，跳过 rviz 启动")
            return
        rviz_env = self._build_rviz_env()
        self.get_logger().info(
            f"准备启动 rviz2 | DISPLAY={rviz_env.get('DISPLAY', '(未设置)')} "
            f"XAUTHORITY={rviz_env.get('XAUTHORITY', '(未设置)')} "
            f"QT_QPA_PLATFORM_PLUGIN_PATH="
            f"{rviz_env.get('QT_QPA_PLATFORM_PLUGIN_PATH', '(系统默认)')}")
        log_path = os.path.join(self._debug_dir, "rviz2.log")
        try:
            with open(log_path, "w") as log_f:
                self._rviz_proc = subprocess.Popen(
                    ["rviz2", "-d", cfg],
                    env=rviz_env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            proc = self._rviz_proc
            self.get_logger().info(
                f"已启动 rviz2 (pid={proc.pid})，加载配置: {cfg}，"
                f"输出见 {log_path}")
            # 后台观察：rviz2 若异常退出（如显示连接失败），把它的输出打出来
            threading.Thread(
                target=self._watch_rviz, args=(proc, log_path),
                daemon=True).start()
        except FileNotFoundError:
            self.get_logger().warn(
                "未找到 rviz2 命令，跳过（请确认已 source ROS 环境）")
            self._rviz_proc = None
        except Exception as e:
            self.get_logger().warn(f"rviz2 启动失败: {e}")
            self._rviz_proc = None

    @staticmethod
    def _build_rviz_env() -> dict:
        """构造 rviz2 子进程环境，清除 conda/cv2 污染的 Qt 变量。

        在 conda 环境里 `import cv2` 会把 QT_QPA_PLATFORM_PLUGIN_PATH 指向
        cv2 自带的 Qt 插件目录，该插件与系统 Qt 版本不兼容，导致 rviz2
        启动即 abort（rc=-6）。这里改回系统 Qt 插件目录，并防御性清理
        LD_LIBRARY_PATH 中的 conda 库路径。
        """
        env = os.environ.copy()
        for k in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR",
                  "QT_PLUGIN_PATH"):
            env.pop(k, None)
        system_plugins = "/usr/lib/x86_64-linux-gnu/qt5/plugins"
        if os.path.isdir(system_plugins):
            env["QT_QPA_PLATFORM_PLUGIN_PATH"] = system_plugins
        libs = env.get("LD_LIBRARY_PATH", "")
        clean = [
            p for p in libs.split(":")
            if p and not any(s in p for s in ("/miniconda3/", "/anaconda3/",
                                              "/conda/"))
        ]
        if clean:
            env["LD_LIBRARY_PATH"] = ":".join(clean)
        return env

    def _watch_rviz(self, proc, log_path):
        """等待 rviz2 退出，异常退出时输出其日志末尾。"""
        rc = proc.wait()
        if self._rviz_stopping:
            return
        if rc != 0:
            self.get_logger().warn(
                f"rviz2 异常退出 rc={rc}，输出见 {log_path}:\n"
                f"{self._read_tail(log_path)}")
        else:
            self.get_logger().info(f"rviz2 已退出（rc=0），输出见 {log_path}")

    @staticmethod
    def _read_tail(path: str, n: int = 20) -> str:
        """读取文件末尾 n 行，用于输出 rviz2 的报错。"""
        try:
            with open(path) as f:
                lines = f.readlines()
            return "".join(lines[-n:]).strip() or "(日志为空)"
        except Exception:
            return "(无法读取 rviz2 日志)"

    def _stop_rviz(self):
        """节点退出时关闭自动启动的 rviz2，避免残留孤儿进程。"""
        if self._rviz_proc is not None and self._rviz_proc.poll() is None:
            self._rviz_stopping = True
            self.get_logger().info("节点退出，关闭 rviz2")
            self._rviz_proc.terminate()
        self._rviz_proc = None

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
                    f"[cloud_cb] stamp={stamp_text(msg.header.stamp)} "
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

        t_img = stamp_ns(color_stamp)
        best_idx = -1
        best_d = float("inf")
        for i, (st, _fr, _xyz, _rgb) in enumerate(self._cloud_buffer):
            d = abs(stamp_ns(st) - t_img)
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
                f"[sync] color={stamp_text(color_stamp)} cloud={stamp_text(cloud_stamp)} "
                f"delta=None buffer={cloud_n} match={matched}")
            return

        delta_ms = delta_ns / 1_000_000.0
        self.get_logger().info(
            f"[sync] color={stamp_text(color_stamp)} "
            f"cloud={stamp_text(cloud_stamp)} "
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
                f"[sync] pending color {stamp_text(self._pending_color_stamp)} "
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
                    stamp_ns(self._pending_color_stamp) <= stamp_ns(color_stamp))

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
                        f"[sync] WAIT_CLOUD color={stamp_text(color_stamp)} "
                        f"cloud_buffer=0 cloud_msgs={self._cloud_msg_count} "
                        f"wait={self._pending_max_wait:.3f}s")
                else:
                    self.get_logger().debug(
                        f"[sync] HOLD color={stamp_text(color_stamp)} "
                        f"cloud={stamp_text(cloud_stamp)} "
                        f"delta={delta_text} buffer={len(self._cloud_buffer)} "
                        f"cloud_msgs={self._cloud_msg_count}")
                return

            self.get_logger().debug(
                f"[sync] DROP color={stamp_text(color_stamp)} "
                f"cloud={stamp_text(cloud_stamp)} "
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

        # ── 5. 调试点云（debug 模式：所有识别物品的点云）──
        if self._debug_mode:
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

        if self._debug_mode and _HAS_MARKER:
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
    node = None
    try:
        node = YOLOInferenceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node._stop_rviz()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
