# cloud_state.py —— 节点状态组件
#
# 拆分计划 Phase 5/6（v2）：
#   CloudSyncMatcher：彩色图/点云时间同步（缓冲 + 最近邻匹配 + 节流日志）
#   CloudCache（Phase 6 加入）：track_id → 点云线程安全缓存
# 生命周期与节点一致，由回调/定时器驱动；不创建 timer/线程。

import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from cloud_utils import stamp_ns, stamp_text


@dataclass
class SyncMatch:
    """一次彩色图 → 点云匹配的结果。"""

    xyz: Optional[np.ndarray] = None
    rgb: Optional[np.ndarray] = None
    cloud_stamp: object = None
    cloud_frame: str = ""
    delta_ns: Optional[int] = None
    matched: bool = False


class CloudSyncMatcher:
    """彩色图/点云时间同步：维护点云缓冲，按时间戳最近邻匹配。"""

    def __init__(self, tolerance_ns: int, buffer_size: int,
                 sync_debug: bool, logger):
        self._tolerance_ns = tolerance_ns
        self._buffer_size = buffer_size
        self._sync_debug = sync_debug
        self._logger = logger
        self._buffer: List[Tuple[object, str, np.ndarray, np.ndarray]] = []
        self._msgs = 0
        self._last_log_time = 0.0

    @property
    def msgs(self) -> int:
        """累计收到的点云消息数。"""
        return self._msgs

    def __len__(self) -> int:
        return len(self._buffer)

    def add_cloud(self, stamp, frame_id, xyz, rgb):
        """回调收到点云后加入缓冲（超长裁剪最早的帧）。"""
        self._buffer.append((stamp, frame_id, xyz, rgb))
        if len(self._buffer) > self._buffer_size:
            self._buffer.pop(0)
        self._msgs += 1

        if self._sync_debug and self._msgs <= 3:
            self._logger.info(
                f"[cloud_cb] stamp={stamp_text(stamp)} "
                f"frame={frame_id or '(empty)'} "
                f"points={len(xyz)}"
            )

    def match(self, color_stamp, fallback_frame: str) -> SyncMatch:
        """为彩色图匹配最接近的点云帧。

        容差内算 matched=True；否则 matched=False 且不带点云（宁可跳过裁剪，
        也不把旧 mask 套到新点云上）。
        """
        if color_stamp is None or not self._buffer:
            if color_stamp is not None and not self._buffer:
                self._logger.warn(
                    "[sync] color 已到但 cloud_buffer 为空，无法匹配",
                    throttle_duration_sec=1.0)
            return SyncMatch(cloud_frame=fallback_frame)

        t_img = stamp_ns(color_stamp)
        best_idx = -1
        best_d = float("inf")
        for i, (st, _fr, _xyz, _rgb) in enumerate(self._buffer):
            d = abs(stamp_ns(st) - t_img)
            if d < best_d:
                best_d = d
                best_idx = i

        if best_idx < 0:
            return SyncMatch(cloud_frame=fallback_frame)

        cloud_stamp, cloud_frame, xyz, rgb = self._buffer[best_idx]
        matched = best_d <= self._tolerance_ns
        return SyncMatch(
            xyz=xyz, rgb=rgb, cloud_stamp=cloud_stamp,
            cloud_frame=cloud_frame or fallback_frame,
            delta_ns=best_d, matched=matched)

    def log(self, color_stamp, cloud_stamp, delta_ns, matched, cloud_n):
        """节流输出 [sync] 日志（0.2s 内只打一条）。"""
        if not self._sync_debug:
            return
        now = time.time()
        if now - self._last_log_time < 0.2:
            return
        self._last_log_time = now

        if delta_ns is None:
            self._logger.warn(
                f"[sync] color={stamp_text(color_stamp)} "
                f"cloud={stamp_text(cloud_stamp)} "
                f"delta=None buffer={cloud_n} match={matched}")
            return

        delta_ms = delta_ns / 1_000_000.0
        self._logger.info(
            f"[sync] color={stamp_text(color_stamp)} "
            f"cloud={stamp_text(cloud_stamp)} "
            f"delta={delta_ms:.2f}ms buffer={cloud_n} match={matched}"
        )


class CloudCache:
    """track_id → 点云 (N,6) 的线程安全缓存。"""

    def __init__(self):
        self._cache: Dict[int, np.ndarray] = {}
        self._lock = threading.Lock()

    def get(self, track_id: int) -> Optional[np.ndarray]:
        with self._lock:
            return self._cache.get(track_id)

    def update(self, active_ids, new_clouds: Dict[int, np.ndarray]):
        """清理不在 active_ids 的旧条目，并写入本帧新点云。"""
        with self._lock:
            for k in list(self._cache):
                if k not in active_ids:
                    del self._cache[k]
            self._cache.update(new_clouds)

    def clear(self):
        with self._lock:
            self._cache.clear()

    def keys(self) -> List[int]:
        with self._lock:
            return list(self._cache.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
