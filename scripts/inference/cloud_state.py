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
    """一次彩色图 → 点云匹配的结果（acquire 时携带待处理帧）。"""

    image: Optional[np.ndarray] = None
    color_stamp: object = None
    xyz: Optional[np.ndarray] = None
    rgb: Optional[np.ndarray] = None
    cloud_stamp: object = None
    cloud_frame: str = ""
    delta_ns: Optional[int] = None
    matched: bool = False
    consumed_latest: bool = False


class CloudSyncMatcher:
    """彩色图/点云时间同步：维护点云缓冲，按时间戳最近邻匹配。"""

    def __init__(self, tolerance_ns: int, buffer_size: int,
                 sync_debug: bool, logger,
                 require_synced_cloud: bool = True,
                 pending_max_wait: float = 0.12):
        self._tolerance_ns = tolerance_ns
        self._buffer_size = buffer_size
        self._sync_debug = sync_debug
        self._logger = logger
        self._require_synced_cloud = require_synced_cloud
        self._pending_max_wait = pending_max_wait
        self._buffer: List[Tuple[object, str, np.ndarray, np.ndarray]] = []
        self._msgs = 0
        self._last_log_time = 0.0
        self._pending_color: Optional[np.ndarray] = None
        self._pending_color_stamp = None
        self._pending_color_deadline = 0.0
        self._sync_ok_count = 0
        self._sync_skip_count = 0
        self._hold_count = 0
        self._last_crop_stamp = None
        self._last_crop_frame = "d435i_color_optical_frame"

    @property
    def msgs(self) -> int:
        """累计收到的点云消息数。"""
        return self._msgs

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def sync_ok_count(self) -> int:
        return self._sync_ok_count

    @property
    def sync_skip_count(self) -> int:
        return self._sync_skip_count

    @property
    def hold_count(self) -> int:
        return self._hold_count

    @property
    def last_crop_stamp(self):
        return self._last_crop_stamp

    @property
    def last_crop_frame(self) -> str:
        return self._last_crop_frame

    def acquire(self, latest_color, latest_color_stamp,
                fallback_frame: str) -> Optional[SyncMatch]:
        """决定本轮处理哪个彩色图并完成点云同步匹配（含 pending 等待逻辑）。

        - 上一帧 HOLD 的 pending 帧优先于更新的 latest 帧；
        - 无容差内匹配且 require_synced_cloud 时先 HOLD 一小段，超时 DROP；
        - 返回 None 表示本轮无可处理帧（无新帧 / HOLD / DROP）；
        - 返回 SyncMatch 时，image/color_stamp 为待处理帧，
          consumed_latest=True 表示节点应清掉 latest 彩色图。
        """
        now_mono = time.perf_counter()

        # 如果上一帧暂时没等到同步点云，先让 pending 帧继续匹配，
        # 而不是把这一帧直接丢掉，从而减少 RViz 中偶发的一拍冻结。
        if self._pending_color is not None and now_mono > self._pending_color_deadline:
            self._logger.debug(
                f"[sync] pending color {stamp_text(self._pending_color_stamp)} "
                "超时，丢弃该帧")
            self._pending_color = None
            self._pending_color_stamp = None

        image = latest_color
        color_stamp = latest_color_stamp

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
            consumed_latest = False
        elif image is not None:
            # 有更新的彩色图，且没有更早的 pending 帧，直接处理新帧。
            self._pending_color = None
            self._pending_color_stamp = None
            consumed_latest = True
        else:
            return None

        # 点云时间同步：没有容差内匹配时，宁可本帧不裁点云，
        # 也不要把旧 mask 套到新点云上。
        sync = self.match(color_stamp, fallback_frame)

        self.log(color_stamp, sync.cloud_stamp, sync.delta_ns, sync.matched,
                 len(self))

        if sync.matched:
            self._sync_ok_count += 1
        else:
            self._sync_skip_count += 1

        delta_text = ("None" if sync.delta_ns is None
                      else f"{sync.delta_ns / 1e6:.2f}ms")

        # 没有同步点云时默认等待一小段时间，等待对应点云回调到达。
        # 超过 pending_max_wait 后仍不匹配，再丢弃该帧，不无限等待。
        if not sync.matched and self._require_synced_cloud:
            if self._pending_max_wait > 0.0:
                if not use_pending:
                    self._pending_color = image
                    self._pending_color_stamp = color_stamp
                    self._pending_color_deadline = now_mono + self._pending_max_wait
                self._hold_count += 1
                if len(self) == 0:
                    self._logger.debug(
                        f"[sync] WAIT_CLOUD color={stamp_text(color_stamp)} "
                        f"cloud_buffer=0 cloud_msgs={self.msgs} "
                        f"wait={self._pending_max_wait:.3f}s")
                else:
                    self._logger.debug(
                        f"[sync] HOLD color={stamp_text(color_stamp)} "
                        f"cloud={stamp_text(sync.cloud_stamp)} "
                        f"delta={delta_text} buffer={len(self)} "
                        f"cloud_msgs={self.msgs}")
                return None

            self._logger.debug(
                f"[sync] DROP color={stamp_text(color_stamp)} "
                f"cloud={stamp_text(sync.cloud_stamp)} "
                f"delta={delta_text} buffer={len(self)} "
                f"cloud_msgs={self.msgs}")

        # 只有真正采用匹配点云时，才更新“最后一次裁剪使用的点云时间戳”。
        if sync.matched and sync.cloud_stamp is not None:
            self._last_crop_stamp = sync.cloud_stamp
            self._last_crop_frame = sync.cloud_frame
        elif not sync.matched:
            self._last_crop_stamp = None
            self._last_crop_frame = fallback_frame

        # 成功进入推理时，清除 pending 帧，避免下一轮重复处理。
        self._pending_color = None
        self._pending_color_stamp = None

        if not sync.matched and self._require_synced_cloud:
            sync.xyz = None
            sync.rgb = None

        sync.image = image
        sync.color_stamp = color_stamp
        sync.consumed_latest = consumed_latest
        return sync

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
