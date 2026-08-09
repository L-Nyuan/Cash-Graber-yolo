"""
IoU-based multi-object tracker for YOLO detections.

跨帧追踪：为每个检测分配稳定 ID，供决策系统按 ID 请求点云。
"""
import numpy as np
from typing import List, Dict, Tuple, Optional


class Track:
    """单个追踪目标。"""

    __slots__ = ("id", "bbox", "class_name", "confidence",
                 "age", "hits", "time_since_update", "det_idx")

    def __init__(self, track_id: int, bbox: np.ndarray,
                 class_name: str, confidence: float):
        self.id = track_id
        self.bbox = bbox.astype(np.float32).copy()  # [x1, y1, x2, y2]
        self.class_name = class_name
        self.confidence = confidence
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.det_idx = -1          # 当前帧匹配到的 detection 索引，-1 表示未匹配


class ObjectTracker:
    """简单 IoU 匹配多目标追踪器。

    每帧用 IoU 贪心匹配已有 track 和新检测；
    新检测创建 track（需 min_hits 帧连续命中才算"确认"）；
    超过 max_age 帧未匹配的 track 被删除。
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3,
                 iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._tracks: List[Track] = []
        self._next_id = 0

    def update(self, detections: List[Dict]) -> List[Track]:
        """
        Args:
            detections: [{"bbox": [x1,y1,x2,y2], "class_name": str, "confidence": float}, ...]

        Returns:
            已确认的活跃 track 列表（hits >= min_hits）。
            每个 track.det_idx 指向当前帧 detections 中的索引（-1 = 未匹配）。
        """
        # 1. 预测：所有 track 年龄 +1，det_idx 重置
        for t in self._tracks:
            t.age += 1
            t.time_since_update += 1
            t.det_idx = -1

        # 2. IoU 匹配
        matches, unmatched_dets, _unmatched_tracks = self._match(detections)

        # 3. 更新匹配到的 track
        for t_idx, d_idx in matches:
            det = detections[d_idx]
            t = self._tracks[t_idx]
            t.bbox = np.array(det["bbox"], dtype=np.float32)
            t.class_name = det["class_name"]
            t.confidence = det["confidence"]
            t.hits += 1
            t.time_since_update = 0
            t.det_idx = d_idx          # ← 记录匹配到了哪个 detection

        # 4. 未匹配的检测 → 新建 track
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            t = Track(self._next_id,
                      np.array(det["bbox"], dtype=np.float32),
                      det["class_name"], det["confidence"])
            t.det_idx = d_idx
            self._tracks.append(t)
            self._next_id += 1

        # 5. 删除长期未命中的 track
        self._tracks = [t for t in self._tracks
                        if t.time_since_update <= self.max_age]

        # 6. 返回已确认的 track（包含 det_idx 映射）
        return [t for t in self._tracks if t.hits >= self.min_hits]

    # ── 内部方法 ────────────────────────────────────────────

    def _match(self, detections: List[Dict]) -> Tuple[
        List[Tuple[int, int]], List[int], List[int]
    ]:
        """IoU 贪心匹配。返回 (matches, unmatched_dets, unmatched_tracks)。"""
        n_tracks = len(self._tracks)
        n_dets = len(detections)

        if n_tracks == 0:
            return [], list(range(n_dets)), []
        if n_dets == 0:
            return [], [], list(range(n_tracks))

        # 计算 IoU 矩阵
        iou_mat = np.zeros((n_tracks, n_dets), dtype=np.float32)
        for t in range(n_tracks):
            for d in range(n_dets):
                iou_mat[t, d] = self._iou(self._tracks[t].bbox,
                                          np.array(detections[d]["bbox"]))

        # 贪心匹配：按 IoU 降序
        flat_idx = np.argsort(-iou_mat.ravel())
        matched_t = set()
        matched_d = set()
        matches = []

        for idx in flat_idx:
            t, d = divmod(idx, n_dets)
            if iou_mat[t, d] < self.iou_threshold:
                break
            if t not in matched_t and d not in matched_d:
                matches.append((t, d))
                matched_t.add(t)
                matched_d.add(d)

        unmatched_t = [t for t in range(n_tracks) if t not in matched_t]
        unmatched_d = [d for d in range(n_dets) if d not in matched_d]

        return matches, unmatched_d, unmatched_t

    @staticmethod
    def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """两矩形框 IoU。box: [x1, y1, x2, y2]"""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0