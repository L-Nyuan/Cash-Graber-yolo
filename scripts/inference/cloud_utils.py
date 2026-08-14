# cloud_utils.py —— YOLO 节点纯函数工具
#
# 点云编解码、mask 缩放/腐蚀、点云裁剪、SOR、时间戳等无状态纯函数。
# 不依赖 rclpy Node，可离线单测（见 test_cloud_utils.py）。
#
# 注意：cv2 保持函数内延迟导入（避免 import 时污染 Qt 插件环境，
# 见 rviz_launcher 相关的环境修复说明）。
from typing import Tuple

import numpy as np
from sensor_msgs.msg import CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header


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


# ============================================================
# 时间戳与点云后处理小工具
# ============================================================

def stamp_ns(s) -> int:
    """ROS 时间戳 → 纳秒整数。"""
    return int(s.sec) * 1_000_000_000 + int(s.nanosec)


def stamp_text(s) -> str:
    """把 ROS 时间转成 sec.nanosec 字符串，方便读日志。"""
    if s is None:
        return "None"
    return f"{int(s.sec)}.{int(s.nanosec):09d}"


def erode_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    """mask 腐蚀（3x3 核），返回 bool 掩码。"""
    import cv2
    return cv2.erode(
        mask.astype(np.uint8),
        np.ones((3, 3), np.uint8),
        iterations=iterations) > 0


def remove_outliers_sor(cloud: np.ndarray, o3d,
                        nb_neighbors: int = 20,
                        std_ratio: float = 1.0) -> np.ndarray:
    """open3d 统计离群点移除，返回过滤后的 (N,6) 点云。"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
    _, idx = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return cloud[np.asarray(idx)]
