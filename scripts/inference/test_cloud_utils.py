# test_cloud_utils.py —— 点云工具离线冒烟测试
#
# Phase 0 创建：验证 build_pointcloud2 → realsense_cloud_to_xyzrgb 往返一致。
# Phase 2 之后 cloud_utils.py 存在，自动切换从新模块导入；此前从节点模块导入。
import numpy as np

try:
    from cloud_utils import build_pointcloud2, realsense_cloud_to_xyzrgb
except ImportError:
    from yolo_inference_node_cloud import (
        build_pointcloud2, realsense_cloud_to_xyzrgb,
    )


def _make_header():
    from std_msgs.msg import Header

    h = Header()
    h.frame_id = "test_cloud"
    h.stamp.sec = 123
    h.stamp.nanosec = 456
    return h


def test_roundtrip():
    n = 100
    rng = np.random.default_rng(42)
    xyz = rng.uniform(-0.5, 0.5, (n, 3)).astype(np.float32)
    xyz[:, 2] = np.abs(xyz[:, 2]) + 0.3          # z > 0.1，保持有效
    rgb = rng.integers(0, 256, (n, 3)).astype(np.float32)
    xyz_rgb = np.concatenate([xyz, rgb], axis=1).astype(np.float32)

    msg = build_pointcloud2(xyz_rgb, _make_header())
    xyz2, rgb2 = realsense_cloud_to_xyzrgb(msg)

    assert xyz2.shape == (n, 3), f"xyz shape {xyz2.shape} != {(n, 3)}"
    assert rgb2.shape == (n, 3), f"rgb shape {rgb2.shape} != {(n, 3)}"
    assert np.allclose(xyz2, xyz, atol=1e-4), "xyz 不一致"
    assert np.allclose(rgb2, rgb, atol=0.51), "rgb 不一致"
    print(f"[OK] 往返测试: {n} 点 xyz/rgb 一致")


def test_empty():
    empty = np.empty((0, 6), dtype=np.float32)
    msg = build_pointcloud2(empty, _make_header())
    xyz, rgb = realsense_cloud_to_xyzrgb(msg)
    assert xyz.shape[0] == 0 and rgb.shape[0] == 0
    print("[OK] 空点云测试: 0 点")


if __name__ == "__main__":
    test_roundtrip()
    test_empty()
    print("test_cloud_utils.py 全部通过")
