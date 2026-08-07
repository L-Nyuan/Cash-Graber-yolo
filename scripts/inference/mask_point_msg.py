from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np

def _build_pointcloud2(self, points, header, frame_id=""):
    """numpy (N,3) 点云 → PointCloud2 消息。

    Args:
        points:    np.ndarray (N,3) float32，[X, Y, Z] 米
        header:    std_msgs/Header，使用其 stamp
        frame_id:  坐标系，如 "Wrist_Camera_color_optical_frame"

    Returns:
        sensor_msgs/PointCloud2
    """
    msg = PointCloud2()
    msg.header = header
    if frame_id:
        msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points.shape[0]
    msg.is_bigendian = False
    msg.point_step = 16              # 4 fields × 4 bytes = 16
    msg.row_step = msg.point_step * msg.width

    msg.fields = [
        PointField(name="x",  offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name="y",  offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name="z",  offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    # 压入数据：每点 [X, Y, Z, intensity]
    buf = np.zeros(points.shape[0], dtype=[
        ("x", np.float32), ("y", np.float32),
        ("z", np.float32), ("intensity", np.float32)
    ])
    buf["x"] = points[:, 0]
    buf["y"] = points[:, 1]
    buf["z"] = points[:, 2]

    msg.data = buf.tobytes()
    return msg