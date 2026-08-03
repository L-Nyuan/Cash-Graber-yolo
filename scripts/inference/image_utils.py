import cv2
import numpy as np
from sensor_msgs.msg import Image

# encoding → (numpy dtype, channel 数量) 映射
# 用来处理收到的ros图像消息，转换为numpy数组

_ENCODING_MAP = {
    "rgb8":   (np.uint8,   3),
    "rgba8":  (np.uint8,   4),
    "bgr8":   (np.uint8,   3),
    "bgra8":  (np.uint8,   4),
    "mono8":  (np.uint8,   1),
    "8UC1":   (np.uint8,   1),
    "16UC1":  (np.uint16,  1),
    "32FC1":  (np.float32, 1),
}


def ros_image_to_numpy(msg: Image) -> np.ndarray:
    """ROS sensor_msgs/Image → numpy 数组，不依赖 cv_bridge(很变态，发现居然会冲突)。

    Args:
        msg: sensor_msgs/Image 消息
    Returns:
        shape (H, W) 或 (H, W, C) 的 numpy 数组 BGR格式，与yolo保持一致
    """
    if msg.encoding not in _ENCODING_MAP:
        raise ValueError(
            f"不支持的 encoding: {msg.encoding}, "
            f"已知: {list(_ENCODING_MAP.keys())}"
        )

    dtype, channels = _ENCODING_MAP[msg.encoding]
    data = np.frombuffer(msg.data, dtype=dtype)# 按照数据格式直接读缓存区内存，零拷贝赋值

    if channels == 1:
        return data.reshape((msg.height, msg.width))

    image = data.reshape((msg.height, msg.width, channels))

    # 转换为 BGR 顺序
    if msg.encoding == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif msg.encoding == "bgr8":
        pass  # 已经是 BGR
    else:
        raise ValueError(f"不支持的 encoding: {msg.encoding}")
    return image