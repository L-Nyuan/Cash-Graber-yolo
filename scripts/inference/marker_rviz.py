from visualization_msgs.msg import Marker, MarkerArray
import numpy as np

def _publish_markers(objects, header, publisher):
    """为每个检测物体发一个 3D 包围框 marker。"""
    markers = MarkerArray()
    colors = [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),
              (0,1,1),(1,0.5,0),(0.5,0,1),(0,1,0.5)]

    for i, obj in enumerate(objects):
        if "cloud" not in obj:
            continue
        cloud = obj["cloud"]
        if cloud.shape[0] < 10:
            continue

        xyz = np.mean(cloud, axis=0)   # 点云质心作为 marker 位置

        marker = Marker()
        marker.header = header
        marker.ns = "yolo_objects"
        marker.id = i
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = float(xyz[0])
        marker.pose.position.y = float(xyz[1])
        marker.pose.position.z = float(xyz[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        c = colors[i % len(colors)]
        marker.color.r = c[0]
        marker.color.g = c[1]
        marker.color.b = c[2]
        marker.color.a = 0.8

        marker.lifetime.sec = 1   # 1 秒后消失，等待下一帧刷新

        marker_text = Marker()
        marker_text.header = header
        marker_text.ns = "yolo_labels"
        marker_text.id = i
        marker_text.type = Marker.TEXT_VIEW_FACING
        marker_text.action = Marker.ADD

        marker_text.pose.position.x = float(xyz[0])
        marker_text.pose.position.y = float(xyz[1])
        marker_text.pose.position.z = float(xyz[2]) + 0.05
        marker_text.text = f"{obj['class_name']} {obj['confidence']:.2f}"
        marker_text.scale.z = 0.03
        marker_text.color.a = 1.0
        marker_text.color.r = 1.0
        marker_text.color.g = 1.0
        marker_text.color.b = 1.0
        marker_text.lifetime.sec = 1

        markers.markers.append(marker)
        markers.markers.append(marker_text)

    publisher.publish(markers)