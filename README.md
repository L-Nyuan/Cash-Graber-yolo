# Cash-Graber-yolo
2026埃斯顿机器人抓取比赛的视觉部分代码仓库

`clean_dataset`是对不同视角和不同光照条件数据集的清理，因为在训练过程中发现数据集有冗余，遂添加

`debug_centers`现在的`Seg_To_Txt`的K-means聚类存在问题，所以添加该调试模块，显示经过颜色压缩后的聚类的中心

`interacitve_color_picker`因为K-means聚类的问题和数据集关系比较大，懒得解决，所以直接手动取颜色作为聚类中心

`preview_letterbox`防止yolo网络内部对图像压缩太狠（目前是640*640），导致无法看到小物品，所以提前预览

`visualize_labels`绘制txt数据集和中包围点segmentation，检查制作成果

`Seg_To_Txt`从官方数据集到Yolo可用数据格式（文件格式需要另外调整）

`yolo_train`训练主函数


官方数据集中有很多多余的深度图像，删掉就行了

python -c "
import cv2, sys
sys.path.insert(0, '/root/yolo/scripts/inference')
from yolo_inference import YOLOSegInference
from visualization_utils import save_debug_image

yolo = YOLOSegInference('/root/yolo/result/final/last.pt')

img_bgr = cv2.imread('/root/ros2_ws/dataset_real/images/val/37_010_jpg.rf.NiwQfaRabHYHhxKNzA3m.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
print(f'图像尺寸: {img_rgb.shape}')

r = yolo.predict(img_rgb)
print(f\"推理 {r['inference_time_ms']:.1f}ms, 检测到 {len(r['objects'])} 个物体\")
for o in r['objects']:
    print(f\"  {o['class_name']:20s} conf={o['confidence']:.2f}  pixels={o['mask'].sum()}\")

save_debug_image(img_rgb, r['objects'], '/root/yolo/debug_test4501.jpg')
print('可视化已保存: /root/yolo/debug_test.jpg')
"

python yolo_inference_node.py --ros-args -p mode:=debug


## 主推理节点：yolo_inference_node_cloud.py

实机（RealSense D435i + 机械臂腕部相机）的主入口节点，功能是：订阅彩色图和彩色点云，运行 YOLO11-seg 检测与分割，用 mask 裁剪出每个目标的点云，并用 IoU 追踪器稳定目标 ID。位置：

`scripts/inference/yolo_inference_node_cloud.py`

### 运行方式

先启动相机（示例）：

```bash
ros2 launch realsense2_camera rs_launch.py \
  pointcloud.enable:=true \
  align_depth.enable:=true \
  enable_color:=true \
  enable_depth:=true \
  rgb_camera.color_profile:=640x480x30 \
  depth_module.depth_profile:=640x480x30 \
  camera_name:=d435i \
  camera_namespace:="Wrist_Camera"
```

再启动推理节点（建议先激活 yolo 虚拟环境）：

```bash
python scripts/inference/yolo_inference_node_cloud.py --ros-args -p mode:=production
```

调试模式（打开同步调试日志、发布调试点云）：

```bash
python scripts/inference/yolo_inference_node_cloud.py --ros-args \
  -p mode:=debug \
  -p publish_debug_cloud:=true \
  -p sync_debug:=true \
  --log-level yolo_inference_node:=debug
```

如果相机时间戳存在固定偏差、彩色图和点云匹配不上，可以放宽/收紧同步容差：

```bash
python scripts/inference/yolo_inference_node_cloud.py --ros-args -p sync_tolerance:=0.08
```

### 命令行参数（`-p 参数名:=值`）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `model_path` | `/root/yolo/result/final/best.pt` | YOLO 权重路径 |
| `imgsz` | `640` | 推理输入尺寸（边长） |
| `conf` | `0.8` | 置信度阈值 |
| `iou` | `0.7` | NMS 的 IoU 阈值 |
| `mode` | `production` | `production` / `debug`；debug 模式会无条件发布调试点云和 RViz 标记，并输出更多日志 |
| `publish_debug_cloud` | `false` | 发布裁剪后目标点云到 `/yolo/debug_cloud` |
| `publish_markers` | `false` | 发布检测框标记到 `/yolo/markers`（RViz 用） |
| `debug` | `false` | 调试开关；为 true 且环境有可视化依赖时创建 `debug_dir` 目录（当前版本仅建目录，图片保存逻辑尚未接入） |
| `debug_cloud_hz` | `15.0` | `/yolo/debug_cloud` 的最大发布频率（Hz） |
| `debug_dir` | `/root/yolo/yolo_debug` | 调试图片保存目录 |
| `tracker_max_age` | `30` | 追踪器允许目标消失的最大帧数，超过则丢弃该 ID |
| `tracker_min_hits` | `3` | 新目标需要连续命中多少帧才确认（避免闪烁 ID） |
| `tracker_iou_threshold` | `0.3` | 追踪关联时判断“同一目标”的 IoU 阈值 |
| `mask_erode` | `1` | 分割 mask 腐蚀迭代次数，用于剔除目标边缘的噪声点云点 |
| `cloud_sor` | `true` | 对裁剪出的点云做统计离群点移除（需要 open3d；缺失时自动跳过） |
| `sync_tolerance` | `0.05` | 彩色图与点云时间戳匹配容差（秒） |
| `require_synced_cloud` | `true` | 为 true 时只裁剪与彩色图同步的点云，找不到就跳过本帧；为 false 时退回使用最近一帧点云 |
| `sync_debug` | `true` | 输出同步/裁剪过程的 DEBUG 级日志（需配合 `--log-level` 使用） |
| `cloud_buffer_size` | `30` | 点云时间戳缓存的最大帧数，用于与彩色图匹配 |
| `pending_max_wait` | `0.12` | 彩色图最多等待点云多少秒，超时丢弃该帧，避免死等 |
| `cloud_topic` | `/Wrist_Camera/d435i/depth/color/points` | 彩色点云话题 |
| `color_topic` | `/Wrist_Camera/d435i/color/image_raw` | 彩色图话题 |
| `info_topic` | `/Wrist_Camera/d435i/color/camera_info` | 相机内参话题（点云投影裁剪用） |

### 话题与 QoS

订阅：

| 话题 | 类型 |
| --- | --- |
| `cloud_topic` | `sensor_msgs/msg/PointCloud2`（RELIABLE，与 RealSense 发布端一致） |
| `color_topic` | `sensor_msgs/msg/Image` |
| `info_topic` | `sensor_msgs/msg/CameraInfo` |
| `/yolo/request_object_cloud` | `std_msgs/msg/Int32`（请求指定 track_id 的最新目标点云） |

发布：

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| `/yolo/detections` | `std_msgs/msg/String` | 检测结果 JSON（id、类别、置信度、中心点、bbox） |
| `/yolo/object_cloud` | `sensor_msgs/msg/PointCloud2` | 请求到的目标点云 |
| `/yolo/debug_cloud` | `sensor_msgs/msg/PointCloud2` | 调试点云（`publish_debug_cloud:=true` 或 debug 模式） |
| `/yolo/markers` | `visualization_msgs/msg/MarkerArray` | RViz 标记（`publish_markers:=true` 或 debug 模式） |

注意：点云订阅使用 RELIABLE / KEEP_LAST(10)，与 RealSense 发布端 QoS 保持一致。若改成 BEST_EFFORT，在部分 RMW（如 CycloneDDS）下会导致订阅匹配不上、收不到点云。这是此前排查过的一个坑，改 QoS 前先确认发布端配置。
