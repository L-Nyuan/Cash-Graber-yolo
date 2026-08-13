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

再启动推理节点（建议先激活 yolo 虚拟环境）。输出内容由 `mode` 统一控制：

| 模式 | 输出 |
| --- | --- |
| `production`（默认） | 只输出识别标签 `/yolo/detections` 和按需点云 `/yolo/object_cloud`，无调试发布 |
| `debug` | 在 production 基础上，额外发布所有识别物品的点云 `/yolo/debug_cloud`、RViz 标记 `/yolo/markers`，并默认打开同步/裁剪调试日志 |

生产模式：

```bash
python scripts/inference/yolo_inference_node_cloud.py --ros-args
```

调试模式：

```bash
python scripts/inference/yolo_inference_node_cloud.py --ros-args \
  -p mode:=debug \
  --log-level yolo_inference_node:=debug
```

调试模式下 `/yolo/detections` 的 JSON 会额外带每个物品的 `cloud_points`（裁剪点数）和 `cloud_centroid`（点云质心），方便核对“所有物品的点云”是否裁到。

调参示例（参数按前缀分组，写法统一为 `-p 组名.参数:=值`）：

```bash
# 手腕相机移动快时收紧同步容差
python scripts/inference/yolo_inference_node_cloud.py --ros-args -p sync.tolerance:=0.03

# 降低置信度阈值 / 切换模型
python scripts/inference/yolo_inference_node_cloud.py --ros-args \
  -p model.conf:=0.5 -p model.path:=/root/yolo/result/final/last.pt
```

### 命令行参数（`-p 组名.参数:=值`）

| 分组 | 参数 | 默认值 | 说明 |
| --- | --- | --- | --- |
| — | `mode` | `production` | `production` / `debug`，统一决定输出内容 |
| `model` | `.path` | `/root/yolo/result/final/best.pt` | YOLO 权重路径 |
| `model` | `.imgsz` | `640` | 推理输入尺寸（边长） |
| `model` | `.conf` | `0.8` | 置信度阈值 |
| `model` | `.iou` | `0.7` | NMS 的 IoU 阈值 |
| `tracker` | `.max_age` | `30` | 目标消失超过该帧数则丢弃 ID |
| `tracker` | `.min_hits` | `3` | 新目标需连续命中多少帧才确认（避免闪烁 ID） |
| `tracker` | `.iou` | `0.3` | 追踪关联时判断“同一目标”的 IoU 阈值 |
| `cloud` | `.mask_erode` | `1` | 分割 mask 腐蚀迭代次数，剔除目标边缘噪声点 |
| `cloud` | `.sor` | `true` | 对裁剪点云做统计离群点移除（需 open3d，缺失自动跳过） |
| `sync` | `.tolerance` | `0.05` | 彩色图与点云时间戳匹配容差（秒） |
| `sync` | `.require_cloud` | `true` | true 时只裁剪同步点云，找不到则跳过本帧；false 时退回最近一帧 |
| `sync` | `.buffer_size` | `30` | 点云时间戳缓存最大帧数 |
| `sync` | `.pending_wait` | `0.12` | 彩色图最多等待点云多少秒，超时丢弃该帧 |
| `sync` | `.debug` | `false` | 同步/裁剪 DEBUG 日志；debug 模式下默认开启 |
| `topic` | `.cloud` | `/Wrist_Camera/d435i/depth/color/points` | 彩色点云话题 |
| `topic` | `.color` | `/Wrist_Camera/d435i/color/image_raw` | 彩色图话题 |
| `topic` | `.info` | `/Wrist_Camera/d435i/color/camera_info` | 相机内参话题（投影裁剪用） |
| `debug` | `.cloud_hz` | `15.0` | `/yolo/debug_cloud` 最大发布频率（Hz），仅 debug 模式生效 |
| `debug` | `.dir` | `/root/yolo/yolo_debug` | 调试输出目录，仅 debug 模式生效 |

### 话题与 QoS

订阅：

| 话题 | 类型 |
| --- | --- |
| `topic.cloud` | `sensor_msgs/msg/PointCloud2`（RELIABLE，与 RealSense 发布端一致） |
| `topic.color` | `sensor_msgs/msg/Image` |
| `topic.info` | `sensor_msgs/msg/CameraInfo` |
| `/yolo/request_object_cloud` | `std_msgs/msg/Int32`（请求指定 track_id 的最新目标点云） |

发布（始终）：

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| `/yolo/detections` | `std_msgs/msg/String` | 识别标签 JSON（id、类别、置信度、中心点、bbox；debug 模式附加 `cloud_points`/`cloud_centroid`） |
| `/yolo/object_cloud` | `sensor_msgs/msg/PointCloud2` | 按请求返回的单个目标点云 |

发布（仅 debug 模式）：

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| `/yolo/debug_cloud` | `sensor_msgs/msg/PointCloud2` | 本帧所有识别物品的裁剪点云（合并） |
| `/yolo/markers` | `visualization_msgs/msg/MarkerArray` | RViz 检测框标记 |

注意：点云订阅使用 RELIABLE / KEEP_LAST(10)，与 RealSense 发布端 QoS 保持一致。若改成 BEST_EFFORT，在部分 RMW（如 CycloneDDS）下会导致订阅匹配不上、收不到点云。这是此前排查过的一个坑，改 QoS 前先确认发布端配置。
