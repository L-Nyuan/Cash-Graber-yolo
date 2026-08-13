# Cash-Graber-yolo
2026 埃斯顿机器人抓取比赛的视觉部分代码仓库：从 CEPB/真实数据集制作、YOLO11-seg 训练，到 ROS2 实时推理（检测 + 分割 + 彩色点云裁剪）。

## 工作区结构

```text
/root/yolo
├── AGENTS.md                       # 开发约定与易错点（AI/协作使用）
├── README.md                       # 本文件
├── scripts/
│   ├── CEPB/                       # CEPB 官方数据集 → YOLO 格式
│   ├── Real_rec/                   # 真实数据采集与处理
│   ├── inference/                  # ROS2 推理节点与配套工具
│   │   ├── yolo_inference_node_cloud.py   # 主入口节点（见下文）
│   │   ├── yolo_inference.py              # YOLO 推理封装
│   │   ├── object_tracker.py              # IoU 多目标追踪
│   │   ├── image_utils.py                 # 不依赖 cv_bridge 的 ROS Image 解码
│   │   ├── marker_rviz.py                 # RViz 标记发布
│   │   ├── visualization_utils.py         # 调试图保存
│   │   └── mask_point_msg.py              # 未使用的旧版点云消息工具
│   ├── request_point_debug.py     # 按需点云请求调试脚本
│   └── train/                     # 训练脚本
│       ├── yolo_train.py          # 阶段 1：CEPB 基线训练
│       ├── train_freeze.py        # 阶段 2：冻结 backbone 微调
│       └── train_final.py         # 阶段 3：解冻全模型微调
├── dataset_real_remapped/          # 真实 + CEPB 混合 YOLO 数据集（images/labels + data.yaml）
├── dataset_temp/                   # 少量 CEPB 示例数据
├── result/
│   ├── freeze/                     # 阶段 2 权重
│   └── final/                      # 阶段 3 最终权重（best.pt / last.pt）
├── runs/                           # ultralytics 训练日志与验证输出
├── test_output/                    # Seg_To_Txt 转换测试输出（images/labels）
├── debug_log/                      # 调试记录（如点云 QoS 问题排查）
├── yolo_debug/                     # 推理节点调试输出目录
└── rgb_test/                       # 相机 RGB 通道排查图片
```

## 数据集处理

### 1. CEPB 官方数据集 → YOLO 格式（`scripts/CEPB/`）

官方数据集中有很多多余的深度图像，转换前删掉即可。

| 脚本 | 作用 |
| --- | --- |
| `Seg_To_Txt.py` | 主转换：把 CEPB 官方 yaml GT + RGB/segmentation 图转成 YOLO 分割格式（images/labels/dataset.yaml），以 GT 2D 质心为初始中心做 KMeans 聚类生成 polygon。输出 `/root/dataset_seg/` |
| `clean_dataset.py` | 清理冗余：同一场景同一视角的多光源图（dir/point/spot）只保留一张，避免数据集重复 |
| `debug_centers.py` | 调试：可视化 KMeans 聚类中心颜色与量化后 seg 图，排查聚类错位 |
| `interactive_color_picker.py` | 手动取色：交互式点击 seg 图取 RGB 作为聚类中心（KMeans 效果不好时的替代方案） |
| `preview_letterbox.py` | 预览 YOLO 640×640 letterbox 效果，防止小物体被压缩丢失 |
| `visualize_labels.py` | 把 YOLO polygon 标签画回图像，检查标注是否正确 |
| `check_labels.py` | 从训练集每类随机抽 N 张绘制标签，检查标注错位 |

### 2. 真实数据集采集与处理（`scripts/Real_rec/`）

| 脚本 | 作用 |
| --- | --- |
| `record_realsense.py` | 用 RealSense 录制 RGB 视频为 H.264 MP4（可指定分辨率/时长） |
| `extract_frames.py` | 从视频按时间/帧数间隔抽帧，带模糊过滤 |
| `capture_click.py` | RealSense 实时预览，按键逐帧保存（命名 `MM_NNN.jpg`） |
| `remap_labels.py` | 把 Roboflow 导出的字母序 class id 重映射为目标数据集顺序 |
| `sample_cepb_to_real.py` | 从 CEPB 按类别均匀采样 N 张复制进真实数据集（加 `cepb_` 前缀防重名） |

真实数据流程：采集视频 → 抽帧/按键存图 → 标注清洗 → `remap_labels` 对齐类别 → `sample_cepb_to_real` 混入 CEPB → 训练。

### 3. 数据集处理正确性检查

下面这个命令用训练好的模型对一张验证图跑推理并保存可视化，用来确认「数据集 → 训练 → 模型」整条链路是否正常，也常用来检查标注/分割结果是否正确：

```python
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
print('可视化已保存: /root/yolo/debug_test4501.jpg')
"
```

## 训练（`scripts/train/`）

三阶段策略：

| 脚本 | 阶段 | 说明 |
| --- | --- | --- |
| `scripts/train/yolo_train.py` | 1 | CEPB 数据上的 yolo11m-seg 基线训练 |
| `scripts/train/train_freeze.py` | 2 | 冻结 backbone，在混合数据（真实 + CEPB）上微调 |
| `scripts/train/train_final.py` | 3 | 解冻全模型、低学习率精细微调，产出 `result/final/` |

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
| `debug` | 在 production 基础上，额外发布所有识别物品的点云 `/yolo/debug_cloud`、RViz 标记 `/yolo/markers`，默认打开同步/裁剪调试日志，并自动启动 rviz2 窗口（加载 `debug.rviz_config`） |

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

debug 模式自动打开的 rviz2 会把输出写到 `debug.dir`（默认 `yolo_debug/rviz2.log`）；如果 rviz2 启动后异常退出（如显示连接失败），节点日志会直接打出它的报错末尾，便于排查窗口未出现的问题。

注意：在 conda 环境下 `import cv2` 会把 `QT_QPA_PLATFORM_PLUGIN_PATH` 指向 cv2 自带的 Qt 插件，导致 rviz2 加载到不兼容的 xcb 插件、启动即崩溃（rc=-6）。节点拉起 rviz2 前会自动把该变量指回系统 Qt 插件目录（`/usr/lib/x86_64-linux-gnu/qt5/plugins`）并清理 LD_LIBRARY_PATH 中的 conda 路径。

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
| `debug` | `.launch_rviz` | `true` | debug 模式下自动启动 rviz2 窗口；无显示器环境可设为 false |
| `debug` | `.rviz_config` | `/root/yolo/rviz/debug.rviz` | rviz2 启动时加载的配置文件 |

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
