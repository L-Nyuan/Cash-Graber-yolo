# AGENTS.md — Cash-Graber-yolo

本文件是 YOLO 视觉模块的自动上下文文件。新加入的 Agent 应先读完本文，再动手改代码或排查问题。

## 0. 项目速览

- 项目名称：Cash-Graber-yolo
- 上级任务：2026 埃斯顿机器人抓取比赛，视觉感知部分
- 本仓库职责：从 RealSense 相机图像出发，用 YOLO11-seg 做实例分割，追踪目标，并向下游 GraspNet 输出按需请求的物体彩色点云
- 当前状态：视觉模型已训练完成；推理基本链路已打通；仍在修复若干小 bug 和整理脚本
- 上游官方基础文档目录：`/root/ros2_ws/埃斯顿比赛`
- 本仓库路径：`/root/yolo`

## 1. 整体数据流

```text
RealSense D435i
  ├─ RGB 图 / color/image_raw
  ├─ 深度图 / aligned_depth_to_color/image_raw（旧版节点用）
  ├─ CameraInfo / color/camera_info
  └─ 彩色点云 / depth/color/points（cloud 版节点用）
              │
              ▼
YOLO11-seg 推理
  ├─ mask / bbox / class / confidence
  ├─ IoU 多目标追踪 → track_id
  └─ 物体点云裁剪或反投影
              │
              ▼
发布：
  /yolo/detections              std_msgs/String     每帧 JSON 检测元数据
  /yolo/object_cloud            sensor_msgs/PointCloud2  按请求返回目标点云
  /yolo/debug_cloud             sensor_msgs/PointCloud2  调试用
  /yolo/markers                 visualization_msgs/MarkerArray  RViz 调试

订阅：
  /yolo/request_object_cloud    std_msgs/Int32      下游请求某个 track_id
```

下游决策系统先看 `/yolo/detections` 选目标，再向 `/yolo/request_object_cloud` 发对应 `track_id`，本节点从缓存取点云发布到 `/yolo/object_cloud`。检测 JSON 本身不包含点云。

## 2. 环境与运行

### 2.1 环境

```bash
conda activate yolo
source /opt/ros/humble/setup.bash
```

如果 `python` 命令不存在，先确认已激活 `yolo` conda 环境；代码目前按 `python` 调用。

本项目设计为“零编译”：只使用 ROS2 标准消息类型，不需要 `colcon build`。

### 2.2 启动 RealSense

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

需要日志时，启动方式为
```bash
ros2 launch realsense2_camera rs_launch.py     pointcloud.enable:=true     align_depth.enable:=true     enable_color:=true     enable_depth:=true     rgb_camera.color_profile:=640x480x30     depth_module.depth_profile:=640x480x30     camera_name:=d435i     camera_namespace:="Wrist_Camera"     log_level:=debug     output:=log
```

### 2.3 启动推理节点

本地 import 依赖 `scripts/inference` 下的模块，所以要在该目录运行：

```bash
cd /root/yolo/scripts/inference

# 当前主要调试版本：RealSense 彩色点云直取 + 时间同步
python yolo_inference_node_cloud.py \
  --ros-args -p mode:=debug -p sync_debug:=true -p publish_debug_cloud:=true

# 早期版本：depth + 反投影
python yolo_inference_node.py --ros-args -p mode:=debug
```

### 2.4 调试点云请求

```bash
cd /root/yolo/scripts
python request_point_debug.py \
  --ros-args -p track_id:=1 -p save_cloud:=true -p save_format:=ply
```
这里的trck_id表示识别到物体的编号
## 3. 目录地图

```text
/root/yolo
├── AGENTS.md                       本文件
├── README.md                       旧版项目说明与单帧推理示例
├── dataset_real_remapped/          真实数据 YOLO 数据集（train/val + data.yaml）
├── dataset_temp/                   少量 CEPB 示例数据
├── result/
│   ├── final/                      最终部署权重：best.pt、last.pt、各 epoch
│   └── freeze/                     阶段 2 冻结微调权重
├── runs/                           推理/验证输出图片
├── rgb_test/                       相机 RGB 通道历史排查图片
├── test_output/                    CEPB 标注转换测试输出
└── scripts/
    ├── inference/
    │   ├── yolo_inference_node_cloud.py   当前主要节点：彩色点云直取+同步
    │   ├── yolo_inference_node.py         旧版节点：深度反投影
    │   ├── yolo_inference.py              YOLO 推理封装
    │   ├── object_tracker.py              IoU 多目标追踪
    │   ├── image_utils.py                 不依赖 cv_bridge 的 ROS Image 解码
    │   ├── marker_rviz.py                 RViz marker
    │   ├── mask_point_msg.py              旧版点云消息工具
    │   ├── visualization_utils.py         调试图保存
    │   └── test_ros2_caramer.py           相机通信测试脚本
    ├── request_point_debug.py             点云请求/保存调试脚本
    ├── yolo_train.py                      第一阶段 CEPB 训练脚本
    ├── train_freeze.py                    第二阶段冻结 backbone 微调脚本
    ├── train_final.py                     第三阶段解冻精细微调脚本
    ├── CEPB/                               CEPB 分割图转 YOLO 标签工具链
    └── Real_rec/                           真实数据采集与标签重映射工具链
```

## 4. 数据与模型

### 4.1 类别

真实数据集配置文件：`/root/yolo/dataset_real_remapped/data.yaml`

```text
0 Cheez-it
1 Starkist_Tuna
2 Scissors
3 Frenchs_Mustard
4 Tomato_Soup
5 Foam_Brick
6 Clamp
7 Plastic_Banana
8 Mug
9 meat_can
```

### 4.2 训练策略

采用三阶段 Sim-to-Real：

1. CEPB 合成数据预训练，学习物体基础形状与姿态。
2. 冻结 backbone，用少量真实数据 + CEPB 锚点混合微调，避免过拟合。
3. 解冻全模型，用极低学习率做最后域适应。

当前部署权重：

- 最终部署：`/root/yolo/result/final/best.pt`
- 最终 last：`/root/yolo/result/final/last.pt`
- 阶段 2 权重：`/root/yolo/result/freeze/best.pt`、`/root/yolo/result/freeze/last.pt`

## 5. 当前推理实现

### 5.1 `yolo_inference_node_cloud.py`（当前主要版本）

特点：

- 直接从 RealSense 的 `/depth/color/points` 彩色点云裁剪，而不是自己反投影深度。
- 用时间戳在彩色图和点云之间做最近邻匹配，超过 `sync_tolerance` 时默认不裁剪，避免旧 mask 套新点云。
- 有 `cloud_buffer_size` 缓冲、`pending_max_wait` 短等待、离群点移除和 mask 腐蚀。
- 点云缓存按 `track_id` 维护；未同步时清空旧缓存。

关键参数可在运行时用 `--ros-args -p name:=value` 覆盖。

### 5.2 `yolo_inference_node.py`（旧版）

订阅对齐深度图，用 `mask_to_pointcloud()` 反投影生成点云。功能可用，但当前调试重点已经转向 cloud 版；二者维护时要避免重复修同一处逻辑。


## 6. 重要约定与易错点

- **不要引入 `cv_bridge`**：当前环境 NumPy 版本与 cv_bridge 存在冲突，手动解码是有意为之。
- **不要擅自改 QoS**：相机类订阅与 RealSense 发布端必须兼容；cloud 节点中点云订阅使用 `BEST_EFFORT`，其余检测/点云发布使用 `RELIABLE`。
- **相机话题带命名空间**：当前默认是 `/Wrist_Camera/d435i/...`，不是无前缀的 `/d435i/...`；cloud 节点已参数化，改话题用 `cloud_topic`、`color_topic`、`info_topic`。
- **YOLO mask 分辨率必须与点云/相机内参对齐**：cloud 版会先 `ensure_mask_resolution()` 再投影裁剪；旧版会打印形状不匹配错误。
- **单线程执行器**：不要在回调里做长阻塞操作；节点已改成 50ms 定时器驱动，`_latest_*` 缓存并配合同步逻辑。
- **按需点云协议**：`/yolo/detections` 只给元数据，下游必须用 `/yolo/request_object_cloud` 发 `Int32(track_id)` 再收 `/yolo/object_cloud`。

## 7. 参考文档

优先阅读：

- `/root/ros2_ws/埃斯顿比赛/yolo/项目介绍.md`
- `/root/ros2_ws/埃斯顿比赛/yolo/知识总结.md`
- `/root/ros2_ws/埃斯顿比赛/yolo/YOLO11-seg-ROS2推理节点指导.md`
- `/root/ros2_ws/埃斯顿比赛/yolo/YOLO11-seg实例分割训练指南.md`
- `/root/ros2_ws/埃斯顿比赛/yolo/YOLO检测到点云提取-ROS2发布指南.md`
- `/root/ros2_ws/埃斯顿比赛/yolo/YOLO分割到GraspNet点云输入流程.md`
- `/root/ros2_ws/埃斯顿比赛/yolo/三阶段训练策略-CEPB到真实数据迁移.md`

这些文档是设计和调参背景，本 `AGENTS.md` 是执行时的快速入口。
