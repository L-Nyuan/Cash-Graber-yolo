# YOLO 点云订阅 QoS 不匹配问题调试记录

## 日期

2026-08-13

## 问题现象

在相机启动但 RViz 关闭，或 RViz 打开但没有订阅点云话题时，YOLO 推理节点出现异常：

1. 早期版本在首个点云未到达时会因 `None:.2f` 格式化错误直接崩溃。
2. 修复格式化错误并增加等待逻辑后，节点不再崩溃，但出现新问题：
   - 当相机初始不对准待识别物体时，点云能正常到达，推理正常。
   - 当相机初始对准待识别物体时，`cloud_msgs` 始终为 0，日志持续输出 `WAIT_CLOUD`，无法恢复。

### 关键日志

```text
[sync] WAIT_CLOUD color=... cloud_buffer=0 cloud_msgs=0 wait=0.120s
[sync] pending color ... 超时，丢弃该帧
```

YOLO 节点一直收不到 `/Wrist_Camera/d435i/depth/color/points`。

## 排查过程

### 1. 排除 YOLO 订阅发现失败

```bash
ros2 topic info /Wrist_Camera/d435i/depth/color/points -v
```

结果：

```text
Publisher count: 1
Subscription count: 1

Publisher QoS:
  Reliability: RELIABLE
  History (Depth): KEEP_LAST (10)
  Durability: VOLATILE

Subscriber:
  Node name: yolo_inference_node
  Reliability: BEST_EFFORT
  History (Depth): KEEP_LAST (5)
  Durability: VOLATILE
```

说明发布器和订阅器在 topic graph 中都已经存在。

### 2. 测试点云话题是否真正发布

使用 BEST_EFFORT 订阅点云：

```bash
ros2 topic echo /Wrist_Camera/d435i/depth/color/points \
  --qos-reliability best_effort \
  --qos-depth 5 \
  --qos-durability volatile \
  --no-arr --no-str --once --spin-time 15
```

结果：收不到点云。

使用 RELIABLE 订阅点云：

```bash
ros2 topic echo /Wrist_Camera/d435i/depth/color/points \
  --qos-reliability reliable \
  --qos-depth 10 \
  --qos-durability volatile \
  --no-arr --no-str --once --spin-time 15
```

结果：成功收到点云。

示例输出：

```text
height: 1
width: 117270
point_step: 20
row_step: 2345400
is_dense: true
```

### 3. RealSense 调试日志确认点云已生成

日志文件：

```text
/root/.ros/log/realsense2_camera_node_258410_1786628389466.log
```

日志显示：

```text
List of frameset before applying filters: size: 2
Frameset contain (Color, 0, RGB8)
Frameset contain (Depth, 0, Z16)

List of frameset after applying filters: size: 3
Frameset contain (Depth, 0, XYZ32F)
Frameset contain (Depth, 0, Z16)
Frameset contain (Color, 0, RGB8)
```

`XYZ32F` 表示 RealSense 内部已经生成点云。因此问题不在深度流或点云生成，而在点云发布环节。

## 根因判断

YOLO 节点点云订阅使用了：

```python
qos_cloud = QoSProfile(
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE)
```

RealSense 点云发布端使用：

```text
Reliability: RELIABLE
History: KEEP_LAST (10)
Durability: VOLATILE
```

在当前 `rmw_cyclonedds_cpp` 环境下，该 BEST_EFFORT 订阅没有与 RELIABLE 发布器形成有效 DDS 匹配。因此 RealSense 点云滤波器中的订阅计数门控：

```cpp
if ((!_pointcloud_publisher) || (!(_pointcloud_publisher->get_subscription_count())))
    return;
```

认为没有有效订阅者，点云生成后被直接跳过，不会真正发布到 topic。

## 修改内容

文件：

```text
/root/yolo/scripts/inference/yolo_inference_node_cloud.py
```

修改：

```python
qos_cloud = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE)
```

提交：

```text
8121f59 将YOLO点云订阅QoS改为RELIABLE以匹配RealSense发布端
```

## 验证结果

执行：

```bash
python3 -m py_compile /root/yolo/scripts/inference/yolo_inference_node_cloud.py
```

通过。

运行 YOLO 节点：

```bash
cd /root/yolo/scripts/inference
python yolo_inference_node_cloud.py \
  --ros-args \
  -p mode:=debug \
  -p publish_debug_cloud:=true \
  -p sync_debug:=true \
  --log-level yolo_inference_node:=debug
```

现象：

```text
[sync] WAIT_CLOUD ...（仅启动初期）
[cloud_cb] stamp=... frame=d435i_depth_optical_frame points=...
[sync] color=... cloud=... delta=... match=True
```

点云能够正常到达并参与同步，推理正常，问题解决。

## 后续注意事项

1. YOLO 的 color、camera_info、pointcloud 三个订阅应尽量使用一致的 QoS。
2. RealSense 点云话题属于较大的 PointCloud2 消息，不建议使用 BEST_EFFORT 订阅，否则可能出现发布端订阅计数门控导致点云不发布。
3. 再次遇到“topic info 中有订阅，但实际收不到数据”的问题，应分别用 RELIABLE 和 BEST_EFFORT 的 `ros2 topic echo` 做对比验证。
