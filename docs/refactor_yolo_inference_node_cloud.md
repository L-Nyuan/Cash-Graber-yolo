# `yolo_inference_node_cloud.py` 拆分计划（v2）

> 本文档是拆分任务的唯一事实来源（single source of truth）。
> 每个 Codex 会话开工前先读本文档；每完成一个步骤就更新"进度追踪"勾选表并提交 git。

创建日期：2026-08-13
修订记录：
- **v2（2026-08-14）**：按用户意见合并子模块，新模块从 8 个减为 4 个；分类原则改为"按职责域合并，不按函数粒度拆分"。
- v1（2026-08-13）：初版细粒度拆分方案。

目标文件：`/root/yolo/scripts/inference/yolo_inference_node_cloud.py`（当前 1022 行）
基线提交：`dcbd771`

---

## 0. 文档目的

把 1022 行的 ROS2 推理节点拆成一组职责清晰、方便管理的模块，同时保证：

1. **行为完全不变**：话题、参数名、QoS、消息格式、日志风格一律不动。
2. **每步都可运行**：每一步结束都有可执行的验收命令，且必须有 git 提交，随时可回滚。
3. **可跨会话接力**：你需要在中间做真机测试，文档 + git 提交记录就是连续性保障——任何一次新会话都可以从"进度追踪"表继续，不依赖对话历史。
4. **不过度拆分**：模块数量固定为目标结构，不因某个函数大小再拆；只有当某个模块超过约 400 行时才考虑二次拆分。

---

## 1. 现状分析（2026-08-13）

### 1.1 文件内部结构

| 行范围 | 内容 | 大致行数 | 拆分去向（v2） |
|---|---|---|---|
| 1–60 | 模块 docstring + import | 60 | 精简后保留 |
| 62–128 | `realsense_cloud_to_xyzrgb`（点云解码） | 67 | `cloud_utils.py` |
| 130–141 | `ensure_mask_resolution` | 12 | `cloud_utils.py` |
| 142–173 | `crop_cloud_by_mask` | 32 | `cloud_utils.py` |
| 174–228 | `build_pointcloud2` | 55 | `cloud_utils.py` |
| 229–242 | `_get_bbox` | 14 | 可选：`extract_bbox` → `yolo_inference.py` |
| 244–254 | `_box_iou` | 11 | **死代码，删除**（`ObjectTracker` 已有同名实现） |
| 255–269 | `_stamp_ns` / `_stamp_text` | 15 | `cloud_utils.py` |
| 271–465 | `YOLOInferenceNode.__init__`（参数/QoS/订阅发布/状态） | 195 | 参数部分进 `node_config.py` |
| 467–564 | rviz2 生命周期 5 个方法 | 98 | `rviz_launcher.py` |
| 566–610 | `_color_cb` / `_info_cb` / `_cloud_cb` | 45 | 节点内精简（`_cloud_cb` 变薄） |
| 612–666 | 时间同步匹配 + 日志 | 55 | `cloud_state.py`（`CloudSyncMatcher`） |
| 668–693 | `_on_request_cloud` | 26 | 节点内保留，缓存逻辑用 `CloudCache` |
| 695–948 | `_inference_loop`（254 行，最大单函数） | 254 | Phase 7 拆成 4 个方法 |
| 950–985 | `_publish_detections` | 36 | **留在节点内**（不抽模块，仅缩为薄方法） |
| 987–1005 | `_print_status` | 19 | 节点内保留 |
| 1007–1021 | `main()` | 15 | 节点内保留 |

### 1.2 已有模块（拆分初期已完成的成果，本次尽量不动）

| 模块 | 职责 | 状态 |
|---|---|---|
| `yolo_inference.py` | YOLO11-seg 推理封装 | 保留，可加 `extract_bbox` 函数 |
| `object_tracker.py` | IoU 多目标追踪 | 保留 |
| `image_utils.py` | ROS Image 手动解码 | 保留 |
| `marker_rviz.py` | RViz Marker 发布 | 保留 |
| `visualization_utils.py` | 调试图保存（README 有独立调试用法） | **保留，勿删** |
| `mask_point_msg.py` | 旧版点云消息工具 | 当前无人 import，收尾阶段处理 |

### 1.3 已发现的问题

- `_box_iou` 定义后从未被调用，是死代码（`object_tracker.py` 内部有自己的 `_iou`）。
- `_inference_loop` 一个函数承载了取帧、同步、推理、追踪、裁剪、缓存、发布、日志全部逻辑。
- `__init__` 里参数声明 + 解析占约 195 行，适合整体搬出。
- `mask_point_msg.py` 已无引用，属于历史遗留，收尾阶段确认后清理。
- `visualization_utils.py` 虽是旧模块，但 README 中有独立调试用法（`save_debug_image` 单测），**保留不删**。

---

## 2. 拆分原则（红线，任何时候不得违反）

1. **只搬移，不重写**：函数按原样搬进新模块（含 docstring、注释、日志文案），禁止顺手"优化"逻辑。
2. **参数、话题、QoS、消息格式零变化**：命令行 `-p model.conf:=...` 等覆盖方式必须继续生效。
3. **保持平级 import**：节点以 `cd /root/yolo/scripts/inference && python yolo_inference_node_cloud.py` 方式运行，新模块放同目录，用 `from cloud_utils import ...` 即可。不要建 ROS package、不要改 `sys.path`、不要动 `setup.py`。
4. **保持 cv2 延迟导入**：`ensure_mask_resolution` 等函数内 `import cv2` 是有意为之（避免 import 时污染 Qt 插件环境），搬移时原样保留。
5. **单线程执行器约束**：抽出的类不要自己创建 timer / spin / 阻塞循环；只有 `RvizLauncher` 的进程观察线程（daemon）例外。
6. **不加新依赖**：不 pip 安装任何包；open3d 保持现有的"可用则用，不可用则跳过"逻辑。
7. **每步一个 commit**：任何一步没通过验收就不提交；提交信息按本文档模板写，便于 `git log` 对照进度表。
8. **先复制后删除**：搬函数时先在节点里 `import` 并确认能用，再删旧定义，避免"删了才发现没搬全"。
9. **模块数固定**：目标结构就是最终结构。合并后每个模块都有独立的存在理由（见 3.1），不要再按单个函数拆新文件；单个模块超过约 400 行时才考虑二拆。

---

## 3. 目标结构（v2：合并后）

### 3.1 分类原则

合并依据是"职责域 + 生命周期归属"，而不是函数大小：

| 类别 | 判断标准 | 归入模块 |
|---|---|---|
| 纯函数工具 | 无节点状态、不依赖 rclpy Node、可离线单测 | `cloud_utils.py` |
| 节点状态组件 | 生命周期与节点一致，由回调/定时器驱动 | `cloud_state.py` |
| 外部进程管理 | 管理子进程生命周期（rviz2） | `rviz_launcher.py` |
| 配置加载 | 纯声明与解析，不参与运行逻辑 | `node_config.py` |
| 节点编排 | 组装上述组件、订阅发布、推理循环 | `yolo_inference_node_cloud.py` |

### 3.2 目标文件树

```text
scripts/inference/
├── yolo_inference_node_cloud.py   # 入口节点：配置加载 + 订阅发布 + 推理编排（目标 ≈ 350 行）
├── cloud_utils.py                 # 纯函数：点云解码/编码、mask 缩放腐蚀、裁剪、SOR、时间戳（≈ 230 行）
├── cloud_state.py                 # 节点状态组件：CloudSyncMatcher（时间同步）+ CloudCache（点云缓存）（≈ 150 行）
├── node_config.py                 # 全部参数声明与解析（NodeConfig）（≈ 110 行）
├── rviz_launcher.py               # debug 模式 rviz2 启动/监控/退出（RvizLauncher）（≈ 110 行）
├── yolo_inference.py              # 已有：可新增 extract_bbox
├── object_tracker.py              # 已有，不动
├── image_utils.py                 # 已有，不动
├── marker_rviz.py                 # 已有，不动
└── test_cloud_utils.py            # 新增离线冒烟测试（可选但推荐）
```

对比 v1：`cloud_io.py + cloud_processing.py + ros_utils.py` 合并为 `cloud_utils.py`；`cloud_sync.py + cloud_cache.py` 合并为 `cloud_state.py`；`detections_msg.py` 取消（JSON 构造保留在节点内）。新模块从 8 个减到 4 个。

### 3.3 依赖关系（只允许向下依赖）

```text
node ─→ node_config（一次性加载）
node ─→ rviz_launcher（进程生命周期）
node ─→ cloud_state（同步 + 缓存）
node ─→ cloud_utils（纯函数）
cloud_state ─→ cloud_utils
node ─→ yolo_inference / object_tracker / image_utils / marker_rviz（已有）
```

规则：`cloud_utils.py` 不 import 节点、不 import rclpy 的 Node；`cloud_state.py`、`node_config.py`、`rviz_launcher.py` 允许接触节点上下文（通过构造参数注入 logger / node）。

---

## 4. 逐步拆分流程

### 全局验收命令（每一阶段结束后都要跑）

```bash
# ① 语法检查（不需要 ROS / conda，任何 python3 都行）
cd /root/yolo/scripts/inference
python3 -m py_compile yolo_inference_node_cloud.py

# ② import 冒烟（需要 ROS + yolo 环境）
conda activate yolo
source /opt/ros/humble/setup.bash
cd /root/yolo/scripts/inference
python -c "import yolo_inference_node_cloud as m; print('IMPORT OK')"
```

② 通过后，无相机的快速运行验收：

```bash
# 生产模式跑 8 秒，应看到"等待首帧..."状态日志且无 traceback
timeout 8 python yolo_inference_node_cloud.py --ros-args

# debug 模式不启动 rviz 跑 8 秒
timeout 8 python yolo_inference_node_cloud.py --ros-args \
  -p mode:=debug -p debug.launch_rviz:=false
```

有相机的完整验收见各阶段"真机验收"栏。

---

### Phase 0：基线准备（约 15 分钟）

**做什么：**

1. 确认工作区干净：`git status --short` 无输出（2026-08-13 已确认）。
2. 打基线标签：`git tag refactor-baseline`（保护点，永远可回到拆分前）。
3. 新建 `test_cloud_utils.py`：用合成数据做 `build_pointcloud2` → `realsense_cloud_to_xyzrgb` 往返测试（点数、xyz、rgb 值一致），作为 Phase 2 的离线安全网。
4. 在本文档"进度追踪"勾选 Phase 0。

**验收：** `python test_cloud_utils.py` 通过；`git tag` 存在。

**提交：** `chore(node): refactor Phase 0 基线标签 + 点云往返冒烟测试`

---

### Phase 1：删除死代码（约 10 分钟，最安全的一刀）

**做什么：**

1. 删除 `_box_iou`（从未被调用）。
2. 顺带确认没有其他未使用定义（`rg -n "_box_iou" .` 应为空）。
3. 不要顺手删"看起来没用"的 import（`DurabilityPolicy` 等目前都在用）；注释掉的调试代码也不在本阶段处理。

**验收：** 全局验收 ①② 通过。

**提交：** `refactor(node): step1 删除未使用的 _box_iou`

---

### Phase 2：抽取纯函数工具 `cloud_utils.py`（约 50 分钟）

**做什么：**

1. 新建 `cloud_utils.py`，原样搬入：
   - 点云编解码：`realsense_cloud_to_xyzrgb`、`build_pointcloud2`
   - mask/裁剪：`ensure_mask_resolution`、`crop_cloud_by_mask`
   - 时间戳：`stamp_ns`、`stamp_text`（去掉下划线前缀，公开命名）
   - 新增小函数：`erode_mask(mask, iterations)`（把推理循环里的 3 行 cv2.erode 搬出来）、`remove_outliers_sor(cloud, o3d, nb_neighbors=20, std_ratio=1.0)`（把循环里的 SOR 7 行搬出来）
2. 节点改为 `from cloud_utils import ...`，更新全部调用点后删除旧定义。
3. **本阶段先不重构推理循环**：循环里的 erode/SOR 暂时直接调用新函数，只改调用点。
4. cv2 的 `import cv2` 保持在函数内部（红线 4）。

**验收：**

- 全局验收 ①② 通过。
- `python test_cloud_utils.py` 通过（往返测试）。
- `rg -n "def (realsense_cloud_to_xyzrgb|build_pointcloud2|ensure_mask_resolution|crop_cloud_by_mask|_stamp_)" yolo_inference_node_cloud.py` 无输出（函数已全部搬走）。

**提交：** 建议拆两个 commit：`refactor(node): step2a 点云编解码与裁剪 → cloud_utils.py`、`refactor(node): step2b 时间戳与mask/SOR工具 → cloud_utils.py`

---

### Phase 3：抽取参数配置 `node_config.py`（约 40 分钟）

**做什么：**

1. 新建 `node_config.py`，定义一个 `NodeConfig` 类：
   - `@classmethod load(node)` 内集中 `declare_parameter` + `get_parameter`，字段名与现在 `self._xxx` 一一对应（`mode`、`cloud_topic`、`sync_tolerance_ns`、`debug_dir`……）。
   - 模式校验逻辑（未知 mode 回退 production）原样搬入。
2. 节点 `__init__` 改为：
   ```python
   cfg = NodeConfig.load(self)
   self._mode = cfg.mode
   self._cloud_topic = cfg.cloud_topic
   ...
   ```
3. **参数名、默认值必须逐字保留**（`model.path` 默认 `/root/yolo/result/final/best.pt`、`sync.tolerance` 默认 0.05 等）。

**验收：**

- 全局验收 ①②。
- 命令行验收：用几条 `-p` 覆盖启动（如 `-p sync.tolerance:=0.02 -p model.conf:=0.5`），日志里能看到覆盖后的值。

**提交：** `refactor(node): step3 参数声明解析 → node_config.py`

---

### Phase 4：抽取 rviz2 生命周期 `rviz_launcher.py`（约 30 分钟）

**做什么：**

1. 新建 `rviz_launcher.py`，定义 `RvizLauncher` 类：
   - `__init__(self, rviz_config, debug_dir, logger)`（logger 传节点，用 `get_logger()`）
   - `start()` → 原 `_launch_rviz_window` 逻辑（含 `_build_rviz_env` 的环境清理）
   - `stop()` → 原 `_stop_rviz` 逻辑
   - `watch()` / `read_tail()` → 原 `_watch_rviz` / `_read_tail`
   - 属性 `proc`、`stopping` 相应搬入
2. 节点 `__init__` 里变成 `self._rviz = RvizLauncher(...)` + 条件启动；`main()` 的 `node._stop_rviz()` 改为 `node._rviz.stop()`（或节点留一个薄封装方法）。
3. **不要动** `_build_rviz_env` 里清除 conda Qt 变量、清理 `LD_LIBRARY_PATH` 的逻辑，逐行搬。

**验收：**

- 全局验收 ①②。
- 无显示器：`-p mode:=debug -p debug.launch_rviz:=false` 正常。
- 有显示器：debug 模式 rviz2 能自动打开、节点退出后 rviz2 自动关闭（验证 `stop()`）。

**提交：** `refactor(node): step4 rviz2 生命周期 → rviz_launcher.py`

---

### Phase 5：抽取时间同步 `cloud_state.py` 之 `CloudSyncMatcher`（约 40 分钟）

**做什么：**

1. 新建 `cloud_state.py`，定义 `CloudSyncMatcher`：
   - `__init__(self, tolerance_ns, buffer_size, sync_debug, logger)`
   - `add_cloud(stamp, frame_id, xyz, rgb)`：原 `_cloud_cb` 里的缓冲追加 + 超长裁剪逻辑
   - `match(color_stamp)`：原 `_match_cloud_for_color`，返回一个小 dataclass（`xyz/rgb/cloud_stamp/cloud_frame/delta_ns/matched`）
   - `log(...)`：原 `_log_sync`（含节流逻辑）
   - `__len__()` / `msgs` 计数等供状态日志使用
2. 节点 `_cloud_cb` 变薄：解码后调用 `self._sync.add_cloud(...)`；`_inference_loop` 里同步相关行改为调用 `matcher.match(...)` 与 `matcher.log(...)`。
3. **pending 帧逻辑（`_pending_color*`）先留在节点**，Phase 7 再决定是否搬入。

**验收：**

- 全局验收 ①②。
- 真机验收（重点）：开相机跑 debug 模式，观察 `[sync]` 日志与之前一致（`delta`、`buffer`、`match` 字段不变）；`sync_ok/sync_skip` 计数正常增长。

**提交：** `refactor(node): step5 时间同步 → cloud_state.py`

---

### Phase 6：抽取点云缓存 `cloud_state.py` 之 `CloudCache`（约 20 分钟）

**做什么：**

1. 在 `cloud_state.py` 中新增 `CloudCache`：
   - 内部持有 `dict` + `threading.Lock`
   - `get(track_id)`、`update(active_ids, new_clouds)`（清过期 + 写新）、`clear()`、`keys()`、`__len__()`
2. 节点 `_inference_loop` 里缓存相关 5 处（清空、清理过期、写入、请求读取）全部改为调用 `self._cloud_cache`。
3. `_on_request_cloud` 里"缓存为空发空点云"的逻辑留在节点（涉及发布），只把 `cache.get` 换掉。

**验收：**

- 全局验收 ①②。
- 真机验收（重点）：`python request_point_debug.py --ros-args -p track_id:=1 -p save_cloud:=true` 能按 ID 取到点云；连续请求两次结果一致。

**提交：** `refactor(node): step6 点云缓存 → cloud_state.py`

---

### Phase 7：拆分 `_inference_loop`（约 1.5 小时，最大的一步，务必拆小步提交）

目标：254 行的 `_inference_loop` 变成约 60 行的"编排层"，只按顺序调用以下方法：

```python
def _inference_loop(self):            # 编排：取帧 → 推理 → 裁剪 → 发布
    frame = self._acquire_frame()     # 7.1
    if frame is None:
        return
    tracked = self._run_inference(frame.image)                    # 7.2
    new_clouds = self._crop_tracked_clouds(tracked, frame.cloud)  # 7.3
    self._update_cloud_cache(tracked, new_clouds, frame.matched)  # 7.4
    self._publish_detections(tracked)
    self._publish_debug_outputs(tracked)                          # 7.5
    self._log_frame(frame, tracked, t0)
```

说明：`_publish_detections` 不再抽成独立模块，保留为节点方法（约 30 行，内部直接构造 JSON）；若日后嫌节点大，可把 payload 构造下沉为 `cloud_utils.build_detections_json` 纯函数（可选，不在本次范围）。

**7.1 `_acquire_frame()`（约 80 行逻辑）**

- pending 帧超时判定、latest/pending 二选一、`_match_cloud_for_color` 调用、`_log_sync`、hold/drop 判定、`_last_crop_stamp/_last_crop_frame` 更新、首帧诊断日志。
- 返回一个小 dataclass（`image / color_stamp / cloud_xyz / cloud_rgb / matched / cloud_stamp / cloud_frame`）或 `None`。
- 验收：真机跑 2 分钟，`sync` 日志、pending 行为（偶发等待一拍）与拆分前一致。
- 提交：`refactor(node): step7.1 取帧+同步决策 → _acquire_frame`

**7.2 `_run_inference(image)`（约 25 行逻辑）**

- YOLO predict → `_get_bbox` 组 dets → `tracker.update` → 组装 `tracked_objects` 并写入 `obj["track_id"]`、`obj["cloud"]` 占位。
- 提交：`refactor(node): step7.2 推理+追踪 → _run_inference`

**7.3 `_crop_tracked_clouds(tracked, frame)`（约 85 行逻辑，重点）**

- 逐 track 循环：mask 缩放 → `erode_mask` → `crop_cloud_by_mask` → `remove_outliers_sor` → `[crop]` 调试日志 → `new_clouds` 收集。
- 此时循环内所有 cv2/open3d 细节都已换成 `cloud_utils` 的函数，本步只是把循环整体搬进方法（行为零变化）。
- 验收：真机看 `[crop]` 日志的 `px`、`crop`、`after_sor` 数字与拆分前一致。
- 提交：`refactor(node): step7.3 点云裁剪循环 → _crop_tracked_clouds`

**7.4 `_update_cloud_cache(...)`（约 20 行）**

- 把"清空旧缓存 / 清理过期 + 写入新点云"两段逻辑搬成方法（内部调用 `CloudCache`）。

**7.5 `_publish_debug_outputs(tracked)`（约 35 行）**

- debug 模式：合并点云发 `/yolo/debug_cloud`（含节流）+ `_publish_markers`。
- 提交：`refactor(node): step7.4-7.5 缓存更新与debug发布拆方法`

**7.6 收尾校验**

- `_inference_loop` 行数确认 ≤ 70；`wc -l` 整文件应降到约 500 行以内。
- 完整真机验收（见第 5 节"完整验收清单"）。

---

### Phase 8：收尾（约 30 分钟）

1. 更新 `yolo_inference_node_cloud.py` 顶部 docstring：新模块地图 + 运行示例不变。
2. `visualization_utils.py` 保留（README 调试用法，勿删）；确认 `mask_point_msg.py` 仍无引用后删除（先 `rg -n "mask_point_msg" /root/yolo/scripts` 确认，再 `git rm`，可随时从 git 恢复）。
3. 更新 `/root/yolo/AGENTS.md` 的目录地图，把新模块加进去（AGENTS.md 是 Codex 每次会话的自动上下文，必须同步）。
4. 全量验收：语法 + import + 生产/调试模式无相机跑 10 秒 + 有相机完整流程。
5. 在本文档记录最终行数、最终 commit，并把进度表全部勾完。

**提交：** `refactor(node): step8 收尾清理与文档同步`

---

## 5. 验证体系（三层）

### 第一层：静态（每步必跑，秒级）

```bash
cd /root/yolo/scripts/inference
python3 -m py_compile yolo_inference_node_cloud.py
```

### 第二层：import 冒烟（每步必跑，几秒）

```bash
conda activate yolo
source /opt/ros/humble/setup.bash
cd /root/yolo/scripts/inference
python -c "import yolo_inference_node_cloud as m; print('IMPORT OK')"
```

### 第三层：运行验收

**无相机快速验收（每步可选）：**

```bash
timeout 8 python yolo_inference_node_cloud.py --ros-args
timeout 8 python yolo_inference_node_cloud.py --ros-args -p mode:=debug -p debug.launch_rviz:=false
```

**有相机完整验收（Phase 3/4/5/6/7 后必跑）：**

```bash
# 1. 启动相机（AGENTS.md 2.2 的命令）
# 2. 启动节点
cd /root/yolo/scripts/inference
python yolo_inference_node_cloud.py --ros-args -p mode:=debug
# 3. 看检测 JSON
ros2 topic echo /yolo/detections --once
# 4. 按 ID 请求点云并保存
cd /root/yolo/scripts
python request_point_debug.py --ros-args -p track_id:=1 -p save_cloud:=true -p save_format:=ply
# 5. 对照项：
#    - RViz 正常显示、退出节点后 rviz2 自动关闭
#    - [sync] 日志字段与拆分前一致，sync_ok 增长
#    - [crop] 日志的 px/crop/after_sor 数值合理
#    - /yolo/object_cloud 能收到点，点数 > 0
#    - /yolo/detections 的 JSON 字段无变化
```

---

## 6. 进度追踪（跨会话连续性机制）

> **新会话恢复流程：**
> 1. 读本文件；
> 2. 跑 `git log --oneline -15`，对照下面的勾选表确认已完成的步骤；
> 3. 从第一个未勾选步骤开始；
> 4. 完成该步骤全部验收后再勾选，随代码一起 commit。

### 勾选表（v2）

- [x] Phase 0：基线标签 + `test_cloud_utils.py`
- [x] Phase 1：删除 `_box_iou`
- [x] Phase 2a：点云编解码与裁剪 → `cloud_utils.py`
- [x] Phase 2b：时间戳与 mask/SOR 工具 → `cloud_utils.py`
- [x] Phase 3：参数 → `node_config.py`
- [x] Phase 4：rviz → `rviz_launcher.py`
- [x] Phase 5：同步 → `cloud_state.py`（`CloudSyncMatcher`）
- [x] Phase 6：缓存 → `cloud_state.py`（`CloudCache`）
- [ ] Phase 7.1：`_acquire_frame`
- [ ] Phase 7.2：`_run_inference`
- [ ] Phase 7.3：`_crop_tracked_clouds`
- [ ] Phase 7.4–7.5：缓存更新 + debug 发布
- [ ] Phase 7.6：`_inference_loop` ≤ 70 行，完整验收
- [ ] Phase 8：收尾清理 + AGENTS.md 更新

### 里程碑记录

| 里程碑 | 行数 | commit |
|---|---|---|
| 拆分前（2026-08-13） | 1022 | `dcbd771` |
| Phase 2 完成（纯函数搬出） | 节点 835 / cloud_utils 216 | `f8f4b58` + `d46caf7` |
| Phase 3 完成（配置搬出） | 节点 792 / node_config 123 | `f9b43fd` |
| Phase 4 完成（rviz 搬出） | 节点 693 / rviz_launcher 136 | `7b76164` |
| Phase 5 完成（同步搬出） | 节点 634 / cloud_state 117 | `7df47f5` |
| Phase 6 完成（缓存搬出） | 节点 626 / cloud_state 150 | （本次提交） |
| Phase 7 完成（循环拆分） | ≈ 350 | （待填） |
| 拆分完成 | 节点 ≈ 350 + 4 个模块 | （待填） |

---

## 7. 风险与回滚

| 风险 | 缓解措施 |
|---|---|
| 搬函数时漏改调用点 | 每步结束 `rg` 旧函数名确认零残留；import 冒烟兜底 |
| 参数默认值/名字写错 | Phase 3 逐字对照原文件，验收时用 `-p` 覆盖验证 |
| 同步逻辑行为漂移 | Phase 5 只搬不改；真机对比 `[sync]` 日志 |
| 点云内容变化（颜色/坐标） | `test_cloud_utils.py` 往返测试 + 真机对比 `[crop]` 日志 |
| rviz2 启动行为变化 | Phase 4 后必须在有显示器环境验证打开/关闭 |
| 合并后模块职责模糊（"杂物箱"倾向） | 每个模块的类别归属在 3.1 有明确判断标准；超 400 行才二拆 |
| 会话中断 | 本文档 + git commit 每步一个，随时可续 |

**回滚方法：**

```bash
# 回滚单步
git revert <step_commit>          # 推荐，保留历史
# 或回到拆分前
git checkout refactor-baseline -- scripts/inference/
```

---

## 8. 完成标准（Definition of Done）

1. `yolo_inference_node_cloud.py` ≤ 360 行，`_inference_loop` ≤ 70 行。
2. 新模块共 4 个：`cloud_utils.py` / `cloud_state.py` / `node_config.py` / `rviz_launcher.py`，各自职责见 3.1，无循环依赖。
3. 全量验收通过：语法 / import / 无相机运行 / 有相机完整流程（检测、点云、请求、rviz、退出清理）。
4. `/yolo/detections` JSON 字段、话题名、参数名、QoS 与拆分前完全一致。
5. `AGENTS.md` 目录地图已更新，本文档进度表全部勾选，里程碑行数已填写。
