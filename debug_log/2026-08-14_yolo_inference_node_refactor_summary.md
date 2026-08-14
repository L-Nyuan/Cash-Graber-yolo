# 2026-08-14 YOLO 推理节点拆分重构总结

> 背景：`yolo_inference_node_cloud.py` 增长到 1022 行，单函数 `_inference_loop` 达 244 行，
> 按"职责域合并、不按函数粒度拆分"的原则重构为"入口节点 + 4 个支撑模块"。
> 本总结替代原拆分计划文档（`docs/refactor_yolo_inference_node_cloud.md`，已完成使命后删除）。

## 1. 最终结构

| 文件 | 行数 | 职责 |
|---|---|---|
| `yolo_inference_node_cloud.py` | 591（原 1022，−42%） | 入口编排：配置加载、订阅发布、推理循环；`_inference_loop` 46 行 |
| `cloud_utils.py` | 216 | 点云编解码、mask 缩放/腐蚀、裁剪、SOR、时间戳等纯函数 |
| `cloud_state.py` | 293 | `CloudSyncMatcher`（同步 + pending 等待 + 节流日志）+ `CloudCache`（点云缓存） |
| `node_config.py` | 169（含参数详注） | 全部参数声明与解析（`NodeConfig`） |
| `rviz_launcher.py` | 136 | debug 模式 rviz2 子进程生命周期（含 Qt 环境修复） |
| `test_cloud_utils.py` | 54 | 点云编解码往返离线冒烟测试 |

其余既有模块（`yolo_inference.py` / `object_tracker.py` / `image_utils.py` /
`marker_rviz.py` / `visualization_utils.py`）保持不变；`mask_point_msg.py` 因无引用已删除。

## 2. 拆分后节点方法划分

```text
_inference_loop            # 46 行编排层
  ├─ _acquire_frame        # 相机信息检查 + CloudSyncMatcher.acquire（含 pending 决策）
  ├─ _run_inference        # YOLO 推理 + IoU 追踪 + 目标组装
  ├─ _crop_tracked_clouds  # 逐目标 mask 缩放/腐蚀/裁剪/SOR + [crop] 日志
  ├─ _update_cloud_cache   # 无匹配清空 / 有匹配清理过期并写入
  ├─ _publish_detections   # JSON 检测元数据（保留在节点，按用户决定不抽模块）
  └─ _publish_debug_outputs # debug 模式合并点云 + RViz markers
```

## 3. 执行过程（Phase 0–8，20 个提交）

| 阶段 | 内容 | 提交 |
|---|---|---|
| Phase 0 | 基线标签 `refactor-baseline` + 点云往返冒烟测试 | `511e212` |
| Phase 1 | 删除死代码 `_box_iou` | `affe562` |
| Phase 2 | 点云/时间戳纯函数 → `cloud_utils.py` | `f8f4b58` `d46caf7` |
| Phase 3 | 参数声明解析 → `node_config.py` | `f9b43fd` |
| Phase 4 | rviz2 生命周期 → `rviz_launcher.py`（stop 增加超时强杀兜底） | `7b76164` |
| Phase 5 | 时间同步 → `cloud_state.py`（`CloudSyncMatcher`） | `7df47f5` |
| Phase 6 | 点云缓存 → `cloud_state.py`（`CloudCache`） | `0cd57b0` |
| Phase 7 | `_inference_loop` 拆 5 个方法（244 → 46 行） | `cb3fb2d` `acc09b2` `64374bb` `9906fa0` |
| 可选② | pending 决策并入 `CloudSyncMatcher.acquire` | `7e97e9a` |
| Phase 8 | 删除 `mask_point_msg.py`，更新 AGENTS/README，注释清理，参数详注 | `fe375b9` `c917eba` `c2d9348` |

## 4. 关键决策与偏差

1. **v1 → v2 合并**：初版按函数粒度拆 8 个新模块，用户反馈过碎；合并为 4 个，
   分类依据为"纯函数工具 / 节点状态组件 / 外部进程管理 / 配置加载"。
2. **节点行数目标未达 v1 预估**（500/350）：v2 取消 `detections_msg.py`（JSON 构造留在节点），
   且 Phase 7 把循环逻辑拆成同文件方法（增加方法定义开销）。实际节点 591 行，
   `_inference_loop` 46 行达标；文件瘦身主要来自 Phase 2–6。
3. **JSON 构造不抽模块**：用户明确决定保留在节点（唯一剩余可选优化，约 −40 行）。
4. **rviz 退出修复**：测试发现 rviz2 收到 SIGTERM 后不退出（残留孤儿进程），
   `RvizLauncher.stop()` 增加 5 秒等待 + SIGKILL 兜底。
5. **死代码清理**：`_box_iou`（从未调用）、`color_for_cloud`（赋值未使用）在重构中删除。

## 5. 验证记录

- 每阶段：`py_compile` + import 冒烟 + 离线往返测试（`test_cloud_utils.py`）全部通过。
- 运行时（相机在线）：生产/调试模式各多次 8–12 秒运行，检测、裁剪、同步、
  debug 发布、rviz 自动打开/关闭均正常；状态日志计数（sync_ok/skip/hold）正常。
- 命令行覆盖：`-p sync.tolerance:=0.02 -p model.conf:=0.5` 生效并检测到 3 目标；
  `-p mode:=foo` 正确回退 production。
- 按需点云链路：空缓存请求返回空点云 ✓；用户真机请求测试（含保存 .ply）通过 ✓。

## 6. 遗留事项

- `/root/yolo/debug_clouds/`（真机测试保存的点云）未纳入 git，建议加入 `.gitignore`。
- 唯一可选优化：`_publish_detections` 的 JSON 构造下沉为 `cloud_utils.build_detections_json`
  （约 −40 行），用户决定暂不执行。
- 基线标签 `refactor-baseline` 保留在 git 中，可随时对照拆分前后代码。
