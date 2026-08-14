# node_config.py —— 节点参数声明与解析
#
# 拆分计划 Phase 3（v2）：
#   集中 declare_parameter + get_parameter，节点 __init__ 只消费 NodeConfig。
#   参数名与默认值必须与拆分前逐字一致（命令行 -p 覆盖方式保持不变）。


class NodeConfig:
    """YOLOInferenceNode 全部参数。

    字段名与节点原 self._xxx 一一对应；值全部来自 load() 的
    declare_parameter + get_parameter，默认值唯一来源是 declare_parameter。
    """

    mode: str
    debug_mode: bool
    cloud_topic: str
    color_topic: str
    info_topic: str
    sync_tolerance_ns: int
    require_synced_cloud: bool
    sync_debug: bool
    cloud_buffer_size: int
    pending_max_wait: float
    model_path: str
    model_imgsz: int
    model_conf: float
    model_iou: float
    debug_cloud_interval: float
    debug_dir: str
    launch_rviz: bool
    rviz_config: str
    mask_erode: int
    cloud_sor: bool
    tracker_max_age: int
    tracker_min_hits: int
    tracker_iou: float

    @classmethod
    def load(cls, node) -> "NodeConfig":
        """在节点上声明全部参数并解析成 NodeConfig。"""
        # ── 参数（按前缀分组，命令行更规整）────────────────
        # mode: production | debug
        node.declare_parameter("mode", "production")
        # model.* 模型
        node.declare_parameter("model.path", "/root/yolo/result/final/best.pt")
        node.declare_parameter("model.imgsz", 640)
        node.declare_parameter("model.conf", 0.8)
        node.declare_parameter("model.iou", 0.7)
        # tracker.* IoU 追踪器
        node.declare_parameter("tracker.max_age", 30)
        node.declare_parameter("tracker.min_hits", 3)
        node.declare_parameter("tracker.iou", 0.3)
        # cloud.* 点云后处理
        node.declare_parameter("cloud.mask_erode", 1)
        node.declare_parameter("cloud.sor", True)
        # sync.* 彩色图/点云时间同步
        node.declare_parameter("sync.tolerance", 0.05)
        node.declare_parameter("sync.require_cloud", True)
        node.declare_parameter("sync.buffer_size", 30)
        node.declare_parameter("sync.pending_wait", 0.12)
        node.declare_parameter("sync.debug", False)
        # topic.* 话题名（如需改命名空间可覆盖）
        node.declare_parameter(
            "topic.cloud", "/Wrist_Camera/d435i/depth/color/points")
        node.declare_parameter(
            "topic.color", "/Wrist_Camera/d435i/color/image_raw")
        node.declare_parameter(
            "topic.info", "/Wrist_Camera/d435i/color/camera_info")
        # debug.* 调试输出（仅 mode=debug 生效）
        node.declare_parameter("debug.cloud_hz", 15.0)
        node.declare_parameter("debug.dir", "/root/yolo/yolo_debug")
        node.declare_parameter("debug.launch_rviz", True)
        node.declare_parameter(
            "debug.rviz_config", "/root/yolo/rviz/debug.rviz")

        cfg = cls()

        # ── 模式：统一决定输出内容 ────────────────────────
        mode = str(node.get_parameter("mode").value).strip().lower()
        if mode not in ("production", "debug"):
            node.get_logger().warn(
                f"未知 mode={mode}，回退到 production")
            mode = "production"
        cfg.mode = mode
        cfg.debug_mode = (mode == "debug")

        cfg.cloud_topic = node.get_parameter("topic.cloud").value
        cfg.color_topic = node.get_parameter("topic.color").value
        cfg.info_topic = node.get_parameter("topic.info").value

        cfg.sync_tolerance_ns = int(
            node.get_parameter("sync.tolerance").value * 1_000_000_000)
        cfg.require_synced_cloud = bool(
            node.get_parameter("sync.require_cloud").value)
        # 同步调试日志：debug 模式默认开启，production 默认关闭
        cfg.sync_debug = (bool(node.get_parameter("sync.debug").value)
                          or cfg.debug_mode)
        cfg.cloud_buffer_size = int(
            node.get_parameter("sync.buffer_size").value)
        cfg.pending_max_wait = float(
            node.get_parameter("sync.pending_wait").value)

        cfg.model_path = node.get_parameter("model.path").value
        cfg.model_imgsz = node.get_parameter("model.imgsz").value
        cfg.model_conf = node.get_parameter("model.conf").value
        cfg.model_iou = node.get_parameter("model.iou").value

        cfg.debug_cloud_interval = (
            1.0 / max(float(node.get_parameter("debug.cloud_hz").value), 1.0))
        cfg.debug_dir = node.get_parameter("debug.dir").value
        cfg.launch_rviz = bool(
            node.get_parameter("debug.launch_rviz").value)
        cfg.rviz_config = node.get_parameter("debug.rviz_config").value

        cfg.mask_erode = int(node.get_parameter("cloud.mask_erode").value)
        cfg.cloud_sor = bool(node.get_parameter("cloud.sor").value)

        cfg.tracker_max_age = node.get_parameter("tracker.max_age").value
        cfg.tracker_min_hits = node.get_parameter("tracker.min_hits").value
        cfg.tracker_iou = node.get_parameter("tracker.iou").value

        return cfg
