# node_config.py —— 节点参数声明与解析
#
# 集中 declare_parameter + get_parameter，节点 __init__ 只消费 NodeConfig。
# 参数名与默认值以 declare_parameter 为准（命令行 -p 覆盖方式保持不变）。
#
# ── 如何修改参数 ──────────────────────────────────────────
# 两种方式等价，推荐直接改这里的默认值（改动会随代码提交、对所有启动方式生效）：
#   1) 改本文件的默认值（如 node.declare_parameter("model.conf", 0.8)）；
#   2) 启动时用 --ros-args -p 组名.参数:=值 覆盖（临时生效，不改代码）。
# 注意：yolo_inference_node_cloud.py 只消费 NodeConfig，不直接读参数，
# 因此这里是最新值的唯一来源；改完后新启动的节点即生效。


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
        # mode: 节点工作模式。
        #   production —— 只发布 /yolo/detections + 按需 /yolo/object_cloud；
        #   debug      —— 额外发布 /yolo/debug_cloud + /yolo/markers，
        #                  且默认自动打开 rviz2（可用 debug.launch_rviz 关闭）。
        # 传了其他值会告警并回退到 production。
        node.declare_parameter("mode", "production")
        # model.* —— YOLO11-seg 模型
        # model.path: 权重文件路径。最终部署权重；切换实验权重时改这里。
        node.declare_parameter("model.path", "/root/yolo/result/final/best.pt")
        # model.imgsz: 推理输入正方形边长（像素）。640 是 YOLO11 常用值；
        #   调大对小目标更友好但推理更慢，调小反之。
        node.declare_parameter("model.imgsz", 640)
        # model.conf: 置信度阈值（0~1）。越高误检越少但易漏检；实机场景
        #   空旷时降为 0.5 可显著提高召回（当前相机视野常见 0.8 漏检）。
        node.declare_parameter("model.conf", 0.8)
        # model.iou: NMS 的 IoU 阈值（0~1）。越大，重叠度高的框越可能都被保留。
        node.declare_parameter("model.iou", 0.7)
        # tracker.* —— IoU 多目标追踪器
        # tracker.max_age: track 连续多少帧未匹配后删除（帧数）。越大 ID 越稳定，
        #   但目标离开视野后旧 ID 存活越久、恢复越慢。
        node.declare_parameter("tracker.max_age", 30)
        # tracker.min_hits: 新 track 需连续命中多少帧才转为"确认"并输出。
        #   用于抑制偶发误检造成的闪烁假目标。
        node.declare_parameter("tracker.min_hits", 3)
        # tracker.iou: 帧间匹配的 IoU 阈值（0~1）。手腕相机移动快时建议调低
        #   （如 0.2），相机相对静止可保持 0.3。
        node.declare_parameter("tracker.iou", 0.3)
        # cloud.* —— 点云后处理
        # cloud.mask_erode: mask 腐蚀迭代次数（3x3 核）。>=1 可削掉分割边缘的
        #   误报像素，但太大会把物体真实边缘点一起削掉。
        node.declare_parameter("cloud.mask_erode", 1)
        # cloud.sor: 是否启用 open3d 统计离群点移除（Statistical Outlier Removal）。
        #   环境没有 open3d 时会自动跳过，不影响主流程。
        node.declare_parameter("cloud.sor", True)
        # sync.* —— 彩色图与点云的时间同步
        # sync.tolerance: 彩色图时间戳与点云时间戳的最大容差（秒）。
        #   相机/手腕移动快时收紧（如 0.03s），静止可放宽；超出容差宁可跳过裁剪，
        #   也不把旧 mask 套到新点云上。
        node.declare_parameter("sync.tolerance", 0.05)
        # sync.require_cloud: true 时，没有容差内匹配点云就跳过本帧裁剪
        #   （先短暂等待，超时丢弃）；false 时即使时间戳超出容差，
        #   也会用缓冲里最近的一帧点云继续裁剪（不等待、不丢弃）。
        node.declare_parameter("sync.require_cloud", True)
        # sync.buffer_size: 点云缓冲帧数（用于最近邻时间匹配）。
        #   默认 30 ≈ 1 秒 @30fps；调大可抗时间抖动，但占用内存更多。
        node.declare_parameter("sync.buffer_size", 30)
        # sync.pending_wait: 无匹配点云时最多等待的秒数，等对应点云回调到达；
        #   超时则丢弃该帧（避免无限等待）。设 0 表示不等待、直接丢弃。
        node.declare_parameter("sync.pending_wait", 0.12)
        # sync.debug: 是否输出 [sync]/[crop] 调试日志。debug 模式强制开启，
        #   production 默认关闭；排查同步问题时可在 production 手动开 true。
        node.declare_parameter("sync.debug", False)
        # topic.* —— 话题名（改相机命名空间时覆盖）
        # topic.cloud: RealSense 彩色点云话题（带相机命名空间）。
        node.declare_parameter(
            "topic.cloud", "/Wrist_Camera/d435i/depth/color/points")
        # topic.color: 彩色图像话题。
        node.declare_parameter(
            "topic.color", "/Wrist_Camera/d435i/color/image_raw")
        # topic.info: CameraInfo 话题（提供相机内参，用于 mask 投影裁剪）。
        node.declare_parameter(
            "topic.info", "/Wrist_Camera/d435i/color/camera_info")
        # debug.* —— 调试输出（仅 mode=debug 生效）
        # debug.cloud_hz: /yolo/debug_cloud 的发布频率上限（Hz），节流用；
        #   内部按 max(值, 1.0) 计算间隔，设 0 也不会死循环。
        node.declare_parameter("debug.cloud_hz", 15.0)
        # debug.dir: 调试输出目录（rviz2 日志、调试图等）。debug 模式自动创建。
        node.declare_parameter("debug.dir", "/root/yolo/yolo_debug")
        # debug.launch_rviz: debug 模式是否自动打开 rviz2。无显示器时设 false。
        node.declare_parameter("debug.launch_rviz", True)
        # debug.rviz_config: 自动启动 rviz2 时加载的配置文件路径。
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
