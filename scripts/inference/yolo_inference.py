# yolo_inference.py —— YOLO11-seg 推理封装
import time
import cv2
import numpy as np
from ultralytics import YOLO


class YOLOSegInference:
    """YOLO11-seg 推理器。"""

    def __init__(
        self,
        model_path: str = "/root/yolo/result/exp_2/last.pt",     # 模型权重路径
        imgsz: int = 640,                       # 输入图像尺寸
        conf: float = 0.5,                      # 置信度阈值
        iou: float = 0.4,                       # 非极大抑制 IoU 阈值
        device: str = "cuda",
    ):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device

        # 预热：因为第一次推理总是慢（CUDA context 初始化），所以先跑一帧废图
        print(f"[YOLO] 加载: {model_path}, 类别: {len(self.model.names)}, "
              f"device={device}")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = self.model(dummy, imgsz=imgsz, verbose=False, device=device)# 关闭日志
        print("[YOLO] 预热完成")

    def predict(self, image: np.ndarray):
        """单帧分割推理。

        Args:
            image: (H, W, 3) uint8, RGB 顺序

        Returns:
            {"objects": [{class_name, class_id, confidence, mask}, ...],
             "inference_time_ms": float}
            mask 是 (H, W) bool，原图尺寸
        """

        # # 用来检查输入的图像是否正确，曾出现过回调函数rgb通道错误的问题
        # cv2.imwrite(
        # '/root/yolo/rgb_test/test_ros.png',
        # image)

        # # 添加转换步骤  该步骤已补充到 ros_image_to_numpy() 函数中
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(
        # '/root/yolo/rgb_test/test_ros_turn.png',
        # image) 

        t0 = time.perf_counter()

        results = self.model(
            image,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            retina_masks=True,      # 输出高分辨率masks
            verbose=False,
        )

        elapsed = (time.perf_counter() - t0) * 1000
        result = results[0]

        objects = []
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy().astype(bool)        # 从显存中取masks张量到cpu，转换为numpy数组，按0.5b转bool类型
            # 三维 masks: (N, H, W)，N是检测到的目标数量，H、W是原图尺寸。内容全是 True/False，只有轮廓上是true

            classes = result.boxes.cls.cpu().numpy().astype(int)        # 取类别id
            confs = result.boxes.conf.cpu().numpy()                     # 取置信度

            for i in range(len(masks)):
                objects.append({
                    "class_name": self.model.names[classes[i]],
                    "class_id": classes[i],
                    "confidence": float(confs[i]),
                    "mask": masks[i],          # (H, W) bool
                })

        return {"objects": objects, "inference_time_ms": elapsed}