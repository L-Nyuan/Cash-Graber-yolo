from ultralytics import YOLO

# ========== 配置区 ==========
MODEL = "yolo11n-seg.pt"          # 预训练权重
DATA  = "/root/dataset_yolo/dataset.yaml"
PROJECT = "runs"             # 所有实验存这里
NAME    = "exp01_baseline"        # 本次实验名

EPOCHS   = 120
IMSZ     = 640                  # 图像尺寸
BATCH    = 32                   # 按 GPU 显存调
DEVICE   = 0                    # 多卡写 [0,1]，单卡写 0
WORKERS  = 8                    # 做图像预处理的cpu数量，防止gpu空闲
# =============================

model = YOLO(MODEL)

results = model.train(
    data    = DATA,
    epochs  = EPOCHS,
    imgsz   = IMSZ,
    batch   = BATCH,
    device  = DEVICE,
    workers = WORKERS,
    project = PROJECT,
    name    = NAME,

    # 分割相关
    task    = "segment",

    # 数据增强（先默认）
    hsv_h=0.015,  hsv_s=0.7,  hsv_v=0.55,               # 色调、饱和度、明度的扰动
    degrees=0.0,  translate=0.1,                        # 随机旋转角度范围、平移范围
    scale=0.5,                                          # 随机缩放
    fliplr=0.5,                                         # 随机水平翻转
    mosaic=1.0,                                         # 随机马赛克增强,1.0表示100%概率使用

    # 优化器
    optimizer="AdamW",  lr0=0.001,  lrf=0.01,                   # 初始学习率、最终学习率
    momentum=0.937,  weight_decay=0.0005,  warmup_epochs=3,     # 动量、L2正则化、预热轮数

    # 保存策略
    save=False,  save_period=10,     # save=Ture表示全程保存模型，save_period=10表示每10轮保存一次
    exist_ok=True,                  # 覆盖同名目录
    resume=False,                   # 断点续训改成 True
)