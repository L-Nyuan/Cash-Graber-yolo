#!/usr/bin/env python3
"""阶段 3: 解冻全模型精细微调 — 阶段 2 best.pt → 极低 lr 全域调整。

用法:
    python train_stage3_unfreeze.py
"""

from ultralytics import YOLO

# ========== 配置区 ==========
STAGE2_BEST = "/root/yolo/scripts/runs/stage2_freeze/weights/best.pt"  # 阶段 2 产出
DATA  = "/root/yolo/dataset_real_remapped/data.yaml"                   # 混合数据集
PROJECT = "runs"
NAME    = "stage3_unfreeze"

EPOCHS   = 20
IMSZ     = 640
BATCH    = 16
DEVICE   = 0
WORKERS  = 8
# =============================

model = YOLO(STAGE2_BEST)

results = model.train(
    data    = DATA,
    epochs  = EPOCHS,
    imgsz   = IMSZ,
    batch   = BATCH,
    device  = DEVICE,
    workers = WORKERS,
    project = PROJECT,
    name    = NAME,

    # 分割任务
    task    = "segment",

    # ── 全解冻 ──
    freeze  = 0,              # backbone + neck + head 全部可训练

    # ── 学习率（极低：阶段 2 的 1/4）──
    lr0     = 0.00005,
    lrf     = 0.001,
    optimizer = "AdamW",
    momentum  = 0.937,
    weight_decay = 0.0005,
    warmup_epochs = 1,

    # ── 数据增强（进一步降低强度，防止 backbone 过拟合）──
    hsv_h     = 0.01,         # 比阶段 2 更低
    hsv_s     = 0.5,
    hsv_v     = 0.3,
    degrees   = 3.0,
    translate = 0.05,
    scale     = 0.3,
    shear     = 1.0,
    fliplr    = 0.5,
    mosaic    = 0,
    close_mosaic = 10,        # 最后 10 epoch 关闭
    mixup     = 0.0,
    copy_paste = 0.0,

    # ── 保存 ──
    save      = True,
    save_period = 5,
    exist_ok  = True,
    resume    = False,

    cache     = "disk",
)