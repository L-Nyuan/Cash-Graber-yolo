#!/usr/bin/env python3
"""阶段 2: 冻结 backbone 微调 — CEPB checkpoint → 混合数据（真实 150 + CEPB 150）。

用法:
    python train_stage2_freeze.py
"""

from ultralytics import YOLO

# ========== 配置区 ==========
CEPB_WEIGHTS = "/root/gpufree-data/runs/exp01_baseline/weights/last.pt"  # 阶段 1 产出
DATA  = "/root/yolo/dataset_real_remapped/data.yaml"                     # 混合数据集
PROJECT = "runs"
NAME    = "stage2_freeze"

EPOCHS   = 50
IMSZ     = 640
BATCH    = 16           # 混合数据 ~300 张，batch 小一点
DEVICE   = 0
WORKERS  = 8
# =============================

model = YOLO(CEPB_WEIGHTS)   # 从 CEPB 训练结果继续，不是 yolo11m-seg.pt

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

    # ── 冻结 backbone ──
    freeze  = 12,             # backbone 权重不更新，只训 neck + head

    # ── 学习率（关键：远低于 CEPB 训练）──
    lr0     = 0.0002,         # CEPB 训练的 1/10
    lrf     = 0.01,
    optimizer = "AdamW",
    momentum  = 0.937,
    weight_decay = 0.0005,
    warmup_epochs = 3,

    # ── 数据增强（真实域变化小，降低增强强度）──
    hsv_h     = 0.015,
    hsv_s     = 0.7,
    hsv_v     = 0.4,          # 比 CEPB 低，真实场景亮度变化不宜太大
    degrees   = 5.0,          # 允许小角度旋转
    translate = 0.1,
    scale     = 0.5,
    shear     = 2.0,
    fliplr    = 0.5,
    mosaic    = 1.0,          # 开启 mosaic 增强小数据集
    close_mosaic = 30,        # 最后 30 epoch 关闭，保证分割边界质量
    mixup     = 0.0,
    copy_paste = 0.0,

    # ── 保存 ──
    save      = True,
    save_period = 10,
    exist_ok  = True,
    resume    = False,

    cache     = "disk",
)