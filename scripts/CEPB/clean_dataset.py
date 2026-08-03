import os
import re
import random
from pathlib import Path

# ==================== 配置区 ====================
TARGET_DIRS = [
    "/root/gpufree-data/dataset_yolo/images/train",
    "/root/gpufree-data/dataset_yolo/images/val",
]
DRY_RUN = False          # True=只打印不真删，False=真删
SEED = 42               # 随机种子，固定可复现
# =================================================

# 匹配: {id}_view_{0|1|2}_{dir|point|spot}.jpg
PATTERN = re.compile(r"^(.+_view_\d+)_(dir|point|spot)\.jpg$")

random.seed(SEED)

for target_dir in TARGET_DIRS:
    if not os.path.isdir(target_dir):
        print(f"[SKIP] 目录不存在: {target_dir}")
        continue

    # 按 (场景_视角) 分组
    groups: dict[str, list[Path]] = {}
    for f in Path(target_dir).iterdir():
        if not f.is_file():
            continue
        m = PATTERN.match(f.name)
        if not m:
            continue
        base = m.group(1)          # e.g. "1000_view_0"
        groups.setdefault(base, []).append(f)

    kept = 0
    deleted = 0

    for base, files in groups.items():
        if len(files) <= 1:
            continue               # 只有一种光照，无需操作

        # 随机保留一个
        keep = random.choice(files)
        for f in files:
            if f == keep:
                print(f"[KEEP]  {f}")
                kept += 1
            else:
                if DRY_RUN:
                    print(f"[DEL?]  {f}")
                else:
                    f.unlink()
                    print(f"[DEL]   {f}")
                deleted += 1

    print(f"\n{'[DRY RUN] ' if DRY_RUN else ''}{target_dir}:"
          f" 保留 {kept} 张, 删除 {deleted} 张\n")