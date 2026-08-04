#!/usr/bin/env python3
"""将 Roboflow 导出数据集（字母序标签）的 class id 重新映射为目标数据集顺序。

用法:
    python remap_labels.py \
        -s "/root/yolo/My First Project.yolov11" \
        -t /root/dataset_yolo/data.yaml \
        -o /root/yolo/My_First_Project_remapped

    python remap_labels.py -s ./src -t ./target/data.yaml    # 原地覆盖（备份为 .bak）
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def parse_names(yaml_path):
    """从 YAML 读 names 字段，返回 {class_id: name}。"""
    names = {}
    with open(yaml_path) as f:
        in_names = False
        for line in f:
            line = line.rstrip()
            if line.startswith("names"):
                in_names = True
                continue
            if in_names:
                stripped = line.strip()
                if not stripped:
                    break
                # 格式: "0: Cheez-it" 或 "0: meat_can" 或 "'Cheez-it'," (列表格式)
                if ":" in stripped:
                    k, v = stripped.split(":", 1)
                    names[int(k.strip())] = v.strip().strip("'\"")
                else:
                    # 方括号列表格式: ['Cheez-it', 'Clamp', ...]
                    pass
    return names


def parse_names_from_list(yaml_path):
    """处理列表格式的 names: ['Cheez-it', 'Clamp', ...]"""
    import re
    with open(yaml_path) as f:
        content = f.read()

    # 找 names: 后面的列表
    m = re.search(r'names:\s*\[([^\]]+)\]', content)
    if not m:
        return {}

    items = re.findall(r"'([^']+)'|\"([^\"]+)\"", m.group(1))
    names = {}
    for i, (s1, s2) in enumerate(items):
        names[i] = s1 or s2
    return names


def load_names(yaml_path):
    """自动检测格式并解析。"""
    names = parse_names_from_list(yaml_path)
    if names:
        return names
    return parse_names(yaml_path)


def build_mapping(src_names, tgt_names):
    """按名称匹配，返回 {src_id: tgt_id}。

    src_names: {id: name}
    tgt_names: {id: name}
    """
    # 反转目标：{name: id}
    tgt_by_name = {v: k for k, v in tgt_names.items()}

    mapping = {}
    for src_id, name in src_names.items():
        if name in tgt_by_name:
            mapping[src_id] = tgt_by_name[name]
        else:
            print(f"  ⚠ 源类别 '{name}' (id={src_id}) 在目标数据集中不存在，跳过")

    return mapping


def remap_file(src_path, dst_path, mapping):
    """读取一个 label 文件，重映射 class id，写入新文件。"""
    with open(src_path) as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        old_id = int(parts[0])
        if old_id in mapping:
            parts[0] = str(mapping[old_id])
            new_lines.append(" ".join(parts) + "\n")
        # 如果不在 mapping 中 → 跳过该标注（未知类别）

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w") as f:
        f.writelines(new_lines)


def main():
    parser = argparse.ArgumentParser(
        description="YOLO 标签 class id 重映射（按名称匹配）"
    )
    parser.add_argument("-s", "--src", required=True,
                        help="源数据集目录（含 data.yaml, images/, labels/）")
    parser.add_argument("-t", "--target-yaml", required=True,
                        help="目标 data.yaml 路径（定义目标 class 顺序）")
    parser.add_argument("-o", "--output",
                        help="输出目录（默认：源目录原地覆盖，保留 .bak 备份）")
    parser.add_argument("--splits", nargs="+",
                        default=["train", "val", "test"],
                        help="要处理的 split 名称，默认 train val test")
    args = parser.parse_args()

    # ── 加载源 names ──
    src_yaml = os.path.join(args.src, "data.yaml")
    if not os.path.exists(src_yaml):
        sys.exit(f"❌ 未找到: {src_yaml}")
    src_names = load_names(src_yaml)
    print(f"源: {len(src_names)} 类")
    for k, v in src_names.items():
        print(f"  {k}: {v}")

    # ── 加载目标 names ──
    if not os.path.exists(args.target_yaml):
        sys.exit(f"❌ 未找到: {args.target_yaml}")
    tgt_names = load_names(args.target_yaml)
    print(f"\n目标: {len(tgt_names)} 类")
    for k, v in sorted(tgt_names.items()):
        print(f"  {k}: {v}")

    # ── 建立映射 ──
    mapping = build_mapping(src_names, tgt_names)
    if not mapping:
        sys.exit("❌ 没有找到任何可映射的类别，请检查 names")

    print(f"\n映射: {len(mapping)} 个类别")
    for src_id, tgt_id in sorted(mapping.items()):
        print(f"  {src_names[src_id]}: {src_id} → {tgt_id}")

    # ── 确认 ──
    print()
    inplace = args.output is None
    if inplace:
        print("⚠  原地覆盖模式 — 源文件将备份为 .bak")
    else:
        print(f"输出目录: {args.output}")

    # ── 处理每个 split ──
    total_files = 0
    for split in args.splits:
        src_label_dir = os.path.join(args.src, "labels", split)
        if not os.path.isdir(src_label_dir):
            print(f"  ⏭  跳过 {split}（目录不存在: {src_label_dir}）")
            continue

        label_files = list(Path(src_label_dir).glob("*.txt"))
        if not label_files:
            print(f"  ⏭  {split}: 无 .txt 文件")
            continue

        for lf in label_files:
            if inplace:
                # 先备份
                bak = str(lf) + ".bak"
                if not os.path.exists(bak):
                    shutil.copy2(lf, bak)
                remap_file(lf, lf, mapping)
            else:
                rel = lf.relative_to(os.path.join(args.src, "labels"))
                dst = os.path.join(args.output, "labels", str(rel))
                remap_file(lf, dst, mapping)
            total_files += 1

        # 同时复制 images
        src_img_dir = os.path.join(args.src, "images", split)
        if os.path.isdir(src_img_dir) and not inplace:
            dst_img_dir = os.path.join(args.output, "images", split)
            os.makedirs(dst_img_dir, exist_ok=True)
            for img in Path(src_img_dir).iterdir():
                if img.is_file():
                    dst_img = os.path.join(dst_img_dir, img.name)
                    if not os.path.exists(dst_img):
                        os.symlink(os.path.abspath(img), dst_img)
                        # shutil.copy2  if you prefer copy over symlink
                        # shutil.copy2(img, dst_img)

        print(f"  ✓ {split}: {len(label_files)} 个标签文件")

    # ── 生成新的 data.yaml ──
    if args.output:
        # 复制源 yaml 但替换 names
        dst_yaml = os.path.join(args.output, "data.yaml")
        # 按目标顺序重建 names，只保留实际存在的类别
        with open(dst_yaml, "w") as f:
            f.write(f"path: {os.path.abspath(args.output)}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            if os.path.isdir(os.path.join(args.src, "labels", "test")):
                f.write("test: images/test\n")
            f.write(f"\nnc: {len(tgt_names)}\n")
            f.write("names:\n")
            for i in sorted(tgt_names.keys()):
                f.write(f"    {i}: {tgt_names[i]}\n")
        print(f"  ✓ 新 data.yaml → {dst_yaml}")

    print(f"\n✓ 完成: {total_files} 个标签文件，映射 {len(mapping)} 个类别")
    print("\n映射汇总:")
    for src_id, tgt_id in sorted(mapping.items()):
        print(f"  {src_names[src_id]:20s}  {src_id} → {tgt_id}")


if __name__ == "__main__":
    main()