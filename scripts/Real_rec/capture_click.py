#!/usr/bin/env python3
"""RealSense 预览 + 按键保存当前帧。

用法:
    python capture_click.py                           # 默认 640x480, 输出 ./captures/
    python capture_click.py -o /root/yolo/dataset_real/scene01 -W 1280 -H 720

操作:
    空格 / s  → 保存当前帧
    q / Esc   → 退出

命名规则: MM_NNN.jpg（MM=分钟, NNN=该分钟内序号，自动重置）
例如: 32_001.jpg, 32_002.jpg, 33_001.jpg ...

依赖: pyrealsense2, opencv-python, numpy
"""

import argparse
import os
import signal
import sys
from datetime import datetime

import cv2
import numpy as np

# ── 全局状态 ──
_save_dir = ""
_seq = 0
_last_minute = -1
_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    print(f"\n  收到 {signal.Signals(signum).name}，退出...")
    _shutdown = True


def _save_frame(frame):
    """保存当前帧，分钟+序号命名。"""
    global _seq, _last_minute

    now = datetime.now()
    minute = now.minute

    if minute != _last_minute:
        _last_minute = minute
        _seq = 0

    _seq += 1
    fname = f"{minute:02d}_{_seq:03d}.jpg"
    fpath = os.path.join(_save_dir, fname)

    cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    ts = now.strftime("%H:%M:%S")
    print(f"  ✓ {ts}  →  {fname}")


def main():
    parser = argparse.ArgumentParser(
        description="RealSense 预览 + 按键保存帧"
    )
    parser.add_argument("-o", "--output", default="/root/yolo/dataset_real",
                        help="保存目录")
    parser.add_argument("-W", "--width", type=int, default=640)
    parser.add_argument("-H", "--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    global _shutdown, _save_dir
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _handle_signal)

    # ── 初始化 RealSense ──
    try:
        import pyrealsense2 as rs
    except ImportError:
        sys.exit("❌ 未安装 pyrealsense2: pip install pyrealsense2")

    ctx = rs.context()
    if len(ctx.query_devices()) == 0:
        sys.exit("❌ 未检测到 RealSense 设备")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height,
                         rs.format.bgr8, args.fps)
    profile = pipeline.start(config)
    intrinsics = profile.get_stream(rs.stream.color) \
                        .as_video_stream_profile().get_intrinsics()
    w, h = intrinsics.width, intrinsics.height
    print(f"✓ 设备就绪: {w}x{h}  |  fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")

    for _ in range(30):
        pipeline.wait_for_frames()
    print("✓ 自动曝光稳定完成")

    # ── 输出目录 ──
    _save_dir = args.output
    os.makedirs(_save_dir, exist_ok=True)
    print(f"✓ 保存目录: {os.path.abspath(_save_dir)}")

    total = 0
    print(f"\n  [空格/s] 保存  [q/Esc] 退出\n")

    save_flash = 0   # 保存后的视觉反馈帧数

    try:
        while not _shutdown:
            frame = np.asanyarray(
                pipeline.wait_for_frames().get_color_frame().get_data()
            )

            # 显示
            display = frame.copy()

            # 保存后短暂闪绿提示
            if save_flash > 0:
                cv2.putText(display, f"SAVED ({total})", (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                save_flash -= 1

            cv2.imshow("CAPTURE | Space/s=保存 q=退出", display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                print("\n  退出")
                break
            elif key in (ord(" "), ord("s")):   # 空格 或 s
                _save_frame(frame)
                total += 1
                save_flash = 15

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"✓ 共保存 {total} 张 → {os.path.abspath(_save_dir)}")


if __name__ == "__main__":
    main()