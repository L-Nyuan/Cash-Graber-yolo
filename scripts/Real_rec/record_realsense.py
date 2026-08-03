#!/usr/bin/env python3
"""录制 RealSense D435i RGB 视频为 H.264 MP4。

用法:
    python record_realsense.py                           # 默认 640x480@30fps H.264
    python record_realsense.py -o /root/videos/scene01.mp4 -d 120
    python record_realsense.py -W 1280 -H 720 --fps 15

停止方式（任意一种）:  q / Ctrl+C / 关终端
编码: H.264 (libx264) via ffmpeg，VSCode / 浏览器可直接预览。

依赖: pyrealsense2, opencv-python, numpy, ffmpeg
"""

import argparse
import os
import signal
import subprocess as sp
import sys
import time
from datetime import datetime

import cv2
import numpy as np

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    print(f"\n  收到 {signal.Signals(signum).name}，安全关闭...")
    _shutdown = True


def build_ffmpeg_cmd(out_path, w, h, fps, codec="libx264", crf=23):
    """构造 ffmpeg 命令行，stdin 喂 raw BGR 帧。"""
    return [
        "ffmpeg",
        "-y",                        # 覆盖已有文件
        "-f", "rawvideo",            # 输入格式：原始帧
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "bgr24",         # OpenCV/RealSense 原生格式
        "-r", str(fps),
        "-i", "-",                   # stdin
        "-c:v", codec,
        "-preset", "ultrafast",      # 实时编码，低延迟
        "-crf", str(crf),            # 质量 (0-51, 越小越好, 23 常用)
        "-pix_fmt", "yuv420p",       # 输出像素格式，保证兼容性
        out_path,
    ]


def main():
    parser = argparse.ArgumentParser(description="录制 RealSense D435i RGB 视频 (H.264)")
    parser.add_argument("-o", "--output", default="/root/yolo/videos",
                        help="输出目录（自动命名）或指定 .mp4 路径")
    parser.add_argument("-W", "--width", type=int, default=640)
    parser.add_argument("-H", "--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("-d", "--duration", type=int, default=0,
                        help="录制时长（秒），0=手动停止")
    parser.add_argument("--crf", type=int, default=23,
                        help="H.264 质量，越小越好 (18-28)，默认 23")
    parser.add_argument("--no-preview", action="store_true",
                        help="不显示预览窗口")
    args = parser.parse_args()

    # ── 信号处理 ──
    global _shutdown
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
    print(f"✓ 设备就绪: {w}x{h}")

    for _ in range(30):
        pipeline.wait_for_frames()
    print("✓ 自动曝光稳定完成")

    # ── 输出路径 ──
    out = args.output
    if os.path.isdir(out) or not out.endswith(".mp4"):
        os.makedirs(out, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out, f"realsense_{ts}.mp4")
    else:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        out_path = out
    print(f"✓ 输出: {out_path}")

    # ── 检查 ffmpeg + 编码器 ──
    try:
        sp.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, sp.CalledProcessError):
        sys.exit("❌ 未安装 ffmpeg: apt install ffmpeg -y")

    # 探测可用编码器
    result = sp.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
    if "libx264" in result.stdout:
        codec, codec_label = "libx264", "H.264 (libx264)"
    else:
        codec, codec_label = "mpeg4", "MPEG-4 (libx264 不可用)"
    print(f"✓ 编码: {codec_label}")

    # ── 启动 ffmpeg 子进程 ──
    ffmpeg_cmd = build_ffmpeg_cmd(out_path, w, h, args.fps, codec, args.crf)
    ffmpeg_proc = sp.Popen(ffmpeg_cmd, stdin=sp.PIPE, stderr=sp.DEVNULL)

    # ── 录制 ──
    stops = []
    if not args.no_preview:
        stops.append("q")
    stops.append("Ctrl+C/关终端" if args.duration == 0
                 else f"{args.duration}s后自动")
    print(f"\n  开始录制 [{', '.join(stops)}]\n")

    t0 = time.time()
    cnt = 0
    last_print = t0

    try:
        while not _shutdown:
            frame = np.asanyarray(
                pipeline.wait_for_frames().get_color_frame().get_data()
            )
            ffmpeg_proc.stdin.write(frame.tobytes())
            cnt += 1

            now = time.time()
            if now - last_print >= 1.0:
                elapsed = now - t0
                print(f"\r  已录 {cnt:6d} 帧  |  {elapsed:6.1f}s  |  "
                      f"{cnt / elapsed:.1f} fps", end="")
                last_print = now

            if not args.no_preview:
                cv2.imshow("REC (q 停止)", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    print("\n  q 按下，停止录制")
                    break

            if args.duration > 0 and (now - t0) >= args.duration:
                print("\n  到达设定时长")
                break

    except KeyboardInterrupt:
        pass
    except BrokenPipeError:
        print("\n  ffmpeg 管道断开（编码失败）")

    finally:
        # 关闭 ffmpeg stdin → ffmpeg 完成封装 → 视频文件可播放
        try:
            ffmpeg_proc.stdin.close()
        except Exception:
            pass
        ffmpeg_proc.wait(timeout=5)

        pipeline.stop()
        cv2.destroyAllWindows()

        elapsed = time.time() - t0
        if cnt > 0 and os.path.exists(out_path):
            mb = os.path.getsize(out_path) / (1024 * 1024)
            real_fps = cnt / elapsed if elapsed > 0 else 0
            print(f"\n  ✓ 完成: {out_path}")
            print(f"    帧数: {cnt}  |  时长: {elapsed:.1f}s  |  "
                  f"大小: {mb:.1f}MB  |  FPS: {real_fps:.1f}")


if __name__ == "__main__":
    main()