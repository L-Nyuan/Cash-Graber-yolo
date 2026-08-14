# rviz_launcher.py —— debug 模式 rviz2 子进程生命周期管理
#
# 拆分计划 Phase 4（v2）：
#   管理 rviz2 的启动、后台观察与退出清理；含 Qt 插件环境修复
#   （conda/cv2 污染 QT_QPA_PLATFORM_PLUGIN_PATH 导致 rviz2 abort 的兼容处理）。
#   生命周期与节点一致；watch 线程为 daemon，不阻塞节点退出。

import os
import subprocess
import threading
from typing import Optional


class RvizLauncher:
    """管理 rviz2 子进程：启动、后台观察、退出清理。"""

    def __init__(self, rviz_config: str, debug_dir: str, logger):
        self._rviz_config = rviz_config
        self._debug_dir = debug_dir
        self._logger = logger
        self._rviz_proc: Optional[subprocess.Popen] = None
        self._rviz_stopping = False

    @property
    def config(self) -> str:
        return self._rviz_config

    @property
    def running(self) -> bool:
        """rviz2 是否仍在运行。"""
        return self._rviz_proc is not None and self._rviz_proc.poll() is None

    def start(self):
        """debug 模式下自动启动 rviz2 并加载调试配置。"""
        cfg = self._rviz_config
        if not os.path.isfile(cfg):
            self._logger.warn(
                f"debug.launch_rviz=true 但配置不存在: {cfg}，跳过 rviz 启动")
            return
        rviz_env = self._build_rviz_env()
        self._logger.info(
            f"准备启动 rviz2 | DISPLAY={rviz_env.get('DISPLAY', '(未设置)')} "
            f"XAUTHORITY={rviz_env.get('XAUTHORITY', '(未设置)')} "
            f"QT_QPA_PLATFORM_PLUGIN_PATH="
            f"{rviz_env.get('QT_QPA_PLATFORM_PLUGIN_PATH', '(系统默认)')}")
        log_path = os.path.join(self._debug_dir, "rviz2.log")
        try:
            with open(log_path, "w") as log_f:
                self._rviz_proc = subprocess.Popen(
                    ["rviz2", "-d", cfg],
                    env=rviz_env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            proc = self._rviz_proc
            self._logger.info(
                f"已启动 rviz2 (pid={proc.pid})，加载配置: {cfg}，"
                f"输出见 {log_path}")
            # 后台观察：rviz2 若异常退出（如显示连接失败），把它的输出打出来
            threading.Thread(
                target=self._watch_rviz, args=(proc, log_path),
                daemon=True).start()
        except FileNotFoundError:
            self._logger.warn(
                "未找到 rviz2 命令，跳过（请确认已 source ROS 环境）")
            self._rviz_proc = None
        except Exception as e:
            self._logger.warn(f"rviz2 启动失败: {e}")
            self._rviz_proc = None

    @staticmethod
    def _build_rviz_env() -> dict:
        """构造 rviz2 子进程环境，清除 conda/cv2 污染的 Qt 变量。

        在 conda 环境里 `import cv2` 会把 QT_QPA_PLATFORM_PLUGIN_PATH 指向
        cv2 自带的 Qt 插件目录，该插件与系统 Qt 版本不兼容，导致 rviz2
        启动即 abort（rc=-6）。这里改回系统 Qt 插件目录，并防御性清理
        LD_LIBRARY_PATH 中的 conda 库路径。
        """
        env = os.environ.copy()
        for k in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR",
                  "QT_PLUGIN_PATH"):
            env.pop(k, None)
        system_plugins = "/usr/lib/x86_64-linux-gnu/qt5/plugins"
        if os.path.isdir(system_plugins):
            env["QT_QPA_PLATFORM_PLUGIN_PATH"] = system_plugins
        libs = env.get("LD_LIBRARY_PATH", "")
        clean = [
            p for p in libs.split(":")
            if p and not any(s in p for s in ("/miniconda3/", "/anaconda3/",
                                              "/conda/"))
        ]
        if clean:
            env["LD_LIBRARY_PATH"] = ":".join(clean)
        return env

    def _watch_rviz(self, proc, log_path):
        """等待 rviz2 退出，异常退出时输出其日志末尾。"""
        rc = proc.wait()
        if self._rviz_stopping:
            return
        if rc != 0:
            self._logger.warn(
                f"rviz2 异常退出 rc={rc}，输出见 {log_path}:\n"
                f"{self._read_tail(log_path)}")
        else:
            self._logger.info(f"rviz2 已退出（rc=0），输出见 {log_path}")

    @staticmethod
    def _read_tail(path: str, n: int = 20) -> str:
        """读取文件末尾 n 行，用于输出 rviz2 的报错。"""
        try:
            with open(path) as f:
                lines = f.readlines()
            return "".join(lines[-n:]).strip() or "(日志为空)"
        except Exception:
            return "(无法读取 rviz2 日志)"

    def stop(self):
        """节点退出时关闭自动启动的 rviz2，避免残留孤儿进程。"""
        proc = self._rviz_proc
        if proc is not None and proc.poll() is None:
            self._rviz_stopping = True
            self._logger.info("节点退出，关闭 rviz2")
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._logger.warn("rviz2 未在 5 秒内退出，强制结束")
                proc.kill()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
        self._rviz_proc = None
