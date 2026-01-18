"""
進程守護模組 - 防止孤兒進程

確保所有 multiprocessing 子進程在主進程結束時正確清理，
即使是被 Ctrl+C 中斷。

使用方式：
    from src.utils.process_guard import ProcessGuard

    # 方式 1: 自動守護（推薦）
    with ProcessGuard():
        pool = multiprocessing.Pool(4)
        # ... 使用 pool ...

    # 方式 2: 全域啟用
    ProcessGuard.install()  # 在程式入口呼叫一次
"""

import atexit
import signal
import os
import logging
from typing import Set, Optional
from multiprocessing import current_process
import weakref

logger = logging.getLogger(__name__)


class ProcessGuard:
    """進程守護器

    追蹤所有子進程，確保在主進程結束時清理。
    """

    _instance: Optional['ProcessGuard'] = None
    _installed: bool = False
    _child_pids: Set[int] = set()
    _original_sigint: Optional[signal.Handlers] = None
    _original_sigterm: Optional[signal.Handlers] = None

    def __new__(cls):
        """單例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not ProcessGuard._installed:
            self._setup_handlers()

    def _setup_handlers(self):
        """設定信號處理器和 atexit"""
        if ProcessGuard._installed:
            return

        # 只在主進程設定
        if current_process().name != 'MainProcess':
            return

        # 保存原始處理器
        ProcessGuard._original_sigint = signal.getsignal(signal.SIGINT)
        ProcessGuard._original_sigterm = signal.getsignal(signal.SIGTERM)

        # 設定新處理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # 註冊 atexit
        atexit.register(self._cleanup_all)

        ProcessGuard._installed = True
        logger.debug("ProcessGuard 已安裝")

    def _signal_handler(self, signum, frame):
        """信號處理器"""
        logger.info(f"收到信號 {signum}，清理子進程...")
        self._cleanup_all()

        # 呼叫原始處理器
        if signum == signal.SIGINT and ProcessGuard._original_sigint:
            if callable(ProcessGuard._original_sigint):
                ProcessGuard._original_sigint(signum, frame)
            else:
                raise KeyboardInterrupt
        elif signum == signal.SIGTERM and ProcessGuard._original_sigterm:
            if callable(ProcessGuard._original_sigterm):
                ProcessGuard._original_sigterm(signum, frame)

    def _cleanup_all(self):
        """清理所有已知的子進程"""
        if not ProcessGuard._child_pids:
            return

        logger.info(f"清理 {len(ProcessGuard._child_pids)} 個子進程")

        for pid in list(ProcessGuard._child_pids):
            try:
                os.kill(pid, signal.SIGTERM)
                logger.debug(f"已終止進程 {pid}")
            except (ProcessLookupError, PermissionError):
                pass  # 進程已結束或無權限
            except Exception as e:
                logger.warning(f"終止進程 {pid} 失敗: {e}")

        ProcessGuard._child_pids.clear()

    @classmethod
    def register_pid(cls, pid: int):
        """註冊子進程 PID"""
        cls._child_pids.add(pid)
        logger.debug(f"註冊子進程 PID: {pid}")

    @classmethod
    def unregister_pid(cls, pid: int):
        """取消註冊子進程 PID"""
        cls._child_pids.discard(pid)
        logger.debug(f"取消註冊子進程 PID: {pid}")

    @classmethod
    def install(cls):
        """全域安裝進程守護"""
        cls()

    @classmethod
    def cleanup_orphans(cls) -> int:
        """清理當前用戶的孤兒 Python multiprocessing 進程

        Returns:
            清理的進程數量
        """
        import subprocess

        try:
            # 找出 multiprocessing 孤兒進程 (PPID=1)
            result = subprocess.run(
                ['pgrep', '-f', 'multiprocessing'],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                return 0

            pids = result.stdout.strip().split('\n')
            cleaned = 0

            for pid_str in pids:
                if not pid_str:
                    continue

                pid = int(pid_str)

                # 檢查 PPID 是否為 1（孤兒）
                try:
                    ppid_result = subprocess.run(
                        ['ps', '-o', 'ppid=', '-p', str(pid)],
                        capture_output=True,
                        text=True
                    )
                    ppid = int(ppid_result.stdout.strip())

                    if ppid == 1:  # 孤兒進程
                        os.kill(pid, signal.SIGTERM)
                        cleaned += 1
                        logger.info(f"已清理孤兒進程: {pid}")

                except (ValueError, ProcessLookupError, PermissionError):
                    continue

            return cleaned

        except Exception as e:
            logger.warning(f"清理孤兒進程失敗: {e}")
            return 0

    def __enter__(self):
        """Context manager 進入"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 退出"""
        self._cleanup_all()
        return False


# 便利函數
def install_process_guard():
    """安裝進程守護（建議在程式入口呼叫）"""
    ProcessGuard.install()


def cleanup_orphan_processes() -> int:
    """清理孤兒進程

    Returns:
        清理的進程數量
    """
    return ProcessGuard.cleanup_orphans()


# ============= 測試 =============

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("測試 ProcessGuard")
    print("=" * 40)

    # 測試清理孤兒進程
    cleaned = cleanup_orphan_processes()
    print(f"清理了 {cleaned} 個孤兒進程")

    # 測試 context manager
    print("\n測試 context manager...")
    with ProcessGuard():
        print("ProcessGuard 已啟用")

    print("\n✓ 測試完成")
