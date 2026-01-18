"""
系統狀態監控

使用 psutil 監控 CPU、記憶體使用情況。
"""

import asyncio
import logging
from typing import Optional, Callable, Awaitable
from dataclasses import dataclass

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)


@dataclass
class SystemSnapshot:
    """系統狀態快照"""
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    gpu_percent: Optional[float] = None
    active_workers: int = 0


class SystemStats:
    """系統狀態監控器"""

    def __init__(self, interval_seconds: float = 1.0):
        """
        Args:
            interval_seconds: 監控間隔（秒）
        """
        self.interval = interval_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._callback: Optional[Callable[[SystemSnapshot], Awaitable[None]]] = None

    def get_snapshot(self) -> SystemSnapshot:
        """獲取當前系統狀態快照"""
        if not HAS_PSUTIL:
            return SystemSnapshot(
                cpu_percent=0.0,
                memory_mb=0.0,
                memory_percent=0.0,
            )

        # CPU 使用率
        cpu_percent = psutil.cpu_percent(interval=None)

        # 記憶體使用
        memory = psutil.virtual_memory()
        memory_mb = memory.used / (1024 * 1024)
        memory_percent = memory.percent

        # GPU（如果可用）
        gpu_percent = self._get_gpu_percent()

        # 活動 worker 數量
        active_workers = self._count_workers()

        return SystemSnapshot(
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            gpu_percent=gpu_percent,
            active_workers=active_workers,
        )

    def _get_gpu_percent(self) -> Optional[float]:
        """嘗試獲取 GPU 使用率"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=1.0
            )
            if result.returncode == 0:
                return float(result.stdout.strip().split('\n')[0])
        except Exception:
            pass
        return None

    def _count_workers(self) -> int:
        """計算活動的 worker 進程數"""
        if not HAS_PSUTIL:
            return 0

        try:
            current_process = psutil.Process()
            children = current_process.children(recursive=True)
            return len([c for c in children if c.is_running()])
        except Exception:
            return 0

    async def start_monitoring(
        self,
        callback: Callable[[SystemSnapshot], Awaitable[None]]
    ):
        """開始監控循環

        Args:
            callback: 每次快照後呼叫的回調函數
        """
        if self._running:
            logger.warning("監控已在運行中")
            return

        self._callback = callback
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(f"系統監控已啟動，間隔: {self.interval}s")

    async def stop_monitoring(self):
        """停止監控"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("系統監控已停止")

    async def _monitor_loop(self):
        """監控循環"""
        while self._running:
            try:
                snapshot = self.get_snapshot()
                if self._callback:
                    await self._callback(snapshot)
            except Exception as e:
                logger.error(f"監控錯誤: {e}")

            await asyncio.sleep(self.interval)


# ============= 測試 =============

if __name__ == "__main__":
    async def print_snapshot(snap: SystemSnapshot):
        print(f"CPU: {snap.cpu_percent:.1f}% | "
              f"RAM: {snap.memory_mb:.0f}MB ({snap.memory_percent:.1f}%) | "
              f"Workers: {snap.active_workers}")

    async def test():
        stats = SystemStats(interval_seconds=1.0)
        await stats.start_monitoring(print_snapshot)
        await asyncio.sleep(5)
        await stats.stop_monitoring()

    asyncio.run(test())
