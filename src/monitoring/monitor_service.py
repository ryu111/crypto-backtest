"""
監控服務

FastAPI + WebSocket 即時推送回測進度。
"""

import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from src.monitoring.events import (
    EventType,
    IterationEvent,
    SystemStatsEvent,
    LoopStateEvent,
    BestStrategyEvent,
    StrategyDistribution,
)
from src.monitoring.system_stats import SystemStats, SystemSnapshot

logger = logging.getLogger(__name__)


@dataclass
class MonitorState:
    """監控狀態（累積統計）"""
    status: str = "idle"  # idle, running, completed, error
    start_time: Optional[datetime] = None
    current_iteration: int = 0
    total_iterations: int = 0

    # 績效統計
    best_sharpe: float = 0.0
    best_iteration: int = 0
    best_strategy_name: str = ""
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_return: float = 0.0
    best_drawdown: float = 0.0

    # 累積統計
    sharpe_history: List[float] = field(default_factory=list)
    success_count: int = 0
    error_count: int = 0

    # 策略分佈
    strategy_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    strategy_sharpe_sum: Dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def update_iteration(self, event: IterationEvent):
        """更新迭代結果"""
        self.current_iteration = event.iteration

        if event.sharpe is not None:
            self.sharpe_history.append(event.sharpe)
            self.success_count += 1

            # 更新策略統計
            if event.strategy_name:
                self.strategy_counts[event.strategy_name] += 1
                self.strategy_sharpe_sum[event.strategy_name] += event.sharpe

            # 檢查是否為新最佳
            if event.sharpe > self.best_sharpe:
                self.best_sharpe = event.sharpe
                self.best_iteration = event.iteration
                self.best_strategy_name = event.strategy_name or ""
                self.best_params = event.params or {}
                self.best_return = event.total_return or 0.0
                self.best_drawdown = event.max_drawdown or 0.0
                return True  # 有新最佳

        return False

    def get_avg_sharpe(self) -> float:
        """獲取平均 Sharpe"""
        if not self.sharpe_history:
            return 0.0
        return sum(self.sharpe_history) / len(self.sharpe_history)

    def get_success_rate(self) -> float:
        """獲取成功率"""
        total = self.success_count + self.error_count
        if total == 0:
            return 0.0
        return self.success_count / total * 100

    def get_elapsed_seconds(self) -> float:
        """獲取已經過時間"""
        if not self.start_time:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds()

    def get_estimated_remaining(self) -> Optional[float]:
        """估計剩餘時間"""
        if self.current_iteration == 0:
            return None
        elapsed = self.get_elapsed_seconds()
        avg_per_iter = elapsed / self.current_iteration
        remaining = self.total_iterations - self.current_iteration
        return avg_per_iter * remaining

    def get_strategy_distribution(self) -> StrategyDistribution:
        """獲取策略分佈"""
        avg_sharpe = {}
        for name, count in self.strategy_counts.items():
            if count > 0:
                avg_sharpe[name] = self.strategy_sharpe_sum[name] / count

        return StrategyDistribution(
            strategy_counts=dict(self.strategy_counts),
            strategy_avg_sharpe=avg_sharpe,
        )

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典（用於 JSON）"""
        return {
            'status': self.status,
            'startTime': self.start_time.isoformat() if self.start_time else None,
            'currentIteration': self.current_iteration,
            'totalIterations': self.total_iterations,
            'elapsedSeconds': self.get_elapsed_seconds(),
            'estimatedRemaining': self.get_estimated_remaining(),
            'bestSharpe': self.best_sharpe,
            'bestIteration': self.best_iteration,
            'bestStrategyName': self.best_strategy_name,
            'bestParams': self.best_params,
            'bestReturn': self.best_return,
            'bestDrawdown': self.best_drawdown,
            'avgSharpe': self.get_avg_sharpe(),
            'successRate': self.get_success_rate(),
            'successCount': self.success_count,
            'errorCount': self.error_count,
            'sharpeHistory': self.sharpe_history[-100:],  # 最近 100 個
            'strategyDistribution': self.get_strategy_distribution().to_dict(),
        }


class MonitorService:
    """監控服務"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.app = FastAPI(title="UltimateLoop Monitor")
        self.connections: List[WebSocket] = []
        self.state = MonitorState()
        self.system_stats = SystemStats(interval_seconds=1.0)
        self._setup_routes()

    def _setup_routes(self):
        """設定路由"""
        # 靜態檔案
        ui_path = Path(__file__).parent.parent.parent / "ui" / "monitor"
        if ui_path.exists():
            self.app.mount("/static", StaticFiles(directory=str(ui_path)), name="static")

        @self.app.get("/")
        async def index():
            """主頁面"""
            index_file = ui_path / "index.html"
            if index_file.exists():
                return FileResponse(str(index_file))
            return HTMLResponse("<h1>UltimateLoop Monitor</h1><p>UI files not found.</p>")

        @self.app.get("/api/state")
        async def get_state():
            """獲取當前狀態"""
            return self.state.to_dict()

        @self.app.get("/api/system-stats")
        async def get_system_stats():
            """獲取系統狀態（CPU/RAM）"""
            snapshot = self.system_stats.get_snapshot()
            return {
                'cpuPercent': snapshot.cpu_percent,
                'memoryMb': snapshot.memory_mb,
                'memoryPercent': snapshot.memory_percent,
                'gpuPercent': snapshot.gpu_percent,
                'activeWorkers': snapshot.active_workers,
            }

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket 連接"""
            await websocket.accept()
            self.connections.append(websocket)
            logger.info(f"WebSocket 連接建立，當前連接數: {len(self.connections)}")

            try:
                # 發送當前狀態
                await websocket.send_json({
                    'type': 'state_sync',
                    'data': self.state.to_dict()
                })

                # 保持連接
                while True:
                    try:
                        # 接收心跳或控制命令
                        data = await asyncio.wait_for(
                            websocket.receive_text(),
                            timeout=30.0
                        )
                        if data == "ping":
                            await websocket.send_text("pong")
                    except asyncio.TimeoutError:
                        # 發送心跳
                        await websocket.send_text("ping")

            except WebSocketDisconnect:
                pass
            finally:
                if websocket in self.connections:
                    self.connections.remove(websocket)
                logger.info(f"WebSocket 斷開，當前連接數: {len(self.connections)}")

    async def broadcast(self, event_type: str, data: Dict[str, Any]):
        """廣播事件到所有連接"""
        if not self.connections:
            return

        message = json.dumps({
            'type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        })

        # 移除斷開的連接
        disconnected = []
        for ws in self.connections:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            if ws in self.connections:
                self.connections.remove(ws)

    async def _on_system_stats(self, snapshot: SystemSnapshot):
        """系統狀態回調"""
        await self.broadcast('system_stats', {
            'cpuPercent': snapshot.cpu_percent,
            'memoryMb': snapshot.memory_mb,
            'memoryPercent': snapshot.memory_percent,
            'gpuPercent': snapshot.gpu_percent,
            'activeWorkers': snapshot.active_workers,
        })

    # ============= 外部調用接口 =============

    async def on_loop_start(self, total_iterations: int):
        """Loop 開始"""
        self.state = MonitorState()
        self.state.status = "running"
        self.state.start_time = datetime.now()
        self.state.total_iterations = total_iterations

        await self.broadcast('loop_start', {
            'totalIterations': total_iterations,
        })

        # 啟動系統監控
        await self.system_stats.start_monitoring(self._on_system_stats)

    async def on_loop_complete(self):
        """Loop 完成"""
        self.state.status = "completed"
        await self.system_stats.stop_monitoring()

        await self.broadcast('loop_complete', self.state.to_dict())

    async def on_loop_error(self, error: str):
        """Loop 錯誤"""
        self.state.status = "error"
        await self.system_stats.stop_monitoring()

        await self.broadcast('loop_error', {
            'error': error,
            'state': self.state.to_dict()
        })

    async def on_iteration_start(self, iteration: int):
        """迭代開始"""
        self.state.current_iteration = iteration

        await self.broadcast('iteration_start', {
            'iteration': iteration,
            'total': self.state.total_iterations,
        })

    async def on_iteration_complete(
        self,
        iteration: int,
        strategy_name: str,
        sharpe: float,
        total_return: float,
        max_drawdown: float,
        params: Dict[str, Any],
        duration_seconds: float,
    ):
        """迭代完成"""
        event = IterationEvent(
            type=EventType.ITERATION_COMPLETE,
            iteration=iteration,
            total=self.state.total_iterations,
            strategy_name=strategy_name,
            sharpe=sharpe,
            total_return=total_return,
            max_drawdown=max_drawdown,
            params=params,
            duration_seconds=duration_seconds,
        )

        is_new_best = self.state.update_iteration(event)

        await self.broadcast('iteration_complete', {
            'iteration': iteration,
            'total': self.state.total_iterations,
            'strategyName': strategy_name,
            'sharpe': sharpe,
            'totalReturn': total_return,
            'maxDrawdown': max_drawdown,
            'params': params,
            'duration': duration_seconds,
            'isNewBest': is_new_best,
            'avgSharpe': self.state.get_avg_sharpe(),
            'successRate': self.state.get_success_rate(),
        })

        # 如果有新最佳，額外發送事件
        if is_new_best:
            await self.broadcast('best_strategy_update', {
                'iteration': iteration,
                'strategyName': strategy_name,
                'sharpe': sharpe,
                'totalReturn': total_return,
                'maxDrawdown': max_drawdown,
                'params': params,
            })

    async def on_iteration_error(self, iteration: int, error: str):
        """迭代錯誤"""
        self.state.error_count += 1
        self.state.current_iteration = iteration

        await self.broadcast('iteration_error', {
            'iteration': iteration,
            'error': error,
        })

    def run(self):
        """啟動服務（阻塞）"""
        import uvicorn
        logger.info(f"啟動監控服務: http://{self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="warning")

    async def start(self):
        """非阻塞啟動"""
        import uvicorn
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning"
        )
        server = uvicorn.Server(config)
        logger.info(f"啟動監控服務: http://{self.host}:{self.port}")
        await server.serve()


# ============= 單獨執行 =============

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    service = MonitorService()
    service.run()
