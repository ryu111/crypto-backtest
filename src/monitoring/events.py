"""
監控事件類型定義
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any, Literal
from enum import Enum


class EventType(str, Enum):
    """事件類型"""
    LOOP_START = "loop_start"
    LOOP_COMPLETE = "loop_complete"
    LOOP_ERROR = "loop_error"
    ITERATION_START = "iteration_start"
    ITERATION_COMPLETE = "iteration_complete"
    ITERATION_ERROR = "iteration_error"
    SYSTEM_STATS = "system_stats"
    BEST_STRATEGY_UPDATE = "best_strategy_update"


@dataclass
class MonitorEvent:
    """基礎事件類型"""
    type: EventType
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['type'] = self.type.value
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class IterationEvent(MonitorEvent):
    """迭代事件"""
    iteration: int = 0
    total: int = 0
    strategy_name: Optional[str] = None
    sharpe: Optional[float] = None
    total_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    params: Optional[Dict[str, Any]] = None
    duration_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        # 移除 None 值
        return {k: v for k, v in result.items() if v is not None}


@dataclass
class SystemStatsEvent(MonitorEvent):
    """系統狀態事件"""
    type: EventType = field(default=EventType.SYSTEM_STATS)
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    gpu_percent: Optional[float] = None
    active_workers: int = 0


@dataclass
class LoopStateEvent(MonitorEvent):
    """Loop 狀態事件"""
    status: Literal['running', 'paused', 'completed', 'error'] = 'running'
    current_iteration: int = 0
    total_iterations: int = 0
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: Optional[float] = None
    best_sharpe: float = 0.0
    avg_sharpe: float = 0.0
    success_count: int = 0
    error_count: int = 0


@dataclass
class BestStrategyEvent(MonitorEvent):
    """最佳策略更新事件"""
    type: EventType = field(default=EventType.BEST_STRATEGY_UPDATE)
    iteration: int = 0
    strategy_name: str = ""
    sharpe: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyDistribution:
    """策略分佈統計"""
    strategy_counts: Dict[str, int] = field(default_factory=dict)
    strategy_avg_sharpe: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
