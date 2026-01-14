"""
統一型別模組

集中管理所有專案中的資料型別定義，避免 dict 裸傳遞，
確保型別安全和 IDE 支援。

使用範例:
    from src.types import BacktestResult, ExperimentRecord, StrategyInfo
    from src.types import ExperimentStatus, Grade, StrategyType

參考：
    ~/.claude/skills/dev/SKILL.md - 資料契約規範
"""

# Enum 定義（禁止硬編碼，使用這些 Enum）
from .enums import (
    ExperimentStatus,
    Grade,
    StrategyType,
    OptimizationMethod,
    ObjectiveMetric,
    BackendType,
    LessonType,
    ParamType,
)

# 回測和驗證結果
from .results import (
    BacktestResult,
    ValidationResult,
    ExperimentRecord,
    PerformanceMetrics,
)

# 配置類型
from .configs import (
    BacktestConfig,
    LoopConfig,
    OptimizationConfig,
)

# 策略相關
from .strategies import (
    StrategyInfo,
    ParamSpace,
    StrategyStats,
)

__all__ = [
    # Enums
    'ExperimentStatus',
    'Grade',
    'StrategyType',
    'OptimizationMethod',
    'ObjectiveMetric',
    'BackendType',
    'LessonType',
    'ParamType',
    # Results
    'BacktestResult',
    'ValidationResult',
    'ExperimentRecord',
    'PerformanceMetrics',
    # Configs
    'BacktestConfig',
    'LoopConfig',
    'OptimizationConfig',
    # Strategies
    'StrategyInfo',
    'ParamSpace',
    'StrategyStats',
]
