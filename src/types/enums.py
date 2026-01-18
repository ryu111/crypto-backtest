"""
集中的 Enum 定義模組

所有專案使用的枚舉型別，避免硬編碼和重複定義。

使用範例:
    from src.types import ExperimentStatus, StrategyType, Grade

    # ❌ 錯誤：硬編碼
    if status == "completed": ...
    if strategy_type == "trend": ...

    # ✅ 正確：使用 Enum
    if status == ExperimentStatus.COMPLETED: ...
    if strategy_type == StrategyType.TREND: ...

    # JSON 序列化
    data = {"status": ExperimentStatus.COMPLETED}  # 自動轉為 "completed"
"""
from enum import Enum


class ExperimentStatus(str, Enum):
    """實驗狀態

    用於標記實驗記錄的執行狀態。
    """
    COMPLETED = "completed"
    FAILED = "failed"
    RUNNING = "running"


class Grade(str, Enum):
    """驗證等級

    策略驗證結果的評級，從 A（最佳）到 F（失敗）。
    """
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


class StrategyType(str, Enum):
    """策略類型

    支援的交易策略類型分類。
    """
    TREND = "trend"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    FUNDING_RATE = "funding_rate"
    COMPOSITE = "composite"
    # 預留（未來擴展）
    BREAKOUT = "breakout"
    VOLATILITY = "volatility"


class OptimizationMethod(str, Enum):
    """優化方法

    參數優化使用的算法類型。
    """
    BAYESIAN = "bayesian"
    GRID = "grid"
    RANDOM = "random"


class ObjectiveMetric(str, Enum):
    """目標指標

    優化時使用的目標函數指標。
    """
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    RETURN_PCT = "return_pct"


class BackendType(str, Enum):
    """後端類型

    回測引擎使用的計算後端。
    """
    MLX = "mlx"
    MPS = "mps"  # PyTorch Metal Performance Shaders
    CPU = "cpu"
    NUMPY = "numpy"
    METAL = "metal"


class LessonType(str, Enum):
    """洞察類型

    自動學習系統記錄的洞察分類。
    """
    EXCEPTIONAL = "exceptional_performance"
    POOR = "unexpected_poor_performance"
    OVERFIT = "overfit_warning"
    RISK = "risk_event"
    SENSITIVITY = "parameter_sensitivity"


class ParamType(str, Enum):
    """參數類型

    用於 ParamSpace 定義參數的數值類型。
    """
    INT = "int"
    FLOAT = "float"
    LOG = "log"


class DirectionMethod(str, Enum):
    """方向性計算方法

    用於 Regime Detection 的方向性計算。
    """
    COMPOSITE = "composite"
    ADX = "adx"
    ELDER = "elder"


class StrategySelectionMode(str, Enum):
    """策略選擇模式

    用於決定如何選擇策略組合。
    """
    REGIME_AWARE = "regime_aware"
    RANDOM = "random"
    EXPLOIT = "exploit"


class AggregationMode(str, Enum):
    """信號聚合模式

    多個策略信號的聚合方法。
    """
    WEIGHTED = "weighted"
    VOTING = "voting"
    RANKED = "ranked"
    UNANIMOUS = "unanimous"


class ParetoSelectMethod(str, Enum):
    """Pareto 解選擇方法

    多目標優化中選擇 Pareto 前緣解的方法。
    """
    KNEE = "knee"
    CROWDING = "crowding"
    RANDOM = "random"


__all__ = [
    "ExperimentStatus",
    "Grade",
    "StrategyType",
    "OptimizationMethod",
    "ObjectiveMetric",
    "BackendType",
    "LessonType",
    "ParamType",
    "DirectionMethod",
    "StrategySelectionMode",
    "AggregationMode",
    "ParetoSelectMethod",
]
