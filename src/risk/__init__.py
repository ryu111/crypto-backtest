"""
風險管理模組

提供部位管理、相關性分析、自適應槓桿等風險控制功能。
"""

from .correlation import (
    CorrelationAnalyzer,
    CorrelationMatrix,
    RollingCorrelation,
    TailCorrelation,
)
from .position_sizing import (
    kelly_criterion,
    KellyPositionSizer,
    PositionSizeResult,
)
from .adaptive_leverage import (
    AdaptiveLeverageConfig,
    AdaptiveLeverageController,
)

__all__ = [
    "CorrelationAnalyzer",
    "CorrelationMatrix",
    "RollingCorrelation",
    "TailCorrelation",
    "kelly_criterion",
    "KellyPositionSizer",
    "PositionSizeResult",
    "AdaptiveLeverageConfig",
    "AdaptiveLeverageController",
]
