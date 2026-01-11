"""
回測引擎核心

提供基於 VectorBT Pro 的完整回測功能，支援永續合約交易。

主要功能：
- 永續合約回測（資金費率、槓桿）
- 完整績效指標計算
- 靈活的配置系統
"""

# 永續合約模組（無依賴，可獨立使用）
from .perpetual import (
    PerpetualCalculator,
    PerpetualPosition,
    PerpetualRiskMonitor
)

# 滑點模組（無依賴，可獨立使用）
from .slippage import (
    SlippageCalculator,
    SlippageConfig,
    SlippageModel,
    OrderType,
    create_fixed_slippage,
    create_dynamic_slippage,
    create_market_impact_slippage
)

# 流動性模組（無依賴，可獨立使用）
from .liquidity import (
    LiquidityCalculator,
    LiquidityConfig,
    LiquidityModel,
    LiquidityLevel,
    create_linear_liquidity,
    create_square_root_liquidity,
    create_logarithmic_liquidity
)

# 回測引擎（需要 VectorBT）
try:
    from .engine import BacktestEngine, BacktestConfig, BacktestResult
    from .metrics import MetricsCalculator
    _has_vectorbt = True
except ImportError:
    _has_vectorbt = False
    BacktestEngine = None
    BacktestConfig = None
    BacktestResult = None
    MetricsCalculator = None

__all__ = [
    # 永續合約
    'PerpetualCalculator',
    'PerpetualPosition',
    'PerpetualRiskMonitor',
    # 滑點
    'SlippageCalculator',
    'SlippageConfig',
    'SlippageModel',
    'OrderType',
    'create_fixed_slippage',
    'create_dynamic_slippage',
    'create_market_impact_slippage',
    # 流動性
    'LiquidityCalculator',
    'LiquidityConfig',
    'LiquidityModel',
    'LiquidityLevel',
    'create_linear_liquidity',
    'create_square_root_liquidity',
    'create_logarithmic_liquidity',
]

if _has_vectorbt:
    __all__.extend([
        'BacktestEngine',
        'BacktestConfig',
        'BacktestResult',
        'MetricsCalculator',
    ])

__version__ = '1.0.0'
