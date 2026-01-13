"""市場狀態偵測模組"""

from .analyzer import (
    MarketRegime,
    MarketState,
    MarketStateAnalyzer,
    RegimeValidator,
    calculate_direction_score,
    adx_direction_score,
    elder_power_score,
    volatility_score_atr,
    volatility_score_bbw,
    choppiness_index,
)

from .switch import (
    StrategyConfig,
    StrategySwitch,
    setup_default_switch,
)

__all__ = [
    # Analyzer
    'MarketRegime',
    'MarketState',
    'MarketStateAnalyzer',
    'RegimeValidator',
    'calculate_direction_score',
    'adx_direction_score',
    'elder_power_score',
    'volatility_score_atr',
    'volatility_score_bbw',
    'choppiness_index',
    # Switch
    'StrategyConfig',
    'StrategySwitch',
    'setup_default_switch',
]
