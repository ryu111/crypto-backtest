"""
策略模組

提供交易策略的基礎架構、註冊表和預設策略實作。

主要元件：
- BaseStrategy: 策略基礎抽象類別
- TrendStrategy: 趨勢策略基礎類別
- MeanReversionStrategy: 均值回歸策略基礎類別
- MomentumStrategy: 動量策略基礎類別
- StrategyRegistry: 策略註冊與管理

使用範例：
    # 1. 定義策略
    from src.strategies import BaseStrategy, register_strategy

    @register_strategy('ma_cross')
    class MACrossStrategy(BaseStrategy):
        params = {'fast': 10, 'slow': 30}
        strategy_type = 'trend'

        def calculate_indicators(self, data):
            return {
                'sma_fast': data['close'].rolling(self.params['fast']).mean(),
                'sma_slow': data['close'].rolling(self.params['slow']).mean()
            }

        def generate_signals(self, data):
            indicators = self.calculate_indicators(data)
            fast = indicators['sma_fast']
            slow = indicators['sma_slow']

            long_entry = (fast > slow) & (fast.shift(1) <= slow.shift(1))
            long_exit = (fast < slow) & (fast.shift(1) >= slow.shift(1))

            return long_entry, long_exit, None, None

    # 2. 使用策略
    from src.strategies import create_strategy

    strategy = create_strategy('ma_cross', fast=12, slow=26)
    long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(data)

    # 3. 查詢策略
    from src.strategies import list_strategies, get_strategy

    all_strategies = list_strategies()
    strategy_class = get_strategy('ma_cross')
"""

from .base import (
    BaseStrategy,
    TrendStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    StatisticalArbStrategy,
    FundingRateStrategy
)

from .registry import (
    StrategyRegistry,
    register_strategy,
    get_strategy,
    list_strategies,
    create_strategy
)

# 匯入策略以觸發註冊
from .momentum import RSIStrategy, MACDStrategy, StochasticStrategy
from .trend import MACrossStrategy, SupertrendStrategy, DonchianStrategy
from .mean_reversion import BollingerMeanReversionStrategy, RSIReversionStrategy
from .statistical_arb import ETHBTCPairsStrategy, BasisArbStrategy
from .funding_rate import FundingArbStrategy, SettlementTradeStrategy

__all__ = [
    # 基礎類別
    'BaseStrategy',
    'TrendStrategy',
    'MeanReversionStrategy',
    'MomentumStrategy',
    'StatisticalArbStrategy',
    'FundingRateStrategy',

    # 註冊表
    'StrategyRegistry',

    # 便利函數
    'register_strategy',
    'get_strategy',
    'list_strategies',
    'create_strategy',

    # 動量策略
    'RSIStrategy',
    'MACDStrategy',
    'StochasticStrategy',

    # 趨勢策略
    'MACrossStrategy',
    'SupertrendStrategy',
    'DonchianStrategy',

    # 均值回歸策略
    'BollingerMeanReversionStrategy',
    'RSIReversionStrategy',

    # 統計套利策略
    'ETHBTCPairsStrategy',
    'BasisArbStrategy',

    # 資金費率策略
    'FundingArbStrategy',
    'SettlementTradeStrategy',
]

__version__ = '1.0.0'
