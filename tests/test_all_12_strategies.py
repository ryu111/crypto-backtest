"""
測試 Polars 重構的三個核心策略 + 驗證所有策略的實例化

專注於：RSI, MA Cross, MACD
同時確保其他策略也能正常運作。
"""

import pytest
import pandas as pd
import numpy as np


def test_refactored_3_strategies_can_instantiate():
    """測試本次重構的 3 個策略能正常實例化"""

    from src.strategies.momentum.rsi import RSIStrategy
    from src.strategies.momentum.macd import MACDStrategy
    from src.strategies.trend.ma_cross import MACrossStrategy

    strategies = [
        ('RSI', RSIStrategy()),
        ('MACD', MACDStrategy()),
        ('MA Cross', MACrossStrategy()),
    ]

    print("\n" + "=" * 60)
    print("測試本次重構的 3 個策略")
    print("=" * 60)

    for name, strategy in strategies:
        assert strategy is not None
        assert hasattr(strategy, 'name')
        assert hasattr(strategy, 'params')
        assert hasattr(strategy, 'param_space')
        assert hasattr(strategy, '__init__')
        print(f"✓ {name:20s} - {strategy.name}")

    print("=" * 60)
    print(f"✓ 所有 {len(strategies)} 個策略實例化成功")
    print("=" * 60)


def test_all_strategies_can_instantiate():
    """測試所有可用策略都能正常實例化（包括 Polars 重構的 3 個）"""

    from src.strategies.momentum.rsi import RSIStrategy
    from src.strategies.momentum.macd import MACDStrategy
    from src.strategies.momentum.stochastic import StochasticStrategy
    from src.strategies.trend.ma_cross import MACrossStrategy
    from src.strategies.trend.supertrend import SupertrendStrategy
    from src.strategies.trend.donchian import DonchianStrategy
    from src.strategies.mean_reversion.bollinger import BollingerMeanReversionStrategy

    strategies = [
        ('RSI', RSIStrategy()),
        ('MACD', MACDStrategy()),
        ('Stochastic', StochasticStrategy()),
        ('MA Cross', MACrossStrategy()),
        ('SuperTrend', SupertrendStrategy()),
        ('Donchian', DonchianStrategy()),
        ('Bollinger', BollingerMeanReversionStrategy()),
    ]

    print("\n" + "=" * 60)
    print("測試所有 12 個策略的實例化")
    print("=" * 60)

    for name, strategy in strategies:
        assert strategy is not None
        assert hasattr(strategy, 'name')
        assert hasattr(strategy, 'params')
        assert hasattr(strategy, 'param_space')
        print(f"✓ {name:20s} - {strategy.name}")

    print("=" * 60)
    print(f"✓ 所有 {len(strategies)} 個策略實例化成功")
    print("=" * 60)


def test_refactored_strategies_have_custom_init():
    """測試本次重構的 3 個策略都有自訂 __init__ 方法"""

    from src.strategies.momentum.rsi import RSIStrategy
    from src.strategies.momentum.macd import MACDStrategy
    from src.strategies.trend.ma_cross import MACrossStrategy

    strategy_classes = [
        RSIStrategy,
        MACDStrategy,
        MACrossStrategy,
    ]

    print("\n" + "=" * 60)
    print("測試重構策略的 __init__ 方法")
    print("=" * 60)

    for cls in strategy_classes:
        # 確認有 __init__ 方法
        assert hasattr(cls, '__init__')
        # 確認 __init__ 不是繼承自 object 的預設版本
        assert cls.__init__ is not object.__init__
        print(f"✓ {cls.__name__:30s} has custom __init__")

    print("=" * 60)


def test_all_strategies_have_init_method():
    """測試所有策略都有 __init__ 方法"""

    from src.strategies.momentum.rsi import RSIStrategy
    from src.strategies.momentum.macd import MACDStrategy
    from src.strategies.momentum.stochastic import StochasticStrategy
    from src.strategies.trend.ma_cross import MACrossStrategy
    from src.strategies.trend.supertrend import SupertrendStrategy
    from src.strategies.trend.donchian import DonchianStrategy
    from src.strategies.mean_reversion.bollinger import BollingerMeanReversionStrategy

    strategy_classes = [
        RSIStrategy,
        MACDStrategy,
        StochasticStrategy,
        MACrossStrategy,
        SupertrendStrategy,
        DonchianStrategy,
        BollingerMeanReversionStrategy,
    ]

    for cls in strategy_classes:
        # 確認有 __init__ 方法
        assert hasattr(cls, '__init__')
        # 確認 __init__ 不是繼承自 object 的預設版本
        assert cls.__init__ is not object.__init__
        print(f"✓ {cls.__name__} has custom __init__")


@pytest.fixture
def sample_data():
    """產生測試用數據"""
    np.random.seed(42)
    n = 300

    trend = np.linspace(100, 150, n)
    noise = np.random.randn(n) * 5
    close = trend + noise

    high = close + np.random.rand(n) * 2
    low = close - np.random.rand(n) * 2
    open_price = close + np.random.randn(n) * 1
    volume = np.random.randint(1000, 10000, n)

    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h'),
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })

    return data


def test_all_strategies_can_generate_signals(sample_data):
    """測試所有策略都能生成訊號"""

    from src.strategies.momentum.rsi import RSIStrategy
    from src.strategies.momentum.macd import MACDStrategy
    from src.strategies.momentum.stochastic import StochasticStrategy
    from src.strategies.trend.ma_cross import MACrossStrategy
    from src.strategies.trend.supertrend import SupertrendStrategy
    from src.strategies.trend.donchian import DonchianStrategy
    from src.strategies.mean_reversion.bollinger import BollingerMeanReversionStrategy

    strategies = [
        RSIStrategy(),
        MACDStrategy(),
        StochasticStrategy(),
        MACrossStrategy(),
        SupertrendStrategy(),
        DonchianStrategy(),
        BollingerMeanReversionStrategy(),
    ]

    print("\n" + "=" * 60)
    print("測試所有策略的訊號生成")
    print("=" * 60)

    for strategy in strategies:
        signals = strategy.generate_signals(sample_data)

        # 檢查返回 4 個訊號
        assert len(signals) == 4

        # 檢查所有訊號都是 Series
        for signal in signals:
            assert isinstance(signal, pd.Series)
            assert len(signal) == len(sample_data)
            assert signal.dtype == bool

        print(f"✓ {strategy.name:30s} - 訊號生成正常")

    print("=" * 60)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
