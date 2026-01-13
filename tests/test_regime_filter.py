"""
測試 Regime Filter 功能

驗證 BaseStrategy.apply_regime_filter() 整合是否正確。
"""

import pytest
import pandas as pd
import numpy as np
from src.strategies.base import BaseStrategy
from src.regime import MarketStateAnalyzer, StrategyConfig


class DummyStrategy(BaseStrategy):
    """測試用假策略"""

    name = "dummy_strategy"
    strategy_type = "test"
    version = "1.0"
    description = "Test strategy for regime filter"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params.setdefault('period', 20)

    def calculate_indicators(self, data):
        return {'sma': data['close'].rolling(self.params['period']).mean()}

    def generate_signals(self, data):
        # 簡單策略：價格 > SMA 做多，< SMA 做空
        sma = data['close'].rolling(self.params['period']).mean()
        long_entry = data['close'] > sma
        long_exit = data['close'] < sma
        short_entry = data['close'] < sma
        short_exit = data['close'] > sma
        return long_entry, long_exit, short_entry, short_exit


def create_test_data(trend='bull', volatility='high', periods=100):
    """
    建立測試數據

    Args:
        trend: 'bull', 'bear', 'neutral'
        volatility: 'high', 'low'
        periods: 數據長度
    """
    np.random.seed(42)

    # 基礎趨勢
    if trend == 'bull':
        base = np.linspace(100, 150, periods)
    elif trend == 'bear':
        base = np.linspace(150, 100, periods)
    else:  # neutral
        base = np.ones(periods) * 125

    # 波動
    if volatility == 'high':
        noise = np.random.randn(periods) * 5
    else:  # low
        noise = np.random.randn(periods) * 1

    close = base + noise
    high = close + np.abs(np.random.randn(periods) * 2)
    low = close - np.abs(np.random.randn(periods) * 2)
    open_ = close + np.random.randn(periods) * 0.5
    volume = np.random.randint(1000, 10000, periods)

    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


class TestRegimeFilter:
    """Regime Filter 測試套件"""

    def test_regime_filter_disabled_by_default(self):
        """測試：預設不啟用 Regime Filter"""
        strategy = DummyStrategy()
        data = create_test_data('bull', 'high')

        long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(data)
        filtered = strategy.apply_filters(data, long_entry, long_exit, short_entry, short_exit)

        # 不啟用時，訊號應該不變
        assert filtered[0].equals(long_entry)
        assert filtered[1].equals(long_exit)
        assert filtered[2].equals(short_entry)
        assert filtered[3].equals(short_exit)

    def test_regime_filter_enabled_strong_bull(self):
        """測試：強牛市過濾"""
        strategy = DummyStrategy(
            use_regime_filter=True,
            regime_config={
                'direction_range': (3, 10),   # 只接受強牛
                'volatility_range': (0, 10)
            }
        )

        # 建立強牛市數據
        data = create_test_data('bull', 'high')

        long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(data)
        filtered_long, filtered_long_exit, filtered_short, filtered_short_exit = strategy.apply_filters(
            data, long_entry, long_exit, short_entry, short_exit
        )

        # 驗證市場狀態
        analyzer = MarketStateAnalyzer()
        state = analyzer.calculate_state(data)

        if state.direction > 3:
            # 強牛市：訊號應保留
            assert filtered_long.sum() > 0
        else:
            # 非強牛市：進場訊號應被清除
            assert filtered_long.sum() == 0
            assert filtered_short.sum() == 0

    def test_regime_filter_enabled_strong_bear(self):
        """測試：強熊市過濾"""
        strategy = DummyStrategy(
            use_regime_filter=True,
            regime_config={
                'direction_range': (-10, -3),  # 只接受強熊
                'volatility_range': (0, 10)
            }
        )

        # 建立強熊市數據
        data = create_test_data('bear', 'high')

        long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(data)
        filtered_long, filtered_long_exit, filtered_short, filtered_short_exit = strategy.apply_filters(
            data, long_entry, long_exit, short_entry, short_exit
        )

        # 驗證市場狀態
        analyzer = MarketStateAnalyzer()
        state = analyzer.calculate_state(data)

        if state.direction < -3:
            # 強熊市：訊號應保留
            assert filtered_short.sum() > 0 or filtered_long.sum() > 0
        else:
            # 非強熊市：進場訊號應被清除
            assert filtered_long.sum() == 0
            assert filtered_short.sum() == 0

    def test_regime_filter_neutral_market(self):
        """測試：中性市場過濾"""
        strategy = DummyStrategy(
            use_regime_filter=True,
            regime_config={
                'direction_range': (-3, 3),    # 只接受中性
                'volatility_range': (0, 10)
            }
        )

        # 建立中性市場數據
        data = create_test_data('neutral', 'low')

        long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(data)
        filtered_long, filtered_long_exit, filtered_short, filtered_short_exit = strategy.apply_filters(
            data, long_entry, long_exit, short_entry, short_exit
        )

        # 驗證市場狀態
        analyzer = MarketStateAnalyzer()
        state = analyzer.calculate_state(data)

        if -3 <= state.direction <= 3:
            # 中性市場：訊號應保留（可能沒有訊號，但不應該全被過濾）
            # 至少檢查不會出錯
            assert isinstance(filtered_long, pd.Series)
        else:
            # 非中性市場：進場訊號應被清除
            assert filtered_long.sum() == 0
            assert filtered_short.sum() == 0

    def test_regime_filter_exit_signals_preserved(self):
        """測試：出場訊號保留"""
        strategy = DummyStrategy(
            use_regime_filter=True,
            regime_config={
                'direction_range': (5, 10),    # 極窄範圍（大概率不符合）
                'volatility_range': (0, 10)
            }
        )

        data = create_test_data('neutral', 'low')

        long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(data)
        filtered_long, filtered_long_exit, filtered_short, filtered_short_exit = strategy.apply_filters(
            data, long_entry, long_exit, short_entry, short_exit
        )

        # 出場訊號應保留原值
        assert filtered_long_exit.equals(long_exit)
        assert filtered_short_exit.equals(short_exit)

    def test_regime_filter_custom_analyzer_params(self):
        """測試：自定義 Analyzer 參數"""
        strategy = DummyStrategy(
            use_regime_filter=True,
            regime_config={
                'direction_range': (-10, 10),
                'volatility_range': (0, 10),
                'analyzer_params': {
                    'direction_threshold_strong': 4.0,
                    'direction_threshold_weak': 1.5,
                    'volatility_threshold': 6.0,
                    'direction_method': 'adx'
                }
            }
        )

        data = create_test_data('bull', 'high')

        long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(data)

        # 應該不會出錯
        filtered = strategy.apply_filters(data, long_entry, long_exit, short_entry, short_exit)
        assert len(filtered) == 4

    def test_regime_filter_backward_compatible(self):
        """測試：向後相容性"""
        # 舊策略不使用 regime_config
        old_strategy = DummyStrategy()
        data = create_test_data('bull', 'high')

        long_entry, long_exit, short_entry, short_exit = old_strategy.generate_signals(data)
        filtered = old_strategy.apply_filters(data, long_entry, long_exit, short_entry, short_exit)

        # 應該完全不受影響
        assert filtered[0].equals(long_entry)
        assert filtered[1].equals(long_exit)
        assert filtered[2].equals(short_entry)
        assert filtered[3].equals(short_exit)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
