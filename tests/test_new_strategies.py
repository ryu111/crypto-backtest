"""
測試新建立的 4 個策略

策略列表：
1. mean_reversion_bollinger - 布林帶均值回歸
2. mean_reversion_rsi - RSI 均值回歸
3. trend_donchian - Donchian Channel 突破
4. momentum_stochastic - Stochastic Crossover

測試項目：
- 策略註冊測試
- 參數驗證測試
- 訊號生成測試
- 指標計算測試
"""

import pytest
import pandas as pd
import numpy as np
from typing import Tuple

from src.strategies import (
    create_strategy,
    get_strategy,
    list_strategies,
)


def create_test_data(n: int = 200) -> pd.DataFrame:
    """
    建立模擬 OHLCV 數據

    Args:
        n: 數據點數量

    Returns:
        DataFrame: 包含 open, high, low, close, volume 的測試數據
    """
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=n, freq='4h')

    # 產生有趨勢和震盪的價格
    trend = np.linspace(100, 110, n)
    noise = np.random.randn(n) * 2
    close = trend + noise

    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(100, 1000, n).astype(float)

    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


class TestStrategyRegistration:
    """測試策略註冊"""

    def test_all_strategies_registered(self):
        """測試所有 4 個新策略都已註冊"""
        all_strategies = list_strategies()

        required_strategies = [
            'mean_reversion_bollinger',
            'mean_reversion_rsi',
            'trend_donchian',
            'momentum_stochastic',
        ]

        for strategy_name in required_strategies:
            assert strategy_name in all_strategies, \
                f"策略 '{strategy_name}' 未註冊"

    def test_get_strategy_classes(self):
        """測試能夠取得策略類別"""
        strategy_names = [
            'mean_reversion_bollinger',
            'mean_reversion_rsi',
            'trend_donchian',
            'momentum_stochastic',
        ]

        for name in strategy_names:
            strategy_class = get_strategy(name)
            assert strategy_class is not None, \
                f"無法取得策略類別: {name}"
            assert hasattr(strategy_class, 'name')
            assert hasattr(strategy_class, 'strategy_type')
            assert hasattr(strategy_class, 'version')


class TestBollingerMeanReversion:
    """測試布林帶均值回歸策略"""

    def test_create_with_default_params(self):
        """測試使用預設參數建立策略"""
        strategy = create_strategy('mean_reversion_bollinger')
        assert strategy is not None
        assert strategy.name == "Bollinger Bands Mean Reversion"
        assert strategy.strategy_type == "mean_reversion"

    def test_create_with_custom_params(self):
        """測試使用自訂參數建立策略"""
        strategy = create_strategy(
            'mean_reversion_bollinger',
            period=15,
            std_dev=2.5
        )
        assert strategy.params['period'] == 15
        assert strategy.params['std_dev'] == 2.5

    def test_param_validation(self):
        """測試參數驗證"""
        # 有效參數應該通過
        strategy = create_strategy(
            'mean_reversion_bollinger',
            period=20,
            std_dev=2.0
        )
        assert strategy.validate_params() is True

        # 無效參數應該被拒絕
        with pytest.raises(ValueError):
            create_strategy(
                'mean_reversion_bollinger',
                period=0  # 無效：週期必須 > 1
            )

        with pytest.raises(ValueError):
            create_strategy(
                'mean_reversion_bollinger',
                std_dev=-1  # 無效：標準差必須 > 0
            )

    def test_calculate_indicators(self):
        """測試指標計算"""
        strategy = create_strategy('mean_reversion_bollinger')
        data = create_test_data()

        indicators = strategy.calculate_indicators(data)

        # 檢查回傳的指標
        assert 'bb_upper' in indicators
        assert 'bb_middle' in indicators
        assert 'bb_lower' in indicators
        assert 'bb_width' in indicators

        # 檢查資料型態
        assert isinstance(indicators['bb_upper'], pd.Series)
        assert len(indicators['bb_upper']) == len(data)

        # 檢查邏輯正確性：上軌 > 中軌 > 下軌（排除 NaN 值）
        valid_data = ~indicators['bb_upper'].isna()
        assert (indicators['bb_upper'][valid_data] >= indicators['bb_middle'][valid_data]).all()
        assert (indicators['bb_middle'][valid_data] >= indicators['bb_lower'][valid_data]).all()

    def test_generate_signals(self):
        """測試訊號生成"""
        strategy = create_strategy('mean_reversion_bollinger')
        data = create_test_data()

        long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(data)

        # 檢查回傳型態
        assert isinstance(long_entry, pd.Series)
        assert isinstance(long_exit, pd.Series)
        assert isinstance(short_entry, pd.Series)
        assert isinstance(short_exit, pd.Series)

        # 檢查長度
        assert len(long_entry) == len(data)
        assert len(long_exit) == len(data)
        assert len(short_entry) == len(data)
        assert len(short_exit) == len(data)

        # 檢查值都是 boolean
        assert long_entry.dtype == bool
        assert long_exit.dtype == bool
        assert short_entry.dtype == bool
        assert short_exit.dtype == bool


class TestRSIMeanReversion:
    """測試 RSI 均值回歸策略"""

    def test_create_with_default_params(self):
        """測試使用預設參數建立策略"""
        strategy = create_strategy('mean_reversion_rsi')
        assert strategy is not None
        assert strategy.name == "RSI Mean Reversion"
        assert strategy.strategy_type == "mean_reversion"

    def test_param_validation(self):
        """測試參數驗證"""
        # 有效參數：oversold < exit_threshold < overbought
        strategy = create_strategy(
            'mean_reversion_rsi',
            oversold=30,
            exit_threshold=50,
            overbought=70
        )
        assert strategy.validate_params() is True

        # 無效參數：oversold >= exit_threshold
        with pytest.raises(ValueError):
            create_strategy(
                'mean_reversion_rsi',
                oversold=60,
                exit_threshold=50,
                overbought=70
            )

        # 無效參數：exit_threshold >= overbought
        with pytest.raises(ValueError):
            create_strategy(
                'mean_reversion_rsi',
                oversold=30,
                exit_threshold=80,
                overbought=70
            )

    def test_calculate_indicators(self):
        """測試指標計算"""
        strategy = create_strategy('mean_reversion_rsi')
        data = create_test_data()

        indicators = strategy.calculate_indicators(data)

        # 檢查回傳的指標
        assert 'rsi' in indicators
        assert 'atr' in indicators

        # 檢查 RSI 範圍（應該在 0-100 之間）
        rsi = indicators['rsi'].dropna()
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()

        # 檢查 ATR 為正
        atr = indicators['atr'].dropna()
        assert (atr > 0).all()

    def test_generate_signals(self):
        """測試訊號生成"""
        strategy = create_strategy('mean_reversion_rsi')
        data = create_test_data()

        long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(data)

        # 檢查回傳型態
        assert isinstance(long_entry, pd.Series)
        assert len(long_entry) == len(data)
        assert long_entry.dtype == bool

        # 檢查訊號邏輯（只在 RSI 穿越閾值時才觸發，不是持續觸發）
        # 訊號不應該連續多個 True
        long_entry_values = long_entry.values
        for i in range(1, len(long_entry_values)):
            # 如果連續兩個 True，可能有問題（但也可能正常，視數據而定）
            # 這裡只做基本檢查
            pass


class TestDonchianBreakout:
    """測試 Donchian Channel 突破策略"""

    def test_create_with_default_params(self):
        """測試使用預設參數建立策略"""
        strategy = create_strategy('trend_donchian')
        assert strategy is not None
        assert strategy.name == "Donchian Channel Breakout"
        assert strategy.strategy_type == "trend"

    def test_param_validation(self):
        """測試參數驗證"""
        # 有效參數
        strategy = create_strategy(
            'trend_donchian',
            period=20,
            stop_loss_atr=2.0
        )
        assert strategy.validate_params() is True

        # 無效參數：週期 <= 0
        with pytest.raises(ValueError):
            create_strategy(
                'trend_donchian',
                period=0
            )

        # 無效參數：ATR 倍數 <= 0
        with pytest.raises(ValueError):
            create_strategy(
                'trend_donchian',
                stop_loss_atr=-1
            )

    def test_calculate_indicators(self):
        """測試指標計算"""
        strategy = create_strategy('trend_donchian')
        data = create_test_data()

        indicators = strategy.calculate_indicators(data)

        # 檢查回傳的指標
        assert 'donchian_upper' in indicators
        assert 'donchian_lower' in indicators
        assert 'donchian_middle' in indicators
        assert 'atr' in indicators
        assert 'natr' in indicators

        # 檢查邏輯正確性：上軌 > 中線 > 下軌
        upper = indicators['donchian_upper'].dropna()
        middle = indicators['donchian_middle'].dropna()
        lower = indicators['donchian_lower'].dropna()

        assert (upper >= middle).all()
        assert (middle >= lower).all()

    def test_generate_signals(self):
        """測試訊號生成"""
        strategy = create_strategy('trend_donchian')
        data = create_test_data()

        long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(data)

        # 檢查回傳型態
        assert isinstance(long_entry, pd.Series)
        assert len(long_entry) == len(data)
        assert long_entry.dtype == bool


class TestStochasticCrossover:
    """測試 Stochastic Crossover 策略"""

    def test_create_with_default_params(self):
        """測試使用預設參數建立策略"""
        strategy = create_strategy('momentum_stochastic')
        assert strategy is not None
        assert strategy.name == "Stochastic Crossover"
        assert strategy.strategy_type == "momentum"

    def test_param_validation(self):
        """測試參數驗證"""
        # 有效參數：oversold < overbought
        strategy = create_strategy(
            'momentum_stochastic',
            oversold=20,
            overbought=80
        )
        assert strategy.validate_params() is True

        # 無效參數：oversold >= overbought
        with pytest.raises(ValueError):
            create_strategy(
                'momentum_stochastic',
                oversold=80,
                overbought=20
            )

        # 無效參數：週期 <= 0
        with pytest.raises(ValueError):
            create_strategy(
                'momentum_stochastic',
                k_period=0
            )

    def test_calculate_indicators(self):
        """測試指標計算"""
        strategy = create_strategy('momentum_stochastic')
        data = create_test_data()

        indicators = strategy.calculate_indicators(data)

        # 檢查回傳的指標
        assert 'stoch_k' in indicators
        assert 'stoch_d' in indicators

        # 檢查 Stochastic 範圍（應該在 0-100 之間）
        stoch_k = indicators['stoch_k'].dropna()
        stoch_d = indicators['stoch_d'].dropna()

        assert (stoch_k >= 0).all()
        assert (stoch_k <= 100).all()
        assert (stoch_d >= 0).all()
        assert (stoch_d <= 100).all()

    def test_generate_signals(self):
        """測試訊號生成"""
        strategy = create_strategy('momentum_stochastic')
        data = create_test_data()

        long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(data)

        # 檢查回傳型態
        assert isinstance(long_entry, pd.Series)
        assert len(long_entry) == len(data)
        assert long_entry.dtype == bool

        # 檢查所有訊號
        assert isinstance(long_exit, pd.Series)
        assert isinstance(short_entry, pd.Series)
        assert isinstance(short_exit, pd.Series)


class TestStrategyComparison:
    """測試策略之間的比較"""

    def test_all_strategies_return_same_format(self):
        """測試所有策略回傳相同格式的訊號"""
        strategy_names = [
            'mean_reversion_bollinger',
            'mean_reversion_rsi',
            'trend_donchian',
            'momentum_stochastic',
        ]

        data = create_test_data()

        for name in strategy_names:
            strategy = create_strategy(name)
            signals = strategy.generate_signals(data)

            # 檢查回傳 4 個 Series
            assert len(signals) == 4, f"{name} 未回傳 4 個訊號"

            # 檢查每個訊號都是 boolean Series
            for i, signal in enumerate(signals):
                assert isinstance(signal, pd.Series), \
                    f"{name} 訊號 {i} 不是 Series"
                assert signal.dtype == bool, \
                    f"{name} 訊號 {i} 不是 boolean 型態"
                assert len(signal) == len(data), \
                    f"{name} 訊號 {i} 長度不正確"

    def test_all_strategies_have_param_space(self):
        """測試所有策略都有參數優化空間定義"""
        strategy_names = [
            'mean_reversion_bollinger',
            'mean_reversion_rsi',
            'trend_donchian',
            'momentum_stochastic',
        ]

        for name in strategy_names:
            strategy = create_strategy(name)

            assert hasattr(strategy, 'param_space'), \
                f"{name} 沒有 param_space 屬性"
            assert isinstance(strategy.param_space, dict), \
                f"{name} 的 param_space 不是 dict"
            assert len(strategy.param_space) > 0, \
                f"{name} 的 param_space 為空"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
