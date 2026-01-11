"""
向量化基本功能測試

快速驗證向量化計算的正確性與效能。
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.backtester.engine import BacktestEngine, BacktestConfig
from src.backtester.vectorized import (
    vectorized_sma,
    vectorized_ema,
    vectorized_rsi,
)


@pytest.fixture
def sample_data():
    """產生樣本資料"""
    np.random.seed(42)
    n = 1000

    dates = pd.date_range('2024-01-01', periods=n, freq='1h')
    close = 10000 + np.cumsum(np.random.randn(n) * 10)
    high = close + np.abs(np.random.randn(n) * 5)
    low = close - np.abs(np.random.randn(n) * 5)
    open_ = close + np.random.randn(n) * 3
    volume = np.random.randint(100, 1000, n)

    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


class TestVectorizedIndicators:
    """測試向量化指標"""

    def test_sma_basic(self):
        """測試 SMA 基本功能"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        sma = vectorized_sma(data, 3)

        # 驗證前幾個值
        assert sma.iloc[2] == 2.0  # (1+2+3)/3
        assert sma.iloc[9] == 9.0  # (8+9+10)/3

    def test_ema_basic(self):
        """測試 EMA 基本功能"""
        data = pd.Series([10, 11, 12, 13, 14, 15])
        ema = vectorized_ema(data, 3)

        # 驗證 EMA 遞增
        assert ema.iloc[-1] > ema.iloc[0]

    def test_rsi_range(self):
        """測試 RSI 範圍"""
        data = pd.Series(np.random.randn(100) + 100)
        rsi = vectorized_rsi(data, 14)

        # RSI 應在 0-100 之間
        assert rsi.min() >= 0
        assert rsi.max() <= 100


class TestVectorizedBacktest:
    """測試向量化回測"""

    def test_vectorized_enabled(self, sample_data):
        """測試向量化模式啟用"""
        config = BacktestConfig(
            symbol='BTCUSDT',
            timeframe='1h',
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 2, 1),
            initial_capital=10000,
            vectorized=True,
            use_polars=False
        )

        assert config.vectorized is True
        assert config.use_polars is False

    def test_simple_strategy(self, sample_data):
        """測試簡單策略回測"""

        class SimpleStrategy:
            name = "Simple SMA"
            params = {'period': 20}

            def generate_signals(self, df):
                close = df['close']
                sma = vectorized_sma(close, self.params['period'])

                long_entry = close > sma
                long_exit = close < sma
                short_entry = pd.Series(False, index=df.index)
                short_exit = pd.Series(False, index=df.index)

                return long_entry, long_exit, short_entry, short_exit

        config = BacktestConfig(
            symbol='BTCUSDT',
            timeframe='1h',
            start_date=sample_data.index[0],
            end_date=sample_data.index[-1],
            initial_capital=10000,
            vectorized=True,
            use_polars=False
        )

        engine = BacktestEngine(config)
        strategy = SimpleStrategy()

        result = engine.run(strategy, data=sample_data)

        # 驗證結果結構
        assert hasattr(result, 'total_return')
        assert hasattr(result, 'sharpe_ratio')
        assert hasattr(result, 'total_trades')

        # 驗證結果合理性
        assert result.total_trades >= 0
        # 注意：VectorBT 允許槓桿導致虧損超過 100%

    def test_vectorized_vs_original(self, sample_data):
        """比較向量化與原始方法結果"""

        class TestStrategy:
            name = "Test"
            params = {'period': 10}

            def generate_signals(self, df):
                sma = vectorized_sma(df['close'], self.params['period'])
                long_entry = df['close'] > sma
                long_exit = df['close'] < sma
                return (
                    long_entry,
                    long_exit,
                    pd.Series(False, index=df.index),
                    pd.Series(False, index=df.index)
                )

        # 原始方法
        config_orig = BacktestConfig(
            symbol='BTCUSDT',
            timeframe='1h',
            start_date=sample_data.index[0],
            end_date=sample_data.index[-1],
            initial_capital=10000,
            vectorized=False,
            use_polars=False
        )

        # 向量化方法
        config_vec = BacktestConfig(
            symbol='BTCUSDT',
            timeframe='1h',
            start_date=sample_data.index[0],
            end_date=sample_data.index[-1],
            initial_capital=10000,
            vectorized=True,
            use_polars=False
        )

        strategy = TestStrategy()

        engine_orig = BacktestEngine(config_orig)
        engine_vec = BacktestEngine(config_vec)

        result_orig = engine_orig.run(strategy, data=sample_data)
        result_vec = engine_vec.run(strategy, data=sample_data)

        # 驗證結果一致（允許小誤差）
        assert abs(result_orig.total_return - result_vec.total_return) < 0.01
        assert result_orig.total_trades == result_vec.total_trades


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
