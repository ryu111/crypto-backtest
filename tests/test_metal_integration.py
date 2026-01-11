"""
Metal GPU 引擎整合測試

測試與專案其他模組的整合。
"""

import pytest
import numpy as np
import pandas as pd

from src.backtester.metal_engine import MetalBacktestEngine


class TestIntegration:
    """整合測試"""

    def test_with_pandas_series(self):
        """測試：與 pandas Series 整合"""
        engine = MetalBacktestEngine()

        # Pandas Series
        prices_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

        # 轉為 numpy 並計算
        sma = engine.gpu_sma(prices_series.values, period=3)

        assert len(sma) == len(prices_series)
        assert np.isnan(sma[0])
        assert np.isclose(sma[2], 2.0)

    def test_with_pandas_dataframe(self):
        """測試：與 pandas DataFrame 整合"""
        engine = MetalBacktestEngine()

        # DataFrame
        df = pd.DataFrame({
            'open': [99, 100, 101],
            'high': [101, 102, 103],
            'low': [98, 99, 100],
            'close': [100, 101, 102],
        })

        # 提取收盤價並計算
        close_prices = df['close'].values
        sma = engine.gpu_sma(close_prices, period=2)

        assert len(sma) == len(df)
        assert np.isnan(sma[0])
        assert np.isclose(sma[1], 100.5)

    def test_real_world_strategy(self):
        """測試：真實策略回測"""
        engine = MetalBacktestEngine()

        # 生成模擬真實資料
        np.random.seed(42)
        T = 1000
        prices = 100 + np.cumsum(np.random.randn(T) * 0.5)
        prices = prices.reshape(-1, 1)

        def momentum_strategy(prices, lookback=20, threshold=0.02):
            """動量策略"""
            prices_1d = prices[:, 0]
            signals = np.zeros(len(prices))

            for i in range(lookback, len(prices)):
                # 計算動量
                momentum = (prices_1d[i] - prices_1d[i - lookback]) / prices_1d[i - lookback]

                # 超過閾值才開倉
                signals[i] = 1.0 if momentum > threshold else 0.0

            return signals

        # 多組參數
        param_grid = [
            {"lookback": lb, "threshold": th}
            for lb in [10, 20, 30]
            for th in [0.01, 0.02, 0.03]
        ]

        results = engine.batch_backtest(prices, param_grid, momentum_strategy)

        # 驗證
        assert len(results) == len(param_grid)
        assert all(r.total_return != 0 for r in results)  # 應該有報酬
        assert all(r.execution_time_ms > 0 for r in results)

    def test_empty_signals_handling(self):
        """測試：空訊號處理"""
        engine = MetalBacktestEngine()

        prices = np.random.randn(100, 1) * 10 + 100

        def always_zero_strategy(prices):
            """永遠不開倉"""
            return np.zeros(len(prices))

        param_grid = [{}]
        results = engine.batch_backtest(prices, param_grid, always_zero_strategy)

        # 沒有交易，報酬應該是 0
        assert results[0].total_return == 0.0

    def test_large_parameter_grid(self):
        """測試：大參數網格（壓力測試）"""
        engine = MetalBacktestEngine()

        prices = np.random.randn(500, 1) * 10 + 100

        def simple_strategy(prices, threshold):
            signals = np.zeros(len(prices))
            signals[prices[:, 0] > threshold] = 1.0
            return signals

        # 100 組參數
        param_grid = [
            {"threshold": threshold}
            for threshold in np.linspace(90, 110, 100)
        ]

        results = engine.batch_backtest(prices, param_grid, simple_strategy)

        assert len(results) == 100
        assert all(isinstance(r.total_return, float) for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
