"""
向量化效能基準測試

驗證 Polars 後端的效能提升（目標：5-10x）
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None

from src.backtester.engine import BacktestEngine, BacktestConfig
from src.backtester.vectorized import (
    vectorized_sma,
    vectorized_ema,
    vectorized_rsi,
    vectorized_positions,
    vectorized_pnl,
    pandas_to_polars,
)


# ============================================================================
# 測試資料生成
# ============================================================================

def generate_test_data(n_rows: int = 10000) -> pd.DataFrame:
    """產生測試用 OHLCV 資料"""
    dates = pd.date_range(start='2020-01-01', periods=n_rows, freq='1h')

    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
    high = close + np.abs(np.random.randn(n_rows))
    low = close - np.abs(np.random.randn(n_rows))
    open_ = close + np.random.randn(n_rows) * 0.5
    volume = np.random.randint(1000, 10000, n_rows)

    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


# ============================================================================
# 效能基準測試
# ============================================================================

@pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")
class TestVectorizedPerformance:
    """向量化效能測試"""

    @pytest.fixture
    def small_data(self):
        """小資料集（1萬行）"""
        return generate_test_data(10_000)

    @pytest.fixture
    def medium_data(self):
        """中資料集（10萬行）"""
        return generate_test_data(100_000)

    @pytest.fixture
    def large_data(self):
        """大資料集（100萬行）"""
        return generate_test_data(1_000_000)

    def test_sma_performance(self, medium_data):
        """測試 SMA 向量化效能"""
        close_pd = medium_data['close']
        close_pl = pl.Series(close_pd)

        # Pandas
        start = time.perf_counter()
        sma_pd = vectorized_sma(close_pd, 20)
        pandas_time = time.perf_counter() - start

        # Polars
        start = time.perf_counter()
        sma_pl = vectorized_sma(close_pl, 20)
        polars_time = time.perf_counter() - start

        speedup = pandas_time / polars_time
        print(f"\nSMA Speedup: {speedup:.2f}x")
        print(f"Pandas: {pandas_time*1000:.2f}ms | Polars: {polars_time*1000:.2f}ms")

        assert speedup >= 1.5, f"Polars should be at least 1.5x faster, got {speedup:.2f}x"

    def test_ema_performance(self, medium_data):
        """測試 EMA 向量化效能"""
        close_pd = medium_data['close']
        close_pl = pl.Series(close_pd)

        # Pandas
        start = time.perf_counter()
        ema_pd = vectorized_ema(close_pd, 20)
        pandas_time = time.perf_counter() - start

        # Polars
        start = time.perf_counter()
        ema_pl = vectorized_ema(close_pl, 20)
        polars_time = time.perf_counter() - start

        speedup = pandas_time / polars_time
        print(f"\nEMA Speedup: {speedup:.2f}x")
        print(f"Pandas: {pandas_time*1000:.2f}ms | Polars: {polars_time*1000:.2f}ms")

        assert speedup >= 1.5

    def test_rsi_performance(self, medium_data):
        """測試 RSI 向量化效能"""
        close_pd = medium_data['close']
        close_pl = pl.Series(close_pd)

        # Pandas
        start = time.perf_counter()
        rsi_pd = vectorized_rsi(close_pd, 14)
        pandas_time = time.perf_counter() - start

        # Polars
        start = time.perf_counter()
        rsi_pl = vectorized_rsi(close_pl, 14)
        polars_time = time.perf_counter() - start

        speedup = pandas_time / polars_time
        print(f"\nRSI Speedup: {speedup:.2f}x")
        print(f"Pandas: {pandas_time*1000:.2f}ms | Polars: {polars_time*1000:.2f}ms")

        assert speedup >= 1.2

    def test_position_calculation_performance(self, medium_data):
        """測試部位計算效能"""
        # 產生隨機訊號
        np.random.seed(42)
        signals_pd = pd.Series(
            np.random.choice([1, -1, 0], size=len(medium_data), p=[0.1, 0.1, 0.8]),
            index=medium_data.index
        )
        signals_pl = pl.Series(signals_pd)

        # Pandas
        start = time.perf_counter()
        pos_pd = vectorized_positions(signals_pd)
        pandas_time = time.perf_counter() - start

        # Polars
        start = time.perf_counter()
        pos_pl = vectorized_positions(signals_pl)
        polars_time = time.perf_counter() - start

        speedup = pandas_time / polars_time
        print(f"\nPosition Calculation Speedup: {speedup:.2f}x")
        print(f"Pandas: {pandas_time*1000:.2f}ms | Polars: {polars_time*1000:.2f}ms")

        assert speedup >= 2.0, f"Expected 2x speedup, got {speedup:.2f}x"

    def test_pnl_calculation_performance(self, medium_data):
        """測試損益計算效能"""
        np.random.seed(42)
        positions_pd = pd.Series(
            np.random.choice([1, -1, 0], size=len(medium_data)),
            index=medium_data.index
        )
        positions_pl = pl.Series(positions_pd)
        prices_pd = medium_data['close']
        prices_pl = pl.Series(prices_pd)

        # Pandas
        start = time.perf_counter()
        pnl_pd = vectorized_pnl(positions_pd, prices_pd)
        pandas_time = time.perf_counter() - start

        # Polars
        start = time.perf_counter()
        pnl_pl = vectorized_pnl(positions_pl, prices_pl)
        polars_time = time.perf_counter() - start

        speedup = pandas_time / polars_time
        print(f"\nPnL Calculation Speedup: {speedup:.2f}x")
        print(f"Pandas: {pandas_time*1000:.2f}ms | Polars: {polars_time*1000:.2f}ms")

        assert speedup >= 2.0

    def test_dataframe_conversion_performance(self, large_data):
        """測試資料轉換效能"""
        # Pandas → Polars
        start = time.perf_counter()
        df_pl = pandas_to_polars(large_data)
        conversion_time = time.perf_counter() - start

        print(f"\nPandas → Polars conversion: {conversion_time*1000:.2f}ms")
        print(f"Rows: {len(large_data):,}")

        # 轉換應該很快（< 100ms for 1M rows）
        assert conversion_time < 0.1, f"Conversion too slow: {conversion_time*1000:.2f}ms"

    @pytest.mark.slow
    def test_full_backtest_performance(self, small_data):
        """測試完整回測效能（小資料集）"""
        # 建立簡單策略
        class SimpleStrategy:
            name = "Simple SMA Cross"
            params = {'fast': 10, 'slow': 20}

            def generate_signals(self, df):
                if isinstance(df, pl.DataFrame):
                    fast_sma = df['close'].rolling_mean(window_size=10)
                    slow_sma = df['close'].rolling_mean(window_size=20)

                    long_entry = fast_sma > slow_sma
                    long_exit = fast_sma < slow_sma
                    short_entry = pl.Series([False] * len(df))
                    short_exit = pl.Series([False] * len(df))
                else:
                    fast_sma = df['close'].rolling(10).mean()
                    slow_sma = df['close'].rolling(20).mean()

                    long_entry = fast_sma > slow_sma
                    long_exit = fast_sma < slow_sma
                    short_entry = pd.Series(False, index=df.index)
                    short_exit = pd.Series(False, index=df.index)

                return long_entry, long_exit, short_entry, short_exit

        strategy = SimpleStrategy()

        # 測試配置
        base_config = {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'start_date': datetime(2020, 1, 1),
            'end_date': datetime(2020, 12, 31),
            'initial_capital': 10000.0,
            'leverage': 1
        }

        # Pandas + VectorBT（基準）
        config_pandas = BacktestConfig(**base_config, use_polars=False, vectorized=False)
        engine_pandas = BacktestEngine(config_pandas)

        start = time.perf_counter()
        result_pandas = engine_pandas.run(strategy, data=small_data)
        pandas_time = time.perf_counter() - start

        # Pandas + Vectorized
        config_vec_pd = BacktestConfig(**base_config, use_polars=False, vectorized=True)
        engine_vec_pd = BacktestEngine(config_vec_pd)

        start = time.perf_counter()
        result_vec_pd = engine_vec_pd.run(strategy, data=small_data)
        vec_pd_time = time.perf_counter() - start

        # Polars + Vectorized
        config_polars = BacktestConfig(**base_config, use_polars=True, vectorized=True)
        engine_polars = BacktestEngine(config_polars)

        start = time.perf_counter()
        result_polars = engine_polars.run(strategy, data=small_data)
        polars_time = time.perf_counter() - start

        # 計算提升倍數
        vec_pd_speedup = pandas_time / vec_pd_time
        polars_speedup = pandas_time / polars_time

        print(f"\n{'='*60}")
        print("完整回測效能比較（10,000 rows）")
        print(f"{'='*60}")
        print(f"Pandas + VectorBT:       {pandas_time*1000:>8.2f}ms (baseline)")
        print(f"Pandas + Vectorized:     {vec_pd_time*1000:>8.2f}ms ({vec_pd_speedup:.2f}x)")
        print(f"Polars + Vectorized:     {polars_time*1000:>8.2f}ms ({polars_speedup:.2f}x)")
        print(f"{'='*60}")

        # 驗證結果一致性（允許小誤差）
        assert abs(result_pandas.total_return - result_polars.total_return) < 0.01

        # 效能目標：5x 提升
        assert polars_speedup >= 2.0, f"Expected 2x+ speedup, got {polars_speedup:.2f}x"


# ============================================================================
# 正確性測試
# ============================================================================

class TestVectorizedCorrectness:
    """驗證向量化計算的正確性"""

    def test_sma_correctness(self):
        """驗證 SMA 計算正確"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        sma_pd = vectorized_sma(data, 3)
        expected = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

        pd.testing.assert_series_equal(sma_pd, expected)

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")
    def test_polars_pandas_equivalence(self):
        """驗證 Polars 和 Pandas 結果一致"""
        data_pd = pd.Series(np.random.randn(1000))
        data_pl = pl.Series(data_pd)

        # SMA
        sma_pd = vectorized_sma(data_pd, 20)
        sma_pl = vectorized_sma(data_pl, 20)
        assert np.allclose(sma_pd.values, sma_pl.to_numpy(), rtol=1e-5, equal_nan=True)

        # EMA
        ema_pd = vectorized_ema(data_pd, 20)
        ema_pl = vectorized_ema(data_pl, 20)
        assert np.allclose(ema_pd.values, ema_pl.to_numpy(), rtol=1e-5, equal_nan=True)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
