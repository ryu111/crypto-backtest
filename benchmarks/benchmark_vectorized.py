"""
向量化效能基準測試

真實場景：完整回測流程的效能提升
"""

import pandas as pd
import numpy as np
from datetime import datetime
import time

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None

import sys
sys.path.insert(0, '/Users/sbu/Desktop/side project/合約交易')

from src.backtester.engine import BacktestEngine, BacktestConfig


def generate_test_data(n_rows: int = 100000) -> pd.DataFrame:
    """產生測試用 OHLCV 資料"""
    dates = pd.date_range(start='2020-01-01', periods=n_rows, freq='1h')

    np.random.seed(42)
    close = 10000 + np.cumsum(np.random.randn(n_rows) * 50)
    high = close + np.abs(np.random.randn(n_rows) * 30)
    low = close - np.abs(np.random.randn(n_rows) * 30)
    open_ = close + np.random.randn(n_rows) * 20
    volume = np.random.randint(100, 1000, n_rows)

    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


class TrendFollowingStrategy:
    """趨勢跟蹤策略（支援 Polars）"""

    name = "Trend Following"
    params = {'fast': 10, 'slow': 30, 'rsi_period': 14}

    def generate_signals(self, df):
        """產生交易訊號"""
        if isinstance(df, pl.DataFrame):
            # Polars 實作：建立新 DataFrame 並 select
            result_df = df.select([
                pl.col('close'),
                pl.col('close').rolling_mean(window_size=self.params['fast'], min_samples=1).alias('fast_sma'),
                pl.col('close').rolling_mean(window_size=self.params['slow'], min_samples=1).alias('slow_sma'),
            ])

            # RSI 計算
            delta = result_df['close'].diff()
            gain = delta.apply(lambda x: x if x > 0 else 0)
            loss = delta.apply(lambda x: -x if x < 0 else 0)
            avg_gain = gain.ewm_mean(span=self.params['rsi_period'])
            avg_loss = loss.ewm_mean(span=self.params['rsi_period'])
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            # 訊號（轉為布林 Series）
            long_entry = (result_df['fast_sma'] > result_df['slow_sma']) & (rsi < 70)
            long_exit = (result_df['fast_sma'] < result_df['slow_sma']) | (rsi > 80)
            short_entry = pl.Series('short_entry', [False] * len(df))
            short_exit = pl.Series('short_exit', [False] * len(df))

        else:
            # Pandas 實作
            close = df['close']

            fast_sma = close.rolling(window=self.params['fast'], min_periods=1).mean()
            slow_sma = close.rolling(window=self.params['slow'], min_periods=1).mean()

            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.ewm(span=self.params['rsi_period'], adjust=False).mean()
            avg_loss = loss.ewm(span=self.params['rsi_period'], adjust=False).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            # 訊號
            long_entry = (fast_sma > slow_sma) & (rsi < 70)
            long_exit = (fast_sma < slow_sma) | (rsi > 80)
            short_entry = pd.Series(False, index=df.index)
            short_exit = pd.Series(False, index=df.index)

        return long_entry, long_exit, short_entry, short_exit


def benchmark_backtest(data_size: int = 100000):
    """效能基準測試"""

    print(f"\n{'='*70}")
    print(f"向量化回測效能基準測試（{data_size:,} rows）")
    print(f"{'='*70}\n")

    # 產生測試資料
    print("產生測試資料...")
    data = generate_test_data(data_size)
    print(f"  資料大小: {len(data):,} rows × {len(data.columns)} columns")
    print(f"  記憶體: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

    # 策略
    strategy = TrendFollowingStrategy()

    # 測試配置
    base_config = {
        'symbol': 'BTCUSDT',
        'timeframe': '1h',
        'start_date': datetime(2020, 1, 1),
        'end_date': datetime(2021, 12, 31),
        'initial_capital': 10000.0,
        'leverage': 3
    }

    results = {}

    # ========================================================================
    # 1. Pandas + VectorBT（基準）
    # ========================================================================
    print("1️⃣  Pandas + VectorBT (baseline)")
    config_pandas = BacktestConfig(**base_config, use_polars=False, vectorized=False)
    engine_pandas = BacktestEngine(config_pandas)

    start = time.perf_counter()
    result_pandas = engine_pandas.run(strategy, data=data)
    pandas_time = time.perf_counter() - start

    results['pandas'] = {
        'time': pandas_time,
        'result': result_pandas
    }

    print(f"  執行時間: {pandas_time*1000:.2f} ms")
    print(f"  報酬率: {result_pandas.total_return:.2%}")
    print(f"  交易次數: {result_pandas.total_trades}\n")

    # ========================================================================
    # 2. Pandas + Vectorized
    # ========================================================================
    print("2️⃣  Pandas + Vectorized")
    config_vec_pd = BacktestConfig(**base_config, use_polars=False, vectorized=True)
    engine_vec_pd = BacktestEngine(config_vec_pd)

    start = time.perf_counter()
    result_vec_pd = engine_vec_pd.run(strategy, data=data)
    vec_pd_time = time.perf_counter() - start

    results['vec_pandas'] = {
        'time': vec_pd_time,
        'result': result_vec_pd
    }

    vec_pd_speedup = pandas_time / vec_pd_time
    print(f"  執行時間: {vec_pd_time*1000:.2f} ms")
    print(f"  加速比: {vec_pd_speedup:.2f}x")
    print(f"  報酬率: {result_vec_pd.total_return:.2%}\n")

    # ========================================================================
    # 3. Polars + Vectorized
    # ========================================================================
    if POLARS_AVAILABLE:
        print("3️⃣  Polars + Vectorized")
        config_polars = BacktestConfig(**base_config, use_polars=True, vectorized=True)
        engine_polars = BacktestEngine(config_polars)

        start = time.perf_counter()
        result_polars = engine_polars.run(strategy, data=data)
        polars_time = time.perf_counter() - start

        results['polars'] = {
            'time': polars_time,
            'result': result_polars
        }

        polars_speedup = pandas_time / polars_time
        print(f"  執行時間: {polars_time*1000:.2f} ms")
        print(f"  加速比: {polars_speedup:.2f}x")
        print(f"  報酬率: {result_polars.total_return:.2%}\n")
    else:
        print("3️⃣  Polars + Vectorized: SKIPPED (Polars not installed)\n")

    # ========================================================================
    # 總結
    # ========================================================================
    print(f"{'='*70}")
    print("總結")
    print(f"{'='*70}")
    print(f"{'方法':<25} {'時間':>12} {'加速比':>10} {'記憶體':>12}")
    print(f"{'-'*70}")

    for name, data_dict in results.items():
        t = data_dict['time']
        speedup = pandas_time / t
        print(f"{name:<25} {t*1000:>10.2f} ms {speedup:>8.2f}x      -")

    print(f"{'='*70}\n")

    # 驗證結果一致性
    if 'polars' in results:
        pandas_return = results['pandas']['result'].total_return
        polars_return = results['polars']['result'].total_return
        diff = abs(pandas_return - polars_return)

        if diff < 0.01:
            print("✅ 結果驗證：Polars 與 Pandas 結果一致")
        else:
            print(f"⚠️  結果驗證：差異 {diff:.4f} ({diff/pandas_return:.2%})")

    print()


if __name__ == '__main__':
    # 測試不同資料規模
    for size in [10_000, 50_000, 100_000]:
        benchmark_backtest(size)
        print("\n")
