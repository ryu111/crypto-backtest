"""
Benchmark Runners - 效能測試執行器

提供三類效能測試：
1. DataFrameRunner - Pandas vs Polars 操作效能
2. EngineRunner - 回測引擎效能比較
3. GPURunner - GPU 批量優化效能
"""

from typing import List, Optional, Callable, Dict, Any
import numpy as np
import pandas as pd
import logging

from .framework import BenchmarkSuite, BenchmarkReport

# 嘗試導入可選依賴
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None

# DataFrameOps
try:
    from ..strategies.utils.dataframe_ops import DataFrameOps
except ImportError:
    try:
        from src.strategies.utils.dataframe_ops import DataFrameOps
    except ImportError:
        DataFrameOps = None  # type: ignore

# GPU 批量優化器
try:
    from ..optimizer.gpu_batch import GPUBatchOptimizer
except ImportError:
    try:
        from src.optimizer.gpu_batch import GPUBatchOptimizer
    except ImportError:
        GPUBatchOptimizer = None  # type: ignore

logger = logging.getLogger(__name__)


# ============================================================================
# 資料生成常數
# ============================================================================

# 模擬資料生成參數
PRICE_START = 100.0        # 起始價格
PRICE_VOLATILITY = 0.01    # 日波動率（1%）
INTRADAY_NOISE = 0.005     # 日內波動（0.5%）
VOLUME_MIN = 1000          # 最小成交量
VOLUME_MAX = 100000        # 最大成交量

# ============================================================================
# 資料生成工具
# ============================================================================

def generate_ohlcv_data(size: int, seed: Optional[int] = None) -> pd.DataFrame:
    """
    生成模擬 OHLCV 資料

    Args:
        size: 資料長度
        seed: 隨機種子

    Returns:
        包含 open, high, low, close, volume 的 DataFrame
    """
    if seed is not None:
        np.random.seed(seed)

    # 生成收盤價（隨機遊走）
    returns = np.random.randn(size) * PRICE_VOLATILITY
    close = PRICE_START * np.exp(np.cumsum(returns))

    # 生成 OHLC（加入日內波動）
    noise = np.random.randn(size, 3) * INTRADAY_NOISE
    open_prices = close * (1 + noise[:, 0])
    high_prices = close * (1 + np.abs(noise[:, 1]))
    low_prices = close * (1 - np.abs(noise[:, 2]))

    # 確保 OHLC 合理（high >= low, close 在 high/low 之間）
    high_prices = np.maximum(high_prices, close)
    high_prices = np.maximum(high_prices, open_prices)
    low_prices = np.minimum(low_prices, close)
    low_prices = np.minimum(low_prices, open_prices)

    # 生成成交量
    volume = np.random.randint(VOLUME_MIN, VOLUME_MAX, size=size).astype(float)

    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close,
        'volume': volume
    })

    return df


# ============================================================================
# DataFrame 效能測試
# ============================================================================

class DataFrameRunner:
    """
    DataFrame 操作效能測試

    測試 Pandas vs Polars 在常見操作上的效能差異：
    - 滾動平均 (rolling mean)
    - 條件選擇 (where)
    - 指數加權平均 (ewm)
    """

    def __init__(
        self,
        warmup: int = 3,
        iterations: int = 10,
        track_memory: bool = False
    ):
        """
        初始化 DataFrameRunner

        Args:
            warmup: Warmup 次數
            iterations: 測量次數
            track_memory: 是否追蹤記憶體使用
        """
        self.warmup = warmup
        self.iterations = iterations
        self.track_memory = track_memory

    def benchmark_rolling_mean(
        self,
        sizes: List[int],
        window: int = 20
    ) -> BenchmarkReport:
        """
        測試滾動平均效能

        Args:
            sizes: 資料大小列表 (e.g., [10000, 50000, 100000])
            window: 滾動視窗大小

        Returns:
            BenchmarkReport
        """
        suite = BenchmarkSuite(
            warmup=self.warmup,
            iterations=self.iterations,
            track_memory=self.track_memory,
            metadata={
                'operation': 'rolling_mean',
                'window': window,
                'sizes': sizes
            }
        )

        for size in sizes:
            data = generate_ohlcv_data(size, seed=42)

            # Pandas
            suite.add_benchmark(
                f"Pandas_{size}",
                self._pandas_rolling_mean,
                data['close'],
                window
            )

            # Polars (如果可用)
            if POLARS_AVAILABLE and pl is not None:
                suite.add_benchmark(
                    f"Polars_{size}",
                    self._polars_rolling_mean,
                    pl.Series(data['close']),
                    window
                )

            # DataFrameOps (如果可用)
            if DataFrameOps is not None:
                df_ops = DataFrameOps(data)
                suite.add_benchmark(
                    f"DataFrameOps_Pandas_{size}",
                    self._dataframe_ops_rolling_mean,
                    df_ops,
                    'close',
                    window
                )

                if POLARS_AVAILABLE and pl is not None:
                    df_ops_polars = DataFrameOps(pl.from_pandas(data))
                    suite.add_benchmark(
                        f"DataFrameOps_Polars_{size}",
                        self._dataframe_ops_rolling_mean,
                        df_ops_polars,
                        'close',
                        window
                    )

        return suite.run()

    def benchmark_where(
        self,
        sizes: List[int],
        threshold: float = 100.0
    ) -> BenchmarkReport:
        """
        測試條件選擇效能

        Args:
            sizes: 資料大小列表
            threshold: 條件閾值

        Returns:
            BenchmarkReport
        """
        suite = BenchmarkSuite(
            warmup=self.warmup,
            iterations=self.iterations,
            track_memory=self.track_memory,
            metadata={
                'operation': 'where',
                'threshold': threshold,
                'sizes': sizes
            }
        )

        for size in sizes:
            data = generate_ohlcv_data(size, seed=42)

            # Pandas
            suite.add_benchmark(
                f"Pandas_{size}",
                self._pandas_where,
                data['close'],
                threshold
            )

            # Polars (如果可用)
            if POLARS_AVAILABLE and pl is not None:
                suite.add_benchmark(
                    f"Polars_{size}",
                    self._polars_where,
                    pl.Series(data['close']),
                    threshold
                )

        return suite.run()

    def benchmark_ewm(
        self,
        sizes: List[int],
        span: int = 20
    ) -> BenchmarkReport:
        """
        測試指數加權平均效能

        Args:
            sizes: 資料大小列表
            span: EWM span

        Returns:
            BenchmarkReport
        """
        suite = BenchmarkSuite(
            warmup=self.warmup,
            iterations=self.iterations,
            track_memory=self.track_memory,
            metadata={
                'operation': 'ewm',
                'span': span,
                'sizes': sizes
            }
        )

        for size in sizes:
            data = generate_ohlcv_data(size, seed=42)

            # Pandas
            suite.add_benchmark(
                f"Pandas_{size}",
                self._pandas_ewm,
                data['close'],
                span
            )

            # Polars (如果可用)
            if POLARS_AVAILABLE and pl is not None:
                suite.add_benchmark(
                    f"Polars_{size}",
                    self._polars_ewm,
                    pl.Series(data['close']),
                    span
                )

        return suite.run()

    # ========== Pandas 實作 ==========

    def _pandas_rolling_mean(self, series: pd.Series, window: int):
        """Pandas 滾動平均"""
        return series.rolling(window=window, min_periods=1).mean()

    def _pandas_where(self, series: pd.Series, threshold: float):
        """Pandas 條件選擇"""
        return series.where(series > threshold, 0)

    def _pandas_ewm(self, series: pd.Series, span: int):
        """Pandas 指數加權平均"""
        return series.ewm(span=span, adjust=False).mean()

    # ========== Polars 實作 ==========

    def _polars_rolling_mean(self, series, window: int):
        """Polars 滾動平均"""
        if pl is None:
            raise ImportError("Polars not available")
        return series.rolling_mean(window_size=window, min_periods=1)

    def _polars_where(self, series, threshold: float):
        """Polars 條件選擇"""
        # Polars 使用 when().then().otherwise() 語法
        if pl is None:
            raise ImportError("Polars not available")
        return pl.when(series > threshold).then(series).otherwise(0)

    def _polars_ewm(self, series, span: int):
        """Polars 指數加權平均"""
        if pl is None:
            raise ImportError("Polars not available")
        return series.ewm_mean(span=span)

    # ========== DataFrameOps 實作 ==========

    def _dataframe_ops_rolling_mean(self, df_ops: Any, col: str, window: int):
        """DataFrameOps 滾動平均"""
        return df_ops[col].rolling_mean(window)


# ============================================================================
# 回測引擎效能測試
# ============================================================================

class EngineRunner:
    """
    回測引擎效能測試

    比較不同回測引擎的效能：
    - Vectorized Pandas
    - Vectorized Polars
    - VectorBT (如果可用)
    """

    def __init__(
        self,
        warmup: int = 1,
        iterations: int = 5,
        track_memory: bool = True
    ):
        """
        初始化 EngineRunner

        Args:
            warmup: Warmup 次數（回測較慢，減少 warmup）
            iterations: 測量次數
            track_memory: 是否追蹤記憶體使用
        """
        self.warmup = warmup
        self.iterations = iterations
        self.track_memory = track_memory

    def benchmark_backtest(
        self,
        data_sizes: List[int],
        strategy: str = "ma_cross"
    ) -> BenchmarkReport:
        """
        測試完整回測效能

        Args:
            data_sizes: 資料大小列表
            strategy: 策略類型 ('ma_cross', 'rsi')

        Returns:
            BenchmarkReport
        """
        suite = BenchmarkSuite(
            warmup=self.warmup,
            iterations=self.iterations,
            track_memory=self.track_memory,
            metadata={
                'operation': 'backtest',
                'strategy': strategy,
                'sizes': data_sizes
            }
        )

        for size in data_sizes:
            data = generate_ohlcv_data(size, seed=42)

            # Vectorized Pandas
            suite.add_benchmark(
                f"Vectorized_Pandas_{size}",
                self._vectorized_pandas_backtest,
                data,
                strategy
            )

            # Vectorized Polars (如果可用)
            if POLARS_AVAILABLE and pl is not None:
                suite.add_benchmark(
                    f"Vectorized_Polars_{size}",
                    self._vectorized_polars_backtest,
                    pl.from_pandas(data),
                    strategy
                )

        return suite.run()

    def _vectorized_pandas_backtest(self, data: pd.DataFrame, strategy: str):
        """Pandas 向量化回測"""
        if strategy == "ma_cross":
            # 簡單 MA Cross 策略
            fast_ma = data['close'].rolling(10).mean()
            slow_ma = data['close'].rolling(30).mean()

            signals = pd.Series(0, index=data.index)
            signals[fast_ma > slow_ma] = 1
            signals[fast_ma < slow_ma] = -1

            # 計算報酬
            returns: pd.Series = data['close'].pct_change() * signals.shift(1)
            cumulative_returns: pd.Series = returns.add(1).cumprod()

            return cumulative_returns
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _vectorized_polars_backtest(self, data, strategy: str):
        """Polars 向量化回測"""
        if pl is None:
            raise ImportError("Polars not available")

        if strategy == "ma_cross":
            # 簡單 MA Cross 策略
            data = data.with_columns([
                pl.col('close').rolling_mean(10).alias('fast_ma'),
                pl.col('close').rolling_mean(30).alias('slow_ma')
            ])

            # 生成信號
            data = data.with_columns(
                pl.when(pl.col('fast_ma') > pl.col('slow_ma'))
                .then(1)
                .when(pl.col('fast_ma') < pl.col('slow_ma'))
                .then(-1)
                .otherwise(0)
                .alias('signal')
            )

            # 計算報酬
            data = data.with_columns(
                (pl.col('close').pct_change() * pl.col('signal').shift(1))
                .alias('returns')
            )

            data = data.with_columns(
                (1 + pl.col('returns')).cum_prod().alias('cumulative_returns')
            )

            return data.select('cumulative_returns')
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


# ============================================================================
# GPU 批量優化效能測試
# ============================================================================

class GPURunner:
    """
    GPU 批量優化效能測試

    測試 GPU 加速效果：
    - MLX (Apple Silicon)
    - MPS (PyTorch Metal)
    - CPU (baseline)
    """

    def __init__(
        self,
        warmup: int = 0,
        iterations: int = 3,
        track_memory: bool = True
    ):
        """
        初始化 GPURunner

        Args:
            warmup: Warmup 次數（GPU 優化較慢，跳過 warmup）
            iterations: 測量次數
            track_memory: 是否追蹤記憶體使用
        """
        self.warmup = warmup
        self.iterations = iterations
        self.track_memory = track_memory

        # 檢測可用的 GPU 後端
        self._available_backends = self._detect_backends()

    def _detect_backends(self) -> List[str]:
        """
        檢測可用的 GPU 後端

        Returns:
            可用的後端列表，至少包含 'cpu'
        """
        backends = ['cpu']  # CPU 總是可用

        if GPUBatchOptimizer is None:
            logger.warning("GPUBatchOptimizer not available")
            return backends

        # 嘗試 MLX
        try:
            optimizer = GPUBatchOptimizer(prefer_mlx=True, fallback_to_cpu=False)
            if optimizer.is_gpu_available():
                backends.append('mlx')
                logger.debug("MLX backend detected")
        except ImportError as e:
            logger.debug(f"MLX not available: {e}")
        except RuntimeError as e:
            logger.debug(f"MLX initialization failed: {e}")
        except ValueError as e:
            logger.debug(f"MLX configuration error: {e}")

        # 嘗試 MPS
        try:
            optimizer = GPUBatchOptimizer(prefer_mlx=False, fallback_to_cpu=False)
            if optimizer.is_gpu_available():
                backends.append('mps')
                logger.debug("MPS backend detected")
        except ImportError as e:
            logger.debug(f"MPS not available: {e}")
        except RuntimeError as e:
            logger.debug(f"MPS initialization failed: {e}")
        except ValueError as e:
            logger.debug(f"MPS configuration error: {e}")

        logger.info(f"Detected backends: {backends}")
        return backends

    @property
    def available_backends(self) -> List[str]:
        """取得可用的後端列表"""
        return self._available_backends

    def benchmark_batch_optimization(
        self,
        batch_sizes: List[int],
        data_size: int = 1000
    ) -> BenchmarkReport:
        """
        測試批量優化效能

        Args:
            batch_sizes: 批次大小列表
            data_size: 回測資料大小

        Returns:
            BenchmarkReport
        """
        if GPUBatchOptimizer is None:
            raise ImportError("GPUBatchOptimizer not available")

        suite = BenchmarkSuite(
            warmup=self.warmup,
            iterations=self.iterations,
            track_memory=self.track_memory,
            metadata={
                'operation': 'batch_optimization',
                'batch_sizes': batch_sizes,
                'data_size': data_size,
                'available_backends': self._available_backends
            }
        )

        # 準備測試資料
        data = generate_ohlcv_data(data_size, seed=42)
        price_data = data[['open', 'high', 'low', 'close', 'volume']].values.astype(np.float32)

        # 簡單策略函數
        def simple_strategy(prices: np.ndarray, params: Dict) -> np.ndarray:
            """簡單 MA Cross 策略"""
            close = prices[:, 3]  # close price
            fast_period = params['fast_period']
            slow_period = params['slow_period']

            # 簡單滾動平均（向量化）
            fast_ma = np.convolve(close, np.ones(fast_period)/fast_period, mode='same')
            slow_ma = np.convolve(close, np.ones(slow_period)/slow_period, mode='same')

            signals = np.zeros(len(close))
            signals[fast_ma > slow_ma] = 1
            signals[fast_ma < slow_ma] = -1

            return signals

        # 參數空間
        param_space = {
            'fast_period': {'type': 'int', 'low': 5, 'high': 20},
            'slow_period': {'type': 'int', 'low': 20, 'high': 50}
        }

        for batch_size in batch_sizes:
            # CPU baseline
            if 'cpu' in self._available_backends:
                suite.add_benchmark(
                    f"CPU_batch{batch_size}",
                    self._run_cpu_optimization,
                    price_data,
                    simple_strategy,
                    param_space,
                    batch_size
                )

            # MLX
            if 'mlx' in self._available_backends:
                suite.add_benchmark(
                    f"MLX_batch{batch_size}",
                    self._run_gpu_optimization,
                    price_data,
                    simple_strategy,
                    param_space,
                    batch_size,
                    prefer_mlx=True
                )

            # MPS
            if 'mps' in self._available_backends:
                suite.add_benchmark(
                    f"MPS_batch{batch_size}",
                    self._run_gpu_optimization,
                    price_data,
                    simple_strategy,
                    param_space,
                    batch_size,
                    prefer_mlx=False
                )

        return suite.run()

    def _run_cpu_optimization(
        self,
        price_data: np.ndarray,
        strategy_fn: Callable,
        param_space: Dict,
        batch_size: int
    ):
        """
        執行 CPU 優化（基準測試）

        Args:
            price_data: 價格資料陣列
            strategy_fn: 策略函數
            param_space: 參數空間
            batch_size: 批次大小

        Returns:
            優化結果

        Raises:
            ImportError: 當 GPUBatchOptimizer 不可用時
        """
        if GPUBatchOptimizer is None:
            raise ImportError("GPUBatchOptimizer not available")

        # 使用 fallback_to_cpu=True 並禁用所有 GPU 選項
        # prefer_mlx=False 且沒有 GPU 可用時會使用 CPU
        optimizer = GPUBatchOptimizer(
            prefer_mlx=False,
            fallback_to_cpu=True,
            verbose=False
        )

        # 注意：fallback_to_cpu=True 意味著當 GPU 不可用時會自動降級
        # 這個函數的目的是測試 CPU 效能作為基準線

        result = optimizer.batch_optimize(
            strategy_fn=strategy_fn,
            price_data=price_data,
            param_space=param_space,
            n_trials=batch_size,
            batch_size=batch_size,
            metric='sharpe_ratio'
        )

        return result

    def _run_gpu_optimization(
        self,
        price_data: np.ndarray,
        strategy_fn: Callable,
        param_space: Dict,
        batch_size: int,
        prefer_mlx: bool = True
    ):
        """
        執行 GPU 優化

        Args:
            price_data: 價格資料陣列
            strategy_fn: 策略函數
            param_space: 參數空間
            batch_size: 批次大小
            prefer_mlx: 是否優先使用 MLX（Apple Silicon）

        Returns:
            優化結果

        Raises:
            ImportError: 當 GPUBatchOptimizer 不可用時
            RuntimeError: 當 GPU 不可用時
        """
        if GPUBatchOptimizer is None:
            raise ImportError("GPUBatchOptimizer not available")

        optimizer = GPUBatchOptimizer(
            prefer_mlx=prefer_mlx,
            fallback_to_cpu=False,
            verbose=False
        )

        # 確保使用 GPU
        if not optimizer.is_gpu_available():
            raise RuntimeError("GPU not available")

        result = optimizer.batch_optimize(
            strategy_fn=strategy_fn,
            price_data=price_data,
            param_space=param_space,
            n_trials=batch_size,
            batch_size=batch_size,
            metric='sharpe_ratio'
        )

        return result


# ============================================================================
# 便利函數
# ============================================================================

def run_all_benchmarks(
    data_sizes: List[int] = [10000, 50000, 100000],
    batch_sizes: List[int] = [10, 50, 100],
    save_reports: bool = True
) -> Dict[str, BenchmarkReport]:
    """
    執行所有效能測試

    Args:
        data_sizes: 資料大小列表
        batch_sizes: 批次大小列表（GPU 測試用）
        save_reports: 是否儲存報告

    Returns:
        所有測試報告的字典
    """
    reports = {}

    # DataFrame 效能測試
    print("=" * 60)
    print("DataFrame Operations Benchmark")
    print("=" * 60)

    df_runner = DataFrameRunner()

    print("\n1. Rolling Mean")
    reports['rolling_mean'] = df_runner.benchmark_rolling_mean(data_sizes)
    print(reports['rolling_mean'].summary())

    print("\n2. Where (Conditional Selection)")
    reports['where'] = df_runner.benchmark_where(data_sizes)
    print(reports['where'].summary())

    print("\n3. EWM (Exponential Weighted Mean)")
    reports['ewm'] = df_runner.benchmark_ewm(data_sizes)
    print(reports['ewm'].summary())

    # 回測引擎效能測試
    print("\n" + "=" * 60)
    print("Backtest Engine Benchmark")
    print("=" * 60)

    engine_runner = EngineRunner()
    reports['backtest'] = engine_runner.benchmark_backtest(data_sizes)
    print(reports['backtest'].summary())

    # GPU 效能測試
    print("\n" + "=" * 60)
    print("GPU Batch Optimization Benchmark")
    print("=" * 60)

    gpu_runner = GPURunner()
    print(f"Available backends: {gpu_runner.available_backends}")

    if len(gpu_runner.available_backends) > 1:  # 不只 CPU
        reports['gpu_batch'] = gpu_runner.benchmark_batch_optimization(batch_sizes)
        print(reports['gpu_batch'].summary())
    else:
        print("⚠️ No GPU backends available, skipping GPU benchmark")

    # 儲存報告
    if save_reports:
        from pathlib import Path

        output_dir = Path("benchmark_results")
        output_dir.mkdir(exist_ok=True)

        for name, report in reports.items():
            # Markdown
            md_file = output_dir / f"{name}_report.md"
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(report.to_markdown())

            # JSON
            json_file = output_dir / f"{name}_report.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                f.write(report.to_json())

        print(f"\n✅ Reports saved to {output_dir}")

    return reports


if __name__ == "__main__":
    """執行所有效能測試"""
    run_all_benchmarks(
        data_sizes=[10000, 50000],
        batch_sizes=[10, 50],
        save_reports=True
    )
