"""
測試 src/benchmark/runners.py 的修復

驗證項目：
1. generate_ohlcv_data 函數正確性
2. DataFrameRunner 基本功能
3. EngineRunner 基本功能
4. GPURunner 基本功能
5. Polars 函數一致性錯誤處理
6. 常數使用正確性
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 添加專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.benchmark.runners import (
    generate_ohlcv_data,
    DataFrameRunner,
    EngineRunner,
    GPURunner,
    PRICE_START,
    PRICE_VOLATILITY,
    INTRADAY_NOISE,
    VOLUME_MIN,
    VOLUME_MAX,
    POLARS_AVAILABLE
)

try:
    import polars as pl
except ImportError:
    pl = None


# ============================================================================
# 1. 測試 generate_ohlcv_data
# ============================================================================

class TestGenerateOHLCV:
    """測試 OHLCV 資料生成"""

    def test_basic_generation(self):
        """測試基本生成功能"""
        data = generate_ohlcv_data(100, seed=42)

        # 檢查欄位
        assert isinstance(data, pd.DataFrame)
        assert list(data.columns) == ['open', 'high', 'low', 'close', 'volume']

        # 檢查大小
        assert len(data) == 100

        # 檢查沒有 NaN
        assert not data.isna().any().any()

    def test_ohlc_validity(self):
        """測試 OHLC 資料合理性"""
        data = generate_ohlcv_data(1000, seed=42)

        # high >= close
        assert (data['high'] >= data['close']).all()

        # high >= open
        assert (data['high'] >= data['open']).all()

        # low <= close
        assert (data['low'] <= data['close']).all()

        # low <= open
        assert (data['low'] <= data['open']).all()

    def test_volume_range(self):
        """測試成交量範圍"""
        data = generate_ohlcv_data(1000, seed=42)

        # 成交量在合理範圍
        assert (data['volume'] >= VOLUME_MIN).all()
        assert (data['volume'] < VOLUME_MAX).all()

    def test_reproducibility(self):
        """測試可重現性（相同 seed 產生相同資料）"""
        data1 = generate_ohlcv_data(100, seed=123)
        data2 = generate_ohlcv_data(100, seed=123)

        pd.testing.assert_frame_equal(data1, data2)

    def test_randomness(self):
        """測試隨機性（不同 seed 產生不同資料）"""
        data1 = generate_ohlcv_data(100, seed=123)
        data2 = generate_ohlcv_data(100, seed=456)

        # 至少有一些值不同
        assert not data1.equals(data2)


# ============================================================================
# 2. 測試 DataFrameRunner
# ============================================================================

class TestDataFrameRunner:
    """測試 DataFrame 操作效能測試器"""

    @pytest.fixture
    def runner(self):
        """建立測試用 runner（快速設定）"""
        return DataFrameRunner(warmup=1, iterations=2, track_memory=False)

    def test_runner_initialization(self, runner):
        """測試初始化"""
        assert runner.warmup == 1
        assert runner.iterations == 2
        assert runner.track_memory is False

    def test_benchmark_rolling_mean(self, runner):
        """測試滾動平均 benchmark"""
        report = runner.benchmark_rolling_mean(
            sizes=[1000],
            window=20
        )

        # 檢查報告結構
        assert report is not None
        assert hasattr(report, 'timing_results')
        assert len(report.timing_results) > 0

        # 至少有 Pandas 測試
        assert any('Pandas' in b.name for b in report.timing_results)

    def test_benchmark_where(self, runner):
        """測試條件選擇 benchmark"""
        report = runner.benchmark_where(
            sizes=[1000],
            threshold=100.0
        )

        assert report is not None
        assert len(report.timing_results) > 0
        assert any('Pandas' in b.name for b in report.timing_results)

    def test_benchmark_ewm(self, runner):
        """測試指數加權平均 benchmark"""
        report = runner.benchmark_ewm(
            sizes=[1000],
            span=20
        )

        assert report is not None
        assert len(report.timing_results) > 0
        assert any('Pandas' in b.name for b in report.timing_results)

    def test_pandas_rolling_mean(self, runner):
        """測試 Pandas 滾動平均實作"""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = runner._pandas_rolling_mean(series, window=3)

        assert isinstance(result, pd.Series)
        assert len(result) == len(series)
        assert not result.isna().all()  # 應該有值

    def test_pandas_where(self, runner):
        """測試 Pandas where 實作"""
        series = pd.Series([50, 100, 150, 200])
        result = runner._pandas_where(series, threshold=100.0)

        assert isinstance(result, pd.Series)
        assert result.iloc[0] == 0  # 50 < 100
        assert result.iloc[1] == 0  # 100 = 100 (不大於)
        assert result.iloc[2] == 150  # 150 > 100
        assert result.iloc[3] == 200  # 200 > 100

    def test_pandas_ewm(self, runner):
        """測試 Pandas EWM 實作"""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = runner._pandas_ewm(series, span=3)

        assert isinstance(result, pd.Series)
        assert len(result) == len(series)
        assert not result.isna().any()

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
    def test_polars_rolling_mean(self, runner):
        """測試 Polars 滾動平均實作"""
        series = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = runner._polars_rolling_mean(series, window=3)

        assert isinstance(result, pl.Series)
        assert len(result) == len(series)

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
    def test_polars_where(self, runner):
        """測試 Polars where 實作"""
        series = pl.Series([50, 100, 150, 200])
        result = runner._polars_where(series, threshold=100.0)

        # Polars 的 when().then().otherwise() 回傳 Expr
        # 需要在 DataFrame 中使用
        assert result is not None

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
    def test_polars_ewm(self, runner):
        """測試 Polars EWM 實作"""
        series = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = runner._polars_ewm(series, span=3)

        assert isinstance(result, pl.Series)
        assert len(result) == len(series)

    def test_polars_error_handling_when_not_available(self, runner):
        """測試當 Polars 不可用時的錯誤處理"""
        if not POLARS_AVAILABLE:
            with pytest.raises(ImportError, match="Polars not available"):
                runner._polars_rolling_mean(None, window=3)

            with pytest.raises(ImportError, match="Polars not available"):
                runner._polars_where(None, threshold=100.0)

            with pytest.raises(ImportError, match="Polars not available"):
                runner._polars_ewm(None, span=3)


# ============================================================================
# 3. 測試 EngineRunner
# ============================================================================

class TestEngineRunner:
    """測試回測引擎效能測試器"""

    @pytest.fixture
    def runner(self):
        """建立測試用 runner"""
        return EngineRunner(warmup=1, iterations=2, track_memory=False)

    def test_runner_initialization(self, runner):
        """測試初始化"""
        assert runner.warmup == 1
        assert runner.iterations == 2
        assert runner.track_memory is False

    def test_benchmark_backtest(self, runner):
        """測試回測 benchmark"""
        report = runner.benchmark_backtest(
            data_sizes=[1000],
            strategy='ma_cross'
        )

        assert report is not None
        assert len(report.timing_results) > 0
        assert any('Pandas' in b.name for b in report.timing_results)

    def test_vectorized_pandas_backtest(self, runner):
        """測試 Pandas 向量化回測"""
        data = generate_ohlcv_data(1000, seed=42)
        result = runner._vectorized_pandas_backtest(data, strategy='ma_cross')

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        assert not result.isna().all()

    def test_unknown_strategy(self, runner):
        """測試未知策略"""
        data = generate_ohlcv_data(100, seed=42)

        with pytest.raises(ValueError, match="Unknown strategy"):
            runner._vectorized_pandas_backtest(data, strategy='unknown')

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not available")
    def test_vectorized_polars_backtest(self, runner):
        """測試 Polars 向量化回測"""
        data = generate_ohlcv_data(1000, seed=42)
        pl_data = pl.from_pandas(data)
        result = runner._vectorized_polars_backtest(pl_data, strategy='ma_cross')

        assert result is not None
        # Polars 回傳 DataFrame
        assert isinstance(result, pl.DataFrame)


# ============================================================================
# 4. 測試 GPURunner
# ============================================================================

class TestGPURunner:
    """測試 GPU 批量優化效能測試器"""

    @pytest.fixture
    def runner(self):
        """建立測試用 runner"""
        return GPURunner(warmup=0, iterations=1, track_memory=False)

    def test_runner_initialization(self, runner):
        """測試初始化"""
        assert runner.warmup == 0
        assert runner.iterations == 1
        assert runner.track_memory is False

    def test_backend_detection(self, runner):
        """測試後端檢測"""
        backends = runner.available_backends

        # 至少有 CPU
        assert 'cpu' in backends
        assert isinstance(backends, list)

    def test_cpu_always_available(self, runner):
        """測試 CPU 總是可用"""
        assert 'cpu' in runner.available_backends


# ============================================================================
# 5. 測試常數使用
# ============================================================================

class TestConstants:
    """測試常數定義與使用"""

    def test_constants_defined(self):
        """測試所有常數都已定義"""
        assert PRICE_START == 100.0
        assert PRICE_VOLATILITY == 0.01
        assert INTRADAY_NOISE == 0.005
        assert VOLUME_MIN == 1000
        assert VOLUME_MAX == 100000

    def test_constants_used_in_generation(self):
        """測試常數在生成函數中使用（間接測試）"""
        # 生成資料
        data = generate_ohlcv_data(1000, seed=42)

        # 價格應該在合理範圍（起始價 ± 一些波動）
        # 使用 PRICE_START 作為基準
        assert data['close'].min() > PRICE_START * 0.5
        assert data['close'].max() < PRICE_START * 2.0

        # 成交量範圍
        assert (data['volume'] >= VOLUME_MIN).all()
        assert (data['volume'] < VOLUME_MAX).all()


# ============================================================================
# 6. 整合測試
# ============================================================================

class TestIntegration:
    """整合測試"""

    def test_full_dataframe_benchmark(self):
        """測試完整 DataFrame benchmark 流程"""
        runner = DataFrameRunner(warmup=1, iterations=2, track_memory=False)

        # 執行所有操作
        reports = {
            'rolling_mean': runner.benchmark_rolling_mean([1000]),
            'where': runner.benchmark_where([1000]),
            'ewm': runner.benchmark_ewm([1000])
        }

        # 所有報告都應該成功
        for name, report in reports.items():
            assert report is not None
            assert len(report.timing_results) > 0
            assert hasattr(report, 'summary')

    def test_full_engine_benchmark(self):
        """測試完整 Engine benchmark 流程"""
        runner = EngineRunner(warmup=1, iterations=2, track_memory=False)
        report = runner.benchmark_backtest([1000])

        assert report is not None
        assert len(report.timing_results) > 0

    def test_report_generation(self):
        """測試報告生成"""
        runner = DataFrameRunner(warmup=1, iterations=2, track_memory=False)
        report = runner.benchmark_rolling_mean([1000])

        # 測試 summary
        summary = report.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

        # 測試 to_markdown
        markdown = report.to_markdown()
        assert isinstance(markdown, str)
        assert '##' in markdown

        # 測試 to_json
        json_str = report.to_json()
        assert isinstance(json_str, str)
        assert '{' in json_str


# ============================================================================
# 執行測試
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
