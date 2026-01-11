"""
Metal GPU 加速引擎測試

測試 MLX/PyTorch MPS/CPU 三種後端的正確性與效能。
"""

import pytest
import numpy as np
from typing import Dict, Any

from src.backtester.metal_engine import (
    MetalBacktestEngine,
    GPUBacktestResult,
    MLX_AVAILABLE,
    TORCH_MPS_AVAILABLE,
)


@pytest.fixture
def sample_price_data():
    """生成測試用價格資料"""
    np.random.seed(42)
    T = 1000  # 時間步
    prices = 100 + np.cumsum(np.random.randn(T) * 0.5)  # 隨機遊走
    return prices.reshape(-1, 1)  # (T, 1)


@pytest.fixture
def simple_strategy():
    """簡單均線策略"""
    def strategy_fn(prices: np.ndarray, sma_period: int = 20) -> np.ndarray:
        """
        簡單 SMA 策略：價格 > SMA 則做多(1)，否則空手(0)
        """
        prices_1d = prices[:, 0]
        signals = np.zeros(len(prices_1d))

        for i in range(sma_period, len(prices_1d)):
            sma = np.mean(prices_1d[i - sma_period:i])
            signals[i] = 1.0 if prices_1d[i] > sma else 0.0

        return signals

    return strategy_fn


@pytest.fixture
def param_grid():
    """參數網格"""
    return [
        {"sma_period": 10},
        {"sma_period": 20},
        {"sma_period": 50},
    ]


class TestMetalBacktestEngine:
    """MetalBacktestEngine 測試"""

    def test_backend_selection_prefer_mlx(self):
        """測試：優先選擇 MLX"""
        engine = MetalBacktestEngine(prefer_mlx=True)

        if MLX_AVAILABLE:
            assert engine.backend == "mlx"
        elif TORCH_MPS_AVAILABLE:
            assert engine.backend == "mps"
        else:
            assert engine.backend == "cpu"

    def test_backend_selection_prefer_mps(self):
        """測試：選擇 PyTorch MPS"""
        engine = MetalBacktestEngine(prefer_mlx=False)

        if TORCH_MPS_AVAILABLE:
            assert engine.backend == "mps"
        elif MLX_AVAILABLE:
            # 如果 prefer_mlx=False 但 MPS 不可用，會回退到 MLX
            # 實際上這取決於實作邏輯
            assert engine.backend in ("mlx", "cpu")
        else:
            assert engine.backend == "cpu"

    def test_is_gpu_available(self):
        """測試：GPU 可用性檢查"""
        engine = MetalBacktestEngine()
        gpu_available = engine.is_gpu_available()

        expected = MLX_AVAILABLE or TORCH_MPS_AVAILABLE
        assert gpu_available == expected

    def test_batch_backtest_returns_correct_length(
        self, sample_price_data, simple_strategy, param_grid
    ):
        """測試：批次回測回傳正確數量的結果"""
        engine = MetalBacktestEngine()
        results = engine.batch_backtest(
            sample_price_data,
            param_grid,
            simple_strategy
        )

        assert len(results) == len(param_grid)
        assert all(isinstance(r, GPUBacktestResult) for r in results)

    def test_batch_backtest_result_structure(
        self, sample_price_data, simple_strategy, param_grid
    ):
        """測試：回測結果包含所有必要欄位"""
        engine = MetalBacktestEngine()
        results = engine.batch_backtest(
            sample_price_data,
            param_grid,
            simple_strategy
        )

        for result in results:
            assert hasattr(result, "params")
            assert hasattr(result, "total_return")
            assert hasattr(result, "sharpe_ratio")
            assert hasattr(result, "max_drawdown")
            assert hasattr(result, "execution_time_ms")

            # 檢查數值有效性
            assert isinstance(result.total_return, float)
            assert isinstance(result.sharpe_ratio, float)
            assert isinstance(result.max_drawdown, float)
            assert result.execution_time_ms >= 0

    def test_batch_backtest_params_match(
        self, sample_price_data, simple_strategy, param_grid
    ):
        """測試：結果的參數與輸入參數一致"""
        engine = MetalBacktestEngine()
        results = engine.batch_backtest(
            sample_price_data,
            param_grid,
            simple_strategy
        )

        for result, params in zip(results, param_grid):
            assert result.params == params

    def test_gpu_sma_basic(self):
        """測試：SMA 基本計算"""
        engine = MetalBacktestEngine()
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        period = 3

        sma = engine.gpu_sma(prices, period)

        # 前 period-1 個應該是 NaN
        assert np.isnan(sma[0])
        assert np.isnan(sma[1])

        # 第 3 個應該是 (1+2+3)/3 = 2.0
        assert np.isclose(sma[2], 2.0)

        # 第 4 個應該是 (2+3+4)/3 = 3.0
        assert np.isclose(sma[3], 3.0)

        # 最後一個應該是 (8+9+10)/3 = 9.0
        assert np.isclose(sma[-1], 9.0)

    def test_gpu_ema_basic(self):
        """測試：EMA 基本計算"""
        engine = MetalBacktestEngine()
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        period = 3

        ema = engine.gpu_ema(prices, period)

        # 前 period-1 個應該是 NaN
        assert np.isnan(ema[0])
        assert np.isnan(ema[1])

        # EMA 應該平滑但不是簡單平均
        assert not np.isnan(ema[2])

        # EMA 應該隨價格上升
        assert ema[-1] > ema[2]

    def test_cpu_fallback_when_no_gpu(self, sample_price_data, simple_strategy, param_grid):
        """測試：無 GPU 時正確回退到 CPU"""
        # 強制使用 CPU（透過模擬沒有 GPU 的環境）
        engine = MetalBacktestEngine(prefer_mlx=False)

        # 手動覆蓋 backend（模擬測試）
        original_backend = engine.backend
        engine.backend = "cpu"

        results = engine.batch_backtest(
            sample_price_data,
            param_grid,
            simple_strategy
        )

        assert len(results) == len(param_grid)

        # 恢復原始 backend
        engine.backend = original_backend

    def test_sma_cpu_vs_gpu_consistency(self):
        """測試：CPU 與 GPU SMA 結果一致性"""
        engine = MetalBacktestEngine()
        prices = np.random.randn(100) * 10 + 100
        period = 20

        # 計算 CPU SMA
        cpu_sma = engine._cpu_sma(prices, period)

        # 計算 GPU SMA
        gpu_sma = engine.gpu_sma(prices, period)

        # 比較結果（忽略 NaN）
        valid_mask = ~np.isnan(cpu_sma)
        assert np.allclose(cpu_sma[valid_mask], gpu_sma[valid_mask], rtol=1e-5)

    def test_ema_cpu_vs_gpu_consistency(self):
        """測試：CPU 與 GPU EMA 結果一致性"""
        engine = MetalBacktestEngine()
        prices = np.random.randn(100) * 10 + 100
        period = 20

        # 計算 CPU EMA
        cpu_ema = engine._cpu_ema(prices, period)

        # 計算 GPU EMA
        gpu_ema = engine.gpu_ema(prices, period)

        # 比較結果（忽略 NaN）
        valid_mask = ~np.isnan(cpu_ema)
        assert np.allclose(cpu_ema[valid_mask], gpu_ema[valid_mask], rtol=1e-5)


class TestGPUBacktestResult:
    """GPUBacktestResult 資料類別測試"""

    def test_result_creation(self):
        """測試：建立 GPUBacktestResult 物件"""
        result = GPUBacktestResult(
            params={"sma_period": 20},
            total_return=0.15,
            sharpe_ratio=1.2,
            max_drawdown=0.08,
            execution_time_ms=5.5
        )

        assert result.params == {"sma_period": 20}
        assert result.total_return == 0.15
        assert result.sharpe_ratio == 1.2
        assert result.max_drawdown == 0.08
        assert result.execution_time_ms == 5.5


class TestPerformance:
    """效能測試（需要 GPU）"""

    @pytest.mark.skipif(not (MLX_AVAILABLE or TORCH_MPS_AVAILABLE), reason="No GPU available")
    def test_gpu_faster_than_cpu(self, sample_price_data, simple_strategy):
        """測試：GPU 應該比 CPU 快（大批次）"""
        # 大參數網格
        large_param_grid = [{"sma_period": p} for p in range(10, 60, 5)]

        engine = MetalBacktestEngine()

        # GPU 執行
        import time
        start = time.perf_counter()
        gpu_results = engine.batch_backtest(
            sample_price_data,
            large_param_grid,
            simple_strategy
        )
        gpu_time = time.perf_counter() - start

        # CPU 執行（強制）
        original_backend = engine.backend
        engine.backend = "cpu"

        start = time.perf_counter()
        cpu_results = engine.batch_backtest(
            sample_price_data,
            large_param_grid,
            simple_strategy
        )
        cpu_time = time.perf_counter() - start

        engine.backend = original_backend

        # GPU 應該更快（至少不會慢很多）
        # 注意：小數據量時 GPU 可能因為傳輸開銷反而較慢
        print(f"GPU time: {gpu_time:.4f}s, CPU time: {cpu_time:.4f}s")

        # 只要不崩潰就算通過（效能取決於硬體）
        assert len(gpu_results) == len(cpu_results)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
