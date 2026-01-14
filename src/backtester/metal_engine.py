"""
Metal GPU 加速回測引擎

支援 Apple Silicon M4 Max 的 GPU 加速。
使用 MLX（優先）或 PyTorch MPS（備選）。
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import logging
import time

from src.types.enums import BackendType

logger = logging.getLogger(__name__)

# MLX 是可選依賴
MLX_AVAILABLE = False
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
    logger.info("MLX GPU backend available")
except ImportError:
    mx = None  # type: ignore[assignment]
    logger.debug("MLX not available, will fallback to PyTorch MPS or CPU")

# PyTorch MPS 是可選依賴
TORCH_MPS_AVAILABLE = False
try:
    import torch
    TORCH_MPS_AVAILABLE = torch.backends.mps.is_available()
    if TORCH_MPS_AVAILABLE:
        logger.info("PyTorch MPS backend available")
except ImportError:
    torch = None  # type: ignore[assignment]
    logger.debug("PyTorch not available, will fallback to CPU")


@dataclass
class GPUBacktestResult:
    """GPU 回測結果"""
    params: Dict[str, Any]
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    execution_time_ms: float


class MetalBacktestEngine:
    """Metal GPU 加速回測引擎"""

    def __init__(self, prefer_mlx: bool = True):
        """
        Args:
            prefer_mlx: 優先使用 MLX（否則使用 PyTorch MPS）
        """
        self.prefer_mlx = prefer_mlx
        self.backend: BackendType = self._select_backend()
        logger.info(f"MetalBacktestEngine initialized with backend: {self.backend.value}")

    def _select_backend(self) -> BackendType:
        """選擇最佳可用後端"""
        if self.prefer_mlx and MLX_AVAILABLE:
            return BackendType.MLX
        elif TORCH_MPS_AVAILABLE:
            return BackendType.MPS
        else:
            logger.warning("No GPU backend available, using CPU")
            return BackendType.CPU

    def is_gpu_available(self) -> bool:
        """檢查 GPU 是否可用"""
        return self.backend in (BackendType.MLX, BackendType.MPS)

    def batch_backtest(
        self,
        price_data: np.ndarray,
        param_grid: List[Dict[str, Any]],
        strategy_fn: Callable
    ) -> List[GPUBacktestResult]:
        """
        批次回測（GPU 並行）

        Args:
            price_data: 價格資料 (T, N) 其中 T=時間步, N=特徵數
            param_grid: 參數組合列表
            strategy_fn: 策略函數 fn(prices, **params) -> signals

        Returns:
            回測結果列表
        """
        start_time = time.perf_counter()

        if self.backend == BackendType.MLX:
            results_array = self._mlx_backtest(price_data, param_grid, strategy_fn)
        elif self.backend == BackendType.MPS:
            results_array = self._torch_backtest(price_data, param_grid, strategy_fn)
        else:
            results_array = self._cpu_backtest(price_data, param_grid, strategy_fn)

        execution_time = (time.perf_counter() - start_time) * 1000  # ms

        # 將結果轉換為 GPUBacktestResult 物件
        results = []
        for i, params in enumerate(param_grid):
            results.append(GPUBacktestResult(
                params=params,
                total_return=float(results_array[i, 0]),
                sharpe_ratio=float(results_array[i, 1]),
                max_drawdown=float(results_array[i, 2]),
                execution_time_ms=execution_time / len(param_grid)
            ))

        logger.info(
            f"Batch backtest completed: {len(param_grid)} combinations "
            f"in {execution_time:.2f}ms ({self.backend.value})"
        )

        return results

    def _mlx_backtest(
        self,
        prices: np.ndarray,
        param_grid: List[Dict[str, Any]],
        strategy_fn: Callable
    ) -> np.ndarray:
        """
        MLX 回測實作

        Returns:
            results (N, 3): [total_return, sharpe_ratio, max_drawdown]
        """
        if not MLX_AVAILABLE or mx is None:
            raise RuntimeError("MLX not available")

        # 類型縮窄（讓 Pyright 理解 mx 不是 None）
        _mx = mx  # type: ignore[assignment]

        # 將 numpy 轉為 MLX array
        prices_mx = _mx.array(prices)

        results = []
        for params in param_grid:
            # 執行策略
            signals = strategy_fn(prices, **params)
            signals_mx = _mx.array(signals)

            # 計算報酬
            returns = self._calculate_returns_mlx(prices_mx, signals_mx)

            # 計算指標
            total_return = float(_mx.sum(returns))
            sharpe = float(self._calculate_sharpe_mlx(returns))
            max_dd = float(self._calculate_max_drawdown_mlx(returns))

            results.append([total_return, sharpe, max_dd])

        return np.array(results)

    def _torch_backtest(
        self,
        prices: np.ndarray,
        param_grid: List[Dict[str, Any]],
        strategy_fn: Callable
    ) -> np.ndarray:
        """
        PyTorch MPS 回測實作

        Returns:
            results (N, 3): [total_return, sharpe_ratio, max_drawdown]
        """
        if not TORCH_MPS_AVAILABLE or torch is None:
            raise RuntimeError("PyTorch MPS not available")

        # 類型縮窄（讓 Pyright 理解 torch 不是 None）
        _torch = torch  # type: ignore[assignment]

        # 將 numpy 轉為 torch tensor 並放到 MPS
        # MPS 不支援 float64，必須使用 float32
        device = _torch.device("mps")
        prices_tensor = _torch.from_numpy(prices).float().to(device)

        results = []
        for params in param_grid:
            # 執行策略
            signals = strategy_fn(prices, **params)
            signals_tensor = _torch.from_numpy(signals).float().to(device)

            # 計算報酬
            returns = self._calculate_returns_torch(prices_tensor, signals_tensor)

            # 計算指標
            total_return = float(returns.sum().cpu())
            sharpe = float(self._calculate_sharpe_torch(returns).cpu())
            max_dd = float(self._calculate_max_drawdown_torch(returns).cpu())

            results.append([total_return, sharpe, max_dd])

        return np.array(results)

    def _cpu_backtest(
        self,
        prices: np.ndarray,
        param_grid: List[Dict[str, Any]],
        strategy_fn: Callable
    ) -> np.ndarray:
        """CPU 回測實作（回退方案）"""
        results = []
        for params in param_grid:
            signals = strategy_fn(prices, **params)
            returns = self._calculate_returns_cpu(prices, signals)

            total_return = float(np.sum(returns))
            sharpe = float(self._calculate_sharpe_cpu(returns))
            max_dd = float(self._calculate_max_drawdown_cpu(returns))

            results.append([total_return, sharpe, max_dd])

        return np.array(results)

    # ========== MLX 計算函數 ==========
    # 注意：這些函數只在 MLX 可用時被調用，由 _mlx_backtest 保證

    def _calculate_returns_mlx(self, prices: Any, signals: Any) -> Any:
        """計算報酬（MLX）

        使用 t 時刻的信號預測 t+1 的漲跌，報酬 = 信號 × 價格變化率。
        正確處理信號長度可能與價格長度不同的情況。
        """
        _mx = mx  # type: ignore[assignment]
        # 價格變化：diff 後長度 N-1
        price_changes = _mx.diff(prices[:, 0])  # 假設第一列是收盤價

        # 信號對齊：信號長度可能是 N（與價格相同）或 N-1（已經 shift 過）
        # 我們使用 min_len 來確保正確對齊
        signal_len = signals.shape[0]
        change_len = price_changes.shape[0]

        # 如果信號長度等於價格長度，使用 signals[:-1]
        # 如果信號長度等於價格變化長度，直接使用
        if signal_len > change_len:
            aligned_signals = signals[:change_len]
        else:
            aligned_signals = signals

        min_len = min(change_len, aligned_signals.shape[0])
        returns = price_changes[:min_len] * aligned_signals[:min_len]
        return returns

    def _calculate_sharpe_mlx(self, returns: Any, periods_per_year: int = 252) -> Any:
        """計算 Sharpe Ratio（MLX）"""
        _mx = mx  # type: ignore[assignment]
        mean_return = _mx.mean(returns)
        std_return = _mx.std(returns)

        # 避免除以零
        if float(std_return) < 1e-8:
            return _mx.array(0.0)

        sharpe = mean_return / std_return * _mx.sqrt(_mx.array(periods_per_year))
        return sharpe

    def _calculate_max_drawdown_mlx(self, returns: Any) -> Any:
        """計算最大回撤（MLX）"""
        _mx = mx  # type: ignore[assignment]
        cumulative = _mx.cumsum(returns)
        running_max = _mx.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_dd = _mx.max(drawdown)
        return max_dd

    # ========== PyTorch 計算函數 ==========
    # 注意：這些函數只在 PyTorch MPS 可用時被調用，由 _torch_backtest 保證

    def _calculate_returns_torch(self, prices: Any, signals: Any) -> Any:
        """計算報酬（PyTorch）

        使用 t 時刻的信號預測 t+1 的漲跌，報酬 = 信號 × 價格變化率。
        正確處理信號長度可能與價格長度不同的情況。
        """
        _torch = torch  # type: ignore[assignment]
        # 價格變化：diff 後長度 N-1
        price_changes = _torch.diff(prices[:, 0])

        # 信號對齊：信號長度可能是 N（與價格相同）或 N-1（已經 shift 過）
        signal_len = signals.shape[0]
        change_len = price_changes.shape[0]

        # 如果信號長度等於價格長度，使用 signals[:-1]
        # 如果信號長度等於價格變化長度，直接使用
        if signal_len > change_len:
            aligned_signals = signals[:change_len]
        else:
            aligned_signals = signals

        min_len = min(change_len, aligned_signals.shape[0])
        returns = price_changes[:min_len] * aligned_signals[:min_len]
        return returns

    def _calculate_sharpe_torch(self, returns: Any, periods_per_year: int = 252) -> Any:
        """計算 Sharpe Ratio（PyTorch）"""
        _torch = torch  # type: ignore[assignment]
        mean_return = returns.mean()
        std_return = returns.std()

        if float(std_return) < 1e-8:
            return _torch.tensor(0.0, device=returns.device)

        sharpe = mean_return / std_return * _torch.sqrt(_torch.tensor(periods_per_year, device=returns.device))
        return sharpe

    def _calculate_max_drawdown_torch(self, returns: Any) -> Any:
        """計算最大回撤（PyTorch）"""
        cumulative = torch.cumsum(returns, dim=0)
        running_max = torch.cummax(cumulative, dim=0)[0]
        drawdown = running_max - cumulative
        max_dd = torch.max(drawdown)
        return max_dd

    # ========== CPU 計算函數 ==========

    def _calculate_returns_cpu(self, prices: np.ndarray, signals: np.ndarray) -> np.ndarray:
        """計算報酬（CPU）"""
        price_changes = np.diff(prices[:, 0])
        returns = price_changes * signals[:-1]
        return returns

    def _calculate_sharpe_cpu(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """計算 Sharpe Ratio（CPU）"""
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return < 1e-8:
            return 0.0

        sharpe = mean_return / std_return * np.sqrt(periods_per_year)
        return sharpe

    def _calculate_max_drawdown_cpu(self, returns: np.ndarray) -> float:
        """計算最大回撤（CPU）"""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_dd = np.max(drawdown)
        return max_dd

    # ========== 技術指標（GPU 加速）==========

    def gpu_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        GPU 計算 SMA（簡單移動平均）

        Args:
            prices: 價格陣列 (T,)
            period: 週期

        Returns:
            SMA 陣列 (T,)
        """
        if self.backend == BackendType.MLX:
            return self._mlx_sma(prices, period)
        elif self.backend == BackendType.MPS:
            return self._torch_sma(prices, period)
        else:
            return self._cpu_sma(prices, period)

    def gpu_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        GPU 計算 EMA（指數移動平均）

        Args:
            prices: 價格陣列 (T,)
            period: 週期

        Returns:
            EMA 陣列 (T,)
        """
        if self.backend == BackendType.MLX:
            return self._mlx_ema(prices, period)
        elif self.backend == BackendType.MPS:
            return self._torch_ema(prices, period)
        else:
            return self._cpu_ema(prices, period)

    def _mlx_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """MLX SMA 實作"""
        if not MLX_AVAILABLE:
            return self._cpu_sma(prices, period)

        prices_mx = mx.array(prices)

        # 使用 convolution 計算 SMA
        kernel = mx.ones(period) / period
        # 注意：MLX 的 conv 需要 (batch, in_channels, length) 格式
        prices_reshaped = mx.expand_dims(mx.expand_dims(prices_mx, 0), 0)
        kernel_reshaped = mx.expand_dims(mx.expand_dims(kernel, 0), 0)

        # 使用手動捲積代替（MLX API 可能不同）
        result = []
        for i in range(len(prices)):
            if i < period - 1:
                result.append(float('nan'))
            else:
                result.append(float(mx.mean(prices_mx[i - period + 1:i + 1])))

        return np.array(result)

    def _torch_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """PyTorch MPS SMA 實作"""
        if not TORCH_MPS_AVAILABLE:
            return self._cpu_sma(prices, period)

        device = torch.device("mps")
        prices_tensor = torch.from_numpy(prices).float().to(device)

        # 使用 unfold 實作移動窗口
        result = torch.full((len(prices),), float('nan'), device=device)

        for i in range(period - 1, len(prices)):
            result[i] = prices_tensor[i - period + 1:i + 1].mean()

        return result.cpu().numpy()

    def _cpu_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """CPU SMA 實作"""
        result = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            result[i] = np.mean(prices[i - period + 1:i + 1])
        return result

    def _mlx_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """MLX EMA 實作"""
        if not MLX_AVAILABLE:
            return self._cpu_ema(prices, period)

        prices_mx = mx.array(prices)
        alpha = 2.0 / (period + 1)

        result = [float('nan')] * (period - 1)
        ema = float(mx.mean(prices_mx[:period]))  # 初始 EMA = SMA
        result.append(ema)

        for i in range(period, len(prices)):
            ema = alpha * float(prices_mx[i]) + (1 - alpha) * ema
            result.append(ema)

        return np.array(result)

    def _torch_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """PyTorch MPS EMA 實作"""
        if not TORCH_MPS_AVAILABLE:
            return self._cpu_ema(prices, period)

        device = torch.device("mps")
        prices_tensor = torch.from_numpy(prices).float().to(device)
        alpha = 2.0 / (period + 1)

        result = torch.full((len(prices),), float('nan'), device=device)
        ema = prices_tensor[:period].mean()  # 初始 EMA = SMA
        result[period - 1] = ema

        for i in range(period, len(prices)):
            ema = alpha * prices_tensor[i] + (1 - alpha) * ema
            result[i] = ema

        return result.cpu().numpy()

    def _cpu_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """CPU EMA 實作"""
        alpha = 2.0 / (period + 1)
        result = np.full(len(prices), np.nan)

        ema = np.mean(prices[:period])  # 初始 EMA = SMA
        result[period - 1] = ema

        for i in range(period, len(prices)):
            ema = alpha * prices[i] + (1 - alpha) * ema
            result[i] = ema

        return result
