"""
GPU 批量參數優化器

整合 Optuna 貝葉斯優化與 GPU 批量回測：
1. Optuna 生成參數候選批次
2. GPU 批量執行回測
3. 更新 Optuna study
4. 重複直到達到試驗次數

支援 MLX（Apple Silicon）和 PyTorch MPS 後端。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, TYPE_CHECKING
import numpy as np
import logging
import time

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# Optuna 導入（為型別檢查器提供 stub）
OPTUNA_AVAILABLE = False
optuna: Any = None
TPESampler: Any = None

try:
    import optuna as _optuna
    from optuna.samplers import TPESampler as _TPESampler
    optuna = _optuna
    TPESampler = _TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    logger.warning("Optuna 未安裝，GPU 批量優化功能不可用")

# Metal Engine 導入（延遲導入避免測試時出錯）
try:
    from ..backtester.metal_engine import MetalBacktestEngine
except ImportError:
    # 測試時使用絕對導入
    try:
        from src.backtester.metal_engine import MetalBacktestEngine
    except ImportError:
        MetalBacktestEngine = None  # type: ignore


@dataclass
class GPUBatchResult:
    """GPU 批量回測結果"""
    params: Dict[str, Any]
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    execution_time_ms: float


@dataclass
class GPUOptimizationResult:
    """GPU 優化結果"""
    best_params: Dict[str, Any]
    best_sharpe: float
    best_return: float

    n_trials: int
    n_batches: int
    total_time_seconds: float

    all_results: List[GPUBatchResult] = field(default_factory=list)

    # Optuna study（如果使用）
    study: Optional[Any] = None

    def summary(self) -> str:
        """生成摘要報告"""
        return f"""
GPU 優化結果摘要
{'='*60}
最佳 Sharpe Ratio: {self.best_sharpe:.4f}
最佳總報酬: {self.best_return:.2%}
最佳參數: {self.best_params}

統計資訊
{'-'*60}
總試驗次數: {self.n_trials}
批次數: {self.n_batches}
總優化時間: {self.total_time_seconds:.2f} 秒
平均每批次: {self.total_time_seconds / self.n_batches:.2f} 秒
"""


class GPUBatchOptimizer:
    """GPU 批量參數優化器

    將 Optuna 貝葉斯優化與 GPU 批量回測結合：
    1. Optuna 生成參數候選批次
    2. GPU 批量執行回測
    3. 更新 Optuna study
    4. 重複直到達到試驗次數
    """

    def __init__(
        self,
        prefer_mlx: bool = True,
        fallback_to_cpu: bool = True,
        verbose: bool = False
    ):
        """初始化 GPU 批量優化器

        Args:
            prefer_mlx: 優先使用 MLX（Apple Silicon）
            fallback_to_cpu: GPU 不可用時是否降級到 CPU
            verbose: 是否顯示詳細資訊
        """
        self.prefer_mlx = prefer_mlx
        self.fallback_to_cpu = fallback_to_cpu
        self.verbose = verbose

        # 初始化 Metal Engine（先設為 None）
        self._metal_engine = None

        # 檢測可用後端（會設置 _metal_engine）
        self._backend = self._detect_backend()

        if self.verbose:
            logger.info(f"GPUBatchOptimizer initialized with backend: {self._backend}")

    def _detect_backend(self) -> str:
        """檢測可用的計算後端"""
        # 檢查 Metal Engine 是否可用
        if MetalBacktestEngine is None:
            if self.fallback_to_cpu:
                logger.warning("MetalBacktestEngine 不可用，降級到 CPU")
                return 'cpu'
            else:
                raise RuntimeError("MetalBacktestEngine 不可用且未啟用 CPU fallback")

        # 嘗試初始化 Metal Engine
        try:
            engine = MetalBacktestEngine(prefer_mlx=self.prefer_mlx)

            if engine.is_gpu_available():
                # GPU 可用，保存 engine 並返回 backend
                self._metal_engine = engine
                return engine.backend  # 'mlx' 或 'mps'
            elif self.fallback_to_cpu:
                # GPU 不可用，降級到 CPU（不保存 engine）
                logger.warning("GPU 不可用，降級到 CPU")
                self._metal_engine = None
                return 'cpu'
            else:
                raise RuntimeError("GPU 不可用且未啟用 CPU fallback")
        except Exception as e:
            if self.fallback_to_cpu:
                logger.warning(f"GPU 初始化失敗，降級到 CPU: {e}")
                self._metal_engine = None
                return 'cpu'
            else:
                raise RuntimeError(f"GPU 初始化失敗: {e}")

    def is_gpu_available(self) -> bool:
        """GPU 是否可用"""
        return self._backend in ('mlx', 'mps')

    def batch_optimize(
        self,
        strategy_fn: Callable[[np.ndarray, Dict], np.ndarray],
        price_data: np.ndarray,
        param_space: Dict[str, Dict],
        n_trials: int = 100,
        batch_size: int = 50,
        metric: str = 'sharpe_ratio',
        direction: str = 'maximize',
        seed: Optional[int] = None
    ) -> GPUOptimizationResult:
        """執行 GPU 批量優化

        Args:
            strategy_fn: 策略函數 (data, params) -> signals
            price_data: OHLCV 資料 (N, 5) float32
            param_space: 參數空間定義
            n_trials: 總試驗次數
            batch_size: 每批次試驗數
            metric: 優化目標 ('sharpe_ratio', 'total_return', 'calmar')
            direction: 優化方向 ('maximize', 'minimize')
            seed: 隨機種子

        Returns:
            GPUOptimizationResult
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("需要安裝 Optuna: pip install optuna")

        start_time = time.time()

        # 建立 Optuna study
        study = self._create_study(param_space, direction, seed)

        all_results = []
        n_batches = (n_trials + batch_size - 1) // batch_size  # 向上取整

        if self.verbose:
            logger.info(f"開始 GPU 批量優化：{n_trials} 試驗，{n_batches} 批次")

        for batch_idx in range(n_batches):
            # 計算本批次的試驗數量
            remaining_trials = n_trials - batch_idx * batch_size
            current_batch_size = min(batch_size, remaining_trials)

            if self.verbose:
                logger.info(f"批次 {batch_idx + 1}/{n_batches}：{current_batch_size} 試驗")

            # 採樣參數批次
            param_batch = self._sample_param_batch(study, param_space, current_batch_size)

            # 執行批量回測
            if self.is_gpu_available():
                batch_results = self._execute_batch_gpu(strategy_fn, price_data, param_batch)
            else:
                batch_results = self._execute_batch_cpu(strategy_fn, price_data, param_batch)

            # 更新 Optuna study
            self._update_study(study, batch_results, param_batch, metric)

            all_results.extend(batch_results)

        total_time = time.time() - start_time

        # 找出最佳結果
        if direction == 'maximize':
            best_result = max(all_results, key=lambda r: getattr(r, metric))
        else:
            best_result = min(all_results, key=lambda r: getattr(r, metric))

        return GPUOptimizationResult(
            best_params=best_result.params,
            best_sharpe=best_result.sharpe_ratio,
            best_return=best_result.total_return,
            n_trials=n_trials,
            n_batches=n_batches,
            total_time_seconds=total_time,
            all_results=all_results,
            study=study
        )

    def _create_study(
        self,
        param_space: Dict,
        direction: str,
        seed: Optional[int]
    ) -> Any:
        """建立 Optuna study"""
        if not OPTUNA_AVAILABLE:
            raise ImportError("需要安裝 Optuna: pip install optuna")

        # 這裡 optuna 和 TPESampler 保證已導入（由 OPTUNA_AVAILABLE 保護）
        sampler = TPESampler(seed=seed)

        study = optuna.create_study(
            direction=direction,
            sampler=sampler
        )

        return study

    def _sample_param_batch(
        self,
        study,
        param_space: Dict,
        batch_size: int
    ) -> List[Dict[str, Any]]:
        """從 Optuna 採樣參數批次"""
        param_batch = []

        for _ in range(batch_size):
            # 要求 Optuna 採樣但不執行
            trial = study.ask()

            # 從參數空間採樣
            params = {}
            for param_name, config in param_space.items():
                param_type = config['type']

                if param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        low=config['low'],
                        high=config['high'],
                        step=config.get('step', 1)
                    )
                elif param_type == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        low=config['low'],
                        high=config['high'],
                        step=config.get('step')
                    )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        choices=config['choices']
                    )
                else:
                    raise ValueError(f"不支援的參數類型: {param_type}")

            param_batch.append({
                'trial': trial,
                'params': params
            })

        return param_batch

    def _execute_batch_gpu(
        self,
        strategy_fn: Callable,
        price_data: np.ndarray,
        param_batch: List[Dict]
    ) -> List[GPUBatchResult]:
        """GPU 批量執行回測"""
        if self._metal_engine is None:
            raise RuntimeError("Metal Engine 未初始化")

        # 準備參數網格
        param_grid = [item['params'] for item in param_batch]

        # 執行批量回測
        gpu_results = self._metal_engine.batch_backtest(
            price_data=price_data,
            param_grid=param_grid,
            strategy_fn=strategy_fn
        )

        # 轉換為 GPUBatchResult（加入 win_rate）
        batch_results = []
        for gpu_result in gpu_results:
            # 簡單的勝率計算（假設 GPU 回測包含信號）
            # 實際應該由 Metal Engine 提供
            win_rate = 0.5  # placeholder

            batch_results.append(GPUBatchResult(
                params=gpu_result.params,
                total_return=gpu_result.total_return,
                sharpe_ratio=gpu_result.sharpe_ratio,
                max_drawdown=gpu_result.max_drawdown,
                win_rate=win_rate,
                execution_time_ms=gpu_result.execution_time_ms
            ))

        return batch_results

    def _execute_batch_cpu(
        self,
        strategy_fn: Callable,
        price_data: np.ndarray,
        param_batch: List[Dict]
    ) -> List[GPUBatchResult]:
        """CPU 批量執行回測（fallback）"""
        batch_results = []

        for item in param_batch:
            params = item['params']

            start_time = time.perf_counter()

            # 執行策略
            signals = strategy_fn(price_data, params)

            # 計算指標（簡化版本）
            returns = self._calculate_returns(price_data, signals)
            total_return = float(np.sum(returns))
            sharpe_ratio = self._calculate_sharpe(returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            win_rate = self._calculate_win_rate(returns)

            execution_time = (time.perf_counter() - start_time) * 1000

            batch_results.append(GPUBatchResult(
                params=params,
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                execution_time_ms=execution_time
            ))

        return batch_results

    def _update_study(
        self,
        study,
        results: List[GPUBatchResult],
        param_batch: List[Dict],
        metric: str
    ):
        """更新 Optuna study"""
        for i, result in enumerate(results):
            trial = param_batch[i]['trial']
            metric_value = getattr(result, metric)

            # 告訴 Optuna 這個 trial 的結果
            study.tell(trial, metric_value)

    # ========== CPU 計算函數 ==========

    def _calculate_returns(self, price_data: np.ndarray, signals: np.ndarray) -> np.ndarray:
        """計算報酬"""
        price_changes = np.diff(price_data[:, 0])  # 假設第一列是收盤價
        returns = price_changes * signals[:-1]
        return returns

    def _calculate_sharpe(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """計算 Sharpe Ratio"""
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return < 1e-8:
            return 0.0

        sharpe = mean_return / std_return * np.sqrt(periods_per_year)
        return float(sharpe)

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """計算最大回撤"""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_dd = np.max(drawdown)
        return float(max_dd)

    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """計算勝率"""
        winning_trades = np.sum(returns > 0)
        total_trades = len(returns[returns != 0])

        if total_trades == 0:
            return 0.0

        return float(winning_trades / total_trades)


# 便利函數
def gpu_optimize_strategy(
    strategy,
    data: np.ndarray,
    n_trials: int = 100,
    batch_size: int = 50,
    metric: str = 'sharpe_ratio'
) -> GPUOptimizationResult:
    """快速 GPU 優化策略

    Args:
        strategy: 策略實例（需有 generate_signals 和 param_space）
        data: OHLCV 資料
        n_trials: 試驗次數
        batch_size: 批次大小
        metric: 優化目標

    Returns:
        GPUOptimizationResult
    """
    # 檢查策略是否有必要屬性
    if not hasattr(strategy, 'generate_signals'):
        raise ValueError("策略必須有 generate_signals 方法")

    if not hasattr(strategy, 'param_space'):
        raise ValueError("策略必須定義 param_space")

    # 包裝策略函數
    def strategy_fn(price_data: np.ndarray, params: Dict) -> np.ndarray:
        # 更新策略參數
        for key, value in params.items():
            setattr(strategy, key, value)

        # 生成信號
        import pandas as pd
        df = pd.DataFrame(price_data)
        df.columns = pd.Index(['open', 'high', 'low', 'close', 'volume'])  # type: ignore[assignment]
        signals = strategy.generate_signals(df)
        return np.asarray(signals)

    # 建立優化器
    optimizer = GPUBatchOptimizer(verbose=True)

    # 執行優化
    result = optimizer.batch_optimize(
        strategy_fn=strategy_fn,
        price_data=data,
        param_space=strategy.param_space,
        n_trials=n_trials,
        batch_size=batch_size,
        metric=metric
    )

    return result


if __name__ == "__main__":
    """測試 GPU 批量優化器"""
    import sys
    from pathlib import Path

    # 添加專案根目錄到 Python path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    # 測試後端檢測
    print("=" * 60)
    print("測試 GPU 批量優化器")
    print("=" * 60)

    optimizer = GPUBatchOptimizer(verbose=True)
    print(f"Backend: {optimizer._backend}")
    print(f"GPU available: {optimizer.is_gpu_available()}")

    # 測試參數採樣
    if OPTUNA_AVAILABLE:
        param_space = {
            'fast_period': {'type': 'int', 'low': 5, 'high': 20},
            'slow_period': {'type': 'int', 'low': 20, 'high': 50},
            'threshold': {'type': 'float', 'low': 0.01, 'high': 0.1, 'step': 0.01}
        }

        study = optimizer._create_study(param_space, 'maximize', seed=42)
        param_batch = optimizer._sample_param_batch(study, param_space, batch_size=3)

        print("\n參數採樣測試：")
        for i, item in enumerate(param_batch):
            print(f"  Trial {i}: {item['params']}")

    print("\n✅ GPU 批量優化器測試完成")
