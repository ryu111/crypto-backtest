"""
HyperLoopController - 高效能並行回測 Loop 主控制器

整合 Phase 1-3 的模組，建立支援三層並行的高效能回測系統：
1. 策略層 (L1): ProcessPoolExecutor 並行執行不同策略
2. 參數層 (L2): GPU 批量優化或 Optuna 並行
3. 資料層 (L3): SharedDataPool 零拷貝共享

特性：
- 零拷貝資料共享（SharedDataPool）
- GPU 批量參數優化（GPUBatchOptimizer）
- 智能調度（ExecutionScheduler）
- 完整實驗記錄（ExperimentRecorder）
- 錯誤恢復與超時處理
"""

import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from datetime import datetime

import pandas as pd
import numpy as np

from ..data.shared_pool import (
    SharedDataPool,
    create_shared_pool,
    attach_to_pool
)
from ..optimizer.gpu_batch import (
    GPUBatchOptimizer,
    GPUOptimizationResult
)
from ..automation.scheduler import (
    ExecutionScheduler,
    BacktestTask,
    TaskType,
    ExecutionPlan,
    ExecutorType
)
from ..automation.orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    LoopSummary
)
from ..learning.recorder import ExperimentRecorder
from ..strategies.registry import StrategyRegistry

logger = logging.getLogger(__name__)


@dataclass
class HyperLoopConfig:
    """高效能 Loop 配置"""

    # CPU 設定
    max_workers: int = 8

    # GPU 設定
    use_gpu: bool = True
    gpu_batch_size: int = 50

    # 資料設定
    symbols: List[str] = field(default_factory=lambda: ['BTCUSDT', 'ETHUSDT'])
    timeframes: List[str] = field(default_factory=lambda: [
        '1m', '3m', '5m', '15m', '30m',      # 短線
        '1h', '2h', '4h', '6h', '8h',        # 中線（8h 對齊資金費率）
        '12h', '1d', '3d', '1w'              # 長線
    ])
    data_dir: str = "data"

    # 優化設定
    n_trials: int = 100
    param_sweep_threshold: int = 100  # 超過此值使用 GPU

    # 驗證設定
    min_sharpe: float = 1.0
    min_stages: int = 3
    max_overfit: float = 0.5

    # 交易設定
    leverage: int = 5
    initial_capital: float = 10000.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0004

    # 執行設定
    timeout_per_iteration: int = 600  # 每次迭代超時時間（秒）
    max_retries: int = 3              # 失敗重試次數


@dataclass
class IterationTask:
    """單次迭代任務定義"""
    task_id: str
    strategy_name: str
    symbol: str
    timeframe: str

    # 參數空間
    param_space: Dict[str, Any]
    param_count: int

    # 執行設定
    priority: int = 0
    retry_count: int = 0


@dataclass
class HyperLoopSummary:
    """HyperLoop 執行摘要"""

    # 基本統計
    total_iterations: int
    successful_iterations: int
    failed_iterations: int
    timeout_iterations: int

    # 執行統計
    total_duration_seconds: float
    avg_iteration_time: float

    # 資源使用
    peak_memory_mb: float
    total_gpu_time: float
    total_cpu_time: float

    # 最佳結果
    best_strategy: Optional[str] = None
    best_params: Optional[Dict] = None
    best_sharpe: float = 0.0

    # 詳細記錄
    iteration_results: List[Dict] = field(default_factory=list)

    def summary_text(self) -> str:
        """生成摘要報告"""
        success_rate = (
            self.successful_iterations / self.total_iterations * 100
            if self.total_iterations > 0 else 0
        )

        lines = [
            "\n" + "="*70,
            "HyperLoop 執行摘要",
            "="*70,
            f"總迭代次數: {self.total_iterations}",
            f"成功: {self.successful_iterations} ({success_rate:.1f}%)",
            f"失敗: {self.failed_iterations}",
            f"超時: {self.timeout_iterations}",
            "",
            "效能統計:",
            "-"*70,
            f"總執行時間: {self.total_duration_seconds/60:.1f} 分鐘",
            f"平均每次迭代: {self.avg_iteration_time:.1f} 秒",
            f"峰值記憶體: {self.peak_memory_mb:.0f} MB",
            f"GPU 時間: {self.total_gpu_time:.1f} 秒",
            f"CPU 時間: {self.total_cpu_time:.1f} 秒",
            "",
            "最佳結果:",
            "-"*70,
            f"策略: {self.best_strategy or 'N/A'}",
            f"Sharpe Ratio: {self.best_sharpe:.2f}",
            f"參數: {self.best_params or {}}",
            "="*70,
        ]

        return "\n".join(lines)


class HyperLoopController:
    """高效能並行回測 Loop 控制器

    三層並行架構：
    1. 策略層 (L1): ProcessPoolExecutor 並行執行不同策略
    2. 參數層 (L2): GPU 批量優化或 Optuna 並行
    3. 資料層 (L3): SharedDataPool 零拷貝共享

    使用範例：
        config = HyperLoopConfig(
            max_workers=8,
            use_gpu=True,
            symbols=['BTCUSDT', 'ETHUSDT'],
            n_trials=100
        )

        controller = HyperLoopController(config)

        # 執行 Loop
        summary = await controller.run_loop(n_iterations=50)
        print(summary.summary_text())

        # 清理資源
        controller.cleanup()
    """

    def __init__(
        self,
        config: Optional[HyperLoopConfig] = None,
        verbose: bool = True
    ):
        """初始化 HyperLoop 控制器

        Args:
            config: 配置（None 則使用預設值）
            verbose: 是否顯示詳細資訊
        """
        self.config = config or HyperLoopConfig()
        self.verbose = verbose

        # 初始化子模組
        self.scheduler = ExecutionScheduler(
            max_cpu_workers=self.config.max_workers,
            gpu_available=self.config.use_gpu,
            verbose=verbose
        )

        self.gpu_optimizer = GPUBatchOptimizer(
            prefer_mlx=True,
            fallback_to_cpu=True,
            verbose=verbose
        ) if self.config.use_gpu else None

        self.recorder = ExperimentRecorder()

        # 共享資料池（稍後初始化）
        self.shared_pool: Optional[SharedDataPool] = None
        # 使用短名稱避免 POSIX 共享記憶體名稱限制（macOS 約 31 字元）
        # 使用 PID + 實例計數器確保唯一性，避免高頻回測時名稱衝突
        import os
        self._instance_counter = getattr(HyperLoopController, '_instance_counter', 0)
        HyperLoopController._instance_counter = self._instance_counter + 1
        self.pool_name = f"hl_{os.getpid() % 10000}_{self._instance_counter}"

        # 統計資訊
        self.summary = HyperLoopSummary(
            total_iterations=0,
            successful_iterations=0,
            failed_iterations=0,
            timeout_iterations=0,
            total_duration_seconds=0.0,
            avg_iteration_time=0.0,
            peak_memory_mb=0.0,
            total_gpu_time=0.0,
            total_cpu_time=0.0
        )

        # 清理狀態標誌（防止重複清理）
        self._cleaned_up: bool = False

        if self.verbose:
            logger.info(
                f"HyperLoopController initialized: "
                f"workers={self.config.max_workers}, "
                f"GPU={self.config.use_gpu}"
            )

    async def run_loop(self, n_iterations: int) -> HyperLoopSummary:
        """執行高效能回測 Loop

        流程：
        1. 預載所有資料到共享記憶體
        2. 建立策略批次
        3. 並行執行
        4. 清理資源

        Args:
            n_iterations: 迭代次數

        Returns:
            HyperLoopSummary: 執行摘要
        """
        start_time = time.time()

        if self.verbose:
            logger.info(f"\n{'='*70}")
            logger.info(f"開始 HyperLoop (共 {n_iterations} 次迭代)")
            logger.info(f"{'='*70}")

        try:
            # 1. 預載所有資料到共享記憶體
            await self._preload_all_data()

            # 2. 建立策略批次
            batches = self._create_strategy_batches(n_iterations)

            # 3. 並行執行
            for batch_idx, batch in enumerate(batches):
                if self.verbose:
                    logger.info(f"\n執行批次 {batch_idx + 1}/{len(batches)} (共 {len(batch)} 個任務)")

                results = await self._execute_batch_parallel(batch)
                self._process_and_record_results(results)

            # 4. 計算統計
            self.summary.total_duration_seconds = time.time() - start_time
            self.summary.avg_iteration_time = (
                self.summary.total_duration_seconds / n_iterations
                if n_iterations > 0 else 0
            )

            if self.verbose:
                logger.info(self.summary.summary_text())

            return self.summary

        finally:
            # 確保清理資源
            self._cleanup()

    async def _preload_all_data(self):
        """預載所有資料到共享記憶體"""
        if self.verbose:
            logger.info("\n[1/4] 預載資料到共享記憶體...")

        # 先清理舊的共享池（如果存在）
        if self.shared_pool is not None:
            try:
                self.shared_pool.cleanup()
            except Exception:
                pass  # 忽略清理錯誤
            self.shared_pool = None

        # 重置清理標誌，允許新一輪清理
        self._cleaned_up = False

        # 建立共享資料池
        self.shared_pool = create_shared_pool(
            data_dir=self.config.data_dir,
            symbols=self.config.symbols,
            timeframes=self.config.timeframes,
            pool_name=self.pool_name,
            include_funding=True
        )

        # 顯示載入資訊
        if self.verbose:
            total_mb = self.shared_pool.get_total_size_mb()
            logger.info(f"      已載入 {len(self.shared_pool.list_keys())} 個資料集")
            logger.info(f"      共享記憶體大小: {total_mb:.2f} MB")

            # 更新峰值記憶體
            self.summary.peak_memory_mb = max(self.summary.peak_memory_mb, total_mb)

    def _create_strategy_batches(
        self,
        n_iterations: int
    ) -> List[List[IterationTask]]:
        """建立策略批次

        根據 max_workers 分批執行策略

        Args:
            n_iterations: 總迭代次數

        Returns:
            List[List[IterationTask]]: 批次列表
        """
        if self.verbose:
            logger.info("\n[2/4] 建立策略批次...")

        # 取得所有已註冊策略
        all_strategies = StrategyRegistry.list_all()

        if not all_strategies:
            raise ValueError("沒有已註冊的策略")

        # 為每次迭代建立任務
        tasks = []
        for i in range(n_iterations):
            # 輪流選擇策略
            strategy_name = all_strategies[i % len(all_strategies)]

            # 輪流選擇標的和時間框架
            symbol = self.config.symbols[i % len(self.config.symbols)]
            timeframe = self.config.timeframes[i % len(self.config.timeframes)]

            # 取得參數空間
            param_space = StrategyRegistry.get_param_space(strategy_name)
            param_count = self._estimate_param_count(param_space)

            task = IterationTask(
                task_id=f"iter_{i}_{strategy_name}_{symbol}_{timeframe}",
                strategy_name=strategy_name,
                symbol=symbol,
                timeframe=timeframe,
                param_space=param_space,
                param_count=param_count,
                priority=0
            )

            tasks.append(task)

        # 分批（每批最多 max_workers 個任務）
        batches = []
        batch_size = self.config.max_workers

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batches.append(batch)

        if self.verbose:
            logger.info(f"      建立 {len(batches)} 個批次（每批最多 {batch_size} 個任務）")

        return batches

    async def _execute_batch_parallel(
        self,
        batch: List[IterationTask]
    ) -> List[Dict[str, Any]]:
        """並行執行策略批次

        使用 ProcessPoolExecutor 並行執行多個策略

        Args:
            batch: 任務批次

        Returns:
            List[Dict]: 執行結果列表
        """
        if self.verbose:
            logger.info(f"\n[3/4] 並行執行 {len(batch)} 個任務...")

        results = []

        # 使用 ProcessPoolExecutor 並行執行
        with ProcessPoolExecutor(max_workers=len(batch)) as executor:
            # 提交所有任務
            future_to_task = {
                executor.submit(
                    _worker_execute_task,
                    task,
                    self.pool_name,
                    self.config,
                    self.verbose
                ): task
                for task in batch
            }

            # 收集結果
            for future in as_completed(future_to_task):
                task = future_to_task[future]

                try:
                    # 等待結果（設定超時）
                    result = future.result(timeout=self.config.timeout_per_iteration)
                    results.append(result)

                    # 更新統計
                    self.summary.successful_iterations += 1

                    if result.get('used_gpu', False):
                        self.summary.total_gpu_time += result.get('duration', 0)
                    else:
                        self.summary.total_cpu_time += result.get('duration', 0)

                    if self.verbose:
                        logger.info(
                            f"      ✓ {task.task_id}: "
                            f"Sharpe={result.get('sharpe', 0):.2f}, "
                            f"{result.get('duration', 0):.1f}s"
                        )

                except TimeoutError:
                    # 超時
                    self.summary.timeout_iterations += 1
                    logger.warning(f"      ✗ {task.task_id}: 超時")

                    results.append({
                        'task_id': task.task_id,
                        'success': False,
                        'error': 'timeout'
                    })

                except Exception as e:
                    # 執行失敗
                    self.summary.failed_iterations += 1
                    logger.error(f"      ✗ {task.task_id}: {type(e).__name__}: {e}")

                    results.append({
                        'task_id': task.task_id,
                        'success': False,
                        'error': str(e)
                    })

                finally:
                    self.summary.total_iterations += 1

        return results

    def _process_and_record_results(self, results: List[Dict[str, Any]]):
        """處理並記錄結果

        Args:
            results: 執行結果列表
        """
        if self.verbose:
            logger.info(f"\n[4/4] 處理並記錄結果...")

        for result in results:
            if not result.get('success', False):
                continue

            # 記錄到摘要
            self.summary.iteration_results.append(result)

            # 更新最佳結果
            sharpe = result.get('sharpe', 0)
            if sharpe > self.summary.best_sharpe:
                self.summary.best_strategy = result.get('strategy_name')
                self.summary.best_params = result.get('best_params')
                self.summary.best_sharpe = sharpe

            if self.verbose:
                logger.info(
                    f"      已記錄 {result.get('task_id')}: "
                    f"Sharpe={sharpe:.2f}"
                )

    def _cleanup(self):
        """清理所有共享記憶體（只執行一次）"""
        # 防止重複清理
        if self._cleaned_up:
            return

        if self.verbose:
            logger.info("\n清理資源...")

        if self.shared_pool:
            self.shared_pool.cleanup()
            self.shared_pool = None

        # 標記清理完成
        self._cleaned_up = True

        if self.verbose:
            logger.info("✓ 資源已清理")

    def _estimate_param_count(self, param_space: Dict[str, Any]) -> int:
        """估算參數組合數量

        Args:
            param_space: 參數空間定義

        Returns:
            int: 估算的組合數
        """
        total = 1
        for param_name, config in param_space.items():
            param_type = config.get('type')

            if param_type == 'int':
                total *= (config['high'] - config['low'] + 1)
            elif param_type == 'float':
                # 假設 10 個採樣點
                total *= 10
            elif param_type == 'categorical':
                total *= len(config.get('choices', []))

        return total

    def __enter__(self):
        """Context manager 進入"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 退出（自動清理）"""
        self._cleanup()

    def __del__(self):
        """析構函數

        注意：在多進程環境中，不應在 __del__ 中調用 _cleanup()，
        因為 Python GC 可能在不可預期的時間點調用 __del__。
        共享記憶體的清理應該由 run_loop() 的 finally 塊或
        顯式調用 _cleanup() 來處理。
        """
        # 不在 __del__ 中調用 _cleanup()，避免意外清理共享記憶體
        pass


# ===== Worker 函數（在子進程中執行） =====

def _worker_execute_task(
    task: IterationTask,
    pool_name: str,
    config: HyperLoopConfig,
    verbose: bool
) -> Dict[str, Any]:
    """Worker 執行單個任務

    在子進程中執行，使用共享資料池

    Args:
        task: 迭代任務
        pool_name: 共享資料池名稱
        config: HyperLoop 配置
        verbose: 是否顯示詳細資訊

    Returns:
        Dict: 執行結果
    """
    import logging
    logger = logging.getLogger(__name__)

    start_time = time.time()

    pool = None
    try:
        # 1. 附加到共享資料池
        pool = attach_to_pool(pool_name)

        # 2. 取得資料
        data_key = f"{task.symbol}_{task.timeframe}"
        if data_key not in pool.list_keys():
            raise ValueError(f"找不到資料: {data_key}")

        data_df = pool.get_dataframe(data_key)

        # 3. 建立策略實例
        strategy_class = StrategyRegistry.get(task.strategy_name)
        strategy = strategy_class()

        # 4. 執行優化
        # 判斷是否使用 GPU
        use_gpu = (
            config.use_gpu and
            task.param_count >= config.param_sweep_threshold
        )

        if use_gpu:
            # GPU 批量優化
            optimizer = GPUBatchOptimizer(verbose=verbose)

            opt_result = optimizer.batch_optimize(
                strategy_fn=_create_strategy_fn(strategy),
                price_data=data_df.values.astype(np.float32),
                param_space=task.param_space,
                n_trials=config.n_trials,
                batch_size=config.gpu_batch_size,
                metric='sharpe_ratio',
                direction='maximize'
            )

            best_params = opt_result.best_params
            best_sharpe = opt_result.best_sharpe

            # 從 all_results 中找到最佳結果的完整資訊
            best_max_dd = 0.5  # 預設值
            best_win_rate = 0.5  # 預設值
            best_total_return = 0.0  # 預設值
            best_total_trades = 0  # 預設值
            if opt_result.all_results:
                # 找到 sharpe 最高的結果
                best_result = max(opt_result.all_results, key=lambda r: r.sharpe_ratio)
                best_max_dd = best_result.max_drawdown
                best_win_rate = best_result.win_rate
                best_total_return = getattr(best_result, 'total_return', 0.0)
                # 從非零 returns 估算交易數
                best_total_trades = getattr(best_result, 'total_trades', len(opt_result.all_results))

        else:
            # CPU 優化（簡化版本）
            # TODO: 整合 BayesianOptimizer 實現完整優化
            # 目前使用隨機採樣作為 fallback
            best_params = _sample_random_params(task.param_space)

            # 應用參數並生成信號
            for key, value in best_params.items():
                setattr(strategy, key, value)

            signal_result = strategy.generate_signals(data_df)

            # 處理不同返回類型
            if isinstance(signal_result, tuple):
                signals = np.asarray(signal_result[0])  # 取第一個元素（信號）
            else:
                signals = np.asarray(signal_result)

            # 計算 Sharpe（簡化版本）
            prices = np.asarray(data_df['close'].values)
            returns = _calculate_returns(prices, signals)
            best_sharpe = _calculate_sharpe(returns)

            # CPU 暫時使用預設值
            best_max_dd = 0.5
            best_win_rate = 0.5
            best_total_return = 0.0
            best_total_trades = 0

        duration = time.time() - start_time

        # 返回結果
        return {
            'task_id': task.task_id,
            'success': True,
            'strategy_name': task.strategy_name,
            'symbol': task.symbol,
            'timeframe': task.timeframe,
            'best_params': best_params,
            'sharpe': best_sharpe,
            'max_drawdown': best_max_dd,
            'win_rate': best_win_rate,
            'total_return': best_total_return,
            'total_trades': best_total_trades,
            'duration': duration,
            'used_gpu': use_gpu
        }

    except Exception as e:
        logger.error(f"Worker 執行失敗: {type(e).__name__}: {e}")

        return {
            'task_id': task.task_id,
            'success': False,
            'error': str(e),
            'duration': time.time() - start_time
        }

    finally:
        # 確保分離共享記憶體引用（不刪除共享記憶體本身）
        if pool is not None:
            pool.detach()


def _create_strategy_fn(strategy) -> Callable:
    """建立策略函數（用於 GPU 優化器）

    Args:
        strategy: 策略實例

    Returns:
        Callable: 策略函數 (data, params) -> signals
                  或 (data, **kwargs) -> signals

    Note:
        支援兩種調用方式：
        1. strategy_fn(price_data, params_dict)  - GPU batch optimizer
        2. strategy_fn(price_data, **params)     - Metal engine

        策略返回格式處理：
        - 如果返回 tuple (long_entry, short_entry, ...)：轉換為 +1/-1/0 信號
        - 如果返回單一 Series/ndarray：直接使用
    """
    def strategy_fn(price_data: np.ndarray, params: Optional[Dict] = None, **kwargs) -> np.ndarray:
        # 處理兩種調用方式
        # 1. strategy_fn(data, {'rsi_period': 14})  -> params 是字典
        # 2. strategy_fn(data, rsi_period=14)       -> kwargs 包含參數
        if params is None:
            params = kwargs
        elif kwargs:
            # 如果兩者都有，合併（kwargs 優先）
            params = {**params, **kwargs}

        # 更新策略參數
        for key, value in params.items():
            strategy.params[key] = value

        # 生成信號
        df = pd.DataFrame(price_data)
        df.columns = pd.Index(['open', 'high', 'low', 'close', 'volume'])
        result = strategy.generate_signals(df)

        # 處理不同的返回格式
        if isinstance(result, tuple):
            # 策略返回 (long_entry, short_entry, long_exit, short_exit)
            # 轉換為 +1 (做多) / -1 (做空) / 0 (無持倉) 信號
            long_entry = np.asarray(result[0]).astype(float)
            short_entry = np.asarray(result[1]).astype(float)

            # 簡單信號：long_entry 為 +1，short_entry 為 -1
            # 注意：這是簡化版本，不考慮 exit 信號
            signals = long_entry - short_entry
        else:
            signals = np.asarray(result)

        return signals

    return strategy_fn


def _sample_random_params(param_space: Dict[str, Any]) -> Dict[str, Any]:
    """隨機採樣參數

    Args:
        param_space: 參數空間定義

    Returns:
        Dict: 採樣的參數
    """
    import random

    params = {}
    for param_name, config in param_space.items():
        param_type = config.get('type')

        if param_type == 'int':
            params[param_name] = random.randint(config['low'], config['high'])
        elif param_type == 'float':
            params[param_name] = random.uniform(config['low'], config['high'])
        elif param_type == 'categorical':
            params[param_name] = random.choice(config['choices'])

    return params


def _calculate_returns(prices: np.ndarray, signals: np.ndarray) -> np.ndarray:
    """計算報酬

    使用 t 時刻的信號預測 t+1 的漲跌，報酬 = 信號 × 價格變化率

    Args:
        prices: 收盤價陣列 [p_0, p_1, ..., p_n]
        signals: 信號陣列 (1=做多, 0=空倉, -1=做空)

    Returns:
        np.ndarray: 報酬陣列
    """
    if len(prices) < 2:
        return np.array([0.0])

    # 價格變化率: (p_{t+1} - p_t) / p_t
    price_changes = np.diff(prices) / prices[:-1]

    # 信號對齊：使用 t 時刻的信號預測 t+1 的漲跌
    # 因此 signals[:n] 對應 price_changes[:n]
    min_len = min(len(price_changes), len(signals) - 1 if len(signals) > 1 else 0)
    if min_len <= 0:
        return np.array([0.0])

    # 報酬 = 信號 × 價格變化率
    returns = price_changes[:min_len] * signals[:min_len]
    return returns


def _calculate_sharpe(returns: np.ndarray, periods_per_year: int = 365) -> float:
    """計算 Sharpe Ratio

    Args:
        returns: 報酬陣列
        periods_per_year: 年化週期數

    Returns:
        float: Sharpe Ratio
    """
    mean_return = np.mean(returns)
    std_return = np.std(returns)

    if std_return < 1e-8:
        return 0.0

    sharpe = mean_return / std_return * np.sqrt(periods_per_year)
    return float(sharpe)


# ===== 便利函數 =====

def create_hyperloop(
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> HyperLoopController:
    """建立 HyperLoop 控制器

    Args:
        config: 配置字典（轉換為 HyperLoopConfig）
        verbose: 是否顯示詳細資訊

    Returns:
        HyperLoopController: 控制器實例
    """
    if config:
        config_obj = HyperLoopConfig(**config)
    else:
        config_obj = HyperLoopConfig()

    return HyperLoopController(config_obj, verbose)


async def run_hyperloop(
    n_iterations: int,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> HyperLoopSummary:
    """便利函數：執行 HyperLoop

    Args:
        n_iterations: 迭代次數
        config: 配置字典
        verbose: 是否顯示詳細資訊

    Returns:
        HyperLoopSummary: 執行摘要
    """
    controller = create_hyperloop(config, verbose)

    try:
        summary = await controller.run_loop(n_iterations)
        return summary
    finally:
        controller._cleanup()


if __name__ == "__main__":
    """測試 HyperLoopController"""
    import asyncio
    import sys

    # 設定 logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 70)
    print("測試 HyperLoopController")
    print("=" * 70)

    # 建立測試配置
    test_config = HyperLoopConfig(
        max_workers=2,
        use_gpu=False,  # 測試時關閉 GPU
        symbols=['BTCUSDT'],
        timeframes=['1h'],
        n_trials=10,
        data_dir='data'
    )

    # 執行測試
    async def test():
        controller = HyperLoopController(test_config, verbose=True)

        try:
            summary = await controller.run_loop(n_iterations=5)
            print(summary.summary_text())
        finally:
            controller._cleanup()

    # 執行
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(test())

    print("\n✅ HyperLoopController 測試完成")
