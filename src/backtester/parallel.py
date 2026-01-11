"""
多核心並行回測器

支援 Apple M4 Max 16 核心並行處理。
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import cpu_count
from typing import Any, Callable, Dict, List, Literal, Optional
import time
import logging
from itertools import product


logger = logging.getLogger(__name__)


@dataclass
class ParallelTask:
    """並行任務"""
    task_id: str
    params: Dict[str, Any]
    strategy_name: str


@dataclass
class ParallelResult:
    """並行結果"""
    task_id: str
    result: Any
    execution_time: float
    worker_id: int
    success: bool = True
    error_message: Optional[str] = None


class ParallelBacktester:
    """多核心並行回測器"""

    def __init__(
        self,
        n_workers: Optional[int] = None,
        backend: Literal['multiprocessing', 'concurrent'] = 'concurrent'
    ):
        """
        Args:
            n_workers: 工作程序數量（預設使用所有 CPU）
            backend: 並行後端
        """
        self.n_workers = n_workers or cpu_count()
        self.backend = backend
        logger.info(f"初始化並行回測器: {self.n_workers} workers, backend={backend}")

    def run_parallel(
        self,
        tasks: List[ParallelTask],
        backtest_fn: Callable[[ParallelTask], Any],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[ParallelResult]:
        """並行執行多個回測任務

        Args:
            tasks: 任務列表
            backtest_fn: 回測函數，接收 ParallelTask，返回任意結果
            progress_callback: 進度回調函數 (completed, total)

        Returns:
            List[ParallelResult]: 所有任務的結果
        """
        results: List[ParallelResult] = []
        total_tasks = len(tasks)
        completed = 0

        logger.info(f"開始並行執行 {total_tasks} 個任務")

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # 提交所有任務
            future_to_task = {
                executor.submit(_worker_fn, task, backtest_fn): task
                for task in tasks
            }

            # 收集結果
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1

                    if progress_callback:
                        progress_callback(completed, total_tasks)

                    if result.success:
                        logger.debug(f"任務完成: {task.task_id} ({result.execution_time:.2f}s)")
                    else:
                        logger.warning(f"任務失敗: {task.task_id} - {result.error_message}")

                except Exception as e:
                    # 處理 future 本身的異常
                    logger.error(f"任務執行異常: {task.task_id} - {str(e)}")
                    results.append(ParallelResult(
                        task_id=task.task_id,
                        result=None,
                        execution_time=0.0,
                        worker_id=-1,
                        success=False,
                        error_message=str(e)
                    ))
                    completed += 1

                    if progress_callback:
                        progress_callback(completed, total_tasks)

        logger.info(f"並行執行完成: {total_tasks} 個任務")
        return results

    def parameter_sweep(
        self,
        data: Any,
        strategy_fn: Callable,
        param_grid: Dict[str, List],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[ParallelResult]:
        """參數掃描（自動並行化）

        Args:
            data: 回測資料
            strategy_fn: 策略函數，接收 (data, **params)
            param_grid: 參數網格，例如 {'period': [10, 20, 30], 'threshold': [0.01, 0.02]}
            progress_callback: 進度回調函數

        Returns:
            List[ParallelResult]: 所有參數組合的回測結果
        """
        # 產生所有參數組合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        tasks: List[ParallelTask] = []
        for idx, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            task = ParallelTask(
                task_id=f"sweep_{idx}",
                params={'data': data, 'strategy_fn': strategy_fn, **params},
                strategy_name=strategy_fn.__name__
            )
            tasks.append(task)

        logger.info(f"參數掃描: {len(tasks)} 個組合")

        # 使用頂層函數避免序列化問題
        return self.run_parallel(tasks, _parameter_sweep_worker, progress_callback)

    def multi_strategy_backtest(
        self,
        data: Any,
        strategies: List[Callable],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, ParallelResult]:
        """多策略並行回測

        Args:
            data: 回測資料
            strategies: 策略函數列表，每個接收 data
            progress_callback: 進度回調函數

        Returns:
            Dict[str, ParallelResult]: 策略名稱 -> 結果
        """
        tasks: List[ParallelTask] = []
        for strategy_fn in strategies:
            task = ParallelTask(
                task_id=strategy_fn.__name__,
                params={'data': data, 'strategy_fn': strategy_fn},
                strategy_name=strategy_fn.__name__
            )
            tasks.append(task)

        logger.info(f"多策略回測: {len(tasks)} 個策略")

        # 使用頂層函數避免序列化問題
        results = self.run_parallel(tasks, _multi_strategy_worker, progress_callback)

        # 轉換為字典
        return {result.task_id: result for result in results}


def _worker_fn(task: ParallelTask, backtest_fn: Callable) -> ParallelResult:
    """工作程序函數

    Args:
        task: 並行任務
        backtest_fn: 回測函數

    Returns:
        ParallelResult: 執行結果
    """
    import os
    worker_id = os.getpid()
    start_time = time.time()

    try:
        result = backtest_fn(task)
        execution_time = time.time() - start_time

        return ParallelResult(
            task_id=task.task_id,
            result=result,
            execution_time=execution_time,
            worker_id=worker_id,
            success=True
        )
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Worker {worker_id} 執行失敗: {task.task_id} - {str(e)}")

        return ParallelResult(
            task_id=task.task_id,
            result=None,
            execution_time=execution_time,
            worker_id=worker_id,
            success=False,
            error_message=str(e)
        )


def _parameter_sweep_worker(task: ParallelTask) -> Any:
    """參數掃描專用 worker（頂層函數，可序列化）"""
    strategy_fn = task.params.pop('strategy_fn')
    return strategy_fn(**task.params)


def _multi_strategy_worker(task: ParallelTask) -> Any:
    """多策略專用 worker（頂層函數，可序列化）"""
    strategy_fn = task.params['strategy_fn']
    data = task.params['data']
    return strategy_fn(data)


# 便利函數

def run_parameter_sweep(
    data: Any,
    strategy_fn: Callable,
    param_grid: Dict[str, List],
    n_workers: Optional[int] = None,
    progress: bool = True
) -> List[ParallelResult]:
    """便利函數：執行參數掃描

    Args:
        data: 回測資料
        strategy_fn: 策略函數
        param_grid: 參數網格
        n_workers: 工作程序數量
        progress: 是否顯示進度

    Returns:
        List[ParallelResult]: 所有結果
    """
    backtester = ParallelBacktester(n_workers=n_workers)

    progress_callback = None
    if progress:
        def callback(completed: int, total: int):
            pct = completed / total * 100
            print(f"\rProgress: {completed}/{total} ({pct:.1f}%)", end='', flush=True)
        progress_callback = callback

    results = backtester.parameter_sweep(data, strategy_fn, param_grid, progress_callback)

    if progress:
        print()  # 換行

    return results


def run_multi_strategy(
    data: Any,
    strategies: List[Callable],
    n_workers: Optional[int] = None,
    progress: bool = True
) -> Dict[str, ParallelResult]:
    """便利函數：執行多策略回測

    Args:
        data: 回測資料
        strategies: 策略列表
        n_workers: 工作程序數量
        progress: 是否顯示進度

    Returns:
        Dict[str, ParallelResult]: 策略名稱 -> 結果
    """
    backtester = ParallelBacktester(n_workers=n_workers)

    progress_callback = None
    if progress:
        def callback(completed: int, total: int):
            pct = completed / total * 100
            print(f"\rProgress: {completed}/{total} ({pct:.1f}%)", end='', flush=True)
        progress_callback = callback

    results = backtester.multi_strategy_backtest(data, strategies, progress_callback)

    if progress:
        print()  # 換行

    return results
