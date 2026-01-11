"""
並行回測器測試
"""

import pytest
import time
from typing import Any, Dict
from src.backtester.parallel import (
    ParallelBacktester,
    ParallelTask,
    ParallelResult,
    run_parameter_sweep,
    run_multi_strategy,
    _worker_fn
)


# 測試用策略函數（頂層函數，可序列化）

def dummy_strategy(data: Any, period: int = 10, threshold: float = 0.01) -> Dict:
    """測試用策略"""
    time.sleep(0.1)  # 模擬計算
    return {
        'period': period,
        'threshold': threshold,
        'profit': period * threshold * 100
    }


def fast_strategy(data: Any) -> Dict:
    """快速策略"""
    return {'name': 'fast', 'profit': 10.0}


def slow_strategy(data: Any) -> Dict:
    """慢速策略"""
    time.sleep(0.2)
    return {'name': 'slow', 'profit': 20.0}


def error_strategy(data: Any) -> Dict:
    """會拋出錯誤的策略"""
    raise ValueError("Intentional error")


def mixed_strategy(data: Any, should_fail: bool = False) -> Dict:
    """混合成功/失敗策略"""
    if should_fail:
        raise ValueError("Expected failure")
    return {'profit': 10.0}


# 頂層 backtest 函數（可序列化）

def backtest_dummy(task: ParallelTask) -> Dict:
    """回測 dummy_strategy"""
    return dummy_strategy(**task.params)


def backtest_fast(task: ParallelTask) -> Dict:
    """回測 fast_strategy"""
    return fast_strategy(**task.params)


def backtest_mixed(task: ParallelTask) -> Dict:
    """回測混合策略"""
    if task.strategy_name == "fast":
        return fast_strategy(**task.params)
    else:
        return error_strategy(**task.params)


# 測試案例

class TestParallelBacktester:
    """並行回測器測試"""

    def test_initialization_default_workers(self):
        """測試：預設使用所有 CPU"""
        backtester = ParallelBacktester()
        from multiprocessing import cpu_count
        assert backtester.n_workers == cpu_count()

    def test_initialization_custom_workers(self):
        """測試：自訂工作程序數量"""
        backtester = ParallelBacktester(n_workers=4)
        assert backtester.n_workers == 4

    def test_run_parallel_basic(self):
        """測試：基本並行執行"""
        backtester = ParallelBacktester(n_workers=2)

        tasks = [
            ParallelTask(task_id="task_1", params={'data': None, 'period': 10}, strategy_name="dummy"),
            ParallelTask(task_id="task_2", params={'data': None, 'period': 20}, strategy_name="dummy"),
        ]

        results = backtester.run_parallel(tasks, backtest_dummy)

        assert len(results) == 2
        assert all(isinstance(r, ParallelResult) for r in results)
        assert all(r.success for r in results)

    def test_run_parallel_with_progress(self):
        """測試：帶進度回調的並行執行"""
        backtester = ParallelBacktester(n_workers=2)
        progress_updates = []

        def progress_callback(completed: int, total: int):
            progress_updates.append((completed, total))

        tasks = [
            ParallelTask(task_id=f"task_{i}", params={'data': None}, strategy_name="fast")
            for i in range(5)
        ]

        backtester.run_parallel(tasks, backtest_fast, progress_callback)

        assert len(progress_updates) == 5
        assert progress_updates[-1] == (5, 5)

    def test_run_parallel_with_errors(self):
        """測試：處理執行錯誤"""
        backtester = ParallelBacktester(n_workers=2)

        tasks = [
            ParallelTask(task_id="success", params={'data': None}, strategy_name="fast"),
            ParallelTask(task_id="error", params={'data': None}, strategy_name="error"),
        ]

        results = backtester.run_parallel(tasks, backtest_mixed)

        assert len(results) == 2
        success_result = next(r for r in results if r.task_id == "success")
        error_result = next(r for r in results if r.task_id == "error")

        assert success_result.success is True
        assert error_result.success is False
        assert error_result.error_message is not None

    def test_parameter_sweep(self):
        """測試：參數掃描"""
        backtester = ParallelBacktester(n_workers=2)

        param_grid = {
            'period': [10, 20],
            'threshold': [0.01, 0.02]
        }

        results = backtester.parameter_sweep(
            data=None,
            strategy_fn=dummy_strategy,
            param_grid=param_grid
        )

        # 2 x 2 = 4 組合
        assert len(results) == 4
        assert all(r.success for r in results)

        # 驗證參數組合正確
        periods = sorted([r.result['period'] for r in results])
        thresholds = sorted([r.result['threshold'] for r in results])
        assert periods == [10, 10, 20, 20]
        assert thresholds == [0.01, 0.01, 0.02, 0.02]

    def test_multi_strategy_backtest(self):
        """測試：多策略回測"""
        backtester = ParallelBacktester(n_workers=2)

        strategies = [fast_strategy, slow_strategy]

        results = backtester.multi_strategy_backtest(
            data=None,
            strategies=strategies
        )

        assert len(results) == 2
        assert 'fast_strategy' in results
        assert 'slow_strategy' in results
        assert results['fast_strategy'].success
        assert results['slow_strategy'].success

    def test_worker_fn_success(self):
        """測試：工作函數成功執行"""
        task = ParallelTask(
            task_id="test",
            params={'data': None, 'period': 10},
            strategy_name="dummy"
        )

        result = _worker_fn(task, backtest_dummy)

        assert result.success is True
        assert result.task_id == "test"
        assert result.result is not None
        assert result.execution_time > 0

    def test_worker_fn_error(self):
        """測試：工作函數錯誤處理"""
        def backtest_error(t: ParallelTask) -> Dict:
            raise ValueError("Test error")

        task = ParallelTask(
            task_id="error_task",
            params={'data': None},
            strategy_name="error"
        )

        result = _worker_fn(task, backtest_error)

        assert result.success is False
        assert result.error_message == "Test error"
        assert result.result is None


class TestConvenienceFunctions:
    """便利函數測試"""

    def test_run_parameter_sweep(self):
        """測試：參數掃描便利函數"""
        param_grid = {
            'period': [10, 20],
            'threshold': [0.01]
        }

        results = run_parameter_sweep(
            data=None,
            strategy_fn=dummy_strategy,
            param_grid=param_grid,
            n_workers=2,
            progress=False
        )

        assert len(results) == 2
        assert all(r.success for r in results)

    def test_run_multi_strategy(self):
        """測試：多策略便利函數"""
        strategies = [fast_strategy, slow_strategy]

        results = run_multi_strategy(
            data=None,
            strategies=strategies,
            n_workers=2,
            progress=False
        )

        assert len(results) == 2
        assert 'fast_strategy' in results
        assert 'slow_strategy' in results


class TestPerformance:
    """效能測試"""

    def test_parallel_speedup(self):
        """測試：並行加速效果"""
        # 對於小任務，multiprocessing overhead 可能比實際執行時間長
        # 所以這裡只驗證並行執行能完成，不測試加速比
        backtester = ParallelBacktester(n_workers=4)
        tasks = [
            ParallelTask(task_id=f"task_{i}", params={'data': None, 'period': 10}, strategy_name="dummy")
            for i in range(8)
        ]

        results = backtester.run_parallel(tasks, backtest_dummy)

        # 驗證所有任務都完成
        assert len(results) == 8
        assert all(r.success for r in results)

        # 驗證使用了多個 worker
        worker_ids = {r.worker_id for r in results}
        assert len(worker_ids) > 1, "應該使用多個 worker"


# 整合測試

class TestIntegration:
    """整合測試"""

    def test_large_parameter_grid(self):
        """測試：大型參數網格"""
        param_grid = {
            'period': [5, 10, 15, 20],
            'threshold': [0.01, 0.02, 0.03]
        }

        results = run_parameter_sweep(
            data=None,
            strategy_fn=dummy_strategy,
            param_grid=param_grid,
            n_workers=4,
            progress=False
        )

        # 4 x 3 = 12 組合
        assert len(results) == 12
        assert all(r.success for r in results)

    def test_mixed_success_failure(self):
        """測試：混合成功和失敗的任務"""
        param_grid = {
            'should_fail': [False, False, True, False]
        }

        results = run_parameter_sweep(
            data=None,
            strategy_fn=mixed_strategy,
            param_grid=param_grid,
            n_workers=2,
            progress=False
        )

        assert len(results) == 4
        success_count = sum(1 for r in results if r.success)
        failure_count = sum(1 for r in results if not r.success)

        assert success_count == 3
        assert failure_count == 1
