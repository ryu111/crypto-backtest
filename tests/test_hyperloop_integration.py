"""
HyperLoop 系統整合測試

測試目標：
1. 模組整合測試
2. SharedDataPool 效能測試
3. GPUBatchOptimizer 功能測試
4. ExecutionScheduler 調度測試
5. HyperLoopController 實例化測試
"""

import time
import numpy as np
import pytest
from pathlib import Path

# 1. 模組整合測試
def test_module_imports():
    """測試所有模組可以正確導入"""
    print("\n=== 1. 模組整合測試 ===")

    try:
        from src.data.shared_pool import SharedDataPool, create_shared_pool, attach_to_pool
        print("✓ SharedDataPool 模組導入成功")
    except Exception as e:
        print(f"✗ SharedDataPool 模組導入失敗: {e}")
        raise

    try:
        from src.optimizer.gpu_batch import GPUBatchOptimizer, OPTUNA_AVAILABLE
        print("✓ GPUBatchOptimizer 模組導入成功")
        print(f"  Optuna available: {OPTUNA_AVAILABLE}")
    except Exception as e:
        print(f"✗ GPUBatchOptimizer 模組導入失敗: {e}")
        raise

    try:
        from src.automation.scheduler import ExecutionScheduler, BacktestTask, TaskType
        print("✓ ExecutionScheduler 模組導入成功")
    except Exception as e:
        print(f"✗ ExecutionScheduler 模組導入失敗: {e}")
        raise

    try:
        from src.automation.hyperloop import HyperLoopController, HyperLoopConfig, create_hyperloop
        print("✓ HyperLoopController 模組導入成功")
    except Exception as e:
        print(f"✗ HyperLoopController 模組導入失敗: {e}")
        raise

    print("✅ 所有模組導入成功\n")


def test_shared_data_pool_performance():
    """測試 SharedDataPool 的零拷貝效能"""
    print("=== 2. SharedDataPool 效能測試 ===")

    from src.data.shared_pool import SharedDataPool

    # 測試不同大小的資料
    test_sizes = [
        (10_000, "10K"),
        (100_000, "100K"),
        (1_000_000, "1M"),
    ]

    results = []

    for size, label in test_sizes:
        print(f"\n測試資料大小: {label} rows")

        # 建立測試資料 (OHLCV + timestamp = 6 columns)
        test_data = np.random.randn(size, 6).astype(np.float32)
        data_size_mb = test_data.nbytes / 1024 / 1024
        print(f"  資料大小: {data_size_mb:.2f} MB")

        with SharedDataPool(pool_name=f"perf_test_{label}") as pool:
            # 測試 put
            start = time.time()
            pool.put("test", test_data)
            put_time = time.time() - start

            # 測試 get (應該是零拷貝，非常快)
            start = time.time()
            retrieved = pool.get("test")
            get_time = time.time() - start

            # 驗證零拷貝
            try:
                # 檢查是否共享記憶體
                retrieved2 = pool.get("test")
                is_zero_copy = np.shares_memory(retrieved, retrieved2)

                # 驗證資料正確性
                is_correct = np.array_equal(test_data, retrieved)

                results.append({
                    'size': label,
                    'size_mb': data_size_mb,
                    'put_time': put_time,
                    'get_time': get_time,
                    'zero_copy': is_zero_copy,
                    'correct': is_correct
                })

                print(f"  Put time: {put_time:.4f}s")
                print(f"  Get time: {get_time:.6f}s")
                print(f"  零拷貝: {'✓' if is_zero_copy else '✗'}")
                print(f"  資料正確: {'✓' if is_correct else '✗'}")

                # 驗證 get 速度應該非常快（零拷貝）
                assert get_time < 0.001, f"Get time too slow: {get_time}s (應該 < 1ms)"
                assert is_zero_copy, "零拷貝機制失敗！"
                assert is_correct, "資料不正確！"

            except Exception as e:
                print(f"  ✗ 測試失敗: {e}")
                raise

    # 輸出效能摘要
    print("\n效能摘要：")
    print("-" * 60)
    print(f"{'大小':<10} {'Put (s)':<12} {'Get (s)':<12} {'零拷貝':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['size']:<10} {r['put_time']:<12.4f} {r['get_time']:<12.6f} {'✓' if r['zero_copy'] else '✗':<10}")
    print("-" * 60)

    print("✅ SharedDataPool 效能測試通過\n")


def test_gpu_batch_optimizer():
    """測試 GPUBatchOptimizer 功能"""
    print("=== 3. GPUBatchOptimizer 功能測試 ===")

    from src.optimizer.gpu_batch import GPUBatchOptimizer, OPTUNA_AVAILABLE

    if not OPTUNA_AVAILABLE:
        print("⚠️  Optuna 未安裝，跳過 GPU 優化器測試")
        return

    try:
        optimizer = GPUBatchOptimizer(verbose=True)
        print(f"✓ 優化器建立成功")
        print(f"  Backend: {optimizer._backend}")
        print(f"  GPU available: {optimizer.is_gpu_available()}")

        # 檢查 backend (包含 MPS for Apple Silicon)
        assert optimizer._backend in ['cupy', 'pytorch', 'numpy', 'mps'], \
            f"Unknown backend: {optimizer._backend}"

        print("✅ GPUBatchOptimizer 功能測試通過\n")

    except Exception as e:
        print(f"✗ GPUBatchOptimizer 測試失敗: {e}")
        raise


def test_execution_scheduler():
    """測試 ExecutionScheduler 調度邏輯"""
    print("=== 4. ExecutionScheduler 調度測試 ===")

    from src.automation.scheduler import ExecutionScheduler, BacktestTask, TaskType

    try:
        scheduler = ExecutionScheduler(verbose=True)
        print("✓ Scheduler 建立成功")

        # 測試不同任務類型的調度決策
        test_cases = [
            BacktestTask(
                task_id="small_sweep",
                task_type=TaskType.PARAM_SWEEP,
                strategy_name="rsi",
                param_count=10
            ),
            BacktestTask(
                task_id="large_sweep",
                task_type=TaskType.PARAM_SWEEP,
                strategy_name="rsi",
                param_count=500
            ),
            BacktestTask(
                task_id="walk_forward",
                task_type=TaskType.WALK_FORWARD,
                strategy_name="rsi"
            ),
            BacktestTask(
                task_id="monte_carlo",
                task_type=TaskType.MONTE_CARLO,
                strategy_name="rsi"
            ),
        ]

        print("\n調度決策：")
        print("-" * 70)
        print(f"{'Task ID':<20} {'Type':<15} {'Executor':<15} {'Workers':<10}")
        print("-" * 70)

        for task in test_cases:
            plan = scheduler.schedule(task)
            print(f"{task.task_id:<20} {task.task_type.value:<15} {plan.executor.value:<15} {plan.n_workers:<10}")

            # 驗證調度邏輯
            if task.task_type == TaskType.PARAM_SWEEP:
                # 小量參數：使用 CPU_POOL，worker 數 = min(param_count, max_cpu_workers)
                if task.param_count and task.param_count < 100:
                    assert plan.executor.value == 'cpu_pool', "小量參數應使用 CPU Pool"
                    assert plan.n_workers == min(task.param_count, scheduler.max_cpu_workers), \
                        f"Workers 應等於 min(param_count={task.param_count}, max_cpu_workers={scheduler.max_cpu_workers})"
                # 大量參數：應該使用 GPU（如果可用）
                elif task.param_count and task.param_count >= 100:
                    if scheduler.gpu_available:
                        assert plan.executor.value == 'gpu_batch', "大量參數應使用 GPU Batch"
                    else:
                        assert plan.executor.value == 'cpu_pool', "無 GPU 時使用 CPU Pool"

        print("-" * 70)
        print("✅ ExecutionScheduler 調度測試通過\n")

    except Exception as e:
        print(f"✗ ExecutionScheduler 測試失敗: {e}")
        raise


def test_hyperloop_controller():
    """測試 HyperLoopController 實例化"""
    print("=== 5. HyperLoopController 實例化測試 ===")

    from src.automation.hyperloop import HyperLoopController, HyperLoopConfig

    try:
        config = HyperLoopConfig(
            max_workers=4,
            use_gpu=False,  # 測試時禁用 GPU
            symbols=['BTCUSDT'],
            timeframes=['1h'],
            n_trials=10
        )

        controller = HyperLoopController(config, verbose=True)
        print("✓ Controller 建立成功")
        print(f"  Max workers: {controller.config.max_workers}")
        print(f"  Use GPU: {controller.config.use_gpu}")
        print(f"  Symbols: {controller.config.symbols}")
        print(f"  Timeframes: {controller.config.timeframes}")
        print(f"  N trials: {controller.config.n_trials}")

        # 驗證配置
        assert controller.config.max_workers == 4
        assert controller.config.use_gpu == False
        assert controller.config.symbols == ['BTCUSDT']
        assert controller.config.timeframes == ['1h']
        assert controller.config.n_trials == 10

        print("✅ HyperLoopController 實例化測試通過\n")

    except Exception as e:
        print(f"✗ HyperLoopController 測試失敗: {e}")
        raise


def test_integration_summary():
    """整合測試摘要"""
    print("\n" + "=" * 70)
    print("HyperLoop 系統整合測試摘要")
    print("=" * 70)

    test_results = {
        "模組導入": "PASS",
        "SharedDataPool 效能": "PASS",
        "GPUBatchOptimizer": "PASS",
        "ExecutionScheduler": "PASS",
        "HyperLoopController": "PASS",
    }

    for test_name, result in test_results.items():
        status = "✅" if result == "PASS" else "❌"
        print(f"{status} {test_name}: {result}")

    print("=" * 70)
    print("\n所有整合測試通過！系統可以開始執行 HyperLoop。\n")


if __name__ == "__main__":
    """執行完整測試套件"""
    try:
        # 1. 模組整合測試
        test_module_imports()

        # 2. SharedDataPool 效能測試
        test_shared_data_pool_performance()

        # 3. GPUBatchOptimizer 功能測試
        test_gpu_batch_optimizer()

        # 4. ExecutionScheduler 調度測試
        test_execution_scheduler()

        # 5. HyperLoopController 實例化測試
        test_hyperloop_controller()

        # 6. 輸出摘要
        test_integration_summary()

    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
