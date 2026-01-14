"""
測試 DuckDB Repository

驗證基本 CRUD 操作是否正常。
"""

from datetime import datetime
from src.db import Repository, QueryFilters
from src.types import ExperimentRecord, StrategyStats


def test_basic_operations():
    """測試基本操作"""
    print("=== 測試 DuckDB Repository ===\n")

    # 建立測試資料庫
    with Repository("test_experiments.duckdb") as repo:
        # 1. 建立測試實驗記錄
        print("1. 建立測試實驗記錄...")
        experiment = ExperimentRecord(
            id="exp_test_001",
            timestamp=datetime.now(),
            strategy={
                'name': 'ma_cross',
                'type': 'trend',
                'version': '1.0',
                'params': {'fast_period': 10, 'slow_period': 30}
            },
            config={
                'symbol': 'BTCUSDT',
                'timeframe': '4h',
                'start_date': '2023-01-01',
                'end_date': '2024-01-01',
            },
            results={
                'sharpe_ratio': 2.1,
                'total_return': 45.5,
                'max_drawdown': 12.3,
                'win_rate': 0.62,
                'profit_factor': 1.8,
                'total_trades': 120,
            },
            validation={
                'grade': 'A',
                'stages_passed': [1, 2, 3, 4, 5],
            },
            status='completed',
            tags=['backtest', 'production'],
        )

        repo.insert_experiment(experiment)
        print("✓ 實驗記錄已插入\n")

        # 2. 查詢單一實驗
        print("2. 查詢單一實驗...")
        retrieved = repo.get_experiment("exp_test_001")
        if retrieved:
            print(f"✓ 找到實驗: {retrieved.id}")
            print(f"  策略: {retrieved.strategy_name}")
            print(f"  Sharpe: {retrieved.sharpe_ratio}")
            print(f"  等級: {retrieved.grade}\n")
        else:
            print("✗ 找不到實驗\n")

        # 3. 查詢實驗（使用 filters）
        print("3. 查詢實驗（Sharpe > 2.0）...")
        filters = QueryFilters(min_sharpe=2.0)
        results = repo.query_experiments(filters)
        print(f"✓ 找到 {len(results)} 筆記錄\n")

        # 4. 獲取最佳實驗
        print("4. 獲取最佳 3 個實驗...")
        best = repo.get_best_experiments(metric="sharpe_ratio", n=3)
        for i, exp in enumerate(best, 1):
            print(f"  {i}. {exp.strategy_name}: Sharpe {exp.sharpe_ratio}")
        print()

        # 5. 更新策略統計
        print("5. 更新策略統計...")
        stats = StrategyStats(
            name='ma_cross',
            attempts=10,
            successes=3,
            avg_sharpe=1.5,
            best_sharpe=2.1,
            worst_sharpe=0.8,
            best_params={'fast_period': 10, 'slow_period': 30},
            last_attempt=datetime.now(),
        )
        repo.update_strategy_stats(stats)
        print("✓ 策略統計已更新\n")

        # 6. 查詢策略統計
        print("6. 查詢策略統計...")
        retrieved_stats = repo.get_strategy_stats('ma_cross')
        if retrieved_stats:
            print(f"✓ 找到策略統計:")
            print(f"  嘗試次數: {retrieved_stats.attempts}")
            print(f"  成功次數: {retrieved_stats.successes}")
            print(f"  平均 Sharpe: {retrieved_stats.avg_sharpe}")
            print(f"  最佳 Sharpe: {retrieved_stats.best_sharpe}")
        print()

    print("=== 測試完成 ===")
    print("\n清理測試檔案...")
    import os
    if os.path.exists("test_experiments.duckdb"):
        os.remove("test_experiments.duckdb")
        print("✓ 測試檔案已刪除")


if __name__ == "__main__":
    test_basic_operations()
