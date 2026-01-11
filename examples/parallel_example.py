"""
並行回測範例
"""

import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
from src.backtester.parallel import (
    ParallelBacktester,
    ParallelTask,
    run_parameter_sweep,
    run_multi_strategy
)


# 策略函數必須定義在頂層（multiprocessing 需要）

def simple_strategy(data, period: int, threshold: float):
    """簡單策略"""
    time.sleep(0.05)  # 模擬計算
    profit = period * threshold * 1000
    return {
        'period': period,
        'threshold': threshold,
        'profit': profit,
        'sharpe': profit / 100
    }


def trend_following(data):
    """趨勢跟蹤策略"""
    time.sleep(0.1)
    return {
        'name': 'Trend Following',
        'profit': 1250.0,
        'sharpe': 1.8,
        'max_dd': -0.15
    }


def mean_reversion(data):
    """均值回歸策略"""
    time.sleep(0.1)
    return {
        'name': 'Mean Reversion',
        'profit': 980.0,
        'sharpe': 1.5,
        'max_dd': -0.12
    }


def momentum(data):
    """動量策略"""
    time.sleep(0.1)
    return {
        'name': 'Momentum',
        'profit': 1450.0,
        'sharpe': 2.1,
        'max_dd': -0.18
    }


def breakout(data):
    """突破策略"""
    time.sleep(0.1)
    return {
        'name': 'Breakout',
        'profit': 1100.0,
        'sharpe': 1.6,
        'max_dd': -0.20
    }


def complex_backtest(task: ParallelTask):
    """複雜回測函數"""
    symbol = task.params['symbol']
    period = task.params['period']

    time.sleep(0.05)

    # 模擬不同幣種的結果
    base_profit = {'BTC': 1000, 'ETH': 800, 'BNB': 600, 'SOL': 700}
    profit = base_profit.get(symbol, 500) * (period / 10)

    return {
        'symbol': symbol,
        'period': period,
        'profit': profit,
        'trades': period * 10
    }


def unreliable_strategy(data, fail_rate: float = 0.0):
    """不穩定策略（隨機失敗）"""
    import random

    if random.random() < fail_rate:
        raise ValueError(f"Strategy failed (fail_rate={fail_rate})")

    return {
        'profit': 1000 * (1 - fail_rate),
        'fail_rate': fail_rate
    }


# 範例 1: 參數掃描
def example_parameter_sweep():
    """範例：參數掃描"""
    print("範例 1: 參數掃描\n")

    # 定義參數網格
    param_grid = {
        'period': [5, 10, 15, 20, 25],
        'threshold': [0.01, 0.02, 0.03, 0.04, 0.05]
    }

    # 執行參數掃描（5 x 5 = 25 組合）
    print(f"測試 {len(param_grid['period'])} x {len(param_grid['threshold'])} = "
          f"{len(param_grid['period']) * len(param_grid['threshold'])} 組參數")

    results = run_parameter_sweep(
        data=None,
        strategy_fn=simple_strategy,
        param_grid=param_grid,
        n_workers=4,
        progress=True
    )

    # 找出最佳參數
    best = max(results, key=lambda r: r.result['profit'])
    print(f"\n最佳參數:")
    print(f"  Period: {best.result['period']}")
    print(f"  Threshold: {best.result['threshold']}")
    print(f"  Profit: ${best.result['profit']:.2f}")
    print(f"  Sharpe: {best.result['sharpe']:.2f}")


# 範例 2: 多策略比較
def example_multi_strategy():
    """範例：多策略比較"""
    print("\n\n範例 2: 多策略比較\n")

    strategies = [trend_following, mean_reversion, momentum, breakout]

    # 執行多策略回測
    print(f"測試 {len(strategies)} 個策略")

    results = run_multi_strategy(
        data=None,
        strategies=strategies,
        n_workers=4,
        progress=True
    )

    # 顯示結果
    print("\n策略比較:")
    print(f"{'策略名稱':<20} {'獲利':>10} {'Sharpe':>8} {'最大回撤':>10}")
    print("-" * 52)

    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].result['sharpe'],
        reverse=True
    )

    for name, result in sorted_results:
        data = result.result
        print(f"{data['name']:<20} ${data['profit']:>9.2f} {data['sharpe']:>8.2f} {data['max_dd']:>9.1%}")


# 範例 3: 自訂並行任務
def example_custom_parallel():
    """範例：自訂並行任務"""
    print("\n\n範例 3: 自訂並行任務\n")

    # 建立自訂任務
    tasks = []
    for symbol in ['BTC', 'ETH', 'BNB', 'SOL']:
        for period in [10, 20, 30]:
            task = ParallelTask(
                task_id=f"{symbol}_{period}",
                params={'symbol': symbol, 'period': period},
                strategy_name="complex"
            )
            tasks.append(task)

    # 並行執行
    backtester = ParallelBacktester(n_workers=4)
    print(f"執行 {len(tasks)} 個自訂任務")

    progress_count = [0]

    def progress_callback(completed, total):
        progress_count[0] = completed
        pct = completed / total * 100
        print(f"\rProgress: {completed}/{total} ({pct:.1f}%)", end='', flush=True)

    results = backtester.run_parallel(tasks, complex_backtest, progress_callback)
    print()

    # 按幣種分組顯示結果
    print("\n結果（按幣種分組）:")
    for symbol in ['BTC', 'ETH', 'BNB', 'SOL']:
        symbol_results = [r for r in results if r.result['symbol'] == symbol]
        total_profit = sum(r.result['profit'] for r in symbol_results)
        print(f"  {symbol}: ${total_profit:,.2f}")


# 範例 4: 錯誤處理
def example_error_handling():
    """範例：錯誤處理"""
    print("\n\n範例 4: 錯誤處理\n")

    param_grid = {
        'fail_rate': [0.0, 0.2, 0.5, 0.8, 1.0]
    }

    results = run_parameter_sweep(
        data=None,
        strategy_fn=unreliable_strategy,
        param_grid=param_grid,
        n_workers=2,
        progress=False
    )

    # 統計成功/失敗
    success = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print(f"總任務數: {len(results)}")
    print(f"成功: {len(success)}")
    print(f"失敗: {len(failed)}")

    if failed:
        print("\n失敗的任務:")
        for r in failed:
            print(f"  {r.task_id}: {r.error_message}")


if __name__ == '__main__':
    # 執行所有範例
    example_parameter_sweep()
    example_multi_strategy()
    example_custom_parallel()
    example_error_handling()

    print("\n\n所有範例執行完成！")
