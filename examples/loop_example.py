"""
Loop 控制器使用範例

展示如何使用 LoopController 進行持續的策略優化。
"""

import sys
from pathlib import Path
from datetime import datetime

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.automation import (
    LoopController,
    LoopMode,
    IterationResult,
    IterationStatus
)
import pandas as pd
import numpy as np


def example_simple_iteration():
    """範例：最簡單的迭代"""

    iteration_count = 0

    def simple_iteration() -> IterationResult:
        """模擬單次迭代"""
        nonlocal iteration_count
        iteration_count += 1

        # 模擬優化過程（實際會是貝葉斯優化）
        sharpe = np.random.uniform(0.5, 2.5)
        total_return = np.random.uniform(0.1, 0.6)
        max_drawdown = np.random.uniform(-0.05, -0.20)

        return IterationResult(
            iteration=iteration_count,
            timestamp=datetime.now(),
            status=IterationStatus.SUCCESS,
            sharpe_ratio=sharpe,
            total_return=total_return,
            max_drawdown=max_drawdown,
            strategy_name="MA Cross",
            best_params={'fast': 10, 'slow': 30},
            experiment_id=f"exp_{iteration_count}"
        )

    # 建立控制器
    controller = LoopController(
        iteration_callback=simple_iteration,
        auto_save=True
    )

    print("\n===== 範例 1: 執行 5 次迭代 =====\n")

    # 執行 5 次迭代
    controller.start(
        mode=LoopMode.N_ITERATIONS,
        target=5
    )

    # 顯示摘要
    print("\n最終摘要:")
    print(controller.get_summary())


def example_with_callbacks():
    """範例：使用回調函數"""

    def iteration_func() -> IterationResult:
        """模擬迭代"""
        sharpe = np.random.uniform(0.5, 2.5)
        return IterationResult(
            iteration=0,
            timestamp=datetime.now(),
            status=IterationStatus.SUCCESS,
            sharpe_ratio=sharpe,
            total_return=0.3,
            max_drawdown=-0.1,
            strategy_name="Test Strategy",
            best_params={'param1': 10}
        )

    # 定義回調函數
    def on_new_best(result: IterationResult):
        print(f"🎉 新的最佳 Sharpe: {result.sharpe_ratio:.4f}")

    def on_iteration_end(iteration_num):
        print(f"迭代 #{iteration_num} 完成")

    callbacks = {
        'on_new_best': on_new_best,
        'on_iteration_end': on_iteration_end
    }

    controller = LoopController(
        iteration_callback=iteration_func,
        callbacks=callbacks
    )

    print("\n===== 範例 2: 使用回調函數 =====\n")

    controller.start(
        mode=LoopMode.N_ITERATIONS,
        target=3
    )


def example_until_target():
    """範例：執行直到達到目標"""

    iteration_count = 0

    def improving_iteration() -> IterationResult:
        """模擬逐步改進的迭代"""
        nonlocal iteration_count
        iteration_count += 1

        # Sharpe 逐步提升（帶隨機性）
        base_sharpe = 0.5 + (iteration_count * 0.3)
        sharpe = base_sharpe + np.random.uniform(-0.2, 0.2)

        return IterationResult(
            iteration=iteration_count,
            timestamp=datetime.now(),
            status=IterationStatus.SUCCESS,
            sharpe_ratio=sharpe,
            total_return=0.3,
            max_drawdown=-0.1,
            strategy_name="Improving Strategy",
            best_params={'iter': iteration_count}
        )

    controller = LoopController(
        iteration_callback=improving_iteration
    )

    print("\n===== 範例 3: 執行直到 Sharpe >= 2.0 =====\n")

    controller.start(
        mode=LoopMode.UNTIL_TARGET,
        target=2.0
    )

    print(f"\n達到目標！總共執行 {iteration_count} 次迭代")


def example_pause_resume():
    """範例：暫停和恢復"""

    iteration_count = 0

    def iteration_func() -> IterationResult:
        nonlocal iteration_count
        iteration_count += 1

        # 在第 3 次迭代時暫停
        if iteration_count == 3:
            print("\n⏸️  手動暫停（實際會從外部控制）")
            # controller.pause()  # 實際使用時從外部呼叫

        sharpe = np.random.uniform(0.5, 2.5)
        return IterationResult(
            iteration=iteration_count,
            timestamp=datetime.now(),
            status=IterationStatus.SUCCESS,
            sharpe_ratio=sharpe,
            total_return=0.3,
            max_drawdown=-0.1,
            strategy_name="Pausable Strategy",
            best_params={}
        )

    controller = LoopController(
        iteration_callback=iteration_func
    )

    print("\n===== 範例 4: 暫停和恢復（概念示範） =====\n")
    print("實際使用時可透過信號或外部控制暫停/恢復\n")

    controller.start(
        mode=LoopMode.N_ITERATIONS,
        target=5
    )


def example_get_progress():
    """範例：取得進度資訊"""

    iteration_count = 0

    def iteration_func() -> IterationResult:
        nonlocal iteration_count
        iteration_count += 1

        # 顯示進度
        if iteration_count % 2 == 0:
            progress = controller.get_progress()
            print(f"\n📊 進度報告:")
            print(f"   完成: {progress['completed_iterations']}/{progress.get('estimated_remaining', '?')}")
            print(f"   成功率: {progress['success_rate']:.1%}")
            print(f"   最佳 Sharpe: {progress['best_sharpe']:.4f}")
            print(f"   已執行: {progress['elapsed_time']}\n")

        sharpe = np.random.uniform(0.5, 2.5)
        return IterationResult(
            iteration=iteration_count,
            timestamp=datetime.now(),
            status=IterationStatus.SUCCESS,
            sharpe_ratio=sharpe,
            total_return=0.3,
            max_drawdown=-0.1,
            strategy_name="Progress Strategy",
            best_params={}
        )

    controller = LoopController(
        iteration_callback=iteration_func
    )

    print("\n===== 範例 5: 進度追蹤 =====\n")

    controller.start(
        mode=LoopMode.N_ITERATIONS,
        target=6
    )


def example_iteration_history():
    """範例：取得迭代歷史"""

    iteration_count = 0

    def iteration_func() -> IterationResult:
        nonlocal iteration_count
        iteration_count += 1

        sharpe = np.random.uniform(0.5, 2.5)
        return IterationResult(
            iteration=iteration_count,
            timestamp=datetime.now(),
            status=IterationStatus.SUCCESS,
            sharpe_ratio=sharpe,
            total_return=np.random.uniform(0.1, 0.5),
            max_drawdown=np.random.uniform(-0.05, -0.20),
            strategy_name=f"Strategy_{iteration_count}",
            best_params={'iteration': iteration_count}
        )

    controller = LoopController(
        iteration_callback=iteration_func
    )

    print("\n===== 範例 6: 迭代歷史分析 =====\n")

    controller.start(
        mode=LoopMode.N_ITERATIONS,
        target=5
    )

    # 取得迭代歷史
    history_df = controller.get_iteration_history()

    print("\n迭代歷史 DataFrame:")
    print(history_df[['iteration', 'sharpe_ratio', 'total_return', 'strategy_name']])

    print("\n統計資訊:")
    print(f"平均 Sharpe: {history_df['sharpe_ratio'].mean():.4f}")
    print(f"最大 Sharpe: {history_df['sharpe_ratio'].max():.4f}")
    print(f"最小 Sharpe: {history_df['sharpe_ratio'].min():.4f}")


if __name__ == '__main__':
    print("="*60)
    print("Loop 控制器使用範例")
    print("="*60)

    # 執行所有範例
    example_simple_iteration()
    example_with_callbacks()
    example_until_target()
    example_pause_resume()
    example_get_progress()
    example_iteration_history()

    print("\n" + "="*60)
    print("範例執行完成")
    print("="*60)
