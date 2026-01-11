"""
簡單的 Loop 控制器測試

快速驗證 Loop 控制器的基本功能。
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
import numpy as np


def main():
    """主函數"""

    print("="*60)
    print("Loop 控制器簡單測試")
    print("="*60)

    iteration_count = 0

    def simple_iteration() -> IterationResult:
        """模擬單次迭代"""
        nonlocal iteration_count
        iteration_count += 1

        # 模擬優化過程（隨機 Sharpe）
        sharpe = np.random.uniform(0.5, 2.5)
        total_return = np.random.uniform(0.1, 0.6)
        max_drawdown = np.random.uniform(-0.05, -0.20)

        print(f"  執行迭代 {iteration_count}...")
        print(f"  Sharpe: {sharpe:.4f}")

        return IterationResult(
            iteration=iteration_count,
            timestamp=datetime.now(),
            status=IterationStatus.SUCCESS,
            sharpe_ratio=sharpe,
            total_return=total_return,
            max_drawdown=max_drawdown,
            strategy_name="Test Strategy",
            best_params={'test_param': iteration_count},
            experiment_id=f"exp_{iteration_count:03d}"
        )

    # 建立控制器
    print("\n建立 Loop 控制器...")
    controller = LoopController(
        iteration_callback=simple_iteration,
        auto_save=True
    )

    # 測試 1: 執行 5 次迭代
    print("\n【測試 1】執行 5 次迭代")
    print("-"*60)

    controller.start(
        mode=LoopMode.N_ITERATIONS,
        target=5
    )

    # 顯示摘要
    print("\n執行摘要:")
    print(controller.get_summary())

    # 測試 2: 進度資訊
    print("\n【測試 2】進度資訊")
    print("-"*60)

    progress = controller.get_progress()
    print(f"完成迭代: {progress['completed_iterations']}")
    print(f"成功率: {progress['success_rate']:.1%}")
    print(f"最佳 Sharpe: {progress['best_sharpe']:.4f}")
    print(f"最佳策略: {progress['best_strategy']}")

    # 測試 3: 迭代歷史
    print("\n【測試 3】迭代歷史")
    print("-"*60)

    history_df = controller.get_iteration_history()
    print("\n前 3 筆記錄:")
    print(history_df[['iteration', 'sharpe_ratio', 'total_return', 'status']].head(3))

    print("\n統計資訊:")
    print(f"平均 Sharpe: {history_df['sharpe_ratio'].mean():.4f}")
    print(f"最大 Sharpe: {history_df['sharpe_ratio'].max():.4f}")
    print(f"最小 Sharpe: {history_df['sharpe_ratio'].min():.4f}")

    # 測試 4: 狀態檔案
    print("\n【測試 4】狀態檔案")
    print("-"*60)

    state_file = controller.state_file
    if state_file.exists():
        print(f"✓ 狀態檔案已建立: {state_file}")
        print(f"  檔案大小: {state_file.stat().st_size} bytes")

        # 載入並驗證
        loaded_state = controller.load_state()
        print(f"  載入狀態成功")
        print(f"  完成迭代: {loaded_state.completed_iterations}")
        print(f"  最佳 Sharpe: {loaded_state.best_sharpe:.4f}")
    else:
        print("✗ 狀態檔案未找到")

    # 清理
    print("\n清理狀態檔案...")
    controller.clear_state()

    print("\n" + "="*60)
    print("測試完成！")
    print("="*60)


if __name__ == '__main__':
    main()
