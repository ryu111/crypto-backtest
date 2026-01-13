"""
多目標優化器進階功能範例

展示以下功能：
1. 約束條件支援
2. 暖啟動
3. 膝點檢測
4. Pareto 前沿篩選
5. 3D 視覺化與平行座標圖
"""

import numpy as np
from typing import Dict
from src.optimizer.multi_objective import MultiObjectiveOptimizer


def main():
    # ===== 1. 定義參數空間 =====
    param_space = {
        'fast_period': {'type': 'int', 'low': 5, 'high': 20},
        'slow_period': {'type': 'int', 'low': 20, 'high': 50},
        'stop_loss_atr': {'type': 'float', 'low': 1.0, 'high': 3.0, 'step': 0.1},
        'leverage': {'type': 'int', 'low': 1, 'high': 15}
    }

    # ===== 2. 定義評估函數（模擬） =====
    def evaluate_strategy(params: Dict) -> Dict[str, float]:
        """模擬策略評估（實際應該是回測引擎）"""
        # 模擬計算 Sharpe Ratio
        sharpe = np.random.normal(
            1.0 + (params['fast_period'] - params['slow_period']) / 100,
            0.3
        )

        # 模擬計算 Max Drawdown（%）
        max_dd = np.random.uniform(0.05, 0.30) * (params['leverage'] / 5)

        # 模擬計算 Sortino Ratio
        sortino = sharpe * np.random.uniform(0.9, 1.2)

        # 模擬計算 Win Rate
        win_rate = np.random.uniform(0.45, 0.65)

        return {
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'sortino_ratio': sortino,
            'win_rate': win_rate
        }

    # ===== 3. 定義約束條件 =====
    def leverage_constraint(params: Dict) -> float:
        """槓桿不得超過 10x"""
        return max(0, params.get('leverage', 1) - 10)

    def period_constraint(params: Dict) -> float:
        """快週期必須小於慢週期"""
        fast = params.get('fast_period', 10)
        slow = params.get('slow_period', 30)
        return max(0, fast - slow + 5)  # 至少相差 5

    constraints = [leverage_constraint, period_constraint]

    # ===== 4. 建立優化器（含約束） =====
    print("建立多目標優化器...")
    optimizer = MultiObjectiveOptimizer(
        objectives=[
            ('sharpe_ratio', 'maximize'),
            ('max_drawdown', 'minimize'),
            ('sortino_ratio', 'maximize'),
            ('win_rate', 'maximize')
        ],
        n_trials=100,
        seed=42,
        constraints=constraints,
        verbose=True
    )

    # ===== 5. 暖啟動（使用歷史優良解） =====
    print("\n暖啟動：載入歷史優良解...")
    optimizer.warm_start([
        {'fast_period': 10, 'slow_period': 30, 'stop_loss_atr': 2.0, 'leverage': 3},
        {'fast_period': 12, 'slow_period': 26, 'stop_loss_atr': 1.5, 'leverage': 5},
        {'fast_period': 8, 'slow_period': 35, 'stop_loss_atr': 2.5, 'leverage': 2}
    ])

    # ===== 6. 執行優化 =====
    print("\n開始優化...")
    result = optimizer.optimize(
        param_space=param_space,
        evaluate_fn=evaluate_strategy,
        show_progress_bar=True
    )

    # ===== 7. 基本結果 =====
    print("\n" + "=" * 60)
    print("優化結果摘要")
    print("=" * 60)
    print(f"Pareto 前沿解數量: {len(result.pareto_front)}")
    print(f"完成試驗數: {result.n_completed_trials}")
    print(f"失敗試驗數: {result.n_failed_trials}")
    print(f"優化時間: {result.optimization_time:.2f} 秒")

    # ===== 8. 膝點檢測 =====
    print("\n" + "=" * 60)
    print("膝點檢測（最佳平衡解）")
    print("=" * 60)
    knee_solution = result.find_knee_point()
    if knee_solution:
        print(f"膝點參數: {knee_solution.params}")
        for obj in knee_solution.objectives:
            print(f"  {obj.name}: {obj.value:.4f} ({obj.direction})")
    else:
        print("無法找到膝點（Pareto 前沿解太少）")

    # ===== 9. 加權最佳解 =====
    print("\n" + "=" * 60)
    print("加權最佳解（自定義權重）")
    print("=" * 60)
    best_solution = result.get_best_solution(weights={
        'sharpe_ratio': 0.3,
        'max_drawdown': 0.3,
        'sortino_ratio': 0.2,
        'win_rate': 0.2
    })
    if best_solution:
        print(f"最佳解參數: {best_solution.params}")
        for obj in best_solution.objectives:
            print(f"  {obj.name}: {obj.value:.4f} ({obj.direction})")

    # ===== 10. Pareto 前沿篩選 =====
    print("\n" + "=" * 60)
    print("Pareto 前沿篩選")
    print("=" * 60)

    # 多樣性最大化
    diverse_solutions = result.filter_pareto_front('crowding', n_select=5)
    print(f"\n多樣性最大化（crowding）- 選擇 {len(diverse_solutions)} 個解:")
    for i, sol in enumerate(diverse_solutions, 1):
        print(f"  解 {i}: CD={sol.crowding_distance:.4f}, {sol.params}")

    # 膝點附近
    balanced_solutions = result.filter_pareto_front('knee', n_select=5)
    print(f"\n膝點附近（knee）- 選擇 {len(balanced_solutions)} 個解:")
    for i, sol in enumerate(balanced_solutions, 1):
        sharpe = sol.get_objective_value('sharpe_ratio')
        max_dd = sol.get_objective_value('max_drawdown')
        print(f"  解 {i}: Sharpe={sharpe:.3f}, MaxDD={max_dd:.3f}")

    # 極值解
    extreme_solutions = result.filter_pareto_front('extreme', n_select=8)
    print(f"\n極值解（extreme）- 選擇 {len(extreme_solutions)} 個解:")
    for i, sol in enumerate(extreme_solutions, 1):
        print(f"  解 {i}: {sol.params}")

    # ===== 11. 視覺化 =====
    print("\n" + "=" * 60)
    print("視覺化輸出")
    print("=" * 60)

    # 2D Pareto 前沿
    result.plot_pareto_front_2d(
        'sharpe_ratio',
        'max_drawdown',
        save_path='results/pareto_2d.html'
    )
    print("✓ 2D Pareto 前沿: results/pareto_2d.html")

    # 3D Pareto 前沿
    result.plot_pareto_front_3d(
        'sharpe_ratio',
        'max_drawdown',
        'sortino_ratio',
        save_path='results/pareto_3d.html'
    )
    print("✓ 3D Pareto 前沿: results/pareto_3d.html")

    # 平行座標圖
    result.plot_parallel_coordinates(
        save_path='results/parallel_coordinates.html',
        max_params=6
    )
    print("✓ 平行座標圖: results/parallel_coordinates.html")

    # ===== 12. 匯出結果 =====
    df = result.to_dataframe()
    df.to_csv('results/pareto_front.csv', index=False)
    print("✓ Pareto 前沿資料: results/pareto_front.csv")

    print("\n" + "=" * 60)
    print("優化完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
