"""
多目標優化器使用範例

展示如何使用 NSGA-II 進行多目標策略優化。
"""

import sys
from pathlib import Path

# 加入專案根目錄到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from typing import Dict

from src.optimizer.multi_objective import (
    MultiObjectiveOptimizer,
    optimize_multi_objective
)


# ============================================================================
# 範例 1: 簡單的雙目標優化
# ============================================================================

def example_1_basic_optimization():
    """
    範例 1: 基本雙目標優化

    目標：同時最小化兩個相互衝突的目標
    - f1(x) = x^2
    - f2(x) = (x - 2)^2
    """
    print("=" * 60)
    print("範例 1: 基本雙目標優化")
    print("=" * 60)

    # 參數空間
    param_space = {
        'x': {'type': 'float', 'low': -5.0, 'high': 5.0}
    }

    # 評估函數
    def evaluate(params: Dict) -> Dict[str, float]:
        x = params['x']
        return {
            'f1': x**2,           # 最小化 (最優解在 x=0)
            'f2': (x - 2)**2      # 最小化 (最優解在 x=2)
        }

    # 執行優化
    optimizer = MultiObjectiveOptimizer(
        objectives=[('f1', 'minimize'), ('f2', 'minimize')],
        n_trials=100,
        seed=42
    )

    result = optimizer.optimize(
        param_space=param_space,
        evaluate_fn=evaluate
    )

    # 顯示結果
    print(result.summary())
    print()

    # 取得平衡解
    best = result.get_best_solution()
    print(f"最佳平衡解: x={best.params['x']:.4f}")
    print(f"  f1={best.get_objective_value('f1'):.4f}")
    print(f"  f2={best.get_objective_value('f2'):.4f}")
    print()


# ============================================================================
# 範例 2: 交易策略多目標優化
# ============================================================================

def example_2_trading_strategy():
    """
    範例 2: 交易策略多目標優化

    目標：同時優化多個交易指標
    - Sharpe Ratio (最大化)
    - Max Drawdown (最小化)
    - Win Rate (最大化)
    """
    print("=" * 60)
    print("範例 2: 交易策略多目標優化")
    print("=" * 60)

    # 參數空間
    param_space = {
        'fast_period': {'type': 'int', 'low': 5, 'high': 20},
        'slow_period': {'type': 'int', 'low': 20, 'high': 50},
        'stop_loss': {'type': 'float', 'low': 0.01, 'high': 0.05, 'step': 0.005},
        'take_profit': {'type': 'float', 'low': 0.02, 'high': 0.1, 'step': 0.01}
    }

    # 模擬評估函數（實際應使用回測引擎）
    def evaluate(params: Dict) -> Dict[str, float]:
        fast = params['fast_period']
        slow = params['slow_period']
        sl = params['stop_loss']
        tp = params['take_profit']

        # 虛擬策略評估（實際應該執行回測）
        # 這裡只是示範多目標優化的概念

        # Sharpe Ratio: 期望 slow-fast 接近 20，TP/SL 比例適中
        sharpe = 2.0 - abs(slow - fast - 20) / 20.0 + (tp / sl) / 2.0
        sharpe += np.random.normal(0, 0.1)  # 加入噪音模擬市場變化

        # Max Drawdown: 與止損大小相關
        max_dd = sl * 5.0 + np.random.uniform(0, 0.05)

        # Win Rate: 與止盈止損比例相關
        win_rate = 0.5 + (tp - sl) / (tp + sl) * 0.3
        win_rate += np.random.normal(0, 0.05)
        win_rate = np.clip(win_rate, 0.3, 0.7)

        return {
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate
        }

    # 執行優化
    result = optimize_multi_objective(
        param_space=param_space,
        evaluate_fn=evaluate,
        objectives=[
            ('sharpe_ratio', 'maximize'),
            ('max_drawdown', 'minimize'),
            ('win_rate', 'maximize')
        ],
        n_trials=200,
        seed=42,
        verbose=True
    )

    # 顯示 Pareto 前緣
    print("\nPareto 前緣（前 5 個解）:")
    print("-" * 60)
    for i, solution in enumerate(result.pareto_front[:5], 1):
        print(f"\n解 {i}:")
        print(f"  參數: {solution.params}")
        print(f"  Sharpe Ratio: {solution.get_objective_value('sharpe_ratio'):.4f}")
        print(f"  Max Drawdown: {solution.get_objective_value('max_drawdown'):.4f}")
        print(f"  Win Rate: {solution.get_objective_value('win_rate'):.4f}")

    # 使用不同權重選擇解
    print("\n" + "=" * 60)
    print("不同偏好的最佳解:")
    print("=" * 60)

    # 偏重 Sharpe Ratio
    best_sharpe = result.get_best_solution(
        weights={'sharpe_ratio': 0.6, 'max_drawdown': 0.2, 'win_rate': 0.2}
    )
    print("\n1. 偏重 Sharpe Ratio:")
    print(f"   參數: {best_sharpe.params}")
    print(f"   Sharpe: {best_sharpe.get_objective_value('sharpe_ratio'):.4f}")

    # 偏重風控（低回撤）
    best_dd = result.get_best_solution(
        weights={'sharpe_ratio': 0.2, 'max_drawdown': 0.6, 'win_rate': 0.2}
    )
    print("\n2. 偏重風控（低回撤）:")
    print(f"   參數: {best_dd.params}")
    print(f"   Max DD: {best_dd.get_objective_value('max_drawdown'):.4f}")

    # 偏重勝率
    best_wr = result.get_best_solution(
        weights={'sharpe_ratio': 0.2, 'max_drawdown': 0.2, 'win_rate': 0.6}
    )
    print("\n3. 偏重勝率:")
    print(f"   參數: {best_wr.params}")
    print(f"   Win Rate: {best_wr.get_objective_value('win_rate'):.4f}")

    # 平衡解
    best_balanced = result.get_best_solution()
    print("\n4. 平衡解（均等權重）:")
    print(f"   參數: {best_balanced.params}")
    print(f"   Sharpe: {best_balanced.get_objective_value('sharpe_ratio'):.4f}")
    print(f"   Max DD: {best_balanced.get_objective_value('max_drawdown'):.4f}")
    print(f"   Win Rate: {best_balanced.get_objective_value('win_rate'):.4f}")
    print()

    # 轉為 DataFrame
    df = result.to_dataframe()
    print("\nPareto 前緣 DataFrame:")
    print(df.head(10))
    print()

    # 儲存結果（可選）
    try:
        df.to_csv('pareto_front.csv', index=False)
        print("Pareto 前緣已儲存至 pareto_front.csv")
    except Exception as e:
        print(f"儲存失敗: {e}")


# ============================================================================
# 範例 3: 整合回測引擎
# ============================================================================

def example_3_with_backtest_engine():
    """
    範例 3: 整合回測引擎的完整範例

    展示如何將多目標優化器與回測引擎結合使用。
    """
    print("=" * 60)
    print("範例 3: 整合回測引擎")
    print("=" * 60)
    print("\n註: 此範例需要實際的市場資料和回測引擎")
    print("     這裡展示整合架構\n")

    # 假設我們有回測引擎和策略
    # from src.backtester.engine import BacktestEngine
    # from src.strategies.ma_cross import MovingAverageCross

    # 參數空間
    param_space = {
        'fast_period': {'type': 'int', 'low': 5, 'high': 20},
        'slow_period': {'type': 'int', 'low': 20, 'high': 50},
        'stop_loss_atr': {'type': 'float', 'low': 1.0, 'high': 3.0, 'step': 0.5}
    }

    # 評估函數（整合回測引擎）
    def evaluate_with_backtest(params: Dict) -> Dict[str, float]:
        """
        使用回測引擎評估策略參數

        Args:
            params: 策略參數

        Returns:
            目標指標字典
        """
        # engine = BacktestEngine(...)
        # strategy = MovingAverageCross()
        # result = engine.run(strategy=strategy, params=params, data=market_data)

        # 模擬回測結果
        result_sharpe = 1.5 + np.random.normal(0, 0.3)
        result_max_dd = 0.15 + np.random.uniform(0, 0.1)
        result_sortino = 2.0 + np.random.normal(0, 0.4)
        result_calmar = 5.0 + np.random.normal(0, 1.0)

        return {
            'sharpe_ratio': result_sharpe,
            'max_drawdown': result_max_dd,
            'sortino_ratio': result_sortino,
            'calmar_ratio': result_calmar
        }

    # 執行多目標優化
    optimizer = MultiObjectiveOptimizer(
        objectives=[
            ('sharpe_ratio', 'maximize'),
            ('max_drawdown', 'minimize'),
            ('sortino_ratio', 'maximize'),
            ('calmar_ratio', 'maximize')
        ],
        n_trials=50,
        seed=42,
        verbose=True
    )

    result = optimizer.optimize(
        param_space=param_space,
        evaluate_fn=evaluate_with_backtest
    )

    print("\n優化完成!")
    print(f"Pareto 前緣解數量: {len(result.pareto_front)}")
    print(f"優化時間: {result.optimization_time:.2f} 秒")

    # 取得最佳解
    best = result.get_best_solution(
        weights={
            'sharpe_ratio': 0.3,
            'max_drawdown': 0.3,
            'sortino_ratio': 0.2,
            'calmar_ratio': 0.2
        }
    )

    print(f"\n最佳解參數: {best.params}")
    print("目標值:")
    for obj in best.objectives:
        print(f"  {obj.name}: {obj.value:.4f}")


# ============================================================================
# 主程式
# ============================================================================

if __name__ == '__main__':
    # 執行範例
    example_1_basic_optimization()
    print("\n" * 2)

    example_2_trading_strategy()
    print("\n" * 2)

    example_3_with_backtest_engine()
