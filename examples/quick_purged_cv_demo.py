"""
Combinatorial Purged CV - 快速示範

展示核心功能的最小範例。
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 添加專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validator.walk_forward import combinatorial_purged_cv


def demo_strategy(data, params):
    """簡單的動量策略"""
    threshold = params.get('threshold', 0.001)

    # 生成信號
    signals = (data['returns'] > threshold).astype(int).shift(1)
    strategy_returns = signals * data['returns']
    strategy_returns = strategy_returns.dropna()

    # 計算績效
    total_return = strategy_returns.sum()
    sharpe = (
        strategy_returns.mean() / (strategy_returns.std() + 1e-8)
        * np.sqrt(252 * 24)
    )

    return {
        'return': total_return,
        'sharpe': sharpe,
        'trades': int(signals.diff().abs().sum())
    }


def main():
    print("=" * 70)
    print("Combinatorial Purged CV - 快速示範")
    print("=" * 70)
    print()

    # 1. 生成測試資料
    print("1. 生成測試資料...")
    dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='1h')
    np.random.seed(42)

    returns = pd.Series(
        np.random.normal(0.0001, 0.01, len(dates)),
        index=dates,
        name='returns'
    )
    print(f"   資料期間: {returns.index[0]} ~ {returns.index[-1]}")
    print(f"   資料筆數: {len(returns)}")
    print()

    # 2. 執行 Combinatorial Purged CV
    print("2. 執行 Combinatorial Purged CV...")
    print("   配置: n_splits=5, n_test_groups=2, purge_gap=24, embargo=1%")
    print()

    result = combinatorial_purged_cv(
        returns=returns,
        strategy_func=demo_strategy,
        n_splits=5,
        n_test_groups=2,
        purge_gap=24,
        embargo_pct=0.01,
        param_grid={'threshold': [0.0005, 0.001, 0.002]}
    )

    # 3. 顯示結果
    print("3. 驗證結果")
    print("=" * 70)
    print(f"組合數量: {result.n_combinations}")
    print()

    print(f"平均訓練報酬: {result.mean_train_return:>10.2%}")
    print(f"平均測試報酬: {result.mean_test_return:>10.2%}")
    print(f"測試報酬標準差: {result.test_return_std:>10.2%}")
    print()

    print(f"平均訓練夏普: {result.mean_train_sharpe:>10.2f}")
    print(f"平均測試夏普: {result.mean_test_sharpe:>10.2f}")
    print(f"測試夏普標準差: {result.test_sharpe_std:>10.2f}")
    print()

    # 4. 穩健性分析
    print("4. 穩健性分析")
    print("-" * 70)

    print(f"過擬合比例: {result.overfitting_ratio:>10.2%}")
    if result.overfitting_ratio >= 0.8:
        print("   評估: ✓ 穩健")
    elif result.overfitting_ratio >= 0.5:
        print("   評估: ⚠ 輕微過擬合")
    else:
        print("   評估: ✗ 嚴重過擬合")
    print()

    print(f"測試集勝率: {result.consistency:>10.2%}")
    if result.consistency >= 0.6:
        print("   評估: ✓ 穩健")
    elif result.consistency >= 0.4:
        print("   評估: ⚠ 中等")
    else:
        print("   評估: ✗ 不穩定")
    print()

    # 5. 每個 fold 詳情
    print("5. Fold 詳情")
    print("-" * 70)
    print(f"{'Fold':<6} {'訓練報酬':<12} {'測試報酬':<12} {'最佳參數':<20}")
    print("-" * 70)

    for fold in result.folds[:5]:  # 只顯示前 5 個
        params_str = f"threshold={fold.best_params.get('threshold', 'N/A')}"
        print(f"{fold.fold_id:<6} {fold.train_return:<12.2%} "
              f"{fold.test_return:<12.2%} {params_str:<20}")

    if len(result.folds) > 5:
        print(f"... 還有 {len(result.folds) - 5} 個 fold")

    print()
    print("=" * 70)
    print("示範完成！")
    print()
    print("更多範例請執行: python examples/purged_cv_example.py")
    print()


if __name__ == '__main__':
    main()
