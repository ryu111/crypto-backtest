"""
Combinatorial Purged Cross-Validation 使用範例

展示如何使用 PurgedKFold 和 CombinatorialPurgedCV 來驗證策略穩健性。
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# 添加專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validator.walk_forward import (
    PurgedKFold,
    CombinatorialPurgedCV,
    combinatorial_purged_cv
)


def generate_sample_data():
    """生成範例市場資料"""
    print("生成範例資料...")

    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1h')
    np.random.seed(42)

    # 模擬價格資料（帶趨勢 + 噪音）
    trend = np.linspace(40000, 45000, len(dates))
    noise = np.random.normal(0, 200, len(dates))
    close = trend + noise

    # 計算報酬率
    returns = pd.Series(close).pct_change()

    data = pd.DataFrame({
        'close': close,
        'returns': returns,
        'volume': np.random.uniform(100, 200, len(dates))
    }, index=dates)

    print(f"  資料期間: {data.index[0]} ~ {data.index[-1]}")
    print(f"  資料筆數: {len(data)}")
    print()

    return data


def simple_momentum_strategy(data: pd.DataFrame, params: dict) -> dict:
    """
    簡單動量策略

    Args:
        data: 包含 'returns' 欄位的 DataFrame
        params: {'threshold': float, 'lookback': int}

    Returns:
        {'return': float, 'sharpe': float, 'trades': int}
    """
    threshold = params.get('threshold', 0.001)
    lookback = params.get('lookback', 24)

    # 計算動量信號（過去 N 小時累積報酬）
    momentum = data['returns'].rolling(lookback).sum()

    # 生成交易信號
    signals = (momentum > threshold).astype(int)
    signals = signals.shift(1)  # 避免 look-ahead bias

    # 計算策略報酬
    strategy_returns = signals * data['returns']
    strategy_returns = strategy_returns.dropna()

    if len(strategy_returns) == 0:
        return {'return': 0.0, 'sharpe': 0.0, 'trades': 0}

    # 計算績效指標
    total_return = strategy_returns.sum()
    sharpe_ratio = (
        strategy_returns.mean() / (strategy_returns.std() + 1e-8)
        * np.sqrt(252 * 24)  # 年化（小時資料）
    )
    num_trades = signals.diff().abs().sum()

    return {
        'return': total_return,
        'sharpe': sharpe_ratio,
        'trades': int(num_trades)
    }


def example_1_purged_kfold():
    """範例 1: PurgedKFold 基礎使用"""
    print("=" * 70)
    print("範例 1: PurgedKFold 基礎使用")
    print("=" * 70)

    data = generate_sample_data()

    # 建立 PurgedKFold
    kfold = PurgedKFold(
        n_splits=5,
        purge_gap=24,  # 24 小時 purge gap
        embargo_pct=0.01  # 1% embargo
    )

    print(f"配置:")
    print(f"  K-Fold: {kfold.n_splits}")
    print(f"  Purge Gap: {kfold.purge_gap}")
    print(f"  Embargo: {kfold.embargo_pct:.2%}")
    print()

    # 獲取 splits
    splits = kfold.split(data)
    print(f"產生 {len(splits)} 個 folds\n")

    # 檢視每個 fold 的資訊
    for i in range(min(3, len(splits))):
        fold_info = kfold.get_fold_info(data, fold_id=i)

        print(f"Fold {i + 1}:")
        print(f"  訓練期間: {fold_info['train_start']} ~ {fold_info['train_end']}")
        print(f"  測試期間: {fold_info['test_start']} ~ {fold_info['test_end']}")
        print(f"  訓練樣本數: {fold_info['train_size']}")
        print(f"  測試樣本數: {fold_info['test_size']}")

        if fold_info['purge_start']:
            print(f"  Purge 期間: {fold_info['purge_start']} ~ {fold_info['purge_end']}")

        if fold_info['embargo_start']:
            print(f"  Embargo 期間: {fold_info['embargo_start']} ~ {fold_info['embargo_end']}")

        print()

    print()


def example_2_combinatorial_cv():
    """範例 2: Combinatorial Purged CV"""
    print("=" * 70)
    print("範例 2: Combinatorial Purged CV")
    print("=" * 70)

    data = generate_sample_data()

    # 建立 CombinatorialPurgedCV
    cv = CombinatorialPurgedCV(
        n_splits=5,
        n_test_groups=2,
        purge_gap=24,
        embargo_pct=0.01
    )

    print(f"配置:")
    print(f"  K-Fold: {cv.n_splits}")
    print(f"  測試組數: {cv.n_test_groups}")
    print(f"  Purge Gap: {cv.purge_gap}")
    print(f"  Embargo: {cv.embargo_pct:.2%}")
    print()

    # 定義參數網格
    param_grid = {
        'threshold': [0.001, 0.002, 0.005],
        'lookback': [12, 24, 48]
    }

    print(f"參數網格:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")
    print()

    # 執行驗證
    result = cv.validate(
        data=data,
        strategy_func=simple_momentum_strategy,
        param_grid=param_grid,
        optimize_metric='sharpe',
        verbose=True
    )

    # 顯示結果
    print("\n" + result.summary())
    print()


def example_3_convenience_function():
    """範例 3: 使用便捷函數"""
    print("=" * 70)
    print("範例 3: 使用便捷函數")
    print("=" * 70)

    data = generate_sample_data()

    # 使用便捷函數
    result = combinatorial_purged_cv(
        returns=data,
        strategy_func=simple_momentum_strategy,
        n_splits=4,
        n_test_groups=1,
        purge_gap=12,
        embargo_pct=0.02,
        param_grid={'threshold': [0.001, 0.005]}
    )

    # 顯示結果
    print(result.summary())
    print()

    # 詳細分析
    print("詳細分析:")
    print("-" * 70)
    print(f"過擬合比例: {result.overfitting_ratio:.2%}")
    print(f"  > 100%: 測試優於訓練（unlikely but possible）")
    print(f"  80-100%: 穩健")
    print(f"  50-80%: 輕微過擬合")
    print(f"  < 50%: 嚴重過擬合")
    print()

    print(f"測試集勝率: {result.consistency:.2%}")
    print(f"  > 60%: 穩健")
    print(f"  40-60%: 中等")
    print(f"  < 40%: 不穩定")
    print()

    print(f"測試報酬標準差: {result.test_return_std:.2%}")
    print(f"  低標準差 → 穩定性高")
    print(f"  高標準差 → 結果波動大")
    print()


def example_4_manual_fold_usage():
    """範例 4: 手動使用 fold splits"""
    print("=" * 70)
    print("範例 4: 手動使用 fold splits")
    print("=" * 70)

    data = generate_sample_data()

    kfold = PurgedKFold(n_splits=3, purge_gap=24, embargo_pct=0.01)
    splits = kfold.split(data)

    fold_results = []

    for fold_id, (train_idx, test_idx) in enumerate(splits, 1):
        # 提取訓練和測試資料
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]

        # 在訓練集上測試策略
        train_metrics = simple_momentum_strategy(
            train_data,
            params={'threshold': 0.002, 'lookback': 24}
        )

        # 在測試集上測試策略
        test_metrics = simple_momentum_strategy(
            test_data,
            params={'threshold': 0.002, 'lookback': 24}
        )

        fold_results.append({
            'fold': fold_id,
            'train_return': train_metrics['return'],
            'test_return': test_metrics['return'],
            'train_sharpe': train_metrics['sharpe'],
            'test_sharpe': test_metrics['sharpe']
        })

        print(f"Fold {fold_id}:")
        print(f"  訓練: 報酬={train_metrics['return']:.2%}, 夏普={train_metrics['sharpe']:.2f}")
        print(f"  測試: 報酬={test_metrics['return']:.2%}, 夏普={test_metrics['sharpe']:.2f}")
        print()

    # 計算整體統計
    avg_train_return = np.mean([r['train_return'] for r in fold_results])
    avg_test_return = np.mean([r['test_return'] for r in fold_results])
    efficiency = avg_test_return / avg_train_return if avg_train_return != 0 else 0

    print("整體統計:")
    print(f"  平均訓練報酬: {avg_train_return:.2%}")
    print(f"  平均測試報酬: {avg_test_return:.2%}")
    print(f"  效率: {efficiency:.2%}")
    print()


def example_5_compare_configurations():
    """範例 5: 比較不同配置"""
    print("=" * 70)
    print("範例 5: 比較不同配置")
    print("=" * 70)

    data = generate_sample_data()

    configurations = [
        {
            'name': '無 Purge/Embargo',
            'n_splits': 5,
            'n_test_groups': 2,
            'purge_gap': 0,
            'embargo_pct': 0.0
        },
        {
            'name': '有 Purge (24h)',
            'n_splits': 5,
            'n_test_groups': 2,
            'purge_gap': 24,
            'embargo_pct': 0.0
        },
        {
            'name': '有 Purge + Embargo',
            'n_splits': 5,
            'n_test_groups': 2,
            'purge_gap': 24,
            'embargo_pct': 0.01
        }
    ]

    results_comparison = []

    for config in configurations:
        print(f"\n測試配置: {config['name']}")
        print("-" * 70)

        cv = CombinatorialPurgedCV(
            n_splits=config['n_splits'],
            n_test_groups=config['n_test_groups'],
            purge_gap=config['purge_gap'],
            embargo_pct=config['embargo_pct']
        )

        result = cv.validate(
            data=data,
            strategy_func=simple_momentum_strategy,
            param_grid={'threshold': [0.001, 0.005]},
            verbose=False
        )

        results_comparison.append({
            'name': config['name'],
            'mean_test_return': result.mean_test_return,
            'overfitting_ratio': result.overfitting_ratio,
            'consistency': result.consistency
        })

        print(f"測試報酬: {result.mean_test_return:.2%}")
        print(f"過擬合比例: {result.overfitting_ratio:.2%}")
        print(f"勝率: {result.consistency:.2%}")

    print("\n" + "=" * 70)
    print("配置比較摘要")
    print("=" * 70)
    print(f"{'配置':<20} {'測試報酬':<15} {'過擬合比例':<15} {'勝率':<10}")
    print("-" * 70)

    for r in results_comparison:
        print(f"{r['name']:<20} {r['mean_test_return']:<15.2%} "
              f"{r['overfitting_ratio']:<15.2%} {r['consistency']:<10.2%}")

    print()


def main():
    """執行所有範例"""
    examples = [
        ("PurgedKFold 基礎使用", example_1_purged_kfold),
        ("Combinatorial Purged CV", example_2_combinatorial_cv),
        ("使用便捷函數", example_3_convenience_function),
        ("手動使用 fold splits", example_4_manual_fold_usage),
        ("比較不同配置", example_5_compare_configurations)
    ]

    print("\n" + "=" * 70)
    print("Combinatorial Purged Cross-Validation 範例")
    print("=" * 70)
    print()

    for i, (name, func) in enumerate(examples, 1):
        print(f"\n執行範例 {i}: {name}\n")
        try:
            func()
        except Exception as e:
            print(f"錯誤: {e}")
            import traceback
            traceback.print_exc()
        print("\n" + "-" * 70 + "\n")

    print("所有範例執行完成！")


if __name__ == '__main__':
    main()
