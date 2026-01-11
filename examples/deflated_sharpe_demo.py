"""
Deflated Sharpe Ratio 使用範例

示範如何使用 Deflated Sharpe Ratio 校正多重檢定偏差。
"""

import sys
from pathlib import Path

# 加入專案根目錄到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.validator.sharpe_correction import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    minimum_backtest_length,
    print_deflated_sharpe_report,
    print_pbo_report,
)
from src.validator.statistical_tests import calculate_sharpe


def main():
    print("=" * 80)
    print("Deflated Sharpe Ratio 範例".center(80))
    print("=" * 80)

    # ========== 範例 1: 基本使用 ==========
    print("\n" + "=" * 80)
    print("範例 1: Deflated Sharpe Ratio 基本計算")
    print("=" * 80)

    # 模擬一年的交易資料（Sharpe ≈ 2.0）
    np.random.seed(42)
    returns = np.random.normal(0.00126, 0.01, 252)  # 日收益

    # 計算原始 Sharpe
    sharpe = calculate_sharpe(returns)
    print(f"\n原始 Sharpe Ratio: {sharpe:.2f}")

    # 假設測試了 100 個策略
    n_trials = 100

    # 計算 Deflated Sharpe
    dsr_result = deflated_sharpe_ratio(
        sharpe=sharpe,
        n_trials=n_trials,
        returns=returns,
        t_years=1.0
    )

    print_deflated_sharpe_report(dsr_result)

    # ========== 範例 2: PBO 檢測 ==========
    print("\n" + "=" * 80)
    print("範例 2: Probability of Backtest Overfitting (PBO)")
    print("=" * 80)

    # 模擬 20 個策略的 In-Sample 和 Out-of-Sample Sharpe
    np.random.seed(123)
    n_strategies = 20

    # In-Sample: 優化期間（2022-2023）
    is_sharpe = np.random.uniform(1.0, 3.0, n_strategies)

    # Out-of-Sample: 測試期間（2024）
    # 情境 A: 過擬合（OOS 表現差）
    oos_sharpe_overfit = is_sharpe * 0.4 + np.random.normal(0, 0.2, n_strategies)

    print("\n[情境 A: 過擬合]")
    pbo_overfit = probability_of_backtest_overfitting(
        is_sharpe, oos_sharpe_overfit, n_trials=n_strategies
    )
    print_pbo_report(pbo_overfit)

    # 情境 B: 穩健策略（OOS 表現接近 IS）
    oos_sharpe_robust = is_sharpe * 0.9 + np.random.normal(0, 0.1, n_strategies)

    print("\n[情境 B: 穩健策略]")
    pbo_robust = probability_of_backtest_overfitting(
        is_sharpe, oos_sharpe_robust, n_trials=n_strategies
    )
    print_pbo_report(pbo_robust)

    # ========== 範例 3: 最小回測長度 ==========
    print("\n" + "=" * 80)
    print("範例 3: 計算達到統計顯著的最小回測長度")
    print("=" * 80)

    scenarios = [
        ("高 Sharpe (3.0), 少量嘗試 (10)", 3.0, 10),
        ("中 Sharpe (2.0), 中等嘗試 (100)", 2.0, 100),
        ("低 Sharpe (1.0), 大量嘗試 (500)", 1.0, 500),
    ]

    for desc, target_sharpe, n_trials in scenarios:
        result = minimum_backtest_length(
            target_sharpe=target_sharpe,
            n_trials=n_trials,
            confidence=0.95
        )

        print(f"\n[{desc}]")
        print(f"  最小回測年數: {result.min_years:.2f} 年")
        print(f"  最小觀察次數: {result.min_observations} 天")
        print(f"  約 {result.min_observations / 252:.1f} 年的日資料")

    # ========== 範例 4: 比較不同嘗試次數 ==========
    print("\n" + "=" * 80)
    print("範例 4: 多重檢定偏差的影響")
    print("=" * 80)

    sharpe_observed = 2.5
    print(f"\n觀察到的 Sharpe Ratio: {sharpe_observed}")
    print(f"\n{'嘗試次數':<15} {'預期最大 SR':<15} {'DSR':<15} {'顯著性'}")
    print("-" * 60)

    for n in [1, 10, 100, 500, 1000]:
        result = deflated_sharpe_ratio(
            sharpe=sharpe_observed,
            n_trials=n,
            variance=0.004,
            t_years=1.0
        )

        significance = "✓" if result.is_significant else "✗"
        print(f"{n:<15} {result.expected_max_sharpe:<15.3f} "
              f"{result.deflated_sharpe:<15.3f} {significance}")

    print("\n說明：")
    print("  - 嘗試越多，預期最大 Sharpe 越高（多重檢定偏差）")
    print("  - DSR 考慮了這個偏差，調整顯著性判斷")
    print("  - 當 DSR > 1.96，策略具有統計顯著性（95% 信賴水準）")

    # ========== 範例 5: 實際應用建議 ==========
    print("\n" + "=" * 80)
    print("實際應用建議")
    print("=" * 80)

    print("""
1. 記錄所有嘗試次數
   - 包含失敗的策略
   - 包含參數調整次數
   - 建議使用 learning/experiments.json 記錄

2. 使用 Deflated Sharpe 評估
   - 將 n_trials 設為實際嘗試次數
   - 檢查 DSR > 1.96 (95% 信賴)
   - 參考 p-value < 0.05

3. 使用 PBO 驗證
   - 將資料分為 IS (70%) 和 OOS (30%)
   - 計算所有策略在兩期間的 Sharpe
   - PBO < 0.3 為低風險

4. 確保足夠的回測長度
   - 使用 minimum_backtest_length 計算
   - 目標：至少 2-3 年資料（永續合約）
   - 包含多種市場環境（牛市、熊市、震盪）

5. 持續監控
   - 定期更新 OOS 驗證
   - Walk-Forward 分析
   - 實盤交易後比對實際表現
    """)

    print("\n" + "=" * 80)
    print("範例執行完畢")
    print("=" * 80)


if __name__ == "__main__":
    main()
