"""
統計檢定模組使用範例

展示如何使用 Bootstrap Test 和 Permutation Test 驗證策略績效的顯著性。
"""

import numpy as np
from src.validator.statistical_tests import (
    bootstrap_sharpe,
    permutation_test,
    block_bootstrap,
    run_statistical_tests,
    print_test_report,
)


def main():
    """執行統計檢定範例"""

    print("=" * 80)
    print("統計檢定模組使用範例".center(80))
    print("=" * 80)

    # ========== 模擬策略收益 ==========
    print("\n[1] 生成模擬策略收益...")

    np.random.seed(42)

    # 策略 A: 正收益策略（Sharpe ~ 2.0）
    returns_a = np.random.randn(252) * 0.01 + 0.002  # 日收益，平均 +0.2%
    print(f"策略 A: 平均日收益 = {np.mean(returns_a):.4f}, 標準差 = {np.std(returns_a):.4f}")

    # 策略 B: 隨機遊走（Sharpe ~ 0）
    returns_b = np.random.randn(252) * 0.01
    print(f"策略 B: 平均日收益 = {np.mean(returns_b):.4f}, 標準差 = {np.std(returns_b):.4f}")

    # ========== Bootstrap Test ==========
    print("\n" + "=" * 80)
    print("[2] Bootstrap Test - 計算 Sharpe Ratio 信賴區間")
    print("=" * 80)

    result_bootstrap = bootstrap_sharpe(
        returns_a,
        n_bootstrap=10000,
        confidence=0.95,
        n_jobs=-1  # 使用所有 CPU 核心
    )

    print(f"\nBootstrap 結果（策略 A）:")
    print(f"  Sharpe Ratio 估計: {result_bootstrap.sharpe_mean:.3f}")
    print(f"  95% 信賴區間: ({result_bootstrap.ci_lower:.3f}, {result_bootstrap.ci_upper:.3f})")
    print(f"  Bootstrap 標準誤: {result_bootstrap.sharpe_std:.3f}")
    print(f"  p-value (H0: Sharpe ≤ 0): {result_bootstrap.p_value:.4f}")

    if result_bootstrap.p_value < 0.05:
        print("  結論: Sharpe 顯著大於 0 ✓")
    else:
        print("  結論: Sharpe 不顯著")

    # ========== Permutation Test ==========
    print("\n" + "=" * 80)
    print("[3] Permutation Test - 檢定策略績效顯著性")
    print("=" * 80)

    result_perm = permutation_test(
        returns_a,
        n_permutations=10000,
        n_jobs=-1
    )

    print(f"\nPermutation Test 結果（策略 A）:")
    print(f"  實際 Sharpe: {result_perm.actual_sharpe:.3f}")
    print(f"  虛無假設分布平均: {result_perm.null_mean:.3f}")
    print(f"  虛無假設分布標準差: {result_perm.null_std:.3f}")
    print(f"  p-value (H1: Sharpe > 隨機): {result_perm.p_value:.4f}")
    print(f"  顯著性 (α=0.05): {'是 ✓' if result_perm.is_significant else '否 ✗'}")

    # ========== Block Bootstrap ==========
    print("\n" + "=" * 80)
    print("[4] Block Bootstrap - 保留時間序列相關性")
    print("=" * 80)

    # 生成有自相關的收益序列
    np.random.seed(42)
    autocorr_returns = np.zeros(252)
    autocorr_returns[0] = np.random.randn() * 0.01

    for i in range(1, 252):
        autocorr_returns[i] = 0.3 * autocorr_returns[i-1] + np.random.randn() * 0.01 + 0.001

    result_block = block_bootstrap(
        autocorr_returns,
        block_size=20,  # 約 1 個月
        n_bootstrap=10000,
        confidence=0.95
    )

    print(f"\nBlock Bootstrap 結果（有自相關的策略）:")
    print(f"  Sharpe Ratio 估計: {result_block.sharpe_mean:.3f}")
    print(f"  95% 信賴區間: ({result_block.ci_lower:.3f}, {result_block.ci_upper:.3f})")
    print(f"  區塊大小: 20 天")

    # ========== 完整統計檢定 ==========
    print("\n" + "=" * 80)
    print("[5] 完整統計檢定報告")
    print("=" * 80)

    # 策略 A
    print("\n[策略 A - 正收益策略]")
    report_a = run_statistical_tests(
        returns_a,
        n_bootstrap=10000,
        n_permutations=10000,
        confidence=0.95,
        n_jobs=-1
    )
    print_test_report(report_a)

    # 策略 B
    print("\n[策略 B - 隨機遊走]")
    report_b = run_statistical_tests(
        returns_b,
        n_bootstrap=10000,
        n_permutations=10000,
        confidence=0.95,
        n_jobs=-1
    )
    print_test_report(report_b)

    # ========== 結論 ==========
    print("\n" + "=" * 80)
    print("統計檢定結論".center(80))
    print("=" * 80)

    print(f"\n策略 A:")
    print(f"  - 統計顯著性: {'通過 ✓' if report_a.is_statistically_significant else '未通過 ✗'}")
    print(f"  - Bootstrap Sharpe: {report_a.bootstrap_sharpe:.3f} {report_a.bootstrap_ci}")
    print(f"  - Permutation p-value: {report_a.permutation_p_value:.4f}")

    print(f"\n策略 B:")
    print(f"  - 統計顯著性: {'通過 ✓' if report_b.is_statistically_significant else '未通過 ✗'}")
    print(f"  - Bootstrap Sharpe: {report_b.bootstrap_sharpe:.3f} {report_b.bootstrap_ci}")
    print(f"  - Permutation p-value: {report_b.permutation_p_value:.4f}")

    print("\n" + "=" * 80)
    print("範例執行完畢".center(80))
    print("=" * 80)


if __name__ == "__main__":
    main()
