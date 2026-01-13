#!/usr/bin/env python3
"""
BacktestValidator 使用範例

示範如何使用回測驗證器驗證回測系統的正確性。
"""

import sys
from pathlib import Path

# 加入專案根目錄到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtester import BacktestValidator

def main():
    print("=" * 60)
    print("BacktestValidator 使用範例")
    print("=" * 60)
    print()

    # 建立驗證器
    validator = BacktestValidator(
        tolerance=1e-6,    # 數值比較容差
        random_seed=42     # 隨機種子（確保可重現）
    )

    # ========== 範例 1: 執行所有驗證 ==========
    print("範例 1: 執行所有驗證")
    print("-" * 60)

    report = validator.validate_all()
    print(report.summary())
    print()

    # ========== 範例 2: 只驗證特定層級 ==========
    print("\n範例 2: 只驗證 L1（過程正確性）")
    print("-" * 60)

    l1_report = validator.validate_level("L1")
    print(l1_report.summary())
    print()

    # ========== 範例 3: 執行單一驗證 ==========
    print("\n範例 3: 執行單一驗證測試")
    print("-" * 60)

    result = validator.validate_sharpe_calculation()
    print(result)
    print(f"  執行時間: {result.duration_ms:.2f} ms")
    if result.details:
        print(f"  詳細資訊: {result.details}")
    print()

    # ========== 範例 4: 驗證特定策略 ==========
    print("\n範例 4: 驗證特定策略的訊號一致性")
    print("-" * 60)

    strategies = ["trend_ma_cross", "momentum_rsi", "trend_supertrend"]

    for strategy_name in strategies:
        try:
            result = validator.validate_signal_consistency(strategy_name)
            status = "✅" if result.success else "❌"
            print(f"{status} {strategy_name}: {result.message}")
        except Exception as e:
            print(f"❌ {strategy_name}: 測試失敗 - {e}")

    print()

    # ========== 範例 5: 檢查驗證報告統計 ==========
    print("\n範例 5: 驗證報告統計")
    print("-" * 60)

    print(f"總測試數: {report.total}")
    print(f"通過: {report.passed}")
    print(f"失敗: {report.failed}")
    print(f"通過率: {report.pass_rate:.1%}")
    print()

    # 按層級統計
    l1_tests = [r for r in report.results if r.level == "L1"]
    l2_tests = [r for r in report.results if r.level == "L2"]
    l3_tests = [r for r in report.results if r.level == "L3"]

    print("按層級統計:")
    print(f"  L1（過程正確性）: {sum(r.success for r in l1_tests)}/{len(l1_tests)} 通過")
    print(f"  L2（數值正確性）: {sum(r.success for r in l2_tests)}/{len(l2_tests)} 通過")
    print(f"  L3（統計正確性）: {sum(r.success for r in l3_tests)}/{len(l3_tests)} 通過")
    print()

    print("=" * 60)

if __name__ == '__main__':
    main()
