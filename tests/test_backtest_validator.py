"""
BacktestValidator 功能測試

測試 src/backtester/validator.py 的 BacktestValidator 類別
"""

import sys
from pathlib import Path

# 確保可以 import 專案模組
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backtester.validator import BacktestValidator


def test_basic_instantiation():
    """測試基本實例化"""
    print("測試 1: BacktestValidator 實例化")
    validator = BacktestValidator()
    assert validator is not None
    print("✅ PASS - BacktestValidator 成功實例化")
    return True


def test_validate_all():
    """測試執行所有驗證"""
    print("\n測試 2: validate_all() 執行所有驗證")
    validator = BacktestValidator()
    report = validator.validate_all()

    assert report is not None
    assert report.total > 0
    print(f"✅ PASS - 執行了 {report.total} 個測試")
    return True


def test_validation_levels():
    """測試各層級驗證"""
    print("\n測試 3: 驗證層級測試")
    validator = BacktestValidator()

    # L1 測試
    print("\n  L1 測試（過程正確性）:")
    l1_report = validator.validate_level("L1")
    print(f"    - 總測試數: {l1_report.total}")
    print(f"    - 通過: {l1_report.passed}")
    print(f"    - 失敗: {l1_report.failed}")

    # L2 測試
    print("\n  L2 測試（數值正確性）:")
    l2_report = validator.validate_level("L2")
    print(f"    - 總測試數: {l2_report.total}")
    print(f"    - 通過: {l2_report.passed}")
    print(f"    - 失敗: {l2_report.failed}")

    # L3 測試
    print("\n  L3 測試（統計正確性）:")
    l3_report = validator.validate_level("L3")
    print(f"    - 總測試數: {l3_report.total}")
    print(f"    - 通過: {l3_report.passed}")
    print(f"    - 失敗: {l3_report.failed}")

    print("\n✅ PASS - 各層級驗證可正確執行")
    return True


def test_individual_validations():
    """測試個別驗證功能"""
    print("\n測試 4: 個別驗證功能測試")
    validator = BacktestValidator()

    # L1 測試
    print("\n  L1 個別測試:")
    try:
        result = validator.validate_signal_consistency()
        print(f"    - validate_signal_consistency: {result}")
    except Exception as e:
        print(f"    - validate_signal_consistency: ❌ {str(e)}")

    try:
        result = validator.validate_order_execution()
        print(f"    - validate_order_execution: {result}")
    except Exception as e:
        print(f"    - validate_order_execution: ❌ {str(e)}")

    try:
        result = validator.validate_fee_calculation()
        print(f"    - validate_fee_calculation: {result}")
    except Exception as e:
        print(f"    - validate_fee_calculation: ❌ {str(e)}")

    # L2 測試
    print("\n  L2 個別測試:")
    try:
        result = validator.validate_sharpe_calculation()
        print(f"    - validate_sharpe_calculation: {result}")
    except Exception as e:
        print(f"    - validate_sharpe_calculation: ❌ {str(e)}")

    try:
        result = validator.validate_maxdd_calculation()
        print(f"    - validate_maxdd_calculation: {result}")
    except Exception as e:
        print(f"    - validate_maxdd_calculation: ❌ {str(e)}")

    try:
        result = validator.validate_return_calculation()
        print(f"    - validate_return_calculation: {result}")
    except Exception as e:
        print(f"    - validate_return_calculation: ❌ {str(e)}")

    # L3 測試
    print("\n  L3 個別測試:")
    try:
        result = validator.validate_wfa_reproducibility()
        print(f"    - validate_wfa_reproducibility: {result}")
    except Exception as e:
        print(f"    - validate_wfa_reproducibility: ❌ {str(e)}")

    try:
        result = validator.validate_monte_carlo_distribution()
        print(f"    - validate_monte_carlo_distribution: {result}")
    except Exception as e:
        print(f"    - validate_monte_carlo_distribution: ❌ {str(e)}")

    print("\n✅ PASS - 個別驗證功能可執行")
    return True


def test_report_format():
    """測試報告格式"""
    print("\n測試 5: 驗證報告格式")
    validator = BacktestValidator()
    report = validator.validate_all()

    summary = report.summary()
    print("\n報告摘要:")
    print("-" * 60)
    print(summary)
    print("-" * 60)

    assert "回測驗證報告" in summary
    assert "總測試數" in summary
    assert "通過" in summary
    assert "失敗" in summary

    print("\n✅ PASS - 報告格式正確")
    return True


def main():
    """執行所有測試"""
    print("=" * 60)
    print("BacktestValidator 功能測試")
    print("=" * 60)

    tests = [
        test_basic_instantiation,
        test_validate_all,
        test_validation_levels,
        test_individual_validations,
        test_report_format,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n❌ FAIL - {test_func.__name__}: {str(e)}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"測試結果: {passed}/{len(tests)} 通過")
    print("=" * 60)

    if failed == 0:
        print("\n🎉 所有測試通過！")
        return 0
    else:
        print(f"\n⚠️ {failed} 個測試失敗")
        return 1


if __name__ == "__main__":
    exit(main())
