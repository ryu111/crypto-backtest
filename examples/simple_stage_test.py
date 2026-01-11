"""
簡單的 5 階段驗證測試

不依賴實際資料，展示驗證器結構。
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def test_stage_structure():
    """測試驗證器結構"""
    print("="*60)
    print("5 階段驗證器結構測試")
    print("="*60)

    try:
        from src.validator.stages import (
            StageValidator,
            ValidationGrade,
            ValidationResult,
            StageResult
        )

        print("\n✓ 成功導入驗證器模組")

        # 建立驗證器
        validator = StageValidator()
        print("✓ 成功建立 StageValidator 實例")

        # 檢查門檻值
        print("\n門檻值配置:")
        for stage, thresholds in validator.thresholds.items():
            print(f"\n{stage}:")
            for key, value in thresholds.items():
                print(f"  {key}: {value}")

        # 測試評級計算
        print("\n評級計算測試:")
        for i in range(6):
            grade = validator._calculate_grade(i)
            print(f"  通過 {i} 階段 → 評級: {grade.value}")

        # 測試建議生成
        print("\n建議生成測試:")
        stage_results = {}
        for grade in ValidationGrade:
            rec = validator._generate_recommendation(grade, stage_results)
            print(f"\n評級 {grade.value}:")
            print(f"{rec[:100]}...")

        print("\n" + "="*60)
        print("✓ 所有結構測試通過")
        print("="*60)

        return True

    except ImportError as e:
        print(f"\n✗ 導入失敗: {e}")
        print("提示：可能缺少 vectorbt 依賴")
        return False

    except Exception as e:
        print(f"\n✗ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stage_result_creation():
    """測試 StageResult 建立"""
    print("\n測試 StageResult 建立...")

    try:
        from src.validator.stages import StageResult

        result = StageResult(
            passed=True,
            score=85.0,
            details={'test': 'value'},
            message="測試通過",
            threshold={'min': 0.5}
        )

        print(f"  StageResult: {result}")
        print(f"  Passed: {result.passed}")
        print(f"  Score: {result.score}")
        print("  ✓ StageResult 建立成功")

        return True

    except Exception as e:
        print(f"  ✗ 失敗: {e}")
        return False


def test_validation_result_creation():
    """測試 ValidationResult 建立"""
    print("\n測試 ValidationResult 建立...")

    try:
        from src.validator.stages import ValidationResult, ValidationGrade, StageResult

        stage_results = {
            '階段1': StageResult(
                passed=True,
                score=90.0,
                details={},
                message="通過",
                threshold={}
            ),
            '階段2': StageResult(
                passed=True,
                score=85.0,
                details={},
                message="通過",
                threshold={}
            ),
        }

        result = ValidationResult(
            grade=ValidationGrade.B,
            passed_stages=4,
            stage_results=stage_results,
            recommendation="良好策略"
        )

        print(f"  Grade: {result.grade.value}")
        print(f"  Passed Stages: {result.passed_stages}")
        print(f"  Stage Results: {len(result.stage_results)}")
        print("  ✓ ValidationResult 建立成功")

        # 測試 summary
        summary = result.summary()
        print(f"\n  Summary 長度: {len(summary)} 字元")
        print("  ✓ Summary 生成成功")

        return True

    except Exception as e:
        print(f"  ✗ 失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_validation_workflow():
    """展示驗證流程"""
    print("\n" + "="*60)
    print("5 階段驗證流程說明")
    print("="*60)

    stages = [
        ("階段 1", "基礎回測", [
            "total_return > 0",
            "total_trades >= 30",
            "sharpe_ratio > 0.5",
            "max_drawdown < 30%",
            "profit_factor > 1.0"
        ]),
        ("階段 2", "統計檢驗", [
            "t-test p < 0.05",
            "Sharpe 95% CI 不含 0",
            "偏態 |skew| < 2"
        ]),
        ("階段 3", "穩健性測試", [
            "參數敏感度 < 30%",
            "時間一致性（前後半期皆獲利）",
            "標的一致性（BTC/ETH 皆獲利）"
        ]),
        ("階段 4", "Walk-Forward 分析", [
            "WFA Efficiency >= 50%",
            "OOS 勝率 > 50%",
            "無單窗口 > -10%"
        ]),
        ("階段 5", "Monte Carlo 模擬", [
            "5th percentile > 0",
            "1st percentile > -30%",
            "Median > Original × 50%"
        ]),
    ]

    for stage_num, stage_name, criteria in stages:
        print(f"\n{stage_num}：{stage_name}")
        print("-" * 40)
        for criterion in criteria:
            print(f"  • {criterion}")

    print("\n" + "="*60)
    print("評級標準")
    print("="*60)
    grades = [
        ("A", "5/5", "優秀，可實盤測試"),
        ("B", "4/5", "良好，降低倉位"),
        ("C", "3/5", "及格，需改進"),
        ("D", "1-2/5", "不及格，重新優化"),
        ("F", "0/5", "失敗，重新設計"),
    ]

    for grade, stages_passed, desc in grades:
        print(f"  {grade} 級 | {stages_passed:6s} | {desc}")

    print("="*60)


if __name__ == '__main__':
    # 測試 1: 結構測試
    success1 = test_stage_structure()

    if success1:
        # 測試 2: StageResult 建立
        success2 = test_stage_result_creation()

        # 測試 3: ValidationResult 建立
        success3 = test_validation_result_creation()

        if success2 and success3:
            print("\n✓ 所有測試通過")
        else:
            print("\n△ 部分測試失敗")

    # 展示流程
    show_validation_workflow()

    print("\n提示：")
    print("  - 完整範例請參考 examples/stage_validation_example.py")
    print("  - 詳細說明請參考 src/validator/README.md")
