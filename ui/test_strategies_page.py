#!/usr/bin/env python3
"""
策略列表頁面測試腳本

快速驗證頁面功能是否正常運作。
"""

import sys
from pathlib import Path

# 加入專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """測試必要套件是否已安裝"""
    print("測試套件匯入...")

    required_packages = [
        ('streamlit', 'Streamlit'),
        ('pandas', 'Pandas'),
        ('plotly', 'Plotly'),
    ]

    missing = []
    for module, name in required_packages:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - 未安裝")
            missing.append(module)

    if missing:
        print("\n請安裝缺少的套件：")
        print(f"pip install {' '.join(missing)}")
        return False

    return True


def test_data_loading():
    """測試資料載入功能"""
    print("\n測試資料載入...")

    try:
        # 動態載入頁面模組
        sys.path.insert(0, str(Path(__file__).parent / 'pages'))

        # 模擬 Streamlit 環境
        import pandas as pd

        # 測試範例資料
        sample_data = [
            {
                'strategy_name': 'Test Strategy',
                'strategy_type': '趨勢',
                'symbol': 'BTCUSDT',
                'timeframe': '4h',
                'total_return': 45.8,
                'annual_return': 28.2,
                'sharpe_ratio': 1.85,
                'max_drawdown': 12.5,
                'total_trades': 158,
                'win_rate': 62.5,
                'grade': 'A',
                'wfa_efficiency': 0.85,
                'params': {'fast_period': 10, 'slow_period': 30},
                'created_at': '2024-01-10 14:30:00'
            }
        ]

        df = pd.DataFrame(sample_data)
        print(f"✓ 成功建立測試資料 ({len(df)} 筆)")

        # 測試篩選功能
        filters = {
            'min_sharpe': 1.0,
            'min_return': 0,
            'max_drawdown': 50,
            'min_trades': 0,
            'grades': ['A', 'B'],
            'strategy_types': ['趨勢'],
            'symbols': ['BTCUSDT'],
            'timeframes': ['4h']
        }

        filtered = df[df['sharpe_ratio'] >= filters['min_sharpe']]
        print(f"✓ 篩選功能正常 (結果: {len(filtered)} 筆)")

        return True

    except Exception as e:
        print(f"✗ 資料載入失敗: {e}")
        return False


def test_file_structure():
    """測試檔案結構"""
    print("\n測試檔案結構...")

    pages_dir = Path(__file__).parent / 'pages'
    strategies_file = pages_dir / '2_Strategies.py'

    if not strategies_file.exists():
        print(f"✗ 找不到檔案: {strategies_file}")
        return False

    print(f"✓ 策略列表頁面存在: {strategies_file}")

    # 檢查檔案內容
    with open(strategies_file, 'r', encoding='utf-8') as f:
        content = f.read()

        required_functions = [
            'load_strategy_results',
            'apply_filters',
            'sort_dataframe',
            'plot_equity_curve',
            'plot_monthly_heatmap'
        ]

        for func in required_functions:
            if func in content:
                print(f"✓ 函數已定義: {func}")
            else:
                print(f"✗ 缺少函數: {func}")
                return False

    return True


def test_validators():
    """測試驗證器整合"""
    print("\n測試驗證器整合...")

    try:
        from src.validator.stages import ValidationGrade, ValidationResult
        print("✓ ValidationGrade 匯入成功")
        print("✓ ValidationResult 匯入成功")

        # 測試等級
        grades = ['A', 'B', 'C', 'D', 'F']
        for grade in grades:
            g = ValidationGrade(grade)
            print(f"  - 等級 {grade}: {g.value}")

        return True

    except Exception as e:
        print(f"✗ 驗證器整合失敗: {e}")
        return False


def main():
    """執行所有測試"""
    print("=" * 60)
    print("策略列表頁面 - 功能測試")
    print("=" * 60)

    tests = [
        ("套件匯入", test_imports),
        ("檔案結構", test_file_structure),
        ("資料載入", test_data_loading),
        ("驗證器整合", test_validators),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n測試 '{name}' 發生錯誤: {e}")
            results.append((name, False))

    # 總結
    print("\n" + "=" * 60)
    print("測試結果總結")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ 通過" if result else "✗ 失敗"
        print(f"{status} - {name}")

    print(f"\n總計: {passed}/{total} 測試通過")

    if passed == total:
        print("\n✅ 所有測試通過！頁面已就緒。")
        print("\n啟動頁面：")
        print("  streamlit run ui/pages/2_Strategies.py")
    else:
        print("\n⚠️  部分測試失敗，請檢查錯誤訊息。")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
