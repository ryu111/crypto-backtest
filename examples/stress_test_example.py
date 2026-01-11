"""
黑天鵝壓力測試範例

展示如何使用 StressTester 進行壓力測試。
"""

import sys
from pathlib import Path

# 加入專案根目錄到 Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from src.validator import StressTester, HISTORICAL_EVENTS


def main():
    """執行黑天鵝壓力測試範例"""
    print("=" * 70)
    print("黑天鵝壓力測試範例")
    print("=" * 70)

    # 1. 建立模擬策略報酬
    print("\n1. 產生模擬策略報酬（252 天 ≈ 1 年）")
    np.random.seed(42)
    strategy_returns = pd.Series(
        np.random.normal(loc=0.002, scale=0.015, size=252),
        index=pd.date_range('2023-01-01', periods=252, freq='D')
    )

    original_return = (1 + strategy_returns).prod() - 1
    print(f"原始策略總報酬: {original_return:.2%}")
    print(f"平均日報酬: {strategy_returns.mean():.4f}")
    print(f"日報酬標準差: {strategy_returns.std():.4f}")

    # 2. 建立壓力測試器
    print("\n2. 建立壓力測試器")
    tester = StressTester(
        survival_threshold=-0.5,  # 虧損 50% 視為爆倉
        risk_free_rate=0.02       # 2% 無風險利率
    )
    print(f"存活閾值: {tester.survival_threshold:.1%}")
    print(f"無風險利率: {tester.risk_free_rate:.1%}")

    # 3. 重播歷史黑天鵝事件
    print("\n3. 重播歷史黑天鵝事件")
    print("-" * 70)

    for event_name in HISTORICAL_EVENTS:
        result = tester.replay_historical_event(
            strategy_returns=strategy_returns,
            event_name=event_name
        )

        print(f"\n事件: {result.event_name}")
        print(f"  衝擊: {result.drop_percentage:.1%} ({result.duration_days} 天)")
        print(f"  總報酬: {result.total_return:>8.2%}")
        print(f"  最大回撤: {result.max_drawdown:>8.2%}")
        print(f"  Sharpe: {result.sharpe_ratio:>8.2f}")
        if result.recovery_days:
            print(f"  恢復天數: {result.recovery_days:>6} 天")
        else:
            print(f"  恢復天數: {'未恢復':>8}")

    # 4. 測試自定義情境
    print("\n4. 測試自定義極端情境")
    print("-" * 70)

    custom_scenarios = [
        {
            'name': '輕微衝擊',
            'drop': -0.15,
            'duration': 3,
            'description': '15% 下跌，持續 3 天'
        },
        {
            'name': '中等衝擊',
            'drop': -0.35,
            'duration': 7,
            'description': '35% 下跌，持續 7 天'
        },
        {
            'name': '極端衝擊',
            'drop': -0.70,
            'duration': 14,
            'description': '70% 下跌，持續 14 天（極端黑天鵝）'
        }
    ]

    for scenario in custom_scenarios:
        result = tester.run_scenario(
            strategy_returns=strategy_returns,
            scenario=scenario
        )

        print(f"\n{result.event_name}")
        print(f"  {result.description}")
        print(f"  總報酬: {result.total_return:>8.2%}")
        print(f"  最大回撤: {result.max_drawdown:>8.2%}")
        print(f"  勝率: {result.win_rate:>8.1%}")

    # 5. 產生完整壓力測試報告
    print("\n5. 產生完整壓力測試報告")
    print("=" * 70)

    report = tester.generate_stress_report(
        strategy_returns=strategy_returns,
        custom_scenarios=custom_scenarios
    )

    # 使用內建的格式化輸出
    StressTester.print_report(report)

    # 6. 詳細檢視最差情境
    print("\n6. 詳細檢視最差情境")
    print("=" * 70)

    worst_result = min(report.test_results, key=lambda x: x.total_return)
    StressTester.print_result(worst_result)

    # 7. 分析結論
    print("\n7. 壓力測試分析結論")
    print("=" * 70)

    if report.survival_rate >= 0.8:
        print("✓ 策略在大多數黑天鵝事件中存活（存活率 ≥ 80%）")
    elif report.survival_rate >= 0.5:
        print("⚠ 策略在部分黑天鵝事件中可能爆倉（存活率 50-80%）")
    else:
        print("✗ 策略在黑天鵝事件中高風險（存活率 < 50%）")

    if report.profit_rate >= 0.5:
        print("✓ 策略在半數以上情境仍能獲利")
    else:
        print("⚠ 策略在多數黑天鵝事件中虧損")

    if report.average_recovery_days != float('inf'):
        if report.average_recovery_days <= 30:
            print(f"✓ 策略恢復速度快（平均 {report.average_recovery_days:.0f} 天）")
        else:
            print(f"⚠ 策略恢復速度慢（平均 {report.average_recovery_days:.0f} 天）")
    else:
        print("✗ 策略在某些情境無法恢復")

    print(f"\n最大風險情境: {report.worst_scenario}")
    print(f"該情境下報酬: {report.worst_return:.2%}")
    print(f"該情境下最大回撤: {report.worst_drawdown:.2%}")


if __name__ == '__main__':
    main()
