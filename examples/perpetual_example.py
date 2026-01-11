"""
永續合約模組使用範例

展示如何在回測中使用永續合約計算功能。
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.backtester.perpetual import (
    PerpetualCalculator,
    PerpetualPosition,
    PerpetualRiskMonitor
)


def example_1_basic_calculations():
    """範例 1：基本計算功能"""
    print("\n" + "=" * 60)
    print("範例 1：永續合約基本計算")
    print("=" * 60)

    calc = PerpetualCalculator()

    # 情境：開倉做多 1 BTC @ $50,000，使用 10x 槓桿
    entry_price = 50000
    leverage = 10
    size = 1.0

    # 計算所需保證金
    margin = calc.calculate_initial_margin(size, entry_price, leverage)
    print(f"\n開倉資訊:")
    print(f"  價格: ${entry_price:,}")
    print(f"  數量: {size} BTC")
    print(f"  槓桿: {leverage}x")
    print(f"  所需保證金: ${margin:,}")

    # 計算強平價格
    liq_price = calc.calculate_liquidation_price(entry_price, leverage, 1)
    print(f"  強平價格: ${liq_price:,}")
    print(f"  強平距離: {((liq_price - entry_price) / entry_price * 100):.2f}%")

    # 模擬價格變化
    scenarios = [
        ("上漲 4%", 52000),
        ("下跌 4%", 48000),
        ("接近強平", 46000),
    ]

    print("\n價格變化情境:")
    for scenario_name, current_price in scenarios:
        pnl = calc.calculate_unrealized_pnl(entry_price, current_price, size, 1)
        pnl_pct = calc.calculate_pnl_percentage(pnl, margin)
        is_liq = calc.check_liquidation(current_price, entry_price, leverage, 1)

        print(f"\n  {scenario_name} → ${current_price:,}")
        print(f"    未實現盈虧: ${pnl:,.0f} ({pnl_pct:+.1f}%)")
        print(f"    狀態: {'💀 已爆倉' if is_liq else '✅ 安全'}")


def example_2_funding_rate_impact():
    """範例 2：資金費率影響"""
    print("\n" + "=" * 60)
    print("範例 2：資金費率對收益的影響")
    print("=" * 60)

    calc = PerpetualCalculator()

    # 情境：持倉 30 天，每 8 小時結算一次
    position_value = 50000
    holding_days = 30
    funding_intervals = (holding_days * 24) // 8  # 90 次結算

    print(f"\n持倉資訊:")
    print(f"  持倉價值: ${position_value:,}")
    print(f"  持倉時長: {holding_days} 天")
    print(f"  結算次數: {funding_intervals} 次")

    # 不同費率情境
    scenarios = [
        ("正常市場", 0.0001),
        ("牛市", 0.0005),
        ("極端牛市", 0.001),
        ("熊市", -0.0001),
    ]

    print("\n不同費率情境下的成本:")
    for scenario_name, avg_rate in scenarios:
        total_cost = 0
        for _ in range(funding_intervals):
            cost = calc.calculate_funding_cost(position_value, avg_rate, 1)
            total_cost += cost

        annualized = calc.annualized_funding_rate(avg_rate)

        print(f"\n  {scenario_name} (費率 {avg_rate * 100:.2f}%)")
        print(f"    總成本: ${total_cost:,.2f}")
        print(f"    佔持倉: {(total_cost / position_value * 100):.2f}%")
        print(f"    年化影響: {annualized * 100:.2f}%")


def example_3_position_management():
    """範例 3：倉位管理與風險監控"""
    print("\n" + "=" * 60)
    print("範例 3：倉位管理與風險監控")
    print("=" * 60)

    calc = PerpetualCalculator()
    monitor = PerpetualRiskMonitor(
        warning_threshold=0.02,  # 距離強平 2%
        critical_threshold=0.01   # 距離強平 1%
    )

    # 建立倉位
    position = PerpetualPosition(
        entry_price=50000,
        size=1.0,
        leverage=10,
        entry_time=datetime.now(),
        margin=5000
    )

    print(f"\n倉位資訊:")
    print(f"  方向: {'做多 📈' if position.is_long else '做空 📉'}")
    print(f"  入場價: ${position.entry_price:,}")
    print(f"  數量: {abs(position.size)} BTC")
    print(f"  槓桿: {position.leverage}x")
    print(f"  保證金: ${position.margin:,}")

    # 模擬價格下跌過程
    print("\n價格下跌過程中的風險監控:")
    print("-" * 60)

    prices = [50000, 49000, 48000, 47000, 46000, 45500, 45000]

    for price in prices:
        report = monitor.generate_risk_report(position, price)

        # 風險等級顏色
        risk_colors = {
            'safe': '🟢',
            'warning': '🟡',
            'critical': '🔴',
            'liquidated': '💀'
        }

        icon = risk_colors.get(report['risk_level'], '⚪')

        print(f"\n當前價格: ${price:,} {icon}")
        print(f"  風險等級: {report['risk_level'].upper()}")
        print(f"  距離強平: {abs(report['distance_to_liquidation_pct']):.2f}%")
        print(f"  未實現盈虧: ${report['unrealized_pnl']:,.0f}")
        print(f"  保證金率: {report['margin_ratio'] * 100:.2f}%")

        if report['risk_level'] == 'liquidated':
            print("  ⚠️ 已觸發強制平倉！")
            break


def example_4_leverage_comparison():
    """範例 4：不同槓桿對比"""
    print("\n" + "=" * 60)
    print("範例 4：不同槓桿倍數對比")
    print("=" * 60)

    calc = PerpetualCalculator()

    entry_price = 50000
    available_capital = 10000

    print(f"\n可用資金: ${available_capital:,}")
    print(f"當前價格: ${entry_price:,}")
    print("\n不同槓桿倍數比較:")
    print("-" * 60)

    leverages = [1, 5, 10, 20, 50]

    for leverage in leverages:
        # 計算最大倉位
        max_size = calc.estimate_max_position_size(
            available_capital,
            entry_price,
            leverage
        )

        # 計算強平價格
        liq_price = calc.calculate_liquidation_price(entry_price, leverage, 1)
        liq_distance = ((liq_price - entry_price) / entry_price * 100)

        # 計算價格漲跌 10% 的盈虧
        price_up = entry_price * 1.10
        price_down = entry_price * 0.90

        pnl_up = calc.calculate_unrealized_pnl(entry_price, price_up, max_size, 1)
        pnl_down = calc.calculate_unrealized_pnl(entry_price, price_down, max_size, 1)

        print(f"\n{leverage}x 槓桿:")
        print(f"  最大倉位: {max_size:.4f} BTC (${max_size * entry_price:,.0f})")
        print(f"  強平價格: ${liq_price:,.0f} ({liq_distance:.2f}%)")
        print(f"  價格 +10%: ${pnl_up:+,.0f} ({pnl_up/available_capital*100:+.1f}%)")
        print(f"  價格 -10%: ${pnl_down:+,.0f} ({pnl_down/available_capital*100:+.1f}%)")


def example_5_funding_rate_strategy():
    """範例 5：資金費率套利策略模擬"""
    print("\n" + "=" * 60)
    print("範例 5：資金費率套利策略（Delta Neutral）")
    print("=" * 60)

    calc = PerpetualCalculator()

    # 策略：現貨做多 + 永續做空
    capital = 20000
    spot_investment = capital / 2  # $10,000 買現貨
    perp_margin = capital / 2      # $10,000 做永續保證金

    spot_price = 50000
    perp_price = 50100  # 永續溢價 0.2%

    spot_size = spot_investment / spot_price
    perp_size = spot_size  # 等量做空

    print(f"\n策略設置:")
    print(f"  總資金: ${capital:,}")
    print(f"  現貨投資: ${spot_investment:,} ({spot_size:.4f} BTC)")
    print(f"  永續保證金: ${perp_margin:,}")
    print(f"  永續倉位: 做空 {perp_size:.4f} BTC")
    print(f"  基差: ${perp_price - spot_price} ({(perp_price/spot_price - 1)*100:.2f}%)")

    # 模擬持倉 30 天，收取資金費率
    holding_days = 30
    funding_intervals = (holding_days * 24) // 8
    avg_funding_rate = 0.0003  # 平均 0.03%

    print(f"\n持倉期間: {holding_days} 天")
    print(f"結算次數: {funding_intervals} 次")
    print(f"平均費率: {avg_funding_rate * 100:.3f}%")

    # 計算總收入（做空收取正費率）
    total_funding = 0
    perp_position_value = perp_size * perp_price

    for _ in range(funding_intervals):
        # 做空時，正費率收取（direction = -1）
        funding = calc.calculate_funding_cost(
            perp_position_value,
            avg_funding_rate,
            -1
        )
        total_funding += funding

    print(f"\n收益分析:")
    print(f"  總收取資金費率: ${abs(total_funding):,.2f}")
    print(f"  佔資金比例: {abs(total_funding) / capital * 100:.2f}%")

    annualized_return = (abs(total_funding) / capital) * (365 / holding_days)
    print(f"  年化收益率: {annualized_return * 100:.2f}%")

    # 考慮風險
    print(f"\n風險因素:")
    print(f"  ✓ 價格風險: 對沖（Delta Neutral）")
    print(f"  ✓ 強平風險: 使用低槓桿或全倉模式")
    print(f"  ⚠️ 費率風險: 費率可能轉負")
    print(f"  ⚠️ 基差風險: 平倉時基差可能不利")


def example_6_risk_monitoring_system():
    """範例 6：完整風險監控系統"""
    print("\n" + "=" * 60)
    print("範例 6：即時風險監控系統")
    print("=" * 60)

    calc = PerpetualCalculator()
    monitor = PerpetualRiskMonitor()

    # 建立多個倉位
    positions = [
        PerpetualPosition(
            entry_price=50000,
            size=1.0,
            leverage=10,
            entry_time=datetime.now(),
            margin=5000
        ),
        PerpetualPosition(
            entry_price=3000,
            size=-5.0,  # 做空
            leverage=5,
            entry_time=datetime.now(),
            margin=3000
        ),
    ]

    # 模擬市場行情
    btc_price = 49000
    eth_price = 3100

    print("\n倉位風險總覽:")
    print("-" * 60)

    total_margin = 0
    total_unrealized_pnl = 0
    risk_summary = {'safe': 0, 'warning': 0, 'critical': 0, 'liquidated': 0}

    for i, position in enumerate(positions, 1):
        current_price = btc_price if i == 1 else eth_price
        symbol = "BTC" if i == 1 else "ETH"

        report = monitor.generate_risk_report(position, current_price)

        print(f"\n倉位 {i} - {symbol}:")
        print(f"  方向: {'做多 📈' if position.is_long else '做空 📉'}")
        print(f"  入場價: ${position.entry_price:,}")
        print(f"  當前價: ${current_price:,}")
        print(f"  槓桿: {position.leverage}x")
        print(f"  風險: {report['risk_level'].upper()}")
        print(f"  距離強平: {abs(report['distance_to_liquidation_pct']):.2f}%")
        print(f"  未實現盈虧: ${report['unrealized_pnl']:+,.0f}")

        total_margin += position.margin
        total_unrealized_pnl += report['unrealized_pnl']
        risk_summary[report['risk_level']] += 1

    print("\n" + "=" * 60)
    print("投資組合總結:")
    print(f"  總保證金: ${total_margin:,}")
    print(f"  總未實現盈虧: ${total_unrealized_pnl:+,.0f}")
    print(f"  總權益: ${total_margin + total_unrealized_pnl:,.0f}")
    print(f"\n風險分布:")
    print(f"  🟢 安全: {risk_summary['safe']}")
    print(f"  🟡 警告: {risk_summary['warning']}")
    print(f"  🔴 危急: {risk_summary['critical']}")
    print(f"  💀 爆倉: {risk_summary['liquidated']}")


if __name__ == '__main__':
    # 執行所有範例
    example_1_basic_calculations()
    example_2_funding_rate_impact()
    example_3_position_management()
    example_4_leverage_comparison()
    example_5_funding_rate_strategy()
    example_6_risk_monitoring_system()

    print("\n" + "=" * 60)
    print("✅ 所有範例執行完成！")
    print("=" * 60)
