"""
流動性影響模組使用範例

展示如何使用流動性計算器評估大單對市場的價格衝擊。
"""

import sys
from pathlib import Path

# 加入專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtester import (
    create_square_root_liquidity,
    create_linear_liquidity,
    create_logarithmic_liquidity,
    LiquidityLevel
)
import pandas as pd
import numpy as np


def create_sample_data(periods=1000):
    """建立模擬市場資料"""
    dates = pd.date_range('2024-01-01', periods=periods, freq='1h')

    # 模擬 BTC 價格（50,000 USDT 起）
    np.random.seed(42)
    close_prices = 50000 + np.cumsum(np.random.randn(periods) * 100)
    volumes = np.random.uniform(100, 1000, periods)  # BTC

    data = pd.DataFrame({
        'close': close_prices,
        'volume': volumes,
    }, index=dates)

    return data


def example_1_basic_usage():
    """範例 1：基本使用"""
    print("=" * 60)
    print("範例 1：基本流動性衝擊計算")
    print("=" * 60)

    data = create_sample_data()

    # 建立平方根模型計算器（推薦，學術標準）
    calc = create_square_root_liquidity(
        impact_coefficient=0.3,  # 衝擊係數（通常 0.1-0.5）
        adv_window=30,           # ADV 計算窗口（30天）
        use_volatility=True      # 考慮波動率
    )

    # 計算不同訂單大小的價格衝擊
    order_sizes = [10000, 50000, 100000, 500000]

    for size in order_sizes:
        impact = calc.calculate_impact(data, size, index=500)
        exec_price = calc.estimate_execution_price(
            data['close'].iloc[500], impact, direction=1
        )
        level = calc.get_liquidity_score(data, size, index=500)

        print(f"\n訂單: ${size:,.0f}")
        print(f"  價格衝擊: {impact:.4%}")
        print(f"  執行價格: ${exec_price:,.2f} (市價: ${data['close'].iloc[500]:,.2f})")
        print(f"  流動性等級: {level.value.upper()}")


def example_2_max_order_size():
    """範例 2：計算最大訂單"""
    print("\n" + "=" * 60)
    print("範例 2：計算最大可執行訂單")
    print("=" * 60)

    data = create_sample_data()
    calc = create_square_root_liquidity()

    # 不同價格容忍度的最大訂單
    tolerances = [0.005, 0.01, 0.02]  # 0.5%, 1%, 2%

    for tol in tolerances:
        max_order = calc.calculate_max_order_size(data, tol, index=500)
        print(f"\n容忍度 {tol:.2%}: 最大訂單 ${max_order:,.0f}")

        # 驗證：計算該訂單的實際衝擊
        actual_impact = calc.calculate_impact(data, max_order, index=500)
        print(f"  實際衝擊: {actual_impact:.4%} (目標: {tol:.4%})")


def example_3_model_comparison():
    """範例 3：不同模型比較"""
    print("\n" + "=" * 60)
    print("範例 3：流動性模型比較")
    print("=" * 60)

    data = create_sample_data()
    order_size = 50000

    # 建立不同模型
    linear = create_linear_liquidity(impact_coefficient=0.2)
    sqrt = create_square_root_liquidity(impact_coefficient=0.3)
    log = create_logarithmic_liquidity(impact_coefficient=0.4)

    models = [
        ("線性模型", linear, "衝擊 ∝ Q"),
        ("平方根模型", sqrt, "衝擊 ∝ √Q (推薦)"),
        ("對數模型", log, "衝擊 ∝ log(Q)")
    ]

    print(f"\n訂單大小: ${order_size:,.0f}\n")

    for name, calc, formula in models:
        impact = calc.calculate_impact(data, order_size, index=500)
        max_order = calc.calculate_max_order_size(data, 0.01, index=500)

        print(f"{name} ({formula})")
        print(f"  價格衝擊: {impact:.4%}")
        print(f"  最大訂單 (1% 容忍): ${max_order:,.0f}\n")


def example_4_vectorized_calculation():
    """範例 4：向量化計算（回測用）"""
    print("=" * 60)
    print("範例 4：向量化計算（回測場景）")
    print("=" * 60)

    data = create_sample_data()
    calc = create_square_root_liquidity()

    # 模擬一系列訂單
    order_sizes = pd.Series(
        np.random.uniform(10000, 100000, len(data)),
        index=data.index
    )

    # 向量化計算所有衝擊
    impacts = calc.calculate_vectorized(data, order_sizes)

    print(f"\n總訂單數: {len(order_sizes)}")
    print(f"平均衝擊: {impacts.mean():.4%}")
    print(f"最大衝擊: {impacts.max():.4%}")
    print(f"最小衝擊: {impacts.min():.4%}")

    # 統計不同流動性等級的訂單分布
    levels_count = {level: 0 for level in LiquidityLevel}

    for i in range(0, len(data), 50):  # 取樣避免過慢
        level = calc.get_liquidity_score(data, order_sizes.iloc[i], index=i)
        levels_count[level] += 1

    print("\n流動性等級分布:")
    for level, count in levels_count.items():
        if count > 0:
            print(f"  {level.value.upper()}: {count}")


def example_5_liquidity_analysis():
    """範例 5：流動性分析"""
    print("\n" + "=" * 60)
    print("範例 5：流動性分析")
    print("=" * 60)

    data = create_sample_data()
    calc = create_square_root_liquidity()

    # 分析不同訂單大小的流動性衝擊曲線
    analysis = calc.analyze_liquidity(
        data,
        order_sizes=[10000, 50000, 100000, 500000, 1000000]
    )

    print("\n不同訂單大小的平均衝擊:")
    for col in analysis.columns:
        size = col.replace('size_', '')
        avg_impact = analysis[col].mean()
        max_impact = analysis[col].max()
        print(f"  ${int(size):>9,}: 平均 {avg_impact:.4%}, 最大 {max_impact:.4%}")


if __name__ == "__main__":
    example_1_basic_usage()
    example_2_max_order_size()
    example_3_model_comparison()
    example_4_vectorized_calculation()
    example_5_liquidity_analysis()

    print("\n" + "=" * 60)
    print("所有範例執行完成！")
    print("=" * 60)
