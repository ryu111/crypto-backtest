"""
滑點模擬使用範例

展示如何使用滑點模組進行真實交易成本估算。
"""

import pandas as pd
import numpy as np
from datetime import datetime

from src.backtester.slippage import (
    SlippageCalculator,
    SlippageConfig,
    SlippageModel,
    OrderType,
    create_fixed_slippage,
    create_dynamic_slippage,
    create_market_impact_slippage
)


def generate_sample_data(periods: int = 1000) -> pd.DataFrame:
    """產生模擬 OHLCV 資料"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='1h')

    # 模擬價格走勢（帶趨勢和隨機波動）
    trend = np.linspace(50000, 55000, periods)
    noise = np.random.randn(periods) * 500
    close_prices = trend + noise

    data = pd.DataFrame({
        'open': close_prices + np.random.randn(periods) * 50,
        'high': close_prices + np.abs(np.random.randn(periods) * 100),
        'low': close_prices - np.abs(np.random.randn(periods) * 100),
        'close': close_prices,
        'volume': 100 + np.random.randn(periods) * 20
    }, index=dates)

    # 確保 high 和 low 正確
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    data['volume'] = data['volume'].clip(lower=10)  # 確保成交量為正

    return data


def example_1_basic_usage():
    """範例 1：基本使用"""
    print("\n" + "="*70)
    print("範例 1：基本滑點計算")
    print("="*70)

    data = generate_sample_data()

    # 建立固定滑點計算器（0.05%）
    calculator = create_fixed_slippage(0.0005)

    # 計算單筆交易滑點
    slippage = calculator.calculate(
        data=data,
        order_size=10000,  # 10,000 USDT 訂單
        order_type=OrderType.MARKET
    )

    print(f"\n固定滑點: {slippage:.4%}")

    # 估算執行價格
    current_price = data['close'].iloc[-1]
    exec_price_long = calculator.estimate_execution_price(
        current_price, slippage, direction=1
    )
    exec_price_short = calculator.estimate_execution_price(
        current_price, slippage, direction=-1
    )

    print(f"\n當前市價: ${current_price:,.2f}")
    print(f"做多執行價: ${exec_price_long:,.2f} (上滑 ${exec_price_long - current_price:.2f})")
    print(f"做空執行價: ${exec_price_short:,.2f} (下滑 ${current_price - exec_price_short:.2f})")


def example_2_dynamic_slippage():
    """範例 2：動態滑點（根據市場波動調整）"""
    print("\n" + "="*70)
    print("範例 2：動態滑點計算")
    print("="*70)

    data = generate_sample_data()

    # 建立動態滑點計算器
    calculator = create_dynamic_slippage(
        base_slippage=0.0005,
        volatility_factor=1.5,  # 波動率影響係數
        max_slippage=0.01
    )

    # 比較不同時間點的滑點
    indices = [100, 300, 500, 700, 900]
    print("\n不同時間點的動態滑點：")
    print(f"{'時間':<20} {'收盤價':>12} {'滑點':>10} {'執行價差':>12}")
    print("-" * 60)

    for idx in indices:
        slippage = calculator.calculate(
            data=data,
            order_size=10000,
            index=idx
        )

        price = data['close'].iloc[idx]
        price_diff = price * slippage

        print(
            f"{str(data.index[idx]):<20} "
            f"${price:>11,.2f} "
            f"{slippage:>9.4%} "
            f"${price_diff:>11.2f}"
        )


def example_3_market_impact():
    """範例 3：市場衝擊（訂單大小影響）"""
    print("\n" + "="*70)
    print("範例 3：市場衝擊滑點（訂單大小影響）")
    print("="*70)

    data = generate_sample_data()

    # 建立市場衝擊滑點計算器
    calculator = create_market_impact_slippage(
        base_slippage=0.0005,
        market_impact_coeff=0.1,
        max_slippage=0.01
    )

    # 比較不同訂單大小的滑點
    order_sizes = [1000, 5000, 10000, 50000, 100000]

    print("\n不同訂單大小的滑點：")
    print(f"{'訂單金額 (USDT)':>20} {'滑點':>10} {'成本':>12}")
    print("-" * 50)

    for size in order_sizes:
        slippage = calculator.calculate(
            data=data,
            order_size=size,
            index=-1  # 使用最新資料
        )

        cost = size * slippage

        print(
            f"{size:>20,} "
            f"{slippage:>9.4%} "
            f"${cost:>11.2f}"
        )


def example_4_order_types():
    """範例 4：不同訂單類型的滑點"""
    print("\n" + "="*70)
    print("範例 4：不同訂單類型的滑點")
    print("="*70)

    data = generate_sample_data()

    config = SlippageConfig(
        model=SlippageModel.DYNAMIC,
        base_slippage=0.0005,
        stop_order_multiplier=1.5  # 止損單滑點倍數
    )
    calculator = SlippageCalculator(config)

    order_types = [
        OrderType.MARKET,
        OrderType.LIMIT,
        OrderType.STOP,
    ]

    print("\n不同訂單類型的滑點（10,000 USDT 訂單）：")
    print(f"{'訂單類型':<15} {'滑點':>10} {'說明'}")
    print("-" * 60)

    for order_type in order_types:
        slippage = calculator.calculate(
            data=data,
            order_size=10000,
            order_type=order_type
        )

        descriptions = {
            OrderType.MARKET: "市價單：完整滑點",
            OrderType.LIMIT: "限價單：無滑點（但可能不成交）",
            OrderType.STOP: "止損單：滑點較高（1.5x）"
        }

        print(
            f"{order_type.value:<15} "
            f"{slippage:>9.4%} "
            f"{descriptions[order_type]}"
        )


def example_5_vectorized_calculation():
    """範例 5：向量化計算（回測場景）"""
    print("\n" + "="*70)
    print("範例 5：向量化滑點計算（回測場景）")
    print("="*70)

    data = generate_sample_data(periods=100)

    calculator = create_dynamic_slippage()

    # 模擬交易序列（隨機訂單大小）
    np.random.seed(42)
    order_sizes = pd.Series(
        np.random.uniform(5000, 20000, len(data)),
        index=data.index
    )

    # 向量化計算所有滑點
    slippages = calculator.calculate_vectorized(data, order_sizes)

    # 統計分析
    valid_slippages = slippages.dropna()

    print(f"\n共計算 {len(valid_slippages)} 筆交易滑點")
    print(f"\n滑點統計：")
    print(f"  平均滑點: {valid_slippages.mean():.4%}")
    print(f"  最大滑點: {valid_slippages.max():.4%}")
    print(f"  最小滑點: {valid_slippages.min():.4%}")
    print(f"  標準差:   {valid_slippages.std():.4%}")

    # 計算總成本
    total_cost = (order_sizes * slippages).sum()
    print(f"\n總滑點成本: ${total_cost:,.2f}")


def example_6_impact_analysis():
    """範例 6：滑點影響分析"""
    print("\n" + "="*70)
    print("範例 6：滑點影響分析")
    print("="*70)

    data = generate_sample_data()

    calculator = create_market_impact_slippage()

    # 模擬交易記錄
    trades = pd.DataFrame({
        'entry_time': data.index[::50][:10],  # 每 50 小時一筆交易
        'size': [10000, 15000, 8000, 20000, 12000,
                 9000, 18000, 11000, 13000, 16000]
    })

    # 分析滑點影響
    analysis = calculator.analyze_impact(data, trades)

    print(f"\n交易滑點分析（{len(trades)} 筆交易）：")
    print(f"  總成本:     ${analysis['total_cost']:,.2f}")
    print(f"  平均滑點:   {analysis['avg_slippage']:.4%}")
    print(f"  最大滑點:   {analysis['max_slippage']:.4%}")
    print(f"  最小滑點:   {analysis['min_slippage']:.4%}")
    print(f"  中位數滑點: {analysis['median_slippage']:.4%}")
    print(f"  標準差:     {analysis['std_slippage']:.4%}")


def example_7_slippage_curve():
    """範例 7：滑點曲線分析"""
    print("\n" + "="*70)
    print("範例 7：滑點曲線分析")
    print("="*70)

    data = generate_sample_data(periods=100)

    calculator = create_market_impact_slippage()

    # 產生不同訂單大小的滑點曲線
    curve = calculator.get_slippage_curve(
        data,
        order_sizes=[1000, 5000, 10000, 50000, 100000]
    )

    print(f"\n滑點曲線統計（{len(curve)} 個時間點）：")
    print(f"\n{'訂單大小':>15} {'平均滑點':>12} {'最大滑點':>12} {'最小滑點':>12}")
    print("-" * 60)

    for col in curve.columns:
        size = int(col.split('_')[1])
        print(
            f"${size:>14,} "
            f"{curve[col].mean():>11.4%} "
            f"{curve[col].max():>11.4%} "
            f"{curve[col].min():>11.4%}"
        )


def example_8_custom_slippage():
    """範例 8：自定義滑點函數"""
    print("\n" + "="*70)
    print("範例 8：自定義滑點函數")
    print("="*70)

    data = generate_sample_data()

    # 定義自定義滑點函數（基於時間的滑點）
    def time_based_slippage(data, order_size, index):
        """
        根據交易時間調整滑點
        假設：交易活躍時段滑點較低
        """
        hour = data.index[index].hour
        base = 0.0005

        # 交易活躍時段 (8:00 - 20:00) 滑點較低
        if 8 <= hour < 20:
            return base
        else:
            # 非活躍時段滑點增加 50%
            return base * 1.5

    calculator = SlippageCalculator()
    calculator.set_custom_function(time_based_slippage)

    # 測試不同時段的滑點
    print("\n不同時段的滑點（自定義函數）：")
    print(f"{'時間':<20} {'小時':>6} {'滑點':>10} {'說明'}")
    print("-" * 60)

    test_indices = [8, 32, 56, 80]  # 不同小時
    for idx in test_indices:
        slippage = calculator.calculate(
            data=data,
            order_size=10000,
            index=idx
        )

        hour = data.index[idx].hour
        is_active = 8 <= hour < 20

        print(
            f"{str(data.index[idx]):<20} "
            f"{hour:>6} "
            f"{slippage:>9.4%} "
            f"{'活躍時段' if is_active else '非活躍時段'}"
        )


def main():
    """執行所有範例"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "滑點模擬模組使用範例" + " "*15 + "║")
    print("╚" + "="*68 + "╝")

    example_1_basic_usage()
    example_2_dynamic_slippage()
    example_3_market_impact()
    example_4_order_types()
    example_5_vectorized_calculation()
    example_6_impact_analysis()
    example_7_slippage_curve()
    example_8_custom_slippage()

    print("\n" + "="*70)
    print("所有範例執行完成！")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
