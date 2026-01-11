"""
動量策略使用範例

展示如何使用 RSI 和 MACD 策略產生交易訊號。
"""

import sys
from pathlib import Path

# 添加專案根目錄到 Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.strategies import create_strategy, list_strategies, StrategyRegistry


def create_sample_data(periods=200):
    """建立模擬 OHLCV 資料"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=periods, freq='h')

    # 模擬價格走勢（帶趨勢和波動）
    trend = np.linspace(0, 20, periods)
    noise = np.cumsum(np.random.randn(periods) * 2)
    close_prices = 100 + trend + noise

    data = pd.DataFrame({
        'open': close_prices + np.random.randn(periods) * 0.5,
        'high': close_prices + abs(np.random.randn(periods) * 1),
        'low': close_prices - abs(np.random.randn(periods) * 1),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, periods)
    }, index=dates)

    return data


def demo_rsi_strategy():
    """展示 RSI 策略"""
    print("=" * 60)
    print("RSI 策略範例")
    print("=" * 60)

    # 建立測試資料
    data = create_sample_data()
    print(f"\n資料期間: {data.index[0]} 至 {data.index[-1]}")
    print(f"資料筆數: {len(data)}")
    print(f"價格範圍: {data['close'].min():.2f} - {data['close'].max():.2f}")

    # 建立 RSI 策略（使用預設參數）
    strategy = create_strategy('momentum_rsi')
    print(f"\n策略: {strategy.name}")
    print(f"參數: {strategy.params}")

    # 計算指標
    indicators = strategy.calculate_indicators(data)
    rsi = indicators['rsi']
    print(f"\nRSI 統計:")
    print(f"  範圍: {rsi.min():.2f} - {rsi.max():.2f}")
    print(f"  平均: {rsi.mean():.2f}")
    print(f"  標準差: {rsi.std():.2f}")

    # 產生訊號
    long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(data)

    print(f"\n訊號統計:")
    print(f"  多單進場: {long_entry.sum()} 次")
    print(f"  多單出場: {long_exit.sum()} 次")
    print(f"  空單進場: {short_entry.sum()} 次")
    print(f"  空單出場: {short_exit.sum()} 次")

    # 顯示前幾個訊號
    if long_entry.sum() > 0:
        first_long = data.index[long_entry][0]
        print(f"\n第一個多單進場時間: {first_long}")
        print(f"當時價格: {data.loc[first_long, 'close']:.2f}")
        print(f"當時 RSI: {rsi.loc[first_long]:.2f}")

    return strategy, data, indicators


def demo_macd_strategy():
    """展示 MACD 策略"""
    print("\n" + "=" * 60)
    print("MACD 策略範例")
    print("=" * 60)

    # 建立測試資料
    data = create_sample_data()
    print(f"\n資料期間: {data.index[0]} 至 {data.index[-1]}")
    print(f"資料筆數: {len(data)}")

    # 建立 MACD 策略（自訂參數）
    strategy = create_strategy(
        'momentum_macd',
        fast_period=12,
        slow_period=26,
        signal_period=9,
        use_histogram=True
    )
    print(f"\n策略: {strategy.name}")
    print(f"參數: {strategy.params}")

    # 計算指標
    indicators = strategy.calculate_indicators(data)
    macd = indicators['macd']
    signal = indicators['signal']
    histogram = indicators['histogram']

    print(f"\nMACD 統計:")
    print(f"  MACD 範圍: {macd.min():.2f} - {macd.max():.2f}")
    print(f"  Signal 範圍: {signal.min():.2f} - {signal.max():.2f}")
    print(f"  Histogram 範圍: {histogram.min():.2f} - {histogram.max():.2f}")

    # 產生訊號
    long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(data)

    print(f"\n訊號統計:")
    print(f"  多單進場: {long_entry.sum()} 次")
    print(f"  多單出場: {long_exit.sum()} 次")
    print(f"  空單進場: {short_entry.sum()} 次")
    print(f"  空單出場: {short_exit.sum()} 次")

    # 顯示交叉點
    golden_cross = (macd > signal) & (macd.shift(1) <= signal.shift(1))
    death_cross = (macd < signal) & (macd.shift(1) >= signal.shift(1))

    print(f"\nMACD 穿越:")
    print(f"  黃金交叉: {golden_cross.sum()} 次")
    print(f"  死亡交叉: {death_cross.sum()} 次")

    return strategy, data, indicators


def demo_strategy_comparison():
    """比較兩個策略"""
    print("\n" + "=" * 60)
    print("策略比較")
    print("=" * 60)

    data = create_sample_data()

    # 建立兩個策略
    rsi_strategy = create_strategy('momentum_rsi')
    macd_strategy = create_strategy('momentum_macd')

    # 產生訊號
    rsi_long, rsi_long_exit, rsi_short, rsi_short_exit = rsi_strategy.generate_signals(data)
    macd_long, macd_long_exit, macd_short, macd_short_exit = macd_strategy.generate_signals(data)

    print("\n訊號比較:")
    print(f"{'策略':<20} {'多單進場':<12} {'多單出場':<12} {'空單進場':<12} {'空單出場':<12}")
    print("-" * 60)
    print(f"{'RSI':<20} {rsi_long.sum():<12} {rsi_long_exit.sum():<12} {rsi_short.sum():<12} {rsi_short_exit.sum():<12}")
    print(f"{'MACD':<20} {macd_long.sum():<12} {macd_long_exit.sum():<12} {macd_short.sum():<12} {macd_short_exit.sum():<12}")


def demo_parameter_optimization():
    """展示參數空間"""
    print("\n" + "=" * 60)
    print("參數優化空間")
    print("=" * 60)

    # 取得 RSI 策略參數空間
    rsi_param_space = StrategyRegistry.get_param_space('momentum_rsi')
    print("\nRSI 策略參數空間:")
    for param, config in rsi_param_space.items():
        print(f"  {param}: {config}")

    # 取得 MACD 策略參數空間
    macd_param_space = StrategyRegistry.get_param_space('momentum_macd')
    print("\nMACD 策略參數空間:")
    for param, config in macd_param_space.items():
        print(f"  {param}: {config}")


if __name__ == '__main__':
    print("\n動量策略測試範例\n")

    # 列出所有已註冊策略
    print("已註冊策略:", list_strategies())
    print()

    # 執行各個範例
    demo_rsi_strategy()
    demo_macd_strategy()
    demo_strategy_comparison()
    demo_parameter_optimization()

    print("\n" + "=" * 60)
    print("測試完成！")
    print("=" * 60)
