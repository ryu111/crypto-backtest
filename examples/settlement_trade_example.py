"""
Settlement Trade Strategy 使用範例

展示如何使用結算時段交易策略進行訊號生成。
"""

import sys
from pathlib import Path

# 加入專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.strategies.funding_rate import SettlementTradeStrategy
from src.strategies.registry import create_strategy


def create_sample_data(days=7):
    """
    建立範例數據（包含結算時間）

    Args:
        days: 數據天數

    Returns:
        tuple: (ohlcv_data, funding_rates)
    """
    # 建立小時級別的時間序列（7天 = 168 小時）
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = pd.date_range(
        start=start_time,
        periods=days * 24,
        freq='h',
        tz='UTC'
    )

    # 建立 OHLCV 數據
    np.random.seed(42)
    base_price = 40000

    data = pd.DataFrame({
        'open': base_price + np.random.normal(0, 500, len(timestamps)),
        'high': base_price + np.random.normal(200, 500, len(timestamps)),
        'low': base_price + np.random.normal(-200, 500, len(timestamps)),
        'close': base_price + np.random.normal(0, 500, len(timestamps)),
        'volume': np.random.uniform(1000, 5000, len(timestamps))
    }, index=timestamps)

    # 確保 high >= low
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)

    # 建立資金費率數據（模擬極端費率在結算前出現）
    funding_rates = pd.Series(
        np.random.normal(0.00005, 0.00003, len(timestamps)),
        index=timestamps
    )

    # 在一些結算前時段設定極端費率
    for i in range(len(timestamps)):
        hour = timestamps[i].hour

        # 08:00 結算前 1 小時（07:00）設定高費率
        if hour == 7:
            if np.random.random() > 0.5:
                funding_rates.iloc[i] = np.random.uniform(0.0002, 0.0005)

        # 16:00 結算前 1 小時（15:00）設定負費率
        elif hour == 15:
            if np.random.random() > 0.5:
                funding_rates.iloc[i] = np.random.uniform(-0.0005, -0.0002)

        # 00:00 結算前 1 小時（23:00）設定高費率
        elif hour == 23:
            if np.random.random() > 0.5:
                funding_rates.iloc[i] = np.random.uniform(0.0002, 0.0004)

    return data, funding_rates


def example_basic_usage():
    """範例 1: 基本使用"""
    print("=" * 60)
    print("範例 1: 基本使用")
    print("=" * 60)

    # 建立策略實例
    strategy = SettlementTradeStrategy(
        rate_threshold=0.0001,
        hours_before_settlement=1
    )

    print(f"策略: {strategy}")
    print(f"參數: {strategy.params}")
    print()

    # 建立範例數據
    data, funding_rates = create_sample_data(days=7)

    # 生成交易訊號
    long_entry, long_exit, short_entry, short_exit = \
        strategy.generate_signals_with_funding(data, funding_rates)

    # 統計訊號
    print(f"數據期間: {data.index[0]} 至 {data.index[-1]}")
    print(f"總 K 線數: {len(data)}")
    print()
    print("訊號統計:")
    print(f"  多單進場: {long_entry.sum()} 次")
    print(f"  多單出場: {long_exit.sum()} 次")
    print(f"  空單進場: {short_entry.sum()} 次")
    print(f"  空單出場: {short_exit.sum()} 次")
    print()

    # 顯示一些觸發訊號的時間點
    if long_entry.sum() > 0:
        print("多單進場時間點（前 5 個）:")
        entry_times = data.index[long_entry][:5]
        for t in entry_times:
            rate = funding_rates.loc[t]
            price = data.loc[t, 'close']
            print(f"  {t} | 費率: {rate:.6f} | 價格: {price:.2f}")
        print()


def example_registry_usage():
    """範例 2: 透過註冊表使用"""
    print("=" * 60)
    print("範例 2: 透過註冊表使用")
    print("=" * 60)

    # 透過註冊表建立策略
    strategy = create_strategy(
        'funding_rate_settlement',
        rate_threshold=0.0002,
        hours_before_settlement=2
    )

    print(f"策略: {strategy}")
    print(f"策略類型: {strategy.strategy_type}")
    print()

    # 建立範例數據
    data, funding_rates = create_sample_data(days=3)

    # 生成訊號
    long_entry, long_exit, short_entry, short_exit = \
        strategy.generate_signals_with_funding(data, funding_rates)

    print(f"訊號統計（3 天數據）:")
    print(f"  多單進場: {long_entry.sum()} 次")
    print(f"  空單進場: {short_entry.sum()} 次")
    print()


def example_parameter_testing():
    """範例 3: 測試不同參數"""
    print("=" * 60)
    print("範例 3: 測試不同參數組合")
    print("=" * 60)

    # 建立範例數據
    data, funding_rates = create_sample_data(days=30)

    # 測試不同參數組合
    param_combinations = [
        {'rate_threshold': 0.00005, 'hours_before_settlement': 1},
        {'rate_threshold': 0.0001, 'hours_before_settlement': 1},
        {'rate_threshold': 0.0002, 'hours_before_settlement': 1},
        {'rate_threshold': 0.0001, 'hours_before_settlement': 2},
        {'rate_threshold': 0.0001, 'hours_before_settlement': 3},
    ]

    print(f"數據期間: 30 天")
    print(f"總 K 線數: {len(data)}")
    print()
    print("不同參數組合的訊號數量:")
    print(f"{'閾值':<10} {'提前小時':<10} {'多單':<8} {'空單':<8} {'總訊號':<8}")
    print("-" * 60)

    for params in param_combinations:
        strategy = SettlementTradeStrategy(**params)

        long_entry, long_exit, short_entry, short_exit = \
            strategy.generate_signals_with_funding(data, funding_rates)

        total_signals = long_entry.sum() + short_entry.sum()

        print(
            f"{params['rate_threshold']:<10.5f} "
            f"{params['hours_before_settlement']:<10} "
            f"{long_entry.sum():<8} "
            f"{short_entry.sum():<8} "
            f"{total_signals:<8}"
        )

    print()


def example_settlement_timing():
    """範例 4: 驗證結算時間判斷"""
    print("=" * 60)
    print("範例 4: 結算時間判斷")
    print("=" * 60)

    strategy = SettlementTradeStrategy()

    # 測試不同時間點
    test_times = [
        datetime(2024, 1, 1, 0, 0, 0),   # 結算時間
        datetime(2024, 1, 1, 7, 0, 0),   # 結算前 1 小時
        datetime(2024, 1, 1, 8, 0, 0),   # 結算時間
        datetime(2024, 1, 1, 10, 0, 0),  # 非結算時間
        datetime(2024, 1, 1, 15, 0, 0),  # 結算前 1 小時
        datetime(2024, 1, 1, 16, 0, 0),  # 結算時間
        datetime(2024, 1, 1, 23, 0, 0),  # 結算前 1 小時
    ]

    print("結算時間判斷（hours_before=1）:")
    print(f"{'時間':<20} {'是否接近結算':<15} {'說明'}")
    print("-" * 60)

    for t in test_times:
        ts = pd.Timestamp(t, tz='UTC')
        is_settlement = strategy.is_settlement_hour(ts, hours_before=1)

        explanation = ""
        if t.hour in [0, 8, 16]:
            explanation = "✓ 結算時間"
        elif t.hour in [7, 15, 23]:
            explanation = "✓ 結算前 1 小時"
        else:
            explanation = "非結算期間"

        print(f"{str(t):<20} {str(is_settlement):<15} {explanation}")

    print()


def main():
    """執行所有範例"""
    print("\n" + "=" * 60)
    print("Settlement Trade Strategy 使用範例")
    print("=" * 60 + "\n")

    example_basic_usage()
    print()

    example_registry_usage()
    print()

    example_parameter_testing()
    print()

    example_settlement_timing()
    print()

    print("=" * 60)
    print("範例執行完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
