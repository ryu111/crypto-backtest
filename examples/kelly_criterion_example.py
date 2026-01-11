"""
Kelly Criterion Position Sizing 使用範例

展示如何使用 Kelly Criterion 計算最佳部位大小。
"""

import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.risk import kelly_criterion, KellyPositionSizer


def example_basic_kelly():
    """範例 1: 基本 Kelly Criterion 計算"""
    print("=" * 60)
    print("範例 1: 基本 Kelly Criterion 計算")
    print("=" * 60)

    # 策略統計
    win_rate = 0.55  # 55% 勝率
    win_loss_ratio = 1.5  # 盈虧比 1.5:1

    # 計算最佳資金比例
    optimal_fraction = kelly_criterion(win_rate, win_loss_ratio)

    print(f"勝率: {win_rate:.1%}")
    print(f"盈虧比: {win_loss_ratio:.2f}")
    print(f"最佳資金比例 (Full Kelly): {optimal_fraction:.2%}")
    print()


def example_half_kelly():
    """範例 2: Half Kelly (推薦用於實際交易)"""
    print("=" * 60)
    print("範例 2: Half Kelly Position Sizing")
    print("=" * 60)

    # 初始化 Half Kelly Sizer
    sizer = KellyPositionSizer(
        kelly_fraction=0.5,  # Half Kelly
        max_position_fraction=0.25,  # 最大 25% 部位
        min_win_rate=0.45,  # 最低 45% 勝率
        min_win_loss_ratio=1.2  # 最低盈虧比 1.2
    )

    # 計算部位大小
    result = sizer.calculate_position_size(
        capital=100000,  # 10 萬 USD
        win_rate=0.58,
        avg_win=350,  # 平均獲利 $350
        avg_loss=200,  # 平均虧損 $200
        enforce_min_requirements=True
    )

    print(result)
    print()


def example_from_trade_history():
    """範例 3: 從交易歷史計算"""
    print("=" * 60)
    print("範例 3: 從真實交易歷史計算部位大小")
    print("=" * 60)

    # 真實交易記錄 (USD)
    winning_trades = [
        320, 180, 450, 220, 280,  # 前 5 筆獲利
        310, 190, 380, 240, 160   # 後 5 筆獲利
    ]

    losing_trades = [
        150, 120, 180, 110, 140,  # 5 筆虧損
    ]

    # 初始化 Sizer
    sizer = KellyPositionSizer(kelly_fraction=0.5)

    # 從交易記錄計算
    result = sizer.calculate_from_trades(
        capital=50000,  # 5 萬 USD
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        enforce_min_requirements=False
    )

    print(f"總交易數: {len(winning_trades) + len(losing_trades)}")
    print(f"獲利交易: {len(winning_trades)} 筆")
    print(f"虧損交易: {len(losing_trades)} 筆")
    print(f"平均獲利: ${sum(winning_trades) / len(winning_trades):.2f}")
    print(f"平均虧損: ${sum(losing_trades) / len(losing_trades):.2f}")
    print()
    print(result)
    print()


def example_conservative_vs_aggressive():
    """範例 4: 保守 vs 激進策略比較"""
    print("=" * 60)
    print("範例 4: 不同風險偏好比較")
    print("=" * 60)

    # 策略績效
    capital = 100000
    win_rate = 0.6
    avg_win = 400
    avg_loss = 250

    # Quarter Kelly (保守)
    conservative = KellyPositionSizer(kelly_fraction=0.25)
    result_conservative = conservative.calculate_position_size(
        capital, win_rate, avg_win, avg_loss, enforce_min_requirements=False
    )

    # Half Kelly (平衡)
    balanced = KellyPositionSizer(kelly_fraction=0.5)
    result_balanced = balanced.calculate_position_size(
        capital, win_rate, avg_win, avg_loss, enforce_min_requirements=False
    )

    # Full Kelly (激進)
    aggressive = KellyPositionSizer(kelly_fraction=1.0)
    result_aggressive = aggressive.calculate_position_size(
        capital, win_rate, avg_win, avg_loss, enforce_min_requirements=False
    )

    print(f"策略績效: 勝率 {win_rate:.1%}, 盈虧比 {avg_win/avg_loss:.2f}")
    print()
    print(f"Quarter Kelly: {result_conservative.position_size:,.0f} USD "
          f"({result_conservative.optimal_fraction:.1%})")
    print(f"Half Kelly:    {result_balanced.position_size:,.0f} USD "
          f"({result_balanced.optimal_fraction:.1%})")
    print(f"Full Kelly:    {result_aggressive.position_size:,.0f} USD "
          f"({result_aggressive.optimal_fraction:.1%})")
    print()
    print("建議: Half Kelly 提供風險與報酬的最佳平衡")
    print()


def example_dynamic_adjustment():
    """範例 5: 根據績效動態調整"""
    print("=" * 60)
    print("範例 5: 根據績效動態調整 Kelly 乘數")
    print("=" * 60)

    # 初始設定
    sizer = KellyPositionSizer(kelly_fraction=0.5)
    capital = 100000

    print(f"初始設定: {sizer.kelly_type}")

    # 階段 1: 策略表現良好
    print("\n[階段 1] 策略表現穩定，保持 Half Kelly")
    result1 = sizer.calculate_position_size(
        capital, win_rate=0.58, avg_win=350, avg_loss=200,
        enforce_min_requirements=False
    )
    print(f"建議部位: ${result1.position_size:,.0f}")

    # 階段 2: 策略進入回撤，降低風險
    print("\n[階段 2] 策略進入回撤期，降低至 Quarter Kelly")
    sizer.adjust_kelly_fraction(0.25)
    result2 = sizer.calculate_position_size(
        capital * 0.95,  # 資金略微下降
        win_rate=0.52, avg_win=300, avg_loss=220,
        enforce_min_requirements=False
    )
    print(f"建議部位: ${result2.position_size:,.0f}")

    # 階段 3: 策略恢復，提高至 Half Kelly
    print("\n[階段 3] 策略恢復表現，提高至 Half Kelly")
    sizer.adjust_kelly_fraction(0.5)
    result3 = sizer.calculate_position_size(
        capital * 1.05,  # 資金增長
        win_rate=0.60, avg_win=380, avg_loss=200,
        enforce_min_requirements=False
    )
    print(f"建議部位: ${result3.position_size:,.0f}")
    print()


def main():
    """執行所有範例"""
    example_basic_kelly()
    example_half_kelly()
    example_from_trade_history()
    example_conservative_vs_aggressive()
    example_dynamic_adjustment()

    print("=" * 60)
    print("重要提醒")
    print("=" * 60)
    print("1. Kelly Criterion 假設無限次重複交易")
    print("2. 實際使用建議 Half Kelly 或更保守")
    print("3. 需要配合嚴格的風險管理（止損、資金管理）")
    print("4. 過去績效不代表未來表現")
    print("5. 建議定期檢視並調整參數")
    print()


if __name__ == "__main__":
    main()
