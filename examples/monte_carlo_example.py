"""
Monte Carlo 模擬範例

展示如何使用 MonteCarloSimulator 評估策略穩健性。
"""

import sys
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from src.validator import MonteCarloSimulator


def create_sample_trades(n_trades: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    建立範例交易記錄

    Args:
        n_trades: 交易數量
        seed: 隨機種子

    Returns:
        交易記錄 DataFrame
    """
    np.random.seed(seed)

    # 模擬交易 PnL（有正偏態，勝率 55%）
    win_prob = 0.55
    avg_win = 100
    avg_loss = -80

    pnl = []
    for _ in range(n_trades):
        if np.random.rand() < win_prob:
            # 獲利交易
            pnl.append(np.random.exponential(avg_win))
        else:
            # 虧損交易
            pnl.append(-np.random.exponential(-avg_loss))

    trades = pd.DataFrame({
        'pnl': pnl,
        'timestamp': pd.date_range('2024-01-01', periods=n_trades, freq='1H')
    })

    return trades


def example_basic_simulation():
    """範例 1：基本 Monte Carlo 模擬"""
    print("=" * 70)
    print("範例 1：基本 Monte Carlo 模擬")
    print("=" * 70)

    # 建立範例交易
    trades = create_sample_trades(100)

    # 建立模擬器
    simulator = MonteCarloSimulator(seed=42)

    # 執行模擬
    result = simulator.simulate(
        trades=trades,
        n_simulations=10000,
        method='shuffle'
    )

    # 輸出結果
    simulator.print_result(result)

    # 繪製分布圖
    simulator.plot_distribution(result)


def example_compare_methods():
    """範例 2：比較不同模擬方法"""
    print("\n" + "=" * 70)
    print("範例 2：比較不同模擬方法")
    print("=" * 70)

    # 建立範例交易
    trades = create_sample_trades(200)

    # 建立模擬器
    simulator = MonteCarloSimulator(seed=42)

    methods = ['shuffle', 'bootstrap', 'block_bootstrap']
    results = {}

    for method in methods:
        print(f"\n執行 {method} 模擬...")
        result = simulator.simulate(
            trades=trades,
            n_simulations=5000,
            method=method,
            block_size=10
        )
        results[method] = result

    # 比較結果
    print("\n" + "=" * 70)
    print("方法比較")
    print("=" * 70)
    print(f"{'指標':<20} {'Shuffle':>12} {'Bootstrap':>12} {'Block':>12}")
    print("-" * 70)

    metrics = [
        ('原始報酬', 'original_return'),
        ('平均報酬', 'mean'),
        ('標準差', 'std'),
        ('VaR (95%)', 'var_95'),
        ('CVaR (95%)', 'cvar_95'),
        ('獲利機率', 'probability_profitable'),
    ]

    for label, attr in metrics:
        values = [getattr(results[m], attr) for m in methods]
        if 'probability' in attr:
            print(f"{label:<20} {values[0]:>11.2%} {values[1]:>11.2%} {values[2]:>11.2%}")
        else:
            print(f"{label:<20} {values[0]:>12.2f} {values[1]:>12.2f} {values[2]:>12.2f}")


def example_equity_paths():
    """範例 3：權益曲線路徑模擬"""
    print("\n" + "=" * 70)
    print("範例 3：權益曲線路徑模擬")
    print("=" * 70)

    # 建立範例交易
    trades = create_sample_trades(150)

    # 建立模擬器
    simulator = MonteCarloSimulator(seed=42)

    # 產生權益曲線路徑
    equity_paths, original_path = simulator.generate_equity_paths(
        trades=trades,
        n_simulations=1000,
        method='bootstrap'
    )

    print(f"\n產生 {len(equity_paths)} 條權益曲線路徑")
    print(f"每條路徑長度: {len(original_path)}")
    print(f"\n原始最終報酬: {original_path[-1]:.2f}")
    print(f"模擬平均最終報酬: {equity_paths[:, -1].mean():.2f}")
    print(f"模擬最終報酬標準差: {equity_paths[:, -1].std():.2f}")

    # 繪製路徑
    simulator.plot_paths(
        equity_paths=equity_paths,
        original_path=original_path,
        n_paths_to_plot=200
    )


def example_risk_analysis():
    """範例 4：風險分析"""
    print("\n" + "=" * 70)
    print("範例 4：風險分析")
    print("=" * 70)

    # 建立範例交易（較高風險）
    np.random.seed(42)
    n_trades = 100

    # 高波動交易
    pnl = np.random.normal(loc=20, scale=150, size=n_trades)

    trades = pd.DataFrame({
        'pnl': pnl,
        'timestamp': pd.date_range('2024-01-01', periods=n_trades, freq='1H')
    })

    # 建立模擬器
    simulator = MonteCarloSimulator(seed=42)

    # 執行模擬
    result = simulator.simulate(
        trades=trades,
        n_simulations=10000,
        method='bootstrap'
    )

    # 風險分析
    print("\n風險分析")
    print("-" * 70)
    print(f"原始報酬:       {result.original_return:>12.2f}")
    print(f"預期報酬:       {result.mean:>12.2f}")
    print(f"標準差:         {result.std:>12.2f}")
    print()

    print("下行風險")
    print("-" * 70)
    print(f"VaR (95%):     {result.var_95:>12.2f}")
    print(f"CVaR (95%):    {result.cvar_95:>12.2f}")
    print(f"最差 1%:       {result.percentile_1:>12.2f}")
    print()

    print("上行潛力")
    print("-" * 70)
    print(f"最佳 1%:       {result.percentile_99:>12.2f}")
    print(f"第 95 百分位:  {result.percentile_95:>12.2f}")
    print()

    print("機率評估")
    print("-" * 70)
    print(f"獲利機率:       {result.probability_profitable:>11.2%}")
    print(f"超越原始機率:   {result.probability_beat_original:>11.2%}")

    # 計算其他風險指標
    loss_prob = 1 - result.probability_profitable
    expected_gain = result.simulated_returns[result.simulated_returns > 0].mean()
    expected_loss = result.simulated_returns[result.simulated_returns < 0].mean()

    print(f"虧損機率:       {loss_prob:>11.2%}")
    if not np.isnan(expected_gain):
        print(f"平均獲利:       {expected_gain:>12.2f}")
    if not np.isnan(expected_loss):
        print(f"平均虧損:       {expected_loss:>12.2f}")
        if expected_loss != 0:
            profit_loss_ratio = expected_gain / abs(expected_loss)
            print(f"盈虧比:         {profit_loss_ratio:>12.2f}")

    # 繪製分布
    simulator.plot_distribution(result)


def example_strategy_robustness():
    """範例 5：策略穩健性測試"""
    print("\n" + "=" * 70)
    print("範例 5：策略穩健性測試")
    print("=" * 70)

    # 建立兩種策略的交易記錄

    # 策略 A：穩定型（低波動）
    np.random.seed(42)
    trades_a = pd.DataFrame({
        'pnl': np.random.normal(loc=50, scale=30, size=200)
    })

    # 策略 B：激進型（高波動）
    np.random.seed(43)
    trades_b = pd.DataFrame({
        'pnl': np.random.normal(loc=50, scale=100, size=200)
    })

    simulator = MonteCarloSimulator(seed=42)

    # 執行模擬
    result_a = simulator.simulate(trades_a, n_simulations=5000, method='bootstrap')
    result_b = simulator.simulate(trades_b, n_simulations=5000, method='bootstrap')

    # 比較策略
    print("\n策略比較")
    print("-" * 70)
    print(f"{'指標':<25} {'策略 A (穩定)':>15} {'策略 B (激進)':>15}")
    print("-" * 70)

    comparisons = [
        ('原始報酬', 'original_return', False),
        ('預期報酬', 'mean', False),
        ('報酬標準差', 'std', False),
        ('VaR (95%)', 'var_95', False),
        ('CVaR (95%)', 'cvar_95', False),
        ('獲利機率', 'probability_profitable', True),
        ('超越原始機率', 'probability_beat_original', True),
    ]

    for label, attr, is_pct in comparisons:
        val_a = getattr(result_a, attr)
        val_b = getattr(result_b, attr)
        if is_pct:
            print(f"{label:<25} {val_a:>14.2%} {val_b:>14.2%}")
        else:
            print(f"{label:<25} {val_a:>15.2f} {val_b:>15.2f}")

    # 穩健性評估
    print("\n穩健性評估")
    print("-" * 70)

    # 計算變異係數（CV = std / mean）
    cv_a = result_a.std / abs(result_a.mean) if result_a.mean != 0 else float('inf')
    cv_b = result_b.std / abs(result_b.mean) if result_b.mean != 0 else float('inf')

    print(f"變異係數 (CV)")
    print(f"  策略 A: {cv_a:.4f}")
    print(f"  策略 B: {cv_b:.4f}")
    print(f"  {'策略 A 較穩定' if cv_a < cv_b else '策略 B 較穩定'}")
    print()

    # 計算下行偏差百分比
    downside_a = abs(result_a.cvar_95) / result_a.mean if result_a.mean > 0 else float('inf')
    downside_b = abs(result_b.cvar_95) / result_b.mean if result_b.mean > 0 else float('inf')

    print(f"下行風險比率 (CVaR / Mean)")
    print(f"  策略 A: {downside_a:.4f}")
    print(f"  策略 B: {downside_b:.4f}")
    print(f"  {'策略 A 下行風險較低' if downside_a < downside_b else '策略 B 下行風險較低'}")


def main():
    """執行所有範例"""
    examples = [
        ("基本模擬", example_basic_simulation),
        ("比較方法", example_compare_methods),
        ("權益路徑", example_equity_paths),
        ("風險分析", example_risk_analysis),
        ("策略穩健性", example_strategy_robustness),
    ]

    print("Monte Carlo 模擬範例")
    print("=" * 70)
    print("\n可用範例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    print("0. 執行所有範例")

    choice = input("\n請選擇要執行的範例 (0-5): ").strip()

    if choice == "0":
        for _, func in examples:
            func()
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        examples[int(choice) - 1][1]()
    else:
        print("無效選擇")


if __name__ == "__main__":
    main()
