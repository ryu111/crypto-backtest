"""
策略組合優化範例

展示如何使用 PortfolioOptimizer 進行多策略組合配置。
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加 src 到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimizer.portfolio import PortfolioOptimizer


def generate_sample_strategy_returns(n_strategies: int = 5, n_days: int = 252):
    """
    生成模擬策略回報資料

    Args:
        n_strategies: 策略數量
        n_days: 交易日數量

    Returns:
        策略回報 DataFrame
    """
    np.random.seed(42)

    returns_dict = {}

    # 模擬不同特性的策略
    strategies_config = [
        {'name': 'Trend Following', 'mean': 0.0008, 'std': 0.015},
        {'name': 'Mean Reversion', 'mean': 0.0006, 'std': 0.012},
        {'name': 'Breakout', 'mean': 0.0010, 'std': 0.018},
        {'name': 'Statistical Arb', 'mean': 0.0005, 'std': 0.008},
        {'name': 'Momentum', 'mean': 0.0009, 'std': 0.016}
    ]

    for i, config in enumerate(strategies_config[:n_strategies]):
        returns_dict[config['name']] = np.random.normal(
            config['mean'],
            config['std'],
            n_days
        )

    df = pd.DataFrame(returns_dict)

    # 添加一些策略間的相關性（更真實）
    if 'Trend Following' in df.columns and 'Momentum' in df.columns:
        df['Momentum'] = df['Momentum'] + 0.4 * df['Trend Following']

    if 'Mean Reversion' in df.columns and 'Statistical Arb' in df.columns:
        df['Statistical Arb'] = df['Statistical Arb'] + 0.3 * df['Mean Reversion']

    return df


def main():
    """主函數"""
    print("="*80)
    print("策略組合優化範例")
    print("="*80)

    # 1. 準備資料
    print("\n1. 生成模擬策略回報資料...")
    returns = generate_sample_strategy_returns(n_strategies=5, n_days=252)

    print(f"   策略數量: {len(returns.columns)}")
    print(f"   資料期間: {len(returns)} 天")
    print(f"   策略名稱: {list(returns.columns)}")

    # 顯示基本統計
    print("\n   策略績效統計（年化）:")
    print("   " + "-"*70)
    for col in returns.columns:
        annual_return = returns[col].mean() * 252
        annual_vol = returns[col].std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        print(f"   {col:20s} | 報酬: {annual_return*100:6.2f}% | "
              f"波動: {annual_vol*100:6.2f}% | Sharpe: {sharpe:6.3f}")

    # 2. 建立優化器
    print("\n2. 建立組合優化器...")
    optimizer = PortfolioOptimizer(
        returns=returns,
        risk_free_rate=0.0,  # 無風險利率
        frequency=252,        # 年化頻率（每日資料）
        use_ledoit_wolf=True  # 使用 Ledoit-Wolf 協方差估計
    )

    # 3. 相關性分析
    print("\n3. 策略相關性矩陣:")
    corr_matrix = optimizer.get_correlation_matrix()
    print(corr_matrix.round(3))

    # 4. 基準：等權重組合
    print("\n4. 基準組合 - 等權重配置:")
    print("   " + "-"*70)
    equal_weight = optimizer.equal_weight_portfolio()
    print(equal_weight.summary())

    # 5. 反波動率加權
    print("\n5. 反波動率加權組合:")
    print("   " + "-"*70)
    inv_vol = optimizer.inverse_volatility_portfolio()
    print(inv_vol.summary())

    # 6. 最大化 Sharpe Ratio
    print("\n6. 最大化 Sharpe Ratio 組合:")
    print("   " + "-"*70)
    max_sharpe = optimizer.max_sharpe_optimize(
        max_weight=0.5,  # 每個策略最多 50%
        min_weight=0.0   # 允許 0 權重
    )
    print(max_sharpe.summary())

    # 7. 風險平價配置
    print("\n7. 風險平價組合:")
    print("   " + "-"*70)
    risk_parity = optimizer.risk_parity_optimize(
        max_weight=0.6,
        min_weight=0.05  # 每個策略至少 5%
    )
    print(risk_parity.summary())

    # 8. Mean-Variance 優化（目標報酬）
    print("\n8. Mean-Variance 優化 (目標報酬 15%):")
    print("   " + "-"*70)
    target_return_portfolio = optimizer.mean_variance_optimize(
        target_return=0.15,  # 15% 年化報酬
        max_weight=0.6,
        min_weight=0.0
    )
    print(target_return_portfolio.summary())

    # 9. 計算效率前緣
    print("\n9. 計算效率前緣...")
    frontier = optimizer.efficient_frontier(
        n_points=20,
        max_weight=1.0,
        min_weight=0.0
    )
    print(f"   成功計算 {len(frontier)} 個前緣點")

    # 找出最大 Sharpe Ratio 的前緣點
    max_sharpe_frontier = max(frontier, key=lambda p: p.sharpe_ratio)
    print(f"\n   前緣上的最大 Sharpe Ratio 點:")
    print(f"   報酬: {max_sharpe_frontier.expected_return*100:.2f}%")
    print(f"   風險: {max_sharpe_frontier.expected_volatility*100:.2f}%")
    print(f"   Sharpe: {max_sharpe_frontier.sharpe_ratio:.4f}")

    # 10. 比較所有方法
    print("\n10. 方法比較:")
    print("   " + "-"*70)

    methods = [
        ('等權重', equal_weight),
        ('反波動率', inv_vol),
        ('最大 Sharpe', max_sharpe),
        ('風險平價', risk_parity),
        ('目標報酬 15%', target_return_portfolio)
    ]

    print(f"   {'方法':<20s} | {'報酬':>8s} | {'風險':>8s} | {'Sharpe':>8s}")
    print("   " + "-"*70)

    for name, portfolio in methods:
        print(f"   {name:<20s} | "
              f"{portfolio.expected_return*100:7.2f}% | "
              f"{portfolio.expected_volatility*100:7.2f}% | "
              f"{portfolio.sharpe_ratio:8.4f}")

    # 11. 視覺化（可選）
    print("\n11. 繪製效率前緣...")
    try:
        fig = optimizer.plot_efficient_frontier(
            frontier=frontier,
            save_path='examples/efficient_frontier.png',
            show_assets=True
        )
        if fig:
            print("   ✓ 效率前緣圖表已儲存至: examples/efficient_frontier.png")
    except Exception as e:
        print(f"   ✗ 繪圖失敗: {e}")

    print("\n" + "="*80)
    print("範例完成！")
    print("="*80)


if __name__ == '__main__':
    main()
