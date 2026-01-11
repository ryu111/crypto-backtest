"""
多策略相關性分析使用範例

展示如何使用 CorrelationAnalyzer 分析策略相關性。
"""

import numpy as np
import pandas as pd
from src.risk.correlation import CorrelationAnalyzer


def create_sample_strategies(n_days: int = 252):
    """建立範例策略收益率"""
    np.random.seed(42)

    # 市場因子（共同影響）
    market = np.random.randn(n_days) * 0.015

    # 策略 1: 趨勢跟隨（與市場正相關）
    trend = pd.Series(
        market * 0.8 + np.random.randn(n_days) * 0.005,
        name='trend_following'
    )

    # 策略 2: 均值回歸（與市場負相關）
    mean_revert = pd.Series(
        -market * 0.6 + np.random.randn(n_days) * 0.005,
        name='mean_reversion'
    )

    # 策略 3: 動能策略（與市場正相關但較強）
    momentum = pd.Series(
        market * 0.7 + np.random.randn(n_days) * 0.006,
        name='momentum'
    )

    # 策略 4: 統計套利（與市場低相關）
    stat_arb = pd.Series(
        market * 0.2 + np.random.randn(n_days) * 0.004,
        name='statistical_arbitrage'
    )

    return {
        'trend_following': trend,
        'mean_reversion': mean_revert,
        'momentum': momentum,
        'statistical_arbitrage': stat_arb
    }


def example_correlation_matrix():
    """範例 1: 計算策略相關性矩陣"""
    print("=" * 60)
    print("範例 1: 策略相關性矩陣")
    print("=" * 60)

    # 建立分析器
    analyzer = CorrelationAnalyzer(window=60)

    # 建立策略收益率
    strategies = create_sample_strategies()

    # 計算相關性矩陣
    result = analyzer.calculate_correlation_matrix(strategies)

    print("\n相關性矩陣:")
    print(result.matrix.round(3))

    print(f"\n平均相關性: {result.mean_correlation:.3f}")
    print(f"最大相關性: {result.max_correlation:.3f}")
    print(f"最小相關性: {result.min_correlation:.3f}")
    print(f"分散比率: {result.diversification_ratio:.3f}")

    # 解讀
    if result.mean_correlation < 0.3:
        print("\n✅ 策略組合分散良好（低相關性）")
    elif result.mean_correlation < 0.6:
        print("\n⚠️  策略組合有一定相關性")
    else:
        print("\n❌ 策略組合相關性過高，缺乏分散效果")


def example_rolling_correlation():
    """範例 2: 滾動相關性分析"""
    print("\n" + "=" * 60)
    print("範例 2: 滾動相關性分析")
    print("=" * 60)

    analyzer = CorrelationAnalyzer(window=60)
    strategies = create_sample_strategies()

    # 分析趨勢跟隨 vs 均值回歸
    rolling = analyzer.rolling_correlation(
        strategies['trend_following'],
        strategies['mean_reversion']
    )

    print(f"\n平均相關性: {rolling.mean:.3f}")
    print(f"相關性標準差: {rolling.std:.3f}")
    print(f"相關性範圍: [{rolling.min:.3f}, {rolling.max:.3f}]")
    print(f"趨勢變化次數: {rolling.regime_changes}")

    # 顯示最近的相關性
    print("\n最近 5 天的滾動相關性:")
    print(rolling.correlation.tail().round(3))

    if rolling.std > 0.3:
        print("\n⚠️  相關性波動較大，需要動態調整權重")
    else:
        print("\n✅ 相關性穩定")


def example_tail_correlation():
    """範例 3: 尾部相關性分析（危機時期）"""
    print("\n" + "=" * 60)
    print("範例 3: 尾部相關性分析")
    print("=" * 60)

    analyzer = CorrelationAnalyzer(window=60)
    strategies = create_sample_strategies(n_days=500)  # 更長的歷史

    # 分析動能 vs 趨勢跟隨的尾部相關性
    tail = analyzer.tail_correlation(
        strategies['momentum'],
        strategies['trend_following'],
        threshold=-0.02  # 2% 下跌閾值
    )

    print(f"\n正常時期相關性: {tail.normal:.3f}")
    print(f"左尾相關性（下跌時）: {tail.left_tail:.3f} (樣本數: {tail.left_tail_count})")
    print(f"右尾相關性（上漲時）: {tail.right_tail:.3f} (樣本數: {tail.right_tail_count})")
    print(f"危機相關性（極端下跌）: {tail.crisis_correlation:.3f}")

    # 危機時期相關性警報
    if tail.crisis_correlation > tail.normal + 0.2:
        print("\n❌ 警告：危機時期相關性大幅上升！")
        print("   → 在極端市場環境下，分散效果會減弱")
    else:
        print("\n✅ 策略在危機時期保持低相關性")


def example_portfolio_diversification():
    """範例 4: 投資組合分散分析"""
    print("\n" + "=" * 60)
    print("範例 4: 投資組合分散分析")
    print("=" * 60)

    analyzer = CorrelationAnalyzer(window=60)
    strategies = create_sample_strategies()

    # 方案 1: 等權重
    print("\n方案 1: 等權重配置")
    result1 = analyzer.analyze_portfolio_diversification(strategies)
    print(f"組合標準差: {result1['portfolio_std']:.4f}")
    print(f"加權平均標準差: {result1['weighted_avg_std']:.4f}")
    print(f"分散效益: {result1['diversification_benefit']:.2%}")

    # 方案 2: 趨勢策略為主
    print("\n方案 2: 趨勢策略為主")
    weights_trend = {
        'trend_following': 0.5,
        'mean_reversion': 0.2,
        'momentum': 0.2,
        'statistical_arbitrage': 0.1
    }
    result2 = analyzer.analyze_portfolio_diversification(strategies, weights_trend)
    print(f"組合標準差: {result2['portfolio_std']:.4f}")
    print(f"分散效益: {result2['diversification_benefit']:.2%}")

    # 方案 3: 均衡配置（利用負相關）
    print("\n方案 3: 均衡配置")
    weights_balanced = {
        'trend_following': 0.35,
        'mean_reversion': 0.35,  # 增加負相關策略
        'momentum': 0.15,
        'statistical_arbitrage': 0.15
    }
    result3 = analyzer.analyze_portfolio_diversification(strategies, weights_balanced)
    print(f"組合標準差: {result3['portfolio_std']:.4f}")
    print(f"分散效益: {result3['diversification_benefit']:.2%}")

    # 比較
    print("\n最佳方案:")
    benefits = {
        '等權重': result1['diversification_benefit'],
        '趨勢為主': result2['diversification_benefit'],
        '均衡配置': result3['diversification_benefit']
    }
    best = max(benefits, key=benefits.get)
    print(f"→ {best} (分散效益: {benefits[best]:.2%})")


def example_stress_test():
    """範例 5: 壓力測試 - 市場崩盤情境"""
    print("\n" + "=" * 60)
    print("範例 5: 壓力測試 - 市場崩盤情境")
    print("=" * 60)

    np.random.seed(42)
    n_days = 300
    strategies = create_sample_strategies(n_days)

    # 模擬市場崩盤（連續下跌）
    crash_start = 150
    crash_days = 10
    crash_magnitude = -0.05  # 5% 日跌幅

    for strat_name, returns in strategies.items():
        for i in range(crash_start, crash_start + crash_days):
            if 'mean_reversion' in strat_name:
                # 均值回歸策略在崩盤時可能受益
                returns.iloc[i] = -crash_magnitude * 0.3
            else:
                # 其他策略跟隨市場
                returns.iloc[i] = crash_magnitude * np.random.uniform(0.8, 1.2)

    analyzer = CorrelationAnalyzer(window=60)

    # 分析崩盤前後的相關性
    pre_crash = {k: v.iloc[:crash_start] for k, v in strategies.items()}
    during_crash = {k: v.iloc[crash_start:crash_start + crash_days] for k, v in strategies.items()}
    post_crash = {k: v.iloc[crash_start + crash_days:] for k, v in strategies.items()}

    print("\n崩盤前平均相關性:")
    result_pre = analyzer.calculate_correlation_matrix(pre_crash)
    print(f"  {result_pre.mean_correlation:.3f}")

    print("\n崩盤期間平均相關性:")
    result_during = analyzer.calculate_correlation_matrix(during_crash)
    print(f"  {result_during.mean_correlation:.3f}")

    print("\n崩盤後平均相關性:")
    result_post = analyzer.calculate_correlation_matrix(post_crash)
    print(f"  {result_post.mean_correlation:.3f}")

    # 相關性變化
    corr_change = result_during.mean_correlation - result_pre.mean_correlation
    print(f"\n崩盤期間相關性變化: {corr_change:+.3f}")

    if corr_change > 0.2:
        print("❌ 危機時期策略相關性大幅上升！")
    else:
        print("✅ 策略在危機時期保持分散")


if __name__ == '__main__':
    # 執行所有範例
    example_correlation_matrix()
    example_rolling_correlation()
    example_tail_correlation()
    example_portfolio_diversification()
    example_stress_test()

    print("\n" + "=" * 60)
    print("所有範例執行完成！")
    print("=" * 60)
