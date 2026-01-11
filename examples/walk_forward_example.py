"""
Walk-Forward 分析使用範例

展示如何使用 WalkForwardAnalyzer 驗證策略穩健性。
"""

import pandas as pd
from datetime import datetime

from src.backtester.engine import BacktestConfig
from src.optimizer import WalkForwardAnalyzer
from src.strategies.momentum.rsi import RSIStrategy


def main():
    """主程式"""

    # 1. 準備回測配置
    config = BacktestConfig(
        symbol='BTCUSDT',
        timeframe='1h',
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_capital=10000,
        leverage=3,
        maker_fee=0.0002,
        taker_fee=0.0004,
        slippage=0.0001
    )

    # 2. 載入資料（這裡使用假資料示範）
    # 實際使用時應從 DataFetcher 載入真實資料
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1h')
    data = pd.DataFrame({
        'open': 40000 + pd.Series(range(len(dates))) * 0.1,
        'high': 40100 + pd.Series(range(len(dates))) * 0.1,
        'low': 39900 + pd.Series(range(len(dates))) * 0.1,
        'close': 40000 + pd.Series(range(len(dates))) * 0.1,
        'volume': 100 + pd.Series(range(len(dates))) * 0.01
    }, index=dates)

    # 3. 建立策略
    strategy = RSIStrategy()

    # 4. 定義參數網格
    param_grid = {
        'rsi_period': [10, 14, 20],
        'rsi_oversold': [25, 30, 35],
        'rsi_overbought': [65, 70, 75]
    }

    # 5. 建立 Walk-Forward 分析器
    print("\n" + "="*60)
    print("Walk-Forward Analysis - Rolling Window Mode")
    print("="*60)

    analyzer = WalkForwardAnalyzer(
        config=config,
        mode='rolling',
        optimize_metric='sharpe_ratio'
    )

    # 6. 執行分析
    result = analyzer.analyze(
        strategy=strategy,
        data=data,
        param_grid=param_grid,
        n_windows=5,
        is_ratio=0.7,
        min_trades=10,
        verbose=True
    )

    # 7. 顯示結果
    print("\n" + result.summary())

    # 8. 分析衰退
    print("\n" + "="*60)
    print("Performance Degradation Analysis")
    print("="*60)

    degradation = analyzer.analyze_degradation(result)
    for metric, value in degradation.items():
        print(f"{metric:.<40} {value:>8.2%}")

    # 9. 判斷策略品質
    print("\n" + "="*60)
    print("Strategy Quality Assessment")
    print("="*60)

    # WFA 效率判斷基準
    if result.efficiency >= 0.9:
        efficiency_rating = "優秀 (Excellent)"
    elif result.efficiency >= 0.7:
        efficiency_rating = "良好 (Good)"
    elif result.efficiency >= 0.5:
        efficiency_rating = "普通 (Fair)"
    else:
        efficiency_rating = "不佳 (Poor)"

    # OOS 一致性判斷基準
    if result.consistency >= 0.7:
        consistency_rating = "優秀 (Excellent)"
    elif result.consistency >= 0.5:
        consistency_rating = "良好 (Good)"
    else:
        consistency_rating = "不佳 (Poor)"

    print(f"WFA Efficiency: {result.efficiency:.2%} - {efficiency_rating}")
    print(f"OOS Consistency: {result.consistency:.2%} - {consistency_rating}")

    if result.efficiency >= 0.7 and result.consistency >= 0.5:
        print("\n✓ 策略通過 Walk-Forward 驗證，可考慮實盤測試")
    else:
        print("\n✗ 策略未通過 Walk-Forward 驗證，需進一步優化")

    # 10. 繪製結果（可選）
    try:
        analyzer.plot_results(result, save_path='wfa_results.png')
    except ImportError:
        print("\n提示: 安裝 matplotlib 以繪製圖表 (pip install matplotlib)")

    # 11. 不同窗口模式比較
    print("\n" + "="*60)
    print("Comparing Different Window Modes")
    print("="*60)

    modes = ['rolling', 'expanding', 'anchored']
    mode_results = {}

    for mode in modes:
        print(f"\n分析 {mode} 模式...")
        analyzer_mode = WalkForwardAnalyzer(
            config=config,
            mode=mode,
            optimize_metric='sharpe_ratio'
        )

        try:
            mode_result = analyzer_mode.analyze(
                strategy=strategy,
                data=data,
                param_grid=param_grid,
                n_windows=3,  # 減少窗口數加快執行
                is_ratio=0.7,
                min_trades=5,
                verbose=False
            )
            mode_results[mode] = mode_result
        except Exception as e:
            print(f"  {mode} 模式失敗: {e}")
            continue

    # 比較結果
    if mode_results:
        print("\n模式比較:")
        print(f"{'Mode':<12} {'Efficiency':<12} {'Consistency':<12} {'OOS Mean':<12}")
        print("-"*60)
        for mode, res in mode_results.items():
            print(f"{mode:<12} {res.efficiency:<12.2%} {res.consistency:<12.2%} {res.oos_mean_return:<12.2%}")


def example_bayesian_optimization():
    """
    範例：與貝氏優化結合

    WFA 的參數網格可以先用貝氏優化縮小搜尋空間。
    """
    # 1. 使用貝氏優化找出有潛力的參數範圍
    # bayesian_optimizer = BayesianOptimizer(...)
    # best_params = bayesian_optimizer.optimize(...)

    # 2. 在該範圍內建立更細緻的網格
    # param_grid = {
    #     'rsi_period': [best_params['rsi_period'] - 2,
    #                    best_params['rsi_period'],
    #                    best_params['rsi_period'] + 2],
    #     ...
    # }

    # 3. 執行 WFA 驗證
    # wfa_result = analyzer.analyze(...)

    print("此範例需要先實作 BayesianOptimizer")


def example_multi_strategy_comparison():
    """
    範例：比較多個策略的 WFA 表現
    """
    # strategies = [RSIStrategy(), MACDStrategy(), SuperTrendStrategy()]
    # wfa_results = {}

    # for strategy in strategies:
    #     result = analyzer.analyze(strategy, ...)
    #     wfa_results[strategy.name] = result

    # 找出最佳策略
    # best_strategy = max(wfa_results.items(), key=lambda x: x[1].efficiency)

    print("此範例需要匯入多個策略類別")


if __name__ == '__main__':
    main()
