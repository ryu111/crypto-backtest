"""
5 階段策略驗證範例

展示如何使用 StageValidator 驗證策略有效性。
"""

import pandas as pd
from datetime import datetime, timedelta
from src.strategies.momentum.rsi import RSIStrategy
from src.validator.stages import StageValidator
from src.data.fetcher import DataFetcher


def main():
    """執行 5 階段驗證範例"""

    print("="*60)
    print("5 階段策略驗證範例")
    print("="*60)

    # 1. 準備資料
    print("\n[1/4] 載入市場資料...")
    fetcher = DataFetcher()

    # 載入 BTC 和 ETH 資料（至少 6 個月）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    data_btc = fetcher.fetch_binance(
        symbol='BTCUSDT',
        interval='1h',
        start_time=start_date,
        end_time=end_date
    )

    data_eth = fetcher.fetch_binance(
        symbol='ETHUSDT',
        interval='1h',
        start_time=start_date,
        end_time=end_date
    )

    print(f"BTC 資料: {len(data_btc)} 筆")
    print(f"ETH 資料: {len(data_eth)} 筆")

    # 2. 建立策略
    print("\n[2/4] 建立策略...")
    strategy = RSIStrategy()

    # 策略參數
    params = {
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
    }

    print(f"策略: {strategy.name}")
    print(f"參數: {params}")

    # 3. 執行 5 階段驗證
    print("\n[3/4] 執行 5 階段驗證...")
    print("-"*60)

    validator = StageValidator()

    result = validator.validate(
        strategy=strategy,
        data_btc=data_btc,
        data_eth=data_eth,
        params=params
    )

    # 4. 顯示結果
    print("\n[4/4] 驗證結果")
    print(result.summary())

    # 詳細資訊
    print("\n詳細指標:")
    print("-"*60)

    for stage_name, stage_result in result.stage_results.items():
        print(f"\n{stage_name}:")
        print(f"  通過: {'✓' if stage_result.passed else '✗'}")
        print(f"  分數: {stage_result.score:.1f}/100")

        # 顯示關鍵指標
        if 'details' in stage_result.__dict__:
            details = stage_result.details

            if stage_name == '階段1_基礎回測':
                print(f"  總報酬: {details.get('total_return', 0):.2%}")
                print(f"  總交易: {details.get('total_trades', 0)}")
                print(f"  夏普: {details.get('sharpe_ratio', 0):.2f}")
                print(f"  最大回撤: {details.get('max_drawdown', 0):.2%}")

            elif stage_name == '階段2_統計檢驗':
                print(f"  t-統計量: {details.get('t_statistic', 0):.2f}")
                print(f"  p-值: {details.get('p_value', 0):.4f}")
                print(f"  偏態: {details.get('skewness', 0):.2f}")

            elif stage_name == '階段3_穩健性':
                print(f"  參數敏感度: {details.get('param_sensitivity_pct', 0):.1f}%")
                print(f"  時間一致性: {'✓' if details.get('time_consistent', False) else '✗'}")
                print(f"  標的一致性: {'✓' if details.get('asset_consistent', False) else '✗'}")

            elif stage_name == '階段4_WalkForward':
                print(f"  WFA 效率: {details.get('efficiency', 0):.1%}")
                print(f"  OOS 勝率: {details.get('oos_win_rate', 0):.1%}")
                print(f"  最大 OOS 回撤: {details.get('max_oos_dd', 0):.2%}")

            elif stage_name == '階段5_MonteCarlo':
                print(f"  原始報酬: {details.get('original_return', 0):.2%}")
                print(f"  1% 分位: {details.get('p1', 0):.2%}")
                print(f"  5% 分位: {details.get('p5', 0):.2%}")
                print(f"  中位數: {details.get('median', 0):.2%}")

    # 決策建議
    print("\n" + "="*60)
    print("決策建議:")
    print("="*60)

    if result.grade.value in ['A', 'B']:
        print("✓ 策略驗證通過，可考慮實盤測試")
        print("  建議：")
        print("  - 使用小倉位開始")
        print("  - 持續監控實盤表現")
        print("  - 定期重新驗證")

    elif result.grade.value == 'C':
        print("△ 策略需要改進")
        print("  建議：")
        print("  - 優化參數提高穩健性")
        print("  - 延長回測期間")
        print("  - 暫緩實盤測試")

    else:
        print("✗ 策略不建議實盤")
        print("  建議：")
        print("  - 重新設計策略邏輯")
        print("  - 檢查資料品質")
        print("  - 尋找新的交易邊際")

    print("="*60)


def validate_multiple_strategies():
    """批次驗證多個策略"""

    from src.strategies.momentum.macd import MACDStrategy
    from src.strategies.trend.ma_cross import MovingAverageCrossStrategy

    strategies = [
        (RSIStrategy(), {'rsi_period': 14}),
        (MACDStrategy(), {'fast': 12, 'slow': 26}),
        (MovingAverageCrossStrategy(), {'fast': 10, 'slow': 30}),
    ]

    fetcher = DataFetcher()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    data_btc = fetcher.fetch_binance('BTCUSDT', '1h', start_date, end_date)
    data_eth = fetcher.fetch_binance('ETHUSDT', '1h', start_date, end_date)

    validator = StageValidator()
    results = []

    print("\n批次驗證多個策略")
    print("="*60)

    for strategy, params in strategies:
        print(f"\n驗證策略: {strategy.name}")

        result = validator.validate(
            strategy=strategy,
            data_btc=data_btc,
            data_eth=data_eth,
            params=params
        )

        results.append({
            'name': strategy.name,
            'grade': result.grade.value,
            'passed_stages': result.passed_stages,
            'params': params
        })

        print(f"評級: {result.grade.value}")
        print(f"通過階段: {result.passed_stages}/5")

    # 排序結果
    results.sort(key=lambda x: x['passed_stages'], reverse=True)

    print("\n" + "="*60)
    print("策略排名:")
    print("="*60)

    for i, r in enumerate(results, 1):
        print(f"{i}. {r['name']}")
        print(f"   評級: {r['grade']} | 通過階段: {r['passed_stages']}/5")
        print(f"   參數: {r['params']}")
        print()


if __name__ == '__main__':
    # 單一策略驗證
    main()

    # 批次驗證（可選）
    # validate_multiple_strategies()
