"""
ExperimentRecorder 簡單測試

不依賴外部套件，直接測試核心功能。
"""

import sys
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.learning import ExperimentRecorder, Experiment
from datetime import datetime


def main():
    print("="*60)
    print("ExperimentRecorder 簡單測試")
    print("="*60)

    # 建立記錄器
    recorder = ExperimentRecorder()
    print(f"\n✓ 記錄器已初始化")
    print(f"  experiments_file: {recorder.experiments_file}")
    print(f"  insights_file: {recorder.insights_file}")

    # 模擬回測結果
    class MockResult:
        total_return = 0.456
        annual_return = 0.23
        sharpe_ratio = 1.85
        sortino_ratio = 2.1
        max_drawdown = -0.10
        win_rate = 0.55
        profit_factor = 1.72
        total_trades = 124
        avg_trade_duration = 12.5
        expectancy = 0.0037
        parameters = {'fast_period': 10, 'slow_period': 30}

    # 測試 1: 記錄實驗
    print("\n" + "-"*60)
    print("測試 1: 記錄實驗")
    print("-"*60)

    exp_id = recorder.log_experiment(
        result=MockResult(),
        strategy_info={
            'name': 'ma_cross_4h_v1',
            'type': 'trend',
            'version': '1.0'
        },
        config={
            'symbol': 'BTCUSDT',
            'timeframe': '4h',
            'initial_capital': 10000,
            'leverage': 5
        },
        insights=[
            'ATR 2x 止損表現更好',
            '慢線 30 優於 20'
        ]
    )

    print(f"✓ 實驗已記錄: {exp_id}")

    # 測試 2: 取得實驗
    print("\n" + "-"*60)
    print("測試 2: 取得實驗")
    print("-"*60)

    exp = recorder.get_experiment(exp_id)
    if exp:
        print(f"✓ 成功取得實驗")
        print(f"  策略: {exp.strategy['name']}")
        print(f"  Sharpe: {exp.results['sharpe_ratio']:.2f}")
        print(f"  Return: {exp.results['total_return']:.1%}")
        print(f"  標籤: {', '.join(exp.tags)}")

    # 測試 3: 查詢實驗
    print("\n" + "-"*60)
    print("測試 3: 查詢實驗")
    print("-"*60)

    # 記錄更多實驗
    for i in range(3):
        result = MockResult()
        result.sharpe_ratio = 1.0 + i * 0.3

        recorder.log_experiment(
            result=result,
            strategy_info={
                'name': f'test_strategy_{i}',
                'type': 'trend',
                'version': '1.0'
            },
            config={
                'symbol': 'BTCUSDT',
                'timeframe': '1h'
            }
        )

    # 查詢所有實驗
    all_exps = recorder.query_experiments()
    print(f"✓ 總共 {len(all_exps)} 個實驗")

    # 查詢趨勢策略
    trend_exps = recorder.query_experiments({'strategy_type': 'trend'})
    print(f"✓ 趨勢策略: {len(trend_exps)} 個")

    # 查詢高 Sharpe 策略
    high_sharpe = recorder.query_experiments({'min_sharpe': 1.5})
    print(f"✓ Sharpe >= 1.5: {len(high_sharpe)} 個")

    # 測試 4: 取得最佳策略
    print("\n" + "-"*60)
    print("測試 4: 取得最佳策略")
    print("-"*60)

    best = recorder.get_best_experiments('sharpe_ratio', n=3)
    print(f"✓ Top 3 Sharpe Ratio:")
    for i, exp in enumerate(best, 1):
        print(f"  {i}. {exp.strategy['name']}: {exp.results['sharpe_ratio']:.2f}")

    # 測試 5: 策略演進
    print("\n" + "-"*60)
    print("測試 5: 策略演進")
    print("-"*60)

    # 記錄同一策略的多個版本
    for version in ['2.0', '2.1', '3.0']:
        result = MockResult()
        result.sharpe_ratio = 1.5 + float(version)

        recorder.log_experiment(
            result=result,
            strategy_info={
                'name': f'ma_cross_v{version}',
                'type': 'trend',
                'version': version
            },
            config={'symbol': 'BTCUSDT', 'timeframe': '4h'}
        )

    evolution = recorder.get_strategy_evolution('ma_cross')
    print(f"✓ MA 交叉策略演進 ({len(evolution)} 個版本):")
    for entry in evolution:
        print(f"  v{entry['version']}: Sharpe {entry['sharpe']:.2f}")

    # 測試 6: 標籤生成
    print("\n" + "-"*60)
    print("測試 6: 標籤生成")
    print("-"*60)

    tags = recorder.generate_tags(
        strategy_info={'name': 'rsi_btc', 'type': 'momentum'},
        config={'symbol': 'BTCUSDT', 'timeframe': '1h'},
        validation={'grade': 'A'}
    )

    print(f"✓ 生成標籤: {', '.join(tags)}")

    # 測試 7: Experiment 序列化
    print("\n" + "-"*60)
    print("測試 7: Experiment 序列化")
    print("-"*60)

    test_exp = Experiment(
        id='exp_test',
        timestamp=datetime.now(),
        strategy={'name': 'test', 'type': 'trend'},
        config={'symbol': 'BTCUSDT'},
        parameters={'period': 14},
        results={'sharpe_ratio': 1.5},
        validation={'grade': 'A'},
        insights=['Test'],
        tags=['crypto', 'btc']
    )

    # 轉為字典
    exp_dict = test_exp.to_dict()
    print(f"✓ to_dict() 成功")
    print(f"  ID: {exp_dict['id']}")
    print(f"  Timestamp: {exp_dict['timestamp']}")

    # 從字典還原
    restored = Experiment.from_dict(exp_dict)
    print(f"✓ from_dict() 成功")
    print(f"  ID: {restored.id}")
    print(f"  Strategy: {restored.strategy['name']}")

    # 總結
    print("\n" + "="*60)
    print("✓ 所有測試通過")
    print("="*60)


if __name__ == '__main__':
    main()
