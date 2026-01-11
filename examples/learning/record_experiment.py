"""
實驗記錄器使用範例

展示如何使用 ExperimentRecorder 記錄回測結果、查詢實驗、分析演進。
"""

import sys
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.learning import ExperimentRecorder, Experiment
from src.validator.stages import ValidationGrade
from datetime import datetime


def example_log_experiment():
    """範例：記錄實驗"""
    print("\n===== 記錄實驗 =====\n")

    recorder = ExperimentRecorder()

    # 模擬回測結果（實際使用時會是 BacktestResult 物件）
    class MockBacktestResult:
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

    result = MockBacktestResult()

    # 策略資訊
    strategy_info = {
        'name': 'trend_ma_cross_4h_v2',
        'type': 'trend',
        'version': '2.0'
    }

    # 配置
    config = {
        'symbol': 'BTCUSDT',
        'timeframe': '4h',
        'period': {
            'start': '2024-01-01',
            'end': '2025-12-31'
        },
        'initial_capital': 10000,
        'leverage': 5
    }

    # 洞察
    insights = [
        '止損 2x ATR 在高波動期間表現更好',
        '慢線 30 優於 20，減少假訊號'
    ]

    # 記錄實驗
    exp_id = recorder.log_experiment(
        result=result,
        strategy_info=strategy_info,
        config=config,
        insights=insights
    )

    print(f"✓ 實驗已記錄: {exp_id}")


def example_log_with_validation():
    """範例：記錄帶驗證結果的實驗"""
    print("\n===== 記錄帶驗證的實驗 =====\n")

    recorder = ExperimentRecorder()

    # 模擬回測結果
    class MockBacktestResult:
        total_return = 0.52
        annual_return = 0.26
        sharpe_ratio = 2.1
        sortino_ratio = 2.5
        max_drawdown = -0.08
        win_rate = 0.58
        profit_factor = 1.95
        total_trades = 156
        avg_trade_duration = 10.2
        expectancy = 0.0042
        parameters = {'period': 14, 'multiplier': 3.0}

    # 模擬驗證結果
    from src.validator.stages import ValidationResult, StageResult

    stage1 = StageResult(
        passed=True,
        score=100.0,
        details={'total_return': 0.52},
        message="基礎績效符合要求",
        threshold={}
    )

    stage4 = StageResult(
        passed=True,
        score=85.0,
        details={'efficiency': 0.68},
        message="Walk-Forward 驗證通過",
        threshold={}
    )

    stage5 = StageResult(
        passed=True,
        score=90.0,
        details={'p5': 0.15},
        message="Monte Carlo 模擬通過，風險可控",
        threshold={}
    )

    validation = ValidationResult(
        grade=ValidationGrade.A,
        passed_stages=5,
        stage_results={
            '階段1_基礎回測': stage1,
            '階段4_WalkForward': stage4,
            '階段5_MonteCarlo': stage5,
        },
        recommendation="優秀！策略通過所有驗證階段。"
    )

    # 策略資訊
    strategy_info = {
        'name': 'supertrend_4h_v1',
        'type': 'trend',
        'version': '1.0'
    }

    config = {
        'symbol': 'BTCUSDT',
        'timeframe': '4h',
        'initial_capital': 10000,
        'leverage': 5
    }

    insights = [
        'Multiplier 3.0 在高波動市場表現穩定',
        'Period 14 相對 10 可減少假訊號'
    ]

    # 記錄實驗
    exp_id = recorder.log_experiment(
        result=MockBacktestResult(),
        strategy_info=strategy_info,
        config=config,
        validation_result=validation,
        insights=insights
    )

    print(f"✓ 實驗已記錄（含驗證）: {exp_id}")
    print(f"  驗證等級: {validation.grade.value}")
    print(f"  通過階段: {validation.passed_stages}/5")


def example_query_experiments():
    """範例：查詢實驗"""
    print("\n===== 查詢實驗 =====\n")

    recorder = ExperimentRecorder()

    # 1. 查詢所有趨勢策略
    print("1. 所有趨勢策略:")
    trend_exps = recorder.query_experiments({
        'strategy_type': 'trend'
    })
    print(f"   找到 {len(trend_exps)} 個趨勢策略實驗")

    # 2. 查詢 BTC 高 Sharpe 策略
    print("\n2. BTC Sharpe >= 1.5 的策略:")
    btc_high_sharpe = recorder.query_experiments({
        'symbol': 'BTCUSDT',
        'min_sharpe': 1.5
    })
    for exp in btc_high_sharpe[:3]:
        print(f"   {exp.id}: {exp.strategy['name']} (Sharpe: {exp.results['sharpe_ratio']:.2f})")

    # 3. 查詢驗證通過的策略
    print("\n3. 驗證通過（A/B 等級）:")
    validated = recorder.query_experiments({
        'grade': ['A', 'B']
    })
    for exp in validated[:3]:
        grade = exp.validation.get('grade', 'N/A')
        print(f"   {exp.id}: {exp.strategy['name']} (Grade: {grade})")

    # 4. 查詢最近一週的實驗
    print("\n4. 最近實驗:")
    from datetime import datetime, timedelta
    recent = recorder.query_experiments({
        'date_range': (
            datetime.now() - timedelta(days=7),
            datetime.now()
        )
    })
    print(f"   找到 {len(recent)} 個最近實驗")


def example_get_best():
    """範例：取得最佳策略"""
    print("\n===== 最佳策略 =====\n")

    recorder = ExperimentRecorder()

    # 按 Sharpe Ratio 排序
    print("1. Top 5 Sharpe Ratio:")
    best_sharpe = recorder.get_best_experiments('sharpe_ratio', n=5)
    for i, exp in enumerate(best_sharpe, 1):
        print(f"   {i}. {exp.strategy['name']}")
        print(f"      Sharpe: {exp.results['sharpe_ratio']:.2f}")
        print(f"      Return: {exp.results['total_return']:.1%}")

    # 按 Total Return 排序
    print("\n2. Top 5 Total Return:")
    best_return = recorder.get_best_experiments('total_return', n=5)
    for i, exp in enumerate(best_return, 1):
        print(f"   {i}. {exp.strategy['name']}")
        print(f"      Return: {exp.results['total_return']:.1%}")
        print(f"      Sharpe: {exp.results['sharpe_ratio']:.2f}")

    # 僅查詢驗證通過的最佳策略
    print("\n3. 驗證通過的最佳策略:")
    best_validated = recorder.get_best_experiments(
        'sharpe_ratio',
        n=5,
        filters={'grade': ['A', 'B']}
    )
    for i, exp in enumerate(best_validated, 1):
        grade = exp.validation.get('grade', 'N/A')
        print(f"   {i}. {exp.strategy['name']} (Grade: {grade})")
        print(f"      Sharpe: {exp.results['sharpe_ratio']:.2f}")


def example_strategy_evolution():
    """範例：追蹤策略演進"""
    print("\n===== 策略演進追蹤 =====\n")

    recorder = ExperimentRecorder()

    # 追蹤 MA 交叉策略的演進
    evolution = recorder.get_strategy_evolution('trend_ma_cross')

    if evolution:
        print(f"MA 交叉策略演進歷史 ({len(evolution)} 個版本):\n")

        for entry in evolution:
            print(f"版本 {entry['version']} - {entry['date'].strftime('%Y-%m-%d')}")
            print(f"  實驗 ID: {entry['exp_id']}")
            print(f"  Sharpe: {entry['sharpe']:.2f}")
            print(f"  Return: {entry['return']:.1%}")

            if entry['improvement'] is not None:
                improvement_pct = entry['improvement'] * 100
                direction = '↑' if entry['improvement'] > 0 else '↓'
                print(f"  改進: {direction} {abs(improvement_pct):.1f}%")

            if entry['changes']:
                print(f"  變更: {entry['changes'][0]}")
            print()
    else:
        print("尚無 MA 交叉策略的記錄")


def example_get_experiment():
    """範例：取得單一實驗"""
    print("\n===== 取得單一實驗 =====\n")

    recorder = ExperimentRecorder()

    # 先取得所有實驗
    all_exps = recorder.query_experiments()

    if all_exps:
        # 取得第一個實驗
        first_exp = all_exps[0]
        exp_id = first_exp.id

        print(f"查詢實驗: {exp_id}\n")

        # 重新取得實驗
        exp = recorder.get_experiment(exp_id)

        if exp:
            print(f"策略: {exp.strategy['name']}")
            print(f"類型: {exp.strategy['type']}")
            print(f"標的: {exp.config['symbol']}")
            print(f"時間框架: {exp.config['timeframe']}")
            print(f"\n績效:")
            print(f"  Total Return: {exp.results['total_return']:.1%}")
            print(f"  Sharpe Ratio: {exp.results['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {exp.results['max_drawdown']:.1%}")
            print(f"  Win Rate: {exp.results['win_rate']:.1%}")

            if exp.validation:
                print(f"\n驗證:")
                print(f"  等級: {exp.validation.get('grade', 'N/A')}")
                print(f"  通過階段: {exp.validation.get('passed_stages', 0)}/5")

            if exp.insights:
                print(f"\n洞察:")
                for insight in exp.insights:
                    print(f"  - {insight}")

            if exp.tags:
                print(f"\n標籤: {', '.join(exp.tags)}")
        else:
            print("找不到實驗")
    else:
        print("尚無實驗記錄")


if __name__ == '__main__':
    print("="*60)
    print("ExperimentRecorder 使用範例")
    print("="*60)

    # 執行範例
    example_log_experiment()
    example_log_with_validation()
    example_query_experiments()
    example_get_best()
    example_strategy_evolution()
    example_get_experiment()

    print("\n" + "="*60)
    print("範例執行完成")
    print("="*60)
