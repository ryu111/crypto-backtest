"""
型別系統使用範例

展示如何使用 src/types 模組進行型別安全的資料傳遞。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime
import json
from src.types import (
    BacktestConfig,
    BacktestResult,
    ValidationResult,
    ExperimentRecord,
    StrategyInfo,
    StrategyStats,
    ParamSpace,
    PerformanceMetrics,
    LoopConfig,
)


def example_1_basic_usage():
    """範例 1：基本使用"""
    print("=== 範例 1：基本使用 ===\n")

    # 建立配置
    config = BacktestConfig(
        symbol="BTCUSDT",
        timeframe="4h",
        start_date="2020-01-01",
        end_date="2024-01-01",
    )
    print(f"配置: {config.symbol} {config.timeframe}")
    print(f"期間: {config.start_date} ~ {config.end_date}\n")

    # 建立策略資訊
    strategy = StrategyInfo(
        name="trend_ma_cross",
        type="trend",
        version="1.0",
        params={'fast_period': 10, 'slow_period': 30}
    )
    print(f"策略: {strategy.name} ({strategy.type})")
    print(f"參數: {strategy.params}\n")


def example_2_serialization():
    """範例 2：序列化和反序列化"""
    print("=== 範例 2：序列化和反序列化 ===\n")

    # 建立配置
    config = BacktestConfig(
        symbol="ETHUSDT",
        timeframe="1h",
        start_date="2021-01-01",
        end_date="2024-01-01",
        initial_capital=10000,
        leverage=2,
    )

    # 序列化
    data = config.to_dict()
    print("序列化為 dict:")
    print(json.dumps(data, indent=2))

    # 反序列化
    config2 = BacktestConfig.from_dict(data)
    print(f"\n反序列化成功: {config.symbol == config2.symbol}")
    print(f"數據完整性: {config == config2}\n")


def example_3_backtest_result():
    """範例 3：回測結果記錄"""
    print("=== 範例 3：回測結果記錄 ===\n")

    # 建立績效指標
    metrics = PerformanceMetrics(
        sharpe_ratio=1.85,
        total_return=0.92,
        max_drawdown=0.18,
        win_rate=0.58,
        profit_factor=2.1,
        total_trades=245,
        sortino_ratio=2.3,
        calmar_ratio=5.1,
    )

    # 建立回測結果
    result = BacktestResult(
        metrics=metrics,
        execution_time=15.3,
    )

    print("績效指標:")
    print(f"  Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
    print(f"  Total Return: {result.metrics.total_return:.2%}")
    print(f"  Max Drawdown: {result.metrics.max_drawdown:.2%}")
    print(f"  Win Rate: {result.metrics.win_rate:.2%}")
    print(f"  Profit Factor: {result.metrics.profit_factor:.2f}")
    print(f"  Total Trades: {result.metrics.total_trades}")
    print(f"  執行時間: {result.execution_time:.1f}s\n")


def example_4_validation_result():
    """範例 4：驗證結果"""
    print("=== 範例 4：驗證結果 ===\n")

    # 建立驗證結果
    validation = ValidationResult(
        grade="B",
        stages_passed=[1, 2, 3],
        efficiency=0.88,
        overfit_probability=0.12,
    )

    print(f"評級: {validation.grade}")
    print(f"通過階段: {validation.stages_passed}")
    print(f"效率: {validation.efficiency:.2%}")
    print(f"過擬合機率: {validation.overfit_probability:.2%}")
    print(f"是否通過: {validation.is_passing}\n")


def example_5_complete_experiment():
    """範例 5：完整實驗記錄"""
    print("=== 範例 5：完整實驗記錄 ===\n")

    # 策略資訊
    strategy = StrategyInfo(
        name="trend_ma_cross",
        type="trend",
        version="1.0",
        params={'fast_period': 12, 'slow_period': 35}
    )

    # 配置
    config = BacktestConfig(
        symbol="BTCUSDT",
        timeframe="4h",
        start_date="2020-01-01",
        end_date="2024-01-01",
    )

    # 績效
    metrics = PerformanceMetrics(
        sharpe_ratio=2.1,
        total_return=1.35,
        max_drawdown=0.12,
        win_rate=0.62,
        profit_factor=2.5,
        total_trades=180,
    )
    result = BacktestResult(metrics=metrics, execution_time=10.2)

    # 驗證
    validation = ValidationResult(
        grade="A",
        stages_passed=[1, 2, 3, 4, 5],
        efficiency=0.95,
        overfit_probability=0.05,
    )

    # 建立實驗記錄
    now = datetime.now()
    experiment = ExperimentRecord(
        id=f"exp_{now.strftime('%Y%m%d_%H%M%S')}_{config.symbol}_{strategy.name}",
        timestamp=now,
        strategy=strategy.to_dict(),
        config=config.to_dict(),
        results=result.to_dict(),
        validation=validation.to_dict(),
        status="completed",
        insights=["快慢均線期間比 1:3 最佳", "ATR 2x 止損表現穩定"],
        tags=["optimized", "trend", "high_sharpe"],
    )

    print(f"實驗 ID: {experiment.id}")
    print(f"時間戳記: {experiment.timestamp}")
    print(f"策略: {experiment.strategy['name']}")
    print(f"配置: {experiment.config['symbol']} {experiment.config['timeframe']}")
    print(f"Sharpe Ratio: {experiment.sharpe_ratio:.2f}")
    print(f"評級: {experiment.validation['grade']}")
    print(f"是否成功: {experiment.is_success}")
    print(f"洞察: {experiment.insights}")
    print(f"標籤: {experiment.tags}\n")

    # 序列化（可存入 experiments.json）
    data = experiment.to_dict()
    print("JSON 格式（部分）:")
    print(json.dumps({k: v for k, v in data.items() if k in ['id', 'status']}, indent=2))
    print()


def example_6_strategy_stats():
    """範例 6：策略統計追蹤"""
    print("=== 範例 6：策略統計追蹤 ===\n")

    # 建立策略統計
    stats = StrategyStats(
        name="trend_ma_cross",
        attempts=0,
        successes=0,
    )

    # 模擬 5 次實驗
    experiments = [
        (1.2, True, {'fast_period': 10, 'slow_period': 30}),
        (1.8, True, {'fast_period': 12, 'slow_period': 35}),
        (0.8, False, {'fast_period': 5, 'slow_period': 20}),
        (2.1, True, {'fast_period': 15, 'slow_period': 40}),
        (1.5, True, {'fast_period': 10, 'slow_period': 35}),
    ]

    print("更新統計...")
    for sharpe, passed, params in experiments:
        stats.update_from_experiment(sharpe, passed, params)
        print(f"  Sharpe {sharpe:.1f}, 通過: {passed}")

    print(f"\n策略統計:")
    print(f"  總嘗試: {stats.attempts}")
    print(f"  成功次數: {stats.successes}")
    print(f"  成功率: {stats.success_rate:.2%}")
    print(f"  平均 Sharpe: {stats.avg_sharpe:.2f}")
    print(f"  最佳 Sharpe: {stats.best_sharpe:.2f}")
    print(f"  最差 Sharpe: {stats.worst_sharpe:.2f}")
    print(f"  最佳參數: {stats.best_params}")

    # 計算 UCB 評分
    ucb = stats.calculate_ucb(total_attempts=100, exploration_weight=2.0)
    print(f"  UCB 評分: {ucb:.2f}\n")


def example_7_param_space():
    """範例 7：參數空間定義和採樣"""
    print("=== 範例 7：參數空間定義和採樣 ===\n")

    # 定義參數空間
    param_space = ParamSpace(
        params={
            'fast_period': (5, 50, 'int'),
            'slow_period': (20, 200, 'int'),
            'atr_multiplier': (1.0, 3.0, 'float'),
        },
        constraints=[
            lambda p: p['fast_period'] < p['slow_period']
        ]
    )

    print("參數空間:")
    for name, (min_val, max_val, param_type) in param_space.params.items():
        print(f"  {name}: [{min_val}, {max_val}] ({param_type})")

    print("\n隨機採樣 5 組參數:")
    for i in range(5):
        params = param_space.sample_random()
        print(f"  {i+1}. {params}")
    print()


def example_8_loop_config():
    """範例 8：自動化循環配置"""
    print("=== 範例 8：自動化循環配置 ===\n")

    config = LoopConfig(
        max_iterations=100,
        stop_on_target=True,
        target_sharpe=2.0,
        exploit_ratio=0.8,
        symbols=["BTCUSDT", "ETHUSDT"],
        timeframes=["1h", "4h"],
        log_interval=10,
        save_checkpoints=True,
    )

    print("循環配置:")
    print(f"  最大迭代: {config.max_iterations}")
    print(f"  目標 Sharpe: {config.target_sharpe}")
    print(f"  Exploit 比例: {config.exploit_ratio:.0%}")
    print(f"  標的: {config.symbols}")
    print(f"  時間框架: {config.timeframes}")
    print(f"  記錄間隔: 每 {config.log_interval} 次")
    print(f"  儲存檢查點: {config.save_checkpoints}\n")


if __name__ == "__main__":
    # 執行所有範例
    example_1_basic_usage()
    example_2_serialization()
    example_3_backtest_result()
    example_4_validation_result()
    example_5_complete_experiment()
    example_6_strategy_stats()
    example_7_param_space()
    example_8_loop_config()

    print("=== 所有範例執行完畢 ===")
