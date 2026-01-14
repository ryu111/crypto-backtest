"""
統一型別系統測試

測試所有 src/types 模組的功能。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    OptimizationConfig,
)


def test_backtest_config():
    """測試 BacktestConfig"""
    print("測試 BacktestConfig...")

    # 建立
    config = BacktestConfig(
        symbol="BTCUSDT",
        timeframe="4h",
        start_date="2020-01-01",
        end_date="2024-01-01",
    )

    # 序列化
    data = config.to_dict()
    assert data['symbol'] == "BTCUSDT"
    assert data['timeframe'] == "4h"

    # 反序列化
    config2 = BacktestConfig.from_dict(data)
    assert config == config2

    print("  ✅ BacktestConfig 通過")


def test_performance_metrics():
    """測試 PerformanceMetrics"""
    print("測試 PerformanceMetrics...")

    metrics = PerformanceMetrics(
        sharpe_ratio=1.85,
        total_return=0.92,
        max_drawdown=0.18,
        win_rate=0.58,
        profit_factor=2.1,
        total_trades=245,
    )

    # 序列化
    data = metrics.to_dict()
    assert data['sharpe_ratio'] == 1.85

    # 反序列化
    metrics2 = PerformanceMetrics.from_dict(data)
    assert metrics == metrics2

    print("  ✅ PerformanceMetrics 通過")


def test_backtest_result():
    """測試 BacktestResult"""
    print("測試 BacktestResult...")

    metrics = PerformanceMetrics(
        sharpe_ratio=1.85,
        total_return=0.92,
        max_drawdown=0.18,
        win_rate=0.58,
        profit_factor=2.1,
        total_trades=245,
    )

    result = BacktestResult(
        metrics=metrics,
        execution_time=15.3,
    )

    # 序列化（不含 pd.Series）
    data = result.to_dict()
    assert 'sharpe_ratio' in data
    assert data['execution_time'] == 15.3

    # 反序列化
    result2 = BacktestResult.from_dict(data)
    assert result.metrics.sharpe_ratio == result2.metrics.sharpe_ratio

    print("  ✅ BacktestResult 通過")


def test_validation_result():
    """測試 ValidationResult"""
    print("測試 ValidationResult...")

    validation = ValidationResult(
        grade="B",
        stages_passed=[1, 2, 3],
        efficiency=0.88,
        overfit_probability=0.12,
    )

    # 檢查 is_passing 屬性
    assert validation.is_passing is True

    # 序列化
    data = validation.to_dict()
    assert data['grade'] == "B"

    # 反序列化
    validation2 = ValidationResult.from_dict(data)
    assert validation.grade == validation2.grade

    # 測試 F 評級
    validation_f = ValidationResult(grade="F", stages_passed=[])
    assert validation_f.is_passing is False

    print("  ✅ ValidationResult 通過")


def test_strategy_info():
    """測試 StrategyInfo"""
    print("測試 StrategyInfo...")

    strategy = StrategyInfo(
        name="trend_ma_cross",
        type="trend",
        version="1.0",
        params={'fast_period': 10, 'slow_period': 30}
    )

    # 序列化
    data = strategy.to_dict()
    assert data['name'] == "trend_ma_cross"
    assert data['params']['fast_period'] == 10

    # 反序列化
    strategy2 = StrategyInfo.from_dict(data)
    assert strategy.name == strategy2.name

    print("  ✅ StrategyInfo 通過")


def test_experiment_record():
    """測試 ExperimentRecord"""
    print("測試 ExperimentRecord...")

    now = datetime.now()

    experiment = ExperimentRecord(
        id="exp_test_001",
        timestamp=now,
        strategy={'name': 'test', 'type': 'trend', 'version': '1.0', 'params': {}},
        config={'symbol': 'BTCUSDT', 'timeframe': '4h', 'start_date': '2020-01-01', 'end_date': '2024-01-01'},
        results={'sharpe_ratio': 1.5, 'total_return': 0.85, 'max_drawdown': 0.15, 'win_rate': 0.55, 'profit_factor': 1.8, 'total_trades': 100},
        validation={'grade': 'B', 'stages_passed': [1, 2, 3]},
        status="completed",
    )

    # 測試屬性
    assert experiment.sharpe_ratio == 1.5
    assert experiment.is_success is True

    # 序列化
    data = experiment.to_dict()
    assert isinstance(data['timestamp'], str)  # datetime 轉為 ISO 格式

    # 反序列化
    experiment2 = ExperimentRecord.from_dict(data)
    assert experiment.id == experiment2.id

    print("  ✅ ExperimentRecord 通過")


def test_strategy_stats():
    """測試 StrategyStats"""
    print("測試 StrategyStats...")

    stats = StrategyStats(name="trend_ma_cross")

    # 初始狀態
    assert stats.attempts == 0
    assert stats.success_rate == 0.0

    # 更新統計
    stats.update_from_experiment(1.2, True, {'fast': 10, 'slow': 30})
    assert stats.attempts == 1
    assert stats.successes == 1
    assert stats.avg_sharpe == 1.2

    stats.update_from_experiment(1.8, True, {'fast': 12, 'slow': 35})
    assert stats.attempts == 2
    assert stats.successes == 2
    assert stats.avg_sharpe == 1.5  # (1.2 + 1.8) / 2
    assert stats.best_sharpe == 1.8

    stats.update_from_experiment(0.8, False, {'fast': 5, 'slow': 20})
    assert stats.attempts == 3
    assert stats.successes == 2
    assert stats.success_rate == 2/3

    # UCB 計算
    ucb = stats.calculate_ucb(total_attempts=100, exploration_weight=2.0)
    assert ucb > 0

    print("  ✅ StrategyStats 通過")


def test_param_space():
    """測試 ParamSpace"""
    print("測試 ParamSpace...")

    param_space = ParamSpace(
        params={
            'fast_period': (5, 50, 'int'),
            'slow_period': (20, 200, 'int'),
        },
        constraints=[
            lambda p: p['fast_period'] < p['slow_period']
        ]
    )

    # 測試隨機採樣
    for _ in range(10):
        params = param_space.sample_random()
        assert 5 <= params['fast_period'] <= 50
        assert 20 <= params['slow_period'] <= 200
        assert params['fast_period'] < params['slow_period']  # 約束

    # 序列化
    data = param_space.to_dict()
    assert 'params' in data

    print("  ✅ ParamSpace 通過")


def test_loop_config():
    """測試 LoopConfig"""
    print("測試 LoopConfig...")

    config = LoopConfig(
        max_iterations=100,
        target_sharpe=2.0,
        symbols=["BTCUSDT", "ETHUSDT"],
    )

    # 預設值
    assert config.exploit_ratio == 0.8
    assert config.optimization is not None

    # 序列化
    data = config.to_dict()
    assert data['max_iterations'] == 100

    # 反序列化
    config2 = LoopConfig.from_dict(data)
    assert config.max_iterations == config2.max_iterations

    print("  ✅ LoopConfig 通過")


def test_json_roundtrip():
    """測試完整的 JSON 往返"""
    print("測試 JSON 往返...")

    # 建立完整的實驗記錄
    strategy = StrategyInfo(
        name="trend_ma_cross",
        type="trend",
        params={'fast_period': 10, 'slow_period': 30}
    )

    config = BacktestConfig(
        symbol="BTCUSDT",
        timeframe="4h",
        start_date="2020-01-01",
        end_date="2024-01-01",
    )

    metrics = PerformanceMetrics(
        sharpe_ratio=1.85,
        total_return=0.92,
        max_drawdown=0.18,
        win_rate=0.58,
        profit_factor=2.1,
        total_trades=245,
    )
    result = BacktestResult(metrics=metrics)

    validation = ValidationResult(
        grade="B",
        stages_passed=[1, 2, 3],
    )

    experiment = ExperimentRecord(
        id="exp_test_json",
        timestamp=datetime.now(),
        strategy=strategy.to_dict(),
        config=config.to_dict(),
        results=result.to_dict(),
        validation=validation.to_dict(),
    )

    # 轉為 JSON
    json_str = json.dumps(experiment.to_dict())

    # 從 JSON 還原
    data = json.loads(json_str)
    experiment2 = ExperimentRecord.from_dict(data)

    # 驗證
    assert experiment.id == experiment2.id
    assert experiment.strategy['name'] == experiment2.strategy['name']
    assert experiment.sharpe_ratio == experiment2.sharpe_ratio

    print("  ✅ JSON 往返通過")


def run_all_tests():
    """執行所有測試"""
    print("\n" + "=" * 50)
    print("統一型別系統測試")
    print("=" * 50 + "\n")

    tests = [
        test_backtest_config,
        test_performance_metrics,
        test_backtest_result,
        test_validation_result,
        test_strategy_info,
        test_experiment_record,
        test_strategy_stats,
        test_param_space,
        test_loop_config,
        test_json_roundtrip,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {test.__name__} 失敗: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__} 錯誤: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"測試完成: {passed} 通過, {failed} 失敗")
    print("=" * 50 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
