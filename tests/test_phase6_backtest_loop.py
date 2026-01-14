"""
Phase 6 - Backtest Loop 模組測試

測試項目：
1. loop_config.py - BacktestLoopConfig, LoopResult, IterationSummary
2. validation_runner.py - ValidationRunner, ValidationResult, StageResult
3. backtest_loop.py - BacktestLoop, convenience functions
4. __init__.py - 模組匯出

測試策略：
- 使用 Mock 避免實際回測（專注邏輯測試）
- 快速執行（< 30 秒）
- 檢查核心邏輯正確性
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock, patch

# ===== 測試 1: 模組匯入 =====

def test_module_imports():
    """測試所有模組是否能正確匯入"""

    # 測試 loop_config 匯入
    from src.automation.loop_config import (
        BacktestLoopConfig,
        LoopResult,
        IterationSummary,
        SelectionMode,
        ValidationStage,
        create_default_config,
        create_quick_config,
        create_production_config
    )

    assert BacktestLoopConfig is not None
    assert LoopResult is not None
    assert IterationSummary is not None
    assert SelectionMode is not None
    assert ValidationStage is not None
    assert callable(create_default_config)
    assert callable(create_quick_config)
    assert callable(create_production_config)

    # 測試 validation_runner 匯入
    from src.automation.validation_runner import (
        ValidationRunner,
        ValidationResult,
        StageResult
    )

    assert ValidationRunner is not None
    assert ValidationResult is not None
    assert StageResult is not None

    # 測試 backtest_loop 匯入
    from src.automation.backtest_loop import (
        BacktestLoop,
        run_backtest_loop,
        quick_optimize,
        validate_strategy
    )

    assert BacktestLoop is not None
    assert callable(run_backtest_loop)
    assert callable(quick_optimize)
    assert callable(validate_strategy)

    # 測試 __init__ 匯出
    from src.automation import (
        BacktestLoopConfig,
        SelectionMode,
        ValidationStage,
        IterationSummary,
        LoopResult,
        ValidationRunner,
        ValidationResult,
        StageResult,
        BacktestLoop,
        run_backtest_loop,
        quick_optimize,
        validate_strategy
    )

    assert all([
        BacktestLoopConfig,
        SelectionMode,
        ValidationStage,
        IterationSummary,
        LoopResult,
        ValidationRunner,
        ValidationResult,
        StageResult,
        BacktestLoop,
        run_backtest_loop,
        quick_optimize,
        validate_strategy
    ])


# ===== 測試 2: BacktestLoopConfig 配置驗證 =====

def test_backtest_loop_config_defaults():
    """測試預設配置"""
    from src.automation.loop_config import BacktestLoopConfig

    config = BacktestLoopConfig()

    # 檢查預設值
    assert config.strategies == []
    assert config.symbols == ['BTCUSDT', 'ETHUSDT']
    # 預設包含多個時間框架
    expected_timeframes = [
        '1m', '3m', '5m', '15m', '30m',      # 短線
        '1h', '2h', '4h', '6h', '8h',        # 中線（8h 對齊資金費率）
        '12h', '1d', '3d', '1w'              # 長線
    ]
    assert config.timeframes == expected_timeframes
    assert config.n_iterations == 100
    assert config.selection_mode == 'epsilon_greedy'
    assert config.epsilon == 0.2
    assert config.ucb_c == 2.0
    assert config.validation_stages == [4, 5]
    assert config.min_sharpe == 1.0
    assert config.max_drawdown == 0.30
    assert config.min_trades == 30
    assert config.max_workers == 8
    assert config.use_gpu == True
    assert config.gpu_batch_size == 50
    assert config.n_trials == 100
    assert config.leverage == 5
    assert config.initial_capital == 10000.0

    # 配置應該有效
    config.validate()  # 不應拋出異常


def test_backtest_loop_config_validation():
    """測試配置驗證邏輯"""
    from src.automation.loop_config import BacktestLoopConfig

    # 測試無效的 symbols
    with pytest.raises(ValueError, match="symbols 不能為空"):
        config = BacktestLoopConfig(symbols=[])
        config.validate()

    # 測試無效的 timeframes
    with pytest.raises(ValueError, match="timeframes 不能為空"):
        config = BacktestLoopConfig(timeframes=[])
        config.validate()

    # 測試無效的 n_iterations
    with pytest.raises(ValueError, match="n_iterations 必須大於 0"):
        config = BacktestLoopConfig(n_iterations=0)
        config.validate()

    # 測試無效的 selection_mode
    with pytest.raises(ValueError, match="selection_mode 必須是"):
        config = BacktestLoopConfig(selection_mode='invalid_mode')
        config.validate()

    # 測試無效的 epsilon
    with pytest.raises(ValueError, match="epsilon 必須在"):
        config = BacktestLoopConfig(epsilon=1.5)
        config.validate()

    # 測試無效的 validation_stages
    with pytest.raises(ValueError, match="validation_stages 包含無效階段"):
        config = BacktestLoopConfig(validation_stages=[1, 2, 6])  # 6 不存在
        config.validate()

    # 測試無效的 min_sharpe
    with pytest.raises(ValueError, match="min_sharpe 必須大於等於 0"):
        config = BacktestLoopConfig(min_sharpe=-1.0)
        config.validate()

    # 測試無效的 max_drawdown
    with pytest.raises(ValueError, match="max_drawdown 必須在"):
        config = BacktestLoopConfig(max_drawdown=1.5)
        config.validate()

    # 測試無效的 leverage
    with pytest.raises(ValueError, match="leverage 必須大於 0"):
        config = BacktestLoopConfig(leverage=0)
        config.validate()


def test_backtest_loop_config_convenience_functions():
    """測試便利函數"""
    from src.automation.loop_config import (
        create_default_config,
        create_quick_config,
        create_production_config
    )

    # 測試預設配置
    default = create_default_config()
    default.validate()
    assert default.n_iterations == 100

    # 測試快速配置
    quick = create_quick_config(strategies=['ma_cross'], n_iterations=10)
    quick.validate()
    assert quick.strategies == ['ma_cross']
    assert quick.n_iterations == 10
    assert quick.symbols == ['BTCUSDT']
    assert quick.timeframes == ['1h']
    assert quick.validation_stages == [1, 4]  # 只執行基礎 + WF

    # 測試生產配置
    prod = create_production_config(strategies=['ma_cross', 'rsi'], n_iterations=100)
    prod.validate()
    assert prod.strategies == ['ma_cross', 'rsi']
    assert prod.n_iterations == 100
    assert prod.validation_stages == [1, 2, 3, 4, 5]  # 全部五個階段


# ===== 測試 3: LoopResult 和 IterationSummary =====

def test_iteration_summary():
    """測試 IterationSummary 資料類別"""
    from src.automation.loop_config import IterationSummary

    summary = IterationSummary(
        iteration=1,
        strategy_name='ma_cross',
        symbol='BTCUSDT',
        timeframe='1h',
        best_params={'fast_period': 10, 'slow_period': 30},
        sharpe_ratio=2.3,
        total_return=0.45,
        max_drawdown=0.12,
        validation_grade='A',
        duration_seconds=45.2,
        timestamp=datetime.now(),
        wf_sharpe=2.1,
        mc_p5=1.8,
        passed=True,
        experiment_id='exp_test'
    )

    assert summary.iteration == 1
    assert summary.strategy_name == 'ma_cross'
    assert summary.symbol == 'BTCUSDT'
    assert summary.sharpe_ratio == 2.3
    assert summary.passed == True


def test_loop_result_summary():
    """測試 LoopResult 摘要生成"""
    from src.automation.loop_config import LoopResult, IterationSummary

    # 建立假結果
    summaries = [
        IterationSummary(
            iteration=i,
            strategy_name='ma_cross',
            symbol='BTCUSDT',
            timeframe='1h',
            best_params={'fast_period': 10},
            sharpe_ratio=1.5 + i * 0.1,
            total_return=0.3 + i * 0.05,
            max_drawdown=0.15,
            validation_grade='A',
            duration_seconds=30.0,
            timestamp=datetime.now(),
            wf_sharpe=1.3 + i * 0.1,
            passed=True
        )
        for i in range(5)
    ]

    result = LoopResult(
        iterations_completed=5,
        total_iterations=10,
        best_strategies=summaries,
        failed_strategies=[],
        experiment_ids=['exp1', 'exp2'],
        duration_seconds=150.0,
        avg_sharpe=1.7,
        best_sharpe=1.9,
        avg_wf_sharpe=1.5,
        pass_rate=1.0,
        strategy_counts={'ma_cross': 5},
        strategy_win_rates={'ma_cross': 1.0}
    )

    # 測試 summary
    summary_text = result.summary()
    assert '完成迭代: 5 / 10' in summary_text
    assert '通過率: 100.0%' in summary_text
    assert '平均 Sharpe: 1.70' in summary_text
    assert '最佳 Sharpe: 1.90' in summary_text

    # 測試 to_dict
    result_dict = result.to_dict()
    assert result_dict['iterations_completed'] == 5
    assert result_dict['avg_sharpe'] == 1.7
    assert len(result_dict['best_strategies']) == 5


def test_loop_result_serialization():
    """測試 LoopResult 序列化/反序列化"""
    from src.automation.loop_config import LoopResult, IterationSummary

    # 建立原始結果
    original = LoopResult(
        iterations_completed=2,
        total_iterations=5,
        best_strategies=[
            IterationSummary(
                iteration=1,
                strategy_name='test',
                symbol='BTCUSDT',
                timeframe='1h',
                best_params={'p': 10},
                sharpe_ratio=1.5,
                total_return=0.3,
                max_drawdown=0.1,
                validation_grade='A',
                duration_seconds=10.0,
                timestamp=datetime.now(),
                wf_sharpe=1.3,
                passed=True
            )
        ],
        duration_seconds=20.0,
        avg_sharpe=1.5,
        best_sharpe=1.5,
        pass_rate=1.0
    )

    # 序列化
    data = original.to_dict()

    # 反序列化
    restored = LoopResult.from_dict(data)

    assert restored.iterations_completed == original.iterations_completed
    assert restored.avg_sharpe == original.avg_sharpe
    assert len(restored.best_strategies) == len(original.best_strategies)


# ===== 測試 4: ValidationRunner 邏輯測試 =====

@pytest.fixture
def mock_engine():
    """建立 Mock BacktestEngine"""
    engine = Mock()

    # Mock config
    engine.config = Mock()
    engine.config.initial_capital = 10000.0

    # Mock run 方法
    mock_result = Mock()
    mock_result.sharpe_ratio = 1.5
    mock_result.total_return = 0.35
    mock_result.max_drawdown = 0.15
    mock_result.total_trades = 50
    mock_result.win_rate = 0.6
    mock_result.volatility = 0.02

    # Mock daily_returns
    np.random.seed(42)
    mock_result.daily_returns = pd.Series(np.random.randn(100) * 0.01 + 0.001)

    # Mock trades
    mock_result.trades = pd.DataFrame({
        'PnL': np.random.randn(50) * 100 + 20
    })

    engine.run.return_value = mock_result

    return engine


@pytest.fixture
def mock_data():
    """建立假市場資料"""
    dates = pd.date_range('2020-01-01', periods=1000, freq='1h')
    data = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 102,
        'low': np.random.randn(1000).cumsum() + 98,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    return data


def test_validation_runner_basic(mock_engine, mock_data):
    """測試 ValidationRunner 基本流程"""
    from src.automation.validation_runner import ValidationRunner
    from src.strategies.base import BaseStrategy

    # Mock 策略
    strategy = Mock(spec=BaseStrategy)
    strategy.name = 'test_strategy'

    # 建立驗證器（只執行階段 1）
    runner = ValidationRunner(
        engine=mock_engine,
        stages=[1]
    )

    params = {'period': 20}

    # 執行驗證
    result = runner.validate(
        strategy=strategy,
        params=params,
        data=mock_data,
        symbol='BTCUSDT',
        timeframe='1h'
    )

    # 檢查結果
    assert result.strategy_name == 'test_strategy'
    assert result.params == params
    assert result.symbol == 'BTCUSDT'
    assert result.timeframe == '1h'
    assert result.sharpe_ratio == 1.5
    assert result.total_return == 0.35
    assert result.max_drawdown == 0.15
    assert len(result.stages) >= 1

    # 檢查階段 1 結果
    stage1 = result.stages[0]
    assert stage1.stage == 1
    assert stage1.name == 'Basic Backtest'
    assert stage1.passed == True  # 50 交易 > 10


def test_validation_runner_stage_1_basic(mock_engine, mock_data):
    """測試 Stage 1: 基礎回測驗證"""
    from src.automation.validation_runner import ValidationRunner
    from src.strategies.base import BaseStrategy

    strategy = Mock(spec=BaseStrategy)
    strategy.name = 'test'

    runner = ValidationRunner(engine=mock_engine, stages=[1])
    result = runner.validate(strategy, {}, mock_data, 'BTCUSDT', '1h')

    stage1 = result.stages[0]

    # 應該通過（50 交易 > 10, 0.15 DD < 0.9）
    assert stage1.passed == True
    assert stage1.details['total_trades'] == 50
    assert stage1.details['sharpe_ratio'] == 1.5


def test_validation_runner_stage_2_statistical(mock_engine, mock_data):
    """測試 Stage 2: 統計檢定"""
    from src.automation.validation_runner import ValidationRunner
    from src.strategies.base import BaseStrategy

    strategy = Mock(spec=BaseStrategy)
    strategy.name = 'test'

    runner = ValidationRunner(engine=mock_engine, stages=[2])
    result = runner.validate(strategy, {}, mock_data, 'BTCUSDT', '1h')

    stage2 = result.stages[0]

    # 檢查有執行 t-test
    assert stage2.stage == 2
    assert stage2.name == 'Statistical Test'
    assert 't_statistic' in stage2.details
    assert 'p_value' in stage2.details
    assert stage2.details['n_samples'] == 100


def test_validation_runner_grade_calculation():
    """測試評級邏輯"""
    from src.automation.validation_runner import ValidationRunner, StageResult

    runner = ValidationRunner(engine=Mock(), stages=[])

    # 測試 F 級（未通過基礎驗證）
    results = [
        StageResult(stage=1, name='Basic', passed=False, score=0.0, details={})
    ]
    grade = runner._calculate_grade(results)
    assert grade == 'F'

    # 測試 A 級（WF Sharpe > 1.0, MC P5 > 0.5）
    results = [
        StageResult(stage=1, name='Basic', passed=True, score=1.0, details={}),
        StageResult(stage=4, name='WF', passed=True, score=1.0, details={'oos_mean_sharpe': 1.5}),
        StageResult(stage=5, name='MC', passed=True, score=1.0, details={'p5_sharpe': 0.6})
    ]
    grade = runner._calculate_grade(results)
    assert grade == 'A'

    # 測試 B 級（WF Sharpe > 0.5, MC P5 > 0.3）
    results[1].details['oos_mean_sharpe'] = 0.7
    results[2].details['p5_sharpe'] = 0.4
    grade = runner._calculate_grade(results)
    assert grade == 'B'

    # 測試 C 級
    results[1].details['oos_mean_sharpe'] = 0.4
    grade = runner._calculate_grade(results)
    assert grade == 'C'


# ===== 測試 5: BacktestLoop Context Manager =====

def test_backtest_loop_context_manager():
    """測試 BacktestLoop Context Manager"""
    from src.automation.loop_config import create_quick_config
    from src.automation.backtest_loop import BacktestLoop

    config = create_quick_config(
        strategies=['ma_cross'],
        n_iterations=1,
        use_gpu=False
    )

    # 測試 __enter__ / __exit__
    with patch('src.automation.backtest_loop.ExperimentRecorder'):
        with patch('src.automation.backtest_loop.StrategySelector'):
            with patch('src.automation.backtest_loop.BacktestEngine'):
                with BacktestLoop(config) as loop:
                    # 檢查內部元件已初始化
                    assert loop._engine is not None
                    assert loop._selector is not None
                    assert loop._recorder is not None

                # 退出後應該清理
                assert loop._engine is None
                assert loop._selector is None


def test_backtest_loop_state_management():
    """測試 BacktestLoop 狀態管理"""
    from src.automation.loop_config import create_quick_config
    from src.automation.backtest_loop import BacktestLoop

    config = create_quick_config(n_iterations=5)

    with patch('src.automation.backtest_loop.ExperimentRecorder'):
        with patch('src.automation.backtest_loop.StrategySelector'):
            with patch('src.automation.backtest_loop.BacktestEngine'):
                with BacktestLoop(config) as loop:
                    # 初始狀態
                    assert loop.is_running == False
                    assert loop.is_paused == False
                    assert loop.current_iteration == 0

                    # 測試狀態控制
                    loop.pause()
                    assert loop.is_paused == True

                    loop.resume()
                    assert loop.is_paused == False

                    loop.stop()
                    assert loop.is_running == False


# ===== 測試 6: 整合測試（簡化版）=====

@patch('src.automation.validation_runner.ValidationRunner')
@patch('src.automation.backtest_loop.BacktestEngine')
@patch('src.automation.backtest_loop.StrategySelector')
@patch('src.automation.backtest_loop.ExperimentRecorder')
def test_backtest_loop_integration(
    mock_recorder,
    mock_selector,
    mock_engine,
    mock_validation_runner
):
    """測試 BacktestLoop 整合流程（使用 Mock）"""
    from src.automation.loop_config import create_quick_config
    from src.automation.backtest_loop import BacktestLoop
    from src.automation.validation_runner import ValidationResult, StageResult

    # 設定 Mock
    mock_selector_instance = Mock()
    mock_selector_instance.select.return_value = 'ma_cross'
    mock_selector.return_value = mock_selector_instance

    mock_engine_instance = Mock()
    mock_engine_instance.config = Mock()
    mock_engine_instance.config.symbol = 'BTCUSDT'
    mock_engine_instance.config.timeframe = '1h'

    # Mock run 方法
    mock_backtest_result = Mock()
    mock_backtest_result.sharpe_ratio = 1.8
    mock_backtest_result.total_return = 0.4
    mock_backtest_result.max_drawdown = 0.12
    mock_backtest_result.total_trades = 60
    mock_backtest_result.win_rate = 0.65
    mock_backtest_result.volatility = 0.02
    mock_engine_instance.run.return_value = mock_backtest_result

    # Mock load_data
    mock_data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100
    })
    mock_engine_instance.load_data.return_value = mock_data

    mock_engine.return_value = mock_engine_instance

    # Mock ValidationRunner
    mock_validator_instance = Mock()
    mock_validation_result = ValidationResult(
        strategy_name='ma_cross',
        params={'fast': 10, 'slow': 30},
        symbol='BTCUSDT',
        timeframe='1h',
        sharpe_ratio=1.8,
        total_return=0.4,
        max_drawdown=0.12,
        stages=[
            StageResult(stage=1, name='Basic', passed=True, score=1.0, details={})
        ],
        grade='A',
        passed=True,
        wf_sharpe=1.6,
        mc_p5_sharpe=1.4
    )
    mock_validator_instance.validate.return_value = mock_validation_result
    mock_validation_runner.return_value = mock_validator_instance

    # Mock StrategyRegistry
    with patch('src.automation.backtest_loop.StrategyRegistry') as mock_registry:
        mock_strategy_class = Mock()
        mock_strategy_class.param_space = {
            'fast_period': {'type': 'int', 'low': 5, 'high': 20},
            'slow_period': {'type': 'int', 'low': 20, 'high': 50}
        }
        mock_registry.get.return_value = mock_strategy_class

        # Mock create_strategy
        with patch('src.strategies.create_strategy') as mock_create:
            mock_strategy = Mock()
            mock_strategy.name = 'ma_cross'
            mock_create.return_value = mock_strategy

            # 執行測試
            config = create_quick_config(
                strategies=['ma_cross'],
                n_iterations=2,
                use_gpu=False
            )

            with BacktestLoop(config) as loop:
                result = loop.run()

            # 驗證結果
            assert result.iterations_completed == 2
            assert result.total_iterations == 2
            assert result.pass_rate > 0  # 應該有通過的策略
            assert len(result.best_strategies) > 0


# ===== 測試 7: 便利函數 =====

def test_validate_strategy_placeholder():
    """測試 validate_strategy 佔位符實作"""
    from src.automation.backtest_loop import validate_strategy

    result = validate_strategy(
        'ma_cross',
        params={'fast': 10, 'slow': 30},
        symbol='BTCUSDT',
        timeframe='1h'
    )

    # 檢查回傳格式
    assert 'passed' in result
    assert 'grade' in result
    assert 'sharpe_ratio' in result
    assert 'max_drawdown' in result
    assert 'params' in result
    assert 'validation_details' in result

    # 檢查參數
    assert result['params'] == {'fast': 10, 'slow': 30}


# ===== 執行測試報告 =====

if __name__ == '__main__':
    import sys

    # 執行測試並產生報告
    print("=" * 70)
    print("Phase 6 - Backtest Loop 模組測試")
    print("=" * 70)

    # 執行 pytest
    exit_code = pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--color=yes',
        '-k', 'test_'  # 執行所有 test_ 開頭的函數
    ])

    sys.exit(exit_code)
