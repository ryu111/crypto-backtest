"""
ExperimentRecorder 測試

測試實驗記錄器的核心功能。
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime

from src.learning import ExperimentRecorder, Experiment
from src.validator.stages import ValidationGrade


class MockBacktestResult:
    """模擬回測結果"""
    def __init__(self):
        self.total_return = 0.456
        self.annual_return = 0.23
        self.sharpe_ratio = 1.85
        self.sortino_ratio = 2.1
        self.max_drawdown = -0.10
        self.win_rate = 0.55
        self.profit_factor = 1.72
        self.total_trades = 124
        self.avg_trade_duration = 12.5
        self.expectancy = 0.0037
        self.parameters = {'fast': 10, 'slow': 30}


@pytest.fixture
def temp_recorder():
    """建立臨時記錄器"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        exp_file = tmpdir_path / 'experiments.json'
        insights_file = tmpdir_path / 'insights.md'

        recorder = ExperimentRecorder(
            experiments_file=exp_file,
            insights_file=insights_file
        )

        yield recorder


def test_init_files(temp_recorder):
    """測試檔案初始化"""
    assert temp_recorder.experiments_file.exists()
    assert temp_recorder.insights_file.exists()

    # 檢查初始內容
    data = json.loads(temp_recorder.experiments_file.read_text())
    assert data['metadata']['total_experiments'] == 0
    assert len(data['experiments']) == 0


def test_log_experiment(temp_recorder):
    """測試記錄實驗"""
    result = MockBacktestResult()
    strategy_info = {
        'name': 'test_strategy',
        'type': 'trend',
        'version': '1.0'
    }
    config = {
        'symbol': 'BTCUSDT',
        'timeframe': '1h'
    }

    exp_id = temp_recorder.log_experiment(
        result=result,
        strategy_info=strategy_info,
        config=config,
        insights=['Test insight']
    )

    # 驗證實驗 ID 格式
    assert exp_id.startswith('exp_')

    # 驗證記錄已保存
    data = json.loads(temp_recorder.experiments_file.read_text())
    assert data['metadata']['total_experiments'] == 1
    assert len(data['experiments']) == 1

    # 驗證實驗內容
    exp = data['experiments'][0]
    assert exp['id'] == exp_id
    assert exp['strategy']['name'] == 'test_strategy'
    assert exp['results']['sharpe_ratio'] == 1.85


def test_get_experiment(temp_recorder):
    """測試取得單一實驗"""
    result = MockBacktestResult()
    strategy_info = {'name': 'test', 'type': 'trend'}
    config = {'symbol': 'BTCUSDT', 'timeframe': '1h'}

    exp_id = temp_recorder.log_experiment(result, strategy_info, config)

    # 取得實驗
    exp = temp_recorder.get_experiment(exp_id)

    assert exp is not None
    assert exp.id == exp_id
    assert exp.strategy['name'] == 'test'
    assert exp.results['sharpe_ratio'] == 1.85


def test_query_experiments_empty(temp_recorder):
    """測試查詢空實驗列表"""
    exps = temp_recorder.query_experiments()
    assert len(exps) == 0


def test_query_experiments_with_filters(temp_recorder):
    """測試過濾查詢"""
    result = MockBacktestResult()

    # 記錄多個實驗
    temp_recorder.log_experiment(
        result,
        {'name': 'strategy1', 'type': 'trend'},
        {'symbol': 'BTCUSDT', 'timeframe': '1h'}
    )

    result.sharpe_ratio = 0.5  # 低 Sharpe
    temp_recorder.log_experiment(
        result,
        {'name': 'strategy2', 'type': 'momentum'},
        {'symbol': 'ETHUSDT', 'timeframe': '4h'}
    )

    # 過濾趨勢策略
    trend_exps = temp_recorder.query_experiments({
        'strategy_type': 'trend'
    })
    assert len(trend_exps) == 1
    assert trend_exps[0].strategy['name'] == 'strategy1'

    # 過濾 BTC 標的
    btc_exps = temp_recorder.query_experiments({
        'symbol': 'BTCUSDT'
    })
    assert len(btc_exps) == 1

    # 過濾 Sharpe >= 1.0
    high_sharpe = temp_recorder.query_experiments({
        'min_sharpe': 1.0
    })
    assert len(high_sharpe) == 1


def test_get_best_experiments(temp_recorder):
    """測試取得最佳實驗"""
    result = MockBacktestResult()

    # 記錄多個不同 Sharpe 的實驗
    sharpe_values = [1.5, 2.0, 0.8, 1.8, 1.2]

    for i, sharpe in enumerate(sharpe_values):
        result.sharpe_ratio = sharpe
        temp_recorder.log_experiment(
            result,
            {'name': f'strategy_{i}', 'type': 'trend'},
            {'symbol': 'BTCUSDT', 'timeframe': '1h'}
        )

    # 取得 Top 3
    best = temp_recorder.get_best_experiments('sharpe_ratio', n=3)

    assert len(best) == 3
    assert best[0].results['sharpe_ratio'] == 2.0
    assert best[1].results['sharpe_ratio'] == 1.8
    assert best[2].results['sharpe_ratio'] == 1.5


def test_generate_tags(temp_recorder):
    """測試標籤生成"""
    strategy_info = {
        'name': 'ma_cross_btc',
        'type': 'trend'
    }
    config = {
        'symbol': 'BTCUSDT',
        'timeframe': '4h'
    }
    validation = {
        'grade': 'A'
    }

    tags = temp_recorder.generate_tags(strategy_info, config, validation)

    assert 'crypto' in tags
    assert 'btc' in tags
    assert 'trend' in tags
    assert 'ma' in tags
    assert '4h' in tags
    assert 'validated' in tags


def test_strategy_evolution(temp_recorder):
    """測試策略演進追蹤"""
    result = MockBacktestResult()

    # 記錄同一策略的不同版本
    versions = ['1.0', '1.1', '2.0']
    sharpes = [1.5, 1.7, 2.0]

    for version, sharpe in zip(versions, sharpes):
        result.sharpe_ratio = sharpe
        temp_recorder.log_experiment(
            result,
            {'name': 'ma_cross_v' + version, 'type': 'trend', 'version': version},
            {'symbol': 'BTCUSDT', 'timeframe': '1h'}
        )

    # 追蹤演進
    evolution = temp_recorder.get_strategy_evolution('ma_cross')

    assert len(evolution) == 3
    assert evolution[0]['version'] == '1.0'
    assert evolution[1]['version'] == '1.1'
    assert evolution[2]['version'] == '2.0'

    # 檢查 Sharpe 遞增
    assert evolution[0]['sharpe'] == 1.5
    assert evolution[1]['sharpe'] == 1.7
    assert evolution[2]['sharpe'] == 2.0


def test_experiment_to_dict():
    """測試 Experiment 序列化"""
    exp = Experiment(
        id='exp_test',
        timestamp=datetime.now(),
        strategy={'name': 'test', 'type': 'trend'},
        config={'symbol': 'BTCUSDT'},
        parameters={'period': 14},
        results={'sharpe_ratio': 1.5},
        validation={'grade': 'A'},
        insights=['Test insight'],
        tags=['crypto', 'btc']
    )

    # 轉換為字典
    exp_dict = exp.to_dict()

    assert exp_dict['id'] == 'exp_test'
    assert exp_dict['strategy']['name'] == 'test'
    assert isinstance(exp_dict['timestamp'], str)  # 應該轉換為 ISO 格式


def test_experiment_from_dict():
    """測試 Experiment 反序列化"""
    exp_dict = {
        'id': 'exp_test',
        'timestamp': '2026-01-11T10:00:00',
        'strategy': {'name': 'test', 'type': 'trend'},
        'config': {'symbol': 'BTCUSDT'},
        'parameters': {'period': 14},
        'results': {'sharpe_ratio': 1.5},
        'validation': {'grade': 'A'},
        'insights': ['Test insight'],
        'tags': ['crypto', 'btc']
    }

    exp = Experiment.from_dict(exp_dict)

    assert exp.id == 'exp_test'
    assert exp.strategy['name'] == 'test'
    assert isinstance(exp.timestamp, datetime)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
