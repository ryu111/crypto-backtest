"""
ExperimentRecorder 測試

測試實驗記錄器的核心功能。
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from src.learning import ExperimentRecorder, Experiment
from src.validator.stages import ValidationGrade


class MockBacktestResult:
    """模擬回測結果"""
    def __init__(self, with_timeseries=True):
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

        # 時間序列資料（新增）
        if with_timeseries:
            dates = pd.date_range('2024-01-01', periods=100, freq='1h')
            self.equity_curve = pd.Series(
                np.cumsum(np.random.randn(100) * 0.01) + 10000,
                index=dates
            )
            self.daily_returns = pd.Series(
                np.random.randn(100) * 0.01,
                index=dates
            )
            self.trades = pd.DataFrame({
                'entry_time': dates[:10],
                'exit_time': dates[10:20],
                'pnl': np.random.randn(10) * 100,
                'size': [1.0] * 10
            })
        else:
            self.equity_curve = None
            self.daily_returns = None
            self.trades = None


@pytest.fixture
def temp_recorder():
    """建立臨時記錄器"""
    # 使用專案內的臨時目錄（符合安全路徑驗證）
    import shutil
    from pathlib import Path

    # 取得專案根目錄
    project_root = Path(__file__).parent.parent
    temp_dir = project_root / '.test_temp' / f'test_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        exp_file = temp_dir / 'experiments.json'
        insights_file = temp_dir / 'insights.md'

        recorder = ExperimentRecorder(
            experiments_file=exp_file,
            insights_file=insights_file
        )

        yield recorder
    finally:
        # 清理臨時目錄
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


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


def test_save_timeseries_data(temp_recorder):
    """測試時間序列資料儲存"""
    result = MockBacktestResult(with_timeseries=True)
    strategy_info = {'name': 'test_strategy', 'type': 'trend'}
    config = {'symbol': 'BTCUSDT', 'timeframe': '1h'}

    exp_id = temp_recorder.log_experiment(result, strategy_info, config)

    # 驗證目錄已建立
    exp_dir = temp_recorder.project_root / 'results' / exp_id
    assert exp_dir.exists()
    assert exp_dir.is_dir()

    # 驗證檔案已儲存
    assert (exp_dir / 'equity_curve.csv').exists()
    assert (exp_dir / 'daily_returns.csv').exists()
    assert (exp_dir / 'trades.csv').exists()


def test_load_equity_curve(temp_recorder):
    """測試載入權益曲線"""
    result = MockBacktestResult(with_timeseries=True)
    strategy_info = {'name': 'test_strategy', 'type': 'trend'}
    config = {'symbol': 'BTCUSDT', 'timeframe': '1h'}

    exp_id = temp_recorder.log_experiment(result, strategy_info, config)

    # 載入權益曲線
    loaded_curve = temp_recorder.load_equity_curve(exp_id)

    assert loaded_curve is not None
    assert isinstance(loaded_curve, pd.Series)
    assert len(loaded_curve) == len(result.equity_curve)

    # 驗證數值一致（允許浮點誤差）
    # 注意：CSV 儲存會丟失 freq 屬性，所以 check_freq=False
    pd.testing.assert_series_equal(
        loaded_curve,
        result.equity_curve,
        check_names=False,
        check_freq=False
    )


def test_load_daily_returns(temp_recorder):
    """測試載入每日收益率"""
    result = MockBacktestResult(with_timeseries=True)
    strategy_info = {'name': 'test_strategy', 'type': 'trend'}
    config = {'symbol': 'BTCUSDT', 'timeframe': '1h'}

    exp_id = temp_recorder.log_experiment(result, strategy_info, config)

    # 載入收益率
    loaded_returns = temp_recorder.load_daily_returns(exp_id)

    assert loaded_returns is not None
    assert isinstance(loaded_returns, pd.Series)
    assert len(loaded_returns) == len(result.daily_returns)

    # 驗證數值一致
    pd.testing.assert_series_equal(
        loaded_returns,
        result.daily_returns,
        check_names=False,
        check_freq=False
    )


def test_load_trades(temp_recorder):
    """測試載入交易記錄"""
    result = MockBacktestResult(with_timeseries=True)
    strategy_info = {'name': 'test_strategy', 'type': 'trend'}
    config = {'symbol': 'BTCUSDT', 'timeframe': '1h'}

    exp_id = temp_recorder.log_experiment(result, strategy_info, config)

    # 載入交易記錄
    loaded_trades = temp_recorder.load_trades(exp_id)

    assert loaded_trades is not None
    assert isinstance(loaded_trades, pd.DataFrame)
    assert len(loaded_trades) == len(result.trades)
    assert list(loaded_trades.columns) == list(result.trades.columns)


def test_date_index_preservation(temp_recorder):
    """測試日期索引保留（關鍵測試）"""
    result = MockBacktestResult(with_timeseries=True)
    strategy_info = {'name': 'test_strategy', 'type': 'trend'}
    config = {'symbol': 'BTCUSDT', 'timeframe': '1h'}

    # 記錄原始日期索引
    original_dates = result.equity_curve.index
    original_returns_dates = result.daily_returns.index

    exp_id = temp_recorder.log_experiment(result, strategy_info, config)

    # 載入並比較日期索引
    loaded_curve = temp_recorder.load_equity_curve(exp_id)
    loaded_returns = temp_recorder.load_daily_returns(exp_id)

    # 驗證索引類型為 DatetimeIndex
    assert isinstance(loaded_curve.index, pd.DatetimeIndex)
    assert isinstance(loaded_returns.index, pd.DatetimeIndex)

    # 驗證日期完全一致
    pd.testing.assert_index_equal(loaded_curve.index, original_dates)
    pd.testing.assert_index_equal(loaded_returns.index, original_returns_dates)


def test_load_nonexistent_experiment(temp_recorder):
    """測試載入不存在的實驗（邊界測試）"""
    # 嘗試載入不存在的實驗
    result = temp_recorder.load_equity_curve('exp_nonexistent')
    assert result is None

    result = temp_recorder.load_daily_returns('exp_nonexistent')
    assert result is None

    result = temp_recorder.load_trades('exp_nonexistent')
    assert result is None


def test_save_with_none_timeseries(temp_recorder):
    """測試 None 時間序列資料（邊界測試）"""
    result = MockBacktestResult(with_timeseries=False)
    strategy_info = {'name': 'test_none_series', 'type': 'trend'}
    config = {'symbol': 'BTCUSDT', 'timeframe': '1h'}

    # 關鍵測試：None 時間序列不應導致崩潰
    # 應該不會拋出異常
    try:
        exp_id = temp_recorder.log_experiment(result, strategy_info, config)
        # 成功記錄，目錄建立
        exp_dir = temp_recorder.project_root / 'results' / exp_id
        assert exp_dir.exists()

        # 嘗試載入，應該能正確處理（返回 None 或空）
        loaded = temp_recorder.load_equity_curve(exp_id)
        # 不拋出異常即為通過
        assert True
    except Exception as e:
        pytest.fail(f"處理 None 時間序列時拋出異常: {e}")


def test_save_with_empty_timeseries(temp_recorder):
    """測試空時間序列資料（邊界測試）"""
    result = MockBacktestResult(with_timeseries=True)

    # 建立空的時間序列
    result.equity_curve = pd.Series(dtype=float)
    result.daily_returns = pd.Series(dtype=float)
    result.trades = pd.DataFrame()

    strategy_info = {'name': 'test_empty_series', 'type': 'trend'}
    config = {'symbol': 'BTCUSDT', 'timeframe': '1h'}

    # 關鍵測試：空時間序列不應導致崩潰
    try:
        exp_id = temp_recorder.log_experiment(result, strategy_info, config)

        # 嘗試載入，應該能正確處理空資料
        loaded_curve = temp_recorder.load_equity_curve(exp_id)
        loaded_returns = temp_recorder.load_daily_returns(exp_id)
        loaded_trades = temp_recorder.load_trades(exp_id)

        # 能成功載入（不論是空 Series 還是 None）即為通過
        # 不應該拋出異常
        assert True
    except Exception as e:
        pytest.fail(f"處理空時間序列時拋出異常: {e}")


def test_timeseries_roundtrip_accuracy(temp_recorder):
    """測試序列化/反序列化精確度"""
    result = MockBacktestResult(with_timeseries=True)

    # 建立特定的測試資料
    dates = pd.date_range('2024-01-01 10:30:00', periods=5, freq='1h')
    result.equity_curve = pd.Series([10000.0, 10050.5, 10025.25, 10100.0, 10150.75], index=dates)
    result.daily_returns = pd.Series([0.0, 0.00505, -0.00251, 0.00747, 0.00505], index=dates)

    strategy_info = {'name': 'test_strategy', 'type': 'trend'}
    config = {'symbol': 'BTCUSDT', 'timeframe': '1h'}

    exp_id = temp_recorder.log_experiment(result, strategy_info, config)

    # 載入並驗證精確度
    loaded_curve = temp_recorder.load_equity_curve(exp_id)
    loaded_returns = temp_recorder.load_daily_returns(exp_id)

    # 完全相等測試（包含索引和數值）
    # 注意：CSV 儲存會丟失 freq 屬性
    pd.testing.assert_series_equal(loaded_curve, result.equity_curve, check_names=False, check_freq=False)
    pd.testing.assert_series_equal(loaded_returns, result.daily_returns, check_names=False, check_freq=False)

    # 驗證特定數值
    assert loaded_curve.iloc[0] == 10000.0
    assert loaded_curve.iloc[-1] == 10150.75
    assert loaded_returns.iloc[1] == pytest.approx(0.00505, rel=1e-9)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
