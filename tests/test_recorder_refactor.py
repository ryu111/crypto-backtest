"""
recorder.py 重構測試

測試重構後的三個類別：
- TimeSeriesStorage (storage.py)
- InsightsManager (insights.py)
- ExperimentRecorder (recorder.py)

確認：
1. 單元測試：每個類別獨立運作正常
2. 整合測試：三個類別正確協作
3. 回歸測試：外部 API 不變，現有功能正常
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from src.learning import ExperimentRecorder, Experiment
from src.learning.storage import TimeSeriesStorage
from src.learning.insights import InsightsManager


# ===== Mock 資料 =====

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

        # 時間序列資料
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
def temp_dir():
    """建立臨時目錄"""
    project_root = Path(__file__).parent.parent
    temp = project_root / '.test_temp' / f'test_refactor_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
    temp.mkdir(parents=True, exist_ok=True)

    yield temp

    # 清理
    if temp.exists():
        shutil.rmtree(temp)


@pytest.fixture
def project_root():
    """專案根目錄"""
    return Path(__file__).parent.parent


# ===== TimeSeriesStorage 單元測試 =====

def test_storage_init(temp_dir):
    """測試 Storage 初始化"""
    storage = TimeSeriesStorage(temp_dir)

    assert storage.project_root == temp_dir
    assert storage.results_dir == temp_dir / 'results'
    assert storage.results_dir.exists()


def test_storage_save_equity_curve(temp_dir):
    """測試儲存權益曲線"""
    storage = TimeSeriesStorage(temp_dir)
    result = MockBacktestResult(with_timeseries=True)
    exp_id = 'test_exp_001'

    # 儲存
    storage.save(exp_id, result)

    # 驗證檔案存在
    exp_dir = temp_dir / 'results' / exp_id
    assert (exp_dir / 'equity_curve.csv').exists()

    # 驗證內容
    df = pd.read_csv(exp_dir / 'equity_curve.csv', index_col='date', parse_dates=True)
    assert len(df) == len(result.equity_curve)
    assert 'equity' in df.columns


def test_storage_save_daily_returns(temp_dir):
    """測試儲存每日收益率"""
    storage = TimeSeriesStorage(temp_dir)
    result = MockBacktestResult(with_timeseries=True)
    exp_id = 'test_exp_002'

    storage.save(exp_id, result)

    # 驗證檔案存在
    exp_dir = temp_dir / 'results' / exp_id
    assert (exp_dir / 'daily_returns.csv').exists()

    # 驗證內容
    df = pd.read_csv(exp_dir / 'daily_returns.csv', index_col='date', parse_dates=True)
    assert len(df) == len(result.daily_returns)
    assert 'return' in df.columns


def test_storage_save_trades(temp_dir):
    """測試儲存交易記錄"""
    storage = TimeSeriesStorage(temp_dir)
    result = MockBacktestResult(with_timeseries=True)
    exp_id = 'test_exp_003'

    storage.save(exp_id, result)

    # 驗證檔案存在
    exp_dir = temp_dir / 'results' / exp_id
    assert (exp_dir / 'trades.csv').exists()

    # 驗證內容
    df = pd.read_csv(exp_dir / 'trades.csv')
    assert len(df) == len(result.trades)


def test_storage_load_equity_curve(temp_dir):
    """測試載入權益曲線"""
    storage = TimeSeriesStorage(temp_dir)
    result = MockBacktestResult(with_timeseries=True)
    exp_id = 'test_exp_004'

    # 先儲存
    storage.save(exp_id, result)

    # 再載入
    loaded = storage.load_equity_curve(exp_id)

    assert loaded is not None
    assert isinstance(loaded, pd.Series)
    assert len(loaded) == len(result.equity_curve)

    # 驗證數值一致
    pd.testing.assert_series_equal(
        loaded,
        result.equity_curve,
        check_names=False,
        check_freq=False
    )


def test_storage_load_daily_returns(temp_dir):
    """測試載入每日收益率"""
    storage = TimeSeriesStorage(temp_dir)
    result = MockBacktestResult(with_timeseries=True)
    exp_id = 'test_exp_005'

    storage.save(exp_id, result)
    loaded = storage.load_daily_returns(exp_id)

    assert loaded is not None
    assert isinstance(loaded, pd.Series)
    assert len(loaded) == len(result.daily_returns)


def test_storage_load_trades(temp_dir):
    """測試載入交易記錄"""
    storage = TimeSeriesStorage(temp_dir)
    result = MockBacktestResult(with_timeseries=True)
    exp_id = 'test_exp_006'

    storage.save(exp_id, result)
    loaded = storage.load_trades(exp_id)

    assert loaded is not None
    assert isinstance(loaded, pd.DataFrame)
    assert len(loaded) == len(result.trades)


def test_storage_load_nonexistent(temp_dir):
    """測試載入不存在的資料（邊界測試）"""
    storage = TimeSeriesStorage(temp_dir)

    result = storage.load_equity_curve('nonexistent_exp')
    assert result is None

    result = storage.load_daily_returns('nonexistent_exp')
    assert result is None

    result = storage.load_trades('nonexistent_exp')
    assert result is None


def test_storage_handle_none_timeseries(temp_dir):
    """測試處理 None 時間序列（邊界測試）"""
    storage = TimeSeriesStorage(temp_dir)
    result = MockBacktestResult(with_timeseries=False)
    exp_id = 'test_exp_none'

    # 不應該拋出異常
    storage.save(exp_id, result)

    # 目錄應該存在
    exp_dir = temp_dir / 'results' / exp_id
    assert exp_dir.exists()


# ===== InsightsManager 單元測試 =====

def test_insights_init(temp_dir):
    """測試 InsightsManager 初始化"""
    insights_file = temp_dir / 'insights.md'
    manager = InsightsManager(insights_file)

    assert manager.insights_file == insights_file
    assert insights_file.exists()

    # 檢查初始內容
    content = insights_file.read_text(encoding='utf-8')
    assert '# 交易策略洞察彙整' in content
    assert '## 策略類型洞察' in content


def test_insights_update_header(temp_dir):
    """測試更新標題資訊"""
    insights_file = temp_dir / 'insights.md'
    manager = InsightsManager(insights_file)

    # 建立實驗
    exp = Experiment(
        id='exp_test',
        timestamp=datetime.now(),
        strategy={'name': 'test_strategy', 'type': 'trend'},
        config={'symbol': 'BTCUSDT'},
        parameters={},
        results={'sharpe_ratio': 1.5, 'total_return': 0.45},
        validation={'grade': 'A'},
        insights=['Test insight']
    )

    # 更新
    manager.update(exp, total_experiments=1)

    # 檢查內容
    content = insights_file.read_text(encoding='utf-8')
    assert '> 總實驗數：1' in content
    assert f"> 最後更新：{datetime.now().strftime('%Y-%m-%d')}" in content


def test_insights_update_strategy_section_trend(temp_dir):
    """測試更新趨勢策略區塊"""
    insights_file = temp_dir / 'insights.md'
    manager = InsightsManager(insights_file)

    exp = Experiment(
        id='exp_trend',
        timestamp=datetime.now(),
        strategy={'name': 'ma_cross', 'type': 'trend'},
        config={'symbol': 'BTCUSDT'},
        parameters={'fast': 10, 'slow': 30},
        results={'sharpe_ratio': 2.0, 'total_return': 0.50},
        validation={'grade': 'A'},
        insights=['MA cross 效果良好']
    )

    manager.update(exp, total_experiments=1)

    content = insights_file.read_text(encoding='utf-8')

    # 應該不再有「尚無記錄」
    assert '### 趨勢跟隨策略' in content
    # 檢查是否有策略資訊
    assert 'ma_cross' in content
    assert 'Sharpe 2.00' in content


def test_insights_update_strategy_section_momentum(temp_dir):
    """測試更新動量策略區塊"""
    insights_file = temp_dir / 'insights.md'
    manager = InsightsManager(insights_file)

    exp = Experiment(
        id='exp_momentum',
        timestamp=datetime.now(),
        strategy={'name': 'rsi_reversal', 'type': 'momentum'},
        config={'symbol': 'ETHUSDT'},
        parameters={'period': 14},
        results={'sharpe_ratio': 1.8, 'total_return': 0.40},
        validation={'grade': 'B'},
        insights=['RSI 反轉策略表現穩定']
    )

    manager.update(exp, total_experiments=1)

    content = insights_file.read_text(encoding='utf-8')

    assert '### 動量策略' in content
    assert 'rsi_reversal' in content


def test_insights_update_failure_section(temp_dir):
    """測試更新失敗教訓區塊"""
    insights_file = temp_dir / 'insights.md'
    manager = InsightsManager(insights_file)

    exp = Experiment(
        id='exp_fail',
        timestamp=datetime.now(),
        strategy={'name': 'bad_strategy', 'type': 'trend'},
        config={'symbol': 'BTCUSDT'},
        parameters={},
        results={'sharpe_ratio': 0.2, 'total_return': -0.10},
        validation={'grade': 'F', 'passed_stages': 0}
    )

    manager.update(exp, total_experiments=1)

    content = insights_file.read_text(encoding='utf-8')

    assert '## 過擬合教訓' in content
    assert 'bad_strategy' in content
    assert '基礎績效不達標' in content


# ===== ExperimentRecorder 整合測試 =====

@pytest.fixture
def recorder(temp_dir):
    """建立 ExperimentRecorder"""
    exp_file = temp_dir / 'experiments.json'
    insights_file = temp_dir / 'insights.md'

    return ExperimentRecorder(
        experiments_file=exp_file,
        insights_file=insights_file
    )


def test_recorder_init_components(recorder):
    """測試 Recorder 正確初始化子元件"""
    # 檢查子元件已初始化
    assert recorder.storage is not None
    assert isinstance(recorder.storage, TimeSeriesStorage)

    assert recorder.insights_manager is not None
    assert isinstance(recorder.insights_manager, InsightsManager)


def test_recorder_log_experiment_integration(recorder):
    """測試 log_experiment 端到端流程（整合測試）"""
    result = MockBacktestResult(with_timeseries=True)
    strategy_info = {
        'name': 'integration_test',
        'type': 'trend',
        'version': '1.0'
    }
    config = {
        'symbol': 'BTCUSDT',
        'timeframe': '1h'
    }

    # 執行記錄
    exp_id = recorder.log_experiment(
        result=result,
        strategy_info=strategy_info,
        config=config,
        insights=['Integration test insight']
    )

    # 驗證 experiments.json 已更新
    data = json.loads(recorder.experiments_file.read_text())
    assert data['metadata']['total_experiments'] == 1
    assert len(data['experiments']) == 1
    assert data['experiments'][0]['id'] == exp_id

    # 驗證時間序列已儲存（storage 被呼叫）
    equity_curve = recorder.load_equity_curve(exp_id)
    assert equity_curve is not None

    # 驗證 insights.md 已更新（insights_manager 被呼叫）
    insights_content = recorder.insights_manager.insights_file.read_text(encoding='utf-8')
    assert 'integration_test' in insights_content


def test_recorder_delegation_to_storage(recorder):
    """測試 Recorder 正確委派給 Storage"""
    result = MockBacktestResult(with_timeseries=True)
    strategy_info = {'name': 'delegate_test', 'type': 'trend'}
    config = {'symbol': 'BTCUSDT', 'timeframe': '1h'}

    exp_id = recorder.log_experiment(result, strategy_info, config)

    # 測試委派方法
    equity = recorder.load_equity_curve(exp_id)
    returns = recorder.load_daily_returns(exp_id)
    trades = recorder.load_trades(exp_id)

    # 驗證結果來自 storage
    assert equity is not None
    assert returns is not None
    assert trades is not None


def test_recorder_delegation_to_insights(recorder):
    """測試 Recorder 正確委派給 InsightsManager"""
    result = MockBacktestResult()
    strategy_info = {'name': 'insights_test', 'type': 'momentum'}
    config = {'symbol': 'ETHUSDT', 'timeframe': '4h'}

    # 記錄有 insights 的實驗
    recorder.log_experiment(
        result,
        strategy_info,
        config,
        insights=['Important insight']
    )

    # 檢查 insights.md 已更新
    content = recorder.insights_manager.insights_file.read_text(encoding='utf-8')
    assert 'insights_test' in content
    assert '### 動量策略' in content


# ===== API 向後相容性測試（回歸測試）=====

def test_api_log_experiment_signature(recorder):
    """測試 log_experiment API 簽章不變"""
    result = MockBacktestResult()

    # 原有的呼叫方式應該仍然有效
    exp_id = recorder.log_experiment(
        result=result,
        strategy_info={'name': 'api_test', 'type': 'trend'},
        config={'symbol': 'BTCUSDT', 'timeframe': '1h'},
        validation_result=None,
        insights=['API test'],
        parent_experiment=None
    )

    assert exp_id.startswith('exp_')


def test_api_query_experiments(recorder):
    """測試 query_experiments API 不變"""
    result = MockBacktestResult()

    # 記錄實驗
    recorder.log_experiment(
        result,
        {'name': 'query_test', 'type': 'trend'},
        {'symbol': 'BTCUSDT', 'timeframe': '1h'}
    )

    # 查詢應該正常工作
    experiments = recorder.query_experiments()
    assert len(experiments) == 1

    # 過濾查詢
    filtered = recorder.query_experiments({
        'strategy_type': 'trend'
    })
    assert len(filtered) == 1


def test_api_get_best_experiments(recorder):
    """測試 get_best_experiments API 不變"""
    result = MockBacktestResult()

    # 記錄多個實驗
    for i in range(3):
        result.sharpe_ratio = 1.0 + i * 0.5
        recorder.log_experiment(
            result,
            {'name': f'best_test_{i}', 'type': 'trend'},
            {'symbol': 'BTCUSDT', 'timeframe': '1h'}
        )

    # 取得最佳實驗
    best = recorder.get_best_experiments('sharpe_ratio', n=2)

    assert len(best) == 2
    assert best[0].results['sharpe_ratio'] == 2.0
    assert best[1].results['sharpe_ratio'] == 1.5


def test_api_load_timeseries_methods(recorder):
    """測試時間序列載入 API 不變"""
    result = MockBacktestResult(with_timeseries=True)

    exp_id = recorder.log_experiment(
        result,
        {'name': 'timeseries_api', 'type': 'trend'},
        {'symbol': 'BTCUSDT', 'timeframe': '1h'}
    )

    # 原有的 API 應該仍然可用
    equity = recorder.load_equity_curve(exp_id)
    returns = recorder.load_daily_returns(exp_id)
    trades = recorder.load_trades(exp_id)

    assert equity is not None
    assert returns is not None
    assert trades is not None


# ===== 錯誤處理測試 =====

def test_storage_corrupted_csv(temp_dir):
    """測試處理損壞的 CSV 檔案"""
    storage = TimeSeriesStorage(temp_dir)
    exp_id = 'test_corrupted'

    # 建立損壞的檔案
    exp_dir = temp_dir / 'results' / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'equity_curve.csv').write_text('corrupted,data\n1,2,3,4,5')

    # 應該返回 None 而不是拋出異常
    result = storage.load_equity_curve(exp_id)
    assert result is None


def test_insights_handle_empty_strategy_type(temp_dir):
    """測試處理空策略類型"""
    insights_file = temp_dir / 'insights.md'
    manager = InsightsManager(insights_file)

    exp = Experiment(
        id='exp_empty_type',
        timestamp=datetime.now(),
        strategy={'name': 'unknown_strategy', 'type': ''},  # 空類型
        config={'symbol': 'BTCUSDT'},
        parameters={},
        results={'sharpe_ratio': 1.0, 'total_return': 0.1},
        validation={}
    )

    # 不應該拋出異常
    manager.update(exp, total_experiments=1)

    # insights.md 應該保持完整
    assert insights_file.exists()


# ===== 效能測試 =====

def test_storage_batch_save_performance(temp_dir):
    """測試批量儲存效能"""
    storage = TimeSeriesStorage(temp_dir)

    # 儲存多個實驗
    import time
    start = time.time()

    for i in range(10):
        result = MockBacktestResult(with_timeseries=True)
        storage.save(f'exp_{i:03d}', result)

    elapsed = time.time() - start

    # 10 個實驗應該在合理時間內完成（< 5 秒）
    assert elapsed < 5.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
