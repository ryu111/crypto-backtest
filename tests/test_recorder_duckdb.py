"""
ExperimentRecorder DuckDB 整合測試

測試重構後的 DuckDB 版本：
1. 基本功能（建立、記錄、查詢實驗）
2. Context manager 支援
3. 與 DuckDB 的整合
4. 遷移功能（JSON → DuckDB）
5. 時間序列資料儲存/載入
"""

import pytest
import json
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from src.learning.recorder import ExperimentRecorder
from src.types import ExperimentRecord
from src.validator.stages import ValidationGrade


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
        self.params = {'fast': 10, 'slow': 30}  # 新格式

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


class MockValidationResult:
    """模擬驗證結果"""
    def __init__(self, grade='A', passed_stages=5):
        self.grade = ValidationGrade(grade)
        # passed_stages 應該是 list，不是 int
        self.passed_stages = list(range(1, passed_stages + 1)) if isinstance(passed_stages, int) else passed_stages
        self.stage_results = {
            '階段1_基礎績效': type('obj', (object,), {
                'passed': True,
                'score': 0.9,
                'message': 'Pass',
                'details': {}
            })(),
            '階段4_WalkForward': type('obj', (object,), {
                'passed': True,
                'score': 0.85,
                'message': 'Pass',
                'details': {'efficiency': 0.85}
            })(),
            '階段5_MonteCarlo': type('obj', (object,), {
                'passed': True,
                'score': 0.88,
                'message': 'Pass',
                'details': {'p5': 0.05}
            })(),
        }


@pytest.fixture
def temp_dir():
    """建立臨時目錄"""
    project_root = Path(__file__).parent.parent
    temp = project_root / '.test_temp' / f'test_duckdb_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
    temp.mkdir(parents=True, exist_ok=True)

    yield temp

    # 清理
    if temp.exists():
        shutil.rmtree(temp)


@pytest.fixture
def db_path(temp_dir):
    """DuckDB 資料庫路徑"""
    return temp_dir / 'experiments.duckdb'


@pytest.fixture
def insights_file(temp_dir):
    """Insights 檔案路徑"""
    return temp_dir / 'insights.md'


# ===== 1. 基本功能測試 =====

def test_recorder_init(temp_dir, db_path, insights_file):
    """測試 Recorder 初始化"""
    recorder = ExperimentRecorder(
        db_path=db_path,
        insights_file=insights_file
    )

    try:
        # 檢查路徑設定
        assert recorder.db_path == db_path
        assert recorder.repo is not None
        assert recorder.storage is not None
        assert recorder.insights_manager is not None

        # 檢查檔案建立
        assert db_path.exists()
        assert insights_file.exists()
    finally:
        recorder.close()


def test_log_experiment_basic(temp_dir, db_path, insights_file):
    """測試記錄基本實驗"""
    with ExperimentRecorder(db_path=db_path, insights_file=insights_file) as recorder:
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

        # 記錄實驗
        exp_id = recorder.log_experiment(
            result=result,
            strategy_info=strategy_info,
            config=config
        )

        # 驗證實驗 ID 格式
        assert exp_id.startswith('exp_')

        # 驗證能查詢到
        exp = recorder.get_experiment(exp_id)
        assert exp is not None
        assert exp.id == exp_id
        assert exp.strategy['name'] == 'test_strategy'
        assert exp.sharpe_ratio == 1.85


def test_log_experiment_with_validation(temp_dir, db_path, insights_file):
    """測試記錄帶驗證結果的實驗"""
    with ExperimentRecorder(db_path=db_path, insights_file=insights_file) as recorder:
        result = MockBacktestResult()
        validation = MockValidationResult(grade='A', passed_stages=5)

        exp_id = recorder.log_experiment(
            result=result,
            strategy_info={'name': 'validated_strategy', 'type': 'trend'},
            config={'symbol': 'BTCUSDT', 'timeframe': '4h'},
            validation_result=validation,
            insights=['Validation successful']
        )

        # 驗證記錄
        exp = recorder.get_experiment(exp_id)
        assert exp.grade == 'A'
        assert len(exp.validation.get('stages_passed', [])) == 5
        assert len(exp.insights) == 1


def test_get_experiment_nonexistent(temp_dir, db_path, insights_file):
    """測試查詢不存在的實驗"""
    with ExperimentRecorder(db_path=db_path, insights_file=insights_file) as recorder:
        result = recorder.get_experiment('nonexistent_exp')
        assert result is None


def test_query_experiments_empty(temp_dir, db_path, insights_file):
    """測試查詢空資料庫"""
    with ExperimentRecorder(db_path=db_path, insights_file=insights_file) as recorder:
        experiments = recorder.query_experiments()
        assert len(experiments) == 0


def test_query_experiments_with_filters(temp_dir, db_path, insights_file):
    """測試過濾查詢"""
    with ExperimentRecorder(db_path=db_path, insights_file=insights_file) as recorder:
        result = MockBacktestResult()

        # 記錄多個實驗
        recorder.log_experiment(
            result,
            {'name': 'strategy1', 'type': 'trend'},
            {'symbol': 'BTCUSDT', 'timeframe': '1h'}
        )

        result.sharpe_ratio = 0.5  # 低 Sharpe
        recorder.log_experiment(
            result,
            {'name': 'strategy2', 'type': 'momentum'},
            {'symbol': 'ETHUSDT', 'timeframe': '4h'}
        )

        # 過濾趨勢策略
        trend_exps = recorder.query_experiments({
            'strategy_type': 'trend'
        })
        assert len(trend_exps) == 1
        assert trend_exps[0].strategy['name'] == 'strategy1'

        # 過濾 BTC 標的
        btc_exps = recorder.query_experiments({
            'symbol': 'BTCUSDT'
        })
        assert len(btc_exps) == 1

        # 過濾 Sharpe >= 1.0
        high_sharpe = recorder.query_experiments({
            'min_sharpe': 1.0
        })
        assert len(high_sharpe) == 1
        assert high_sharpe[0].sharpe_ratio >= 1.0


def test_get_best_experiments(temp_dir, db_path, insights_file):
    """測試取得最佳實驗"""
    with ExperimentRecorder(db_path=db_path, insights_file=insights_file) as recorder:
        result = MockBacktestResult()

        # 記錄多個不同 Sharpe 的實驗
        sharpe_values = [1.5, 2.0, 0.8, 1.8, 1.2]

        for i, sharpe in enumerate(sharpe_values):
            result.sharpe_ratio = sharpe
            recorder.log_experiment(
                result,
                {'name': f'strategy_{i}', 'type': 'trend'},
                {'symbol': 'BTCUSDT', 'timeframe': '1h'}
            )

        # 取得 Top 3
        best = recorder.get_best_experiments('sharpe_ratio', n=3)

        assert len(best) == 3
        assert best[0].sharpe_ratio == 2.0
        assert best[1].sharpe_ratio == 1.8
        assert best[2].sharpe_ratio == 1.5


# ===== 2. Context Manager 測試 =====

def test_context_manager_normal_exit(temp_dir, db_path, insights_file):
    """測試 Context Manager 正常退出"""
    with ExperimentRecorder(db_path=db_path, insights_file=insights_file) as recorder:
        result = MockBacktestResult()
        exp_id = recorder.log_experiment(
            result,
            {'name': 'context_test', 'type': 'trend'},
            {'symbol': 'BTCUSDT', 'timeframe': '1h'}
        )
        assert exp_id.startswith('exp_')

    # 退出後，資源應該已關閉
    # 再次開啟應該能讀取到剛才的記錄
    with ExperimentRecorder(db_path=db_path, insights_file=insights_file) as recorder:
        exp = recorder.get_experiment(exp_id)
        assert exp is not None


def test_context_manager_exception_exit(temp_dir, db_path, insights_file):
    """測試 Context Manager 異常退出"""
    exp_id = None

    try:
        with ExperimentRecorder(db_path=db_path, insights_file=insights_file) as recorder:
            result = MockBacktestResult()
            exp_id = recorder.log_experiment(
                result,
                {'name': 'exception_test', 'type': 'trend'},
                {'symbol': 'BTCUSDT', 'timeframe': '1h'}
            )
            # 模擬異常
            raise ValueError("Test exception")
    except ValueError:
        pass

    # 即使異常，資料應該已經保存
    with ExperimentRecorder(db_path=db_path, insights_file=insights_file) as recorder:
        exp = recorder.get_experiment(exp_id)
        assert exp is not None


def test_manual_close(temp_dir, db_path, insights_file):
    """測試手動關閉資源"""
    recorder = ExperimentRecorder(db_path=db_path, insights_file=insights_file)

    try:
        result = MockBacktestResult()
        exp_id = recorder.log_experiment(
            result,
            {'name': 'manual_test', 'type': 'trend'},
            {'symbol': 'BTCUSDT', 'timeframe': '1h'}
        )
        assert exp_id.startswith('exp_')
    finally:
        recorder.close()

    # 關閉後重新開啟，資料應該存在
    with ExperimentRecorder(db_path=db_path, insights_file=insights_file) as recorder:
        exp = recorder.get_experiment(exp_id)
        assert exp is not None


# ===== 3. 時間序列資料測試 =====

def test_save_and_load_equity_curve(temp_dir, db_path, insights_file):
    """測試儲存和載入權益曲線"""
    with ExperimentRecorder(db_path=db_path, insights_file=insights_file) as recorder:
        result = MockBacktestResult(with_timeseries=True)

        exp_id = recorder.log_experiment(
            result,
            {'name': 'timeseries_test', 'type': 'trend'},
            {'symbol': 'BTCUSDT', 'timeframe': '1h'}
        )

        # 載入權益曲線
        loaded_curve = recorder.load_equity_curve(exp_id)

        assert loaded_curve is not None
        assert isinstance(loaded_curve, pd.Series)
        assert len(loaded_curve) == len(result.equity_curve)

        # 驗證數值一致
        pd.testing.assert_series_equal(
            loaded_curve,
            result.equity_curve,
            check_names=False,
            check_freq=False
        )


def test_save_and_load_daily_returns(temp_dir, db_path, insights_file):
    """測試儲存和載入每日收益率"""
    with ExperimentRecorder(db_path=db_path, insights_file=insights_file) as recorder:
        result = MockBacktestResult(with_timeseries=True)

        exp_id = recorder.log_experiment(
            result,
            {'name': 'returns_test', 'type': 'trend'},
            {'symbol': 'BTCUSDT', 'timeframe': '1h'}
        )

        # 載入收益率
        loaded_returns = recorder.load_daily_returns(exp_id)

        assert loaded_returns is not None
        assert isinstance(loaded_returns, pd.Series)
        assert len(loaded_returns) == len(result.daily_returns)


def test_save_and_load_trades(temp_dir, db_path, insights_file):
    """測試儲存和載入交易記錄"""
    with ExperimentRecorder(db_path=db_path, insights_file=insights_file) as recorder:
        result = MockBacktestResult(with_timeseries=True)

        exp_id = recorder.log_experiment(
            result,
            {'name': 'trades_test', 'type': 'trend'},
            {'symbol': 'BTCUSDT', 'timeframe': '1h'}
        )

        # 載入交易記錄
        loaded_trades = recorder.load_trades(exp_id)

        assert loaded_trades is not None
        assert isinstance(loaded_trades, pd.DataFrame)
        assert len(loaded_trades) == len(result.trades)


def test_load_nonexistent_timeseries(temp_dir, db_path, insights_file):
    """測試載入不存在的時間序列"""
    with ExperimentRecorder(db_path=db_path, insights_file=insights_file) as recorder:
        result = recorder.load_equity_curve('nonexistent_exp')
        assert result is None

        result = recorder.load_daily_returns('nonexistent_exp')
        assert result is None

        result = recorder.load_trades('nonexistent_exp')
        assert result is None


# ===== 4. 遷移功能測試 =====

def test_migrate_from_json(temp_dir, db_path, insights_file):
    """測試從 JSON 遷移到 DuckDB"""
    # 建立 JSON 檔案
    json_file = temp_dir / 'experiments.json'
    json_data = {
        'version': '1.0',
        'metadata': {
            'total_experiments': 2,
            'last_updated': '2026-01-11T10:00:00'
        },
        'experiments': [
            {
                'id': 'exp_001',
                'timestamp': '2026-01-11T10:00:00',
                'strategy': {'name': 'strategy1', 'type': 'trend', 'version': '1.0', 'params': {}},
                'config': {'symbol': 'BTCUSDT', 'timeframe': '1h'},
                'results': {'sharpe_ratio': 1.5, 'total_return': 0.3, 'max_drawdown': -0.1},
                'validation': {'grade': 'A', 'stages_passed': []},
                'status': 'completed',
                'insights': [],
                'tags': ['crypto', 'btc']
            },
            {
                'id': 'exp_002',
                'timestamp': '2026-01-11T11:00:00',
                'strategy': {'name': 'strategy2', 'type': 'momentum', 'version': '1.0', 'params': {}},
                'config': {'symbol': 'ETHUSDT', 'timeframe': '4h'},
                'results': {'sharpe_ratio': 1.8, 'total_return': 0.4, 'max_drawdown': -0.08},
                'validation': {'grade': 'B', 'stages_passed': []},
                'status': 'completed',
                'insights': [],
                'tags': ['crypto', 'eth']
            }
        ]
    }

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f)

    # 建立 Recorder（指定 JSON 檔案）
    with ExperimentRecorder(
        db_path=db_path,
        insights_file=insights_file,
        experiments_file=json_file
    ) as recorder:
        # 手動觸發遷移（自動遷移已在 __init__ 執行）
        # 這裡只是驗證結果

        # 驗證資料已遷移
        all_exps = recorder.query_experiments()
        assert len(all_exps) == 2

        # 驗證實驗內容
        exp1 = recorder.get_experiment('exp_001')
        assert exp1 is not None
        assert exp1.strategy['name'] == 'strategy1'
        assert exp1.sharpe_ratio == 1.5

        exp2 = recorder.get_experiment('exp_002')
        assert exp2 is not None
        assert exp2.strategy['name'] == 'strategy2'
        assert exp2.sharpe_ratio == 1.8


def test_auto_migrate_on_init(temp_dir, db_path, insights_file):
    """測試初始化時自動遷移"""
    # 建立 JSON 檔案
    json_file = temp_dir / 'experiments.json'
    json_data = {
        'version': '1.0',
        'metadata': {'total_experiments': 1},
        'experiments': [
            {
                'id': 'exp_auto',
                'timestamp': '2026-01-11T10:00:00',
                'strategy': {'name': 'auto_migrate', 'type': 'trend', 'version': '1.0', 'params': {}},
                'config': {'symbol': 'BTCUSDT', 'timeframe': '1h'},
                'results': {'sharpe_ratio': 1.0, 'total_return': 0.1, 'max_drawdown': -0.05},
                'validation': {'grade': 'C', 'stages_passed': []},
                'status': 'completed',
                'insights': [],
                'tags': []
            }
        ]
    }

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f)

    # 初始化 Recorder（應該自動遷移）
    with ExperimentRecorder(
        db_path=db_path,
        insights_file=insights_file,
        experiments_file=json_file
    ) as recorder:
        # 驗證已遷移
        exp = recorder.get_experiment('exp_auto')
        assert exp is not None
        assert exp.strategy['name'] == 'auto_migrate'

    # 驗證原 JSON 已重命名為備份
    backup_file = json_file.with_suffix('.json.migrated')
    assert backup_file.exists()
    assert not json_file.exists()


# ===== 5. 策略演進測試 =====

def test_get_strategy_evolution(temp_dir, db_path, insights_file):
    """測試策略演進追蹤"""
    with ExperimentRecorder(db_path=db_path, insights_file=insights_file) as recorder:
        result = MockBacktestResult()

        # 記錄同一策略的不同版本
        versions = ['1.0', '1.1', '2.0']
        sharpes = [1.5, 1.7, 2.0]

        for version, sharpe in zip(versions, sharpes):
            result.sharpe_ratio = sharpe
            recorder.log_experiment(
                result,
                {'name': f'ma_cross_v{version}', 'type': 'trend', 'version': version},
                {'symbol': 'BTCUSDT', 'timeframe': '1h'}
            )

        # 追蹤演進（使用前綴匹配）
        evolution = recorder.get_strategy_evolution('ma_cross')

        assert len(evolution) == 3
        assert evolution[0]['version'] == '1.0'
        assert evolution[1]['version'] == '1.1'
        assert evolution[2]['version'] == '2.0'

        # 檢查 Sharpe 遞增
        assert evolution[0]['sharpe'] == 1.5
        assert evolution[1]['sharpe'] == 1.7
        assert evolution[2]['sharpe'] == 2.0


def test_get_strategy_stats(temp_dir, db_path, insights_file):
    """測試策略統計"""
    with ExperimentRecorder(db_path=db_path, insights_file=insights_file) as recorder:
        result = MockBacktestResult()

        # 記錄多個實驗
        sharpes = [1.5, 2.0, 1.8]
        for i, sharpe in enumerate(sharpes):
            result.sharpe_ratio = sharpe
            validation = MockValidationResult(grade='A' if sharpe >= 1.8 else 'C')
            recorder.log_experiment(
                result,
                {'name': f'test_strategy_{i}', 'type': 'trend'},
                {'symbol': 'BTCUSDT', 'timeframe': '1h'},
                validation_result=validation
            )

        # 取得統計
        stats = recorder.get_strategy_stats('test_strategy')

        assert stats is not None
        assert stats['attempts'] == 3
        assert stats['successes'] == 2  # A/B 評級
        assert stats['avg_sharpe'] == pytest.approx((1.5 + 2.0 + 1.8) / 3, rel=0.01)
        assert stats['best_sharpe'] == 2.0


# ===== 6. 標籤生成測試 =====

def test_generate_tags(temp_dir, db_path, insights_file):
    """測試標籤自動生成"""
    with ExperimentRecorder(db_path=db_path, insights_file=insights_file) as recorder:
        strategy_info = {
            'name': 'ma_cross_rsi',
            'type': 'trend'
        }
        config = {
            'symbol': 'BTCUSDT',
            'timeframe': '4h'
        }
        validation = {
            'grade': 'A'
        }

        tags = recorder.generate_tags(strategy_info, config, validation)

        assert 'crypto' in tags
        assert 'btc' in tags
        assert 'trend' in tags
        assert 'ma' in tags
        assert 'rsi' in tags
        assert '4h' in tags
        assert 'validated' in tags


# ===== 7. 匯出功能測試 =====

def test_export_to_json(temp_dir, db_path, insights_file):
    """測試匯出到 JSON（備份功能）"""
    with ExperimentRecorder(db_path=db_path, insights_file=insights_file) as recorder:
        result = MockBacktestResult()

        # 記錄實驗
        recorder.log_experiment(
            result,
            {'name': 'export_test', 'type': 'trend'},
            {'symbol': 'BTCUSDT', 'timeframe': '1h'}
        )

        # 匯出到 JSON
        output_file = temp_dir / 'backup.json'
        exported_path = recorder.export_to_json(output_file)

        # 驗證檔案存在
        assert exported_path.exists()

        # 驗證內容
        with open(exported_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert data['metadata']['total_experiments'] == 1
        assert len(data['experiments']) == 1
        assert data['experiments'][0]['strategy']['name'] == 'export_test'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
