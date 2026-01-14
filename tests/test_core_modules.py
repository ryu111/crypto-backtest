"""
核心模組測試

測試 BaseStrategy, ExperimentRecorder, StrategySelector 的核心功能。
"""

import pytest
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock

# BaseStrategy 測試
from src.strategies.base import BaseStrategy


class SimpleStrategy(BaseStrategy):
    """測試用策略（重新命名避免 pytest 收集警告）"""
    name = "test_strategy"
    strategy_type = "test"

    def __init__(self, **kwargs):
        # 先呼叫父類別（會初始化 params 和 param_space 為空字典）
        super().__init__()

        # 然後設定參數
        self.params = {'period': kwargs.get('period', 20)}
        self.param_space = {'period': (10, 50, 5)}

    def calculate_indicators(self, data):
        return {'sma': data['close'].rolling(self.params['period']).mean()}

    def generate_signals(self, data):
        sma = self.calculate_indicators(data)['sma']
        long_entry = data['close'] > sma
        long_exit = data['close'] < sma
        short_entry = data['close'] < sma
        short_exit = data['close'] > sma
        return long_entry, long_exit, short_entry, short_exit


class TestBaseStrategyCore:
    """測試 BaseStrategy 核心功能"""

    def test_params_not_shared_between_instances(self):
        """驗證 params 不在實例間共享"""
        s1 = SimpleStrategy(period=10)
        s2 = SimpleStrategy(period=20)

        assert s1.params['period'] == 10
        assert s2.params['period'] == 20

        s1.params['period'] = 30
        assert s2.params['period'] == 20  # 未改變

    def test_param_space_independence(self):
        """驗證 param_space 獨立性"""
        s1 = SimpleStrategy()
        s2 = SimpleStrategy()

        # 因為 BaseStrategy.__init__ 會重置 param_space 為空字典
        # 子類別需要在 __init__ 中設定，所以每個實例都獨立
        assert 'period' in s1.param_space
        assert 'period' in s2.param_space

        # 修改 s1 不影響 s2
        s1.param_space['period'] = (5, 30, 5)
        assert s2.param_space['period'] == (10, 50, 5)

    def test_position_size_calculation(self):
        """測試部位大小計算"""
        strategy = SimpleStrategy()

        size = strategy.position_size(
            capital=10000,
            entry_price=100,
            stop_loss_price=95,
            risk_per_trade=0.02
        )

        # 風險 = 10000 * 0.02 = 200
        # 止損距離 = 5
        # 部位 = 200 / 5 = 40
        assert size == 40

    def test_position_size_zero_stop_distance(self):
        """測試零止損距離"""
        strategy = SimpleStrategy()
        size = strategy.position_size(
            capital=10000,
            entry_price=100,
            stop_loss_price=100
        )
        assert size == 0

    def test_signal_generation(self):
        """測試訊號生成"""
        strategy = SimpleStrategy(period=3)
        data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100, 101, 102, 103, 104],
            'volume': [1000] * 5
        })

        long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(data)

        assert isinstance(long_entry, pd.Series)
        assert len(long_entry) == len(data)
        assert long_entry.dtype == bool


# ExperimentRecorder 測試
from src.learning.recorder import ExperimentRecorder
from src.types.results import ExperimentRecord


class MockBacktestResult:
    def __init__(self, sharpe=1.5):
        self.total_return = 0.456
        self.annual_return = 0.23
        self.sharpe_ratio = sharpe
        self.sortino_ratio = 2.1
        self.max_drawdown = -0.10
        self.win_rate = 0.55
        self.profit_factor = 1.72
        self.total_trades = 124
        self.avg_trade_duration = 12.5
        self.expectancy = 0.0037
        self.parameters = {'period': 20}


class TestExperimentRecorderCore:
    """測試 ExperimentRecorder 核心功能（DuckDB 版本）"""

    def test_log_and_retrieve_experiment(self):
        """測試記錄和取得實驗"""
        import shutil
        # 使用專案內的臨時目錄
        project_root = Path(__file__).parent.parent
        test_dir = project_root / 'tests' / '.test_data'
        test_dir.mkdir(exist_ok=True)

        db_file = test_dir / 'test_experiments.duckdb'
        insights_file = test_dir / 'test_insights.md'

        try:
            # 使用 context manager 確保資源正確關閉（Path 物件，不是字串）
            with ExperimentRecorder(db_path=db_file, insights_file=insights_file) as recorder:
                result = MockBacktestResult()
                strategy_info = {'name': 'test', 'type': 'trend'}
                config = {'symbol': 'BTCUSDT', 'timeframe': '1h'}

                exp_id = recorder.log_experiment(result, strategy_info, config)

                # 驗證記錄
                assert exp_id.startswith('exp_')

                # 取得實驗（返回 ExperimentRecord dataclass）
                exp = recorder.get_experiment(exp_id)
                assert exp is not None
                assert exp.strategy['name'] == 'test'
                assert exp.results['sharpe_ratio'] == 1.5

        finally:
            # 清理（使用 shutil 刪除整個目錄）
            if test_dir.exists():
                shutil.rmtree(test_dir, ignore_errors=True)

    def test_database_initialization(self):
        """測試資料庫初始化（替代 JSON 錯誤處理測試）"""
        import shutil
        project_root = Path(__file__).parent.parent
        test_dir = project_root / 'tests' / '.test_data'
        test_dir.mkdir(exist_ok=True)

        db_file = test_dir / 'test_init.duckdb'

        try:
            # 建立新資料庫應該成功（Path 物件）
            with ExperimentRecorder(db_path=db_file) as recorder:
                # 查詢空資料庫應該返回 0 筆記錄
                all_exps = recorder.query_experiments()
                assert len(all_exps) == 0

        finally:
            if test_dir.exists():
                shutil.rmtree(test_dir, ignore_errors=True)

    def test_query_experiments(self):
        """測試實驗查詢"""
        import shutil
        project_root = Path(__file__).parent.parent
        test_dir = project_root / 'tests' / '.test_data'
        test_dir.mkdir(exist_ok=True)

        db_file = test_dir / 'test_query.duckdb'
        insights_file = test_dir / 'test_insights.md'

        try:
            with ExperimentRecorder(db_path=db_file, insights_file=insights_file) as recorder:
                # 記錄兩個實驗
                result1 = MockBacktestResult(sharpe=1.5)
                recorder.log_experiment(
                    result1,
                    {'name': 's1', 'type': 'trend'},
                    {'symbol': 'BTCUSDT', 'timeframe': '1h'}
                )

                result2 = MockBacktestResult(sharpe=0.5)
                recorder.log_experiment(
                    result2,
                    {'name': 's2', 'type': 'momentum'},
                    {'symbol': 'ETHUSDT', 'timeframe': '4h'}
                )

                # 查詢所有實驗
                all_exps = recorder.query_experiments()
                assert len(all_exps) == 2

                # TODO: DuckDB 版本的查詢 API 可能不同，需要檢查實際 API
                # 暫時測試能成功查詢即可

        finally:
            if test_dir.exists():
                shutil.rmtree(test_dir, ignore_errors=True)


# StrategySelector 測試
from src.automation.selector import StrategySelector, StrategyStats


class MockRegistry:
    def __init__(self):
        self.strategies = ['s_a', 's_b', 's_c']

    def list_strategies(self):
        return self.strategies.copy()

    def list_all(self):
        """DuckDB 版本需要的方法"""
        return self.strategies.copy()


class MockRecorder:
    def __init__(self):
        self.stats_db = {}

    def record_strategy_stats(self, name, stats):
        self.stats_db[name] = stats

    def get_strategy_stats(self, name):
        return self.stats_db.get(name)


class TestStrategySelectorCore:
    """測試 StrategySelector 核心功能"""

    def test_epsilon_greedy_exploitation(self, monkeypatch):
        """測試 epsilon-greedy 利用模式"""
        registry = MockRegistry()
        recorder = MockRecorder()
        selector = StrategySelector(registry, recorder)

        # Mock random 返回高值（利用模式）
        monkeypatch.setattr('random.random', lambda: 0.9)

        selector._stats_cache = {
            's_a': StrategyStats('s_a', attempts=10, avg_sharpe=1.5),
            's_b': StrategyStats('s_b', attempts=10, avg_sharpe=2.5),  # 最佳
            's_c': StrategyStats('s_c', attempts=10, avg_sharpe=0.8),
        }
        selector._cache_updated = True

        selected = selector._epsilon_greedy()
        assert selected == 's_b'

    def test_ucb_untried_strategy(self):
        """測試 UCB 對未嘗試策略的優先權"""
        registry = MockRegistry()
        recorder = MockRecorder()
        selector = StrategySelector(registry, recorder)

        selector._stats_cache = {
            's_a': StrategyStats('s_a', attempts=10, avg_sharpe=2.0),
            's_b': StrategyStats('s_b', attempts=0, avg_sharpe=0.0),  # 未嘗試
            's_c': StrategyStats('s_c', attempts=5, avg_sharpe=1.5),
        }
        selector._cache_updated = True

        selected = selector._ucb()
        assert selected == 's_b'

    def test_update_stats(self):
        """測試統計更新"""
        registry = MockRegistry()
        recorder = MockRecorder()
        selector = StrategySelector(registry, recorder)

        result = {
            'passed': True,
            'sharpe_ratio': 1.8,
            'params': {'period': 20}
        }

        selector.update_stats('test_strategy', result)

        stat = selector._stats_cache['test_strategy']
        assert stat.attempts == 1
        assert stat.successes == 1
        assert stat.avg_sharpe == 1.8
        assert stat.best_sharpe == 1.8

    def test_update_stats_incremental(self):
        """測試增量統計更新"""
        registry = MockRegistry()
        recorder = MockRecorder()
        selector = StrategySelector(registry, recorder)

        # 第一次
        selector.update_stats('test', {'passed': True, 'sharpe_ratio': 2.0, 'params': {}})

        # 第二次
        selector.update_stats('test', {'passed': False, 'sharpe_ratio': 1.0, 'params': {}})

        stat = selector._stats_cache['test']
        assert stat.attempts == 2
        assert stat.successes == 1
        assert stat.avg_sharpe == 1.5  # (2.0 + 1.0) / 2

    def test_exploration_stats(self):
        """測試探索統計"""
        registry = MockRegistry()
        recorder = MockRecorder()
        selector = StrategySelector(registry, recorder)

        selector._stats_cache = {
            's_a': StrategyStats('s_a', attempts=10, best_sharpe=1.8),
            's_b': StrategyStats('s_b', attempts=5, best_sharpe=2.2),
            's_c': StrategyStats('s_c', attempts=0, best_sharpe=0.0),
        }
        selector._cache_updated = True

        stats = selector.get_exploration_stats()

        assert stats['total_attempts'] == 15
        assert stats['strategies_tried'] == 2
        assert stats['best_strategy'] == 's_b'
        assert stats['best_sharpe'] == 2.2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
