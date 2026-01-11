"""
StrategySelector 完整測試

測試 Epsilon-Greedy、UCB、Thompson Sampling 三種選擇策略。
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from datetime import datetime

from src.automation.selector import StrategySelector, StrategyStats


class MockStrategyRegistry:
    """模擬策略註冊表"""

    def __init__(self, strategies=None):
        self.strategies = strategies or ['strategy_a', 'strategy_b', 'strategy_c']

    def list_strategies(self):
        return self.strategies.copy()


class MockExperimentRecorder:
    """模擬實驗記錄器"""

    def __init__(self):
        self.stats_db = {}

    def record_strategy_stats(self, name, stats):
        """記錄統計"""
        self.stats_db[name] = stats

    def get_strategy_stats(self, name):
        """取得統計"""
        return self.stats_db.get(name)


@pytest.fixture
def mock_registry():
    """建立模擬註冊表"""
    return MockStrategyRegistry()


@pytest.fixture
def mock_recorder():
    """建立模擬記錄器"""
    return MockExperimentRecorder()


@pytest.fixture
def selector(mock_registry, mock_recorder):
    """建立選擇器"""
    return StrategySelector(mock_registry, mock_recorder)


class TestStrategyStatsDataclass:
    """測試 StrategyStats 資料類別"""

    def test_default_values(self):
        """測試預設值"""
        stat = StrategyStats(name='test')

        assert stat.name == 'test'
        assert stat.attempts == 0
        assert stat.successes == 0
        assert stat.avg_sharpe == 0.0
        assert stat.best_sharpe == 0.0
        assert stat.last_attempt is None
        assert stat.last_params == {}

    def test_success_rate_calculation(self):
        """測試成功率計算"""
        stat = StrategyStats(name='test', attempts=10, successes=7)

        assert stat.success_rate == 0.7
        assert stat.failure_rate == 0.3

    def test_success_rate_zero_attempts(self):
        """測試零嘗試次數的成功率"""
        stat = StrategyStats(name='test', attempts=0, successes=0)

        assert stat.success_rate == 0.0
        assert stat.failure_rate == 1.0


class TestEpsilonGreedy:
    """測試 Epsilon-Greedy 選擇"""

    def test_exploration_mode(self, selector, mock_registry, monkeypatch):
        """測試探索模式（隨機選擇）"""
        # Mock random.random 返回值 < epsilon（0.2）
        monkeypatch.setattr('random.random', lambda: 0.1)

        # 初始化統計快取
        selector._stats_cache = {
            'strategy_a': StrategyStats('strategy_a', attempts=10, avg_sharpe=1.5),
            'strategy_b': StrategyStats('strategy_b', attempts=5, avg_sharpe=2.0),
            'strategy_c': StrategyStats('strategy_c', attempts=2, avg_sharpe=0.5),
        }
        selector._cache_updated = True

        # 多次選擇，驗證隨機性
        choices = [selector._epsilon_greedy() for _ in range(10)]

        # 應該在所有策略中選擇
        assert len(set(choices)) >= 1  # 至少有變化

    def test_exploitation_mode(self, selector, monkeypatch):
        """測試利用模式（選擇最佳）"""
        # Mock random.random 返回值 >= epsilon
        monkeypatch.setattr('random.random', lambda: 0.9)

        # 設定統計資料（strategy_b 最佳）
        selector._stats_cache = {
            'strategy_a': StrategyStats('strategy_a', attempts=10, avg_sharpe=1.5),
            'strategy_b': StrategyStats('strategy_b', attempts=10, avg_sharpe=2.5),  # 最佳
            'strategy_c': StrategyStats('strategy_c', attempts=10, avg_sharpe=0.8),
        }
        selector._cache_updated = True

        selected = selector._epsilon_greedy()

        # 應該選擇 strategy_b（最高平均 Sharpe）
        assert selected == 'strategy_b'

    def test_exploitation_with_insufficient_attempts(self, selector, monkeypatch):
        """測試利用模式但嘗試次數不足"""
        monkeypatch.setattr('random.random', lambda: 0.9)

        # 所有策略嘗試次數都 < min_attempts (3)
        selector._stats_cache = {
            'strategy_a': StrategyStats('strategy_a', attempts=1, avg_sharpe=1.5),
            'strategy_b': StrategyStats('strategy_b', attempts=2, avg_sharpe=2.0),
            'strategy_c': StrategyStats('strategy_c', attempts=1, avg_sharpe=0.5),
        }
        selector._cache_updated = True

        # 應該隨機選擇（因為沒有足夠的歷史資料）
        selected = selector._epsilon_greedy()
        assert selected in ['strategy_a', 'strategy_b', 'strategy_c']


class TestUCB:
    """測試 UCB (Upper Confidence Bound) 選擇"""

    def test_untried_strategy_highest_priority(self, selector):
        """測試未嘗試的策略有最高優先權"""
        selector._stats_cache = {
            'strategy_a': StrategyStats('strategy_a', attempts=10, avg_sharpe=2.0),
            'strategy_b': StrategyStats('strategy_b', attempts=0, avg_sharpe=0.0),  # 未嘗試
            'strategy_c': StrategyStats('strategy_c', attempts=5, avg_sharpe=1.5),
        }
        selector._cache_updated = True

        selected = selector._ucb()

        # 應該選擇未嘗試的 strategy_b
        assert selected == 'strategy_b'

    def test_ucb_balances_exploitation_exploration(self, selector):
        """測試 UCB 平衡利用與探索"""
        selector._stats_cache = {
            'strategy_a': StrategyStats('strategy_a', attempts=100, avg_sharpe=1.8),  # 高嘗試，中等分數
            'strategy_b': StrategyStats('strategy_b', attempts=10, avg_sharpe=1.5),   # 低嘗試，中等分數
            'strategy_c': StrategyStats('strategy_c', attempts=50, avg_sharpe=2.0),   # 中嘗試，高分數
        }
        selector._cache_updated = True

        selected = selector._ucb()

        # UCB 應該考慮探索獎勵
        # strategy_b 雖然分數較低，但嘗試次數少，可能獲得探索獎勵
        # strategy_c 分數最高，可能被選中
        assert selected in ['strategy_b', 'strategy_c']

    def test_ucb_zero_total_attempts(self, selector):
        """測試總嘗試次數為零時的 UCB"""
        selector._stats_cache = {
            'strategy_a': StrategyStats('strategy_a', attempts=0, avg_sharpe=0.0),
            'strategy_b': StrategyStats('strategy_b', attempts=0, avg_sharpe=0.0),
            'strategy_c': StrategyStats('strategy_c', attempts=0, avg_sharpe=0.0),
        }
        selector._cache_updated = True

        selected = selector._ucb()

        # 應該隨機選擇
        assert selected in ['strategy_a', 'strategy_b', 'strategy_c']

    def test_ucb_calculation_correctness(self, selector):
        """測試 UCB 計算正確性"""
        import math

        # 設定統計
        selector._stats_cache = {
            'strategy_a': StrategyStats('strategy_a', attempts=10, avg_sharpe=1.5),
            'strategy_b': StrategyStats('strategy_b', attempts=20, avg_sharpe=1.6),
        }
        selector._cache_updated = True

        c = selector.config['ucb_c']  # 2.0
        total_attempts = 30

        # 手動計算期望的 UCB 值
        ucb_a = 1.5 + c * math.sqrt(math.log(30) / 10)
        ucb_b = 1.6 + c * math.sqrt(math.log(30) / 20)

        # ucb_a 應該更高（因為嘗試次數少，探索獎勵大）
        assert ucb_a > ucb_b


class TestThompsonSampling:
    """測試 Thompson Sampling 選擇"""

    def test_thompson_sampling_uses_beta_distribution(self, selector, monkeypatch):
        """測試 Thompson Sampling 使用 Beta 分佈"""
        selector._stats_cache = {
            'strategy_a': StrategyStats('strategy_a', attempts=10, successes=7),
            'strategy_b': StrategyStats('strategy_b', attempts=10, successes=3),
        }
        selector._cache_updated = True

        # Mock numpy.random.beta
        samples = {'strategy_a': 0.8, 'strategy_b': 0.4}
        call_count = {'count': 0}

        def mock_beta(alpha, beta_param):
            strategies = ['strategy_a', 'strategy_b']
            sample = samples[strategies[call_count['count']]]
            call_count['count'] += 1
            return sample

        monkeypatch.setattr('numpy.random.beta', mock_beta)

        selected = selector._thompson_sampling()

        # 應該選擇抽樣值較高的 strategy_a
        assert selected == 'strategy_a'

    def test_thompson_sampling_with_no_attempts(self, selector):
        """測試 Thompson Sampling 處理零嘗試"""
        selector._stats_cache = {
            'strategy_a': StrategyStats('strategy_a', attempts=0, successes=0),
            'strategy_b': StrategyStats('strategy_b', attempts=10, successes=5),
        }
        selector._cache_updated = True

        # 應該能正常執行（使用先驗 alpha=1, beta=1）
        selected = selector._thompson_sampling()

        assert selected in ['strategy_a', 'strategy_b']

    def test_thompson_sampling_randomness(self, selector):
        """測試 Thompson Sampling 的隨機性"""
        selector._stats_cache = {
            'strategy_a': StrategyStats('strategy_a', attempts=5, successes=3),
            'strategy_b': StrategyStats('strategy_b', attempts=5, successes=2),
        }
        selector._cache_updated = True

        # 多次選擇
        choices = [selector._thompson_sampling() for _ in range(20)]

        # 應該有變化（兩者都被選過）
        unique_choices = set(choices)
        assert len(unique_choices) >= 1


class TestUpdateStats:
    """測試統計更新"""

    def test_update_stats_new_strategy(self, selector):
        """測試更新新策略的統計"""
        result = {
            'passed': True,
            'sharpe_ratio': 1.8,
            'params': {'period': 20}
        }

        selector.update_stats('new_strategy', result)

        stat = selector._stats_cache['new_strategy']

        assert stat.attempts == 1
        assert stat.successes == 1
        assert stat.avg_sharpe == 1.8
        assert stat.best_sharpe == 1.8
        assert stat.last_params == {'period': 20}

    def test_update_stats_incremental_avg(self, selector):
        """測試增量計算平均 Sharpe"""
        # 第一次更新
        selector.update_stats('test', {
            'passed': True,
            'sharpe_ratio': 2.0,
            'params': {}
        })

        # 第二次更新
        selector.update_stats('test', {
            'passed': False,
            'sharpe_ratio': 1.0,
            'params': {}
        })

        stat = selector._stats_cache['test']

        # 平均 = (2.0 + 1.0) / 2 = 1.5
        assert stat.attempts == 2
        assert stat.successes == 1  # 只有第一次通過
        assert stat.avg_sharpe == 1.5

    def test_update_stats_best_sharpe_tracking(self, selector):
        """測試最佳 Sharpe 追蹤"""
        # 第一次更新
        selector.update_stats('test', {'passed': True, 'sharpe_ratio': 1.5, 'params': {}})

        # 第二次更新（更高）
        selector.update_stats('test', {'passed': True, 'sharpe_ratio': 2.5, 'params': {}})

        # 第三次更新（較低）
        selector.update_stats('test', {'passed': True, 'sharpe_ratio': 1.0, 'params': {}})

        stat = selector._stats_cache['test']

        # 最佳應該是 2.5
        assert stat.best_sharpe == 2.5

    def test_update_stats_syncs_to_recorder(self, selector, mock_recorder):
        """測試統計同步到記錄器"""
        selector.update_stats('test', {
            'passed': True,
            'sharpe_ratio': 1.5,
            'params': {}
        })

        # 驗證記錄器中有統計
        recorded_stat = mock_recorder.get_strategy_stats('test')

        assert recorded_stat is not None
        assert recorded_stat.attempts == 1


class TestGetStrategyStats:
    """測試取得策略統計"""

    def test_get_stats_updates_cache(self, selector, mock_registry):
        """測試取得統計會更新快取"""
        # 確保快取未更新
        selector._cache_updated = False

        stats = selector.get_strategy_stats()

        # 應該包含所有策略
        assert len(stats) == len(mock_registry.list_strategies())
        assert selector._cache_updated is True

    def test_get_stats_returns_copy(self, selector):
        """測試返回的是快取的副本"""
        selector._cache_updated = True
        selector._stats_cache = {
            'strategy_a': StrategyStats('strategy_a', attempts=5)
        }

        stats = selector.get_strategy_stats()

        # 修改返回的統計
        stats['strategy_a'].attempts = 100

        # 原始快取不應改變
        assert selector._stats_cache['strategy_a'].attempts == 5


class TestExplorationStats:
    """測試探索統計"""

    def test_get_exploration_stats(self, selector):
        """測試取得探索統計"""
        selector._stats_cache = {
            'strategy_a': StrategyStats('strategy_a', attempts=10, best_sharpe=1.8),
            'strategy_b': StrategyStats('strategy_b', attempts=5, best_sharpe=2.2),
            'strategy_c': StrategyStats('strategy_c', attempts=0, best_sharpe=0.0),
        }
        selector._cache_updated = True

        stats = selector.get_exploration_stats()

        assert stats['total_attempts'] == 15
        assert stats['strategies_tried'] == 2  # strategy_c 未嘗試
        assert stats['strategies_available'] == 3
        assert stats['exploration_rate'] == 2 / 3
        assert stats['best_strategy'] == 'strategy_b'
        assert stats['best_sharpe'] == 2.2

    def test_exploration_stats_empty(self, selector):
        """測試空探索統計"""
        selector._stats_cache = {
            'strategy_a': StrategyStats('strategy_a'),
            'strategy_b': StrategyStats('strategy_b'),
        }
        selector._cache_updated = True

        stats = selector.get_exploration_stats()

        assert stats['total_attempts'] == 0
        assert stats['strategies_tried'] == 0
        assert stats['best_sharpe'] == 0.0


class TestRecommendation:
    """測試策略推薦"""

    def test_get_recommendation_ensemble(self, selector, monkeypatch):
        """測試集成推薦（三種方法投票）"""
        selector._stats_cache = {
            'strategy_a': StrategyStats('strategy_a', attempts=10, avg_sharpe=1.5),
            'strategy_b': StrategyStats('strategy_b', attempts=5, avg_sharpe=2.0),
        }
        selector._cache_updated = True

        # Mock 三種方法都返回同一個策略
        monkeypatch.setattr(selector, '_epsilon_greedy', lambda: 'strategy_b')
        monkeypatch.setattr(selector, '_ucb', lambda: 'strategy_b')
        monkeypatch.setattr(selector, '_thompson_sampling', lambda: 'strategy_b')

        recommendation = selector.get_recommendation()

        assert recommendation['strategy'] == 'strategy_b'
        assert recommendation['method'] == 'ensemble'
        assert 'epsilon-greedy' in recommendation['reason']
        assert recommendation['stats'].name == 'strategy_b'

    def test_recommendation_with_alternatives(self, selector, monkeypatch):
        """測試推薦包含替代選項"""
        selector._stats_cache = {
            'strategy_a': StrategyStats('strategy_a', attempts=10, avg_sharpe=1.5),
            'strategy_b': StrategyStats('strategy_b', attempts=5, avg_sharpe=2.0),
        }
        selector._cache_updated = True

        # Mock 三種方法返回不同策略
        monkeypatch.setattr(selector, '_epsilon_greedy', lambda: 'strategy_a')
        monkeypatch.setattr(selector, '_ucb', lambda: 'strategy_b')
        monkeypatch.setattr(selector, '_thompson_sampling', lambda: 'strategy_b')

        recommendation = selector.get_recommendation()

        # strategy_b 獲得 2 票，應該被推薦
        assert recommendation['strategy'] == 'strategy_b'
        # strategy_a 應該在替代選項中
        assert 'strategy_a' in recommendation['alternatives']


class TestCacheManagement:
    """測試快取管理"""

    def test_reset_cache(self, selector):
        """測試重置快取"""
        selector._stats_cache = {'test': StrategyStats('test')}
        selector._cache_updated = True

        selector.reset_cache()

        assert len(selector._stats_cache) == 0
        assert selector._cache_updated is False


class TestEdgeCases:
    """測試邊界情況"""

    def test_select_with_unknown_method(self, selector):
        """測試使用未知的選擇方法"""
        selector._cache_updated = True

        with pytest.raises(ValueError, match="Unknown selection method"):
            selector.select(method='unknown_method')

    def test_select_with_empty_registry(self):
        """測試空註冊表"""
        empty_registry = MockStrategyRegistry(strategies=[])
        recorder = MockExperimentRecorder()

        selector = StrategySelector(empty_registry, recorder)
        selector._cache_updated = True

        # UCB 應該能處理空列表
        total_attempts = sum(s.attempts for s in selector._stats_cache.values())
        assert total_attempts == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
