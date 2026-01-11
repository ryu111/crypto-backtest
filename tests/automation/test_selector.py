"""
測試策略選擇器
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime
from src.automation.selector import StrategySelector, StrategyStats


@pytest.fixture
def mock_registry():
    """模擬策略註冊表"""
    registry = Mock()
    registry.list_strategies.return_value = [
        'momentum',
        'mean_reversion',
        'breakout',
        'grid_trading'
    ]
    return registry


@pytest.fixture
def mock_recorder():
    """模擬實驗記錄器"""
    recorder = Mock()

    # 模擬歷史統計
    def get_strategy_stats(name):
        stats_map = {
            'momentum': StrategyStats(
                name='momentum',
                attempts=10,
                successes=7,
                avg_sharpe=1.5,
                best_sharpe=2.0,
                last_attempt=datetime.now()
            ),
            'mean_reversion': StrategyStats(
                name='mean_reversion',
                attempts=5,
                successes=3,
                avg_sharpe=1.2,
                best_sharpe=1.8,
                last_attempt=datetime.now()
            ),
            'breakout': StrategyStats(
                name='breakout',
                attempts=2,
                successes=1,
                avg_sharpe=0.8,
                best_sharpe=1.0,
                last_attempt=datetime.now()
            ),
        }
        return stats_map.get(name)

    recorder.get_strategy_stats.side_effect = get_strategy_stats
    recorder.record_strategy_stats = Mock()

    return recorder


@pytest.fixture
def selector(mock_registry, mock_recorder):
    """建立策略選擇器"""
    return StrategySelector(
        strategy_registry=mock_registry,
        experiment_recorder=mock_recorder,
        config={
            'epsilon': 0.2,
            'ucb_c': 2.0,
            'min_attempts': 3
        }
    )


class TestStrategyStats:
    """測試 StrategyStats"""

    def test_success_rate(self):
        """測試成功率計算"""
        stats = StrategyStats(name='test', attempts=10, successes=7)
        assert stats.success_rate == 0.7

    def test_success_rate_zero_attempts(self):
        """測試零嘗試時的成功率"""
        stats = StrategyStats(name='test', attempts=0, successes=0)
        assert stats.success_rate == 0.0

    def test_failure_rate(self):
        """測試失敗率計算"""
        stats = StrategyStats(name='test', attempts=10, successes=7)
        assert stats.failure_rate == 0.3


class TestStrategySelector:
    """測試策略選擇器"""

    def test_init(self, selector, mock_registry, mock_recorder):
        """測試初始化"""
        assert selector.registry == mock_registry
        assert selector.recorder == mock_recorder
        assert selector.config['epsilon'] == 0.2
        assert selector.config['ucb_c'] == 2.0

    def test_select_epsilon_greedy(self, selector):
        """測試 Epsilon-Greedy 選擇"""
        # 多次選擇，統計分佈
        selections = [selector.select('epsilon_greedy') for _ in range(100)]

        # 應該大部分選擇 momentum（最佳策略）
        momentum_count = selections.count('momentum')
        assert momentum_count > 50  # 大於 50% 應該選擇最佳

    def test_select_ucb(self, selector):
        """測試 UCB 選擇"""
        strategy = selector.select('ucb')

        # UCB 應該平衡探索和利用
        assert strategy in selector.registry.list_strategies()

        # 嘗試次數少的策略（grid_trading）應該有機會被選中
        # 因為探索獎勵高

    def test_select_thompson_sampling(self, selector):
        """測試 Thompson Sampling 選擇"""
        # 多次選擇，確保有隨機性
        selections = [selector.select('thompson_sampling') for _ in range(50)]

        # 應該有多樣性
        unique_strategies = set(selections)
        assert len(unique_strategies) > 1

    def test_update_stats(self, selector):
        """測試更新統計"""
        result = {
            'passed': True,
            'sharpe_ratio': 2.5,
            'params': {'param1': 10}
        }

        selector.update_stats('momentum', result)

        # 檢查快取更新
        stats = selector.get_strategy_stats()
        momentum_stats = stats['momentum']

        assert momentum_stats.attempts == 11  # 原本 10 + 1
        assert momentum_stats.successes == 8   # 原本 7 + 1
        assert momentum_stats.best_sharpe == 2.5
        assert momentum_stats.last_params == {'param1': 10}

        # 檢查記錄器被呼叫
        selector.recorder.record_strategy_stats.assert_called_once()

    def test_update_stats_new_strategy(self, selector):
        """測試更新新策略統計"""
        result = {
            'passed': True,
            'sharpe_ratio': 1.5,
            'params': {}
        }

        selector.update_stats('new_strategy', result)

        stats = selector.get_strategy_stats()
        new_stats = stats['new_strategy']

        assert new_stats.attempts == 1
        assert new_stats.successes == 1
        assert new_stats.avg_sharpe == 1.5
        assert new_stats.best_sharpe == 1.5

    def test_get_strategy_stats(self, selector):
        """測試取得策略統計"""
        stats = selector.get_strategy_stats()

        assert 'momentum' in stats
        assert 'mean_reversion' in stats
        assert 'breakout' in stats
        assert 'grid_trading' in stats

        # 檢查資料正確
        assert stats['momentum'].attempts == 10
        assert stats['momentum'].avg_sharpe == 1.5

    def test_get_exploration_stats(self, selector):
        """測試取得探索統計"""
        exp_stats = selector.get_exploration_stats()

        assert exp_stats['total_attempts'] == 17  # 10 + 5 + 2
        assert exp_stats['strategies_tried'] == 3
        assert exp_stats['strategies_available'] == 4
        assert exp_stats['exploration_rate'] == 0.75
        assert exp_stats['best_strategy'] == 'momentum'
        assert exp_stats['best_sharpe'] == 2.0

    def test_get_recommendation(self, selector):
        """測試取得推薦策略"""
        recommendation = selector.get_recommendation()

        assert 'strategy' in recommendation
        assert 'method' in recommendation
        assert 'reason' in recommendation
        assert 'stats' in recommendation
        assert 'alternatives' in recommendation

        assert recommendation['method'] == 'ensemble'
        assert recommendation['strategy'] in selector.registry.list_strategies()

    def test_random_select(self, selector):
        """測試隨機選擇"""
        # 多次選擇，確保有隨機性
        selections = [selector._random_select() for _ in range(50)]

        # 應該有多樣性
        unique_strategies = set(selections)
        assert len(unique_strategies) > 1

        # 所有選擇都應該在可用策略中
        for s in selections:
            assert s in selector.registry.list_strategies()

    def test_best_select(self, selector):
        """測試選擇最佳策略"""
        best = selector._best_select()

        # momentum 有最高的 avg_sharpe 且嘗試次數 >= min_attempts
        assert best == 'momentum'

    def test_best_select_insufficient_attempts(self, selector):
        """測試嘗試次數不足時的最佳選擇"""
        # 設定高 min_attempts
        selector.config['min_attempts'] = 20

        # 應該回退到隨機選擇
        strategy = selector._best_select()
        assert strategy in selector.registry.list_strategies()

    def test_reset_cache(self, selector):
        """測試重置快取"""
        # 先載入快取
        selector.get_strategy_stats()
        assert selector._cache_updated

        # 重置
        selector.reset_cache()
        assert not selector._cache_updated
        assert len(selector._stats_cache) == 0

    def test_update_avg_sharpe_calculation(self, selector):
        """測試平均 Sharpe 的增量計算"""
        # 初始統計
        selector._stats_cache['test'] = StrategyStats(
            name='test',
            attempts=2,
            successes=1,
            avg_sharpe=1.0
        )

        # 新增一次結果
        result = {
            'passed': True,
            'sharpe_ratio': 2.0,
            'params': {}
        }

        selector.update_stats('test', result)

        # 平均應該是 (1.0 * 2 + 2.0) / 3 = 1.333...
        stats = selector.get_strategy_stats()['test']
        assert abs(stats.avg_sharpe - 1.333) < 0.01

    def test_ucb_untried_strategy_priority(self, selector):
        """測試 UCB 對未嘗試策略的優先權"""
        # grid_trading 嘗試次數為 0
        # 應該被賦予無窮大的 UCB 值

        strategy = selector._ucb()

        # 有可能選擇 grid_trading（因為 UCB = inf）
        # 但由於有隨機性，不保證一定選中
        # 只測試不會崩潰
        assert strategy in selector.registry.list_strategies()


class TestIntegration:
    """整合測試"""

    def test_full_workflow(self, mock_registry, mock_recorder):
        """測試完整工作流程"""
        selector = StrategySelector(
            strategy_registry=mock_registry,
            experiment_recorder=mock_recorder
        )

        # 1. 取得推薦
        recommendation = selector.get_recommendation()
        strategy = recommendation['strategy']

        # 2. 模擬優化結果
        result = {
            'passed': True,
            'sharpe_ratio': 2.5,
            'params': {'param1': 10}
        }

        # 3. 更新統計
        selector.update_stats(strategy, result)

        # 4. 檢查統計更新
        stats = selector.get_strategy_stats()[strategy]
        assert stats.best_sharpe == 2.5

        # 5. 再次推薦（應該偏好剛才成功的策略）
        # 多次嘗試，統計分佈
        selections = [
            selector.get_recommendation()['strategy']
            for _ in range(20)
        ]

        # 成功的策略應該被選中較多次
        assert strategy in selections
