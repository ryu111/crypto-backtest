"""
Integration tests for GPStrategyAdapter with real components

Tests the full integration:
- ExpressionConverter + PrimitiveSetFactory
- GPStrategyAdapter
- Generated strategy execution
"""

import pytest
import numpy as np
import pandas as pd

# 只在 DEAP 可用時執行測試
pytest.importorskip("deap", reason="DEAP is required for integration tests")

from src.gp.primitives import PrimitiveSetFactory
from src.gp.converter import ExpressionConverter
from src.automation.gp_integration import GPStrategyAdapter
from src.strategies.base import BaseStrategy


class TestGPAdapterIntegration:
    """Integration tests for GPStrategyAdapter"""

    @pytest.fixture
    def pset(self):
        """建立 DEAP primitive set"""
        factory = PrimitiveSetFactory()
        return factory.create_standard_set()

    @pytest.fixture
    def converter(self, pset):
        """建立 expression converter"""
        return ExpressionConverter(pset)

    @pytest.fixture
    def adapter(self, converter):
        """建立 strategy adapter"""
        return GPStrategyAdapter(converter)

    @pytest.fixture
    def simple_individual(self, pset):
        """建立簡單的 GP 個體（mock）"""
        # 使用 mock individual 來避免複雜的樹生成
        class MockIndividual:
            """Mock GP individual for testing"""
            def __str__(self):
                return "GT(RSI(close, period_14), rsi_overbought)"

        # 添加必要屬性供 ExpressionConverter 使用
        individual = MockIndividual()

        return individual

    @pytest.fixture
    def sample_data(self):
        """建立測試數據"""
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        np.random.seed(42)
        close = np.cumsum(np.random.randn(100)) + 100
        high = close + np.abs(np.random.randn(100))
        low = close - np.abs(np.random.randn(100))
        open_ = close + np.random.randn(100)
        volume = np.random.randint(1000, 10000, 100)

        return pd.DataFrame({
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)

    def test_adapter_creates_valid_strategy_class(self, adapter, simple_individual):
        """應該建立有效的策略類別"""
        strategy_class = adapter.create_strategy_class(
            individual=simple_individual,
            strategy_name='gp_integration_test_001',
            fitness=1.85,
            generation=5
        )

        # 驗證類別
        assert isinstance(strategy_class, type)
        assert issubclass(strategy_class, BaseStrategy)

    def test_strategy_instance_has_correct_attributes(self, adapter, simple_individual):
        """策略實例應該有正確的屬性"""
        strategy_class = adapter.create_strategy_class(
            individual=simple_individual,
            strategy_name='test_strategy',
            fitness=2.5,
            generation=10
        )

        strategy = strategy_class()

        assert strategy.name == 'test_strategy'
        assert strategy.fitness_score == 2.5
        assert strategy.generation == 10
        assert strategy.expression != ""  # 應該有表達式字串

    def test_strategy_can_generate_signals(self, adapter, simple_individual, sample_data):
        """策略應該能產生交易訊號"""
        strategy_class = adapter.create_strategy_class(
            individual=simple_individual,
            strategy_name='test_signals',
            fitness=1.0,
            generation=0
        )

        strategy = strategy_class()

        # 產生訊號
        long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(sample_data)

        # 驗證訊號格式
        assert isinstance(long_entry, pd.Series)
        assert isinstance(long_exit, pd.Series)
        assert isinstance(short_entry, pd.Series)
        assert isinstance(short_exit, pd.Series)

        # 驗證長度
        assert len(long_entry) == len(sample_data)
        assert len(long_exit) == len(sample_data)

        # 驗證布林型別
        assert long_entry.dtype == bool
        assert long_exit.dtype == bool

    def test_strategy_signals_are_valid(self, adapter, simple_individual, sample_data):
        """策略訊號應該是合理的"""
        strategy_class = adapter.create_strategy_class(
            individual=simple_individual,
            strategy_name='test_valid_signals',
            fitness=1.0,
            generation=0
        )

        strategy = strategy_class()
        long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(sample_data)

        # 訊號應該有 True 和 False（不全是同一個值）
        # 注意：某些極端情況可能所有值都相同，但機率很低
        assert long_entry.dtype == bool
        assert long_exit.dtype == bool

        # 多單進場和出場應該互補（對於簡單策略）
        # 注意：這取決於 EvolvedStrategy 的實作
        total_signals = long_entry.sum() + long_exit.sum()
        assert total_signals > 0  # 至少應該有一些訊號

    def test_multiple_strategies_can_coexist(self, adapter, simple_individual):
        """應該能建立多個不同的策略實例"""
        strategy_class_1 = adapter.create_strategy_class(
            individual=simple_individual,
            strategy_name='strategy_001',
            fitness=1.5,
            generation=1
        )

        strategy_class_2 = adapter.create_strategy_class(
            individual=simple_individual,
            strategy_name='strategy_002',
            fitness=2.0,
            generation=2
        )

        strategy_1 = strategy_class_1()
        strategy_2 = strategy_class_2()

        # 驗證獨立性
        assert strategy_1.name != strategy_2.name
        assert strategy_1.fitness_score != strategy_2.fitness_score
        assert strategy_1.generation != strategy_2.generation

    def test_strategy_get_info_method(self, adapter, simple_individual):
        """get_info() 應該返回完整資訊"""
        strategy_class = adapter.create_strategy_class(
            individual=simple_individual,
            strategy_name='test_info',
            fitness=1.8,
            generation=7
        )

        strategy = strategy_class()
        info = strategy.get_info()

        # 驗證基本資訊
        assert info['name'] == 'test_info'
        assert info['type'] == 'evolved'
        assert info['version'] == '1.0'

        # 驗證演化元資料
        assert 'expression' in info
        assert info['fitness_score'] == 1.8
        assert info['generation'] == 7
        assert 'evolved_at' in info

    def test_error_handling_missing_data_columns(self, adapter, simple_individual):
        """應該處理缺少欄位的數據"""
        strategy_class = adapter.create_strategy_class(
            individual=simple_individual,
            strategy_name='test_error',
            fitness=1.0,
            generation=0
        )

        strategy = strategy_class()

        # 缺少 'close' 欄位的數據
        bad_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97]
            # 缺少 'close'
        })

        with pytest.raises(KeyError) as exc_info:
            strategy.generate_signals(bad_data)

        assert 'close' in str(exc_info.value).lower()


if __name__ == '__main__':
    pytest.main([__file__, '-xvs'])
