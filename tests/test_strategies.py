"""
策略基礎架構單元測試
"""

import sys
from pathlib import Path

# 將專案根目錄加入路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import unittest
import pandas as pd
import numpy as np

from src.strategies import (
    BaseStrategy,
    TrendStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    register_strategy,
    list_strategies,
    create_strategy,
    get_strategy,
    StrategyRegistry
)


class TestBaseStrategy(unittest.TestCase):
    """測試策略基礎類別"""

    def setUp(self):
        """設定測試環境"""
        # 清空註冊表
        StrategyRegistry.clear()

        # 建立測試策略
        @register_strategy('test_strategy')
        class TestStrategy(BaseStrategy):
            params = {'period': 10}
            param_space = {
                'period': {'type': 'int', 'low': 5, 'high': 20}
            }
            strategy_type = 'test'

            def calculate_indicators(self, data):
                return {'ma': data['close'].rolling(self.params['period']).mean()}

            def generate_signals(self, data):
                ma = self.calculate_indicators(data)['ma']
                long_entry = data['close'] > ma
                long_exit = data['close'] < ma
                return long_entry, long_exit, None, None

        self.TestStrategy = TestStrategy

    def test_strategy_creation(self):
        """測試策略建立"""
        strategy = create_strategy('test_strategy')
        self.assertIsInstance(strategy, BaseStrategy)
        self.assertEqual(strategy.params['period'], 10)

    def test_param_override(self):
        """測試參數覆寫"""
        strategy = create_strategy('test_strategy', period=20)
        self.assertEqual(strategy.params['period'], 20)

    def test_signal_generation(self):
        """測試訊號產生"""
        data = self._create_test_data()
        strategy = create_strategy('test_strategy')
        long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(data)

        self.assertIsInstance(long_entry, pd.Series)
        self.assertEqual(len(long_entry), len(data))

    def test_position_sizing(self):
        """測試部位計算"""
        strategy = create_strategy('test_strategy')
        size = strategy.position_size(
            capital=10000,
            entry_price=50000,
            stop_loss_price=49000,
            risk_per_trade=0.02
        )
        self.assertGreater(size, 0)

    def _create_test_data(self, n=100):
        """建立測試資料"""
        np.random.seed(42)
        return pd.DataFrame({
            'open': np.random.randn(n).cumsum() + 100,
            'high': np.random.randn(n).cumsum() + 101,
            'low': np.random.randn(n).cumsum() + 99,
            'close': np.random.randn(n).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, n)
        })


class TestStrategyRegistry(unittest.TestCase):
    """測試策略註冊表"""

    def setUp(self):
        """設定測試環境"""
        StrategyRegistry.clear()

    def test_register_strategy(self):
        """測試策略註冊"""
        @register_strategy('test_reg')
        class TestReg(BaseStrategy):
            def calculate_indicators(self, data):
                return {}
            def generate_signals(self, data):
                return None, None, None, None

        self.assertTrue(StrategyRegistry.exists('test_reg'))

    def test_duplicate_registration(self):
        """測試重複註冊"""
        @register_strategy('test_dup')
        class TestDup1(BaseStrategy):
            def calculate_indicators(self, data):
                return {}
            def generate_signals(self, data):
                return None, None, None, None

        with self.assertRaises(ValueError):
            @register_strategy('test_dup')
            class TestDup2(BaseStrategy):
                def calculate_indicators(self, data):
                    return {}
                def generate_signals(self, data):
                    return None, None, None, None

    def test_get_strategy(self):
        """測試取得策略"""
        @register_strategy('test_get')
        class TestGet(BaseStrategy):
            def calculate_indicators(self, data):
                return {}
            def generate_signals(self, data):
                return None, None, None, None

        strategy_class = get_strategy('test_get')
        self.assertEqual(strategy_class, TestGet)

    def test_list_strategies(self):
        """測試列出策略"""
        @register_strategy('test_list1')
        class TestList1(BaseStrategy):
            def calculate_indicators(self, data):
                return {}
            def generate_signals(self, data):
                return None, None, None, None

        @register_strategy('test_list2')
        class TestList2(BaseStrategy):
            def calculate_indicators(self, data):
                return {}
            def generate_signals(self, data):
                return None, None, None, None

        strategies = list_strategies()
        self.assertEqual(len(strategies), 2)
        self.assertIn('test_list1', strategies)
        self.assertIn('test_list2', strategies)

    def test_list_by_type(self):
        """測試按類型過濾"""
        @register_strategy('trend1')
        class Trend1(TrendStrategy):
            def calculate_indicators(self, data):
                return {}
            def generate_signals(self, data):
                return None, None, None, None

        @register_strategy('momentum1')
        class Momentum1(MomentumStrategy):
            def calculate_indicators(self, data):
                return {}
            def generate_signals(self, data):
                return None, None, None, None

        trend_strategies = StrategyRegistry.list_by_type('trend')
        self.assertEqual(len(trend_strategies), 1)
        self.assertIn('trend1', trend_strategies)


class TestStrategyTypes(unittest.TestCase):
    """測試策略類型輔助類別"""

    def setUp(self):
        """設定測試資料"""
        np.random.seed(42)
        self.data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })

    def test_momentum_strategy_rsi(self):
        """測試動量策略 RSI 計算"""
        # 建立具體實作類別
        class ConcreteMomentum(MomentumStrategy):
            def calculate_indicators(self, data):
                return {}
            def generate_signals(self, data):
                return None, None, None, None

        strategy = ConcreteMomentum()
        rsi = strategy.calculate_rsi(self.data['close'], period=14)
        self.assertIsInstance(rsi, pd.Series)
        self.assertTrue(all((rsi >= 0) & (rsi <= 100) | rsi.isna()))

    def test_mean_reversion_bollinger(self):
        """測試均值回歸策略布林帶計算"""
        # 建立具體實作類別
        class ConcreteMeanReversion(MeanReversionStrategy):
            def calculate_indicators(self, data):
                return {}
            def generate_signals(self, data):
                return None, None, None, None

        strategy = ConcreteMeanReversion()
        upper, middle, lower = strategy.calculate_bollinger_bands(
            self.data['close'], period=20, std_dev=2.0
        )
        self.assertIsInstance(upper, pd.Series)
        # 移除 NaN 後檢查
        valid_mask = ~(upper.isna() | middle.isna() | lower.isna())
        self.assertTrue(all(upper[valid_mask] >= middle[valid_mask]))
        self.assertTrue(all(middle[valid_mask] >= lower[valid_mask]))


if __name__ == '__main__':
    # 執行測試
    unittest.main(verbosity=2)
