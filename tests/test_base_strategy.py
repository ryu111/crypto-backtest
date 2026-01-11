"""
BaseStrategy 測試

測試策略基礎類別的核心功能。
"""

import pytest
import pandas as pd
import numpy as np
from src.strategies.base import BaseStrategy, TrendStrategy, MomentumStrategy


class ConcreteStrategy(BaseStrategy):
    """具體策略實作（用於測試）"""

    name = "test_strategy"
    strategy_type = "test"
    version = "1.0"
    description = "Test strategy for unit testing"

    def __init__(self, **kwargs):
        # 設定預設參數
        default_params = {
            'period': 20,
            'threshold': 0.5
        }
        # 在呼叫父類別之前設定實例屬性
        self.params = {}
        self.param_space = {
            'period': (10, 50, 5),
            'threshold': (0.1, 1.0, 0.1)
        }

        # 合併預設參數
        self.params.update(default_params)
        self.params.update(kwargs)

        # 呼叫父類別初始化（會執行驗證）
        super().__init__(**self.params)

    def calculate_indicators(self, data: pd.DataFrame):
        """計算測試指標"""
        return {
            'sma': data['close'].rolling(self.params['period']).mean()
        }

    def generate_signals(self, data: pd.DataFrame):
        """生成測試訊號"""
        indicators = self.calculate_indicators(data)
        sma = indicators['sma']

        long_entry = data['close'] > sma
        long_exit = data['close'] < sma
        short_entry = data['close'] < sma
        short_exit = data['close'] > sma

        return long_entry, long_exit, short_entry, short_exit


class TestBaseStrategyInstanceAttributes:
    """測試實例屬性（不共享）"""

    def test_params_not_shared_between_instances(self):
        """驗證 params 是實例屬性，不在實例間共享"""
        strategy1 = ConcreteStrategy(period=10)
        strategy2 = ConcreteStrategy(period=20)

        # 驗證各自擁有獨立的 params
        assert strategy1.params['period'] == 10
        assert strategy2.params['period'] == 20

        # 修改其中一個不應影響另一個
        strategy1.params['period'] = 30
        assert strategy1.params['period'] == 30
        assert strategy2.params['period'] == 20  # 未改變

    def test_param_space_not_shared_between_instances(self):
        """驗證 param_space 是實例屬性"""
        strategy1 = ConcreteStrategy()
        strategy2 = ConcreteStrategy()

        # 先確認初始值相同
        assert strategy1.param_space['period'] == (10, 50, 5)
        assert strategy2.param_space['period'] == (10, 50, 5)

        # 修改其中一個的 param_space
        strategy1.param_space['period'] = (5, 30, 5)

        # 驗證另一個未受影響（因為是獨立的實例屬性）
        assert strategy1.param_space['period'] == (5, 30, 5)
        assert strategy2.param_space['period'] == (10, 50, 5)

    def test_multiple_instances_independent(self):
        """驗證多個實例完全獨立"""
        strategies = [
            ConcreteStrategy(period=10 * i, threshold=0.1 * i)
            for i in range(1, 5)
        ]

        # 驗證每個實例的參數正確
        for i, strategy in enumerate(strategies, start=1):
            assert strategy.params['period'] == 10 * i
            assert strategy.params['threshold'] == 0.1 * i


class TestBaseStrategyInheritance:
    """測試子類別繼承"""

    def test_concrete_strategy_inherits_base(self):
        """驗證子類別正確繼承 BaseStrategy"""
        strategy = ConcreteStrategy()

        assert isinstance(strategy, BaseStrategy)
        assert hasattr(strategy, 'calculate_indicators')
        assert hasattr(strategy, 'generate_signals')
        assert hasattr(strategy, 'position_size')

    def test_trend_strategy_inherits_base(self):
        """驗證 TrendStrategy 正確繼承"""

        class ConcreteTrendStrategy(TrendStrategy):
            name = "test_trend"

            def __init__(self, **kwargs):
                self.params = kwargs
                self.param_space = {}
                super().__init__(**kwargs)

            def calculate_indicators(self, data):
                return {}

            def generate_signals(self, data):
                return pd.Series([False]), pd.Series([False]), pd.Series([False]), pd.Series([False])

        strategy = ConcreteTrendStrategy()

        assert isinstance(strategy, BaseStrategy)
        assert strategy.strategy_type == "trend"
        assert hasattr(strategy, 'apply_trend_filter')

    def test_momentum_strategy_inherits_base(self):
        """驗證 MomentumStrategy 正確繼承"""

        class ConcreteMomentumStrategy(MomentumStrategy):
            name = "test_momentum"

            def __init__(self, **kwargs):
                self.params = kwargs
                self.param_space = {}
                super().__init__(**kwargs)

            def calculate_indicators(self, data):
                return {}

            def generate_signals(self, data):
                return pd.Series([False]), pd.Series([False]), pd.Series([False]), pd.Series([False])

        strategy = ConcreteMomentumStrategy()

        assert isinstance(strategy, BaseStrategy)
        assert strategy.strategy_type == "momentum"
        assert hasattr(strategy, 'calculate_rsi')
        assert hasattr(strategy, 'calculate_macd')


class TestBaseStrategyMethods:
    """測試基礎方法"""

    def test_position_size_calculation(self):
        """測試部位大小計算"""
        strategy = ConcreteStrategy()

        # 正常情況
        size = strategy.position_size(
            capital=10000,
            entry_price=100,
            stop_loss_price=95,
            risk_per_trade=0.02
        )

        # 風險金額 = 10000 * 0.02 = 200
        # 止損距離 = 100 - 95 = 5
        # 部位大小 = 200 / 5 = 40
        assert size == 40

    def test_position_size_zero_stop_distance(self):
        """測試止損距離為零的情況"""
        strategy = ConcreteStrategy()

        size = strategy.position_size(
            capital=10000,
            entry_price=100,
            stop_loss_price=100,  # 同價格
            risk_per_trade=0.02
        )

        assert size == 0

    def test_position_size_max_position_limit(self):
        """測試最大部位限制"""
        strategy = ConcreteStrategy()

        size = strategy.position_size(
            capital=10000,
            entry_price=100,
            stop_loss_price=99.99,  # 極小止損距離
            risk_per_trade=0.02,
            max_position_pct=0.5  # 最大 50% 部位
        )

        # 最大部位 = 10000 * 0.5 / 100 = 50
        assert size <= 50

    def test_validate_params_all_valid(self):
        """測試參數驗證（全部有效）"""
        strategy = ConcreteStrategy(period=20, threshold=0.5)

        assert strategy.validate_params() is True

    def test_validate_params_none_value(self):
        """測試參數驗證（包含 None）"""
        # 建立策略時會呼叫驗證，None 應該失敗
        with pytest.raises(ValueError, match="Invalid parameters"):
            ConcreteStrategy(period=None)

    def test_get_info(self):
        """測試取得策略資訊"""
        strategy = ConcreteStrategy(period=25)

        info = strategy.get_info()

        assert info['name'] == 'test_strategy'
        assert info['type'] == 'test'
        assert info['version'] == '1.0'
        assert info['description'] == 'Test strategy for unit testing'
        assert info['params']['period'] == 25
        assert 'param_space' in info

    def test_apply_filters_default_no_filtering(self):
        """測試預設過濾器（不過濾）"""
        strategy = ConcreteStrategy()

        # 建立測試資料
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104]
        })

        long_entry = pd.Series([True, False, True, False, False])
        long_exit = pd.Series([False, True, False, True, False])
        short_entry = pd.Series([False, False, False, False, True])
        short_exit = pd.Series([False, False, False, False, False])

        # 應用過濾器（預設不過濾）
        filtered = strategy.apply_filters(
            data, long_entry, long_exit, short_entry, short_exit
        )

        # 驗證未改變
        pd.testing.assert_series_equal(filtered[0], long_entry)
        pd.testing.assert_series_equal(filtered[1], long_exit)
        pd.testing.assert_series_equal(filtered[2], short_entry)
        pd.testing.assert_series_equal(filtered[3], short_exit)

    def test_repr_and_str(self):
        """測試字串表示"""
        strategy = ConcreteStrategy(period=25)

        repr_str = repr(strategy)
        str_str = str(strategy)

        assert 'test_strategy' in repr_str
        assert 'test' in repr_str  # strategy_type
        assert 'period' in repr_str

        assert 'test_strategy' in str_str
        assert '1.0' in str_str  # version


class TestBaseStrategySignalGeneration:
    """測試訊號生成"""

    def test_generate_signals_with_real_data(self):
        """測試使用真實資料生成訊號"""
        strategy = ConcreteStrategy(period=5)

        # 建立測試資料
        data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 104, 103, 102, 101],
            'high': [101, 102, 103, 104, 105, 106, 105, 104, 103, 102],
            'low': [99, 100, 101, 102, 103, 104, 103, 102, 101, 100],
            'close': [100, 101, 102, 103, 104, 105, 104, 103, 102, 101],
            'volume': [1000] * 10
        })

        long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(data)

        # 驗證訊號是 boolean Series
        assert isinstance(long_entry, pd.Series)
        assert isinstance(long_exit, pd.Series)
        assert isinstance(short_entry, pd.Series)
        assert isinstance(short_exit, pd.Series)

        # 驗證長度正確
        assert len(long_entry) == len(data)
        assert len(long_exit) == len(data)
        assert len(short_entry) == len(data)
        assert len(short_exit) == len(data)

        # 驗證訊號類型為 bool
        assert long_entry.dtype == bool
        assert long_exit.dtype == bool

    def test_calculate_indicators(self):
        """測試指標計算"""
        strategy = ConcreteStrategy(period=3)

        data = pd.DataFrame({
            'close': [100, 102, 104, 106, 108]
        })

        indicators = strategy.calculate_indicators(data)

        assert 'sma' in indicators
        assert isinstance(indicators['sma'], pd.Series)

        # 驗證 SMA 計算正確（前兩個為 NaN）
        assert pd.isna(indicators['sma'].iloc[0])
        assert pd.isna(indicators['sma'].iloc[1])

        # 第三個值 = (100 + 102 + 104) / 3
        assert abs(indicators['sma'].iloc[2] - 102.0) < 0.01


class TestTrendStrategy:
    """測試趨勢策略基礎類別"""

    def test_apply_trend_filter(self):
        """測試趨勢過濾器"""

        class TestTrend(TrendStrategy):
            name = "test_trend"

            def __init__(self, **kwargs):
                self.params = kwargs
                self.param_space = {}
                super().__init__(**kwargs)

            def calculate_indicators(self, data):
                return {}

            def generate_signals(self, data):
                return pd.Series([False]), pd.Series([False]), pd.Series([False]), pd.Series([False])

        strategy = TestTrend()

        # 建立趨勢資料（向上）
        data = pd.DataFrame({
            'close': [100, 102, 104, 106, 108, 110]
        })

        uptrend, downtrend = strategy.apply_trend_filter(data, period=3)

        assert isinstance(uptrend, pd.Series)
        assert isinstance(downtrend, pd.Series)

        # 後期應該為上升趨勢（價格高於 MA）
        assert uptrend.iloc[-1] is True or uptrend.iloc[-1] == True


class TestMomentumStrategy:
    """測試動量策略基礎類別"""

    def test_calculate_rsi(self):
        """測試 RSI 計算"""

        class TestMomentum(MomentumStrategy):
            name = "test_momentum"

            def __init__(self, **kwargs):
                self.params = kwargs
                self.param_space = {}
                super().__init__(**kwargs)

            def calculate_indicators(self, data):
                return {}

            def generate_signals(self, data):
                return pd.Series([False]), pd.Series([False]), pd.Series([False]), pd.Series([False])

        strategy = TestMomentum()

        # 建立測試資料（先上漲再下跌）
        close_prices = [100] + [100 + i for i in range(1, 15)] + [113 - i for i in range(1, 10)]
        data = pd.DataFrame({'close': close_prices})

        rsi = strategy.calculate_rsi(data['close'], period=14)

        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(data)

        # RSI 應該在 0-100 之間（排除 NaN）
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_calculate_macd(self):
        """測試 MACD 計算"""

        class TestMomentum(MomentumStrategy):
            name = "test_momentum"

            def __init__(self, **kwargs):
                self.params = kwargs
                self.param_space = {}
                super().__init__(**kwargs)

            def calculate_indicators(self, data):
                return {}

            def generate_signals(self, data):
                return pd.Series([False]), pd.Series([False]), pd.Series([False]), pd.Series([False])

        strategy = TestMomentum()

        # 建立測試資料
        close_prices = [100 + i * 0.5 for i in range(50)]
        data = pd.DataFrame({'close': close_prices})

        macd_line, signal_line, histogram = strategy.calculate_macd(
            data['close'], fast=12, slow=26, signal=9
        )

        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert isinstance(histogram, pd.Series)

        assert len(macd_line) == len(data)
        assert len(signal_line) == len(data)
        assert len(histogram) == len(data)

        # Histogram = MACD - Signal
        pd.testing.assert_series_equal(
            histogram,
            macd_line - signal_line,
            check_names=False
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
