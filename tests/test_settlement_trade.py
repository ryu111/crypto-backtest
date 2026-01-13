"""
測試 Settlement Trade 策略

驗證策略邏輯、參數驗證、訊號生成的正確性。
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.strategies.funding_rate import SettlementTradeStrategy
from src.strategies.registry import StrategyRegistry


class TestSettlementTradeStrategy:
    """Settlement Trade 策略測試套件"""

    @pytest.fixture
    def strategy(self):
        """建立策略實例"""
        return SettlementTradeStrategy(
            rate_threshold=0.0001,
            hours_before_settlement=1
        )

    @pytest.fixture
    def sample_data(self):
        """建立測試數據（包含結算時間）"""
        # 建立 24 小時的小時級數據（包含 3 個結算時間：0, 8, 16）
        timestamps = pd.date_range(
            start='2024-01-01 00:00:00',
            periods=24,
            freq='h',
            tz='UTC'
        )

        data = pd.DataFrame({
            'open': np.random.uniform(40000, 41000, 24),
            'high': np.random.uniform(41000, 42000, 24),
            'low': np.random.uniform(39000, 40000, 24),
            'close': np.random.uniform(40000, 41000, 24),
            'volume': np.random.uniform(1000, 2000, 24)
        }, index=timestamps)

        return data

    @pytest.fixture
    def sample_funding_rates(self, sample_data):
        """建立測試資金費率數據"""
        # 建立與 sample_data 長度相同的費率數據
        rates = pd.Series([0.0] * len(sample_data), index=sample_data.index)

        # 設定一些高費率（在結算前 1 小時）
        rates.iloc[7] = 0.0003   # 08:00 前 1 小時（07:00）-> 高費率
        rates.iloc[15] = -0.0002  # 16:00 前 1 小時（15:00）-> 負費率
        rates.iloc[23] = 0.00015  # 00:00 前 1 小時（23:00）-> 高費率

        return rates

    def test_strategy_initialization(self):
        """測試策略初始化"""
        strategy = SettlementTradeStrategy(
            rate_threshold=0.0001,
            hours_before_settlement=2
        )

        assert strategy.name == "settlement_trade"
        assert strategy.strategy_type == "funding_rate"
        assert strategy.params['rate_threshold'] == 0.0001
        assert strategy.params['hours_before_settlement'] == 2

    def test_param_space_definition(self, strategy):
        """測試參數空間定義"""
        param_space = strategy.param_space

        # 驗證 rate_threshold
        assert 'rate_threshold' in param_space
        assert param_space['rate_threshold']['type'] == 'float'
        assert param_space['rate_threshold']['low'] == 0.00005
        assert param_space['rate_threshold']['high'] == 0.0005

        # 驗證 hours_before_settlement
        assert 'hours_before_settlement' in param_space
        assert param_space['hours_before_settlement']['type'] == 'int'
        assert param_space['hours_before_settlement']['low'] == 1
        assert param_space['hours_before_settlement']['high'] == 4

    def test_validate_params_success(self):
        """測試參數驗證（成功案例）"""
        strategy = SettlementTradeStrategy(
            rate_threshold=0.0002,
            hours_before_settlement=2
        )
        assert strategy.validate_params() is True

    def test_validate_params_failure_negative_threshold(self):
        """測試參數驗證（負閾值應失敗）"""
        with pytest.raises(ValueError):
            SettlementTradeStrategy(rate_threshold=-0.0001)

    def test_validate_params_failure_invalid_hours(self):
        """測試參數驗證（無效小時數應失敗）"""
        with pytest.raises(ValueError):
            SettlementTradeStrategy(hours_before_settlement=0)

        with pytest.raises(ValueError):
            SettlementTradeStrategy(hours_before_settlement=5)

    def test_calculate_indicators(self, strategy, sample_data):
        """測試指標計算（應返回空字典）"""
        indicators = strategy.calculate_indicators(sample_data)
        assert indicators == {}

    def test_generate_signals_without_funding(self, strategy, sample_data):
        """測試無資金費率數據時的訊號生成（應全部為 False）"""
        long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(
            sample_data
        )

        assert long_entry.sum() == 0
        assert long_exit.sum() == 0
        assert short_entry.sum() == 0
        assert short_exit.sum() == 0

    def test_generate_signals_with_funding_high_rate(
        self, strategy, sample_data, sample_funding_rates
    ):
        """測試高資金費率時的訊號生成（應做多）"""
        long_entry, long_exit, short_entry, short_exit = \
            strategy.generate_signals_with_funding(
                sample_data,
                sample_funding_rates
            )

        # 07:00 應觸發多單進場（費率 0.0003 > 0.0001）
        assert long_entry.iloc[7] == True

        # 08:00 應觸發多單出場（結算時間）
        assert long_exit.iloc[8] == True

    def test_generate_signals_with_funding_negative_rate(
        self, strategy, sample_data, sample_funding_rates
    ):
        """測試負資金費率時的訊號生成（應做空）"""
        long_entry, long_exit, short_entry, short_exit = \
            strategy.generate_signals_with_funding(
                sample_data,
                sample_funding_rates
            )

        # 15:00 應觸發空單進場（費率 -0.0002 < -0.0001）
        assert short_entry.iloc[15] == True

        # 16:00 應觸發空單出場（結算時間）
        assert short_exit.iloc[16] == True

    def test_is_settlement_hour(self, strategy):
        """測試結算時間判斷"""
        # 測試結算時間（0, 8, 16）
        assert strategy.is_settlement_hour(
            pd.Timestamp('2024-01-01 00:00:00', tz='UTC'),
            hours_before=0
        ) is True

        assert strategy.is_settlement_hour(
            pd.Timestamp('2024-01-01 08:00:00', tz='UTC'),
            hours_before=0
        ) is True

        assert strategy.is_settlement_hour(
            pd.Timestamp('2024-01-01 16:00:00', tz='UTC'),
            hours_before=0
        ) is True

        # 測試結算前 1 小時（7, 15, 23）
        assert strategy.is_settlement_hour(
            pd.Timestamp('2024-01-01 07:00:00', tz='UTC'),
            hours_before=1
        ) is True

        assert strategy.is_settlement_hour(
            pd.Timestamp('2024-01-01 15:00:00', tz='UTC'),
            hours_before=1
        ) is True

        # 測試非結算時間
        assert strategy.is_settlement_hour(
            pd.Timestamp('2024-01-01 10:00:00', tz='UTC'),
            hours_before=1
        ) is False

    def test_data_funding_length_mismatch(
        self, strategy, sample_data
    ):
        """測試數據與費率長度不匹配（應拋出錯誤）"""
        wrong_rates = pd.Series([0.0001] * 10)  # 長度不匹配

        with pytest.raises(ValueError, match="length mismatch"):
            strategy.generate_signals_with_funding(
                sample_data,
                wrong_rates
            )

    def test_strategy_registration(self):
        """測試策略是否正確註冊"""
        assert StrategyRegistry.exists('funding_rate_settlement')

        strategy_class = StrategyRegistry.get('funding_rate_settlement')
        assert strategy_class == SettlementTradeStrategy

        # 驗證可以透過註冊表建立實例
        strategy = StrategyRegistry.create(
            'funding_rate_settlement',
            rate_threshold=0.0002
        )
        assert isinstance(strategy, SettlementTradeStrategy)
        assert strategy.params['rate_threshold'] == 0.0002

    def test_repr_and_str(self, strategy):
        """測試字串表示"""
        repr_str = repr(strategy)
        assert 'SettlementTradeStrategy' in repr_str
        assert '0.00010' in repr_str
        assert 'hours_before=1' in repr_str

        str_str = str(strategy)
        assert 'settlement_trade' in str_str
        assert '1.0' in str_str

    def test_extreme_rate_values(self, strategy, sample_data):
        """測試極端費率值"""
        # 建立極端費率數據
        extreme_rates = pd.Series([0.0] * len(sample_data), index=sample_data.index)
        extreme_rates.iloc[7] = 0.01  # 1% 極高費率
        extreme_rates.iloc[15] = -0.01  # -1% 極低費率

        long_entry, long_exit, short_entry, short_exit = \
            strategy.generate_signals_with_funding(
                sample_data,
                extreme_rates
            )

        # 即使是極端值也應該正常工作
        assert long_entry.iloc[7] == True
        assert short_entry.iloc[15] == True

    def test_nan_funding_rate_handling(self, strategy, sample_data):
        """測試 NaN 費率處理"""
        # 建立包含 NaN 的費率數據
        nan_rates = pd.Series([0.0001] * len(sample_data), index=sample_data.index)
        nan_rates.iloc[7] = np.nan  # NaN 值
        nan_rates.iloc[15] = 0.0002

        long_entry, long_exit, short_entry, short_exit = \
            strategy.generate_signals_with_funding(
                sample_data,
                nan_rates
            )

        # NaN 位置應該被跳過（不觸發訊號）
        assert long_entry.iloc[7] == False
        assert short_entry.iloc[7] == False

    def test_settlement_exit_timing(self, strategy, sample_data):
        """測試結算時的出場時機"""
        rates = pd.Series([0.0] * len(sample_data), index=sample_data.index)

        # 在所有結算前 1 小時設定高費率
        rates.iloc[7] = 0.0002   # 07:00
        rates.iloc[15] = 0.0002  # 15:00
        rates.iloc[23] = 0.0002  # 23:00

        long_entry, long_exit, short_entry, short_exit = \
            strategy.generate_signals_with_funding(
                sample_data,
                rates
            )

        # 驗證出場時機在結算時（0, 8, 16）
        settlement_hours = [0, 8, 16]
        for hour in settlement_hours:
            assert long_exit.iloc[hour] == True
            assert short_exit.iloc[hour] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
