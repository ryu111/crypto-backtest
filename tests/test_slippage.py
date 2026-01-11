"""
滑點模組測試
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.backtester.slippage import (
    SlippageCalculator,
    SlippageConfig,
    SlippageModel,
    OrderType,
    create_fixed_slippage,
    create_dynamic_slippage,
    create_market_impact_slippage
)


@pytest.fixture
def sample_data():
    """產生測試用 OHLCV 資料"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')

    data = pd.DataFrame({
        'open': 50000 + np.random.randn(100) * 500,
        'high': 50000 + np.random.randn(100) * 500 + 100,
        'low': 50000 + np.random.randn(100) * 500 - 100,
        'close': 50000 + np.random.randn(100) * 500,
        'volume': 100 + np.random.randn(100) * 10
    }, index=dates)

    # 確保 high/low 正確
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    return data


@pytest.fixture
def calculator():
    """預設滑點計算器"""
    return SlippageCalculator()


class TestSlippageConfig:
    """測試滑點配置"""

    def test_default_config(self):
        """測試預設配置"""
        config = SlippageConfig()

        assert config.model == SlippageModel.DYNAMIC
        assert config.base_slippage == 0.0005
        assert config.max_slippage == 0.01

    def test_custom_config(self):
        """測試自定義配置"""
        config = SlippageConfig(
            model=SlippageModel.FIXED,
            base_slippage=0.001,
            max_slippage=0.02
        )

        assert config.model == SlippageModel.FIXED
        assert config.base_slippage == 0.001
        assert config.max_slippage == 0.02

    def test_validation_negative_slippage(self):
        """測試負數滑點驗證"""
        with pytest.raises(ValueError, match="基礎滑點不能為負數"):
            SlippageConfig(base_slippage=-0.001)

    def test_validation_max_less_than_min(self):
        """測試最大滑點小於最小滑點"""
        with pytest.raises(ValueError, match="最大滑點不能小於最小滑點"):
            SlippageConfig(max_slippage=0.001, min_slippage=0.002)

    def test_validation_max_too_high(self):
        """測試最大滑點過高警告"""
        with pytest.raises(ValueError, match="最大滑點不應超過 10%"):
            SlippageConfig(max_slippage=0.15)


class TestSlippageCalculatorFixed:
    """測試固定滑點模型"""

    def test_fixed_slippage(self, sample_data):
        """測試固定滑點"""
        config = SlippageConfig(
            model=SlippageModel.FIXED,
            base_slippage=0.001
        )
        calculator = SlippageCalculator(config)

        slippage = calculator.calculate(
            data=sample_data,
            order_size=10000,
            order_type=OrderType.MARKET
        )

        assert slippage == 0.001

    def test_limit_order_no_slippage(self, sample_data):
        """測試限價單無滑點"""
        config = SlippageConfig(model=SlippageModel.FIXED)
        calculator = SlippageCalculator(config)

        slippage = calculator.calculate(
            data=sample_data,
            order_size=10000,
            order_type=OrderType.LIMIT
        )

        assert slippage == 0.0

    def test_stop_order_multiplier(self, sample_data):
        """測試止損單滑點倍數"""
        config = SlippageConfig(
            model=SlippageModel.FIXED,
            base_slippage=0.001,
            stop_order_multiplier=2.0
        )
        calculator = SlippageCalculator(config)

        slippage = calculator.calculate(
            data=sample_data,
            order_size=10000,
            order_type=OrderType.STOP
        )

        assert slippage == 0.002  # 0.001 * 2.0

    def test_convenience_function(self, sample_data):
        """測試便捷函數"""
        calculator = create_fixed_slippage(0.002)

        slippage = calculator.calculate(
            data=sample_data,
            order_size=10000,
            order_type=OrderType.MARKET
        )

        assert slippage == 0.002


class TestSlippageCalculatorDynamic:
    """測試動態滑點模型"""

    def test_dynamic_slippage_basic(self, sample_data):
        """測試基本動態滑點"""
        config = SlippageConfig(
            model=SlippageModel.DYNAMIC,
            base_slippage=0.001
        )
        calculator = SlippageCalculator(config)

        slippage = calculator.calculate(
            data=sample_data,
            order_size=10000,
            order_type=OrderType.MARKET,
            index=50
        )

        # 應該是正數且受限於 max_slippage
        assert 0 <= slippage <= config.max_slippage

    def test_dynamic_slippage_high_volatility(self):
        """測試高波動率情況"""
        # 建立高波動資料
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        np.random.seed(42)

        # 高波動率資料（10% 標準差）
        high_vol_data = pd.DataFrame({
            'close': 50000 * (1 + np.random.randn(100) * 0.1),
            'volume': 100 + np.random.randn(100) * 10
        }, index=dates)

        high_vol_data['open'] = high_vol_data['close']
        high_vol_data['high'] = high_vol_data['close']
        high_vol_data['low'] = high_vol_data['close']

        config = SlippageConfig(
            model=SlippageModel.DYNAMIC,
            base_slippage=0.001,
            volatility_factor=2.0
        )
        calculator = SlippageCalculator(config)

        slippage = calculator.calculate(
            data=high_vol_data,
            order_size=10000,
            index=50
        )

        # 高波動率應該產生較高滑點
        assert slippage >= config.base_slippage

    def test_dynamic_slippage_max_limit(self, sample_data):
        """測試最大滑點限制"""
        config = SlippageConfig(
            model=SlippageModel.DYNAMIC,
            base_slippage=0.001,
            volatility_factor=100.0,  # 極高係數
            max_slippage=0.005
        )
        calculator = SlippageCalculator(config)

        slippage = calculator.calculate(
            data=sample_data,
            order_size=10000,
            index=50
        )

        # 應該受限於 max_slippage
        assert slippage <= config.max_slippage

    def test_convenience_function_dynamic(self, sample_data):
        """測試動態滑點便捷函數"""
        calculator = create_dynamic_slippage(
            base_slippage=0.001,
            volatility_factor=1.5
        )

        slippage = calculator.calculate(
            data=sample_data,
            order_size=10000,
            index=50
        )

        assert slippage >= 0


class TestSlippageCalculatorMarketImpact:
    """測試市場衝擊模型"""

    def test_market_impact_small_order(self, sample_data):
        """測試小單市場衝擊"""
        config = SlippageConfig(
            model=SlippageModel.MARKET_IMPACT,
            base_slippage=0.001
        )
        calculator = SlippageCalculator(config)

        slippage = calculator.calculate(
            data=sample_data,
            order_size=1000,  # 小單
            index=50
        )

        # 小單衝擊應該接近基礎滑點
        assert slippage >= config.base_slippage

    def test_market_impact_large_order(self, sample_data):
        """測試大單市場衝擊"""
        config = SlippageConfig(
            model=SlippageModel.MARKET_IMPACT,
            base_slippage=0.001,
            market_impact_coeff=0.5
        )
        calculator = SlippageCalculator(config)

        # 大單
        large_slippage = calculator.calculate(
            data=sample_data,
            order_size=100000,
            index=50
        )

        # 小單
        small_slippage = calculator.calculate(
            data=sample_data,
            order_size=1000,
            index=50
        )

        # 大單滑點應該明顯高於小單
        assert large_slippage > small_slippage

    def test_convenience_function_market_impact(self, sample_data):
        """測試市場衝擊便捷函數"""
        calculator = create_market_impact_slippage(
            base_slippage=0.001,
            market_impact_coeff=0.2
        )

        slippage = calculator.calculate(
            data=sample_data,
            order_size=50000,
            index=50
        )

        assert slippage >= 0


class TestSlippageCalculatorCustom:
    """測試自定義滑點函數"""

    def test_custom_function(self, sample_data):
        """測試自定義滑點函數"""
        def custom_slippage(data, order_size, index):
            # 簡單：訂單越大滑點越高
            return order_size / 10000000  # 除以更大的數，避免超過 max_slippage

        calculator = SlippageCalculator()
        calculator.set_custom_function(custom_slippage)

        slippage_10k = calculator.calculate(
            data=sample_data,
            order_size=10000,
            index=50
        )

        slippage_50k = calculator.calculate(
            data=sample_data,
            order_size=50000,
            index=50
        )

        # 50k 訂單應該有更高滑點
        assert slippage_50k > slippage_10k
        assert slippage_10k == pytest.approx(0.001)  # 10000/10000000
        assert slippage_50k == pytest.approx(0.005)  # 50000/10000000

    def test_custom_function_not_set_error(self, sample_data):
        """測試未設定自定義函數錯誤"""
        config = SlippageConfig(model=SlippageModel.CUSTOM)
        calculator = SlippageCalculator(config)

        with pytest.raises(ValueError, match="自定義模型需要先設定計算函數"):
            calculator.calculate(
                data=sample_data,
                order_size=10000
            )


class TestSlippageCalculatorVectorized:
    """測試向量化計算"""

    def test_vectorized_calculation(self, sample_data):
        """測試向量化計算"""
        calculator = create_fixed_slippage(0.001)

        order_sizes = pd.Series(10000, index=sample_data.index)
        slippages = calculator.calculate_vectorized(sample_data, order_sizes)

        assert len(slippages) == len(sample_data)
        assert all(slippages == 0.001)  # 固定滑點

    def test_vectorized_with_different_sizes(self, sample_data):
        """測試不同訂單大小的向量化計算"""
        calculator = create_market_impact_slippage()

        # 隨機訂單大小
        np.random.seed(42)
        order_sizes = pd.Series(
            np.random.uniform(1000, 100000, len(sample_data)),
            index=sample_data.index
        )

        slippages = calculator.calculate_vectorized(sample_data, order_sizes)

        assert len(slippages) == len(sample_data)
        # 忽略 NaN 值（窗口不足時）
        valid_slippages = slippages.dropna()
        assert all(valid_slippages >= 0)

    def test_vectorized_with_limit_orders(self, sample_data):
        """測試限價單向量化計算"""
        calculator = create_fixed_slippage(0.001)

        order_sizes = pd.Series(10000, index=sample_data.index)
        order_types = pd.Series(OrderType.LIMIT, index=sample_data.index)

        slippages = calculator.calculate_vectorized(
            sample_data, order_sizes, order_types
        )

        # 全部應該是 0（限價單無滑點）
        assert all(slippages == 0.0)


class TestExecutionPrice:
    """測試執行價格估算"""

    def test_execution_price_long(self):
        """測試做多執行價格"""
        calculator = create_fixed_slippage(0.001)

        execution_price = calculator.estimate_execution_price(
            current_price=50000,
            slippage=0.001,
            direction=1  # 做多
        )

        # 做多時價格上滑
        assert execution_price == 50000 * 1.001

    def test_execution_price_short(self):
        """測試做空執行價格"""
        calculator = create_fixed_slippage(0.001)

        execution_price = calculator.estimate_execution_price(
            current_price=50000,
            slippage=0.001,
            direction=-1  # 做空
        )

        # 做空時價格下滑
        assert execution_price == 50000 * 0.999


class TestSlippageCurve:
    """測試滑點曲線分析"""

    def test_slippage_curve_generation(self, sample_data):
        """測試滑點曲線產生"""
        calculator = create_market_impact_slippage()

        curve = calculator.get_slippage_curve(sample_data)

        assert isinstance(curve, pd.DataFrame)
        assert len(curve) == len(sample_data)
        assert 'size_1000' in curve.columns
        assert 'size_100000' in curve.columns

    def test_slippage_curve_custom_sizes(self, sample_data):
        """測試自定義大小的滑點曲線"""
        calculator = create_dynamic_slippage()

        curve = calculator.get_slippage_curve(
            sample_data,
            order_sizes=[5000, 10000, 20000]
        )

        assert 'size_5000' in curve.columns
        assert 'size_10000' in curve.columns
        assert 'size_20000' in curve.columns


class TestImpactAnalysis:
    """測試滑點影響分析"""

    def test_impact_analysis(self, sample_data):
        """測試滑點影響分析"""
        calculator = create_fixed_slippage(0.001)

        # 建立測試交易記錄
        trades = pd.DataFrame({
            'entry_time': sample_data.index[:10],
            'size': [10000] * 10
        })

        analysis = calculator.analyze_impact(sample_data, trades)

        assert 'total_cost' in analysis
        assert 'avg_slippage' in analysis
        assert 'max_slippage' in analysis
        # 使用 pytest.approx 處理浮點數精度問題
        assert analysis['avg_slippage'] == pytest.approx(0.001)

    def test_impact_analysis_empty_trades(self, sample_data):
        """測試空交易記錄"""
        calculator = create_fixed_slippage()

        trades = pd.DataFrame({
            'entry_time': [],
            'size': []
        })

        analysis = calculator.analyze_impact(sample_data, trades)

        assert analysis['total_cost'] == 0.0
        assert analysis['avg_slippage'] == 0.0


class TestEdgeCases:
    """測試邊界情況"""

    def test_zero_volatility(self):
        """測試零波動率情況"""
        # 建立零波動資料
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        flat_data = pd.DataFrame({
            'open': 50000,
            'high': 50000,
            'low': 50000,
            'close': 50000,
            'volume': 100
        }, index=dates)

        calculator = create_dynamic_slippage()

        slippage = calculator.calculate(
            data=flat_data,
            order_size=10000,
            index=50
        )

        # 零波動應該回退到基礎滑點
        assert slippage >= 0

    def test_zero_volume(self):
        """測試零成交量情況"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        zero_vol_data = pd.DataFrame({
            'open': 50000,
            'high': 50000,
            'low': 50000,
            'close': 50000,
            'volume': 0
        }, index=dates)

        calculator = create_market_impact_slippage()

        slippage = calculator.calculate(
            data=zero_vol_data,
            order_size=10000,
            index=50
        )

        # 應該回退到基礎滑點
        assert slippage >= 0

    def test_first_index(self, sample_data):
        """測試資料起始點計算"""
        calculator = create_dynamic_slippage()

        # 在起始點計算（窗口不足）
        slippage = calculator.calculate(
            data=sample_data,
            order_size=10000,
            index=0
        )

        # 窗口不足時應該回退到基礎滑點
        assert slippage >= 0 or pd.isna(slippage) == False

    def test_last_index_default(self, sample_data):
        """測試預設使用最後一筆資料"""
        calculator = create_fixed_slippage(0.001)

        slippage = calculator.calculate(
            data=sample_data,
            order_size=10000
            # index=None（預設）
        )

        assert slippage == 0.001


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
