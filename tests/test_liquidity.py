"""
流動性模組測試

測試覆蓋：
1. 基本功能測試
2. 模型計算測試（線性、平方根、對數）
3. 向量化計算測試
4. 邊界條件測試
5. 錯誤處理測試
"""

import pytest
import pandas as pd
import numpy as np
from src.backtester.liquidity import (
    LiquidityCalculator,
    LiquidityConfig,
    LiquidityModel,
    LiquidityLevel,
    create_linear_liquidity,
    create_square_root_liquidity,
    create_logarithmic_liquidity
)


@pytest.fixture
def sample_data():
    """建立測試資料"""
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')

    # 模擬 BTC 價格資料
    np.random.seed(42)
    close_prices = 50000 + np.cumsum(np.random.randn(100) * 100)
    volumes = np.random.uniform(100, 1000, 100)  # BTC

    data = pd.DataFrame({
        'close': close_prices,
        'volume': volumes,
        'high': close_prices * 1.01,
        'low': close_prices * 0.99,
        'open': close_prices
    }, index=dates)

    return data


@pytest.fixture
def calculator():
    """建立預設計算器"""
    config = LiquidityConfig(
        model=LiquidityModel.SQUARE_ROOT,
        impact_coefficient=0.3,
        adv_window=30
    )
    return LiquidityCalculator(config)


class TestLiquidityConfig:
    """測試配置類別"""

    def test_default_config(self):
        """測試預設配置"""
        config = LiquidityConfig()

        assert config.model == LiquidityModel.SQUARE_ROOT
        assert config.impact_coefficient == 0.3
        assert config.adv_window == 30
        assert config.use_volatility is True

    def test_config_validation(self):
        """測試配置驗證"""
        # 負數衝擊係數
        with pytest.raises(ValueError, match="衝擊係數不能為負數"):
            LiquidityConfig(impact_coefficient=-0.1)

        # 最大衝擊小於最小衝擊
        with pytest.raises(ValueError, match="最大衝擊不能小於最小衝擊"):
            LiquidityConfig(max_impact=0.01, min_impact=0.05)

        # 最大衝擊過大
        with pytest.raises(ValueError, match="最大價格衝擊不應超過"):
            LiquidityConfig(max_impact=0.15)

        # ADV 窗口無效
        with pytest.raises(ValueError, match="ADV 窗口不能小於 1"):
            LiquidityConfig(adv_window=0)

    def test_custom_config(self):
        """測試自訂配置"""
        config = LiquidityConfig(
            model=LiquidityModel.LINEAR,
            impact_coefficient=0.5,
            adv_window=60,
            max_impact=0.03
        )

        assert config.model == LiquidityModel.LINEAR
        assert config.impact_coefficient == 0.5
        assert config.adv_window == 60
        assert config.max_impact == 0.03


class TestLiquidityCalculator:
    """測試流動性計算器"""

    def test_initialization(self):
        """測試初始化"""
        # 預設配置
        calc1 = LiquidityCalculator()
        assert calc1.config.model == LiquidityModel.SQUARE_ROOT

        # 自訂配置
        config = LiquidityConfig(model=LiquidityModel.LINEAR)
        calc2 = LiquidityCalculator(config)
        assert calc2.config.model == LiquidityModel.LINEAR

    def test_calculate_impact_basic(self, calculator, sample_data):
        """測試基本價格衝擊計算"""
        impact = calculator.calculate_impact(
            data=sample_data,
            order_size_usd=10000,
            index=50
        )

        assert isinstance(impact, float)
        assert 0 <= impact <= calculator.config.max_impact
        assert impact > 0  # 應該有正衝擊

    def test_linear_model(self, sample_data):
        """測試線性模型"""
        calc = create_linear_liquidity(impact_coefficient=0.2)

        impact_10k = calc.calculate_impact(sample_data, 10000, index=50)
        impact_20k = calc.calculate_impact(sample_data, 20000, index=50)

        # 線性模型：2倍訂單 = 2倍衝擊（在未達上限時）
        if impact_20k < calc.config.max_impact:
            assert abs(impact_20k / impact_10k - 2.0) < 0.1  # 允許小誤差

    def test_square_root_model(self, sample_data):
        """測試平方根模型"""
        calc = create_square_root_liquidity(impact_coefficient=0.3)

        impact_10k = calc.calculate_impact(sample_data, 10000, index=50)
        impact_40k = calc.calculate_impact(sample_data, 40000, index=50)

        # 平方根模型：4倍訂單 = 2倍衝擊（√4 = 2）
        if impact_40k < calc.config.max_impact:
            ratio = impact_40k / impact_10k
            assert 1.8 < ratio < 2.2  # 應該接近 2

    def test_logarithmic_model(self, sample_data):
        """測試對數模型"""
        calc = create_logarithmic_liquidity(impact_coefficient=0.4)

        impact_small = calc.calculate_impact(sample_data, 1000, index=50)
        impact_large = calc.calculate_impact(sample_data, 100000, index=50)

        # 對數模型：大單的邊際衝擊遞減
        assert impact_large > impact_small
        assert impact_large / impact_small < 100  # 衝擊增長遠小於訂單增長

    def test_direction_effect(self, calculator, sample_data):
        """測試交易方向（目前模型中方向不影響衝擊大小，只影響價格方向）"""
        impact_long = calculator.calculate_impact(
            sample_data, 10000, direction=1, index=50
        )
        impact_short = calculator.calculate_impact(
            sample_data, 10000, direction=-1, index=50
        )

        # 衝擊大小應該相同（方向只影響執行價格）
        assert impact_long == impact_short

    def test_impact_limits(self, sample_data):
        """測試衝擊限制"""
        config = LiquidityConfig(
            model=LiquidityModel.SQUARE_ROOT,
            impact_coefficient=0.3,
            max_impact=0.02,
            min_impact=0.001
        )
        calc = LiquidityCalculator(config)

        # 極小訂單
        impact_tiny = calc.calculate_impact(sample_data, 1, index=50)
        assert impact_tiny >= config.min_impact

        # 極大訂單
        impact_huge = calc.calculate_impact(sample_data, 1_000_000_000, index=50)
        assert impact_huge <= config.max_impact

    def test_custom_function(self, sample_data):
        """測試自定義函數"""
        def custom_impact(order_size, adv, volatility):
            # 簡單的線性模型
            return 0.001 * (order_size / adv)

        calc = LiquidityCalculator()
        calc.set_custom_function(custom_impact)

        impact = calc.calculate_impact(sample_data, 10000, index=50)
        assert isinstance(impact, float)
        assert impact > 0

    def test_custom_function_error(self, sample_data):
        """測試未設定自定義函數的錯誤"""
        config = LiquidityConfig(model=LiquidityModel.CUSTOM)
        calc = LiquidityCalculator(config)

        with pytest.raises(ValueError, match="自定義模型需要先設定計算函數"):
            calc.calculate_impact(sample_data, 10000, index=50)


class TestExecutionPrice:
    """測試執行價格估算"""

    def test_estimate_execution_price_long(self, calculator):
        """測試做多執行價格"""
        current_price = 50000.0
        impact = 0.002  # 0.2%

        exec_price = calculator.estimate_execution_price(
            current_price, impact, direction=1
        )

        expected = 50000 * 1.002
        assert abs(exec_price - expected) < 0.01

    def test_estimate_execution_price_short(self, calculator):
        """測試做空執行價格"""
        current_price = 50000.0
        impact = 0.002  # 0.2%

        exec_price = calculator.estimate_execution_price(
            current_price, impact, direction=-1
        )

        expected = 50000 * 0.998
        assert abs(exec_price - expected) < 0.01


class TestMaxOrderSize:
    """測試最大訂單大小計算"""

    def test_calculate_max_order_size_square_root(self, sample_data):
        """測試平方根模型的最大訂單"""
        calc = create_square_root_liquidity(impact_coefficient=0.3)

        max_size = calc.calculate_max_order_size(
            sample_data,
            price_tolerance=0.01,  # 1% 容忍度
            index=50
        )

        assert max_size > 0
        assert isinstance(max_size, float)

        # 驗證：計算出的訂單大小應該產生接近容忍度的衝擊
        impact = calc.calculate_impact(sample_data, max_size, index=50)
        assert abs(impact - 0.01) < 0.002  # 允許小誤差

    def test_calculate_max_order_size_linear(self, sample_data):
        """測試線性模型的最大訂單"""
        calc = create_linear_liquidity(impact_coefficient=0.2)

        max_size = calc.calculate_max_order_size(
            sample_data,
            price_tolerance=0.005,
            index=50
        )

        assert max_size > 0

        # 驗證
        impact = calc.calculate_impact(sample_data, max_size, index=50)
        assert abs(impact - 0.005) < 0.001

    def test_max_order_size_different_tolerances(self, calculator, sample_data):
        """測試不同容忍度的最大訂單"""
        max_1pct = calculator.calculate_max_order_size(
            sample_data, price_tolerance=0.01, index=50
        )
        max_2pct = calculator.calculate_max_order_size(
            sample_data, price_tolerance=0.02, index=50
        )

        # 容忍度越高，可下單金額越大
        assert max_2pct > max_1pct


class TestLiquidityScore:
    """測試流動性評級"""

    def test_get_liquidity_score(self, calculator, sample_data):
        """測試流動性評級"""
        # 極小訂單 -> HIGH
        score_tiny = calculator.get_liquidity_score(sample_data, 100, index=50)
        assert score_tiny == LiquidityLevel.HIGH

        # 極大訂單 -> LOW 或 VERY_LOW
        score_huge = calculator.get_liquidity_score(sample_data, 10_000_000, index=50)
        assert score_huge in [LiquidityLevel.LOW, LiquidityLevel.VERY_LOW]

    def test_liquidity_levels(self, sample_data):
        """測試不同流動性等級"""
        config = LiquidityConfig(
            high_liquidity_threshold=0.001,
            medium_liquidity_threshold=0.01,
            low_liquidity_threshold=0.05
        )
        calc = LiquidityCalculator(config)

        # 根據 ADV 計算不同大小的訂單等級
        adv = calc._calculate_adv(sample_data, 50)

        # HIGH: < 0.1% ADV
        score_high = calc.get_liquidity_score(
            sample_data, adv * 0.0005, index=50
        )
        assert score_high == LiquidityLevel.HIGH

        # MEDIUM: 0.1% - 1% ADV
        score_medium = calc.get_liquidity_score(
            sample_data, adv * 0.005, index=50
        )
        assert score_medium == LiquidityLevel.MEDIUM

        # LOW: 1% - 5% ADV
        score_low = calc.get_liquidity_score(
            sample_data, adv * 0.03, index=50
        )
        assert score_low == LiquidityLevel.LOW

        # VERY_LOW: > 5% ADV
        score_very_low = calc.get_liquidity_score(
            sample_data, adv * 0.1, index=50
        )
        assert score_very_low == LiquidityLevel.VERY_LOW


class TestVectorized:
    """測試向量化計算"""

    def test_calculate_vectorized(self, calculator, sample_data):
        """測試向量化計算"""
        order_sizes = pd.Series(10000, index=sample_data.index)
        impacts = calculator.calculate_vectorized(sample_data, order_sizes)

        assert len(impacts) == len(sample_data)
        assert all(impacts >= 0)
        assert all(impacts <= calculator.config.max_impact)

    def test_vectorized_variable_sizes(self, calculator, sample_data):
        """測試不同訂單大小的向量化計算"""
        # 建立隨機訂單大小
        np.random.seed(42)
        order_sizes = pd.Series(
            np.random.uniform(1000, 100000, len(sample_data)),
            index=sample_data.index
        )

        impacts = calculator.calculate_vectorized(sample_data, order_sizes)

        assert len(impacts) == len(sample_data)
        assert all(impacts > 0)

    def test_vectorized_with_directions(self, calculator, sample_data):
        """測試帶方向的向量化計算"""
        order_sizes = pd.Series(10000, index=sample_data.index)
        directions = pd.Series(
            [1 if i % 2 == 0 else -1 for i in range(len(sample_data))],
            index=sample_data.index
        )

        impacts = calculator.calculate_vectorized(
            sample_data, order_sizes, directions
        )

        assert len(impacts) == len(sample_data)
        assert all(impacts >= 0)


class TestAnalysis:
    """測試分析功能"""

    def test_analyze_liquidity(self, calculator, sample_data):
        """測試流動性分析"""
        analysis = calculator.analyze_liquidity(
            sample_data,
            order_sizes=[1000, 10000, 100000]
        )

        assert isinstance(analysis, pd.DataFrame)
        assert len(analysis) == len(sample_data)
        assert 'size_1000' in analysis.columns
        assert 'size_10000' in analysis.columns
        assert 'size_100000' in analysis.columns

        # 大訂單應該有更大的衝擊
        assert (analysis['size_100000'] >= analysis['size_10000']).all()
        assert (analysis['size_10000'] >= analysis['size_1000']).all()


class TestEdgeCases:
    """測試邊界條件"""

    def test_zero_volume(self, calculator):
        """測試零成交量"""
        data = pd.DataFrame({
            'close': [50000] * 10,
            'volume': [0] * 10,
            'high': [50000] * 10,
            'low': [50000] * 10,
            'open': [50000] * 10
        })

        impact = calculator.calculate_impact(data, 10000, index=5)

        # 無成交量應該返回最大衝擊
        assert impact == calculator.config.max_impact

    def test_single_data_point(self, calculator):
        """測試單一資料點"""
        data = pd.DataFrame({
            'close': [50000],
            'volume': [100],
            'high': [50000],
            'low': [50000],
            'open': [50000]
        })

        impact = calculator.calculate_impact(data, 10000, index=0)
        assert isinstance(impact, float)
        assert impact >= 0

    def test_very_small_order(self, calculator, sample_data):
        """測試極小訂單"""
        impact = calculator.calculate_impact(sample_data, 0.01, index=50)

        assert impact >= calculator.config.min_impact
        assert impact < 0.01  # 應該很小

    def test_nan_handling(self, calculator):
        """測試 NaN 處理"""
        data = pd.DataFrame({
            'close': [50000, np.nan, 50100],
            'volume': [100, 200, 150],
            'high': [50000, np.nan, 50100],
            'low': [50000, np.nan, 50100],
            'open': [50000, np.nan, 50100]
        })

        # 應該不會拋出錯誤
        impact = calculator.calculate_impact(data, 10000, index=2)
        assert isinstance(impact, float)
        assert not np.isnan(impact)


class TestConvenienceFunctions:
    """測試便捷函數"""

    def test_create_linear_liquidity(self):
        """測試線性流動性建立函數"""
        calc = create_linear_liquidity(impact_coefficient=0.2, adv_window=60)

        assert calc.config.model == LiquidityModel.LINEAR
        assert calc.config.impact_coefficient == 0.2
        assert calc.config.adv_window == 60

    def test_create_square_root_liquidity(self):
        """測試平方根流動性建立函數"""
        calc = create_square_root_liquidity(
            impact_coefficient=0.3,
            adv_window=30,
            use_volatility=False
        )

        assert calc.config.model == LiquidityModel.SQUARE_ROOT
        assert calc.config.impact_coefficient == 0.3
        assert calc.config.use_volatility is False

    def test_create_logarithmic_liquidity(self):
        """測試對數流動性建立函數"""
        calc = create_logarithmic_liquidity(
            impact_coefficient=0.4,
            adv_window=45
        )

        assert calc.config.model == LiquidityModel.LOGARITHMIC
        assert calc.config.impact_coefficient == 0.4
        assert calc.config.adv_window == 45


class TestIntegration:
    """整合測試"""

    def test_realistic_scenario(self, sample_data):
        """測試真實場景"""
        # 建立計算器（平方根模型）
        calc = create_square_root_liquidity(impact_coefficient=0.3)

        # 計算不同大小訂單的衝擊
        order_size = 50000  # $50k USDT

        # 1. 計算價格衝擊
        impact = calc.calculate_impact(sample_data, order_size, index=50)

        # 2. 估算執行價格
        current_price = sample_data['close'].iloc[50]
        exec_price_long = calc.estimate_execution_price(current_price, impact, 1)
        exec_price_short = calc.estimate_execution_price(current_price, impact, -1)

        # 3. 評估流動性等級
        liquidity_level = calc.get_liquidity_score(sample_data, order_size, index=50)

        # 4. 計算最大訂單（1% 容忍度）
        max_order = calc.calculate_max_order_size(sample_data, 0.01, index=50)

        # 驗證結果合理性
        assert 0 < impact < 0.05
        assert exec_price_long > current_price
        assert exec_price_short < current_price
        assert liquidity_level in LiquidityLevel
        # 如果最大訂單小於當前訂單，流動性應該不高
        # 或者如果流動性高，最大訂單應該大於當前訂單
        if max_order < order_size:
            assert liquidity_level not in [LiquidityLevel.HIGH]

    def test_model_comparison(self, sample_data):
        """測試不同模型比較"""
        order_size = 50000

        linear = create_linear_liquidity(0.2)
        sqrt = create_square_root_liquidity(0.3)
        log = create_logarithmic_liquidity(0.4)

        impact_linear = linear.calculate_impact(sample_data, order_size, index=50)
        impact_sqrt = sqrt.calculate_impact(sample_data, order_size, index=50)
        impact_log = log.calculate_impact(sample_data, order_size, index=50)

        # 所有模型都應該返回正值
        assert impact_linear > 0
        assert impact_sqrt > 0
        assert impact_log > 0

        # 對於相同參數，線性模型通常給出最保守的估計（最大衝擊）
        # 但實際結果取決於係數設定
        assert all(isinstance(x, float) for x in [impact_linear, impact_sqrt, impact_log])
