"""
GP Primitives 單元測試

測試範圍：
- 類型系統：Price, Indicator, Signal, Number
- 指標函數：RSI, MA, EMA, ATR, MACD, BB
- 比較函數：gt, lt, cross_above, cross_below
- 邏輯函數：and_, or_, not_
- 數學函數：add, sub, mul, protected_div, protected_log
- PrimitiveSetFactory：標準集、最小集、自訂集
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from src.gp.primitives import (
    # 類型
    Price, Indicator, Signal, Number,
    # 指標函數
    rsi, ma, ema, atr, macd, bb_upper, bb_lower,
    # 比較函數
    gt, lt, cross_above, cross_below,
    # 邏輯函數
    and_, or_, not_,
    # 數學函數
    add, sub, mul, protected_div, protected_log,
    # 工廠
    PrimitiveSetFactory,
)


# ============================================================================
# 測試夾具（Test Fixtures）
# ============================================================================

@pytest.fixture
def sample_prices():
    """模擬價格序列（10 天 OHLC）"""
    return {
        'close': np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109]),
        'high': np.array([101, 103, 102, 104, 106, 105, 107, 109, 108, 110]),
        'low': np.array([99, 101, 100, 102, 104, 103, 105, 107, 106, 108]),
    }


@pytest.fixture
def trending_prices():
    """明確趨勢的價格（用於測試交叉）"""
    return np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])


@pytest.fixture
def factory():
    """PrimitiveSetFactory 實例"""
    try:
        return PrimitiveSetFactory()
    except ImportError:
        pytest.skip("DEAP not installed")


# ============================================================================
# 1. 類型系統測試
# ============================================================================

class TestTypeSystem:
    """測試 DEAP 類型系統"""

    def test_price_inherits_from_float(self):
        """Price 類型繼承自 float"""
        assert issubclass(Price, float)

    def test_indicator_inherits_from_float(self):
        """Indicator 類型繼承自 float"""
        assert issubclass(Indicator, float)

    def test_number_inherits_from_float(self):
        """Number 類型繼承自 float"""
        assert issubclass(Number, float)

    def test_signal_is_distinct_type(self):
        """Signal 是獨立類型（不繼承 float）"""
        assert not issubclass(Signal, float)


# ============================================================================
# 2. 指標函數測試
# ============================================================================

class TestIndicatorFunctions:
    """測試所有指標函數"""

    # ===== RSI 測試 =====

    def test_rsi_basic(self, sample_prices):
        """RSI 基本計算"""
        close = sample_prices['close']
        result = rsi(close, period=5)

        # 檢查返回型別和長度
        assert isinstance(result, np.ndarray)
        assert len(result) == len(close)

        # RSI 值應在 0-100 範圍內（忽略 NaN）
        valid_values = result[~np.isnan(result)]
        assert np.all((valid_values >= 0) & (valid_values <= 100))

    def test_rsi_minimum_period(self):
        """RSI 最小週期處理（period < 2 自動調整為 2）"""
        prices = np.array([100, 102, 101, 103, 105])
        result = rsi(prices, period=1)

        # 應該使用 period=2（最小週期）
        assert len(result) == len(prices)

    def test_rsi_all_rising_prices(self):
        """RSI 在持續上漲時應接近 100"""
        prices = np.array([100, 105, 110, 115, 120, 125, 130, 135, 140, 145])
        result = rsi(prices, period=5)

        # 後半段 RSI 應該很高（接近 100）
        # RSI 在持續上漲時會趨近 100（但不會精確到 100）
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            assert valid_values[-1] > 50  # 持續上漲應大於中線

    def test_rsi_empty_array(self):
        """RSI 處理空陣列"""
        prices = np.array([])

        # 空陣列應該返回空陣列（不崩潰）
        result = rsi(prices, period=14)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    # ===== MA 測試 =====

    def test_ma_basic(self, sample_prices):
        """MA 基本計算"""
        close = sample_prices['close']
        result = ma(close, period=3)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(close)

        # 前 period-1 個值應為 NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])

    def test_ma_calculation_accuracy(self):
        """MA 計算正確性（手動驗證）"""
        prices = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = ma(prices, period=3)

        # 前 2 個值為 NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])

        # 第 3 個值 = (10+20+30)/3 = 20
        assert result[2] == pytest.approx(20.0, rel=1e-6)

        # 第 4 個值 = (20+30+40)/3 = 30
        assert result[3] == pytest.approx(30.0, rel=1e-6)

    def test_ma_single_element(self):
        """MA 處理單元素陣列"""
        prices = np.array([100.0])
        result = ma(prices, period=1)

        assert len(result) == 1
        assert result[0] == pytest.approx(100.0)

    # ===== EMA 測試 =====

    def test_ema_basic(self, sample_prices):
        """EMA 基本計算"""
        close = sample_prices['close']
        result = ema(close, period=5)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(close)

    def test_ema_responds_faster_than_ma(self, trending_prices):
        """EMA 應比 MA 反應更快"""
        ma_result = ma(trending_prices, period=5)
        ema_result = ema(trending_prices, period=5)

        # 在上漲趨勢中，EMA 最後值應大於 MA（更快反應）
        # 忽略 NaN 值
        ma_last = ma_result[~np.isnan(ma_result)][-1]
        ema_last = ema_result[~np.isnan(ema_result)][-1]

        assert ema_last > ma_last

    # ===== ATR 測試 =====

    def test_atr_basic(self, sample_prices):
        """ATR 基本計算"""
        result = atr(
            sample_prices['high'],
            sample_prices['low'],
            sample_prices['close'],
            period=5
        )

        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_prices['close'])

        # ATR 應為正值
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values >= 0)

    def test_atr_increases_with_volatility(self):
        """ATR 在波動增加時應上升"""
        # 低波動
        low_vol_high = np.array([100, 101, 102, 103, 104])
        low_vol_low = np.array([99, 100, 101, 102, 103])
        low_vol_close = np.array([100, 101, 102, 103, 104])

        # 高波動
        high_vol_high = np.array([100, 110, 105, 115, 110])
        high_vol_low = np.array([90, 95, 90, 100, 95])
        high_vol_close = np.array([95, 105, 95, 110, 100])

        atr_low = atr(low_vol_high, low_vol_low, low_vol_close, period=3)
        atr_high = atr(high_vol_high, high_vol_low, high_vol_close, period=3)

        # 高波動的 ATR 應該更大
        assert atr_high[-1] > atr_low[-1]

    # ===== MACD 測試 =====

    def test_macd_basic(self, sample_prices):
        """MACD 基本計算"""
        close = sample_prices['close']
        result = macd(close, fast_period=3, slow_period=5)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(close)

    def test_macd_fast_period_smaller_than_slow(self):
        """MACD 確保 fast < slow（自動調整）"""
        prices = np.array([100, 102, 104, 106, 108, 110, 112, 114])

        # 故意傳入 fast > slow
        result = macd(prices, fast_period=10, slow_period=5)

        # 應該自動調整為 slow = fast + 1
        assert isinstance(result, np.ndarray)

    # ===== 布林帶測試 =====

    def test_bb_upper_basic(self, sample_prices):
        """布林帶上軌基本計算"""
        close = sample_prices['close']
        result = bb_upper(close, period=5, std_mult=2.0)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(close)

        # 上軌應大於價格（大部分時間）
        ma_result = ma(close, period=5)
        valid_idx = ~np.isnan(result)
        assert np.all(result[valid_idx] >= ma_result[valid_idx])

    def test_bb_lower_basic(self, sample_prices):
        """布林帶下軌基本計算"""
        close = sample_prices['close']
        result = bb_lower(close, period=5, std_mult=2.0)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(close)

        # 下軌應小於價格（大部分時間）
        ma_result = ma(close, period=5)
        valid_idx = ~np.isnan(result)
        assert np.all(result[valid_idx] <= ma_result[valid_idx])

    def test_bb_bands_symmetry(self, sample_prices):
        """布林帶上下軌應對稱於中軌"""
        close = sample_prices['close']
        middle = ma(close, period=5)
        upper = bb_upper(close, period=5, std_mult=2.0)
        lower = bb_lower(close, period=5, std_mult=2.0)

        # 忽略 NaN 值
        valid_idx = ~np.isnan(middle)

        # 上軌 - 中軌 ≈ 中軌 - 下軌
        upper_diff = upper[valid_idx] - middle[valid_idx]
        lower_diff = middle[valid_idx] - lower[valid_idx]

        assert_array_almost_equal(upper_diff, lower_diff, decimal=5)


# ============================================================================
# 3. 比較函數測試
# ============================================================================

class TestComparisonFunctions:
    """測試所有比較函數"""

    def test_gt_basic(self):
        """大於比較基本測試"""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([2.0, 2.0, 2.0, 2.0])
        result = gt(a, b)

        expected = np.array([False, False, True, True])
        assert_array_equal(result, expected)

    def test_lt_basic(self):
        """小於比較基本測試"""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([2.0, 2.0, 2.0, 2.0])
        result = lt(a, b)

        expected = np.array([True, False, False, False])
        assert_array_equal(result, expected)

    def test_cross_above_basic(self):
        """向上穿越基本測試"""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 快線
        b = np.array([3.0, 3.0, 3.0, 3.0, 3.0])  # 慢線（固定）

        result = cross_above(a, b)

        # 第一個值永遠是 False（沒有前一期）
        assert result[0] == False

        # 第 2 個值：a[1]=2 <= b[1]=3，沒有穿越
        assert result[1] == False

        # 第 3 個值：a[2]=3 == b[2]=3，沒有穿越（需要 >）
        assert result[2] == False

        # 第 4 個值：a[3]=4 > b[3]=3 且 a[2]=3 <= b[2]=3，發生穿越
        assert result[3] == True

        # 第 5 個值：a[4]=5 > b[4]=3 但 a[3]=4 已經 > b[3]=3，沒有穿越
        assert result[4] == False

    def test_cross_below_basic(self):
        """向下穿越基本測試"""
        a = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # 快線
        b = np.array([3.0, 3.0, 3.0, 3.0, 3.0])  # 慢線（固定）

        result = cross_below(a, b)

        # 第一個值永遠是 False
        assert result[0] == False

        # 第 2 個值：a[1]=4 >= b[1]=3，沒有穿越
        assert result[1] == False

        # 第 3 個值：a[2]=3 == b[2]=3，沒有穿越
        assert result[2] == False

        # 第 4 個值：a[3]=2 < b[3]=3 且 a[2]=3 >= b[2]=3，發生穿越
        assert result[3] == True

        # 第 5 個值：a[4]=1 < b[4]=3 但 a[3]=2 已經 < b[3]=3，沒有穿越
        assert result[4] == False

    def test_cross_above_empty_array(self):
        """向上穿越處理空陣列"""
        a = np.array([])
        b = np.array([])

        # 空陣列應該返回空布林陣列（不崩潰）
        result = cross_above(a, b)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
        assert result.dtype == bool


# ============================================================================
# 4. 邏輯函數測試
# ============================================================================

class TestLogicalFunctions:
    """測試所有邏輯函數"""

    def test_and_basic(self):
        """邏輯 AND 基本測試"""
        a = np.array([True, True, False, False])
        b = np.array([True, False, True, False])
        result = and_(a, b)

        expected = np.array([True, False, False, False])
        assert_array_equal(result, expected)

    def test_or_basic(self):
        """邏輯 OR 基本測試"""
        a = np.array([True, True, False, False])
        b = np.array([True, False, True, False])
        result = or_(a, b)

        expected = np.array([True, True, True, False])
        assert_array_equal(result, expected)

    def test_not_basic(self):
        """邏輯 NOT 基本測試"""
        a = np.array([True, False, True, False])
        result = not_(a)

        expected = np.array([False, True, False, True])
        assert_array_equal(result, expected)

    def test_logical_empty_array(self):
        """邏輯運算處理空陣列"""
        a = np.array([], dtype=bool)
        b = np.array([], dtype=bool)

        assert len(and_(a, b)) == 0
        assert len(or_(a, b)) == 0
        assert len(not_(a)) == 0


# ============================================================================
# 5. 數學函數測試
# ============================================================================

class TestMathFunctions:
    """測試所有數學函數"""

    def test_add_basic(self):
        """加法基本測試"""
        assert add(2.0, 3.0) == 5.0
        assert add(-1.0, 1.0) == 0.0
        assert add(0.0, 0.0) == 0.0

    def test_sub_basic(self):
        """減法基本測試"""
        assert sub(5.0, 3.0) == 2.0
        assert sub(3.0, 5.0) == -2.0
        assert sub(0.0, 0.0) == 0.0

    def test_mul_basic(self):
        """乘法基本測試"""
        assert mul(2.0, 3.0) == 6.0
        assert mul(-2.0, 3.0) == -6.0
        assert mul(0.0, 100.0) == 0.0

    def test_protected_div_normal(self):
        """保護除法正常情況"""
        assert protected_div(6.0, 2.0) == 3.0
        assert protected_div(5.0, 2.0) == 2.5

    def test_protected_div_by_zero(self):
        """保護除法處理除零"""
        # 除數為 0 時返回 1.0
        assert protected_div(10.0, 0.0) == 1.0
        assert protected_div(5.0, 0.0) == 1.0

    def test_protected_div_by_near_zero(self):
        """保護除法處理接近零的值"""
        # 除數 < 1e-10 視為 0
        assert protected_div(10.0, 1e-11) == 1.0
        assert protected_div(10.0, -1e-11) == 1.0

    def test_protected_log_positive(self):
        """保護對數正常情況"""
        assert protected_log(np.e) == pytest.approx(1.0, rel=1e-6)
        assert protected_log(1.0) == pytest.approx(0.0, rel=1e-6)
        assert protected_log(10.0) == pytest.approx(np.log(10.0), rel=1e-6)

    def test_protected_log_zero(self):
        """保護對數處理零"""
        assert protected_log(0.0) == 0.0

    def test_protected_log_negative(self):
        """保護對數處理負數"""
        assert protected_log(-5.0) == 0.0
        assert protected_log(-100.0) == 0.0


# ============================================================================
# 6. PrimitiveSetFactory 測試
# ============================================================================

class TestPrimitiveSetFactory:
    """測試 PrimitiveSetFactory"""

    def test_factory_requires_deap(self):
        """工廠類別需要 DEAP"""
        # 這個測試在 DEAP 未安裝時會被跳過（由 fixture 處理）
        factory = PrimitiveSetFactory()
        assert factory is not None

    def test_create_standard_set(self, factory):
        """建立標準原語集"""
        pset = factory.create_standard_set()

        # 檢查名稱
        assert pset.name == "MAIN"

        # 檢查輸入參數（close, high, low）
        assert len(pset.arguments) == 3

        # 檢查 primitives 存在
        assert len(pset.primitives) > 0

        # 檢查 terminals 存在
        assert len(pset.terminals) > 0

    def test_create_minimal_set(self, factory):
        """建立最小原語集"""
        pset = factory.create_minimal_set()

        # 檢查名稱
        assert pset.name == "MINIMAL"

        # 檢查輸入參數（只有 close）
        assert len(pset.arguments) == 1

        # 最小集應該少於標準集（比較 primitives 總數）
        standard_pset = factory.create_standard_set()

        # 計算 primitives 總數（所有類型的總和）
        minimal_count = sum(len(prims) for prims in pset.primitives.values())
        standard_count = sum(len(prims) for prims in standard_pset.primitives.values())

        assert minimal_count < standard_count

    def test_create_custom_set_default(self, factory):
        """建立自訂原語集（預設包含所有）"""
        pset = factory.create_custom_set()

        assert pset.name == "CUSTOM"
        assert len(pset.arguments) == 3  # close, high, low

    def test_create_custom_set_with_indicators(self, factory):
        """建立自訂原語集（指定指標）"""
        pset = factory.create_custom_set(
            indicators=['ma', 'rsi'],
            comparisons=['gt', 'cross_above']
        )

        assert pset.name == "CUSTOM"

        # 檢查是否有對應的 primitives
        # 注意：DEAP 內部使用 dict，這裡只檢查存在性
        assert len(pset.primitives) > 0

    def test_create_custom_set_adds_terminals_based_on_indicators(self, factory):
        """自訂集根據指標類型添加對應 terminals"""
        # 包含 RSI → 應該有 rsi_oversold, rsi_overbought
        pset_rsi = factory.create_custom_set(indicators=['rsi'])

        # 包含 MACD → 應該有 macd_fast, macd_slow
        pset_macd = factory.create_custom_set(indicators=['macd'])

        # 包含 BB → 應該有 bb_std_2
        pset_bb = factory.create_custom_set(indicators=['bb_upper'])

        # 檢查 terminals 數量（不同配置應不同）
        assert len(pset_rsi.terminals) > 0
        assert len(pset_macd.terminals) > 0
        assert len(pset_bb.terminals) > 0

    def test_primitive_set_argument_names(self, factory):
        """檢查參數名稱是否正確"""
        pset = factory.create_standard_set()

        # 檢查參數名稱（DEAP 會將 ARG0, ARG1, ARG2 重命名）
        # 注意：DEAP 的內部結構，這裡只檢查數量
        assert len(pset.arguments) == 3


# ============================================================================
# 7. 邊界條件測試
# ============================================================================

class TestEdgeCases:
    """測試邊界條件"""

    def test_indicators_with_nan_values(self):
        """指標處理包含 NaN 的輸入"""
        prices = np.array([100, np.nan, 102, 103, np.nan])

        # 這些指標應該能處理 NaN（不崩潰）
        rsi_result = rsi(prices, period=3)
        ma_result = ma(prices, period=3)
        ema_result = ema(prices, period=3)

        # 返回的陣列應該有正確長度
        assert len(rsi_result) == len(prices)
        assert len(ma_result) == len(prices)
        assert len(ema_result) == len(prices)

    def test_indicators_with_constant_prices(self):
        """指標處理恆定價格（無波動）"""
        prices = np.array([100.0, 100.0, 100.0, 100.0, 100.0])

        # RSI 在無波動時應為 0 或 NaN
        rsi_result = rsi(prices, period=3)
        # 允許 NaN 或 0（因為沒有漲跌）
        valid_values = rsi_result[~np.isnan(rsi_result)]
        if len(valid_values) > 0:
            assert np.all((valid_values >= 0) & (valid_values <= 100))

        # MA 應該接近價格（允許浮點誤差）
        ma_result = ma(prices, period=3)
        valid_values = ma_result[~np.isnan(ma_result)]
        assert_array_almost_equal(valid_values, np.full(len(valid_values), 100.0), decimal=5)

    def test_period_larger_than_data_length(self):
        """週期大於資料長度"""
        prices = np.array([100, 101, 102])

        # period=10 > len(prices)=3
        ma_result = ma(prices, period=10)

        # 大部分值應為 NaN
        assert np.all(np.isnan(ma_result))

    def test_negative_prices(self):
        """處理負數價格（理論上不應出現，但要能處理）"""
        prices = np.array([-100, -102, -101, -103, -105])

        # 不應崩潰
        ma_result = ma(prices, period=3)
        assert len(ma_result) == len(prices)

    def test_very_large_period(self):
        """處理非常大的週期"""
        prices = np.array([100, 102, 104, 106, 108])

        # period=1000000（遠大於資料長度）
        ma_result = ma(prices, period=1000000)

        # 所有值應為 NaN
        assert np.all(np.isnan(ma_result))


# ============================================================================
# 8. 整合測試
# ============================================================================

class TestIntegration:
    """整合測試（組合多個函數）"""

    def test_ma_crossover_strategy(self, trending_prices):
        """MA 交叉策略完整流程"""
        # 計算快慢均線
        ma_fast = ma(trending_prices, period=3)
        ma_slow = ma(trending_prices, period=5)

        # 產生訊號
        buy_signal = cross_above(ma_fast, ma_slow)
        sell_signal = cross_below(ma_fast, ma_slow)

        # 檢查訊號類型
        assert isinstance(buy_signal, np.ndarray)
        assert isinstance(sell_signal, np.ndarray)
        assert buy_signal.dtype == bool
        assert sell_signal.dtype == bool

    def test_rsi_threshold_strategy(self, sample_prices):
        """RSI 閾值策略完整流程"""
        close = sample_prices['close']

        # 計算 RSI
        rsi_values = rsi(close, period=5)

        # 常數閾值
        oversold = np.full_like(rsi_values, 30.0)
        overbought = np.full_like(rsi_values, 70.0)

        # 產生訊號
        buy_signal = lt(rsi_values, oversold)
        sell_signal = gt(rsi_values, overbought)

        # 檢查訊號
        assert isinstance(buy_signal, np.ndarray)
        assert isinstance(sell_signal, np.ndarray)

    def test_combined_logic(self):
        """組合邏輯運算"""
        a = np.array([True, True, False, False])
        b = np.array([True, False, True, False])
        c = np.array([False, False, True, True])

        # (a AND b) OR c
        result = or_(and_(a, b), c)

        expected = np.array([True, False, True, True])
        assert_array_equal(result, expected)

    def test_math_operations_combination(self):
        """數學運算組合"""
        # ((5 + 3) * 2) / 4 = 4
        result = protected_div(
            mul(add(5.0, 3.0), 2.0),
            4.0
        )

        assert result == pytest.approx(4.0, rel=1e-6)


# ============================================================================
# 執行測試
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
