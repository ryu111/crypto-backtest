"""
GP Primitives（遺傳規劃原語）

定義 DEAP Strongly Typed GP 的所有原語（primitives）：
- 指標原語：RSI, MA, EMA, ATR, MACD, BB
- 比較原語：gt, lt, cross_above, cross_below
- 邏輯原語：and_, or_, not_
- 數學原語：add, sub, mul, protected_div, protected_log
- 工廠類別：PrimitiveSetFactory

所有運算都是向量化（numpy），適用於時間序列回測。
"""

from typing import List, Callable, Optional, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

try:
    from deap import gp
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    gp = None

if TYPE_CHECKING:
    from deap.gp import PrimitiveSetTyped


# ============================================================================
# 類型系統（DEAP Strongly Typed GP）
# ============================================================================

class Price(float):
    """價格序列類型（OHLC）"""
    pass


class Indicator(float):
    """指標值類型"""
    pass


class Signal:
    """交易訊號類型（布林）"""
    pass


class Number(float):
    """數值參數類型"""
    pass


# ============================================================================
# 指標原語（Task 2.1）
# ============================================================================

def rsi(prices: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """
    相對強弱指標（RSI）

    Args:
        prices: 價格序列（收盤價）
        period: 計算週期（建議 14）

    Returns:
        RSI 值（0-100）

    Note:
        返回值可能包含 NaN（前 period 個值）
    """
    # 處理空陣列
    if len(prices) == 0:
        return np.array([])

    period = max(2, int(period))  # 最小週期 2

    # 計算價格變化
    delta = np.diff(prices, prepend=prices[0])

    # 分離漲跌
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    # 計算平均漲跌（使用 EMA）
    avg_gain = _ema(gains, period)
    avg_loss = _ema(losses, period)

    # 計算 RS 和 RSI（處理除零）
    # 當 avg_loss = 0（持續上漲）→ RSI = 100
    # 當 avg_gain = 0（持續下跌）→ RSI = 0
    rsi_values = np.full_like(prices, 50.0)  # 預設中線

    # 正常情況：有漲有跌
    mask_normal = (avg_loss > 1e-10) & (avg_gain > 1e-10)
    rs = avg_gain[mask_normal] / avg_loss[mask_normal]
    rsi_values[mask_normal] = 100.0 - (100.0 / (1.0 + rs))

    # 特殊情況：只漲不跌
    mask_only_gain = (avg_gain > 1e-10) & (avg_loss <= 1e-10)
    rsi_values[mask_only_gain] = 100.0

    # 特殊情況：只跌不漲
    mask_only_loss = (avg_gain <= 1e-10) & (avg_loss > 1e-10)
    rsi_values[mask_only_loss] = 0.0

    return rsi_values


def ma(prices: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """
    簡單移動平均（SMA）

    Args:
        prices: 價格序列
        period: 計算週期

    Returns:
        SMA 值
    """
    period = max(1, int(period))

    # 使用 numpy convolve 實現高效 SMA
    weights = np.ones(period) / period
    sma_values = np.convolve(prices, weights, mode='full')[:len(prices)]

    # 前 period-1 個值設為 NaN
    sma_values[:period-1] = np.nan

    return sma_values


def ema(prices: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """
    指數移動平均（EMA）

    Args:
        prices: 價格序列
        period: 計算週期

    Returns:
        EMA 值
    """
    return _ema(prices, max(1, int(period)))


def atr(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int
) -> NDArray[np.float64]:
    """
    平均真實波幅（ATR）

    Args:
        high: 最高價序列
        low: 最低價序列
        close: 收盤價序列
        period: 計算週期（建議 14）

    Returns:
        ATR 值
    """
    period = max(1, int(period))

    # 真實波動幅度（True Range）
    tr1 = high - low

    # 使用 np.roll 避免第一值問題
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]  # 第一天沒有前一日，用自己

    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)

    # 取三者最大值
    tr = np.maximum.reduce([tr1, tr2, tr3])

    # 計算 ATR（使用 EMA）
    return _ema(tr, int(max(1, period)))


def macd(
    prices: NDArray[np.float64],
    fast_period: int,
    slow_period: int
) -> NDArray[np.float64]:
    """
    MACD 指標（返回 MACD 線）

    Args:
        prices: 價格序列
        fast_period: 快線週期（建議 12）
        slow_period: 慢線週期（建議 26）

    Returns:
        MACD 線值（fast_ema - slow_ema）

    Note:
        此函數只返回 MACD 線，不返回訊號線和柱狀圖
    """
    fast_period = max(1, int(fast_period))
    slow_period = max(fast_period + 1, int(slow_period))

    # 計算快慢 EMA
    fast_ema = ema(prices, fast_period)
    slow_ema = ema(prices, slow_period)

    # MACD 線
    macd_line = fast_ema - slow_ema

    return macd_line


def bb_upper(
    prices: NDArray[np.float64],
    period: int,
    std_mult: float
) -> NDArray[np.float64]:
    """
    布林帶上軌

    Args:
        prices: 價格序列
        period: 計算週期（建議 20）
        std_mult: 標準差倍數（建議 2.0）

    Returns:
        布林帶上軌值
    """
    period = max(2, int(period))
    std_mult = max(0.1, float(std_mult))

    # 計算中軌（SMA）
    middle_band = ma(prices, period)

    # 計算標準差
    std_dev = _rolling_std(prices, period)

    # 上軌 = 中軌 + std_mult * std
    upper_band = middle_band + std_mult * std_dev

    return upper_band


def bb_lower(
    prices: NDArray[np.float64],
    period: int,
    std_mult: float
) -> NDArray[np.float64]:
    """
    布林帶下軌

    Args:
        prices: 價格序列
        period: 計算週期（建議 20）
        std_mult: 標準差倍數（建議 2.0）

    Returns:
        布林帶下軌值
    """
    period = max(2, int(period))
    std_mult = max(0.1, float(std_mult))

    # 計算中軌（SMA）
    middle_band = ma(prices, period)

    # 計算標準差
    std_dev = _rolling_std(prices, period)

    # 下軌 = 中軌 - std_mult * std
    lower_band = middle_band - std_mult * std_dev

    return lower_band


# ============================================================================
# 比較原語（Task 2.2）
# ============================================================================

def gt(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.bool_]:
    """
    大於比較（a > b）

    Args:
        a: 指標 A
        b: 指標 B

    Returns:
        布林陣列
    """
    return a > b


def lt(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.bool_]:
    """
    小於比較（a < b）

    Args:
        a: 指標 A
        b: 指標 B

    Returns:
        布林陣列
    """
    return a < b


def cross_above(
    a: NDArray[np.float64],
    b: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """
    向上穿越（a 從下方穿越 b）

    Args:
        a: 指標 A（快線）
        b: 指標 B（慢線）

    Returns:
        布林陣列（穿越點為 True）

    Example:
        金叉：cross_above(ma_fast, ma_slow)
    """
    # 處理空陣列
    if len(a) == 0:
        return np.array([], dtype=bool)

    # 當前：a > b
    # 前一期：a <= b
    current_above = a > b
    previous_below = np.roll(a, 1) <= np.roll(b, 1)

    # 第一個值設為 False（沒有前一期）
    cross = current_above & previous_below
    cross[0] = False

    return cross


def cross_below(
    a: NDArray[np.float64],
    b: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """
    向下穿越（a 從上方穿越 b）

    Args:
        a: 指標 A（快線）
        b: 指標 B（慢線）

    Returns:
        布林陣列（穿越點為 True）

    Example:
        死叉：cross_below(ma_fast, ma_slow)
    """
    # 處理空陣列
    if len(a) == 0:
        return np.array([], dtype=bool)

    # 當前：a < b
    # 前一期：a >= b
    current_below = a < b
    previous_above = np.roll(a, 1) >= np.roll(b, 1)

    # 第一個值設為 False（沒有前一期）
    cross = current_below & previous_above
    cross[0] = False

    return cross


# ============================================================================
# 邏輯原語（Task 2.3）
# ============================================================================

def and_(a: NDArray[np.bool_], b: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """
    邏輯 AND（a AND b）

    Args:
        a: 訊號 A
        b: 訊號 B

    Returns:
        布林陣列
    """
    return a & b


def or_(a: NDArray[np.bool_], b: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """
    邏輯 OR（a OR b）

    Args:
        a: 訊號 A
        b: 訊號 B

    Returns:
        布林陣列
    """
    return a | b


def not_(a: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """
    邏輯 NOT（NOT a）

    Args:
        a: 訊號 A

    Returns:
        布林陣列
    """
    return ~a


# ============================================================================
# 數學原語（Task 2.4）
# ============================================================================

def protected_div(a: float, b: float) -> float:
    """
    保護除法（避免除零）

    Args:
        a: 分子
        b: 分母

    Returns:
        a / b（b=0 時返回 1.0）
    """
    if abs(b) < 1e-10:
        return 1.0
    return a / b


def protected_log(a: float) -> float:
    """
    保護對數（避免負數和零）

    Args:
        a: 輸入值

    Returns:
        log(a)（a<=0 時返回 0.0）
    """
    if a <= 0:
        return 0.0
    return np.log(a)


def add(a: float, b: float) -> float:
    """
    加法（a + b）

    Args:
        a: 數值 A
        b: 數值 B

    Returns:
        a + b
    """
    return a + b


def sub(a: float, b: float) -> float:
    """
    減法（a - b）

    Args:
        a: 數值 A
        b: 數值 B

    Returns:
        a - b
    """
    return a - b


def mul(a: float, b: float) -> float:
    """
    乘法（a * b）

    Args:
        a: 數值 A
        b: 數值 B

    Returns:
        a * b
    """
    return a * b


# ============================================================================
# PrimitiveSetFactory（Task 2.5）
# ============================================================================

class PrimitiveSetFactory:
    """
    DEAP PrimitiveSetTyped 工廠類別

    負責建立不同配置的原語集（primitive set），供 GP 演化使用。

    使用範例：
        factory = PrimitiveSetFactory()
        pset = factory.create_standard_set()
        # pset 可直接傳給 DEAP GP
    """

    def __init__(self):
        if not DEAP_AVAILABLE:
            raise ImportError(
                "DEAP is required for GP primitives. "
                "Install it with: pip install deap"
            )

    def create_standard_set(self) -> "PrimitiveSetTyped":
        """
        建立標準原語集（包含所有原語）

        包含：
        - 指標：RSI, MA, EMA, ATR, MACD, BB
        - 比較：gt, lt, cross_above, cross_below
        - 邏輯：and_, or_, not_
        - 數學：add, sub, mul, protected_div, protected_log

        Returns:
            DEAP PrimitiveSetTyped 實例
        """
        # 建立類型化原語集
        # 輸入：close, high, low（3 個價格序列）
        # 輸出：Signal（交易訊號）
        pset = gp.PrimitiveSetTyped(
            "MAIN",
            [Price, Price, Price],  # close, high, low
            Signal
        )

        # 重命名輸入參數（方便理解）
        pset.renameArguments(ARG0='close', ARG1='high', ARG2='low')

        # ===== 指標原語 =====
        # RSI(prices, period) -> Indicator
        pset.addPrimitive(rsi, [Price, Number], Indicator, name='RSI')

        # MA(prices, period) -> Indicator
        pset.addPrimitive(ma, [Price, Number], Indicator, name='MA')

        # EMA(prices, period) -> Indicator
        pset.addPrimitive(ema, [Price, Number], Indicator, name='EMA')

        # ATR(high, low, close, period) -> Indicator
        pset.addPrimitive(
            atr,
            [Price, Price, Price, Number],
            Indicator,
            name='ATR'
        )

        # MACD(prices, fast, slow) -> Indicator
        pset.addPrimitive(
            macd,
            [Price, Number, Number],
            Indicator,
            name='MACD'
        )

        # BB_UPPER(prices, period, std_mult) -> Indicator
        pset.addPrimitive(
            bb_upper,
            [Price, Number, Number],
            Indicator,
            name='BB_UPPER'
        )

        # BB_LOWER(prices, period, std_mult) -> Indicator
        pset.addPrimitive(
            bb_lower,
            [Price, Number, Number],
            Indicator,
            name='BB_LOWER'
        )

        # ===== 比較原語 =====
        # GT(a, b) -> Signal
        pset.addPrimitive(gt, [Indicator, Indicator], Signal, name='GT')

        # LT(a, b) -> Signal
        pset.addPrimitive(lt, [Indicator, Indicator], Signal, name='LT')

        # CROSS_ABOVE(a, b) -> Signal
        pset.addPrimitive(
            cross_above,
            [Indicator, Indicator],
            Signal,
            name='CROSS_ABOVE'
        )

        # CROSS_BELOW(a, b) -> Signal
        pset.addPrimitive(
            cross_below,
            [Indicator, Indicator],
            Signal,
            name='CROSS_BELOW'
        )

        # ===== 邏輯原語 =====
        # AND(a, b) -> Signal
        pset.addPrimitive(and_, [Signal, Signal], Signal, name='AND')

        # OR(a, b) -> Signal
        pset.addPrimitive(or_, [Signal, Signal], Signal, name='OR')

        # NOT(a) -> Signal
        pset.addPrimitive(not_, [Signal], Signal, name='NOT')

        # ===== 數學原語（用於指標參數） =====
        # ADD(a, b) -> Number
        pset.addPrimitive(add, [Number, Number], Number, name='ADD')

        # SUB(a, b) -> Number
        pset.addPrimitive(sub, [Number, Number], Number, name='SUB')

        # MUL(a, b) -> Number
        pset.addPrimitive(mul, [Number, Number], Number, name='MUL')

        # PROTECTED_DIV(a, b) -> Number
        pset.addPrimitive(
            protected_div,
            [Number, Number],
            Number,
            name='PROTECTED_DIV'
        )

        # PROTECTED_LOG(a) -> Number
        pset.addPrimitive(
            protected_log,
            [Number],
            Number,
            name='PROTECTED_LOG'
        )

        # ===== 終端（常數） =====
        # 常用指標週期
        pset.addTerminal(7, Number, name='period_7')
        pset.addTerminal(14, Number, name='period_14')
        pset.addTerminal(20, Number, name='period_20')
        pset.addTerminal(30, Number, name='period_30')
        pset.addTerminal(50, Number, name='period_50')

        # MACD 快慢線週期
        pset.addTerminal(12, Number, name='macd_fast')
        pset.addTerminal(26, Number, name='macd_slow')
        pset.addTerminal(9, Number, name='macd_signal')

        # 布林帶標準差倍數
        pset.addTerminal(2.0, Number, name='bb_std_2')
        pset.addTerminal(2.5, Number, name='bb_std_2_5')

        # RSI 閾值
        pset.addTerminal(30, Number, name='rsi_oversold')
        pset.addTerminal(70, Number, name='rsi_overbought')

        return pset

    def create_minimal_set(self) -> "PrimitiveSetTyped":
        """
        建立最小原語集（只有基本指標）

        包含：
        - 指標：MA, RSI
        - 比較：gt, lt, cross_above
        - 邏輯：and_, or_

        適用於快速測試或受限環境。

        Returns:
            DEAP PrimitiveSetTyped 實例
        """
        pset = gp.PrimitiveSetTyped(
            "MINIMAL",
            [Price],  # 只需 close
            Signal
        )

        pset.renameArguments(ARG0='close')

        # 基本指標
        pset.addPrimitive(ma, [Price, Number], Indicator, name='MA')
        pset.addPrimitive(rsi, [Price, Number], Indicator, name='RSI')

        # 基本比較
        pset.addPrimitive(gt, [Indicator, Indicator], Signal, name='GT')
        pset.addPrimitive(lt, [Indicator, Indicator], Signal, name='LT')
        pset.addPrimitive(
            cross_above,
            [Indicator, Indicator],
            Signal,
            name='CROSS_ABOVE'
        )

        # 基本邏輯
        pset.addPrimitive(and_, [Signal, Signal], Signal, name='AND')
        pset.addPrimitive(or_, [Signal, Signal], Signal, name='OR')

        # 基本終端
        pset.addTerminal(10, Number, name='period_10')
        pset.addTerminal(20, Number, name='period_20')
        pset.addTerminal(14, Number, name='rsi_period')
        pset.addTerminal(30, Number, name='rsi_oversold')
        pset.addTerminal(70, Number, name='rsi_overbought')

        return pset

    def create_custom_set(
        self,
        indicators: Optional[List[str]] = None,
        comparisons: Optional[List[str]] = None
    ) -> "PrimitiveSetTyped":
        """
        建立自定義原語集

        Args:
            indicators: 指標名稱列表（可選）
                可用：'rsi', 'ma', 'ema', 'atr', 'macd', 'bb_upper', 'bb_lower'
            comparisons: 比較運算名稱列表（可選）
                可用：'gt', 'lt', 'cross_above', 'cross_below'

        Returns:
            DEAP PrimitiveSetTyped 實例

        Example:
            factory = PrimitiveSetFactory()
            pset = factory.create_custom_set(
                indicators=['ma', 'rsi'],
                comparisons=['gt', 'cross_above']
            )
        """
        # 預設包含所有
        if indicators is None:
            indicators = ['rsi', 'ma', 'ema', 'atr', 'macd', 'bb_upper', 'bb_lower']

        if comparisons is None:
            comparisons = ['gt', 'lt', 'cross_above', 'cross_below']

        # 建立原語集
        pset = gp.PrimitiveSetTyped(
            "CUSTOM",
            [Price, Price, Price],
            Signal
        )

        pset.renameArguments(ARG0='close', ARG1='high', ARG2='low')

        # 指標映射
        indicator_map = {
            'rsi': (rsi, [Price, Number], 'RSI'),
            'ma': (ma, [Price, Number], 'MA'),
            'ema': (ema, [Price, Number], 'EMA'),
            'atr': (atr, [Price, Price, Price, Number], 'ATR'),
            'macd': (macd, [Price, Number, Number], 'MACD'),
            'bb_upper': (bb_upper, [Price, Number, Number], 'BB_UPPER'),
            'bb_lower': (bb_lower, [Price, Number, Number], 'BB_LOWER'),
        }

        # 添加指標
        for ind_name in indicators:
            if ind_name in indicator_map:
                func, types, name = indicator_map[ind_name]
                pset.addPrimitive(func, types, Indicator, name=name)

        # 比較映射
        comparison_map = {
            'gt': (gt, 'GT'),
            'lt': (lt, 'LT'),
            'cross_above': (cross_above, 'CROSS_ABOVE'),
            'cross_below': (cross_below, 'CROSS_BELOW'),
        }

        # 添加比較
        for comp_name in comparisons:
            if comp_name in comparison_map:
                func, name = comparison_map[comp_name]
                pset.addPrimitive(
                    func,
                    [Indicator, Indicator],
                    Signal,
                    name=name
                )

        # 總是包含邏輯運算（必需）
        pset.addPrimitive(and_, [Signal, Signal], Signal, name='AND')
        pset.addPrimitive(or_, [Signal, Signal], Signal, name='OR')
        pset.addPrimitive(not_, [Signal], Signal, name='NOT')

        # 添加終端（根據指標類型）
        pset.addTerminal(14, Number, name='period_14')
        pset.addTerminal(20, Number, name='period_20')

        if 'rsi' in indicators:
            pset.addTerminal(30, Number, name='rsi_oversold')
            pset.addTerminal(70, Number, name='rsi_overbought')

        if 'macd' in indicators:
            pset.addTerminal(12, Number, name='macd_fast')
            pset.addTerminal(26, Number, name='macd_slow')
            pset.addTerminal(9, Number, name='macd_signal')

        if 'bb_upper' in indicators or 'bb_lower' in indicators:
            pset.addTerminal(2.0, Number, name='bb_std_2')

        return pset


# ============================================================================
# 輔助函數（內部使用）
# ============================================================================

def _ema(data: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """
    指數移動平均（內部實現，向量化）

    Args:
        data: 輸入數據
        period: 週期

    Returns:
        EMA 值
    """
    import pandas as pd
    return pd.Series(data).ewm(span=period, adjust=False).mean().values


def _rolling_std(data: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """
    滾動標準差（內部實現，向量化）

    Args:
        data: 輸入數據
        period: 週期

    Returns:
        標準差值
    """
    import pandas as pd
    return pd.Series(data).rolling(period).std(ddof=1).values


# ============================================================================
# 公開 API
# ============================================================================

__all__ = [
    # 類型
    'Price',
    'Indicator',
    'Signal',
    'Number',
    # 指標原語
    'rsi',
    'ma',
    'ema',
    'atr',
    'macd',
    'bb_upper',
    'bb_lower',
    # 比較原語
    'gt',
    'lt',
    'cross_above',
    'cross_below',
    # 邏輯原語
    'and_',
    'or_',
    'not_',
    # 數學原語
    'protected_div',
    'protected_log',
    'add',
    'sub',
    'mul',
    # 工廠
    'PrimitiveSetFactory',
]
