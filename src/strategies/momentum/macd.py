"""
MACD 動量策略

基於 MACD 指標的趨勢跟隨策略。
當 MACD 線穿越訊號線時產生交易訊號。
"""

from typing import Dict, Tuple
from pandas import Series, DataFrame

from ..base import MomentumStrategy
from ..registry import register_strategy


@register_strategy('momentum_macd')
class MACDStrategy(MomentumStrategy):
    """
    MACD 交叉策略

    策略邏輯：
    - 多單進場：MACD 線向上穿越訊號線（黃金交叉）
    - 多單出場：MACD 線向下穿越訊號線（死亡交叉）
    - 空單進場：MACD 線向下穿越訊號線（死亡交叉）
    - 空單出場：MACD 線向上穿越訊號線（黃金交叉）
    - 柱狀圖方向確認：可選擇性使用柱狀圖確認訊號強度

    Parameters:
        fast_period (int): 快線 EMA 週期，預設 12
        slow_period (int): 慢線 EMA 週期，預設 26
        signal_period (int): 訊號線 EMA 週期，預設 9
        use_histogram (bool): 是否使用柱狀圖確認，預設 True
        histogram_threshold (float): 柱狀圖閾值，預設 0

    Example:
        >>> strategy = MACDStrategy(fast_period=12, slow_period=26, signal_period=9)
        >>> long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(data)
    """

    name = "MACD Crossover"
    strategy_type = "momentum"
    version = "1.0"
    description = "MACD line and signal line crossover strategy with histogram confirmation"

    # 預設參數
    params = {
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9,
        'use_histogram': True,
        'histogram_threshold': 0,
    }

    # 參數優化空間
    param_space = {
        'fast_period': {
            'type': 'int',
            'low': 8,
            'high': 20,
        },
        'slow_period': {
            'type': 'int',
            'low': 20,
            'high': 40,
        },
        'signal_period': {
            'type': 'int',
            'low': 5,
            'high': 15,
        },
        'histogram_threshold': {
            'type': 'float',
            'low': -0.1,
            'high': 0.1,
        },
    }

    def __init__(self, **kwargs):
        """初始化策略"""
        self.params = self.__class__.params.copy()
        self.param_space = self.__class__.param_space.copy()
        self.params.update(kwargs)
        if not self.validate_params():
            raise ValueError(f"Invalid parameters for {self.name}")

    def validate_params(self) -> bool:
        """
        驗證參數有效性

        Returns:
            bool: 參數是否有效
        """
        # 基礎驗證
        if not super().validate_params():
            return False

        # 快線必須 < 慢線
        if self.params['fast_period'] >= self.params['slow_period']:
            return False

        # 所有週期必須 > 0
        if self.params['fast_period'] <= 0:
            return False
        if self.params['slow_period'] <= 0:
            return False
        if self.params['signal_period'] <= 0:
            return False

        return True

    def calculate_indicators(self, data: DataFrame) -> Dict[str, Series]:
        """
        計算 MACD 指標

        Args:
            data: OHLCV DataFrame

        Returns:
            dict: {
                'macd': MACD 線,
                'signal': 訊號線,
                'histogram': 柱狀圖
            }
        """
        # 計算 MACD
        macd_line, signal_line, histogram = self.calculate_macd(
            data['close'],
            fast=self.params['fast_period'],
            slow=self.params['slow_period'],
            signal=self.params['signal_period']
        )

        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram,
        }

    def generate_signals(
        self,
        data: DataFrame
    ) -> Tuple[Series, Series, Series, Series]:
        """
        產生交易訊號

        Args:
            data: OHLCV DataFrame

        Returns:
            tuple: (long_entry, long_exit, short_entry, short_exit)
        """
        # 計算指標
        indicators = self.calculate_indicators(data)
        macd = indicators['macd']
        signal = indicators['signal']
        histogram = indicators['histogram']

        # 初始化訊號（使用 BaseStrategy 提供的輔助方法）
        long_entry = self._create_signal_series(data)
        long_exit = self._create_signal_series(data)
        short_entry = self._create_signal_series(data)
        short_exit = self._create_signal_series(data)

        # MACD 穿越條件
        # 黃金交叉：MACD 線由下往上穿越訊號線
        golden_cross = (macd > signal) & (macd.shift(1) <= signal.shift(1))

        # 死亡交叉：MACD 線由上往下穿越訊號線
        death_cross = (macd < signal) & (macd.shift(1) >= signal.shift(1))

        # 柱狀圖確認（可選）
        if self.params['use_histogram']:
            threshold = self.params['histogram_threshold']

            # 柱狀圖轉正/轉負（計算但未直接使用，保留供未來策略變種）
            _histogram_bullish = (histogram > threshold) & (histogram.shift(1) <= threshold)
            _histogram_bearish = (histogram < -threshold) & (histogram.shift(1) >= -threshold)
            del _histogram_bullish, _histogram_bearish  # 明確釋放未使用變數

            # 結合穿越訊號與柱狀圖確認
            long_entry = golden_cross & (histogram > threshold)
            short_entry = death_cross & (histogram < -threshold)
        else:
            # 只使用穿越訊號
            long_entry = golden_cross
            short_entry = death_cross

        # 出場訊號：反向穿越
        long_exit = death_cross
        short_exit = golden_cross

        return long_entry, long_exit, short_entry, short_exit

    def apply_filters(
        self,
        data: DataFrame,  # noqa: ARG002 - interface requirement
        long_entry: Series,
        long_exit: Series,
        short_entry: Series,
        short_exit: Series
    ) -> Tuple[Series, Series, Series, Series]:
        """
        應用額外過濾器（可選覆寫）

        常見的 MACD 過濾器包括：
        - MACD 零軸過濾：只在 MACD > 0 時做多，< 0 時做空
        - 趨勢過濾：結合長期均線確認趨勢
        - 背離檢測：價格與 MACD 背離時反向操作

        Args:
            data: OHLCV DataFrame
            long_entry: 多單進場訊號
            long_exit: 多單出場訊號
            short_entry: 空單進場訊號
            short_exit: 空單出場訊號

        Returns:
            tuple: 過濾後的訊號
        """
        # 預設不加入額外過濾
        # 子類別可覆寫此方法加入自訂過濾邏輯
        return long_entry, long_exit, short_entry, short_exit

    def detect_divergence(
        self,
        data: DataFrame,
        lookback: int = 14
    ) -> Tuple[Series, Series]:
        """
        檢測 MACD 背離

        背離是價格與指標走勢不一致的現象，可能預示趨勢反轉：
        - 看漲背離：價格創新低，但 MACD 未創新低
        - 看跌背離：價格創新高，但 MACD 未創新高

        Args:
            data: OHLCV DataFrame
            lookback: 回溯週期

        Returns:
            tuple: (bullish_divergence, bearish_divergence)
        """
        indicators = self.calculate_indicators(data)
        macd = indicators['macd']

        # 使用 DataFrameOps 進行高效滾動計算
        ops = self._wrap_data(data)
        close_ops = ops['close']

        # 計算滾動最高/最低（使用 DataFrameOps）
        price_high = close_ops.rolling_max(lookback).to_pandas()
        price_low = close_ops.rolling_min(lookback).to_pandas()

        # MACD 滾動極值（用於完整背離檢測，當前簡化版本使用 shift 替代）
        # macd_high = macd.rolling(lookback).max()  # 完整版可用
        # macd_low = macd.rolling(lookback).min()   # 完整版可用

        # 價格 Series（轉為 Pandas 供後續比較）
        close = close_ops.to_pandas()

        # 看漲背離：價格新低但 MACD 未新低
        bullish_div = (
            (close == price_low) &
            (close < close.shift(lookback)) &
            (macd > macd.shift(lookback))
        )

        # 看跌背離：價格新高但 MACD 未新高
        bearish_div = (
            (close == price_high) &
            (close > close.shift(lookback)) &
            (macd < macd.shift(lookback))
        )

        return bullish_div, bearish_div
