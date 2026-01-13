"""
Donchian Channel Breakout 策略

策略邏輯：
- 進場：突破 N 日最高價做多，跌破 N 日最低價做空
- 出場：反向突破或 ATR 止損
- 過濾：可選 ATR 過濾器（排除低波動期）

適用市場：
- 突破行情（breakout market）
- 中長期持倉（4H, 1D 時間框）

參數說明：
- period: Donchian Channel 週期（建議 10-50）
- stop_loss_atr: 止損距離（ATR 倍數，建議 1.0-4.0）
- use_atr_filter: 是否使用 ATR 過濾低波動期
- atr_threshold: NATR 閾值（建議 0.01 = 1%）
"""

import pandas as pd
from typing import Dict, Tuple
from pandas import Series, DataFrame

from ..base import TrendStrategy
from ..registry import register_strategy


@register_strategy('trend_donchian')
class DonchianStrategy(TrendStrategy):
    """
    Donchian Channel Breakout 策略

    使用 Donchian Channel 的突破訊號進行交易。
    突破上軌做多，突破下軌做空，反向突破出場。
    """

    name = "Donchian Channel Breakout"
    strategy_type = "trend"
    version = "1.0"
    description = "Donchian Channel 突破策略，適用於突破行情"

    # 預設參數
    params = {
        'period': 20,
        'stop_loss_atr': 2.0,
        'use_atr_filter': False,
        'atr_threshold': 0.01,
    }

    # Optuna 優化空間
    param_space = {
        'period': {'type': 'int', 'low': 10, 'high': 50},
        'stop_loss_atr': {'type': 'float', 'low': 1.0, 'high': 4.0},
    }

    def __init__(self, **kwargs):
        """
        初始化策略

        Args:
            **kwargs: 覆寫預設參數
        """
        # 複製類別預設參數到實例
        self.params = self.__class__.params.copy()
        self.param_space = self.__class__.param_space.copy()

        # 更新傳入的參數
        self.params.update(kwargs)

        # 驗證參數
        if not self.validate_params():
            raise ValueError(f"Invalid parameters for {self.name}")

    def validate_params(self) -> bool:
        """驗證參數有效性"""
        # 週期必須為正
        if self.params['period'] <= 0:
            return False

        # ATR 倍數必須為正
        if self.params['stop_loss_atr'] <= 0:
            return False

        # ATR 閾值必須在合理範圍
        if self.params['use_atr_filter'] and self.params['atr_threshold'] <= 0:
            return False

        return super().validate_params()

    def calculate_indicators(self, data: DataFrame) -> Dict[str, Series]:
        """
        計算策略所需指標

        Args:
            data: OHLCV DataFrame

        Returns:
            dict: 包含以下指標
                - donchian_upper: Donchian 上軌
                - donchian_lower: Donchian 下軌
                - donchian_middle: Donchian 中線
                - atr: 平均真實波幅
                - natr: 標準化 ATR（ATR / close）
        """
        close = data['close']
        high = data['high']
        low = data['low']
        period = self.params['period']

        indicators = {}

        # 計算 Donchian Channel
        indicators['donchian_upper'] = high.rolling(window=period).max()
        indicators['donchian_lower'] = low.rolling(window=period).min()
        indicators['donchian_middle'] = (
            indicators['donchian_upper'] + indicators['donchian_lower']
        ) / 2

        # 計算 ATR（用於止損和波動過濾）
        indicators['atr'] = self._calculate_atr(
            high, low, close, period=14
        )

        # 計算標準化 ATR（NATR）
        indicators['natr'] = indicators['atr'] / close

        return indicators

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

        close = data['close']
        high = data['high']
        low = data['low']

        upper = indicators['donchian_upper']
        lower = indicators['donchian_lower']
        natr = indicators['natr']

        # 初始化訊號
        long_entry = pd.Series(False, index=data.index)
        long_exit = pd.Series(False, index=data.index)
        short_entry = pd.Series(False, index=data.index)
        short_exit = pd.Series(False, index=data.index)

        # 突破訊號（檢測穿越，避免持續觸發）
        # 多單進場：突破上軌（當前價格 > 上軌，但前一根未突破）
        long_breakout = (high > upper.shift(1)) & (high.shift(1) <= upper.shift(2))

        # 空單進場：跌破下軌（當前價格 < 下軌，但前一根未跌破）
        short_breakout = (low < lower.shift(1)) & (low.shift(1) >= lower.shift(2))

        # 進場訊號
        long_entry = long_breakout
        short_entry = short_breakout

        # 出場訊號（反向突破）
        long_exit = short_breakout
        short_exit = long_breakout

        # 應用 ATR 波動過濾器
        if self.params['use_atr_filter']:
            atr_filter = natr >= self.params['atr_threshold']
            long_entry = long_entry & atr_filter
            short_entry = short_entry & atr_filter

        return long_entry, long_exit, short_entry, short_exit

    def _calculate_atr(
        self,
        high: Series,
        low: Series,
        close: Series,
        period: int = 14
    ) -> Series:
        """
        計算平均真實波幅（ATR）

        Args:
            high: 最高價
            low: 最低價
            close: 收盤價
            period: 計算週期

        Returns:
            Series: ATR 值
        """
        # 計算真實波幅（TR）
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 計算 ATR（簡單移動平均）
        atr = tr.rolling(window=period).mean()

        return atr

    def calculate_stop_loss(
        self,
        data: DataFrame,
        entry_price: float,
        position_type: str = 'long'
    ) -> float:
        """
        計算止損價格

        Args:
            data: OHLCV DataFrame
            entry_price: 入場價格
            position_type: 部位類型（'long' 或 'short'）

        Returns:
            float: 止損價格
        """
        # 計算當前 ATR
        indicators = self.calculate_indicators(data)
        current_atr = indicators['atr'].iloc[-1]

        # 計算止損距離
        stop_distance = self.params['stop_loss_atr'] * current_atr

        # 根據部位類型計算止損價
        if position_type == 'long':
            stop_loss = entry_price - stop_distance
        else:  # short
            stop_loss = entry_price + stop_distance

        return stop_loss

    def get_entry_reason(self, data: DataFrame, index: int) -> str:
        """
        取得進場原因說明

        Args:
            data: OHLCV DataFrame
            index: 資料索引

        Returns:
            str: 進場原因
        """
        indicators = self.calculate_indicators(data)

        close = data['close'].iloc[index]
        upper = indicators['donchian_upper'].iloc[index]
        lower = indicators['donchian_lower'].iloc[index]
        middle = indicators['donchian_middle'].iloc[index]

        if close > middle:
            return f"突破上軌：價格({close:.2f}) > 上軌({upper:.2f})"
        else:
            return f"跌破下軌：價格({close:.2f}) < 下軌({lower:.2f})"
