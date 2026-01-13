"""
Supertrend 策略

策略邏輯：
- 進場：Supertrend 方向翻轉時進場
- 出場：Supertrend 反向翻轉或固定止盈止損
- 止損：Supertrend 線本身作為動態止損

適用市場：
- 趨勢市場（trending market）
- 中短期持倉（1H, 4H 時間框）

參數說明：
- period: ATR 計算週期（建議 7-14）
- multiplier: ATR 倍數（建議 2.0-4.0）
- use_volume_filter: 是否使用成交量過濾

Supertrend 指標說明：
Supertrend 是基於 ATR 的趨勢跟隨指標，計算方式：
- 上軌 = (H+L)/2 + multiplier × ATR
- 下軌 = (H+L)/2 - multiplier × ATR
- 當價格突破上軌，視為上升趨勢
- 當價格跌破下軌，視為下降趨勢
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from pandas import Series, DataFrame

from ..base import TrendStrategy
from ..registry import register_strategy


@register_strategy('trend_supertrend')
class SupertrendStrategy(TrendStrategy):
    """
    Supertrend 趨勢策略

    使用 Supertrend 指標識別趨勢並產生交易訊號。
    Supertrend 同時提供趨勢方向和動態止損位置。
    """

    name = "Supertrend"
    strategy_type = "trend"
    version = "1.0"
    description = "Supertrend 趨勢跟隨策略，提供動態止損"

    # 預設參數
    params = {
        'period': 10,
        'multiplier': 3.0,
        'use_volume_filter': False,
        'volume_ma_period': 20,
        'volume_threshold': 1.0,
    }

    # Optuna 優化空間
    param_space = {
        'period': {'type': 'int', 'low': 7, 'high': 14},
        'multiplier': {'type': 'float', 'low': 2.0, 'high': 4.0},
    }

    def __init__(self, **kwargs):
        """初始化策略"""
        self.params = self.__class__.params.copy()
        self.param_space = self.__class__.param_space.copy()
        self.params.update(kwargs)
        if not self.validate_params():
            raise ValueError(f"Invalid parameters for {self.name}")

    def validate_params(self) -> bool:
        """驗證參數有效性"""
        # ATR 週期必須為正整數
        if self.params['period'] <= 0:
            return False

        # 倍數必須為正數
        if self.params['multiplier'] <= 0:
            return False

        # 成交量閾值必須為正數
        if self.params['volume_threshold'] <= 0:
            return False

        return super().validate_params()

    def calculate_indicators(self, data: DataFrame) -> Dict[str, Series]:
        """
        計算策略所需指標

        Args:
            data: OHLCV DataFrame

        Returns:
            dict: 包含以下指標
                - supertrend: Supertrend 線
                - direction: 趨勢方向（1=上升, -1=下降）
                - upper_band: 上軌
                - lower_band: 下軌
                - atr: ATR 值
                - volume_ma: 成交量均線（可選）
        """
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']

        indicators = {}

        # 計算 ATR
        indicators['atr'] = self._calculate_atr(
            high, low, close,
            period=self.params['period']
        )

        # 計算 Supertrend
        supertrend, direction, upper_band, lower_band = self._calculate_supertrend(
            high, low, close,
            period=self.params['period'],
            multiplier=self.params['multiplier']
        )

        indicators['supertrend'] = supertrend
        indicators['direction'] = direction
        indicators['upper_band'] = upper_band
        indicators['lower_band'] = lower_band

        # 成交量過濾器（可選）
        if self.params['use_volume_filter']:
            indicators['volume_ma'] = volume.rolling(
                window=self.params['volume_ma_period']
            ).mean()

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

        direction = indicators['direction']

        # 初始化訊號
        long_entry = pd.Series(False, index=data.index)
        long_exit = pd.Series(False, index=data.index)
        short_entry = pd.Series(False, index=data.index)
        short_exit = pd.Series(False, index=data.index)

        # Supertrend 方向變化訊號
        # 從下降轉為上升 → 做多
        long_entry = (direction == 1) & (direction.shift(1) == -1)

        # 從上升轉為下降 → 做空
        short_entry = (direction == -1) & (direction.shift(1) == 1)

        # 出場訊號（趨勢反轉）
        long_exit = short_entry
        short_exit = long_entry

        # 應用成交量過濾器
        if self.params['use_volume_filter']:
            long_entry, long_exit, short_entry, short_exit = self.apply_filters(
                data, long_entry, long_exit, short_entry, short_exit
            )

        return long_entry, long_exit, short_entry, short_exit

    def apply_filters(
        self,
        data: DataFrame,
        long_entry: Series,
        long_exit: Series,
        short_entry: Series,
        short_exit: Series
    ) -> Tuple[Series, Series, Series, Series]:
        """
        應用成交量過濾器

        只在成交量放大時進場，提高訊號可靠性。

        Args:
            data: OHLCV DataFrame
            long_entry: 多單進場訊號
            long_exit: 多單出場訊號
            short_entry: 空單進場訊號
            short_exit: 空單出場訊號

        Returns:
            tuple: 過濾後的訊號
        """
        if not self.params['use_volume_filter']:
            return long_entry, long_exit, short_entry, short_exit

        volume = data['volume']
        indicators = self.calculate_indicators(data)
        volume_ma = indicators['volume_ma']

        # 成交量大於均值的一定倍數
        high_volume = volume > (volume_ma * self.params['volume_threshold'])

        # 只在高成交量時進場
        long_entry = long_entry & high_volume
        short_entry = short_entry & high_volume

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

    def _calculate_supertrend(
        self,
        high: Series,
        low: Series,
        close: Series,
        period: int = 10,
        multiplier: float = 3.0
    ) -> Tuple[Series, Series, Series, Series]:
        """
        計算 Supertrend 指標

        Args:
            high: 最高價
            low: 最低價
            close: 收盤價
            period: ATR 週期
            multiplier: ATR 倍數

        Returns:
            tuple: (supertrend, direction, upper_band, lower_band)
                - supertrend: Supertrend 線
                - direction: 趨勢方向（1=上升, -1=下降）
                - upper_band: 上軌
                - lower_band: 下軌
        """
        # 計算 ATR
        atr = self._calculate_atr(high, low, close, period)

        # 計算基礎帶
        hl2 = (high + low) / 2

        # 初始化上下軌
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        # 初始化 Supertrend 和方向
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=int)

        # 設定初始值
        supertrend.iloc[0] = lower_band.iloc[0]
        direction.iloc[0] = 1

        # 逐步計算 Supertrend
        for i in range(1, len(close)):
            # 當前趨勢方向
            curr_dir = direction.iloc[i-1]

            # 更新上下軌（確保不會頻繁翻轉）
            if curr_dir == 1:  # 上升趨勢
                # 下軌不能低於前一根
                lower_band.iloc[i] = max(lower_band.iloc[i], lower_band.iloc[i-1])

                # 檢查是否轉為下降
                if close.iloc[i] <= lower_band.iloc[i]:
                    direction.iloc[i] = -1
                    supertrend.iloc[i] = upper_band.iloc[i]
                else:
                    direction.iloc[i] = 1
                    supertrend.iloc[i] = lower_band.iloc[i]

            else:  # 下降趨勢
                # 上軌不能高於前一根
                upper_band.iloc[i] = min(upper_band.iloc[i], upper_band.iloc[i-1])

                # 檢查是否轉為上升
                if close.iloc[i] >= upper_band.iloc[i]:
                    direction.iloc[i] = 1
                    supertrend.iloc[i] = lower_band.iloc[i]
                else:
                    direction.iloc[i] = -1
                    supertrend.iloc[i] = upper_band.iloc[i]

        return supertrend, direction, upper_band, lower_band

    def get_stop_loss(self, data: DataFrame, position_type: str = 'long') -> float:
        """
        取得當前止損價格（Supertrend 線）

        Args:
            data: OHLCV DataFrame
            position_type: 部位類型（'long' 或 'short'）

        Returns:
            float: 止損價格
        """
        indicators = self.calculate_indicators(data)
        supertrend = indicators['supertrend'].iloc[-1]

        return supertrend

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

        direction = indicators['direction'].iloc[index]
        supertrend = indicators['supertrend'].iloc[index]
        close = data['close'].iloc[index]

        if direction == 1:
            return f"Supertrend 轉多：價格({close:.2f}) > Supertrend({supertrend:.2f})"
        else:
            return f"Supertrend 轉空：價格({close:.2f}) < Supertrend({supertrend:.2f})"

    def calculate_target_price(
        self,
        entry_price: float,
        risk_reward_ratio: float = 2.0,
        position_type: str = 'long',
        data: DataFrame = None
    ) -> float:
        """
        計算目標價格（基於風險回報比）

        Args:
            entry_price: 入場價格
            risk_reward_ratio: 風險回報比（預設 2:1）
            position_type: 部位類型
            data: OHLCV DataFrame（用於計算止損）

        Returns:
            float: 目標價格
        """
        if data is None:
            raise ValueError("需要提供 data 以計算止損")

        # 取得止損價格
        stop_loss = self.get_stop_loss(data, position_type)

        # 計算風險
        risk = abs(entry_price - stop_loss)

        # 計算目標價格
        if position_type == 'long':
            target = entry_price + (risk * risk_reward_ratio)
        else:  # short
            target = entry_price - (risk * risk_reward_ratio)

        return target
