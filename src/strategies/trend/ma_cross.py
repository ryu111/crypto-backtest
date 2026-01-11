"""
雙均線交叉策略 (Moving Average Cross)

策略邏輯：
- 進場：快線向上穿越慢線（金叉）做多，向下穿越（死叉）做空
- 出場：反向穿越或 ATR 止損
- 過濾：可選趨勢過濾器（200MA）

適用市場：
- 趨勢市場（trending market）
- 中長期持倉（4H, 1D 時間框）

參數說明：
- fast_period: 快線週期（建議 5-20）
- slow_period: 慢線週期（建議 20-60）
- stop_loss_atr: 止損距離（ATR 倍數，建議 1.5-3.0）
- use_trend_filter: 是否使用趨勢過濾（200MA）
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from pandas import Series, DataFrame

from ..base import TrendStrategy
from ..registry import register_strategy


@register_strategy('trend_ma_cross')
class MACrossStrategy(TrendStrategy):
    """
    雙均線交叉策略

    使用兩條簡單移動平均線（SMA）的交叉訊號進行交易。
    快線向上穿越慢線時做多，向下穿越時做空。
    """

    name = "MA Cross"
    strategy_type = "trend"
    version = "1.0"
    description = "雙均線交叉策略，適用於趨勢市場"

    # 預設參數
    params = {
        'fast_period': 10,
        'slow_period': 30,
        'stop_loss_atr': 2.0,
        'use_trend_filter': False,
        'trend_filter_period': 200,
    }

    # Optuna 優化空間
    param_space = {
        'fast_period': {'type': 'int', 'low': 5, 'high': 20},
        'slow_period': {'type': 'int', 'low': 20, 'high': 60},
        'stop_loss_atr': {'type': 'float', 'low': 1.0, 'high': 4.0},
    }

    def validate_params(self) -> bool:
        """驗證參數有效性"""
        # 快線必須小於慢線
        if self.params['fast_period'] >= self.params['slow_period']:
            return False

        # ATR 倍數必須為正
        if self.params['stop_loss_atr'] <= 0:
            return False

        return super().validate_params()

    def calculate_indicators(self, data: DataFrame) -> Dict[str, Series]:
        """
        計算策略所需指標

        Args:
            data: OHLCV DataFrame

        Returns:
            dict: 包含以下指標
                - sma_fast: 快線 SMA
                - sma_slow: 慢線 SMA
                - atr: 平均真實波幅
                - trend_ma: 趨勢過濾均線（可選）
        """
        close = data['close']
        high = data['high']
        low = data['low']

        indicators = {}

        # 計算雙均線
        indicators['sma_fast'] = close.rolling(
            window=self.params['fast_period']
        ).mean()

        indicators['sma_slow'] = close.rolling(
            window=self.params['slow_period']
        ).mean()

        # 計算 ATR（用於止損）
        indicators['atr'] = self._calculate_atr(
            high, low, close, period=14
        )

        # 趨勢過濾器（可選）
        if self.params['use_trend_filter']:
            indicators['trend_ma'] = close.rolling(
                window=self.params['trend_filter_period']
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

        fast_ma = indicators['sma_fast']
        slow_ma = indicators['sma_slow']
        atr = indicators['atr']

        # 初始化訊號
        long_entry = pd.Series(False, index=data.index)
        long_exit = pd.Series(False, index=data.index)
        short_entry = pd.Series(False, index=data.index)
        short_exit = pd.Series(False, index=data.index)

        # 均線交叉訊號
        # 金叉：快線向上穿越慢線
        golden_cross = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))

        # 死叉：快線向下穿越慢線
        death_cross = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

        # 進場訊號
        long_entry = golden_cross
        short_entry = death_cross

        # 出場訊號（反向交叉）
        long_exit = death_cross
        short_exit = golden_cross

        # 應用趨勢過濾器
        if self.params['use_trend_filter']:
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
        應用趨勢過濾器

        只允許在主趨勢方向交易：
        - 多單：價格在 200MA 之上
        - 空單：價格在 200MA 之下

        Args:
            data: OHLCV DataFrame
            long_entry: 多單進場訊號
            long_exit: 多單出場訊號
            short_entry: 空單進場訊號
            short_exit: 空單出場訊號

        Returns:
            tuple: 過濾後的訊號
        """
        if not self.params['use_trend_filter']:
            return long_entry, long_exit, short_entry, short_exit

        # 計算趨勢
        uptrend, downtrend = self.apply_trend_filter(
            data,
            period=self.params['trend_filter_period']
        )

        # 只在趨勢方向交易
        long_entry = long_entry & uptrend
        short_entry = short_entry & downtrend

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

        fast_ma = indicators['sma_fast'].iloc[index]
        slow_ma = indicators['sma_slow'].iloc[index]

        if fast_ma > slow_ma:
            return f"金叉：快線({fast_ma:.2f}) > 慢線({slow_ma:.2f})"
        else:
            return f"死叉：快線({fast_ma:.2f}) < 慢線({slow_ma:.2f})"
