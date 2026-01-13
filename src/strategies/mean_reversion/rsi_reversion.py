"""
RSI 均值回歸策略 (RSI Mean Reversion)

策略邏輯：
- 進場：RSI < oversold（超賣）做多，RSI > overbought（超買）做空
- 出場：RSI 回到 exit_threshold（通常是中性區 50）
- 風險管理：ATR 止損

適用市場：
- 震盪市場（ranging market）
- 短中期持倉（1H, 4H 時間框）

參數說明：
- rsi_period: RSI 計算週期（建議 7-21）
- oversold: 超賣閾值（建議 20-40）
- overbought: 超買閾值（建議 60-80）
- exit_threshold: 出場閾值（建議 40-60）
- stop_loss_atr: 止損距離（ATR 倍數，建議 1.5-3.0）
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from pandas import Series, DataFrame

from ..base import MeanReversionStrategy
from ..registry import register_strategy


@register_strategy('mean_reversion_rsi')
class RSIReversionStrategy(MeanReversionStrategy):
    """
    RSI 均值回歸策略

    利用 RSI 指標識別超買超賣區域，在極端位置反向進場，
    等待價格回歸均值時出場。
    """

    name = "RSI Mean Reversion"
    strategy_type = "mean_reversion"
    version = "1.0"
    description = "RSI 均值回歸策略，適用於震盪市場"

    # 預設參數
    params = {
        'rsi_period': 14,
        'oversold': 30,
        'overbought': 70,
        'exit_threshold': 50,
        'stop_loss_atr': 2.0,
    }

    # Optuna 優化空間
    param_space = {
        'rsi_period': {'type': 'int', 'low': 7, 'high': 21},
        'oversold': {'type': 'int', 'low': 20, 'high': 40},
        'overbought': {'type': 'int', 'low': 60, 'high': 80},
        'stop_loss_atr': {'type': 'float', 'low': 1.5, 'high': 3.0},
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
        # oversold < exit_threshold < overbought
        if not (
            self.params['oversold']
            < self.params['exit_threshold']
            < self.params['overbought']
        ):
            return False

        # RSI period 必須為正
        if self.params['rsi_period'] <= 0:
            return False

        # ATR 倍數必須為正
        if self.params['stop_loss_atr'] <= 0:
            return False

        # oversold 和 overbought 必須在 0-100 範圍內
        if not (0 < self.params['oversold'] < 100):
            return False
        if not (0 < self.params['overbought'] < 100):
            return False

        return super().validate_params()

    def calculate_indicators(self, data: DataFrame) -> Dict[str, Series]:
        """
        計算策略所需指標

        Args:
            data: OHLCV DataFrame

        Returns:
            dict: 包含以下指標
                - rsi: RSI 指標值
                - atr: 平均真實波幅（用於止損）
        """
        close = data['close']
        high = data['high']
        low = data['low']

        indicators = {}

        # 計算 RSI
        indicators['rsi'] = self._calculate_rsi(
            close,
            period=self.params['rsi_period']
        )

        # 計算 ATR（用於止損）
        indicators['atr'] = self._calculate_atr(
            high, low, close, period=14
        )

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

        rsi = indicators['rsi']

        # 取得參數
        oversold = self.params['oversold']
        overbought = self.params['overbought']
        exit_threshold = self.params['exit_threshold']

        # 初始化訊號
        long_entry = pd.Series(False, index=data.index)
        long_exit = pd.Series(False, index=data.index)
        short_entry = pd.Series(False, index=data.index)
        short_exit = pd.Series(False, index=data.index)

        # 進場訊號
        # 做多：RSI 進入超賣區（< oversold）
        long_entry = (rsi < oversold) & (rsi.shift(1) >= oversold)

        # 做空：RSI 進入超買區（> overbought）
        short_entry = (rsi > overbought) & (rsi.shift(1) <= overbought)

        # 出場訊號
        # 多單出場：RSI 回到中性區（> exit_threshold）
        long_exit = (rsi > exit_threshold) & (rsi.shift(1) <= exit_threshold)

        # 空單出場：RSI 回到中性區（< exit_threshold）
        short_exit = (rsi < exit_threshold) & (rsi.shift(1) >= exit_threshold)

        return long_entry, long_exit, short_entry, short_exit

    def _calculate_rsi(
        self,
        close: Series,
        period: int = 14
    ) -> Series:
        """
        計算 RSI 指標

        Args:
            close: 收盤價 Series
            period: RSI 週期

        Returns:
            Series: RSI 值 (0-100)
        """
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

        # 避免除以零
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi

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

        rsi = indicators['rsi'].iloc[index]
        oversold = self.params['oversold']
        overbought = self.params['overbought']

        if rsi < oversold:
            return f"超賣進場：RSI({rsi:.2f}) < {oversold}"
        elif rsi > overbought:
            return f"超買進場：RSI({rsi:.2f}) > {overbought}"
        else:
            return f"RSI: {rsi:.2f}"
