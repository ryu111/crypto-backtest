"""
RSI 動量策略

基於 RSI 指標的超買超賣均值回歸策略。
當 RSI 超賣時做多，超買時做空。
可選擇性加入趨勢過濾器，只在趨勢方向交易。
"""

import pandas as pd
from typing import Dict, Tuple
from pandas import Series, DataFrame

from ..base import MomentumStrategy
from ..registry import register_strategy


@register_strategy('momentum_rsi')
class RSIStrategy(MomentumStrategy):
    """
    RSI 均值回歸策略

    策略邏輯：
    - 多單進場：RSI < oversold（超賣）
    - 多單出場：RSI > overbought 或 RSI 回到中線（50）
    - 空單進場：RSI > overbought（超買）
    - 空單出場：RSI < oversold 或 RSI 回到中線（50）
    - 可選趨勢過濾：只在趨勢方向交易

    Parameters:
        rsi_period (int): RSI 計算週期，預設 14
        oversold (float): 超賣閾值，預設 30
        overbought (float): 超買閾值，預設 70
        trend_filter (bool): 是否啟用趨勢過濾器，預設 True
        trend_period (int): 趨勢判斷均線週期，預設 200

    Example:
        >>> strategy = RSIStrategy(rsi_period=14, oversold=30, overbought=70)
        >>> long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(data)
    """

    name = "RSI Mean Reversion"
    strategy_type = "momentum"
    version = "1.0"
    description = "RSI-based mean reversion strategy with optional trend filter"

    # 預設參數
    params = {
        'rsi_period': 14,
        'oversold': 30,
        'overbought': 70,
        'trend_filter': True,
        'trend_period': 200,
    }

    # 參數優化空間
    param_space = {
        'rsi_period': {
            'type': 'int',
            'low': 7,
            'high': 28,
        },
        'oversold': {
            'type': 'int',
            'low': 20,
            'high': 40,
        },
        'overbought': {
            'type': 'int',
            'low': 60,
            'high': 80,
        },
        'trend_period': {
            'type': 'int',
            'low': 100,
            'high': 300,
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

        # RSI 週期必須 > 0
        if self.params['rsi_period'] <= 0:
            return False

        # 超賣必須 < 超買
        if self.params['oversold'] >= self.params['overbought']:
            return False

        # 超賣和超買必須在 0-100 範圍內
        if not (0 < self.params['oversold'] < 100):
            return False
        if not (0 < self.params['overbought'] < 100):
            return False

        # 趨勢週期必須 > 0
        if self.params['trend_period'] <= 0:
            return False

        return True

    def calculate_indicators(self, data: DataFrame) -> Dict[str, Series]:
        """
        計算 RSI 指標

        Args:
            data: OHLCV DataFrame

        Returns:
            dict: {
                'rsi': RSI 值,
                'trend_ma': 趨勢均線（如果啟用趨勢過濾）
            }
        """
        indicators = {}

        # 計算 RSI
        rsi = self.calculate_rsi(
            data['close'],
            period=self.params['rsi_period']
        )
        indicators['rsi'] = rsi

        # 如果啟用趨勢過濾，計算趨勢均線
        if self.params['trend_filter']:
            trend_ma = data['close'].rolling(self.params['trend_period']).mean()
            indicators['trend_ma'] = trend_ma

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
        close = data['close']

        # 初始化訊號
        long_entry = pd.Series(False, index=data.index)
        long_exit = pd.Series(False, index=data.index)
        short_entry = pd.Series(False, index=data.index)
        short_exit = pd.Series(False, index=data.index)

        # RSI 超賣超買條件
        rsi_oversold = rsi < self.params['oversold']
        rsi_overbought = rsi > self.params['overbought']
        rsi_neutral = (rsi >= 50) & (rsi.shift(1) < 50)  # RSI 回到中線以上
        rsi_neutral_down = (rsi <= 50) & (rsi.shift(1) > 50)  # RSI 回到中線以下

        # 趨勢過濾
        if self.params['trend_filter']:
            trend_ma = indicators['trend_ma']
            uptrend = close > trend_ma
            downtrend = close < trend_ma
        else:
            uptrend = pd.Series(True, index=data.index)
            downtrend = pd.Series(True, index=data.index)

        # 多單訊號：RSI 超賣且在上升趨勢（或無趨勢過濾）
        long_entry = rsi_oversold & uptrend

        # 多單出場：RSI 超買或回到中線
        long_exit = rsi_overbought | rsi_neutral

        # 空單訊號：RSI 超買且在下降趨勢（或無趨勢過濾）
        short_entry = rsi_overbought & downtrend

        # 空單出場：RSI 超賣或回到中線
        short_exit = rsi_oversold | rsi_neutral_down

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
        應用額外過濾器（可選覆寫）

        這裡可以加入額外的過濾條件，例如：
        - 成交量過濾
        - 波動度過濾
        - 時間過濾等

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
        return long_entry, long_exit, short_entry, short_exit
