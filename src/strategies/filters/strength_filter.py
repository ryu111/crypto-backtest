"""信號強度過濾器 - 只保留強信號"""

from typing import Tuple
import pandas as pd
import numpy as np
from .base_filter import BaseSignalFilter


class SignalStrengthFilter(BaseSignalFilter):
    """信號強度過濾器 - 只保留強信號

    過濾條件：
    - RSI 必須距離超買/超賣區足夠遠
    - MACD 強度必須足夠大
    - 價格移動幅度必須足夠大
    """

    name = "strength_filter"
    priority = 10

    def __init__(
        self,
        min_rsi_distance: float = 5.0,
        min_macd_strength: float = 0.002,
        min_price_move: float = 0.005
    ):
        """初始化強度過濾器

        Args:
            min_rsi_distance: RSI 距離超買/超賣區的最小距離
            min_macd_strength: MACD 的最小強度（相對於價格）
            min_price_move: 最小價格移動幅度（百分比）
        """
        self.min_rsi_distance = min_rsi_distance
        self.min_macd_strength = min_macd_strength
        self.min_price_move = min_price_move

    def filter(
        self,
        data: pd.DataFrame,
        long_entry: pd.Series,
        short_entry: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """過濾弱信號，只保留強信號"""

        # 檢查必要的欄位
        has_rsi = 'rsi' in data.columns
        has_macd = 'macd' in data.columns
        has_close = 'close' in data.columns

        # 初始化過濾條件（全部為 True）
        long_filter = pd.Series(True, index=data.index)
        short_filter = pd.Series(True, index=data.index)

        # RSI 強度過濾
        if has_rsi:
            rsi = data['rsi']
            # 做多：RSI 必須遠離超買區（< 70 - min_distance）
            long_filter &= rsi < (70 - self.min_rsi_distance)
            # 做空：RSI 必須遠離超賣區（> 30 + min_distance）
            short_filter &= rsi > (30 + self.min_rsi_distance)

        # MACD 強度過濾
        if has_macd and has_close:
            macd = data['macd']
            close = data['close']
            macd_strength = abs(macd / close)

            # 做多：MACD 必須足夠正
            long_filter &= (macd > 0) & (macd_strength >= self.min_macd_strength)
            # 做空：MACD 必須足夠負
            short_filter &= (macd < 0) & (macd_strength >= self.min_macd_strength)

        # 價格移動幅度過濾
        if has_close:
            close = data['close']
            # 計算過去 3 根 K 棒的價格變化
            price_change = close.pct_change(periods=3).abs()

            # 必須有足夠的價格移動
            long_filter &= price_change >= self.min_price_move
            short_filter &= price_change >= self.min_price_move

        # 應用過濾條件
        filtered_long = long_entry & long_filter
        filtered_short = short_entry & short_filter

        return filtered_long, filtered_short

    def should_apply(self, data: pd.DataFrame) -> bool:
        """檢查是否有足夠的數據來應用此過濾器"""
        return (
            'rsi' in data.columns or
            'macd' in data.columns or
            'close' in data.columns
        )
