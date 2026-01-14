"""確認過濾器 - 要求多指標確認"""

from typing import Tuple
import pandas as pd
import numpy as np
from .base_filter import BaseSignalFilter


class ConfirmationFilter(BaseSignalFilter):
    """確認過濾器 - 要求多指標確認

    過濾條件：
    - 趨勢對齊：價格趨勢與信號方向一致
    - 成交量確認：成交量高於平均
    - 最小確認數量：至少 N 個指標同時確認
    """

    name = "confirmation_filter"
    priority = 20

    def __init__(
        self,
        require_trend_alignment: bool = True,
        require_volume_confirm: bool = True,
        min_confirmations: int = 2,
        ma_period: int = 20
    ):
        """初始化確認過濾器

        Args:
            require_trend_alignment: 是否要求趨勢對齊
            require_volume_confirm: 是否要求成交量確認
            min_confirmations: 最小確認數量
            ma_period: 移動平均週期（用於趨勢判斷）
        """
        self.require_trend_alignment = require_trend_alignment
        self.require_volume_confirm = require_volume_confirm
        self.min_confirmations = min_confirmations
        self.ma_period = ma_period

    def filter(
        self,
        data: pd.DataFrame,
        long_entry: pd.Series,
        short_entry: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """過濾未確認的信號"""

        confirmations_long = pd.Series(0, index=data.index)
        confirmations_short = pd.Series(0, index=data.index)

        # 1. 趨勢確認
        if self.require_trend_alignment and 'close' in data.columns:
            close = data['close']
            ma_raw = close.rolling(window=self.ma_period).mean()
            # 確保 ma 是 Series 類型以使用 shift
            ma = pd.Series(ma_raw, index=data.index)

            # 做多確認：價格在均線之上且均線上升
            trend_up = (close > ma) & (ma > ma.shift(1))
            confirmations_long += trend_up.astype(int)

            # 做空確認：價格在均線之下且均線下降
            trend_down = (close < ma) & (ma < ma.shift(1))
            confirmations_short += trend_down.astype(int)

        # 2. 成交量確認
        if self.require_volume_confirm and 'volume' in data.columns:
            volume = data['volume']
            avg_volume = volume.rolling(window=self.ma_period).mean()

            # 成交量高於平均
            volume_confirm = volume > avg_volume
            confirmations_long += volume_confirm.astype(int)
            confirmations_short += volume_confirm.astype(int)

        # 3. RSI 確認
        if 'rsi' in data.columns:
            rsi = data['rsi']

            # 做多確認：RSI 上升且在 30-70 之間
            rsi_long = (rsi > rsi.shift(1)) & (rsi > 30) & (rsi < 70)
            confirmations_long += rsi_long.astype(int)

            # 做空確認：RSI 下降且在 30-70 之間
            rsi_short = (rsi < rsi.shift(1)) & (rsi > 30) & (rsi < 70)
            confirmations_short += rsi_short.astype(int)

        # 4. MACD 確認
        if 'macd' in data.columns and 'macd_signal' in data.columns:
            macd = data['macd']
            signal = data['macd_signal']

            # 做多確認：MACD 在信號線之上
            macd_long = macd > signal
            confirmations_long += macd_long.astype(int)

            # 做空確認：MACD 在信號線之下
            macd_short = macd < signal
            confirmations_short += macd_short.astype(int)

        # 應用最小確認數量過濾
        filtered_long = long_entry & (confirmations_long >= self.min_confirmations)
        filtered_short = short_entry & (confirmations_short >= self.min_confirmations)

        return filtered_long, filtered_short

    def should_apply(self, data: pd.DataFrame) -> bool:
        """檢查是否有足夠的數據來應用此過濾器"""
        # 至少需要價格數據
        return 'close' in data.columns
