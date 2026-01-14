"""成交量過濾器 - 只在高成交量時交易"""

from typing import Tuple
import pandas as pd
import numpy as np
from .base_filter import BaseSignalFilter


class VolumeFilter(BaseSignalFilter):
    """成交量過濾器 - 只在高成交量時交易

    過濾條件：
    - 成交量必須高於平均成交量的 N 倍
    - 可選：成交量必須持續放大
    """

    name = "volume_filter"
    priority = 15

    def __init__(
        self,
        volume_multiplier: float = 1.5,
        lookback: int = 20,
        require_increasing_volume: bool = False,
        increasing_periods: int = 2
    ):
        """初始化成交量過濾器

        Args:
            volume_multiplier: 成交量倍數（相對於平均）
            lookback: 計算平均成交量的回顧期
            require_increasing_volume: 是否要求成交量持續放大
            increasing_periods: 成交量持續放大的週期數
        """
        self.volume_multiplier = volume_multiplier
        self.lookback = lookback
        self.require_increasing_volume = require_increasing_volume
        self.increasing_periods = increasing_periods

    def filter(
        self,
        data: pd.DataFrame,
        long_entry: pd.Series,
        short_entry: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """過濾低成交量時段的信號"""

        # 檢查是否有成交量數據
        if 'volume' not in data.columns:
            return long_entry, short_entry

        volume = data['volume']

        # 計算平均成交量
        avg_volume = volume.rolling(window=self.lookback).mean()

        # 基本過濾：成交量必須高於平均的 N 倍
        volume_filter = volume >= (avg_volume * self.volume_multiplier)

        # 可選：要求成交量持續放大
        if self.require_increasing_volume:
            volume_increasing = pd.Series(True, index=data.index)

            for i in range(1, self.increasing_periods + 1):
                volume_increasing &= volume > volume.shift(i)

            volume_filter &= volume_increasing

        # 應用成交量過濾（對多空都一樣）
        filtered_long = long_entry & volume_filter
        filtered_short = short_entry & volume_filter

        return filtered_long, filtered_short

    def should_apply(self, data: pd.DataFrame) -> bool:
        """檢查是否有成交量數據"""
        return 'volume' in data.columns and len(data) >= self.lookback

    def get_volume_stats(self, data: pd.DataFrame) -> dict:
        """獲取成交量統計資訊

        Args:
            data: OHLCV 數據

        Returns:
            成交量統計字典
        """
        if 'volume' not in data.columns:
            return {}

        volume = data['volume']
        avg_volume = volume.rolling(window=self.lookback).mean()

        # 安全獲取最後一個值
        try:
            current_vol = float(np.array(volume)[-1])
        except (IndexError, TypeError):
            current_vol = 0.0

        try:
            avg_vol = float(np.array(avg_volume)[-1])
        except (IndexError, TypeError):
            avg_vol = 0.0

        return {
            'avg_volume': float(avg_volume.mean()),
            'current_volume': current_vol if pd.notna(current_vol) else 0.0,
            'volume_multiplier': current_vol / avg_vol if pd.notna(avg_vol) and avg_vol > 0 else 0.0,
            'high_volume_periods': int((volume >= avg_volume * self.volume_multiplier).sum()),
            'total_periods': len(volume),
        }
