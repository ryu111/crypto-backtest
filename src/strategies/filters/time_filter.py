"""時間過濾器 - 避開特定時段"""

from typing import List, Tuple, Optional
import pandas as pd
from datetime import timedelta
from .base_filter import BaseSignalFilter


class TimeFilter(BaseSignalFilter):
    """時間過濾器 - 避開特定時段

    過濾條件：
    - 避開資金費率結算時段
    - 避開週末（可選）
    - 避開自定義時段
    """

    name = "time_filter"
    priority = 5  # 優先級最高，最先執行

    def __init__(
        self,
        avoid_funding_hours: Optional[List[int]] = None,
        funding_buffer_minutes: int = 30,
        avoid_weekend: bool = False,
        avoid_custom_hours: Optional[List[Tuple[int, int]]] = None
    ):
        """初始化時間過濾器

        Args:
            avoid_funding_hours: 要避開的資金費率結算小時（UTC）
            funding_buffer_minutes: 資金費率結算前後的緩衝時間（分鐘）
            avoid_weekend: 是否避開週末
            avoid_custom_hours: 自定義避開時段，格式 [(start_hour, end_hour), ...]
        """
        self.avoid_funding_hours = avoid_funding_hours or [0, 8, 16]
        self.funding_buffer_minutes = funding_buffer_minutes
        self.avoid_weekend = avoid_weekend
        self.avoid_custom_hours = avoid_custom_hours or []

    def filter(
        self,
        data: pd.DataFrame,
        long_entry: pd.Series,
        short_entry: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """過濾特定時段的信號"""

        # 確保 index 是 DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            # 如果有 timestamp 欄位，嘗試使用它
            if 'timestamp' in data.columns:
                timestamps = pd.to_datetime(data['timestamp'])
            else:
                # 無法判斷時間，不過濾
                return long_entry, short_entry
        else:
            timestamps = data.index

        # 初始化允許交易的時段（全部為 True）
        allowed_time = pd.Series(True, index=data.index)

        # 1. 過濾資金費率結算時段
        for funding_hour in self.avoid_funding_hours:
            # 計算緩衝時間範圍
            buffer_td = timedelta(minutes=self.funding_buffer_minutes)

            for i, ts in enumerate(timestamps):
                # 檢查是否在資金費率結算時段前後的緩衝區內
                funding_time = ts.replace(hour=funding_hour, minute=0, second=0, microsecond=0)

                # 如果在緩衝區內，禁止交易
                if abs(ts - funding_time) <= buffer_td:
                    allowed_time.iloc[i] = False

        # 2. 過濾週末
        if self.avoid_weekend:
            # 週六 = 5, 週日 = 6
            if isinstance(timestamps, pd.DatetimeIndex):
                # 轉換為 Series 使用 .dt 訪問器
                ts_series = pd.Series(timestamps, index=data.index)
                is_weekend = ts_series.dt.dayofweek.isin([5, 6])
            else:
                is_weekend = timestamps.dt.dayofweek.isin([5, 6])
            allowed_time &= ~is_weekend

        # 3. 過濾自定義時段
        for start_hour, end_hour in self.avoid_custom_hours:
            if isinstance(timestamps, pd.DatetimeIndex):
                # 轉換為 Series 使用 .dt 訪問器
                ts_series = pd.Series(timestamps, index=data.index)
                hour = ts_series.dt.hour
            else:
                hour = timestamps.dt.hour
            if start_hour <= end_hour:
                # 同一天內的時段
                in_custom_period = (hour >= start_hour) & (hour < end_hour)
            else:
                # 跨日的時段
                in_custom_period = (hour >= start_hour) | (hour < end_hour)

            allowed_time &= ~in_custom_period

        # 應用時間過濾
        filtered_long = long_entry & allowed_time
        filtered_short = short_entry & allowed_time

        return filtered_long, filtered_short

    def should_apply(self, data: pd.DataFrame) -> bool:
        """檢查是否有時間資訊"""
        return (
            isinstance(data.index, pd.DatetimeIndex) or
            'timestamp' in data.columns
        )
