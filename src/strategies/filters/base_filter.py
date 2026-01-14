"""信號過濾器基礎類別"""

from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd


class BaseSignalFilter(ABC):
    """信號過濾器基礎類別

    所有過濾器都應該繼承這個類別並實作 filter 方法。
    過濾器只影響進場信號，不影響出場信號。
    """

    name: str = "base_filter"
    priority: int = 100  # 優先級（數字越小越先執行）

    @abstractmethod
    def filter(
        self,
        data: pd.DataFrame,
        long_entry: pd.Series,
        short_entry: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """過濾信號（只過濾進場，不影響出場）

        Args:
            data: OHLCV 數據及指標
            long_entry: 原始做多進場信號
            short_entry: 原始做空進場信號

        Returns:
            過濾後的 (long_entry, short_entry)
        """
        pass

    def should_apply(self, data: pd.DataFrame) -> bool:
        """判斷是否應該應用此過濾器

        Args:
            data: OHLCV 數據

        Returns:
            是否應該應用此過濾器
        """
        return True

    def get_filter_stats(
        self,
        original_long: pd.Series,
        original_short: pd.Series,
        filtered_long: pd.Series,
        filtered_short: pd.Series
    ) -> dict:
        """計算過濾統計

        Args:
            original_long: 原始做多信號
            original_short: 原始做空信號
            filtered_long: 過濾後做多信號
            filtered_short: 過濾後做空信號

        Returns:
            包含過濾統計的字典
        """
        original_long_count = original_long.sum() if isinstance(original_long.sum(), (int, float)) else 0
        original_short_count = original_short.sum() if isinstance(original_short.sum(), (int, float)) else 0
        filtered_long_count = filtered_long.sum() if isinstance(filtered_long.sum(), (int, float)) else 0
        filtered_short_count = filtered_short.sum() if isinstance(filtered_short.sum(), (int, float)) else 0

        return {
            'filter_name': self.name,
            'original_long_signals': int(original_long_count),
            'original_short_signals': int(original_short_count),
            'filtered_long_signals': int(filtered_long_count),
            'filtered_short_signals': int(filtered_short_count),
            'long_reduction_rate': (
                (original_long_count - filtered_long_count) / original_long_count
                if original_long_count > 0 else 0.0
            ),
            'short_reduction_rate': (
                (original_short_count - filtered_short_count) / original_short_count
                if original_short_count > 0 else 0.0
            ),
        }
