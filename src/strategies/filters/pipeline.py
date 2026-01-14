"""信號過濾管道"""

from typing import List, Dict, Tuple, Optional
import pandas as pd
import logging
from .base_filter import BaseSignalFilter
from .strength_filter import SignalStrengthFilter
from .confirmation_filter import ConfirmationFilter
from .time_filter import TimeFilter
from .volume_filter import VolumeFilter

logger = logging.getLogger(__name__)


class FilterPipeline:
    """信號過濾管道

    將多個過濾器串聯起來，依序過濾信號。
    只過濾進場信號，不影響出場信號。
    """

    def __init__(self, filters: Optional[List[BaseSignalFilter]] = None):
        """初始化過濾管道

        Args:
            filters: 過濾器列表，如果為 None 則建立空管道
        """
        self.filters = sorted(filters or [], key=lambda f: f.priority)
        self._stats = {}
        self._last_data_length = 0

    def add_filter(self, filter: BaseSignalFilter) -> 'FilterPipeline':
        """新增過濾器

        Args:
            filter: 要新增的過濾器

        Returns:
            self（支援鏈式調用）
        """
        self.filters.append(filter)
        self.filters.sort(key=lambda f: f.priority)
        return self

    def process(
        self,
        data: pd.DataFrame,
        long_entry: pd.Series,
        long_exit: pd.Series,
        short_entry: pd.Series,
        short_exit: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """執行過濾管道（只過濾進場，保留出場）

        Args:
            data: OHLCV 數據及指標
            long_entry: 原始做多進場信號
            long_exit: 做多出場信號（不會被過濾）
            short_entry: 原始做空進場信號
            short_exit: 做空出場信號（不會被過濾）

        Returns:
            (filtered_long_entry, long_exit, filtered_short_entry, short_exit)
        """
        self._stats = {}
        self._last_data_length = len(data)

        # 記錄原始信號數量
        original_long_count = long_entry.sum()
        original_short_count = short_entry.sum()

        logger.info(f"開始過濾管道，共 {len(self.filters)} 個過濾器")
        logger.info(f"原始信號：做多 {original_long_count}，做空 {original_short_count}")

        # 依序應用每個過濾器
        current_long = long_entry.copy()
        current_short = short_entry.copy()

        for i, filter in enumerate(self.filters, 1):
            # 檢查是否應該應用此過濾器
            if not filter.should_apply(data):
                logger.warning(f"過濾器 {filter.name} 不適用，跳過")
                continue

            # 記錄過濾前的狀態
            before_long = current_long.copy()
            before_short = current_short.copy()

            # 應用過濾器
            try:
                current_long, current_short = filter.filter(data, current_long, current_short)

                # 計算並記錄統計資訊
                stats = filter.get_filter_stats(
                    before_long, before_short,
                    current_long, current_short
                )
                self._stats[filter.name] = stats

                logger.info(
                    f"[{i}/{len(self.filters)}] {filter.name}: "
                    f"做多 {stats['original_long_signals']} → {stats['filtered_long_signals']} "
                    f"({stats['long_reduction_rate']:.1%} 減少), "
                    f"做空 {stats['original_short_signals']} → {stats['filtered_short_signals']} "
                    f"({stats['short_reduction_rate']:.1%} 減少)"
                )

            except Exception as e:
                logger.error(f"過濾器 {filter.name} 執行失敗: {e}")
                # 發生錯誤時保持原樣
                continue

        # 記錄最終結果
        final_long_count = current_long.sum()
        final_short_count = current_short.sum()

        total_long_reduction = (
            (original_long_count - final_long_count) / original_long_count
            if original_long_count > 0 else 0.0
        )
        total_short_reduction = (
            (original_short_count - final_short_count) / original_short_count
            if original_short_count > 0 else 0.0
        )

        logger.info(
            f"過濾完成：做多 {original_long_count} → {final_long_count} ({total_long_reduction:.1%} 減少), "
            f"做空 {original_short_count} → {final_short_count} ({total_short_reduction:.1%} 減少)"
        )

        # 儲存總體統計
        self._stats['_total'] = {
            'original_long_signals': int(original_long_count),
            'original_short_signals': int(original_short_count),
            'final_long_signals': int(final_long_count),
            'final_short_signals': int(final_short_count),
            'total_long_reduction_rate': float(total_long_reduction),
            'total_short_reduction_rate': float(total_short_reduction),
        }

        return current_long, long_exit, current_short, short_exit

    def get_stats(self) -> Dict:
        """獲取過濾統計

        Returns:
            包含所有過濾器統計的字典
        """
        return self._stats.copy()

    def get_summary(self) -> str:
        """獲取過濾統計摘要

        Returns:
            統計摘要字串
        """
        if not self._stats:
            return "尚未執行過濾"

        total = self._stats.get('_total', {})

        lines = [
            "=" * 60,
            "過濾管道統計摘要",
            "=" * 60,
            f"數據長度: {self._last_data_length}",
            f"過濾器數量: {len(self.filters)}",
            "",
            "原始信號:",
            f"  做多: {total.get('original_long_signals', 0)}",
            f"  做空: {total.get('original_short_signals', 0)}",
            "",
            "最終信號:",
            f"  做多: {total.get('final_long_signals', 0)} ({total.get('total_long_reduction_rate', 0):.1%} 減少)",
            f"  做空: {total.get('final_short_signals', 0)} ({total.get('total_short_reduction_rate', 0):.1%} 減少)",
            "",
            "各過濾器統計:",
        ]

        for filter in self.filters:
            if filter.name in self._stats:
                stats = self._stats[filter.name]
                lines.append(f"  {filter.name}:")
                lines.append(f"    做多: {stats['original_long_signals']} → {stats['filtered_long_signals']} ({stats['long_reduction_rate']:.1%})")
                lines.append(f"    做空: {stats['original_short_signals']} → {stats['filtered_short_signals']} ({stats['short_reduction_rate']:.1%})")

        lines.append("=" * 60)

        return "\n".join(lines)

    @classmethod
    def create_default(
        cls,
        strength_enabled: bool = True,
        confirmation_enabled: bool = True,
        time_enabled: bool = True,
        volume_enabled: bool = False
    ) -> 'FilterPipeline':
        """建立預設過濾管道

        Args:
            strength_enabled: 是否啟用強度過濾器
            confirmation_enabled: 是否啟用確認過濾器
            time_enabled: 是否啟用時間過濾器
            volume_enabled: 是否啟用成交量過濾器

        Returns:
            配置好的 FilterPipeline
        """
        filters = []

        if time_enabled:
            filters.append(TimeFilter(
                avoid_funding_hours=[0, 8, 16],
                funding_buffer_minutes=30,
                avoid_weekend=False
            ))

        if strength_enabled:
            filters.append(SignalStrengthFilter(
                min_rsi_distance=5.0,
                min_macd_strength=0.002,
                min_price_move=0.005
            ))

        if volume_enabled:
            filters.append(VolumeFilter(
                volume_multiplier=1.5,
                lookback=20,
                require_increasing_volume=False
            ))

        if confirmation_enabled:
            filters.append(ConfirmationFilter(
                require_trend_alignment=True,
                require_volume_confirm=True,
                min_confirmations=2,
                ma_period=20
            ))

        logger.info(f"建立預設過濾管道，包含 {len(filters)} 個過濾器")

        return cls(filters=filters)

    @classmethod
    def create_aggressive(cls) -> 'FilterPipeline':
        """建立激進模式（過濾較少）"""
        return cls.create_default(
            strength_enabled=False,
            confirmation_enabled=False,
            time_enabled=True,
            volume_enabled=False
        )

    @classmethod
    def create_conservative(cls) -> 'FilterPipeline':
        """建立保守模式（過濾較多）"""
        filters = [
            TimeFilter(
                avoid_funding_hours=[0, 8, 16],
                funding_buffer_minutes=60,  # 更長的緩衝時間
                avoid_weekend=True
            ),
            SignalStrengthFilter(
                min_rsi_distance=10.0,  # 更嚴格的 RSI 要求
                min_macd_strength=0.005,  # 更高的 MACD 強度
                min_price_move=0.01  # 更大的價格移動
            ),
            VolumeFilter(
                volume_multiplier=2.0,  # 更高的成交量要求
                lookback=20,
                require_increasing_volume=True
            ),
            ConfirmationFilter(
                require_trend_alignment=True,
                require_volume_confirm=True,
                min_confirmations=3,  # 要求更多確認
                ma_period=20
            )
        ]

        logger.info("建立保守過濾管道，包含更嚴格的過濾條件")

        return cls(filters=filters)
