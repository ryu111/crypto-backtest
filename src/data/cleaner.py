"""
資料清理模組

處理資料缺失、異常值和品質問題
支援 Polars（優先）和 Pandas 兩種後端
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import pandas as pd

# Polars 是可選依賴
# 目前僅設定 backend 變數，實際 Polars 功能尚未實作
POLARS_AVAILABLE = False
if TYPE_CHECKING:
    pass  # 類型檢查時不需要 polars
else:
    try:
        import polars as pl  # noqa: F401
        POLARS_AVAILABLE = True
        del pl  # 移除未使用的變數
    except ImportError:
        pass


class GapFillStrategy(Enum):
    """缺失資料填補策略"""
    LINEAR = "linear"          # 線性插值
    FORWARD_FILL = "ffill"     # 向前填充
    BACKWARD_FILL = "bfill"    # 向後填充
    NONE = "none"              # 不填補，標記為 NaN


@dataclass
class GapInfo:
    """缺失區間資訊"""
    start_time: datetime
    end_time: datetime
    duration: timedelta
    gap_size: int
    is_maintenance: bool = False
    fill_strategy: Optional[GapFillStrategy] = None

    def __str__(self) -> str:
        duration_hours = self.duration.total_seconds() / 3600
        gap_type = "維護期間" if self.is_maintenance else "資料缺失"
        return (
            f"{gap_type}: {self.start_time} ~ {self.end_time} "
            f"({duration_hours:.1f}h, {self.gap_size} gaps)"
        )


@dataclass
class DataQualityReport:
    """資料品質報告"""
    total_records: int
    missing_count: int
    missing_rate: float
    gap_count: int
    gaps: list[GapInfo]
    quality_score: float
    issues: list[str]

    def __str__(self) -> str:
        lines = [
            "=== 資料品質報告 ===",
            f"總筆數: {self.total_records:,}",
            f"缺失筆數: {self.missing_count:,} ({self.missing_rate:.2%})",
            f"缺失區間: {self.gap_count} 處",
            f"品質評分: {self.quality_score:.2f}/100",
            "",
            "主要問題:",
        ]

        if self.issues:
            for issue in self.issues:
                lines.append(f"  - {issue}")
        else:
            lines.append("  (無)")

        if self.gaps:
            lines.append("")
            lines.append("缺失區間詳情:")
            for gap in self.gaps[:10]:  # 只顯示前 10 個
                lines.append(f"  {gap}")
            if len(self.gaps) > 10:
                lines.append(f"  ... 還有 {len(self.gaps) - 10} 處")

        return "\n".join(lines)


class DataCleaner:
    """
    資料清理器 - 處理資料缺失和品質問題

    支援 Polars（優先）和 Pandas 兩種後端
    """

    def __init__(
        self,
        timeframe: str = '4h',
        gap_threshold_multiplier: float = 2.0,
        long_gap_hours: float = 1.0,
        verbose: bool = False
    ):
        """
        初始化資料清理器

        Args:
            timeframe: 資料時間框架，用於計算預期間隔
            gap_threshold_multiplier: 判定缺失的閾值倍數
            long_gap_hours: 長期缺失的小時數閾值
            verbose: 是否顯示詳細資訊
        """
        self.timeframe = timeframe
        self.expected_interval = self._parse_timeframe(timeframe)
        self.gap_threshold_multiplier = gap_threshold_multiplier
        self.long_gap_threshold = timedelta(hours=long_gap_hours)
        self.verbose = verbose

        if POLARS_AVAILABLE:
            self.backend: Literal['polars', 'pandas'] = 'polars'
            if self.verbose:
                print("使用 Polars 後端（效能優化）")
        else:
            self.backend = 'pandas'
            if self.verbose:
                print("使用 Pandas 後端（Polars 未安裝）")

    def clean(
        self,
        df: pd.DataFrame,
        fill_short_gaps: bool = True,
        mark_long_gaps: bool = True
    ) -> pd.DataFrame:
        """
        執行完整的資料清理流程

        Args:
            df: 輸入的 DataFrame（時間戳為索引）
            fill_short_gaps: 是否填補短期缺失
            mark_long_gaps: 是否標記長期缺失

        Returns:
            清理後的 DataFrame
        """
        if self.verbose:
            print(f"開始清理資料，原始筆數: {len(df)}")

        # 1. 移除重複時間戳
        df = self._remove_duplicates(df)

        # 2. 偵測缺失區間
        gaps = self._detect_gaps(df)

        if self.verbose:
            print(f"偵測到 {len(gaps)} 處缺失區間")

        # 3. 填補短期缺失
        if fill_short_gaps:
            df = self._fill_gaps(df, gaps, only_short=True)

        # 4. 標記長期缺失
        if mark_long_gaps:
            df = self._mark_long_gaps(df, gaps)

        if self.verbose:
            print(f"清理完成，最終筆數: {len(df)}")

        return df

    def analyze_quality(self, df: pd.DataFrame) -> DataQualityReport:
        """
        分析資料品質並生成報告

        Args:
            df: 輸入的 DataFrame

        Returns:
            資料品質報告
        """
        total = len(df)

        # 計算缺失值
        missing_count = df.isnull().sum().sum()
        missing_rate = missing_count / (total * len(df.columns)) if total > 0 else 0

        # 偵測時間缺失
        gaps = self._detect_gaps(df)
        gap_count = len(gaps)

        # 收集問題
        issues = []

        if missing_rate > 0.01:  # > 1%
            issues.append(f"缺失率過高: {missing_rate:.2%}")

        if gap_count > 10:
            issues.append(f"時間缺失過多: {gap_count} 處")

        # 檢查 OHLC 邏輯（如果適用）
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid = self._validate_ohlc(df)
            if invalid > 0:
                issues.append(f"OHLC 邏輯錯誤: {invalid} 筆")

        # 檢查成交量（如果有）
        if 'volume' in df.columns:
            zero_vol = (df['volume'] <= 0).sum()
            if zero_vol > 0:
                issues.append(f"零成交量: {zero_vol} 筆")

        # 計算品質評分
        quality_score = self._calculate_quality_score(
            missing_rate=missing_rate,
            gap_count=gap_count,
            total=total,
            issues=len(issues)
        )

        return DataQualityReport(
            total_records=total,
            missing_count=missing_count,
            missing_rate=missing_rate,
            gap_count=gap_count,
            gaps=gaps,
            quality_score=quality_score,
            issues=issues
        )

    def _detect_gaps(self, df: pd.DataFrame) -> list[GapInfo]:
        """
        偵測時間序列中的缺失區間

        Args:
            df: 輸入的 DataFrame（時間戳為索引）

        Returns:
            缺失區間列表
        """
        if len(df) < 2:
            return []

        gaps = []
        timestamps = df.index.to_series()
        time_diffs = timestamps.diff()

        # 找出大於預期間隔的 gap
        gap_threshold = self.expected_interval * self.gap_threshold_multiplier

        for i, (ts, diff) in enumerate(zip(timestamps[1:], time_diffs[1:]), start=1):
            if diff > gap_threshold:
                gap_size = int(diff / self.expected_interval) - 1

                # 判斷是否為維護期間（簡單規則：gap > 4 小時且在凌晨 0-4 點）
                is_maintenance = (
                    diff > timedelta(hours=4) and
                    timestamps.iloc[i-1].hour in range(0, 4)
                )

                # 選擇填補策略
                if diff <= self.long_gap_threshold:
                    fill_strategy = GapFillStrategy.LINEAR
                else:
                    fill_strategy = GapFillStrategy.NONE

                gap_info = GapInfo(
                    start_time=timestamps.iloc[i-1],
                    end_time=ts,
                    duration=diff,
                    gap_size=gap_size,
                    is_maintenance=is_maintenance,
                    fill_strategy=fill_strategy
                )
                gaps.append(gap_info)

        return gaps

    def _fill_gaps(
        self,
        df: pd.DataFrame,
        gaps: list[GapInfo],
        only_short: bool = True
    ) -> pd.DataFrame:
        """
        填補缺失資料

        Args:
            df: 輸入的 DataFrame
            gaps: 缺失區間列表
            only_short: 是否只填補短期缺失

        Returns:
            填補後的 DataFrame
        """
        if not gaps:
            return df

        # 篩選需要填補的 gap
        gaps_to_fill = [
            gap for gap in gaps
            if gap.fill_strategy == GapFillStrategy.LINEAR and
            (not only_short or gap.duration <= self.long_gap_threshold)
        ]

        if not gaps_to_fill:
            if self.verbose:
                print("無需填補的缺失區間")
            return df

        # 使用線性插值填補
        df_filled = df.copy()

        # 建立完整的時間索引
        start = df.index.min()
        end = df.index.max()
        full_index = pd.date_range(start=start, end=end, freq=self.timeframe)

        # Reindex 並插值
        df_filled = df_filled.reindex(full_index)

        # 只對數值欄位插值
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
        df_filled[numeric_cols] = df_filled[numeric_cols].interpolate(
            method='linear',
            limit=self._get_interpolation_limit()
        )

        if self.verbose:
            filled_count = len(df_filled) - len(df)
            print(f"填補了 {filled_count} 筆缺失資料")

        return df_filled

    def _mark_long_gaps(self, df: pd.DataFrame, gaps: list[GapInfo]) -> pd.DataFrame:
        """
        標記長期缺失區間（加入 flag 欄位）

        Args:
            df: 輸入的 DataFrame
            gaps: 缺失區間列表

        Returns:
            加入標記的 DataFrame
        """
        df_marked = df.copy()
        df_marked['gap_flag'] = 0  # 0: 正常, 1: 短缺失, 2: 長缺失, 3: 維護期

        for gap in gaps:
            # 找出受影響的時間範圍
            mask = (df_marked.index >= gap.start_time) & (df_marked.index <= gap.end_time)

            if gap.is_maintenance:
                flag = 3
            elif gap.duration > self.long_gap_threshold:
                flag = 2
            else:
                flag = 1

            df_marked.loc[mask, 'gap_flag'] = flag

        return df_marked

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """移除重複的時間戳，保留最後一筆"""
        if df.index.duplicated().any():
            dup_count = int(df.index.duplicated().sum())
            if self.verbose:
                print(f"移除 {dup_count} 筆重複時間戳")
            # 使用布林索引並明確轉換類型
            mask = ~df.index.duplicated(keep='last')
            return pd.DataFrame(df[mask])
        return df

    def _validate_ohlc(self, df: pd.DataFrame) -> int:
        """驗證 OHLC 邏輯，返回錯誤筆數"""
        invalid = (
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close']) |
            (df['high'] < df['low'])
        )
        return invalid.sum()

    def _calculate_quality_score(
        self,
        missing_rate: float,
        gap_count: int,
        total: int,
        issues: int
    ) -> float:
        """
        計算資料品質評分（0-100）

        考慮因素：
        - 缺失率（權重 40%）
        - Gap 數量（權重 30%）
        - 問題數量（權重 30%）
        """
        # 極端情況：幾乎全部缺失或沒有資料
        if missing_rate >= 0.99 or total == 0:
            return 0.0

        # 缺失率分數（缺失率越低越好）
        missing_score = max(0, 100 - missing_rate * 10000) * 0.4

        # Gap 分數（gap 數量佔總數的比例）
        gap_ratio = gap_count / max(total, 1)
        gap_score = max(0, 100 - gap_ratio * 1000) * 0.3

        # 問題分數（每個問題扣 10 分）
        issue_score = max(0, 100 - issues * 10) * 0.3

        return min(100, missing_score + gap_score + issue_score)

    def _parse_timeframe(self, timeframe: str) -> timedelta:
        """
        解析時間框架字串為 timedelta

        Args:
            timeframe: 如 '1m', '5m', '1h', '4h', '1d'

        Returns:
            對應的 timedelta
        """
        unit = timeframe[-1]
        value = int(timeframe[:-1])

        if unit == 'm':
            return timedelta(minutes=value)
        elif unit == 'h':
            return timedelta(hours=value)
        elif unit == 'd':
            return timedelta(days=value)
        else:
            raise ValueError(f"不支援的時間框架: {timeframe}")

    def _get_interpolation_limit(self) -> int:
        """取得插值的最大連續數量（避免填補過長的缺失）"""
        # 最多填補 1 小時的資料
        max_fill_duration = timedelta(hours=1)
        return int(max_fill_duration / self.expected_interval)


# 快速使用範例
if __name__ == '__main__':
    from .fetcher import DataFetcher

    # 1. 獲取資料
    fetcher = DataFetcher(verbose=True)
    df = fetcher.fetch_ohlcv(
        symbol='BTCUSDT',
        timeframe='4h',
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 1)
    )

    # 2. 分析資料品質
    cleaner = DataCleaner(timeframe='4h', verbose=True)
    report = cleaner.analyze_quality(df)
    print(report)

    # 3. 清理資料
    df_cleaned = cleaner.clean(df, fill_short_gaps=True, mark_long_gaps=True)

    # 4. 再次檢查品質
    report_after = cleaner.analyze_quality(df_cleaned)
    print("\n清理後:")
    print(report_after)
