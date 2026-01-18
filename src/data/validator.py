"""
資料驗證模組

在回測前驗證資料品質，避免因資料問題導致錯誤的回測結果。

參考：
- .claude/skills/資料管道/SKILL.md

使用範例：
    validator = DataValidator()

    # 回測前驗證
    issues = validator.validate(data)

    if validator.has_fatal_issues(issues):
        raise ValueError("資料有嚴重問題，無法執行回測")

    # 自動修復
    if validator.has_fixable_issues(issues):
        data = validator.auto_fix(data, issues)
"""

from dataclasses import dataclass
from typing import List, Optional, Any, Literal
from enum import Enum
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class IssueLevel(Enum):
    """問題嚴重等級"""
    FATAL = "fatal"       # 致命：阻止回測
    WARNING = "warning"   # 警告：記錄並可修復
    INFO = "info"         # 資訊：僅記錄


@dataclass
class DataIssue:
    """資料問題記錄"""
    level: IssueLevel
    issue_type: str
    count: int
    details: Any
    fixable: bool = False

    @property
    def message(self) -> str:
        """問題訊息"""
        level_emoji = {
            IssueLevel.FATAL: "❌",
            IssueLevel.WARNING: "⚠️",
            IssueLevel.INFO: "ℹ️"
        }
        emoji = level_emoji.get(self.level, "")
        fix_hint = " (可自動修復)" if self.fixable else ""
        return f"{emoji} [{self.level.value.upper()}] {self.issue_type}: {self.count} 筆{fix_hint}"


class DataValidator:
    """
    資料驗證器

    驗證 OHLCV 資料品質，識別並修復常見問題。

    問題優先級：
    - FATAL：缺失值、OHLC 邏輯錯誤（阻止回測）
    - WARNING：時間間隙、重複時間戳（記錄並修復）
    - INFO：低成交量（僅記錄）
    """

    def __init__(
        self,
        allow_auto_fix: bool = True,
        low_volume_threshold: float = 0.1  # 低於平均的 10%
    ):
        """
        初始化驗證器

        Args:
            allow_auto_fix: 是否允許自動修復
            low_volume_threshold: 低成交量閾值（相對於平均）
        """
        self.allow_auto_fix = allow_auto_fix
        self.low_volume_threshold = low_volume_threshold

    def validate(self, df: pd.DataFrame) -> List[DataIssue]:
        """
        驗證資料品質

        Args:
            df: OHLCV DataFrame

        Returns:
            問題列表
        """
        issues = []

        # 1. 檢查必要欄位
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            issues.append(DataIssue(
                level=IssueLevel.FATAL,
                issue_type="missing_columns",
                count=len(missing_cols),
                details=list(missing_cols),
                fixable=False
            ))
            return issues  # 缺少必要欄位，無法繼續驗證

        # 2. 檢查缺失值（FATAL）
        null_counts = df[required_cols].isnull().sum()
        total_nulls = null_counts.sum()
        if total_nulls > 0:
            issues.append(DataIssue(
                level=IssueLevel.FATAL,
                issue_type="null_values",
                count=int(total_nulls),
                details=null_counts.to_dict(),
                fixable=True  # 可用前向填充修復
            ))

        # 3. 檢查 OHLC 邏輯（FATAL）
        ohlc_errors = self._check_ohlc_logic(df)
        if ohlc_errors > 0:
            issues.append(DataIssue(
                level=IssueLevel.FATAL,
                issue_type="invalid_ohlc_logic",
                count=ohlc_errors,
                details="H < max(O,C) 或 L > min(O,C)",
                fixable=True  # 可修正
            ))

        # 4. 檢查時間連續性（WARNING）
        time_gaps = self._check_time_continuity(df)
        if time_gaps > 0:
            issues.append(DataIssue(
                level=IssueLevel.WARNING,
                issue_type="time_gaps",
                count=time_gaps,
                details=self._find_gaps(df),
                fixable=True  # 可插補
            ))

        # 5. 檢查重複時間戳（WARNING）
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            issues.append(DataIssue(
                level=IssueLevel.WARNING,
                issue_type="duplicate_timestamps",
                count=int(duplicates),
                details=df.index[df.index.duplicated()].tolist()[:10],
                fixable=True  # 可移除
            ))

        # 6. 檢查成交量（INFO）
        zero_volume = (df['volume'] <= 0).sum()
        if zero_volume > 0:
            issues.append(DataIssue(
                level=IssueLevel.WARNING,
                issue_type="zero_or_negative_volume",
                count=int(zero_volume),
                details="成交量為零或負數",
                fixable=False
            ))

        # 7. 檢查低成交量（INFO）
        low_volume = self._check_low_volume(df)
        if low_volume > 0:
            issues.append(DataIssue(
                level=IssueLevel.INFO,
                issue_type="low_volume",
                count=low_volume,
                details=f"低於平均的 {self.low_volume_threshold*100}%",
                fixable=False
            ))

        return issues

    def _check_ohlc_logic(self, df: pd.DataFrame) -> int:
        """檢查 OHLC 邏輯"""
        invalid = (
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close']) |
            (df['high'] < df['low'])
        )
        return int(invalid.sum())

    def _check_time_continuity(self, df: pd.DataFrame) -> int:
        """檢查時間連續性"""
        if len(df) < 2:
            return 0

        time_diff = df.index.to_series().diff()
        expected_freq = time_diff.mode().iloc[0] if len(time_diff.mode()) > 0 else time_diff.median()

        # 允許 10% 的誤差
        gaps = time_diff > expected_freq * 1.5
        return int(gaps.sum())

    def _find_gaps(self, df: pd.DataFrame) -> List[dict]:
        """找出時間間隙"""
        if len(df) < 2:
            return []

        time_diff = df.index.to_series().diff()
        expected_freq = time_diff.mode().iloc[0] if len(time_diff.mode()) > 0 else time_diff.median()

        gaps = time_diff > expected_freq * 1.5
        gap_indices = df.index[gaps]

        return [
            {
                'start': str(df.index[i-1]) if i > 0 else None,
                'end': str(idx),
                'gap_size': str(time_diff.iloc[i])
            }
            for i, idx in enumerate(df.index) if idx in gap_indices
        ][:10]  # 最多顯示 10 個

    def _check_low_volume(self, df: pd.DataFrame) -> int:
        """檢查低成交量"""
        if 'volume' not in df.columns:
            return 0

        avg_volume = df['volume'].mean()
        threshold = avg_volume * self.low_volume_threshold
        low_volume = df['volume'] < threshold
        return int(low_volume.sum())

    def validate_before_backtest(self, df: pd.DataFrame) -> bool:
        """
        回測前強制驗證

        Args:
            df: OHLCV DataFrame

        Returns:
            是否可以執行回測

        Raises:
            ValueError: 如果有 FATAL 問題
        """
        issues = self.validate(df)

        # 記錄所有問題
        for issue in issues:
            if issue.level == IssueLevel.FATAL:
                logger.error(issue.message)
            elif issue.level == IssueLevel.WARNING:
                logger.warning(issue.message)
            else:
                logger.info(issue.message)

        # 檢查是否有 FATAL 問題
        fatal_issues = [i for i in issues if i.level == IssueLevel.FATAL]
        if fatal_issues:
            if not self.allow_auto_fix or not all(i.fixable for i in fatal_issues):
                raise ValueError(
                    f"資料有 {len(fatal_issues)} 個嚴重問題，無法執行回測：" +
                    ", ".join(i.issue_type for i in fatal_issues)
                )
            return False  # 需要修復

        return True

    def auto_fix(
        self,
        df: pd.DataFrame,
        issues: Optional[List[DataIssue]] = None
    ) -> pd.DataFrame:
        """
        自動修復資料問題

        Args:
            df: OHLCV DataFrame
            issues: 問題列表（可選，否則自動驗證）

        Returns:
            修復後的 DataFrame
        """
        if issues is None:
            issues = self.validate(df)

        df_fixed = df.copy()

        for issue in issues:
            if not issue.fixable:
                continue

            if issue.issue_type == "duplicate_timestamps":
                # 移除重複時間戳
                df_fixed = df_fixed[~df_fixed.index.duplicated(keep='first')]
                logger.info(f"已移除 {issue.count} 個重複時間戳")

            elif issue.issue_type == "null_values":
                # 前向填充缺失值
                df_fixed = df_fixed.ffill()
                # 如果開頭有 NaN，用後向填充
                df_fixed = df_fixed.bfill()
                logger.info(f"已填充 {issue.count} 個缺失值")

            elif issue.issue_type == "invalid_ohlc_logic":
                # 修正 OHLC 邏輯
                df_fixed['high'] = df_fixed[['open', 'high', 'close']].max(axis=1)
                df_fixed['low'] = df_fixed[['open', 'low', 'close']].min(axis=1)
                logger.info(f"已修正 {issue.count} 個 OHLC 邏輯錯誤")

            elif issue.issue_type == "time_gaps":
                # 插補缺失的時間戳
                df_fixed = self._fill_time_gaps(df_fixed)
                logger.info(f"已插補 {issue.count} 個時間間隙")

        # 確保排序
        df_fixed = df_fixed.sort_index()

        return df_fixed

    def _fill_time_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """填補時間間隙"""
        try:
            # 推斷頻率
            freq = pd.infer_freq(df.index)
            if freq is None:
                # 使用最常見的時間差
                time_diff = df.index.to_series().diff()
                freq = time_diff.mode().iloc[0]

            # 生成完整的時間索引
            full_index = pd.date_range(
                start=df.index.min(),
                end=df.index.max(),
                freq=freq
            )

            # 重新索引並前向填充
            df_filled = df.reindex(full_index, method='ffill')
            return df_filled

        except Exception as e:
            logger.warning(f"無法填補時間間隙: {e}")
            return df

    def has_fatal_issues(self, issues: List[DataIssue]) -> bool:
        """檢查是否有 FATAL 問題"""
        return any(i.level == IssueLevel.FATAL for i in issues)

    def has_fixable_issues(self, issues: List[DataIssue]) -> bool:
        """檢查是否有可修復的問題"""
        return any(i.fixable for i in issues)

    def summary(self, issues: List[DataIssue]) -> str:
        """生成問題摘要"""
        if not issues:
            return "✅ 資料品質良好，無問題"

        fatal = [i for i in issues if i.level == IssueLevel.FATAL]
        warning = [i for i in issues if i.level == IssueLevel.WARNING]
        info = [i for i in issues if i.level == IssueLevel.INFO]

        lines = ["📊 資料品質報告", "=" * 40]

        if fatal:
            lines.append(f"\n❌ 嚴重問題 ({len(fatal)})")
            for i in fatal:
                lines.append(f"  - {i.issue_type}: {i.count} 筆")

        if warning:
            lines.append(f"\n⚠️ 警告 ({len(warning)})")
            for i in warning:
                lines.append(f"  - {i.issue_type}: {i.count} 筆")

        if info:
            lines.append(f"\nℹ️ 資訊 ({len(info)})")
            for i in info:
                lines.append(f"  - {i.issue_type}: {i.count} 筆")

        fixable = [i for i in issues if i.fixable]
        if fixable:
            lines.append(f"\n🔧 可自動修復: {len(fixable)} 項")

        return "\n".join(lines)


def validate_ohlcv(df: pd.DataFrame) -> List[DataIssue]:
    """
    便捷函數：驗證 OHLCV 資料

    Args:
        df: OHLCV DataFrame

    Returns:
        問題列表
    """
    validator = DataValidator()
    return validator.validate(df)
