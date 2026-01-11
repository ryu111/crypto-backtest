"""
DataCleaner 單元測試

測試資料清理功能的正確性
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.data.cleaner import DataCleaner


class TestDataCleaner(unittest.TestCase):
    """測試 DataCleaner 類別"""

    def setUp(self) -> None:
        """設定測試環境"""
        self.cleaner = DataCleaner(timeframe='4h', verbose=False)

    def _create_sample_data(
        self, gaps: list[int] | None = None
    ) -> pd.DataFrame:
        """
        建立測試用的範例資料

        Args:
            gaps: 缺失位置列表（索引）

        Returns:
            測試用 DataFrame
        """
        # 建立連續時間序列
        start = datetime(2024, 1, 1)
        periods = 100
        freq = '4h'

        timestamps = pd.date_range(start=start, periods=periods, freq=freq)
        df = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, periods),
            'high': np.random.uniform(45000, 50000, periods),
            'low': np.random.uniform(35000, 40000, periods),
            'close': np.random.uniform(40000, 45000, periods),
            'volume': np.random.uniform(1000, 5000, periods)
        }, index=timestamps)

        # 確保 OHLC 邏輯正確
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)

        # 加入 gaps - 使用列表轉換確保類型正確
        if gaps is not None:
            indices_to_drop: list[datetime] = list(df.index[gaps])
            df = df.drop(index=indices_to_drop)

        return df

    def test_parse_timeframe(self):
        """測試時間框架解析"""
        cleaner_1m = DataCleaner(timeframe='1m')
        self.assertEqual(cleaner_1m.expected_interval, timedelta(minutes=1))

        cleaner_4h = DataCleaner(timeframe='4h')
        self.assertEqual(cleaner_4h.expected_interval, timedelta(hours=4))

        cleaner_1d = DataCleaner(timeframe='1d')
        self.assertEqual(cleaner_1d.expected_interval, timedelta(days=1))

    def test_remove_duplicates(self):
        """測試移除重複時間戳"""
        df = self._create_sample_data()

        # 加入重複時間戳
        dup_row = df.iloc[10:11].copy()
        df = pd.concat([df, dup_row])

        self.assertTrue(df.index.duplicated().any())

        # 執行清理
        df_cleaned = self.cleaner._remove_duplicates(df)

        self.assertFalse(df_cleaned.index.duplicated().any())

    def test_detect_gaps(self):
        """測試缺失偵測"""
        # 建立有缺失的資料（移除索引 10-12）
        df = self._create_sample_data(gaps=[10, 11, 12])

        gaps = self.cleaner._detect_gaps(df)

        self.assertGreater(len(gaps), 0)
        self.assertEqual(gaps[0].gap_size, 3)

    def test_validate_ohlc(self):
        """測試 OHLC 驗證"""
        df = self._create_sample_data()

        # 正常資料應該沒有錯誤
        invalid_count = self.cleaner._validate_ohlc(df)
        self.assertEqual(invalid_count, 0)

        # 加入錯誤資料
        df.iloc[0, df.columns.get_loc('high')] = df.iloc[0]['low'] - 100

        invalid_count = self.cleaner._validate_ohlc(df)
        self.assertGreater(invalid_count, 0)

    def test_analyze_quality(self):
        """測試品質分析"""
        df = self._create_sample_data()

        report = self.cleaner.analyze_quality(df)

        self.assertEqual(report.total_records, len(df))
        self.assertGreaterEqual(report.quality_score, 0)
        self.assertLessEqual(report.quality_score, 100)
        self.assertIsInstance(report.issues, list)

    def test_clean_with_gaps(self):
        """測試資料清理（有缺失）"""
        # 建立有缺失的資料（移除 3 筆以產生明顯 gap）
        df = self._create_sample_data(gaps=[10, 11, 12])

        original_len = len(df)

        # 先檢查是否有 gap
        gaps = self.cleaner._detect_gaps(df)
        self.assertGreater(len(gaps), 0, "應該偵測到 gap")

        # 執行清理
        df_cleaned = self.cleaner.clean(
            df,
            fill_short_gaps=True,
            mark_long_gaps=False
        )

        # 清理後應該填補缺失（或至少不會變少）
        self.assertGreaterEqual(len(df_cleaned), original_len)

    def test_clean_without_gaps(self):
        """測試資料清理（無缺失）"""
        df = self._create_sample_data()

        original_len = len(df)

        # 執行清理
        df_cleaned = self.cleaner.clean(
            df,
            fill_short_gaps=True,
            mark_long_gaps=False
        )

        # 無缺失的資料應該保持長度
        self.assertEqual(len(df_cleaned), original_len)

    def test_mark_long_gaps(self):
        """測試長缺失標記"""
        # 建立有長缺失的資料（移除 10-20）
        df = self._create_sample_data(gaps=list(range(10, 21)))

        gaps = self.cleaner._detect_gaps(df)

        # 標記缺失
        df_marked = self.cleaner._mark_long_gaps(df, gaps)

        # 應該有 gap_flag 欄位
        self.assertIn('gap_flag', df_marked.columns)

        # 應該有標記（非全為 0）
        self.assertGreater(df_marked['gap_flag'].sum(), 0)

    def test_quality_score_calculation(self):
        """測試品質評分計算"""
        # 完美資料
        score_perfect = self.cleaner._calculate_quality_score(
            missing_rate=0.0,
            gap_count=0,
            total=1000,
            issues=0
        )
        self.assertEqual(score_perfect, 100.0)

        # 有問題的資料
        score_bad = self.cleaner._calculate_quality_score(
            missing_rate=0.1,
            gap_count=100,
            total=1000,
            issues=5
        )
        self.assertLess(score_bad, score_perfect)


class TestGapInfo(unittest.TestCase):
    """測試 GapInfo 資料類別"""

    def test_gap_info_str(self):
        """測試 GapInfo 字串表示"""
        from src.data.cleaner import GapInfo

        gap = GapInfo(
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 1, 8, 0),
            duration=timedelta(hours=8),
            gap_size=2,
            is_maintenance=False
        )

        gap_str = str(gap)
        self.assertIn("資料缺失", gap_str)
        self.assertIn("8.0h", gap_str)


class TestDataQualityReport(unittest.TestCase):
    """測試 DataQualityReport 資料類別"""

    def test_report_str(self):
        """測試報告字串表示"""
        from src.data.cleaner import DataQualityReport, GapInfo

        report = DataQualityReport(
            total_records=1000,
            missing_count=10,
            missing_rate=0.01,
            gap_count=2,
            gaps=[],
            quality_score=95.5,
            issues=["測試問題"]
        )

        report_str = str(report)
        self.assertIn("總筆數: 1,000", report_str)
        self.assertIn("品質評分: 95.50", report_str)
        self.assertIn("測試問題", report_str)


if __name__ == '__main__':
    unittest.main()
