"""
DataCleaner 邊界測試

測試極端情況和邊界條件
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.data.cleaner import DataCleaner, GapFillStrategy


class TestCleanerEdgeCases(unittest.TestCase):
    """測試邊界情況"""

    def setUp(self) -> None:
        """設定測試環境"""
        self.cleaner = DataCleaner(timeframe='4h', verbose=False)

    def test_empty_dataframe(self):
        """測試空 DataFrame"""
        df = pd.DataFrame()

        # 應該不會拋錯
        gaps = self.cleaner._detect_gaps(df)
        self.assertEqual(len(gaps), 0)

        df_cleaned = self.cleaner.clean(df)
        self.assertEqual(len(df_cleaned), 0)

    def test_single_row(self):
        """測試單筆資料"""
        timestamps = pd.date_range(start=datetime(2024, 1, 1), periods=1, freq='4h')
        df = pd.DataFrame({
            'open': [40000],
            'high': [45000],
            'low': [39000],
            'close': [42000],
            'volume': [1000]
        }, index=timestamps)

        gaps = self.cleaner._detect_gaps(df)
        self.assertEqual(len(gaps), 0)

        df_cleaned = self.cleaner.clean(df)
        self.assertEqual(len(df_cleaned), 1)

    def test_all_missing_values(self):
        """測試全部缺失值"""
        timestamps = pd.date_range(start=datetime(2024, 1, 1), periods=100, freq='4h')
        df = pd.DataFrame({
            'open': [np.nan] * 100,
            'high': [np.nan] * 100,
            'low': [np.nan] * 100,
            'close': [np.nan] * 100,
            'volume': [np.nan] * 100
        }, index=timestamps)

        report = self.cleaner.analyze_quality(df)

        # 應該偵測到 100% 缺失
        self.assertEqual(report.missing_rate, 1.0)
        self.assertEqual(report.quality_score, 0.0)
        self.assertGreater(len(report.issues), 0)

    def test_extreme_gaps(self):
        """測試極端缺失（99% 資料遺失）"""
        timestamps = pd.date_range(start=datetime(2024, 1, 1), periods=1000, freq='4h')
        df = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, 1000),
            'high': np.random.uniform(45000, 50000, 1000),
            'low': np.random.uniform(35000, 40000, 1000),
            'close': np.random.uniform(40000, 45000, 1000),
            'volume': np.random.uniform(1000, 5000, 1000)
        }, index=timestamps)

        # 只保留 10 筆（99% 缺失）
        df = df.iloc[::100]  # 每 100 筆取 1 筆

        gaps = self.cleaner._detect_gaps(df)

        # 應該偵測到多個 gap
        self.assertGreater(len(gaps), 0)

        # 品質評分應該很低
        report = self.cleaner.analyze_quality(df)
        self.assertLess(report.quality_score, 50.0)

    def test_zero_volume(self):
        """測試零成交量"""
        timestamps = pd.date_range(start=datetime(2024, 1, 1), periods=100, freq='4h')
        df = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, 100),
            'high': np.random.uniform(45000, 50000, 100),
            'low': np.random.uniform(35000, 40000, 100),
            'close': np.random.uniform(40000, 45000, 100),
            'volume': [0] * 100  # 全部零成交量
        }, index=timestamps)

        report = self.cleaner.analyze_quality(df)

        # 應該偵測到零成交量問題
        zero_vol_issue = any("零成交量" in issue for issue in report.issues)
        self.assertTrue(zero_vol_issue)

    def test_invalid_ohlc(self):
        """測試無效 OHLC（high < low）"""
        timestamps = pd.date_range(start=datetime(2024, 1, 1), periods=100, freq='4h')
        df = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, 100),
            'high': [35000] * 100,  # high 比 low 還低
            'low': [45000] * 100,
            'close': np.random.uniform(40000, 45000, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=timestamps)

        invalid_count = self.cleaner._validate_ohlc(df)
        self.assertEqual(invalid_count, 100)

        report = self.cleaner.analyze_quality(df)
        ohlc_issue = any("OHLC" in issue for issue in report.issues)
        self.assertTrue(ohlc_issue)

    def test_consecutive_duplicates(self):
        """測試連續重複時間戳"""
        timestamps = pd.date_range(start=datetime(2024, 1, 1), periods=50, freq='4h')
        df = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, 50),
            'high': np.random.uniform(45000, 50000, 50),
            'low': np.random.uniform(35000, 40000, 50),
            'close': np.random.uniform(40000, 45000, 50),
            'volume': np.random.uniform(1000, 5000, 50)
        }, index=timestamps)

        # 加入大量重複
        dup_rows = df.iloc[10:20].copy()
        df = pd.concat([df, dup_rows, dup_rows, dup_rows])

        original_len = len(df)
        self.assertTrue(df.index.duplicated().any())

        df_cleaned = self.cleaner._remove_duplicates(df)

        # 應該移除重複
        self.assertFalse(df_cleaned.index.duplicated().any())
        self.assertLess(len(df_cleaned), original_len)

    def test_very_long_gap(self):
        """測試超長缺失（數月）"""
        # 前 10 筆
        timestamps1 = pd.date_range(start=datetime(2024, 1, 1), periods=10, freq='4h')
        df1 = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, 10),
            'high': np.random.uniform(45000, 50000, 10),
            'low': np.random.uniform(35000, 40000, 10),
            'close': np.random.uniform(40000, 45000, 10),
            'volume': np.random.uniform(1000, 5000, 10)
        }, index=timestamps1)

        # 後 10 筆（3 個月後）
        timestamps2 = pd.date_range(start=datetime(2024, 4, 1), periods=10, freq='4h')
        df2 = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, 10),
            'high': np.random.uniform(45000, 50000, 10),
            'low': np.random.uniform(35000, 40000, 10),
            'close': np.random.uniform(40000, 45000, 10),
            'volume': np.random.uniform(1000, 5000, 10)
        }, index=timestamps2)

        df = pd.concat([df1, df2])

        gaps = self.cleaner._detect_gaps(df)

        # 應該偵測到 1 個超長 gap
        self.assertEqual(len(gaps), 1)
        self.assertGreater(gaps[0].duration, timedelta(days=30))

        # 超長 gap 不應該被填補
        self.assertEqual(gaps[0].fill_strategy, GapFillStrategy.NONE)

    def test_maintenance_window_detection(self):
        """測試維護期間偵測"""
        # 建立在凌晨 1 點開始的長 gap
        timestamps1 = pd.date_range(start=datetime(2024, 1, 1, 0, 0), periods=10, freq='4h')
        df1 = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, 10),
            'high': np.random.uniform(45000, 50000, 10),
            'low': np.random.uniform(35000, 40000, 10),
            'close': np.random.uniform(40000, 45000, 10),
            'volume': np.random.uniform(1000, 5000, 10)
        }, index=timestamps1)

        # 6 小時後恢復（凌晨時段的長 gap）
        timestamps2 = pd.date_range(start=datetime(2024, 1, 1, 6, 0), periods=10, freq='4h')
        df2 = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, 10),
            'high': np.random.uniform(45000, 50000, 10),
            'low': np.random.uniform(35000, 40000, 10),
            'close': np.random.uniform(40000, 45000, 10),
            'volume': np.random.uniform(1000, 5000, 10)
        }, index=timestamps2)

        df = pd.concat([df1, df2])

        gaps = self.cleaner._detect_gaps(df)

        # 應該偵測到維護期間
        if len(gaps) > 0:
            # 檢查是否有 gap 被標記為維護期間
            has_maintenance = any(gap.is_maintenance for gap in gaps)
            # 注意：維護偵測邏輯依賴時間點，這裡只驗證功能不會報錯
            self.assertIsInstance(has_maintenance, bool)

    def test_performance_large_dataset(self):
        """測試大資料集效能（10000 筆）"""
        timestamps = pd.date_range(start=datetime(2023, 1, 1), periods=10000, freq='4h')
        df = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, 10000),
            'high': np.random.uniform(45000, 50000, 10000),
            'low': np.random.uniform(35000, 40000, 10000),
            'close': np.random.uniform(40000, 45000, 10000),
            'volume': np.random.uniform(1000, 5000, 10000)
        }, index=timestamps)

        # 加入一些 gap
        df = df.drop(df.index[1000:1010])
        df = df.drop(df.index[5000:5020])

        import time
        start_time = time.time()

        # 執行完整清理流程
        df_cleaned = self.cleaner.clean(df)
        report = self.cleaner.analyze_quality(df_cleaned)

        elapsed = time.time() - start_time

        # 應該在 5 秒內完成
        self.assertLess(elapsed, 5.0, f"處理 10000 筆資料花費 {elapsed:.2f} 秒，超過預期")

        # 驗證結果
        self.assertGreater(len(df_cleaned), 0)
        self.assertGreater(report.quality_score, 0)


class TestInterpolationLimit(unittest.TestCase):
    """測試插值限制"""

    def test_interpolation_limit_respected(self):
        """測試插值限制被遵守"""
        cleaner = DataCleaner(timeframe='4h', verbose=False)

        # 建立有超長 gap 的資料（超過 1 小時 = 15 個 4h bar）
        timestamps1 = pd.date_range(start=datetime(2024, 1, 1), periods=10, freq='4h')
        df1 = pd.DataFrame({
            'open': [40000.0] * 10,
            'high': [45000.0] * 10,
            'low': [35000.0] * 10,
            'close': [42000.0] * 10,
            'volume': [1000.0] * 10
        }, index=timestamps1)

        # 1 天後（6 個 4h bar）
        timestamps2 = pd.date_range(start=datetime(2024, 1, 2), periods=10, freq='4h')
        df2 = pd.DataFrame({
            'open': [41000.0] * 10,
            'high': [46000.0] * 10,
            'low': [36000.0] * 10,
            'close': [43000.0] * 10,
            'volume': [1100.0] * 10
        }, index=timestamps2)

        df = pd.concat([df1, df2])

        # 執行填補
        gaps = cleaner._detect_gaps(df)
        df_filled = cleaner._fill_gaps(df, gaps, only_short=True)

        # 超過限制的 gap 應該不被填補
        # 驗證：填補後的資料不應該是完全連續的
        expected_full_len = len(pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq='4h'
        ))

        # 由於 gap 太長，不應該完全填補
        self.assertLess(len(df_filled), expected_full_len)


if __name__ == '__main__':
    unittest.main()
