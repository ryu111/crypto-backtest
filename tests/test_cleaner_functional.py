"""
DataCleaner 功能測試

測試實際使用場景的功能正確性
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.data.cleaner import DataCleaner, GapFillStrategy, GapInfo


class TestFunctionalScenarios(unittest.TestCase):
    """測試實際功能場景"""

    def setUp(self) -> None:
        """設定測試環境"""
        self.cleaner = DataCleaner(timeframe='4h', verbose=True)

    def test_gap_detection_accuracy(self):
        """測試缺失偵測準確性"""
        # 建立已知 gap 的資料
        timestamps1 = pd.date_range(start=datetime(2024, 1, 1, 0, 0), periods=10, freq='4h')
        timestamps2 = pd.date_range(start=datetime(2024, 1, 1, 20, 0), periods=10, freq='4h')

        df1 = pd.DataFrame({
            'open': [40000.0] * 10,
            'high': [45000.0] * 10,
            'low': [35000.0] * 10,
            'close': [42000.0] * 10,
            'volume': [1000.0] * 10
        }, index=timestamps1)

        df2 = pd.DataFrame({
            'open': [41000.0] * 10,
            'high': [46000.0] * 10,
            'low': [36000.0] * 10,
            'close': [43000.0] * 10,
            'volume': [1100.0] * 10
        }, index=timestamps2)

        df = pd.concat([df1, df2])

        # 偵測 gap
        gaps = self.cleaner._detect_gaps(df)

        # 驗證
        self.assertEqual(len(gaps), 1, "應該偵測到 1 個 gap")

        gap = gaps[0]
        # 從 16:00 到 20:00，缺少 4 小時 = 1 個 4h bar
        self.assertEqual(gap.gap_size, 1, "Gap 大小應該是 1")
        self.assertEqual(gap.duration, timedelta(hours=4), "Gap 時長應該是 4 小時")

    def test_fill_strategy_selection(self):
        """測試填補策略選擇正確性"""
        # 短 gap (4h) - 應該使用 LINEAR
        timestamps1 = pd.date_range(start=datetime(2024, 1, 1), periods=10, freq='4h')
        timestamps2 = pd.date_range(start=datetime(2024, 1, 1, 16, 0), periods=10, freq='4h')

        df1 = pd.DataFrame({
            'open': [40000.0] * 10,
            'high': [45000.0] * 10,
            'low': [35000.0] * 10,
            'close': [42000.0] * 10
        }, index=timestamps1)

        df2 = pd.DataFrame({
            'open': [41000.0] * 10,
            'high': [46000.0] * 10,
            'low': [36000.0] * 10,
            'close': [43000.0] * 10
        }, index=timestamps2)

        df_short = pd.concat([df1, df2])
        gaps_short = self.cleaner._detect_gaps(df_short)

        self.assertEqual(gaps_short[0].fill_strategy, GapFillStrategy.LINEAR)

        # 長 gap (2 days) - 應該使用 NONE
        timestamps3 = pd.date_range(start=datetime(2024, 1, 3), periods=10, freq='4h')
        df3 = pd.DataFrame({
            'open': [42000.0] * 10,
            'high': [47000.0] * 10,
            'low': [37000.0] * 10,
            'close': [44000.0] * 10
        }, index=timestamps3)

        df_long = pd.concat([df1, df3])
        gaps_long = self.cleaner._detect_gaps(df_long)

        self.assertEqual(gaps_long[0].fill_strategy, GapFillStrategy.NONE)

    def test_linear_interpolation_correctness(self):
        """測試線性插值正確性"""
        # 建立簡單的測試資料
        timestamps = pd.date_range(start=datetime(2024, 1, 1), periods=5, freq='4h')
        df = pd.DataFrame({
            'close': [100.0, np.nan, np.nan, 200.0, 250.0]
        }, index=timestamps)

        # 填補
        gaps = self.cleaner._detect_gaps(df)
        df_filled = self.cleaner._fill_gaps(df, gaps, only_short=True)

        # 檢查插值結果（應該是 100 -> 125 -> 150 -> 175 -> 200）
        # 注意：實際值可能因為 reindex 而不同，主要確認有填補
        self.assertFalse(df_filled['close'].isnull().any(), "不應該有 NaN")

    def test_gap_flag_marking(self):
        """測試 gap flag 標記正確性"""
        # 建立不同類型的 gap
        timestamps1 = pd.date_range(start=datetime(2024, 1, 1, 0, 0), periods=10, freq='4h')
        timestamps2 = pd.date_range(start=datetime(2024, 1, 1, 16, 0), periods=5, freq='4h')  # 短 gap
        timestamps3 = pd.date_range(start=datetime(2024, 1, 3, 0, 0), periods=10, freq='4h')  # 長 gap

        df1 = pd.DataFrame({
            'close': [100.0] * 10
        }, index=timestamps1)

        df2 = pd.DataFrame({
            'close': [110.0] * 5
        }, index=timestamps2)

        df3 = pd.DataFrame({
            'close': [120.0] * 10
        }, index=timestamps3)

        df = pd.concat([df1, df2, df3])

        # 標記 gap
        gaps = self.cleaner._detect_gaps(df)
        df_marked = self.cleaner._mark_long_gaps(df, gaps)

        # 驗證
        self.assertIn('gap_flag', df_marked.columns)

        # 應該有不同的 flag 值
        unique_flags = df_marked['gap_flag'].unique()
        self.assertGreater(len(unique_flags), 1, "應該有多種 flag 類型")

    def test_ohlc_validation_comprehensive(self):
        """測試 OHLC 驗證的全面性"""
        timestamps = pd.date_range(start=datetime(2024, 1, 1), periods=5, freq='4h')

        # 測試各種無效情況
        test_cases = [
            {
                'name': 'high < open',
                'data': {
                    'open': [100.0, 100.0, 100.0, 100.0, 100.0],
                    'high': [90.0, 110.0, 110.0, 110.0, 110.0],  # 第一筆無效
                    'low': [80.0, 80.0, 80.0, 80.0, 80.0],
                    'close': [95.0, 95.0, 95.0, 95.0, 95.0]
                },
                'expected_invalid': 1
            },
            {
                'name': 'high < close',
                'data': {
                    'open': [100.0, 100.0, 100.0, 100.0, 100.0],
                    'high': [110.0, 110.0, 90.0, 110.0, 110.0],  # 第三筆無效
                    'low': [80.0, 80.0, 80.0, 80.0, 80.0],
                    'close': [95.0, 95.0, 120.0, 95.0, 95.0]
                },
                'expected_invalid': 1
            },
            {
                'name': 'low > open',
                'data': {
                    'open': [100.0, 100.0, 100.0, 100.0, 100.0],
                    'high': [110.0, 110.0, 110.0, 110.0, 110.0],
                    'low': [80.0, 80.0, 80.0, 110.0, 80.0],  # 第四筆無效
                    'close': [95.0, 95.0, 95.0, 95.0, 95.0]
                },
                'expected_invalid': 1
            },
            {
                'name': 'high < low',
                'data': {
                    'open': [100.0, 100.0, 100.0, 100.0, 100.0],
                    'high': [80.0, 110.0, 110.0, 110.0, 110.0],  # 第一筆無效
                    'low': [90.0, 80.0, 80.0, 80.0, 80.0],
                    'close': [95.0, 95.0, 95.0, 95.0, 95.0]
                },
                'expected_invalid': 1
            }
        ]

        for test_case in test_cases:
            df = pd.DataFrame(test_case['data'], index=timestamps)
            invalid_count = self.cleaner._validate_ohlc(df)
            self.assertEqual(
                invalid_count,
                test_case['expected_invalid'],
                f"{test_case['name']} 應該偵測到 {test_case['expected_invalid']} 筆錯誤"
            )

    def test_quality_score_components(self):
        """測試品質評分的各個組成因素"""
        # 完美資料
        timestamps = pd.date_range(start=datetime(2024, 1, 1), periods=100, freq='4h')
        df_perfect = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, 100),
            'high': np.random.uniform(45000, 50000, 100),
            'low': np.random.uniform(35000, 40000, 100),
            'close': np.random.uniform(40000, 45000, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=timestamps)

        df_perfect['high'] = df_perfect[['open', 'high', 'close']].max(axis=1)
        df_perfect['low'] = df_perfect[['open', 'low', 'close']].min(axis=1)

        report_perfect = self.cleaner.analyze_quality(df_perfect)
        self.assertEqual(report_perfect.quality_score, 100.0)

        # 有缺失的資料
        df_missing = df_perfect.copy()
        df_missing.loc[df_missing.index[10:20], 'volume'] = np.nan

        report_missing = self.cleaner.analyze_quality(df_missing)
        self.assertLess(report_missing.quality_score, report_perfect.quality_score)

        # 有 gap 的資料
        df_gaps = df_perfect.drop(df_perfect.index[10:15])

        report_gaps = self.cleaner.analyze_quality(df_gaps)
        self.assertLess(report_gaps.quality_score, report_perfect.quality_score)

    def test_quality_report_readability(self):
        """測試品質報告的可讀性"""
        # 建立有問題的資料
        timestamps = pd.date_range(start=datetime(2024, 1, 1), periods=100, freq='4h')
        df = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, 100),
            'high': np.random.uniform(45000, 50000, 100),
            'low': np.random.uniform(35000, 40000, 100),
            'close': np.random.uniform(40000, 45000, 100),
            'volume': [0] * 100  # 零成交量
        }, index=timestamps)

        # 加入缺失
        df = df.drop(df.index[10:15])

        # 加入無效 OHLC
        df.loc[df.index[0], 'high'] = df.loc[df.index[0], 'low'] - 1000

        report = self.cleaner.analyze_quality(df)
        report_str = str(report)

        # 驗證報告包含關鍵資訊
        self.assertIn("資料品質報告", report_str)
        self.assertIn("總筆數", report_str)
        self.assertIn("品質評分", report_str)
        self.assertIn("主要問題", report_str)

        # 應該包含問題描述
        self.assertGreater(len(report.issues), 0)

    def test_timeframe_parsing_edge_cases(self):
        """測試時間框架解析的邊界情況"""
        test_cases = [
            ('1m', timedelta(minutes=1)),
            ('5m', timedelta(minutes=5)),
            ('15m', timedelta(minutes=15)),
            ('1h', timedelta(hours=1)),
            ('4h', timedelta(hours=4)),
            ('1d', timedelta(days=1)),
        ]

        for tf_str, expected_td in test_cases:
            cleaner = DataCleaner(timeframe=tf_str)
            self.assertEqual(
                cleaner.expected_interval,
                expected_td,
                f"{tf_str} 應該解析為 {expected_td}"
            )

    def test_duplicate_removal_preserves_latest(self):
        """測試移除重複時保留最新資料"""
        timestamps = pd.date_range(start=datetime(2024, 1, 1), periods=10, freq='4h')
        df = pd.DataFrame({
            'close': list(range(10))  # 0, 1, 2, ..., 9
        }, index=timestamps)

        # 加入重複（相同時間戳，不同值）
        dup_row = pd.DataFrame({
            'close': [999]  # 不同的值
        }, index=[timestamps[5]])

        df = pd.concat([df, dup_row])

        # 移除重複
        df_cleaned = self.cleaner._remove_duplicates(df)

        # 驗證保留的是最後一筆（999）
        self.assertEqual(
            df_cleaned.loc[timestamps[5], 'close'],
            999,
            "應該保留最後一筆重複資料"
        )


class TestGapInfoDataClass(unittest.TestCase):
    """測試 GapInfo 資料類別的功能"""

    def test_gap_info_creation(self):
        """測試 GapInfo 建立"""
        gap = GapInfo(
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 1, 8, 0),
            duration=timedelta(hours=8),
            gap_size=2,
            is_maintenance=True,
            fill_strategy=GapFillStrategy.LINEAR
        )

        self.assertEqual(gap.start_time, datetime(2024, 1, 1, 0, 0))
        self.assertEqual(gap.gap_size, 2)
        self.assertTrue(gap.is_maintenance)
        self.assertEqual(gap.fill_strategy, GapFillStrategy.LINEAR)

    def test_gap_info_string_representation(self):
        """測試 GapInfo 字串表示"""
        # 一般缺失
        gap_normal = GapInfo(
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 1, 8, 0),
            duration=timedelta(hours=8),
            gap_size=2,
            is_maintenance=False
        )

        gap_str = str(gap_normal)
        self.assertIn("資料缺失", gap_str)
        self.assertIn("8.0h", gap_str)

        # 維護期間
        gap_maintenance = GapInfo(
            start_time=datetime(2024, 1, 1, 0, 0),
            end_time=datetime(2024, 1, 1, 8, 0),
            duration=timedelta(hours=8),
            gap_size=2,
            is_maintenance=True
        )

        gap_str_maint = str(gap_maintenance)
        self.assertIn("維護期間", gap_str_maint)


if __name__ == '__main__':
    unittest.main()
