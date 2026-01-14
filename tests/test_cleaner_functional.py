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
        # gap_threshold = expected_interval * gap_threshold_multiplier = 4h * 2.0 = 8h
        # 需要 > 8h 才會被偵測為 gap
        # 建立已知 gap 的資料
        # timestamps1: 2024-01-01 00:00 ~ 2024-01-02 12:00 (10 筆，每筆 4h)
        timestamps1 = pd.date_range(start=datetime(2024, 1, 1, 0, 0), periods=10, freq='4h')

        # timestamps2 從 2024-01-03 00:00 開始，與 timestamps1 最後一筆（2024-01-02 12:00）有 12 小時 gap
        # Gap: 2024-01-02 12:00 ~ 2024-01-03 00:00 (12h > 8h threshold)
        timestamps2 = pd.date_range(start=datetime(2024, 1, 3, 0, 0), periods=10, freq='4h')

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

        df = pd.concat([df1, df2]).sort_index()

        # 偵測 gap
        gaps = self.cleaner._detect_gaps(df)

        # 驗證
        self.assertEqual(len(gaps), 1, "應該偵測到 1 個 gap")

        gap = gaps[0]
        # Gap 從 12:00 到 00:00，時長 12 小時，缺少 (12/4) - 1 = 2 個 4h bar
        self.assertEqual(gap.gap_size, 2, "Gap 大小應該是 2（中間缺少的 bar 數）")
        self.assertEqual(gap.duration, timedelta(hours=12), "Gap 時長應該是 12 小時")

    def test_fill_strategy_selection(self):
        """測試填補策略選擇正確性"""
        # 測試 1：短 gap (<= long_gap_threshold=1h) - 應該使用 LINEAR
        # 使用 15m timeframe 來建立符合條件的短 gap
        cleaner_short = DataCleaner(timeframe='15m', long_gap_hours=1.0, verbose=False)

        # timestamps1: 最後一筆 = 2024-01-01 00:00 + 9*15m = 2024-01-01 02:15
        timestamps1 = pd.date_range(start=datetime(2024, 1, 1, 0, 0), periods=10, freq='15min')
        # timestamps2: 從 2024-01-01 03:00 開始，gap = 45min < 1h
        # gap_threshold = 15min * 2 = 30min，45min > 30min，會被偵測
        timestamps2 = pd.date_range(start=datetime(2024, 1, 1, 3, 0), periods=10, freq='15min')

        df1 = pd.DataFrame({'close': [100.0] * 10}, index=timestamps1)
        df2 = pd.DataFrame({'close': [110.0] * 10}, index=timestamps2)

        df_short = pd.concat([df1, df2]).sort_index()
        gaps_short = cleaner_short._detect_gaps(df_short)

        self.assertGreater(len(gaps_short), 0, "應該偵測到至少 1 個 gap")
        # 45min < 1h，應該使用 LINEAR
        self.assertEqual(gaps_short[0].fill_strategy, GapFillStrategy.LINEAR)

        # 測試 2：長 gap (> long_gap_threshold) - 應該使用 NONE
        # 使用原本的 4h cleaner
        timestamps3 = pd.date_range(start=datetime(2024, 1, 1, 0, 0), periods=10, freq='4h')
        # timestamps4 從 2024-01-03 00:00 開始，gap = 2 天 - 12h = 36h > 1h
        timestamps4 = pd.date_range(start=datetime(2024, 1, 3, 0, 0), periods=10, freq='4h')

        df3 = pd.DataFrame({'close': [120.0] * 10}, index=timestamps3)
        df4 = pd.DataFrame({'close': [130.0] * 10}, index=timestamps4)

        df_long = pd.concat([df3, df4]).sort_index()
        gaps_long = self.cleaner._detect_gaps(df_long)

        self.assertGreater(len(gaps_long), 0, "應該偵測到至少 1 個 gap")
        # 36h > 1h，應該使用 NONE
        self.assertEqual(gaps_long[0].fill_strategy, GapFillStrategy.NONE)

    def test_linear_interpolation_correctness(self):
        """測試線性插值正確性"""
        # 使用 15m timeframe 建立可填補的 gap（注意格式是 '15m' 不是 '15min'）
        cleaner_fill = DataCleaner(timeframe='15m', long_gap_hours=1.0, verbose=False)

        # timestamps1: 2024-01-01 00:00 ~ 2024-01-01 00:30 (3 筆)
        timestamps1 = pd.date_range(start=datetime(2024, 1, 1, 0, 0), periods=3, freq='15min')
        # timestamps2: 從 2024-01-01 01:15 開始，gap = 45min > 30min threshold，且 < 1h，會填補
        timestamps2 = pd.date_range(start=datetime(2024, 1, 1, 1, 15), periods=2, freq='15min')

        df1 = pd.DataFrame({
            'close': [100.0, 110.0, 120.0]
        }, index=timestamps1)

        df2 = pd.DataFrame({
            'close': [200.0, 210.0]
        }, index=timestamps2)

        df = pd.concat([df1, df2]).sort_index()

        gaps = cleaner_fill._detect_gaps(df)

        self.assertGreater(len(gaps), 0, "應該偵測到至少 1 個 gap")

        # 填補
        df_filled = cleaner_fill._fill_gaps(df, gaps, only_short=True)

        # 檢查插值結果：應該填補了中間的時間點
        # reindex 會建立完整索引，interpolate 會填充數值
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

        max_values = df_perfect[['open', 'high', 'close']].max(axis=1)
        min_values = df_perfect[['open', 'low', 'close']].min(axis=1)
        df_perfect['high'] = max_values
        df_perfect['low'] = min_values

        report_perfect = self.cleaner.analyze_quality(df_perfect)
        self.assertEqual(report_perfect.quality_score, 100.0)

        # 有缺失的資料
        df_missing = df_perfect.copy()
        missing_index = df_missing.index[10:20].tolist()
        df_missing.loc[missing_index, 'volume'] = np.nan

        report_missing = self.cleaner.analyze_quality(df_missing)
        self.assertLess(report_missing.quality_score, report_perfect.quality_score)

        # 有 gap 的資料
        df_gaps = df_perfect.drop(index=df_perfect.index[10:15].tolist())

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
        df = df.drop(index=df.index[10:15].tolist())

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
            cleaner_test = DataCleaner(timeframe=tf_str)
            self.assertEqual(
                cleaner_test.expected_interval,
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
        dup_index = pd.DatetimeIndex([timestamps[5]])
        dup_row = pd.DataFrame({
            'close': [999]  # 不同的值
        }, index=dup_index)

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
