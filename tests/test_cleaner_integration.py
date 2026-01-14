"""
DataCleaner 整合測試

測試與 DataFetcher 的整合
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.data.cleaner import DataCleaner


class TestCleanerIntegration(unittest.TestCase):
    """測試與其他模組的整合"""

    def setUp(self) -> None:
        """設定測試環境"""
        self.cleaner = DataCleaner(timeframe='4h', verbose=False)

    def test_real_world_workflow(self):
        """測試真實世界的完整工作流程"""
        # 1. 模擬從交易所取得的資料（有缺失、有重複）
        timestamps = pd.date_range(start=datetime(2024, 1, 1), periods=100, freq='4h')
        df = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, 100),
            'high': np.random.uniform(45000, 50000, 100),
            'low': np.random.uniform(35000, 40000, 100),
            'close': np.random.uniform(40000, 45000, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=timestamps)

        # 確保 OHLC 邏輯正確
        max_values = df[['open', 'high', 'close']].max(axis=1)
        min_values = df[['open', 'low', 'close']].min(axis=1)
        df['high'] = max_values
        df['low'] = min_values

        # 加入問題：
        # - 移除一些資料（gap）
        df = df.drop(index=df.index[10:13].tolist())
        df = df.drop(index=df.index[50:52].tolist())

        # - 加入重複
        dup_rows = df.iloc[20:22].copy()
        df = pd.concat([df, dup_rows])

        # - 加入一些缺失值
        df.loc[df.index[5], 'volume'] = np.nan

        # 2. 分析品質
        report_before = self.cleaner.analyze_quality(df)

        self.assertGreater(report_before.missing_count, 0)
        self.assertGreater(report_before.gap_count, 0)

        # 3. 清理資料
        df_cleaned = self.cleaner.clean(
            df,
            fill_short_gaps=True,
            mark_long_gaps=True
        )

        # 4. 驗證清理效果
        self.assertFalse(df_cleaned.index.duplicated().any(), "應該移除重複")
        self.assertIn('gap_flag', df_cleaned.columns, "應該有 gap_flag 欄位")

        # 5. 再次分析品質
        report_after = self.cleaner.analyze_quality(df_cleaned)

        # 品質應該有改善（或至少不變差）
        self.assertGreaterEqual(
            report_after.quality_score,
            report_before.quality_score - 5,  # 允許小幅誤差
            "清理後品質不應該變差"
        )

    def test_pipeline_with_multiple_symbols(self):
        """測試處理多個交易對的流程"""
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        results = {}

        for symbol in symbols:
            # 建立測試資料
            timestamps = pd.date_range(start=datetime(2024, 1, 1), periods=50, freq='4h')
            df = pd.DataFrame({
                'open': np.random.uniform(100, 200, 50),
                'high': np.random.uniform(200, 300, 50),
                'low': np.random.uniform(50, 100, 50),
                'close': np.random.uniform(100, 200, 50),
                'volume': np.random.uniform(1000, 5000, 50)
            }, index=timestamps)

            max_vals = df[['open', 'high', 'close']].max(axis=1)
            min_vals = df[['open', 'low', 'close']].min(axis=1)
            df['high'] = max_vals
            df['low'] = min_vals

            # 隨機加入一些 gap
            if symbol == 'BTCUSDT':
                df = df.drop(index=df.index[10:12].tolist())
            elif symbol == 'ETHUSDT':
                df = df.drop(index=df.index[20:25].tolist())

            # 清理
            df_cleaned = self.cleaner.clean(df)
            report = self.cleaner.analyze_quality(df_cleaned)

            results[symbol] = {
                'df': df_cleaned,
                'report': report
            }

        # 驗證所有交易對都成功處理
        self.assertEqual(len(results), 3)
        for symbol, result in results.items():
            self.assertGreater(len(result['df']), 0, f"{symbol} 應該有資料")
            self.assertIsNotNone(result['report'], f"{symbol} 應該有品質報告")

    def test_incremental_cleaning(self):
        """測試增量清理（模擬實時資料接收）"""
        # 初始資料
        timestamps1 = pd.date_range(start=datetime(2024, 1, 1), periods=50, freq='4h')
        df1 = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, 50),
            'high': np.random.uniform(45000, 50000, 50),
            'low': np.random.uniform(35000, 40000, 50),
            'close': np.random.uniform(40000, 45000, 50),
            'volume': np.random.uniform(1000, 5000, 50)
        }, index=timestamps1)

        max_values1 = df1[['open', 'high', 'close']].max(axis=1)
        min_values1 = df1[['open', 'low', 'close']].min(axis=1)
        df1['high'] = max_values1
        df1['low'] = min_values1

        # 第一次清理
        df1_cleaned = self.cleaner.clean(df1)

        # 新增資料
        timestamps2 = pd.date_range(
            start=timestamps1[-1] + timedelta(hours=4),
            periods=20,
            freq='4h'
        )
        df2 = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, 20),
            'high': np.random.uniform(45000, 50000, 20),
            'low': np.random.uniform(35000, 40000, 20),
            'close': np.random.uniform(40000, 45000, 20),
            'volume': np.random.uniform(1000, 5000, 20)
        }, index=timestamps2)

        max_values2 = df2[['open', 'high', 'close']].max(axis=1)
        min_values2 = df2[['open', 'low', 'close']].min(axis=1)
        df2['high'] = max_values2
        df2['low'] = min_values2

        # 合併並清理
        df_combined = pd.concat([df1_cleaned, df2])
        df_combined_cleaned = self.cleaner.clean(df_combined)

        # 驗證合併後的資料
        self.assertGreater(len(df_combined_cleaned), len(df1_cleaned))
        self.assertFalse(df_combined_cleaned.index.duplicated().any())

    def test_compatibility_with_different_timeframes(self):
        """測試不同時間框架的相容性"""
        timeframes = ['1h', '4h', '1d']

        for tf in timeframes:
            cleaner = DataCleaner(timeframe=tf, verbose=False)

            # 建立對應時間框架的資料
            timestamps = pd.date_range(start=datetime(2024, 1, 1), periods=100, freq=tf)
            df = pd.DataFrame({
                'open': np.random.uniform(40000, 45000, 100),
                'high': np.random.uniform(45000, 50000, 100),
                'low': np.random.uniform(35000, 40000, 100),
                'close': np.random.uniform(40000, 45000, 100),
                'volume': np.random.uniform(1000, 5000, 100)
            }, index=timestamps)

            max_vals_tf = df[['open', 'high', 'close']].max(axis=1)
            min_vals_tf = df[['open', 'low', 'close']].min(axis=1)
            df['high'] = max_vals_tf
            df['low'] = min_vals_tf

            # 移除一些資料
            df = df.drop(index=df.index[10:12].tolist())

            # 執行清理
            df_cleaned = cleaner.clean(df)
            report = cleaner.analyze_quality(df_cleaned)

            # 驗證
            self.assertGreater(len(df_cleaned), 0, f"{tf} 應該有資料")
            self.assertIsNotNone(report, f"{tf} 應該有品質報告")

    def test_error_recovery(self):
        """測試錯誤恢復能力"""
        # 建立有問題的資料
        timestamps = pd.date_range(start=datetime(2024, 1, 1), periods=50, freq='4h')
        df = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, 50),
            'high': np.random.uniform(45000, 50000, 50),
            'low': np.random.uniform(35000, 40000, 50),
            'close': np.random.uniform(40000, 45000, 50),
            'volume': np.random.uniform(1000, 5000, 50)
        }, index=timestamps)

        # 加入無效 OHLC
        df.loc[df.index[10], 'high'] = df.loc[df.index[10], 'low'] - 1000

        # 加入負數價格
        df.loc[df.index[20], 'close'] = -100

        # 加入無效成交量
        df.loc[df.index[30], 'volume'] = -1000

        # 即使有錯誤資料，清理流程也應該能完成
        try:
            df_cleaned = self.cleaner.clean(df)
            report = self.cleaner.analyze_quality(df_cleaned)

            # 應該偵測到問題
            self.assertGreater(len(report.issues), 0)
            self.assertLess(report.quality_score, 100)

        except Exception as e:
            self.fail(f"清理流程不應該拋出例外: {e}")


class TestCleanerMemoryEfficiency(unittest.TestCase):
    """測試記憶體效率"""

    def test_large_dataset_memory(self):
        """測試大資料集不會產生記憶體洩漏"""
        cleaner = DataCleaner(timeframe='4h', verbose=False)

        # 處理多批資料
        for batch in range(5):
            timestamps = pd.date_range(
                start=datetime(2024, 1, 1) + timedelta(days=batch * 10),
                periods=1000,
                freq='4h'
            )
            df = pd.DataFrame({
                'open': np.random.uniform(40000, 45000, 1000),
                'high': np.random.uniform(45000, 50000, 1000),
                'low': np.random.uniform(35000, 40000, 1000),
                'close': np.random.uniform(40000, 45000, 1000),
                'volume': np.random.uniform(1000, 5000, 1000)
            }, index=timestamps)

            # 清理
            df_cleaned = cleaner.clean(df)

            # 刪除以釋放記憶體
            del df
            del df_cleaned

        # 測試應該能順利完成，不會 OOM
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
