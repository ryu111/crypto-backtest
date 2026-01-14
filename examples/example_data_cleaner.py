"""
資料清理模組測試範例

展示如何使用 DataCleaner 處理資料品質問題
"""

import sys
from pathlib import Path
from datetime import datetime

# 加入專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import DataFetcher, DataCleaner


def main():
    print("=== 資料清理測試 ===\n")

    # 1. 獲取測試資料
    print("步驟 1: 獲取資料")
    print("-" * 50)
    fetcher = DataFetcher(verbose=False)

    try:
        df = fetcher.fetch_ohlcv(
            symbol='BTCUSDT',
            timeframe='4h',
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 3, 1)
        )
        print(f"✓ 成功獲取 {len(df)} 筆資料")
        print(f"  時間範圍: {df.index.min()} ~ {df.index.max()}")
    except Exception as e:
        print(f"✗ 獲取資料失敗: {e}")
        return

    # 2. 初始品質分析
    print("\n步驟 2: 初始品質分析")
    print("-" * 50)
    cleaner = DataCleaner(timeframe='4h', verbose=False)
    report_before = cleaner.analyze_quality(df)
    print(report_before)

    # 3. 執行清理
    print("\n步驟 3: 執行資料清理")
    print("-" * 50)
    df_cleaned = cleaner.clean(
        df,
        fill_short_gaps=True,
        mark_long_gaps=True
    )
    print(f"✓ 清理完成")
    print(f"  原始筆數: {len(df)}")
    print(f"  清理後筆數: {len(df_cleaned)}")
    print(f"  新增筆數: {len(df_cleaned) - len(df)}")

    # 4. 清理後品質分析
    print("\n步驟 4: 清理後品質分析")
    print("-" * 50)
    report_after = cleaner.analyze_quality(df_cleaned)
    print(report_after)

    # 5. 品質改善對比
    print("\n步驟 5: 品質改善對比")
    print("-" * 50)
    print(f"缺失率: {report_before.missing_rate:.2%} → {report_after.missing_rate:.2%}")
    print(f"Gap 數量: {report_before.gap_count} → {report_after.gap_count}")
    print(f"品質評分: {report_before.quality_score:.2f} → {report_after.quality_score:.2f}")

    improvement = report_after.quality_score - report_before.quality_score
    if improvement > 0:
        print(f"\n✓ 品質提升 {improvement:.2f} 分")
    elif improvement < 0:
        print(f"\n⚠ 品質下降 {abs(improvement):.2f} 分")
    else:
        print(f"\n- 品質無變化")

    # 6. 檢視標記資訊（如果有）
    if 'gap_flag' in df_cleaned.columns:
        print("\n步驟 6: Gap 標記統計")
        print("-" * 50)
        flag_counts = df_cleaned['gap_flag'].value_counts().sort_index()
        labels = {0: "正常", 1: "短缺失", 2: "長缺失", 3: "維護期"}

        for flag, count in flag_counts.items():
            label = labels.get(flag, "未知")
            percentage = count / len(df_cleaned) * 100
            print(f"  {label} (flag={flag}): {count:,} 筆 ({percentage:.2f}%)")

    # 7. 儲存清理後的資料（選擇性）
    save_option = input("\n是否儲存清理後的資料？(y/n): ").strip().lower()
    if save_option == 'y':
        output_path = project_root / 'data' / 'cleaned' / 'BTCUSDT_4h_cleaned.parquet'
        fetcher.save_to_parquet(df_cleaned, str(output_path))
        print(f"✓ 已儲存到 {output_path}")

    print("\n=== 測試完成 ===")


if __name__ == '__main__':
    main()
