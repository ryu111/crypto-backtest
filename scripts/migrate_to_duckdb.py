#!/usr/bin/env python3
"""
遷移 experiments.json 到 DuckDB

將現有的 JSON 格式實驗記錄遷移到 DuckDB 資料庫。

用法:
    python scripts/migrate_to_duckdb.py
    python scripts/migrate_to_duckdb.py --dry-run  # 測試不實際寫入
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from src.db import Repository
from src.types import ExperimentRecord


def migrate_experiments(
    json_path: str = "learning/experiments.json",
    db_path: str = "data/experiments.duckdb",
    dry_run: bool = False
):
    """
    執行遷移

    Args:
        json_path: JSON 檔案路徑
        db_path: DuckDB 資料庫路徑
        dry_run: 只測試不實際寫入
    """
    json_file = Path(json_path)
    db_file = Path(db_path)

    print("=== 遷移 experiments.json 到 DuckDB ===\n")
    print(f"來源: {json_file}")
    print(f"目標: {db_file}")
    print(f"模式: {'測試模式' if dry_run else '正式遷移'}\n")

    # 檢查 JSON 檔案
    if not json_file.exists():
        print(f"❌ JSON 檔案不存在: {json_file}")
        return

    # 讀取 JSON
    print("讀取 JSON 檔案...")
    with open(json_file) as f:
        data = json.load(f)

    total = len(data)
    print(f"✓ 找到 {total} 筆記錄\n")

    if dry_run:
        print("=== 測試模式：解析記錄 ===")
        success = 0
        failed = 0

        for i, exp_data in enumerate(data, 1):
            try:
                experiment = ExperimentRecord.from_dict(exp_data)
                print(f"✓ {i}/{total}: {experiment.id}")
                success += 1
            except Exception as e:
                print(f"✗ {i}/{total}: 解析失敗 - {e}")
                failed += 1

        print(f"\n成功: {success}, 失敗: {failed}")
        return

    # 正式遷移
    print("=== 開始遷移 ===")

    # 備份舊資料庫（如果存在）
    if db_file.exists():
        backup_path = db_file.with_suffix(f".backup_{datetime.now():%Y%m%d_%H%M%S}.duckdb")
        print(f"備份現有資料庫到: {backup_path}")
        import shutil
        shutil.copy(db_file, backup_path)

    # 建立資料庫
    with Repository(str(db_file)) as repo:
        success = 0
        failed = 0
        skipped = 0

        for i, exp_data in enumerate(data, 1):
            try:
                experiment = ExperimentRecord.from_dict(exp_data)

                # 檢查是否已存在
                existing = repo.get_experiment(experiment.id)
                if existing:
                    print(f"⊖ {i}/{total}: {experiment.id} - 已存在，跳過")
                    skipped += 1
                    continue

                # 插入記錄
                repo.insert_experiment(experiment)
                print(f"✓ {i}/{total}: {experiment.id}")
                success += 1

            except Exception as e:
                print(f"✗ {i}/{total}: 遷移失敗 - {e}")
                failed += 1

    print("\n=== 遷移完成 ===")
    print(f"成功: {success}")
    print(f"跳過: {skipped}")
    print(f"失敗: {failed}")

    if failed > 0:
        print("\n⚠️  有失敗記錄，請檢查日誌")


def main():
    parser = argparse.ArgumentParser(description="遷移 experiments.json 到 DuckDB")
    parser.add_argument(
        "--json",
        default="learning/experiments.json",
        help="JSON 檔案路徑（預設: learning/experiments.json）"
    )
    parser.add_argument(
        "--db",
        default="data/experiments.duckdb",
        help="DuckDB 資料庫路徑（預設: data/experiments.duckdb）"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="測試模式：只解析不實際寫入"
    )

    args = parser.parse_args()
    migrate_experiments(args.json, args.db, args.dry_run)


if __name__ == "__main__":
    main()
