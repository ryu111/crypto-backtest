"""
JSON 到 DuckDB 遷移腳本

將 learning/experiments.json (622 筆記錄) 遷移到 DuckDB。

執行範例:
    python scripts/migrate_json_to_duckdb.py
    python scripts/migrate_json_to_duckdb.py --dry-run  # 不實際寫入
    python scripts/migrate_json_to_duckdb.py --db-path data/test.duckdb
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import argparse

# 將專案根目錄加入 sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.types import ExperimentRecord
from src.db.repository import Repository

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MigrationStats:
    """遷移統計"""

    def __init__(self):
        self.total = 0
        self.success = 0
        self.failed = 0
        self.skipped = 0
        self.errors: List[Dict[str, Any]] = []

    def add_success(self):
        self.success += 1

    def add_failed(self, record_id: str, error: str):
        self.failed += 1
        self.errors.append({
            'id': record_id,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })

    def add_skipped(self):
        self.skipped += 1

    def report(self) -> str:
        """生成報告"""
        report = [
            "\n" + "="*60,
            "遷移統計報告",
            "="*60,
            f"總筆數: {self.total}",
            f"成功:   {self.success} ({self.success/self.total*100:.1f}%)" if self.total > 0 else "成功:   0",
            f"失敗:   {self.failed} ({self.failed/self.total*100:.1f}%)" if self.total > 0 else "失敗:   0",
            f"跳過:   {self.skipped} ({self.skipped/self.total*100:.1f}%)" if self.total > 0 else "跳過:   0",
        ]

        if self.errors:
            report.append("\n失敗詳情:")
            for err in self.errors[:10]:  # 只顯示前 10 個錯誤
                report.append(f"  - {err['id']}: {err['error']}")
            if len(self.errors) > 10:
                report.append(f"  ... 還有 {len(self.errors) - 10} 個錯誤")

        report.append("="*60)
        return "\n".join(report)


def normalize_record(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    標準化記錄格式（處理舊格式相容性）

    轉換規則:
    - 舊格式使用 'parameters' → 轉為 'strategy.params'
    - 舊格式使用 'notes' → 轉為 'insights'
    - 確保必要欄位存在

    Args:
        data: 原始記錄

    Returns:
        標準化後的記錄
    """
    normalized = data.copy()

    # 向後相容：parameters → strategy.params
    if 'parameters' in normalized and 'params' not in normalized.get('strategy', {}):
        if 'strategy' not in normalized:
            normalized['strategy'] = {}
        normalized['strategy']['params'] = normalized.pop('parameters')

    # 向後相容：notes → insights
    if 'notes' in normalized and 'insights' not in normalized:
        normalized['insights'] = normalized.pop('notes')

    # 確保必要欄位存在
    if 'insights' not in normalized:
        normalized['insights'] = []

    if 'tags' not in normalized:
        normalized['tags'] = []

    # 確保 strategy 結構完整
    if 'strategy' in normalized:
        strategy = normalized['strategy']
        if 'version' not in strategy:
            strategy['version'] = '1.0'

    return normalized


def deduplicate_experiments(experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    去除重複的實驗記錄（保留最新）

    Args:
        experiments: 實驗記錄列表

    Returns:
        去重後的記錄列表
    """
    # 使用字典，key 為 id，value 為記錄
    # 後面的記錄會覆蓋前面的（保留最新）
    unique = {}
    duplicates = 0

    for exp in experiments:
        exp_id = exp.get('id')
        if exp_id in unique:
            duplicates += 1
        unique[exp_id] = exp

    if duplicates > 0:
        logger.info(f"發現 {duplicates} 筆重複記錄，已去重")

    return list(unique.values())


def migrate_json_to_duckdb(
    json_path: Path,
    db_path: Path,
    dry_run: bool = False,
    deduplicate: bool = True
) -> MigrationStats:
    """
    執行遷移

    Args:
        json_path: experiments.json 路徑
        db_path: DuckDB 資料庫路徑
        dry_run: 是否為乾跑模式（不實際寫入）
        deduplicate: 是否去重（預設: True）

    Returns:
        遷移統計
    """
    stats = MigrationStats()

    # 讀取 JSON 檔案
    logger.info(f"讀取 {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"讀取 JSON 失敗: {e}")
        return stats

    experiments = data.get('experiments', [])
    logger.info(f"原始記錄: {len(experiments)} 筆")

    # 去重
    if deduplicate:
        experiments = deduplicate_experiments(experiments)

    stats.total = len(experiments)
    logger.info(f"待遷移記錄: {stats.total} 筆")

    if dry_run:
        logger.info("【乾跑模式】不實際寫入資料庫")

    # 連接 DuckDB
    if not dry_run:
        logger.info(f"連接 DuckDB: {db_path}")
        try:
            repo = Repository(str(db_path))
        except Exception as e:
            logger.error(f"連接資料庫失敗: {e}")
            return stats

    # 遷移每筆記錄
    for i, exp_data in enumerate(experiments, 1):
        record_id = exp_data.get('id', f'unknown_{i}')

        try:
            # 標準化格式
            normalized = normalize_record(exp_data)

            # 轉換為 ExperimentRecord
            record = ExperimentRecord.from_dict(normalized)

            # 插入資料庫（如果不是乾跑）
            if not dry_run:
                repo.insert_experiment(record)

            stats.add_success()

            # 每 50 筆記錄顯示進度
            if i % 50 == 0:
                logger.info(f"進度: {i}/{stats.total} ({i/stats.total*100:.1f}%)")

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"記錄 {record_id} 遷移失敗: {error_msg}")
            stats.add_failed(record_id, error_msg)

    # 關閉連接
    if not dry_run:
        repo.close()

    return stats


def verify_migration(json_path: Path, db_path: Path, deduplicate: bool = True) -> bool:
    """
    驗證遷移結果

    驗證項目:
    1. 筆數是否一致（考慮去重）
    2. 隨機抽樣檢查資料完整性

    Args:
        json_path: 原始 JSON 路徑
        db_path: DuckDB 路徑
        deduplicate: 是否啟用了去重

    Returns:
        是否驗證通過
    """
    logger.info("\n開始驗證遷移結果...")

    # 讀取 JSON 筆數
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    experiments = data.get('experiments', [])
    if deduplicate:
        experiments = deduplicate_experiments(experiments)

    json_count = len(experiments)

    # 讀取 DuckDB 筆數
    repo = Repository(str(db_path))
    db_result = repo.conn.execute("SELECT COUNT(*) FROM experiments").fetchone()
    db_count = db_result[0] if db_result else 0
    repo.close()

    logger.info(f"JSON 筆數: {json_count}")
    logger.info(f"DuckDB 筆數: {db_count}")

    if json_count != db_count:
        logger.error(f"❌ 筆數不一致！差異: {abs(json_count - db_count)}")
        return False

    logger.info("✅ 筆數驗證通過")

    # 隨機抽樣檢查（抽取 5 筆）
    logger.info("\n隨機抽樣檢查...")
    import random
    sample_ids = random.sample([exp['id'] for exp in experiments], min(5, json_count))

    repo = Repository(str(db_path))
    all_passed = True

    for exp_id in sample_ids:
        # 從 JSON 找記錄
        json_record = next((exp for exp in experiments if exp['id'] == exp_id), None)

        # 從 DuckDB 找記錄
        db_record = repo.get_experiment(exp_id)

        if not db_record:
            logger.error(f"❌ {exp_id}: DuckDB 中找不到")
            all_passed = False
            continue

        # 檢查關鍵欄位
        json_sharpe = json_record['results'].get('sharpe_ratio')
        db_sharpe = db_record.sharpe_ratio

        if abs(json_sharpe - db_sharpe) > 1e-6:
            logger.error(f"❌ {exp_id}: Sharpe 不一致 (JSON: {json_sharpe}, DB: {db_sharpe})")
            all_passed = False
        else:
            logger.info(f"✅ {exp_id}: 資料一致")

    repo.close()

    if all_passed:
        logger.info("\n✅ 抽樣驗證通過")
    else:
        logger.error("\n❌ 抽樣驗證失敗")

    return all_passed


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='遷移 experiments.json 到 DuckDB')
    parser.add_argument(
        '--json-path',
        type=str,
        default='learning/experiments.json',
        help='JSON 檔案路徑（預設: learning/experiments.json）'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default='data/experiments.duckdb',
        help='DuckDB 路徑（預設: data/experiments.duckdb）'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='乾跑模式（不實際寫入）'
    )
    parser.add_argument(
        '--skip-verify',
        action='store_true',
        help='跳過驗證步驟'
    )
    parser.add_argument(
        '--no-deduplicate',
        action='store_true',
        help='不進行去重（預設會去重）'
    )

    args = parser.parse_args()

    # 轉換為絕對路徑
    json_path = project_root / args.json_path
    db_path = project_root / args.db_path

    # 檢查 JSON 檔案存在
    if not json_path.exists():
        logger.error(f"JSON 檔案不存在: {json_path}")
        sys.exit(1)

    # 開始遷移
    logger.info("="*60)
    logger.info("JSON → DuckDB 遷移")
    logger.info("="*60)
    logger.info(f"來源: {json_path}")
    logger.info(f"目標: {db_path}")

    start_time = datetime.now()
    stats = migrate_json_to_duckdb(
        json_path,
        db_path,
        dry_run=args.dry_run,
        deduplicate=not args.no_deduplicate
    )
    end_time = datetime.now()

    # 顯示統計報告
    print(stats.report())
    logger.info(f"執行時間: {(end_time - start_time).total_seconds():.2f} 秒")

    # 驗證遷移結果
    if not args.dry_run and not args.skip_verify and stats.success > 0:
        verify_passed = verify_migration(
            json_path,
            db_path,
            deduplicate=not args.no_deduplicate
        )
        if not verify_passed:
            logger.error("\n遷移驗證失敗，請檢查錯誤日誌")
            sys.exit(1)

    # 判斷是否成功
    if stats.failed > 0:
        logger.error(f"\n遷移完成，但有 {stats.failed} 筆失敗")
        sys.exit(1)

    logger.info("\n✅ 遷移成功完成！")


if __name__ == '__main__':
    main()
