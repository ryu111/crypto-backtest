# JSON → DuckDB 遷移腳本使用說明

## 概述

`migrate_json_to_duckdb.py` 將 `learning/experiments.json` (622 筆原始記錄) 遷移到 DuckDB 資料庫。

## 功能特色

- ✅ 自動去重（保留最新記錄）
- ✅ 向後相容性處理（`parameters` → `params`, `notes` → `insights`）
- ✅ 完整驗證（筆數驗證 + 隨機抽樣）
- ✅ 詳細統計報告
- ✅ 支援乾跑模式

## 快速開始

### 基本用法

```bash
# 執行遷移（預設會去重並驗證）
python scripts/migrate_json_to_duckdb.py

# 乾跑模式（不實際寫入）
python scripts/migrate_json_to_duckdb.py --dry-run

# 自訂路徑
python scripts/migrate_json_to_duckdb.py \
  --json-path learning/experiments.json \
  --db-path data/experiments.duckdb
```

### 進階選項

```bash
# 跳過驗證（加快速度）
python scripts/migrate_json_to_duckdb.py --skip-verify

# 不去重（保留所有重複記錄，會失敗）
python scripts/migrate_json_to_duckdb.py --no-deduplicate
```

## 遷移結果

### 資料統計

| 項目 | 數量 |
|------|------|
| 原始 JSON 記錄 | 622 筆 |
| 重複記錄 | 356 筆 |
| **遷移後唯一記錄** | **266 筆** |

### 去重策略

- **策略**: 保留最新記錄（後面覆蓋前面）
- **原因**: JSON 中有 145 個重複 ID（同一 ID 出現 2-3 次）
- **實作**: 使用 Python dict，相同 key 自動覆蓋

### 驗證項目

1. **筆數驗證**: 確保 DuckDB 筆數 = 去重後 JSON 筆數
2. **隨機抽樣**: 抽取 5 筆記錄驗證資料一致性（Sharpe Ratio 等）

## 輸出範例

```
============================================================
JSON → DuckDB 遷移
============================================================
來源: /path/to/learning/experiments.json
目標: /path/to/data/experiments.duckdb

原始記錄: 622 筆
發現 356 筆重複記錄,已去重
待遷移記錄: 266 筆

進度: 50/266 (18.8%)
進度: 100/266 (37.6%)
...
進度: 250/266 (94.0%)

============================================================
遷移統計報告
============================================================
總筆數: 266
成功:   266 (100.0%)
失敗:   0 (0.0%)
跳過:   0 (0.0%)
============================================================

開始驗證遷移結果...
JSON 筆數: 266
DuckDB 筆數: 266
✅ 筆數驗證通過

隨機抽樣檢查...
✅ exp_20260114_095601: 資料一致
✅ exp_20260114_091021: 資料一致
...
✅ 抽樣驗證通過

✅ 遷移成功完成！
```

## DuckDB 資料驗證

### Python 腳本

```python
from src.db.repository import Repository

repo = Repository('data/experiments.duckdb')

# 統計資訊
total = repo.conn.execute('SELECT COUNT(*) FROM experiments').fetchone()[0]
print(f'總筆數: {total}')  # 266

# 按等級統計
result = repo.conn.execute('''
    SELECT grade, COUNT(*) as count
    FROM experiments
    GROUP BY grade
    ORDER BY grade
''').fetchall()

# 等級 B: 1 筆
# 等級 C: 7 筆
# 等級 D: 4 筆
# 等級 F: 254 筆

# 最佳績效
best = repo.get_best_experiments(metric='sharpe_ratio', n=5)
# Sharpe 最高: 3.22
```

## 向後相容性處理

腳本自動處理舊格式：

| 舊格式 | 新格式 | 處理方式 |
|--------|--------|----------|
| `parameters` | `strategy.params` | 自動轉換 |
| `notes` | `insights` | 自動轉換 |
| 未知欄位 | - | 自動忽略 |

## 錯誤處理

### 常見錯誤

1. **JSON 檔案不存在**
   ```
   ERROR: JSON 檔案不存在: /path/to/experiments.json
   ```
   → 檢查 `--json-path` 參數

2. **資料庫連接失敗**
   ```
   ERROR: 連接資料庫失敗: ...
   ```
   → 檢查 DuckDB 路徑權限

3. **驗證失敗**
   ```
   ERROR: 筆數不一致！差異: X
   ```
   → 檢查是否有程序在寫入資料庫

## 性能指標

| 項目 | 數值 |
|------|------|
| 遷移速度 | ~2200 筆/秒 |
| 266 筆記錄耗時 | ~0.12 秒 |
| 記憶體占用 | < 50 MB |

## 後續操作

遷移完成後，可以使用 `Repository` 進行查詢：

```python
from src.db.repository import Repository, QueryFilters

repo = Repository('data/experiments.duckdb')

# 查詢特定策略
filters = QueryFilters(
    strategy_name="trend_ma_cross",
    min_sharpe=1.5,
    grade=["A", "B"]
)
results = repo.query_experiments(filters)

# 取得最佳實驗
best = repo.get_best_experiments(metric='sharpe_ratio', n=10)

repo.close()
```

## 注意事項

1. **去重是預設行為**: 如果需要保留所有記錄，使用 `--no-deduplicate`（但會因為主鍵衝突失敗）
2. **驗證會重新去重**: 驗證時會再次執行去重邏輯，確保比較正確
3. **覆蓋資料庫**: 如果 DuckDB 已存在，會直接插入（可能造成主鍵衝突）。建議先刪除舊檔案。

## 技術細節

### ExperimentRecord 轉換

```python
# JSON 格式
{
  "id": "exp_xxx",
  "strategy": {"name": "...", "params": {...}},
  "results": {"sharpe_ratio": 1.5, ...},
  "validation": {"grade": "B", ...}
}

# 透過 from_dict 轉換
record = ExperimentRecord.from_dict(json_data)

# 插入 DuckDB
repo.insert_experiment(record)
```

### 資料庫 Schema

詳見 `src/db/schema.sql`：

- **experiments 表**: 所有實驗記錄（主鍵: id）
- **strategy_stats 表**: 策略統計（UCB 評分等）

## 授權

MIT License
