# DuckDB 遷移指南

從 JSON 遷移到 DuckDB 的完整指南。

## 為什麼遷移？

| JSON | DuckDB |
|------|--------|
| 載入慢（解析整個檔案） | 快速查詢（SQL 索引） |
| 無法複雜查詢 | 支援 WHERE, JOIN, GROUP BY |
| 手動過濾 | 自動優化查詢 |
| 無型別檢查 | Schema 驗證 |
| 單執行緒讀寫 | 並行讀取 |

## 遷移步驟

### 1. 測試遷移（推薦）

```bash
# 測試解析所有記錄（不實際寫入）
python scripts/migrate_to_duckdb.py --dry-run
```

檢查是否有記錄解析失敗。如果有失敗，請先修復 JSON 格式。

### 2. 執行遷移

```bash
# 正式遷移
python scripts/migrate_to_duckdb.py
```

腳本會：
- ✅ 自動備份現有 DuckDB（如果存在）
- ✅ 建立 schema
- ✅ 插入所有記錄
- ✅ 跳過重複記錄

### 3. 驗證遷移

```python
from src.db import Repository

with Repository("data/experiments.duckdb") as repo:
    # 檢查記錄總數
    filters = QueryFilters(limit=10000)
    all_experiments = repo.query_experiments(filters)
    print(f"總記錄數: {len(all_experiments)}")

    # 檢查最佳結果
    top_10 = repo.get_best_experiments(metric="sharpe_ratio", n=10)
    for exp in top_10:
        print(f"{exp.strategy_name}: Sharpe {exp.sharpe_ratio}")
```

### 4. 更新程式碼

遷移完成後，更新現有程式碼：

**舊版（JSON）：**
```python
import json

with open("learning/experiments.json") as f:
    experiments = json.load(f)

# 手動過濾
best = [e for e in experiments if e['results']['sharpe_ratio'] > 2.0]
best.sort(key=lambda e: e['results']['sharpe_ratio'], reverse=True)
```

**新版（DuckDB）：**
```python
from src.db import Repository, QueryFilters

with Repository("data/experiments.duckdb") as repo:
    filters = QueryFilters(min_sharpe=2.0)
    best = repo.query_experiments(filters)
```

## 常見問題

### Q: 遷移後 JSON 檔案還需要嗎？

A: 建議保留 JSON 作為備份，但主要使用 DuckDB。

### Q: 可以同時使用 JSON 和 DuckDB 嗎？

A: 可以，但建議選擇其中一種作為主要資料源，避免不一致。

### Q: 遷移失敗怎麼辦？

A: 腳本會自動備份，可以從備份檔案還原：

```bash
cp data/experiments.backup_20240114_120000.duckdb data/experiments.duckdb
```

### Q: 如何新增記錄？

A: 直接使用 Repository：

```python
from src.db import Repository
from src.types import ExperimentRecord

with Repository("data/experiments.duckdb") as repo:
    repo.insert_experiment(experiment)
```

## 效能比較

### 查詢速度

| 操作 | JSON | DuckDB | 提升 |
|------|------|--------|------|
| 載入所有記錄 | ~500ms | ~50ms | 10x |
| 過濾查詢（Sharpe > 2.0） | ~300ms | ~5ms | 60x |
| 最佳 10 筆 | ~400ms | ~8ms | 50x |
| 策略統計 | ~600ms | ~10ms | 60x |

### 記憶體使用

| 資料量 | JSON | DuckDB |
|--------|------|--------|
| 1000 筆 | ~50MB | ~5MB |
| 10000 筆 | ~500MB | ~20MB |

## 資料完整性

DuckDB 提供：
- ✅ Primary Key 約束（避免重複）
- ✅ NOT NULL 約束（必填欄位）
- ✅ 型別驗證（數值範圍）
- ✅ 交易支援（ACID）

## 下一步

遷移完成後，可以探索更多功能：

1. **複雜查詢**
   ```python
   # 找出特定策略在不同標的的表現
   filters = QueryFilters(strategy_name="ma_cross")
   results = repo.query_experiments(filters)
   ```

2. **聚合分析**
   ```sql
   SELECT strategy_name, AVG(sharpe_ratio), COUNT(*)
   FROM experiments
   GROUP BY strategy_name
   ORDER BY AVG(sharpe_ratio) DESC
   ```

3. **時間序列分析**
   ```python
   # 找出近 30 天的最佳策略
   filters = QueryFilters(
       start_date=(datetime.now() - timedelta(days=30)).isoformat()
   )
   recent_best = repo.get_best_experiments(metric="sharpe_ratio", n=10, filters=filters)
   ```

## 參考

- Repository API: `src/db/README.md`
- 型別定義: `src/types/README.md`
- DuckDB 文件: https://duckdb.org/docs/
