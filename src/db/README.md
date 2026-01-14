# DuckDB 資料層

實驗記錄和策略統計的資料庫操作模組。

## 功能

- ✅ 實驗記錄 CRUD 操作
- ✅ 策略統計追蹤
- ✅ 複雜條件查詢
- ✅ 最佳結果查詢
- ✅ JSON 欄位支援
- ✅ 自動 Schema 初始化

## 快速開始

### 1. 基本使用

```python
from src.db import Repository, QueryFilters
from src.types import ExperimentRecord, StrategyStats

# 建立 Repository
with Repository("data/experiments.duckdb") as repo:
    # 插入實驗記錄
    repo.insert_experiment(experiment)

    # 查詢實驗
    filters = QueryFilters(strategy_name="ma_cross", min_sharpe=1.5)
    results = repo.query_experiments(filters)

    # 獲取最佳結果
    top_10 = repo.get_best_experiments(metric="sharpe_ratio", n=10)
```

### 2. 插入實驗記錄

```python
from datetime import datetime
from src.types import ExperimentRecord

experiment = ExperimentRecord(
    id="exp_001",
    timestamp=datetime.now(),
    strategy={
        'name': 'ma_cross',
        'type': 'trend',
        'version': '1.0',
        'params': {'fast_period': 10, 'slow_period': 30}
    },
    config={
        'symbol': 'BTCUSDT',
        'timeframe': '4h',
        'start_date': '2023-01-01',
        'end_date': '2024-01-01',
    },
    results={
        'sharpe_ratio': 2.1,
        'total_return': 45.5,
        'max_drawdown': 12.3,
        'win_rate': 0.62,
        'profit_factor': 1.8,
        'total_trades': 120,
    },
    validation={
        'grade': 'A',
        'stages_passed': [1, 2, 3, 4, 5],
    }
)

with Repository("data/experiments.duckdb") as repo:
    repo.insert_experiment(experiment)
```

### 3. 查詢實驗

```python
from src.db import QueryFilters

# 基本查詢
filters = QueryFilters(
    strategy_name="ma_cross",
    min_sharpe=1.5,
    grade=["A", "B"],
    limit=50
)

with Repository("data/experiments.duckdb") as repo:
    results = repo.query_experiments(filters)
    for exp in results:
        print(f"{exp.id}: Sharpe {exp.sharpe_ratio}")
```

### 4. 獲取最佳結果

```python
# 獲取最佳 10 個實驗（依 Sharpe Ratio）
with Repository("data/experiments.duckdb") as repo:
    top_10 = repo.get_best_experiments(metric="sharpe_ratio", n=10)

    # 加入過濾條件
    filters = QueryFilters(strategy_type="trend")
    best_trend = repo.get_best_experiments(
        metric="sharpe_ratio",
        n=5,
        filters=filters
    )
```

### 5. 策略統計

```python
from src.types import StrategyStats

# 更新策略統計
stats = StrategyStats(
    name='ma_cross',
    attempts=10,
    successes=3,
    avg_sharpe=1.5,
    best_sharpe=2.1,
    best_params={'fast_period': 10, 'slow_period': 30}
)

with Repository("data/experiments.duckdb") as repo:
    repo.update_strategy_stats(stats)

    # 查詢策略統計
    stats = repo.get_strategy_stats('ma_cross')
    print(f"成功率: {stats.success_rate * 100}%")
```

## QueryFilters 參數

| 參數 | 類型 | 說明 |
|------|------|------|
| `strategy_name` | `str` | 策略名稱 |
| `strategy_type` | `str` | 策略類型 (trend/mean_reversion/...) |
| `symbol` | `str` | 交易標的 (BTCUSDT, ETHUSDT) |
| `timeframe` | `str` | 時間框架 (1h, 4h, 1d) |
| `min_sharpe` | `float` | 最小 Sharpe Ratio |
| `max_drawdown` | `float` | 最大回撤上限 |
| `grade` | `List[str]` | 驗證等級 (["A", "B"]) |
| `tags` | `List[str]` | 標籤篩選 |
| `start_date` | `str` | 開始時間 |
| `end_date` | `str` | 結束時間 |
| `limit` | `int` | 回傳數量上限 (預設 100) |
| `offset` | `int` | 偏移量（分頁用） |

## 資料庫結構

### experiments 表

```sql
CREATE TABLE experiments (
    id VARCHAR PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    strategy_name VARCHAR NOT NULL,
    strategy_type VARCHAR NOT NULL,
    params JSON,
    symbol VARCHAR NOT NULL,
    timeframe VARCHAR NOT NULL,
    sharpe_ratio DOUBLE,
    total_return DOUBLE,
    max_drawdown DOUBLE,
    grade VARCHAR,
    ...
);
```

### strategy_stats 表

```sql
CREATE TABLE strategy_stats (
    name VARCHAR PRIMARY KEY,
    attempts INTEGER,
    successes INTEGER,
    avg_sharpe DOUBLE,
    best_sharpe DOUBLE,
    best_params JSON,
    ucb_score DOUBLE,
    ...
);
```

## 效能考量

### 索引

已自動建立以下索引：

- `idx_experiments_symbol` - 標的查詢
- `idx_experiments_strategy` - 策略查詢
- `idx_experiments_grade` - 等級查詢
- `idx_experiments_timestamp` - 時間查詢
- `idx_experiments_sharpe` - 績效排序

### 查詢優化

```python
# 好：使用索引欄位過濾
filters = QueryFilters(strategy_name="ma_cross", min_sharpe=1.5)

# 好：限制回傳數量
filters = QueryFilters(limit=50)

# 避免：過度使用 tags 過濾（JSON 查詢較慢）
```

## 遷移自 JSON

從 `learning/experiments.json` 遷移到 DuckDB：

```python
import json
from src.db import Repository
from src.types import ExperimentRecord

# 讀取舊 JSON
with open("learning/experiments.json") as f:
    data = json.load(f)

# 遷移到 DuckDB
with Repository("data/experiments.duckdb") as repo:
    for exp_data in data:
        experiment = ExperimentRecord.from_dict(exp_data)
        repo.insert_experiment(experiment)
```

## 故障排除

### 模組未安裝

```bash
pip install duckdb
```

### Schema 錯誤

刪除資料庫檔案重新建立：

```bash
rm data/experiments.duckdb
# Repository 會自動重建 schema
```

### 查詢太慢

檢查是否使用索引欄位：

```python
# 查看查詢計劃
repo.conn.execute("EXPLAIN SELECT * FROM experiments WHERE sharpe_ratio > 1.5")
```

## 參考

- DuckDB 文件: https://duckdb.org/docs/
- 型別定義: `src/types/`
- 測試範例: `test_db_repository.py`
