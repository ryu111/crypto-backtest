# SharedDataPool 共享資料池使用指南

## 概述

`SharedDataPool` 提供跨進程零拷貝資料共享，適用於多進程回測場景。使用 Python `multiprocessing.shared_memory` 實現，所有 Worker 進程可以直接存取預載的資料，無需複製。

## 核心優勢

| 優勢 | 說明 |
|------|------|
| **零拷貝** | 多進程共享同一塊記憶體，不重複載入資料 |
| **自動轉換** | Parquet → float32 NumPy 陣列 |
| **進程安全** | 支援多進程並發讀取 |
| **易用性** | Context Manager 自動管理資源 |
| **跨進程** | 使用 JSON registry 共享 metadata |

## 快速開始

### 1. 基本使用（單進程）

```python
from src.data import create_shared_pool

# 建立並預載資料
pool = create_shared_pool(
    data_dir="data",
    symbols=["BTCUSDT", "ETHUSDT"],
    timeframes=["1h", "4h"]
)

# 使用資料（零拷貝）
btc_1h = pool.get("BTCUSDT_1h")
print(btc_1h.shape)  # (N, 5) - OHLCV

# 清理
pool.cleanup()
```

### 2. 多進程使用

**主進程（建立者）：**

```python
from src.data import create_shared_pool
import multiprocessing as mp

# 建立共享資料池
pool = create_shared_pool(
    symbols=["BTCUSDT"],
    timeframes=["1h"],
    pool_name="my_backtest"
)

# 啟動 Worker 進程
with mp.Pool(processes=4) as workers:
    workers.map(run_backtest, range(4))

# 清理（會 unlink 共享記憶體）
pool.cleanup()
```

**Worker 進程（使用者）：**

```python
from src.data import attach_to_pool

def run_backtest(worker_id):
    # 附加到現有共享資料池
    pool = attach_to_pool("my_backtest")

    # 零拷貝存取資料
    data = pool.get("BTCUSDT_1h")

    # 執行回測...

    # 清理（只 close，不 unlink）
    pool.cleanup()
```

### 3. Context Manager 用法

```python
from src.data import create_shared_pool

# 自動清理
with create_shared_pool(symbols=["BTCUSDT"]) as pool:
    btc_data = pool.get("BTCUSDT_1h")
    # ... 使用資料
# 離開時自動 cleanup
```

## API 參考

### create_shared_pool()

建立並預載共享資料池（最常用）。

```python
pool = create_shared_pool(
    data_dir="data",                    # 資料目錄
    symbols=["BTCUSDT", "ETHUSDT"],     # 標的列表
    timeframes=["5m", "1h", "4h"],      # 時間框架
    pool_name="backtest_data",          # 共享記憶體名稱
    include_funding=True                # 是否包含資金費率
)
```

### attach_to_pool()

附加到現有共享資料池（供子進程使用）。

```python
pool = attach_to_pool(pool_name="backtest_data")
```

### SharedDataPool 方法

| 方法 | 說明 |
|------|------|
| `put(key, data, columns)` | 放入資料到共享記憶體 |
| `get(key)` | 取得資料（零拷貝 NumPy 陣列） |
| `get_dataframe(key, columns)` | 取得 DataFrame 格式 |
| `list_keys()` | 列出所有資料鍵 |
| `get_info(key)` | 取得資料詳細資訊 |
| `get_total_size_mb()` | 取得總大小（MB） |
| `cleanup()` | 清理共享記憶體 |

## 高級用法

### 預載特定資料

```python
from src.data import SharedDataPool

pool = SharedDataPool(pool_name="custom")

# 只預載 OHLCV
pool.preload_ohlcv("data", ["BTCUSDT"], ["1h", "4h"])

# 只預載資金費率
pool.preload_funding_rates("data", ["BTCUSDT"])

# 手動放入自定義資料
import numpy as np
custom_data = np.random.randn(1000, 6).astype(np.float32)
pool.put("custom_key", custom_data, columns=["a", "b", "c", "d", "e", "f"])

pool.cleanup()
```

### 取得 DataFrame 格式

```python
# 取得 DataFrame（底層仍是零拷貝）
df = pool.get_dataframe("BTCUSDT_1h")
print(df.columns)  # ['open', 'high', 'low', 'close', 'volume']

# 指定欄位名
df = pool.get_dataframe("custom_key", columns=["col1", "col2", ...])
```

### 查詢資料資訊

```python
# 列出所有可用資料
keys = pool.list_keys()
print(keys)  # ['BTCUSDT_1h', 'BTCUSDT_4h', 'ETHUSDT_1h', ...]

# 查詢詳細資訊
info = pool.get_info("BTCUSDT_1h")
print(info)
# SharedDataInfo(key=BTCUSDT_1h, shape=(55610, 5), dtype=float32, size=1.06MB)

# 總共享記憶體大小
total_mb = pool.get_total_size_mb()
print(f"總大小: {total_mb:.2f} MB")
```

## 資料格式

### OHLCV 資料

- **檔案格式**：`data/ohlcv/{SYMBOL}_{TIMEFRAME}.parquet`
- **欄位**：`[open, high, low, close, volume]`
- **資料類型**：自動轉換為 `float32`
- **Shape**：`(N, 5)` - N 是資料點數量

### 資金費率資料

- **檔案格式**：`data/funding/{SYMBOL}_funding.parquet`
- **欄位**：`[rate]` 或其他（視檔案而定）
- **資料類型**：自動轉換為 `float32`

## 效能特性

### 記憶體使用

```python
# 範例：預載 2 年 BTC/ETH 多時間框架資料
pool = create_shared_pool(
    symbols=["BTCUSDT", "ETHUSDT"],
    timeframes=["5m", "15m", "30m", "1h", "4h", "1d"]
)

# 總大小約 20-30 MB（取決於資料點數量）
# 所有 Worker 共享這一份資料，無論啟動多少個 Worker
```

### 效能對比

| 模式 | 4 個 Worker 的記憶體使用 | 資料載入時間 |
|------|-------------------------|-------------|
| **傳統方式**（每個 Worker 載入） | 4x 資料大小 | 4x 載入時間 |
| **SharedDataPool** | 1x 資料大小 | 1x 載入時間 |

## 注意事項

### 1. 建立者 vs 使用者

- **建立者**（主進程）：使用 `create_shared_pool()`，負責 `unlink` 共享記憶體
- **使用者**（子進程）：使用 `attach_to_pool()`，只 `close`，不 `unlink`

### 2. 清理順序

```python
# 正確順序：
# 1. 所有子進程先 cleanup（close）
# 2. 主進程最後 cleanup（close + unlink）

# 錯誤順序：
# 主進程先 unlink → 子進程無法存取
```

### 3. Registry 檔案

- 位置：`/tmp/shared_data_pool/{pool_name}_registry.json`
- 用途：儲存 metadata 供子進程載入
- 清理：建立者 cleanup 時自動刪除

### 4. 資料是唯讀的

共享記憶體中的資料應視為**唯讀**。如果需要修改，請先複製：

```python
data = pool.get("BTCUSDT_1h")
modified = data.copy()  # 複製後再修改
modified[:, 0] *= 1.1   # 修改複製的資料
```

## 故障排除

### 問題 1：子進程找不到資料

**錯誤**：`KeyError: '找不到資料: BTCUSDT_1h'`

**原因**：子進程啟動時 registry 檔案尚未建立

**解決**：確保主進程完成 `create_shared_pool()` 後才啟動子進程

### 問題 2：記憶體洩漏警告

**警告**：`resource_tracker: There appear to be N leaked shared_memory objects`

**原因**：未正確 cleanup

**解決**：
```python
# 使用 context manager
with create_shared_pool(...) as pool:
    # 使用資料

# 或確保呼叫 cleanup
pool = create_shared_pool(...)
try:
    # 使用資料
finally:
    pool.cleanup()
```

### 問題 3：共享記憶體已被釋放

**錯誤**：`RuntimeError: 共享記憶體已被釋放`

**原因**：主進程過早 cleanup

**解決**：確保所有子進程完成後才 cleanup

## 完整範例

參見 `examples/shared_pool_example.py`，展示完整的多進程使用流程。

```bash
python examples/shared_pool_example.py
```

## 技術細節

### 零拷貝原理

```python
# 第一次 get 和第二次 get 共享記憶體
data1 = pool.get("BTCUSDT_1h")
data2 = pool.get("BTCUSDT_1h")

import numpy as np
assert np.shares_memory(data1, data2)  # True
```

### Registry 結構

```json
{
  "BTCUSDT_1h": {
    "key": "BTCUSDT_1h",
    "shape": [55610, 5],
    "dtype": "float32",
    "shm_name": "backtest_data_BTCUSDT_1h",
    "nbytes": 1113200,
    "columns": ["open", "high", "low", "close", "volume"]
  }
}
```

## 與其他模組整合

### 與 DataFetcher 整合

```python
from src.data import DataFetcher, create_shared_pool

# 1. 下載資料
fetcher = DataFetcher()
btc_data = fetcher.fetch_ohlcv("BTCUSDT", "1h")
fetcher.save_to_parquet(btc_data, "data/ohlcv/BTCUSDT_1h.parquet")

# 2. 預載到共享記憶體
pool = create_shared_pool(symbols=["BTCUSDT"], timeframes=["1h"])

# 3. 使用
data = pool.get("BTCUSDT_1h")
```

## 總結

`SharedDataPool` 是多進程回測的關鍵模組，透過零拷貝共享資料大幅降低記憶體使用並提升效能。適用於需要並行執行大量回測的場景。
