---
name: data-pipeline
description: 資料管道管理。抓取、清洗、驗證歷史資料。當需要取得交易資料、處理資料品質問題、準備回測資料時使用。
---

# 資料管道

BTC/ETH 永續合約歷史資料的獲取、處理和驗證。

## Quick Start

```python
from src.data import DataPipeline

# 下載資料
pipeline = DataPipeline()
data = pipeline.fetch(
    symbol="BTCUSDT",
    timeframe="4h",
    start="2024-01-01",
    end="2025-12-31"
)

# 驗證資料品質
pipeline.validate(data)

# 儲存
pipeline.save(data, "data/ohlcv/BTCUSDT_4h.parquet")
```

## 資料來源比較

| 來源 | 優點 | 缺點 | 推薦度 |
|------|------|------|--------|
| Binance API | 免費、即時 | 有限歷史（約 2 年） | ⭐⭐⭐⭐ |
| CCXT | 支援多交易所 | 需處理差異 | ⭐⭐⭐⭐ |
| CoinAPI | 完整歷史、訂單簿 | 付費 | ⭐⭐⭐⭐⭐ |
| CoinGlass | 資金費率完整 | 付費進階功能 | ⭐⭐⭐⭐ |

## 必要資料類型

| 資料類型 | 用途 | 頻率 | 必要性 |
|----------|------|------|--------|
| OHLCV | 價格回測 | 1m/5m/1h/4h/1d | 必須 |
| Funding Rate | 持倉成本 | 8h | 永續必須 |
| Open Interest | 市場情緒 | 1h | 建議 |
| Liquidation | 極端風險 | 即時 | 可選 |

## 資料抓取

### Binance API

```python
import ccxt

exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

# 抓取 OHLCV
ohlcv = exchange.fetch_ohlcv(
    symbol='BTC/USDT',
    timeframe='4h',
    since=exchange.parse8601('2024-01-01T00:00:00Z'),
    limit=1000
)

# 抓取資金費率
funding = exchange.fetch_funding_rate_history(
    symbol='BTC/USDT',
    since=exchange.parse8601('2024-01-01T00:00:00Z'),
    limit=1000
)
```

### 批量下載腳本

```python
import pandas as pd
from datetime import datetime, timedelta

def fetch_all_ohlcv(symbol, timeframe, start_date, end_date):
    """批量下載 OHLCV 資料"""
    all_data = []
    current = start_date

    while current < end_date:
        ohlcv = exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=int(current.timestamp() * 1000),
            limit=1000
        )

        if not ohlcv:
            break

        all_data.extend(ohlcv)
        current = datetime.fromtimestamp(ohlcv[-1][0] / 1000)

        # 避免 rate limit
        time.sleep(0.1)

    df = pd.DataFrame(
        all_data,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    return df
```

## 資料品質檢查

```python
def validate_ohlcv(df):
    """驗證 OHLCV 資料品質"""
    issues = []

    # 1. 檢查缺失值
    if df.isnull().any().any():
        issues.append(f"缺失值: {df.isnull().sum().sum()} 筆")

    # 2. 檢查時間連續性
    expected_freq = pd.infer_freq(df.index)
    gaps = df.index.to_series().diff()
    if gaps.nunique() > 2:  # 允許一點誤差
        issues.append(f"時間不連續: {(gaps > gaps.median() * 2).sum()} 處")

    # 3. 檢查 OHLC 邏輯
    invalid_ohlc = (
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close']) |
        (df['high'] < df['low'])
    )
    if invalid_ohlc.any():
        issues.append(f"OHLC 邏輯錯誤: {invalid_ohlc.sum()} 筆")

    # 4. 檢查成交量
    if (df['volume'] <= 0).any():
        issues.append(f"成交量異常: {(df['volume'] <= 0).sum()} 筆")

    # 5. 檢查重複時間戳
    if df.index.duplicated().any():
        issues.append(f"重複時間戳: {df.index.duplicated().sum()} 筆")

    return issues
```

## 資料清洗

```python
def clean_ohlcv(df):
    """清洗 OHLCV 資料"""

    # 移除重複
    df = df[~df.index.duplicated(keep='first')]

    # 排序
    df = df.sort_index()

    # 填補缺失（前向填充）
    df = df.ffill()

    # 修正 OHLC 異常
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    # 填補缺失的時間戳
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=pd.infer_freq(df.index)
    )
    df = df.reindex(full_index, method='ffill')

    return df
```

## 資金費率處理

```python
def fetch_funding_rates(symbol, start_date, end_date):
    """抓取資金費率歷史"""
    all_rates = []
    current = start_date

    while current < end_date:
        rates = exchange.fetch_funding_rate_history(
            symbol=symbol,
            since=int(current.timestamp() * 1000),
            limit=1000
        )

        if not rates:
            break

        all_rates.extend(rates)
        current = datetime.fromtimestamp(rates[-1]['timestamp'] / 1000)
        time.sleep(0.1)

    df = pd.DataFrame(all_rates)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['fundingRate']].rename(columns={'fundingRate': 'rate'})

    return df
```

## 資料儲存格式

| 格式 | 優點 | 缺點 | 推薦場景 |
|------|------|------|----------|
| Parquet | 壓縮好、讀取快 | 需安裝 pyarrow | 大量資料 |
| CSV | 通用、可讀 | 大檔案慢 | 小量資料 |
| HDF5 | 支援大數據 | 複雜 | 進階分析 |

```python
# 儲存為 Parquet（推薦）
df.to_parquet("data/ohlcv/BTCUSDT_4h.parquet")

# 讀取
df = pd.read_parquet("data/ohlcv/BTCUSDT_4h.parquet")
```

## 資料更新策略

```python
def update_data(symbol, timeframe, data_path):
    """增量更新資料"""
    # 讀取現有資料
    if os.path.exists(data_path):
        existing = pd.read_parquet(data_path)
        last_timestamp = existing.index.max()
    else:
        existing = pd.DataFrame()
        last_timestamp = datetime(2020, 1, 1)

    # 抓取新資料
    new_data = fetch_all_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        start_date=last_timestamp,
        end_date=datetime.now()
    )

    # 合併
    combined = pd.concat([existing, new_data])
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.sort_index()

    # 儲存
    combined.to_parquet(data_path)

    return combined
```

## 資料品質檢查清單

- [ ] 無缺失值（或已插補）
- [ ] 時間戳連續
- [ ] OHLC 邏輯正確（H >= O,C >= L）
- [ ] 成交量 > 0
- [ ] 無重複時間戳
- [ ] 時區一致（建議 UTC）
- [ ] 資金費率與 K 線時間對齊

## 與其他 Skills 關係

### 被調用（上游）

| Skill | 場景 |
|-------|------|
| **回測核心** | 提供 OHLCV 和資金費率資料 |

### 本 Skill 提供

- OHLCV 歷史資料
- 資金費率歷史
- 資料品質驗證
- 資料清洗和補缺

### 資料流整合

```
資料管道
    ├─→ 抓取 OHLCV（CCXT/Binance API）
    ├─→ 抓取資金費率
    ├─→ 品質驗證和清洗
    └─→ 儲存（Parquet）
    ↓
回測核心（載入資料）
```

For 資料來源詳細比較 → read `references/data-sources.md`
