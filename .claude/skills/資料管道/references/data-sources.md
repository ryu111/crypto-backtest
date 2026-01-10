# 資料來源詳細比較

## 免費資料來源

### Binance API

```python
import ccxt

exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

# OHLCV
ohlcv = exchange.fetch_ohlcv('BTC/USDT', '4h', limit=1000)

# 資金費率
funding = exchange.fetch_funding_rate_history('BTC/USDT', limit=1000)
```

**優點**：
- 免費
- 即時數據
- API 穩定

**缺點**：
- 歷史限制約 2 年
- 需處理分頁

**適用場景**：日常回測、即時監控

### CoinGecko API

```python
import requests

url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
params = {"vs_currency": "usd", "days": 365}
response = requests.get(url, params=params)
```

**優點**：
- 免費額度足夠
- 多幣種支援

**缺點**：
- 只有 OHLCV
- 精度較低

**適用場景**：快速原型、概念驗證

## 付費資料來源

### CoinAPI

```python
import requests

headers = {"X-CoinAPI-Key": "YOUR_API_KEY"}
url = "https://rest.coinapi.io/v1/ohlcv/BINANCE_SPOT_BTC_USDT/history"
params = {
    "period_id": "4HRS",
    "time_start": "2024-01-01T00:00:00",
    "time_end": "2025-12-31T23:59:59"
}
response = requests.get(url, headers=headers, params=params)
```

**優點**：
- 完整歷史（2010+）
- 訂單簿數據
- 多交易所整合

**價格**：$79-499/月

**適用場景**：專業回測、學術研究

### CoinGlass（資金費率專用）

**優點**：
- 最完整的資金費率歷史
- 清算數據
- Open Interest

**價格**：$49-199/月

**適用場景**：永續合約策略

## 資料格式標準

### OHLCV 欄位

| 欄位 | 類型 | 說明 |
|------|------|------|
| timestamp | datetime | UTC 時間 |
| open | float | 開盤價 |
| high | float | 最高價 |
| low | float | 最低價 |
| close | float | 收盤價 |
| volume | float | 成交量 |

### 資金費率欄位

| 欄位 | 類型 | 說明 |
|------|------|------|
| timestamp | datetime | 結算時間 |
| rate | float | 費率（如 0.0001 = 0.01%） |
| mark_price | float | 標記價格 |

## 資料下載腳本

```python
# scripts/fetch_binance.py

import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time

def fetch_ohlcv_history(symbol, timeframe, start_date, end_date, save_path):
    """下載完整 OHLCV 歷史"""

    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })

    all_data = []
    current = start_date

    while current < end_date:
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=int(current.timestamp() * 1000),
                limit=1000
            )

            if not ohlcv:
                break

            all_data.extend(ohlcv)
            current = datetime.fromtimestamp(ohlcv[-1][0] / 1000) + timedelta(hours=4)

            print(f"Downloaded until {current}")
            time.sleep(0.5)  # Rate limit

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
            continue

    df = pd.DataFrame(
        all_data,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]

    df.to_parquet(save_path)
    print(f"Saved {len(df)} rows to {save_path}")

    return df

if __name__ == "__main__":
    fetch_ohlcv_history(
        symbol="BTC/USDT",
        timeframe="4h",
        start_date=datetime(2024, 1, 1),
        end_date=datetime.now(),
        save_path="data/ohlcv/BTCUSDT_4h.parquet"
    )
```
