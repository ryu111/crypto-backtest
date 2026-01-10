# VectorBT 基礎教學

## 安裝

```bash
# 開源版
pip install vectorbt

# Pro 版（推薦）
pip install vectorbtpro
```

## 核心概念

### 向量化運算

VectorBT 的核心優勢是向量化運算，比傳統逐 bar 回測快 100-1000 倍。

```python
import vectorbtpro as vbt
import numpy as np

# 傳統方式：逐 bar 計算（慢）
for i in range(len(prices)):
    if prices[i] > ma[i]:
        signal[i] = 1

# VectorBT 方式：向量化（快）
signal = prices > ma
```

### 資料載入

```python
# 從 Binance 載入
data = vbt.BinanceData.pull(
    symbols=["BTCUSDT", "ETHUSDT"],
    start="2024-01-01",
    end="2025-12-31",
    timeframe="4h"
)

# 從 CSV 載入
data = vbt.CSVData.pull(
    paths="data/ohlcv/BTCUSDT_4h.csv",
    parse_dates=True,
    index_col="timestamp"
)

# 取得 close 價格
close = data.get("Close")
```

### 指標計算

```python
# 內建指標
ma_fast = vbt.MA.run(close, window=10)
ma_slow = vbt.MA.run(close, window=30)
rsi = vbt.RSI.run(close, window=14)
macd = vbt.MACD.run(close)
bbands = vbt.BBANDS.run(close, window=20, alpha=2)

# 取得指標值
fast_ma = ma_fast.ma
slow_ma = slow_ma.ma
rsi_value = rsi.rsi
```

### 訊號產生

```python
# 交叉訊號
entries = ma_fast.ma_crossed_above(ma_slow.ma)
exits = ma_fast.ma_crossed_below(ma_slow.ma)

# RSI 超買超賣
long_entries = rsi.rsi_crossed_below(30)
long_exits = rsi.rsi_crossed_above(70)
short_entries = rsi.rsi_crossed_above(70)
short_exits = rsi.rsi_crossed_below(30)
```

### Portfolio 建立

```python
pf = vbt.Portfolio.from_signals(
    close=close,
    entries=entries,
    exits=exits,

    # 做空支援
    short_entries=short_entries,
    short_exits=short_exits,

    # 資金設定
    init_cash=10000,

    # 槓桿
    leverage=5.0,
    leverage_mode='lazy',  # lazy: 需要時才用槓桿

    # 費用
    fees=0.0006,  # 0.06%
    slippage=0.001,  # 0.1%

    # 部位大小
    size=0.1,  # 10% 資金
    size_type='percent',

    # 頻率
    freq='4h'
)
```

### 績效分析

```python
# 完整統計
print(pf.stats())

# 關鍵指標
print(f"Total Return: {pf.total_return():.2%}")
print(f"Sharpe Ratio: {pf.sharpe_ratio():.2f}")
print(f"Max Drawdown: {pf.max_drawdown():.2%}")
print(f"Win Rate: {pf.trades.win_rate:.2%}")
print(f"Profit Factor: {pf.trades.profit_factor:.2f}")

# 交易記錄
trades = pf.trades.records_readable
print(trades.head(10))

# 月度報酬
monthly = pf.returns.resample('M').sum()
```

### 視覺化

```python
# 權益曲線
pf.plot().show()

# 交易標記
pf.trades.plot().show()

# 回撤圖
pf.drawdowns.plot().show()

# 月度熱力圖
pf.returns.vbt.heatmap(
    x_level='month',
    y_level='year'
).show()
```

## 參數優化

```python
# 測試多組參數
fast_windows = np.arange(5, 20, 2)
slow_windows = np.arange(20, 50, 5)

# 產生所有組合的 MA
fast_ma, slow_ma = vbt.MA.run_combs(
    close,
    window=np.concatenate([fast_windows, slow_windows]),
    r=2,  # 兩兩組合
    short_names=['fast', 'slow']
)

# 批量回測
entries = fast_ma.ma_crossed_above(slow_ma.ma)
exits = fast_ma.ma_crossed_below(slow_ma.ma)

pf = vbt.Portfolio.from_signals(close, entries, exits, fees=0.0006)

# 找最佳參數
best_idx = pf.sharpe_ratio().idxmax()
print(f"Best params: {best_idx}")

# 熱力圖
pf.total_return().vbt.heatmap().show()
```

## 自定義指標

```python
from vectorbtpro.indicators.factory import IndicatorFactory

# 建立自定義指標
SuperTrend = IndicatorFactory(
    class_name='SuperTrend',
    short_name='st',
    input_names=['high', 'low', 'close'],
    param_names=['period', 'multiplier'],
    output_names=['supertrend', 'direction']
).with_custom_func(supertrend_func)

# 使用自定義指標
st = SuperTrend.run(high, low, close, period=10, multiplier=3.0)
```

## 效能優化技巧

1. **使用 Numba 加速**：
```python
from numba import njit

@njit
def custom_signal(close, ma):
    result = np.empty(len(close), dtype=np.bool_)
    for i in range(len(close)):
        result[i] = close[i] > ma[i]
    return result
```

2. **分批處理大數據**：
```python
# 分割時間段
chunks = vbt.split(data, n=10)
results = []
for chunk in chunks:
    pf = vbt.Portfolio.from_signals(chunk, ...)
    results.append(pf.stats())
```

3. **使用 Ray 分散式**：
```python
import ray
ray.init()

# 分散式參數優化
results = vbt.Portfolio.from_signals(
    ...,
    ray_workers=4
)
```

## 常見問題

### Q: 如何處理缺失資料？
```python
# 前向填充
close = close.ffill()
# 或刪除
close = close.dropna()
```

### Q: 如何避免前瞻偏差？
```python
# 使用 shift 延遲訊號
entries = entries.shift(1)  # 延遲一根 K 線
```

### Q: 如何模擬限價單？
```python
pf = vbt.Portfolio.from_signals(
    ...,
    price=open_price,  # 用開盤價成交
    val_price=close_price  # 用收盤價估值
)
```
