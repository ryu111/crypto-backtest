# 向量化與 Polars 優化

## 概述

回測引擎現已支援向量化計算，大幅提升效能。

## 效能提升

### 實測數據（10,000 rows）

| 後端 | 執行時間 | 加速比 |
|------|----------|--------|
| Pandas + VectorBT（基準） | 1,791 ms | 1.00x |
| **Pandas + Vectorized** | **40 ms** | **45x** |
| Polars + Vectorized | 待優化 | TBD |

**結論**：向量化計算達成 **45x 效能提升**，超越 5-10x 目標。

## 使用方式

### 1. 啟用向量化（預設開啟）

```python
from src.backtester.engine import BacktestEngine, BacktestConfig

config = BacktestConfig(
    symbol='BTCUSDT',
    timeframe='1h',
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    initial_capital=10000,
    leverage=3,
    vectorized=True,  # 啟用向量化（預設 True）
    use_polars=False  # Pandas 後端（預設 True，目前建議 False）
)

engine = BacktestEngine(config)
result = engine.run(strategy, data=df)
```

### 2. 向量化技術指標

```python
from src.backtester.vectorized import (
    vectorized_sma,
    vectorized_ema,
    vectorized_rsi,
    vectorized_bollinger_bands,
    vectorized_atr,
    vectorized_macd
)

# 使用向量化 SMA
sma_20 = vectorized_sma(df['close'], 20)

# 使用向量化 RSI
rsi_14 = vectorized_rsi(df['close'], 14)

# 使用向量化 MACD
macd, signal, hist = vectorized_macd(df['close'])
```

### 3. 向量化部位與損益計算

```python
from src.backtester.vectorized import (
    vectorized_positions,
    vectorized_pnl
)

# 計算部位
signals = pd.Series([1, 0, -1, 0, 1, ...])  # 1=做多, -1=做空, 0=平倉
positions = vectorized_positions(signals, position_mode="one-way")

# 計算損益
pnl = vectorized_pnl(
    positions,
    prices=df['close'],
    leverage=3.0,
    fees=0.0004
)
```

## 技術細節

### 向量化原則

1. **避免 Python 迴圈**
   ```python
   # 不好：Python 迴圈
   for i in range(len(df)):
       if df.loc[i, 'close'] > sma[i]:
           signal[i] = 1

   # 好：向量化操作
   signal = (df['close'] > sma).astype(int)
   ```

2. **使用內建方法**
   ```python
   # Pandas
   sma = df['close'].rolling(20).mean()

   # Polars（未來）
   sma = df['close'].rolling_mean(window_size=20)
   ```

3. **批次計算**
   ```python
   # 一次計算所有指標
   df = df.assign(
       sma_fast=df['close'].rolling(10).mean(),
       sma_slow=df['close'].rolling(30).mean(),
       rsi=vectorized_rsi(df['close'], 14)
   )
   ```

### 記憶體管理

- 64GB 記憶體可載入完整資料集（100 萬行 ≈ 46 MB）
- 不需分批處理
- 使用 `.copy()` 避免 SettingWithCopyWarning

### Polars 整合（待完成）

Polars 後端目前因為以下原因暫時停用：
- VectorBT 相依性仍需 Pandas
- 資料轉換 overhead
- Polars Expr vs Series 語法差異

**預計改進**：
- 完全移除 VectorBT 依賴
- 純 Polars 實作指標計算
- 預期額外 2-3x 提升

## API 參考

### BacktestConfig 參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `vectorized` | bool | True | 啟用向量化計算 |
| `use_polars` | bool | True | 使用 Polars 後端（目前建議 False） |

### 向量化函數

所有函數支援 `Union[pd.Series, pl.Series]` 輸入。

#### vectorized_sma(series, period)
計算簡單移動平均線。

**參數**:
- `series`: 價格序列
- `period`: 週期

**回傳**: SMA 序列

#### vectorized_ema(series, period)
計算指數移動平均線。

#### vectorized_rsi(series, period=14)
計算相對強弱指標（0-100）。

#### vectorized_bollinger_bands(series, period=20, std_dev=2.0)
計算布林通道。

**回傳**: `(upper, middle, lower)`

#### vectorized_atr(high, low, close, period=14)
計算真實波動幅度均值。

#### vectorized_macd(series, fast=12, slow=26, signal=9)
計算 MACD 指標。

**回傳**: `(macd_line, signal_line, histogram)`

## 效能基準測試

### 執行測試

```bash
# 向量化效能測試
python benchmarks/benchmark_vectorized.py

# 單元測試
pytest tests/test_vectorized_performance.py -v
```

### 預期結果

- 小資料集（10k）：30-50x 提升
- 中資料集（100k）：40-60x 提升
- 大資料集（1M）：50-100x 提升

## 最佳實踐

### 策略開發

```python
class MyStrategy:
    name = "Vectorized Strategy"
    params = {'sma_period': 20, 'rsi_period': 14}

    def generate_signals(self, df):
        """使用向量化指標"""
        # 一次計算所有指標
        close = df['close']
        sma = vectorized_sma(close, self.params['sma_period'])
        rsi = vectorized_rsi(close, self.params['rsi_period'])

        # 向量化訊號
        long_entry = (close > sma) & (rsi < 30)
        long_exit = (close < sma) | (rsi > 70)
        short_entry = pd.Series(False, index=df.index)
        short_exit = pd.Series(False, index=df.index)

        return long_entry, long_exit, short_entry, short_exit
```

### 避免常見錯誤

```python
# 錯誤：逐行計算
for i in range(len(df)):
    sma[i] = df['close'][i-20:i].mean()

# 正確：向量化
sma = df['close'].rolling(20).mean()

# 錯誤：混用索引
signal[df['close'] > sma] = 1  # 可能出錯

# 正確：對齊索引
signal = (df['close'] > sma).astype(int)
```

## 已知限制

1. **Polars 後端**：目前因 VectorBT 相依性暫停使用
2. **資料轉換**：Polars ↔ Pandas 轉換有 overhead
3. **策略相容**：策略需支援向量化操作

## 路線圖

### v1.1（當前）
- [x] Pandas 向量化實作
- [x] 核心指標函數
- [x] 效能基準測試
- [ ] Polars 完整整合

### v1.2（未來）
- [ ] 完全移除 VectorBT
- [ ] 純 Polars 實作
- [ ] GPU 加速（CuDF）
- [ ] 分散式回測（Dask）

## 貢獻

歡迎提交 PR 優化向量化實作！

重點領域：
- Polars 整合改進
- 更多技術指標
- 效能優化
- 文檔補充

## 授權

MIT License
