---
name: indicator-lib
description: 技術指標庫。所有可用指標的參考和計算方法。當需要選擇指標、理解指標意義、建立自定義指標時使用。
---

# 技術指標庫

BTC/ETH 交易策略常用技術指標參考。

## 指標分類速查

### 趨勢指標

| 指標 | 全名 | 用途 | 常用參數 | 訊號 |
|------|------|------|----------|------|
| MA | Moving Average | 趨勢方向 | 20, 50, 200 | 價格 vs MA |
| EMA | Exponential MA | 趨勢（快速） | 12, 26 | 價格 vs EMA |
| MACD | Moving Average Convergence Divergence | 趨勢動能 | 12, 26, 9 | 交叉、柱狀 |
| ADX | Average Directional Index | 趨勢強度 | 14 | > 25 強趨勢 |
| Supertrend | - | 趨勢+止損 | 10, 3.0 | 方向翻轉 |
| Ichimoku | 一目均衡表 | 趨勢綜合 | 9, 26, 52 | 雲層突破 |

### 動量指標

| 指標 | 全名 | 用途 | 常用參數 | 訊號 |
|------|------|------|----------|------|
| RSI | Relative Strength Index | 超買超賣 | 14 | < 30 / > 70 |
| Stochastic | Stochastic Oscillator | 超買超賣 | 14, 3, 3 | K/D 交叉 |
| CCI | Commodity Channel Index | 價格偏離 | 20 | < -100 / > 100 |
| Williams %R | - | 超買超賣 | 14 | < -80 / > -20 |
| MFI | Money Flow Index | 資金流動 | 14 | < 20 / > 80 |
| ROC | Rate of Change | 動量變化 | 12 | 穿越 0 軸 |

### 波動率指標

| 指標 | 全名 | 用途 | 常用參數 | 應用 |
|------|------|------|----------|------|
| ATR | Average True Range | 波動幅度 | 14 | 止損距離 |
| BB | Bollinger Bands | 價格區間 | 20, 2 | 觸及帶邊緣 |
| Keltner | Keltner Channel | 價格區間 | 20, 2 | 突破 |
| NATR | Normalized ATR | 相對波動 | 14 | 跨幣種比較 |

### 成交量指標

| 指標 | 全名 | 用途 | 常用參數 | 訊號 |
|------|------|------|----------|------|
| OBV | On Balance Volume | 量價關係 | - | 趨勢確認 |
| VWAP | Volume Weighted Average Price | 均價 | - | 支撐阻力 |
| CMF | Chaikin Money Flow | 資金流向 | 20 | > 0 / < 0 |
| Volume MA | - | 成交量趨勢 | 20 | 放量確認 |

### 市場狀態偵測（Regime Detection）

| 方法 | 類型 | 用途 | 輸出 |
|------|------|------|------|
| 方向×波動矩陣 | 可解釋指標 | 即時策略切換 | 方向(-10~+10), 波動(0~10) |
| HMM | 學術方法 | 狀態機率估計 | 狀態機率分佈 |
| HSMM | 學術方法 | 含持續時間建模 | 狀態+預期持續 |
| Jump Model | 學術方法 | 突變點偵測 | 狀態轉換點 |

**方向分數指標**：MA位置、MA斜率、RSI偏離、MACD柱狀、ADX +DI/-DI、Elder Power

**波動分數指標**：ATR百分位、Bollinger Band Width、Choppiness Index

## VectorBT 內建指標

```python
import vectorbtpro as vbt

# 趨勢指標
ma = vbt.MA.run(close, window=20)
ema = vbt.MA.run(close, window=20, ewm=True)
macd = vbt.MACD.run(close, fast_window=12, slow_window=26, signal_window=9)

# 動量指標
rsi = vbt.RSI.run(close, window=14)
stoch = vbt.STOCH.run(high, low, close, k_window=14, d_window=3)

# 波動率指標
atr = vbt.ATR.run(high, low, close, window=14)
bbands = vbt.BBANDS.run(close, window=20, alpha=2)
```

## 常用指標組合

### 組合 1：趨勢確認

```python
# MA 交叉 + RSI 過濾
ma_fast = vbt.MA.run(close, window=10).ma
ma_slow = vbt.MA.run(close, window=30).ma
rsi = vbt.RSI.run(close, window=14).rsi

long_signal = (ma_fast > ma_slow) & (rsi > 50) & (rsi < 70)
short_signal = (ma_fast < ma_slow) & (rsi < 50) & (rsi > 30)
```

### 組合 2：突破策略

```python
# 布林帶突破 + 成交量確認
bb = vbt.BBANDS.run(close, window=20, alpha=2)
vol_ma = vbt.MA.run(volume, window=20).ma

long_signal = (close > bb.upper) & (volume > vol_ma * 1.5)
short_signal = (close < bb.lower) & (volume > vol_ma * 1.5)
```

### 組合 3：動量策略

```python
# MACD + RSI
macd = vbt.MACD.run(close)
rsi = vbt.RSI.run(close, window=14).rsi

long_signal = (macd.macd > macd.signal) & (rsi > 50)
short_signal = (macd.macd < macd.signal) & (rsi < 50)
```

## 指標計算公式

### RSI

```python
def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
```

### ATR

```python
def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr
```

### Supertrend

```python
def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)

    for i in range(period, len(close)):
        if close.iloc[i] > upper_band.iloc[i-1]:
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i-1]:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
        else:
            supertrend.iloc[i] = supertrend.iloc[i-1]
            direction.iloc[i] = direction.iloc[i-1]

    return supertrend, direction
```

## 指標選擇指南

### 趨勢市場

| 推薦指標 | 理由 |
|----------|------|
| MA/EMA | 識別趨勢方向 |
| MACD | 確認趨勢動能 |
| ADX | 確認趨勢強度 |
| Supertrend | 趨勢+動態止損 |

### 震盪市場

| 推薦指標 | 理由 |
|----------|------|
| RSI | 超買超賣反轉 |
| Bollinger Bands | 區間上下界 |
| Stochastic | 短期反轉訊號 |

### 高波動市場

| 推薦指標 | 理由 |
|----------|------|
| ATR | 動態止損距離 |
| Keltner Channel | 波動過濾 |
| NATR | 標準化比較 |

## 指標陷阱

| 陷阱 | 說明 | 解決方案 |
|------|------|----------|
| 過度擬合 | 太多指標 | 限制 3-4 個 |
| 延遲 | 滯後訊號 | 縮短週期或用前瞻指標 |
| 同質化 | 指標類似 | 選擇不同類型指標 |
| 參數敏感 | 小變化大影響 | 參數穩健性測試 |

## 加密貨幣特殊考量

1. **24/7 交易**：無收盤缺口，部分指標需調整
2. **高波動性**：ATR 數值較大，止損需放寬
3. **週期性**：資金費率結算時常有波動
4. **流動性**：小時框低流動性時段指標可能失真

## 與其他 Skills 關係

### 被調用（上游）

| Skill | 場景 |
|-------|------|
| **策略開發** | 選擇和計算策略所需指標 |

### 本 Skill 提供

- 趨勢指標（MA, EMA, MACD, ADX, Supertrend）
- 動量指標（RSI, Stochastic, CCI, MFI）
- 波動率指標（ATR, Bollinger Bands, Keltner）
- 成交量指標（OBV, VWAP, CMF）

### 指標整合

```
策略開發（設計策略）
    ↓
指標庫
    ├─→ 選擇合適指標
    ├─→ VectorBT 計算
    └─→ 組合應用
    ↓
策略訊號產生
```

For 趨勢指標詳解 → read `references/trend-indicators.md`
For 動量指標詳解 → read `references/momentum-indicators.md`
For 市場狀態偵測與策略切換 → read `references/regime-detection.md`
