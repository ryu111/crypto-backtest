# 趨勢指標詳解

## Moving Average (MA)

### 簡單移動平均 (SMA)

```python
def sma(close, period):
    return close.rolling(period).mean()
```

**用途**：識別趨勢方向
**常用週期**：20, 50, 200

### 指數移動平均 (EMA)

```python
def ema(close, period):
    return close.ewm(span=period, adjust=False).mean()
```

**用途**：更快反應價格變化
**常用週期**：12, 26

### 交叉訊號

```python
def ma_cross_signal(close, fast_period, slow_period):
    fast_ma = sma(close, fast_period)
    slow_ma = sma(close, slow_period)

    golden_cross = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    death_cross = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

    return golden_cross, death_cross
```

## MACD

### 計算

```python
def macd(close, fast=12, slow=26, signal=9):
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()

    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram
```

### 訊號

| 訊號 | 條件 | 意義 |
|------|------|------|
| 金叉 | MACD > Signal | 看多 |
| 死叉 | MACD < Signal | 看空 |
| 柱狀增加 | Histogram 增加 | 動能增強 |
| 零軸上方 | MACD > 0 | 多頭市場 |

## ADX (Average Directional Index)

### 計算

```python
def adx(high, low, close, period=14):
    # True Range
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)

    # Directional Movement
    plus_dm = ((high - high.shift(1)) > (low.shift(1) - low)) * (high - high.shift(1))
    plus_dm = plus_dm.clip(lower=0)
    minus_dm = ((low.shift(1) - low) > (high - high.shift(1))) * (low.shift(1) - low)
    minus_dm = minus_dm.clip(lower=0)

    # Smoothed
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()

    return adx, plus_di, minus_di
```

### 解讀

| ADX 值 | 趨勢強度 |
|--------|----------|
| 0-20 | 無趨勢/弱 |
| 20-40 | 中等趨勢 |
| 40-60 | 強趨勢 |
| 60+ | 極強趨勢 |

## Supertrend

### 計算

```python
def supertrend(high, low, close, period=10, multiplier=3.0):
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)

    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = -1

    for i in range(1, len(close)):
        if close.iloc[i] > supertrend.iloc[i-1]:
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        elif close.iloc[i] < supertrend.iloc[i-1]:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
        else:
            supertrend.iloc[i] = supertrend.iloc[i-1]
            direction.iloc[i] = direction.iloc[i-1]

            if direction.iloc[i] == 1:
                supertrend.iloc[i] = max(supertrend.iloc[i], lower_band.iloc[i])
            else:
                supertrend.iloc[i] = min(supertrend.iloc[i], upper_band.iloc[i])

    return supertrend, direction
```

### 優點

- 內建動態止損
- 趨勢反轉明確
- 適合加密貨幣高波動

## 趨勢指標組合建議

### 組合 1：MA + ADX

```python
# 只在強趨勢時交易
trend_up = close > sma(close, 50)
strong_trend = adx > 25

long_signal = trend_up & strong_trend
```

### 組合 2：MACD + Supertrend

```python
# MACD 確認方向，Supertrend 止損
macd_bullish = macd_line > signal_line
st_direction = supertrend_direction == 1

long_signal = macd_bullish & st_direction
```
