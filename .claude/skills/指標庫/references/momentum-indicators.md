# 動量指標詳解

## RSI (Relative Strength Index)

### 計算

```python
def rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi
```

### 訊號

| 條件 | 訊號 | 可靠度 |
|------|------|--------|
| RSI < 30 | 超賣，考慮做多 | 需確認 |
| RSI > 70 | 超買，考慮做空 | 需確認 |
| RSI 背離 | 趨勢反轉 | 較可靠 |
| RSI 中軸 | 50 為多空分界 | 趨勢確認 |

### RSI 背離

```python
def rsi_divergence(close, rsi, lookback=14):
    """檢測 RSI 背離"""

    # 價格新低但 RSI 沒新低 = 看漲背離
    price_lower = close < close.rolling(lookback).min().shift(1)
    rsi_higher = rsi > rsi.rolling(lookback).min().shift(1)
    bullish_div = price_lower & rsi_higher

    # 價格新高但 RSI 沒新高 = 看跌背離
    price_higher = close > close.rolling(lookback).max().shift(1)
    rsi_lower = rsi < rsi.rolling(lookback).max().shift(1)
    bearish_div = price_higher & rsi_lower

    return bullish_div, bearish_div
```

## Stochastic Oscillator

### 計算

```python
def stochastic(high, low, close, k_period=14, d_period=3, smooth=3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()

    # Fast %K
    fast_k = 100 * (close - lowest_low) / (highest_high - lowest_low)

    # Slow %K (smoothed)
    slow_k = fast_k.rolling(smooth).mean()

    # %D
    d = slow_k.rolling(d_period).mean()

    return slow_k, d
```

### 訊號

| 條件 | 訊號 |
|------|------|
| %K < 20, %D < 20 | 超賣區 |
| %K > 80, %D > 80 | 超買區 |
| %K 上穿 %D (超賣區) | 買入訊號 |
| %K 下穿 %D (超買區) | 賣出訊號 |

## CCI (Commodity Channel Index)

### 計算

```python
def cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())

    cci = (tp - sma_tp) / (0.015 * mad)
    return cci
```

### 訊號

| CCI 值 | 意義 |
|--------|------|
| > 100 | 超買 |
| < -100 | 超賣 |
| 穿越 100 | 進入超買 |
| 穿越 -100 | 進入超賣 |

## Williams %R

### 計算

```python
def williams_r(high, low, close, period=14):
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()

    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr
```

### 訊號

| 條件 | 訊號 |
|------|------|
| %R > -20 | 超買 |
| %R < -80 | 超賣 |

## MFI (Money Flow Index)

### 計算

```python
def mfi(high, low, close, volume, period=14):
    tp = (high + low + close) / 3
    mf = tp * volume

    positive_mf = mf.where(tp > tp.shift(1), 0)
    negative_mf = mf.where(tp < tp.shift(1), 0)

    positive_sum = positive_mf.rolling(period).sum()
    negative_sum = negative_mf.rolling(period).sum()

    mfi = 100 - (100 / (1 + positive_sum / negative_sum))
    return mfi
```

### 與 RSI 的差異

| 指標 | 考慮成交量 | 適用場景 |
|------|------------|----------|
| RSI | 否 | 價格動量 |
| MFI | 是 | 資金流動 |

## 動量指標組合建議

### 組合 1：RSI + MA 趨勢濾網

```python
# 只在趨勢方向做超買超賣
uptrend = close > sma(close, 200)

# 上升趨勢中只做多
long_signal = uptrend & (rsi < 30)

# 下降趨勢中只做空
downtrend = close < sma(close, 200)
short_signal = downtrend & (rsi > 70)
```

### 組合 2：Stochastic + RSI

```python
# 雙重確認
stoch_oversold = (stoch_k < 20) & (stoch_d < 20)
rsi_oversold = rsi < 30

strong_buy = stoch_oversold & rsi_oversold
```
