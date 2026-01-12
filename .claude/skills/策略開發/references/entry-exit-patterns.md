# 進出場模式

交易策略的進場和出場訊號模式。

## 進場模式分類

| 類型 | 說明 | 適用場景 |
|------|------|----------|
| Breakout | 價格突破關鍵位 | 趨勢啟動 |
| Pullback | 趨勢中回調 | 趨勢延續 |
| Reversal | 趨勢反轉 | 趨勢結束 |
| Momentum | 動量訊號 | 短期方向 |

## Breakout 模式

### 有效突破條件

| 條件 | 標準 | 說明 |
|------|------|------|
| 成交量 | > 20日均量 50% | 量能確認 |
| 價格移動 | > 3% | 有效幅度 |
| K 線實體 | > 70% | 收盤確認 |
| 無長影線 | 影線 < 實體 50% | 避免假突破 |

### 常見形態

```
三角形突破
-----------
     /\
    /  \
   /    \
  /      \
 /________\
      ↓
   突破向下

或

 \________/
  \      /
   \    /
    \  /
     \/
      ↓
   突破向上


旗形突破
---------
      /
     /
    /
   /___     ← 整理區間
   \__/
      \
       \
        ↓
      突破延續
```

### Python 實作

```python
import pandas as pd
import numpy as np

def breakout_signal(
    data: pd.DataFrame,
    lookback: int = 20,
    volume_mult: float = 1.5,
    price_threshold: float = 0.03
) -> tuple:
    """
    突破訊號產生器

    Args:
        data: OHLCV 資料
        lookback: 回看期間
        volume_mult: 成交量倍數
        price_threshold: 價格突破閾值

    Returns:
        (long_entries, short_entries)
    """
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']

    # 計算區間高低點
    resistance = high.rolling(lookback).max()
    support = low.rolling(lookback).min()

    # 成交量條件
    vol_ma = volume.rolling(lookback).mean()
    high_volume = volume > vol_ma * volume_mult

    # 價格變動
    price_change = close.pct_change()

    # 多頭突破：收盤突破阻力 + 高成交量 + 足夠漲幅
    long_entries = (
        (close > resistance.shift(1)) &
        high_volume &
        (price_change > price_threshold)
    )

    # 空頭突破：收盤跌破支撐 + 高成交量 + 足夠跌幅
    short_entries = (
        (close < support.shift(1)) &
        high_volume &
        (price_change < -price_threshold)
    )

    return long_entries, short_entries
```

### 假突破過濾

```python
def filter_false_breakouts(
    data: pd.DataFrame,
    entries: pd.Series,
    confirmation_bars: int = 2,
    retest_threshold: float = 0.005
) -> pd.Series:
    """
    過濾假突破

    等待確認 K 線，價格未回落至突破點
    """
    filtered = entries.copy()
    close = data['close']

    for i in range(confirmation_bars, len(entries)):
        if entries.iloc[i - confirmation_bars]:
            breakout_price = close.iloc[i - confirmation_bars]

            # 檢查確認期間是否回落
            min_close = close.iloc[i - confirmation_bars + 1:i + 1].min()

            if min_close < breakout_price * (1 - retest_threshold):
                # 回落過多，視為假突破
                filtered.iloc[i - confirmation_bars] = False

    return filtered
```

## Pullback 模式

### 核心概念

```
趨勢中回調
-----------
        ↗ 高點3
       /
      ↗ 高點2
     / ↘
    /   回調（入場點）
   ↗ 高點1
  /
 ↗ 趨勢起點


突破後回測
-----------
        ───────→ 突破後上漲
       /
阻力位 ─────────
      ↘ ↗      ← 回測（新支撐，入場點）
       X
      /
```

### Pullback vs Reversal 區別

| 特徵 | Pullback | Reversal |
|------|----------|----------|
| 成交量 | 低/中 | 高 |
| 動量 | 減弱 | 強烈反轉 |
| 結構 | 不破前低 | 破前低 |
| 持續時間 | 短（3-10根K線） | 長 |

### Python 實作

```python
def pullback_signal(
    data: pd.DataFrame,
    trend_period: int = 50,
    pullback_threshold: float = 0.382,  # 斐波那契
    rsi_period: int = 14,
    rsi_oversold: float = 40
) -> tuple:
    """
    回調進場訊號

    Args:
        trend_period: 趨勢判斷期間
        pullback_threshold: 回調比例（斐波那契）
        rsi_period: RSI 週期
        rsi_oversold: RSI 超賣閾值
    """
    close = data['close']

    # 趨勢判斷
    ma = close.rolling(trend_period).mean()
    uptrend = close > ma
    downtrend = close < ma

    # 計算近期高低點
    recent_high = close.rolling(20).max()
    recent_low = close.rolling(20).min()

    # 回調幅度
    pullback_depth = (recent_high - close) / (recent_high - recent_low)

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # 多頭回調進場
    long_entries = (
        uptrend &
        (pullback_depth >= pullback_threshold) &
        (pullback_depth <= 0.618) &  # 不超過 61.8%
        (rsi < rsi_oversold + 10) &
        (rsi > rsi_oversold)  # RSI 開始回升
    )

    # 空頭回調進場
    pullback_depth_short = (close - recent_low) / (recent_high - recent_low)
    short_entries = (
        downtrend &
        (pullback_depth_short >= pullback_threshold) &
        (pullback_depth_short <= 0.618) &
        (rsi > 100 - rsi_oversold - 10) &
        (rsi < 100 - rsi_oversold)
    )

    return long_entries, short_entries
```

## Reversal 模式

### 反轉訊號

| 訊號 | 描述 | 可靠度 |
|------|------|--------|
| 量價背離 | 價格新高但成交量下降 | 高 |
| RSI 背離 | 價格新高但 RSI 較低 | 高 |
| MACD 背離 | 價格新高但 MACD 較低 | 中高 |
| 頭肩形態 | 三峰形態，中間最高 | 高 |
| 雙頂/雙底 | 兩次測試失敗 | 中 |

### 背離檢測

```python
def detect_divergence(
    price: pd.Series,
    indicator: pd.Series,
    lookback: int = 14,
    min_bars: int = 5
) -> tuple:
    """
    檢測價格與指標背離

    Returns:
        (bullish_divergence, bearish_divergence)
    """
    # 找局部高低點
    price_highs = find_local_extremes(price, lookback, 'high')
    price_lows = find_local_extremes(price, lookback, 'low')

    bullish_div = pd.Series(False, index=price.index)
    bearish_div = pd.Series(False, index=price.index)

    for i in range(lookback, len(price)):
        # 看跌背離：價格新高，指標較低
        if price_highs.iloc[i]:
            prev_high_idx = find_previous_extreme(price_highs, i, min_bars)
            if prev_high_idx is not None:
                if (price.iloc[i] > price.iloc[prev_high_idx] and
                    indicator.iloc[i] < indicator.iloc[prev_high_idx]):
                    bearish_div.iloc[i] = True

        # 看漲背離：價格新低，指標較高
        if price_lows.iloc[i]:
            prev_low_idx = find_previous_extreme(price_lows, i, min_bars)
            if prev_low_idx is not None:
                if (price.iloc[i] < price.iloc[prev_low_idx] and
                    indicator.iloc[i] > indicator.iloc[prev_low_idx]):
                    bullish_div.iloc[i] = True

    return bullish_div, bearish_div

def find_local_extremes(series: pd.Series, lookback: int, extreme_type: str) -> pd.Series:
    """找局部極值"""
    result = pd.Series(False, index=series.index)

    for i in range(lookback, len(series) - lookback):
        window = series.iloc[i-lookback:i+lookback+1]

        if extreme_type == 'high' and series.iloc[i] == window.max():
            result.iloc[i] = True
        elif extreme_type == 'low' and series.iloc[i] == window.min():
            result.iloc[i] = True

    return result
```

### 頭肩頂形態

```python
def head_and_shoulders(
    data: pd.DataFrame,
    window: int = 20,
    tolerance: float = 0.02
) -> pd.Series:
    """
    頭肩頂形態檢測
    """
    high = data['high']
    close = data['close']

    signals = pd.Series(False, index=data.index)

    for i in range(window * 3, len(data)):
        # 找三個高點
        segment = high.iloc[i-window*3:i]

        # 左肩、頭、右肩
        left_shoulder = segment.iloc[:window].max()
        head = segment.iloc[window:window*2].max()
        right_shoulder = segment.iloc[window*2:].max()

        # 頸線
        left_low = segment.iloc[:window].min()
        right_low = segment.iloc[window*2:].min()
        neckline = (left_low + right_low) / 2

        # 驗證形態
        if (head > left_shoulder * (1 + tolerance) and
            head > right_shoulder * (1 + tolerance) and
            abs(left_shoulder - right_shoulder) / left_shoulder < tolerance * 2):

            # 突破頸線
            if close.iloc[i] < neckline:
                signals.iloc[i] = True

    return signals
```

## Momentum 訊號

### RSI 超買超賣

```python
def rsi_signals(
    close: pd.Series,
    period: int = 14,
    oversold: float = 30,
    overbought: float = 70
) -> tuple:
    """RSI 訊號"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # 基本訊號
    long_entries = (rsi < oversold) & (rsi.shift(1) >= oversold)
    short_entries = (rsi > overbought) & (rsi.shift(1) <= overbought)

    return long_entries, short_entries
```

### MACD 交叉

```python
def macd_signals(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> tuple:
    """MACD 訊號"""
    fast_ema = close.ewm(span=fast_period).mean()
    slow_ema = close.ewm(span=slow_period).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_period).mean()

    # 金叉做多，死叉做空
    long_entries = (macd > signal) & (macd.shift(1) <= signal.shift(1))
    short_entries = (macd < signal) & (macd.shift(1) >= signal.shift(1))

    return long_entries, short_entries
```

## K 線形態

### 錘子線 (Hammer)

```python
def hammer_pattern(data: pd.DataFrame, body_ratio: float = 0.3) -> pd.Series:
    """
    錘子線：小實體 + 長下影線
    """
    open_price = data['open']
    high = data['high']
    low = data['low']
    close = data['close']

    body = abs(close - open_price)
    full_range = high - low
    lower_shadow = np.minimum(open_price, close) - low
    upper_shadow = high - np.maximum(open_price, close)

    is_hammer = (
        (body / full_range < body_ratio) &  # 小實體
        (lower_shadow > body * 2) &          # 長下影線
        (upper_shadow < body * 0.5)          # 短上影線
    )

    return is_hammer
```

### 吞噬形態 (Engulfing)

```python
def engulfing_pattern(data: pd.DataFrame) -> tuple:
    """
    吞噬形態

    Returns:
        (bullish_engulfing, bearish_engulfing)
    """
    open_price = data['open']
    close = data['close']

    prev_open = open_price.shift(1)
    prev_close = close.shift(1)

    # 看漲吞噬：前一根陰線，當前陽線完全吞噬
    bullish = (
        (prev_close < prev_open) &  # 前一根陰線
        (close > open_price) &       # 當前陽線
        (open_price < prev_close) &  # 開盤低於前收盤
        (close > prev_open)          # 收盤高於前開盤
    )

    # 看跌吞噬
    bearish = (
        (prev_close > prev_open) &
        (close < open_price) &
        (open_price > prev_close) &
        (close < prev_open)
    )

    return bullish, bearish
```

## 組合策略

### 趨勢 + 回調 + 動量

```python
def trend_pullback_momentum(
    data: pd.DataFrame,
    trend_period: int = 50,
    rsi_period: int = 14,
    rsi_threshold: float = 40
) -> tuple:
    """
    組合策略：趨勢確認 + 回調入場 + 動量過濾
    """
    close = data['close']

    # 趨勢過濾
    ma = close.rolling(trend_period).mean()
    uptrend = close > ma
    downtrend = close < ma

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rsi = 100 - (100 / (1 + gain / loss))

    # 回調條件：RSI 低於閾值但趨勢向上
    long_entries = uptrend & (rsi < rsi_threshold) & (rsi > rsi.shift(1))
    short_entries = downtrend & (rsi > 100 - rsi_threshold) & (rsi < rsi.shift(1))

    return long_entries, short_entries
```

## 出場模式

### 出場類型比較

| 類型 | 優點 | 缺點 |
|------|------|------|
| 固定止損 | 簡單 | 不適應波動 |
| ATR 止損 | 適應波動 | 需選擇倍數 |
| 移動止損 | 保護利潤 | 可能過早出場 |
| 訊號反轉 | 系統完整 | 可能延遲 |

### 詳細實作參考

For 止損策略詳解 → read `../風險管理/references/position-sizing.md`

## 參考資料

- [Fidelity: Entry and Exit Points](https://www.fidelity.com/bin-public/060_www_fidelity_com/documents/learning-center/Deck_Entry-and-exit-points.pdf)
- [Day Trading Patterns 101](https://highstrike.com/day-trading-patterns/)
- [Reversal Patterns Guide](https://forextester.com/blog/reversal-patterns/)
