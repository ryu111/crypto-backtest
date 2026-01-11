# 動量策略

基於動量指標的交易策略實作。

## 策略列表

### 1. RSI 均值回歸策略 (`momentum_rsi`)

基於 RSI 指標的超買超賣策略，適合震盪市場。

**策略邏輯:**
- **多單進場**: RSI < 30（超賣）且價格在上升趨勢
- **多單出場**: RSI > 70（超買）或 RSI 回到 50
- **空單進場**: RSI > 70（超買）且價格在下降趨勢
- **空單出場**: RSI < 30（超賣）或 RSI 回到 50

**參數:**
```python
{
    'rsi_period': 14,       # RSI 計算週期
    'oversold': 30,         # 超賣閾值
    'overbought': 70,       # 超買閾值
    'trend_filter': True,   # 是否啟用趨勢過濾
    'trend_period': 200,    # 趨勢判斷均線週期
}
```

**使用範例:**
```python
from src.strategies import create_strategy

# 使用預設參數
strategy = create_strategy('momentum_rsi')

# 自訂參數
strategy = create_strategy(
    'momentum_rsi',
    rsi_period=14,
    oversold=30,
    overbought=70,
    trend_filter=True
)

# 產生訊號
long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(data)
```

**優點:**
- 適合震盪市場
- 訊號明確，容易理解
- 可選趨勢過濾，降低假訊號

**缺點:**
- 趨勢市場表現較差
- 可能過早進場（RSI 可以長時間超買/超賣）

**最佳應用:**
- 橫盤整理市場
- 日內交易
- 搭配其他趨勢指標使用

---

### 2. MACD 交叉策略 (`momentum_macd`)

基於 MACD 指標的趨勢跟隨策略，適合趨勢市場。

**策略邏輯:**
- **多單進場**: MACD 線向上穿越訊號線（黃金交叉）且柱狀圖 > 0
- **多單出場**: MACD 線向下穿越訊號線（死亡交叉）
- **空單進場**: MACD 線向下穿越訊號線（死亡交叉）且柱狀圖 < 0
- **空單出場**: MACD 線向上穿越訊號線（黃金交叉）

**參數:**
```python
{
    'fast_period': 12,          # 快線 EMA 週期
    'slow_period': 26,          # 慢線 EMA 週期
    'signal_period': 9,         # 訊號線 EMA 週期
    'use_histogram': True,      # 是否使用柱狀圖確認
    'histogram_threshold': 0,   # 柱狀圖閾值
}
```

**使用範例:**
```python
from src.strategies import create_strategy

# 使用預設參數
strategy = create_strategy('momentum_macd')

# 自訂參數
strategy = create_strategy(
    'momentum_macd',
    fast_period=12,
    slow_period=26,
    signal_period=9,
    use_histogram=True
)

# 產生訊號
long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(data)

# 檢測背離
bullish_div, bearish_div = strategy.detect_divergence(data, lookback=14)
```

**優點:**
- 趨勢跟隨效果好
- 訊號延遲較低
- 柱狀圖提供額外確認

**缺點:**
- 震盪市場假訊號多
- 可能錯過趨勢初期
- 需要搭配過濾器

**最佳應用:**
- 趨勢市場
- 中長線交易
- 搭配趨勢過濾器

**進階功能:**
- `detect_divergence()`: 檢測價格與 MACD 背離，預示可能的趨勢反轉

---

## 完整使用範例

```python
import pandas as pd
from src.strategies import create_strategy, list_strategies

# 1. 查看可用策略
print(list_strategies())
# 輸出: ['momentum_rsi', 'momentum_macd', ...]

# 2. 準備資料
data = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# 3. 建立並使用 RSI 策略
rsi_strategy = create_strategy('momentum_rsi', rsi_period=14)
long_entry, long_exit, short_entry, short_exit = rsi_strategy.generate_signals(data)

# 4. 建立並使用 MACD 策略
macd_strategy = create_strategy('momentum_macd')
long_entry, long_exit, short_entry, short_exit = macd_strategy.generate_signals(data)

# 5. 查看策略資訊
print(rsi_strategy.get_info())
```

## 參數優化

策略提供參數優化空間定義，可與 Optuna 整合：

```python
from src.strategies import StrategyRegistry

# 取得參數空間
rsi_param_space = StrategyRegistry.get_param_space('momentum_rsi')
print(rsi_param_space)
# {
#     'rsi_period': {'type': 'int', 'low': 7, 'high': 28},
#     'oversold': {'type': 'int', 'low': 20, 'high': 40},
#     'overbought': {'type': 'int', 'low': 60, 'high': 80},
#     'trend_period': {'type': 'int', 'low': 100, 'high': 300}
# }
```

## 策略組合建議

### 趨勢市場
```python
# MACD + 趨勢過濾
macd = create_strategy('momentum_macd', use_histogram=True)
```

### 震盪市場
```python
# RSI + 關閉趨勢過濾
rsi = create_strategy('momentum_rsi', trend_filter=False, oversold=25, overbought=75)
```

### 混合市場
```python
# 兩者結合，MACD 確認趨勢，RSI 找進場點
macd = create_strategy('momentum_macd')
rsi = create_strategy('momentum_rsi', trend_filter=True)
```

## 測試範例

執行完整測試範例：

```bash
python examples/momentum_strategies_demo.py
```

## 相關檔案

- `rsi.py`: RSI 策略實作
- `macd.py`: MACD 策略實作
- `__init__.py`: 模組匯出
- `../base.py`: 策略基礎類別
- `../registry.py`: 策略註冊表

## 技術指標計算

這些策略使用的指標計算方法定義在 `BaseStrategy` 和 `MomentumStrategy` 基礎類別中：

- `calculate_rsi()`: RSI 計算
- `calculate_macd()`: MACD 計算
- `apply_trend_filter()`: 趨勢過濾

## 下一步

1. 回測這些策略以評估績效
2. 使用 Optuna 進行參數優化
3. 結合其他策略建立組合
4. 加入風控規則（止損、止盈）
