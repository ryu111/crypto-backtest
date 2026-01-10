---
name: strategy-dev
description: 策略開發指南。設計、編寫、測試交易策略。當需要建立新策略、修改現有策略、理解策略邏輯時使用。
---

# 策略開發

BTC/ETH 永續合約交易策略的設計與實作。

## Quick Start

```python
from src.strategies import BaseStrategy

class MACrossStrategy(BaseStrategy):
    params = {
        'fast_period': 10,
        'slow_period': 30,
        'stop_loss_atr': 2.0,
    }

    def generate_signals(self, data):
        fast_ma = data['close'].rolling(self.params['fast_period']).mean()
        slow_ma = data['close'].rolling(self.params['slow_period']).mean()

        long_entry = fast_ma > slow_ma
        long_exit = fast_ma < slow_ma

        return long_entry, long_exit, None, None
```

## 策略分類

| 類型 | 適用市場 | 持倉時間 | 複雜度 | 勝率期望 |
|------|----------|----------|--------|----------|
| 趨勢跟隨 | 趨勢市 | 中長期 | 低 | 40-50% |
| 均值回歸 | 震盪市 | 短期 | 中 | 55-65% |
| 動量突破 | 波動市 | 短中期 | 中 | 45-55% |
| 統計套利 | 任何 | 極短 | 高 | 50-60% |
| 資金費率 | 高費率期 | 持有 | 低 | 70%+ |

## 策略結構模板

```python
class BaseStrategy:
    """策略基礎類別"""

    # 1. 參數定義
    params = {
        'param1': 10,
        'param2': 20,
    }

    def __init__(self, **kwargs):
        # 覆寫預設參數
        self.params = {**self.params, **kwargs}

    # 2. 指標計算
    def calculate_indicators(self, data):
        """計算策略所需指標"""
        indicators = {}
        # 實作指標計算
        return indicators

    # 3. 進場訊號
    def generate_entries(self, data, indicators):
        """產生進場訊號"""
        long_entries = None
        short_entries = None
        return long_entries, short_entries

    # 4. 出場訊號
    def generate_exits(self, data, indicators):
        """產生出場訊號"""
        long_exits = None
        short_exits = None
        return long_exits, short_exits

    # 5. 部位大小
    def position_size(self, data, capital, risk_per_trade=0.02):
        """計算部位大小"""
        # 基於風險的部位計算
        return size

    # 6. 完整訊號
    def generate_signals(self, data):
        """產生完整交易訊號"""
        indicators = self.calculate_indicators(data)
        long_entry, short_entry = self.generate_entries(data, indicators)
        long_exit, short_exit = self.generate_exits(data, indicators)
        return long_entry, long_exit, short_entry, short_exit
```

## 常見策略範例

### 趨勢策略：雙均線交叉

```python
class DualMACross(BaseStrategy):
    params = {
        'fast_period': 10,
        'slow_period': 30,
    }

    def generate_signals(self, data):
        close = data['close']
        fast_ma = close.rolling(self.params['fast_period']).mean()
        slow_ma = close.rolling(self.params['slow_period']).mean()

        # 金叉做多，死叉做空
        long_entry = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        long_exit = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        short_entry = long_exit
        short_exit = long_entry

        return long_entry, long_exit, short_entry, short_exit
```

### 動量策略：RSI 超買超賣

```python
class RSIMeanReversion(BaseStrategy):
    params = {
        'rsi_period': 14,
        'oversold': 30,
        'overbought': 70,
    }

    def generate_signals(self, data):
        close = data['close']

        # 計算 RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(self.params['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.params['rsi_period']).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # 超賣做多，超買做空
        long_entry = rsi < self.params['oversold']
        long_exit = rsi > 50
        short_entry = rsi > self.params['overbought']
        short_exit = rsi < 50

        return long_entry, long_exit, short_entry, short_exit
```

### 突破策略：布林帶突破

```python
class BollingerBreakout(BaseStrategy):
    params = {
        'period': 20,
        'std_dev': 2.0,
        'volume_mult': 1.5,
    }

    def generate_signals(self, data):
        close = data['close']
        volume = data['volume']

        # 布林帶
        ma = close.rolling(self.params['period']).mean()
        std = close.rolling(self.params['period']).std()
        upper = ma + self.params['std_dev'] * std
        lower = ma - self.params['std_dev'] * std

        # 成交量確認
        vol_ma = volume.rolling(self.params['period']).mean()
        high_volume = volume > vol_ma * self.params['volume_mult']

        # 突破訊號
        long_entry = (close > upper) & high_volume
        long_exit = close < ma
        short_entry = (close < lower) & high_volume
        short_exit = close > ma

        return long_entry, long_exit, short_entry, short_exit
```

## 進場模式

| 模式 | 說明 | 適用策略 |
|------|------|----------|
| 交叉進場 | 快線穿越慢線 | 趨勢跟隨 |
| 閾值進場 | 指標達到閾值 | 超買超賣 |
| 突破進場 | 價格突破區間 | 動量突破 |
| 回調進場 | 趨勢中的回調 | 趨勢跟隨 |
| 形態進場 | K 線形態識別 | 技術分析 |

## 出場模式

| 模式 | 說明 | 優點 | 缺點 |
|------|------|------|------|
| 固定止損 | 入場價 ±X% | 簡單 | 不適應波動 |
| ATR 止損 | 入場價 ± N×ATR | 適應波動 | 需選擇 N |
| 追蹤止損 | 跟隨最高/低價 | 保護利潤 | 可能過早出場 |
| 時間出場 | 持有 N 根 K 線 | 減少暴露 | 可能錯過行情 |
| 訊號反轉 | 反向訊號觸發 | 完整系統 | 可能延遲 |

## 策略過濾器

### 趨勢過濾

```python
def trend_filter(data, period=200):
    """只在趨勢方向交易"""
    ma_200 = data['close'].rolling(period).mean()
    uptrend = data['close'] > ma_200
    downtrend = data['close'] < ma_200
    return uptrend, downtrend
```

### 波動率過濾

```python
def volatility_filter(data, atr_period=14, min_atr=0.01):
    """過濾低波動期間"""
    atr = calculate_atr(data['high'], data['low'], data['close'], atr_period)
    natr = atr / data['close']
    return natr > min_atr
```

### 時間過濾

```python
def time_filter(data, avoid_hours=[0, 8, 16]):
    """避開資金費率結算時段"""
    hour = data.index.hour
    return ~hour.isin(avoid_hours)
```

## 策略命名規範

```
{type}_{indicator}_{timeframe}_{version}
```

範例：
- `trend_ma_cross_4h_v1`
- `momentum_rsi_1h_v2`
- `breakout_bb_4h_v1`

## 策略開發流程

```
1. 定義假設
   └→ 市場行為假設、預期優勢

2. 選擇指標
   └→ 根據假設選擇合適指標

3. 設計規則
   └→ 進場、出場、過濾條件

4. 初步回測
   └→ 驗證基本有效性

5. 參數優化
   └→ 使用 optimization skill

6. 穩健性驗證
   └→ 使用 validation skill

7. 風險評估
   └→ 使用 risk-management skill

8. 記錄學習
   └→ 使用 learning-system skill
```

## 策略檢查清單

- [ ] 策略邏輯清晰、可解釋
- [ ] 進出場規則明確
- [ ] 包含止損機制
- [ ] 考慮交易成本
- [ ] 避免前瞻偏差
- [ ] 參數數量適中（<= 5 個）
- [ ] 包含過濾條件
- [ ] 適用於 BTC 和 ETH

For 策略類型詳解 → read `references/strategy-types.md`
For 進出場模式 → read `references/entry-exit-patterns.md`
