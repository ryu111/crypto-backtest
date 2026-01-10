# 永續合約機制

## 永續合約 vs 現貨

| 特性 | 現貨 | 永續合約 |
|------|------|----------|
| 到期日 | 無 | 無（永不到期） |
| 槓桿 | 1x | 1-125x |
| 做空 | 需借幣 | 原生支援 |
| 資金費率 | 無 | 有 |
| 強制平倉 | 無 | 有 |

## 資金費率機制

### 計算公式

```
資金費率 = 溢價指數 + clamp(利率 - 溢價指數, -0.05%, 0.05%)
```

### 結算週期

- **Binance**：每 8 小時（00:00, 08:00, 16:00 UTC）
- **Bybit**：每 8 小時
- **部分交易所**：已改為動態週期（1h/4h/8h）

### 費率方向

| 資金費率 | 市場情緒 | 持倉成本 |
|----------|----------|----------|
| 正費率（>0） | 看多情緒強 | 多方付空方 |
| 負費率（<0） | 看空情緒強 | 空方付多方 |

### 費率計算

```python
def calculate_funding_cost(position_value, funding_rate, direction):
    """
    計算資金費率成本

    Args:
        position_value: 持倉價值 (USDT)
        funding_rate: 資金費率 (如 0.0001 = 0.01%)
        direction: 1 = 做多, -1 = 做空

    Returns:
        cost: 正數 = 支付, 負數 = 收取
    """
    cost = position_value * funding_rate * direction
    return cost

# 範例：持倉 $10,000 做多，費率 0.01%
cost = calculate_funding_cost(10000, 0.0001, 1)
# cost = 1.0 USDT（支付）
```

### 回測中整合資金費率

```python
import pandas as pd
import numpy as np

def apply_funding_rate(equity_curve, positions, funding_rates, timestamps):
    """
    在權益曲線中扣除資金費率

    Args:
        equity_curve: 權益曲線 Series
        positions: 持倉方向 Series (1=多, -1=空, 0=無)
        funding_rates: 資金費率 DataFrame
        timestamps: 時間戳 Index

    Returns:
        adjusted_equity: 調整後權益曲線
    """
    adjusted_equity = equity_curve.copy()

    for ts in funding_rates.index:
        if ts in timestamps:
            pos = positions.loc[ts]
            if pos != 0:
                rate = funding_rates.loc[ts, 'rate']
                pos_value = adjusted_equity.loc[ts]
                cost = pos_value * rate * pos
                adjusted_equity.loc[ts:] -= cost

    return adjusted_equity
```

## 槓桿機制

### 槓桿類型

| 類型 | 說明 | 風險 |
|------|------|------|
| 逐倉 (Isolated) | 每個倉位獨立保證金 | 單倉爆倉不影響其他 |
| 全倉 (Cross) | 共用帳戶餘額 | 爆倉會損失全部 |

### 保證金計算

```python
def calculate_margin(position_size, leverage, entry_price):
    """
    計算所需保證金
    """
    notional_value = position_size * entry_price
    initial_margin = notional_value / leverage
    return initial_margin

# 範例：1 BTC @ $50,000，10x 槓桿
margin = calculate_margin(1, 10, 50000)
# margin = 5000 USDT
```

### 保證金率

```python
def margin_ratio(equity, position_value):
    """
    計算保證金率
    """
    return equity / position_value

# 維持保證金率通常為 0.4% - 0.5%
MAINTENANCE_MARGIN_RATE = 0.005  # 0.5%
```

## 強制平倉機制

### 強平價格計算

```python
def liquidation_price_long(entry_price, leverage, maintenance_margin_rate=0.005):
    """
    計算做多強平價格
    """
    liq_price = entry_price * (1 - 1/leverage + maintenance_margin_rate)
    return liq_price

def liquidation_price_short(entry_price, leverage, maintenance_margin_rate=0.005):
    """
    計算做空強平價格
    """
    liq_price = entry_price * (1 + 1/leverage - maintenance_margin_rate)
    return liq_price

# 範例：$50,000 做多，10x 槓桿
liq_long = liquidation_price_long(50000, 10)
# liq_long ≈ $45,250（跌 9.5% 爆倉）

liq_short = liquidation_price_short(50000, 10)
# liq_short ≈ $54,750（漲 9.5% 爆倉）
```

### 回測中模擬強平

```python
def check_liquidation(current_price, entry_price, direction, leverage, mmr=0.005):
    """
    檢查是否觸發強平

    Args:
        current_price: 當前價格
        entry_price: 入場價格
        direction: 1 = 多, -1 = 空
        leverage: 槓桿倍數
        mmr: 維持保證金率

    Returns:
        is_liquidated: bool
    """
    if direction == 1:  # 做多
        liq_price = entry_price * (1 - 1/leverage + mmr)
        return current_price <= liq_price
    else:  # 做空
        liq_price = entry_price * (1 + 1/leverage - mmr)
        return current_price >= liq_price
```

## Mark Price

### 用途

- 計算未實現盈虧
- 判斷強平
- 避免惡意操縱

### 計算方式

```
Mark Price = 現貨指數價格 + 移動平均基差
```

### 為什麼重要

使用 Mark Price 而非 Last Price 計算強平，可以：
- 防止閃崩時被惡意爆倉
- 減少極端行情的誤殺

## 回測實作範例

```python
class PerpetualBacktester:
    def __init__(self, initial_capital, leverage, fees):
        self.capital = initial_capital
        self.leverage = leverage
        self.fees = fees
        self.position = 0
        self.entry_price = 0

    def open_position(self, price, size, direction):
        """開倉"""
        notional = size * price
        margin_required = notional / self.leverage
        fee = notional * self.fees['taker']

        if margin_required + fee > self.capital:
            return False

        self.capital -= margin_required + fee
        self.position = size * direction
        self.entry_price = price
        return True

    def close_position(self, price):
        """平倉"""
        if self.position == 0:
            return 0

        pnl = self.position * (price - self.entry_price)
        notional = abs(self.position) * price
        margin = notional / self.leverage
        fee = notional * self.fees['taker']

        self.capital += margin + pnl - fee
        self.position = 0
        self.entry_price = 0
        return pnl

    def apply_funding(self, funding_rate, mark_price):
        """應用資金費率"""
        if self.position == 0:
            return 0

        position_value = abs(self.position) * mark_price
        cost = position_value * funding_rate * np.sign(self.position)
        self.capital -= cost
        return cost

    def check_liquidation(self, current_price):
        """檢查強平"""
        if self.position == 0:
            return False

        direction = np.sign(self.position)
        return check_liquidation(
            current_price,
            self.entry_price,
            direction,
            self.leverage
        )
```

## 交易所差異

| 交易所 | Maker Fee | Taker Fee | 最大槓桿 | 資金費率週期 |
|--------|-----------|-----------|----------|--------------|
| Binance | 0.02% | 0.04% | 125x | 8h |
| Bybit | 0.01% | 0.06% | 100x | 8h |
| OKX | 0.02% | 0.05% | 125x | 8h |
| Bitget | 0.02% | 0.06% | 125x | 8h |

## 回測注意事項

1. **資金費率數據**：需單獨獲取歷史資金費率
2. **Mark Price**：理想情況使用 Mark Price，但 OHLCV 通常只有 Last Price
3. **強平滑點**：實際強平可能有額外滑點
4. **ADL 機制**：自動減倉在回測中通常忽略
5. **保險基金**：回測中通常不考慮
