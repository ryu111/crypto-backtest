---
name: perpetual-specific
description: 永續合約專用知識。資金費率、基差、強平機制、交易所特性。當需要理解永續合約特性、設計合約專用策略時使用。
---

# 永續合約專用知識

BTC/ETH 永續合約特有的交易機制和策略。

## 資金費率機制

### 基本概念

資金費率是永續合約維持與現貨價格錨定的機制。

| 資金費率 | 市場情緒 | 多方 | 空方 |
|----------|----------|------|------|
| 正費率（>0） | 看多情緒強 | 付費 | 收費 |
| 負費率（<0） | 看空情緒強 | 收費 | 付費 |

### 結算週期

| 交易所 | 結算時間 (UTC) | 週期 |
|--------|----------------|------|
| Binance | 00:00, 08:00, 16:00 | 8h |
| Bybit | 00:00, 08:00, 16:00 | 8h |
| OKX | 00:00, 08:00, 16:00 | 8h |

### 費率計算

```python
# 資金費率成本計算
def funding_cost(position_value, funding_rate, direction):
    """
    Args:
        position_value: 持倉價值 (USDT)
        funding_rate: 資金費率 (如 0.0001 = 0.01%)
        direction: 1 = 多, -1 = 空

    Returns:
        cost: 正 = 支付, 負 = 收取
    """
    return position_value * funding_rate * direction

# 年化收益估算
def annualized_funding_return(avg_rate):
    """
    假設每 8 小時結算一次
    一年 = 365 * 3 = 1095 次
    """
    return avg_rate * 1095
```

### 費率範圍參考

| 市場狀態 | 典型費率 | 年化影響 |
|----------|----------|----------|
| 正常 | 0.01% | ~11% |
| 牛市 | 0.05-0.1% | ~55-110% |
| 極端牛市 | 0.1-0.3% | ~110-330% |
| 熊市 | -0.01-0% | ~-11-0% |
| 極端熊市 | -0.1%以下 | < -110% |

## 資金費率策略

### 策略 1：費率套利（Delta Neutral）

```python
class FundingArbitrage:
    """
    現貨做多 + 永續做空
    賺取正資金費率
    """

    def setup(self, capital, symbol):
        # 一半資金買現貨
        spot_size = capital * 0.5 / spot_price

        # 一半資金做空永續（等量）
        perp_size = spot_size
        perp_margin = capital * 0.5

        return {
            'spot': {'side': 'long', 'size': spot_size},
            'perp': {'side': 'short', 'size': perp_size}
        }

    def should_enter(self, funding_rate):
        # 費率 > 0.05% 時進場
        return funding_rate > 0.0005

    def should_exit(self, funding_rate):
        # 費率 < 0.01% 或轉負時出場
        return funding_rate < 0.0001
```

### 策略 2：費率預測

```python
class FundingPredictor:
    """
    預測費率方向進行交易
    """

    def predict_funding_direction(self, data):
        # 高溢價 → 費率將升高
        premium = (data['perp_price'] - data['spot_price']) / data['spot_price']

        # OI 增加 + 溢價上升 → 費率將升高
        oi_change = data['open_interest'].pct_change(24)

        if premium > 0.001 and oi_change > 0.05:
            return 'rising'  # 做空永續
        elif premium < -0.001 and oi_change < -0.05:
            return 'falling'  # 做多永續
        else:
            return 'neutral'
```

### 策略 3：費率結算交易

```python
class FundingSettlementTrade:
    """
    利用結算前後價格波動
    """

    def pre_settlement_signal(self, current_hour, funding_rate):
        # 結算前 1 小時
        settlement_hours = [23, 7, 15]  # UTC-1

        if current_hour in settlement_hours:
            if funding_rate > 0.0003:
                # 高正費率，結算前可能下跌
                return 'short'
            elif funding_rate < -0.0003:
                # 高負費率，結算前可能上漲
                return 'long'

        return None
```

## 基差交易

### 基差定義

```
基差 = 永續價格 - 現貨價格
```

### 基差狀態

| 狀態 | 含義 | 機會 |
|------|------|------|
| 正基差（溢價） | 永續 > 現貨 | 做空永續 |
| 負基差（折價） | 永續 < 現貨 | 做多永續 |
| 基差收斂 | 價格回歸 | 套利空間 |

### 基差交易策略

```python
class BasisTrade:
    def __init__(self, entry_threshold=0.002, exit_threshold=0.0005):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def calculate_basis(self, perp_price, spot_price):
        return (perp_price - spot_price) / spot_price

    def generate_signal(self, perp_price, spot_price):
        basis = self.calculate_basis(perp_price, spot_price)

        if basis > self.entry_threshold:
            # 高溢價，做空永續
            return 'short_perp'
        elif basis < -self.entry_threshold:
            # 高折價，做多永續
            return 'long_perp'
        elif abs(basis) < self.exit_threshold:
            # 基差收斂，平倉
            return 'close'

        return None
```

## 強平級聯識別

### 清算熱力圖

```python
def identify_liquidation_zones(price, open_interest, leverage_distribution):
    """
    識別潛在清算密集區
    """
    zones = []

    # 假設平均槓桿
    avg_leverage = 10

    # 多頭清算區（價格下方）
    long_liq_price = price * (1 - 1/avg_leverage + 0.005)

    # 空頭清算區（價格上方）
    short_liq_price = price * (1 + 1/avg_leverage - 0.005)

    return {
        'long_liquidation_zone': long_liq_price,
        'short_liquidation_zone': short_liq_price
    }
```

### 清算級聯預警

```python
def liquidation_cascade_risk(oi_change, price_change, volume):
    """
    評估清算級聯風險

    警示信號：
    - OI 大幅下降
    - 價格快速移動
    - 成交量暴增
    """
    risk_score = 0

    if oi_change < -0.05:  # OI 下降 5%+
        risk_score += 30

    if abs(price_change) > 0.03:  # 價格變動 3%+
        risk_score += 30

    if volume > volume.rolling(24).mean() * 3:  # 成交量 3x+
        risk_score += 40

    return risk_score  # 0-100
```

## 交易所特性

| 交易所 | Maker Fee | Taker Fee | 最大槓桿 | 特點 |
|--------|-----------|-----------|----------|------|
| Binance | 0.02% | 0.04% | 125x | 流動性最佳 |
| Bybit | 0.01% | 0.06% | 100x | 介面友善 |
| OKX | 0.02% | 0.05% | 125x | 產品多元 |
| Bitget | 0.02% | 0.06% | 125x | 跟單功能 |

### 流動性比較

| 交易對 | Binance | Bybit | OKX |
|--------|---------|-------|-----|
| BTCUSDT | 最佳 | 良好 | 良好 |
| ETHUSDT | 最佳 | 良好 | 良好 |
| 山寨幣 | 較好 | 中等 | 較好 |

## 永續合約回測注意事項

### 必須考慮

1. **資金費率**：每 8 小時扣除/收取
2. **滑點**：高槓桿時影響放大
3. **強平**：模擬爆倉機制
4. **交易費用**：Maker/Taker 差異

### 常見錯誤

| 錯誤 | 影響 | 解決方案 |
|------|------|----------|
| 忽略資金費率 | 高估收益 | 整合費率數據 |
| 固定滑點 | 低估成本 | 動態滑點模型 |
| 無強平機制 | 虛假績效 | 實作強平邏輯 |
| 用現貨費率 | 成本不準 | 用正確的費率 |

## 永續 vs 現貨策略差異

| 面向 | 現貨策略 | 永續策略 |
|------|----------|----------|
| 做空 | 需借幣/不可 | 原生支援 |
| 成本 | 交易費 | 交易費 + 資金費率 |
| 風險 | 最大虧完 | 可能爆倉 |
| 資金效率 | 1x | Nx |
| 持有成本 | 無 | 資金費率 |

For 資金費率策略詳解 → read `references/funding-rate-strategies.md`
For 基差交易詳解 → read `references/basis-trading.md`
