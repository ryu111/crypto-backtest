# 資金費率策略詳解

## 策略 1：Delta Neutral 套利

### 原理

同時持有現貨多頭和永續空頭，對沖價格風險，純收取資金費率。

### 實作

```python
class DeltaNeutralArbitrage:
    """
    Delta Neutral 資金費率套利

    操作：
    1. 現貨買入 1 BTC
    2. 永續做空 1 BTC
    3. 每 8 小時收取資金費率（正費率時）
    """

    def __init__(self, capital, min_rate=0.0005):
        self.capital = capital
        self.min_rate = min_rate  # 最低進場費率 0.05%

    def should_enter(self, funding_rate):
        """判斷是否進場"""
        return funding_rate >= self.min_rate

    def should_exit(self, funding_rate):
        """判斷是否出場"""
        return funding_rate < 0.0001  # 費率過低

    def calculate_position(self, spot_price, perp_price):
        """計算持倉"""
        # 一半資金買現貨
        spot_capital = self.capital * 0.5
        spot_size = spot_capital / spot_price

        # 一半資金做空永續（等量）
        perp_margin = self.capital * 0.5
        perp_size = spot_size  # 等量對沖

        return {
            'spot': {'side': 'long', 'size': spot_size},
            'perp': {'side': 'short', 'size': perp_size, 'margin': perp_margin}
        }

    def calculate_return(self, funding_rate, holding_hours):
        """計算收益"""
        settlements = holding_hours // 8
        total_return = funding_rate * settlements
        annualized = (1 + total_return) ** (8760 / holding_hours) - 1
        return total_return, annualized
```

### 風險

| 風險 | 說明 | 對策 |
|------|------|------|
| 費率翻負 | 開始付費 | 設定出場條件 |
| 基差風險 | 現貨永續價差 | 監控基差 |
| 強平風險 | 永續倉位 | 控制槓桿 |
| 交易成本 | 開平倉費用 | 計算盈虧平衡 |

### 年化收益估算

| 平均費率 | 年化收益（扣費前） |
|----------|-------------------|
| 0.01% | ~11% |
| 0.03% | ~33% |
| 0.05% | ~55% |
| 0.10% | ~110% |

## 策略 2：費率預測

### 原理

預測費率方向，在費率變化前佈局。

### 預測因子

| 因子 | 與費率關係 |
|------|------------|
| 溢價率 | 正相關 |
| OI 增加 | 正相關 |
| 多空比 | 正相關 |
| 大戶持倉 | 參考 |

### 實作

```python
class FundingRatePredictor:
    """資金費率預測交易"""

    def predict_funding(self, data):
        """
        預測下一期費率方向

        使用因子：
        - 當前溢價率
        - OI 變化
        - 成交量變化
        """
        # 溢價率
        premium = (data['perp_close'] - data['spot_close']) / data['spot_close']

        # OI 變化
        oi_change = data['open_interest'].pct_change(24)

        # 成交量變化
        vol_change = data['volume'].pct_change(24)

        # 簡單規則
        if premium > 0.002 and oi_change > 0.05:
            return 'rising', 0.8  # 費率可能上升
        elif premium < -0.002 and oi_change < -0.05:
            return 'falling', 0.8  # 費率可能下降
        else:
            return 'neutral', 0.5

    def generate_signal(self, prediction, current_rate):
        """產生交易訊號"""
        direction, confidence = prediction

        if direction == 'rising' and current_rate > 0.0003:
            # 費率上升，做空永續（收取費率）
            return 'short_perp'
        elif direction == 'falling' and current_rate < -0.0003:
            # 費率下降，做多永續（收取費率）
            return 'long_perp'

        return None
```

## 策略 3：結算時點交易

### 原理

利用結算前後的價格波動規律。

### 觀察規律

| 情境 | 結算前 | 結算後 |
|------|--------|--------|
| 高正費率 | 可能下跌（多頭平倉避費） | 可能反彈 |
| 高負費率 | 可能上漲（空頭平倉避費） | 可能回落 |

### 實作

```python
class FundingSettlementStrategy:
    """結算時點交易策略"""

    def __init__(self):
        self.settlement_hours = [0, 8, 16]  # UTC

    def should_trade(self, current_hour, funding_rate):
        """判斷是否交易"""
        # 結算前 1-2 小時
        pre_settlement = [(h - 2) % 24 for h in self.settlement_hours]

        if current_hour in pre_settlement:
            if funding_rate > 0.0005:  # 高正費率
                return 'short', 'pre_settlement_high_positive'
            elif funding_rate < -0.0005:  # 高負費率
                return 'long', 'pre_settlement_high_negative'

        # 結算後
        post_settlement = [(h + 1) % 24 for h in self.settlement_hours]

        if current_hour in post_settlement:
            if funding_rate > 0.0005:
                return 'long', 'post_settlement_reversal'
            elif funding_rate < -0.0005:
                return 'short', 'post_settlement_reversal'

        return None, None
```

## 費率數據獲取

### Binance API

```python
import ccxt

exchange = ccxt.binance({'options': {'defaultType': 'future'}})

# 當前費率
ticker = exchange.fetch_ticker('BTC/USDT')
current_rate = ticker.get('info', {}).get('lastFundingRate')

# 歷史費率
funding_history = exchange.fetch_funding_rate_history(
    'BTC/USDT',
    since=exchange.parse8601('2024-01-01T00:00:00Z'),
    limit=1000
)
```

### CoinGlass（推薦）

```python
import requests

url = "https://open-api.coinglass.com/public/v2/funding"
params = {
    "symbol": "BTC",
    "time_type": "h8"
}
headers = {"coinglassSecret": "YOUR_API_KEY"}

response = requests.get(url, params=params, headers=headers)
```

## 回測注意事項

1. **費率精確性**：使用歷史實際費率
2. **結算時間**：準確對齊結算時點
3. **交易成本**：包含現貨和永續費用
4. **滑點**：高費率期間滑點可能更大
5. **強平風險**：永續倉位的強平模擬
