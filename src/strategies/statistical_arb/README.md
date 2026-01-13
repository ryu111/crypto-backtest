# 統計套利策略

統計套利策略基於統計方法找出價格偏離並等待回歸。

## 策略列表

### 1. Basis Arbitrage（基差套利）

**策略 ID**: `statistical_arb_basis`

#### 策略邏輯

基於永續合約相對於現貨的基差進行套利：

```
基差 = (永續價格 - 現貨價格) / 現貨價格
```

- **做空永續**（基差 > entry_threshold）：永續溢價過高，預期回歸
- **做多永續**（基差 < -entry_threshold）：永續折價過多，預期回歸
- **出場**：基差回到 ±exit_threshold 範圍內

#### 參數

| 參數 | 類型 | 預設值 | 範圍 | 說明 |
|------|------|--------|------|------|
| `entry_threshold` | float | 0.005 | 0.003-0.01 | 進場閾值（0.5% = 0.005） |
| `exit_threshold` | float | 0.001 | 0.0005-0.003 | 出場閾值（0.1% = 0.001） |
| `period` | int | 20 | 10-50 | 移動平均週期 |
| `use_ma` | bool | True | - | 是否使用 MA 平滑基差 |

#### 使用範例

```python
from src.strategies import create_strategy

# 建立策略
strategy = create_strategy(
    'statistical_arb_basis',
    entry_threshold=0.005,  # 0.5%
    exit_threshold=0.001,   # 0.1%
    period=20,
    use_ma=True
)

# 雙標的模式（永續 + 現貨）
long_entry, long_exit, short_entry, short_exit = strategy.generate_signals_dual(
    perp_data,  # 永續合約 OHLCV
    spot_data   # 現貨 OHLCV
)

# 單標的模式（僅永續，會回傳空訊號）
signals = strategy.generate_signals(perp_data)  # 不建議使用
```

#### 訊號說明

| 訊號 | 說明 | 對沖操作 |
|------|------|----------|
| `long_entry=True` | 基差 < -0.5%，做多永續 | 同時做空現貨 |
| `long_exit=True` | 基差 > -0.1%，多單出場 | 平倉現貨空單 |
| `short_entry=True` | 基差 > 0.5%，做空永續 | 同時做多現貨 |
| `short_exit=True` | 基差 < 0.1%，空單出場 | 平倉現貨多單 |

#### 風險提示

1. **需要雙邊對沖**：必須同時操作永續和現貨以鎖定基差
2. **資金費率影響**：永續合約有資金費率，需計入成本
3. **流動性風險**：需要足夠流動性同時建立雙邊部位
4. **極端行情**：市場劇烈波動時基差可能持續偏離

#### 適用場景

- **套利交易**：低風險套利，賺取基差回歸收益
- **市場中性**：Delta Neutral 策略
- **資金利用**：現貨+永續組合策略

---

## 開發指南

### 新增統計套利策略

```python
from ..base import StatisticalArbStrategy
from ..registry import register_strategy

@register_strategy('statistical_arb_my_strategy')
class MyStatArbStrategy(StatisticalArbStrategy):
    name = "My Statistical Arbitrage"
    strategy_type = "statistical_arbitrage"
    version = "1.0"

    params = {
        'param1': 10,
        'param2': 20,
    }

    param_space = {
        'param1': {'type': 'int', 'low': 5, 'high': 20},
        'param2': {'type': 'int', 'low': 10, 'high': 50},
    }

    def calculate_indicators(self, data):
        # 計算指標
        return {'indicator': data['close'].rolling(10).mean()}

    def generate_signals_dual(self, data_primary, data_secondary):
        # 產生雙標的訊號
        # ...
        return long_entry, long_exit, short_entry, short_exit
```

### 測試策略

```python
# tests/test_statistical_arb.py
import pytest
from src.strategies import create_strategy

def test_my_strategy():
    strategy = create_strategy('statistical_arb_my_strategy')
    assert strategy.name == "My Statistical Arbitrage"
    # ...
```
