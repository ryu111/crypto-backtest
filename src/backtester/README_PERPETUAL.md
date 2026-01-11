# 永續合約計算模組

提供永續合約特有的計算功能，支援回測和實盤交易中的各種永續合約相關計算。

## 功能概覽

### 核心類別

| 類別 | 用途 |
|------|------|
| `PerpetualCalculator` | 永續合約數學計算 |
| `PerpetualPosition` | 倉位資料結構 |
| `PerpetualRiskMonitor` | 風險監控系統 |

## 快速開始

```python
from src.backtester.perpetual import (
    PerpetualCalculator,
    PerpetualPosition,
    PerpetualRiskMonitor
)

# 建立計算器
calc = PerpetualCalculator()

# 計算強平價格
liq_price = calc.calculate_liquidation_price(
    entry_price=50000,
    leverage=10,
    direction=1  # 1=做多，-1=做空
)

# 計算未實現盈虧
pnl = calc.calculate_unrealized_pnl(
    entry_price=50000,
    mark_price=52000,
    size=1.0,
    direction=1
)
```

## 功能詳解

### 1. 資金費率計算

```python
# 計算單次資金費率成本
funding_cost = calc.calculate_funding_cost(
    position_value=10000,  # 持倉價值 $10,000
    funding_rate=0.0001,   # 費率 0.01%
    direction=1            # 做多
)
# 結果: 1.0 USDT（做多支付）

# 計算年化影響
annualized = calc.annualized_funding_rate(0.0001)
# 結果: 0.1095（10.95% 年化）

# 應用到權益曲線
adjusted_equity = calc.apply_funding_to_equity(
    equity_curve,
    positions,
    funding_rates,
    position_sizes
)
```

#### 資金費率結算時間

| 交易所 | 結算時間 (UTC) | 週期 |
|--------|---------------|------|
| Binance | 00:00, 08:00, 16:00 | 8h |
| Bybit | 00:00, 08:00, 16:00 | 8h |
| OKX | 00:00, 08:00, 16:00 | 8h |

### 2. 保證金計算

```python
# 計算初始保證金
margin = calc.calculate_initial_margin(
    position_size=1.0,
    entry_price=50000,
    leverage=10
)
# 結果: 5000.0 USDT

# 計算保證金率
ratio = calc.calculate_margin_ratio(
    equity=5500,
    position_value=50000
)
# 結果: 0.11（11%）

# 計算可用保證金
available = calc.calculate_available_margin(
    total_equity=10000,
    used_margin=5000
)
# 結果: 5000 USDT
```

### 3. 強平計算

```python
# 計算強平價格
liq_price_long = calc.calculate_liquidation_price(
    entry_price=50000,
    leverage=10,
    direction=1  # 做多
)
# 結果: 45250.0（跌 9.5% 爆倉）

liq_price_short = calc.calculate_liquidation_price(
    entry_price=50000,
    leverage=10,
    direction=-1  # 做空
)
# 結果: 54750.0（漲 9.5% 爆倉）

# 檢查是否爆倉
is_liquidated = calc.check_liquidation(
    current_price=45000,
    entry_price=50000,
    leverage=10,
    direction=1
)
# 結果: True

# 計算距離強平的距離
distance_pct, distance_price = calc.calculate_liquidation_distance(
    current_price=48000,
    entry_price=50000,
    leverage=10,
    direction=1
)
# 結果: (-5.73, -2750.0)
```

#### 強平公式

**做多:**
```
強平價 = 入場價 × (1 - 1/槓桿 + 維持保證金率)
```

**做空:**
```
強平價 = 入場價 × (1 + 1/槓桿 - 維持保證金率)
```

### 4. 盈虧計算

```python
# 計算未實現盈虧
unrealized_pnl = calc.calculate_unrealized_pnl(
    entry_price=50000,
    mark_price=52000,
    size=1.0,
    direction=1  # 做多
)
# 結果: 2000.0 USDT

# 計算盈虧百分比
pnl_pct = calc.calculate_pnl_percentage(
    pnl=500,
    margin=5000
)
# 結果: 10.0（10% ROI）
```

### 5. Mark Price 和基差

```python
# 計算基差
basis_abs, basis_pct = calc.calculate_basis(
    perp_price=50500,
    spot_price=50000
)
# 結果: (500.0, 1.0)（溢價 $500，1%）

# 計算 Mark Price（簡化版）
mark_price = calc.calculate_mark_price(
    spot_price=50000,
    perp_price=50100
)
```

### 6. 風險指標

```python
# 計算有效槓桿
eff_leverage = calc.calculate_effective_leverage(
    position_value=50000,
    equity=5500
)
# 結果: 9.09x

# 計算破產價格（100% 虧損）
bankruptcy_price = calc.calculate_bankruptcy_price(
    entry_price=50000,
    leverage=10,
    direction=1
)
# 結果: 45000.0

# 估算最大倉位
max_size = calc.estimate_max_position_size(
    available_capital=10000,
    price=50000,
    leverage=10
)
# 結果: 1.9992 BTC
```

### 7. 倉位管理

```python
from datetime import datetime

# 建立倉位
position = PerpetualPosition(
    entry_price=50000,
    size=1.0,           # 正數=做多，負數=做空
    leverage=10,
    entry_time=datetime.now(),
    margin=5000
)

# 倉位屬性
print(position.direction)       # 1（做多）
print(position.is_long)         # True
print(position.notional_value)  # 50000.0
```

### 8. 風險監控

```python
# 建立監控器
monitor = PerpetualRiskMonitor(
    warning_threshold=0.02,   # 距離強平 2%
    critical_threshold=0.01   # 距離強平 1%
)

# 評估風險等級
risk_level = monitor.assess_risk_level(
    position=position,
    current_price=48000
)
# 結果: "safe" | "warning" | "critical" | "liquidated"

# 生成完整風險報告
report = monitor.generate_risk_report(
    position=position,
    current_price=48000
)
# 回傳字典包含：
# - risk_level: 風險等級
# - liquidation_price: 強平價格
# - distance_to_liquidation_pct: 距離強平百分比
# - margin_ratio: 保證金率
# - unrealized_pnl: 未實現盈虧
```

## 實際應用案例

### 案例 1：回測中整合資金費率

```python
import pandas as pd

# 建立資金費率數據
funding_rates = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=90, freq='8H'),
    'rate': [0.0001] * 90
})

# 應用到權益曲線
calc = PerpetualCalculator()
adjusted_equity = calc.apply_funding_to_equity(
    equity_curve,
    positions,
    funding_rates,
    position_sizes
)
```

### 案例 2：資金費率套利策略

```python
# Delta Neutral 策略：現貨做多 + 永續做空
capital = 20000
spot_investment = capital / 2
perp_margin = capital / 2

# 計算收益
holding_days = 30
funding_intervals = (holding_days * 24) // 8
avg_funding_rate = 0.0003

total_funding = 0
for _ in range(funding_intervals):
    funding = calc.calculate_funding_cost(
        position_value=10000,
        funding_rate=avg_funding_rate,
        direction=-1  # 做空收取
    )
    total_funding += funding

# 年化收益率
annualized_return = (abs(total_funding) / capital) * (365 / holding_days)
```

### 案例 3：動態風險管理

```python
# 即時監控倉位風險
positions = [...]  # 多個倉位

for position in positions:
    report = monitor.generate_risk_report(position, current_price)

    if report['risk_level'] == 'critical':
        # 減倉或平倉
        print(f"警告：倉位風險過高，距離強平僅 {report['distance_to_liquidation_pct']:.2f}%")

    elif report['risk_level'] == 'warning':
        # 提醒注意
        print(f"注意：倉位接近風險區域")
```

## 配置參數

### 維持保證金率

| 交易所 | 主流幣 | 山寨幣 |
|--------|--------|--------|
| Binance | 0.4% | 0.5-2.5% |
| Bybit | 0.5% | 0.5-2.0% |
| OKX | 0.5% | 0.5-5.0% |

預設值：`0.5%`

### 資金費率週期

預設：`8 小時`

可根據交易所調整：
```python
calc = PerpetualCalculator(funding_interval_hours=8)
```

## 注意事項

### 回測中的考慮

1. **資金費率數據**：需要歷史資金費率數據，無法僅用 OHLCV
2. **Mark Price**：理想使用 Mark Price，但通常只有 Last Price
3. **強平滑點**：實際強平可能有額外滑點（1-5%）
4. **ADL 機制**：自動減倉在回測中通常忽略
5. **保險基金**：回測中通常不考慮

### 常見錯誤

| 錯誤 | 影響 | 解決方案 |
|------|------|----------|
| 忽略資金費率 | 高估收益 10-100%+ | 整合費率數據 |
| 固定滑點 | 低估成本 | 動態滑點模型 |
| 無強平機制 | 虛假績效 | 實作強平邏輯 |
| 用現貨費率 | 成本不準 | 用永續費率 |

### 性能優化

對於大規模回測：

```python
# 使用 NumPy 向量化計算
positions_array = np.array([...])
prices_array = np.array([...])

# 批次計算盈虧
pnls = calc.calculate_unrealized_pnl(
    entry_prices_array,
    mark_prices_array,
    sizes_array,
    directions_array
)
```

## 測試

執行測試：
```bash
# 單元測試
python -m pytest tests/test_perpetual.py -v

# 功能範例
PYTHONPATH=. python examples/perpetual_example.py
```

## 參考文件

- `.claude/skills/永續合約/SKILL.md` - 永續合約專用知識
- `.claude/skills/回測核心/references/perpetual-mechanics.md` - 永續合約機制

## API 完整列表

### PerpetualCalculator

**資金費率:**
- `calculate_funding_cost(position_value, funding_rate, direction)`
- `calculate_total_funding(trades, funding_rates)`
- `annualized_funding_rate(avg_rate)`
- `apply_funding_to_equity(equity_curve, positions, funding_rates, position_sizes)`

**保證金:**
- `calculate_initial_margin(position_size, entry_price, leverage)`
- `calculate_margin_ratio(equity, position_value)`
- `calculate_available_margin(total_equity, used_margin)`

**強平:**
- `calculate_liquidation_price(entry_price, leverage, direction, maintenance_margin_rate=None)`
- `check_liquidation(current_price, entry_price, leverage, direction, maintenance_margin_rate=None)`
- `calculate_liquidation_distance(current_price, entry_price, leverage, direction)`

**盈虧:**
- `calculate_unrealized_pnl(entry_price, mark_price, size, direction)`
- `calculate_realized_pnl(entry_price, exit_price, size, direction)`
- `calculate_pnl_percentage(pnl, margin)`

**Mark Price:**
- `calculate_mark_price(spot_price, perp_price, window_size=30)`
- `calculate_basis(perp_price, spot_price)`

**風險:**
- `calculate_effective_leverage(position_value, equity)`
- `calculate_bankruptcy_price(entry_price, leverage, direction)`
- `estimate_max_position_size(available_capital, price, leverage, fee_rate=0.0004)`

### PerpetualPosition

**屬性:**
- `entry_price: float`
- `size: float`（正=做多，負=做空）
- `leverage: int`
- `entry_time: datetime`
- `margin: float`
- `unrealized_pnl: float`
- `total_funding_paid: float`

**計算屬性:**
- `direction: int`（1=做多，-1=做空，0=無）
- `notional_value: float`
- `is_long: bool`
- `is_short: bool`

### PerpetualRiskMonitor

**方法:**
- `assess_risk_level(position, current_price)` → "safe" | "warning" | "critical" | "liquidated"
- `generate_risk_report(position, current_price)` → dict

## 授權

MIT License
