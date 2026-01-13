# Settlement Trade Strategy

## 策略概述

**策略類型**：Funding Rate
**註冊名稱**：`funding_rate_settlement`
**檔案位置**：`src/strategies/funding_rate/settlement_trade.py`

結算時段交易策略利用永續合約資金費率結算前的市場行為進行短期交易。

## 策略邏輯

### 理論基礎

永續合約透過資金費率機制保持價格與現貨接近：
- **正費率**：多頭支付給空頭（多頭成本 ↑）
- **負費率**：空頭支付給多頭（空頭成本 ↑）

當費率極端時，持倉方可能在結算前平倉以避免支付費用，造成短期價格波動。

### 交易邏輯

```
結算前 hours_before 小時
    ↓
檢查資金費率
    │
    ├── rate > +threshold → 做多
    │   （預期多頭平倉 → 價格上漲）
    │
    └── rate < -threshold → 做空
        （預期空頭平倉 → 價格下跌）
    ↓
結算時立即平倉
```

### 進出場規則

| 條件 | 操作 |
|------|------|
| 結算前 N 小時 + 高費率 | 開多單 |
| 結算前 N 小時 + 負費率 | 開空單 |
| 到達結算時間 | 平倉所有部位 |

## 參數說明

### rate_threshold

- **類型**：float
- **預設值**：0.0001（0.01%）
- **優化範圍**：0.00005 - 0.0005
- **說明**：費率閾值，超過此絕對值才觸發交易

**參數影響**：
- 過小 → 訊號過多，交易成本高
- 過大 → 訊號過少，錯過機會

### hours_before_settlement

- **類型**：int
- **預設值**：1
- **優化範圍**：1 - 4
- **說明**：結算前幾小時開始進場

**參數影響**：
- 1 小時 → 持倉時間短，風險低但可能錯過波動
- 4 小時 → 持倉時間長，捕捉更多波動但風險較高

## 使用範例

### 基本使用

```python
from src.strategies.funding_rate import SettlementTradeStrategy

# 建立策略實例
strategy = SettlementTradeStrategy(
    rate_threshold=0.0002,
    hours_before_settlement=2
)

# 生成訊號
long_entry, long_exit, short_entry, short_exit = \
    strategy.generate_signals_with_funding(data, funding_rates)
```

### 透過註冊表使用

```python
from src.strategies import create_strategy

strategy = create_strategy(
    'funding_rate_settlement',
    rate_threshold=0.00015,
    hours_before_settlement=1
)
```

### 回測範例

```python
from src.backtester import BacktestEngine

# 準備數據
data = load_ohlcv_data('BTCUSDT', '1h')
funding_rates = load_funding_rates('BTCUSDT', '1h')

# 建立策略
strategy = SettlementTradeStrategy(
    rate_threshold=0.0001,
    hours_before_settlement=1
)

# 執行回測
engine = BacktestEngine(
    initial_capital=10000,
    leverage=3,
    maker_fee=0.0002,
    taker_fee=0.0005
)

results = engine.run(
    strategy=strategy,
    data=data,
    funding_rates=funding_rates
)
```

## 資金費率數據格式

### 要求

1. **時間對齊**：必須與 OHLCV 數據的時間索引一致
2. **數據頻率**：建議小時級別（配合結算時間）
3. **費率格式**：小數表示（0.0001 = 0.01%）

### 範例

```python
import pandas as pd

# OHLCV 數據
data = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
}, index=pd.DatetimeIndex([...], tz='UTC'))

# 資金費率數據（必須時間對齊）
funding_rates = pd.Series(
    [0.0001, 0.0002, -0.0001, 0.0003, ...],
    index=data.index
)
```

## 結算時間

### 主流交易所

| 交易所 | 結算時間（UTC） | 頻率 |
|--------|----------------|------|
| Binance | 00:00, 08:00, 16:00 | 每 8 小時 |
| OKX | 00:00, 08:00, 16:00 | 每 8 小時 |
| Bybit | 00:00, 08:00, 16:00 | 每 8 小時 |

策略已內建這些結算時間，會自動識別並執行平倉。

## 績效指標

### 預期表現

| 指標 | 目標值 | 說明 |
|------|--------|------|
| 勝率 | > 55% | 短期交易建議勝率較高 |
| 平均持倉時間 | 1-4 小時 | 結算前進場，結算時平倉 |
| 最大回撤 | < 15% | 短期策略應控制回撤 |
| Sharpe Ratio | > 1.5 | 風險調整後收益 |

### 影響因素

1. **費率極端程度**：越極端效果越明顯
2. **市場波動度**：高波動期表現較好
3. **交易成本**：頻繁交易需注意手續費
4. **滑價**：結算時刻可能流動性降低

## 風險管理

### 主要風險

| 風險類型 | 說明 | 對策 |
|---------|------|------|
| 交易成本 | 頻繁進出場 | 提高費率閾值 |
| 滑價風險 | 結算時刻流動性下降 | 提前幾秒平倉 |
| 反向波動 | 市場未如預期 | 設定止損 |
| 費率突變 | 費率在持倉期間變化 | 縮短持倉時間 |

### 建議配置

```python
# 風險管理參數
RISK_CONFIG = {
    'max_position_size': 0.3,      # 最大部位 30%
    'stop_loss_pct': 0.02,         # 2% 止損
    'max_leverage': 3,             # 最大槓桿 3x
    'min_liquidity': 1000000       # 最小流動性要求
}
```

## 參數優化建議

### 優化方法

1. **網格搜尋**：
```python
param_grid = {
    'rate_threshold': [0.00005, 0.0001, 0.0002, 0.0003],
    'hours_before_settlement': [1, 2, 3, 4]
}
```

2. **貝葉斯優化**：
```python
from src.optimizer import BayesianOptimizer

optimizer = BayesianOptimizer(
    strategy='funding_rate_settlement',
    n_iterations=50
)

best_params = optimizer.optimize(data, funding_rates)
```

### 穩健性測試

```python
from src.validator import WalkForwardValidator

validator = WalkForwardValidator(
    strategy=strategy,
    train_period=90,   # 90 天訓練
    test_period=30     # 30 天測試
)

results = validator.validate(data, funding_rates)
```

## 回測注意事項

### 數據品質

- [ ] OHLCV 數據無缺失
- [ ] 資金費率數據完整
- [ ] 時間對齊正確（UTC）
- [ ] 包含至少 3 個月數據

### 交易成本

| 成本項目 | 建議值 | 說明 |
|---------|--------|------|
| Maker Fee | 0.02% | 掛單手續費 |
| Taker Fee | 0.05% | 吃單手續費 |
| 滑價 | 0.01-0.05% | 視市場深度 |
| 資金費率 | 實際費率 | 跨結算持倉需計入 |

### 市場環境

測試不同市場狀態：
- **牛市**（2021 Q1）
- **熊市**（2022 Q2）
- **震盪市**（2023 Q3）

## 實戰部署

### 部署清單

- [ ] 確認交易所支援資金費率 API
- [ ] 設定費率閾值（根據回測結果）
- [ ] 配置風險管理參數
- [ ] 設定監控告警
- [ ] 準備應急平倉機制

### 監控指標

```python
# 即時監控
metrics = {
    'current_funding_rate': rate,
    'hours_to_settlement': hours,
    'open_positions': positions,
    'unrealized_pnl': pnl,
    'daily_trades': count
}
```

## 延伸開發

### 改進方向

1. **動態閾值**：
   - 根據歷史費率分佈自動調整
   - 使用百分位數而非固定值

2. **多標的組合**：
   - 同時監控 BTC、ETH 等多個標的
   - 分散風險，提高資金利用率

3. **機器學習增強**：
   - 預測結算前價格走勢
   - 優化進出場時機

### 相關策略

- **Funding Arbitrage**：Delta Neutral 資金費率套利
- **Basis Trade**：永續/現貨基差套利
- **Liquidation Hunter**：強平獵取策略

---

**版本**：1.0
**最後更新**：2026-01-13
**測試狀態**：✅ 16/16 測試通過
