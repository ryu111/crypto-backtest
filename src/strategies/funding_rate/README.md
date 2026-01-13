# 資金費率策略模組

提供基於永續合約資金費率的交易策略。

## 策略列表

### 1. FundingArbStrategy - Delta Neutral 資金費率套利

**策略邏輯：**
- 當資金費率過高時（> entry_rate），做空永續合約並做多現貨以收取資金費率
- 當資金費率過低時（< -entry_rate），做多永續合約並做空現貨以收取資金費率
- 當資金費率回到正常水平時（|rate| < exit_rate）且持有時間足夠時出場

**參數：**
- `entry_rate`: 進場費率閾值（絕對值），預設 0.0003 (0.03%)
- `exit_rate`: 出場費率閾值（絕對值），預設 0.0001 (0.01%)
- `min_holding_periods`: 最少持有結算次數，預設 1

**使用範例：**
```python
from src.strategies import create_strategy

# 建立策略實例
strategy = create_strategy(
    'funding_rate_arb',
    entry_rate=0.0003,
    exit_rate=0.0001,
    min_holding_periods=1
)

# 產生訊號（需要資金費率數據）
long_entry, long_exit, short_entry, short_exit = (
    strategy.generate_signals_with_funding(data, funding_rates)
)

# 計算預期年化收益
annual_return = strategy.calculate_expected_annual_return(
    funding_rates,
    settlements_per_day=3
)
print(f"預期年化收益: {annual_return*100:.2f}%")
```

**參數優化空間：**
```python
{
    'entry_rate': {
        'type': 'float',
        'min': 0.0001,
        'max': 0.001,
        'default': 0.0003
    },
    'exit_rate': {
        'type': 'float',
        'min': 0.00005,
        'max': 0.0003,
        'default': 0.0001
    },
    'min_holding_periods': {
        'type': 'int',
        'min': 1,
        'max': 3,
        'default': 1
    }
}
```

**特點：**
- ✅ 市場中性（Delta Neutral）
- ✅ 低風險套利策略
- ✅ 穩定收益來源
- ⚠️ 需要資金費率數據
- ⚠️ 需要對沖功能（永續 + 現貨）

---

### 2. SettlementTradeStrategy - 結算時段交易

**策略邏輯：**
- 在資金費率結算前 N 小時監測費率
- 高資金費率時做多（預期多頭在結算前平倉，推高價格）
- 負資金費率時做空（預期空頭在結算前平倉，壓低價格）
- 結算後立即平倉

**參數：**
- `rate_threshold`: 費率閾值，預設 0.0001 (0.01%)
- `hours_before_settlement`: 結算前幾小時進場，預設 1

**使用範例：**
```python
from src.strategies import create_strategy

# 建立策略實例
strategy = create_strategy(
    'funding_rate_settlement',
    rate_threshold=0.0001,
    hours_before_settlement=1
)

# 產生訊號（需要資金費率數據）
long_entry, long_exit, short_entry, short_exit = (
    strategy.generate_signals_with_funding(data, funding_rates)
)
```

**參數優化空間：**
```python
{
    'rate_threshold': {
        'type': 'float',
        'low': 0.00005,
        'high': 0.0005,
        'log': True  # 對數空間搜尋
    },
    'hours_before_settlement': {
        'type': 'int',
        'low': 1,
        'high': 4
    }
}
```

**特點：**
- ✅ 短期交易策略
- ✅ 利用結算行為獲利
- ⚠️ 需要精確的時間數據
- ⚠️ 市場波動風險較高

---

## 資金費率數據要求

兩個策略都需要資金費率數據（Pandas Series），格式如下：

```python
# 資金費率 Series
funding_rates = pd.Series(
    [0.0001, 0.0002, -0.0001, 0.0005, ...],
    index=pd.DatetimeIndex([...])  # 必須與 OHLCV 數據時間對齊
)
```

**重要：**
- 必須使用 `generate_signals_with_funding()` 而非 `generate_signals()`
- 資金費率時間必須與 OHLCV 數據對齊
- 費率單位為小數（0.0001 = 0.01%）

---

## 回測整合

### 基本回測

```python
from src.backtester import BacktestEngine
from src.strategies import create_strategy

# 建立策略
strategy = create_strategy('funding_rate_arb', entry_rate=0.0003)

# 建立回測引擎
engine = BacktestEngine(
    strategy=strategy,
    initial_capital=100000,
    fee_rate=0.0004,
    slippage=0.0001
)

# 執行回測（需要資金費率數據）
result = engine.run(
    data=price_data,
    funding_rates=funding_rates  # 傳入資金費率
)

print(result.summary())
```

### 參數優化

```python
from src.optimizer import BayesianOptimizer

# 建立優化器
optimizer = BayesianOptimizer(
    strategy_name='funding_rate_arb',
    data=price_data,
    funding_rates=funding_rates,  # 傳入資金費率
    n_iterations=50
)

# 執行優化
best_params, best_result = optimizer.optimize()

print(f"最佳參數: {best_params}")
print(f"最佳績效: {best_result.sharpe_ratio}")
```

---

## 測試

執行測試腳本：

```bash
python test_funding_strategies.py
```

測試內容：
- ✅ 策略基本功能
- ✅ 訊號產生邏輯
- ✅ 參數驗證
- ✅ 策略註冊
- ✅ 策略建立

---

## 注意事項

1. **資金費率數據來源**
   - Binance API: `/fapi/v1/fundingRate`
   - 歷史數據需要批次下載
   - 建議存儲在資料庫中

2. **回測限制**
   - 無法完全模擬對沖成本（永續 + 現貨）
   - 忽略了資金費率變化的滑價
   - 假設永遠可以進出場

3. **實盤考量**
   - 需要同時持有永續和現貨部位
   - 需要監控對沖比例（Delta）
   - 需要考慮資金利用率

---

## 參考資料

- [Binance 資金費率機制](https://www.binance.com/en/support/faq/360033525031)
- [資金費率套利策略原理](https://academy.binance.com/en/articles/what-is-funding-rate-arbitrage)
