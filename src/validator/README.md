# 策略驗證模組

完整的 5 階段策略驗證系統，確保策略真實有效。

## 目錄結構

```
src/validator/
├── __init__.py          # 模組導出
├── stages.py            # 5 階段驗證系統
├── monte_carlo.py       # Monte Carlo 模擬器
└── README.md            # 本檔案
```

## 5 階段驗證流程

### 階段 1：基礎回測
驗證策略基本獲利能力。

**門檻值：**
- `total_return > 0`（有獲利）
- `total_trades >= 30`（足夠樣本）
- `sharpe_ratio > 0.5`（風險調整報酬）
- `max_drawdown < 30%`（可承受風險）
- `profit_factor > 1.0`（盈虧比）

### 階段 2：統計檢驗
確認結果非隨機產生。

**檢驗項目：**
- t-test `p < 0.05`（顯著異於 0）
- Sharpe 95% CI 不包含 0
- 偏態 `|skew| < 2`（避免極端分布）

### 階段 3：穩健性測試
驗證策略在不同條件下仍有效。

**測試項目：**
- 參數敏感度 < 30%
- 時間一致性（前後半期皆獲利）
- 標的一致性（BTC/ETH 皆獲利）

### 階段 4：Walk-Forward 分析
驗證樣本外表現。

**門檻值：**
- WFA Efficiency >= 50%
- OOS 勝率 > 50%
- 無單窗口 > -10%

### 階段 5：Monte Carlo 模擬
評估風險分布。

**門檻值：**
- 5th percentile > 0
- 1st percentile > -30%
- Median > Original × 50%

## 使用方式

### 基本用法

```python
from src.validator import StageValidator
from src.strategies.momentum.rsi import RSIStrategy

# 1. 建立驗證器
validator = StageValidator()

# 2. 準備資料
data_btc = ...  # BTC OHLCV DataFrame
data_eth = ...  # ETH OHLCV DataFrame

# 3. 建立策略
strategy = RSIStrategy()
params = {'rsi_period': 14}

# 4. 執行驗證
result = validator.validate(
    strategy=strategy,
    data_btc=data_btc,
    data_eth=data_eth,
    params=params
)

# 5. 查看結果
print(result.summary())
print(f"評級: {result.grade.value}")
print(f"通過階段: {result.passed_stages}/5")
```

### 評級說明

| 評級 | 通過階段 | 說明 |
|------|----------|------|
| **A** | 5/5 | 優秀，可實盤測試 |
| **B** | 4/5 | 良好，降低倉位測試 |
| **C** | 3/5 | 及格，需改進 |
| **D** | 1-2/5 | 不及格，重新優化 |
| **F** | 0/5 | 失敗，重新設計 |

### 批次驗證

```python
strategies = [
    (RSIStrategy(), {'rsi_period': 14}),
    (MACDStrategy(), {'fast': 12, 'slow': 26}),
    (MAStrategy(), {'fast': 10, 'slow': 30}),
]

validator = StageValidator()
results = []

for strategy, params in strategies:
    result = validator.validate(
        strategy=strategy,
        data_btc=data_btc,
        data_eth=data_eth,
        params=params
    )

    results.append({
        'name': strategy.name,
        'grade': result.grade.value,
        'passed_stages': result.passed_stages,
    })

# 排序
results.sort(key=lambda x: x['passed_stages'], reverse=True)
```

## 驗證結果解讀

### 通過所有階段 (A 級)
✅ **建議行動：**
- 可以進入實盤測試（小倉位）
- 持續監控實盤表現
- 定期重新驗證（每季度）

### 通過 4 階段 (B 級)
⚠️ **建議行動：**
- Monte Carlo 風險較高
- 降低倉位（50%）
- 加入額外風控（止損）
- 謹慎進入實盤

### 通過 3 階段 (C 級)
🔶 **建議行動：**
- Walk-Forward 表現不佳
- 優化參數
- 延長測試期
- 暫緩實盤

### 未通過 3 階段 (D/F 級)
❌ **建議行動：**
- 策略邏輯有問題
- 重新設計
- 不建議實盤

## 詳細指標說明

### 階段 1 指標

```python
result.stage_results['階段1_基礎回測'].details
{
    'total_return': 0.35,      # 總報酬 35%
    'total_trades': 50,        # 交易次數
    'sharpe_ratio': 1.2,       # 夏普比率
    'max_drawdown': -0.15,     # 最大回撤 -15%
    'profit_factor': 1.5,      # 獲利因子
}
```

### 階段 2 指標

```python
result.stage_results['階段2_統計檢驗'].details
{
    't_statistic': 2.5,        # t 統計量
    'p_value': 0.013,          # p 值（< 0.05 顯著）
    'sharpe_ratio': 1.2,
    'sharpe_ci': (0.3, 2.1),   # 95% 信賴區間
    'skewness': -0.5,          # 偏態
}
```

### 階段 3 指標

```python
result.stage_results['階段3_穩健性'].details
{
    'param_sensitivity_pct': 15.0,    # 參數敏感度 15%
    'time_consistent': True,          # 時間一致性
    'asset_consistent': True,         # 標的一致性
}
```

### 階段 4 指標

```python
result.stage_results['階段4_WalkForward'].details
{
    'efficiency': 0.65,         # WFA 效率 65%
    'oos_win_rate': 0.6,        # OOS 勝率 60%
    'max_oos_dd': -0.08,        # 最大 OOS 回撤 -8%
    'oos_returns': [...],       # 各窗口 OOS 報酬
}
```

### 階段 5 指標

```python
result.stage_results['階段5_MonteCarlo'].details
{
    'original_return': 0.35,    # 原始報酬 35%
    'p1': -0.15,                # 1% 分位 -15%
    'p5': 0.05,                 # 5% 分位 5%
    'median': 0.32,             # 中位數 32%
    'p95': 0.60,                # 95% 分位 60%
}
```

## 進階配置

### 自訂門檻值

```python
validator = StageValidator()

# 修改門檻值
validator.thresholds['stage1']['sharpe_ratio'] = 1.0  # 提高要求
validator.thresholds['stage3']['param_sensitivity'] = 0.2  # 降低容忍度

result = validator.validate(...)
```

### 自訂回測配置

```python
from src.backtester.engine import BacktestConfig

config = BacktestConfig(
    symbol='BTCUSDT',
    timeframe='1h',
    start_date=start,
    end_date=end,
    initial_capital=10000,
    leverage=2,
    maker_fee=0.0002,
    taker_fee=0.0004,
)

result = validator.validate(
    strategy=strategy,
    data_btc=data_btc,
    data_eth=data_eth,
    params=params,
    config=config  # 使用自訂配置
)
```

## 常見問題

### Q1: 為什麼需要 5 階段驗證？

A: 單純回測容易過擬合。5 階段驗證從不同角度確保策略真實有效：
- 階段 1-2：基礎有效性
- 階段 3：穩健性
- 階段 4：樣本外表現
- 階段 5：風險評估

### Q2: 資料需要多長？

A: 建議至少 6 個月（約 4000 小時棒）：
- Walk-Forward 需要分割多個窗口
- 統計檢驗需要足夠樣本
- 時間一致性需要前後期對比

### Q3: 可以只跑部分階段嗎？

A: 可以，直接呼叫單一階段方法：

```python
validator = StageValidator()

# 只跑階段 1
stage1 = validator.stage1_basic_backtest(backtest_result)

# 只跑階段 2
stage2 = validator.stage2_statistical_tests(returns)
```

### Q4: Monte Carlo 模擬次數可調整嗎？

A: 可以：

```python
stage5 = validator.stage5_monte_carlo(
    trades=trades,
    n_simulations=5000  # 預設 1000
)
```

## 範例

完整範例請參考：
- `examples/stage_validation_example.py`
- `tests/test_stage_validator.py`

## 參考文獻

驗證方法論參考：
1. `.claude/skills/策略驗證/SKILL.md`
2. Walk-Forward Analysis (Pardo, 1992)
3. Monte Carlo Simulation in Trading (Burns, 2006)
4. Statistical Significance in Trading (Aronson, 2006)
