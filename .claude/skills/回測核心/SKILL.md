---
name: backtest-core
description: 永續合約回測核心引擎。執行回測、產生報告、分析績效。當需要執行策略回測、查看回測結果、理解回測機制時使用。
---

# 回測核心引擎

基於 VectorBT 的 BTC/ETH 永續合約回測系統。

## Quick Start

```python
from src.backtester import run_backtest

# 執行回測
result = run_backtest(
    strategy="ma_cross",
    symbol="BTCUSDT",
    timeframe="4h",
    period="2024-01-01:2025-12-31",
    leverage=5,
    initial_capital=10000
)

# 查看績效
print(result.stats())
result.plot()
```

## 回測流程

```
1. 載入資料 → data-pipeline skill
2. 計算指標 → indicator-lib skill
3. 產生訊號 → strategy-dev skill
4. 執行回測 → Portfolio.from_signals()
5. 計算績效 → 績效指標
6. 產生報告 → HTML/PDF
```

## 永續合約特性

| 特性 | 說明 | 處理方式 |
|------|------|----------|
| 資金費率 | 每 8 小時結算 | 計入持倉成本 |
| 槓桿 | 1-125x | 影響保證金和爆倉價 |
| 雙向持倉 | 做多/做空 | 分開計算 PnL |
| 強制平倉 | 保證金不足時 | 模擬爆倉機制 |
| Mark Price | 標記價格 | 計算未實現盈虧 |

## 績效指標

| 指標 | 說明 | 目標值 |
|------|------|--------|
| Total Return | 總報酬率 | > 基準 |
| Annual Return | 年化報酬 | > 20% |
| Sharpe Ratio | 風險調整報酬 | > 1.5 |
| Sortino Ratio | 下行風險調整 | > 2.0 |
| Calmar Ratio | 報酬/回撤比 | > 2.0 |
| Max Drawdown | 最大回撤 | < 20% |
| Win Rate | 勝率 | > 50% |
| Profit Factor | 獲利因子 | > 1.5 |
| Total Trades | 交易次數 | >= 30 |

## VectorBT 核心 API

```python
import vectorbtpro as vbt

# 建立 Portfolio
pf = vbt.Portfolio.from_signals(
    close=price_data,
    entries=long_entries,
    exits=long_exits,
    short_entries=short_entries,
    short_exits=short_exits,
    leverage=leverage,
    fees=0.0006,  # 0.06% taker fee
    slippage=0.001,  # 0.1% 滑點
    init_cash=initial_capital,
    freq='4h'
)

# 績效統計
stats = pf.stats()
trades = pf.trades.records_readable

# 視覺化
pf.plot().show()
pf.trades.plot().show()
```

## 回測配置模板

```yaml
# config/backtest.yaml
backtest:
  symbol: BTCUSDT
  timeframe: 4h
  period:
    start: "2024-01-01"
    end: "2025-12-31"

  capital:
    initial: 10000
    currency: USDT

  leverage:
    default: 5
    max: 20

  fees:
    maker: 0.0002  # 0.02%
    taker: 0.0004  # 0.04%
    funding_rate: true  # 計入資金費率

  slippage:
    model: "percentage"
    value: 0.001  # 0.1%

  risk:
    max_position_size: 0.5  # 最大 50% 資金
    stop_loss: true
    take_profit: true
```

## 回測報告內容

1. **績效摘要**：總報酬、Sharpe、回撤
2. **交易統計**：勝率、盈虧比、平均持倉
3. **時間分析**：月度報酬、回撤期間
4. **風險分析**：VaR、波動率、最大連續虧損
5. **圖表**：權益曲線、回撤圖、交易分布

## 注意事項

- 永續合約需自建資金費率模組
- 強平機制需單獨實作
- 高槓桿回測需謹慎處理爆倉
- 滑點在高波動期間可能更大

## 與其他 Skills 關係

### 本 Skill 調用（下游）

| Skill | 調用場景 |
|-------|----------|
| **資料管道** | 獲取 OHLCV 和資金費率資料 |
| **風險管理** | 部位大小計算、止損執行 |
| **永續合約** | 資金費率扣除、強平模擬 |

### 被調用（上游）

| Skill | 場景 |
|-------|------|
| **AI自動化** | 自動化回測執行 |
| **策略開發** | 測試策略有效性 |
| **參數優化** | 每次試驗執行回測 |
| **策略驗證** | 各階段驗證回測 |

### 回測執行流程

```
策略訊號輸入
    ↓
回測核心
    ├─→ 資料管道（載入 OHLCV）
    ├─→ 永續合約（資金費率、強平）
    └─→ 風險管理（部位、止損）
    ↓
輸出績效報告
```

For VectorBT 詳細用法 → read `references/vectorbt-basics.md`
For 永續合約機制 → read `references/perpetual-mechanics.md`
