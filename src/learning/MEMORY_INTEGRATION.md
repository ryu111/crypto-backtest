# Memory MCP 整合快速指南

## 概述

Memory MCP 整合提供跨專案的語義化知識存儲，用於累積交易策略開發經驗。

## 核心特性

- ✅ 策略洞察存儲（最佳參數、績效、適用條件）
- ✅ 市場洞察記錄（標的特性、市場狀態）
- ✅ 失敗教訓追蹤（過擬合、失敗原因）
- ✅ 自動標籤推斷（從策略名稱、標的推斷）
- ✅ 語義搜尋建議（產生檢索查詢）
- ✅ Wrapper 設計（格式化後由 Claude 執行 MCP 呼叫）

## 快速開始

### 1. 存儲策略洞察

```python
from src.learning import MemoryIntegration, StrategyInsight

memory = MemoryIntegration()

# 建立洞察
insight = StrategyInsight(
    strategy_name="MA Cross",
    symbol="BTCUSDT",
    timeframe="4h",
    best_params={"fast": 10, "slow": 30},
    sharpe_ratio=1.85,
    total_return=0.456,
    max_drawdown=-0.12,
    win_rate=0.58,
    wfa_efficiency=0.68,
    wfa_grade="A"
)

# 格式化並印出範例
content, metadata = memory.format_strategy_insight(insight)
memory.print_storage_example(content, metadata)
```

### 2. 檢索歷史最佳

```python
# 產生檢索建議
query = memory.suggest_best_params_query(
    strategy_type="ma-cross",
    symbol="BTCUSDT",
    timeframe="4h"
)

tags = memory.suggest_tags_for_search(
    strategy_type="trend",
    symbol="BTCUSDT",
    timeframe="4h",
    status="validated"
)

# 印出檢索範例
memory.print_retrieval_example(query, tags)
```

### 3. 存儲失敗教訓

```python
from src.learning import TradingLesson

lesson = TradingLesson(
    strategy_name="RSI Mean Reversion",
    symbol="BTCUSDT",
    timeframe="1h",
    failure_type="overfitting",
    description="參數在 OOS 期間完全失效",
    symptoms="WFA Efficiency 僅 15%，Grade F",
    prevention="增加 WFA 窗口數量，檢查參數穩定性"
)

content, metadata = memory.format_trading_lesson(lesson)
memory.print_storage_example(content, metadata)
```

## AI Loop 整合

```python
from src.learning import retrieve_best_params_guide

# Step 1: 優化前查詢歷史
retrieve_best_params_guide(
    memory,
    strategy_type="ma-cross",
    symbol="BTCUSDT",
    timeframe="4h"
)

# Step 2: 執行優化與驗證
# ... BayesianOptimizer + WalkForwardAnalyzer ...

# Step 3: 存儲結果
from src.learning import store_successful_experiment, store_failed_experiment

if wfa_result.grade in ["A", "B", "C"]:
    store_successful_experiment(
        memory=memory,
        strategy_name="MA Cross",
        symbol="BTCUSDT",
        timeframe="4h",
        opt_result=optimization_result,
        wfa_result=wfa_result,
        market_conditions="趨勢明確的牛市"
    )
else:
    store_failed_experiment(
        memory=memory,
        strategy_name="MA Cross",
        symbol="BTCUSDT",
        timeframe="4h",
        opt_result=optimization_result,
        wfa_result=wfa_result,
        failure_description="參數不穩定",
        prevention_advice="增加驗證期"
    )
```

## 標籤系統

### 自動推斷

策略名稱 → 策略類型標籤：
- "MA Cross" → `ma-cross`, `trend`
- "RSI Momentum" → `momentum`
- "Bollinger Bands" → `mean-reversion`

交易標的 → 資產標籤：
- "BTCUSDT" → `btc`, `crypto`
- "ETHUSDT" → `eth`, `crypto`

市場類型 → 市場標籤：
- "bull" → `bull`
- "volatile" → `volatile`

### 預定義標籤

```python
from src.learning import MemoryTags

# 策略類型
MemoryTags.STRATEGY_TREND           # "trend"
MemoryTags.STRATEGY_MOMENTUM        # "momentum"
MemoryTags.STRATEGY_MEAN_REVERSION  # "mean-reversion"

# 狀態
MemoryTags.STATUS_VALIDATED  # "validated"
MemoryTags.STATUS_FAILED     # "failed"
```

## Memory MCP 呼叫

### 存儲

```python
mcp__memory-service__store_memory(
    content="策略洞察內容...",
    metadata={
        "tags": "crypto,btc,4h,validated,trend",
        "type": "trading-insight"
    }
)
```

### 檢索

```python
# 語義搜尋（推薦）
mcp__memory-service__retrieve_with_quality_boost(
    query="MA Cross BTCUSDT 4h best parameters",
    n_results=5,
    quality_weight=0.3
)

# 標籤搜尋
mcp__memory-service__search_by_tag(
    tags=["trend", "btc", "validated"]
)
```

## 架構說明

```
Python Code (格式化)
    ↓
  print_storage_example()
    ↓
顯示 MCP 呼叫範例
    ↓
Claude 看到範例
    ↓
執行 MCP 呼叫
    ↓
Memory MCP Service (語義存儲)
```

**關鍵**：此模組是 wrapper，不直接呼叫 MCP API，而是產生格式化內容和範例供 Claude 執行。

## 測試

```bash
# 執行測試
pytest tests/test_memory_standalone.py -v

# 所有測試應該通過（25 個測試）
```

## 檔案位置

| 檔案 | 說明 |
|------|------|
| `src/learning/memory.py` | 主要實作 |
| `src/learning/__init__.py` | 匯出介面 |
| `tests/test_memory_standalone.py` | 單元測試 |
| `examples/memory_integration_example.py` | 使用範例 |
| `src/learning/README.md` | 完整文件 |

## 進一步閱讀

- `src/learning/memory.py` - 完整原始碼和文件字串
- `src/learning/README.md` - 學習系統完整說明
- `tests/test_memory_standalone.py` - 測試案例作為使用範例
