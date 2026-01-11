# AI Learning System - Memory MCP 整合

## 概述

Memory MCP 整合模組提供跨專案知識存儲與檢索功能，用於累積交易策略開發的經驗。

## 架構

```
learning/
├── recorder.py         # 實驗記錄器
├── memory.py          # Memory MCP 整合
└── __init__.py

專案根目錄/
└── learning/
    ├── experiments.json    # 實驗記錄（結構化）
    └── insights.md        # 洞察彙整（人類可讀）
```

## 雙軌記錄

| 存儲位置 | 用途 | 格式 | 查詢方式 |
|----------|------|------|----------|
| `learning/experiments.json` | 詳細實驗數據 | 結構化 JSON | 精確查詢 |
| `learning/insights.md` | 彙整洞察 | Markdown | 人類閱讀 |
| Memory MCP | 跨專案洞察 | 語義化文字 | 語義搜尋 |

## 快速開始

### 1. 記錄實驗

```python
from src.learning import ExperimentRecorder
from src.backtester.engine import BacktestEngine
from src.validator.stages import StageValidator

# 執行回測
engine = BacktestEngine(config)
result = engine.run(strategy, params, data)

# 驗證策略
validator = StageValidator()
validation = validator.validate(strategy, data_btc, data_eth, params)

# 記錄實驗
recorder = ExperimentRecorder()
exp_id = recorder.log_experiment(
    result=result,
    strategy_info={
        'name': 'ma_cross_4h',
        'type': 'trend',
        'version': '1.0'
    },
    config={
        'symbol': 'BTCUSDT',
        'timeframe': '4h',
        'initial_capital': 10000,
        'leverage': 5
    },
    validation_result=validation,
    insights=[
        'ATR 2x 止損在高波動期間表現更好',
        '慢線 30 優於 20，減少假訊號'
    ]
)

print(f"實驗已記錄: {exp_id}")
```

### 2. 查詢實驗

```python
# 查詢所有趨勢策略
trend_strategies = recorder.query_experiments({
    'strategy_type': 'trend'
})

# 查詢 BTC 高 Sharpe 策略
best_btc = recorder.query_experiments({
    'symbol': 'BTCUSDT',
    'min_sharpe': 1.5,
    'grade': ['A', 'B']
})

# 查詢最近一週的實驗
from datetime import datetime, timedelta
recent = recorder.query_experiments({
    'date_range': (
        datetime.now() - timedelta(days=7),
        datetime.now()
    )
})
```

### 3. 取得最佳策略

```python
# Top 10 Sharpe Ratio
best_sharpe = recorder.get_best_experiments('sharpe_ratio', n=10)

# 僅查詢驗證通過的
best_validated = recorder.get_best_experiments(
    'sharpe_ratio',
    n=10,
    filters={'grade': ['A', 'B']}
)

for exp in best_validated:
    print(f"{exp.strategy['name']}: Sharpe {exp.results['sharpe_ratio']:.2f}")
```

### 4. 追蹤策略演進

```python
# 追蹤 MA 交叉策略的演進
evolution = recorder.get_strategy_evolution('ma_cross')

for entry in evolution:
    print(f"v{entry['version']}: Sharpe {entry['sharpe']:.2f}")
    if entry['improvement']:
        print(f"  改進: {entry['improvement']:.1%}")
```

## 實驗記錄結構

```json
{
  "id": "exp_20260111_120000",
  "timestamp": "2026-01-11T12:00:00",
  "strategy": {
    "name": "ma_cross_4h_v2",
    "type": "trend",
    "version": "2.0"
  },
  "config": {
    "symbol": "BTCUSDT",
    "timeframe": "4h",
    "initial_capital": 10000,
    "leverage": 5
  },
  "parameters": {
    "fast_period": 10,
    "slow_period": 30
  },
  "results": {
    "total_return": 0.456,
    "sharpe_ratio": 1.85,
    "max_drawdown": -0.10,
    "win_rate": 0.55,
    "profit_factor": 1.72,
    "total_trades": 124
  },
  "validation": {
    "grade": "A",
    "passed_stages": 5,
    "walk_forward_efficiency": 0.68,
    "monte_carlo_p5": 0.15
  },
  "insights": [
    "止損 2x ATR 在高波動期間表現更好",
    "慢線 30 優於 20，減少假訊號"
  ],
  "tags": ["crypto", "btc", "trend", "ma", "4h", "validated"]
}
```

## 查詢過濾器

### 可用過濾條件

```python
filters = {
    # 策略類型
    'strategy_type': 'trend',        # trend, momentum, mean_reversion

    # 標的
    'symbol': 'BTCUSDT',             # BTCUSDT, ETHUSDT

    # 績效指標
    'min_sharpe': 1.0,               # Sharpe Ratio >= 1.0
    'max_drawdown': 0.20,            # Max Drawdown <= 20%
    'min_return': 0.3,               # Total Return >= 30%

    # 驗證等級
    'grade': ['A', 'B'],             # A, B, C, D, F

    # 標籤
    'tags': ['validated', 'ma'],     # 必須包含所有標籤

    # 時間範圍
    'date_range': (
        datetime(2026, 1, 1),
        datetime(2026, 1, 11)
    )
}
```

## 標籤系統

### 自動生成標籤

系統會根據策略資訊和配置自動生成標籤：

| 類別 | 標籤範例 | 來源 |
|------|----------|------|
| 資產類別 | crypto | 固定 |
| 標的 | btc, eth | 從 symbol 解析 |
| 策略類型 | trend, momentum | strategy.type |
| 指標 | ma, rsi, macd | 從 strategy.name 解析 |
| 時間框架 | 1h, 4h, 1d | config.timeframe |
| 驗證狀態 | validated, testing, failed | validation.grade |

### 標籤用途

```python
# 查詢所有 BTC 趨勢策略
btc_trend = recorder.query_experiments({
    'tags': ['btc', 'trend']
})

# 查詢驗證通過的 MA 策略
validated_ma = recorder.query_experiments({
    'tags': ['validated', 'ma']
})
```

## 洞察更新

當記錄實驗時，系統會自動更新 `learning/insights.md`：

### 更新區塊

1. **策略類型洞察**：根據 `strategy.type` 更新對應區塊
2. **標的特性**：根據 `config.symbol` 更新標的特性
3. **失敗教訓**：驗證不通過（D/F 等級）時記錄教訓

### 洞察格式

```markdown
### MA 交叉

#### ma_cross_4h_v2
- **最佳參數**：fast_period=10, slow_period=30
- **績效**：Sharpe 1.85, Return 45.6%
- **驗證等級**：A
- **洞察**：止損 2x ATR 在高波動期間表現更好
```

## 策略演進追蹤

### 追蹤同一策略的不同版本

```python
# 記錄第一版
exp_v1 = recorder.log_experiment(
    result=result_v1,
    strategy_info={'name': 'ma_cross', 'type': 'trend', 'version': '1.0'},
    ...
)

# 記錄改進版（指定父實驗）
exp_v2 = recorder.log_experiment(
    result=result_v2,
    strategy_info={'name': 'ma_cross', 'type': 'trend', 'version': '2.0'},
    parent_experiment=exp_v1,  # 追蹤演進
    ...
)

# 查看演進
evolution = recorder.get_strategy_evolution('ma_cross')
for entry in evolution:
    print(f"v{entry['version']}: {entry['improvement']:.1%} improvement")
```

## 整合 Memory MCP

實驗記錄器專注於**詳細的結構化記錄**，而 Memory MCP 用於**跨專案的語義洞察**。

兩者搭配使用：

```python
from src.learning import ExperimentRecorder
from src.learning.memory import store_successful_experiment

# 1. 記錄詳細實驗數據
recorder = ExperimentRecorder()
exp_id = recorder.log_experiment(...)

# 2. 如果驗證通過，存入 Memory MCP
if validation.grade in ['A', 'B']:
    await store_successful_experiment(
        experiment_id=exp_id,
        strategy_info=strategy_info,
        config=config,
        results=result.to_dict(),
        validation=validation,
        insights=insights
    )
```

## 使用範例

完整範例請參考：
- `examples/learning/record_experiment.py` - 完整使用範例
- `tests/test_recorder.py` - 單元測試

## 最佳實踐

### 1. 一致的命名

```python
# 策略命名：{type}_{indicator}_{timeframe}_v{version}
strategy_info = {
    'name': 'trend_ma_cross_4h_v2',
    'type': 'trend',
    'version': '2.0'
}
```

### 2. 完整的洞察

```python
insights = [
    '止損設定：ATR 2x 優於固定百分比',
    '參數範圍：快線 8-12，慢線 25-35 最穩定',
    '市場適用性：趨勢明確時期表現最佳'
]
```

### 3. 追蹤演進

```python
# 記錄改進時，指定父實驗
exp_id_v2 = recorder.log_experiment(
    ...,
    parent_experiment=exp_id_v1
)
```

### 4. 定期分析

```python
# 每週分析最佳策略
best_this_week = recorder.query_experiments({
    'date_range': (week_start, week_end),
    'grade': ['A', 'B']
})

# 比較不同時期的表現
```

## API 參考

### ExperimentRecorder

#### 初始化

```python
recorder = ExperimentRecorder(
    experiments_file=None,  # 預設: learning/experiments.json
    insights_file=None      # 預設: learning/insights.md
)
```

#### 記錄實驗

```python
exp_id = recorder.log_experiment(
    result: BacktestResult,
    strategy_info: Dict[str, Any],
    config: Dict[str, Any],
    validation_result: Optional[ValidationResult] = None,
    insights: Optional[List[str]] = None,
    parent_experiment: Optional[str] = None
) -> str
```

#### 查詢實驗

```python
experiments = recorder.query_experiments(
    filters: Optional[Dict[str, Any]] = None
) -> List[Experiment]
```

#### 取得最佳

```python
best = recorder.get_best_experiments(
    metric: str = 'sharpe_ratio',
    n: int = 10,
    filters: Optional[Dict[str, Any]] = None
) -> List[Experiment]
```

#### 追蹤演進

```python
evolution = recorder.get_strategy_evolution(
    strategy_name: str
) -> List[Dict[str, Any]]
```

## 參考文件

- `.claude/skills/學習系統/SKILL.md` - 完整學習系統規範
- `.claude/skills/學習系統/references/memory-integration.md` - Memory MCP 整合

---

## Memory MCP 整合

### 核心功能

#### 1. 存儲類型

**策略洞察 (Strategy Insight)**
- 最佳參數組合
- 策略適用條件
- 績效表現（Sharpe, Return, WFA Grade）

**市場洞察 (Market Insight)**
- 標的特性
- 市場狀態影響
- 資金費率模式

**失敗教訓 (Trading Lesson)**
- 過擬合案例
- 失敗原因分析
- 避免方法

#### 2. 標籤系統

```python
from src.learning import MemoryTags

# 資產類型
MemoryTags.ASSET_BTC         # "btc"
MemoryTags.ASSET_ETH         # "eth"
MemoryTags.ASSET_CRYPTO      # "crypto"

# 策略類型
MemoryTags.STRATEGY_TREND           # "trend"
MemoryTags.STRATEGY_MOMENTUM        # "momentum"
MemoryTags.STRATEGY_MEAN_REVERSION  # "mean-reversion"

# 時間框架
MemoryTags.TIMEFRAME_1H      # "1h"
MemoryTags.TIMEFRAME_4H      # "4h"
MemoryTags.TIMEFRAME_1D      # "1d"

# 狀態
MemoryTags.STATUS_VALIDATED  # "validated"
MemoryTags.STATUS_FAILED     # "failed"
```

### 使用範例

#### 基本使用

```python
from src.learning import (
    MemoryIntegration,
    StrategyInsight,
    create_memory_integration
)

# 建立 Memory 整合實例
memory = create_memory_integration()

# 建立策略洞察
insight = StrategyInsight(
    strategy_name="MA Cross",
    symbol="BTCUSDT",
    timeframe="4h",
    best_params={"fast": 10, "slow": 30, "atr_stop": 2.0},
    sharpe_ratio=1.85,
    total_return=0.456,
    max_drawdown=-0.12,
    win_rate=0.58,
    wfa_efficiency=0.68,
    wfa_grade="A",
    market_conditions="趨勢明確市場"
)

# 格式化為 Memory MCP 存儲格式
content, metadata = memory.format_strategy_insight(insight)

# 印出範例供 Claude 使用
memory.print_storage_example(content, metadata)
```

#### AI Loop 整合

```python
from src.learning import (
    MemoryIntegration,
    retrieve_best_params_guide
)

memory = MemoryIntegration()

# 優化前：查詢歷史最佳參數
retrieve_best_params_guide(
    memory,
    strategy_type="ma-cross",
    symbol="BTCUSDT",
    timeframe="4h"
)
# 這會印出建議的 Memory MCP 檢索方式
```

### Memory MCP 呼叫範例

#### 存儲

```python
# Claude 會執行以下 MCP 呼叫
mcp__memory-service__store_memory(
    content='''MA Cross 策略最佳實踐 (BTCUSDT 4h):
- 參數: fast 10, slow 30, atr_stop 2.0
- Sharpe: 1.85, Return: 45.6%
- WFA Efficiency: 68%, Grade: A
- 適用: 趨勢明確市場''',
    metadata={
        "tags": "crypto,btc,4h,validated,ma-cross,trend",
        "type": "trading-insight"
    }
)
```

#### 檢索

```python
# 品質加權檢索（推薦）
mcp__memory-service__retrieve_with_quality_boost(
    query="MA Cross BTCUSDT 4h best parameters validated",
    n_results=5,
    quality_weight=0.3
)
```

### Wrapper 設計

此模組是 **wrapper 類別**，不直接呼叫 Memory MCP API，而是：

1. **格式化內容**：將交易洞察格式化為適合存儲的內容
2. **產生元數據**：自動推斷並產生標籤
3. **提供建議**：產生查詢建議供 Claude 參考
4. **印出範例**：顯示 MCP 呼叫範例供 Claude 執行

### 測試

```bash
# 執行測試
pytest tests/test_memory_standalone.py -v
```

詳細文件請參考：`src/learning/memory.py`
