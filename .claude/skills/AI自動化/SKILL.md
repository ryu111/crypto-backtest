---
name: ai-automation
description: AI 自動化回測系統。AI Loop 執行流程、策略選擇、參數優化、驗證記錄。當需要執行自動化回測循環、設計 AI 工作流時使用。
---

# AI 自動化回測系統

AI 主導的自動化回測、優化、驗證、學習循環。

## AI Loop 執行流程

```
┌─────────────────────────────────────────────┐
│              AI Automation Loop              │
├─────────────────────────────────────────────┤
│                                              │
│  1. 策略選擇 (StrategySelector)              │
│     ├─ 查詢 Memory：歷史最佳策略              │
│     └─ 80% 利用 / 20% 探索                   │
│              ↓                               │
│  2. 參數生成                                  │
│     ├─ 歷史最佳為中心                         │
│     └─ ±30% 範圍擴展                         │
│              ↓                               │
│  3. 貝葉斯優化（Optuna）                      │
│     └─ 目標：Sharpe Ratio                    │
│              ↓                               │
│  4. 5 階段驗證                                │
│     ├─ 基礎回測                               │
│     ├─ 統計檢驗                               │
│     ├─ 穩健性測試                             │
│     ├─ Walk-Forward                          │
│     └─ Monte Carlo                           │
│              ↓                               │
│  5. 價值判斷                                  │
│     ├─ passed >= 3 階段？                    │
│     ├─ Sharpe > 1.0？                        │
│     └─ 過擬合率 < 50%？                       │
│              ↓                               │
│  6. 記錄（如通過）                            │
│     ├─ experiments.json                      │
│     ├─ insights.md                           │
│     └─ Memory MCP                            │
│              ↓                               │
│  Loop → 返回步驟 1                            │
│                                              │
└─────────────────────────────────────────────┘
```

## 策略選擇

### 選擇方法

| 方法 | 說明 | 使用情境 |
|------|------|----------|
| Epsilon-Greedy | 80% 最佳 / 20% 隨機 | 預設方法 |
| UCB | 平衡期望與不確定性 | 系統性探索 |
| Thompson Sampling | 貝葉斯後驗抽樣 | 自適應探索 |

### 選擇程式碼

```python
from src.automation.selector import StrategySelector

selector = StrategySelector()

# 選擇策略
strategy_name = selector.select(method='epsilon_greedy')

# 更新統計
selector.update_stats(strategy_name, {
    'passed': True,
    'sharpe_ratio': 1.85,
    'params': best_params
})
```

## 主協調器

### Orchestrator 使用

```python
from src.automation.orchestrator import Orchestrator

# 配置
config = {
    'n_trials': 50,           # Optuna 試驗次數
    'min_sharpe': 1.0,        # 最低 Sharpe
    'min_stages': 3,          # 最低通過階段
    'max_overfit': 0.5,       # 最大過擬合率
    'symbols': ['BTCUSDT', 'ETHUSDT'],
    'timeframes': ['4h'],
    'leverage': 5
}

# 建立協調器
orchestrator = Orchestrator(config, seed=42)

# 執行單次迭代
result = orchestrator.run_iteration(data_btc, data_eth)

# 執行多次迭代
summary = orchestrator.run_loop(10, data_btc, data_eth)
```

### 迭代結果

```python
@dataclass
class IterationResult:
    iteration: int
    strategy_name: str
    best_params: Dict[str, Any]
    best_sharpe: float
    validation_grade: str
    passed_stages: int
    recorded: bool
    duration: float
    error: Optional[str]
```

## Loop 控制器

### 執行模式

| 模式 | 說明 | 參數 |
|------|------|------|
| CONTINUOUS | 持續執行 | 無 |
| N_ITERATIONS | 執行 N 次 | target=100 |
| TIME_BASED | 執行指定時間 | time_limit_minutes=120 |
| UNTIL_TARGET | 達到目標 Sharpe | target=3.0 |

### 使用範例

```python
from src.automation.loop import LoopController, LoopMode

controller = LoopController(orchestrator)

# 執行 100 次
controller.start(mode=LoopMode.N_ITERATIONS, target=100)

# 持續執行直到 Sharpe >= 3.0
controller.start(mode=LoopMode.UNTIL_TARGET, target=3.0)

# 暫停/恢復
controller.pause()
controller.resume()

# 查看進度
progress = controller.get_progress()
print(f"完成: {progress['completed']}/{progress['total']}")
```

### 狀態持久化

```python
# 保存狀態
controller.save_state()

# 載入並恢復
controller.load_state()
controller.start(resume=True)
```

## 價值判斷標準

### 記錄門檻

| 指標 | 門檻 | 說明 |
|------|------|------|
| Sharpe Ratio | > 1.0 | 風險調整報酬 |
| 通過階段 | >= 3 | 驗證穩健性 |
| Max Drawdown | < 25% | 風險控制 |
| WFA Efficiency | >= 50% | 過擬合控制 |
| Total Trades | >= 30 | 統計有效 |

### 判斷程式碼

```python
def should_record(validation_result, metrics):
    """判斷是否記錄實驗"""
    return (
        validation_result.passed_stages >= 3 and
        metrics['sharpe_ratio'] > 1.0 and
        metrics['max_drawdown'] < 0.25 and
        validation_result.efficiency >= 0.5
    )
```

## 記錄流程

### 多渠道記錄

```python
# 1. JSON 記錄（精確查詢）
recorder.log_experiment(result, strategy_info, config)

# 2. Markdown 洞察（人類閱讀）
recorder.update_insights(experiment)

# 3. Memory MCP（語義搜尋）
memory.store_strategy_insight(experiment, insight)
```

## CLI 啟動

```bash
# 執行 100 次迭代
python scripts/run_loop.py --mode n_iterations --target 100

# 持續執行直到 Sharpe >= 3.0
python scripts/run_loop.py --mode until_target --target 3.0

# 從中斷處恢復
python scripts/run_loop.py --resume

# 查看狀態
python scripts/run_loop.py --status
```

## 模組結構

```
src/automation/
├── __init__.py
├── orchestrator.py    # 主協調器
├── selector.py        # 策略選擇器
└── loop.py            # Loop 控制器

scripts/
└── run_loop.py        # CLI 啟動腳本
```

## 與其他 Skills 關係

### 本 Skill 調用（下游）

| Skill | 調用場景 |
|-------|----------|
| **策略開發** | 選擇和配置策略類型 |
| **參數優化** | Bayesian 優化參數 |
| **策略驗證** | 5 階段驗證流程 |
| **回測核心** | 執行回測 |
| **學習系統** | 記錄實驗結果 |

### 本 Skill 作為協調層

AI自動化是整個系統的協調層，統籌所有其他 Skills 的執行順序和資料流。

### 典型工作流

```
AI自動化啟動
    ↓
策略開發 → 選擇策略類型
    ↓
參數優化 → 找最佳參數
    ├─→ 回測核心 → 執行回測
    └─→ 策略驗證 → 驗證結果
    ↓
學習系統 → 記錄洞察
    ↓
Loop → 返回開始
```

For 策略選擇詳解 → read `references/strategy-selection.md`
For Loop 控制詳解 → read `references/loop-control.md`
