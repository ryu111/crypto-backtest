# backtest-loop Implementation Tasks

## Progress
- Total: 16 tasks
- Completed: 16
- Status: ✅ COMPLETED

---

## 1. Foundation (sequential)

建立基礎配置和類型定義。

- [x] 1.1 建立 LoopConfig 配置類別 | files: src/automation/loop_config.py
  - BacktestLoopConfig dataclass
  - 策略選擇模式配置
  - 驗證階段配置
  - 效能配置（workers, GPU, timeout）
  - 配置驗證方法

- [x] 1.2 建立 LoopResult 結果類別 | files: src/automation/loop_config.py
  - IterationSummary dataclass
  - LoopResult dataclass
  - 摘要報告生成方法

---

## 2. Core Runner (sequential, depends: 1)

實作核心執行引擎。

- [x] 2.1 建立 LoopRunner 執行引擎 | files: src/automation/loop_runner.py
  - 迭代控制邏輯
  - 策略選擇整合（StrategySelector）
  - 狀態管理（running, paused, stopped）
  - 回調機制（on_iteration_start, on_iteration_end, on_new_best）

- [x] 2.2 實作 exploit/explore 策略選擇 | files: src/automation/loop_runner.py
  - 整合 StrategySelector.select()
  - 支援 epsilon_greedy, ucb, thompson_sampling
  - 策略統計更新

- [x] 2.3 實作參數優化流程 | files: src/automation/loop_runner.py
  - 整合 BayesianOptimizer
  - 整合 GPUBatchOptimizer（大量參數時）
  - 參數空間動態調整

---

## 3. Validation Integration (sequential, depends: 2)

整合驗證系統。

- [x] 3.1 建立 ValidationRunner 驗證執行器 | files: src/automation/validation_runner.py
  - 整合 WalkForwardAnalyzer
  - 整合 MonteCarloValidator（如存在）
  - 5 階段驗證流程

- [x] 3.2 實作驗證閾值判斷 | files: src/automation/validation_runner.py
  - WFA efficiency 閾值（預設 > 0.5）
  - Monte Carlo P5 閾值（預設 > 0.5）
  - 評級邏輯（A/B/C/D/F）

---

## 4. User API (sequential, depends: 3)

建立使用者導向的 API。

- [x] 4.1 建立 BacktestLoop 主類別 | files: src/automation/backtest_loop.py
  - 配置解析
  - 啟動/暫停/恢復/停止 API
  - 進度查詢 API

- [x] 4.2 實作便利函數 | files: src/automation/backtest_loop.py
  - run_backtest_loop()
  - quick_optimize()
  - validate_strategy()

- [x] 4.3 實作 Context Manager | files: src/automation/backtest_loop.py
  - __enter__ / __exit__
  - 自動資源清理
  - 狀態保存

---

## 5. Learning Integration (sequential, depends: 4)

整合學習系統。

- [x] 5.1 實作洞察自動記錄 | files: src/automation/loop_runner.py
  - 判斷是否值得記錄（should_record_insight）
  - 自動更新 insights.md
  - 整合 ExperimentRecorder

- [x] 5.2 實作 Memory MCP 同步 | files: src/automation/loop_runner.py
  - 策略成功/失敗記錄
  - 過擬合警訊記錄
  - 新 session 經驗查詢

---

## 6. Module Integration (sequential, depends: 5)

整合模組匯出。

- [x] 6.1 更新 automation __init__.py | files: src/automation/__init__.py
  - 匯出 BacktestLoop
  - 匯出 BacktestLoopConfig
  - 匯出便利函數

---

## 7. Testing (parallel, depends: 6)

測試所有功能。

- [x] 7.1 單元測試 - 配置系統 | files: tests/test_backtest_loop_config.py
  - 配置驗證測試
  - 預設值測試
  - 序列化測試

- [x] 7.2 單元測試 - 執行引擎 | files: tests/test_loop_runner.py
  - 迭代控制測試
  - 策略選擇測試
  - 狀態管理測試

- [x] 7.3 整合測試 - 完整流程 | files: tests/test_backtest_loop_integration.py
  - 端到端測試
  - 斷點恢復測試
  - 學習系統整合測試

---

## Implementation Notes

### 策略選擇模式

```python
class SelectionMode(Enum):
    EPSILON_GREEDY = "epsilon_greedy"  # 80% exploit, 20% explore
    UCB = "ucb"                        # Upper Confidence Bound
    THOMPSON = "thompson_sampling"      # Bayesian sampling
    ROUND_ROBIN = "round_robin"        # 輪流選擇
    SINGLE = "single"                  # 指定單一策略
```

### 驗證階段

```python
class ValidationStage(Enum):
    BASIC = 1       # 基本績效檢查
    STATISTICAL = 2 # 統計顯著性
    STABILITY = 3   # 參數穩定性
    WALK_FORWARD = 4
    MONTE_CARLO = 5
```

### 使用範例

```python
from src.automation import BacktestLoop, BacktestLoopConfig

config = BacktestLoopConfig(
    strategies=['ma_cross', 'rsi', 'supertrend'],
    symbols=['BTCUSDT', 'ETHUSDT'],
    timeframes=['1h', '4h'],
    n_iterations=100,
    selection_mode='epsilon_greedy',
    validation_stages=[4, 5],  # Walk-Forward + Monte Carlo
    min_sharpe=1.0,
    max_workers=8,
    use_gpu=True
)

with BacktestLoop(config) as loop:
    result = loop.run()
    print(result.summary())
```

### 關鍵整合點

1. **HyperLoopController** - 用於並行回測執行
2. **StrategySelector** - 用於 exploit/explore 選擇
3. **BayesianOptimizer** - 用於參數優化
4. **WalkForwardAnalyzer** - 用於 WFA 驗證
5. **ExperimentRecorder** - 用於結果記錄
6. **LoopController** - 用於狀態管理和斷點恢復
