# Automation 模組

自動化工具集，包含 AI Loop 執行控制器和遺傳算法特徵工程。

## 模組組成

### 1. Loop Controller（執行控制器）
管理持續的策略優化循環。

### 2. Feature Engineering（特徵工程）
使用遺傳算法自動生成和優化技術指標組合。

---

# 一、自動特徵工程

## 功能特色

### 1. 遺傳算法特徵工程
- **隨機特徵生成**: 組合基礎指標和運算符生成新特徵
- **演化優化**: 透過遺傳算法找到最佳特徵組合
- **適應度評估**: 基於回測績效評估特徵品質

### 2. 特徵選擇
- **重要性評估**: 使用 Random Forest 或互資訊評估特徵重要性
- **相關性過濾**: 移除高度相關的冗餘特徵
- **自動篩選**: 選擇最具預測力的特徵子集

### 3. 自動策略生成
- **規則生成**: 基於特徵自動生成交易規則
- **多種規則類型**: 支援閾值規則、交叉規則等
- **多空對稱**: 自動產生對應的空單策略

## 快速開始

### 基本使用

```python
from src.automation.feature_engineering import (
    create_feature_engineer,
    quick_feature_evolution
)

# 1. 建立特徵工程器
engineer = create_feature_engineer(
    base_indicators=['SMA', 'EMA', 'RSI', 'MACD', 'ATR'],
    population_size=50,
    generations=20
)

# 2. 定義適應度函數（回測績效）
def fitness_function(feature_set, data):
    # 使用特徵回測策略，返回 Sharpe Ratio
    result = run_backtest(feature_set, data)
    return result.sharpe_ratio

# 3. 執行演化
best_features = engineer.evolve(
    data=market_data,
    fitness_function=fitness_function,
    verbose=True
)

# 4. 查看結果
print(f"最佳適應度: {best_features.fitness_score}")
for feature in best_features.features:
    print(f"{feature.name}: {feature.expression}")
```

### 執行 Demo

```bash
python examples/feature_engineering_demo.py
```

### 執行測試

```bash
pytest tests/test_feature_engineering.py -v
```

完整使用說明請參考模組內程式碼文件。

---

# 二、Loop Controller（執行控制器）

## 功能特性

### 核心功能

- **多種執行模式**
  - `CONTINUOUS` - 持續執行直到手動停止
  - `N_ITERATIONS` - 執行指定次數
  - `TIME_BASED` - 執行指定時間
  - `UNTIL_TARGET` - 執行直到達到目標

- **狀態持久化**
  - 每次迭代後自動保存狀態
  - 支援從中斷點恢復
  - JSON 格式存儲（`learning/loop_state.json`）

- **進度報告**
  - 當前迭代 / 總迭代
  - 已完成時間 / 預估剩餘時間
  - 成功率 / 最佳結果
  - 迭代歷史 DataFrame

- **回調機制**
  - `on_iteration_start` - 迭代開始時
  - `on_iteration_end` - 迭代結束時
  - `on_success` - 迭代成功時
  - `on_failure` - 迭代失敗時
  - `on_new_best` - 發現更佳結果時
  - `on_loop_end` - Loop 結束時

- **優雅停止**
  - 支援 SIGINT (Ctrl+C) 和 SIGTERM 信號
  - 自動保存當前狀態
  - 可安全中斷和恢復

## 快速開始

### 基本使用

```python
from src.automation import LoopController, LoopMode, IterationResult, IterationStatus
from datetime import datetime

# 定義迭代回調函數
def run_optimization() -> IterationResult:
    # 執行優化（這裡簡化示範）
    sharpe = 1.85

    return IterationResult(
        iteration=0,  # 會被 controller 覆蓋
        timestamp=datetime.now(),
        status=IterationStatus.SUCCESS,
        sharpe_ratio=sharpe,
        total_return=0.45,
        max_drawdown=-0.12,
        strategy_name="MA Cross",
        best_params={'fast': 10, 'slow': 30},
        experiment_id="exp_001"
    )

# 建立控制器
controller = LoopController(
    iteration_callback=run_optimization,
    auto_save=True
)

# 啟動 Loop（執行 100 次）
controller.start(
    mode=LoopMode.N_ITERATIONS,
    target=100
)
```

### 使用回調函數

```python
def on_new_best(result: IterationResult):
    print(f"🎉 新的最佳 Sharpe: {result.sharpe_ratio:.4f}")
    print(f"   參數: {result.best_params}")

def on_loop_end(state):
    print(f"Loop 結束，總迭代: {state.completed_iterations}")

callbacks = {
    'on_new_best': on_new_best,
    'on_loop_end': on_loop_end
}

controller = LoopController(
    iteration_callback=run_optimization,
    callbacks=callbacks
)
```

### 從中斷處恢復

```python
# 第一次執行
controller.start(
    mode=LoopMode.N_ITERATIONS,
    target=100
)
# 假設在第 50 次迭代時被中斷...

# 恢復執行
controller.start(
    mode=LoopMode.N_ITERATIONS,
    target=100,
    resume=True  # 從上次中斷處繼續
)
```

### 取得進度

```python
# 在迭代過程中或結束後
progress = controller.get_progress()

print(f"完成: {progress['completed_iterations']}")
print(f"成功率: {progress['success_rate']:.1%}")
print(f"最佳 Sharpe: {progress['best_sharpe']:.4f}")
print(f"已執行: {progress['elapsed_time']}")
```

### 迭代歷史分析

```python
# 取得迭代歷史 DataFrame
history_df = controller.get_iteration_history()

print(history_df[['iteration', 'sharpe_ratio', 'total_return']])

# 統計分析
print(f"平均 Sharpe: {history_df['sharpe_ratio'].mean():.4f}")
print(f"最大 Sharpe: {history_df['sharpe_ratio'].max():.4f}")
```

## CLI 腳本使用

### 基本執行

```bash
# 執行 100 次迭代
python scripts/run_loop.py --mode n_iterations --target 100

# 持續執行直到 Sharpe >= 3.0
python scripts/run_loop.py --mode until_target --target 3.0

# 執行 2 小時
python scripts/run_loop.py --mode time_based --time 120

# 持續執行（手動停止）
python scripts/run_loop.py --mode continuous
```

### 進階選項

```bash
# 指定交易對和槓桿
python scripts/run_loop.py \
  --mode n_iterations \
  --target 50 \
  --symbol ETHUSDT \
  --timeframe 4h \
  --leverage 3

# 調整每次優化試驗次數
python scripts/run_loop.py \
  --mode n_iterations \
  --target 100 \
  --trials 100

# 從上次中斷處恢復
python scripts/run_loop.py --resume

# 清除狀態並重新開始
python scripts/run_loop.py --clear
```

## 執行模式詳解

### 1. CONTINUOUS（持續模式）

持續執行直到收到停止信號（Ctrl+C）。

```python
controller.start(mode=LoopMode.CONTINUOUS)
```

適用場景：
- 長期策略挖掘
- 24/7 自動優化
- 雲端部署

### 2. N_ITERATIONS（次數模式）

執行指定次數的迭代。

```python
controller.start(
    mode=LoopMode.N_ITERATIONS,
    target=100  # 執行 100 次
)
```

適用場景：
- 測試和驗證
- 資源限制環境
- 固定批次優化

### 3. TIME_BASED（時間模式）

執行指定時間（分鐘）。

```python
controller.start(
    mode=LoopMode.TIME_BASED,
    time_limit_minutes=120  # 執行 2 小時
)
```

適用場景：
- 時間有限的優化任務
- 每日定時執行
- 資源排程

### 4. UNTIL_TARGET（目標模式）

執行直到達到目標 Sharpe Ratio。

```python
controller.start(
    mode=LoopMode.UNTIL_TARGET,
    target=3.0  # Sharpe >= 3.0
)
```

適用場景：
- 追求特定績效目標
- 自動化策略挖掘
- 品質驅動優化

## 狀態檔案格式

狀態存儲在 `learning/loop_state.json`：

```json
{
  "started_at": "2026-01-11T10:00:00",
  "mode": "n_iterations",
  "target": 100,
  "current_iteration": 45,
  "completed_iterations": 45,
  "successful_iterations": 42,
  "failed_iterations": 3,
  "best_sharpe": 2.15,
  "best_strategy": "MA Cross v2",
  "best_experiment_id": "exp_20260111_100030",
  "best_params": {
    "fast_period": 10,
    "slow_period": 30
  },
  "iteration_history": [
    {
      "iteration": 1,
      "timestamp": "2026-01-11T10:01:23",
      "status": "success",
      "sharpe_ratio": 1.85,
      "total_return": 0.45,
      "max_drawdown": -0.12,
      "strategy_name": "MA Cross",
      "best_params": {...},
      "experiment_id": "exp_001"
    }
  ],
  "is_paused": false,
  "is_stopped": false
}
```

## 與其他模組整合

### 與 Optimizer 整合

```python
from src.backtester.engine import BacktestEngine, BacktestConfig
from src.optimizer.bayesian import BayesianOptimizer
from src.strategies.trend.ma_cross import MovingAverageCross

def optimize_iteration() -> IterationResult:
    # 建立引擎
    config = BacktestConfig(...)
    engine = BacktestEngine(config)

    # 執行優化
    optimizer = BayesianOptimizer(engine, n_trials=50)
    result = optimizer.optimize(
        strategy=MovingAverageCross(),
        data=market_data
    )

    # 返回結果
    return IterationResult(
        iteration=0,
        timestamp=datetime.now(),
        status=IterationStatus.SUCCESS,
        sharpe_ratio=result.best_backtest_result.sharpe_ratio,
        total_return=result.best_backtest_result.total_return,
        max_drawdown=result.best_backtest_result.max_drawdown,
        strategy_name="MA Cross",
        best_params=result.best_params
    )

controller = LoopController(iteration_callback=optimize_iteration)
```

### 與 ExperimentRecorder 整合

```python
from src.learning import ExperimentRecorder

recorder = ExperimentRecorder()

def record_iteration() -> IterationResult:
    # 執行優化
    opt_result = optimizer.optimize(...)

    # 記錄實驗
    exp_id = recorder.log_experiment(
        result=opt_result.best_backtest_result,
        strategy_info={'name': 'MA Cross', 'type': 'trend'},
        config={'symbol': 'BTCUSDT', 'timeframe': '4h'}
    )

    # 返回結果
    return IterationResult(
        ...,
        experiment_id=exp_id
    )
```

## 範例程式

完整範例請參考：

- `examples/loop_example.py` - 基本使用範例
- `scripts/run_loop.py` - CLI 腳本完整實作

## 測試

執行單元測試：

```bash
pytest tests/test_loop.py -v
```

測試涵蓋：
- 各種執行模式
- 回調機制
- 狀態保存/載入
- 進度追蹤
- 迭代歷史

## 最佳實踐

### 1. 設定合理的迭代回調

```python
def robust_iteration() -> IterationResult:
    try:
        # 執行優化
        result = optimizer.optimize(...)

        return IterationResult(
            iteration=0,
            timestamp=datetime.now(),
            status=IterationStatus.SUCCESS,
            sharpe_ratio=result.best_value,
            ...
        )
    except Exception as e:
        # 記錄錯誤並返回失敗結果
        return IterationResult(
            iteration=0,
            timestamp=datetime.now(),
            status=IterationStatus.FAILED,
            sharpe_ratio=float('-inf'),
            total_return=0.0,
            max_drawdown=0.0,
            strategy_name="unknown",
            best_params={},
            error=str(e)
        )
```

### 2. 使用回調監控進度

```python
def on_iteration_end(iteration_num):
    if iteration_num % 10 == 0:
        progress = controller.get_progress()
        print(f"已完成 {progress['completed_iterations']} 次迭代")
        print(f"最佳 Sharpe: {progress['best_sharpe']:.4f}")

callbacks = {'on_iteration_end': on_iteration_end}
```

### 3. 定期保存檢查點

使用 `auto_save=True` 確保每次迭代後保存狀態：

```python
controller = LoopController(
    iteration_callback=run_optimization,
    auto_save=True  # 自動保存
)
```

### 4. 優雅處理中斷

Loop 已內建信號處理，按 Ctrl+C 會優雅停止並保存狀態：

```python
# 執行時按 Ctrl+C
controller.start(mode=LoopMode.CONTINUOUS)
# 自動保存狀態

# 稍後恢復
controller.start(mode=LoopMode.CONTINUOUS, resume=True)
```

## 常見問題

### Q: 如何限制 Loop 的資源使用？

A: 使用 `TIME_BASED` 模式或調整每次迭代的 `n_trials`：

```bash
python scripts/run_loop.py \
  --mode time_based \
  --time 60 \
  --trials 20  # 減少試驗次數
```

### Q: 狀態檔案損壞怎麼辦？

A: 清除狀態並重新開始：

```bash
python scripts/run_loop.py --clear
```

### Q: 如何追蹤歷史最佳策略？

A: 使用 `get_iteration_history()` 取得完整歷史：

```python
history_df = controller.get_iteration_history()
best_iterations = history_df.nlargest(10, 'sharpe_ratio')
print(best_iterations)
```

### Q: 可以同時執行多個 Loop 嗎？

A: 可以，但需指定不同的 `state_file`：

```python
controller1 = LoopController(
    iteration_callback=callback1,
    state_file=Path('learning/loop1_state.json')
)

controller2 = LoopController(
    iteration_callback=callback2,
    state_file=Path('learning/loop2_state.json')
)
```

## 效能考量

- **迭代時間**: 每次迭代時間取決於優化試驗次數（`n_trials`）
- **記憶體使用**: 迭代歷史會累積在記憶體中，長期執行建議定期清理
- **磁碟 I/O**: 每次迭代保存狀態會產生 I/O，可調整 `auto_save=False` 並手動控制

## 授權

MIT License
