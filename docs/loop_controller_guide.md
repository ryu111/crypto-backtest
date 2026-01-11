# Loop 控制器完整指南

## 目錄

1. [概述](#概述)
2. [架構設計](#架構設計)
3. [核心組件](#核心組件)
4. [使用指南](#使用指南)
5. [CLI 工具](#cli-工具)
6. [進階應用](#進階應用)
7. [最佳實踐](#最佳實踐)
8. [疑難排解](#疑難排解)

---

## 概述

Loop 控制器是 AI Loop 系統的執行引擎，負責管理持續的策略優化循環。它提供了完整的狀態管理、進度追蹤和錯誤處理機制。

### 主要特性

- ✅ 多種執行模式（持續/次數/時間/目標）
- ✅ 自動狀態持久化與恢復
- ✅ 完整的回調機制
- ✅ 進度追蹤與報告
- ✅ 優雅停止與信號處理
- ✅ 迭代歷史分析
- ✅ CLI 工具支援

---

## 架構設計

### 系統架構

```
┌─────────────────────────────────────────────────────────┐
│                    Loop Controller                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐   │
│  │  Iteration │───▶│   State    │───▶│  Callback  │   │
│  │  Engine    │    │  Manager   │    │  Handler   │   │
│  └────────────┘    └────────────┘    └────────────┘   │
│         │                 │                  │          │
│         ▼                 ▼                  ▼          │
│  ┌─────────────────────────────────────────────────┐   │
│  │            Persistence Layer                    │   │
│  │         (JSON State File)                       │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 數據流

```
User Request
     │
     ▼
┌─────────────┐
│ Controller  │
│   .start()  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Iteration Loop  │◄─────┐
└────────┬────────┘      │
         │                │
         ▼                │
┌─────────────────┐      │
│ Run Iteration   │      │
│  (Callback)     │      │
└────────┬────────┘      │
         │                │
         ▼                │
┌─────────────────┐      │
│ Update State    │      │
└────────┬────────┘      │
         │                │
         ▼                │
┌─────────────────┐      │
│  Save State     │      │
└────────┬────────┘      │
         │                │
         ▼                │
   Should Stop? ──No─────┘
         │
        Yes
         ▼
    Loop End
```

---

## 核心組件

### 1. LoopController

主控制器類別，管理整個 Loop 生命週期。

**核心方法:**

```python
class LoopController:
    def start(mode, target, resume)         # 啟動 Loop
    def stop()                               # 停止 Loop
    def pause()                              # 暫停 Loop
    def resume()                             # 恢復 Loop
    def save_state()                         # 保存狀態
    def load_state()                         # 載入狀態
    def get_progress()                       # 取得進度
    def get_summary()                        # 取得摘要
    def get_iteration_history()              # 取得歷史
```

### 2. LoopState

狀態管理物件，追蹤 Loop 執行狀態。

```python
@dataclass
class LoopState:
    started_at: datetime
    mode: str
    target: Optional[int]
    current_iteration: int
    completed_iterations: int
    successful_iterations: int
    failed_iterations: int
    best_sharpe: float
    best_strategy: str
    best_experiment_id: str
    best_params: Dict[str, Any]
    iteration_history: List[Dict]
    is_paused: bool
    is_stopped: bool
```

### 3. IterationResult

單次迭代結果物件。

```python
@dataclass
class IterationResult:
    iteration: int
    timestamp: datetime
    status: IterationStatus         # SUCCESS | FAILED
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    strategy_name: str
    best_params: Dict[str, Any]
    experiment_id: Optional[str]
    error: Optional[str]            # 失敗時的錯誤訊息
```

### 4. LoopMode

執行模式枚舉。

```python
class LoopMode(Enum):
    CONTINUOUS = "continuous"       # 持續執行
    N_ITERATIONS = "n_iterations"   # 執行 N 次
    TIME_BASED = "time_based"       # 執行 T 時間
    UNTIL_TARGET = "until_target"   # 執行直到達標
```

---

## 使用指南

### 快速開始

```python
from src.automation import LoopController, LoopMode, IterationResult
from datetime import datetime

# 1. 定義迭代回調
def run_iteration() -> IterationResult:
    # 執行優化...
    return IterationResult(
        iteration=0,
        timestamp=datetime.now(),
        status=IterationStatus.SUCCESS,
        sharpe_ratio=1.85,
        total_return=0.45,
        max_drawdown=-0.12,
        strategy_name="MA Cross",
        best_params={'fast': 10, 'slow': 30}
    )

# 2. 建立控制器
controller = LoopController(
    iteration_callback=run_iteration,
    auto_save=True
)

# 3. 啟動 Loop
controller.start(
    mode=LoopMode.N_ITERATIONS,
    target=100
)
```

### 與優化器整合

```python
from src.backtester.engine import BacktestEngine, BacktestConfig
from src.optimizer.bayesian import BayesianOptimizer
from src.strategies.trend.ma_cross import MovingAverageCross

# 建立配置
config = BacktestConfig(
    symbol='BTCUSDT',
    timeframe='1h',
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2025, 1, 1),
    initial_capital=10000,
    leverage=5
)

# 建立引擎
engine = BacktestEngine(config)

# 載入資料
data = load_market_data()

# 定義迭代回調
def optimize_iteration() -> IterationResult:
    # 建立優化器
    optimizer = BayesianOptimizer(engine, n_trials=50)

    # 執行優化
    result = optimizer.optimize(
        strategy=MovingAverageCross(),
        data=data,
        metric='sharpe_ratio'
    )

    # 返回結果
    backtest = result.best_backtest_result
    return IterationResult(
        iteration=0,
        timestamp=datetime.now(),
        status=IterationStatus.SUCCESS,
        sharpe_ratio=backtest.sharpe_ratio,
        total_return=backtest.total_return,
        max_drawdown=backtest.max_drawdown,
        strategy_name="MA Cross",
        best_params=result.best_params
    )

# 建立控制器並啟動
controller = LoopController(iteration_callback=optimize_iteration)
controller.start(mode=LoopMode.UNTIL_TARGET, target=3.0)
```

### 使用回調函數

```python
# 定義回調
def on_new_best(result: IterationResult):
    print(f"🎉 新最佳 Sharpe: {result.sharpe_ratio:.4f}")
    # 可在此發送通知、記錄日誌等

def on_failure(error: Exception):
    print(f"❌ 迭代失敗: {error}")
    # 可在此記錄錯誤、發送警報等

def on_loop_end(state: LoopState):
    print(f"🏁 Loop 結束，總迭代: {state.completed_iterations}")
    # 可在此產生報告、備份結果等

# 建立控制器
controller = LoopController(
    iteration_callback=run_iteration,
    callbacks={
        'on_new_best': on_new_best,
        'on_failure': on_failure,
        'on_loop_end': on_loop_end
    }
)
```

---

## CLI 工具

### 基本使用

```bash
# 執行 100 次迭代
python scripts/run_loop.py --mode n_iterations --target 100

# 持續執行直到 Sharpe >= 3.0
python scripts/run_loop.py --mode until_target --target 3.0

# 執行 2 小時
python scripts/run_loop.py --mode time_based --time 120
```

### 進階選項

```bash
# 完整配置
python scripts/run_loop.py \
  --mode n_iterations \
  --target 50 \
  --symbol BTCUSDT \
  --timeframe 4h \
  --leverage 5 \
  --trials 100

# 從中斷處恢復
python scripts/run_loop.py --resume

# 清除狀態
python scripts/run_loop.py --clear
```

### 參數說明

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--mode` | 執行模式 | `continuous` |
| `--target` | 目標值（次數或 Sharpe） | - |
| `--time` | 時間限制（分鐘） | - |
| `--symbol` | 交易標的 | `BTCUSDT` |
| `--timeframe` | 時間框架 | `1h` |
| `--leverage` | 槓桿倍數 | `5` |
| `--trials` | 每次優化試驗次數 | `50` |
| `--resume` | 從中斷處恢復 | `False` |
| `--clear` | 清除狀態 | `False` |

---

## 進階應用

### 1. 分散式執行

```python
# 機器 A
controller_a = LoopController(
    iteration_callback=optimize_btc,
    state_file=Path('loop_btc_state.json')
)
controller_a.start(mode=LoopMode.CONTINUOUS)

# 機器 B
controller_b = LoopController(
    iteration_callback=optimize_eth,
    state_file=Path('loop_eth_state.json')
)
controller_b.start(mode=LoopMode.CONTINUOUS)
```

### 2. 自適應優化

```python
def adaptive_iteration() -> IterationResult:
    # 根據歷史表現調整優化策略
    history = controller.get_iteration_history()

    if len(history) > 10:
        recent_sharpe = history['sharpe_ratio'].tail(10).mean()

        if recent_sharpe < 1.0:
            # 表現不佳，增加試驗次數
            n_trials = 100
        else:
            # 表現良好，維持現有設定
            n_trials = 50
    else:
        n_trials = 50

    optimizer = BayesianOptimizer(engine, n_trials=n_trials)
    result = optimizer.optimize(strategy, data)

    # 返回結果...
```

### 3. 多策略輪詢

```python
strategies = [
    MovingAverageCross(),
    SuperTrend(),
    RSIStrategy()
]

strategy_index = 0

def multi_strategy_iteration() -> IterationResult:
    global strategy_index

    # 輪流測試不同策略
    strategy = strategies[strategy_index]
    strategy_index = (strategy_index + 1) % len(strategies)

    # 優化當前策略
    optimizer = BayesianOptimizer(engine, n_trials=50)
    result = optimizer.optimize(strategy, data)

    # 返回結果...
```

### 4. 條件式早停

```python
def early_stopping_iteration() -> IterationResult:
    # 執行優化
    result = optimizer.optimize(strategy, data)

    # 檢查是否應該早停
    if result.best_value < 0.5:
        # Sharpe 過低，跳過此策略
        print("策略表現不佳，跳過")
        raise Exception("Strategy performance too low")

    # 返回結果...

# 使用失敗回調處理早停
def on_failure(error):
    if "too low" in str(error):
        print("自動跳過表現不佳的策略")

controller = LoopController(
    iteration_callback=early_stopping_iteration,
    callbacks={'on_failure': on_failure}
)
```

---

## 最佳實踐

### 1. 設定合理的迭代時間

```python
# 每次迭代不應過長（建議 < 5 分鐘）
optimizer = BayesianOptimizer(
    engine,
    n_trials=50,  # 不要設定過大
    timeout=300   # 5 分鐘超時
)
```

### 2. 定期清理歷史

```python
def cleanup_iteration() -> IterationResult:
    # 執行優化
    result = optimizer.optimize(...)

    # 定期清理（保留最近 1000 筆）
    if len(controller.state.iteration_history) > 1000:
        controller.state.iteration_history = \
            controller.state.iteration_history[-1000:]

    return result
```

### 3. 錯誤處理與重試

```python
def robust_iteration() -> IterationResult:
    max_retries = 3

    for attempt in range(max_retries):
        try:
            result = optimizer.optimize(...)
            return IterationResult(...)

        except Exception as e:
            if attempt == max_retries - 1:
                # 最後一次嘗試，返回失敗結果
                return IterationResult(
                    status=IterationStatus.FAILED,
                    error=str(e),
                    ...
                )
            else:
                # 重試
                time.sleep(5)
                continue
```

### 4. 監控與告警

```python
def monitored_iteration() -> IterationResult:
    # 執行優化
    result = optimizer.optimize(...)

    # 監控指標
    if result.best_value < 0.5:
        send_alert("警告：Sharpe 過低")

    if result.n_failed_trials > result.n_trials * 0.5:
        send_alert("警告：失敗率過高")

    return IterationResult(...)
```

---

## 疑難排解

### Q1: Loop 執行太慢

**原因:**
- 每次迭代的優化試驗次數過多
- 回測資料量過大

**解決方案:**
```python
# 減少試驗次數
optimizer = BayesianOptimizer(engine, n_trials=30)

# 縮短回測期間
config = BacktestConfig(
    start_date=datetime.now() - timedelta(days=180),  # 只用 6 個月
    end_date=datetime.now()
)

# 使用較大時間框架
config.timeframe = '4h'  # 而非 '1h'
```

### Q2: 記憶體使用過高

**原因:**
- 迭代歷史累積過多

**解決方案:**
```python
# 定期清理歷史
if len(controller.state.iteration_history) > 500:
    controller.state.iteration_history = \
        controller.state.iteration_history[-500:]

# 或關閉自動保存
controller = LoopController(
    iteration_callback=run_iteration,
    auto_save=False  # 手動控制保存
)
```

### Q3: 狀態檔案損壞

**原因:**
- 執行過程中強制終止導致寫入不完整

**解決方案:**
```bash
# 清除狀態並重新開始
python scripts/run_loop.py --clear
```

或使用備份機制：
```python
# 定期備份狀態
def on_iteration_end(iteration_num):
    if iteration_num % 10 == 0:
        backup_path = controller.state_file.with_suffix('.bak')
        shutil.copy(controller.state_file, backup_path)
```

### Q4: 無法恢復中斷的 Loop

**原因:**
- 狀態檔案路徑不一致
- 狀態檔案被刪除

**解決方案:**
```python
# 確保使用相同的 state_file
controller = LoopController(
    iteration_callback=run_iteration,
    state_file=Path('learning/loop_state.json')  # 明確指定
)

# 檢查檔案是否存在
if controller.state_file.exists():
    controller.start(resume=True)
else:
    print("找不到狀態檔案，將啟動新的 Loop")
    controller.start(resume=False)
```

---

## 附錄

### A. 完整範例程式碼

請參考：
- `examples/loop_example.py` - 基本使用範例
- `examples/simple_loop_test.py` - 快速測試
- `scripts/run_loop.py` - CLI 完整實作

### B. API 參考

完整 API 文件請參考 `src/automation/README.md`

### C. 測試

執行測試：
```bash
pytest tests/test_loop.py -v
```

### D. 效能基準

| 配置 | 每次迭代時間 | 記憶體使用 |
|------|--------------|------------|
| 50 trials, 1 年資料, 1h | ~30 秒 | ~200 MB |
| 100 trials, 1 年資料, 4h | ~45 秒 | ~150 MB |
| 50 trials, 6 個月資料, 1h | ~15 秒 | ~100 MB |

---

**文件版本**: 1.0
**最後更新**: 2026-01-11
**作者**: AI Development Team
