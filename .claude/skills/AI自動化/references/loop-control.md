# Loop 控制

AI 自動化回測循環的控制機制。

## Event-Driven 架構

### 核心概念

```
┌─────────────────────────────────────────────┐
│              Event-Driven Loop              │
├─────────────────────────────────────────────┤
│                                             │
│   Event Queue                               │
│   ┌─────────────────────────────────┐       │
│   │ IterationEvent                  │       │
│   │ StrategySelectedEvent           │       │
│   │ BacktestCompleteEvent           │       │
│   │ ValidationCompleteEvent         │       │
│   │ RecordEvent                     │       │
│   └─────────────────────────────────┘       │
│              ↓                              │
│   Event Handler                             │
│   ┌─────────────────────────────────┐       │
│   │ on_iteration_start()            │       │
│   │ on_strategy_selected()          │       │
│   │ on_backtest_complete()          │       │
│   │ on_validation_complete()        │       │
│   │ on_record()                     │       │
│   └─────────────────────────────────┘       │
│              ↓                              │
│   Next Event...                             │
│                                             │
└─────────────────────────────────────────────┘
```

### 優勢

| 優勢 | 說明 |
|------|------|
| 模組化 | 每個事件處理獨立 |
| 可擴展 | 新增事件類型容易 |
| 可測試 | 可模擬事件進行測試 |
| 靈活 | 相同代碼可用於回測和實盤 |

## 狀態機設計

### 狀態定義

```python
from enum import Enum, auto

class LoopState(Enum):
    """Loop 狀態"""
    IDLE = auto()       # 閒置，未啟動
    RUNNING = auto()    # 執行中
    PAUSED = auto()     # 暫停
    COMPLETED = auto()  # 完成
    FAILED = auto()     # 失敗
    CANCELLED = auto()  # 取消
```

### 狀態轉換圖

```
                ┌─────────┐
                │  IDLE   │
                └────┬────┘
                     │ start()
                     ↓
                ┌─────────┐
        ┌──────→│ RUNNING │←──────┐
        │       └────┬────┘       │
        │            │            │
   resume()     pause()     resume()
        │            │            │
        │            ↓            │
        │       ┌─────────┐       │
        └───────│ PAUSED  │───────┘
                └────┬────┘
                     │
         ┌───────────┼───────────┐
         │           │           │
         ↓           ↓           ↓
    ┌─────────┐ ┌─────────┐ ┌─────────┐
    │COMPLETED│ │ FAILED  │ │CANCELLED│
    └─────────┘ └─────────┘ └─────────┘
```

### 狀態轉換條件

| 從 | 到 | 條件 |
|----|----|------|
| IDLE | RUNNING | 呼叫 `start()` |
| RUNNING | PAUSED | 呼叫 `pause()` |
| RUNNING | COMPLETED | 達到目標條件 |
| RUNNING | FAILED | 發生錯誤 |
| PAUSED | RUNNING | 呼叫 `resume()` |
| PAUSED | CANCELLED | 呼叫 `cancel()` |

## 迭代控制模式

### 模式定義

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class LoopMode(Enum):
    """Loop 執行模式"""
    CONTINUOUS = "continuous"       # 持續執行
    N_ITERATIONS = "n_iterations"   # 執行 N 次
    TIME_BASED = "time_based"       # 執行指定時間
    UNTIL_TARGET = "until_target"   # 達到目標

@dataclass
class LoopConfig:
    """Loop 配置"""
    mode: LoopMode
    max_iterations: Optional[int] = None      # N_ITERATIONS 模式
    time_limit_minutes: Optional[int] = None  # TIME_BASED 模式
    target_sharpe: Optional[float] = None     # UNTIL_TARGET 模式

    # 通用限制
    max_consecutive_failures: int = 10
    checkpoint_interval: int = 10  # 每 N 次迭代保存狀態
```

### 終止條件檢查

```python
from datetime import datetime, timedelta

class LoopController:
    """Loop 控制器"""

    def __init__(self, config: LoopConfig):
        self.config = config
        self.start_time: Optional[datetime] = None
        self.iteration_count = 0
        self.best_sharpe = 0.0
        self.consecutive_failures = 0

    def should_continue(self) -> bool:
        """檢查是否應該繼續"""

        # 檢查連續失敗
        if self.consecutive_failures >= self.config.max_consecutive_failures:
            return False

        mode = self.config.mode

        if mode == LoopMode.CONTINUOUS:
            return True

        elif mode == LoopMode.N_ITERATIONS:
            return self.iteration_count < self.config.max_iterations

        elif mode == LoopMode.TIME_BASED:
            elapsed = datetime.now() - self.start_time
            limit = timedelta(minutes=self.config.time_limit_minutes)
            return elapsed < limit

        elif mode == LoopMode.UNTIL_TARGET:
            return self.best_sharpe < self.config.target_sharpe

        return False

    def on_iteration_complete(self, success: bool, sharpe: float = 0.0):
        """迭代完成回調"""
        self.iteration_count += 1

        if success:
            self.consecutive_failures = 0
            self.best_sharpe = max(self.best_sharpe, sharpe)
        else:
            self.consecutive_failures += 1

        # 檢查是否需要保存 checkpoint
        if self.iteration_count % self.config.checkpoint_interval == 0:
            self.save_checkpoint()
```

## Orchestrator 流程

### 元件架構

```
┌─────────────────────────────────────────────────────────────┐
│                      Orchestrator                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│   │  Selector   │───→│  Optimizer  │───→│  Validator  │    │
│   │ (策略選擇)  │    │ (參數優化)  │    │ (驗證)      │    │
│   └─────────────┘    └─────────────┘    └─────────────┘    │
│          ↑                  ↓                  ↓            │
│          │            ┌─────────────┐    ┌─────────────┐    │
│          │            │  Backtester │    │  Recorder   │    │
│          │            │  (回測)     │    │  (記錄)     │    │
│          │            └─────────────┘    └─────────────┘    │
│          │                                     ↓            │
│          └─────────────────────────────────────┘            │
│                    (更新策略統計)                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 核心實作

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional
import time

@dataclass
class IterationResult:
    """單次迭代結果"""
    iteration: int
    strategy_name: str
    best_params: Dict[str, Any]
    best_sharpe: float
    validation_grade: str
    passed_stages: int
    recorded: bool
    duration: float
    error: Optional[str] = None

class Orchestrator:
    """主協調器"""

    def __init__(self, config: Dict[str, Any], seed: int = 42):
        self.config = config
        self.selector = StrategySelector(epsilon=0.2)
        self.optimizer = ParameterOptimizer(n_trials=config['n_trials'])
        self.validator = StrategyValidator()
        self.recorder = ExperimentRecorder()

        # 設定隨機種子
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)

    def run_iteration(self, data_btc, data_eth) -> IterationResult:
        """執行單次迭代"""
        start_time = time.time()

        try:
            # 1. 選擇策略
            strategy_name = self.selector.select()

            # 2. 參數優化
            best_params, best_sharpe = self.optimizer.optimize(
                strategy_name, data_btc, data_eth
            )

            # 3. 驗證
            validation = self.validator.validate(
                strategy_name, best_params, data_btc, data_eth
            )

            # 4. 判斷是否記錄
            recorded = False
            if self._should_record(validation, best_sharpe):
                self.recorder.record(strategy_name, best_params, validation)
                recorded = True

            # 5. 更新策略統計
            self.selector.update(strategy_name, best_sharpe)

            return IterationResult(
                iteration=self.iteration_count,
                strategy_name=strategy_name,
                best_params=best_params,
                best_sharpe=best_sharpe,
                validation_grade=validation.grade,
                passed_stages=validation.passed_stages,
                recorded=recorded,
                duration=time.time() - start_time
            )

        except Exception as e:
            return IterationResult(
                iteration=self.iteration_count,
                strategy_name=strategy_name,
                best_params={},
                best_sharpe=0.0,
                validation_grade='F',
                passed_stages=0,
                recorded=False,
                duration=time.time() - start_time,
                error=str(e)
            )

    def _should_record(self, validation, sharpe: float) -> bool:
        """判斷是否記錄"""
        return (
            validation.passed_stages >= self.config['min_stages'] and
            sharpe > self.config['min_sharpe'] and
            validation.efficiency >= (1 - self.config['max_overfit'])
        )
```

## 狀態持久化

### 狀態檔案結構

```json
{
  "loop_id": "loop_20260112_143000",
  "state": "PAUSED",
  "config": {
    "mode": "n_iterations",
    "max_iterations": 100
  },
  "progress": {
    "iteration_count": 45,
    "start_time": "2026-01-12T14:30:00Z",
    "last_checkpoint": "2026-01-12T15:45:00Z",
    "best_sharpe": 2.15,
    "best_strategy": "trend_ma_cross_4h_v3"
  },
  "strategy_stats": {
    "trend_ma_cross_4h": {
      "count": 20,
      "avg_sharpe": 1.45,
      "best_sharpe": 2.15
    }
  }
}
```

### 持久化實作

```python
import json
from pathlib import Path
from datetime import datetime

class LoopStateManager:
    """Loop 狀態管理器"""

    def __init__(self, state_dir: str = "loop_states"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)
        self.current_state_file: Optional[Path] = None

    def save_state(self, controller: LoopController, orchestrator: Orchestrator):
        """保存狀態"""
        state = {
            "loop_id": controller.loop_id,
            "state": controller.state.name,
            "config": {
                "mode": controller.config.mode.value,
                "max_iterations": controller.config.max_iterations,
                "target_sharpe": controller.config.target_sharpe
            },
            "progress": {
                "iteration_count": controller.iteration_count,
                "start_time": controller.start_time.isoformat() if controller.start_time else None,
                "last_checkpoint": datetime.now().isoformat(),
                "best_sharpe": controller.best_sharpe,
                "consecutive_failures": controller.consecutive_failures
            },
            "strategy_stats": orchestrator.selector.strategy_stats
        }

        state_file = self.state_dir / f"{controller.loop_id}.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        self.current_state_file = state_file

    def load_state(self, loop_id: str) -> dict:
        """載入狀態"""
        state_file = self.state_dir / f"{loop_id}.json"

        if not state_file.exists():
            raise FileNotFoundError(f"State file not found: {state_file}")

        with open(state_file) as f:
            return json.load(f)

    def resume(self, loop_id: str, controller: LoopController, orchestrator: Orchestrator):
        """從狀態恢復"""
        state = self.load_state(loop_id)

        # 恢復 controller 狀態
        controller.iteration_count = state['progress']['iteration_count']
        controller.best_sharpe = state['progress']['best_sharpe']
        controller.consecutive_failures = state['progress']['consecutive_failures']

        # 恢復 orchestrator 策略統計
        orchestrator.selector.strategy_stats = state['strategy_stats']

        # 設定狀態為 RUNNING
        controller.state = LoopState.RUNNING
```

## 監控與介入

### 進度查詢

```python
def get_progress(controller: LoopController) -> dict:
    """獲取進度資訊"""
    elapsed = datetime.now() - controller.start_time if controller.start_time else timedelta(0)

    progress = {
        "state": controller.state.name,
        "iteration_count": controller.iteration_count,
        "elapsed_time": str(elapsed),
        "best_sharpe": controller.best_sharpe,
        "consecutive_failures": controller.consecutive_failures
    }

    # 根據模式計算進度百分比
    if controller.config.mode == LoopMode.N_ITERATIONS:
        progress["completion_percentage"] = (
            controller.iteration_count / controller.config.max_iterations * 100
        )
    elif controller.config.mode == LoopMode.UNTIL_TARGET:
        progress["target_progress"] = (
            controller.best_sharpe / controller.config.target_sharpe * 100
        )

    return progress
```

### 介入時機

| 指標 | 警告閾值 | 建議動作 |
|------|----------|----------|
| 連續失敗 | > 5 次 | 檢查資料/策略 |
| 迭代時間 | > 10 分鐘/次 | 檢查效能 |
| Sharpe 停滯 | > 50 次無進展 | 調高探索率 |
| 記憶體使用 | > 80% | 暫停清理 |

## CLI 介面

```bash
# 啟動 Loop
python scripts/run_loop.py --mode n_iterations --target 100

# 查看狀態
python scripts/run_loop.py --status

# 暫停
python scripts/run_loop.py --pause

# 恢復
python scripts/run_loop.py --resume --loop-id loop_20260112_143000

# 取消
python scripts/run_loop.py --cancel
```

## 參考資料

- [QuantStart: Event-Driven Backtesting](https://www.quantstart.com/articles/Choosing-a-Platform-for-Backtesting-and-Automated-Execution/)
- [Freqtrade Bot Basics](https://www.freqtrade.io/en/stable/bot-basics/)
- [arXiv: Orchestration Framework for Financial Agents](https://arxiv.org/html/2512.02227v1)
