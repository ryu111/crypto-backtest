# fix-hardcoding Proposal

## Summary

消除專案中的硬編碼字串和 magic number，使用 Python Enum 和 Literal 型別取代。

## Problem

目前專案存在大量硬編碼問題：

### 1. 狀態字串
```python
# results.py
status: str = "completed"  # completed / failed / running
grade: str  # A/B/C/D/F
```

### 2. 策略類型字串
```python
# strategies.py
type: str  # trend / mean_reversion / breakout / volatility
```

### 3. 配置字串
```python
# configs.py
method: str = "bayesian"  # bayesian / grid / random
objective: str = "sharpe_ratio"  # sharpe_ratio / sortino_ratio / calmar_ratio
```

### 4. 散落各處的比對
```python
# 各檔案中
if strategy == "ma_cross": ...
if backend == "mlx": ...
if lesson_type == 'exceptional_performance': ...
```

## Impact

| 問題 | 後果 |
|------|------|
| Typo 無法被捕捉 | `"stauts"` vs `"status"` → runtime error |
| 無法重構 | 改名時到處漏改 |
| 無自動完成 | 每次都要查文件 |
| 無型別檢查 | 傳錯值沒有警告 |

## Solution

### 建立集中的 Enum 定義

新增 `src/types/enums.py`：

```python
from enum import Enum

class ExperimentStatus(Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    RUNNING = "running"

class Grade(Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"

class StrategyType(Enum):
    TREND = "trend"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    VOLATILITY = "volatility"

class OptimizationMethod(Enum):
    BAYESIAN = "bayesian"
    GRID = "grid"
    RANDOM = "random"

class ObjectiveMetric(Enum):
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"

class BackendType(Enum):
    MLX = "mlx"
    NUMPY = "numpy"
    METAL = "metal"

class LessonType(Enum):
    EXCEPTIONAL = "exceptional_performance"
    POOR = "unexpected_poor_performance"
    OVERFIT = "overfit_warning"
    RISK = "risk_event"
    SENSITIVITY = "parameter_sensitivity"
```

### 更新使用端

```python
# Before
status: str = "completed"
if grade in ['A', 'B']:

# After
from src.types import ExperimentStatus, Grade
status: ExperimentStatus = ExperimentStatus.COMPLETED
if grade in [Grade.A, Grade.B]:
```

## Files Affected

| 檔案 | 變更 |
|------|------|
| `src/types/enums.py` | 新增 |
| `src/types/__init__.py` | 匯出 Enum |
| `src/types/results.py` | 使用 ExperimentStatus, Grade |
| `src/types/strategies.py` | 使用 StrategyType |
| `src/types/configs.py` | 使用 OptimizationMethod, ObjectiveMetric |
| `src/backtester/*.py` | 更新硬編碼比對 |
| `src/automation/*.py` | 更新硬編碼比對 |
| `src/learning/*.py` | 使用 LessonType |

## Scope

- **In Scope**: types 目錄和直接使用這些型別的檔案
- **Out of Scope**: 測試檔案（後續任務）

## Risks

- 需要確保 JSON 序列化/反序列化相容（Enum 轉 string）
- 需要處理既有資料的向後相容

## Timeline

Phase 1: 建立 enums.py 並更新 types/ 目錄
Phase 2: 更新 backtester/ 和 automation/ 目錄
Phase 3: 更新 learning/ 目錄
