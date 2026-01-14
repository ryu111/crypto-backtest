# src/types/ - 統一型別模組

## 目的

集中管理所有專案中的資料型別定義，避免 dict 裸傳遞，確保型別安全和 IDE 支援。

## 規則

參考：`~/.claude/skills/dev/SKILL.md` - 資料契約規範

| 情境 | 要求 |
|------|------|
| 模組間資料傳遞 | **必須**使用 dataclass/TypedDict |
| 函數回傳值（複雜結構） | **必須**定義型別 |
| 臨時內部計算 | 可用 dict，但不傳出模組 |

## 模組結構

```
src/types/
├── __init__.py       # 匯出所有型別
├── results.py        # 回測/驗證結果型別
├── configs.py        # 配置型別
├── strategies.py     # 策略相關型別
└── README.md         # 本檔案
```

## 使用範例

### 1. 基本匯入

```python
from src.types import (
    BacktestResult,
    ValidationResult,
    ExperimentRecord,
    BacktestConfig,
    StrategyInfo,
    StrategyStats,
)
```

### 2. 建立配置

```python
# 回測配置
config = BacktestConfig(
    symbol="BTCUSDT",
    timeframe="4h",
    start_date="2020-01-01",
    end_date="2024-01-01",
    initial_capital=10000,
    leverage=1,
)

# 自動化循環配置
loop_config = LoopConfig(
    max_iterations=100,
    target_sharpe=2.0,
    symbols=["BTCUSDT", "ETHUSDT"],
    exploit_ratio=0.8,
)
```

### 3. 建立策略資訊

```python
# 策略資訊
strategy = StrategyInfo(
    name="trend_ma_cross",
    type="trend",
    version="1.0",
    params={'fast_period': 10, 'slow_period': 30}
)

# 參數空間
param_space = ParamSpace(
    params={
        'fast_period': (5, 50, 'int'),
        'slow_period': (20, 200, 'int'),
    },
    constraints=[
        lambda p: p['fast_period'] < p['slow_period']
    ]
)
```

### 4. 記錄回測結果

```python
from src.types import PerformanceMetrics, BacktestResult

# 績效指標
metrics = PerformanceMetrics(
    sharpe_ratio=1.5,
    total_return=0.85,
    max_drawdown=0.15,
    win_rate=0.55,
    profit_factor=1.8,
    total_trades=150,
)

# 回測結果
result = BacktestResult(
    metrics=metrics,
    daily_returns=daily_returns_series,  # pd.Series
    equity_curve=equity_curve_series,    # pd.Series
    execution_time=12.5,
)

# 序列化（不含 pd.Series）
result_dict = result.to_dict()
```

### 5. 記錄驗證結果

```python
validation = ValidationResult(
    grade="B",
    stages_passed=[1, 2, 3],
    efficiency=0.85,
    overfit_probability=0.15,
)

print(validation.is_passing)  # True (A/B 為通過)
```

### 6. 建立完整實驗記錄

```python
from datetime import datetime

experiment = ExperimentRecord(
    id=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.symbol}_{strategy.name}",
    timestamp=datetime.now(),
    strategy=strategy.to_dict(),
    config=config.to_dict(),
    results=result.to_dict(),
    validation=validation.to_dict(),
    status="completed",
    insights=["ATR 2x 止損表現更好"],
    tags=["optimized", "trend"],
)

# 序列化到 JSON
import json
with open("experiments.json", "a") as f:
    json.dump(experiment.to_dict(), f)
    f.write("\n")

# 反序列化
with open("experiments.json", "r") as f:
    data = json.load(f)
    experiment = ExperimentRecord.from_dict(data)
```

### 7. 策略統計追蹤

```python
stats = StrategyStats(
    name="trend_ma_cross",
    attempts=10,
    successes=3,
    avg_sharpe=1.2,
    best_sharpe=2.1,
    best_params={'fast_period': 12, 'slow_period': 35}
)

# 更新統計
stats.update_from_experiment(
    sharpe=1.8,
    passed=True,
    params={'fast_period': 15, 'slow_period': 40}
)

# 計算 UCB 評分（用於 Exploit/Explore）
ucb_score = stats.calculate_ucb(total_attempts=100, exploration_weight=2.0)
print(f"UCB Score: {ucb_score:.2f}")
print(f"Success Rate: {stats.success_rate:.2%}")
```

## 序列化支援

所有 dataclass 都支援：

```python
# 轉為 dict（JSON 序列化）
data = obj.to_dict()

# 從 dict 建立
obj = ClassName.from_dict(data)
```

**注意**：
- `pd.Series` 和 `pd.DataFrame` **不會**被序列化
- `datetime` 會自動轉換為 ISO 格式字串
- Lambda 函數（如 constraints）無法序列化

## 與 experiments.json 的對應

| JSON 欄位 | 型別 |
|----------|------|
| `id` | ExperimentRecord.id |
| `timestamp` | ExperimentRecord.timestamp |
| `strategy` | StrategyInfo.to_dict() |
| `config` | BacktestConfig.to_dict() |
| `results` | BacktestResult.to_dict() |
| `validation` | ValidationResult.to_dict() |
| `status` | ExperimentRecord.status |

## 整合現有程式碼

### 逐步遷移策略

```python
# 舊寫法（裸 dict）
def run_backtest(symbol, timeframe, params):
    result = {
        'sharpe': 1.5,
        'return': 0.85,
        'drawdown': 0.15,
        # ... 容易打錯 key
    }
    return result

# 新寫法（型別安全）
from src.types import BacktestConfig, BacktestResult, PerformanceMetrics

def run_backtest(config: BacktestConfig, params: dict) -> BacktestResult:
    metrics = PerformanceMetrics(
        sharpe_ratio=1.5,
        total_return=0.85,
        max_drawdown=0.15,
        # ... IDE 會提醒遺漏的欄位
    )
    return BacktestResult(metrics=metrics)
```

### 適配器模式（漸進式遷移）

```python
# 建立適配器函數
def dict_to_backtest_result(data: dict) -> BacktestResult:
    """將舊的 dict 格式轉換為新型別"""
    metrics = PerformanceMetrics(
        sharpe_ratio=data.get('sharpe_ratio', 0.0),
        total_return=data.get('total_return', 0.0),
        max_drawdown=data.get('max_drawdown', 0.0),
        win_rate=data.get('win_rate', 0.0),
        profit_factor=data.get('profit_factor', 1.0),
        total_trades=data.get('total_trades', 0),
    )
    return BacktestResult(metrics=metrics)

# 使用時
old_result = run_old_backtest(...)  # dict
new_result = dict_to_backtest_result(old_result)  # BacktestResult
```

## 開發指南

### 新增型別時

1. 選擇正確的檔案：
   - 回測/驗證結果 → `results.py`
   - 配置 → `configs.py`
   - 策略相關 → `strategies.py`

2. 必須實作的方法：
   ```python
   def to_dict(self) -> Dict[str, Any]:
       """轉換為字典（JSON 序列化）"""
       pass

   @classmethod
   def from_dict(cls, data: Dict[str, Any]) -> 'ClassName':
       """從字典建立"""
       pass
   ```

3. 更新 `__init__.py` 的 `__all__` 列表

### 命名規範

| 類別 | 規則 | 範例 |
|------|------|------|
| 配置 | XxxConfig | BacktestConfig, LoopConfig |
| 結果 | XxxResult | BacktestResult, ValidationResult |
| 資訊 | XxxInfo | StrategyInfo |
| 統計 | XxxStats | StrategyStats |
| 記錄 | XxxRecord | ExperimentRecord |

## 檢查清單

新增型別時：
- [ ] 加入完整的 type hints
- [ ] 實作 `to_dict()` 和 `from_dict()`
- [ ] 處理 Optional 欄位的預設值
- [ ] 欄位名稱與 JSON 一致
- [ ] 加入 docstring 和使用範例
- [ ] 更新 `__init__.py`
- [ ] 撰寫測試（optional but recommended）
