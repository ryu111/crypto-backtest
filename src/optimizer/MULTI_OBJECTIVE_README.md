# 多目標優化器 (Multi-Objective Optimizer)

使用 NSGA-II 演算法進行多目標策略參數優化。

## 目錄

- [功能特色](#功能特色)
- [快速開始](#快速開始)
- [核心概念](#核心概念)
- [使用範例](#使用範例)
- [API 參考](#api-參考)
- [最佳實踐](#最佳實踐)

---

## 功能特色

✅ **NSGA-II 演算法**：業界標準的多目標優化演算法
✅ **Pareto 前緣**：自動找出所有非支配解
✅ **擁擠度距離**：保持解的多樣性
✅ **靈活權重**：根據偏好選擇最佳平衡解
✅ **視覺化**：支援 Pareto 前緣 2D 圖表
✅ **完整類型註解**：型別安全且易於維護

---

## 快速開始

### 安裝依賴

```bash
pip install optuna>=3.0.0
```

### 簡單範例

```python
from src.optimizer.multi_objective import optimize_multi_objective

# 定義參數空間
param_space = {
    'x': {'type': 'float', 'low': 0.0, 'high': 5.0},
    'y': {'type': 'float', 'low': 0.0, 'high': 5.0}
}

# 定義評估函數
def evaluate(params):
    x, y = params['x'], params['y']
    return {
        'f1': x**2,           # 最小化
        'f2': (y - 5)**2      # 最小化
    }

# 執行優化
result = optimize_multi_objective(
    param_space=param_space,
    evaluate_fn=evaluate,
    objectives=[('f1', 'minimize'), ('f2', 'minimize')],
    n_trials=100,
    seed=42
)

# 取得最佳解
best = result.get_best_solution()
print(f"最佳解: {best.params}")
```

---

## 核心概念

### 1. Pareto 前緣 (Pareto Front)

在多目標優化中，通常不存在單一「最佳解」，而是存在一組**非支配解** (non-dominated solutions)，稱為 Pareto 前緣。

**非支配性**：如果解 A 在所有目標上都不比解 B 差，且至少在一個目標上更好，則 A 支配 B。

#### 範例

假設有兩個目標（都要最小化）：

| 解 | Sharpe Ratio | Max Drawdown | 支配關係 |
|----|--------------|--------------|----------|
| A | 1.5 | 0.20 | - |
| B | 1.8 | 0.15 | B 支配 A |
| C | 1.6 | 0.18 | C 支配 A |
| D | 2.0 | 0.25 | 非支配（與 B、C 互不支配） |

Pareto 前緣 = {B, C, D}

### 2. 擁擠度距離 (Crowding Distance)

衡量解在目標空間中的「稀疏程度」，用於保持解的多樣性。

- **高擁擠度距離**：解周圍較少其他解（更稀疏、更有價值）
- **低擁擠度距離**：解周圍密集（可能冗余）
- **無限距離**：邊界解（保證保留）

### 3. 最佳解選擇

從 Pareto 前緣選擇一個「最佳」解，需要考慮用戶偏好（權重）。

```python
# 均等權重（平衡）
best_balanced = result.get_best_solution()

# 偏重 Sharpe Ratio
best_sharpe = result.get_best_solution(
    weights={'sharpe_ratio': 0.6, 'max_drawdown': 0.4}
)

# 偏重風控
best_risk = result.get_best_solution(
    weights={'sharpe_ratio': 0.3, 'max_drawdown': 0.7}
)
```

---

## 使用範例

### 範例 1: 交易策略優化

```python
from src.optimizer.multi_objective import MultiObjectiveOptimizer

# 參數空間
param_space = {
    'fast_period': {'type': 'int', 'low': 5, 'high': 20},
    'slow_period': {'type': 'int', 'low': 20, 'high': 50},
    'stop_loss': {'type': 'float', 'low': 0.01, 'high': 0.05, 'step': 0.005}
}

# 評估函數（整合回測引擎）
def evaluate(params):
    # 執行回測
    result = backtest_engine.run(strategy, params, data)

    return {
        'sharpe_ratio': result.sharpe_ratio,
        'max_drawdown': result.max_drawdown,
        'win_rate': result.win_rate
    }

# 建立優化器
optimizer = MultiObjectiveOptimizer(
    objectives=[
        ('sharpe_ratio', 'maximize'),
        ('max_drawdown', 'minimize'),
        ('win_rate', 'maximize')
    ],
    n_trials=200,
    seed=42
)

# 執行優化
result = optimizer.optimize(
    param_space=param_space,
    evaluate_fn=evaluate
)

# 顯示結果
print(result.summary())

# 轉為 DataFrame
df = result.to_dataframe()
df.to_csv('pareto_front.csv')

# 繪製 2D 圖表
result.plot_pareto_front_2d('sharpe_ratio', 'max_drawdown', 'pareto.html')
```

### 範例 2: 多種偏好選擇

```python
# 取得不同偏好的解
solutions = {
    '激進型': result.get_best_solution(
        weights={'sharpe_ratio': 0.7, 'max_drawdown': 0.3}
    ),
    '穩健型': result.get_best_solution(
        weights={'sharpe_ratio': 0.3, 'max_drawdown': 0.7}
    ),
    '平衡型': result.get_best_solution()  # 均等權重
}

# 比較
for name, sol in solutions.items():
    print(f"{name}:")
    print(f"  參數: {sol.params}")
    print(f"  Sharpe: {sol.get_objective_value('sharpe_ratio'):.4f}")
    print(f"  Max DD: {sol.get_objective_value('max_drawdown'):.4f}")
```

### 範例 3: 整合現有優化流程

```python
from src.optimizer.bayesian import BayesianOptimizer
from src.optimizer.multi_objective import MultiObjectiveOptimizer

# 第一階段：單目標快速篩選
bayesian = BayesianOptimizer(engine, n_trials=50)
bayesian_result = bayesian.optimize(strategy, data, metric='sharpe_ratio')

# 第二階段：多目標精細優化（縮小參數範圍）
best_params = bayesian_result.best_params
param_space_refined = {
    'fast_period': {
        'type': 'int',
        'low': max(5, best_params['fast_period'] - 3),
        'high': min(20, best_params['fast_period'] + 3)
    },
    # ... 其他參數類似縮小
}

multi_obj = MultiObjectiveOptimizer(
    objectives=[
        ('sharpe_ratio', 'maximize'),
        ('max_drawdown', 'minimize'),
        ('sortino_ratio', 'maximize')
    ],
    n_trials=100
)

final_result = multi_obj.optimize(param_space_refined, evaluate_fn)
```

---

## API 參考

### MultiObjectiveOptimizer

```python
MultiObjectiveOptimizer(
    objectives: List[Tuple[str, Literal['maximize', 'minimize']]],
    n_trials: int = 100,
    seed: Optional[int] = None,
    verbose: bool = True,
    population_size: Optional[int] = None,
    mutation_prob: Optional[float] = None,
    crossover_prob: Optional[float] = None
)
```

**參數**：

- `objectives`: 目標列表，格式 `[('name', 'direction'), ...]`
- `n_trials`: 優化試驗次數（建議 100-500）
- `seed`: 隨機種子（可重現性）
- `verbose`: 是否顯示進度
- `population_size`: NSGA-II 族群大小（預設自動計算）
- `mutation_prob`: 突變機率（預設 None）
- `crossover_prob`: 交叉機率（預設 0.9）

**方法**：

```python
optimize(
    param_space: Dict[str, Dict],
    evaluate_fn: Callable[[Dict], Dict[str, float]],
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    show_progress_bar: bool = True
) -> MultiObjectiveResult
```

### MultiObjectiveResult

```python
class MultiObjectiveResult:
    pareto_front: List[ParetoSolution]  # Pareto 前緣
    all_solutions: List[ParetoSolution]  # 所有解
    n_trials: int                        # 試驗次數
    study: optuna.Study                  # Optuna Study 物件
    optimization_time: float             # 優化時間（秒）
    n_completed_trials: int              # 完成的試驗數
    n_failed_trials: int                 # 失敗的試驗數
```

**方法**：

```python
get_best_solution(weights: Optional[Dict[str, float]] = None) -> ParetoSolution
summary() -> str
to_dataframe() -> pd.DataFrame
plot_pareto_front_2d(obj_x: str, obj_y: str, save_path: Optional[str] = None)
```

### ParetoSolution

```python
class ParetoSolution:
    params: Dict[str, Any]               # 參數
    objectives: List[ObjectiveResult]    # 目標值
    rank: int                            # Pareto rank
    crowding_distance: float             # 擁擠度距離
    trial_number: int                    # 試驗編號
```

**方法**：

```python
get_objective_value(name: str) -> Optional[float]
to_dict() -> Dict[str, Any]
```

---

## 最佳實踐

### 1. 選擇合適的目標數量

- **2-3 個目標**：理想，易於可視化和理解
- **4-5 個目標**：可行，但 Pareto 前緣會變大
- **> 5 個目標**：不建議，解的選擇變得困難

### 2. 設定合理的試驗次數

| 參數數量 | 目標數量 | 建議試驗數 |
|----------|----------|------------|
| 2-3 | 2 | 100-200 |
| 4-5 | 2-3 | 200-300 |
| 6+ | 2-3 | 300-500 |

### 3. 目標的方向性

確保目標方向正確：

```python
objectives=[
    ('sharpe_ratio', 'maximize'),      # ✅ 正確
    ('max_drawdown', 'minimize'),      # ✅ 正確（回撤越小越好）
    ('total_return', 'maximize'),      # ✅ 正確
    ('volatility', 'minimize'),        # ✅ 正確
]

# 常見錯誤
objectives=[
    ('sharpe_ratio', 'minimize'),      # ❌ 錯誤！應該最大化
    ('max_drawdown', 'maximize'),      # ❌ 錯誤！應該最小化
]
```

### 4. 處理目標衝突

某些目標可能高度相關，導致冗餘：

```python
# 高度相關（冗餘）
objectives=[
    ('sharpe_ratio', 'maximize'),
    ('sortino_ratio', 'maximize'),     # 與 Sharpe 高度相關
    ('calmar_ratio', 'maximize'),      # 與 Sharpe 高度相關
]

# 改進：選擇互補目標
objectives=[
    ('sharpe_ratio', 'maximize'),      # 收益風險比
    ('max_drawdown', 'minimize'),      # 極端風險
    ('win_rate', 'maximize'),          # 策略穩定性
]
```

### 5. 使用儲存後端

長時間優化建議使用持久化儲存：

```python
result = optimizer.optimize(
    param_space=param_space,
    evaluate_fn=evaluate_fn,
    study_name='my_strategy_optimization',
    storage='sqlite:///optuna_studies.db'  # 儲存到 SQLite
)

# 之後可以載入繼續優化
optimizer2 = MultiObjectiveOptimizer(...)
result2 = optimizer2.optimize(
    ...,
    study_name='my_strategy_optimization',
    storage='sqlite:///optuna_studies.db',  # 載入已有的 study
    n_trials=100  # 繼續優化 100 次
)
```

### 6. 評估函數錯誤處理

```python
def evaluate(params):
    try:
        result = backtest_engine.run(strategy, params, data)
        return {
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown
        }
    except Exception as e:
        # 回傳極差值，讓 Optuna 剪枝
        return {
            'sharpe_ratio': float('-inf'),  # 最小化時用 inf
            'max_drawdown': float('inf')     # 最大化時用 -inf
        }
```

---

## 常見問題

### Q: Pareto 前緣為什麼這麼多解？

A: 這是正常的。多目標優化的目的就是找出所有非支配解，讓用戶根據偏好選擇。使用 `get_best_solution(weights=...)` 可以根據權重選出單一解。

### Q: 如何選擇合適的權重？

A: 根據實際需求：
- 激進型投資者：更高 Sharpe 權重
- 穩健型投資者：更高 Max Drawdown 權重
- 建議嘗試多組權重，比較結果

### Q: 優化時間太長怎麼辦？

A:
1. 減少 `n_trials`
2. 縮小參數空間範圍
3. 使用 `n_jobs > 1` 並行優化（需額外配置）
4. 先用單目標快速篩選，再用多目標精細優化

### Q: 可以超過 3 個目標嗎？

A: 可以，但不建議超過 5 個。目標越多，Pareto 前緣越大，解的選擇越困難。

---

## 完整範例

參考 `examples/multi_objective_example.py` 查看完整使用範例。

```bash
python examples/multi_objective_example.py
```

---

## 參考資料

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [NSGA-II Paper](https://ieeexplore.ieee.org/document/996017)
- [Pareto Efficiency](https://en.wikipedia.org/wiki/Pareto_efficiency)
