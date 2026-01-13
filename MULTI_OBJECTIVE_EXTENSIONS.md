# 多目標優化器擴展功能

## 概述

本次擴展為 `src/optimizer/multi_objective.py` 添加了以下新功能：

1. **約束條件支援（Constraints）**
2. **市場狀態感知（Regime Integration）**
3. **暖啟動支援（Warm Start）**
4. **膝點檢測（Knee Point Detection）**
5. **Pareto 前沿篩選（Pareto Front Filtering）**
6. **額外視覺化方法（3D & Parallel Coordinates）**

所有新功能都是**可選的**，保持**向後相容**。

---

## 1. 約束條件支援

### 功能描述

允許定義參數約束條件，違反約束的參數組合會被自動剔除。

### 使用方法

```python
def leverage_constraint(params: Dict) -> float:
    """槓桿不得超過 10x"""
    # 回傳正值表示違反約束（值越大違反越嚴重）
    # 回傳 0 或負值表示滿足約束
    return max(0, params.get('leverage', 1) - 10)

def period_constraint(params: Dict) -> float:
    """快週期必須小於慢週期至少 5"""
    fast = params.get('fast_period', 10)
    slow = params.get('slow_period', 30)
    return max(0, fast - slow + 5)

optimizer = MultiObjectiveOptimizer(
    objectives=[('sharpe', 'maximize'), ('max_dd', 'minimize')],
    constraints=[leverage_constraint, period_constraint],
    n_trials=100
)
```

### 約束函數規範

- **輸入**: 參數字典 `Dict[str, Any]`
- **輸出**: `float`
  - `> 0`: 違反約束（值表示違反程度）
  - `<= 0`: 滿足約束

### 內部實作

新增 `_check_constraints()` 方法：
- 檢查所有約束條件
- 回傳 `(is_satisfied: bool, violation_sum: float)`
- 在 `_objective()` 中整合檢查

---

## 2. 市場狀態感知（Regime-Aware）

### 功能描述

根據市場狀態（趨勢/震盪/高波動等）動態調整優化目標。

### 使用方法

```python
from src.analysis.regime_detection import MarketStateAnalyzer

# 建立 regime analyzer
regime_analyzer = MarketStateAnalyzer(data)

optimizer = MultiObjectiveOptimizer(
    objectives=[('sharpe', 'maximize'), ('max_dd', 'minimize')],
    regime_aware=True,
    regime_analyzer=regime_analyzer,
    n_trials=100
)
```

### 未來擴展方向

目前 `_evaluate_with_regime()` 是預留接口，未來可擴展為：

```python
def _evaluate_with_regime(self, params, results):
    current_regime = self.regime_analyzer.get_current_regime()

    if current_regime == 'trending':
        # 趨勢市場：提高 Sharpe 權重
        results['sharpe_ratio'] *= 1.1
    elif current_regime == 'ranging':
        # 震盪市場：提高勝率權重
        results['win_rate'] *= 1.1

    return results
```

---

## 3. 暖啟動支援

### 功能描述

使用已知的優良解作為初始搜索點，加速收斂。

### 使用方法

```python
optimizer = MultiObjectiveOptimizer(
    objectives=[('sharpe', 'maximize'), ('max_dd', 'minimize')],
    n_trials=100
)

# 使用歷史最佳參數作為起點
optimizer.warm_start([
    {'fast_period': 10, 'slow_period': 30, 'stop_loss': 2.0},
    {'fast_period': 12, 'slow_period': 26, 'stop_loss': 1.5}
])

result = optimizer.optimize(param_space, evaluate_fn)
```

### 注意事項

- 暖啟動解會被優先評估
- 剩餘試驗次數 = `n_trials - len(initial_solutions)`
- 建議與足夠的 `n_trials` 結合使用，保持探索性
- 可能導致過早收斂到局部最優

---

## 4. 膝點檢測

### 功能描述

自動找出 Pareto 前沿的「膝點」——目標之間的最佳平衡點。

### 使用方法

```python
result = optimizer.optimize(param_space, evaluate_fn)

# 找到膝點
knee = result.find_knee_point()

print(f"最佳平衡解: {knee.params}")
for obj in knee.objectives:
    print(f"  {obj.name}: {obj.value:.4f}")
```

### 演算法原理

使用幾何方法：
1. 標準化所有目標到 [0, 1]（考慮 maximize/minimize 方向）
2. 找到理想點（所有目標最優）和最差點（所有目標最差）
3. 計算每個解到理想-最差連線的距離
4. 距離最大的點即為膝點

### 適用場景

- 需要單一「推薦解」時
- 目標之間有 trade-off 關係
- 不確定如何設定權重時

---

## 5. Pareto 前沿篩選

### 功能描述

從 Pareto 前沿中篩選代表性解，避免結果過多。

### 使用方法

```python
result = optimizer.optimize(param_space, evaluate_fn)

# 方法 1: 擁擠度篩選（多樣性最大化）
diverse_solutions = result.filter_pareto_front('crowding', n_select=10)

# 方法 2: 膝點附近篩選（平衡性優先）
balanced_solutions = result.filter_pareto_front('knee', n_select=5)

# 方法 3: 極值解篩選（展示 trade-off）
extreme_solutions = result.filter_pareto_front('extreme', n_select=6)

# 方法 4: 均勻篩選（沿 Pareto 前沿均勻分佈）
uniform_solutions = result.filter_pareto_front('uniform', n_select=10)
```

### 篩選方法比較

| 方法 | 優點 | 適用場景 |
|------|------|---------|
| **crowding** | 多樣性最大化 | 需要涵蓋整個 Pareto 前沿 |
| **knee** | 平衡性優先 | 需要接近膝點的解 |
| **extreme** | 展示各目標極值 | 需要展示 trade-off |
| **uniform** | 均勻分佈 | 需要等間距的解 |

---

## 6. 額外視覺化方法

### 6.1 3D Pareto 前沿

```python
result.plot_pareto_front_3d(
    obj_x='sharpe_ratio',
    obj_y='max_drawdown',
    obj_z='sortino_ratio',
    save_path='pareto_3d.html'
)
```

**適用場景**: 三個目標的優化問題

### 6.2 平行座標圖

```python
result.plot_parallel_coordinates(
    save_path='parallel_coords.html',
    max_params=10  # 最多顯示 10 個參數
)
```

**特點**:
- 顯示所有目標和參數的關係
- 可觀察參數與目標的關聯
- 適合高維度問題

**適用場景**: 需要同時觀察多個目標和參數的關係

---

## 完整範例

```python
from src.optimizer.multi_objective import MultiObjectiveOptimizer

# 1. 定義約束
def leverage_constraint(params):
    return max(0, params['leverage'] - 10)

# 2. 建立優化器
optimizer = MultiObjectiveOptimizer(
    objectives=[
        ('sharpe_ratio', 'maximize'),
        ('max_drawdown', 'minimize'),
        ('sortino_ratio', 'maximize')
    ],
    n_trials=200,
    constraints=[leverage_constraint],
    seed=42
)

# 3. 暖啟動
optimizer.warm_start([
    {'fast_period': 10, 'slow_period': 30, 'leverage': 5},
    {'fast_period': 12, 'slow_period': 26, 'leverage': 3}
])

# 4. 執行優化
result = optimizer.optimize(param_space, evaluate_fn)

# 5. 分析結果
print(f"Pareto 前沿: {len(result.pareto_front)} 個解")

# 6. 找到膝點
knee = result.find_knee_point()
print(f"膝點: {knee.params}")

# 7. 篩選代表性解
best_solutions = result.filter_pareto_front('knee', n_select=5)

# 8. 視覺化
result.plot_pareto_front_2d('sharpe_ratio', 'max_drawdown', 'pareto_2d.html')
result.plot_pareto_front_3d('sharpe_ratio', 'max_drawdown', 'sortino_ratio', 'pareto_3d.html')
result.plot_parallel_coordinates('parallel.html')
```

---

## API 變更摘要

### MultiObjectiveOptimizer 新增參數

```python
def __init__(
    self,
    objectives: List[Tuple[str, Literal['maximize', 'minimize']]],
    n_trials: int = 100,
    seed: Optional[int] = None,
    verbose: bool = True,
    population_size: Optional[int] = None,
    mutation_prob: Optional[float] = None,
    crossover_prob: Optional[float] = None,
    # ===== 新增參數 =====
    constraints: Optional[List[Callable[[Dict], float]]] = None,
    regime_aware: bool = False,
    regime_analyzer: Optional['MarketStateAnalyzer'] = None
)
```

### MultiObjectiveOptimizer 新增方法

- `warm_start(initial_solutions)`: 暖啟動
- `_check_constraints(params)`: 檢查約束
- `_evaluate_with_regime(params, results)`: Regime 感知評估

### MultiObjectiveResult 新增方法

- `find_knee_point()`: 膝點檢測
- `filter_pareto_front(method, n_select)`: Pareto 前沿篩選
- `plot_pareto_front_3d(obj_x, obj_y, obj_z, save_path)`: 3D 視覺化
- `plot_parallel_coordinates(save_path, max_params)`: 平行座標圖

---

## 測試

完整測試位於 `tests/test_multi_objective_extended.py`：

- ✅ 約束條件支援
- ✅ 暖啟動功能
- ✅ 膝點檢測
- ✅ Pareto 前沿篩選（4 種方法）
- ✅ 視覺化方法

執行測試：
```bash
pytest tests/test_multi_objective_extended.py -v
```

---

## 範例程式

完整範例位於 `examples/multi_objective_advanced.py`：

```bash
python examples/multi_objective_advanced.py
```

輸出：
- `results/pareto_2d.html`: 2D Pareto 前沿
- `results/pareto_3d.html`: 3D Pareto 前沿
- `results/parallel_coordinates.html`: 平行座標圖
- `results/pareto_front.csv`: Pareto 前沿資料

---

## 向後相容性

所有新功能都是**可選的**，不影響現有程式碼：

```python
# 舊版使用方式（仍然有效）
optimizer = MultiObjectiveOptimizer(
    objectives=[('sharpe', 'maximize'), ('max_dd', 'minimize')],
    n_trials=100
)

result = optimizer.optimize(param_space, evaluate_fn)
best = result.get_best_solution()
result.plot_pareto_front_2d('sharpe', 'max_dd')
```

---

## 未來擴展方向

1. **動態約束**: 根據歷史結果動態調整約束條件
2. **多階段優化**: 先粗搜索，再精細化
3. **Regime-based 權重**: 根據市場狀態自動調整目標權重
4. **互動式視覺化**: 使用 Plotly Dash 建立互動介面
5. **解的聚類分析**: 將 Pareto 前沿解分群
6. **不確定性量化**: 評估解的穩健性

---

## 效能考量

- **約束檢查**: O(n_constraints) per trial，影響較小
- **暖啟動**: 初始解評估時間取決於 `evaluate_fn` 複雜度
- **膝點檢測**: O(n_solutions × n_objectives)，對 < 1000 解影響小
- **Pareto 篩選**: O(n_solutions × log n_solutions)，快速
- **視覺化**: 取決於 Plotly，建議 Pareto 前沿 < 1000 解

---

## 依賴

- `optuna >= 3.0.0` (必須)
- `numpy >= 1.20.0` (必須)
- `pandas >= 1.3.0` (必須)
- `plotly >= 5.0.0` (視覺化可選)
- `src.analysis.regime_detection` (Regime 感知可選)

---

## 參考文獻

1. **NSGA-II**: Deb et al. (2002) - "A Fast and Elitist Multiobjective Genetic Algorithm"
2. **Knee Point Detection**: Das (1999) - "Normal-Boundary Intersection"
3. **Crowding Distance**: NSGA-II 論文原始定義

---

## 作者

Claude (Developer Subagent)

## 日期

2026-01-13
