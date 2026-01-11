# Task 2.2: Combinatorial Purged Cross-Validation

## 概述

實作了 **Combinatorial Purged Cross-Validation**，一種專為時間序列資料設計的交叉驗證方法，用於防止資訊洩漏並檢測策略過擬合。

## 核心功能

### 1. PurgedKFold

防止資訊洩漏的 K-Fold Cross-Validation。

**關鍵特性**：
- **Purge Gap**: 訓練集結束和測試集開始之間的緩衝期，避免 look-ahead bias
- **Embargo Period**: 測試集結束後的禁運期，防止資訊反向洩漏
- **時序保持**: 不打亂資料順序，保持時間序列特性

```python
from src.validator.walk_forward import PurgedKFold

# 建立 PurgedKFold
kfold = PurgedKFold(
    n_splits=5,           # K-Fold 數量
    purge_gap=24,         # 24 小時的 purge gap
    embargo_pct=0.01      # 1% embargo 期間
)

# 獲取 splits
splits = kfold.split(data)

# 手動使用每個 fold
for fold_id, (train_idx, test_idx) in enumerate(splits):
    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]
    # ... 訓練和測試
```

### 2. CombinatorialPurgedCV

結合 Purged K-Fold 和組合驗證的完整解決方案。

**工作原理**：
1. 將資料分成 N 個 fold
2. 每次選擇 k 個 fold 作為測試集
3. 剩餘 fold 作為訓練集
4. 測試所有可能的組合（C(N, k) 種）
5. 每個 fold 之間加入 purge gap 和 embargo

**優勢**：
- 更多樣的測試組合
- 每個樣本都有機會被測試多次
- 降低單一測試集的偶然性
- 提供穩健的過擬合檢測

```python
from src.validator.walk_forward import CombinatorialPurgedCV

# 建立 CombinatorialPurgedCV
cv = CombinatorialPurgedCV(
    n_splits=5,           # 5 個 fold
    n_test_groups=2,      # 每次用 2 個 fold 當測試集
    purge_gap=24,         # 24 小時 purge gap
    embargo_pct=0.01      # 1% embargo
)

# 定義策略函數
def my_strategy(data, params):
    # ... 策略邏輯
    return {
        'return': total_return,
        'sharpe': sharpe_ratio,
        'trades': num_trades
    }

# 執行驗證
result = cv.validate(
    data=market_data,
    strategy_func=my_strategy,
    param_grid={'threshold': [0.001, 0.002, 0.005]},
    optimize_metric='sharpe'
)

# 查看結果
print(result.summary())
```

### 3. 便捷函數

快速驗證策略的簡化介面：

```python
from src.validator.walk_forward import combinatorial_purged_cv

result = combinatorial_purged_cv(
    returns=my_returns,
    strategy_func=my_strategy,
    n_splits=5,
    n_test_groups=2,
    purge_gap=24,
    embargo_pct=0.01,
    param_grid={'threshold': [0.001, 0.005]}
)
```

## 結果解讀

### 1. 過擬合比例（Overfitting Ratio）

```
過擬合比例 = 平均測試報酬 / 平均訓練報酬
```

**評判標準**：
- `> 100%`: 測試優於訓練（unlikely but possible）
- `80-100%`: **穩健**
- `50-80%`: 輕微過擬合
- `< 50%`: **嚴重過擬合**

### 2. 測試集勝率（Consistency）

```
勝率 = 測試集正報酬的 fold 數量 / 總 fold 數量
```

**評判標準**：
- `> 60%`: **穩健**
- `40-60%`: 中等
- `< 40%`: 不穩定

### 3. 測試報酬標準差

低標準差 → 穩定性高
高標準差 → 結果波動大

## 使用場景

### 場景 1: 檢測策略過擬合

```python
# 使用多組合驗證
result = combinatorial_purged_cv(
    returns=backtest_returns,
    strategy_func=my_strategy,
    n_splits=5,
    n_test_groups=2
)

# 檢查過擬合比例
if result.overfitting_ratio < 0.5:
    print("⚠️  嚴重過擬合！策略可能不穩健")
elif result.overfitting_ratio < 0.8:
    print("⚠️  輕微過擬合")
else:
    print("✓ 策略穩健")
```

### 場景 2: 參數優化

```python
# 在每個 fold 上優化參數
result = cv.validate(
    data=data,
    strategy_func=my_strategy,
    param_grid={
        'period': [10, 20, 30],
        'threshold': [0.001, 0.005, 0.01]
    },
    optimize_metric='sharpe'
)

# 檢視每個 fold 的最佳參數
for fold in result.folds:
    print(f"Fold {fold.fold_id}: {fold.best_params}")
```

### 場景 3: 比較不同配置

```python
configs = [
    {'purge_gap': 0, 'embargo_pct': 0.0},      # 無防護
    {'purge_gap': 24, 'embargo_pct': 0.0},     # 只有 purge
    {'purge_gap': 24, 'embargo_pct': 0.01}     # 完整防護
]

for config in configs:
    cv = CombinatorialPurgedCV(n_splits=5, n_test_groups=2, **config)
    result = cv.validate(data=data, strategy_func=my_strategy)
    print(f"{config}: 過擬合比例={result.overfitting_ratio:.2%}")
```

## 技術細節

### Purge Gap 的作用

在訓練集結束和測試集開始之間創建緩衝區：

```
訓練集 | Purge Gap | 測試集
───────┼───────────┼───────
  使用  |   移除    |  使用
```

**為什麼需要？**
- 避免 look-ahead bias
- 防止訓練集的最後幾筆交易影響測試集的第一筆
- 模擬真實交易中的訊息傳遞延遲

### Embargo Period 的作用

在測試集結束後創建禁運期：

```
測試集 | Embargo Period | 下一個訓練集
───────┼────────────────┼─────────────
  使用  |      移除      |     使用
```

**為什麼需要？**
- 防止資訊反向洩漏
- 測試集的後期交易可能影響訓練集
- 確保時間序列的因果關係

### 組合數量計算

```python
from math import comb

n_splits = 5
n_test_groups = 2
n_combinations = comb(n_splits, n_test_groups)  # C(5, 2) = 10
```

## 最佳實踐

### 1. Purge Gap 設定

```python
# 根據資料頻率設定
hourly_data: purge_gap = 24   # 1 天
daily_data: purge_gap = 7     # 1 週
minute_data: purge_gap = 1440 # 1 天
```

### 2. Embargo 百分比

```python
# 一般建議
embargo_pct = 0.01  # 1%
# 或
embargo_pct = 0.02  # 2%（更保守）
```

### 3. K-Fold 數量

```python
# 根據資料量選擇
大資料集（> 10000 筆）: n_splits = 5-10
中等資料集（1000-10000）: n_splits = 3-5
小資料集（< 1000）: n_splits = 3
```

### 4. 測試組數量

```python
# 一般建議
n_test_groups = 1  # 快速驗證
n_test_groups = 2  # 標準驗證
n_test_groups = 3  # 深度驗證（但組合數爆炸）
```

## 範例程式

完整範例請參考：

```bash
python examples/purged_cv_example.py
```

範例包含：
1. PurgedKFold 基礎使用
2. CombinatorialPurgedCV 完整流程
3. 使用便捷函數
4. 手動使用 fold splits
5. 比較不同配置

## 測試

執行單元測試：

```bash
pytest tests/test_purged_cv.py -v
```

測試涵蓋：
- PurgedKFold 基礎功能
- Purge Gap 正確性
- Embargo Period 正確性
- CombinatorialPurgedCV 完整流程
- 參數優化
- 過擬合檢測
- 邊界條件

## 參考文獻

1. **Advances in Financial Machine Learning**
   Marcos López de Prado, Chapter 7
   詳細介紹了 Purged K-Fold 和 Combinatorial CV 的理論基礎

2. **The 7 Reasons Most Machine Learning Funds Fail**
   Marcos López de Prado
   討論了回測過擬合和資訊洩漏問題

3. **Cross-Validation in Finance**
   強調時間序列資料中的特殊考量

## API 參考

### PurgedKFold

```python
class PurgedKFold:
    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 0,
        embargo_pct: float = 0.01
    )

    def split(self, data: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]

    def get_fold_info(self, data: pd.DataFrame, fold_id: int) -> Dict[str, Any]
```

### CombinatorialPurgedCV

```python
class CombinatorialPurgedCV:
    def __init__(
        self,
        n_splits: int = 5,
        n_test_groups: int = 2,
        purge_gap: int = 0,
        embargo_pct: float = 0.01
    )

    def validate(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        param_grid: Optional[Dict[str, List]] = None,
        optimize_metric: str = 'sharpe_ratio',
        verbose: bool = True
    ) -> CombinatorialCVResult
```

### 結果物件

```python
@dataclass
class CombinatorialCVResult:
    folds: List[PurgedFoldResult]
    n_splits: int
    n_test_groups: int
    n_combinations: int
    mean_train_return: float
    mean_test_return: float
    mean_train_sharpe: float
    mean_test_sharpe: float
    test_return_std: float
    test_sharpe_std: float
    overfitting_ratio: float
    consistency: float
    purge_gap: int
    embargo_pct: float

    def to_dict(self) -> Dict
    def summary(self) -> str
```

## 整合到現有系統

在 5 階段驗證系統中使用：

```python
from src.validator import (
    get_stage_validator,
    combinatorial_purged_cv
)

# Stage 2: 參數穩健性測試
result = combinatorial_purged_cv(
    returns=backtest_returns,
    strategy_func=my_strategy,
    n_splits=5,
    n_test_groups=2
)

# 整合到驗證報告
if result.overfitting_ratio >= 0.8 and result.consistency >= 0.6:
    print("✓ Stage 2 通過")
else:
    print("✗ Stage 2 失敗")
```

## 效能考量

### 計算複雜度

```python
組合數 = C(n_splits, n_test_groups)
總評估次數 = 組合數 × 參數組合數

# 範例
n_splits = 5, n_test_groups = 2
param_combinations = 3 × 3 = 9
total = C(5, 2) × 9 = 10 × 9 = 90 次評估
```

### 優化建議

1. **減少組合數**：使用較小的 `n_test_groups`
2. **粗網格搜尋**：先用較少的參數組合
3. **並行處理**：未來可實作多執行緒版本
4. **快取結果**：重複的訓練集可快取結果

## 已知限制

1. 計算時間隨組合數指數增長
2. 需要足夠的資料量（建議 > 1000 筆）
3. 策略函數需要設計為可重複呼叫
4. 目前為單執行緒實作

## 未來改進

- [ ] 多執行緒並行處理
- [ ] 支援自定義評估指標
- [ ] 視覺化 fold splits
- [ ] 整合到 UI 介面
- [ ] 支援不同類型的 embargo（固定天數 vs 百分比）
