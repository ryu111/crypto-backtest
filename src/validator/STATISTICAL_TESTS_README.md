# 統計檢定模組 (Statistical Tests)

## 概述

提供進階統計檢定方法，用於驗證交易策略績效的**統計顯著性**。

主要功能：
1. **Bootstrap Test** - 計算 Sharpe Ratio 的信賴區間
2. **Permutation Test** - 檢定策略績效是否顯著優於隨機
3. **Block Bootstrap** - 保留時間序列相關性的 Bootstrap
4. **統計檢定報告** - 整合所有檢定結果的綜合報告

---

## 為什麼需要統計檢定？

策略回測績效良好不代表真的有效，可能只是運氣。統計檢定幫助我們回答：

- **Sharpe Ratio 是否顯著大於 0？** → Bootstrap Test
- **策略是否優於隨機選股？** → Permutation Test
- **信賴區間有多寬？** → Bootstrap CI

---

## 快速開始

### 1. 基礎使用

```python
from src.validator.statistical_tests import bootstrap_sharpe, permutation_test

# 假設你有一個策略的收益序列
returns = strategy.calculate_returns()  # numpy array，日收益

# Bootstrap Test
result = bootstrap_sharpe(returns, n_bootstrap=10000)
print(f"Sharpe: {result.sharpe_mean:.2f}")
print(f"95% CI: ({result.ci_lower:.2f}, {result.ci_upper:.2f})")
print(f"p-value: {result.p_value:.4f}")

# Permutation Test
result = permutation_test(returns, n_permutations=10000)
if result.is_significant:
    print("策略顯著優於隨機 ✓")
```

### 2. 完整統計檢定流程

```python
from src.validator.statistical_tests import run_statistical_tests, print_test_report

# 執行所有檢定
report = run_statistical_tests(
    returns,
    n_bootstrap=10000,
    n_permutations=10000,
    n_jobs=-1  # 使用所有 CPU 核心
)

# 美化輸出
print_test_report(report)

# 判斷顯著性
if report.is_statistically_significant:
    print("策略通過統計檢定 ✓")
    print(f"Sharpe: {report.bootstrap_sharpe:.2f} {report.bootstrap_ci}")
```

---

## API 文件

### Bootstrap Test

計算 Sharpe Ratio 的 Bootstrap 信賴區間。

```python
bootstrap_sharpe(
    returns: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    risk_free_rate: float = 0.0,
    n_jobs: int = -1,
    random_state: Optional[int] = None
) -> BootstrapResult
```

**參數：**
- `returns`: 收益序列（1D numpy array）
- `n_bootstrap`: Bootstrap 重複次數（預設 10000）
- `confidence`: 信賴水準（預設 0.95）
- `risk_free_rate`: 無風險利率，年化（預設 0.0）
- `n_jobs`: 並行工作數，-1 表示使用所有 CPU（預設 -1）
- `random_state`: 隨機種子（可選）

**回傳：**
- `sharpe_mean`: Bootstrap 平均 Sharpe
- `ci_lower, ci_upper`: 信賴區間
- `p_value`: H0: Sharpe ≤ 0 的 p 值
- `sharpe_distribution`: 完整的 Sharpe 分布（用於進階分析）

**範例：**
```python
result = bootstrap_sharpe(returns, n_bootstrap=10000)
if result.p_value < 0.05:
    print(f"Sharpe 顯著 > 0: {result.sharpe_mean:.2f} "
          f"({result.ci_lower:.2f}, {result.ci_upper:.2f})")
```

---

### Permutation Test

檢定策略績效是否顯著優於隨機。

```python
permutation_test(
    returns: np.ndarray,
    n_permutations: int = 10000,
    risk_free_rate: float = 0.0,
    n_jobs: int = -1,
    random_state: Optional[int] = None
) -> PermutationResult
```

**方法說明：**
- 虛無假設 H0: 收益的符號是隨機的（無預測能力）
- 透過隨機翻轉收益符號，建立虛無假設分布
- 計算實際 Sharpe 在虛無假設分布中的百分位

**回傳：**
- `actual_sharpe`: 實際策略的 Sharpe
- `p_value`: 單尾 p 值（H1: actual > null）
- `is_significant`: p < 0.05?

**範例：**
```python
result = permutation_test(returns, n_permutations=10000)
print(f"p-value: {result.p_value:.4f}")
if result.is_significant:
    print("策略顯著優於隨機 ✓")
```

---

### Block Bootstrap

保留時間序列相關性的 Bootstrap。

```python
block_bootstrap(
    returns: np.ndarray,
    block_size: int = 20,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    risk_free_rate: float = 0.0,
    random_state: Optional[int] = None
) -> BootstrapResult
```

**使用時機：**
- 當收益序列有**自相關**（autocorrelation）時使用
- 金融時間序列通常有自相關性（趨勢、週期性）
- Block Bootstrap 以連續區塊為單位抽樣，保留時間依賴性

**參數：**
- `block_size`: 區塊大小，建議 10-30 天（預設 20）
- 其他參數同 `bootstrap_sharpe`

**如何選擇 block_size：**
1. 計算自相關函數（ACF）
2. 找到首次降至 0 的 lag
3. block_size ≈ 該 lag 值

**範例：**
```python
# 有自相關的策略
result = block_bootstrap(returns, block_size=20, n_bootstrap=10000)
print(f"Block Bootstrap Sharpe: {result.sharpe_mean:.2f} "
      f"({result.ci_lower:.2f}, {result.ci_upper:.2f})")
```

---

### 完整統計檢定

同時執行 Bootstrap 和 Permutation Test，產生綜合報告。

```python
run_statistical_tests(
    returns: np.ndarray,
    n_bootstrap: int = 10000,
    n_permutations: int = 10000,
    confidence: float = 0.95,
    use_block_bootstrap: bool = False,
    block_size: int = 20,
    risk_free_rate: float = 0.0,
    n_jobs: int = -1,
    random_state: Optional[int] = None
) -> StatisticalTestReport
```

**回傳報告包含：**
- `bootstrap_sharpe`: Bootstrap Sharpe 估計
- `bootstrap_ci`: Bootstrap 信賴區間
- `bootstrap_p_value`: Bootstrap p-value
- `permutation_p_value`: Permutation p-value
- `is_statistically_significant`: **兩項檢定都通過**才為 True

**顯著性判定規則：**
```python
is_significant = (bootstrap_p_value < 0.05) AND (permutation_p_value < 0.05)
```

**範例：**
```python
report = run_statistical_tests(returns)

if report.is_statistically_significant:
    print("策略通過統計檢定 ✓")
else:
    print("策略未通過統計檢定 ✗")
    if report.bootstrap_p_value >= 0.05:
        print("  原因: Sharpe 不顯著")
    if report.permutation_p_value >= 0.05:
        print("  原因: 績效可能是隨機的")
```

---

## 效能優化

### 多核心並行

所有 Bootstrap 和 Permutation 函數都支援多核心並行：

```python
# 使用所有 CPU 核心
result = bootstrap_sharpe(returns, n_bootstrap=100000, n_jobs=-1)

# 指定核心數
result = bootstrap_sharpe(returns, n_bootstrap=100000, n_jobs=8)

# 單執行緒（測試用）
result = bootstrap_sharpe(returns, n_bootstrap=100000, n_jobs=1)
```

### 效能建議

| 任務 | n_bootstrap | n_permutations | 預計時間（16 核心） |
|------|-------------|----------------|---------------------|
| 快速測試 | 1,000 | 1,000 | ~1 秒 |
| 一般使用 | 10,000 | 10,000 | ~10 秒 |
| 研究級 | 100,000 | 100,000 | ~1-2 分鐘 |

---

## 統計解釋

### p-value 如何解讀？

| p-value | 解釋 | 建議 |
|---------|------|------|
| < 0.001 | 極度顯著 | 強力證據支持策略有效 |
| < 0.01 | 非常顯著 | 策略很可能有效 |
| < 0.05 | 顯著（常用閾值） | 可以考慮實盤 |
| < 0.10 | 邊緣顯著 | 需要更多證據 |
| ≥ 0.10 | 不顯著 | 可能只是運氣，不建議實盤 |

### 信賴區間如何解讀？

```python
result = bootstrap_sharpe(returns)
# Sharpe: 1.5, 95% CI: (0.8, 2.2)

# 解讀：
# - 我們有 95% 的信心，真實 Sharpe 在 0.8 ~ 2.2 之間
# - 區間不包含 0 → 顯著
# - 區間越窄 → 估計越精確
```

---

## 最佳實踐

### 1. 什麼時候用 Bootstrap？

- 想知道 Sharpe Ratio 的**不確定性**
- 需要計算**信賴區間**
- 資料量較小（< 500 筆）時尤其重要

### 2. 什麼時候用 Permutation Test？

- 懷疑策略可能只是運氣
- 需要檢定**績效是否真的優於隨機**
- 配合 Bootstrap 使用，雙重驗證

### 3. 什麼時候用 Block Bootstrap？

- 收益序列有**明顯趨勢**或**週期性**
- 檢定自相關：`from statsmodels.stats.diagnostic import acorr_ljungbox`
- 如果 Ljung-Box p-value < 0.05 → 使用 Block Bootstrap

### 4. 樣本數建議

| 資料筆數 | Bootstrap 次數 | 可靠度 |
|----------|----------------|--------|
| < 100 | 5,000 - 10,000 | 中 |
| 100 - 500 | 10,000 | 高 |
| > 500 | 10,000 - 50,000 | 極高 |

---

## 常見問題

### Q1: 為什麼 Permutation Test 要翻轉符號？

傳統 Permutation Test 打亂順序，但對 Sharpe Ratio 無效（均值和標準差不變）。
我們改為**翻轉符號**，檢定「收益的方向性是否有意義」。

### Q2: Bootstrap 和 Permutation 有什麼區別？

| 方法 | 檢定目標 | 虛無假設 |
|------|----------|----------|
| Bootstrap | Sharpe 是否 > 0 | H0: Sharpe ≤ 0 |
| Permutation | 績效是否優於隨機 | H0: 收益符號隨機 |

兩者互補，建議同時使用。

### Q3: 為什麼需要多核心？

Bootstrap 需要重複計算數萬次，多核心可顯著加速：
- 單核心：100k Bootstrap ≈ 5 分鐘
- 16 核心：100k Bootstrap ≈ 20 秒

---

## 實戰範例

### 範例 1: 驗證趨勢追蹤策略

```python
from src.validator.statistical_tests import run_statistical_tests

# 計算策略收益
returns = trend_strategy.backtest()

# 完整統計檢定
report = run_statistical_tests(returns, n_bootstrap=10000, n_permutations=10000)

if report.is_statistically_significant:
    print(f"✓ 策略有效!")
    print(f"  Sharpe: {report.bootstrap_sharpe:.2f} {report.bootstrap_ci}")
    print(f"  Permutation p-value: {report.permutation_p_value:.4f}")
else:
    print("✗ 策略可能只是運氣，不建議實盤")
```

### 範例 2: 比較多個策略

```python
from src.validator.statistical_tests import bootstrap_sharpe

strategies = {
    'MA Crossover': ma_strategy,
    'Mean Reversion': mr_strategy,
    'Momentum': momentum_strategy
}

for name, strategy in strategies.items():
    returns = strategy.backtest()
    result = bootstrap_sharpe(returns)

    print(f"\n{name}:")
    print(f"  Sharpe: {result.sharpe_mean:.2f} ({result.ci_lower:.2f}, {result.ci_upper:.2f})")
    print(f"  顯著性: {'✓' if result.p_value < 0.05 else '✗'}")
```

### 範例 3: 處理自相關

```python
from src.validator.statistical_tests import block_bootstrap
from statsmodels.stats.diagnostic import acorr_ljungbox

# 檢測自相關
lb_test = acorr_ljungbox(returns, lags=20)
if lb_test['lb_pvalue'][0] < 0.05:
    print("偵測到自相關，使用 Block Bootstrap")
    result = block_bootstrap(returns, block_size=20)
else:
    print("無顯著自相關，使用標準 Bootstrap")
    result = bootstrap_sharpe(returns)
```

---

## 參考文獻

1. Efron, B., & Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*.
2. Politis, D. N., & Romano, J. P. (1994). *The Stationary Bootstrap*.
3. Bailey, D. H., & López de Prado, M. (2014). *The Deflated Sharpe Ratio*.

---

## 技術細節

### 計算公式

**年化 Sharpe Ratio（日收益）:**
```
Sharpe = (mean(returns) - rf/252) / std(returns) * sqrt(252)
```

**Bootstrap 信賴區間（百分位法）:**
```
CI = [percentile(dist, α/2), percentile(dist, 1-α/2)]
```

**Permutation p-value:**
```
p = mean(null_distribution >= actual_sharpe)
```

### 資料類別

**BootstrapResult:**
```python
@dataclass
class BootstrapResult:
    sharpe_mean: float
    sharpe_std: float
    ci_lower: float
    ci_upper: float
    p_value: float
    confidence: float
    n_bootstrap: int
    sharpe_distribution: np.ndarray
```

**PermutationResult:**
```python
@dataclass
class PermutationResult:
    actual_sharpe: float
    null_mean: float
    null_std: float
    p_value: float
    is_significant: bool
    n_permutations: int
    null_distribution: np.ndarray
```

**StatisticalTestReport:**
```python
@dataclass
class StatisticalTestReport:
    bootstrap_sharpe: float
    bootstrap_ci: Tuple[float, float]
    bootstrap_p_value: float
    permutation_p_value: float
    is_statistically_significant: bool
    confidence_level: float
    bootstrap_result: Optional[BootstrapResult]
    permutation_result: Optional[PermutationResult]
```

---

## 下一步

1. 執行範例：`python examples/statistical_tests_demo.py`
2. 閱讀測試：`tests/test_statistical_tests.py`
3. 整合到驗證流程：`src/validator/stages.py`

更多資訊請參考：
- [Monte Carlo 模擬](./monte_carlo.py)
- [5 階段驗證系統](./stages.py)
