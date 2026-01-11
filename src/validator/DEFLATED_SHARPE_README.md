# Deflated Sharpe Ratio 使用指南

## 概述

Deflated Sharpe Ratio (DSR) 是 Bailey & López de Prado (2014) 提出的方法，用於校正**多重檢定偏差** (Multiple Testing Bias)。

當測試多個策略時，即使策略本身無預測能力，也有機率產生高 Sharpe Ratio。DSR 考慮嘗試次數，調整 Sharpe 的統計顯著性。

---

## 核心概念

### 1. 多重檢定問題

```
測試 100 個隨機策略：
→ 預期有 5 個會顯示「顯著」結果（假陽性）
→ 最佳的那個 Sharpe Ratio 可能很高，但只是運氣
```

### 2. Deflated Sharpe Ratio 公式

```
DSR = (SR_observed - E[max SR]) / σ_SR

其中：
- SR_observed: 觀察到的 Sharpe Ratio
- E[max SR]: 預期的最大 Sharpe（考慮嘗試次數）
- σ_SR: Sharpe 的標準誤
```

### 3. 解讀標準

- **DSR > 1.96**: 顯著（95% 信賴水準）
- **DSR > 0**: 優於隨機策略
- **DSR < 0**: 可能是運氣，不是真實技能

---

## 快速開始

### 安裝

無需額外依賴，只需要：
- numpy
- scipy

### 基本使用

```python
from src.validator.sharpe_correction import deflated_sharpe_ratio

# 假設你測試了 100 個策略，最佳的 Sharpe = 2.5
result = deflated_sharpe_ratio(
    sharpe=2.5,
    n_trials=100,        # 測試了 100 個策略
    returns=returns,     # 收益序列（用於計算 variance）
    t_years=1.0         # 回測 1 年
)

print(f"Deflated SR: {result.deflated_sharpe:.2f}")
print(f"p-value: {result.p_value:.4f}")
print(f"Significant: {result.is_significant}")
```

---

## 主要功能

### 1. Deflated Sharpe Ratio

```python
from src.validator.sharpe_correction import (
    deflated_sharpe_ratio,
    print_deflated_sharpe_report
)

result = deflated_sharpe_ratio(
    sharpe=2.0,
    n_trials=100,
    returns=returns,
    t_years=1.0
)

print_deflated_sharpe_report(result)
```

**輸出範例**：
```
======================================================================
                       Deflated Sharpe Ratio 報告
======================================================================

[Sharpe Ratio]
  觀察到的 Sharpe: 2.000
  預期最大 Sharpe (多重檢定): 0.357
  Sharpe 標準誤: 0.141

[Deflated Sharpe Ratio]
  DSR: 11.677
  p-value: 0.0000

結論: 策略具有統計顯著性 ✓
======================================================================
```

### 2. Probability of Backtest Overfitting (PBO)

```python
from src.validator.sharpe_correction import (
    probability_of_backtest_overfitting,
    print_pbo_report
)

# 計算 20 個策略的 IS/OOS Sharpe
is_sharpe = np.array([2.5, 2.3, 1.8, ...])   # In-Sample
oos_sharpe = np.array([1.2, 0.5, 1.5, ...])  # Out-of-Sample

result = probability_of_backtest_overfitting(
    is_sharpe, oos_sharpe, n_trials=20
)

print_pbo_report(result)
```

**解讀**：
- **PBO < 0.3**: 低風險 ✓
- **0.3 ≤ PBO < 0.5**: 中等風險 ⚠
- **PBO ≥ 0.5**: 高風險 ✗（可能過擬合）

### 3. 最小回測長度

```python
from src.validator.sharpe_correction import minimum_backtest_length

# 要達到 Sharpe = 2.0，測試 100 個策略
result = minimum_backtest_length(
    target_sharpe=2.0,
    n_trials=100,
    confidence=0.95
)

print(f"需要至少 {result.min_years:.1f} 年資料")
print(f"約 {result.min_observations} 天")
```

---

## 實際應用流程

### 步驟 1：記錄所有嘗試

```json
// learning/experiments.json
{
  "experiments": [
    {"id": 1, "strategy": "RSI_14", "sharpe": 1.2},
    {"id": 2, "strategy": "RSI_21", "sharpe": 1.5},
    {"id": 3, "strategy": "MA_Cross_50_200", "sharpe": 0.8},
    ...
  ]
}
```

**重點**：
- 包含所有失敗的策略
- 包含參數調整次數
- 不要只記錄「最佳」結果

### 步驟 2：計算 Deflated Sharpe

```python
# 總共嘗試次數
n_trials = len(experiments)

# 最佳策略的 Sharpe
best_sharpe = max(exp["sharpe"] for exp in experiments)

# 計算 DSR
dsr_result = deflated_sharpe_ratio(
    sharpe=best_sharpe,
    n_trials=n_trials,
    returns=returns,
    t_years=1.0
)

if not dsr_result.is_significant:
    print("警告：策略可能是多重檢定的偽陽性！")
```

### 步驟 3：Out-of-Sample 驗證

```python
# 將資料分為 IS (70%) 和 OOS (30%)
split_point = int(len(returns) * 0.7)
is_returns = returns[:split_point]
oos_returns = returns[split_point:]

# 計算所有策略在兩期間的 Sharpe
is_sharpe_list = []
oos_sharpe_list = []

for strategy in all_strategies:
    is_sharpe = calculate_sharpe(strategy.backtest(is_returns))
    oos_sharpe = calculate_sharpe(strategy.backtest(oos_returns))

    is_sharpe_list.append(is_sharpe)
    oos_sharpe_list.append(oos_sharpe)

# 計算 PBO
pbo_result = probability_of_backtest_overfitting(
    np.array(is_sharpe_list),
    np.array(oos_sharpe_list),
    n_trials=len(all_strategies)
)

if pbo_result.pbo > 0.5:
    print("警告：過擬合機率過高！")
```

### 步驟 4：確保足夠的回測長度

```python
# 檢查當前資料是否足夠
min_length = minimum_backtest_length(
    target_sharpe=best_sharpe,
    n_trials=n_trials
)

if current_years < min_length.min_years:
    print(f"警告：需要至少 {min_length.min_years:.1f} 年資料")
    print(f"當前只有 {current_years:.1f} 年")
```

---

## API 參考

### deflated_sharpe_ratio()

計算 Deflated Sharpe Ratio。

**參數**：
- `sharpe` (float): 觀察到的 Sharpe Ratio
- `n_trials` (int): 嘗試的策略數量（包含未通過的）
- `variance` (float, optional): Sharpe 的變異數（如果未提供，從 returns 計算）
- `returns` (np.ndarray, optional): 收益序列（用於計算 variance）
- `t_years` (float): 回測年數（預設 1.0）
- `confidence` (float): 信賴水準（預設 0.95）

**返回**：
- `DeflatedSharpeResult` 包含：
  - `deflated_sharpe`: 校正後的 DSR
  - `p_value`: 顯著性檢定 p 值
  - `expected_max_sharpe`: 預期的最大 Sharpe
  - `is_significant`: 是否顯著（p < 0.05）

### probability_of_backtest_overfitting()

計算 Probability of Backtest Overfitting (PBO)。

**參數**：
- `is_sharpe` (np.ndarray): In-Sample Sharpe Ratios（N 個策略）
- `oos_sharpe` (np.ndarray): Out-of-Sample Sharpe Ratios（對應的 N 個策略）
- `n_trials` (int): 嘗試的策略數量

**返回**：
- `PBOResult` 包含：
  - `pbo`: 過擬合機率（0-1）
  - `rank_correlation`: IS vs OOS 排名相關性
  - `warning`: 警告訊息（如果有）

### minimum_backtest_length()

計算達到統計顯著所需的最小回測長度。

**參數**：
- `target_sharpe` (float): 目標 Sharpe Ratio
- `n_trials` (int): 嘗試的策略數量
- `confidence` (float): 信賴水準（預設 0.95）

**返回**：
- `MinimumBacktestLength` 包含：
  - `min_years`: 最小回測年數
  - `min_observations`: 最小觀察次數（假設日資料）

---

## 常見問題

### Q1: 什麼時候需要用 Deflated Sharpe？

**A**: 當你：
- 測試了多個策略或參數組合
- 進行參數優化（網格搜尋、貝葉斯優化等）
- 需要評估「最佳」策略是否真的顯著

### Q2: n_trials 應該設為多少？

**A**: 包含所有嘗試：
- 所有測試的策略（包含失敗的）
- 所有參數組合
- 所有「微調」和「重試」

範例：
- 測試 5 個策略，每個 20 組參數 → n_trials = 100
- 手動調整 10 次 → n_trials += 10

### Q3: DSR 和原始 Sharpe 差很多，正常嗎？

**A**: DSR 是 Z-score，數值意義不同：
- 原始 Sharpe = 2.0 → 「年化超額報酬/標準差 = 2」
- DSR = 11.7 → 「顯著性統計量（類似 t 統計量）」

兩者不能直接比較，DSR 主要看：
- DSR > 1.96 → 顯著
- DSR < 1.96 → 不顯著

### Q4: PBO 多少才算安全？

**A**:
- **PBO < 0.3**: 低風險，可以接受
- **0.3 ≤ PBO < 0.5**: 中等風險，建議增加 OOS 驗證
- **PBO ≥ 0.5**: 高風險，策略可能過擬合

---

## 進階主題

### 1. 與 Bootstrap Test 結合

```python
from src.validator.statistical_tests import bootstrap_sharpe

# Bootstrap Sharpe
bootstrap_result = bootstrap_sharpe(returns, n_bootstrap=10000)

# Deflated Sharpe
dsr_result = deflated_sharpe_ratio(
    sharpe=bootstrap_result.sharpe_mean,
    n_trials=100,
    returns=returns
)

# 兩者都顯著 → 更有信心
if bootstrap_result.p_value < 0.05 and dsr_result.is_significant:
    print("策略通過多重檢定 ✓")
```

### 2. 與 Walk-Forward 分析結合

```python
from src.validator.walk_forward import combinatorial_purged_cv

# Walk-Forward 分析
wf_result = combinatorial_purged_cv(
    returns=returns,
    n_splits=5,
    n_test_splits=2
)

# 計算每個 fold 的 Sharpe
oos_sharpe_list = [calculate_sharpe(fold.oos_returns) for fold in wf_result.folds]

# 如果所有 fold 都顯著，更有信心
all_significant = all(
    deflated_sharpe_ratio(sr, n_trials=50, variance=0.004).is_significant
    for sr in oos_sharpe_list
)
```

---

## 參考文獻

Bailey, D. H., & López de Prado, M. (2014). **The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality**. *Journal of Portfolio Management*, 40(5), 94-107.

---

## 完整範例

執行 `examples/deflated_sharpe_demo.py` 查看完整使用範例：

```bash
python examples/deflated_sharpe_demo.py
```

包含：
1. 基本 Deflated Sharpe 計算
2. PBO 檢測（過擬合 vs 穩健策略）
3. 最小回測長度計算
4. 多重檢定偏差影響分析
5. 實際應用建議
