# Monte Carlo 模擬器使用指南

Monte Carlo 模擬器用於評估交易策略在不同隨機情境下的表現，幫助了解策略的穩健性和風險特徵。

## 目錄

- [核心概念](#核心概念)
- [快速開始](#快速開始)
- [模擬方法](#模擬方法)
- [結果解讀](#結果解讀)
- [進階用法](#進階用法)
- [實際應用](#實際應用)

## 核心概念

### 什麼是 Monte Carlo 模擬？

Monte Carlo 模擬透過隨機抽樣，產生大量可能的交易序列，用於：

1. **評估穩健性**：策略是否依賴特定交易順序？
2. **量化風險**：最壞情況下可能虧損多少？
3. **估計機率**：獲利的可能性有多高？
4. **壓力測試**：在不同情境下策略的表現如何？

### 為什麼需要？

單一回測結果只代表「一種可能」，Monte Carlo 模擬讓你看到「所有可能」的分布。

## 快速開始

### 基本使用

```python
from src.validator import MonteCarloSimulator
import pandas as pd

# 準備交易記錄（必須包含 'pnl' 欄位）
trades = pd.DataFrame({
    'pnl': [100, -50, 80, -30, 120, -40, 90],
    'timestamp': pd.date_range('2024-01-01', periods=7, freq='1H')
})

# 建立模擬器
simulator = MonteCarloSimulator(seed=42)  # seed 用於可重現性

# 執行模擬（預設 1000 次）
result = simulator.simulate(
    trades=trades,
    n_simulations=10000,
    method='shuffle'
)

# 查看結果
simulator.print_result(result)

# 繪製分布圖
simulator.plot_distribution(result)
```

### 輸出範例

```
Monte Carlo 模擬結果
============================================================
模擬次數: 10000
模擬方法: shuffle

分布統計
------------------------------------------------------------
平均報酬:           270.00
標準差:              89.44
中位數:             270.00

百分位數
------------------------------------------------------------
1%:                100.00
5%:                120.00
25%:               210.00
75%:               330.00
95%:               420.00
99%:               440.00

風險指標
------------------------------------------------------------
VaR (95%):         120.00
CVaR (95%):        110.00

績效比較
------------------------------------------------------------
原始報酬:           270.00
獲利機率:           95.30%
超越原始機率:       50.00%
```

## 模擬方法

### 1. Shuffle（交易順序隨機化）

**原理**：保持交易內容不變，只打亂順序。

**適用場景**：
- 測試策略是否依賴特定交易順序
- 評估序列相關性的影響

**特點**：
- 總報酬固定（所有模擬結果總和相同）
- 變異主要來自順序效應

```python
result = simulator.simulate(
    trades=trades,
    n_simulations=5000,
    method='shuffle'
)
```

### 2. Bootstrap（有放回抽樣）

**原理**：從原始交易中隨機抽取（可重複），產生新的交易序列。

**適用場景**：
- 評估交易樣本的不確定性
- 產生更多樣的可能情境

**特點**：
- 總報酬會變化
- 某些交易可能被抽中多次，某些可能不被抽中

```python
result = simulator.simulate(
    trades=trades,
    n_simulations=5000,
    method='bootstrap'
)
```

### 3. Block Bootstrap（區塊抽樣）

**原理**：將交易分成連續區塊，對區塊進行抽樣，保留時間相關性。

**適用場景**：
- 策略有序列相關性（如趨勢追蹤）
- 需要保持局部時間結構

**參數**：
- `block_size`：區塊大小（預設 5）

```python
result = simulator.simulate(
    trades=trades,
    n_simulations=5000,
    method='block_bootstrap',
    block_size=10  # 每個區塊包含 10 筆交易
)
```

## 結果解讀

### 分布統計

| 指標 | 說明 | 判斷標準 |
|------|------|----------|
| **平均報酬** | 模擬的期望報酬 | 應該接近原始報酬 |
| **標準差** | 報酬的波動程度 | 越小越穩定 |
| **中位數** | 50% 情境的報酬 | 應該接近平均值（對稱分布） |

### 百分位數

| 百分位 | 說明 | 用途 |
|--------|------|------|
| **1%** | 最差 1% 情境 | 極端風險評估 |
| **5%** | 最差 5% 情境 | VaR 計算 |
| **25%** | 下四分位數 | 較差情境 |
| **75%** | 上四分位數 | 較好情境 |
| **95%** | 最佳 5% 情境 | 上行潛力 |
| **99%** | 最佳 1% 情境 | 最佳情境 |

### 風險指標

**VaR (Value at Risk)**
- **定義**：95% 信心水準下的最大損失
- **解讀**：「有 95% 的機會，報酬不會低於這個值」
- **範例**：VaR(95%) = 100，表示只有 5% 機會報酬低於 100

**CVaR (Conditional VaR)**
- **定義**：超過 VaR 的平均損失
- **解讀**：「最差 5% 情境的平均報酬」
- **重要性**：比 VaR 更能反映極端風險

### 機率指標

**獲利機率**
- `P(報酬 > 0)`
- 建議：至少 > 60%

**超越原始機率**
- `P(報酬 > 原始報酬)`
- 應該接近 50%（對稱分布）
- 若明顯偏離，表示分布偏態或樣本偏差

## 進階用法

### 權益曲線路徑模擬

```python
# 產生權益曲線路徑
equity_paths, original_path = simulator.generate_equity_paths(
    trades=trades,
    n_simulations=1000,
    method='bootstrap'
)

# 繪製路徑（顯示 100 條）
simulator.plot_paths(
    equity_paths=equity_paths,
    original_path=original_path,
    n_paths_to_plot=100
)
```

### 比較不同模擬方法

```python
methods = ['shuffle', 'bootstrap', 'block_bootstrap']
results = {}

for method in methods:
    results[method] = simulator.simulate(
        trades=trades,
        n_simulations=5000,
        method=method
    )

# 比較結果
for method, result in results.items():
    print(f"\n{method}:")
    print(f"  平均: {result.mean:.2f}")
    print(f"  標準差: {result.std:.2f}")
    print(f"  VaR(95%): {result.var_95:.2f}")
```

### 自訂風險分析

```python
# 計算其他風險指標
simulated_returns = result.simulated_returns

# 最大損失機率
max_loss = simulated_returns.min()
print(f"最大損失: {max_loss:.2f}")

# 盈虧比
gains = simulated_returns[simulated_returns > 0]
losses = simulated_returns[simulated_returns < 0]
profit_loss_ratio = gains.mean() / abs(losses.mean())
print(f"盈虧比: {profit_loss_ratio:.2f}")

# 偏態係數
from scipy import stats
skewness = stats.skew(simulated_returns)
print(f"偏態: {skewness:.2f}")  # > 0 右偏，< 0 左偏
```

## 實際應用

### 策略穩健性測試

```python
def assess_robustness(trades: pd.DataFrame) -> dict:
    """評估策略穩健性"""
    simulator = MonteCarloSimulator(seed=42)

    # Bootstrap 模擬
    result = simulator.simulate(
        trades=trades,
        n_simulations=10000,
        method='bootstrap'
    )

    # 穩健性指標
    cv = result.std / abs(result.mean)  # 變異係數
    stability = result.probability_profitable  # 獲利穩定性

    # 評級
    if cv < 0.5 and stability > 0.7:
        grade = "優秀"
    elif cv < 1.0 and stability > 0.6:
        grade = "良好"
    elif cv < 1.5 and stability > 0.55:
        grade = "及格"
    else:
        grade = "不及格"

    return {
        'grade': grade,
        'cv': cv,
        'stability': stability,
        'var_95': result.var_95,
        'cvar_95': result.cvar_95,
    }

# 使用
robustness = assess_robustness(trades)
print(f"穩健性評級: {robustness['grade']}")
```

### 策略比較

```python
def compare_strategies(
    strategy_a: pd.DataFrame,
    strategy_b: pd.DataFrame
) -> None:
    """比較兩個策略"""
    simulator = MonteCarloSimulator(seed=42)

    result_a = simulator.simulate(strategy_a, 5000, 'bootstrap')
    result_b = simulator.simulate(strategy_b, 5000, 'bootstrap')

    print(f"{'指標':<20} {'策略 A':>12} {'策略 B':>12} {'較佳':<8}")
    print("-" * 54)

    # 報酬
    print(f"{'平均報酬':<20} {result_a.mean:>12.2f} {result_b.mean:>12.2f} "
          f"{'A' if result_a.mean > result_b.mean else 'B'}")

    # 風險
    print(f"{'標準差':<20} {result_a.std:>12.2f} {result_b.std:>12.2f} "
          f"{'A' if result_a.std < result_b.std else 'B'}")

    # 風險調整報酬
    sharpe_a = result_a.mean / result_a.std
    sharpe_b = result_b.mean / result_b.std
    print(f"{'夏普比率':<20} {sharpe_a:>12.2f} {sharpe_b:>12.2f} "
          f"{'A' if sharpe_a > sharpe_b else 'B'}")

    # 下行風險
    print(f"{'VaR(95%)':<20} {result_a.var_95:>12.2f} {result_b.var_95:>12.2f} "
          f"{'A' if result_a.var_95 > result_b.var_95 else 'B'}")
```

### 風險限制評估

```python
def check_risk_limits(
    trades: pd.DataFrame,
    max_var: float = -1000,
    min_prob_profit: float = 0.6
) -> bool:
    """檢查是否符合風險限制"""
    simulator = MonteCarloSimulator(seed=42)
    result = simulator.simulate(trades, 10000, 'bootstrap')

    # 檢查 VaR
    var_ok = result.var_95 > max_var

    # 檢查獲利機率
    prob_ok = result.probability_profitable > min_prob_profit

    print(f"VaR(95%) 檢查: {result.var_95:.2f} > {max_var:.2f} → "
          f"{'✓' if var_ok else '✗'}")
    print(f"獲利機率檢查: {result.probability_profitable:.2%} > "
          f"{min_prob_profit:.2%} → {'✓' if prob_ok else '✗'}")

    return var_ok and prob_ok

# 使用
passed = check_risk_limits(trades)
print(f"\n{'通過' if passed else '未通過'}風險限制")
```

## 注意事項

### 1. 樣本大小

- **最少交易數**：建議 ≥ 30 筆
- **模擬次數**：通常 1000-10000 次足夠
- **更多不一定更好**：10000 次以上收益遞減

### 2. 方法選擇

| 情境 | 建議方法 |
|------|----------|
| 獨立交易 | Shuffle 或 Bootstrap |
| 趨勢策略 | Block Bootstrap |
| 均值回歸 | Shuffle |
| 未知特性 | 都試試看 |

### 3. 結果解讀

- **分布形狀**：正常應接近常態分布
- **對稱性**：嚴重偏態可能有問題
- **離群值**：檢查是否為異常交易

### 4. 限制

Monte Carlo 模擬**假設未來與過去相似**：
- 無法預測市場結構改變
- 無法模擬未見過的事件
- 不能替代實際驗證

## 完整範例

參考 `examples/monte_carlo_example.py` 查看完整範例，包括：

1. 基本模擬
2. 方法比較
3. 權益曲線路徑
4. 風險分析
5. 策略穩健性測試

執行範例：

```bash
python examples/monte_carlo_example.py
```

## 參考資料

- **測試檔案**：`tests/test_monte_carlo.py`
- **原始碼**：`src/validator/monte_carlo.py`
- **API 文檔**：參考原始碼 docstrings
