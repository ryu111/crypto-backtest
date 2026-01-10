# Monte Carlo 模擬詳解

## 核心概念

Monte Carlo 模擬通過隨機化來評估策略的穩健性和風險分佈。

### 為什麼需要

- 單次回測只是眾多可能結果之一
- 交易順序影響最終結果
- 需要了解風險的概率分佈

## 模擬方法

### 方法 1：交易順序重排

```python
import numpy as np
import pandas as pd

def monte_carlo_trade_shuffle(trades: pd.DataFrame, n_simulations: int = 1000):
    """
    隨機重排交易順序

    Args:
        trades: 交易記錄 DataFrame，需有 'return' 欄位
        n_simulations: 模擬次數

    Returns:
        模擬結果分佈
    """
    results = []

    for _ in range(n_simulations):
        # 隨機打亂交易順序
        shuffled = trades.sample(frac=1, replace=False)

        # 計算累積權益
        cumulative = (1 + shuffled['return']).cumprod()
        final_return = cumulative.iloc[-1] - 1

        # 計算最大回撤
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()

        results.append({
            'final_return': final_return,
            'max_drawdown': max_dd
        })

    return pd.DataFrame(results)
```

### 方法 2：Bootstrap 重抽樣

```python
def monte_carlo_bootstrap(returns: pd.Series, n_simulations: int = 1000):
    """
    Bootstrap 重抽樣

    允許重複抽取，產生不同的報酬序列
    """
    results = []
    n = len(returns)

    for _ in range(n_simulations):
        # 有放回抽樣
        sample = returns.sample(n=n, replace=True)

        # 計算績效
        cumulative = (1 + sample).cumprod()
        final_return = cumulative.iloc[-1] - 1

        results.append(final_return)

    return np.array(results)
```

### 方法 3：參數擾動

```python
def monte_carlo_parameter_perturbation(
    strategy_func,
    data,
    base_params,
    perturbation_range=0.1,
    n_simulations=1000
):
    """
    在最佳參數附近隨機擾動

    評估參數敏感度
    """
    results = []

    for _ in range(n_simulations):
        # 隨機擾動參數
        perturbed_params = {}
        for key, value in base_params.items():
            if isinstance(value, (int, float)):
                delta = value * perturbation_range
                perturbed = value + np.random.uniform(-delta, delta)
                if isinstance(value, int):
                    perturbed = int(round(perturbed))
                perturbed_params[key] = perturbed
            else:
                perturbed_params[key] = value

        # 執行回測
        result = strategy_func(data, perturbed_params)
        results.append({
            'params': perturbed_params,
            'return': result['total_return'],
            'sharpe': result['sharpe_ratio']
        })

    return pd.DataFrame(results)
```

## 結果分析

### 分佈統計

```python
def analyze_monte_carlo_results(simulations):
    """分析 Monte Carlo 結果"""

    returns = simulations['final_return'] if isinstance(simulations, pd.DataFrame) else simulations

    analysis = {
        # 基本統計
        'mean': np.mean(returns),
        'median': np.median(returns),
        'std': np.std(returns),

        # 分位數
        'percentile_1': np.percentile(returns, 1),
        'percentile_5': np.percentile(returns, 5),
        'percentile_25': np.percentile(returns, 25),
        'percentile_75': np.percentile(returns, 75),
        'percentile_95': np.percentile(returns, 95),
        'percentile_99': np.percentile(returns, 99),

        # 風險指標
        'probability_positive': (returns > 0).mean(),
        'probability_loss_10pct': (returns < -0.10).mean(),
        'probability_loss_20pct': (returns < -0.20).mean(),

        # VaR
        'var_95': -np.percentile(returns, 5),
        'var_99': -np.percentile(returns, 1),

        # CVaR (Expected Shortfall)
        'cvar_95': -returns[returns <= np.percentile(returns, 5)].mean()
    }

    return analysis
```

### 視覺化

```python
import matplotlib.pyplot as plt

def plot_monte_carlo_distribution(simulations, original_return=None):
    """繪製 Monte Carlo 結果分佈"""

    returns = simulations['final_return'] if isinstance(simulations, pd.DataFrame) else simulations

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 直方圖
    ax1 = axes[0]
    ax1.hist(returns, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', label='Break-even')
    ax1.axvline(x=np.percentile(returns, 5), color='orange', linestyle='--',
                label=f'5th percentile: {np.percentile(returns, 5):.2%}')
    if original_return:
        ax1.axvline(x=original_return, color='green', linestyle='-',
                    label=f'Original: {original_return:.2%}')
    ax1.set_xlabel('Return')
    ax1.set_ylabel('Density')
    ax1.set_title('Monte Carlo Return Distribution')
    ax1.legend()

    # CDF
    ax2 = axes[1]
    sorted_returns = np.sort(returns)
    cdf = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
    ax2.plot(sorted_returns, cdf)
    ax2.axhline(y=0.05, color='orange', linestyle='--', label='5th percentile')
    ax2.axhline(y=0.50, color='blue', linestyle='--', label='Median')
    ax2.axhline(y=0.95, color='green', linestyle='--', label='95th percentile')
    ax2.axvline(x=0, color='red', linestyle='--')
    ax2.set_xlabel('Return')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Cumulative Distribution Function')
    ax2.legend()

    plt.tight_layout()
    plt.show()
```

## 驗證標準

| 指標 | 通過條件 | 說明 |
|------|----------|------|
| 5th percentile | > 0 | 95% 情境獲利 |
| 1st percentile | > -30% | 極端情境可控 |
| Median | > Original × 50% | 不是純運氣 |
| Prob. positive | > 80% | 大多數獲利 |

## 實務應用

### 策略評估流程

```python
def validate_with_monte_carlo(trades, n_simulations=1000):
    """使用 Monte Carlo 驗證策略"""

    # 執行模擬
    results = monte_carlo_trade_shuffle(trades, n_simulations)

    # 分析結果
    analysis = analyze_monte_carlo_results(results)

    # 判斷
    checks = {
        'p95_profitable': analysis['percentile_5'] > 0,
        'worst_case_acceptable': analysis['percentile_1'] > -0.30,
        'median_reasonable': analysis['median'] > 0,
        'high_prob_positive': analysis['probability_positive'] > 0.8
    }

    passed = all(checks.values())

    return {
        'passed': passed,
        'checks': checks,
        'analysis': analysis,
        'simulations': results
    }
```

### 報告範例

```
Monte Carlo Simulation Report (n=1000)
========================================

Distribution Statistics:
  Mean Return:     45.2%
  Median Return:   42.8%
  Std Dev:         18.5%

Risk Metrics:
  5th Percentile:  12.3%  ✓ (> 0%)
  1st Percentile:  -8.5%  ✓ (> -30%)
  VaR 95:          12.3%
  CVaR 95:         15.8%

Probability Analysis:
  P(Return > 0):   92.3%  ✓
  P(Loss > 10%):   3.2%
  P(Loss > 20%):   0.8%

Validation: PASSED
```

## 注意事項

1. **模擬次數**：至少 1000 次，重要決策 10000 次
2. **交易次數**：原始交易 < 30 筆時結果不可靠
3. **序列相關**：如有趨勢依賴，shuffle 可能不適用
4. **解釋謹慎**：模擬無法預測黑天鵝事件
