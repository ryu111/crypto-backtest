# 統計檢驗

回測結果的統計顯著性驗證方法。

## 為什麼需要統計檢驗？

```
問題：Sharpe = 1.5 是真實技能還是運氣？

統計檢驗回答：
- 這個結果有多大可能是隨機產生的？
- 我們對這個結果有多少信心？
```

## t-test 報酬顯著性

### 單樣本 t-test

**假設檢驗：**
```
H0（虛無假設）：平均報酬 = 0（無優勢）
H1（對立假設）：平均報酬 ≠ 0（有優勢）
```

### 公式

```
t = (x̄ - μ₀) / (s / √n)

其中：
- x̄ = 樣本平均報酬
- μ₀ = 假設平均（通常為 0）
- s = 樣本標準差
- n = 樣本數（交易次數）
```

### Python 實作

```python
import numpy as np
from scipy import stats
from typing import Dict

def return_significance_test(
    returns: np.ndarray,
    null_mean: float = 0,
    alpha: float = 0.05
) -> Dict:
    """
    檢驗報酬是否顯著異於零

    Args:
        returns: 報酬序列
        null_mean: 虛無假設的平均值
        alpha: 顯著水準
    """
    n = len(returns)
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    # t 統計量
    t_stat = (mean_return - null_mean) / (std_return / np.sqrt(n))

    # p-value（雙尾檢驗）
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))

    # 信賴區間
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    margin = t_critical * std_return / np.sqrt(n)
    ci_lower = mean_return - margin
    ci_upper = mean_return + margin

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'mean_return': mean_return,
        'confidence_interval': (ci_lower, ci_upper),
        'n_samples': n
    }
```

### t-statistic 解讀

| t 值 | p-value (約) | 解讀 |
|------|--------------|------|
| > 2.0 | < 0.05 | 顯著（95% 信心） |
| > 2.6 | < 0.01 | 高度顯著（99% 信心） |
| > 3.3 | < 0.001 | 極顯著（99.9% 信心） |
| < 2.0 | > 0.05 | 不顯著 |

## Sharpe Ratio 統計推論

### Sharpe Ratio 的標準誤差

```python
def sharpe_standard_error(
    returns: np.ndarray,
    annualization: float = np.sqrt(252)
) -> Dict:
    """
    計算 Sharpe Ratio 的標準誤差

    考慮報酬的偏態和峰態
    """
    n = len(returns)
    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1)

    # 基本 Sharpe
    sharpe = mean_r / std_r * annualization

    # 偏態和峰態
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns, fisher=True)  # excess kurtosis

    # Sharpe 標準誤差（考慮非常態）
    # 參考: Lo (2002), Bailey & Lopez de Prado (2012)
    se_sharpe = np.sqrt(
        (1 + 0.5 * sharpe**2 - skew * sharpe +
         (kurt / 4) * sharpe**2) / n
    ) * annualization

    return {
        'sharpe': sharpe,
        'standard_error': se_sharpe,
        'skewness': skew,
        'kurtosis': kurt,
        'n_samples': n
    }
```

### Sharpe 信賴區間

```python
def sharpe_confidence_interval(
    returns: np.ndarray,
    alpha: float = 0.05,
    annualization: float = np.sqrt(252)
) -> Dict:
    """
    計算 Sharpe Ratio 的信賴區間
    """
    result = sharpe_standard_error(returns, annualization)
    sharpe = result['sharpe']
    se = result['standard_error']

    z_critical = stats.norm.ppf(1 - alpha/2)
    ci_lower = sharpe - z_critical * se
    ci_upper = sharpe + z_critical * se

    # 檢驗 H0: Sharpe = 0
    z_stat = sharpe / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return {
        'sharpe': sharpe,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'z_statistic': z_stat,
        'p_value': p_value,
        'significant': p_value < alpha
    }
```

### HAC 標準誤差

**HAC = Heteroskedasticity and Autocorrelation Consistent**

處理報酬序列的自相關問題：

```python
def sharpe_hac_se(
    returns: np.ndarray,
    max_lag: int = None
) -> float:
    """
    HAC 標準誤差（Newey-West 估計）
    """
    n = len(returns)
    if max_lag is None:
        max_lag = int(np.floor(4 * (n/100)**(2/9)))

    mean_r = np.mean(returns)
    resid = returns - mean_r

    # Newey-West
    gamma_0 = np.sum(resid**2) / n

    weighted_sum = 0
    for lag in range(1, max_lag + 1):
        weight = 1 - lag / (max_lag + 1)  # Bartlett kernel
        gamma_lag = np.sum(resid[lag:] * resid[:-lag]) / n
        weighted_sum += 2 * weight * gamma_lag

    variance = gamma_0 + weighted_sum
    se = np.sqrt(variance / n)

    return se
```

## Bootstrap 方法

### 為什麼用 Bootstrap？

| 優點 | 說明 |
|------|------|
| 不假設分佈 | 不需要常態假設 |
| 處理複雜統計量 | 適用於 Sharpe、Sortino 等 |
| 提供完整分佈 | 不只點估計 |

### 基本 Bootstrap

```python
def bootstrap_sharpe(
    returns: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05
) -> Dict:
    """
    Bootstrap Sharpe Ratio 信賴區間
    """
    n = len(returns)
    sharpe_samples = []

    for _ in range(n_bootstrap):
        # 重抽樣
        sample = np.random.choice(returns, size=n, replace=True)
        sharpe = np.mean(sample) / np.std(sample, ddof=1) * np.sqrt(252)
        sharpe_samples.append(sharpe)

    sharpe_samples = np.array(sharpe_samples)

    # 原始 Sharpe
    original_sharpe = np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252)

    # 百分位數信賴區間
    ci_lower = np.percentile(sharpe_samples, alpha/2 * 100)
    ci_upper = np.percentile(sharpe_samples, (1 - alpha/2) * 100)

    # Bootstrap p-value
    p_value = np.mean(sharpe_samples <= 0) * 2
    p_value = min(p_value, 2 - p_value)

    return {
        'sharpe': original_sharpe,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_mean': np.mean(sharpe_samples),
        'bootstrap_std': np.std(sharpe_samples),
        'p_value': p_value,
        'n_bootstrap': n_bootstrap
    }
```

### BCa Bootstrap (Bias-Corrected and Accelerated)

```python
def bootstrap_bca(
    returns: np.ndarray,
    statistic_func,
    n_bootstrap: int = 10000,
    alpha: float = 0.05
) -> Dict:
    """
    BCa Bootstrap 信賴區間

    更準確的信賴區間，校正偏差和加速因子
    """
    n = len(returns)
    original_stat = statistic_func(returns)

    # Bootstrap 分佈
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(returns, size=n, replace=True)
        boot_stats.append(statistic_func(sample))
    boot_stats = np.array(boot_stats)

    # 偏差校正因子 z0
    prop_less = np.mean(boot_stats < original_stat)
    z0 = stats.norm.ppf(prop_less) if 0 < prop_less < 1 else 0

    # 加速因子 a (使用 jackknife)
    jackknife_stats = []
    for i in range(n):
        jack_sample = np.delete(returns, i)
        jackknife_stats.append(statistic_func(jack_sample))
    jackknife_stats = np.array(jackknife_stats)

    jack_mean = np.mean(jackknife_stats)
    numerator = np.sum((jack_mean - jackknife_stats)**3)
    denominator = 6 * (np.sum((jack_mean - jackknife_stats)**2))**1.5
    a = numerator / denominator if denominator != 0 else 0

    # 調整後的百分位數
    z_alpha_lower = stats.norm.ppf(alpha/2)
    z_alpha_upper = stats.norm.ppf(1 - alpha/2)

    def adjusted_percentile(z_alpha):
        numerator = z0 + z_alpha
        adjusted = z0 + numerator / (1 - a * numerator)
        return stats.norm.cdf(adjusted) * 100

    ci_lower = np.percentile(boot_stats, adjusted_percentile(z_alpha_lower))
    ci_upper = np.percentile(boot_stats, adjusted_percentile(z_alpha_upper))

    return {
        'statistic': original_stat,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bias_correction': z0,
        'acceleration': a
    }
```

### Block Bootstrap（時間序列）

```python
def block_bootstrap_sharpe(
    returns: np.ndarray,
    block_size: int = 20,
    n_bootstrap: int = 10000,
    alpha: float = 0.05
) -> Dict:
    """
    Block Bootstrap：保留時間序列相關性

    Args:
        block_size: 區塊大小（建議 20-50）
    """
    n = len(returns)
    n_blocks = int(np.ceil(n / block_size))
    sharpe_samples = []

    for _ in range(n_bootstrap):
        # 隨機選擇區塊起點
        starts = np.random.randint(0, n - block_size + 1, size=n_blocks)

        # 組合區塊
        sample = []
        for start in starts:
            sample.extend(returns[start:start + block_size])
        sample = np.array(sample[:n])  # 截斷到原始長度

        sharpe = np.mean(sample) / np.std(sample, ddof=1) * np.sqrt(252)
        sharpe_samples.append(sharpe)

    sharpe_samples = np.array(sharpe_samples)
    original_sharpe = np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252)

    return {
        'sharpe': original_sharpe,
        'ci_lower': np.percentile(sharpe_samples, alpha/2 * 100),
        'ci_upper': np.percentile(sharpe_samples, (1 - alpha/2) * 100),
        'block_size': block_size
    }
```

## 多重測試校正

### 為什麼需要校正？

```
問題：測試 100 個策略，即使全部無效，
     也預期有 5 個顯示 p < 0.05（假陽性）

解決：調整顯著水準或 p-value
```

### Bonferroni 校正

```python
def bonferroni_correction(p_values: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Bonferroni 校正：最保守

    調整後 alpha = alpha / n_tests
    """
    n_tests = len(p_values)
    adjusted_alpha = alpha / n_tests

    significant = p_values < adjusted_alpha

    return {
        'original_alpha': alpha,
        'adjusted_alpha': adjusted_alpha,
        'significant': significant,
        'n_significant': np.sum(significant)
    }
```

### Holm 校正

```python
def holm_correction(p_values: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Holm 校正：比 Bonferroni 更有力

    Step-down 方法
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    significant = np.zeros(n, dtype=bool)

    for i, p in enumerate(sorted_p):
        adjusted_alpha = alpha / (n - i)
        if p < adjusted_alpha:
            significant[sorted_indices[i]] = True
        else:
            break  # 停止，後面的都不顯著

    return {
        'significant': significant,
        'n_significant': np.sum(significant)
    }
```

### Benjamini-Hochberg (FDR 控制)

```python
def benjamini_hochberg(p_values: np.ndarray, fdr: float = 0.05) -> Dict:
    """
    Benjamini-Hochberg 程序

    控制 False Discovery Rate
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # 計算 BH 臨界值
    thresholds = np.arange(1, n+1) / n * fdr

    # 找到最大的 k 使得 p(k) <= k/n * FDR
    significant = np.zeros(n, dtype=bool)
    max_k = 0

    for k in range(n):
        if sorted_p[k] <= thresholds[k]:
            max_k = k + 1

    # 前 max_k 個都是顯著的
    for k in range(max_k):
        significant[sorted_indices[k]] = True

    return {
        'significant': significant,
        'n_significant': np.sum(significant),
        'fdr': fdr
    }
```

### Sharpe Haircut 表

| 測試策略數 | 原始 SR = 1.0 | 原始 SR = 2.0 | 原始 SR = 3.0 |
|------------|---------------|---------------|---------------|
| 10 | 0.71 | 0.85 | 0.92 |
| 50 | 0.51 | 0.71 | 0.82 |
| 100 | 0.42 | 0.63 | 0.76 |
| 500 | 0.28 | 0.49 | 0.64 |
| 1000 | 0.22 | 0.43 | 0.58 |

## 分佈檢驗

### 常態性檢驗

```python
def normality_tests(returns: np.ndarray) -> Dict:
    """
    報酬分佈常態性檢驗
    """
    # Shapiro-Wilk（樣本 < 5000）
    if len(returns) < 5000:
        shapiro_stat, shapiro_p = stats.shapiro(returns)
    else:
        shapiro_stat, shapiro_p = None, None

    # Jarque-Bera
    jb_stat, jb_p = stats.jarque_bera(returns)

    # D'Agostino-Pearson
    dp_stat, dp_p = stats.normaltest(returns)

    # 偏態和峰態
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns, fisher=True)

    return {
        'shapiro': {'statistic': shapiro_stat, 'p_value': shapiro_p},
        'jarque_bera': {'statistic': jb_stat, 'p_value': jb_p},
        'dagostino': {'statistic': dp_stat, 'p_value': dp_p},
        'skewness': skewness,
        'kurtosis': kurtosis,
        'is_normal': jb_p > 0.05 if jb_p else None,
        'skew_ok': abs(skewness) < 2,
        'kurt_ok': abs(kurtosis) < 7
    }
```

### 穩定性檢驗

```python
def stationarity_test(returns: np.ndarray) -> Dict:
    """
    平穩性檢驗（ADF test）
    """
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(returns, autolag='AIC')

    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'used_lag': result[2],
        'n_obs': result[3],
        'critical_values': result[4],
        'is_stationary': result[1] < 0.05
    }
```

## 綜合統計驗證

```python
def comprehensive_statistical_validation(
    returns: np.ndarray,
    n_strategies_tested: int = 1,
    alpha: float = 0.05
) -> Dict:
    """
    綜合統計驗證報告
    """
    results = {}

    # 1. 基本 t-test
    results['t_test'] = return_significance_test(returns, alpha=alpha)

    # 2. Sharpe 信賴區間
    results['sharpe_ci'] = sharpe_confidence_interval(returns, alpha=alpha)

    # 3. Bootstrap
    results['bootstrap'] = bootstrap_sharpe(returns, alpha=alpha)

    # 4. 常態性
    results['normality'] = normality_tests(returns)

    # 5. 多重測試校正（如果測試多個策略）
    if n_strategies_tested > 1:
        # 計算 haircut
        expected_max = np.sqrt(2 * np.log(n_strategies_tested))
        haircut = results['sharpe_ci']['sharpe'] / expected_max
        results['haircut'] = {
            'n_strategies': n_strategies_tested,
            'expected_max_sharpe': expected_max,
            'haircut_factor': min(1, haircut),
            'adjusted_sharpe': results['sharpe_ci']['sharpe'] * min(1, haircut)
        }

    # 6. 綜合判斷
    passed = []
    failed = []

    if results['t_test']['significant']:
        passed.append('t-test')
    else:
        failed.append('t-test')

    if results['sharpe_ci']['ci_lower'] > 0:
        passed.append('sharpe_ci')
    else:
        failed.append('sharpe_ci')

    if results['bootstrap']['ci_lower'] > 0:
        passed.append('bootstrap')
    else:
        failed.append('bootstrap')

    results['summary'] = {
        'passed': passed,
        'failed': failed,
        'all_passed': len(failed) == 0,
        'confidence_level': 1 - alpha
    }

    return results
```

## 解讀指南

### 通過標準

| 檢驗 | 通過條件 | 解讀 |
|------|----------|------|
| t-test | p < 0.05 | 報酬顯著異於零 |
| Sharpe CI | 下界 > 0 | 95% 信心 Sharpe 為正 |
| Bootstrap CI | 下界 > 0 | 不依賴常態假設 |
| 常態性 | \|skew\| < 2 | 分佈不過度偏斜 |

### 警示訊號

| 訊號 | 風險 | 建議 |
|------|------|------|
| Sharpe CI 包含 0 | 可能是運氣 | 增加樣本或重新評估 |
| 高峰態 (kurt > 7) | 尾部風險高 | 使用 bootstrap |
| 多重測試未校正 | 假陽性風險 | 應用 haircut |

## 參考資料

- [Two Sigma: Sharpe Ratio Estimation](https://www.twosigma.com/wp-content/uploads/sharpe-tr-1.pdf)
- [PyBroker Bootstrap Metrics](https://www.pybroker.com/en/latest/notebooks/3.%20Evaluating%20with%20Bootstrap%20Metrics.html)
- [Harvey & Liu: Backtesting](https://people.duke.edu/~charvey/Research/Published_Papers/P120_Backtesting.PDF)
- [Lo (2002): Statistics of Sharpe Ratios](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=377260)
