# 過擬合偵測

回測策略的過擬合風險評估方法。

## 什麼是過擬合？

```
過擬合 = 策略只適應歷史數據的特定模式，無法泛化到未來

典型症狀：
- 回測績效極佳（Sharpe > 3）
- 實盤表現遠差於回測
- 參數微調導致績效大幅波動
- 交易次數過少
```

## PBO (Probability of Backtest Overfitting)

### 核心概念

PBO 量化過擬合風險：比較 In-Sample 最佳參數在 Out-of-Sample 的表現排名。

```
如果 IS 最佳參數在 OOS 也是最佳 → 低 PBO（好）
如果 IS 最佳參數在 OOS 排名很差 → 高 PBO（壞）
```

### CSCV 方法

**Combinatorially Symmetric Cross-Validation**

```
將數據分成 S 個等分
產生所有可能的 S/2 組合作為 IS 和 OOS

例如 S=4:
組合1: IS=[1,2], OOS=[3,4]
組合2: IS=[1,3], OOS=[2,4]
組合3: IS=[1,4], OOS=[2,3]
組合4: IS=[2,3], OOS=[1,4]
組合5: IS=[2,4], OOS=[1,3]
組合6: IS=[3,4], OOS=[1,2]

對每個組合：
1. 在 IS 上找最佳參數
2. 記錄該參數在 OOS 的排名
3. 計算排名低於中位數的比例 → PBO
```

### Python 實作

```python
import numpy as np
from itertools import combinations
from scipy.stats import rankdata
from typing import List, Dict, Callable

def calculate_pbo(
    returns_matrix: np.ndarray,  # shape: (n_periods, n_trials)
    n_splits: int = 8,
    metric: str = 'sharpe'
) -> Dict:
    """
    計算 PBO

    Args:
        returns_matrix: 每個參數組合的報酬序列
        n_splits: 分割數
        metric: 評估指標

    Returns:
        {'pbo': float, 'degradation': float, 'details': list}
    """
    n_periods, n_trials = returns_matrix.shape
    split_size = n_periods // n_splits

    # 產生所有組合
    all_splits = list(range(n_splits))
    is_combos = list(combinations(all_splits, n_splits // 2))

    logits = []

    for is_indices in is_combos:
        oos_indices = [i for i in all_splits if i not in is_indices]

        # 建立 IS 和 OOS 資料
        is_data = np.concatenate([
            returns_matrix[i*split_size:(i+1)*split_size]
            for i in is_indices
        ])

        oos_data = np.concatenate([
            returns_matrix[i*split_size:(i+1)*split_size]
            for i in oos_indices
        ])

        # 計算每個試驗的績效
        is_scores = _calculate_metric(is_data, metric)
        oos_scores = _calculate_metric(oos_data, metric)

        # IS 最佳試驗
        best_trial = np.argmax(is_scores)

        # 該試驗在 OOS 的排名
        oos_ranks = rankdata(-oos_scores)  # 高分低排名
        best_oos_rank = oos_ranks[best_trial]

        # 計算 logit
        w = best_oos_rank / n_trials
        if w > 0 and w < 1:
            logit = np.log(w / (1 - w))
            logits.append(logit)

    # PBO = 排名在中位數以下的比例
    pbo = np.mean([l > 0 for l in logits]) if logits else 0.5

    # 績效衰退
    degradation = _calculate_degradation(returns_matrix, n_splits, metric)

    return {
        'pbo': pbo,
        'degradation': degradation,
        'n_combinations': len(is_combos),
        'logits': logits
    }

def _calculate_metric(returns: np.ndarray, metric: str) -> np.ndarray:
    """計算每個試驗的績效指標"""
    if metric == 'sharpe':
        return np.mean(returns, axis=0) / (np.std(returns, axis=0) + 1e-10) * np.sqrt(252)
    elif metric == 'return':
        return np.sum(returns, axis=0)
    elif metric == 'omega':
        threshold = 0
        gains = np.sum(np.maximum(returns - threshold, 0), axis=0)
        losses = np.sum(np.maximum(threshold - returns, 0), axis=0)
        return gains / (losses + 1e-10)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _calculate_degradation(returns: np.ndarray, n_splits: int, metric: str) -> float:
    """計算 IS 到 OOS 的績效衰退"""
    # 簡化：使用前半和後半
    mid = len(returns) // 2
    is_data = returns[:mid]
    oos_data = returns[mid:]

    is_scores = _calculate_metric(is_data, metric)
    oos_scores = _calculate_metric(oos_data, metric)

    best_is = np.argmax(is_scores)

    is_perf = is_scores[best_is]
    oos_perf = oos_scores[best_is]

    if is_perf > 0:
        return 1 - oos_perf / is_perf
    return 0
```

### PBO 解讀

| PBO 值 | 風險等級 | 建議 |
|--------|----------|------|
| < 25% | 低 | 策略可靠 |
| 25-50% | 中 | 需謹慎 |
| 50-75% | 高 | 很可能過擬合 |
| > 75% | 極高 | 重新設計 |

## Deflated Sharpe Ratio (DSR)

### 問題

多次測試會膨脹 Sharpe Ratio。測試 100 個策略，最佳 Sharpe 可能純粹是運氣。

### DSR 公式

```
DSR = PSR(SR*) × Haircut

其中：
- PSR = Probabilistic Sharpe Ratio
- SR* = 觀察到的 Sharpe
- Haircut = 多重測試校正因子
```

### Sharpe Haircut 表

| 測試次數 | SR = 1.0 | SR = 2.0 | SR = 3.0 |
|----------|----------|----------|----------|
| 10 | 0.71 | 0.85 | 0.92 |
| 50 | 0.51 | 0.71 | 0.82 |
| 100 | 0.42 | 0.63 | 0.76 |
| 500 | 0.28 | 0.49 | 0.64 |

### Python 實作

```python
from scipy import stats
import numpy as np

def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    sample_length: int,
    skewness: float = 0,
    kurtosis: float = 3
) -> Dict:
    """
    計算 Deflated Sharpe Ratio

    Args:
        observed_sharpe: 觀察到的 Sharpe
        n_trials: 測試的參數組合數
        sample_length: 樣本長度（交易次數或時間點）
        skewness: 報酬偏態
        kurtosis: 報酬峰態
    """
    # 計算 PSR (Probabilistic Sharpe Ratio)
    # Sharpe 標準誤差
    sr_std = np.sqrt(
        (1 + 0.5 * observed_sharpe**2 - skewness * observed_sharpe +
         (kurtosis - 3) / 4 * observed_sharpe**2) / sample_length
    )

    # 預期最大 Sharpe（在 n_trials 中）
    expected_max_sharpe = _expected_maximum(n_trials)

    # PSR：觀察 SR 超過基準的機率
    psr = stats.norm.cdf((observed_sharpe - expected_max_sharpe) / sr_std)

    # Haircut
    haircut = observed_sharpe / expected_max_sharpe if expected_max_sharpe > 0 else 1

    # DSR
    dsr = psr * haircut

    return {
        'dsr': dsr,
        'psr': psr,
        'haircut': haircut,
        'sr_std': sr_std,
        'expected_max_sharpe': expected_max_sharpe
    }

def _expected_maximum(n_trials: int) -> float:
    """預期最大值（標準常態分佈）"""
    if n_trials <= 1:
        return 0

    # 使用 Euler-Mascheroni 近似
    gamma = 0.5772156649
    z = stats.norm.ppf(1 - 1/n_trials)

    return (1 - gamma) * stats.norm.ppf(1 - 1/n_trials) + gamma * stats.norm.ppf(1 - 1/(n_trials * np.e))
```

## CPCV vs Walk-Forward

### 比較

| 方法 | PBO 估計 | 穩定性 | 計算成本 |
|------|----------|--------|----------|
| Walk-Forward | 高估 | 低 | 低 |
| CSCV | 準確 | 高 | 中 |
| CPCV | 最準確 | 最高 | 高 |

### CPCV (Combinatorial Purged Cross-Validation)

```python
def cpcv_analysis(
    returns_matrix: np.ndarray,
    n_splits: int = 10,
    purge_ratio: float = 0.01  # 清除比例
) -> Dict:
    """
    CPCV 分析

    Args:
        purge_ratio: 資料點之間的清除比例（避免資訊洩漏）
    """
    n_periods, n_trials = returns_matrix.shape
    split_size = n_periods // n_splits
    purge_size = int(n_periods * purge_ratio)

    results = {
        'is_sharpes': [],
        'oos_sharpes': [],
        'best_trial_oos_ranks': []
    }

    # 產生所有組合
    for k in range(1, n_splits):
        is_combos = combinations(range(n_splits), k)

        for is_indices in is_combos:
            oos_indices = [i for i in range(n_splits) if i not in is_indices]

            # 建立 IS 資料（帶清除）
            is_data = []
            for i in is_indices:
                start = i * split_size + purge_size
                end = (i + 1) * split_size - purge_size
                is_data.append(returns_matrix[start:end])
            is_data = np.concatenate(is_data)

            # 建立 OOS 資料
            oos_data = []
            for i in oos_indices:
                start = i * split_size + purge_size
                end = (i + 1) * split_size - purge_size
                oos_data.append(returns_matrix[start:end])
            oos_data = np.concatenate(oos_data)

            # 計算績效
            is_sharpes = _calculate_metric(is_data, 'sharpe')
            oos_sharpes = _calculate_metric(oos_data, 'sharpe')

            best_trial = np.argmax(is_sharpes)
            oos_rank = rankdata(-oos_sharpes)[best_trial]

            results['is_sharpes'].append(is_sharpes[best_trial])
            results['oos_sharpes'].append(oos_sharpes[best_trial])
            results['best_trial_oos_ranks'].append(oos_rank)

    # 計算 PBO
    pbo = np.mean([r > n_trials/2 for r in results['best_trial_oos_ranks']])

    # 計算衰退
    degradation = 1 - np.mean(results['oos_sharpes']) / np.mean(results['is_sharpes'])

    return {
        'pbo': pbo,
        'degradation': degradation,
        'avg_is_sharpe': np.mean(results['is_sharpes']),
        'avg_oos_sharpe': np.mean(results['oos_sharpes']),
        'n_combinations': len(results['is_sharpes'])
    }
```

## 過擬合指標清單

### 快速檢查

| 指標 | 警戒值 | 說明 |
|------|--------|------|
| IS/OOS 比 | > 2.0 | 績效衰退過大 |
| 參數敏感度 | > 30% | 鄰近參數變異大 |
| 交易次數 | < 30 | 統計無效 |
| Sharpe > 3 | 警戒 | 可能過擬合 |
| PBO | > 50% | 高過擬合風險 |

### 綜合評估

```python
def overfit_risk_assessment(
    is_sharpe: float,
    oos_sharpe: float,
    n_trades: int,
    param_sensitivity: float,
    pbo: float
) -> Dict:
    """
    綜合過擬合風險評估
    """
    risks = []

    # IS/OOS 比
    if is_sharpe > 0 and oos_sharpe > 0:
        is_oos_ratio = is_sharpe / oos_sharpe
        if is_oos_ratio > 2.0:
            risks.append(('is_oos_ratio', 'HIGH', is_oos_ratio))
        elif is_oos_ratio > 1.5:
            risks.append(('is_oos_ratio', 'MEDIUM', is_oos_ratio))
    else:
        risks.append(('is_oos_ratio', 'HIGH', 'OOS negative'))

    # 交易次數
    if n_trades < 30:
        risks.append(('n_trades', 'HIGH', n_trades))
    elif n_trades < 50:
        risks.append(('n_trades', 'MEDIUM', n_trades))

    # 參數敏感度
    if param_sensitivity > 0.30:
        risks.append(('param_sensitivity', 'HIGH', param_sensitivity))
    elif param_sensitivity > 0.20:
        risks.append(('param_sensitivity', 'MEDIUM', param_sensitivity))

    # PBO
    if pbo > 0.50:
        risks.append(('pbo', 'HIGH', pbo))
    elif pbo > 0.25:
        risks.append(('pbo', 'MEDIUM', pbo))

    # 綜合評分
    high_risks = sum(1 for r in risks if r[1] == 'HIGH')
    medium_risks = sum(1 for r in risks if r[1] == 'MEDIUM')

    if high_risks >= 2:
        overall_risk = 'HIGH'
    elif high_risks >= 1 or medium_risks >= 2:
        overall_risk = 'MEDIUM'
    else:
        overall_risk = 'LOW'

    return {
        'overall_risk': overall_risk,
        'details': risks,
        'recommendation': _get_recommendation(overall_risk)
    }

def _get_recommendation(risk_level: str) -> str:
    recommendations = {
        'LOW': '策略可靠，可進行下一步驗證',
        'MEDIUM': '需謹慎，建議增加樣本或簡化參數',
        'HIGH': '高過擬合風險，建議重新設計策略'
    }
    return recommendations.get(risk_level, '')
```

## 最佳實踐

### 避免過擬合

1. **限制參數數量**：<= 5 個可調參數
2. **充足樣本**：交易次數 >= 50
3. **多市場測試**：BTC + ETH 都要獲利
4. **時段穩定性**：前後半都獲利
5. **使用 CPCV**：比 Walk-Forward 更可靠

### 參數選擇原則

```
不要選「最佳」參數，選「穩健」參數

穩健參數 = 鄰近參數也有類似表現的參數
```

## 參考資料

- [Bailey et al. PBO Paper](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
- [PBO by Balaena Quant](https://medium.com/balaena-quant-insights/the-probability-of-backtest-overfitting-pbo-9ba0ac7fb456)
- [R pbo Package](https://cran.r-project.org/web/packages/pbo/vignettes/pbo.html)
- [Harvey & Liu: Backtesting](https://people.duke.edu/~charvey/Research/Published_Papers/P120_Backtesting.PDF)
