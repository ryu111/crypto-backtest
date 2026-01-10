---
name: validation
description: 策略驗證系統。確認策略有效性的完整驗證流程。當需要驗證策略是否可靠、檢查統計顯著性、進行壓力測試時使用。
---

# 策略驗證

確認策略有效性的 5 階段驗證流程。

## 驗證流程總覽

```
階段 1：基礎回測
    ↓ 通過？
階段 2：統計檢驗
    ↓ 通過？
階段 3：穩健性測試
    ↓ 通過？
階段 4：Walk-Forward
    ↓ 通過？
階段 5：Monte Carlo
    ↓ 通過？
最終評級 → A/B/C/D
```

## 階段 1：基礎回測

### 通過門檻

| 指標 | 門檻 | 說明 |
|------|------|------|
| Total Return | > 0 | 必須獲利 |
| Total Trades | >= 30 | 統計有效 |
| Sharpe Ratio | > 0.5 | 風險調整報酬 |
| Max Drawdown | < 30% | 可接受回撤 |
| Profit Factor | > 1.0 | 獲利/虧損 > 1 |

### 檢查程式碼

```python
def stage1_basic_backtest(result):
    """階段 1：基礎回測檢查"""
    checks = {
        'total_return_positive': result['total_return'] > 0,
        'enough_trades': result['total_trades'] >= 30,
        'sharpe_acceptable': result['sharpe_ratio'] > 0.5,
        'drawdown_acceptable': result['max_drawdown'] < 0.30,
        'profit_factor_positive': result['profit_factor'] > 1.0
    }

    passed = all(checks.values())
    return passed, checks
```

## 階段 2：統計檢驗

### 通過門檻

| 檢驗 | 方法 | 通過條件 |
|------|------|----------|
| 報酬顯著性 | t-test | p < 0.05 |
| Sharpe 信賴區間 | Bootstrap | 95% CI 不含 0 |
| 獲利分佈 | 正態檢驗 | 無極端偏態 |

### 檢查程式碼

```python
from scipy import stats
import numpy as np

def stage2_statistical_tests(returns):
    """階段 2：統計檢驗"""
    checks = {}

    # t-test: 平均報酬 > 0
    t_stat, p_value = stats.ttest_1samp(returns, 0)
    checks['t_test_significant'] = p_value < 0.05 and t_stat > 0

    # Sharpe Ratio 信賴區間
    sharpe_samples = []
    for _ in range(1000):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        sharpe_samples.append(sample.mean() / sample.std() * np.sqrt(252))

    ci_lower = np.percentile(sharpe_samples, 2.5)
    ci_upper = np.percentile(sharpe_samples, 97.5)
    checks['sharpe_ci_positive'] = ci_lower > 0

    # 偏態檢查
    skewness = stats.skew(returns)
    checks['skewness_acceptable'] = abs(skewness) < 2

    passed = all(checks.values())
    return passed, checks, {
        'p_value': p_value,
        'sharpe_ci': (ci_lower, ci_upper),
        'skewness': skewness
    }
```

## 階段 3：穩健性測試

### 通過門檻

| 測試 | 方法 | 通過條件 |
|------|------|----------|
| 參數敏感度 | 鄰近參數測試 | 變異 < 30% |
| 時間穩定性 | 分段測試 | 各段一致獲利 |
| 標的一致性 | BTC/ETH 測試 | 兩者皆獲利 |

### 檢查程式碼

```python
def stage3_robustness_tests(strategy, data_btc, data_eth, params):
    """階段 3：穩健性測試"""
    checks = {}

    # 參數敏感度
    sensitivity = calculate_parameter_sensitivity(strategy, params)
    checks['sensitivity_acceptable'] = sensitivity < 0.30

    # 時間穩定性（分前後半）
    mid_point = len(data_btc) // 2
    first_half = run_backtest(strategy, data_btc[:mid_point], params)
    second_half = run_backtest(strategy, data_btc[mid_point:], params)
    checks['time_consistency'] = (
        first_half['total_return'] > 0 and
        second_half['total_return'] > 0
    )

    # 標的一致性
    btc_result = run_backtest(strategy, data_btc, params)
    eth_result = run_backtest(strategy, data_eth, params)
    checks['asset_consistency'] = (
        btc_result['total_return'] > 0 and
        eth_result['total_return'] > 0
    )

    passed = all(checks.values())
    return passed, checks
```

## 階段 4：Walk-Forward Analysis

### 通過門檻

| 指標 | 門檻 | 說明 |
|------|------|------|
| WFA Efficiency | >= 50% | OOS/IS 報酬比 |
| OOS Win Rate | > 50% | 窗口勝率 |
| 無重大虧損 | 單窗口 < -10% | 控制風險 |

### 檢查程式碼

```python
def stage4_walk_forward(strategy, data, params, n_windows=5):
    """階段 4：Walk-Forward 分析"""
    wfa_results, efficiency = walk_forward_analysis(
        data, strategy, params, n_windows=n_windows
    )

    checks = {}

    # 效率門檻
    checks['efficiency_acceptable'] = efficiency >= 0.50

    # OOS 勝率
    oos_returns = [r['oos_return'] for r in wfa_results]
    oos_win_rate = sum(1 for r in oos_returns if r > 0) / len(oos_returns)
    checks['oos_win_rate'] = oos_win_rate > 0.50

    # 無重大單窗口虧損
    checks['no_major_loss'] = all(r > -0.10 for r in oos_returns)

    passed = all(checks.values())
    return passed, checks, {
        'efficiency': efficiency,
        'oos_returns': oos_returns,
        'oos_win_rate': oos_win_rate
    }
```

## 階段 5：Monte Carlo 模擬

### 通過門檻

| 指標 | 門檻 | 說明 |
|------|------|------|
| 95% 情境獲利 | 5th percentile > 0 | 大多數情境獲利 |
| 最差情境 | 1st percentile > -30% | 極端情境可控 |
| 中位數報酬 | > 原始報酬 50% | 不是運氣 |

### 檢查程式碼

```python
def stage5_monte_carlo(trades, n_simulations=1000):
    """階段 5：Monte Carlo 模擬"""

    # 隨機重排交易順序
    simulated_returns = []

    for _ in range(n_simulations):
        # 隨機打亂交易順序
        shuffled = trades.sample(frac=1, replace=False)
        cumulative = (1 + shuffled['return']).cumprod()
        final_return = cumulative.iloc[-1] - 1
        simulated_returns.append(final_return)

    simulated_returns = np.array(simulated_returns)

    checks = {}

    # 95% 情境獲利
    percentile_5 = np.percentile(simulated_returns, 5)
    checks['p95_profitable'] = percentile_5 > 0

    # 最差情境
    percentile_1 = np.percentile(simulated_returns, 1)
    checks['worst_case_acceptable'] = percentile_1 > -0.30

    # 中位數 vs 原始
    median = np.percentile(simulated_returns, 50)
    original_return = (1 + trades['return']).prod() - 1
    checks['median_reasonable'] = median > original_return * 0.5

    passed = all(checks.values())
    return passed, checks, {
        'percentile_1': percentile_1,
        'percentile_5': percentile_5,
        'median': median,
        'percentile_95': np.percentile(simulated_returns, 95)
    }
```

## 最終評級

| 等級 | 通過階段 | 建議 |
|------|----------|------|
| A | 全部 5 階段 | 可實盤 |
| B | 4 階段 | 小資金試運行 |
| C | 3 階段 | 繼續優化 |
| D | < 3 階段 | 重新設計 |

### 評級程式碼

```python
def final_grading(stage_results):
    """計算最終評級"""
    passed_count = sum(1 for s in stage_results.values() if s['passed'])

    if passed_count == 5:
        grade = 'A'
        recommendation = '可實盤，建議先小資金試運行 2-4 週'
    elif passed_count == 4:
        grade = 'B'
        recommendation = '小資金試運行，持續監控'
    elif passed_count == 3:
        grade = 'C'
        recommendation = '繼續優化，重點改善未通過階段'
    else:
        grade = 'D'
        recommendation = '重新設計策略'

    return {
        'grade': grade,
        'passed_stages': passed_count,
        'recommendation': recommendation,
        'details': stage_results
    }
```

## 驗證檢查清單

```markdown
## 策略驗證檢查清單

### 策略名稱：________________
### 日期：________________

### 階段 1：基礎回測
- [ ] 總報酬 > 0
- [ ] 交易次數 >= 30
- [ ] Sharpe > 0.5
- [ ] 最大回撤 < 30%
- [ ] Profit Factor > 1.0

### 階段 2：統計檢驗
- [ ] t-test p < 0.05
- [ ] Sharpe 95% CI > 0
- [ ] 偏態 |skew| < 2

### 階段 3：穩健性測試
- [ ] 參數敏感度 < 30%
- [ ] 前後半期一致
- [ ] BTC/ETH 皆獲利

### 階段 4：Walk-Forward
- [ ] WFA Efficiency >= 50%
- [ ] OOS 勝率 > 50%
- [ ] 無單窗口 > -10%

### 階段 5：Monte Carlo
- [ ] 5th percentile > 0
- [ ] 1st percentile > -30%
- [ ] Median > Original × 50%

### 最終評級：[ A / B / C / D ]

### 改善建議：
1. ________________
2. ________________
```

For 統計檢驗詳解 → read `references/statistical-tests.md`
For Monte Carlo 詳解 → read `references/monte-carlo.md`
