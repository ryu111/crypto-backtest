---
name: optimization
description: 參數優化系統。網格搜尋、貝葉斯優化、Walk-Forward 分析。當需要尋找最佳參數、避免過擬合、驗證參數穩健性時使用。
---

# 參數優化

策略參數的系統化優化和穩健性驗證。

## 優化方法比較

| 方法 | 速度 | 適用場景 | 過擬合風險 | 推薦度 |
|------|------|----------|------------|--------|
| Grid Search | 慢 | 參數少、離散 | 高 | ⭐⭐ |
| Random Search | 中 | 參數多 | 中 | ⭐⭐⭐ |
| Bayesian | 快 | 參數多、連續 | 低 | ⭐⭐⭐⭐⭐ |
| Genetic | 中 | 非凸優化 | 中 | ⭐⭐⭐ |

## VectorBT 參數優化

### Grid Search

```python
import vectorbtpro as vbt
import numpy as np

# 定義參數範圍
fast_windows = np.arange(5, 25, 2)   # 5, 7, 9, ..., 23
slow_windows = np.arange(20, 60, 5)  # 20, 25, 30, ..., 55

# 產生所有 MA 組合
fast_ma, slow_ma = vbt.MA.run_combs(
    close,
    window=np.concatenate([fast_windows, slow_windows]),
    r=2,  # 兩兩組合
    short_names=['fast', 'slow']
)

# 產生訊號
entries = fast_ma.ma_crossed_above(slow_ma.ma)
exits = fast_ma.ma_crossed_below(slow_ma.ma)

# 批量回測
pf = vbt.Portfolio.from_signals(
    close,
    entries,
    exits,
    fees=0.0006,
    freq='4h'
)

# 找最佳參數
sharpe = pf.sharpe_ratio()
best_idx = sharpe.idxmax()
print(f"最佳參數: {best_idx}, Sharpe: {sharpe[best_idx]:.2f}")

# 視覺化
sharpe.vbt.heatmap(
    x_level='fast_window',
    y_level='slow_window'
).show()
```

### Bayesian 優化（推薦）

```python
import optuna

def objective(trial):
    # 定義參數空間
    fast_period = trial.suggest_int('fast_period', 5, 20)
    slow_period = trial.suggest_int('slow_period', 20, 50)
    stop_loss = trial.suggest_float('stop_loss', 0.01, 0.05)

    # 確保 fast < slow
    if fast_period >= slow_period:
        return -999

    # 執行回測
    result = run_backtest(
        fast_period=fast_period,
        slow_period=slow_period,
        stop_loss=stop_loss
    )

    # 優化目標：Sharpe Ratio
    return result['sharpe_ratio']

# 執行優化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# 最佳參數
print(f"最佳參數: {study.best_params}")
print(f"最佳 Sharpe: {study.best_value:.2f}")

# 視覺化
optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()
```

## Walk-Forward Analysis

### 核心概念

將資料分成多個時間窗口，在 In-Sample (IS) 優化，在 Out-of-Sample (OOS) 驗證。

```
|---IS---|--OOS--|  Window 1
     |---IS---|--OOS--|  Window 2
          |---IS---|--OOS--|  Window 3
               |---IS---|--OOS--|  Window 4
```

### 實作

```python
def walk_forward_analysis(data, strategy_func, is_ratio=0.7, n_windows=5):
    """
    Walk-Forward Analysis

    Args:
        data: 完整資料
        strategy_func: 策略函數 (params) -> result
        is_ratio: In-Sample 比例
        n_windows: 窗口數量

    Returns:
        wfa_results: 各窗口結果
        efficiency: WFA 效率
    """
    total_len = len(data)
    window_size = total_len // n_windows
    is_size = int(window_size * is_ratio)
    oos_size = window_size - is_size

    results = []

    for i in range(n_windows):
        start_idx = i * window_size
        is_end = start_idx + is_size
        oos_end = start_idx + window_size

        is_data = data.iloc[start_idx:is_end]
        oos_data = data.iloc[is_end:oos_end]

        # 在 IS 優化參數
        best_params = optimize_on_data(is_data, strategy_func)

        # 在 IS 計算績效
        is_result = strategy_func(is_data, best_params)

        # 在 OOS 測試
        oos_result = strategy_func(oos_data, best_params)

        results.append({
            'window': i + 1,
            'params': best_params,
            'is_return': is_result['total_return'],
            'oos_return': oos_result['total_return'],
            'is_sharpe': is_result['sharpe_ratio'],
            'oos_sharpe': oos_result['sharpe_ratio']
        })

    # 計算 WFA 效率
    is_returns = [r['is_return'] for r in results]
    oos_returns = [r['oos_return'] for r in results]

    efficiency = np.mean(oos_returns) / np.mean(is_returns) if np.mean(is_returns) > 0 else 0

    return results, efficiency
```

### 效率判斷

| WFA Efficiency | 判斷 | 建議 |
|----------------|------|------|
| > 80% | 優秀 | 可信賴 |
| 50-80% | 良好 | 可接受 |
| 30-50% | 一般 | 需謹慎 |
| < 30% | 差 | 可能過擬合 |

## 過擬合偵測

### 指標

| 指標 | 計算 | 警戒值 |
|------|------|--------|
| IS/OOS 比 | Return_IS / Return_OOS | > 2.0 |
| 參數敏感度 | 鄰近參數績效變異 | > 30% |
| 交易次數 | 總交易數 | < 30 |
| PBO | Probability of Backtest Overfitting | > 50% |

### 參數敏感度分析

```python
def parameter_sensitivity(sharpe_matrix, threshold=0.3):
    """
    分析參數敏感度

    Args:
        sharpe_matrix: 參數組合的 Sharpe 矩陣
        threshold: 敏感度閾值

    Returns:
        sensitivity_score: 敏感度分數
        is_sensitive: 是否過度敏感
    """
    # 計算相鄰參數的變異
    diff_x = np.abs(np.diff(sharpe_matrix, axis=0))
    diff_y = np.abs(np.diff(sharpe_matrix, axis=1))

    # 相對變異
    mean_sharpe = np.mean(sharpe_matrix)
    rel_diff_x = diff_x / mean_sharpe if mean_sharpe > 0 else diff_x
    rel_diff_y = diff_y / mean_sharpe if mean_sharpe > 0 else diff_y

    sensitivity = np.mean([rel_diff_x.mean(), rel_diff_y.mean()])

    return sensitivity, sensitivity > threshold
```

### 過擬合概率 (PBO)

```python
def probability_of_backtest_overfitting(is_results, oos_results):
    """
    計算過擬合概率

    使用 CSCV (Combinatorially Symmetric Cross-Validation)
    """
    n = len(is_results)

    # 計算 IS 和 OOS 的排名相關性
    is_ranks = rankdata(is_results)
    oos_ranks = rankdata(oos_results)

    # 計算相關係數
    correlation = np.corrcoef(is_ranks, oos_ranks)[0, 1]

    # PBO = (1 - correlation) / 2
    pbo = (1 - correlation) / 2

    return pbo
```

## 優化最佳實踐

### 參數選擇

| 原則 | 說明 |
|------|------|
| 參數數量 | 限制 3-5 個 |
| 參數範圍 | 基於市場邏輯 |
| 參數間距 | 合理的粒度 |
| 避免極端 | 不選邊界值 |

### 優化流程

```
1. 定義參數空間
   └→ 基於策略邏輯設定範圍

2. 初步網格搜尋
   └→ 粗略找到有效區域

3. 貝葉斯精細優化
   └→ 在有效區域細化

4. Walk-Forward 驗證
   └→ 確認穩健性

5. 敏感度分析
   └→ 檢查是否過擬合

6. 選擇穩健參數
   └→ 不一定選最佳，選穩定
```

### 避免過擬合

1. **充足樣本**：交易次數 >= 30
2. **驗證分離**：使用 OOS 測試
3. **多時段測試**：不同市場狀態
4. **參數簡化**：減少自由度
5. **交叉驗證**：多次 WFA

## 優化報告模板

```python
def generate_optimization_report(study, wfa_results, sensitivity):
    report = {
        'best_params': study.best_params,
        'best_sharpe': study.best_value,
        'n_trials': len(study.trials),

        'walk_forward': {
            'efficiency': wfa_results['efficiency'],
            'is_mean_return': np.mean([r['is_return'] for r in wfa_results]),
            'oos_mean_return': np.mean([r['oos_return'] for r in wfa_results]),
        },

        'sensitivity': {
            'score': sensitivity['score'],
            'is_sensitive': sensitivity['is_sensitive']
        },

        'recommendation': get_recommendation(wfa_results, sensitivity)
    }

    return report
```

For Walk-Forward 詳解 → read `references/walk-forward.md`
For 過擬合偵測 → read `references/overfitting-detection.md`
