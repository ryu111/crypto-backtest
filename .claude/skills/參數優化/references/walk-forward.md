# Walk-Forward Analysis 詳解

## 核心概念

Walk-Forward Analysis (WFA) 是防止過擬合的黃金標準方法。

### 原理

將歷史資料分成多個滾動窗口，每個窗口內：
1. **In-Sample (IS)**：優化參數
2. **Out-of-Sample (OOS)**：測試參數

```
時間 →

|------ IS ------|-- OOS --|  Window 1
         |------ IS ------|-- OOS --|  Window 2
                  |------ IS ------|-- OOS --|  Window 3
```

### 為什麼有效

- 模擬真實交易情境
- 參數必須在未見資料上有效
- 自動偵測過擬合

## 實作

```python
import pandas as pd
import numpy as np
from typing import Callable, Dict, List

def walk_forward_analysis(
    data: pd.DataFrame,
    strategy_func: Callable,
    optimize_func: Callable,
    is_ratio: float = 0.7,
    n_windows: int = 5,
    overlap: float = 0.5
) -> Dict:
    """
    Walk-Forward Analysis

    Args:
        data: 完整資料
        strategy_func: 策略函數 (data, params) -> result
        optimize_func: 優化函數 (data) -> best_params
        is_ratio: In-Sample 比例
        n_windows: 窗口數量
        overlap: 窗口重疊比例

    Returns:
        WFA 結果
    """

    total_len = len(data)
    window_size = int(total_len / (1 + (n_windows - 1) * (1 - overlap)))
    step_size = int(window_size * (1 - overlap))
    is_size = int(window_size * is_ratio)
    oos_size = window_size - is_size

    results = []

    for i in range(n_windows):
        start_idx = i * step_size
        is_end = start_idx + is_size
        oos_end = start_idx + window_size

        if oos_end > total_len:
            break

        is_data = data.iloc[start_idx:is_end]
        oos_data = data.iloc[is_end:oos_end]

        # 在 IS 優化參數
        best_params = optimize_func(is_data)

        # 計算 IS 和 OOS 績效
        is_result = strategy_func(is_data, best_params)
        oos_result = strategy_func(oos_data, best_params)

        results.append({
            'window': i + 1,
            'is_start': is_data.index[0],
            'is_end': is_data.index[-1],
            'oos_start': oos_data.index[0],
            'oos_end': oos_data.index[-1],
            'params': best_params,
            'is_return': is_result['total_return'],
            'oos_return': oos_result['total_return'],
            'is_sharpe': is_result['sharpe_ratio'],
            'oos_sharpe': oos_result['sharpe_ratio'],
            'is_trades': is_result['total_trades'],
            'oos_trades': oos_result['total_trades']
        })

    # 計算整體指標
    is_returns = [r['is_return'] for r in results]
    oos_returns = [r['oos_return'] for r in results]

    # WFA 效率
    is_mean = np.mean(is_returns)
    oos_mean = np.mean(oos_returns)
    efficiency = oos_mean / is_mean if is_mean > 0 else 0

    # OOS 勝率
    oos_win_rate = sum(1 for r in oos_returns if r > 0) / len(oos_returns)

    return {
        'windows': results,
        'is_mean_return': is_mean,
        'oos_mean_return': oos_mean,
        'efficiency': efficiency,
        'oos_win_rate': oos_win_rate,
        'total_oos_return': np.prod([1 + r for r in oos_returns]) - 1
    }
```

## 效率判斷標準

| WFA Efficiency | 判斷 | 建議 |
|----------------|------|------|
| > 80% | 優秀 | 策略穩健，可信賴 |
| 60-80% | 良好 | 可接受，持續監控 |
| 40-60% | 一般 | 需謹慎，可能過擬合 |
| 20-40% | 差 | 高度過擬合嫌疑 |
| < 20% | 極差 | 策略無效 |

## 參數選擇

### 窗口數量

| 資料長度 | 建議窗口數 | 理由 |
|----------|-----------|------|
| 1 年 | 3-4 | 每窗口至少 2-3 個月 |
| 2 年 | 4-6 | 平衡樣本量 |
| 3+ 年 | 5-8 | 更多驗證點 |

### IS/OOS 比例

| 比例 | 優點 | 缺點 |
|------|------|------|
| 80/20 | IS 樣本大 | OOS 太短 |
| 70/30 | 平衡 | 推薦 |
| 60/40 | OOS 充足 | IS 可能不足 |

### 窗口重疊

| 重疊 | 效果 |
|------|------|
| 0% | 獨立窗口，嚴格測試 |
| 25% | 輕微連續性 |
| 50% | 適度連續性，推薦 |

## 視覺化

```python
import matplotlib.pyplot as plt

def plot_wfa_results(wfa_results):
    """繪製 WFA 結果"""

    windows = wfa_results['windows']
    n = len(windows)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. IS vs OOS 報酬
    ax1 = axes[0, 0]
    x = range(1, n + 1)
    ax1.bar([i - 0.2 for i in x], [w['is_return'] for w in windows],
            0.4, label='In-Sample', alpha=0.7)
    ax1.bar([i + 0.2 for i in x], [w['oos_return'] for w in windows],
            0.4, label='Out-of-Sample', alpha=0.7)
    ax1.set_xlabel('Window')
    ax1.set_ylabel('Return')
    ax1.legend()
    ax1.set_title('IS vs OOS Returns by Window')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # 2. 累積 OOS 權益曲線
    ax2 = axes[0, 1]
    oos_returns = [w['oos_return'] for w in windows]
    cumulative = np.cumprod([1 + r for r in oos_returns])
    ax2.plot(x, cumulative, marker='o')
    ax2.set_xlabel('Window')
    ax2.set_ylabel('Cumulative Return')
    ax2.set_title('Cumulative OOS Equity Curve')

    # 3. 效率趨勢
    ax3 = axes[1, 0]
    efficiencies = [w['oos_return'] / w['is_return']
                    if w['is_return'] > 0 else 0 for w in windows]
    ax3.bar(x, efficiencies)
    ax3.axhline(y=0.5, color='red', linestyle='--', label='50% threshold')
    ax3.set_xlabel('Window')
    ax3.set_ylabel('Efficiency (OOS/IS)')
    ax3.set_title('Window Efficiency')
    ax3.legend()

    # 4. 參數穩定性
    ax4 = axes[1, 1]
    # 假設有 'fast_period' 參數
    if 'fast_period' in windows[0]['params']:
        params = [w['params']['fast_period'] for w in windows]
        ax4.plot(x, params, marker='o')
        ax4.set_xlabel('Window')
        ax4.set_ylabel('Fast Period')
        ax4.set_title('Parameter Stability')

    plt.tight_layout()
    plt.show()
```

## 常見問題

### Q: OOS 績效總是比 IS 差？

這是正常的。原因：
- IS 是優化目標
- OOS 測試泛化能力
- 效率 50%+ 已經很好

### Q: 窗口間參數變化大？

可能原因：
- 市場狀態變化
- 參數敏感度高
- 考慮使用更穩健的參數

### Q: 如何選擇最終參數？

選項：
1. 使用最後一個窗口的參數
2. 使用所有窗口參數的平均值
3. 使用表現最穩定的參數
