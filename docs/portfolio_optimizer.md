# 策略組合優化器使用指南

## 概述

`PortfolioOptimizer` 提供基於現代投資組合理論（Modern Portfolio Theory, MPT）的策略組合優化功能。

## 核心功能

### 1. 優化方法

| 方法 | 說明 | 使用場景 |
|------|------|----------|
| **等權重** | 所有策略平均配置 | 基準、多樣化 |
| **反波動率** | 根據波動率反向加權 | 風險控制 |
| **最大 Sharpe** | 最大化風險調整報酬 | 績效優化 |
| **風險平價** | 每個策略風險貢獻相等 | 平衡配置 |
| **Mean-Variance** | 給定目標報酬/風險優化 | 客製化需求 |

### 2. 約束條件

- 單一策略最大/最小權重
- 允許/禁止空頭
- 目標報酬率
- 目標風險水平

## 快速開始

### 基本使用

```python
import pandas as pd
from src.optimizer.portfolio import PortfolioOptimizer

# 準備策略回報資料（每日）
returns = pd.DataFrame({
    'strategy_a': [0.01, 0.02, -0.01, ...],
    'strategy_b': [0.015, -0.005, 0.02, ...],
    'strategy_c': [0.008, 0.012, 0.005, ...]
})

# 建立優化器
optimizer = PortfolioOptimizer(
    returns=returns,
    risk_free_rate=0.0,
    frequency=252  # 年化頻率
)

# 最大化 Sharpe Ratio
weights = optimizer.max_sharpe_optimize()
print(weights.summary())
```

### 帶約束的優化

```python
# 限制單一策略權重，不允許空頭
weights = optimizer.max_sharpe_optimize(
    max_weight=0.5,  # 每個策略最多 50%
    min_weight=0.1,  # 每個策略至少 10%
    allow_short=False
)
```

### 目標報酬優化

```python
# 達成 15% 年化報酬，並最小化風險
weights = optimizer.mean_variance_optimize(
    target_return=0.15,
    max_weight=0.6
)
```

### 風險平價配置

```python
# 讓每個策略對組合風險的貢獻相等
weights = optimizer.risk_parity_optimize(
    max_weight=0.6,
    min_weight=0.05
)
```

## 進階功能

### 效率前緣

```python
# 計算效率前緣（50 個點）
frontier = optimizer.efficient_frontier(n_points=50)

# 找出最大 Sharpe Ratio 點
max_sharpe_point = max(frontier, key=lambda p: p.sharpe_ratio)

# 視覺化
optimizer.plot_efficient_frontier(
    frontier=frontier,
    save_path='frontier.png',
    show_assets=True
)
```

### 相關性分析

```python
# 取得策略間的相關性矩陣
corr_matrix = optimizer.get_correlation_matrix()
print(corr_matrix)
```

### Ledoit-Wolf 協方差估計

```python
# 使用 Ledoit-Wolf 收縮估計（推薦）
optimizer = PortfolioOptimizer(
    returns=returns,
    use_ledoit_wolf=True  # 減少估計誤差
)
```

## PortfolioWeights 物件

優化結果以 `PortfolioWeights` 物件返回：

```python
result = optimizer.max_sharpe_optimize()

# 存取屬性
print(result.weights)              # {'strategy_a': 0.4, 'strategy_b': 0.6}
print(result.expected_return)      # 0.15 (15%)
print(result.expected_volatility)  # 0.12 (12%)
print(result.sharpe_ratio)         # 1.25

# 產生摘要報告
print(result.summary())

# 轉為字典
data = result.to_dict()
```

## 完整範例

```python
import pandas as pd
import numpy as np
from src.optimizer.portfolio import PortfolioOptimizer

# 1. 準備資料（模擬 3 個策略的日回報）
np.random.seed(42)
returns = pd.DataFrame({
    'momentum': np.random.normal(0.001, 0.02, 252),
    'mean_reversion': np.random.normal(0.0008, 0.015, 252),
    'trend_following': np.random.normal(0.0012, 0.018, 252)
})

# 2. 建立優化器
optimizer = PortfolioOptimizer(
    returns=returns,
    risk_free_rate=0.0,
    frequency=252,
    use_ledoit_wolf=True
)

# 3. 比較不同方法
methods = {
    '等權重': optimizer.equal_weight_portfolio(),
    '反波動率': optimizer.inverse_volatility_portfolio(),
    '最大 Sharpe': optimizer.max_sharpe_optimize(),
    '風險平價': optimizer.risk_parity_optimize()
}

# 4. 顯示結果
for name, portfolio in methods.items():
    print(f"\n{name}:")
    print(f"  報酬: {portfolio.expected_return*100:.2f}%")
    print(f"  風險: {portfolio.expected_volatility*100:.2f}%")
    print(f"  Sharpe: {portfolio.sharpe_ratio:.4f}")
    print(f"  權重: {portfolio.weights}")
```

## 參數說明

### PortfolioOptimizer 初始化參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `returns` | pd.DataFrame | 必填 | 策略回報資料 |
| `risk_free_rate` | float | 0.0 | 無風險利率（年化） |
| `frequency` | int | 252 | 年化頻率（252=日, 52=週, 12=月） |
| `use_ledoit_wolf` | bool | True | 使用 Ledoit-Wolf 估計 |

### 優化方法參數

#### `max_sharpe_optimize()`

```python
max_sharpe_optimize(
    max_weight=1.0,      # 單一策略最大權重
    min_weight=0.0,      # 單一策略最小權重
    allow_short=False    # 是否允許空頭
)
```

#### `mean_variance_optimize()`

```python
mean_variance_optimize(
    target_return=None,  # 目標年化報酬
    target_risk=None,    # 目標年化波動
    max_weight=1.0,
    min_weight=0.0,
    allow_short=False
)
```

#### `risk_parity_optimize()`

```python
risk_parity_optimize(
    max_weight=1.0,
    min_weight=0.0
)
```

## 注意事項

### 資料要求

1. **回報資料格式**: DataFrame，columns 為策略名稱，index 為日期
2. **資料頻率一致**: 所有策略使用相同時間粒度（日/週/月）
3. **足夠的樣本數**: 至少 100+ 個觀測值，建議 252+（1 年）
4. **處理缺失值**: 自動填補為 0，建議預處理

### 優化限制

1. **至少 2 個策略**: 組合優化至少需要 2 個標的
2. **協方差矩陣**: 必須正定，建議使用 Ledoit-Wolf
3. **約束可行性**: 確保約束條件有解（如 min_weight * n_assets ≤ 1）

### 最佳實踐

1. **使用 Ledoit-Wolf**: 減少協方差估計誤差
2. **設定合理約束**: 避免過度集中（max_weight ≤ 0.5）
3. **多種方法比較**: 不要只依賴單一優化方法
4. **樣本外驗證**: 在新資料上驗證優化結果
5. **定期重新優化**: 市場條件變化時更新權重

## 技術細節

### 優化演算法

- **求解器**: scipy.optimize.minimize (SLSQP method)
- **協方差估計**: Ledoit-Wolf shrinkage (可選)
- **約束處理**: Linear constraints + Bounds

### 計算複雜度

| 操作 | 時間複雜度 |
|------|-----------|
| 協方差估計 | O(n²m) |
| 單次優化 | O(n³) |
| 效率前緣 | O(kn³) |

n = 策略數量, m = 資料點數, k = 前緣點數

### 依賴套件

```
pandas >= 2.0.0
numpy >= 1.24.0
scipy >= 1.10.0
scikit-learn >= 1.3.0  # Ledoit-Wolf
matplotlib >= 3.7.0    # 視覺化（可選）
```

## 常見問題

### Q: 優化結果不穩定？

A: 使用 Ledoit-Wolf 估計，增加樣本數，或添加正則化約束。

### Q: 權重集中在少數策略？

A: 設定 `max_weight` 約束，或使用風險平價方法。

### Q: 如何處理 NaN？

A: 自動填補為 0，但建議預先處理（前向填補、插值等）。

### Q: 優化失敗怎麼辦？

A: 檢查約束條件是否可行，嘗試放寬約束或使用更簡單的方法。

## 相關資源

- [Modern Portfolio Theory - Wikipedia](https://en.wikipedia.org/wiki/Modern_portfolio_theory)
- [Ledoit-Wolf Covariance Estimation](https://scikit-learn.org/stable/modules/covariance.html#shrunk-covariance)
- [Risk Parity - Investopedia](https://www.investopedia.com/terms/r/risk-parity.asp)

## 更新日誌

### v1.0.0 (2026-01-11)

- ✨ 初始版本
- ✅ 支援 5 種優化方法
- ✅ Ledoit-Wolf 協方差估計
- ✅ 效率前緣計算
- ✅ 完整約束條件支援
