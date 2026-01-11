# Optimizer 優化器模組

防止過擬合、驗證策略穩健性的核心工具。

## 模組概述

```
optimizer/
├── walk_forward.py    # Walk-Forward Analysis
├── bayesian.py        # Bayesian Optimization (待實作)
└── __init__.py
```

## 核心功能

### 1. Walk-Forward Analysis

**目的**: 檢測過擬合，驗證策略在未見資料上的表現

**核心概念**:
- 樣本內 (IS): 用於優化參數
- 樣本外 (OOS): 用於驗證效果
- 滾動窗口: 持續驗證穩健性

**關鍵指標**:
- **WFA Efficiency**: OOS/IS 報酬比 (>= 0.7 佳)
- **Consistency**: OOS 勝率 (>= 0.5 佳)
- **Degradation**: 效能衰退程度

### 2. Bayesian Optimization (計畫中)

**目的**: 高效參數優化

**優勢**:
- 智能搜尋（非暴力網格）
- 減少計算時間
- 找出全域最優

## 快速開始

### 基本 WFA

```python
from src.optimizer import WalkForwardAnalyzer
from src.backtester.engine import BacktestConfig
from src.strategies.momentum.rsi import RSIStrategy

# 配置
config = BacktestConfig(
    symbol='BTCUSDT',
    timeframe='1h',
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    initial_capital=10000,
    leverage=3
)

# 分析器
analyzer = WalkForwardAnalyzer(
    config=config,
    mode='rolling',  # 'rolling', 'expanding', 'anchored'
    optimize_metric='sharpe_ratio'
)

# 執行
result = analyzer.analyze(
    strategy=RSIStrategy(),
    data=market_data,
    param_grid={
        'rsi_period': [10, 14, 20],
        'rsi_oversold': [25, 30, 35],
        'rsi_overbought': [65, 70, 75]
    },
    n_windows=5,
    is_ratio=0.7
)

# 結果
print(result.summary())
print(f"效率: {result.efficiency:.2%}")
print(f"一致性: {result.consistency:.2%}")
```

### 判斷策略品質

```python
if result.efficiency >= 0.7 and result.consistency >= 0.5:
    print("✓ 策略通過驗證")
else:
    print("✗ 策略可能過擬合")
```

## 窗口模式選擇

| 模式 | 說明 | 適用場景 |
|------|------|----------|
| **rolling** | 固定大小窗口滾動 | 一般情況（預設） |
| **expanding** | IS 窗口逐步擴大 | 趨勢市場 |
| **anchored** | 起點固定，逐步延伸 | 長期策略 |

## 參數建議

### 窗口數量 (n_windows)

```python
短期策略: 3-5
中期策略: 5-10
長期策略: 10-20
```

### IS 比例 (is_ratio)

```python
常用: 0.7 (70% IS, 30% OOS)
嚴格: 0.6 (更多 OOS 驗證)
寬鬆: 0.8 (更多訓練資料)
```

### 最小交易次數 (min_trades)

```python
建議: 10-30 筆
確保統計顯著性
```

## 結果解讀

### 優秀策略

```python
✓ Efficiency >= 0.9      # 幾乎無衰退
✓ Consistency >= 0.7     # 高度穩定
✓ OOS Sharpe > 1.0       # 良好風險調整報酬
✓ 衰退 < 30%             # 可控範圍
```

### 過擬合跡象

```python
✗ Efficiency < 0.5       # 嚴重衰退
✗ Consistency < 0.3      # 不穩定
✗ 單窗口大虧損           # 有破產風險
✗ 參數劇烈變化           # 策略不穩健
```

## 進階用法

### 1. 衰退分析

```python
degradation = analyzer.analyze_degradation(result)
print(f"報酬衰退: {degradation['avg_return_degradation']:.2%}")
print(f"夏普衰退: {degradation['avg_sharpe_degradation']:.2%}")
```

### 2. 視覺化

```python
analyzer.plot_results(
    result,
    save_path='wfa_results.png'
)
```

### 3. 多模式比較

```python
modes = ['rolling', 'expanding', 'anchored']
results = {}

for mode in modes:
    analyzer = WalkForwardAnalyzer(config=config, mode=mode)
    results[mode] = analyzer.analyze(...)

# 找最佳
best = max(results.items(), key=lambda x: x[1].efficiency)
```

### 4. 多策略比較

```python
strategies = [RSIStrategy(), MACDStrategy(), SuperTrendStrategy()]
wfa_results = {}

for strategy in strategies:
    result = analyzer.analyze(strategy, ...)
    wfa_results[strategy.name] = result

# 排名
ranking = sorted(
    wfa_results.items(),
    key=lambda x: x[1].efficiency,
    reverse=True
)
```

## 整合工作流

```
1. 策略設計
   ↓
2. 初步回測（單參數組）
   ↓
3. Bayesian Optimization（找有潛力參數範圍）
   ↓
4. Walk-Forward Analysis（驗證穩健性）
   ↓
5. 如果 WFA 通過 → 實盤小倉位測試
   如果 WFA 不通過 → 回到步驟 1
```

## 常見問題

### Q: 為什麼需要 WFA？

回測結果好 ≠ 未來會賺錢。WFA 模擬真實情境，檢測過擬合。

### Q: WFA 效率多少算及格？

- >= 0.9: 優秀
- >= 0.7: 良好
- >= 0.5: 普通
- < 0.5: 不佳（可能過擬合）

### Q: 應該用幾個窗口？

取決於資料長度和策略週期。一般 5-10 個窗口即可。

### Q: WFA 通過就能實盤？

WFA 是必要但非充分條件。還需要：
- 合理的策略邏輯
- 足夠的樣本數
- 壓力測試
- 小倉位驗證

## 效能考量

### 計算時間

```python
時間 = n_windows × n_param_combinations × backtest_time

範例：
- 5 窗口
- 27 種參數組合 (3×3×3)
- 每次回測 0.5 秒
= 5 × 27 × 0.5 = 67.5 秒
```

### 優化建議

```python
# 1. 縮小參數網格
param_grid = {
    'period': [10, 20, 30]  # 而非 [10, 11, 12, ..., 30]
}

# 2. 先用貝氏優化找範圍
bayesian_result = bayesian_optimizer.optimize(...)
# 再用 WFA 驗證

# 3. 減少窗口數（測試時）
n_windows=3  # 快速驗證
n_windows=10 # 完整分析
```

## 測試

```bash
# 執行測試
pytest tests/test_walk_forward.py -v

# 執行範例
python examples/walk_forward_example.py
```

## 文件

- [完整文件](../../docs/optimizer/walk_forward.md)
- [使用範例](../../examples/walk_forward_example.py)
- [測試案例](../../tests/test_walk_forward.py)

## 相關模組

- [Backtester](../backtester/README.md) - 回測引擎
- [Strategies](../strategies/README.md) - 策略開發
- [Data](../data/README.md) - 資料載入

## 待辦事項

- [ ] 實作 BayesianOptimizer
- [ ] 添加參數敏感度分析
- [ ] 支援多目標優化
- [ ] Monte Carlo 模擬
- [ ] 自適應窗口大小

## 參考資料

1. Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies*
2. White, H. (2000). "A Reality Check for Data Snooping"
3. Harvey, C. et al. (2016). "...and the Cross-Section of Expected Returns"
