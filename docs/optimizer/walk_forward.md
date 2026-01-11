# Walk-Forward Analysis

## 概述

Walk-Forward Analysis (WFA) 是防止策略過擬合的核心驗證工具。透過滾動窗口的樣本內/樣本外測試，驗證策略在未見資料上的穩健性。

## 為什麼需要 WFA？

### 過擬合問題

當我們在歷史資料上優化參數時，容易產生「過擬合」：

```
回測結果很完美 ≠ 未來會賺錢

原因：
1. 參數針對歷史資料過度優化
2. 策略記住了「雜訊」而非「規律」
3. 市場環境改變，策略失效
```

### WFA 如何解決？

```
核心思想：模擬真實交易情境

1. 用過去資料優化 (In-Sample)
2. 用未來資料驗證 (Out-of-Sample)
3. 不斷滾動重複
4. 計算 OOS 真實表現
```

## 窗口模式

### 1. Rolling (滾動窗口)

```
時間 ────────────────────────────────────>

Window 1:  |----IS----|--OOS--|
Window 2:       |----IS----|--OOS--|
Window 3:            |----IS----|--OOS--|

特點：
- 固定窗口大小
- 適合穩定市場
- 最常用
```

### 2. Expanding (擴展窗口)

```
時間 ────────────────────────────────────>

Window 1:  |----IS----|--OOS--|
Window 2:  |-------IS--------|--OOS--|
Window 3:  |------------IS------------|--OOS--|

特點：
- IS 逐步增大
- 適合趨勢市場
- 更多訓練資料
```

### 3. Anchored (錨定窗口)

```
時間 ────────────────────────────────────>

Window 1:  |----IS----|--OOS--|
Window 2:  |-------IS--------|--OOS--|
Window 3:  |------------IS------------|--OOS--|

特點：
- 起點固定
- 包含所有歷史
- 適合長期策略
```

## 核心指標

### 1. WFA Efficiency (效率)

```python
Efficiency = avg(OOS_return) / avg(IS_return)

解讀：
>= 0.9: 優秀 - 幾乎無衰退
>= 0.7: 良好 - 可接受範圍
>= 0.5: 普通 - 需改進
<  0.5: 不佳 - 嚴重過擬合
```

### 2. Consistency (一致性)

```python
Consistency = OOS 勝率 (報酬 > 0 的比例)

解讀：
>= 0.7: 優秀 - 高度穩定
>= 0.5: 良好 - 可接受
<  0.5: 不佳 - 不穩定
```

### 3. Degradation (衰退分析)

```python
Return Degradation = (IS_return - OOS_return) / IS_return
Sharpe Degradation = (IS_sharpe - OOS_sharpe) / IS_sharpe

解讀：
越低越好（衰退越少）
```

## 使用方式

### 基本用法

```python
from src.optimizer import WalkForwardAnalyzer
from src.backtester.engine import BacktestConfig
from src.strategies.momentum.rsi import RSIStrategy

# 1. 配置
config = BacktestConfig(
    symbol='BTCUSDT',
    timeframe='1h',
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    initial_capital=10000,
    leverage=3
)

# 2. 建立分析器
analyzer = WalkForwardAnalyzer(
    config=config,
    mode='rolling',
    optimize_metric='sharpe_ratio'
)

# 3. 執行分析
result = analyzer.analyze(
    strategy=RSIStrategy(),
    data=market_data,
    param_grid={
        'rsi_period': [10, 14, 20],
        'rsi_oversold': [25, 30, 35],
        'rsi_overbought': [65, 70, 75]
    },
    n_windows=5,
    is_ratio=0.7,
    verbose=True
)

# 4. 檢視結果
print(result.summary())
```

### 進階分析

```python
# 衰退分析
degradation = analyzer.analyze_degradation(result)
print(f"報酬衰退: {degradation['avg_return_degradation']:.2%}")
print(f"夏普衰退: {degradation['avg_sharpe_degradation']:.2%}")

# 繪製圖表
analyzer.plot_results(result, save_path='wfa_results.png')

# 檢查個別窗口
for window in result.windows:
    print(f"窗口 {window.window_id}:")
    print(f"  IS:  {window.is_return:.2%}")
    print(f"  OOS: {window.oos_return:.2%}")
    print(f"  最佳參數: {window.best_params}")
```

### 比較不同模式

```python
modes = ['rolling', 'expanding', 'anchored']
results = {}

for mode in modes:
    analyzer = WalkForwardAnalyzer(config=config, mode=mode)
    results[mode] = analyzer.analyze(...)

# 找出最佳模式
best_mode = max(results.items(), key=lambda x: x[1].efficiency)
print(f"最佳模式: {best_mode[0]}")
```

## 參數建議

### 窗口數量 (n_windows)

```
短期策略 (日內): 3-5 個窗口
中期策略 (數天): 5-10 個窗口
長期策略 (數週): 10-20 個窗口

權衡：
- 太少：統計不顯著
- 太多：單窗口資料不足
```

### IS 比例 (is_ratio)

```
常用值: 0.6 - 0.8

0.7 (70/30 分割) 最常見：
- IS: 70% 用於優化
- OOS: 30% 用於驗證

較高 IS (0.8):
- 更多訓練資料
- 但 OOS 驗證較弱

較低 IS (0.6):
- 更嚴格的 OOS 測試
- 但 IS 資料可能不足
```

### 最小交易次數 (min_trades)

```
建議: 10-30 筆

太少問題：
- 統計不可靠
- 容易受單筆影響

太多問題：
- 過濾掉有效策略
- 需要更長資料
```

## 實戰經驗法則

### ✓ 好的 WFA 結果

```python
result.efficiency >= 0.7         # 效率良好
result.consistency >= 0.5        # 一致性可接受
result.oos_mean_sharpe > 1.0     # OOS 夏普良好
degradation['avg_return_degradation'] < 0.3  # 衰退可控

→ 可考慮實盤測試
```

### ✗ 差的 WFA 結果

```python
result.efficiency < 0.5          # 嚴重衰退
result.consistency < 0.3         # 不穩定
result.oos_min_return < -0.2     # 出現大虧損窗口
oos_returns 波動過大             # 不穩定

→ 策略過擬合，需重新設計
```

### 警訊檢查

```python
# 1. 單窗口異常
for w in result.windows:
    if w.is_return > 0.5 and w.oos_return < 0:
        print(f"⚠️ 窗口 {w.window_id} 嚴重衰退")

# 2. 參數不穩定
params_list = [w.best_params for w in result.windows]
if all_params_different(params_list):
    print("⚠️ 每個窗口最佳參數都不同，策略不穩定")

# 3. 交易次數異常
for w in result.windows:
    if w.oos_trades < 5:
        print(f"⚠️ 窗口 {w.window_id} OOS 交易過少")
```

## 與其他工具整合

### 1. 與貝氏優化結合

```python
# 先用貝氏優化縮小搜尋空間
from src.optimizer import BayesianOptimizer

bayesian = BayesianOptimizer(config)
best_params = bayesian.optimize(strategy, data, n_trials=100)

# 在該範圍內建立更細緻的網格
param_grid = {
    'rsi_period': [
        best_params['rsi_period'] - 2,
        best_params['rsi_period'],
        best_params['rsi_period'] + 2
    ]
}

# WFA 驗證
result = analyzer.analyze(strategy, data, param_grid)
```

### 2. 多策略比較

```python
strategies = [
    RSIStrategy(),
    MACDStrategy(),
    SuperTrendStrategy()
]

wfa_results = {}
for strategy in strategies:
    result = analyzer.analyze(strategy, data, param_grid)
    wfa_results[strategy.name] = result

# 排名
ranking = sorted(
    wfa_results.items(),
    key=lambda x: x[1].efficiency,
    reverse=True
)

for i, (name, result) in enumerate(ranking, 1):
    print(f"{i}. {name}: 效率={result.efficiency:.2%}")
```

## 常見問題

### Q1: WFA 效率低於 0.5 怎麼辦？

```
可能原因：
1. 過擬合 - 參數網格太細
2. 市場環境改變 - IS/OOS 期間差異大
3. 樣本不足 - 窗口太小

解決方式：
1. 減少參數數量
2. 增加正則化（如最小交易次數）
3. 延長窗口資料
4. 考慮重新設計策略
```

### Q2: 不同窗口最佳參數差異很大？

```
這表示策略不穩定，可能：
1. 參數對市場敏感
2. 缺乏穩健性

建議：
1. 找出參數的「穩定範圍」
2. 使用參數平均值而非單點
3. 添加市場狀態過濾器
```

### Q3: 應該用哪種窗口模式？

```
Rolling: 一般情況
Expanding: 趨勢明顯的市場
Anchored: 長期策略

建議：都試試，比較結果
```

### Q4: WFA 通過就能實盤？

```
WFA 是必要但非充分條件：

還需要：
1. 合理的策略邏輯
2. 足夠的樣本數（至少 100 筆交易）
3. 不同市場環境測試
4. 壓力測試（極端行情）
5. 小倉位實盤驗證

WFA 只是第一關！
```

## 延伸閱讀

- [回測引擎文件](../backtester/engine.md)
- [貝氏優化器文件](./bayesian.md)
- [策略開發指南](../strategies/development.md)

## 參考資料

1. Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies*
2. White, H. (2000). "A Reality Check for Data Snooping"
3. Harvey, C. et al. (2016). "...and the Cross-Section of Expected Returns"
