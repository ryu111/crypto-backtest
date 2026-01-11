# Monte Carlo 模擬器

策略驗證的最後階段，透過 Monte Carlo 模擬評估策略在不同隨機情境下的表現。

## 特色

### 多種模擬方法
- **Shuffle**：交易順序隨機化
- **Bootstrap**：有放回抽樣
- **Block Bootstrap**：區塊抽樣（保留時間相關性）

### 完整風險評估
- VaR / CVaR 計算
- 百分位數分布
- 獲利機率估計
- 權益曲線路徑模擬

### 視覺化支援
- 分布圖（直方圖 + 箱型圖）
- 權益曲線路徑圖
- 風險指標標記

## 快速開始

```python
from src.validator import MonteCarloSimulator
import pandas as pd

# 準備交易記錄
trades = pd.DataFrame({
    'pnl': [100, -50, 80, -30, 120]
})

# 執行模擬
simulator = MonteCarloSimulator(seed=42)
result = simulator.simulate(
    trades=trades,
    n_simulations=10000,
    method='bootstrap'
)

# 查看結果
simulator.print_result(result)

# 繪製圖表
simulator.plot_distribution(result)
```

## 檔案結構

```
src/validator/
├── __init__.py              # 模組匯出
├── monte_carlo.py           # Monte Carlo 模擬器
└── stages.py               # 5 階段驗證系統

tests/
└── test_monte_carlo.py     # 測試（21 個測試，全數通過）

examples/
└── monte_carlo_example.py  # 完整範例（5 個情境）

docs/
└── monte_carlo.md          # 詳細文檔
```

## 核心類別

### MonteCarloSimulator

主要模擬器類別，提供：

**模擬方法**：
- `simulate()` - 執行 Monte Carlo 模擬
- `generate_equity_paths()` - 產生權益曲線路徑

**統計方法**：
- `calculate_statistics()` - 計算分布統計
- `calculate_var()` - 計算風險價值
- `calculate_cvar()` - 計算條件風險價值

**視覺化**：
- `plot_distribution()` - 繪製分布圖
- `plot_paths()` - 繪製權益曲線路徑

### MonteCarloResult

結果容器，包含：
- 分布統計（平均、標準差、百分位數）
- 風險指標（VaR、CVaR）
- 機率指標（獲利機率、超越原始機率）
- 完整模擬資料

## 使用情境

### 1. 策略穩健性測試
評估策略是否依賴特定交易順序。

```python
result = simulator.simulate(trades, method='shuffle')
print(f"標準差: {result.std:.2f}")  # 越小越穩定
```

### 2. 風險評估
了解最壞情況下的表現。

```python
result = simulator.simulate(trades, method='bootstrap')
print(f"VaR(95%): {result.var_95:.2f}")
print(f"CVaR(95%): {result.cvar_95:.2f}")
```

### 3. 策略比較
比較不同策略的風險收益特徵。

```python
result_a = simulator.simulate(strategy_a)
result_b = simulator.simulate(strategy_b)

sharpe_a = result_a.mean / result_a.std
sharpe_b = result_b.mean / result_b.std
print(f"策略 A Sharpe: {sharpe_a:.2f}")
print(f"策略 B Sharpe: {sharpe_b:.2f}")
```

### 4. 權益曲線分析
了解策略路徑的變異性。

```python
equity_paths, original = simulator.generate_equity_paths(trades)
simulator.plot_paths(equity_paths, original)
```

## 方法選擇指南

| 策略類型 | 建議方法 | 理由 |
|----------|----------|------|
| 獨立交易 | Shuffle / Bootstrap | 交易間無相關性 |
| 趨勢追蹤 | Block Bootstrap | 需保留時間結構 |
| 均值回歸 | Shuffle | 順序影響小 |
| 未知特性 | 全部測試 | 觀察差異 |

## 測試結果

```bash
$ pytest tests/test_monte_carlo.py -v

21 passed in 12.65s ✅
```

測試涵蓋：
- 所有模擬方法
- 邊界條件（空交易、單筆交易）
- 統計計算正確性
- 結果可重現性
- 視覺化功能

## 範例執行

```bash
# 執行完整範例
python examples/monte_carlo_example.py

# 選項：
# 1. 基本模擬
# 2. 比較方法
# 3. 權益路徑
# 4. 風險分析
# 5. 策略穩健性
```

## 效能指標

| 交易數 | 模擬次數 | 執行時間 |
|--------|----------|----------|
| 50 | 1,000 | ~0.1s |
| 100 | 5,000 | ~0.5s |
| 200 | 10,000 | ~2s |
| 500 | 10,000 | ~5s |

*在 Apple M1 Pro 上測試

## 依賴套件

- `numpy` - 數值計算
- `pandas` - 資料處理
- `matplotlib` - 視覺化
- `scipy` - 統計函數（選用）

## 文檔

詳細文檔請參考：
- **使用指南**：`docs/monte_carlo.md`
- **API 文檔**：原始碼 docstrings
- **範例程式**：`examples/monte_carlo_example.py`

## 注意事項

1. **樣本大小**：建議至少 30 筆交易
2. **模擬次數**：1000-10000 次通常足夠
3. **方法選擇**：根據策略特性選擇適當方法
4. **結果解讀**：Monte Carlo 假設未來與過去相似

## 整合使用

Monte Carlo 模擬器可與 5 階段驗證系統整合：

```python
from src.validator import MonteCarloSimulator, get_stage_validator

# Stage 5: Monte Carlo 模擬
simulator = MonteCarloSimulator(seed=42)
mc_result = simulator.simulate(trades, n_simulations=10000)

# 整合到 5 階段驗證
validator = get_stage_validator()
full_result = validator.validate(
    strategy=strategy,
    data=data,
    monte_carlo_result=mc_result  # 傳入 MC 結果
)
```

## 授權

MIT License

---

**建立完成** ✅

Monte Carlo 模擬器已完整實作，包含：
- 3 種模擬方法
- 完整測試覆蓋（21 個測試全數通過）
- 5 個實用範例
- 詳細文檔
- 視覺化支援
