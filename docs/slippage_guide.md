# 滑點模擬模組使用指南

## 概述

滑點模擬模組提供真實交易環境中的價格滑動計算，用於提高回測的準確性。

**滑點來源**：
- 市場深度不足（Order Book 深度）
- 市場波動率（波動越大，滑點越高）
- 訂單大小（大單對市場的衝擊）
- 訂單類型（市價單 vs 限價單）

## 快速開始

### 1. 基本使用

```python
from src.backtester.slippage import create_fixed_slippage, OrderType

# 建立固定滑點計算器（0.05%）
calculator = create_fixed_slippage(0.0005)

# 計算滑點
slippage = calculator.calculate(
    data=price_data,           # OHLCV DataFrame
    order_size=10000,          # 訂單金額（USDT）
    order_type=OrderType.MARKET
)

print(f"滑點: {slippage:.4%}")  # 輸出: 滑點: 0.0500%
```

### 2. 動態滑點（推薦）

```python
from src.backtester.slippage import create_dynamic_slippage

# 建立動態滑點計算器
calculator = create_dynamic_slippage(
    base_slippage=0.0005,      # 基礎滑點 0.05%
    volatility_factor=1.5,     # 波動率影響係數
    max_slippage=0.01          # 最大滑點 1%
)

# 計算滑點（會根據市場波動率自動調整）
slippage = calculator.calculate(
    data=price_data,
    order_size=10000,
    index=500  # 指定時間點
)
```

### 3. 市場衝擊模型

```python
from src.backtester.slippage import create_market_impact_slippage

# 建立市場衝擊滑點計算器
calculator = create_market_impact_slippage(
    base_slippage=0.0005,
    market_impact_coeff=0.1,   # 市場衝擊係數
    max_slippage=0.01
)

# 大單會有更高滑點
small_order = calculator.calculate(data, order_size=1000)   # 低滑點
large_order = calculator.calculate(data, order_size=100000) # 高滑點
```

## 滑點模型

### 1. 固定滑點（Fixed）

**特性**：
- 所有交易使用相同滑點
- 計算最快，適合初步測試
- 不考慮市場狀況變化

**公式**：
```
滑點 = 固定值（如 0.05%）
```

**使用場景**：
- 快速回測驗證
- 保守估算
- 基準對照

### 2. 動態滑點（Dynamic）⭐ 推薦

**特性**：
- 根據市場波動率動態調整
- 反映真實市場狀況
- 平衡準確性和計算效率

**公式**：
```
波動率因子 = 當前波動率 / 平均波動率
滑點 = 基礎滑點 × (1 + 波動率係數 × (波動率因子 - 1))
```

**使用場景**：
- 正式回測（推薦）
- 需要考慮市場狀況變化
- 中長期策略評估

**參數調整**：
```python
SlippageConfig(
    model=SlippageModel.DYNAMIC,
    base_slippage=0.0005,        # 基礎滑點
    volatility_factor=1.0,       # 波動率影響（1.0 = 100%）
    volatility_window=20,        # 波動率計算窗口
    max_slippage=0.01            # 最大滑點保護
)
```

### 3. 市場衝擊（Market Impact）

**特性**：
- 考慮訂單大小對市場的衝擊
- 大單滑點顯著增加
- 最真實但計算較複雜

**公式**：
```
市場衝擊 = 市場衝擊係數 × sqrt(訂單大小 / 平均成交量)
滑點 = 基礎滑點 × (1 + 市場衝擊)
```

**使用場景**：
- 大資金量策略
- 高頻交易策略
- 需要精確成本估算

**參數調整**：
```python
SlippageConfig(
    model=SlippageModel.MARKET_IMPACT,
    base_slippage=0.0005,
    market_impact_coeff=0.1,     # 市場衝擊係數（越高影響越大）
    volume_window=20             # 成交量計算窗口
)
```

### 4. 自定義函數（Custom）

**特性**：
- 完全自由定義滑點邏輯
- 適合特殊需求
- 需要自己實作

**範例**：
```python
def my_slippage_func(data, order_size, index):
    """基於交易時間的滑點"""
    hour = data.index[index].hour
    base = 0.0005

    # 非活躍時段增加滑點
    if hour < 8 or hour >= 20:
        return base * 1.5
    return base

calculator = SlippageCalculator()
calculator.set_custom_function(my_slippage_func)
```

## 訂單類型

### 市價單（Market Order）

```python
slippage = calculator.calculate(
    data=price_data,
    order_size=10000,
    order_type=OrderType.MARKET  # 完整滑點
)
```

**特性**：
- 保證成交
- 有完整滑點成本
- 適合緊急進出場

### 限價單（Limit Order）

```python
slippage = calculator.calculate(
    data=price_data,
    order_size=10000,
    order_type=OrderType.LIMIT  # 無滑點
)
# slippage = 0.0
```

**特性**：
- 無滑點成本
- 可能不成交
- 適合非緊急交易

### 止損單（Stop Order）

```python
config = SlippageConfig(
    stop_order_multiplier=1.5  # 止損單滑點倍數
)
calculator = SlippageCalculator(config)

slippage = calculator.calculate(
    data=price_data,
    order_size=10000,
    order_type=OrderType.STOP  # 滑點較高
)
```

**特性**：
- 滑點較高（預設 1.5x）
- 在觸發價成交
- 風控必備

## 向量化計算（回測專用）

```python
import pandas as pd

# 準備訂單序列
order_sizes = pd.Series(10000, index=price_data.index)

# 一次計算所有滑點
slippages = calculator.calculate_vectorized(
    data=price_data,
    order_sizes=order_sizes
)

# 結果是 pd.Series
print(f"平均滑點: {slippages.mean():.4%}")
print(f"最大滑點: {slippages.max():.4%}")
```

## 執行價格估算

```python
current_price = 50000
slippage = 0.001  # 0.1%

# 做多執行價
exec_price_long = calculator.estimate_execution_price(
    current_price, slippage, direction=1
)
print(f"做多執行價: ${exec_price_long:,.2f}")  # $50,050.00

# 做空執行價
exec_price_short = calculator.estimate_execution_price(
    current_price, slippage, direction=-1
)
print(f"做空執行價: ${exec_price_short:,.2f}")  # $49,950.00
```

## 滑點影響分析

```python
# 交易記錄
trades = pd.DataFrame({
    'entry_time': [...],  # 進場時間
    'size': [...]         # 訂單大小
})

# 分析滑點影響
analysis = calculator.analyze_impact(price_data, trades)

print(f"總滑點成本: ${analysis['total_cost']:,.2f}")
print(f"平均滑點: {analysis['avg_slippage']:.4%}")
print(f"最大滑點: {analysis['max_slippage']:.4%}")
print(f"中位數滑點: {analysis['median_slippage']:.4%}")
```

## 滑點曲線分析

```python
# 產生不同訂單大小的滑點曲線
curve = calculator.get_slippage_curve(
    data=price_data,
    order_sizes=[1000, 5000, 10000, 50000, 100000]
)

# curve 是 DataFrame，可以直接繪圖
import matplotlib.pyplot as plt

curve.plot(figsize=(12, 6))
plt.title('Slippage Curve by Order Size')
plt.ylabel('Slippage (%)')
plt.xlabel('Time')
plt.legend(title='Order Size')
plt.show()
```

## 整合到回測引擎

### 方法 1：在策略中計算

```python
from src.backtester.slippage import create_dynamic_slippage

class MyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.slippage_calc = create_dynamic_slippage()

    def generate_signals(self, data):
        # 產生訊號
        long_entry, long_exit, short_entry, short_exit = ...

        # 計算滑點並調整訊號
        for i in range(len(data)):
            if long_entry[i]:
                slippage = self.slippage_calc.calculate(
                    data,
                    order_size=self.position_size,
                    index=i
                )
                # 調整執行價格...

        return long_entry, long_exit, short_entry, short_exit
```

### 方法 2：在回測引擎中整合

```python
# TODO: 未來版本將直接整合到 BacktestEngine
```

## 參數建議

### 保守派（低估滑點）

```python
config = SlippageConfig(
    model=SlippageModel.FIXED,
    base_slippage=0.0002  # 0.02%（低於實際）
)
```

### 務實派（推薦）⭐

```python
config = SlippageConfig(
    model=SlippageModel.DYNAMIC,
    base_slippage=0.0005,     # 0.05%
    volatility_factor=1.0,
    max_slippage=0.01
)
```

### 悲觀派（高估滑點）

```python
config = SlippageConfig(
    model=SlippageModel.MARKET_IMPACT,
    base_slippage=0.001,      # 0.1%（高於實際）
    market_impact_coeff=0.2,
    max_slippage=0.02
)
```

### 幣安 BTC 永續合約參考值

根據實際交易經驗：

| 訂單大小 | 市價單滑點 | 說明 |
|---------|-----------|------|
| < $10k | 0.01-0.03% | 小單，流動性充足 |
| $10k-$50k | 0.03-0.05% | 中單，正常交易 |
| $50k-$100k | 0.05-0.10% | 大單，開始有衝擊 |
| > $100k | 0.10-0.30% | 超大單，明顯衝擊 |

**建議配置**：
```python
# 小資金（< $10k）
config = SlippageConfig(
    model=SlippageModel.DYNAMIC,
    base_slippage=0.0003,     # 0.03%
    volatility_factor=1.0,
    max_slippage=0.005
)

# 中資金（$10k-$100k）
config = SlippageConfig(
    model=SlippageModel.DYNAMIC,
    base_slippage=0.0005,     # 0.05%
    volatility_factor=1.2,
    max_slippage=0.01
)

# 大資金（> $100k）
config = SlippageConfig(
    model=SlippageModel.MARKET_IMPACT,
    base_slippage=0.001,      # 0.1%
    market_impact_coeff=0.15,
    max_slippage=0.02
)
```

## 常見問題

### Q1: 滑點應該設多少？

**A**: 取決於交易金額和市場狀況：
- 小額交易（< $10k）：0.02-0.05%
- 中額交易（$10k-$50k）：0.05-0.10%
- 大額交易（> $50k）：0.10-0.30%

### Q2: 使用哪個模型？

**A**:
- 快速測試 → **固定滑點**
- 正式回測 → **動態滑點**（推薦）
- 大資金 → **市場衝擊**

### Q3: 滑點會影響績效嗎？

**A**: 會，尤其是高頻策略：
```python
# 不考慮滑點
annual_return = 50%

# 考慮 0.05% 滑點，假設每月交易 10 次
slippage_cost = 0.0005 * 2 * 10 * 12 = 12%
real_return = 50% - 12% = 38%
```

### Q4: 如何驗證滑點設定？

**A**:
1. 查看交易所歷史成交明細
2. 比較回測成交價與市價
3. 統計實際滑點分佈
4. 調整模型參數使其符合實際

### Q5: 限價單真的沒有滑點嗎？

**A**:
- 成交時：無滑點（成交在限價）
- 未成交：機會成本（可能錯過行情）
- 建議：統計限價單成交率，計算期望成本

## 完整範例

查看 `examples/slippage_example.py` 獲取完整使用範例。

## API 參考

### SlippageCalculator

**主要方法**：

```python
# 單筆計算
calculator.calculate(data, order_size, order_type, direction, index)

# 向量化計算
calculator.calculate_vectorized(data, order_sizes, order_types, directions)

# 執行價格估算
calculator.estimate_execution_price(current_price, slippage, direction)

# 滑點曲線
calculator.get_slippage_curve(data, order_sizes)

# 影響分析
calculator.analyze_impact(data, trades)
```

### SlippageConfig

**主要參數**：

```python
SlippageConfig(
    model=SlippageModel.DYNAMIC,      # 模型類型
    base_slippage=0.0005,             # 基礎滑點
    volatility_factor=1.0,            # 波動率影響係數
    volume_factor=0.5,                # 成交量影響係數
    market_impact_coeff=0.1,          # 市場衝擊係數
    max_slippage=0.01,                # 最大滑點
    min_slippage=0.0,                 # 最小滑點
    stop_order_multiplier=1.5,        # 止損單滑點倍數
    volatility_window=20,             # 波動率計算窗口
    volume_window=20                  # 成交量計算窗口
)
```

## 測試

```bash
# 執行測試
pytest tests/test_slippage.py -v

# 執行範例
python examples/slippage_example.py
```

## 延伸閱讀

- VectorBT Pro 滑點設定：https://vectorbt.pro/
- 幣安永續合約交易規則：https://www.binance.com/
- Almgren-Chriss 市場衝擊模型：學術論文

---

**版本**: 1.0.0
**最後更新**: 2025-01-11
