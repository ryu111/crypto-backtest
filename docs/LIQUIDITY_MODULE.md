# 流動性影響模組 (Liquidity Impact Module)

## 概述

流動性模組用於模擬大單對市場的價格衝擊，是回測系統資料品質優化的重要組成部分。

## 理論基礎

### 市場衝擊 (Market Impact)

當大額訂單進入市場時，會造成價格偏離當前市價，這種現象稱為「市場衝擊」或「價格滑動」。

**關鍵因素：**
- **訂單大小 (Q)**: 訂單金額越大，衝擊越大
- **市場流動性 (ADV)**: 平均日成交量，衡量市場深度
- **波動率 (σ)**: 市場波動越大，衝擊越大

### 三種流動性模型

#### 1. 線性模型 (Linear Model)
```
價格衝擊 = η × σ × (Q / ADV)
```

**特性：**
- 最簡單的模型
- 衝擊與訂單大小成正比
- 適合快速估算

**使用時機：** 小額訂單、快速原型

---

#### 2. 平方根模型 (Square Root Model) ⭐ 推薦

```
價格衝擊 = η × σ × √(Q / ADV)
```

**特性：**
- 學術界標準模型 (Almgren-Chriss)
- 非線性衝擊（大單的邊際成本遞增）
- 符合真實市場行為

**使用時機：** 生產環境、真實回測

**理論基礎：**
- 市場參與者對大單的反應是非線性的
- 訂單拆分策略的理論基礎
- 大量實證研究支持

---

#### 3. 對數模型 (Logarithmic Model)

```
價格衝擊 = η × σ × log(1 + Q/ADV)
```

**特性：**
- 大單的邊際衝擊遞減
- 更保守的估計
- 適合極端流動性不足的市場

**使用時機：** 低流動性資產、風險厭惡場景

---

## 核心功能

### 1. 價格衝擊計算

```python
from src.backtester import create_square_root_liquidity

calc = create_square_root_liquidity(impact_coefficient=0.3)

# 計算 $50,000 訂單的價格衝擊
impact = calc.calculate_impact(
    data=price_data,
    order_size_usd=50000,
    index=current_index
)
# 結果: 0.012 (1.2%)
```

### 2. 執行價格估算

```python
# 考慮流動性衝擊的執行價格
current_price = 50000
exec_price = calc.estimate_execution_price(
    current_price=current_price,
    impact=impact,
    direction=1  # 1=做多, -1=做空
)
# 做多: 50600 (上漲 1.2%)
# 做空: 49400 (下跌 1.2%)
```

### 3. 最大訂單計算

```python
# 給定 1% 價格容忍度，計算最大可下單金額
max_order = calc.calculate_max_order_size(
    data=price_data,
    price_tolerance=0.01,  # 1%
    index=current_index
)
# 結果: $34,291 (超過此金額衝擊會超過 1%)
```

### 4. 流動性評級

```python
# 評估流動性等級
level = calc.get_liquidity_score(
    data=price_data,
    order_size_usd=50000,
    index=current_index
)
# 結果: LiquidityLevel.MEDIUM
```

**等級定義：**
- `HIGH`: 訂單 < 0.1% ADV（衝擊小）
- `MEDIUM`: 訂單 0.1% - 1% ADV
- `LOW`: 訂單 1% - 5% ADV
- `VERY_LOW`: 訂單 > 5% ADV（高風險）

---

## 配置選項

### LiquidityConfig 參數

```python
from src.backtester import LiquidityConfig, LiquidityModel

config = LiquidityConfig(
    # 模型選擇
    model=LiquidityModel.SQUARE_ROOT,  # 推薦

    # 衝擊係數 (η)
    impact_coefficient=0.3,  # 通常 0.1-0.5

    # ADV 計算
    adv_window=30,          # 30天窗口
    adv_percentile=0.5,     # 使用中位數（更穩定）

    # 波動率
    volatility_window=20,   # 20天窗口
    use_volatility=True,    # 考慮波動率

    # 限制
    max_impact=0.05,        # 最大衝擊 5%
    min_impact=0.0,         # 最小衝擊 0%
)
```

### 衝擊係數調校 (η)

| 市場 | 建議值 | 說明 |
|------|--------|------|
| BTC/USDT | 0.3 | 高流動性現貨 |
| ETH/USDT | 0.3 | 主流幣種 |
| 山寨幣 | 0.5 | 流動性較低 |
| 永續合約 | 0.2 | 通常比現貨流動性好 |

---

## 使用範例

### 範例 1：基本使用

```python
from src.backtester import create_square_root_liquidity
import pandas as pd

# 載入市場資料
data = pd.read_csv('btcusdt.csv', index_col=0, parse_dates=True)

# 建立計算器
calc = create_square_root_liquidity()

# 計算衝擊
impact = calc.calculate_impact(data, order_size_usd=50000, index=100)
print(f"價格衝擊: {impact:.4%}")  # 1.2075%
```

### 範例 2：回測整合

```python
# 向量化計算（高效能）
order_sizes = pd.Series(10000, index=data.index)  # 所有訂單 $10k
impacts = calc.calculate_vectorized(data, order_sizes)

# 調整回測執行價格
for i, impact in impacts.items():
    entry_price = data.loc[i, 'close']
    adjusted_price = entry_price * (1 + impact)  # 做多
    # 使用 adjusted_price 計算真實成本
```

### 範例 3：風險管理

```python
# 計算最大安全訂單大小
max_safe_order = calc.calculate_max_order_size(
    data=data,
    price_tolerance=0.005,  # 0.5% 容忍度
    index=current_index
)

# 限制訂單大小
actual_order = min(desired_order, max_safe_order)
```

---

## 向量化計算（回測優化）

```python
# 批次計算整個時間序列
order_sizes = pd.Series([10000, 20000, 15000, ...], index=data.index)
impacts = calc.calculate_vectorized(data, order_sizes)

# 結果是 pandas Series，可直接用於分析
print(f"平均衝擊: {impacts.mean():.4%}")
print(f"最大衝擊: {impacts.max():.4%}")
```

---

## 測試覆蓋

測試檔案：`tests/test_liquidity.py`

**測試項目：**
- ✅ 基本功能測試（32個測試）
- ✅ 三種模型計算（線性、平方根、對數）
- ✅ 向量化計算
- ✅ 邊界條件（零成交量、NaN 處理）
- ✅ 錯誤處理
- ✅ 整合測試（真實場景）

```bash
# 執行測試
pytest tests/test_liquidity.py -v

# 測試覆蓋率
pytest tests/test_liquidity.py --cov=src/backtester/liquidity
```

---

## 與滑點模組的差異

| 特性 | 滑點模組 | 流動性模組 |
|------|----------|-----------|
| **焦點** | 交易執行的微觀成本 | 市場深度的宏觀影響 |
| **時間尺度** | 瞬時（毫秒級） | 短期（分鐘級） |
| **主要因素** | 訂單簿深度、訂單類型 | ADV、訂單相對大小 |
| **應用** | 所有訂單 | 大額訂單 |
| **模型** | 動態滑點、市場衝擊 | 線性、平方根、對數 |

**實務建議：**
- 小額訂單（< 0.1% ADV）→ 主要考慮滑點
- 中大型訂單（> 0.1% ADV）→ 滑點 + 流動性衝擊
- 組合使用：`總成本 = 滑點成本 + 流動性衝擊`

---

## API 參考

### 類別

- `LiquidityCalculator`: 流動性計算器
- `LiquidityConfig`: 流動性配置
- `LiquidityModel`: 模型類型（Enum）
- `LiquidityLevel`: 流動性等級（Enum）

### 便捷函數

```python
# 建立不同模型的計算器
create_linear_liquidity(impact_coefficient=0.2)
create_square_root_liquidity(impact_coefficient=0.3)  # 推薦
create_logarithmic_liquidity(impact_coefficient=0.4)
```

### 主要方法

```python
# 單次計算
calc.calculate_impact(data, order_size_usd, index)
calc.estimate_execution_price(price, impact, direction)
calc.calculate_max_order_size(data, price_tolerance, index)
calc.get_liquidity_score(data, order_size_usd, index)

# 向量化計算
calc.calculate_vectorized(data, order_sizes, directions)

# 分析工具
calc.analyze_liquidity(data, order_sizes)
```

---

## 最佳實踐

### 1. 模型選擇

✅ **推薦：平方根模型**
- 學術界標準
- 真實市場驗證
- 適合大多數場景

### 2. 參數調校

```python
# 保守估計（風險厭惡）
config = LiquidityConfig(
    impact_coefficient=0.5,  # 較大係數
    max_impact=0.03,         # 較低上限
)

# 激進估計（追求收益）
config = LiquidityConfig(
    impact_coefficient=0.2,  # 較小係數
    max_impact=0.05,         # 較高上限
)
```

### 3. 回測整合

```python
class MyStrategy:
    def __init__(self):
        self.liquidity_calc = create_square_root_liquidity()

    def calculate_position_size(self, data, capital, index):
        # 計算理想部位大小
        ideal_size = capital * self.risk_ratio

        # 限制為流動性容許範圍
        max_size = self.liquidity_calc.calculate_max_order_size(
            data, price_tolerance=0.01, index=index
        )

        return min(ideal_size, max_size)
```

---

## 效能考量

### 向量化計算

```python
# 慢：逐筆計算
for i in range(len(data)):
    impact = calc.calculate_impact(data, size, index=i)

# 快：向量化
impacts = calc.calculate_vectorized(data, order_sizes)  # 快 100x
```

### 快取機制

計算器內建快取機制，重複計算同一時間序列時會自動優化：

```python
# 第一次計算：預計算 ADV 和波動率
impacts1 = calc.calculate_vectorized(data, sizes)

# 後續計算：使用快取（快 10x）
impacts2 = calc.calculate_vectorized(data, sizes)
```

---

## 限制與注意事項

### 1. 資料需求

**必要欄位：**
- `close`: 收盤價
- `volume`: 成交量（單位為資產數量，如 BTC）

**最小資料長度：**
- 至少需要 `adv_window` 天的歷史資料（預設 30 天）

### 2. 模型假設

- 假設市場參與者理性反應
- 不考慮訂單拆分策略
- 不考慮極端市場事件（如閃崩）

### 3. 參數敏感性

衝擊係數 `η` 對結果影響顯著，建議：
- 使用真實交易資料校準
- A/B 測試不同參數
- 保守估計（寧可高估衝擊）

---

## 未來改進

1. **動態係數調整**：根據市場狀態調整 η
2. **訂單簿深度整合**：使用 L2 資料更精確估算
3. **機器學習模型**：訓練預測市場衝擊
4. **高頻場景**：支援微秒級衝擊估算

---

## 參考文獻

1. Almgren, R., & Chriss, N. (2001). "Optimal execution of portfolio transactions"
2. Hasbrouck, J. (2009). "Trading costs and returns for US equities"
3. Grinold, R. C., & Kahn, R. N. (2000). "Active Portfolio Management"

---

## 相關文件

- [滑點模組文件](./SLIPPAGE_MODULE.md)
- [回測引擎文件](./BACKTEST_ENGINE.md)
- [使用範例](../examples/liquidity_usage.py)
- [測試文件](../tests/test_liquidity.py)
