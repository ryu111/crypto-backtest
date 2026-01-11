# Kelly Criterion 部位管理指南

## 概述

Kelly Criterion 是一個數學公式，用於計算在重複博弈中最佳的資金投注比例，可最大化長期資金增長率。

### 公式

```
f* = W - (1-W)/R

其中：
- f*: 最佳資金比例
- W: 勝率 (Win Rate)
- R: 盈虧比 (Win/Loss Ratio = 平均獲利/平均虧損)
```

### 範例

假設策略有：
- 勝率 55%
- 平均獲利 $150
- 平均虧損 $100

計算：
```
盈虧比 R = 150 / 100 = 1.5
f* = 0.55 - (0.45 / 1.5) = 0.55 - 0.3 = 0.25 (25%)
```

建議使用 25% 的資金進行交易。

---

## 使用方式

### 基本用法

```python
from src.risk import kelly_criterion

# 計算最佳資金比例
optimal_fraction = kelly_criterion(
    win_rate=0.55,        # 55% 勝率
    win_loss_ratio=1.5    # 盈虧比 1.5
)
# 結果: 0.25 (25%)
```

### 使用 Position Sizer

```python
from src.risk import KellyPositionSizer

# 初始化 Half Kelly (推薦)
sizer = KellyPositionSizer(
    kelly_fraction=0.5,           # Half Kelly
    max_position_fraction=0.25,   # 最大 25% 部位
    min_win_rate=0.45,            # 最低 45% 勝率
    min_win_loss_ratio=1.2        # 最低盈虧比 1.2
)

# 計算部位大小
result = sizer.calculate_position_size(
    capital=100000,      # 10 萬 USD
    win_rate=0.58,       # 58% 勝率
    avg_win=350,         # 平均獲利 $350
    avg_loss=200         # 平均虧損 $200
)

print(result)
# PositionSizeResult(
#   kelly_type=Half Kelly,
#   optimal_fraction=0.1700 (17.00%),
#   position_size=$17,000.00,
#   ...
# )
```

### 從交易歷史計算

```python
# 真實交易記錄
winning_trades = [320, 180, 450, 220, 280]
losing_trades = [150, 120, 180]

result = sizer.calculate_from_trades(
    capital=50000,
    winning_trades=winning_trades,
    losing_trades=losing_trades
)
```

---

## Kelly Criterion 變體

### Full Kelly (kelly_fraction = 1.0)

- **特性**: 使用完整公式結果
- **優點**: 理論上最大化長期增長率
- **缺點**: 波動極大，實際難以承受
- **建議**: 不建議用於實際交易

### Half Kelly (kelly_fraction = 0.5) ⭐ 推薦

- **特性**: 使用 Full Kelly 的一半
- **優點**:
  - 波動降低 75%
  - 長期報酬率僅降低 25%
  - 風險與報酬最佳平衡
- **建議**: 適合大多數交易者

### Quarter Kelly (kelly_fraction = 0.25)

- **特性**: 使用 Full Kelly 的四分之一
- **優點**: 極保守，波動最小
- **缺點**: 資金利用率較低
- **建議**: 適合保守型交易者或新手

---

## 重要參數說明

### 1. kelly_fraction

Kelly 乘數，決定風險偏好：
```python
KellyPositionSizer(kelly_fraction=0.5)  # Half Kelly
```

### 2. max_position_fraction

最大部位限制（安全上限）：
```python
KellyPositionSizer(max_position_fraction=0.25)  # 最大 25%
```

### 3. min_win_rate

最低勝率要求（過濾差策略）：
```python
KellyPositionSizer(min_win_rate=0.45)  # 最低 45%
```

### 4. min_win_loss_ratio

最低盈虧比要求：
```python
KellyPositionSizer(min_win_loss_ratio=1.2)  # 最低 1.2
```

---

## 動態調整

根據策略表現動態調整 Kelly 乘數：

```python
sizer = KellyPositionSizer(kelly_fraction=0.5)

# 策略表現良好 → 提高至 Full Kelly
if performance_is_excellent:
    sizer.adjust_kelly_fraction(1.0)

# 策略進入回撤 → 降低至 Quarter Kelly
if in_drawdown:
    sizer.adjust_kelly_fraction(0.25)

# 策略恢復 → 回到 Half Kelly
if recovered:
    sizer.adjust_kelly_fraction(0.5)
```

---

## 使用建議

### ✅ 應該做的事

1. **使用 Half Kelly 或更保守**
   - Full Kelly 波動太大，實務上難以承受

2. **設定最大部位限制**
   - 即使 Kelly 計算結果很高，也應限制在合理範圍（如 25%）

3. **定期檢視參數**
   - 市場環境改變，策略績效會變化
   - 建議每月重新計算勝率和盈虧比

4. **配合嚴格風險管理**
   - Kelly 只是部位大小，仍需止損、分散風險

5. **足夠的樣本數**
   - 至少 30 筆以上交易才有統計意義

### ❌ 不應該做的事

1. **不要使用 Full Kelly**
   - 理論最佳 ≠ 實務最佳

2. **不要過度自信**
   - 過去績效不代表未來表現

3. **不要忽略黑天鵝風險**
   - Kelly 假設未來與過去一致
   - 極端事件會打破假設

4. **不要在樣本不足時使用**
   - 少於 20-30 筆交易的統計不可靠

5. **不要忘記資金曲線變化**
   - 資金增長/下降後要重新計算部位

---

## 實際應用範例

### 場景 1: 保守型交易者

```python
# 設定：Quarter Kelly + 嚴格限制
sizer = KellyPositionSizer(
    kelly_fraction=0.25,
    max_position_fraction=0.1,    # 最大 10%
    min_win_rate=0.5,             # 最低 50%
    min_win_loss_ratio=1.5        # 最低 1.5
)
```

### 場景 2: 平衡型交易者 (推薦)

```python
# 設定：Half Kelly + 合理限制
sizer = KellyPositionSizer(
    kelly_fraction=0.5,
    max_position_fraction=0.25,   # 最大 25%
    min_win_rate=0.45,            # 最低 45%
    min_win_loss_ratio=1.2        # 最低 1.2
)
```

### 場景 3: 經驗豐富交易者

```python
# 設定：Full Kelly + 動態調整
sizer = KellyPositionSizer(
    kelly_fraction=1.0,
    max_position_fraction=0.4,    # 最大 40%
    min_win_rate=0.4,             # 最低 40%
    min_win_loss_ratio=1.0        # 最低 1.0
)

# 根據近期表現動態調整
recent_sharpe = calculate_sharpe_ratio()
if recent_sharpe < 1.0:
    sizer.adjust_kelly_fraction(0.5)  # 降低風險
```

---

## 常見問題 (FAQ)

### Q1: Kelly Criterion 計算結果為負數怎麼辦？

A: 表示該策略為負期望值，**不應該交易**。

### Q2: 為什麼建議使用 Half Kelly 而非 Full Kelly？

A: Full Kelly 理論最佳，但實務上：
- 波動太大，心理難以承受
- 對估計誤差極度敏感
- Half Kelly 波動降低 75%，報酬只降低 25%

### Q3: 勝率和盈虧比哪個更重要？

A: 兩者都重要，但盈虧比影響更大：
- 高勝率 + 低盈虧比：可能是負期望值
- 低勝率 + 高盈虧比：可能是正期望值
- 範例：30% 勝率，盈虧比 3.0 → Kelly = 6.7%

### Q4: 多久應該重新計算一次？

A: 建議：
- **定期檢視**: 每月或每季
- **事件驅動**: 策略大幅改變、市場環境改變
- **績效變化**: 勝率或盈虧比明顯變化時

### Q5: 可以用於高頻交易嗎？

A: 可以，但需注意：
- 樣本數夠大（數百或數千筆交易）
- 交易成本要計入平均獲利/虧損
- 高頻策略衰退快，需更頻繁更新

---

## 數學推導

Kelly Criterion 來自於最大化長期資金增長率：

```
假設：
- 每次交易投注資金比例 f
- 勝率 W，獲利 b 倍
- 敗率 (1-W)，虧損全部

期望增長率 G(f):
G(f) = W * log(1 + b*f) + (1-W) * log(1 - f)

求 G(f) 最大值：
dG/df = 0
=> f* = W - (1-W)/b

其中 b 即為盈虧比 R
```

---

## 參考資料

1. **原始論文**
   - Kelly, J. L. (1956). "A New Interpretation of Information Rate"

2. **實務應用**
   - Thorp, E. O. (2008). "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market"

3. **批判與改進**
   - MacLean, L. C., Thorp, E. O., & Ziemba, W. T. (2011). "The Kelly Capital Growth Investment Criterion"

---

## 總結

Kelly Criterion 是強大的數學工具，但需謹慎使用：

✅ **優點**
- 理論上最大化長期增長
- 自動調整部位大小
- 數學基礎嚴謹

⚠️ **限制**
- 假設未來與過去一致
- 對參數估計誤差敏感
- 實務波動可能難以承受

**最佳實踐**：使用 Half Kelly 或 Quarter Kelly，配合嚴格風險管理。
