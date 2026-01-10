# 部位大小計算詳解

## 固定風險法

### 核心公式

```
部位大小 = 風險金額 / 止損距離
風險金額 = 總資金 × 風險比例
```

### 實作

```python
def fixed_risk_size(
    capital: float,
    risk_pct: float,
    entry_price: float,
    stop_loss_price: float
) -> float:
    """
    固定風險部位大小

    Args:
        capital: 總資金 (USDT)
        risk_pct: 單筆風險比例 (如 0.02 = 2%)
        entry_price: 入場價格
        stop_loss_price: 止損價格

    Returns:
        size: 部位大小（合約數量）
    """
    risk_amount = capital * risk_pct
    stop_distance = abs(entry_price - stop_loss_price)

    if stop_distance == 0:
        return 0

    size = risk_amount / stop_distance
    return size


# 範例
capital = 10000  # $10,000
risk_pct = 0.02  # 2%
entry = 50000    # BTC $50,000
stop = 49000     # 止損 $49,000

size = fixed_risk_size(capital, risk_pct, entry, stop)
# risk_amount = 10000 × 0.02 = 200
# stop_distance = 50000 - 49000 = 1000
# size = 200 / 1000 = 0.2 BTC
```

### 考慮槓桿

```python
def fixed_risk_size_leveraged(
    capital: float,
    risk_pct: float,
    entry_price: float,
    stop_loss_price: float,
    leverage: float
) -> tuple:
    """
    考慮槓桿的部位大小

    Returns:
        (size, margin_required)
    """
    # 基本部位大小
    base_size = fixed_risk_size(capital, risk_pct, entry_price, stop_loss_price)

    # 所需保證金
    position_value = base_size * entry_price
    margin_required = position_value / leverage

    # 檢查保證金是否足夠
    if margin_required > capital:
        # 調整部位大小
        max_position_value = capital * leverage
        base_size = max_position_value / entry_price

    return base_size, margin_required
```

## ATR 基準法

### 原理

使用 ATR 作為止損距離，自動適應市場波動。

### 實作

```python
def atr_based_size(
    capital: float,
    risk_pct: float,
    entry_price: float,
    atr: float,
    atr_multiplier: float = 2.0
) -> tuple:
    """
    ATR 基準部位大小

    Args:
        capital: 總資金
        risk_pct: 風險比例
        entry_price: 入場價格
        atr: ATR 值
        atr_multiplier: ATR 倍數

    Returns:
        (size, stop_loss_price)
    """
    risk_amount = capital * risk_pct
    stop_distance = atr * atr_multiplier

    size = risk_amount / stop_distance
    stop_loss = entry_price - stop_distance  # 做多

    return size, stop_loss


# 範例
capital = 10000
risk_pct = 0.02
entry = 50000
atr = 500  # ATR = $500

size, stop = atr_based_size(capital, risk_pct, entry, atr, 2.0)
# stop_distance = 500 × 2 = 1000
# size = 200 / 1000 = 0.2 BTC
# stop = 50000 - 1000 = 49000
```

## Kelly Criterion

### 公式

```
f* = (b × p - q) / b

其中：
f* = 最佳投入比例
b = 盈虧比 (avg_win / avg_loss)
p = 勝率
q = 敗率 = 1 - p
```

### 實作

```python
def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float
) -> float:
    """
    計算 Kelly 比例

    Args:
        win_rate: 勝率 (0-1)
        avg_win: 平均獲利金額
        avg_loss: 平均虧損金額（正數）

    Returns:
        kelly: 最佳投入比例
    """
    if avg_loss == 0:
        return 0

    b = avg_win / avg_loss  # 盈虧比
    p = win_rate
    q = 1 - p

    kelly = (b * p - q) / b
    return max(0, kelly)


def fractional_kelly(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25
) -> float:
    """
    分數 Kelly（推薦）

    使用 1/4 或 1/2 Kelly 降低風險
    """
    full_kelly = kelly_fraction(win_rate, avg_loss, avg_loss)
    return full_kelly * fraction


# 範例
win_rate = 0.55
avg_win = 300
avg_loss = 200

full_kelly = kelly_fraction(win_rate, avg_win, avg_loss)
# b = 300/200 = 1.5
# f* = (1.5 × 0.55 - 0.45) / 1.5 = 0.25 = 25%

# 使用 1/4 Kelly
safe_kelly = fractional_kelly(win_rate, avg_win, avg_loss, 0.25)
# safe_kelly ≈ 6.25%
```

### 為什麼用分數 Kelly

| Kelly 比例 | 風險 | 報酬 | 推薦度 |
|------------|------|------|--------|
| Full Kelly | 高 | 理論最高 | 不推薦 |
| 1/2 Kelly | 中高 | 略低 | 激進 |
| 1/4 Kelly | 中 | 適中 | 推薦 |
| 1/8 Kelly | 低 | 較低 | 保守 |

## 波動率調整

### 原理

波動率高時減少部位，波動率低時增加部位。

### 實作

```python
def volatility_adjusted_size(
    base_size: float,
    current_volatility: float,
    avg_volatility: float,
    max_adjustment: float = 0.5
) -> float:
    """
    波動率調整部位

    Args:
        base_size: 基礎部位大小
        current_volatility: 當前波動率
        avg_volatility: 平均波動率
        max_adjustment: 最大調整幅度

    Returns:
        adjusted_size: 調整後部位
    """
    vol_ratio = avg_volatility / current_volatility

    # 限制調整幅度
    vol_ratio = max(1 - max_adjustment, min(1 + max_adjustment, vol_ratio))

    return base_size * vol_ratio


# 範例
base_size = 0.2
current_vol = 0.05  # 5% 日波動
avg_vol = 0.03      # 平均 3%

adjusted = volatility_adjusted_size(base_size, current_vol, avg_vol)
# vol_ratio = 0.03 / 0.05 = 0.6
# adjusted = 0.2 × 0.6 = 0.12 BTC（減少部位）
```

## 實務建議

### 風險比例選擇

| 交易者類型 | 建議風險 | 理由 |
|------------|----------|------|
| 新手 | 0.5-1% | 學習期間 |
| 一般 | 1-2% | 平衡風險報酬 |
| 專業 | 2-3% | 經驗豐富 |
| 激進 | 3-5% | 高風險高報酬 |

### 檢查清單

- [ ] 單筆風險 <= 2%
- [ ] 考慮槓桿後的保證金充足
- [ ] 止損在強平價之前
- [ ] 總持倉風險 <= 10%
- [ ] 相關資產合併計算
