---
name: risk-management
description: 風險管理系統。部位大小、止損策略、槓桿管理、強平風險。當需要設計風控機制、計算適當部位、管理槓桿風險時使用。
---

# 風險管理

永續合約交易的風險控制和資金管理。

## 部位大小方法

### 方法比較

| 方法 | 公式 | 優點 | 缺點 | 推薦度 |
|------|------|------|------|--------|
| 固定金額 | 每筆固定額度 | 簡單 | 不適應變化 | ⭐⭐ |
| 固定百分比 | 資金 × X% | 簡單 | 不考慮風險 | ⭐⭐⭐ |
| 固定風險 | 風險額 / 止損距離 | 風險固定 | 需要止損 | ⭐⭐⭐⭐ |
| Kelly | 最佳比例公式 | 理論最優 | 過於激進 | ⭐⭐⭐ |
| ATR 基準 | 風險額 / ATR | 適應波動 | 需計算 ATR | ⭐⭐⭐⭐⭐ |

### 固定風險法（推薦）

```python
def fixed_risk_position_size(
    capital,
    risk_per_trade,  # 如 0.02 = 2%
    entry_price,
    stop_loss_price
):
    """
    固定風險部位大小計算

    Args:
        capital: 總資金
        risk_per_trade: 單筆風險比例
        entry_price: 入場價格
        stop_loss_price: 止損價格

    Returns:
        size: 部位大小（合約數量）
    """
    risk_amount = capital * risk_per_trade
    stop_distance = abs(entry_price - stop_loss_price)
    risk_per_unit = stop_distance

    size = risk_amount / risk_per_unit

    return size

# 範例：$10,000 資金，2% 風險，BTC $50,000 進場，$49,000 止損
size = fixed_risk_position_size(10000, 0.02, 50000, 49000)
# size = 200 / 1000 = 0.2 BTC
```

### ATR 基準法

```python
def atr_position_size(
    capital,
    risk_per_trade,
    entry_price,
    atr,
    atr_multiplier=2.0
):
    """
    基於 ATR 的部位大小

    止損距離 = ATR × multiplier
    """
    risk_amount = capital * risk_per_trade
    stop_distance = atr * atr_multiplier
    size = risk_amount / stop_distance

    return size, entry_price - stop_distance  # size, stop_loss_price

# 範例：ATR = 500，使用 2x ATR 止損
size, stop = atr_position_size(10000, 0.02, 50000, 500, 2.0)
# stop_distance = 1000, size = 200 / 1000 = 0.2 BTC
```

### Kelly Criterion

```python
def kelly_fraction(win_rate, avg_win, avg_loss):
    """
    Kelly 公式

    f* = (W/L × p - q) / (W/L)

    其中：
    - p = 勝率
    - q = 敗率 = 1 - p
    - W = 平均獲利
    - L = 平均虧損
    """
    if avg_loss == 0:
        return 0

    win_loss_ratio = avg_win / abs(avg_loss)
    q = 1 - win_rate

    kelly = (win_loss_ratio * win_rate - q) / win_loss_ratio

    return max(0, kelly)

# 範例：55% 勝率，平均獲利 3%，平均虧損 2%
kelly = kelly_fraction(0.55, 0.03, 0.02)
# kelly ≈ 0.325 = 32.5%

# 建議使用 1/4 Kelly 或 1/2 Kelly
conservative_kelly = kelly * 0.25  # ≈ 8%
```

## 止損策略

### 止損類型

| 類型 | 計算 | 優點 | 缺點 |
|------|------|------|------|
| 固定百分比 | 入場價 ×(1-X%) | 簡單 | 不適應波動 |
| ATR 止損 | 入場價 - N×ATR | 適應波動 | 需選 N |
| 結構止損 | 支撐/阻力位 | 有邏輯 | 主觀 |
| 移動止損 | 跟隨最高價 | 保護利潤 | 可能過早 |
| 時間止損 | 持有 N 根 K 線 | 減少暴露 | 可能錯過 |

### ATR 止損實作

```python
def atr_stop_loss(
    entry_price,
    atr,
    direction,  # 1 = 多, -1 = 空
    multiplier=2.0
):
    """ATR 止損計算"""
    stop_distance = atr * multiplier

    if direction == 1:  # 做多
        stop_loss = entry_price - stop_distance
    else:  # 做空
        stop_loss = entry_price + stop_distance

    return stop_loss

# 範例
stop = atr_stop_loss(50000, 500, 1, 2.0)
# stop = 50000 - 1000 = 49000
```

### 移動止損

```python
def trailing_stop(
    highest_price,  # 持倉以來最高價
    atr,
    multiplier=2.0
):
    """移動止損（做多）"""
    return highest_price - atr * multiplier

# 追蹤更新
def update_trailing_stop(current_price, current_stop, atr, multiplier=2.0):
    new_stop = trailing_stop(current_price, atr, multiplier)
    return max(current_stop, new_stop)  # 只能往上調，不能往下
```

## 槓桿管理

### 槓桿風險矩陣

| 槓桿 | 1% 波動影響 | 強平距離 | 風險等級 |
|------|-------------|----------|----------|
| 1x | 1% | 100% | 極低 |
| 3x | 3% | ~33% | 低 |
| 5x | 5% | ~20% | 中 |
| 10x | 10% | ~10% | 高 |
| 20x | 20% | ~5% | 極高 |
| 50x+ | 50%+ | ~2% | 危險 |

### 建議槓桿

| 策略類型 | 建議槓桿 | 理由 |
|----------|----------|------|
| 趨勢跟隨 | 3-5x | 持倉時間長 |
| 波段交易 | 5-10x | 中等持倉 |
| 日內交易 | 10-20x | 短期持倉 |
| 剝頭皮 | 20-50x | 極短持倉 |

### 動態槓桿

```python
def dynamic_leverage(base_leverage, volatility, avg_volatility):
    """
    根據波動率調整槓桿

    波動率高 → 降低槓桿
    波動率低 → 可提高槓桿
    """
    volatility_ratio = volatility / avg_volatility

    if volatility_ratio > 1.5:
        adjusted = base_leverage * 0.5
    elif volatility_ratio > 1.2:
        adjusted = base_leverage * 0.75
    elif volatility_ratio < 0.8:
        adjusted = base_leverage * 1.25
    else:
        adjusted = base_leverage

    return min(adjusted, base_leverage * 1.5)  # 上限
```

## 強平計算

### 強平價格

```python
def liquidation_price(
    entry_price,
    leverage,
    direction,  # 1 = 多, -1 = 空
    maintenance_margin_rate=0.005  # 0.5%
):
    """
    計算強平價格

    做多：當價格跌至此價位時強平
    做空：當價格漲至此價位時強平
    """
    if direction == 1:  # 做多
        liq = entry_price * (1 - 1/leverage + maintenance_margin_rate)
    else:  # 做空
        liq = entry_price * (1 + 1/leverage - maintenance_margin_rate)

    return liq

# 範例：$50,000 做多，10x 槓桿
liq = liquidation_price(50000, 10, 1)
# liq ≈ $45,250（跌 ~9.5% 爆倉）
```

### 安全距離檢查

```python
def check_liquidation_safety(
    entry_price,
    stop_loss,
    leverage,
    direction,
    safety_buffer=0.02  # 2% 安全緩衝
):
    """
    確保止損在強平之前觸發
    """
    liq_price = liquidation_price(entry_price, leverage, direction)

    if direction == 1:  # 做多
        # 止損應該高於強平價格
        safe_stop = liq_price * (1 + safety_buffer)
        is_safe = stop_loss >= safe_stop
    else:  # 做空
        # 止損應該低於強平價格
        safe_stop = liq_price * (1 - safety_buffer)
        is_safe = stop_loss <= safe_stop

    return is_safe, liq_price, safe_stop
```

## 風險限制

### 建議限制

| 限制項目 | 建議值 | 說明 |
|----------|--------|------|
| 單筆風險 | 1-2% | 單一交易最大虧損 |
| 日風險 | 5% | 單日最大虧損 |
| 週風險 | 10% | 單週最大虧損 |
| 最大回撤 | 20% | 觸發暫停 |
| 同時持倉 | 3-5 個 | 分散風險 |
| 相關資產 | 合併計算 | BTC/ETH 相關性高 |

### 風險監控

```python
class RiskMonitor:
    def __init__(self, config):
        self.max_daily_loss = config.get('max_daily_loss', 0.05)
        self.max_weekly_loss = config.get('max_weekly_loss', 0.10)
        self.max_drawdown = config.get('max_drawdown', 0.20)

        self.daily_pnl = 0
        self.weekly_pnl = 0
        self.peak_equity = 0
        self.current_equity = 0

    def update(self, pnl, equity):
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        self.current_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

    def check_limits(self):
        alerts = []

        # 日虧損限制
        if self.daily_pnl < -self.max_daily_loss * self.peak_equity:
            alerts.append('DAILY_LOSS_LIMIT')

        # 週虧損限制
        if self.weekly_pnl < -self.max_weekly_loss * self.peak_equity:
            alerts.append('WEEKLY_LOSS_LIMIT')

        # 最大回撤
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        if drawdown > self.max_drawdown:
            alerts.append('MAX_DRAWDOWN')

        return alerts

    def should_stop_trading(self):
        return len(self.check_limits()) > 0
```

## 風控檢查清單

### 交易前

- [ ] 計算適當部位大小
- [ ] 設定止損位置
- [ ] 確認止損在強平之前
- [ ] 確認不超過日風險限制
- [ ] 確認總持倉風險

### 持倉中

- [ ] 監控未實現盈虧
- [ ] 更新移動止損（如適用）
- [ ] 監控資金費率成本
- [ ] 注意強平風險

### 交易後

- [ ] 記錄交易結果
- [ ] 更新日/週風險統計
- [ ] 檢查是否觸發限制
- [ ] 分析風險管理效果

For 部位大小詳解 → read `references/position-sizing.md`
For 強平計算詳解 → read `references/liquidation-calc.md`
