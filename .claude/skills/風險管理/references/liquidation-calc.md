# 強平計算

永續合約的強制平倉價格計算與風險管理。

## 強平機制概述

```
強平觸發條件：
保證金餘額 ≤ 維持保證金

保證金餘額 = 初始保證金 + 未實現損益
維持保證金 = 部位價值 × 維持保證金率 (MMR)
```

## 隔離保證金模式

### 做多強平價格

```
強平價格 (Long) = 入場價 × (1 - 1/槓桿 + MMR)
```

**推導：**
```
設：
- Entry = 入場價
- Liq = 強平價
- Leverage = 槓桿倍數
- MMR = 維持保證金率

初始保證金 = Entry × Size / Leverage
未實現損益 = (Liq - Entry) × Size  （做多時 Liq < Entry 為虧損）
維持保證金 = Liq × Size × MMR

強平條件：初始保證金 + 未實現損益 = 維持保證金

Entry × Size / Leverage + (Liq - Entry) × Size = Liq × Size × MMR

Entry / Leverage + Liq - Entry = Liq × MMR
Entry / Leverage - Entry = Liq × MMR - Liq
Entry × (1/Leverage - 1) = Liq × (MMR - 1)
Liq = Entry × (1/Leverage - 1) / (MMR - 1)
Liq = Entry × (1 - 1/Leverage + MMR)  （簡化後）
```

### 做空強平價格

```
強平價格 (Short) = 入場價 × (1 + 1/槓桿 - MMR)
```

### Python 實作

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class LiquidationResult:
    """強平計算結果"""
    liquidation_price: float
    bankruptcy_price: float
    distance_percent: float  # 距離強平的百分比
    safe_stop_loss: float    # 安全止損價
    maintenance_margin: float
    initial_margin: float

def calculate_liquidation_price(
    entry_price: float,
    leverage: float,
    position_side: str,  # 'long' or 'short'
    mmr: float = 0.004,  # 維持保證金率（Binance 預設 0.4%）
    safety_buffer: float = 0.02  # 安全緩衝 2%
) -> LiquidationResult:
    """
    計算隔離保證金強平價格

    Args:
        entry_price: 入場價格
        leverage: 槓桿倍數
        position_side: 'long' 或 'short'
        mmr: 維持保證金率
        safety_buffer: 止損安全緩衝

    Returns:
        LiquidationResult
    """
    if position_side == 'long':
        # 做多強平價
        liquidation_price = entry_price * (1 - 1/leverage + mmr)
        # 破產價（保證金歸零）
        bankruptcy_price = entry_price * (1 - 1/leverage)
        # 安全止損（在強平前觸發）
        safe_stop_loss = liquidation_price * (1 + safety_buffer)

    elif position_side == 'short':
        # 做空強平價
        liquidation_price = entry_price * (1 + 1/leverage - mmr)
        # 破產價
        bankruptcy_price = entry_price * (1 + 1/leverage)
        # 安全止損
        safe_stop_loss = liquidation_price * (1 - safety_buffer)

    else:
        raise ValueError("position_side must be 'long' or 'short'")

    # 距離強平的百分比
    distance_percent = abs(entry_price - liquidation_price) / entry_price * 100

    # 保證金計算（假設部位價值為 1）
    initial_margin = entry_price / leverage
    maintenance_margin = entry_price * mmr

    return LiquidationResult(
        liquidation_price=liquidation_price,
        bankruptcy_price=bankruptcy_price,
        distance_percent=distance_percent,
        safe_stop_loss=safe_stop_loss,
        maintenance_margin=maintenance_margin,
        initial_margin=initial_margin
    )
```

### 強平價格範例

| 入場價 | 槓桿 | 方向 | MMR | 強平價 | 距離 |
|--------|------|------|-----|--------|------|
| $50,000 | 10x | Long | 0.4% | $45,200 | 9.6% |
| $50,000 | 20x | Long | 0.4% | $47,700 | 4.6% |
| $50,000 | 50x | Long | 0.4% | $49,200 | 1.6% |
| $50,000 | 10x | Short | 0.4% | $54,800 | 9.6% |
| $50,000 | 20x | Short | 0.4% | $52,300 | 4.6% |

## 維持保證金率 (MMR)

### 階梯式 MMR

**大部位需要更高 MMR（降低交易所風險）**

```python
# Binance BTCUSDT 階梯式 MMR（2024 參考值）
BINANCE_MMR_TIERS = [
    {'max_position': 50_000, 'mmr': 0.004, 'max_leverage': 125},
    {'max_position': 250_000, 'mmr': 0.005, 'max_leverage': 100},
    {'max_position': 1_000_000, 'mmr': 0.01, 'max_leverage': 50},
    {'max_position': 5_000_000, 'mmr': 0.025, 'max_leverage': 20},
    {'max_position': 20_000_000, 'mmr': 0.05, 'max_leverage': 10},
    {'max_position': 50_000_000, 'mmr': 0.10, 'max_leverage': 5},
    {'max_position': 100_000_000, 'mmr': 0.125, 'max_leverage': 4},
    {'max_position': 200_000_000, 'mmr': 0.15, 'max_leverage': 3},
    {'max_position': float('inf'), 'mmr': 0.25, 'max_leverage': 2},
]

def get_mmr_for_position(position_value: float, tiers: list = BINANCE_MMR_TIERS) -> dict:
    """
    根據部位大小獲取 MMR

    Args:
        position_value: 部位價值 (USDT)
        tiers: MMR 階梯表
    """
    for tier in tiers:
        if position_value <= tier['max_position']:
            return {
                'mmr': tier['mmr'],
                'max_leverage': tier['max_leverage'],
                'tier_limit': tier['max_position']
            }

    return tiers[-1]
```

### 交易所 MMR 比較

| 交易所 | 最低 MMR | 最高槓桿 | 特點 |
|--------|----------|----------|------|
| Binance | 0.4% | 125x | 階梯式，大戶 MMR 高 |
| Bybit | 0.5% | 100x | 類似 Binance |
| OKX | 0.4% | 125x | 有全倉模式 |
| Hyperliquid | 1.0% | 50x | 較保守 |

## 全倉保證金模式

### 概念差異

```
隔離保證金：每個倉位獨立計算，虧損上限 = 該倉位保證金
全倉保證金：所有倉位共享保證金，虧損可能影響其他倉位
```

### 全倉強平公式

```python
def cross_margin_liquidation(
    positions: list,  # [{'side', 'size', 'entry', 'mark'}]
    wallet_balance: float,
    mmr: float = 0.004
) -> dict:
    """
    全倉保證金強平計算

    更複雜，需要考慮：
    1. 所有倉位的未實現損益
    2. 總維持保證金需求
    """
    total_unrealized_pnl = 0
    total_maintenance_margin = 0
    total_position_value = 0

    for pos in positions:
        position_value = pos['size'] * pos['mark']
        total_position_value += position_value

        if pos['side'] == 'long':
            unrealized = (pos['mark'] - pos['entry']) * pos['size']
        else:
            unrealized = (pos['entry'] - pos['mark']) * pos['size']

        total_unrealized_pnl += unrealized
        total_maintenance_margin += position_value * mmr

    # 保證金餘額
    margin_balance = wallet_balance + total_unrealized_pnl

    # 保證金率
    margin_ratio = margin_balance / total_maintenance_margin if total_maintenance_margin > 0 else float('inf')

    # 強平觸發：margin_balance <= total_maintenance_margin
    is_liquidatable = margin_balance <= total_maintenance_margin

    return {
        'margin_balance': margin_balance,
        'maintenance_margin': total_maintenance_margin,
        'margin_ratio': margin_ratio,
        'is_liquidatable': is_liquidatable,
        'buffer': margin_balance - total_maintenance_margin
    }
```

## 破產價格 vs 強平價格

```
破產價格：保證金 = 0（完全虧光）
強平價格：保證金 = 維持保證金（提前強平保護交易所）

強平價格更接近入場價！
```

### 視覺化比較

```
做多 $50,000，10x 槓桿：

入場價      $50,000  ────────────────────
                     │
強平價      $45,200  ──── MMR = 0.4%
                     │ ← 這段由交易所承擔風險
破產價      $45,000  ──── 保證金歸零
                     │
價格下跌    ▼
```

## 安全距離計算

### 止損必須在強平前觸發

```python
def calculate_safe_stop_loss(
    entry_price: float,
    leverage: float,
    position_side: str,
    mmr: float = 0.004,
    safety_margin: float = 0.02  # 2% 安全邊際
) -> dict:
    """
    計算安全止損價位

    止損必須在強平價之前觸發，留有安全邊際
    """
    liq = calculate_liquidation_price(entry_price, leverage, position_side, mmr)
    liq_price = liq.liquidation_price

    if position_side == 'long':
        # 做多：止損要高於強平價
        safe_stop = liq_price * (1 + safety_margin)
        max_loss_percent = (entry_price - safe_stop) / entry_price * 100
    else:
        # 做空：止損要低於強平價
        safe_stop = liq_price * (1 - safety_margin)
        max_loss_percent = (safe_stop - entry_price) / entry_price * 100

    return {
        'entry_price': entry_price,
        'liquidation_price': liq_price,
        'safe_stop_loss': safe_stop,
        'safety_margin': safety_margin,
        'max_loss_percent': abs(max_loss_percent),
        'max_loss_with_leverage': abs(max_loss_percent) * leverage
    }
```

### 安全檢查

```python
def validate_stop_loss(
    entry_price: float,
    stop_loss: float,
    leverage: float,
    position_side: str,
    mmr: float = 0.004
) -> dict:
    """
    驗證止損是否安全（在強平前觸發）
    """
    liq = calculate_liquidation_price(entry_price, leverage, position_side, mmr)
    liq_price = liq.liquidation_price

    if position_side == 'long':
        is_safe = stop_loss > liq_price
        distance_to_liq = (stop_loss - liq_price) / liq_price * 100
    else:
        is_safe = stop_loss < liq_price
        distance_to_liq = (liq_price - stop_loss) / liq_price * 100

    return {
        'stop_loss': stop_loss,
        'liquidation_price': liq_price,
        'is_safe': is_safe,
        'distance_to_liquidation': distance_to_liq,
        'warning': None if is_safe else '⚠️ 止損在強平價之後，可能無法觸發！'
    }
```

## 槓桿與風險關係

### 槓桿 vs 強平距離

| 槓桿 | 做多強平距離 | 最大虧損/保證金 |
|------|--------------|-----------------|
| 5x | 19.6% | 100% |
| 10x | 9.6% | 100% |
| 20x | 4.6% | 100% |
| 50x | 1.6% | 100% |
| 100x | 0.6% | 100% |

### 槓桿選擇建議

```python
def recommend_leverage(
    expected_volatility: float,  # 預期波動率（如 5% 日內波動）
    stop_loss_percent: float,    # 計畫止損百分比
    safety_factor: float = 2.0   # 安全係數
) -> dict:
    """
    根據波動率推薦槓桿

    原則：強平距離 > 預期波動 × 安全係數
    """
    # 最大槓桿 = 1 / (波動 × 安全係數)
    max_leverage_by_volatility = 1 / (expected_volatility * safety_factor)

    # 最大槓桿 = 1 / 止損百分比（保證止損在強平前）
    max_leverage_by_stop = 1 / stop_loss_percent * 0.9  # 留 10% 緩衝

    recommended = min(max_leverage_by_volatility, max_leverage_by_stop)
    recommended = max(1, min(20, int(recommended)))  # 限制在 1-20x

    return {
        'recommended_leverage': recommended,
        'max_by_volatility': max_leverage_by_volatility,
        'max_by_stop_loss': max_leverage_by_stop,
        'expected_volatility': expected_volatility,
        'stop_loss_percent': stop_loss_percent
    }
```

## 回測中的強平處理

```python
def check_liquidation_in_backtest(
    entry_price: float,
    low_price: float,  # 期間最低價（做多）
    high_price: float,  # 期間最高價（做空）
    leverage: float,
    position_side: str,
    mmr: float = 0.004
) -> dict:
    """
    回測中檢查是否觸發強平
    """
    liq = calculate_liquidation_price(entry_price, leverage, position_side, mmr)
    liq_price = liq.liquidation_price

    if position_side == 'long':
        liquidated = low_price <= liq_price
        extreme_price = low_price
    else:
        liquidated = high_price >= liq_price
        extreme_price = high_price

    if liquidated:
        # 計算實際虧損（假設在強平價成交）
        if position_side == 'long':
            loss_percent = (entry_price - liq_price) / entry_price
        else:
            loss_percent = (liq_price - entry_price) / entry_price

        return {
            'liquidated': True,
            'liquidation_price': liq_price,
            'extreme_price': extreme_price,
            'loss_percent': loss_percent * 100,
            'loss_with_leverage': loss_percent * leverage * 100
        }

    return {
        'liquidated': False,
        'liquidation_price': liq_price,
        'extreme_price': extreme_price,
        'closest_approach': abs(extreme_price - liq_price) / liq_price * 100
    }
```

## 最佳實踐

### 避免強平的原則

1. **槓桿控制**：初學者建議 ≤ 10x
2. **止損必設**：止損必須在強平價之前
3. **安全緩衝**：止損與強平之間保持 ≥ 2% 距離
4. **部位分散**：不要 all-in 單一倉位
5. **監控保證金率**：保持 margin ratio > 200%

### 檢查清單

```python
def pre_trade_safety_check(
    entry_price: float,
    stop_loss: float,
    leverage: float,
    position_side: str,
    account_balance: float,
    position_size: float
) -> dict:
    """
    交易前安全檢查
    """
    checks = []

    # 1. 止損安全性
    sl_check = validate_stop_loss(entry_price, stop_loss, leverage, position_side)
    checks.append({
        'name': '止損安全性',
        'passed': sl_check['is_safe'],
        'detail': sl_check
    })

    # 2. 部位佔比
    position_value = position_size * entry_price
    position_ratio = position_value / leverage / account_balance
    checks.append({
        'name': '部位佔比',
        'passed': position_ratio <= 0.2,  # 建議不超過 20%
        'detail': {'ratio': position_ratio, 'max_recommended': 0.2}
    })

    # 3. 槓桿合理性
    checks.append({
        'name': '槓桿合理性',
        'passed': leverage <= 20,
        'detail': {'leverage': leverage, 'max_recommended': 20}
    })

    all_passed = all(c['passed'] for c in checks)

    return {
        'all_passed': all_passed,
        'checks': checks,
        'recommendation': '可以交易' if all_passed else '請調整參數'
    }
```

## 參考資料

- [Bybit: Liquidation Price Calculation](https://www.bybit.com/en/help-center/article/Liquidation-Price-USDT-Contract/)
- [Binance: Liquidation Protocol](https://www.binance.com/en/support/faq/liquidation-protocol-360033525271)
- [Hyperliquid: Liquidations](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/liquidations)
- [KuCoin: Liquidation FAQ](https://www.kucoin.com/support/26694703491737)
