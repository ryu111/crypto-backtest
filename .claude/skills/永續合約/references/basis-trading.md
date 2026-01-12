# 基差交易

永續合約的基差套利與資金費率策略。

## 基差定義

```
基差 = 永續合約價格 - 現貨價格

正基差（溢價）：永續 > 現貨 → 多頭市場情緒
負基差（折價）：永續 < 現貨 → 空頭市場情緒
```

### 基差來源

| 來源 | 說明 |
|------|------|
| 市場情緒 | 牛市傾向正基差 |
| 資金費率 | 高費率壓縮正基差 |
| 套利活動 | 套利者縮小基差 |
| 流動性差異 | 兩市場流動性不同 |

## Cash-and-Carry 套利

### 策略原理

```
策略：買現貨 + 賣永續（同等數量）

收益來源：
1. 資金費率（做空收取）
2. 基差收斂（溢價消失）

風險：
- Delta Neutral（市場方向無關）
- 但需要管理保證金
```

### 視覺化

```
時間 T0（建倉）：
┌──────────────────┐    ┌──────────────────┐
│  現貨：買入 1 BTC │    │  永續：做空 1 BTC │
│  價格：$50,000    │    │  價格：$50,100    │
└──────────────────┘    └──────────────────┘
         基差 = +$100（溢價）

時間 T1（收取資金費率）：
每 8 小時收取：部位價值 × 資金費率
假設費率 0.01% → 收取 $5.01

時間 T2（平倉）：
現貨價格變為 $48,000
永續價格變為 $48,050（基差縮小為 $50）

損益：
現貨：$48,000 - $50,000 = -$2,000
永續：$50,100 - $48,050 = +$2,050
淨利：+$50（基差收斂） + 累積資金費率
```

### Python 實作

```python
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class CarryTradePosition:
    """Cash-and-Carry 部位"""
    spot_entry: float        # 現貨入場價
    perp_entry: float        # 永續入場價
    position_size: float     # 部位大小（BTC）
    leverage: float          # 永續槓桿
    entry_time: str

@dataclass
class CarryTradeResult:
    """套利結果"""
    spot_pnl: float
    perp_pnl: float
    funding_received: float
    total_pnl: float
    annualized_return: float
    holding_days: int

class CashAndCarryStrategy:
    """Cash-and-Carry 套利策略"""

    def __init__(
        self,
        min_basis_percent: float = 0.001,  # 最小基差 0.1%
        min_funding_rate: float = 0.0001,  # 最小資金費率 0.01%
        max_leverage: float = 3.0
    ):
        self.min_basis_percent = min_basis_percent
        self.min_funding_rate = min_funding_rate
        self.max_leverage = max_leverage

    def check_entry_conditions(
        self,
        spot_price: float,
        perp_price: float,
        funding_rate: float
    ) -> dict:
        """
        檢查進場條件
        """
        basis = perp_price - spot_price
        basis_percent = basis / spot_price

        conditions = {
            'basis_percent': basis_percent,
            'funding_rate': funding_rate,
            'basis_ok': basis_percent >= self.min_basis_percent,
            'funding_ok': funding_rate >= self.min_funding_rate,
            'entry_signal': False,
            'expected_daily_return': 0
        }

        if conditions['basis_ok'] and conditions['funding_ok']:
            conditions['entry_signal'] = True
            # 預期日報酬 = 資金費率 × 3（每日 3 次）
            conditions['expected_daily_return'] = funding_rate * 3

        return conditions

    def calculate_position_size(
        self,
        capital: float,
        spot_price: float,
        leverage: float = None
    ) -> dict:
        """
        計算部位大小

        需要資金：
        - 現貨全額
        - 永續保證金
        """
        if leverage is None:
            leverage = self.max_leverage

        # 假設 50% 資金買現貨，50% 作為永續保證金
        spot_capital = capital * 0.5
        perp_margin = capital * 0.5

        position_size = min(
            spot_capital / spot_price,
            perp_margin * leverage / spot_price
        )

        return {
            'position_size': position_size,
            'spot_cost': position_size * spot_price,
            'perp_margin': position_size * spot_price / leverage,
            'total_capital_used': position_size * spot_price * (1 + 1/leverage),
            'effective_leverage': leverage
        }

    def calculate_pnl(
        self,
        position: CarryTradePosition,
        spot_exit: float,
        perp_exit: float,
        funding_payments: List[float],
        holding_days: int
    ) -> CarryTradeResult:
        """
        計算套利損益
        """
        # 現貨損益
        spot_pnl = (spot_exit - position.spot_entry) * position.position_size

        # 永續損益（做空）
        perp_pnl = (position.perp_entry - perp_exit) * position.position_size

        # 累積資金費率收入
        funding_received = sum(funding_payments)

        # 總損益
        total_pnl = spot_pnl + perp_pnl + funding_received

        # 年化報酬（基於初始資本）
        initial_capital = position.position_size * position.spot_entry * (1 + 1/position.leverage)
        daily_return = total_pnl / initial_capital / holding_days if holding_days > 0 else 0
        annualized_return = daily_return * 365

        return CarryTradeResult(
            spot_pnl=spot_pnl,
            perp_pnl=perp_pnl,
            funding_received=funding_received,
            total_pnl=total_pnl,
            annualized_return=annualized_return,
            holding_days=holding_days
        )
```

### 歷史表現參考

| 年份 | 平均年化 | 最大回撤 | 備註 |
|------|----------|----------|------|
| 2021 | 25-40% | 5% | 牛市高費率 |
| 2022 | 10-20% | 8% | 熊市費率波動 |
| 2023 | 15-25% | 4% | 費率穩定 |
| 2024 | 10-20% | 6% | ETF 影響 |

## 資金費率套利

### 純資金費率策略

```
與 Cash-and-Carry 的差異：
- 不持有現貨
- 使用兩個永續合約對沖（不同交易所）

策略：
- 交易所 A：做空（收費率）
- 交易所 B：做多（付費率較低或收費率）
```

### 費率差異套利

```python
def funding_rate_arbitrage(
    rate_exchange_a: float,
    rate_exchange_b: float,
    min_spread: float = 0.0005  # 最小利差 0.05%
) -> dict:
    """
    跨交易所資金費率套利

    策略：在費率高的交易所做空，費率低的做多
    """
    spread = rate_exchange_a - rate_exchange_b

    if abs(spread) < min_spread:
        return {
            'signal': False,
            'spread': spread,
            'reason': 'spread too small'
        }

    if spread > 0:
        # A 費率高，做空 A，做多 B
        return {
            'signal': True,
            'short_exchange': 'A',
            'long_exchange': 'B',
            'expected_profit_per_8h': spread,
            'annualized': spread * 3 * 365
        }
    else:
        # B 費率高，做空 B，做多 A
        return {
            'signal': True,
            'short_exchange': 'B',
            'long_exchange': 'A',
            'expected_profit_per_8h': abs(spread),
            'annualized': abs(spread) * 3 * 365
        }
```

### 動態對沖

```python
class DynamicFundingArbitrage:
    """動態資金費率套利"""

    def __init__(
        self,
        entry_threshold: float = 0.0005,  # 進場門檻
        exit_threshold: float = 0.0001,   # 出場門檻
        rebalance_threshold: float = 0.02 # 再平衡門檻
    ):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.rebalance_threshold = rebalance_threshold
        self.position = None

    def should_enter(self, funding_rate: float) -> bool:
        """是否應該進場"""
        return funding_rate > self.entry_threshold

    def should_exit(self, funding_rate: float) -> bool:
        """是否應該出場"""
        return funding_rate < self.exit_threshold

    def should_rebalance(
        self,
        spot_value: float,
        perp_value: float
    ) -> bool:
        """
        是否需要再平衡

        當價格大幅變動時，現貨和永續部位會不對等
        """
        if spot_value == 0:
            return False

        imbalance = abs(spot_value - perp_value) / spot_value
        return imbalance > self.rebalance_threshold

    def calculate_rebalance(
        self,
        spot_value: float,
        perp_value: float,
        current_price: float
    ) -> dict:
        """計算再平衡交易"""
        target_value = (spot_value + perp_value) / 2
        spot_adjustment = target_value - spot_value
        perp_adjustment = target_value - perp_value

        return {
            'spot_trade': spot_adjustment / current_price,  # 正=買入, 負=賣出
            'perp_trade': -perp_adjustment / current_price  # 正=做多, 負=做空
        }
```

## 風險管理

### 主要風險

| 風險 | 說明 | 緩解措施 |
|------|------|----------|
| 強平風險 | 永續部位可能被強平 | 低槓桿、監控保證金 |
| 資金費率反轉 | 費率可能轉負 | 設定出場條件 |
| 流動性風險 | 無法以預期價格成交 | 分批進出 |
| 交易對手風險 | 交易所倒閉 | 分散交易所 |
| 基差擴大 | 基差可能先擴大再收斂 | 足夠保證金 |

### 保證金監控

```python
def monitor_margin_health(
    perp_entry: float,
    perp_mark: float,
    position_size: float,
    margin: float,
    leverage: float,
    position_side: str = 'short'
) -> dict:
    """
    監控永續部位的保證金健康度
    """
    # 計算未實現損益
    if position_side == 'short':
        unrealized_pnl = (perp_entry - perp_mark) * position_size
    else:
        unrealized_pnl = (perp_mark - perp_entry) * position_size

    # 保證金餘額
    margin_balance = margin + unrealized_pnl

    # 維持保證金（假設 MMR = 0.4%）
    mmr = 0.004
    maintenance_margin = perp_mark * position_size * mmr

    # 保證金率
    margin_ratio = margin_balance / maintenance_margin if maintenance_margin > 0 else float('inf')

    return {
        'margin_balance': margin_balance,
        'maintenance_margin': maintenance_margin,
        'margin_ratio': margin_ratio,
        'health': 'safe' if margin_ratio > 3 else 'warning' if margin_ratio > 1.5 else 'danger',
        'unrealized_pnl': unrealized_pnl,
        'add_margin_suggested': margin_ratio < 2
    }
```

### 出場條件

```python
def should_exit_carry_trade(
    current_funding_rate: float,
    avg_funding_rate: float,
    basis_percent: float,
    margin_ratio: float,
    holding_days: int
) -> dict:
    """
    判斷是否應該平倉
    """
    exit_reasons = []

    # 1. 資金費率轉負
    if current_funding_rate < -0.0001:
        exit_reasons.append('negative_funding')

    # 2. 平均費率過低
    if avg_funding_rate < 0.0003:
        exit_reasons.append('low_avg_funding')

    # 3. 基差轉負（折價）
    if basis_percent < -0.001:
        exit_reasons.append('negative_basis')

    # 4. 保證金不足
    if margin_ratio < 1.5:
        exit_reasons.append('low_margin')

    # 5. 持有過久（資金效率）
    if holding_days > 30 and avg_funding_rate < 0.0005:
        exit_reasons.append('low_efficiency')

    return {
        'should_exit': len(exit_reasons) > 0,
        'reasons': exit_reasons,
        'urgency': 'high' if 'low_margin' in exit_reasons else 'normal'
    }
```

## 回測框架

```python
def backtest_carry_trade(
    spot_prices: np.ndarray,
    perp_prices: np.ndarray,
    funding_rates: np.ndarray,  # 每 8 小時
    initial_capital: float = 10000,
    leverage: float = 3.0,
    entry_basis: float = 0.001,
    entry_funding: float = 0.0001,
    exit_funding: float = -0.0001
) -> dict:
    """
    Cash-and-Carry 策略回測
    """
    capital = initial_capital
    position = None
    trades = []
    equity_curve = [capital]
    funding_collected = 0

    for i in range(len(spot_prices)):
        spot = spot_prices[i]
        perp = perp_prices[i]
        funding = funding_rates[i] if i < len(funding_rates) else 0

        basis_pct = (perp - spot) / spot

        if position is None:
            # 進場條件
            if basis_pct >= entry_basis and funding >= entry_funding:
                # 建倉
                position_size = capital * 0.5 / spot
                position = {
                    'spot_entry': spot,
                    'perp_entry': perp,
                    'size': position_size,
                    'entry_idx': i
                }
        else:
            # 計算當前損益
            spot_pnl = (spot - position['spot_entry']) * position['size']
            perp_pnl = (position['perp_entry'] - perp) * position['size']

            # 收取資金費率
            funding_payment = position['size'] * perp * funding
            funding_collected += funding_payment

            # 出場條件
            if funding < exit_funding:
                # 平倉
                total_pnl = spot_pnl + perp_pnl + funding_collected
                capital += total_pnl

                trades.append({
                    'entry_idx': position['entry_idx'],
                    'exit_idx': i,
                    'spot_pnl': spot_pnl,
                    'perp_pnl': perp_pnl,
                    'funding_collected': funding_collected,
                    'total_pnl': total_pnl
                })

                position = None
                funding_collected = 0

        # 更新權益曲線
        if position:
            spot_pnl = (spot - position['spot_entry']) * position['size']
            perp_pnl = (position['perp_entry'] - perp) * position['size']
            equity_curve.append(capital + spot_pnl + perp_pnl + funding_collected)
        else:
            equity_curve.append(capital)

    # 計算績效
    equity_curve = np.array(equity_curve)
    returns = np.diff(equity_curve) / equity_curve[:-1]

    return {
        'final_capital': equity_curve[-1],
        'total_return': (equity_curve[-1] - initial_capital) / initial_capital,
        'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(365 * 3) if np.std(returns) > 0 else 0,
        'max_drawdown': np.min(equity_curve / np.maximum.accumulate(equity_curve) - 1),
        'n_trades': len(trades),
        'avg_trade_pnl': np.mean([t['total_pnl'] for t in trades]) if trades else 0,
        'total_funding': sum(t['funding_collected'] for t in trades),
        'trades': trades,
        'equity_curve': equity_curve
    }
```

## 實務考量

### 交易成本

| 成本項目 | 估計值 | 說明 |
|----------|--------|------|
| 現貨手續費 | 0.1% | 買賣各一次 |
| 永續手續費 | 0.04% | Maker/Taker |
| 滑點 | 0.05-0.1% | 視流動性 |
| 資金成本 | 變動 | 穩定幣借貸利率 |

### 最低收益要求

```
年化報酬 > 交易成本 + 資金成本 + 風險溢價

範例：
- 交易成本：0.3%（進出各一次）
- 資金成本：5% 年化
- 風險溢價：5%
- 最低要求：10.3% 年化
```

### ETF 影響

```
BTC ETF 上市後：
- 傳統套利者進入
- Carry 收益下降
- 費率波動減少
- 預期年化：8-15%（vs 之前 20-40%）
```

## 參考資料

- [HighStrike: Perpetual Futures Guide 2025](https://highstrike.com/perpetual-futures/)
- [Gate.io: Funding Rate Arbitrage](https://www.gate.com/learn/articles/perpetual-contract-funding-rate-arbitrage/2166)
- [BIS: Crypto Carry](https://www.bis.org/publ/work1087.pdf)
- [Paradigm: Funding Rate Mechanics](https://www.paradigm.co/blog/funding-rate-mechanics)
