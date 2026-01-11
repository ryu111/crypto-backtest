"""
永續合約計算模組

提供永續合約特有的計算功能：
- 資金費率計算（每 8 小時結算）
- 強平價格計算
- 保證金追蹤
- Mark Price 計算
- 未實現盈虧計算

參考：
- .claude/skills/永續合約/SKILL.md
- .claude/skills/回測核心/references/perpetual-mechanics.md
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@dataclass
class PerpetualPosition:
    """永續合約倉位"""

    entry_price: float
    size: float  # 正數=多，負數=空
    leverage: int
    entry_time: datetime
    margin: float  # 保證金
    unrealized_pnl: float = 0.0
    total_funding_paid: float = 0.0

    @property
    def direction(self) -> int:
        """倉位方向：1=多，-1=空，0=無"""
        return np.sign(self.size)

    @property
    def notional_value(self) -> float:
        """名義價值"""
        return abs(self.size) * self.entry_price

    @property
    def is_long(self) -> bool:
        """是否做多"""
        return self.size > 0

    @property
    def is_short(self) -> bool:
        """是否做空"""
        return self.size < 0


class PerpetualCalculator:
    """
    永續合約計算器

    提供所有永續合約相關的數學計算功能。

    參數：
        maintenance_margin_rate: 維持保證金率（預設 0.5%）
        funding_interval_hours: 資金費率結算週期（預設 8 小時）
    """

    def __init__(
        self,
        maintenance_margin_rate: float = 0.005,
        funding_interval_hours: int = 8
    ):
        self.maintenance_margin_rate = maintenance_margin_rate
        self.funding_interval_hours = funding_interval_hours
        self.funding_periods_per_year = (365 * 24) // funding_interval_hours

    # ===== 資金費率計算 =====

    def calculate_funding_cost(
        self,
        position_value: float,
        funding_rate: float,
        direction: int
    ) -> float:
        """
        計算單次資金費率成本

        參數：
            position_value: 持倉價值（USDT）
            funding_rate: 資金費率（如 0.0001 = 0.01%）
            direction: 1=做多，-1=做空

        回傳：
            cost: 正數=支付，負數=收取

        範例：
            >>> calc = PerpetualCalculator()
            >>> calc.calculate_funding_cost(10000, 0.0001, 1)
            1.0  # 做多支付 1 USDT
        """
        return position_value * funding_rate * direction

    def calculate_total_funding(
        self,
        trades: pd.DataFrame,
        funding_rates: pd.DataFrame
    ) -> float:
        """
        計算持倉期間總資金費率成本

        參數：
            trades: 交易記錄，需包含欄位：
                - entry_time: 入場時間
                - exit_time: 出場時間
                - position_value: 持倉價值
                - direction: 方向（1=多，-1=空）
            funding_rates: 資金費率記錄，需包含欄位：
                - timestamp: 結算時間
                - rate: 費率

        回傳：
            total_cost: 總資金費率成本
        """
        total_cost = 0.0

        for _, trade in trades.iterrows():
            # 找出該交易期間內的所有資金費率結算點
            mask = (
                (funding_rates['timestamp'] >= trade['entry_time']) &
                (funding_rates['timestamp'] <= trade['exit_time'])
            )

            period_rates = funding_rates[mask]

            for _, funding in period_rates.iterrows():
                cost = self.calculate_funding_cost(
                    trade['position_value'],
                    funding['rate'],
                    trade['direction']
                )
                total_cost += cost

        return total_cost

    def annualized_funding_rate(self, avg_rate: float) -> float:
        """
        計算年化資金費率影響

        參數：
            avg_rate: 平均資金費率

        回傳：
            annualized_rate: 年化費率

        範例：
            >>> calc = PerpetualCalculator()
            >>> calc.annualized_funding_rate(0.0001)
            0.1095  # 10.95% 年化
        """
        return avg_rate * self.funding_periods_per_year

    def apply_funding_to_equity(
        self,
        equity_curve: pd.Series,
        positions: pd.Series,
        funding_rates: pd.DataFrame,
        position_sizes: pd.Series
    ) -> pd.Series:
        """
        在權益曲線中應用資金費率

        參數：
            equity_curve: 權益曲線
            positions: 持倉方向（1=多，-1=空，0=無）
            funding_rates: 資金費率記錄
            position_sizes: 持倉價值

        回傳：
            adjusted_equity: 調整後的權益曲線
        """
        adjusted_equity = equity_curve.copy()

        for ts in funding_rates['timestamp']:
            if ts in equity_curve.index:
                pos_direction = positions.loc[ts]

                if pos_direction != 0:
                    rate = funding_rates.loc[
                        funding_rates['timestamp'] == ts, 'rate'
                    ].iloc[0]

                    pos_value = position_sizes.loc[ts]
                    cost = self.calculate_funding_cost(pos_value, rate, pos_direction)

                    # 從該時間點開始扣除成本
                    adjusted_equity.loc[ts:] -= cost

        return adjusted_equity

    # ===== 保證金計算 =====

    def calculate_initial_margin(
        self,
        position_size: float,
        entry_price: float,
        leverage: int
    ) -> float:
        """
        計算初始保證金

        參數：
            position_size: 倉位大小（合約數量）
            entry_price: 入場價格
            leverage: 槓桿倍數

        回傳：
            initial_margin: 所需初始保證金

        範例：
            >>> calc = PerpetualCalculator()
            >>> calc.calculate_initial_margin(1, 50000, 10)
            5000.0  # 1 BTC @ $50k，10x 槓桿需 $5k 保證金
        """
        notional_value = abs(position_size) * entry_price
        return notional_value / leverage

    def calculate_margin_ratio(
        self,
        equity: float,
        position_value: float
    ) -> float:
        """
        計算保證金率

        參數：
            equity: 當前權益
            position_value: 持倉價值

        回傳：
            margin_ratio: 保證金率

        範例：
            >>> calc = PerpetualCalculator()
            >>> calc.calculate_margin_ratio(5500, 50000)
            0.11  # 11% 保證金率
        """
        if position_value == 0:
            return float('inf')
        return equity / position_value

    def calculate_available_margin(
        self,
        total_equity: float,
        used_margin: float
    ) -> float:
        """
        計算可用保證金

        參數：
            total_equity: 總權益
            used_margin: 已使用保證金

        回傳：
            available_margin: 可用保證金
        """
        return total_equity - used_margin

    # ===== 強平計算 =====

    def calculate_liquidation_price(
        self,
        entry_price: float,
        leverage: int,
        direction: int,
        maintenance_margin_rate: Optional[float] = None
    ) -> float:
        """
        計算強平價格

        參數：
            entry_price: 入場價格
            leverage: 槓桿倍數
            direction: 1=做多，-1=做空
            maintenance_margin_rate: 維持保證金率（可選，使用預設值）

        回傳：
            liquidation_price: 強平價格

        範例：
            >>> calc = PerpetualCalculator()
            >>> calc.calculate_liquidation_price(50000, 10, 1)
            45250.0  # 做多在 $45,250 爆倉（跌 9.5%）

            >>> calc.calculate_liquidation_price(50000, 10, -1)
            54750.0  # 做空在 $54,750 爆倉（漲 9.5%）
        """
        mmr = maintenance_margin_rate or self.maintenance_margin_rate

        if direction == 1:  # 做多
            # 強平價格 = 入場價 × (1 - 1/槓桿 + 維持保證金率)
            liq_price = entry_price * (1 - 1/leverage + mmr)
        else:  # 做空
            # 強平價格 = 入場價 × (1 + 1/槓桿 - 維持保證金率)
            liq_price = entry_price * (1 + 1/leverage - mmr)

        return liq_price

    def check_liquidation(
        self,
        current_price: float,
        entry_price: float,
        leverage: int,
        direction: int,
        maintenance_margin_rate: Optional[float] = None
    ) -> bool:
        """
        檢查是否觸發強平

        參數：
            current_price: 當前價格
            entry_price: 入場價格
            leverage: 槓桿倍數
            direction: 1=做多，-1=做空
            maintenance_margin_rate: 維持保證金率（可選）

        回傳：
            is_liquidated: 是否已爆倉

        範例：
            >>> calc = PerpetualCalculator()
            >>> calc.check_liquidation(45000, 50000, 10, 1)
            True  # 做多已爆倉
        """
        liq_price = self.calculate_liquidation_price(
            entry_price,
            leverage,
            direction,
            maintenance_margin_rate
        )

        if direction == 1:  # 做多
            return current_price <= liq_price
        else:  # 做空
            return current_price >= liq_price

    def calculate_liquidation_distance(
        self,
        current_price: float,
        entry_price: float,
        leverage: int,
        direction: int
    ) -> Tuple[float, float]:
        """
        計算距離強平的距離

        參數：
            current_price: 當前價格
            entry_price: 入場價格
            leverage: 槓桿倍數
            direction: 1=做多，-1=做空

        回傳：
            (distance_percent, distance_price): 距離百分比和絕對價格

        範例：
            >>> calc = PerpetualCalculator()
            >>> calc.calculate_liquidation_distance(48000, 50000, 10, 1)
            (-5.73, -2750.0)  # 距離強平還有 5.73%（$2,750）
        """
        liq_price = self.calculate_liquidation_price(
            entry_price,
            leverage,
            direction
        )

        distance_price = liq_price - current_price if direction == 1 else current_price - liq_price
        distance_percent = (distance_price / current_price) * 100

        return distance_percent, distance_price

    # ===== 盈虧計算 =====

    def calculate_unrealized_pnl(
        self,
        entry_price: float,
        mark_price: float,
        size: float,
        direction: int
    ) -> float:
        """
        計算未實現盈虧

        參數：
            entry_price: 入場價格
            mark_price: 標記價格（Mark Price）
            size: 倉位大小（合約數量）
            direction: 1=做多，-1=做空

        回傳：
            unrealized_pnl: 未實現盈虧（USDT）

        範例：
            >>> calc = PerpetualCalculator()
            >>> calc.calculate_unrealized_pnl(50000, 52000, 1, 1)
            2000.0  # 做多 1 BTC，浮盈 $2,000

            >>> calc.calculate_unrealized_pnl(50000, 52000, 1, -1)
            -2000.0  # 做空 1 BTC，浮虧 $2,000
        """
        price_diff = mark_price - entry_price
        pnl = size * price_diff * direction
        return pnl

    def calculate_realized_pnl(
        self,
        entry_price: float,
        exit_price: float,
        size: float,
        direction: int
    ) -> float:
        """
        計算已實現盈虧

        參數：
            entry_price: 入場價格
            exit_price: 出場價格
            size: 倉位大小
            direction: 1=做多，-1=做空

        回傳：
            realized_pnl: 已實現盈虧
        """
        return self.calculate_unrealized_pnl(entry_price, exit_price, size, direction)

    def calculate_pnl_percentage(
        self,
        pnl: float,
        margin: float
    ) -> float:
        """
        計算盈虧百分比

        參數：
            pnl: 盈虧金額
            margin: 保證金

        回傳：
            pnl_percentage: 盈虧百分比

        範例：
            >>> calc = PerpetualCalculator()
            >>> calc.calculate_pnl_percentage(500, 5000)
            10.0  # 10% ROI
        """
        if margin == 0:
            return 0.0
        return (pnl / margin) * 100

    # ===== Mark Price 計算 =====

    def calculate_mark_price(
        self,
        spot_price: float,
        perp_price: float,
        window_size: int = 30
    ) -> float:
        """
        計算標記價格（簡化版）

        Mark Price = 現貨指數價格 + 移動平均基差

        參數：
            spot_price: 現貨指數價格
            perp_price: 永續合約價格
            window_size: 移動平均窗口（分鐘）

        回傳：
            mark_price: 標記價格

        注意：
            實際交易所的 Mark Price 計算更複雜，這裡提供簡化版本。
            生產環境應使用交易所提供的 Mark Price。
        """
        basis = perp_price - spot_price
        # 實際應該用移動平均，這裡簡化為使用當前基差
        mark_price = spot_price + basis
        return mark_price

    def calculate_basis(
        self,
        perp_price: float,
        spot_price: float
    ) -> Tuple[float, float]:
        """
        計算基差

        參數：
            perp_price: 永續合約價格
            spot_price: 現貨價格

        回傳：
            (basis_abs, basis_percent): 絕對基差和百分比基差

        範例：
            >>> calc = PerpetualCalculator()
            >>> calc.calculate_basis(50500, 50000)
            (500.0, 1.0)  # 溢價 $500（1%）
        """
        basis_abs = perp_price - spot_price
        basis_percent = (basis_abs / spot_price) * 100
        return basis_abs, basis_percent

    # ===== 風險指標 =====

    def calculate_effective_leverage(
        self,
        position_value: float,
        equity: float
    ) -> float:
        """
        計算有效槓桿

        參數：
            position_value: 持倉價值
            equity: 當前權益

        回傳：
            effective_leverage: 有效槓桿

        範例：
            >>> calc = PerpetualCalculator()
            >>> calc.calculate_effective_leverage(50000, 5500)
            9.09  # 有效槓桿 ~9x
        """
        if equity == 0:
            return 0.0
        return position_value / equity

    def calculate_bankruptcy_price(
        self,
        entry_price: float,
        leverage: int,
        direction: int
    ) -> float:
        """
        計算破產價格（100% 虧損）

        參數：
            entry_price: 入場價格
            leverage: 槓桿倍數
            direction: 1=做多，-1=做空

        回傳：
            bankruptcy_price: 破產價格

        範例：
            >>> calc = PerpetualCalculator()
            >>> calc.calculate_bankruptcy_price(50000, 10, 1)
            45000.0  # 做多在 $45k 破產（跌 10%）
        """
        if direction == 1:  # 做多
            return entry_price * (1 - 1/leverage)
        else:  # 做空
            return entry_price * (1 + 1/leverage)

    def estimate_max_position_size(
        self,
        available_capital: float,
        price: float,
        leverage: int,
        fee_rate: float = 0.0004
    ) -> float:
        """
        估算最大可開倉位大小

        參數：
            available_capital: 可用資金
            price: 當前價格
            leverage: 槓桿倍數
            fee_rate: 交易費率（預設 0.04%）

        回傳：
            max_size: 最大倉位大小（合約數量）

        範例：
            >>> calc = PerpetualCalculator()
            >>> calc.estimate_max_position_size(10000, 50000, 10)
            1.999  # 約可開 2 BTC
        """
        # 扣除手續費後的可用資金
        usable_capital = available_capital / (1 + fee_rate)

        # 計算最大名義價值
        max_notional = usable_capital * leverage

        # 計算合約數量
        max_size = max_notional / price

        return max_size


class PerpetualRiskMonitor:
    """
    永續合約風險監控器

    即時監控持倉風險，發出警告信號。
    """

    def __init__(
        self,
        warning_threshold: float = 0.02,  # 距離強平 2%
        critical_threshold: float = 0.01   # 距離強平 1%
    ):
        self.calculator = PerpetualCalculator()
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def assess_risk_level(
        self,
        position: PerpetualPosition,
        current_price: float
    ) -> str:
        """
        評估風險等級

        參數：
            position: 當前倉位
            current_price: 當前價格

        回傳：
            risk_level: "safe" | "warning" | "critical" | "liquidated"
        """
        # 檢查是否已爆倉
        is_liquidated = self.calculator.check_liquidation(
            current_price,
            position.entry_price,
            position.leverage,
            position.direction
        )

        if is_liquidated:
            return "liquidated"

        # 計算距離強平的距離
        distance_pct, _ = self.calculator.calculate_liquidation_distance(
            current_price,
            position.entry_price,
            position.leverage,
            position.direction
        )

        distance_pct = abs(distance_pct) / 100  # 轉為小數

        if distance_pct <= self.critical_threshold:
            return "critical"
        elif distance_pct <= self.warning_threshold:
            return "warning"
        else:
            return "safe"

    def generate_risk_report(
        self,
        position: PerpetualPosition,
        current_price: float
    ) -> dict:
        """
        生成風險報告

        參數：
            position: 當前倉位
            current_price: 當前價格

        回傳：
            risk_report: 包含各項風險指標的字典
        """
        liq_price = self.calculator.calculate_liquidation_price(
            position.entry_price,
            position.leverage,
            position.direction
        )

        distance_pct, distance_price = self.calculator.calculate_liquidation_distance(
            current_price,
            position.entry_price,
            position.leverage,
            position.direction
        )

        unrealized_pnl = self.calculator.calculate_unrealized_pnl(
            position.entry_price,
            current_price,
            position.size,
            position.direction
        )

        margin_ratio = self.calculator.calculate_margin_ratio(
            position.margin + unrealized_pnl,
            abs(position.size) * current_price
        )

        risk_level = self.assess_risk_level(position, current_price)

        return {
            'risk_level': risk_level,
            'liquidation_price': liq_price,
            'distance_to_liquidation_pct': distance_pct,
            'distance_to_liquidation_price': distance_price,
            'margin_ratio': margin_ratio,
            'unrealized_pnl': unrealized_pnl,
            'current_price': current_price,
            'entry_price': position.entry_price
        }
