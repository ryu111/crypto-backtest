"""
永續合約計算模組

提供永續合約特有的計算功能：
- 資金費率計算（每 8 小時結算）
- 強平價格計算
- 強平模擬執行
- 保證金追蹤
- Mark Price 計算
- 未實現盈虧計算

參考：
- .claude/skills/永續合約/SKILL.md
- .claude/skills/風險管理/SKILL.md
- .claude/skills/回測核心/references/perpetual-mechanics.md
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class LiquidationType(Enum):
    """強平類型"""
    NONE = "none"
    MARGIN_CALL = "margin_call"      # 追保通知
    PARTIAL = "partial"               # 部分強平
    FULL = "full"                     # 完全強平


@dataclass
class LiquidationEvent:
    """
    強平事件記錄

    記錄強平發生時的所有相關資訊，用於：
    - 回測績效分析
    - 風險報告生成
    - 策略改進參考
    """
    timestamp: datetime
    liquidation_type: LiquidationType
    entry_price: float
    liquidation_price: float
    position_size: float
    direction: int  # 1=多, -1=空
    leverage: int
    margin_lost: float
    penalty_fee: float

    @property
    def total_loss(self) -> float:
        """總損失（保證金 + 罰金）"""
        return self.margin_lost + self.penalty_fee

    def to_dict(self) -> dict:
        """轉為字典"""
        return {
            'timestamp': self.timestamp,
            'type': self.liquidation_type.value,
            'entry_price': self.entry_price,
            'liquidation_price': self.liquidation_price,
            'position_size': self.position_size,
            'direction': self.direction,
            'leverage': self.leverage,
            'margin_lost': self.margin_lost,
            'penalty_fee': self.penalty_fee,
            'total_loss': self.total_loss
        }


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


class LiquidationSimulator:
    """
    強平模擬器

    在回測中模擬交易所的強平機制，提供更真實的績效評估。

    功能：
    - 每根 K 線檢查強平條件
    - 計算強平罰金（通常為維持保證金）
    - 記錄強平事件
    - 修正權益曲線

    參考：
    - .claude/skills/風險管理/SKILL.md
    - .claude/skills/永續合約/SKILL.md

    使用範例：
        simulator = LiquidationSimulator()

        for bar in data.itertuples():
            if position is not None:
                event = simulator.check_and_execute(
                    position, bar.low, bar.high, bar.Index
                )
                if event:
                    # 處理強平事件
                    equity -= event.total_loss
                    position = None
    """

    def __init__(
        self,
        maintenance_margin_rate: float = 0.005,
        liquidation_penalty_rate: float = 0.0075,  # 0.75% 強平罰金
        enable_partial_liquidation: bool = False
    ):
        """
        初始化強平模擬器

        Args:
            maintenance_margin_rate: 維持保證金率（預設 0.5%）
            liquidation_penalty_rate: 強平罰金率（預設 0.75%）
            enable_partial_liquidation: 是否啟用部分強平（預設否）
        """
        self.calculator = PerpetualCalculator(
            maintenance_margin_rate=maintenance_margin_rate
        )
        self.liquidation_penalty_rate = liquidation_penalty_rate
        self.enable_partial_liquidation = enable_partial_liquidation
        self.events: List[LiquidationEvent] = []

    def check_and_execute(
        self,
        position: PerpetualPosition,
        bar_low: float,
        bar_high: float,
        timestamp: datetime
    ) -> Optional[LiquidationEvent]:
        """
        檢查並執行強平

        在每根 K 線中檢查價格是否觸及強平價格。
        對於做多，檢查最低價；對於做空，檢查最高價。

        Args:
            position: 當前持倉
            bar_low: K 線最低價
            bar_high: K 線最高價
            timestamp: K 線時間戳

        Returns:
            LiquidationEvent: 如果觸發強平則回傳事件，否則 None

        範例：
            >>> pos = PerpetualPosition(50000, 1, 10, datetime.now(), 5000)
            >>> sim = LiquidationSimulator()
            >>> event = sim.check_and_execute(pos, 44000, 51000, datetime.now())
            >>> event is not None  # 做多 10x，價格跌破 45250 會強平
            True
        """
        if position.direction == 0:
            return None

        # 計算強平價格
        liq_price = self.calculator.calculate_liquidation_price(
            position.entry_price,
            position.leverage,
            position.direction
        )

        # 檢查是否觸發強平
        is_liquidated = False
        actual_liq_price = liq_price

        if position.direction == 1:  # 做多
            if bar_low <= liq_price:
                is_liquidated = True
                # 實際強平價格可能比強平線更差（跳空或滑點）
                actual_liq_price = min(liq_price, bar_low)
        else:  # 做空
            if bar_high >= liq_price:
                is_liquidated = True
                actual_liq_price = max(liq_price, bar_high)

        if not is_liquidated:
            return None

        # 執行強平
        event = self._execute_liquidation(
            position, actual_liq_price, timestamp
        )

        self.events.append(event)
        logger.warning(
            f"強平觸發: {timestamp}, "
            f"方向={'做多' if position.direction == 1 else '做空'}, "
            f"入場價={position.entry_price:.2f}, "
            f"強平價={actual_liq_price:.2f}, "
            f"損失={event.total_loss:.2f}"
        )

        return event

    def _execute_liquidation(
        self,
        position: PerpetualPosition,
        liquidation_price: float,
        timestamp: datetime
    ) -> LiquidationEvent:
        """
        執行強平並計算損失

        Args:
            position: 被強平的倉位
            liquidation_price: 強平執行價格
            timestamp: 強平時間

        Returns:
            LiquidationEvent: 強平事件
        """
        # 計算保證金損失
        # 強平時通常會損失大部分保證金
        unrealized_pnl = self.calculator.calculate_unrealized_pnl(
            position.entry_price,
            liquidation_price,
            abs(position.size),
            position.direction
        )

        # 保證金損失 = 初始保證金 + 未實現虧損（已經是負數）
        margin_lost = position.margin + unrealized_pnl
        margin_lost = max(margin_lost, 0)  # 保證金損失不能為負

        # 強平罰金
        notional_value = abs(position.size) * liquidation_price
        penalty_fee = notional_value * self.liquidation_penalty_rate

        return LiquidationEvent(
            timestamp=timestamp,
            liquidation_type=LiquidationType.FULL,
            entry_price=position.entry_price,
            liquidation_price=liquidation_price,
            position_size=abs(position.size),
            direction=position.direction,
            leverage=position.leverage,
            margin_lost=margin_lost,
            penalty_fee=penalty_fee
        )

    def simulate_liquidations(
        self,
        data: pd.DataFrame,
        positions: pd.Series,
        entry_prices: pd.Series,
        leverage: int
    ) -> Tuple[pd.Series, List[LiquidationEvent]]:
        """
        對整個回測資料模擬強平

        Args:
            data: OHLCV DataFrame
            positions: 持倉方向序列（1=多, -1=空, 0=無）
            entry_prices: 入場價格序列
            leverage: 槓桿倍數

        Returns:
            (adjusted_positions, events): 調整後的持倉和強平事件列表
        """
        adjusted_positions = positions.copy()
        events = []

        current_entry_price = None
        position_start_idx = None

        for i, idx in enumerate(data.index):
            pos = positions.iloc[i]

            # 新開倉
            if pos != 0 and (i == 0 or positions.iloc[i-1] == 0):
                current_entry_price = entry_prices.iloc[i]
                position_start_idx = i

            # 有持倉，檢查強平
            if pos != 0 and current_entry_price is not None:
                liq_price = self.calculator.calculate_liquidation_price(
                    current_entry_price, leverage, int(pos)
                )

                bar_low = data['low'].iloc[i]
                bar_high = data['high'].iloc[i]

                is_liquidated = False
                if pos == 1 and bar_low <= liq_price:
                    is_liquidated = True
                    actual_liq_price = min(liq_price, bar_low)
                elif pos == -1 and bar_high >= liq_price:
                    is_liquidated = True
                    actual_liq_price = max(liq_price, bar_high)

                if is_liquidated:
                    # 從強平點開始清除持倉
                    adjusted_positions.iloc[i:] = 0

                    # 記錄事件
                    margin = abs(current_entry_price) / leverage
                    timestamp = idx if isinstance(idx, datetime) else data.index[i]

                    event = LiquidationEvent(
                        timestamp=timestamp,
                        liquidation_type=LiquidationType.FULL,
                        entry_price=current_entry_price,
                        liquidation_price=actual_liq_price,
                        position_size=1.0,  # 標準化為 1
                        direction=int(pos),
                        leverage=leverage,
                        margin_lost=margin,
                        penalty_fee=margin * self.liquidation_penalty_rate * leverage
                    )
                    events.append(event)

                    # 重置狀態
                    current_entry_price = None
                    position_start_idx = None

            # 平倉
            if pos == 0 and i > 0 and positions.iloc[i-1] != 0:
                current_entry_price = None
                position_start_idx = None

        return adjusted_positions, events

    def get_statistics(self) -> dict:
        """
        獲取強平統計

        Returns:
            強平統計字典
        """
        if not self.events:
            return {
                'total_liquidations': 0,
                'total_loss': 0.0,
                'total_penalty': 0.0,
                'avg_loss_per_liquidation': 0.0,
                'liquidation_by_direction': {'long': 0, 'short': 0}
            }

        total_loss = sum(e.total_loss for e in self.events)
        total_penalty = sum(e.penalty_fee for e in self.events)

        by_direction = {
            'long': sum(1 for e in self.events if e.direction == 1),
            'short': sum(1 for e in self.events if e.direction == -1)
        }

        return {
            'total_liquidations': len(self.events),
            'total_loss': total_loss,
            'total_penalty': total_penalty,
            'avg_loss_per_liquidation': total_loss / len(self.events),
            'liquidation_by_direction': by_direction
        }

    def clear_events(self):
        """清除強平事件記錄"""
        self.events = []

    def to_dataframe(self) -> pd.DataFrame:
        """將強平事件轉為 DataFrame"""
        if not self.events:
            return pd.DataFrame()

        return pd.DataFrame([e.to_dict() for e in self.events])


# ===== 資金費率處理 =====


@dataclass
class FundingSettlement:
    """資金費率結算記錄"""
    timestamp: datetime
    rate: float
    position_value: float
    direction: int  # 1=多, -1=空
    cost: float  # 正=支付, 負=收取

    def to_dict(self) -> dict:
        """轉為字典"""
        return {
            'timestamp': self.timestamp,
            'rate': self.rate,
            'position_value': self.position_value,
            'direction': self.direction,
            'cost': self.cost
        }


class FundingRateHandler:
    """
    資金費率處理器

    負責載入、查詢資金費率數據，並計算持倉期間的資金費率成本。

    結算時機（UTC）：
    - 00:00
    - 08:00
    - 16:00

    使用範例：
        handler = FundingRateHandler()

        # 載入費率數據
        handler.load_funding_rates('BTCUSDT', start_date, end_date)

        # 檢查是否為結算時間
        if handler.is_settlement_time(current_time):
            cost = handler.calculate_cost(position_value, direction, current_time)

        # 統計
        print(handler.get_statistics())
    """

    # 結算時間（UTC）
    SETTLEMENT_HOURS = [0, 8, 16]

    def __init__(
        self,
        default_rate: float = 0.0001,  # 預設費率 0.01%
        funding_interval_hours: int = 8
    ):
        """初始化資金費率處理器

        Args:
            default_rate: 預設資金費率（無數據時使用）
            funding_interval_hours: 結算間隔（小時）
        """
        self.default_rate = default_rate
        self.funding_interval_hours = funding_interval_hours

        # 費率數據
        self._rates: Optional[pd.DataFrame] = None
        self._symbol: Optional[str] = None

        # 結算記錄
        self.settlements: List[FundingSettlement] = []

    def load_funding_rates(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        data_source: Optional[str] = None
    ) -> bool:
        """載入資金費率數據

        Args:
            symbol: 交易對符號（如 'BTCUSDT'）
            start_date: 開始日期
            end_date: 結束日期
            data_source: 數據來源（可選）

        Returns:
            bool: 是否載入成功
        """
        self._symbol = symbol

        # 嘗試從本地檔案載入
        rates_path = self._get_rates_path(symbol)

        if rates_path and rates_path.exists():
            try:
                self._rates = pd.read_csv(rates_path)
                self._rates['timestamp'] = pd.to_datetime(self._rates['timestamp'])

                # 過濾日期範圍
                mask = (
                    (self._rates['timestamp'] >= start_date) &
                    (self._rates['timestamp'] <= end_date)
                )
                self._rates = self._rates[mask].reset_index(drop=True)

                logger.info(f"已載入 {symbol} 資金費率: {len(self._rates)} 筆")
                return True

            except Exception as e:
                logger.warning(f"載入資金費率失敗: {e}")

        # 無數據時生成模擬數據
        logger.info(f"使用模擬資金費率數據: 預設費率 {self.default_rate:.4%}")
        self._rates = self._generate_simulated_rates(start_date, end_date)
        return True

    def _get_rates_path(self, symbol: str) -> Optional['Path']:
        """獲取費率數據檔案路徑"""
        from pathlib import Path

        # 嘗試多個可能的路徑
        possible_paths = [
            Path(f"data/funding_rates/{symbol}_funding.csv"),
            Path(f"data/{symbol.lower()}_funding_rates.csv"),
            Path(f"data/raw/{symbol}_funding.csv")
        ]

        for path in possible_paths:
            if path.exists():
                return path

        return None

    def _generate_simulated_rates(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """生成模擬資金費率數據

        在缺乏真實數據時，使用預設費率生成結算時間點。
        """
        timestamps = []
        current = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

        while current <= end_date:
            for hour in self.SETTLEMENT_HOURS:
                ts = current.replace(hour=hour)
                if start_date <= ts <= end_date:
                    timestamps.append(ts)
            current += timedelta(days=1)

        # 生成帶有輕微波動的費率
        np.random.seed(42)  # 確保可重現
        rates = np.random.normal(self.default_rate, self.default_rate * 0.3, len(timestamps))

        return pd.DataFrame({
            'timestamp': timestamps,
            'rate': rates
        })

    def get_rate_at(self, timestamp: datetime) -> float:
        """獲取特定時間的費率

        Args:
            timestamp: 查詢時間

        Returns:
            float: 資金費率
        """
        if self._rates is None or self._rates.empty:
            return self.default_rate

        # 找最近的結算時間點
        mask = self._rates['timestamp'] <= timestamp
        if not mask.any():
            return self.default_rate

        nearest_idx = self._rates.loc[mask, 'timestamp'].idxmax()
        return float(self._rates.loc[nearest_idx, 'rate'])

    def is_settlement_time(self, timestamp: datetime) -> bool:
        """檢查是否為結算時間

        Args:
            timestamp: 要檢查的時間

        Returns:
            bool: 是否為結算時間
        """
        return timestamp.hour in self.SETTLEMENT_HOURS and timestamp.minute == 0

    def get_next_settlement(self, timestamp: datetime) -> datetime:
        """獲取下一個結算時間

        Args:
            timestamp: 當前時間

        Returns:
            datetime: 下一個結算時間
        """
        current_hour = timestamp.hour

        # 找下一個結算小時
        for hour in sorted(self.SETTLEMENT_HOURS):
            if hour > current_hour:
                return timestamp.replace(hour=hour, minute=0, second=0, microsecond=0)

        # 如果今天沒有更多結算時間，返回明天的第一個
        next_day = timestamp + timedelta(days=1)
        return next_day.replace(
            hour=self.SETTLEMENT_HOURS[0],
            minute=0,
            second=0,
            microsecond=0
        )

    def calculate_cost(
        self,
        position_value: float,
        direction: int,
        timestamp: datetime
    ) -> float:
        """計算資金費率成本

        Args:
            position_value: 持倉價值（USDT）
            direction: 持倉方向（1=多, -1=空）
            timestamp: 結算時間

        Returns:
            float: 資金費率成本（正=支付, 負=收取）
        """
        rate = self.get_rate_at(timestamp)
        cost = position_value * rate * direction

        # 記錄結算
        settlement = FundingSettlement(
            timestamp=timestamp,
            rate=rate,
            position_value=position_value,
            direction=direction,
            cost=cost
        )
        self.settlements.append(settlement)

        return cost

    def calculate_period_cost(
        self,
        position_value: float,
        direction: int,
        start_time: datetime,
        end_time: datetime
    ) -> float:
        """計算持倉期間的總資金費率成本

        Args:
            position_value: 持倉價值
            direction: 持倉方向
            start_time: 持倉開始時間
            end_time: 持倉結束時間

        Returns:
            float: 期間總成本
        """
        if self._rates is None:
            return 0.0

        # 找出期間內的所有結算時間點
        mask = (
            (self._rates['timestamp'] >= start_time) &
            (self._rates['timestamp'] <= end_time)
        )
        period_rates = self._rates[mask]

        total_cost = 0.0
        for _, row in period_rates.iterrows():
            cost = self.calculate_cost(
                position_value,
                direction,
                row['timestamp']
            )
            total_cost += cost

        return total_cost

    def get_settlement_times_in_range(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[datetime]:
        """獲取時間範圍內的所有結算時間

        Args:
            start_time: 開始時間
            end_time: 結束時間

        Returns:
            List[datetime]: 結算時間列表
        """
        if self._rates is None:
            return []

        mask = (
            (self._rates['timestamp'] >= start_time) &
            (self._rates['timestamp'] <= end_time)
        )
        return self._rates.loc[mask, 'timestamp'].tolist()

    def get_statistics(self) -> dict:
        """獲取資金費率統計

        Returns:
            dict: 統計資訊
        """
        if not self.settlements:
            return {
                'total_settlements': 0,
                'total_cost': 0.0,
                'total_paid': 0.0,
                'total_received': 0.0,
                'avg_rate': self.default_rate,
                'net_cost': 0.0
            }

        costs = [s.cost for s in self.settlements]
        rates = [s.rate for s in self.settlements]

        total_paid = sum(c for c in costs if c > 0)
        total_received = abs(sum(c for c in costs if c < 0))

        return {
            'total_settlements': len(self.settlements),
            'total_cost': sum(costs),
            'total_paid': total_paid,
            'total_received': total_received,
            'avg_rate': np.mean(rates),
            'net_cost': total_paid - total_received
        }

    def clear_settlements(self):
        """清除結算記錄"""
        self.settlements = []

    def to_dataframe(self) -> pd.DataFrame:
        """將結算記錄轉為 DataFrame"""
        if not self.settlements:
            return pd.DataFrame()

        return pd.DataFrame([s.to_dict() for s in self.settlements])
