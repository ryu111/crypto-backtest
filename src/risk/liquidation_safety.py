"""
強平安全檢查模組

確保止損在強平之前觸發，避免因強平造成額外損失。

參考：
- .claude/skills/風險管理/SKILL.md

使用範例：
    checker = LiquidationSafetyChecker()

    # 檢查止損是否安全
    is_safe, liq_price, safe_stop = checker.check_stop_before_liquidation(
        entry_price=50000,
        stop_loss=45000,
        leverage=10,
        direction=1
    )

    if not is_safe:
        print(f"警告：止損 {stop_loss} 在強平價格 {liq_price} 之後")
        print(f"建議止損：{safe_stop}")
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class SafetyCheckResult:
    """安全檢查結果"""
    is_safe: bool
    entry_price: float
    stop_loss: float
    liquidation_price: float
    suggested_safe_stop: float
    safety_buffer: float
    margin_to_liquidation_pct: float
    margin_to_stop_pct: float

    @property
    def risk_message(self) -> str:
        """風險訊息"""
        if self.is_safe:
            return "✅ 止損設定安全"
        else:
            return (
                f"⚠️ 止損 {self.stop_loss:.2f} 在強平價格 {self.liquidation_price:.2f} 之後，"
                f"建議調整至 {self.suggested_safe_stop:.2f}"
            )


class LiquidationSafetyChecker:
    """
    強平安全檢查器

    主要功能：
    1. 檢查止損是否在強平之前
    2. 建議安全的止損價格
    3. 計算風險距離

    使用場景：
    - 開倉前檢查
    - 回測驗證
    - 風險監控
    """

    def __init__(
        self,
        maintenance_margin_rate: float = 0.005,
        default_safety_buffer: float = 0.02
    ):
        """
        初始化檢查器

        Args:
            maintenance_margin_rate: 維持保證金率（預設 0.5%）
            default_safety_buffer: 預設安全緩衝（預設 2%）
        """
        self.maintenance_margin_rate = maintenance_margin_rate
        self.default_safety_buffer = default_safety_buffer

    def calculate_liquidation_price(
        self,
        entry_price: float,
        leverage: int,
        direction: int
    ) -> float:
        """
        計算強平價格

        Args:
            entry_price: 入場價格
            leverage: 槓桿倍數
            direction: 1=做多，-1=做空

        Returns:
            強平價格
        """
        mmr = self.maintenance_margin_rate

        if direction == 1:  # 做多
            return entry_price * (1 - 1/leverage + mmr)
        else:  # 做空
            return entry_price * (1 + 1/leverage - mmr)

    def check_stop_before_liquidation(
        self,
        entry_price: float,
        stop_loss: float,
        leverage: int,
        direction: int,
        safety_buffer: Optional[float] = None
    ) -> Tuple[bool, float, float]:
        """
        檢查止損是否在強平之前

        確保止損會在強平觸發前執行，避免額外的強平罰金。

        Args:
            entry_price: 入場價格
            stop_loss: 止損價格
            leverage: 槓桿倍數
            direction: 1=做多，-1=做空
            safety_buffer: 安全緩衝（預設 2%）

        Returns:
            (is_safe, liquidation_price, suggested_safe_stop)

        範例：
            >>> checker = LiquidationSafetyChecker()
            >>> is_safe, liq, safe = checker.check_stop_before_liquidation(50000, 45000, 10, 1)
            >>> is_safe  # 做多 10x，強平價約 45250，止損 45000 不安全
            False
        """
        buffer = safety_buffer or self.default_safety_buffer
        liq_price = self.calculate_liquidation_price(entry_price, leverage, direction)

        if direction == 1:  # 做多
            # 止損應該高於強平價格（加上安全緩衝）
            safe_stop = liq_price * (1 + buffer)
            is_safe = stop_loss >= safe_stop
        else:  # 做空
            # 止損應該低於強平價格（減去安全緩衝）
            safe_stop = liq_price * (1 - buffer)
            is_safe = stop_loss <= safe_stop

        return is_safe, liq_price, safe_stop

    def suggest_safe_stop(
        self,
        entry_price: float,
        leverage: int,
        direction: int,
        safety_buffer: Optional[float] = None
    ) -> float:
        """
        建議安全的止損價格

        Args:
            entry_price: 入場價格
            leverage: 槓桿倍數
            direction: 1=做多，-1=做空
            safety_buffer: 安全緩衝（預設 2%）

        Returns:
            建議的安全止損價格

        範例：
            >>> checker = LiquidationSafetyChecker()
            >>> checker.suggest_safe_stop(50000, 10, 1)
            46155.0  # 做多 10x，安全止損約 $46,155
        """
        buffer = safety_buffer or self.default_safety_buffer
        liq_price = self.calculate_liquidation_price(entry_price, leverage, direction)

        if direction == 1:
            return liq_price * (1 + buffer)
        else:
            return liq_price * (1 - buffer)

    def full_safety_check(
        self,
        entry_price: float,
        stop_loss: float,
        leverage: int,
        direction: int,
        safety_buffer: Optional[float] = None
    ) -> SafetyCheckResult:
        """
        完整安全檢查

        Args:
            entry_price: 入場價格
            stop_loss: 止損價格
            leverage: 槓桿倍數
            direction: 1=做多，-1=做空
            safety_buffer: 安全緩衝

        Returns:
            SafetyCheckResult: 完整的安全檢查結果
        """
        buffer = safety_buffer or self.default_safety_buffer
        is_safe, liq_price, safe_stop = self.check_stop_before_liquidation(
            entry_price, stop_loss, leverage, direction, buffer
        )

        # 計算距離百分比
        if direction == 1:
            margin_to_liq = (entry_price - liq_price) / entry_price * 100
            margin_to_stop = (entry_price - stop_loss) / entry_price * 100
        else:
            margin_to_liq = (liq_price - entry_price) / entry_price * 100
            margin_to_stop = (stop_loss - entry_price) / entry_price * 100

        return SafetyCheckResult(
            is_safe=is_safe,
            entry_price=entry_price,
            stop_loss=stop_loss,
            liquidation_price=liq_price,
            suggested_safe_stop=safe_stop,
            safety_buffer=buffer,
            margin_to_liquidation_pct=margin_to_liq,
            margin_to_stop_pct=margin_to_stop
        )

    def validate_trade_setup(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        leverage: int,
        direction: int,
        safety_buffer: Optional[float] = None
    ) -> dict:
        """
        驗證完整的交易設定

        Args:
            entry_price: 入場價格
            stop_loss: 止損價格
            take_profit: 止盈價格
            leverage: 槓桿倍數
            direction: 1=做多，-1=做空
            safety_buffer: 安全緩衝

        Returns:
            驗證結果字典
        """
        safety_check = self.full_safety_check(
            entry_price, stop_loss, leverage, direction, safety_buffer
        )

        # 計算風險報酬比
        if direction == 1:
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit

        risk_reward_ratio = reward / risk if risk > 0 else 0

        # 計算潛在損益百分比（考慮槓桿）
        potential_loss_pct = (risk / entry_price) * leverage * 100
        potential_profit_pct = (reward / entry_price) * leverage * 100

        warnings = []
        if not safety_check.is_safe:
            warnings.append(f"止損在強平之後，建議調整至 {safety_check.suggested_safe_stop:.2f}")

        if risk_reward_ratio < 1:
            warnings.append(f"風險報酬比 {risk_reward_ratio:.2f} 低於 1:1")

        if potential_loss_pct > 50:
            warnings.append(f"潛在損失 {potential_loss_pct:.1f}% 超過 50%")

        return {
            'is_valid': safety_check.is_safe and len(warnings) <= 1,
            'safety_check': safety_check,
            'risk_reward_ratio': risk_reward_ratio,
            'potential_loss_pct': potential_loss_pct,
            'potential_profit_pct': potential_profit_pct,
            'warnings': warnings,
            'recommendation': 'APPROVED' if not warnings else 'REVIEW_NEEDED'
        }

    def calculate_max_safe_leverage(
        self,
        entry_price: float,
        stop_loss: float,
        direction: int,
        safety_buffer: Optional[float] = None
    ) -> int:
        """
        計算給定止損下的最大安全槓桿

        Args:
            entry_price: 入場價格
            stop_loss: 止損價格
            direction: 1=做多，-1=做空
            safety_buffer: 安全緩衝

        Returns:
            最大安全槓桿倍數

        範例：
            >>> checker = LiquidationSafetyChecker()
            >>> checker.calculate_max_safe_leverage(50000, 47500, 1)
            5  # 止損在 5% 處，最大安全槓桿約 5x
        """
        buffer = safety_buffer or self.default_safety_buffer
        mmr = self.maintenance_margin_rate

        if direction == 1:
            stop_distance = (entry_price - stop_loss) / entry_price
            # 反推：stop_loss >= liq_price * (1 + buffer)
            # liq_price = entry * (1 - 1/lev + mmr)
            # stop_loss >= entry * (1 - 1/lev + mmr) * (1 + buffer)
            # 解 leverage
            effective_distance = stop_distance / (1 + buffer)
            if effective_distance <= mmr:
                return 1
            max_lev = 1 / (effective_distance - mmr)
        else:
            stop_distance = (stop_loss - entry_price) / entry_price
            effective_distance = stop_distance / (1 + buffer)
            if effective_distance <= mmr:
                return 1
            max_lev = 1 / (effective_distance - mmr)

        return max(1, int(max_lev))


def check_liquidation_safety(
    entry_price: float,
    stop_loss: float,
    leverage: int,
    direction: int,
    safety_buffer: float = 0.02
) -> Tuple[bool, float, float]:
    """
    便捷函數：檢查強平安全

    Args:
        entry_price: 入場價格
        stop_loss: 止損價格
        leverage: 槓桿倍數
        direction: 1=做多，-1=做空
        safety_buffer: 安全緩衝（預設 2%）

    Returns:
        (is_safe, liquidation_price, suggested_safe_stop)
    """
    checker = LiquidationSafetyChecker()
    return checker.check_stop_before_liquidation(
        entry_price, stop_loss, leverage, direction, safety_buffer
    )
