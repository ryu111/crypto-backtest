"""
策略切換器 - 根據市場狀態自動選擇適合的策略

基於市場方向性和波動度，動態選擇最適合當前市場環境的策略組合。
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .analyzer import MarketState


@dataclass
class StrategyConfig:
    """
    策略配置

    定義策略適用的市場環境範圍和權重。

    Attributes:
        name: 策略名稱
        direction_range: 方向性範圍 (min, max)，-10 到 10
        volatility_range: 波動度範圍 (min, max)，0 到 10
        weight: 策略權重，用於多策略組合，預設 1.0
    """
    name: str
    direction_range: Tuple[float, float]  # (min, max)
    volatility_range: Tuple[float, float]  # (min, max)
    weight: float = 1.0

    def is_active(self, direction: float, volatility: float) -> bool:
        """
        判斷策略是否適用於當前市場狀態

        Args:
            direction: 市場方向性 (-10 到 10)
            volatility: 市場波動度 (0 到 10)

        Returns:
            True 如果策略適用，False 否則
        """
        direction_ok = self.direction_range[0] <= direction <= self.direction_range[1]
        volatility_ok = self.volatility_range[0] <= volatility <= self.volatility_range[1]
        return direction_ok and volatility_ok


class StrategySwitch:
    """
    策略切換管理器

    根據市場狀態動態選擇和權重調整策略組合。

    Examples:
        >>> switch = StrategySwitch()
        >>> switch.register_strategy(
        ...     "trend_following",
        ...     direction_range=(3, 10),
        ...     volatility_range=(3, 10),
        ...     weight=1.0
        ... )
        >>> state = MarketState(direction=5.0, volatility=4.0, ...)
        >>> active = switch.get_active_strategies(state)
        >>> print(active)  # ['trend_following']
    """

    def __init__(self):
        """初始化策略切換器"""
        self.strategies: Dict[str, StrategyConfig] = {}

    def register_strategy(
        self,
        name: str,
        direction_range: Tuple[float, float],
        volatility_range: Tuple[float, float],
        weight: float = 1.0
    ) -> None:
        """
        註冊策略配置

        Args:
            name: 策略名稱
            direction_range: 方向性範圍 (min, max)
            volatility_range: 波動度範圍 (min, max)
            weight: 策略權重，預設 1.0

        Raises:
            ValueError: 如果範圍值無效
        """
        # 驗證範圍
        if not (-10 <= direction_range[0] <= direction_range[1] <= 10):
            raise ValueError(f"Invalid direction_range: {direction_range}")
        if not (0 <= volatility_range[0] <= volatility_range[1] <= 10):
            raise ValueError(f"Invalid volatility_range: {volatility_range}")
        if weight <= 0:
            raise ValueError(f"Weight must be positive: {weight}")

        config = StrategyConfig(
            name=name,
            direction_range=direction_range,
            volatility_range=volatility_range,
            weight=weight
        )
        self.strategies[name] = config

    def get_active_strategies(self, state: MarketState) -> List[str]:
        """
        獲取當前市場狀態下適用的策略列表

        Args:
            state: 市場狀態

        Returns:
            適用策略名稱列表
        """
        active = []
        for name, config in self.strategies.items():
            if config.is_active(state.direction, state.volatility):
                active.append(name)
        return active

    def get_strategy_weights(self, state: MarketState) -> Dict[str, float]:
        """
        獲取當前市場狀態下各策略的權重

        權重會被正規化，使總和為 1.0。

        Args:
            state: 市場狀態

        Returns:
            策略名稱到正規化權重的字典
        """
        weights = {}
        total_weight = 0.0

        # 收集活躍策略的權重
        for name, config in self.strategies.items():
            if config.is_active(state.direction, state.volatility):
                weights[name] = config.weight
                total_weight += config.weight

        # 正規化權重
        if total_weight > 0:
            weights = {name: w / total_weight for name, w in weights.items()}

        return weights


def setup_default_switch() -> StrategySwitch:
    """
    建立預設策略切換器配置

    包含以下策略：
    - trend_following_long: 趨勢追蹤（做多）
    - trend_following_short: 趨勢追蹤（做空）
    - mean_reversion: 均值回歸
    - breakout: 突破策略
    - grid_trading: 網格交易
    - funding_rate_arb: 資金費率套利

    Returns:
        配置好的 StrategySwitch 實例
    """
    switch = StrategySwitch()

    # 趨勢追蹤（做多）- 強方向性 + 高波動
    switch.register_strategy(
        "trend_following_long",
        direction_range=(3, 10),
        volatility_range=(3, 10),
        weight=1.0
    )

    # 趨勢追蹤（做空）- 強反向 + 高波動
    switch.register_strategy(
        "trend_following_short",
        direction_range=(-10, -3),
        volatility_range=(3, 10),
        weight=1.0
    )

    # 均值回歸 - 無方向 + 低波動
    switch.register_strategy(
        "mean_reversion",
        direction_range=(-3, 3),
        volatility_range=(0, 5),
        weight=1.0
    )

    # 突破策略 - 無特定方向 + 低波動（等待突破）
    switch.register_strategy(
        "breakout",
        direction_range=(-5, 5),
        volatility_range=(0, 3),
        weight=1.0
    )

    # 網格交易 - 無方向 + 中高波動
    switch.register_strategy(
        "grid_trading",
        direction_range=(-3, 3),
        volatility_range=(5, 10),
        weight=1.0
    )

    # 資金費率套利 - 任意方向 + 任意波動
    switch.register_strategy(
        "funding_rate_arb",
        direction_range=(-10, 10),
        volatility_range=(0, 10),
        weight=0.5  # 較低權重，作為輔助策略
    )

    return switch
