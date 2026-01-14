"""
自適應槓桿模組

根據市場狀態和策略表現動態調整槓桿倍數。

調整邏輯：
1. 低波動時提高槓桿（穩定期放大收益）
2. 高回撤時降低槓桿（保護資本）
3. 策略表現好時逐步加槓（動量效應）
4. 永遠不超過最大槓桿限制
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveLeverageConfig:
    """自適應槓桿配置"""
    # 基礎設定
    base_leverage: int = 5
    min_leverage: int = 1
    max_leverage: int = 10

    # 波動度模式
    volatility_mode: bool = True
    low_vol_threshold: float = 0.01        # 日波動率 < 1%
    high_vol_threshold: float = 0.03       # 日波動率 > 3%
    low_vol_leverage_mult: float = 1.5     # 低波動時 1.5x
    high_vol_leverage_mult: float = 0.5    # 高波動時 0.5x

    # 回撤模式
    drawdown_mode: bool = True
    dd_leverage_reduction: Dict[float, float] = field(default_factory=lambda: {
        0.05: 0.8,   # 5% DD → 80% 槓桿
        0.10: 0.5,   # 10% DD → 50% 槓桿
        0.15: 0.25,  # 15% DD → 25% 槓桿
    })

    # 表現模式
    performance_mode: bool = True
    winning_streak_threshold: int = 3       # 連勝 3 次後加槓
    losing_streak_threshold: int = 2        # 連虧 2 次後降槓
    streak_adjustment: float = 0.2          # 每次調整 20%

    # 平滑參數
    smoothing_factor: float = 0.3           # EMA 平滑因子（避免劇烈變動）

    def __post_init__(self):
        """驗證配置參數"""
        # 驗證 smoothing_factor 範圍
        if not 0 <= self.smoothing_factor <= 1:
            raise ValueError(
                f"smoothing_factor 必須介於 0 和 1 之間，得到: {self.smoothing_factor}"
            )

        # 驗證槓桿範圍
        if self.min_leverage > self.max_leverage:
            raise ValueError(
                f"min_leverage ({self.min_leverage}) 不能大於 max_leverage ({self.max_leverage})"
            )

        if self.base_leverage > self.max_leverage:
            raise ValueError(
                f"base_leverage ({self.base_leverage}) 不能大於 max_leverage ({self.max_leverage})"
            )


class AdaptiveLeverageController:
    """
    自適應槓桿控制器

    調整邏輯：
    1. 低波動時提高槓桿（穩定期放大收益）
    2. 高 DD 時降低槓桿（保護資本）
    3. 策略表現好時逐步加槓（動量效應）
    4. 永遠不超過最大槓桿限制

    Example:
        >>> config = AdaptiveLeverageConfig(base_leverage=5, max_leverage=10)
        >>> controller = AdaptiveLeverageController(config)
        >>> leverage = controller.calculate_leverage(
        ...     current_volatility=0.02,
        ...     current_drawdown=0.05,
        ...     recent_win_rate=0.65
        ... )
        >>> print(f"建議槓桿: {leverage}x")
    """

    def __init__(self, config: Optional[AdaptiveLeverageConfig] = None):
        """
        初始化自適應槓桿控制器

        Args:
            config: 配置參數，若為 None 則使用預設值
        """
        self.config = config or AdaptiveLeverageConfig()
        self._current_streak = 0  # 正=連勝，負=連虧
        self._recent_trades: List[bool] = []  # True=勝, False=虧
        self._smoothed_leverage: Optional[float] = None
        self._adjustment_history: List[Dict] = []

        logger.info(f"AdaptiveLeverageController 初始化完成，基礎槓桿={self.config.base_leverage}x")

    def calculate_leverage(
        self,
        current_volatility: float,
        current_drawdown: float,
        recent_win_rate: Optional[float] = None
    ) -> int:
        """
        計算當前應使用的槓桿

        Args:
            current_volatility: 當前日波動率（例如 0.02 表示 2%）
            current_drawdown: 當前回撤比例（例如 0.05 表示 5%）
            recent_win_rate: 最近勝率（0.0 ~ 1.0），若為 None 則從內部狀態計算

        Returns:
            建議槓桿倍數（整數）

        Example:
            >>> leverage = controller.calculate_leverage(
            ...     current_volatility=0.015,  # 1.5% 日波動
            ...     current_drawdown=0.03,     # 3% 回撤
            ...     recent_win_rate=0.60       # 60% 勝率
            ... )
        """
        # 參數驗證
        if current_volatility < 0:
            raise ValueError(f"current_volatility 必須 >= 0，得到: {current_volatility}")
        if not 0 <= current_drawdown <= 1:
            raise ValueError(f"current_drawdown 必須介於 0 和 1 之間，得到: {current_drawdown}")
        if recent_win_rate is not None and not 0 <= recent_win_rate <= 1:
            raise ValueError(f"recent_win_rate 必須介於 0 和 1 之間，得到: {recent_win_rate}")

        # 從基礎槓桿開始
        leverage = float(self.config.base_leverage)

        # 1. 應用波動度調整
        if self.config.volatility_mode:
            leverage = self._apply_volatility_adjustment(leverage, current_volatility)

        # 2. 應用回撤調整
        if self.config.drawdown_mode:
            leverage = self._apply_drawdown_adjustment(leverage, current_drawdown)

        # 3. 應用表現調整
        if self.config.performance_mode:
            win_rate = recent_win_rate if recent_win_rate is not None else self.recent_win_rate
            leverage = self._apply_performance_adjustment(leverage, win_rate)

        # 4. 平滑處理（避免頻繁大幅變動）
        leverage = self._smooth_leverage(leverage)

        # 5. 限制範圍
        final_leverage = max(
            self.config.min_leverage,
            min(self.config.max_leverage, int(np.round(leverage)))
        )

        # 記錄調整歷史
        self._record_adjustment(
            volatility=current_volatility,
            drawdown=current_drawdown,
            win_rate=recent_win_rate if recent_win_rate is not None else self.recent_win_rate,
            raw_leverage=leverage,
            final_leverage=final_leverage
        )

        logger.debug(
            f"槓桿計算: 波動={current_volatility:.2%}, DD={current_drawdown:.2%}, "
            f"原始={leverage:.2f}x → 最終={final_leverage}x"
        )

        return final_leverage

    def update_streak(self, trade_won: bool):
        """
        更新連勝/連虧狀態

        Args:
            trade_won: True=獲利交易, False=虧損交易

        Example:
            >>> controller.update_streak(True)   # 記錄一筆獲利
            >>> controller.update_streak(True)   # 又一筆獲利
            >>> controller.update_streak(False)  # 虧損，連勝中斷
        """
        self._recent_trades.append(trade_won)

        # 保留最近 20 筆記錄
        if len(self._recent_trades) > 20:
            self._recent_trades.pop(0)

        # 更新連勝/連虧計數
        if trade_won:
            if self._current_streak >= 0:
                self._current_streak += 1
            else:
                self._current_streak = 1
        else:
            if self._current_streak <= 0:
                self._current_streak -= 1
            else:
                self._current_streak = -1

        logger.debug(f"更新 streak: {'勝' if trade_won else '虧'}, 當前 streak={self._current_streak}")

    def _apply_volatility_adjustment(self, leverage: float, volatility: float) -> float:
        """
        應用波動度調整

        低波動 → 提高槓桿（穩定期可以放大收益）
        高波動 → 降低槓桿（避免強平）
        """
        if volatility <= self.config.low_vol_threshold:
            # 低波動：提高槓桿
            adjusted = leverage * self.config.low_vol_leverage_mult
            logger.debug(f"低波動調整: {leverage:.2f}x → {adjusted:.2f}x (vol={volatility:.2%})")
            return adjusted

        elif volatility >= self.config.high_vol_threshold:
            # 高波動：降低槓桿
            adjusted = leverage * self.config.high_vol_leverage_mult
            logger.debug(f"高波動調整: {leverage:.2f}x → {adjusted:.2f}x (vol={volatility:.2%})")
            return adjusted

        else:
            # 正常波動：線性內插
            # 波動率在 [low, high] 區間時，槓桿倍數在 [high_mult, low_mult] 區間
            vol_range = self.config.high_vol_threshold - self.config.low_vol_threshold

            # 檢查除零情況
            if vol_range == 0:
                # 兩個閾值相同，使用平均倍數
                avg_mult = (self.config.low_vol_leverage_mult + self.config.high_vol_leverage_mult) / 2
                adjusted = leverage * avg_mult
                logger.warning(
                    f"波動閾值相同 ({self.config.low_vol_threshold:.2%})，"
                    f"使用平均倍數 {avg_mult:.2f}"
                )
                return adjusted

            vol_position = (volatility - self.config.low_vol_threshold) / vol_range

            mult_range = self.config.low_vol_leverage_mult - self.config.high_vol_leverage_mult
            multiplier = self.config.low_vol_leverage_mult - (mult_range * vol_position)

            adjusted = leverage * multiplier
            logger.debug(
                f"正常波動調整: {leverage:.2f}x → {adjusted:.2f}x "
                f"(vol={volatility:.2%}, mult={multiplier:.2f})"
            )
            return adjusted

    def _apply_drawdown_adjustment(self, leverage: float, drawdown: float) -> float:
        """
        應用回撤調整

        回撤越大，槓桿越低（保護資本）
        """
        # 檢查配置是否為空
        if not self.config.dd_leverage_reduction:
            logger.warning("dd_leverage_reduction 配置為空，跳過回撤調整")
            return leverage

        # 找到對應的回撤層級
        reduction_factor = 1.0
        for dd_threshold in sorted(self.config.dd_leverage_reduction.keys(), reverse=True):
            if drawdown >= dd_threshold:
                reduction_factor = self.config.dd_leverage_reduction[dd_threshold]
                logger.debug(
                    f"回撤調整: DD={drawdown:.2%} >= {dd_threshold:.2%}, "
                    f"reduction={reduction_factor:.2%}"
                )
                break

        adjusted = leverage * reduction_factor
        if reduction_factor < 1.0:
            logger.info(
                f"⚠️ 回撤保護啟動: {leverage:.2f}x → {adjusted:.2f}x (DD={drawdown:.2%})"
            )

        return adjusted

    def _apply_performance_adjustment(
        self,
        leverage: float,
        win_rate: Optional[float] = None  # noqa: ARG002 - 保留參數供未來擴展
    ) -> float:
        """
        應用表現調整

        連勝 → 逐步增加槓桿（動量效應）
        連虧 → 逐步降低槓桿（防止雪球效應）
        """
        # 連勝調整
        if self._current_streak >= self.config.winning_streak_threshold:
            # 超過閾值的每次連勝都增加
            extra_wins = self._current_streak - self.config.winning_streak_threshold + 1

            # 限制連勝加槓
            max_extra_wins = 5  # 最多計算 5 次額外連勝
            extra_wins = min(extra_wins, max_extra_wins)

            multiplier = 1 + (extra_wins * self.config.streak_adjustment)

            # 限制最大倍數
            max_multiplier = 2.0
            multiplier = min(multiplier, max_multiplier)

            adjusted = leverage * multiplier
            logger.info(
                f"🔥 連勝加槓: {self._current_streak} 連勝 → "
                f"{leverage:.2f}x → {adjusted:.2f}x (mult={multiplier:.2f})"
            )
            return adjusted

        # 連虧調整
        elif abs(self._current_streak) >= self.config.losing_streak_threshold:
            # 超過閾值的每次連虧都減少
            extra_losses = abs(self._current_streak) - self.config.losing_streak_threshold + 1
            reduction = extra_losses * self.config.streak_adjustment

            # 先計算 reduction，再限制下限
            multiplier = max(0.2, 1 - reduction)  # 最多降到 20%

            adjusted = leverage * multiplier
            logger.warning(
                f"⚠️ 連虧降槓: {abs(self._current_streak)} 連虧 → "
                f"{leverage:.2f}x → {adjusted:.2f}x (mult={multiplier:.2f})"
            )
            return adjusted

        # 無明顯連勝/連虧
        return leverage

    def _smooth_leverage(self, leverage: float) -> float:
        """
        平滑槓桿變化（EMA）

        避免槓桿頻繁大幅變動
        """
        if self._smoothed_leverage is None:
            self._smoothed_leverage = leverage
            return leverage

        # EMA: smoothed = alpha * new + (1 - alpha) * smoothed
        alpha = self.config.smoothing_factor
        self._smoothed_leverage = alpha * leverage + (1 - alpha) * self._smoothed_leverage

        return self._smoothed_leverage

    def _record_adjustment(
        self,
        volatility: float,
        drawdown: float,
        win_rate: float,
        raw_leverage: float,
        final_leverage: int
    ):
        """記錄調整歷史"""
        record = {
            'volatility': volatility,
            'drawdown': drawdown,
            'win_rate': win_rate,
            'streak': self._current_streak,
            'raw_leverage': raw_leverage,
            'final_leverage': final_leverage,
        }
        self._adjustment_history.append(record)

        # 保留最近 100 筆記錄
        if len(self._adjustment_history) > 100:
            self._adjustment_history.pop(0)

    def get_leverage_report(self) -> Dict:
        """
        獲取槓桿調整報告

        Returns:
            包含當前狀態和統計資訊的字典

        Example:
            >>> report = controller.get_leverage_report()
            >>> print(f"平均槓桿: {report['avg_leverage']:.2f}x")
            >>> print(f"最大槓桿: {report['max_leverage']}x")
        """
        if not self._adjustment_history:
            return {
                'total_adjustments': 0,
                'avg_leverage': self.config.base_leverage,
                'min_leverage': self.config.base_leverage,
                'max_leverage': self.config.base_leverage,
                'current_streak': self._current_streak,
                'recent_win_rate': self.recent_win_rate,
            }

        leverages = [rec['final_leverage'] for rec in self._adjustment_history]

        report = {
            'total_adjustments': len(self._adjustment_history),
            'avg_leverage': np.mean(leverages),
            'min_leverage': np.min(leverages),
            'max_leverage': np.max(leverages),
            'std_leverage': np.std(leverages),
            'current_streak': self._current_streak,
            'recent_win_rate': self.recent_win_rate,
            'total_trades': len(self._recent_trades),
            'recent_history': self._adjustment_history[-5:],  # 最近 5 筆
        }

        return report

    def reset(self):
        """
        重置狀態

        用於回測時重置到初始狀態

        Example:
            >>> controller.reset()  # 清空所有狀態，重新開始
        """
        self._current_streak = 0
        self._recent_trades.clear()
        self._smoothed_leverage = None
        self._adjustment_history.clear()
        logger.info("AdaptiveLeverageController 狀態已重置")

    @property
    def recent_win_rate(self) -> float:
        """
        計算最近勝率

        Returns:
            勝率（0.0 ~ 1.0）
        """
        if not self._recent_trades:
            return 0.5  # 預設 50%

        wins = sum(1 for trade in self._recent_trades if trade)
        return wins / len(self._recent_trades)

    @property
    def current_streak(self) -> int:
        """
        獲取當前連勝/連虧狀態

        Returns:
            正數=連勝次數, 負數=連虧次數, 0=無連勝連虧
        """
        return self._current_streak

    def __repr__(self) -> str:
        return (
            f"AdaptiveLeverageController("
            f"base={self.config.base_leverage}x, "
            f"range=[{self.config.min_leverage}, {self.config.max_leverage}], "
            f"streak={self._current_streak}, "
            f"win_rate={self.recent_win_rate:.2%})"
        )
