"""
Position Sizing Module

實作 Kelly Criterion 及其變體，用於計算最佳部位大小。

Kelly Criterion 公式：
    f* = W - (1-W)/R

    其中：
    - f*: 最佳資金比例
    - W: 勝率 (win_rate)
    - R: 盈虧比 (win_loss_ratio = avg_win / avg_loss)

變體：
    - Full Kelly: 使用完整公式結果
    - Half Kelly: f* / 2 (更保守，降低波動)
    - Quarter Kelly: f* / 4 (極保守)
"""

from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """部位大小計算結果"""
    optimal_fraction: float  # 最佳資金比例 (0.0 ~ 1.0)
    position_size: float     # 實際部位大小 (USD)
    kelly_type: str          # Kelly 類型 (full/half/quarter)
    win_rate: float          # 勝率
    win_loss_ratio: float    # 盈虧比

    def __str__(self) -> str:
        return (
            f"PositionSizeResult(\n"
            f"  kelly_type={self.kelly_type},\n"
            f"  optimal_fraction={self.optimal_fraction:.4f} ({self.optimal_fraction*100:.2f}%),\n"
            f"  position_size=${self.position_size:,.2f},\n"
            f"  win_rate={self.win_rate:.4f},\n"
            f"  win_loss_ratio={self.win_loss_ratio:.4f}\n"
            f")"
        )


def kelly_criterion(win_rate: float, win_loss_ratio: float) -> float:
    """
    計算 Kelly Criterion 最佳資金比例

    Args:
        win_rate: 勝率 (0.0 ~ 1.0)
        win_loss_ratio: 盈虧比 (平均獲利 / 平均虧損)

    Returns:
        optimal_fraction: 最佳資金比例 (0.0 ~ 1.0)

    Raises:
        ValueError: 當參數不合法時

    Example:
        >>> kelly_criterion(0.55, 1.5)  # 55% 勝率，盈虧比 1.5
        0.25
    """
    # 參數驗證
    if not 0 <= win_rate <= 1:
        raise ValueError(f"win_rate 必須介於 0 和 1 之間，得到: {win_rate}")

    if win_loss_ratio <= 0:
        raise ValueError(f"win_loss_ratio 必須大於 0，得到: {win_loss_ratio}")

    # Kelly Criterion 公式: f* = W - (1-W)/R
    loss_rate = 1 - win_rate
    optimal_fraction = win_rate - (loss_rate / win_loss_ratio)

    # 限制在合理範圍 (0 ~ 1)
    # 如果結果為負數，表示該策略不應交易
    if optimal_fraction < 0:
        logger.warning(
            f"Kelly Criterion 計算結果為負 ({optimal_fraction:.4f})，"
            f"建議不要交易此策略 (win_rate={win_rate}, win_loss_ratio={win_loss_ratio})"
        )
        return 0.0

    if optimal_fraction > 1:
        logger.warning(
            f"Kelly Criterion 計算結果超過 100% ({optimal_fraction:.4f})，"
            f"限制為 100%"
        )
        return 1.0

    return optimal_fraction


class KellyPositionSizer:
    """
    Kelly Criterion 部位管理器

    支援 Full Kelly、Half Kelly、Quarter Kelly 等變體。

    Attributes:
        kelly_fraction: Kelly 乘數 (1.0=Full, 0.5=Half, 0.25=Quarter)
        max_position_fraction: 最大部位比例 (安全上限)
        min_win_rate: 最低勝率要求
        min_win_loss_ratio: 最低盈虧比要求
    """

    def __init__(
        self,
        kelly_fraction: float = 0.5,
        max_position_fraction: float = 0.25,
        min_win_rate: float = 0.4,
        min_win_loss_ratio: float = 1.0
    ):
        """
        初始化 Kelly Position Sizer

        Args:
            kelly_fraction: Kelly 乘數，預設 0.5 (Half Kelly)
                - 1.0: Full Kelly (激進)
                - 0.5: Half Kelly (平衡，推薦)
                - 0.25: Quarter Kelly (保守)
            max_position_fraction: 最大部位比例，預設 0.25 (25%)
            min_win_rate: 最低勝率要求，預設 0.4 (40%)
            min_win_loss_ratio: 最低盈虧比要求，預設 1.0
        """
        if not 0 < kelly_fraction <= 1:
            raise ValueError(f"kelly_fraction 必須介於 0 和 1 之間，得到: {kelly_fraction}")

        if not 0 < max_position_fraction <= 1:
            raise ValueError(f"max_position_fraction 必須介於 0 和 1 之間，得到: {max_position_fraction}")

        self.kelly_fraction = kelly_fraction
        self.max_position_fraction = max_position_fraction
        self.min_win_rate = min_win_rate
        self.min_win_loss_ratio = min_win_loss_ratio

        # 決定 Kelly 類型名稱
        if kelly_fraction == 1.0:
            self.kelly_type = "Full Kelly"
        elif kelly_fraction == 0.5:
            self.kelly_type = "Half Kelly"
        elif kelly_fraction == 0.25:
            self.kelly_type = "Quarter Kelly"
        else:
            self.kelly_type = f"{kelly_fraction}x Kelly"

    def calculate_position_size(
        self,
        capital: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        enforce_min_requirements: bool = True
    ) -> PositionSizeResult:
        """
        計算建議部位大小

        Args:
            capital: 可用資金
            win_rate: 勝率 (0.0 ~ 1.0)
            avg_win: 平均獲利金額
            avg_loss: 平均虧損金額 (正數)
            enforce_min_requirements: 是否強制最低要求 (勝率、盈虧比)

        Returns:
            PositionSizeResult: 部位大小計算結果

        Raises:
            ValueError: 當參數不合法時
        """
        # 參數驗證
        if capital <= 0:
            raise ValueError(f"capital 必須大於 0，得到: {capital}")

        if avg_win <= 0:
            raise ValueError(f"avg_win 必須大於 0，得到: {avg_win}")

        if avg_loss <= 0:
            raise ValueError(f"avg_loss 必須大於 0，得到: {avg_loss}")

        # 計算盈虧比
        win_loss_ratio = avg_win / avg_loss

        # 檢查最低要求
        if enforce_min_requirements:
            if win_rate < self.min_win_rate:
                logger.warning(
                    f"勝率 {win_rate:.2%} 低於最低要求 {self.min_win_rate:.2%}，"
                    f"建議不要交易"
                )
                return PositionSizeResult(
                    optimal_fraction=0.0,
                    position_size=0.0,
                    kelly_type=self.kelly_type,
                    win_rate=win_rate,
                    win_loss_ratio=win_loss_ratio
                )

            if win_loss_ratio < self.min_win_loss_ratio:
                logger.warning(
                    f"盈虧比 {win_loss_ratio:.2f} 低於最低要求 {self.min_win_loss_ratio:.2f}，"
                    f"建議不要交易"
                )
                return PositionSizeResult(
                    optimal_fraction=0.0,
                    position_size=0.0,
                    kelly_type=self.kelly_type,
                    win_rate=win_rate,
                    win_loss_ratio=win_loss_ratio
                )

        # 計算原始 Kelly Criterion
        raw_kelly = kelly_criterion(win_rate, win_loss_ratio)

        # 套用 Kelly 乘數 (Half Kelly / Quarter Kelly)
        adjusted_kelly = raw_kelly * self.kelly_fraction

        # 套用最大部位限制
        optimal_fraction = min(adjusted_kelly, self.max_position_fraction)

        # 計算實際部位大小
        position_size = capital * optimal_fraction

        return PositionSizeResult(
            optimal_fraction=optimal_fraction,
            position_size=position_size,
            kelly_type=self.kelly_type,
            win_rate=win_rate,
            win_loss_ratio=win_loss_ratio
        )

    def calculate_from_trades(
        self,
        capital: float,
        winning_trades: list[float],
        losing_trades: list[float],
        enforce_min_requirements: bool = True
    ) -> PositionSizeResult:
        """
        從實際交易記錄計算部位大小

        Args:
            capital: 可用資金
            winning_trades: 獲利交易列表 (正數)
            losing_trades: 虧損交易列表 (正數)
            enforce_min_requirements: 是否強制最低要求

        Returns:
            PositionSizeResult: 部位大小計算結果

        Raises:
            ValueError: 當交易記錄不足時
        """
        if not winning_trades and not losing_trades:
            raise ValueError("至少需要一筆交易記錄")

        total_trades = len(winning_trades) + len(losing_trades)
        win_count = len(winning_trades)

        # 計算勝率
        win_rate = win_count / total_trades if total_trades > 0 else 0.0

        # 計算平均獲利和虧損
        avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0.0
        avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0.0

        # 處理特殊情況
        if avg_win == 0 and avg_loss == 0:
            raise ValueError("平均獲利和虧損不能同時為 0")

        if avg_win == 0:
            logger.warning("沒有獲利交易，建議不要交易")
            win_loss_ratio = 0.0
            return PositionSizeResult(
                optimal_fraction=0.0,
                position_size=0.0,
                kelly_type=self.kelly_type,
                win_rate=win_rate,
                win_loss_ratio=win_loss_ratio
            )

        if avg_loss == 0:
            # 極端情況：沒有虧損交易
            # 設定一個保守的部位大小
            logger.warning("沒有虧損交易，使用保守估計")
            optimal_fraction = min(0.1, self.max_position_fraction)
            return PositionSizeResult(
                optimal_fraction=optimal_fraction,
                position_size=capital * optimal_fraction,
                kelly_type=self.kelly_type,
                win_rate=win_rate,
                win_loss_ratio=float('inf')
            )

        return self.calculate_position_size(
            capital=capital,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            enforce_min_requirements=enforce_min_requirements
        )

    def adjust_kelly_fraction(self, new_fraction: float) -> None:
        """
        動態調整 Kelly 乘數

        可用於根據績效表現動態調整風險偏好。

        Args:
            new_fraction: 新的 Kelly 乘數 (0.0 ~ 1.0)
        """
        if not 0 < new_fraction <= 1:
            raise ValueError(f"new_fraction 必須介於 0 和 1 之間，得到: {new_fraction}")

        old_fraction = self.kelly_fraction
        self.kelly_fraction = new_fraction

        # 更新類型名稱
        if new_fraction == 1.0:
            self.kelly_type = "Full Kelly"
        elif new_fraction == 0.5:
            self.kelly_type = "Half Kelly"
        elif new_fraction == 0.25:
            self.kelly_type = "Quarter Kelly"
        else:
            self.kelly_type = f"{new_fraction}x Kelly"

        logger.info(
            f"Kelly 乘數已調整: {old_fraction} -> {new_fraction} "
            f"(類型: {self.kelly_type})"
        )
