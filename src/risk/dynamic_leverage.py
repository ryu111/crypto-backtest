"""
動態槓桿管理模組

根據市場波動率動態調整槓桿，高波動時降低槓桿，低波動時可提高槓桿。

參考：
- .claude/skills/風險管理/SKILL.md

使用範例：
    manager = DynamicLeverageManager(base_leverage=5, max_leverage=10)

    # 計算調整後的槓桿
    current_atr = calculate_atr(data, 14)
    avg_atr = data['atr'].rolling(100).mean().iloc[-1]

    adjusted_leverage = manager.calculate_adjusted_leverage(current_atr, avg_atr)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class LeverageAdjustment:
    """槓桿調整記錄"""
    timestamp: pd.Timestamp
    base_leverage: int
    adjusted_leverage: float
    volatility_ratio: float
    adjustment_reason: str

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'base_leverage': self.base_leverage,
            'adjusted_leverage': self.adjusted_leverage,
            'volatility_ratio': self.volatility_ratio,
            'reason': self.adjustment_reason
        }


class DynamicLeverageManager:
    """
    動態槓桿管理器

    根據 ATR（平均真實波幅）波動率調整槓桿倍數。

    調整邏輯（參考風險管理 Skill）：
    - 波動率 > 1.5x 平均：槓桿 × 0.5（高波動，大幅降低）
    - 波動率 > 1.2x 平均：槓桿 × 0.75（中高波動，適度降低）
    - 波動率 < 0.8x 平均：槓桿 × 1.25（低波動，可提高）
    - 其他：維持原槓桿

    使用場景：
    - 趨勢跟隨策略（持倉時間長，需要動態調整）
    - 波段交易
    - 風險控制嚴格的策略
    """

    # 預設調整閾值
    HIGH_VOL_THRESHOLD = 1.5      # 高波動閾值
    MEDIUM_HIGH_VOL_THRESHOLD = 1.2  # 中高波動閾值
    LOW_VOL_THRESHOLD = 0.8       # 低波動閾值

    # 預設調整因子
    HIGH_VOL_FACTOR = 0.5         # 高波動調整因子
    MEDIUM_HIGH_VOL_FACTOR = 0.75 # 中高波動調整因子
    LOW_VOL_FACTOR = 1.25         # 低波動調整因子

    def __init__(
        self,
        base_leverage: int = 5,
        max_leverage: int = 10,
        min_leverage: int = 1,
        atr_period: int = 14,
        lookback_period: int = 100
    ):
        """
        初始化動態槓桿管理器

        Args:
            base_leverage: 基礎槓桿（預設 5x）
            max_leverage: 最大槓桿上限（預設 10x）
            min_leverage: 最小槓桿下限（預設 1x）
            atr_period: ATR 計算週期（預設 14）
            lookback_period: 平均波動率回看週期（預設 100）
        """
        self.base_leverage = base_leverage
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        self.atr_period = atr_period
        self.lookback_period = lookback_period
        self.adjustment_history: List[LeverageAdjustment] = []

    def calculate_atr(self, data: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
        """
        計算 ATR（平均真實波幅）

        Args:
            data: OHLCV DataFrame
            period: ATR 週期（預設使用 self.atr_period）

        Returns:
            ATR Series
        """
        period = period or self.atr_period

        high = data['high']
        low = data['low']
        close = data['close']

        # 真實波幅 = max(H-L, |H-C_prev|, |L-C_prev|)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def calculate_adjusted_leverage(
        self,
        current_atr: float,
        avg_atr: float
    ) -> float:
        """
        根據波動率計算調整後的槓桿

        Args:
            current_atr: 當前 ATR
            avg_atr: 平均 ATR

        Returns:
            調整後的槓桿倍數

        範例：
            >>> mgr = DynamicLeverageManager(base_leverage=5)
            >>> mgr.calculate_adjusted_leverage(1000, 500)  # 高波動
            2.5  # 5 × 0.5
            >>> mgr.calculate_adjusted_leverage(300, 500)  # 低波動
            6.25  # 5 × 1.25，但會被 max_leverage 限制
        """
        if avg_atr == 0:
            return float(self.base_leverage)

        vol_ratio = current_atr / avg_atr

        if vol_ratio > self.HIGH_VOL_THRESHOLD:
            factor = self.HIGH_VOL_FACTOR
            reason = f"高波動 ({vol_ratio:.2f}x)"
        elif vol_ratio > self.MEDIUM_HIGH_VOL_THRESHOLD:
            factor = self.MEDIUM_HIGH_VOL_FACTOR
            reason = f"中高波動 ({vol_ratio:.2f}x)"
        elif vol_ratio < self.LOW_VOL_THRESHOLD:
            factor = self.LOW_VOL_FACTOR
            reason = f"低波動 ({vol_ratio:.2f}x)"
        else:
            factor = 1.0
            reason = f"正常波動 ({vol_ratio:.2f}x)"

        adjusted = self.base_leverage * factor

        # 限制在範圍內
        adjusted = max(self.min_leverage, min(adjusted, self.max_leverage))

        logger.debug(
            f"槓桿調整: {self.base_leverage}x → {adjusted:.2f}x ({reason})"
        )

        return adjusted

    def get_leverage_for_trade(
        self,
        data: pd.DataFrame,
        timestamp: Optional[pd.Timestamp] = None
    ) -> Tuple[float, str]:
        """
        獲取特定時間點的建議槓桿

        Args:
            data: OHLCV DataFrame
            timestamp: 時間戳（預設使用最新數據）

        Returns:
            (adjusted_leverage, adjustment_reason)
        """
        # 計算 ATR
        atr = self.calculate_atr(data)

        # 獲取當前和平均 ATR
        if timestamp is not None and timestamp in atr.index:
            current_atr = atr.loc[timestamp]
        else:
            current_atr = atr.iloc[-1]

        # 計算平均 ATR
        avg_atr = atr.rolling(self.lookback_period).mean()
        if timestamp is not None and timestamp in avg_atr.index:
            avg_atr_value = avg_atr.loc[timestamp]
        else:
            avg_atr_value = avg_atr.iloc[-1]

        # 處理 NaN
        if pd.isna(current_atr) or pd.isna(avg_atr_value):
            return float(self.base_leverage), "數據不足，使用基礎槓桿"

        # 計算調整後槓桿
        vol_ratio = current_atr / avg_atr_value if avg_atr_value > 0 else 1.0
        adjusted = self.calculate_adjusted_leverage(current_atr, avg_atr_value)

        # 產生原因
        if vol_ratio > self.HIGH_VOL_THRESHOLD:
            reason = f"高波動 (ATR {current_atr:.2f} vs 平均 {avg_atr_value:.2f})"
        elif vol_ratio > self.MEDIUM_HIGH_VOL_THRESHOLD:
            reason = f"中高波動"
        elif vol_ratio < self.LOW_VOL_THRESHOLD:
            reason = f"低波動"
        else:
            reason = "正常波動"

        # 記錄調整歷史
        ts = timestamp or data.index[-1]
        self.adjustment_history.append(LeverageAdjustment(
            timestamp=ts,
            base_leverage=self.base_leverage,
            adjusted_leverage=adjusted,
            volatility_ratio=vol_ratio,
            adjustment_reason=reason
        ))

        return adjusted, reason

    def simulate_leverage_series(
        self,
        data: pd.DataFrame
    ) -> pd.Series:
        """
        模擬整個時間序列的動態槓桿

        Args:
            data: OHLCV DataFrame

        Returns:
            動態槓桿 Series
        """
        atr = self.calculate_atr(data)
        avg_atr = atr.rolling(self.lookback_period).mean()

        # 向量化計算波動率比率
        vol_ratio = atr / avg_atr

        # 向量化計算調整因子
        factor = pd.Series(1.0, index=vol_ratio.index)
        factor[vol_ratio > self.HIGH_VOL_THRESHOLD] = self.HIGH_VOL_FACTOR
        factor[(vol_ratio > self.MEDIUM_HIGH_VOL_THRESHOLD) &
               (vol_ratio <= self.HIGH_VOL_THRESHOLD)] = self.MEDIUM_HIGH_VOL_FACTOR
        factor[vol_ratio < self.LOW_VOL_THRESHOLD] = self.LOW_VOL_FACTOR

        # 計算調整後槓桿
        leverage_series = self.base_leverage * factor
        leverage_series = leverage_series.clip(self.min_leverage, self.max_leverage)

        # 填充 NaN
        leverage_series = leverage_series.fillna(self.base_leverage)

        return leverage_series

    def get_statistics(self) -> dict:
        """
        獲取槓桿調整統計

        Returns:
            統計字典
        """
        if not self.adjustment_history:
            return {
                'total_adjustments': 0,
                'avg_leverage': self.base_leverage,
                'min_leverage_used': self.base_leverage,
                'max_leverage_used': self.base_leverage
            }

        leverages = [a.adjusted_leverage for a in self.adjustment_history]

        return {
            'total_adjustments': len(self.adjustment_history),
            'avg_leverage': np.mean(leverages),
            'min_leverage_used': min(leverages),
            'max_leverage_used': max(leverages),
            'std_leverage': np.std(leverages),
            'high_vol_count': sum(1 for a in self.adjustment_history
                                  if a.volatility_ratio > self.HIGH_VOL_THRESHOLD),
            'low_vol_count': sum(1 for a in self.adjustment_history
                                 if a.volatility_ratio < self.LOW_VOL_THRESHOLD)
        }

    def clear_history(self):
        """清除調整歷史"""
        self.adjustment_history = []

    def to_dataframe(self) -> pd.DataFrame:
        """將調整歷史轉為 DataFrame"""
        if not self.adjustment_history:
            return pd.DataFrame()

        return pd.DataFrame([a.to_dict() for a in self.adjustment_history])


def dynamic_leverage(
    base_leverage: int,
    current_volatility: float,
    avg_volatility: float,
    max_leverage: int = 10
) -> float:
    """
    便捷函數：計算動態槓桿

    Args:
        base_leverage: 基礎槓桿
        current_volatility: 當前波動率（ATR 或其他指標）
        avg_volatility: 平均波動率
        max_leverage: 最大槓桿上限

    Returns:
        調整後的槓桿
    """
    if avg_volatility == 0:
        return float(base_leverage)

    vol_ratio = current_volatility / avg_volatility

    if vol_ratio > 1.5:
        factor = 0.5
    elif vol_ratio > 1.2:
        factor = 0.75
    elif vol_ratio < 0.8:
        factor = 1.25
    else:
        factor = 1.0

    adjusted = base_leverage * factor
    return min(adjusted, max_leverage)
