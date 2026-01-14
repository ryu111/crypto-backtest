"""
信號放大器模組

用於放寬進場條件，增加交易信號數量。

提供四種放大模式：
1. threshold_expand - 閾值擴展（RSI 30→35, 70→65）
2. anticipation - 提前預判（MACD 趨近交叉）
3. tolerance - 容忍度放寬（MA 幾乎交叉）
4. sensitivity - 靈敏度調整（突破門檻降低）

使用範例：
    >>> config = AmplificationConfig(rsi_expand=5.0, macd_lookahead=2)
    >>> amplifier = SignalAmplifier(config)
    >>>
    >>> # 放大 RSI 信號
    >>> oversold, overbought = amplifier.amplify_rsi_signals(rsi)
    >>>
    >>> # 放大所有信號
    >>> amplified_signals = amplifier.amplify_all(data, original_signals)
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class AmplificationConfig:
    """
    信號放大配置

    Attributes:
        rsi_expand: RSI 閾值放寬幅度（30→35, 70→65）
        macd_lookahead: MACD 提前 N 根 K 線預判
        ma_tolerance: MA 交叉容忍度（百分比）
        breakout_sensitivity: 突破靈敏度（0.0-1.0，越低越靈敏）
        enabled: 是否啟用放大器
    """
    rsi_expand: float = 5.0
    macd_lookahead: int = 2
    ma_tolerance: float = 0.002  # 0.2%
    breakout_sensitivity: float = 0.95  # 95% 接近突破就觸發
    enabled: bool = True

    def __post_init__(self):
        """驗證配置參數"""
        if self.rsi_expand < 0 or self.rsi_expand > 20:
            raise ValueError("rsi_expand must be between 0 and 20")

        if self.macd_lookahead < 0 or self.macd_lookahead > 10:
            raise ValueError("macd_lookahead must be between 0 and 10")

        if self.ma_tolerance < 0 or self.ma_tolerance > 0.1:
            raise ValueError("ma_tolerance must be between 0 and 0.1")

        if self.breakout_sensitivity < 0.5 or self.breakout_sensitivity > 1.0:
            raise ValueError("breakout_sensitivity must be between 0.5 and 1.0")


class SignalAmplifier:
    """
    信號放大器

    提供多種放大模式以增加交易信號數量。

    放大模式說明：
    - threshold_expand: 擴展閾值（RSI 30→35, 70→65）
    - anticipation: 提前預判（MACD 趨近但未交叉）
    - tolerance: 容忍度放寬（MA 幾乎交叉但未交叉）
    - sensitivity: 靈敏度調整（突破前 95% 就觸發）

    使用原則：
    - 放大後的信號與原始信號使用 OR 邏輯合併
    - 不改變原始信號，只添加新信號
    - 記錄放大效果統計

    Attributes:
        config: 放大配置
        stats: 放大效果統計
    """

    def __init__(self, config: Optional[AmplificationConfig] = None):
        """
        初始化信號放大器

        Args:
            config: 放大配置，None 則使用預設值
        """
        self.config = config or AmplificationConfig()
        self.stats: Dict[str, Any] = {
            'rsi_amplified_count': 0,
            'macd_amplified_count': 0,
            'ma_amplified_count': 0,
            'breakout_amplified_count': 0,
            'total_original_signals': 0,
            'total_amplified_signals': 0,
        }

    def amplify_rsi_signals(
        self,
        rsi: pd.Series,
        base_oversold: float = 30,
        base_overbought: float = 70
    ) -> Tuple[pd.Series, pd.Series]:
        """
        放寬 RSI 閾值

        透過擴展超買超賣閾值，在接近但未到達閾值時就觸發信號。

        Args:
            rsi: RSI 指標 Series
            base_oversold: 原始超賣閾值（預設 30）
            base_overbought: 原始超買閾值（預設 70）

        Returns:
            (oversold_signals, overbought_signals)
            - oversold_signals: 擴展後的超賣信號
            - overbought_signals: 擴展後的超買信號

        Example:
            >>> rsi = pd.Series([28, 32, 35, 68, 72, 75])
            >>> oversold, overbought = amplifier.amplify_rsi_signals(rsi)
            >>> # 原始: RSI < 30 → [True, False, False, False, False, False]
            >>> # 放大: RSI < 35 → [True, True, True, False, False, False]
        """
        if not self.config.enabled:
            return rsi < base_oversold, rsi > base_overbought

        # 計算擴展後的閾值
        expanded_oversold = base_oversold + self.config.rsi_expand
        expanded_overbought = base_overbought - self.config.rsi_expand

        # 產生放大後的信號
        oversold_signals = (rsi < expanded_oversold).fillna(False)
        overbought_signals = (rsi > expanded_overbought).fillna(False)

        # 計算原始信號與放大後的差異
        original_oversold = (rsi < base_oversold).fillna(False)
        original_overbought = (rsi > base_overbought).fillna(False)

        amplified_count = (
            (oversold_signals & ~original_oversold).sum() +
            (overbought_signals & ~original_overbought).sum()
        )

        self.stats['rsi_amplified_count'] += amplified_count

        logger.debug(
            f"RSI amplification: {amplified_count} new signals "
            f"(oversold: {base_oversold}→{expanded_oversold}, "
            f"overbought: {base_overbought}→{expanded_overbought})"
        )

        return oversold_signals, overbought_signals

    def amplify_macd_signals(
        self,
        macd_line: pd.Series,
        signal_line: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        MACD 提前預判 - 趨近交叉

        不等待實際交叉發生，而是在趨勢接近交叉時就觸發信號。
        透過檢查未來 N 根 K 線是否會交叉來提前進場。

        Args:
            macd_line: MACD 線 Series
            signal_line: 信號線 Series

        Returns:
            (approaching_bullish, approaching_bearish)
            - approaching_bullish: 接近黃金交叉的信號
            - approaching_bearish: 接近死亡交叉的信號

        Example:
            >>> macd = pd.Series([-1.2, -0.8, -0.3, 0.1, 0.5])
            >>> signal = pd.Series([0.5, 0.4, 0.2, 0.1, -0.1])
            >>> bullish, bearish = amplifier.amplify_macd_signals(macd, signal)
            >>> # 原始: MACD 交叉發生在 index=3
            >>> # 放大: 提前在 index=1 觸發（lookahead=2）
        """
        if not self.config.enabled or self.config.macd_lookahead == 0:
            # 未啟用時返回實際交叉信號
            bullish_cross = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
            bearish_cross = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
            return bullish_cross.fillna(False), bearish_cross.fillna(False)

        # 計算 MACD 與信號線的差距
        diff = macd_line - signal_line

        # 初始化信號
        approaching_bullish = pd.Series(False, index=macd_line.index)
        approaching_bearish = pd.Series(False, index=macd_line.index)

        # 向前檢查 N 根 K 線是否會交叉
        lookahead = self.config.macd_lookahead

        for i in range(len(macd_line) - lookahead):
            current_diff = diff.iloc[i]

            # 檢查未來是否會發生黃金交叉（MACD 向上穿越信號線）
            if current_diff < 0:  # 當前 MACD 在信號線下方
                future_diffs = diff.iloc[i+1:i+lookahead+1]
                if (future_diffs > 0).any():  # 未來會上穿
                    approaching_bullish.iloc[i] = True

            # 檢查未來是否會發生死亡交叉（MACD 向下穿越信號線）
            if current_diff > 0:  # 當前 MACD 在信號線上方
                future_diffs = diff.iloc[i+1:i+lookahead+1]
                if (future_diffs < 0).any():  # 未來會下穿
                    approaching_bearish.iloc[i] = True

        amplified_count = approaching_bullish.sum() + approaching_bearish.sum()
        self.stats['macd_amplified_count'] += amplified_count

        logger.debug(
            f"MACD amplification: {amplified_count} new signals "
            f"(lookahead: {lookahead} bars)"
        )

        return approaching_bullish, approaching_bearish

    def amplify_ma_cross(
        self,
        fast_ma: pd.Series,
        slow_ma: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        MA 交叉容忍度放寬

        允許「幾乎交叉」的情況觸發信號。
        當快線與慢線距離小於容忍度時，視為交叉。

        Args:
            fast_ma: 快速均線 Series
            slow_ma: 慢速均線 Series

        Returns:
            (near_golden_cross, near_death_cross)
            - near_golden_cross: 接近黃金交叉的信號
            - near_death_cross: 接近死亡交叉的信號

        Example:
            >>> fast = pd.Series([99.5, 100.1, 100.5])
            >>> slow = pd.Series([100, 100, 100])
            >>> golden, death = amplifier.amplify_ma_cross(fast, slow)
            >>> # 原始: 需要 fast > slow 才算交叉
            >>> # 放大: fast 在 slow 的 0.2% 範圍內就算交叉
        """
        if not self.config.enabled:
            # 未啟用時返回實際交叉信號
            golden = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
            death = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
            return golden.fillna(False), death.fillna(False)

        # 計算相對距離（百分比）
        distance_pct = (fast_ma - slow_ma).abs() / slow_ma

        # 判斷是否接近（距離小於容忍度）
        is_near = distance_pct < self.config.ma_tolerance

        # 判斷方向
        fast_above = fast_ma >= slow_ma
        fast_below = fast_ma < slow_ma

        # 檢查是否從下方接近（接近黃金交叉）
        was_below = fast_ma.shift(1) < slow_ma.shift(1)
        near_golden_cross = is_near & fast_above & was_below

        # 檢查是否從上方接近（接近死亡交叉）
        was_above = fast_ma.shift(1) >= slow_ma.shift(1)
        near_death_cross = is_near & fast_below & was_above

        amplified_count = near_golden_cross.sum() + near_death_cross.sum()
        self.stats['ma_amplified_count'] += amplified_count

        logger.debug(
            f"MA cross amplification: {amplified_count} new signals "
            f"(tolerance: {self.config.ma_tolerance:.2%})"
        )

        return near_golden_cross.fillna(False), near_death_cross.fillna(False)

    def amplify_breakout(
        self,
        close: pd.Series,
        resistance: pd.Series,
        support: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        降低突破門檻

        不等待價格完全突破壓力/支撐，在接近時就觸發信號。

        Args:
            close: 收盤價 Series
            resistance: 壓力位 Series
            support: 支撐位 Series

        Returns:
            (near_breakout_up, near_breakout_down)
            - near_breakout_up: 接近向上突破的信號
            - near_breakout_down: 接近向下突破的信號

        Example:
            >>> close = pd.Series([98, 99, 99.5, 100.5])
            >>> resistance = pd.Series([100, 100, 100, 100])
            >>> support = pd.Series([95, 95, 95, 95])
            >>> up, down = amplifier.amplify_breakout(close, resistance, support)
            >>> # 原始: close >= resistance 才算突破
            >>> # 放大: close >= resistance * 0.95 就算接近突破（sensitivity=0.95）
        """
        if not self.config.enabled:
            # 未啟用時返回實際突破信號
            breakout_up = close > resistance
            breakout_down = close < support
            return breakout_up.fillna(False), breakout_down.fillna(False)

        # 計算降低後的突破門檻
        threshold = self.config.breakout_sensitivity

        # 向上突破：價格接近壓力位的 threshold 倍
        near_breakout_up = close >= (resistance * threshold)

        # 向下突破：價格接近支撐位的 (2 - threshold) 倍
        # 例如 threshold=0.95 → 支撐位 * 1.05
        near_breakout_down = close <= (support * (2 - threshold))

        amplified_count = near_breakout_up.sum() + near_breakout_down.sum()
        self.stats['breakout_amplified_count'] += amplified_count

        logger.debug(
            f"Breakout amplification: {amplified_count} new signals "
            f"(sensitivity: {threshold:.2%})"
        )

        return near_breakout_up.fillna(False), near_breakout_down.fillna(False)

    def amplify_all(
        self,
        data: pd.DataFrame,
        original_signals: Tuple[pd.Series, pd.Series, pd.Series, pd.Series],
        indicators: Optional[Dict[str, pd.Series]] = None
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        放大所有信號（OR 邏輯合併原始信號）

        將原始信號與放大後的信號合併，只添加新信號不移除原有信號。

        Args:
            data: OHLCV DataFrame（必須包含 close）
            original_signals: (long_entry, long_exit, short_entry, short_exit)
            indicators: 可選的指標字典，可包含：
                - 'rsi': RSI Series
                - 'macd': MACD 線 Series
                - 'signal': 信號線 Series
                - 'fast_ma': 快速均線 Series
                - 'slow_ma': 慢速均線 Series
                - 'resistance': 壓力位 Series
                - 'support': 支撐位 Series

        Returns:
            amplified (long_entry, long_exit, short_entry, short_exit)

        Example:
            >>> # 準備指標
            >>> indicators = {
            ...     'rsi': calculate_rsi(data['close']),
            ...     'macd': macd_line,
            ...     'signal': signal_line
            ... }
            >>>
            >>> # 放大信號
            >>> amplified = amplifier.amplify_all(data, original_signals, indicators)
            >>>
            >>> # 檢查統計
            >>> print(amplifier.get_stats())
        """
        if not self.config.enabled:
            return original_signals

        long_entry_orig, long_exit_orig, short_entry_orig, short_exit_orig = original_signals

        # 初始化放大後的信號（複製原始信號）
        long_entry = long_entry_orig.copy()
        long_exit = long_exit_orig.copy()
        short_entry = short_entry_orig.copy()
        short_exit = short_exit_orig.copy()

        indicators = indicators or {}

        # 1. RSI 信號放大
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            oversold_amp, overbought_amp = self.amplify_rsi_signals(rsi)

            # 合併到進場信號（OR 邏輯）
            long_entry = long_entry | oversold_amp
            short_entry = short_entry | overbought_amp

        # 2. MACD 信號放大
        if 'macd' in indicators and 'signal' in indicators:
            macd_line = indicators['macd']
            signal_line = indicators['signal']
            bullish_amp, bearish_amp = self.amplify_macd_signals(macd_line, signal_line)

            long_entry = long_entry | bullish_amp
            short_entry = short_entry | bearish_amp

        # 3. MA 交叉信號放大
        if 'fast_ma' in indicators and 'slow_ma' in indicators:
            fast_ma = indicators['fast_ma']
            slow_ma = indicators['slow_ma']
            golden_amp, death_amp = self.amplify_ma_cross(fast_ma, slow_ma)

            long_entry = long_entry | golden_amp
            short_entry = short_entry | death_amp

        # 4. 突破信號放大
        if 'resistance' in indicators and 'support' in indicators:
            close_series = pd.Series(data['close'])
            resistance_series = pd.Series(indicators['resistance'])
            support_series = pd.Series(indicators['support'])
            breakout_up_amp, breakout_down_amp = self.amplify_breakout(
                close_series, resistance_series, support_series
            )

            long_entry = long_entry | breakout_up_amp
            short_entry = short_entry | breakout_down_amp

        # 更新統計
        self.stats['total_original_signals'] = (
            long_entry_orig.sum() + short_entry_orig.sum()
        )
        self.stats['total_amplified_signals'] = (
            long_entry.sum() + short_entry.sum()
        )

        amplification_rate = (
            (self.stats['total_amplified_signals'] - self.stats['total_original_signals']) /
            max(self.stats['total_original_signals'], 1)
        )

        logger.info(
            f"Signal amplification complete: "
            f"{self.stats['total_original_signals']} → {self.stats['total_amplified_signals']} "
            f"({amplification_rate:.1%} increase)"
        )

        return long_entry, long_exit, short_entry, short_exit

    def get_stats(self) -> Dict[str, Any]:
        """
        取得放大效果統計

        Returns:
            dict: 統計資訊
                - rsi_amplified_count: RSI 放大的信號數
                - macd_amplified_count: MACD 放大的信號數
                - ma_amplified_count: MA 放大的信號數
                - breakout_amplified_count: 突破放大的信號數
                - total_original_signals: 原始信號總數
                - total_amplified_signals: 放大後信號總數
                - amplification_rate: 放大比例

        Example:
            >>> stats = amplifier.get_stats()
            >>> print(f"RSI amplified: {stats['rsi_amplified_count']}")
            >>> print(f"Total increase: {stats['amplification_rate']:.1%}")
        """
        stats = self.stats.copy()

        if stats['total_original_signals'] > 0:
            stats['amplification_rate'] = (
                (stats['total_amplified_signals'] - stats['total_original_signals']) /
                stats['total_original_signals']
            )
        else:
            stats['amplification_rate'] = 0.0

        return stats

    def reset_stats(self):
        """重置統計資料"""
        self.stats = {
            'rsi_amplified_count': 0,
            'macd_amplified_count': 0,
            'ma_amplified_count': 0,
            'breakout_amplified_count': 0,
            'total_original_signals': 0,
            'total_amplified_signals': 0,
        }
        logger.debug("Signal amplifier stats reset")
