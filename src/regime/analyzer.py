"""市場狀態分析器 - Market Regime Detection

基於方向×波動矩陣的可解釋市場狀態偵測。
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Union
from datetime import datetime
import numpy as np
import pandas as pd


class MarketRegime(Enum):
    """市場狀態枚舉"""
    STRONG_BULL_HIGH_VOL = "strong_bull_high_vol"     # 強勢上漲，高波動
    STRONG_BULL_LOW_VOL = "strong_bull_low_vol"       # 強勢上漲，低波動
    WEAK_BULL_HIGH_VOL = "weak_bull_high_vol"         # 弱勢上漲，高波動
    WEAK_BULL_LOW_VOL = "weak_bull_low_vol"           # 弱勢上漲，低波動
    NEUTRAL_HIGH_VOL = "neutral_high_vol"             # 中性，高波動
    NEUTRAL_LOW_VOL = "neutral_low_vol"               # 中性，低波動
    WEAK_BEAR_HIGH_VOL = "weak_bear_high_vol"         # 弱勢下跌，高波動
    WEAK_BEAR_LOW_VOL = "weak_bear_low_vol"           # 弱勢下跌，低波動
    STRONG_BEAR_HIGH_VOL = "strong_bear_high_vol"     # 強勢下跌，高波動
    STRONG_BEAR_LOW_VOL = "strong_bear_low_vol"       # 強勢下跌，低波動


@dataclass
class MarketState:
    """市場狀態數據類"""
    direction: float      # -10 到 +10
    volatility: float     # 0 到 10
    regime: MarketRegime
    timestamp: datetime

    def to_dict(self) -> dict:
        """轉換為字典格式"""
        return {
            'direction': self.direction,
            'volatility': self.volatility,
            'regime': self.regime.value,
            'timestamp': self.timestamp.isoformat()
        }


def calculate_direction_score(
    data: pd.DataFrame,
    ma_short: int = 20,
    ma_long: int = 50,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    scale: int = 10
) -> pd.Series:
    """
    計算方向分數 (-scale 到 +scale)

    組合多個指標的綜合方向判斷：
    - MA 位置和斜率
    - RSI 偏離
    - MACD 柱狀

    Args:
        data: OHLCV 數據
        ma_short: 短期 MA 週期
        ma_long: 長期 MA 週期
        rsi_period: RSI 週期
        macd_fast: MACD 快線週期
        macd_slow: MACD 慢線週期
        scale: 縮放係數（通常為 10）

    Returns:
        方向分數序列
    """
    close = data['close']

    # 1. MA 位置 (-1 到 +1)
    ma_s = close.rolling(ma_short).mean()
    ma_l = close.rolling(ma_long).mean()
    ma_position = np.where(close > ma_s, 0.5, -0.5)
    ma_position += np.where(ma_s > ma_l, 0.5, -0.5)

    # 2. MA 斜率 (-1 到 +1)
    ma_shifted = ma_s.shift(5)
    # 安全除法：避免除以零
    with np.errstate(divide='ignore', invalid='ignore'):
        ma_slope = (ma_s - ma_shifted) / ma_shifted
        ma_slope = pd.Series(ma_slope).fillna(0)
    ma_slope_score = np.clip(ma_slope * 20, -1, 1)

    # 3. RSI 偏離 (-1 到 +1)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    # 安全除法：loss=0 時表示全漲，RS 設為極大值（RSI=100）
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain / loss
        rs = pd.Series(rs).replace([np.inf, -np.inf], 100).fillna(0)
    rsi = 100 - (100 / (1 + rs))
    rsi_score = (rsi - 50) / 50

    # 4. MACD 柱狀 (-1 到 +1)
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_hist = macd_line - macd_line.ewm(span=9, adjust=False).mean()
    # 安全除法：以價格標準化 MACD
    with np.errstate(divide='ignore', invalid='ignore'):
        macd_score = macd_hist / close * 100
        macd_score = pd.Series(macd_score).replace([np.inf, -np.inf], 0).fillna(0)
        macd_score = np.clip(macd_score, -1, 1)

    # 綜合分數（加權平均）
    weights = {'ma_position': 0.3, 'ma_slope': 0.2, 'rsi': 0.25, 'macd': 0.25}

    composite = (
        ma_position * weights['ma_position'] +
        ma_slope_score * weights['ma_slope'] +
        rsi_score * weights['rsi'] +
        macd_score * weights['macd']
    )

    return pd.Series(composite * scale, index=data.index, name='direction_score')


def adx_direction_score(
    data: pd.DataFrame,
    period: int = 14,
    scale: int = 10
) -> pd.Series:
    """
    使用 ADX 的 +DI/-DI 計算方向

    優點：直接基於價格動量，較少延遲

    Args:
        data: OHLCV 數據
        period: ADX 週期
        scale: 縮放係數

    Returns:
        方向分數序列
    """
    high, low, close = data['high'], data['low'], data['close']

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # 安全除法：計算 DI（保持原始索引）
    plus_dm_series = pd.Series(plus_dm, index=data.index)
    minus_dm_series = pd.Series(minus_dm, index=data.index)
    plus_dm_avg = plus_dm_series.rolling(period).mean()
    minus_dm_avg = minus_dm_series.rolling(period).mean()

    with np.errstate(divide='ignore', invalid='ignore'):
        # 轉為 numpy 計算，避免索引問題
        atr_arr = np.asarray(atr)
        plus_di_arr = 100 * np.asarray(plus_dm_avg) / atr_arr
        minus_di_arr = 100 * np.asarray(minus_dm_avg) / atr_arr
        plus_di_arr = np.where(np.isinf(plus_di_arr), 0, plus_di_arr)
        plus_di_arr = np.where(np.isnan(plus_di_arr), 0, plus_di_arr)
        minus_di_arr = np.where(np.isinf(minus_di_arr), 0, minus_di_arr)
        minus_di_arr = np.where(np.isnan(minus_di_arr), 0, minus_di_arr)

    # 方向分數：+DI 和 -DI 的差異標準化
    di_diff = plus_di_arr - minus_di_arr
    di_sum = plus_di_arr + minus_di_arr
    with np.errstate(divide='ignore', invalid='ignore'):
        direction_values = np.where(di_sum != 0, (di_diff / di_sum) * scale, 0)
        direction_values = np.where(np.isinf(direction_values), 0, direction_values)
        direction_values = np.where(np.isnan(direction_values), 0, direction_values)
    return pd.Series(direction_values, index=data.index, name='adx_direction')


def elder_power_score(
    data: pd.DataFrame,
    ema_period: int = 13,
    scale: int = 10
) -> pd.Series:
    """
    Elder 的 Bull/Bear Power 方向分數

    Bull Power = High - EMA（多頭力量）
    Bear Power = Low - EMA（空頭力量）

    Args:
        data: OHLCV 數據
        ema_period: EMA 週期
        scale: 縮放係數

    Returns:
        方向分數序列
    """
    close = data['close']
    high = data['high']
    low = data['low']

    ema = close.ewm(span=ema_period, adjust=False).mean()

    bull_power = high - ema
    bear_power = low - ema

    # 標準化（安全除法）
    power_range = (high - low).rolling(20).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        bull_norm = bull_power / power_range
        bear_norm = bear_power / power_range
        bull_norm = pd.Series(bull_norm).replace([np.inf, -np.inf], 0).fillna(0)
        bear_norm = pd.Series(bear_norm).replace([np.inf, -np.inf], 0).fillna(0)

    # 綜合：Bull + Bear
    net_power = (bull_norm + bear_norm) / 2
    direction = np.clip(net_power * scale, -scale, scale)

    return pd.Series(direction, index=data.index, name='elder_power')


def volatility_score_atr(
    data: pd.DataFrame,
    atr_period: int = 14,
    lookback: int = 100,
    scale: int = 10
) -> pd.Series:
    """
    基於 ATR 的波動分數 (0 到 scale)

    將當前 ATR 對比歷史百分位

    Args:
        data: OHLCV 數據
        atr_period: ATR 週期
        lookback: 回溯週期
        scale: 縮放係數

    Returns:
        波動分數序列
    """
    high, low, close = data['high'], data['low'], data['close']

    # ATR 計算
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    # 標準化 ATR (相對於價格，安全除法)
    with np.errstate(divide='ignore', invalid='ignore'):
        natr_values = atr / close
        natr = pd.Series(natr_values, index=data.index).replace([np.inf, -np.inf], 0).fillna(0)

    # 百分位排名
    def percentile_rank(x: pd.Series) -> float:
        return float(pd.Series(x).rank(pct=True).iloc[-1])

    volatility = natr.rolling(lookback).apply(percentile_rank, raw=False)
    return pd.Series(volatility * scale, index=data.index, name='volatility_atr')


def volatility_score_bbw(
    data: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    lookback: int = 100,
    scale: int = 10
) -> pd.Series:
    """
    基於 Bollinger Band Width 的波動分數

    BBW = (Upper - Lower) / Middle

    Args:
        data: OHLCV 數據
        period: BB 週期
        std_dev: 標準差倍數
        lookback: 回溯週期
        scale: 縮放係數

    Returns:
        波動分數序列
    """
    close = data['close']

    # Bollinger Bands
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std

    # Band Width（安全除法）
    with np.errstate(divide='ignore', invalid='ignore'):
        bbw_values = (upper - lower) / middle
        bbw = pd.Series(bbw_values, index=data.index).replace([np.inf, -np.inf], 0).fillna(0)

    # 百分位排名
    def percentile_rank(x: pd.Series) -> float:
        return float(pd.Series(x).rank(pct=True).iloc[-1])

    volatility = bbw.rolling(lookback).apply(percentile_rank, raw=False)
    return pd.Series(volatility * scale, index=data.index, name='volatility_bbw')


def choppiness_index(
    data: pd.DataFrame,
    period: int = 14,
    scale: int = 10
) -> pd.Series:
    """
    Choppiness Index - 趨勢 vs 區間

    高值 = 盤整/震盪
    低值 = 趨勢明確

    Args:
        data: OHLCV 數據
        period: CI 週期
        scale: 縮放係數

    Returns:
        震盪指數序列（0 = 強趨勢, scale = 強震盪）
    """
    high, low, close = data['high'], data['low'], data['close']

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR sum
    atr_sum = tr.rolling(period).sum()

    # Highest High - Lowest Low
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    range_hl = hh - ll

    # Choppiness Index（安全除法）
    with np.errstate(divide='ignore', invalid='ignore'):
        ci_ratio = atr_sum / range_hl
        ci_ratio = pd.Series(ci_ratio).replace([np.inf, -np.inf], 1).fillna(1)
        ci = 100 * np.log10(ci_ratio) / np.log10(period)
        ci = pd.Series(ci).replace([np.inf, -np.inf], 50).fillna(50)

    # 標準化到 0-scale（CI 通常在 38.2 到 61.8 之間，Fibonacci 比例）
    CI_LOW_BOUND = 38.2
    CI_HIGH_BOUND = 61.8
    ci_norm = (ci - CI_LOW_BOUND) / (CI_HIGH_BOUND - CI_LOW_BOUND)
    ci_norm = np.clip(ci_norm, 0, 1) * scale

    return pd.Series(ci_norm, index=data.index, name='choppiness')


class MarketStateAnalyzer:
    """市場狀態分析器"""

    def __init__(
        self,
        direction_threshold_strong: float = 5.0,
        direction_threshold_weak: float = 2.0,
        volatility_threshold: float = 5.0,
        direction_method: str = 'composite'  # 'composite', 'adx', 'elder'
    ):
        """
        初始化市場狀態分析器

        Args:
            direction_threshold_strong: 強方向閾值
            direction_threshold_weak: 弱方向閾值
            volatility_threshold: 波動閾值
            direction_method: 方向計算方法
        """
        self.dir_strong = direction_threshold_strong
        self.dir_weak = direction_threshold_weak
        self.vol_threshold = volatility_threshold
        self.direction_method = direction_method

    def calculate_state(self, data: pd.DataFrame) -> MarketState:
        """
        計算當前市場狀態

        Args:
            data: OHLCV 數據

        Returns:
            MarketState 對象
        """
        direction = self._calculate_direction(data)
        volatility = self._calculate_volatility(data)
        regime = self._determine_regime(direction, volatility)

        # 處理 timestamp 類型
        ts: datetime = datetime.now()  # 預設值
        try:
            last_idx = data.index[-1]
            if isinstance(last_idx, datetime):
                ts = last_idx
            elif isinstance(last_idx, pd.Timestamp):
                converted = last_idx.to_pydatetime()
                # 使用 isinstance 確認類型（排除 NaTType）
                if isinstance(converted, datetime):
                    ts = converted
            else:
                # 嘗試轉換其他類型
                converted = pd.Timestamp(str(last_idx)).to_pydatetime()
                if isinstance(converted, datetime):
                    ts = converted
        except Exception:
            pass  # 使用預設值 datetime.now()

        return MarketState(
            direction=direction,
            volatility=volatility,
            regime=regime,
            timestamp=ts
        )

    def _calculate_direction(self, data: pd.DataFrame) -> float:
        """計算方向分數"""
        if self.direction_method == 'composite':
            score = calculate_direction_score(data)
        elif self.direction_method == 'adx':
            score = adx_direction_score(data)
        elif self.direction_method == 'elder':
            score = elder_power_score(data)
        else:
            score = calculate_direction_score(data)

        return float(score.iloc[-1])

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """計算波動分數"""
        # 綜合 ATR 和 BBW
        vol_atr = volatility_score_atr(data)
        vol_bbw = volatility_score_bbw(data)

        # 加權平均
        volatility = vol_atr.iloc[-1] * 0.6 + vol_bbw.iloc[-1] * 0.4
        return float(volatility)

    def _determine_regime(self, direction: float, volatility: float) -> MarketRegime:
        """
        判斷市場狀態

        根據方向分數和波動分數，分類為 10 種市場狀態之一。
        """
        # 方向分類
        if direction > self.dir_strong:
            dir_class = 'strong_bull'
        elif direction > self.dir_weak:
            dir_class = 'weak_bull'
        elif direction < -self.dir_strong:
            dir_class = 'strong_bear'
        elif direction < -self.dir_weak:
            dir_class = 'weak_bear'
        else:
            dir_class = 'neutral'

        # 波動分類
        vol_class = 'high_vol' if volatility > self.vol_threshold else 'low_vol'

        # 組合並驗證
        regime_name = f"{dir_class}_{vol_class}"
        try:
            return MarketRegime(regime_name)
        except ValueError:
            # 降級處理：預設返回 neutral（防禦性編程）
            return (MarketRegime.NEUTRAL_LOW_VOL
                    if volatility <= self.vol_threshold
                    else MarketRegime.NEUTRAL_HIGH_VOL)


class RegimeValidator:
    """狀態偵測準確度驗證器"""

    def __init__(
        self,
        forward_periods: int = 20,  # 看未來多少根 K 線
        direction_threshold: float = 0.03,  # 方向判定閾值 (3%)
        volatility_threshold: float = 1.5   # 波動判定閾值 (1.5倍)
    ):
        """
        初始化驗證器

        Args:
            forward_periods: 前瞻期數
            direction_threshold: 方向判定閾值
            volatility_threshold: 波動判定閾值
        """
        self.forward_periods = forward_periods
        self.dir_threshold = direction_threshold
        self.vol_threshold = volatility_threshold

    def validate_direction(
        self,
        data: pd.DataFrame,
        states: List[MarketState]
    ) -> Dict:
        """
        驗證方向偵測準確度

        邏輯：
        - 預測 direction > 5（強牛）→ 未來應該漲 > 3%
        - 預測 direction < -5（強熊）→ 未來應該跌 > 3%
        - 預測 |direction| < 3（中性）→ 未來應該盤整 < 3%

        Args:
            data: OHLCV 數據
            states: 市場狀態列表

        Returns:
            驗證結果字典
        """
        results = []

        for i, state in enumerate(states):
            if i + self.forward_periods >= len(data):
                break

            # 計算未來報酬
            future_return = (
                data['close'].iloc[i + self.forward_periods] /
                data['close'].iloc[i] - 1
            )

            # 判斷是否準確
            if state.direction > 5:  # 預測強牛
                accurate = future_return > self.dir_threshold
                prediction = 'strong_bull'
            elif state.direction < -5:  # 預測強熊
                accurate = future_return < -self.dir_threshold
                prediction = 'strong_bear'
            elif state.direction > 2:  # 預測弱牛
                accurate = future_return > 0
                prediction = 'weak_bull'
            elif state.direction < -2:  # 預測弱熊
                accurate = future_return < 0
                prediction = 'weak_bear'
            else:  # 預測中性
                accurate = abs(future_return) < self.dir_threshold
                prediction = 'neutral'

            results.append({
                'timestamp': state.timestamp,
                'direction_score': state.direction,
                'prediction': prediction,
                'future_return': future_return,
                'accurate': accurate
            })

        df = pd.DataFrame(results)
        accuracy = df['accurate'].mean()

        # 分類別準確度
        by_prediction = df.groupby('prediction')['accurate'].mean().to_dict()

        return {
            'overall_accuracy': accuracy,
            'by_prediction': by_prediction,
            'n_samples': len(results),
            'passed': accuracy > 0.6,
            'details': df
        }

    def validate_volatility(
        self,
        data: pd.DataFrame,
        states: List[MarketState]
    ) -> Dict:
        """
        驗證波動偵測準確度

        邏輯：
        - 預測 volatility > 7（高波動）→ 未來波動應該大於平均
        - 預測 volatility < 3（低波動）→ 未來波動應該小於平均

        Args:
            data: OHLCV 數據
            states: 市場狀態列表

        Returns:
            驗證結果字典
        """
        results = []

        # 計算歷史平均波動
        returns = data['close'].pct_change()
        avg_vol = returns.rolling(100).std().mean()

        for i, state in enumerate(states):
            if i + self.forward_periods >= len(data):
                break

            # 計算未來波動
            future_vol = returns.iloc[i:i+self.forward_periods].std()
            vol_ratio = future_vol / avg_vol

            # 判斷是否準確
            if state.volatility > 7:  # 預測高波動
                accurate = vol_ratio > self.vol_threshold
                prediction = 'high_vol'
            elif state.volatility < 3:  # 預測低波動
                accurate = vol_ratio < 1 / self.vol_threshold
                prediction = 'low_vol'
            else:  # 預測中等波動
                accurate = 0.7 < vol_ratio < 1.5
                prediction = 'mid_vol'

            results.append({
                'timestamp': state.timestamp,
                'volatility_score': state.volatility,
                'prediction': prediction,
                'future_vol_ratio': vol_ratio,
                'accurate': accurate
            })

        df = pd.DataFrame(results)
        accuracy = df['accurate'].mean()

        return {
            'overall_accuracy': accuracy,
            'by_prediction': df.groupby('prediction')['accurate'].mean().to_dict(),
            'n_samples': len(results),
            'passed': accuracy > 0.6,
            'details': df
        }

    def validate_stability(
        self,
        states: List[MarketState],
        max_flip_rate: float = 0.2
    ) -> Dict:
        """
        驗證狀態穩定性（不會頻繁翻轉）

        邏輯：
        - 狀態翻轉太頻繁 = 噪音太多，不可靠
        - 每日翻轉率應 < 20%

        Args:
            states: 市場狀態列表
            max_flip_rate: 最大翻轉率

        Returns:
            驗證結果字典
        """
        if len(states) < 2:
            return {'passed': False, 'reason': 'insufficient_data'}

        flips = 0
        for i in range(1, len(states)):
            prev_dir = 'bull' if states[i-1].direction > 2 else \
                      ('bear' if states[i-1].direction < -2 else 'neutral')
            curr_dir = 'bull' if states[i].direction > 2 else \
                      ('bear' if states[i].direction < -2 else 'neutral')

            if prev_dir != curr_dir:
                flips += 1

        flip_rate = flips / len(states)

        return {
            'flip_rate': flip_rate,
            'total_flips': flips,
            'total_states': len(states),
            'passed': flip_rate < max_flip_rate
        }

    def full_validation(
        self,
        data: pd.DataFrame,
        states: List[MarketState]
    ) -> Dict:
        """
        完整驗證報告

        Args:
            data: OHLCV 數據
            states: 市場狀態列表

        Returns:
            完整驗證結果
        """
        dir_result = self.validate_direction(data, states)
        vol_result = self.validate_volatility(data, states)
        stability_result = self.validate_stability(states)

        all_passed = (
            dir_result['passed'] and
            vol_result['passed'] and
            stability_result['passed']
        )

        return {
            'direction': dir_result,
            'volatility': vol_result,
            'stability': stability_result,
            'all_passed': all_passed,
            'recommendation': (
                '✅ 可進行策略匹配' if all_passed
                else '❌ 需調整狀態偵測參數'
            )
        }
