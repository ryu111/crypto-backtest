"""
Stochastic Crossover 動量策略

基於 Stochastic Oscillator 的超買超賣交叉策略。
當 %K 向上穿越 %D 且在超賣區時做多，
當 %K 向下穿越 %D 且在超買區時做空。
"""

import pandas as pd
from typing import Dict, Tuple
from pandas import Series, DataFrame

from ..base import MomentumStrategy
from ..registry import register_strategy


@register_strategy('momentum_stochastic')
class StochasticStrategy(MomentumStrategy):
    """
    Stochastic Crossover 策略

    策略邏輯：
    - 多單進場：%K 向上穿越 %D 且在超賣區（< oversold）
    - 多單出場：%K 離開超買區（從 > overbought 變成 <= overbought）
    - 空單進場：%K 向下穿越 %D 且在超買區（> overbought）
    - 空單出場：%K 離開超賣區（從 < oversold 變成 >= oversold）

    Parameters:
        k_period (int): %K 計算週期，預設 14
        d_period (int): %D 平滑週期，預設 3
        smooth_k (int): %K 平滑週期，預設 3
        overbought (int): 超買閾值，預設 80
        oversold (int): 超賣閾值，預設 20

    Example:
        >>> strategy = StochasticStrategy(k_period=14, d_period=3, smooth_k=3)
        >>> long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(data)
    """

    name = "Stochastic Crossover"
    strategy_type = "momentum"
    version = "1.0"
    description = "Stochastic oscillator crossover strategy with overbought/oversold zones"

    # 預設參數
    params = {
        'k_period': 14,
        'd_period': 3,
        'smooth_k': 3,
        'overbought': 80,
        'oversold': 20,
    }

    # 參數優化空間
    param_space = {
        'k_period': {
            'type': 'int',
            'low': 7,
            'high': 21,
        },
        'd_period': {
            'type': 'int',
            'low': 2,
            'high': 5,
        },
        'smooth_k': {
            'type': 'int',
            'low': 2,
            'high': 5,
        },
        'overbought': {
            'type': 'int',
            'low': 70,
            'high': 90,
        },
        'oversold': {
            'type': 'int',
            'low': 10,
            'high': 30,
        },
    }

    def __init__(self, **kwargs):
        """
        初始化策略

        Args:
            **kwargs: 覆寫預設參數
        """
        # 複製類別預設參數到實例
        self.params = self.__class__.params.copy()
        self.param_space = self.__class__.param_space.copy()

        # 更新傳入的參數
        self.params.update(kwargs)

        # 驗證參數
        if not self.validate_params():
            raise ValueError(f"Invalid parameters for {self.name}")

    def validate_params(self) -> bool:
        """
        驗證參數有效性

        Returns:
            bool: 參數是否有效
        """
        # 基礎驗證
        if not super().validate_params():
            return False

        # 週期必須 > 0
        if self.params['k_period'] <= 0:
            return False
        if self.params['d_period'] <= 0:
            return False
        if self.params['smooth_k'] <= 0:
            return False

        # 超賣必須 < 超買
        if self.params['oversold'] >= self.params['overbought']:
            return False

        # 超賣和超買必須在 0-100 範圍內
        if not (0 < self.params['oversold'] < 100):
            return False
        if not (0 < self.params['overbought'] < 100):
            return False

        return True

    def calculate_stochastic(
        self,
        high: Series,
        low: Series,
        close: Series,
        k_period: int,
        d_period: int,
        smooth_k: int
    ) -> Tuple[Series, Series]:
        """
        計算 Stochastic Oscillator

        Args:
            high: 最高價 Series
            low: 最低價 Series
            close: 收盤價 Series
            k_period: %K 計算週期
            d_period: %D 平滑週期
            smooth_k: %K 平滑週期

        Returns:
            tuple: (slow_k, slow_d)
                   slow_k: Smoothed %K
                   slow_d: %D (signal line)
        """
        # 計算 Raw %K (Fast Stochastic)
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()

        # 處理除零情況
        denominator = highest_high - lowest_low
        # 當 highest_high == lowest_low 時，設定 fast_k = 50（中性值）
        fast_k = pd.Series(50.0, index=close.index)
        valid_mask = denominator > 0
        fast_k[valid_mask] = 100 * (close[valid_mask] - lowest_low[valid_mask]) / denominator[valid_mask]

        # Smoothed %K (Slow Stochastic)
        slow_k = fast_k.rolling(smooth_k).mean()

        # %D (Signal line)
        slow_d = slow_k.rolling(d_period).mean()

        return slow_k, slow_d

    def calculate_indicators(self, data: DataFrame) -> Dict[str, Series]:
        """
        計算 Stochastic 指標

        Args:
            data: OHLCV DataFrame

        Returns:
            dict: {
                'stoch_k': Smoothed %K,
                'stoch_d': %D (signal line)
            }
        """
        slow_k, slow_d = self.calculate_stochastic(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            k_period=self.params['k_period'],
            d_period=self.params['d_period'],
            smooth_k=self.params['smooth_k']
        )

        return {
            'stoch_k': slow_k,
            'stoch_d': slow_d
        }

    def generate_signals(
        self,
        data: DataFrame
    ) -> Tuple[Series, Series, Series, Series]:
        """
        產生交易訊號

        Args:
            data: OHLCV DataFrame

        Returns:
            tuple: (long_entry, long_exit, short_entry, short_exit)
        """
        # 計算指標
        indicators = self.calculate_indicators(data)
        stoch_k = indicators['stoch_k']
        stoch_d = indicators['stoch_d']

        # 初始化訊號
        long_entry = pd.Series(False, index=data.index)
        long_exit = pd.Series(False, index=data.index)
        short_entry = pd.Series(False, index=data.index)
        short_exit = pd.Series(False, index=data.index)

        # 偵測交叉
        # %K 向上穿越 %D（黃金交叉）
        golden_cross = (stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1))

        # %K 向下穿越 %D（死亡交叉）
        death_cross = (stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1))

        # 超買超賣區判定
        in_oversold = stoch_k < self.params['oversold']
        in_overbought = stoch_k > self.params['overbought']

        # 離開超買區（從超買變成不超買）
        exit_overbought = (stoch_k <= self.params['overbought']) & (stoch_k.shift(1) > self.params['overbought'])

        # 離開超賣區（從超賣變成不超賣）
        exit_oversold = (stoch_k >= self.params['oversold']) & (stoch_k.shift(1) < self.params['oversold'])

        # 多單訊號：黃金交叉且在超賣區
        long_entry = golden_cross & in_oversold

        # 多單出場：離開超買區
        long_exit = exit_overbought

        # 空單訊號：死亡交叉且在超買區
        short_entry = death_cross & in_overbought

        # 空單出場：離開超賣區
        short_exit = exit_oversold

        return long_entry, long_exit, short_entry, short_exit

    def apply_filters(
        self,
        data: DataFrame,
        long_entry: Series,
        long_exit: Series,
        short_entry: Series,
        short_exit: Series
    ) -> Tuple[Series, Series, Series, Series]:
        """
        應用額外過濾器（可選覆寫）

        這裡可以加入額外的過濾條件，例如：
        - 成交量過濾
        - 波動度過濾
        - 時間過濾等

        Args:
            data: OHLCV DataFrame
            long_entry: 多單進場訊號
            long_exit: 多單出場訊號
            short_entry: 空單進場訊號
            short_exit: 空單出場訊號

        Returns:
            tuple: 過濾後的訊號
        """
        # 預設不加入額外過濾
        return long_entry, long_exit, short_entry, short_exit
