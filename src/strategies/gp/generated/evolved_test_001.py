"""
GP 演化策略: evolved_test_001

自動生成於: 2026-01-17T18:59:15.531371Z
適應度: 1.8500
表達式: CROSS_BELOW(RSI(high, period_50), MACD(high, bb_std_2, period_20))
"""

from src.strategies.gp.evolved_strategy import EvolvedStrategy
from src.gp.primitives import *
import numpy as np
import pandas as pd


class EvolvedTest001(EvolvedStrategy):
    """
    GP 演化策略

    表達式: CROSS_BELOW(RSI(high, period_50), MACD(high, bb_std_2, period_20))
    適應度: 1.8500
    代數: 10
    """

    name = "evolved_test_001"
    version = "1.0"
    description = "GP evolved strategy with fitness 1.8500"

    # 演化元資料
    expression = "CROSS_BELOW(RSI(high, period_50), MACD(high, bb_std_2, period_20))"
    fitness_score = 1.85
    generation = 10
    evolved_at = "2026-01-17T18:59:15.531371Z"

    def __init__(self, **kwargs):
        """初始化演化策略"""
        super().__init__(**kwargs)
        # 編譯訊號函數
        self._signal_func = self._build_signal_func()

    def _build_signal_func(self):
        """建立訊號函數"""
        def signal_func(close, high, low):
            """
            GP 演化的訊號函數

            Args:
                close: 收盤價序列
                high: 最高價序列
                low: 最低價序列

            Returns:
                布林陣列：True 表示進場訊號
            """
            # GP 表達式
            return CROSS_BELOW(RSI(high, period_50), MACD(high, bb_std_2, period_20))

        return signal_func
