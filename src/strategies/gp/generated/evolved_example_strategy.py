"""
GP 演化策略: evolved_example_strategy

自動生成於: 2026-01-17T19:02:30.762083Z
適應度: 1.8500
表達式: cross_above(bb_lower(low, 20, 50), bb_lower(low, 70, 26))
"""

from src.strategies.gp.evolved_strategy import EvolvedStrategy
from src.gp.primitives import *
import numpy as np
import pandas as pd


class EvolvedExampleStrategy(EvolvedStrategy):
    """
    GP 演化策略

    表達式: cross_above(bb_lower(low, 20, 50), bb_lower(low, 70, 26))
    適應度: 1.8500
    代數: 100
    """

    name = "evolved_example_strategy"
    version = "1.0"
    description = "GP evolved strategy with fitness 1.8500"

    # 演化元資料
    expression = "cross_above(bb_lower(low, 20, 50), bb_lower(low, 70, 26))"
    fitness_score = 1.85
    generation = 100
    evolved_at = "2026-01-17T19:02:30.762083Z"

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
            return cross_above(bb_lower(low, 20, 50), bb_lower(low, 70, 26))

        return signal_func
