"""
布林帶均值回歸策略 (Bollinger Bands Mean Reversion)

策略邏輯：
- 進場：價格觸及下軌時做多（超賣反彈），觸及上軌時做空（超買回落）
- 出場：價格回到中軌時平倉
- 過濾：可選擇在趨勢過濾下只做單邊交易

適用市場：
- 橫盤震盪市場（ranging market）
- 波動率正常至偏低的環境
- 中短期持倉（15m, 1H, 4H 時間框）

參數說明：
- period: 布林帶計算週期（建議 10-30）
- std_dev: 標準差倍數（建議 1.5-3.0），越大帶寬越寬
- exit_at_middle: 是否在中軌出場（True）或反向觸及對側軌道才出場（False）
"""

import pandas as pd
from typing import Dict, Tuple
from pandas import Series, DataFrame

from ..base import MeanReversionStrategy
from ..registry import register_strategy


@register_strategy('mean_reversion_bollinger')
class BollingerMeanReversionStrategy(MeanReversionStrategy):
    """
    布林帶均值回歸策略

    當價格觸及布林帶上軌（超買）時做空，觸及下軌（超賣）時做多，
    預期價格會回歸中軌（移動平均線）。
    """

    name = "Bollinger Bands Mean Reversion"
    strategy_type = "mean_reversion"
    version = "1.0"
    description = "布林帶均值回歸策略，適用於震盪市場"

    # 預設參數
    params = {
        'period': 20,
        'std_dev': 2.0,
        'exit_at_middle': True,
    }

    # Optuna 優化空間
    param_space = {
        'period': {'type': 'int', 'low': 10, 'high': 30},
        'std_dev': {'type': 'float', 'low': 1.5, 'high': 3.0},
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
        """驗證參數有效性"""
        # 週期必須大於 1
        if self.params['period'] <= 1:
            return False

        # 標準差倍數必須為正
        if self.params['std_dev'] <= 0:
            return False

        return super().validate_params()

    def calculate_indicators(self, data: DataFrame) -> Dict[str, Series]:
        """
        計算策略所需指標

        Args:
            data: OHLCV DataFrame

        Returns:
            dict: 包含以下指標
                - bb_upper: 布林帶上軌
                - bb_middle: 布林帶中軌（SMA）
                - bb_lower: 布林帶下軌
                - bb_width: 帶寬（用於識別波動率）
        """
        close = data['close']
        indicators = {}

        # 計算布林帶
        upper, middle, lower = self.calculate_bollinger_bands(
            close,
            period=self.params['period'],
            std_dev=self.params['std_dev']
        )

        indicators['bb_upper'] = upper
        indicators['bb_middle'] = middle
        indicators['bb_lower'] = lower

        # 計算帶寬（上軌 - 下軌）/ 中軌，用於識別波動率環境
        indicators['bb_width'] = (upper - lower) / middle

        return indicators

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

        close = data['close']
        bb_upper = indicators['bb_upper']
        bb_middle = indicators['bb_middle']
        bb_lower = indicators['bb_lower']

        # 初始化訊號
        long_entry = pd.Series(False, index=data.index)
        long_exit = pd.Series(False, index=data.index)
        short_entry = pd.Series(False, index=data.index)
        short_exit = pd.Series(False, index=data.index)

        # 進場訊號
        # 多單：價格觸及或穿越下軌（超賣）
        long_entry = close <= bb_lower

        # 空單：價格觸及或穿越上軌（超買）
        short_entry = close >= bb_upper

        # 出場訊號
        if self.params['exit_at_middle']:
            # 在中軌出場
            long_exit = close >= bb_middle
            short_exit = close <= bb_middle
        else:
            # 在反向觸及對側軌道時出場（更保守）
            long_exit = close >= bb_upper
            short_exit = close <= bb_lower

        return long_entry, long_exit, short_entry, short_exit

    def get_entry_reason(self, data: DataFrame, index: int) -> str:
        """
        取得進場原因說明

        Args:
            data: OHLCV DataFrame
            index: 資料索引

        Returns:
            str: 進場原因
        """
        indicators = self.calculate_indicators(data)

        close = data['close'].iloc[index]
        upper = indicators['bb_upper'].iloc[index]
        middle = indicators['bb_middle'].iloc[index]
        lower = indicators['bb_lower'].iloc[index]

        # 判斷是多單還是空單
        if close <= lower:
            distance_pct = ((lower - close) / close) * 100
            return f"超賣：價格({close:.2f}) 觸及下軌({lower:.2f})，偏離中軌 {distance_pct:.2f}%"
        elif close >= upper:
            distance_pct = ((close - upper) / close) * 100
            return f"超買：價格({close:.2f}) 觸及上軌({upper:.2f})，偏離中軌 {distance_pct:.2f}%"
        else:
            return f"價格({close:.2f}) 在布林帶內，中軌={middle:.2f}"
