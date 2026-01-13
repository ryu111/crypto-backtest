"""
基差套利策略

基於永續合約相對於現貨的基差進行套利交易。
當基差偏離正常範圍時，做反向操作等待回歸。
"""

import pandas as pd
from typing import Dict, Tuple
from pandas import Series, DataFrame

from ..base import StatisticalArbStrategy
from ..registry import register_strategy


@register_strategy('statistical_arb_basis')
class BasisArbStrategy(StatisticalArbStrategy):
    """
    基差套利策略（Basis Arbitrage）

    策略邏輯：
    永續合約相對於現貨的基差 = (永續價格 - 現貨價格) / 現貨價格

    - 基差 > entry_threshold（永續溢價過高）：
      做空永續 + 做多現貨（long = 做多主標的，short = 做空主標的）
    - 基差 < -entry_threshold（永續折價過多）：
      做多永續 + 做空現貨
    - 基差回到 ±exit_threshold 範圍內時出場

    Parameters:
        entry_threshold (float): 基差進場閾值（百分比），預設 0.005 (0.5%)
        exit_threshold (float): 基差出場閾值（百分比），預設 0.001 (0.1%)
        period (int): 移動平均週期（用於平滑基差），預設 20
        use_ma (bool): 是否使用移動平均平滑基差，預設 True

    Example:
        >>> # 雙標的模式（永續 + 現貨）
        >>> strategy = BasisArbStrategy(entry_threshold=0.005, exit_threshold=0.001)
        >>> signals = strategy.generate_signals_dual(perp_data, spot_data)
        >>>
        >>> # 單標的模式（僅永續，無現貨數據）
        >>> signals = strategy.generate_signals(perp_data)  # 回傳空訊號
    """

    name = "Basis Arbitrage"
    strategy_type = "statistical_arbitrage"
    version = "1.0"
    description = "Perpetual-Spot basis arbitrage strategy"

    # 預設參數
    params = {
        'entry_threshold': 0.005,   # 0.5% 基差進場
        'exit_threshold': 0.001,     # 0.1% 基差出場
        'period': 20,                # 移動平均週期
        'use_ma': True,              # 使用移動平均平滑
    }

    # 參數優化空間
    param_space = {
        'entry_threshold': {
            'type': 'float',
            'low': 0.003,
            'high': 0.01,
        },
        'exit_threshold': {
            'type': 'float',
            'low': 0.0005,
            'high': 0.003,
        },
        'period': {
            'type': 'int',
            'low': 10,
            'high': 50,
        },
    }

    def __init__(self, **kwargs):
        """
        初始化基差套利策略

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
        if not all(v is not None for v in self.params.values()):
            return False

        # entry_threshold 必須 > exit_threshold
        if self.params['entry_threshold'] <= self.params['exit_threshold']:
            return False

        # 閾值必須 > 0
        if self.params['entry_threshold'] <= 0 or self.params['exit_threshold'] <= 0:
            return False

        # period 必須 > 0
        if self.params['period'] <= 0:
            return False

        return True

    def calculate_basis(
        self,
        perp_price: Series,
        spot_price: Series,
        use_ma: bool = True,
        period: int = 20
    ) -> Series:
        """
        計算基差（Basis）

        Args:
            perp_price: 永續合約價格 Series
            spot_price: 現貨價格 Series
            use_ma: 是否使用移動平均平滑
            period: 移動平均週期

        Returns:
            Series: 基差百分比 Series
        """
        # 基差 = (永續價格 - 現貨價格) / 現貨價格
        basis = (perp_price - spot_price) / spot_price

        # 使用移動平均平滑（降低噪音）
        if use_ma:
            basis = basis.rolling(period).mean()

        return basis

    def calculate_indicators(self, data: DataFrame) -> Dict[str, Series]:
        """
        計算策略所需指標

        注意：單標的模式下，此方法僅計算永續合約價格相關指標。
        實際基差計算需要在 generate_signals_dual() 中完成。

        Args:
            data: OHLCV DataFrame（永續合約數據）

        Returns:
            dict: {指標名稱: Series}
        """
        indicators = {}

        # 永續合約價格
        indicators['perp_price'] = data['close']

        # 移動平均（用於平滑）
        if self.params['use_ma']:
            indicators['ma'] = data['close'].rolling(self.params['period']).mean()

        return indicators

    def generate_signals(
        self,
        data: DataFrame
    ) -> Tuple[Series, Series, Series, Series]:
        """
        產生交易訊號（單標的模式）

        注意：基差套利需要雙標的（永續 + 現貨），單標的模式無法產生有效訊號。
        建議使用 generate_signals_dual() 方法。

        Args:
            data: OHLCV DataFrame（僅永續合約數據）

        Returns:
            tuple: (long_entry, long_exit, short_entry, short_exit)
                   全部為 False（無有效訊號）
        """
        # 單標的模式無法計算基差，回傳空訊號
        empty_signal = pd.Series(False, index=data.index)
        return empty_signal, empty_signal, empty_signal, empty_signal

    def generate_signals_dual(
        self,
        data_primary: DataFrame,
        data_secondary: DataFrame
    ) -> Tuple[Series, Series, Series, Series]:
        """
        產生雙標的交易訊號

        Args:
            data_primary: 主標的 OHLCV DataFrame（永續合約）
            data_secondary: 次標的 OHLCV DataFrame（現貨）

        Returns:
            tuple: (long_entry, long_exit, short_entry, short_exit)
                   long = 做多永續 + 做空現貨（基差為負）
                   short = 做空永續 + 做多現貨（基差為正）
        """
        # 對齊時間索引
        common_index = data_primary.index.intersection(data_secondary.index)
        if len(common_index) == 0:
            raise ValueError("No overlapping timestamps between primary and secondary data")

        data_primary = data_primary.loc[common_index]
        data_secondary = data_secondary.loc[common_index]

        # 提取價格
        perp_price = data_primary['close']
        spot_price = data_secondary['close']

        # 計算基差
        basis = self.calculate_basis(
            perp_price,
            spot_price,
            use_ma=self.params['use_ma'],
            period=self.params['period']
        )

        # 初始化訊號
        long_entry = pd.Series(False, index=data_primary.index)
        long_exit = pd.Series(False, index=data_primary.index)
        short_entry = pd.Series(False, index=data_primary.index)
        short_exit = pd.Series(False, index=data_primary.index)

        # 訊號條件
        entry_threshold = self.params['entry_threshold']
        exit_threshold = self.params['exit_threshold']

        # 做多訊號：基差 < -entry_threshold（永續折價）
        # 做多永續 + 做空現貨
        long_entry = basis < -entry_threshold

        # 做多出場：基差回到 -exit_threshold 以上
        long_exit = basis > -exit_threshold

        # 做空訊號：基差 > entry_threshold（永續溢價）
        # 做空永續 + 做多現貨
        short_entry = basis > entry_threshold

        # 做空出場：基差回到 exit_threshold 以下
        short_exit = basis < exit_threshold

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

        可以加入：
        - 成交量過濾（避免流動性不足）
        - 波動度過濾（避免異常波動期間）
        - 時間過濾（避開結算時段等）

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

    def get_info(self) -> Dict:
        """
        取得策略詳細資訊

        Returns:
            dict: 策略資訊
        """
        info = super().get_info()
        info['requires_dual_data'] = True
        info['data_primary'] = 'perpetual'
        info['data_secondary'] = 'spot'
        return info
