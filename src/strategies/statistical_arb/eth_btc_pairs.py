"""
ETH/BTC 配對交易策略

基於 ETH/BTC 比率的統計套利策略。
當比率偏離均值時進場，回歸均值時出場。
"""

import pandas as pd
from typing import Dict, Tuple
from pandas import Series, DataFrame

from ..base import StatisticalArbStrategy
from ..registry import register_strategy


@register_strategy('statistical_arb_eth_btc_pairs')
class ETHBTCPairsStrategy(StatisticalArbStrategy):
    """
    ETH/BTC 配對交易策略

    策略邏輯：
    - 計算 ETH/BTC 比率
    - Z-Score > z_threshold 時做空 ETH、做多 BTC（比率偏高，預期回歸）
    - Z-Score < -z_threshold 時做多 ETH、做空 BTC（比率偏低，預期回歸）
    - Z-Score 回到 ±exit_z 範圍內時出場

    Parameters:
        period (int): Z-Score 計算週期，預設 20
        z_threshold (float): 進場閾值，預設 2.0
        exit_z (float): 出場閾值，預設 0.5
        spread_method (str): 比率計算方法，預設 'ratio'

    Example:
        >>> # 單標的模式（只用 ETH 數據，假設 BTC 比率為固定值）
        >>> strategy = ETHBTCPairsStrategy(period=20, z_threshold=2.0)
        >>> long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(data_eth)

        >>> # 雙標的模式（使用 ETH 和 BTC 數據）
        >>> long_entry, long_exit, short_entry, short_exit = strategy.generate_signals_dual(data_eth, data_btc)
    """

    name = "ETH/BTC Pairs Trading"
    strategy_type = "statistical_arbitrage"
    version = "1.0"
    description = "Statistical arbitrage based on ETH/BTC ratio mean reversion"

    # 預設參數
    params = {
        'period': 20,          # Z-Score 計算週期
        'z_threshold': 2.0,    # 進場閾值
        'exit_z': 0.5,         # 出場閾值
        'spread_method': 'ratio',  # 比率計算方法
    }

    # 參數優化空間
    param_space = {
        'period': {
            'type': 'int',
            'low': 10,
            'high': 50,
        },
        'z_threshold': {
            'type': 'float',
            'low': 1.5,
            'high': 3.0,
        },
        'exit_z': {
            'type': 'float',
            'low': 0.0,
            'high': 1.0,
        },
    }

    def __init__(self, **kwargs):
        """
        初始化策略

        Args:
            **kwargs: 覆寫預設參數
        """
        # 合併預設參數與傳入參數
        merged_params = self.params.copy()
        merged_params.update(kwargs)

        # 呼叫父類別初始化（會執行參數驗證）
        super().__init__(**merged_params)

        # 復原 param_space（基礎類別會重置為空字典）
        self.param_space = self.__class__.param_space.copy()

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
        if self.params['period'] <= 0:
            return False

        # z_threshold 必須 > 0
        if self.params['z_threshold'] <= 0:
            return False

        # exit_z 必須在 0 到 z_threshold 之間
        if not (0 <= self.params['exit_z'] < self.params['z_threshold']):
            return False

        # spread_method 必須是支援的方法
        valid_methods = ['ratio', 'diff', 'log_ratio']
        if self.params['spread_method'] not in valid_methods:
            return False

        return True

    def calculate_indicators(self, data: DataFrame) -> Dict[str, Series]:
        """
        計算策略所需指標（單標的模式）

        註：單標的模式下，假設 BTC 比率為固定值（用於簡單回測）
        實際使用建議使用 generate_signals_dual() 雙標的模式

        Args:
            data: ETH OHLCV DataFrame

        Returns:
            dict: {
                'spread': 價差/比率,
                'zscore': Z-Score
            }
        """
        indicators = {}

        # 單標的模式：使用 ETH 價格本身作為 spread
        # （假設 BTC 為固定基準，實際應使用 generate_signals_dual）
        spread = data['close']
        indicators['spread'] = spread

        # 計算 Z-Score
        zscore = self.calculate_zscore(
            spread,
            period=self.params['period']
        )
        indicators['zscore'] = zscore

        return indicators

    def generate_signals(
        self,
        data: DataFrame
    ) -> Tuple[Series, Series, Series, Series]:
        """
        產生交易訊號（單標的模式）

        註：單標的模式僅用於簡單測試。
        實際使用請使用 generate_signals_dual() 雙標的模式。

        Args:
            data: ETH OHLCV DataFrame

        Returns:
            tuple: (long_entry, long_exit, short_entry, short_exit)
        """
        # 計算指標
        indicators = self.calculate_indicators(data)
        zscore = indicators['zscore']

        # Z-Score 條件
        z_high = zscore > self.params['z_threshold']  # 比率偏高
        z_low = zscore < -self.params['z_threshold']  # 比率偏低
        z_neutral_up = (zscore < self.params['exit_z']) & (zscore.shift(1) >= self.params['exit_z'])  # 回歸向下
        z_neutral_down = (zscore > -self.params['exit_z']) & (zscore.shift(1) <= -self.params['exit_z'])  # 回歸向上

        # 多單訊號：Z-Score 偏低時做多 ETH（預期比率上升）
        long_entry = z_low
        long_exit = z_neutral_down

        # 空單訊號：Z-Score 偏高時做空 ETH（預期比率下降）
        short_entry = z_high
        short_exit = z_neutral_up

        return long_entry, long_exit, short_entry, short_exit

    def generate_signals_dual(
        self,
        data_primary: DataFrame,
        data_secondary: DataFrame
    ) -> Tuple[Series, Series, Series, Series]:
        """
        產生雙標的交易訊號（推薦使用）

        Args:
            data_primary: ETH OHLCV DataFrame
            data_secondary: BTC OHLCV DataFrame

        Returns:
            tuple: (long_entry, long_exit, short_entry, short_exit)
                   long = 做多 ETH，做空 BTC（比率偏低，預期上升）
                   short = 做空 ETH，做多 BTC（比率偏高，預期下降）
        """
        # 對齊時間索引
        common_index = data_primary.index.intersection(data_secondary.index)
        if len(common_index) == 0:
            raise ValueError("No common timestamps between primary and secondary data")

        data_primary = data_primary.loc[common_index]
        data_secondary = data_secondary.loc[common_index]

        # 計算價差/比率
        spread = self.calculate_spread(
            data_primary['close'],
            data_secondary['close'],
            method=self.params['spread_method']
        )

        # 計算 Z-Score
        zscore = self.calculate_zscore(
            spread,
            period=self.params['period']
        )

        # 初始化訊號
        long_entry = pd.Series(False, index=common_index)
        long_exit = pd.Series(False, index=common_index)
        short_entry = pd.Series(False, index=common_index)
        short_exit = pd.Series(False, index=common_index)

        # Z-Score 條件
        z_high = zscore > self.params['z_threshold']  # 比率偏高
        z_low = zscore < -self.params['z_threshold']  # 比率偏低
        z_neutral_up = (zscore < self.params['exit_z']) & (zscore.shift(1) >= self.params['exit_z'])  # 回歸向下
        z_neutral_down = (zscore > -self.params['exit_z']) & (zscore.shift(1) <= -self.params['exit_z'])  # 回歸向上

        # 多單訊號：Z-Score 偏低時做多 ETH、做空 BTC（預期比率上升）
        long_entry = z_low
        long_exit = z_neutral_down

        # 空單訊號：Z-Score 偏高時做空 ETH、做多 BTC（預期比率下降）
        short_entry = z_high
        short_exit = z_neutral_up

        return long_entry, long_exit, short_entry, short_exit

    def calculate_correlation(
        self,
        data_primary: DataFrame,
        data_secondary: DataFrame,
        period: int = 60
    ) -> Series:
        """
        計算滾動相關係數（用於評估配對有效性）

        Args:
            data_primary: ETH OHLCV DataFrame
            data_secondary: BTC OHLCV DataFrame
            period: 滾動視窗週期

        Returns:
            Series: 滾動相關係數
        """
        # 對齊時間索引
        common_index = data_primary.index.intersection(data_secondary.index)
        data_primary = data_primary.loc[common_index]
        data_secondary = data_secondary.loc[common_index]

        # 計算滾動相關係數
        correlation = data_primary['close'].rolling(period).corr(data_secondary['close'])
        return correlation

    def get_pair_statistics(
        self,
        data_primary: DataFrame,
        data_secondary: DataFrame
    ) -> Dict:
        """
        取得配對統計資訊

        Args:
            data_primary: ETH OHLCV DataFrame
            data_secondary: BTC OHLCV DataFrame

        Returns:
            dict: 配對統計資訊
        """
        # 計算價差
        spread = self.calculate_spread(
            data_primary['close'],
            data_secondary['close'],
            method=self.params['spread_method']
        )

        # 計算半衰期
        half_life = self.calculate_half_life(spread)

        # 計算相關係數
        correlation = self.calculate_correlation(data_primary, data_secondary)

        return {
            'half_life': half_life,
            'correlation_mean': correlation.mean(),
            'correlation_std': correlation.std(),
            'spread_mean': spread.mean(),
            'spread_std': spread.std(),
            'spread_min': spread.min(),
            'spread_max': spread.max(),
        }
