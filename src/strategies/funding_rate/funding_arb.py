"""
Delta Neutral 資金費率套利策略

策略邏輯：
- 高正費率時：做空永續 + 做多現貨（收取費率）
- 高負費率時：做多永續 + 做空現貨（收取費率）
- 費率回到正常水平時出場

特點：
- 市場中性（Delta Neutral）
- 低風險套利
- 依賴資金費率異常
"""

import pandas as pd
from pandas import Series, DataFrame
from typing import Dict, Tuple

from ..base import FundingRateStrategy
from ..registry import register_strategy


@register_strategy('funding_rate_arb')
class FundingArbStrategy(FundingRateStrategy):
    """
    Delta Neutral 資金費率套利策略

    當資金費率過高時，做空永續並做多現貨以收取費率。
    當資金費率過低（負）時，做多永續並做空現貨以收取費率。
    當費率回歸正常時出場。

    參數：
        entry_rate: 進場費率閾值（絕對值），預設 0.0003 (0.03%)
        exit_rate: 出場費率閾值（絕對值），預設 0.0001 (0.01%)
        min_holding_periods: 最少持有結算次數，預設 1

    參數空間（優化用）：
        entry_rate: float, 0.0001-0.001
        exit_rate: float, 0.00005-0.0003
        min_holding_periods: int, 1-3
    """

    name = "funding_rate_arb"
    strategy_type = "funding_rate"
    version = "1.0"
    description = "Delta Neutral funding rate arbitrage strategy"

    # 預設參數
    params = {
        'entry_rate': 0.0003,
        'exit_rate': 0.0001,
        'min_holding_periods': 1,
    }

    # 參數優化空間
    param_space = {
        'entry_rate': {
            'type': 'float',
            'low': 0.0001,
            'high': 0.001,
        },
        'exit_rate': {
            'type': 'float',
            'low': 0.00005,
            'high': 0.0003,
        },
        'min_holding_periods': {
            'type': 'int',
            'low': 1,
            'high': 3,
        }
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

    def calculate_indicators(self, data: DataFrame) -> Dict[str, Series]:
        """
        計算策略所需指標

        對於純資金費率策略，不需要價格指標。

        Args:
            data: OHLCV DataFrame

        Returns:
            dict: 空字典（此策略不使用價格指標）
        """
        return {}

    def generate_signals(self, data: DataFrame) -> Tuple[Series, Series, Series, Series]:
        """
        產生交易訊號（無資金費率數據）

        如果沒有資金費率數據，無法產生訊號。

        Args:
            data: OHLCV DataFrame

        Returns:
            tuple: 全部為 False 的訊號（因為缺少資金費率）
        """
        # 沒有資金費率數據，無法產生訊號
        length = len(data)
        empty_signal = pd.Series([False] * length, index=data.index)
        return empty_signal, empty_signal, empty_signal, empty_signal

    def generate_signals_with_funding(
        self,
        data: DataFrame,
        funding_rates: Series
    ) -> Tuple[Series, Series, Series, Series]:
        """
        產生考慮資金費率的交易訊號

        策略邏輯：
        - 進場：|funding_rate| >= entry_rate
            - funding_rate > entry_rate → 做空永續（short_entry）
            - funding_rate < -entry_rate → 做多永續（long_entry）
        - 出場：|funding_rate| <= exit_rate 且持有 >= min_holding_periods
            - 有多單 → long_exit
            - 有空單 → short_exit

        Args:
            data: OHLCV DataFrame
            funding_rates: 資金費率 Series

        Returns:
            tuple: (long_entry, long_exit, short_entry, short_exit)
        """
        entry_rate = self.params['entry_rate']
        exit_rate = self.params['exit_rate']
        min_holding_periods = self.params['min_holding_periods']

        # 對齊資金費率與數據
        if len(funding_rates) != len(data):
            # 嘗試使用時間對齊
            funding_rates = funding_rates.reindex(data.index, method='ffill')

        # 初始化訊號
        long_entry = pd.Series([False] * len(data), index=data.index)
        long_exit = pd.Series([False] * len(data), index=data.index)
        short_entry = pd.Series([False] * len(data), index=data.index)
        short_exit = pd.Series([False] * len(data), index=data.index)

        # 追蹤持有狀態和持有期數
        position = 0  # 0=空倉, 1=多單, -1=空單
        holding_periods = 0

        for i in range(len(data)):
            current_rate = funding_rates.iloc[i]

            # 跳過 NaN
            if pd.isna(current_rate):
                continue

            # 無持倉：判斷進場
            if position == 0:
                # 高正費率：做空永續（收取正費率）
                if current_rate >= entry_rate:
                    short_entry.iloc[i] = True
                    position = -1
                    holding_periods = 0

                # 高負費率：做多永續（收取負費率）
                elif current_rate <= -entry_rate:
                    long_entry.iloc[i] = True
                    position = 1
                    holding_periods = 0

            # 有持倉：判斷出場
            else:
                # 累計持有期數（假設每個 bar 代表一個結算週期）
                holding_periods += 1

                # 檢查出場條件
                exit_condition = (
                    abs(current_rate) <= exit_rate
                    and holding_periods >= min_holding_periods
                )

                if exit_condition:
                    if position == 1:
                        long_exit.iloc[i] = True
                        position = 0
                        holding_periods = 0
                    elif position == -1:
                        short_exit.iloc[i] = True
                        position = 0
                        holding_periods = 0

        return long_entry, long_exit, short_entry, short_exit

    def validate_params(self) -> bool:
        """
        驗證參數有效性

        Returns:
            bool: 參數是否有效
        """
        entry_rate = self.params.get('entry_rate')
        exit_rate = self.params.get('exit_rate')
        min_holding_periods = self.params.get('min_holding_periods')

        # 檢查參數存在（允許值為 0）
        if None in [entry_rate, exit_rate, min_holding_periods]:
            return False

        if entry_rate <= 0 or exit_rate <= 0:
            return False

        # 進場門檻應大於出場門檻
        if entry_rate <= exit_rate:
            return False

        # 最少持有期數應為正整數
        if min_holding_periods < 1:
            return False

        return True

    def calculate_expected_annual_return(
        self,
        funding_rates: Series,
        settlements_per_day: int = 3
    ) -> float:
        """
        計算預期年化收益

        基於歷史費率平均值估算年化收益。

        Args:
            funding_rates: 歷史資金費率 Series
            settlements_per_day: 每日結算次數（預設 3）

        Returns:
            float: 預期年化收益率
        """
        # 過濾符合進場條件的費率
        entry_rate = self.params['entry_rate']
        qualified_rates = funding_rates[abs(funding_rates) >= entry_rate]

        if len(qualified_rates) == 0:
            return 0.0

        # 平均費率
        avg_rate = qualified_rates.abs().mean()

        # 年化收益 = 平均費率 × 每日結算次數 × 365
        annual_return = avg_rate * settlements_per_day * 365

        return annual_return

    def __repr__(self) -> str:
        """字串表示"""
        return (
            f"{self.name}(entry_rate={self.params['entry_rate']:.4f}, "
            f"exit_rate={self.params['exit_rate']:.4f}, "
            f"min_holding={self.params['min_holding_periods']})"
        )
