"""
Funding Rate Settlement Trade Strategy

結算時段交易策略 - 利用資金費率結算前的市場行為進行短期交易。

策略邏輯：
1. 在結算前 hours_before 小時監測費率
2. 如果費率 > rate_threshold（高資金費率）：
   - 做多（預期多頭會在結算前平倉，推高價格）
3. 如果費率 < -rate_threshold（負資金費率）：
   - 做空（預期空頭會在結算前平倉，壓低價格）
4. 結算後立即平倉

理論依據：
- 高資金費率時，多頭持有人需支付費用，可能在結算前平倉
- 平倉行為會造成短期價格上漲壓力
- 策略利用這種短期波動獲利
"""

import pandas as pd
from typing import Dict, Tuple
from pandas import Series, DataFrame

from ..base import FundingRateStrategy
from ..registry import register_strategy


@register_strategy('funding_rate_settlement')
class SettlementTradeStrategy(FundingRateStrategy):
    """
    結算時段交易策略

    Attributes:
        rate_threshold (float): 資金費率閾值，超過此值觸發交易
        hours_before_settlement (int): 結算前幾小時開始進場
    """

    name = "settlement_trade"
    version = "1.0"
    description = "Settlement period trading based on funding rate extremes"

    # 類別屬性：參數優化空間（供 StrategyRegistry.get_param_space() 使用）
    param_space = {
        'rate_threshold': {
            'type': 'float',
            'low': 0.00005,
            'high': 0.0005,
            'log': True  # 對數空間搜尋
        },
        'hours_before_settlement': {
            'type': 'int',
            'low': 1,
            'high': 4
        }
    }

    def __init__(
        self,
        rate_threshold: float = 0.0001,
        hours_before_settlement: int = 1,
        **kwargs
    ):
        """
        初始化結算交易策略

        Args:
            rate_threshold: 費率閾值（0.0001 = 0.01%）
            hours_before_settlement: 結算前幾小時進場（1-4）
            **kwargs: 其他基礎參數
        """
        # 準備參數
        params = {
            'rate_threshold': rate_threshold,
            'hours_before_settlement': hours_before_settlement
        }
        params.update(kwargs)

        # 呼叫父類別初始化（會執行驗證）
        super().__init__(**params)

    def calculate_indicators(self, data: DataFrame) -> Dict[str, Series]:
        """
        計算策略指標

        結算交易策略主要依賴資金費率，不需要額外技術指標。

        Args:
            data: OHLCV DataFrame

        Returns:
            dict: 空字典（策略不需要額外指標）
        """
        # 不需要技術指標，只需要資金費率數據
        return {}

    def generate_signals(
        self,
        data: DataFrame
    ) -> Tuple[Series, Series, Series, Series]:
        """
        產生標準交易訊號（無資金費率數據）

        如果沒有資金費率數據，此策略無法運作。
        返回全部 False 的訊號。

        Args:
            data: OHLCV DataFrame

        Returns:
            tuple: (long_entry, long_exit, short_entry, short_exit)
                   全部為 False Series
        """
        length = len(data)
        false_series = pd.Series([False] * length, index=data.index)

        return (
            false_series.copy(),  # long_entry
            false_series.copy(),  # long_exit
            false_series.copy(),  # short_entry
            false_series.copy()   # short_exit
        )

    def generate_signals_with_funding(
        self,
        data: DataFrame,
        funding_rates: Series
    ) -> Tuple[Series, Series, Series, Series]:
        """
        產生考慮資金費率的交易訊號

        Args:
            data: OHLCV DataFrame（必須包含 timestamp 作為 index 或欄位）
            funding_rates: 資金費率 Series（與 data 時間對齊）

        Returns:
            tuple: (long_entry, long_exit, short_entry, short_exit)
        """
        # 確保資料對齊
        if len(data) != len(funding_rates):
            raise ValueError(
                f"Data and funding rates length mismatch: {len(data)} vs {len(funding_rates)}"
            )

        # 初始化訊號 Series
        long_entry = pd.Series([False] * len(data), index=data.index)
        long_exit = pd.Series([False] * len(data), index=data.index)
        short_entry = pd.Series([False] * len(data), index=data.index)
        short_exit = pd.Series([False] * len(data), index=data.index)

        # 提取參數
        rate_threshold = self.params['rate_threshold']
        hours_before = self.params['hours_before_settlement']

        # 確保 data 的 index 是 DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be DatetimeIndex")

        # 遍歷每個時間點
        for i in range(len(data)):
            timestamp = data.index[i]
            rate = funding_rates.iloc[i]

            # 跳過 NaN 費率
            if pd.isna(rate):
                continue

            # 判斷是否接近結算時間
            is_near_settlement = self.is_settlement_hour(
                timestamp,
                hours_before=hours_before
            )

            # 判斷是否在結算時間（用於出場）
            is_settlement = self.is_settlement_hour(
                timestamp,
                hours_before=0
            )

            # 進場邏輯：結算前 + 費率極端值
            if is_near_settlement and not is_settlement:
                # 高費率（多頭支付） → 預期多頭平倉 → 做多
                if rate > rate_threshold:
                    long_entry.iloc[i] = True

                # 負費率（空頭支付） → 預期空頭平倉 → 做空
                elif rate < -rate_threshold:
                    short_entry.iloc[i] = True

            # 出場邏輯：結算時立即平倉
            if is_settlement:
                long_exit.iloc[i] = True
                short_exit.iloc[i] = True

        return long_entry, long_exit, short_entry, short_exit

    def validate_params(self) -> bool:
        """
        驗證參數有效性

        Returns:
            bool: 參數是否有效
        """
        # 呼叫父類別驗證
        if not super().validate_params():
            return False

        # 驗證 rate_threshold
        rate_threshold = self.params.get('rate_threshold')
        if rate_threshold is None or rate_threshold <= 0 or rate_threshold > 0.01:
            return False

        # 驗證 hours_before_settlement
        hours_before = self.params.get('hours_before_settlement')
        if hours_before is None or hours_before < 1 or hours_before > 4:
            return False

        return True

    def __repr__(self) -> str:
        """字串表示"""
        return (
            f"SettlementTradeStrategy("
            f"rate_threshold={self.params['rate_threshold']:.5f}, "
            f"hours_before={self.params['hours_before_settlement']})"
        )
