"""
策略基礎類別

定義所有策略必須實作的介面和共用功能。
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any
from pandas import Series, DataFrame


class BaseStrategy(ABC):
    """
    策略基礎抽象類別

    所有自訂策略必須繼承此類別並實作必要方法。

    Attributes:
        name (str): 策略名稱
        strategy_type (str): 策略類型 (trend, momentum, mean_reversion, statistical_arbitrage, funding_rate)
        params (dict): 策略參數
        param_space (dict): 參數優化空間定義
        version (str): 策略版本
        description (str): 策略描述
    """

    # 子類別必須覆寫的類別屬性
    name: str = "base_strategy"
    strategy_type: str = "base"
    version: str = "1.0"
    description: str = "Base strategy template"

    def __init__(self, **kwargs):
        """
        初始化策略

        Args:
            **kwargs: 覆寫預設參數
        """
        # 初始化實例屬性（避免類別屬性共享問題）
        self.params: Dict[str, Any] = {}
        self.param_space: Dict[str, Any] = {}

        # 合併預設參數與傳入參數
        self.params.update(kwargs)

        # 驗證參數
        if not self.validate_params():
            raise ValueError(f"Invalid parameters for {self.name}")

    @abstractmethod
    def calculate_indicators(self, data: DataFrame) -> Dict[str, Series]:
        """
        計算策略所需指標

        Args:
            data: OHLCV DataFrame，包含 open, high, low, close, volume

        Returns:
            dict: {指標名稱: Series}

        Example:
            {
                'sma_fast': Series([...]),
                'sma_slow': Series([...]),
                'rsi': Series([...])
            }
        """
        pass

    @abstractmethod
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
                   每個都是 boolean Series，True 表示觸發訊號

        Example:
            long_entry = Series([False, True, False, ...])
            long_exit = Series([False, False, True, ...])
            short_entry = Series([False, False, False, ...])
            short_exit = Series([False, True, False, ...])
        """
        pass

    def position_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss_price: float,
        risk_per_trade: float = 0.02,
        max_position_pct: float = 1.0
    ) -> float:
        """
        計算部位大小（基於風險）

        Args:
            capital: 總資金
            entry_price: 入場價格
            stop_loss_price: 止損價格
            risk_per_trade: 單筆交易風險比例 (預設 2%)
            max_position_pct: 最大部位比例 (預設 100%)

        Returns:
            float: 部位大小
        """
        # 風險金額
        risk_amount = capital * risk_per_trade

        # 止損距離
        stop_distance = abs(entry_price - stop_loss_price)

        if stop_distance == 0:
            return 0

        # 基於風險計算部位
        size = risk_amount / stop_distance

        # 限制最大部位
        max_size = capital * max_position_pct / entry_price
        size = min(size, max_size)

        return size

    def validate_params(self) -> bool:
        """
        驗證參數有效性

        子類別可覆寫此方法實作自訂驗證邏輯。

        Returns:
            bool: 參數是否有效
        """
        # 基礎驗證：檢查所有參數值都不是 None
        return all(v is not None for v in self.params.values())

    def get_info(self) -> Dict:
        """
        取得策略詳細資訊

        Returns:
            dict: 策略資訊
        """
        return {
            'name': self.name,
            'type': self.strategy_type,
            'version': self.version,
            'description': self.description,
            'params': self.params,
            'param_space': self.param_space
        }

    def apply_filters(
        self,
        data: DataFrame,
        long_entry: Series,
        long_exit: Series,
        short_entry: Series,
        short_exit: Series
    ) -> Tuple[Series, Series, Series, Series]:
        """
        應用策略過濾器（可選覆寫）

        Args:
            data: OHLCV DataFrame
            long_entry: 多單進場訊號
            long_exit: 多單出場訊號
            short_entry: 空單進場訊號
            short_exit: 空單出場訊號

        Returns:
            tuple: 過濾後的訊號
        """
        # 預設不過濾
        return long_entry, long_exit, short_entry, short_exit

    def __repr__(self) -> str:
        """字串表示"""
        return f"{self.name}(type={self.strategy_type}, params={self.params})"

    def __str__(self) -> str:
        """友善字串表示"""
        return f"{self.name} v{self.version}"


class TrendStrategy(BaseStrategy):
    """趨勢跟隨策略基礎類別"""

    strategy_type = "trend"

    def apply_trend_filter(
        self,
        data: DataFrame,
        period: int = 200
    ) -> Tuple[Series, Series]:
        """
        趨勢過濾器

        Args:
            data: OHLCV DataFrame
            period: 均線週期

        Returns:
            tuple: (uptrend, downtrend) boolean Series
        """
        ma = data['close'].rolling(period).mean()
        uptrend = data['close'] > ma
        downtrend = data['close'] < ma
        return uptrend, downtrend


class MeanReversionStrategy(BaseStrategy):
    """均值回歸策略基礎類別"""

    strategy_type = "mean_reversion"

    def calculate_bollinger_bands(
        self,
        close: Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[Series, Series, Series]:
        """
        計算布林帶

        Args:
            close: 收盤價 Series
            period: 計算週期
            std_dev: 標準差倍數

        Returns:
            tuple: (upper_band, middle_band, lower_band)
        """
        middle = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return upper, middle, lower


class MomentumStrategy(BaseStrategy):
    """動量策略基礎類別"""

    strategy_type = "momentum"

    def calculate_rsi(
        self,
        close: Series,
        period: int = 14
    ) -> Series:
        """
        計算 RSI 指標

        Args:
            close: 收盤價 Series
            period: RSI 週期

        Returns:
            Series: RSI 值 (0-100)
        """
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(
        self,
        close: Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[Series, Series, Series]:
        """
        計算 MACD 指標

        Args:
            close: 收盤價 Series
            fast: 快線週期
            slow: 慢線週期
            signal: 訊號線週期

        Returns:
            tuple: (macd_line, signal_line, histogram)
        """
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
