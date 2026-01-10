"""
策略基礎模板

使用方式：
1. 繼承 BaseStrategy
2. 覆寫 generate_signals 方法
3. 設定 params 參數
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """策略基礎類別"""

    # 預設參數（子類覆寫）
    params = {}

    # 策略元資料
    name = "base_strategy"
    version = "1.0"
    description = "Base strategy template"

    def __init__(self, **kwargs):
        """初始化策略，可覆寫預設參數"""
        self.params = {**self.params, **kwargs}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> tuple:
        """
        產生交易訊號

        Args:
            data: OHLCV DataFrame，包含 open, high, low, close, volume

        Returns:
            tuple: (long_entry, long_exit, short_entry, short_exit)
                   每個都是 boolean Series
        """
        pass

    def calculate_indicators(self, data: pd.DataFrame) -> dict:
        """
        計算策略所需指標（可選覆寫）

        Args:
            data: OHLCV DataFrame

        Returns:
            dict: 指標名稱 -> Series
        """
        return {}

    def position_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss_price: float,
        risk_per_trade: float = 0.02
    ) -> float:
        """
        計算部位大小

        Args:
            capital: 總資金
            entry_price: 入場價格
            stop_loss_price: 止損價格
            risk_per_trade: 單筆風險比例

        Returns:
            float: 部位大小
        """
        risk_amount = capital * risk_per_trade
        stop_distance = abs(entry_price - stop_loss_price)

        if stop_distance == 0:
            return 0

        size = risk_amount / stop_distance
        return size

    def validate_params(self) -> bool:
        """驗證參數有效性"""
        return True

    def get_info(self) -> dict:
        """取得策略資訊"""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'params': self.params
        }


class TrendStrategy(BaseStrategy):
    """趨勢策略模板"""

    params = {
        'fast_period': 10,
        'slow_period': 30,
    }

    name = "trend_template"
    description = "Trend following strategy template"

    def generate_signals(self, data: pd.DataFrame) -> tuple:
        close = data['close']

        # 計算均線
        fast_ma = close.rolling(self.params['fast_period']).mean()
        slow_ma = close.rolling(self.params['slow_period']).mean()

        # 交叉訊號
        long_entry = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        long_exit = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        short_entry = long_exit.copy()
        short_exit = long_entry.copy()

        return long_entry, long_exit, short_entry, short_exit


class MeanReversionStrategy(BaseStrategy):
    """均值回歸策略模板"""

    params = {
        'period': 20,
        'std_dev': 2.0,
    }

    name = "mean_reversion_template"
    description = "Mean reversion strategy template"

    def generate_signals(self, data: pd.DataFrame) -> tuple:
        close = data['close']

        # 布林帶
        ma = close.rolling(self.params['period']).mean()
        std = close.rolling(self.params['period']).std()
        upper = ma + self.params['std_dev'] * std
        lower = ma - self.params['std_dev'] * std

        # 觸及邊界反轉
        long_entry = close < lower
        long_exit = close > ma
        short_entry = close > upper
        short_exit = close < ma

        return long_entry, long_exit, short_entry, short_exit


class MomentumStrategy(BaseStrategy):
    """動量策略模板"""

    params = {
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
    }

    name = "momentum_template"
    description = "Momentum strategy template"

    def calculate_rsi(self, close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def generate_signals(self, data: pd.DataFrame) -> tuple:
        close = data['close']

        # RSI
        rsi = self.calculate_rsi(close, self.params['rsi_period'])

        # 超買超賣
        long_entry = rsi < self.params['rsi_oversold']
        long_exit = rsi > 50
        short_entry = rsi > self.params['rsi_overbought']
        short_exit = rsi < 50

        return long_entry, long_exit, short_entry, short_exit


# 使用範例
if __name__ == "__main__":
    # 建立策略
    strategy = TrendStrategy(fast_period=12, slow_period=26)

    # 查看資訊
    print(strategy.get_info())

    # 模擬資料
    data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })

    # 產生訊號
    long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(data)
    print(f"Long entries: {long_entry.sum()}")
    print(f"Short entries: {short_entry.sum()}")
