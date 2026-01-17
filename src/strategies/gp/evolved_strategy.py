"""
演化策略基類

所有 GP 演化生成的策略都繼承此類別。
"""

from typing import Dict, Tuple, Callable, Optional, Any
import pandas as pd
import numpy as np
from pandas import Series, DataFrame

from src.strategies.base import BaseStrategy


class EvolvedStrategy(BaseStrategy):
    """
    GP 演化策略基類

    繼承 BaseStrategy，提供：
    - 從表達式樹生成的訊號函數
    - 演化元資料（適應度、代數等）
    - 動態編譯支援

    所有由 GP 演化產生的策略都繼承此類別。
    """

    strategy_type = "evolved"

    # 策略配置
    allow_short = False  # 目前 GP 策略只支援多單

    # 演化元資料（子類別覆寫）
    expression: str = ""  # 原始表達式字串
    fitness_score: float = 0.0  # 適應度分數
    generation: int = 0  # 演化代數
    evolved_at: str = ""  # ISO 時間戳

    def __init__(self, signal_func: Optional[Callable] = None, **kwargs):
        """
        初始化演化策略

        Args:
            signal_func: 編譯後的訊號函數（可選，動態載入）
            **kwargs: 其他參數（傳遞給 BaseStrategy）
        """
        super().__init__(**kwargs)
        self._signal_func = signal_func

    def calculate_indicators(self, data: DataFrame) -> Dict[str, Series]:
        """
        計算指標（由 primitives 自動處理）

        GP 策略不需要預先計算指標，
        指標計算在 generate_signals 中隱式完成。

        Args:
            data: OHLCV DataFrame

        Returns:
            dict: 空字典（GP 策略不使用外部指標）
        """
        return {}

    def generate_signals(
        self,
        data: DataFrame
    ) -> Tuple[Series, Series, Series, Series]:
        """
        產生交易訊號

        使用編譯後的 GP 表達式生成訊號。

        Args:
            data: OHLCV DataFrame，必須包含 close, high, low 欄位

        Returns:
            tuple: (long_entry, long_exit, short_entry, short_exit)

        Raises:
            RuntimeError: 如果訊號函數未初始化
            KeyError: 如果 data 缺少必要欄位
        """
        if self._signal_func is None:
            raise RuntimeError(
                f"訊號函數未初始化。請確認 {self.__class__.__name__} "
                "正確實作了 _build_signal_func() 方法。"
            )

        # 驗證必要欄位
        required_columns = ['close', 'high', 'low']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise KeyError(
                f"Data 缺少必要欄位: {missing_columns}。"
                f"可用欄位: {list(data.columns)}"
            )

        # 提取價格序列（numpy 陣列，供 primitives 使用）
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values

        # 呼叫 GP 表達式（返回布林陣列）
        try:
            signal = self._signal_func(close, high, low)
        except Exception as e:
            raise RuntimeError(
                f"執行 GP 表達式失敗: {e}\n"
                f"表達式: {self.expression}"
            ) from e

        # 確保信號是陣列，並處理 NaN
        # 先保持 float 型別以正確處理 NaN
        signal = np.asarray(signal, dtype=float)

        # 處理 NaN（可能由指標計算產生）
        # 將 NaN 轉換為 0.0
        signal = np.nan_to_num(signal, nan=0.0)

        # 最後轉為布林型別
        signal = signal.astype(bool)

        # 轉換為四個訊號 Series
        long_entry = pd.Series(signal, index=data.index)
        long_exit = pd.Series(~signal, index=data.index)  # 相反訊號

        # 根據 allow_short 決定是否產生空單訊號
        if self.allow_short:
            short_entry = pd.Series(~signal, index=data.index)
            short_exit = pd.Series(signal, index=data.index)
        else:
            short_entry = pd.Series(False, index=data.index)
            short_exit = pd.Series(False, index=data.index)

        return long_entry, long_exit, short_entry, short_exit

    def get_info(self) -> Dict[str, Any]:
        """
        取得策略詳細資訊（覆寫 BaseStrategy）

        Returns:
            dict: 策略資訊，包含演化元資料
        """
        info = super().get_info()

        # 添加演化元資料
        info.update({
            'expression': self.expression,
            'fitness_score': self.fitness_score,
            'generation': self.generation,
            'evolved_at': self.evolved_at,
        })

        return info

    def validate_params(self) -> bool:
        """
        驗證參數有效性（覆寫 BaseStrategy）

        GP 策略不需要參數驗證（參數已編碼在表達式中）。

        Returns:
            bool: 總是返回 True
        """
        return True

    def __repr__(self) -> str:
        """字串表示"""
        return (
            f"{self.name}(fitness={self.fitness_score:.4f}, "
            f"generation={self.generation})"
        )

    def __str__(self) -> str:
        """友善字串表示"""
        return f"{self.name} (GP Evolved, Fitness: {self.fitness_score:.4f})"


# 公開 API
__all__ = [
    'EvolvedStrategy',
]
