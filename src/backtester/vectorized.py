"""
向量化計算工具

提供高效能向量化指標計算與資料處理。
支援 Polars 與 Pandas 後端。
"""

from typing import TYPE_CHECKING, Any, Optional, Union
import numpy as np
import pandas as pd

# Polars 是可選依賴
POLARS_AVAILABLE = False
if TYPE_CHECKING:
    import polars as pl
else:
    try:
        import polars as pl
        POLARS_AVAILABLE = True
    except ImportError:
        pl = None  # type: ignore[assignment]


# ============================================================================
# 技術指標向量化計算
# ============================================================================

def vectorized_sma(series: Union[Any, pd.Series], period: int) -> Union[Any, pd.Series]:
    """
    向量化 SMA (Simple Moving Average)

    Args:
        series: 價格序列 (Polars Series 或 Pandas Series)
        period: 週期

    Returns:
        SMA 序列
    """
    if POLARS_AVAILABLE and pl is not None and isinstance(series, pl.Series):
        # 使用 min_periods=1 避免 None 填充問題
        return series.rolling_mean(window_size=period, min_periods=1)
    else:
        return series.rolling(window=period, min_periods=1).mean()


def vectorized_ema(series: Union[Any, pd.Series], period: int) -> Union[Any, pd.Series]:
    """
    向量化 EMA (Exponential Moving Average)

    Args:
        series: 價格序列 (Polars Series 或 Pandas Series)
        period: 週期

    Returns:
        EMA 序列
    """
    if POLARS_AVAILABLE and pl is not None and isinstance(series, pl.Series):
        return series.ewm_mean(span=period)
    else:
        return series.ewm(span=period, adjust=False).mean()


def vectorized_rsi(series: Union[Any, pd.Series], period: int = 14) -> Union[Any, pd.Series]:
    """
    向量化 RSI (Relative Strength Index)

    Args:
        series: 價格序列 (Polars Series 或 Pandas Series)
        period: 週期（預設 14）

    Returns:
        RSI 序列 (0-100)
    """
    if POLARS_AVAILABLE and pl is not None and isinstance(series, pl.Series):
        # Polars 實作
        delta = series.diff()

        gain = pl.when(delta > 0).then(delta).otherwise(0)
        loss = pl.when(delta < 0).then(-delta).otherwise(0)

        avg_gain = gain.ewm_mean(span=period)
        avg_loss = loss.ewm_mean(span=period)

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi
    else:
        # Pandas 實作
        delta = series.diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi


def vectorized_bollinger_bands(
    series: Union[pl.Series, pd.Series],
    period: int = 20,
    std_dev: float = 2.0
) -> tuple:
    """
    向量化布林通道

    Args:
        series: 價格序列
        period: 週期
        std_dev: 標準差倍數

    Returns:
        (upper_band, middle_band, lower_band)
    """
    if isinstance(series, pl.Series):
        middle = series.rolling_mean(window_size=period)
        std = series.rolling_std(window_size=period)

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower
    else:
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower


def vectorized_atr(
    high: Union[pl.Series, pd.Series],
    low: Union[pl.Series, pd.Series],
    close: Union[pl.Series, pd.Series],
    period: int = 14
) -> Union[pl.Series, pd.Series]:
    """
    向量化 ATR (Average True Range)

    Args:
        high: 最高價序列
        low: 最低價序列
        close: 收盤價序列
        period: 週期

    Returns:
        ATR 序列
    """
    if isinstance(high, pl.Series):
        # Polars 實作
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        tr = pl.max_horizontal([tr1, tr2, tr3])
        atr = tr.ewm_mean(span=period)

        return atr
    else:
        # Pandas 實作
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()

        return atr


def vectorized_macd(
    series: Union[pl.Series, pd.Series],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> tuple:
    """
    向量化 MACD (Moving Average Convergence Divergence)

    Args:
        series: 價格序列
        fast: 快線週期
        slow: 慢線週期
        signal: 信號線週期

    Returns:
        (macd_line, signal_line, histogram)
    """
    if isinstance(series, pl.Series):
        fast_ema = series.ewm_mean(span=fast)
        slow_ema = series.ewm_mean(span=slow)

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm_mean(span=signal)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram
    else:
        fast_ema = series.ewm(span=fast, adjust=False).mean()
        slow_ema = series.ewm(span=slow, adjust=False).mean()

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram


# ============================================================================
# 向量化部位與損益計算
# ============================================================================

def vectorized_positions(
    signals: Union[pl.Series, pd.Series],
    position_mode: str = "one-way"
) -> Union[pl.Series, pd.Series]:
    """
    向量化部位計算

    Args:
        signals: 交易訊號序列 (1=做多, -1=做空, 0=平倉)
        position_mode: 持倉模式 ("one-way" or "hedge")

    Returns:
        部位序列
    """
    if isinstance(signals, pl.Series):
        # Polars 實作：使用 cumulative operations
        if position_mode == "one-way":
            # 單向持倉：訊號累積（但需處理反向訊號）
            positions = signals.fill_null(0).shift_and_fill(fill_value=0)
        else:
            # 對沖模式：可同時多空
            positions = signals.fill_null(0).cum_sum()

        return positions
    else:
        # Pandas 實作
        if position_mode == "one-way":
            positions = signals.fillna(0).shift(1, fill_value=0)
        else:
            positions = signals.fillna(0).cumsum()

        return positions


def vectorized_pnl(
    positions: Union[pl.Series, pd.Series],
    prices: Union[pl.Series, pd.Series],
    leverage: float = 1.0,
    fees: float = 0.0004
) -> Union[pl.Series, pd.Series]:
    """
    向量化損益計算

    Args:
        positions: 部位序列
        prices: 價格序列
        leverage: 槓桿倍數
        fees: 交易費率

    Returns:
        累積損益序列
    """
    if isinstance(positions, pl.Series):
        # Polars 實作
        price_changes = prices.diff()

        # 計算持倉損益
        position_pnl = positions * price_changes * leverage

        # 計算交易成本（部位變化時）
        position_changes = positions.diff().abs()
        trading_fees = position_changes * prices * fees

        # 總損益
        pnl = position_pnl - trading_fees
        cumulative_pnl = pnl.cum_sum()

        return cumulative_pnl
    else:
        # Pandas 實作
        price_changes = prices.diff()

        position_pnl = positions * price_changes * leverage

        position_changes = positions.diff().abs()
        trading_fees = position_changes * prices * fees

        pnl = position_pnl - trading_fees
        cumulative_pnl = pnl.cumsum()

        return cumulative_pnl


# ============================================================================
# 資料轉換工具
# ============================================================================

def pandas_to_polars(df: pd.DataFrame) -> pl.DataFrame:
    """
    Pandas DataFrame 轉 Polars DataFrame

    Args:
        df: Pandas DataFrame

    Returns:
        Polars DataFrame
    """
    if not POLARS_AVAILABLE:
        raise ImportError("Polars not installed")

    return pl.from_pandas(df)


def polars_to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    """
    Polars DataFrame 轉 Pandas DataFrame

    Args:
        df: Polars DataFrame

    Returns:
        Pandas DataFrame
    """
    return df.to_pandas()


def ensure_polars(df: Union[pd.DataFrame, pl.DataFrame]) -> pl.DataFrame:
    """
    確保資料為 Polars 格式

    Args:
        df: DataFrame (Pandas or Polars)

    Returns:
        Polars DataFrame
    """
    if isinstance(df, pd.DataFrame):
        return pandas_to_polars(df)
    return df


def ensure_pandas(df: Union[pd.DataFrame, pl.DataFrame]) -> pd.DataFrame:
    """
    確保資料為 Pandas 格式

    Args:
        df: DataFrame (Pandas or Polars)

    Returns:
        Pandas DataFrame
    """
    if isinstance(df, pl.DataFrame):
        return polars_to_pandas(df)
    return df
