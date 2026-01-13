"""
統一 DataFrame 操作層

支援 Pandas 和 Polars 的統一 API，讓策略程式碼無需關心底層格式。
"""

from __future__ import annotations

import pandas as pd

# 嘗試導入 Polars（可選依賴）
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None


class SeriesOps:
    """統一的 Series 操作層"""

    def __init__(self, series, is_polars: bool):
        self._series = series
        self._is_polars = is_polars

    def __repr__(self) -> str:
        """字串表示（方便 debug）"""
        dtype = 'polars' if self._is_polars else 'pandas'
        length = len(self._series) if hasattr(self._series, '__len__') else '?'
        return f"SeriesOps(type={dtype}, len={length})"

    def rolling_mean(self, window: int, min_periods: int = 1) -> 'SeriesOps':
        """滾動平均"""
        if self._is_polars:
            result = self._series.rolling_mean(window_size=window, min_periods=min_periods)
        else:
            result = self._series.rolling(window=window, min_periods=min_periods).mean()
        return SeriesOps(result, self._is_polars)

    def rolling_std(self, window: int, min_periods: int = 1) -> 'SeriesOps':
        """滾動標準差"""
        if self._is_polars:
            result = self._series.rolling_std(window_size=window, min_periods=min_periods)
        else:
            result = self._series.rolling(window=window, min_periods=min_periods).std()
        return SeriesOps(result, self._is_polars)

    def rolling_max(self, window: int, min_periods: int = 1) -> 'SeriesOps':
        """滾動最大值"""
        if self._is_polars:
            result = self._series.rolling_max(window_size=window, min_periods=min_periods)
        else:
            result = self._series.rolling(window=window, min_periods=min_periods).max()
        return SeriesOps(result, self._is_polars)

    def rolling_min(self, window: int, min_periods: int = 1) -> 'SeriesOps':
        """滾動最小值"""
        if self._is_polars:
            result = self._series.rolling_min(window_size=window, min_periods=min_periods)
        else:
            result = self._series.rolling(window=window, min_periods=min_periods).min()
        return SeriesOps(result, self._is_polars)

    def ewm_mean(self, span: int) -> 'SeriesOps':
        """指數加權移動平均"""
        if self._is_polars:
            result = self._series.ewm_mean(span=span)
        else:
            result = self._series.ewm(span=span, adjust=False).mean()
        return SeriesOps(result, self._is_polars)

    def shift(self, n: int = 1) -> 'SeriesOps':
        """位移"""
        result = self._series.shift(n)
        return SeriesOps(result, self._is_polars)

    def diff(self) -> 'SeriesOps':
        """差分"""
        result = self._series.diff()
        return SeriesOps(result, self._is_polars)

    def abs(self) -> 'SeriesOps':
        """絕對值"""
        result = self._series.abs()
        return SeriesOps(result, self._is_polars)

    def fillna(self, value) -> 'SeriesOps':
        """填充 NA 值"""
        if self._is_polars:
            result = self._series.fill_null(value)
        else:
            result = self._series.fillna(value)
        return SeriesOps(result, self._is_polars)

    def where(self, condition, other) -> SeriesOps:
        """
        條件選擇（統一為 Pandas 語意）

        條件為 True 時保留原值，條件為 False 時替換為 other。

        Args:
            condition: 布林條件（SeriesOps 或原生 Series）
            other: 條件為 False 時的替換值

        Returns:
            SeriesOps: 條件選擇結果

        Note:
            此方法統一為 Pandas 語意。Polars 的原生 where() 語意相反，
            本方法內部已處理此差異，使用者無需擔心。
        """
        cond = condition._series if isinstance(condition, SeriesOps) else condition
        other_val = other._series if isinstance(other, SeriesOps) else other

        if self._is_polars:
            # Polars 需要用 zip_with 來模擬 Pandas where 語意
            # zip_with(mask, other): mask 為 True 時取 self，否則取 other
            result = self._series.zip_with(cond, other_val)
        else:
            result = self._series.where(cond, other_val)
        return SeriesOps(result, self._is_polars)

    # 比較運算子
    def __gt__(self, other) -> 'SeriesOps':
        other_val = other._series if isinstance(other, SeriesOps) else other
        return SeriesOps(self._series > other_val, self._is_polars)

    def __lt__(self, other) -> 'SeriesOps':
        other_val = other._series if isinstance(other, SeriesOps) else other
        return SeriesOps(self._series < other_val, self._is_polars)

    def __ge__(self, other) -> 'SeriesOps':
        other_val = other._series if isinstance(other, SeriesOps) else other
        return SeriesOps(self._series >= other_val, self._is_polars)

    def __le__(self, other) -> 'SeriesOps':
        other_val = other._series if isinstance(other, SeriesOps) else other
        return SeriesOps(self._series <= other_val, self._is_polars)

    def eq(self, other) -> 'SeriesOps':
        """相等比較（使用 eq() 而非 __eq__ 以避免與 object.__eq__ 衝突）"""
        other_val = other._series if isinstance(other, SeriesOps) else other
        return SeriesOps(self._series == other_val, self._is_polars)

    def ne(self, other) -> 'SeriesOps':
        """不相等比較"""
        other_val = other._series if isinstance(other, SeriesOps) else other
        return SeriesOps(self._series != other_val, self._is_polars)

    # 算術運算子
    def __add__(self, other) -> 'SeriesOps':
        other_val = other._series if isinstance(other, SeriesOps) else other
        return SeriesOps(self._series + other_val, self._is_polars)

    def __sub__(self, other) -> 'SeriesOps':
        other_val = other._series if isinstance(other, SeriesOps) else other
        return SeriesOps(self._series - other_val, self._is_polars)

    def __mul__(self, other) -> 'SeriesOps':
        other_val = other._series if isinstance(other, SeriesOps) else other
        return SeriesOps(self._series * other_val, self._is_polars)

    def __truediv__(self, other) -> 'SeriesOps':
        other_val = other._series if isinstance(other, SeriesOps) else other
        return SeriesOps(self._series / other_val, self._is_polars)

    def __neg__(self) -> 'SeriesOps':
        return SeriesOps(-self._series, self._is_polars)

    # 邏輯運算子
    def __and__(self, other) -> 'SeriesOps':
        other_val = other._series if isinstance(other, SeriesOps) else other
        return SeriesOps(self._series & other_val, self._is_polars)

    def __or__(self, other) -> 'SeriesOps':
        other_val = other._series if isinstance(other, SeriesOps) else other
        return SeriesOps(self._series | other_val, self._is_polars)

    def __invert__(self) -> 'SeriesOps':
        return SeriesOps(~self._series, self._is_polars)

    @property
    def raw(self):
        """取得原始 Series（用於 VectorBT 等需要原生格式的場景）"""
        return self._series

    def to_pandas(self) -> pd.Series:
        """轉為 Pandas Series"""
        if self._is_polars:
            return self._series.to_pandas()
        return self._series

    def to_numpy(self):
        """轉為 numpy array"""
        if self._is_polars:
            return self._series.to_numpy()
        return self._series.to_numpy()


class DataFrameOps:
    """統一的 DataFrame 操作層"""

    def __init__(self, df):
        """
        初始化 DataFrameOps

        Args:
            df: Pandas DataFrame 或 Polars DataFrame
        """
        self._df = df
        # 檢查是否為 Polars DataFrame（避免 pl=None 時的類型錯誤）
        self._is_polars = False
        if POLARS_AVAILABLE and pl is not None:
            # 使用模組名稱檢查避免直接存取可能為 None 的 pl.DataFrame
            self._is_polars = type(df).__module__.startswith('polars')

    @property
    def is_polars(self) -> bool:
        """是否為 Polars 格式"""
        return self._is_polars

    @property
    def columns(self):
        """取得欄位名稱"""
        return self._df.columns

    def __len__(self) -> int:
        """取得資料長度"""
        return len(self._df)

    def col(self, name: str) -> SeriesOps:
        """取得欄位的 SeriesOps 包裝"""
        return SeriesOps(self._df[name], self._is_polars)

    def __getitem__(self, key: str) -> SeriesOps:
        """支援 df['column'] 語法"""
        return self.col(key)

    @property
    def raw(self):
        """取得原始 DataFrame"""
        return self._df

    def to_pandas(self) -> pd.DataFrame:
        """轉為 Pandas DataFrame"""
        if self._is_polars:
            return self._df.to_pandas()
        return self._df
