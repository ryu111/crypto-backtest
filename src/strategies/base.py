"""
策略基礎類別

定義所有策略必須實作的介面和共用功能。
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any
from pandas import Series, DataFrame
from .utils.dataframe_ops import DataFrameOps, SeriesOps

# Regime Detection（延遲導入以避免循環依賴）
# 這些類型在 apply_regime_filter 中動態導入
REGIME_AVAILABLE = False
MarketStateAnalyzer: Any = None
StrategyConfig: Any = None

def _load_regime_module() -> bool:
    """延遲載入 regime 模組"""
    global REGIME_AVAILABLE, MarketStateAnalyzer, StrategyConfig
    if not REGIME_AVAILABLE:
        try:
            from ..regime import MarketStateAnalyzer as _MSA, StrategyConfig as _SC
            MarketStateAnalyzer = _MSA
            StrategyConfig = _SC
            REGIME_AVAILABLE = True
        except ImportError:
            pass
    return REGIME_AVAILABLE


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

    def _wrap_data(self, data) -> DataFrameOps:
        """
        包裝 DataFrame 為 DataFrameOps

        Args:
            data: Pandas 或 Polars DataFrame

        Returns:
            DataFrameOps: 統一操作層
        """
        return DataFrameOps(data)

    def _to_pandas_series(self, series_ops: SeriesOps) -> pd.Series:
        """
        將 SeriesOps 轉為 Pandas Series（供 VectorBT 使用）

        Args:
            series_ops: SeriesOps 實例

        Returns:
            pd.Series: Pandas Series
        """
        return series_ops.to_pandas()

    def _create_signal_series(self, data, value: bool = False) -> pd.Series:
        """
        建立訊號 Series（統一使用 Pandas 輸出）

        Args:
            data: DataFrame（Pandas 或 Polars）
            value: 初始值（預設 False）

        Returns:
            pd.Series: 訊號 Series
        """
        if hasattr(data, 'to_pandas'):  # Polars DataFrame
            index = pd.RangeIndex(len(data))
        else:  # Pandas DataFrame
            index = data.index
        return pd.Series(value, index=index)

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
        # 如果啟用 Regime Filter，則應用過濾
        if self.params.get('use_regime_filter', False):
            regime_config = self.params.get('regime_config', None)
            return self.apply_regime_filter(
                data, long_entry, long_exit, short_entry, short_exit, regime_config
            )

        # 預設不過濾
        return long_entry, long_exit, short_entry, short_exit

    def apply_regime_filter(
        self,
        data: DataFrame,
        long_entry: Series,
        long_exit: Series,
        short_entry: Series,
        short_exit: Series,
        regime_config: Optional[Dict] = None
    ) -> Tuple[Series, Series, Series, Series]:
        """
        根據市場狀態（Regime）過濾交易訊號

        只在策略適用的市場環境中允許進場。

        Args:
            data: OHLCV DataFrame
            long_entry: 多單進場訊號
            long_exit: 多單出場訊號
            short_entry: 空單進場訊號
            short_exit: 空單出場訊號
            regime_config: Regime 配置
                {
                    'direction_range': (min, max),     # 方向性範圍 -10 到 10
                    'volatility_range': (min, max),    # 波動度範圍 0 到 10
                    'analyzer_params': {               # MarketStateAnalyzer 參數（可選）
                        'direction_threshold_strong': 5.0,
                        'direction_threshold_weak': 2.0,
                        'volatility_threshold': 5.0,
                        'direction_method': 'composite'
                    }
                }

        Returns:
            tuple: (long_entry, long_exit, short_entry, short_exit) 過濾後的訊號

        Raises:
            ImportError: 如果 regime 模組未安裝

        Example:
            >>> # 只在強趨勢高波動環境進場
            >>> strategy.params['use_regime_filter'] = True
            >>> strategy.params['regime_config'] = {
            ...     'direction_range': (3, 10),
            ...     'volatility_range': (5, 10)
            ... }
        """
        # 延遲載入 regime 模組
        if not _load_regime_module():
            raise ImportError(
                "Regime Detection 模組未安裝。請確認 src/regime/ 可用。"
            )

        # 預設配置：適用於趨勢策略
        if regime_config is None:
            regime_config = {
                'direction_range': (-10, 10),  # 接受所有方向
                'volatility_range': (0, 10),   # 接受所有波動
            }

        # 建立 MarketStateAnalyzer
        analyzer_params = regime_config.get('analyzer_params', {})
        analyzer = MarketStateAnalyzer(**analyzer_params)

        # 計算當前市場狀態
        market_state = analyzer.calculate_state(data)

        # 建立 StrategyConfig
        strategy_config = StrategyConfig(
            name=self.name,
            direction_range=regime_config['direction_range'],
            volatility_range=regime_config['volatility_range'],
            weight=1.0
        )

        # 判斷策略是否適用當前市場狀態
        is_active = strategy_config.is_active(
            market_state.direction,
            market_state.volatility
        )

        # 如果不適用，清除所有進場訊號（保留出場訊號）
        if not is_active:
            long_entry = self._create_signal_series(data, value=False)
            short_entry = self._create_signal_series(data, value=False)

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
        趨勢過濾器（使用 DataFrameOps）

        Args:
            data: OHLCV DataFrame
            period: 均線週期

        Returns:
            tuple: (uptrend, downtrend) boolean Series
        """
        ops = self._wrap_data(data)
        ma = ops['close'].rolling_mean(period)

        # 比較並轉為 Pandas
        uptrend = (ops['close'] > ma).to_pandas()
        downtrend = (ops['close'] < ma).to_pandas()

        return uptrend, downtrend


class MeanReversionStrategy(BaseStrategy):
    """均值回歸策略基礎類別"""

    strategy_type = "mean_reversion"

    def calculate_bollinger_bands(
        self,
        close,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[Series, Series, Series]:
        """
        計算布林帶（使用 DataFrameOps）

        Args:
            close: 收盤價（Series 或 SeriesOps）
            period: 計算週期
            std_dev: 標準差倍數

        Returns:
            tuple: (upper_band, middle_band, lower_band)
        """
        from .utils.dataframe_ops import SeriesOps, POLARS_AVAILABLE

        if isinstance(close, SeriesOps):
            series = close
        else:
            is_polars = POLARS_AVAILABLE and type(close).__module__.startswith('polars')
            series = SeriesOps(close, is_polars)

        middle = series.rolling_mean(period)
        std = series.rolling_std(period)
        upper = middle + std * std_dev
        lower = middle - std * std_dev

        return upper.to_pandas(), middle.to_pandas(), lower.to_pandas()


class MomentumStrategy(BaseStrategy):
    """動量策略基礎類別"""

    strategy_type = "momentum"

    def calculate_rsi(
        self,
        close,
        period: int = 14
    ) -> Series:
        """
        計算 RSI 指標（使用 DataFrameOps）

        Args:
            close: 收盤價（Series 或 SeriesOps）
            period: RSI 週期

        Returns:
            Series: RSI 值 (0-100)
        """
        from .utils.dataframe_ops import SeriesOps, POLARS_AVAILABLE

        # 將輸入轉為 SeriesOps
        if isinstance(close, SeriesOps):
            series = close
            is_polars = close._is_polars
        else:
            is_polars = POLARS_AVAILABLE and type(close).__module__.startswith('polars')
            series = SeriesOps(close, is_polars)

        delta = series.diff()

        # 計算 gain 和 loss
        # 使用 SeriesOps.where() 統一處理（已經處理好 Polars/Pandas 差異）
        zero_series = SeriesOps(series.raw * 0, is_polars)  # 全 0 的 SeriesOps

        # gain: 當 delta > 0 時保留，否則為 0
        gain = delta.where(delta > zero_series, zero_series)
        # loss: 當 delta < 0 時取負值，否則為 0
        loss = (-delta).where(delta < zero_series, zero_series)

        avg_gain = gain.rolling_mean(period)
        avg_loss = loss.rolling_mean(period)

        rs = avg_gain / avg_loss
        # 需要先轉換為數值計算（SeriesOps 不支援與 int 的直接運算）
        one = SeriesOps(series.raw * 0 + 1, is_polars)  # 建立全 1 的 SeriesOps
        hundred = SeriesOps(series.raw * 0 + 100, is_polars)  # 建立全 100 的 SeriesOps
        rsi = hundred - (hundred / (one + rs))

        return rsi.to_pandas()

    def calculate_macd(
        self,
        close,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[Series, Series, Series]:
        """
        計算 MACD 指標（使用 DataFrameOps）

        Args:
            close: 收盤價（Series 或 SeriesOps）
            fast: 快線週期
            slow: 慢線週期
            signal: 訊號線週期

        Returns:
            tuple: (macd_line, signal_line, histogram)
        """
        from .utils.dataframe_ops import SeriesOps, POLARS_AVAILABLE

        if isinstance(close, SeriesOps):
            series = close
        else:
            is_polars = POLARS_AVAILABLE and type(close).__module__.startswith('polars')
            series = SeriesOps(close, is_polars)

        ema_fast = series.ewm_mean(fast)
        ema_slow = series.ewm_mean(slow)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm_mean(signal)
        histogram = macd_line - signal_line

        return macd_line.to_pandas(), signal_line.to_pandas(), histogram.to_pandas()


class StatisticalArbStrategy(BaseStrategy):
    """
    統計套利策略基礎類別

    用於配對交易、基差套利等需要雙標的數據的策略。

    特點：
    - 支援雙標的數據輸入（data_primary, data_secondary）
    - 提供價差/比率計算方法
    - 提供 Z-Score 標準化方法

    子類別實作：
    - ETH/BTC 配對交易
    - 永續/現貨基差套利
    """

    strategy_type = "statistical_arbitrage"

    def calculate_spread(
        self,
        price1: Series,
        price2: Series,
        method: str = 'ratio'
    ) -> Series:
        """
        計算兩標的價差

        Args:
            price1: 第一標的價格 Series
            price2: 第二標的價格 Series
            method: 計算方法
                - 'ratio': price1 / price2（比率）
                - 'diff': price1 - price2（價差）
                - 'log_ratio': log(price1) - log(price2)（對數比率）

        Returns:
            Series: 價差/比率 Series
        """
        if method == 'ratio':
            return price1 / price2
        elif method == 'diff':
            return price1 - price2
        elif method == 'log_ratio':
            return np.log(price1) - np.log(price2)
        else:
            raise ValueError(f"Unknown method: {method}")

    def calculate_zscore(
        self,
        spread: Series,
        period: int = 20
    ) -> Series:
        """
        計算 Z-Score（標準化偏離度）

        Args:
            spread: 價差 Series
            period: 滾動視窗週期

        Returns:
            Series: Z-Score Series
        """
        mean = spread.rolling(period).mean()
        std = spread.rolling(period).std()
        zscore = (spread - mean) / std
        return zscore

    def calculate_half_life(
        self,
        spread: Series
    ) -> float:
        """
        計算均值回歸半衰期（用於判斷配對有效性）

        Args:
            spread: 價差 Series

        Returns:
            float: 半衰期（期數）
        """
        spread_clean = spread.dropna()
        if len(spread_clean) < 2:
            return float('inf')

        # 計算 AR(1) 係數
        spread_lag = spread_clean.shift(1).dropna()
        spread_diff = spread_clean.diff().dropna()

        # 對齊長度
        spread_lag = spread_lag.iloc[1:]

        if len(spread_lag) < 2:
            return float('inf')

        # 線性回歸 delta_spread = alpha + beta * spread_lag
        correlation = spread_diff.corr(spread_lag)
        std_diff = spread_diff.std()
        std_lag = spread_lag.std()

        if std_lag == 0:
            return float('inf')

        beta = correlation * (std_diff / std_lag)

        if beta >= 0:
            return float('inf')

        # 半衰期 = -ln(2) / ln(1 + beta)
        half_life = -np.log(2) / np.log(1 + beta)
        return half_life

    def generate_signals_dual(
        self,
        data_primary: DataFrame,
        data_secondary: DataFrame
    ) -> Tuple[Series, Series, Series, Series]:
        """
        產生雙標的交易訊號（子類別必須實作）

        統計套利策略通常需要兩個標的的數據。
        此方法是 generate_signals 的擴展版本。

        Args:
            data_primary: 主標的 OHLCV DataFrame（如 ETH）
            data_secondary: 次標的 OHLCV DataFrame（如 BTC）

        Returns:
            tuple: (long_entry, long_exit, short_entry, short_exit)
                   long = 做多主標的，做空次標的
                   short = 做空主標的，做多次標的
        """
        raise NotImplementedError("Subclass must implement generate_signals_dual()")


class FundingRateStrategy(BaseStrategy):
    """
    資金費率策略基礎類別

    用於基於永續合約資金費率的策略。

    特點：
    - 支援資金費率數據輸入
    - 提供費率成本計算方法
    - 支援結算時間判斷

    子類別實作：
    - Delta Neutral 資金費率套利
    - 結算時段交易
    """

    strategy_type = "funding_rate"

    # 主流交易所結算時間（UTC）
    SETTLEMENT_HOURS = [0, 8, 16]

    def calculate_funding_cost(
        self,
        rate: float,
        position_value: float,
        direction: int
    ) -> float:
        """
        計算資金費率成本

        Args:
            rate: 資金費率（如 0.0001 = 0.01%）
            position_value: 部位價值（USDT）
            direction: 部位方向（1=多，-1=空）

        Returns:
            float: 費率成本（正=支付，負=收取）
        """
        # 多單：費率為正時支付，為負時收取
        # 空單：費率為正時收取，為負時支付
        return rate * position_value * direction

    def is_settlement_hour(
        self,
        timestamp: pd.Timestamp,
        hours_before: int = 1
    ) -> bool:
        """
        判斷是否接近結算時間

        Args:
            timestamp: 時間戳
            hours_before: 結算前幾小時判定為「接近」

        Returns:
            bool: 是否接近結算時間
        """
        hour = timestamp.hour
        for settlement_hour in self.SETTLEMENT_HOURS:
            # 檢查是否在結算前 hours_before 小時內
            hours_until = (settlement_hour - hour) % 24
            if hours_until <= hours_before:
                return True
        return False

    def calculate_annualized_rate(
        self,
        funding_rate: float,
        settlements_per_day: int = 3
    ) -> float:
        """
        計算年化資金費率

        Args:
            funding_rate: 單次結算費率
            settlements_per_day: 每日結算次數（預設 3）

        Returns:
            float: 年化費率
        """
        daily_rate = funding_rate * settlements_per_day
        annual_rate = daily_rate * 365
        return annual_rate

    def calculate_expected_return(
        self,
        funding_rates: Series,
        holding_periods: int = 1
    ) -> float:
        """
        計算預期收益（基於歷史費率）

        Args:
            funding_rates: 歷史資金費率 Series
            holding_periods: 預期持有結算次數

        Returns:
            float: 預期收益率
        """
        if len(funding_rates) < holding_periods:
            return 0.0

        # 使用最近 N 期費率的平均值估算
        recent_rates = funding_rates.tail(holding_periods)
        expected_return = recent_rates.mean() * holding_periods
        return expected_return

    def generate_signals_with_funding(
        self,
        data: DataFrame,
        funding_rates: Series
    ) -> Tuple[Series, Series, Series, Series]:
        """
        產生考慮資金費率的交易訊號（子類別必須實作）

        Args:
            data: OHLCV DataFrame
            funding_rates: 資金費率 Series（與 data 時間對齊）

        Returns:
            tuple: (long_entry, long_exit, short_entry, short_exit)
        """
        raise NotImplementedError("Subclass must implement generate_signals_with_funding()")
