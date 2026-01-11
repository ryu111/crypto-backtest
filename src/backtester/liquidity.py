"""
流動性影響模組

模擬大單對市場的價格衝擊，提供多種流動性模型。

流動性影響來源：
1. 訂單簿深度不足
2. 市場參與者反應
3. 訂單大小相對於成交量的比例
4. 市場波動率

參考：
- Almgren-Chriss 市場衝擊模型
- Square-root law of market impact
"""

from dataclasses import dataclass
from typing import Callable, Literal, Optional
import pandas as pd
import numpy as np
from enum import Enum


class LiquidityModel(Enum):
    """流動性模型類型"""
    LINEAR = "linear"              # 線性模型：衝擊 ∝ Q
    SQUARE_ROOT = "square_root"    # 平方根模型：衝擊 ∝ √Q（學術標準）
    LOGARITHMIC = "logarithmic"    # 對數模型：衝擊 ∝ log(Q)
    CUSTOM = "custom"              # 自定義函數


class LiquidityLevel(Enum):
    """流動性等級"""
    HIGH = "high"          # 高流動性（衝擊小）
    MEDIUM = "medium"      # 中等流動性
    LOW = "low"            # 低流動性（衝擊大）
    VERY_LOW = "very_low"  # 極低流動性（高風險）


@dataclass
class LiquidityConfig:
    """流動性配置"""

    # 基本設定
    model: LiquidityModel = LiquidityModel.SQUARE_ROOT
    impact_coefficient: float = 0.3  # 衝擊係數 η (通常 0.1-0.5)

    # ADV 計算參數
    adv_window: int = 30           # 平均日成交量窗口（天）
    adv_percentile: float = 0.5    # 使用中位數而非平均（更穩定）

    # 波動率參數
    volatility_window: int = 20    # 波動率計算窗口
    use_volatility: bool = True    # 是否考慮波動率

    # 限制
    max_impact: float = 0.05       # 最大價格衝擊 5%
    min_impact: float = 0.0        # 最小價格衝擊 0%

    # 流動性評級門檻（訂單/ADV 比例）
    high_liquidity_threshold: float = 0.001    # 0.1% 以下
    medium_liquidity_threshold: float = 0.01   # 1% 以下
    low_liquidity_threshold: float = 0.05      # 5% 以下

    def __post_init__(self):
        """驗證配置"""
        if self.impact_coefficient < 0:
            raise ValueError("衝擊係數不能為負數")

        if self.max_impact < self.min_impact:
            raise ValueError("最大衝擊不能小於最小衝擊")

        if self.max_impact > 0.1:
            raise ValueError("最大價格衝擊不應超過 10%（建議值）")

        if self.adv_window < 1:
            raise ValueError("ADV 窗口不能小於 1")


class LiquidityCalculator:
    """
    流動性衝擊計算器

    實作多種流動性模型，估算大單對價格的影響。

    使用範例：
        config = LiquidityConfig(
            model=LiquidityModel.SQUARE_ROOT,
            impact_coefficient=0.3,
            adv_window=30
        )

        calculator = LiquidityCalculator(config)
        impact = calculator.calculate_impact(
            data=price_data,
            order_size_usd=50000,
            index=current_index
        )
    """

    def __init__(self, config: Optional[LiquidityConfig] = None):
        """
        初始化流動性計算器

        Args:
            config: 流動性配置，None 則使用預設值
        """
        self.config = config or LiquidityConfig()
        self._custom_func: Optional[Callable[[float, float, float], float]] = None

        # 快取計算結果
        self._adv_cache: Optional[pd.Series] = None
        self._volatility_cache: Optional[pd.Series] = None

    def set_custom_function(self, func: Callable[[float, float, float], float]) -> None:
        """
        設定自定義流動性計算函數

        Args:
            func: 函數簽名 (order_size, adv, volatility) -> float

        範例：
            def my_impact(order_size, adv, volatility):
                return 0.001 * (order_size / adv)  # 線性模型

            calculator.set_custom_function(my_impact)
        """
        self._custom_func = func
        self.config.model = LiquidityModel.CUSTOM

    def calculate_impact(
        self,
        data: pd.DataFrame,
        order_size_usd: float,
        direction: Literal[1, -1] = 1,
        index: Optional[int] = None
    ) -> float:
        """
        計算價格衝擊

        Args:
            data: OHLCV DataFrame（需包含 'close', 'volume'）
            order_size_usd: 訂單金額（USDT）
            direction: 交易方向（1=做多，-1=做空）
            index: 資料索引位置（None 則使用最後一筆）

        Returns:
            impact: 價格衝擊百分比（如 0.002 = 0.2%）

        範例：
            >>> impact = calculator.calculate_impact(
            ...     data=df,
            ...     order_size_usd=50000,
            ...     direction=1,
            ...     index=100
            ... )
            >>> print(f"價格衝擊: {impact:.4%}")
        """
        if index is None:
            index = len(data) - 1

        # 計算 ADV（Average Daily Volume）
        adv = self._calculate_adv(data, index)

        if adv == 0:
            # 無成交量資料，返回最大衝擊（保守估計）
            return self.config.max_impact

        # 計算波動率（可選）
        volatility = 1.0
        if self.config.use_volatility:
            volatility = self._calculate_volatility(data, index)

        # 計算訂單比例（Q / ADV）
        volume_ratio = order_size_usd / adv

        # 根據模型計算市場衝擊
        if self.config.model == LiquidityModel.LINEAR:
            impact = self._calculate_linear(volume_ratio, volatility)

        elif self.config.model == LiquidityModel.SQUARE_ROOT:
            impact = self._calculate_square_root(volume_ratio, volatility)

        elif self.config.model == LiquidityModel.LOGARITHMIC:
            impact = self._calculate_logarithmic(volume_ratio, volatility)

        elif self.config.model == LiquidityModel.CUSTOM:
            if self._custom_func is None:
                raise ValueError("自定義模型需要先設定計算函數")
            impact = self._custom_func(order_size_usd, adv, volatility)

        else:
            raise ValueError(f"不支援的流動性模型: {self.config.model}")

        # 限制範圍
        impact = np.clip(impact, self.config.min_impact, self.config.max_impact)

        return impact

    def calculate_vectorized(
        self,
        data: pd.DataFrame,
        order_sizes: pd.Series,
        directions: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        向量化計算流動性衝擊（用於回測）

        Args:
            data: OHLCV DataFrame
            order_sizes: 訂單金額序列（USDT）
            directions: 交易方向序列（None 則全部視為做多）

        Returns:
            impact_series: 價格衝擊序列

        範例：
            >>> sizes = pd.Series([10000, 20000, 15000], index=df.index)
            >>> impacts = calculator.calculate_vectorized(df, sizes)
        """
        if directions is None:
            directions = pd.Series(1, index=data.index)

        # 預計算 ADV 和波動率
        self._precompute_factors(data)

        # 向量化計算
        impacts = pd.Series(0.0, index=data.index)

        for i in data.index:
            order_size_val = float(order_sizes.loc[i])
            direction_raw = int(directions.loc[i])
            direction_val: Literal[1, -1] = 1 if direction_raw >= 0 else -1

            idx_loc = data.index.get_loc(i)
            if isinstance(idx_loc, (int, np.integer)):
                idx = int(idx_loc)
            else:
                idx = 0

            impacts.loc[i] = self.calculate_impact(
                data=data,
                order_size_usd=order_size_val,
                direction=direction_val,
                index=idx
            )

        return impacts

    def estimate_execution_price(
        self,
        current_price: float,
        impact: float,
        direction: Literal[1, -1] = 1
    ) -> float:
        """
        估算考慮流動性衝擊的執行價格

        Args:
            current_price: 當前市價
            impact: 價格衝擊百分比
            direction: 交易方向（1=做多，-1=做空）

        Returns:
            execution_price: 預期執行價格

        範例：
            >>> impact = calculator.calculate_impact(df, 50000, index=100)
            >>> price = calculator.estimate_execution_price(50000, impact, 1)
            >>> print(f"預期成交價: ${price:.2f}")
        """
        # 做多時價格上漲，做空時價格下跌
        execution_price = current_price * (1 + direction * impact)
        return execution_price

    def calculate_max_order_size(
        self,
        data: pd.DataFrame,
        price_tolerance: float = 0.01,
        index: Optional[int] = None
    ) -> float:
        """
        計算最大可執行訂單大小（給定價格容忍度）

        Args:
            data: OHLCV DataFrame
            price_tolerance: 可接受的價格衝擊（預設 1%）
            index: 資料索引位置

        Returns:
            max_order_size: 最大訂單金額（USDT）

        範例：
            >>> max_size = calculator.calculate_max_order_size(df, price_tolerance=0.005)
            >>> print(f"最大訂單: ${max_size:,.0f}")
        """
        if index is None:
            index = len(data) - 1

        adv = self._calculate_adv(data, index)
        volatility = 1.0
        if self.config.use_volatility:
            volatility = self._calculate_volatility(data, index)

        # 根據模型反推最大訂單大小
        if self.config.model == LiquidityModel.LINEAR:
            # impact = η × σ × (Q / ADV)
            # Q = impact × ADV / (η × σ)
            max_volume_ratio = price_tolerance / (self.config.impact_coefficient * volatility)

        elif self.config.model == LiquidityModel.SQUARE_ROOT:
            # impact = η × σ × √(Q / ADV)
            # Q = ADV × (impact / (η × σ))²
            max_volume_ratio = (price_tolerance / (self.config.impact_coefficient * volatility)) ** 2

        elif self.config.model == LiquidityModel.LOGARITHMIC:
            # impact = η × σ × log(1 + Q/ADV)
            # Q = ADV × (exp(impact / (η × σ)) - 1)
            max_volume_ratio = np.exp(price_tolerance / (self.config.impact_coefficient * volatility)) - 1

        else:
            # 自定義模型無法反推，使用二分搜尋
            return self._binary_search_max_size(data, price_tolerance, index)

        max_order_size = adv * max_volume_ratio
        return max_order_size

    def get_liquidity_score(
        self,
        data: pd.DataFrame,
        order_size_usd: float,
        index: Optional[int] = None
    ) -> LiquidityLevel:
        """
        評估流動性等級

        Args:
            data: OHLCV DataFrame
            order_size_usd: 訂單金額（USDT）
            index: 資料索引位置

        Returns:
            level: 流動性等級（HIGH/MEDIUM/LOW/VERY_LOW）

        範例：
            >>> level = calculator.get_liquidity_score(df, 50000, index=100)
            >>> print(f"流動性等級: {level.value}")
        """
        if index is None:
            index = len(data) - 1

        adv = self._calculate_adv(data, index)

        if adv == 0:
            return LiquidityLevel.VERY_LOW

        volume_ratio = order_size_usd / adv

        if volume_ratio <= self.config.high_liquidity_threshold:
            return LiquidityLevel.HIGH
        elif volume_ratio <= self.config.medium_liquidity_threshold:
            return LiquidityLevel.MEDIUM
        elif volume_ratio <= self.config.low_liquidity_threshold:
            return LiquidityLevel.LOW
        else:
            return LiquidityLevel.VERY_LOW

    def _calculate_adv(self, data: pd.DataFrame, index: int) -> float:
        """
        計算平均日成交量（Average Daily Volume）

        Args:
            data: OHLCV DataFrame
            index: 當前索引

        Returns:
            adv: 平均日成交量（USDT）
        """
        if self._adv_cache is not None:
            return float(self._adv_cache.iloc[index])

        # 計算成交量（金額）
        volume_value = data['volume'] * data['close']

        # 取窗口內的資料
        start_idx = max(0, index - self.config.adv_window)
        window_volume = volume_value.iloc[start_idx:index + 1]

        if len(window_volume) == 0:
            return 0.0

        # 使用百分位數（更穩定）
        adv = float(window_volume.quantile(self.config.adv_percentile))

        return adv

    def _calculate_volatility(self, data: pd.DataFrame, index: int) -> float:
        """
        計算波動率

        Args:
            data: OHLCV DataFrame
            index: 當前索引

        Returns:
            volatility: 年化波動率（標準化）
        """
        if self._volatility_cache is not None:
            vol = float(self._volatility_cache.iloc[index])
            return vol if not np.isnan(vol) else 1.0

        returns = data['close'].pct_change()
        start_idx = max(0, index - self.config.volatility_window)
        window_returns = returns.iloc[start_idx:index + 1]

        if len(window_returns) < 2:
            return 1.0

        volatility = float(window_returns.std())

        # 標準化（相對於平均波動率）
        avg_volatility = float(returns.std())
        if avg_volatility > 0:
            volatility = volatility / avg_volatility
        else:
            volatility = 1.0

        return volatility if not np.isnan(volatility) else 1.0

    def _calculate_linear(self, volume_ratio: float, volatility: float) -> float:
        """
        線性模型：衝擊 ∝ Q

        公式：impact = η × σ × (Q / ADV)
        """
        impact = self.config.impact_coefficient * volatility * volume_ratio
        return impact

    def _calculate_square_root(self, volume_ratio: float, volatility: float) -> float:
        """
        平方根模型：衝擊 ∝ √Q（學術標準）

        公式：impact = η × σ × √(Q / ADV)

        理論基礎：Almgren-Chriss 模型，市場衝擊呈非線性增長
        """
        impact = self.config.impact_coefficient * volatility * np.sqrt(volume_ratio)
        return impact

    def _calculate_logarithmic(self, volume_ratio: float, volatility: float) -> float:
        """
        對數模型：衝擊 ∝ log(Q)

        公式：impact = η × σ × log(1 + Q/ADV)

        特性：大單的邊際衝擊遞減（更保守的估計）
        """
        impact = self.config.impact_coefficient * volatility * np.log(1 + volume_ratio)
        return impact

    def _precompute_factors(self, data: pd.DataFrame):
        """預計算 ADV 和波動率（用於向量化）"""
        # 計算 ADV
        volume_value = data['volume'] * data['close']
        self._adv_cache = volume_value.rolling(
            self.config.adv_window,
            min_periods=1
        ).quantile(self.config.adv_percentile)

        # 計算波動率
        if self.config.use_volatility:
            returns = data['close'].pct_change()
            rolling_vol = returns.rolling(
                self.config.volatility_window,
                min_periods=2
            ).std()
            avg_vol = returns.std()

            if avg_vol > 0:
                self._volatility_cache = rolling_vol / avg_vol
            else:
                self._volatility_cache = pd.Series(1.0, index=data.index)
        else:
            self._volatility_cache = pd.Series(1.0, index=data.index)

    def _binary_search_max_size(
        self,
        data: pd.DataFrame,
        price_tolerance: float,
        index: int
    ) -> float:
        """
        二分搜尋最大訂單大小（用於自定義模型）

        Args:
            data: OHLCV DataFrame
            price_tolerance: 價格容忍度
            index: 資料索引

        Returns:
            max_size: 最大訂單金額
        """
        adv = self._calculate_adv(data, index)

        # 搜尋範圍：0 到 10 倍 ADV
        low, high = 0.0, adv * 10
        tolerance = adv * 0.0001  # 精度

        while high - low > tolerance:
            mid = (low + high) / 2
            impact = self.calculate_impact(data, mid, index=index)

            if impact < price_tolerance:
                low = mid
            else:
                high = mid

        return low

    def analyze_liquidity(
        self,
        data: pd.DataFrame,
        order_sizes: Optional[list] = None
    ) -> pd.DataFrame:
        """
        分析不同訂單大小的流動性衝擊

        Args:
            data: OHLCV DataFrame
            order_sizes: 訂單大小列表（None 則使用預設值）

        Returns:
            analysis_df: 流動性分析結果

        範例：
            >>> analysis = calculator.analyze_liquidity(df)
            >>> analysis.plot()
        """
        if order_sizes is None:
            order_sizes = [1000, 5000, 10000, 50000, 100000, 500000]

        results = {}

        for size in order_sizes:
            sizes = pd.Series(size, index=data.index)
            impacts = self.calculate_vectorized(data, sizes)
            results[f'size_{size}'] = impacts

        analysis_df = pd.DataFrame(results, index=data.index)
        return analysis_df


# 便捷函數
def create_linear_liquidity(
    impact_coefficient: float = 0.2,
    adv_window: int = 30
) -> LiquidityCalculator:
    """
    建立線性流動性計算器

    Args:
        impact_coefficient: 衝擊係數
        adv_window: ADV 計算窗口

    Returns:
        calculator: 流動性計算器
    """
    config = LiquidityConfig(
        model=LiquidityModel.LINEAR,
        impact_coefficient=impact_coefficient,
        adv_window=adv_window
    )
    return LiquidityCalculator(config)


def create_square_root_liquidity(
    impact_coefficient: float = 0.3,
    adv_window: int = 30,
    use_volatility: bool = True
) -> LiquidityCalculator:
    """
    建立平方根流動性計算器（推薦，學術標準）

    Args:
        impact_coefficient: 衝擊係數（通常 0.1-0.5）
        adv_window: ADV 計算窗口
        use_volatility: 是否考慮波動率

    Returns:
        calculator: 流動性計算器
    """
    config = LiquidityConfig(
        model=LiquidityModel.SQUARE_ROOT,
        impact_coefficient=impact_coefficient,
        adv_window=adv_window,
        use_volatility=use_volatility
    )
    return LiquidityCalculator(config)


def create_logarithmic_liquidity(
    impact_coefficient: float = 0.4,
    adv_window: int = 30
) -> LiquidityCalculator:
    """
    建立對數流動性計算器

    Args:
        impact_coefficient: 衝擊係數
        adv_window: ADV 計算窗口

    Returns:
        calculator: 流動性計算器
    """
    config = LiquidityConfig(
        model=LiquidityModel.LOGARITHMIC,
        impact_coefficient=impact_coefficient,
        adv_window=adv_window
    )
    return LiquidityCalculator(config)
