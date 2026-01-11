"""
滑點模擬模組

提供動態滑點計算，模擬真實交易環境中的價格滑動。

滑點來源：
1. 市場深度不足（Order Book 深度）
2. 市場波動率（波動越大，滑點越高）
3. 訂單大小（大單對市場的衝擊）
4. 訂單類型（市價單 vs 限價單）

參考：
- .claude/skills/回測核心/references/slippage-models.md
"""

from dataclasses import dataclass
from typing import Literal, Optional, Callable
import pandas as pd
import numpy as np
from enum import Enum


class OrderType(Enum):
    """訂單類型"""
    MARKET = "market"      # 市價單：完整滑點
    LIMIT = "limit"        # 限價單：無滑點（但可能不成交）
    STOP = "stop"          # 止損單：可能有更高滑點
    STOP_LIMIT = "stop_limit"  # 止損限價單


class SlippageModel(Enum):
    """滑點模型類型"""
    FIXED = "fixed"              # 固定滑點
    DYNAMIC = "dynamic"          # 動態滑點（波動率 + 成交量）
    MARKET_IMPACT = "market_impact"  # 市場衝擊模型
    CUSTOM = "custom"            # 自定義函數


@dataclass
class SlippageConfig:
    """滑點配置"""

    # 基本設定
    model: SlippageModel = SlippageModel.DYNAMIC
    base_slippage: float = 0.0005  # 基礎滑點 0.05%

    # 動態滑點參數
    volatility_factor: float = 1.0  # 波動率影響係數
    volume_factor: float = 0.5      # 成交量影響係數

    # 市場衝擊參數
    market_impact_coeff: float = 0.1  # 市場衝擊係數
    liquidity_threshold: float = 0.01  # 流動性門檻（訂單/成交量）

    # 限制
    max_slippage: float = 0.01      # 最大滑點 1%
    min_slippage: float = 0.0       # 最小滑點 0%

    # 訂單類型調整
    stop_order_multiplier: float = 1.5  # 止損單滑點倍數

    # 窗口參數
    volatility_window: int = 20     # 波動率計算窗口
    volume_window: int = 20         # 成交量計算窗口

    def __post_init__(self):
        """驗證配置"""
        if self.base_slippage < 0:
            raise ValueError("基礎滑點不能為負數")

        if self.max_slippage < self.min_slippage:
            raise ValueError("最大滑點不能小於最小滑點")

        if self.max_slippage > 0.1:
            raise ValueError("最大滑點不應超過 10%（建議值）")


class SlippageCalculator:
    """
    滑點計算器

    提供多種滑點模型計算方法。

    使用範例：
        config = SlippageConfig(
            model=SlippageModel.DYNAMIC,
            base_slippage=0.0005,
            max_slippage=0.01
        )

        calculator = SlippageCalculator(config)
        slippage = calculator.calculate(
            data=price_data,
            order_size=10000,
            order_type=OrderType.MARKET,
            index=current_index
        )
    """

    def __init__(self, config: Optional[SlippageConfig] = None):
        """
        初始化滑點計算器

        Args:
            config: 滑點配置，None 則使用預設值
        """
        self.config = config or SlippageConfig()
        self._custom_func: Optional[Callable] = None

        # 快取計算結果
        self._volatility_cache: Optional[pd.Series] = None
        self._volume_cache: Optional[pd.Series] = None

    def set_custom_function(self, func: Callable):
        """
        設定自定義滑點計算函數

        Args:
            func: 函數簽名 (data, order_size, index) -> float

        範例：
            def my_slippage(data, order_size, index):
                return 0.001  # 固定 0.1%

            calculator.set_custom_function(my_slippage)
        """
        self._custom_func = func
        self.config.model = SlippageModel.CUSTOM

    def calculate(
        self,
        data: pd.DataFrame,
        order_size: float,
        order_type: OrderType = OrderType.MARKET,
        direction: Literal[1, -1] = 1,
        index: Optional[int] = None
    ) -> float:
        """
        計算滑點

        Args:
            data: OHLCV DataFrame（需包含 'close', 'volume'）
            order_size: 訂單金額（USDT）
            order_type: 訂單類型
            direction: 交易方向（1=做多，-1=做空）
            index: 資料索引位置（None 則使用最後一筆）

        Returns:
            slippage: 滑點百分比（如 0.001 = 0.1%）

        範例：
            >>> slippage = calculator.calculate(
            ...     data=df,
            ...     order_size=10000,
            ...     order_type=OrderType.MARKET,
            ...     direction=1,
            ...     index=100
            ... )
            >>> print(f"滑點: {slippage:.4%}")
        """
        # 限價單無滑點
        if order_type == OrderType.LIMIT:
            return 0.0

        # 使用最後一筆資料
        if index is None:
            index = len(data) - 1

        # 根據模型類型計算
        if self.config.model == SlippageModel.FIXED:
            slippage = self._calculate_fixed()

        elif self.config.model == SlippageModel.DYNAMIC:
            slippage = self._calculate_dynamic(data, index)

        elif self.config.model == SlippageModel.MARKET_IMPACT:
            slippage = self._calculate_market_impact(data, order_size, index)

        elif self.config.model == SlippageModel.CUSTOM:
            if self._custom_func is None:
                raise ValueError("自定義模型需要先設定計算函數")
            slippage = self._custom_func(data, order_size, index)

        else:
            raise ValueError(f"不支援的滑點模型: {self.config.model}")

        # 止損單調整
        if order_type == OrderType.STOP:
            slippage *= self.config.stop_order_multiplier

        # 限制範圍
        slippage = np.clip(slippage, self.config.min_slippage, self.config.max_slippage)

        return slippage

    def calculate_vectorized(
        self,
        data: pd.DataFrame,
        order_sizes: pd.Series,
        order_types: Optional[pd.Series] = None,
        directions: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        向量化計算滑點（用於回測）

        Args:
            data: OHLCV DataFrame
            order_sizes: 訂單金額序列
            order_types: 訂單類型序列（None 則全部視為市價單）
            directions: 交易方向序列（None 則全部視為做多）

        Returns:
            slippage_series: 滑點序列

        範例：
            >>> sizes = pd.Series([10000, 20000, 15000], index=df.index)
            >>> slippages = calculator.calculate_vectorized(df, sizes)
        """
        if order_types is None:
            order_types = pd.Series(OrderType.MARKET, index=data.index)

        if directions is None:
            directions = pd.Series(1, index=data.index)

        # 預計算波動率和成交量因子
        if self.config.model in [SlippageModel.DYNAMIC, SlippageModel.MARKET_IMPACT]:
            self._precompute_factors(data)

        # 向量化計算
        slippages = pd.Series(0.0, index=data.index)

        for i in data.index:
            # 從 Series 取值並轉換為原生 Python 類型
            order_size_val = float(order_sizes.loc[i])  # type: ignore[arg-type]

            # 取得 OrderType 值
            order_type_raw = order_types.loc[i]
            if isinstance(order_type_raw, OrderType):
                order_type_val = order_type_raw
            else:
                order_type_val = OrderType.MARKET

            direction_raw = int(directions.loc[i])  # type: ignore[arg-type]
            direction_val: Literal[1, -1] = 1 if direction_raw >= 0 else -1

            idx_loc = data.index.get_loc(i)
            # get_loc 可能返回 int, slice, 或 ndarray，這裡我們預期是 int
            if isinstance(idx_loc, (int, np.integer)):
                idx = int(idx_loc)
            else:
                idx = 0

            slippages.loc[i] = self.calculate(
                data=data,
                order_size=order_size_val,
                order_type=order_type_val,
                direction=direction_val,
                index=idx
            )

        return slippages

    def _calculate_fixed(self) -> float:
        """計算固定滑點"""
        return self.config.base_slippage

    def _calculate_dynamic(
        self,
        data: pd.DataFrame,
        index: int
    ) -> float:
        """
        計算動態滑點

        公式：
        滑點 = 基礎滑點 × 波動率因子 × (1 + 成交量因子)

        其中：
        - 波動率因子 = 當前波動率 / 平均波動率
        - 成交量因子 = 訂單大小 / 平均成交量
        """
        # 計算波動率因子
        if self._volatility_cache is None:
            returns = data['close'].pct_change()
            volatility = returns.rolling(self.config.volatility_window).std()
            avg_volatility_raw = volatility.mean()
            # 轉換為 float 以進行標量比較
            # 安全地轉換為 float（使用 bool() 判斷 pd.isna 的結果）
            is_na = bool(pd.isna(avg_volatility_raw))
            avg_volatility = 0.0 if is_na else float(avg_volatility_raw)

            if avg_volatility == 0.0:
                vol_factor = 1.0
            else:
                current_vol = float(volatility.iloc[index])
                # 處理 NaN 或 0 值
                if np.isnan(current_vol) or current_vol == 0:
                    vol_factor = 1.0
                else:
                    vol_factor = current_vol / avg_volatility
        else:
            vol_factor = float(self._volatility_cache.iloc[index])
            # 處理 NaN
            if np.isnan(vol_factor):
                vol_factor = 1.0

        # 動態滑點
        slippage = (
            self.config.base_slippage *
            (1 + self.config.volatility_factor * (vol_factor - 1))
        )

        return slippage

    def _calculate_market_impact(
        self,
        data: pd.DataFrame,
        order_size: float,
        index: int
    ) -> float:
        """
        計算市場衝擊滑點

        公式：
        滑點 = 基礎滑點 × (1 + 市場衝擊)

        市場衝擊 = 市場衝擊係數 × sqrt(訂單大小 / 平均成交量)

        原理：大單對市場的衝擊呈非線性增長（平方根關係）
        """
        # 計算平均成交量
        avg_volume_value = (
            data['volume'].iloc[max(0, index - self.config.volume_window):index]
            .mean() * data['close'].iloc[index]
        )

        if avg_volume_value == 0:
            volume_ratio = 0
        else:
            volume_ratio = order_size / avg_volume_value

        # 市場衝擊（非線性）
        market_impact = self.config.market_impact_coeff * np.sqrt(volume_ratio)

        # 計算總滑點
        slippage = self.config.base_slippage * (1 + market_impact)

        return slippage

    def _precompute_factors(self, data: pd.DataFrame):
        """預計算波動率和成交量因子（用於向量化）"""
        # 計算波動率因子
        returns = data['close'].pct_change()
        volatility = returns.rolling(self.config.volatility_window).std()
        avg_volatility = volatility.mean()

        if avg_volatility > 0:
            self._volatility_cache = volatility / avg_volatility
        else:
            self._volatility_cache = pd.Series(1.0, index=data.index)

        # 計算成交量因子
        volume_value = data['volume'] * data['close']
        avg_volume = volume_value.rolling(self.config.volume_window).mean()

        self._volume_cache = avg_volume

    def estimate_execution_price(
        self,
        current_price: float,
        slippage: float,
        direction: Literal[1, -1] = 1
    ) -> float:
        """
        估算執行價格

        Args:
            current_price: 當前市價
            slippage: 滑點百分比
            direction: 交易方向（1=做多，-1=做空）

        Returns:
            execution_price: 預期執行價格

        範例：
            >>> price = calculator.estimate_execution_price(50000, 0.001, 1)
            >>> print(f"預期成交價: ${price:.2f}")
            預期成交價: $50050.00  # 做多滑價上漲
        """
        # 做多時滑價上漲，做空時滑價下跌
        execution_price = current_price * (1 + direction * slippage)
        return execution_price

    def get_slippage_curve(
        self,
        data: pd.DataFrame,
        order_sizes: Optional[list] = None
    ) -> pd.DataFrame:
        """
        產生滑點曲線（用於分析）

        Args:
            data: OHLCV DataFrame
            order_sizes: 訂單大小列表（None 則使用預設值）

        Returns:
            slippage_df: 包含不同訂單大小的滑點數據

        範例：
            >>> curve = calculator.get_slippage_curve(df)
            >>> curve.plot()
        """
        if order_sizes is None:
            order_sizes = [1000, 5000, 10000, 50000, 100000]

        results = {}

        for size in order_sizes:
            sizes = pd.Series(size, index=data.index)
            slippages = self.calculate_vectorized(data, sizes)
            results[f'size_{size}'] = slippages

        slippage_df = pd.DataFrame(results, index=data.index)
        return slippage_df

    def analyze_impact(
        self,
        data: pd.DataFrame,
        trades: pd.DataFrame
    ) -> dict:
        """
        分析滑點對交易的影響

        Args:
            data: OHLCV DataFrame
            trades: 交易記錄（需包含 'entry_time', 'size'）

        Returns:
            analysis: 滑點影響分析結果

        範例：
            >>> analysis = calculator.analyze_impact(df, trades_df)
            >>> print(f"平均滑點成本: {analysis['avg_cost']:.2%}")
        """
        total_cost = 0.0
        slippage_list = []

        for _, trade in trades.iterrows():
            try:
                # 嘗試精確匹配
                idx_loc = data.index.get_loc(trade['entry_time'])
                # get_loc 可能返回 int, slice, 或 ndarray
                idx = int(idx_loc) if isinstance(idx_loc, (int, np.integer)) else None
                if idx is None:
                    continue
            except KeyError:
                # 如果精確匹配失敗，使用最接近的時間點
                try:
                    idx_arr = data.index.get_indexer([trade['entry_time']], method='nearest')
                    idx = int(idx_arr[0])
                    if idx == -1:
                        continue
                except Exception:
                    continue

            # 確保 order_size 是 float 類型
            order_size = float(trade.get('size', 0) or 0)

            slippage = self.calculate(
                data=data,
                order_size=order_size,
                index=idx
            )

            cost = order_size * slippage
            total_cost += cost
            slippage_list.append(slippage)

        if len(slippage_list) == 0:
            return {
                'total_cost': 0.0,
                'avg_slippage': 0.0,
                'max_slippage': 0.0,
                'min_slippage': 0.0,
                'std_slippage': 0.0,
                'median_slippage': 0.0
            }

        return {
            'total_cost': total_cost,
            'avg_slippage': np.mean(slippage_list),
            'max_slippage': np.max(slippage_list),
            'min_slippage': np.min(slippage_list),
            'std_slippage': np.std(slippage_list),
            'median_slippage': np.median(slippage_list)
        }


# 便捷函數
def create_fixed_slippage(slippage: float = 0.0005) -> SlippageCalculator:
    """
    建立固定滑點計算器

    Args:
        slippage: 固定滑點百分比

    Returns:
        calculator: 滑點計算器
    """
    config = SlippageConfig(
        model=SlippageModel.FIXED,
        base_slippage=slippage
    )
    return SlippageCalculator(config)


def create_dynamic_slippage(
    base_slippage: float = 0.0005,
    volatility_factor: float = 1.0,
    max_slippage: float = 0.01
) -> SlippageCalculator:
    """
    建立動態滑點計算器

    Args:
        base_slippage: 基礎滑點
        volatility_factor: 波動率影響係數
        max_slippage: 最大滑點

    Returns:
        calculator: 滑點計算器
    """
    config = SlippageConfig(
        model=SlippageModel.DYNAMIC,
        base_slippage=base_slippage,
        volatility_factor=volatility_factor,
        max_slippage=max_slippage
    )
    return SlippageCalculator(config)


def create_market_impact_slippage(
    base_slippage: float = 0.0005,
    market_impact_coeff: float = 0.1,
    max_slippage: float = 0.01
) -> SlippageCalculator:
    """
    建立市場衝擊滑點計算器

    Args:
        base_slippage: 基礎滑點
        market_impact_coeff: 市場衝擊係數
        max_slippage: 最大滑點

    Returns:
        calculator: 滑點計算器
    """
    config = SlippageConfig(
        model=SlippageModel.MARKET_IMPACT,
        base_slippage=base_slippage,
        market_impact_coeff=market_impact_coeff,
        max_slippage=max_slippage
    )
    return SlippageCalculator(config)
