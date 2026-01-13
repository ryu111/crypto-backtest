"""
組合策略（Composite Strategy）

將多個子策略組合成單一策略，支援多種訊號聚合模式和動態權重優化。
"""

from enum import Enum
from typing import Dict, Optional, Tuple, List
import logging

import pandas as pd
import numpy as np
from pandas import DataFrame, Series

from .base import BaseStrategy
from ..optimizer.portfolio import PortfolioOptimizer, PortfolioWeights

logger = logging.getLogger(__name__)


class SignalAggregation(Enum):
    """訊號聚合模式"""

    WEIGHTED = "weighted"      # 加權平均
    VOTING = "voting"          # 多數決
    RANKED = "ranked"          # 排名選擇
    UNANIMOUS = "unanimous"    # 全體一致


class RebalanceTrigger(Enum):
    """再平衡觸發條件"""

    NONE = "none"                      # 不再平衡
    PERIODIC = "periodic"              # 固定週期
    DRIFT = "drift"                    # 權重漂移
    PERFORMANCE = "performance"        # 績效變化
    REGIME_CHANGE = "regime_change"    # 市場狀態改變


class CompositeStrategy(BaseStrategy):
    """
    組合策略類別

    將多個子策略組合成單一策略，提供：
    1. 多種訊號聚合模式（加權、投票、排名、全體一致）
    2. 動態權重優化（整合 PortfolioOptimizer）
    3. 自動再平衡機制

    使用範例：
        # 建立子策略
        ma_cross = MACross(fast=10, slow=30)
        rsi_strategy = RSIStrategy(period=14)
        macd_strategy = MACDStrategy()

        # 建立組合策略（等權重，多數決）
        composite = CompositeStrategy(
            strategies=[ma_cross, rsi_strategy, macd_strategy],
            aggregation=SignalAggregation.VOTING
        )

        # 建立組合策略（自訂權重，加權平均）
        composite = CompositeStrategy(
            strategies=[ma_cross, rsi_strategy, macd_strategy],
            weights={'ma_cross': 0.5, 'rsi_strategy': 0.3, 'macd_strategy': 0.2},
            aggregation=SignalAggregation.WEIGHTED
        )

        # 動態優化權重
        returns_df = pd.DataFrame({...})  # 各策略歷史回報
        composite.optimize_weights(returns_df, method='max_sharpe')

    Attributes:
        strategies: 子策略字典 {name: strategy}
        weights: 策略權重字典 {name: weight}
        aggregation: 訊號聚合模式
        rebalance_trigger: 再平衡觸發條件
    """

    name = "composite_strategy"
    strategy_type = "composite"
    version = "1.0"
    description = "Multi-strategy composite with dynamic aggregation and rebalancing"

    def __init__(
        self,
        strategies: Optional[List[BaseStrategy]] = None,
        weights: Optional[Dict[str, float]] = None,
        aggregation: SignalAggregation = SignalAggregation.WEIGHTED,
        rebalance_trigger: RebalanceTrigger = RebalanceTrigger.NONE,
        weighted_threshold: float = 0.5,
        ranked_top_n: int = 1,
        rebalance_period: int = 20,
        drift_threshold: float = 0.1,
        performance_threshold: float = 0.2,
        **kwargs
    ):
        """
        初始化組合策略

        Args:
            strategies: 子策略列表
            weights: 策略權重字典 {strategy_name: weight}
                     若未指定，則使用等權重
            aggregation: 訊號聚合模式
            rebalance_trigger: 再平衡觸發條件
            weighted_threshold: WEIGHTED 模式的觸發閾值（預設 0.5）
            ranked_top_n: RANKED 模式選擇的策略數量（預設 1）
            rebalance_period: PERIODIC 模式的再平衡週期（bars）
            drift_threshold: DRIFT 模式的權重漂移閾值
            performance_threshold: PERFORMANCE 模式的績效變化閾值
            **kwargs: 其他參數
        """
        # 初始化子策略
        self.strategies: Dict[str, BaseStrategy] = {}
        self.weights: Dict[str, float] = {}
        self.aggregation = aggregation
        self.rebalance_trigger = rebalance_trigger

        # 聚合參數
        self.weighted_threshold = weighted_threshold
        self.ranked_top_n = ranked_top_n

        # 再平衡參數
        self.rebalance_period = rebalance_period
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        self._bars_since_rebalance = 0
        self._initial_weights: Optional[Dict[str, float]] = None
        self._previous_performance: Optional[float] = None

        # 添加初始策略
        if strategies:
            for strategy in strategies:
                self.add_strategy(strategy)

        # 設定權重
        if weights:
            self._set_weights(weights)
        else:
            self._set_equal_weights()

        # 調用父類初始化
        super().__init__(**kwargs)

    def add_strategy(
        self,
        strategy: BaseStrategy,
        weight: Optional[float] = None
    ) -> None:
        """
        添加子策略

        Args:
            strategy: 子策略實例
            weight: 權重（若未指定，稍後會重新計算等權重）

        Raises:
            TypeError: 如果 strategy 不是 BaseStrategy 實例
        """
        # 類型檢查
        if not isinstance(strategy, BaseStrategy):
            raise TypeError(f"strategy 必須是 BaseStrategy 實例，收到 {type(strategy)}")

        strategy_name = strategy.name

        if strategy_name in self.strategies:
            logger.warning(f"策略 {strategy_name} 已存在，將被覆蓋")

        self.strategies[strategy_name] = strategy

        if weight is not None:
            self.weights[strategy_name] = weight
        else:
            # 重新計算等權重
            self._set_equal_weights()

    def remove_strategy(self, name: str) -> None:
        """
        移除子策略

        Args:
            name: 策略名稱
        """
        if name not in self.strategies:
            logger.warning(f"策略 {name} 不存在")
            return

        del self.strategies[name]
        if name in self.weights:
            del self.weights[name]

        # 重新正規化權重
        self._normalize_weights()

    def _set_equal_weights(self) -> None:
        """設定等權重"""
        n = len(self.strategies)
        if n == 0:
            return

        weight = 1.0 / n
        self.weights = {name: weight for name in self.strategies.keys()}

    def _set_weights(self, weights: Dict[str, float]) -> None:
        """
        設定權重並驗證

        Args:
            weights: 權重字典
        """
        # 驗證所有策略都有權重
        for name in self.strategies.keys():
            if name not in weights:
                raise ValueError(f"策略 {name} 缺少權重")

        self.weights = weights.copy()
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """正規化權重（總和為 1）"""
        if len(self.weights) == 0:
            logger.warning("權重字典為空，無法正規化")
            return

        total = sum(self.weights.values())

        if total == 0:
            logger.warning("權重總和為 0，使用等權重")
            self._set_equal_weights()
            return

        self.weights = {
            name: weight / total
            for name, weight in self.weights.items()
        }

    def calculate_indicators(self, data: DataFrame) -> Dict[str, Series]:
        """
        計算所有子策略的指標

        Args:
            data: OHLCV DataFrame

        Returns:
            彙整的指標字典 {strategy_name.indicator_name: Series}
        """
        all_indicators = {}

        for name, strategy in self.strategies.items():
            indicators = strategy.calculate_indicators(data)

            # 添加策略名稱前綴避免衝突
            for indicator_name, values in indicators.items():
                key = f"{name}.{indicator_name}"
                all_indicators[key] = values

        return all_indicators

    def generate_signals(
        self,
        data: DataFrame
    ) -> Tuple[Series, Series, Series, Series]:
        """
        根據聚合模式產生組合訊號

        Args:
            data: OHLCV DataFrame

        Returns:
            (long_entry, long_exit, short_entry, short_exit)
        """
        if len(self.strategies) == 0:
            # 無子策略，回傳全 False
            return self._create_empty_signals(data)

        # 收集所有子策略的訊號
        all_signals = {}
        for name, strategy in self.strategies.items():
            signals = strategy.generate_signals(data)
            all_signals[name] = signals

        # 根據聚合模式產生最終訊號
        if self.aggregation == SignalAggregation.WEIGHTED:
            return self._aggregate_weighted(all_signals, data)
        elif self.aggregation == SignalAggregation.VOTING:
            return self._aggregate_voting(all_signals, data)
        elif self.aggregation == SignalAggregation.RANKED:
            return self._aggregate_ranked(all_signals, data)
        elif self.aggregation == SignalAggregation.UNANIMOUS:
            return self._aggregate_unanimous(all_signals, data)
        else:
            raise ValueError(f"Unknown aggregation mode: {self.aggregation}")

    def _create_empty_signals(self, data: DataFrame) -> Tuple[Series, Series, Series, Series]:
        """建立全 False 的訊號"""
        empty = self._create_signal_series(data, value=False)
        return empty, empty, empty, empty

    def _aggregate_weighted(
        self,
        all_signals: Dict[str, Tuple[Series, Series, Series, Series]],
        data: DataFrame
    ) -> Tuple[Series, Series, Series, Series]:
        """
        加權平均聚合

        將 bool 訊號轉為 float（True=1, False=0），加權平均後大於閾值為 True。
        """
        n_bars = len(data)

        # 初始化加權訊號（float）
        long_entry_weighted = np.zeros(n_bars)
        long_exit_weighted = np.zeros(n_bars)
        short_entry_weighted = np.zeros(n_bars)
        short_exit_weighted = np.zeros(n_bars)

        # 累加加權訊號
        for name, (le, lx, se, sx) in all_signals.items():
            weight = self.weights[name]
            long_entry_weighted += np.asarray(le, dtype=float) * weight
            long_exit_weighted += np.asarray(lx, dtype=float) * weight
            short_entry_weighted += np.asarray(se, dtype=float) * weight
            short_exit_weighted += np.asarray(sx, dtype=float) * weight

        # 轉為 bool（大於閾值）
        long_entry = pd.Series(
            long_entry_weighted > self.weighted_threshold,
            index=data.index
        )
        long_exit = pd.Series(
            long_exit_weighted > self.weighted_threshold,
            index=data.index
        )
        short_entry = pd.Series(
            short_entry_weighted > self.weighted_threshold,
            index=data.index
        )
        short_exit = pd.Series(
            short_exit_weighted > self.weighted_threshold,
            index=data.index
        )

        return long_entry, long_exit, short_entry, short_exit

    def _aggregate_voting(
        self,
        all_signals: Dict[str, Tuple[Series, Series, Series, Series]],
        data: DataFrame
    ) -> Tuple[Series, Series, Series, Series]:
        """
        多數決聚合

        超過半數策略同意即觸發。
        """
        # 空訊號檢查
        if len(all_signals) == 0:
            return self._create_empty_signals(data)

        n_strategies = len(all_signals)
        threshold = n_strategies / 2

        # 初始化投票計數
        n_bars = len(data)
        long_entry_votes = np.zeros(n_bars)
        long_exit_votes = np.zeros(n_bars)
        short_entry_votes = np.zeros(n_bars)
        short_exit_votes = np.zeros(n_bars)

        # 累加投票
        for _name, (le, lx, se, sx) in all_signals.items():
            long_entry_votes += np.asarray(le, dtype=int)
            long_exit_votes += np.asarray(lx, dtype=int)
            short_entry_votes += np.asarray(se, dtype=int)
            short_exit_votes += np.asarray(sx, dtype=int)

        # 判斷是否超過半數
        long_entry = pd.Series(long_entry_votes > threshold, index=data.index)
        long_exit = pd.Series(long_exit_votes > threshold, index=data.index)
        short_entry = pd.Series(short_entry_votes > threshold, index=data.index)
        short_exit = pd.Series(short_exit_votes > threshold, index=data.index)

        return long_entry, long_exit, short_entry, short_exit

    def _aggregate_ranked(
        self,
        all_signals: Dict[str, Tuple[Series, Series, Series, Series]],
        data: DataFrame
    ) -> Tuple[Series, Series, Series, Series]:
        """
        排名選擇聚合

        選擇權重最高的 top_n 策略，只要其中任何一個發出訊號即觸發（OR 邏輯）。

        Note:
            - ranked_top_n=1: 只聽權重最高的策略
            - ranked_top_n=2: 權重前兩高的策略中，任一發出訊號即觸發
            - 如果要「全體一致」，請使用 UNANIMOUS 模式
        """
        # 空訊號檢查
        if len(all_signals) == 0:
            return self._create_empty_signals(data)

        # 按權重排序
        sorted_names = sorted(
            self.weights.keys(),
            key=lambda x: self.weights[x],
            reverse=True
        )

        # 選擇 top_n（自動限制不超過實際策略數）
        actual_top_n = min(self.ranked_top_n, len(sorted_names))
        top_names = sorted_names[:actual_top_n]

        # 對 top_n 策略進行「或」運算
        n_bars = len(data)
        long_entry = np.zeros(n_bars, dtype=bool)
        long_exit = np.zeros(n_bars, dtype=bool)
        short_entry = np.zeros(n_bars, dtype=bool)
        short_exit = np.zeros(n_bars, dtype=bool)

        for name in top_names:
            if name in all_signals:
                le, lx, se, sx = all_signals[name]
                long_entry |= le.values
                long_exit |= lx.values
                short_entry |= se.values
                short_exit |= sx.values

        return (
            pd.Series(long_entry, index=data.index),
            pd.Series(long_exit, index=data.index),
            pd.Series(short_entry, index=data.index),
            pd.Series(short_exit, index=data.index)
        )

    def _aggregate_unanimous(
        self,
        all_signals: Dict[str, Tuple[Series, Series, Series, Series]],
        data: DataFrame
    ) -> Tuple[Series, Series, Series, Series]:
        """
        全體一致聚合

        所有策略都同意才觸發。
        """
        n_bars = len(data)
        long_entry = np.ones(n_bars, dtype=bool)
        long_exit = np.ones(n_bars, dtype=bool)
        short_entry = np.ones(n_bars, dtype=bool)
        short_exit = np.ones(n_bars, dtype=bool)

        for _name, (le, lx, se, sx) in all_signals.items():
            long_entry &= np.asarray(le, dtype=bool)
            long_exit &= np.asarray(lx, dtype=bool)
            short_entry &= np.asarray(se, dtype=bool)
            short_exit &= np.asarray(sx, dtype=bool)

        return (
            pd.Series(long_entry, index=data.index),
            pd.Series(long_exit, index=data.index),
            pd.Series(short_entry, index=data.index),
            pd.Series(short_exit, index=data.index)
        )

    def optimize_weights(
        self,
        returns: pd.DataFrame,
        method: str = 'max_sharpe',
        risk_free_rate: float = 0.0,
        **optimizer_kwargs
    ) -> PortfolioWeights:
        """
        使用 PortfolioOptimizer 優化策略權重

        Args:
            returns: 各策略回報 DataFrame (columns=策略名稱, index=日期)
            method: 優化方法
                - 'max_sharpe': 最大化 Sharpe Ratio
                - 'risk_parity': 風險平價
                - 'equal_weight': 等權重
                - 'inverse_volatility': 反波動率加權
                - 'mean_variance': Mean-Variance 優化
            risk_free_rate: 無風險利率（年化）
            **optimizer_kwargs: 傳遞給優化方法的額外參數

        Returns:
            PortfolioWeights 物件

        Example:
            returns_df = pd.DataFrame({
                'ma_cross': [...],
                'rsi_strategy': [...],
                'macd_strategy': [...]
            })

            composite.optimize_weights(
                returns_df,
                method='max_sharpe',
                max_weight=0.5  # 單一策略最大 50%
            )
        """
        # 驗證策略名稱
        missing_strategies = set(self.strategies.keys()) - set(returns.columns)
        if missing_strategies:
            raise ValueError(f"回報資料缺少策略: {missing_strategies}")

        # 只保留有效策略的回報
        strategy_names = list(self.strategies.keys())
        returns_filtered = pd.DataFrame(returns[strategy_names])

        try:
            # 建立優化器
            optimizer = PortfolioOptimizer(
                returns=returns_filtered,
                risk_free_rate=risk_free_rate
            )

            # 執行優化
            if method == 'max_sharpe':
                result = optimizer.max_sharpe_optimize(**optimizer_kwargs)
            elif method == 'risk_parity':
                result = optimizer.risk_parity_optimize(**optimizer_kwargs)
            elif method == 'equal_weight':
                result = optimizer.equal_weight_portfolio()
            elif method == 'inverse_volatility':
                result = optimizer.inverse_volatility_portfolio()
            elif method == 'mean_variance':
                result = optimizer.mean_variance_optimize(**optimizer_kwargs)
            else:
                raise ValueError(f"Unknown optimization method: {method}")

            # 更新權重
            self.weights = result.weights.copy()

            logger.info(f"權重優化完成 ({method}): {result.weights}")
            logger.info(f"預期 Sharpe: {result.sharpe_ratio:.4f}")

            return result

        except Exception as e:
            logger.warning(f"權重優化失敗: {e}，使用等權重")
            self._set_equal_weights()
            return PortfolioWeights(
                weights=self.weights.copy(),
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                optimization_success=False,
                optimization_message=str(e)
            )

    def rebalance(
        self,
        returns: Optional[pd.DataFrame] = None,
        current_performance: Optional[float] = None
    ) -> bool:
        """
        檢查並執行動態再平衡

        Args:
            returns: 策略回報 DataFrame（DRIFT/PERFORMANCE 模式需要）
            current_performance: 當前績效（PERFORMANCE 模式需要）

        Returns:
            是否執行了再平衡
        """
        should_rebalance = False

        if self.rebalance_trigger == RebalanceTrigger.NONE:
            return False

        # PERIODIC：固定週期
        if self.rebalance_trigger == RebalanceTrigger.PERIODIC:
            self._bars_since_rebalance += 1
            if self._bars_since_rebalance >= self.rebalance_period:
                should_rebalance = True
                self._bars_since_rebalance = 0

        # DRIFT：權重漂移
        elif self.rebalance_trigger == RebalanceTrigger.DRIFT:
            if self._initial_weights is None:
                self._initial_weights = self.weights.copy()

            # 計算權重漂移
            max_drift = max(
                abs(self.weights[name] - self._initial_weights[name])
                for name in self.weights.keys()
            )

            if max_drift > self.drift_threshold:
                should_rebalance = True

        # PERFORMANCE：績效變化
        elif self.rebalance_trigger == RebalanceTrigger.PERFORMANCE:
            if current_performance is not None:
                if self._previous_performance is not None:
                    performance_change = abs(
                        current_performance - self._previous_performance
                    )

                    if performance_change > self.performance_threshold:
                        should_rebalance = True

                self._previous_performance = current_performance

        # REGIME_CHANGE：市場狀態改變
        elif self.rebalance_trigger == RebalanceTrigger.REGIME_CHANGE:
            # 需要外部觸發（透過 trigger_regime_rebalance 方法）
            # 此處不自動觸發
            pass

        # 執行再平衡
        if should_rebalance:
            if returns is None:
                logger.error(f"需要再平衡但缺少 returns 資料 ({self.rebalance_trigger.value})")
                return False

            logger.info(f"觸發再平衡 ({self.rebalance_trigger.value})")
            self.optimize_weights(returns)
            self._initial_weights = self.weights.copy()
            return True

        return False

    def trigger_regime_rebalance(self, returns: pd.DataFrame) -> bool:
        """
        手動觸發 REGIME_CHANGE 再平衡

        當外部檢測到市場狀態改變時，調用此方法觸發再平衡。

        Args:
            returns: 策略回報 DataFrame

        Returns:
            是否成功執行再平衡
        """
        if self.rebalance_trigger != RebalanceTrigger.REGIME_CHANGE:
            logger.warning("rebalance_trigger 不是 REGIME_CHANGE，忽略觸發")
            return False

        logger.info("外部觸發 REGIME_CHANGE 再平衡")
        self.optimize_weights(returns)
        self._initial_weights = self.weights.copy()
        return True

    def get_info(self) -> Dict:
        """
        取得組合策略詳細資訊

        Returns:
            策略資訊字典
        """
        base_info = super().get_info()

        base_info.update({
            'strategies': {
                name: strategy.get_info()
                for name, strategy in self.strategies.items()
            },
            'weights': self.weights,
            'aggregation': self.aggregation.value,
            'rebalance_trigger': self.rebalance_trigger.value,
            'n_strategies': len(self.strategies)
        })

        return base_info

    def validate_params(self) -> bool:
        """驗證參數有效性"""
        if len(self.strategies) == 0:
            logger.warning("組合策略沒有子策略")
            return False

        # 驗證權重總和約為 1
        weight_sum = sum(self.weights.values())
        if not np.isclose(weight_sum, 1.0, atol=1e-6):
            logger.warning(f"權重總和不為 1: {weight_sum}")
            return False

        # 驗證所有子策略都有權重
        if set(self.strategies.keys()) != set(self.weights.keys()):
            logger.warning("策略與權重不匹配")
            return False

        return True

    def __repr__(self) -> str:
        """字串表示"""
        return (
            f"CompositeStrategy("
            f"n_strategies={len(self.strategies)}, "
            f"aggregation={self.aggregation.value})"
        )

    def __str__(self) -> str:
        """友善字串表示"""
        strategies_str = ', '.join(self.strategies.keys())
        return f"Composite[{strategies_str}] ({self.aggregation.value})"
