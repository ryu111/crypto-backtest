"""
統一介面定義 - 解決 API 不匹配問題

使用 Python Protocol 定義預期的介面，
讓不同模組之間有一致的契約。
"""

from typing import Protocol, Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


# =============================================================================
# 策略註冊表介面
# =============================================================================

class IStrategyRegistry(Protocol):
    """策略註冊表介面"""

    def list_all(self) -> List[str]:
        """列出所有策略名稱"""
        ...

    def get(self, name: str) -> Any:
        """取得策略類別"""
        ...

    def exists(self, name: str) -> bool:
        """檢查策略是否存在"""
        ...

    def get_param_space(self, name: str) -> Dict[str, Any]:
        """取得策略參數空間"""
        ...


# =============================================================================
# 實驗記錄器介面
# =============================================================================

@dataclass
class StrategyStatsData:
    """策略統計資料"""
    name: str
    attempts: int = 0
    successes: int = 0
    avg_sharpe: float = 0.0
    best_sharpe: float = 0.0
    last_attempt: Optional[datetime] = None
    last_params: Optional[Dict[str, Any]] = None


class IExperimentRecorder(Protocol):
    """實驗記錄器介面"""

    def log_experiment(self, experiment: Dict[str, Any]) -> str:
        """記錄實驗，返回 experiment_id"""
        ...

    def get_strategy_stats(self, strategy_name: str) -> Optional[StrategyStatsData]:
        """取得策略統計"""
        ...

    def update_strategy_stats(self, strategy_name: str, stats: StrategyStatsData) -> None:
        """更新策略統計"""
        ...

    def query_experiments(
        self,
        strategy: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """查詢實驗記錄"""
        ...


# =============================================================================
# 回測引擎介面
# =============================================================================

@dataclass
class BacktestResultData:
    """回測結果資料"""
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    trade_count: int
    daily_returns: Optional[pd.Series] = None
    equity_curve: Optional[pd.Series] = None


class IBacktestEngine(Protocol):
    """回測引擎介面"""

    def load_data(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """載入或設定市場資料"""
        ...

    def run(self, strategy: Any) -> BacktestResultData:
        """執行回測"""
        ...


# =============================================================================
# 資料獲取介面
# =============================================================================

class IDataFetcher(Protocol):
    """資料獲取介面"""

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 1000,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """獲取 OHLCV 資料"""
        ...

    def fetch_funding_rates(
        self,
        symbol: str,
        limit: int = 500
    ) -> pd.DataFrame:
        """獲取資金費率"""
        ...


# =============================================================================
# 適配器工廠
# =============================================================================

class AdapterFactory:
    """
    適配器工廠 - 將現有類別適配到統一介面

    使用範例:
        registry = StrategyRegistry()
        adapted = AdapterFactory.adapt_registry(registry)
        # 現在 adapted 符合 IStrategyRegistry 介面
    """

    @staticmethod
    def adapt_registry(registry: Any) -> IStrategyRegistry:
        """適配 StrategyRegistry"""
        # 如果已經符合介面，直接返回
        if hasattr(registry, 'list_all'):
            return registry

        # 否則建立適配器
        class RegistryAdapter:
            def __init__(self, inner):
                self._inner = inner

            def list_all(self) -> List[str]:
                if hasattr(self._inner, 'list_all'):
                    return self._inner.list_all()
                elif hasattr(self._inner, '_strategies'):
                    return list(self._inner._strategies.keys())
                else:
                    raise NotImplementedError("Registry doesn't support listing")

            def get(self, name: str) -> Any:
                return self._inner.get(name)

            def exists(self, name: str) -> bool:
                if hasattr(self._inner, 'exists'):
                    return self._inner.exists(name)
                return name in self.list_all()

            def get_param_space(self, name: str) -> Dict[str, Any]:
                if hasattr(self._inner, 'get_param_space'):
                    return self._inner.get_param_space(name)
                strategy_class = self.get(name)
                return getattr(strategy_class, 'param_space', {})

        return RegistryAdapter(registry)

    @staticmethod
    def adapt_recorder(recorder: Any) -> IExperimentRecorder:
        """適配 ExperimentRecorder"""

        class RecorderAdapter:
            def __init__(self, inner):
                self._inner = inner
                self._stats_cache: Dict[str, StrategyStatsData] = {}

            def log_experiment(self, experiment: Dict[str, Any]) -> str:
                return self._inner.log_experiment(experiment)

            def get_strategy_stats(self, strategy_name: str) -> Optional[StrategyStatsData]:
                # 嘗試從快取取得
                if strategy_name in self._stats_cache:
                    return self._stats_cache[strategy_name]

                # 嘗試從內部記錄器建立統計
                try:
                    experiments = self._inner.query_experiments()
                    strategy_exps = [e for e in experiments if e.get('strategy') == strategy_name]

                    if not strategy_exps:
                        return None

                    # 計算統計
                    sharpes = [e.get('sharpe_ratio', 0) for e in strategy_exps]
                    stats = StrategyStatsData(
                        name=strategy_name,
                        attempts=len(strategy_exps),
                        successes=sum(1 for e in strategy_exps if e.get('passed', False)),
                        avg_sharpe=sum(sharpes) / len(sharpes) if sharpes else 0.0,
                        best_sharpe=max(sharpes) if sharpes else 0.0,
                    )
                    self._stats_cache[strategy_name] = stats
                    return stats
                except Exception:
                    return None

            def update_strategy_stats(self, strategy_name: str, stats: StrategyStatsData) -> None:
                self._stats_cache[strategy_name] = stats

            def query_experiments(
                self,
                strategy: Optional[str] = None,
                symbol: Optional[str] = None,
                limit: int = 100
            ) -> List[Dict[str, Any]]:
                return self._inner.query_experiments()

        return RecorderAdapter(recorder)


# =============================================================================
# 使用範例
# =============================================================================

def example_usage():
    """展示如何使用統一介面"""
    from src.strategies import StrategyRegistry
    from src.learning import ExperimentRecorder

    # 原始類別
    registry = StrategyRegistry()
    recorder = ExperimentRecorder()

    # 適配到統一介面
    adapted_registry = AdapterFactory.adapt_registry(registry)
    adapted_recorder = AdapterFactory.adapt_recorder(recorder)

    # 現在可以使用統一的 API
    strategies = adapted_registry.list_all()  # 不再需要知道是 list_all 還是 list_strategies
    stats = adapted_recorder.get_strategy_stats('trend_ma_cross')

    return strategies, stats
