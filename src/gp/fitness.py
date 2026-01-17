"""
GP 適應度函數

將 GP 個體轉換為交易訊號，執行回測，計算適應度。

使用範例:
    from src.gp.fitness import FitnessEvaluator, FitnessConfig

    evaluator = FitnessEvaluator(
        pset=pset,
        backtest_engine=engine,
        data=ohlcv_data,
        config=FitnessConfig(sharpe_weight=0.5)
    )

    fitness = evaluator.evaluate(individual)  # Returns (score,)
"""

from typing import Callable, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from numpy.typing import NDArray

try:
    from deap import base, creator, tools, gp
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    base = creator = tools = gp = None

from ..backtester.engine import BacktestEngine, BacktestResult


# ============================================================================
# 常數定義
# ============================================================================

INVALID_FITNESS = -1e10  # 無效個體適應度
MAX_DRAWDOWN_PENALTY_SCORE = -1.0  # 超過最大回撤限制的懲罰分數


# ============================================================================
# 內部輔助類別
# ============================================================================

class _GPGeneratedStrategy:
    """
    GP 生成策略的簡單包裝

    用於回測引擎執行 GP 演化產生的交易訊號。
    """
    name = "GP_Generated"
    params = {}

    def __init__(self, long_entry: pd.Series, long_exit: pd.Series):
        """
        初始化策略

        Args:
            long_entry: 做多進場訊號
            long_exit: 做多出場訊號
        """
        self.long_entry = long_entry
        self.long_exit = long_exit

    def generate_signals(self, data):
        """生成交易訊號"""
        return (
            self.long_entry,
            self.long_exit,
            pd.Series(False, index=self.long_entry.index),  # short_entry
            pd.Series(False, index=self.long_entry.index),  # short_exit
        )


# ============================================================================
# 適應度配置
# ============================================================================

@dataclass
class FitnessConfig:
    """適應度評估配置

    定義如何將回測結果轉換為適應度分數。

    Attributes:
        sharpe_weight: Sharpe Ratio 權重（預設 0.4）
        return_weight: 報酬率權重（預設 0.2）
        drawdown_weight: 回撤懲罰權重（預設 0.2）
        win_rate_weight: 勝率權重（預設 0.1）
        trade_count_weight: 交易數量權重（預設 0.1）
        min_trades: 最少交易次數（少於則懲罰）
        max_drawdown_limit: 最大回撤限制（超過則返回負分）
        complexity_penalty: 複雜度懲罰係數（每個節點）
    """

    # 主要指標權重（總和應為 1.0）
    sharpe_weight: float = 0.4
    return_weight: float = 0.2
    drawdown_weight: float = 0.2
    win_rate_weight: float = 0.1
    trade_count_weight: float = 0.1

    # 約束參數
    min_trades: int = 10  # 最少交易次數
    max_drawdown_limit: float = 0.5  # 最大回撤限制（50%）
    complexity_penalty: float = 0.01  # 複雜度懲罰係數

    # 正規化範圍
    sharpe_max: float = 3.0  # Sharpe > 3.0 視為滿分
    return_max: float = 2.0  # 200% 報酬視為滿分

    def __post_init__(self):
        """驗證配置"""
        total_weight = (
            self.sharpe_weight +
            self.return_weight +
            self.drawdown_weight +
            self.win_rate_weight +
            self.trade_count_weight
        )
        if not abs(total_weight - 1.0) < 1e-6:
            raise ValueError(
                f"權重總和必須為 1.0，當前為 {total_weight:.10f}"
            )


# ============================================================================
# 適應度評估器
# ============================================================================

class FitnessEvaluator:
    """
    適應度評估器

    將 GP 個體編譯為函數，執行回測，計算適應度分數。

    工作流程:
        1. 編譯 GP 個體為可執行函數
        2. 使用函數生成交易訊號
        3. 通過回測引擎執行回測
        4. 計算複合適應度分數

    使用範例:
        evaluator = FitnessEvaluator(
            pset=pset,
            backtest_engine=engine,
            data=ohlcv_data
        )

        fitness_score = evaluator.evaluate(individual)
    """

    def __init__(
        self,
        pset: 'gp.PrimitiveSetTyped',
        backtest_engine: BacktestEngine,
        data: Union[pd.DataFrame, NDArray[np.float64]],
        config: Optional[FitnessConfig] = None
    ):
        """
        初始化評估器

        Args:
            pset: DEAP 原語集（PrimitiveSetTyped）
            backtest_engine: 回測引擎實例
            data: OHLCV 數據（DataFrame 或 numpy array）
            config: 適應度配置（使用預設值如果未提供）
        """
        if not DEAP_AVAILABLE:
            raise ImportError("需要安裝 DEAP: pip install deap")

        self.pset = pset
        self.engine = backtest_engine
        self.data = data
        self.config = config or FitnessConfig()

        # 提取價格數據（用於訊號生成）
        if isinstance(data, pd.DataFrame):
            self.close_prices = data['close'].values
        else:
            # 假設 numpy array 最後一列是 close
            self.close_prices = data[:, -1]

    def evaluate(self, individual: 'gp.PrimitiveTree') -> Tuple[float]:
        """
        評估單一個體的適應度

        Args:
            individual: DEAP GP 個體（表達式樹）

        Returns:
            tuple: (fitness_score,) DEAP 要求的 tuple 格式
                   分數越高越好，範圍通常在 [-1, 1] 之間

        Note:
            - 無效個體返回 (-1e10,) 極低分數
            - 異常個體不拋出錯誤，避免中斷演化
        """
        try:
            # 1. 編譯個體為可執行函數
            func = gp.compile(individual, self.pset)

            # 2. 執行函數生成訊號
            signals = self._generate_signals(func)

            # 3. 檢查訊號有效性
            if not self._validate_signals(signals):
                return (INVALID_FITNESS,)

            # 4. 回測
            result = self._run_backtest(signals)

            # 5. 計算複合適應度
            fitness = self._calculate_fitness(result, individual)

            return (fitness,)

        except Exception as e:
            # 無效個體返回最低適應度（不中斷演化）
            return (INVALID_FITNESS,)

    def _generate_signals(
        self,
        func: Callable
    ) -> NDArray[np.bool_]:
        """
        從編譯函數生成交易訊號

        Args:
            func: 編譯後的 GP 函數

        Returns:
            布林訊號陣列（True = 進場，False = 出場）

        Note:
            函數接收價格序列，返回布林序列或數值序列（>0 視為 True）
        """
        # 執行 GP 函數
        raw_signals = func(self.close_prices)

        # 轉換為布林訊號
        if isinstance(raw_signals, (np.ndarray, pd.Series)):
            # 數值轉布林（> 0 為 True）
            signals = np.asarray(raw_signals, dtype=float) > 0
        else:
            # 標量擴展為陣列
            signals = np.full(len(self.close_prices), bool(raw_signals))

        return signals.astype(bool)

    def _validate_signals(self, signals: NDArray[np.bool_]) -> bool:
        """
        驗證訊號有效性

        Args:
            signals: 交易訊號陣列

        Returns:
            bool: True 如果訊號有效

        檢查項目:
            - 訊號長度與數據一致
            - 不是全 True 或全 False
            - 至少有最少交易次數
        """
        # 檢查長度
        if len(signals) != len(self.close_prices):
            return False

        # 檢查不是常數訊號
        if signals.all() or (~signals).all():
            return False

        # 檢查交易次數（訊號變化次數）
        changes = np.diff(signals.astype(int))
        num_trades = np.abs(changes).sum()

        if num_trades < self.config.min_trades:
            return False

        return True

    def _run_backtest(self, signals: NDArray[np.bool_]) -> BacktestResult:
        """
        執行回測

        Args:
            signals: 交易訊號（True = 做多，False = 空倉）

        Returns:
            BacktestResult: 回測結果

        Note:
            這裡簡化為純做多策略。
            完整版應支援做多/做空/空倉三態訊號。
        """
        # 將布林訊號轉為進場/出場訊號
        long_entry = pd.Series(False, index=range(len(signals)))
        long_exit = pd.Series(False, index=range(len(signals)))

        # 訊號變化點
        signal_changes = np.diff(signals.astype(int), prepend=0)

        # 0 -> 1 為進場，1 -> 0 為出場
        long_entry[signal_changes == 1] = True
        long_exit[signal_changes == -1] = True

        # 使用模組層級策略類別
        strategy = _GPGeneratedStrategy(long_entry, long_exit)

        # 執行回測
        result = self.engine.run(strategy, data=self.data)

        return result

    def _calculate_fitness(
        self,
        result: BacktestResult,
        individual: 'gp.PrimitiveTree'
    ) -> float:
        """
        計算複合適應度分數

        公式:
            fitness = (
                sharpe_normalized * sharpe_weight +
                return_normalized * return_weight -
                drawdown * drawdown_weight +
                win_rate * win_rate_weight +
                trade_factor * trade_count_weight -
                complexity_penalty * tree_size
            )

        Args:
            result: 回測結果
            individual: GP 個體（用於計算複雜度）

        Returns:
            float: 適應度分數（越高越好）

        Note:
            - 所有指標正規化到 [0, 1] 或 [-1, 1]
            - 超過最大回撤限制直接返回 -1.0
        """
        # 檢查硬約束
        if not self._check_hard_constraints(result):
            return MAX_DRAWDOWN_PENALTY_SCORE

        # 計算各項分數
        score = 0.0
        score += self._calculate_sharpe_score(result)
        score += self._calculate_return_score(result)
        score -= self._calculate_drawdown_penalty(result)
        score += self._calculate_win_rate_score(result)
        score += self._calculate_trade_count_score(result)
        score -= self._calculate_complexity_penalty(individual)

        return score

    def _check_hard_constraints(self, result: BacktestResult) -> bool:
        """
        檢查硬約束

        Args:
            result: 回測結果

        Returns:
            bool: True 如果通過所有硬約束
        """
        # 最大回撤限制
        if abs(result.max_drawdown) > self.config.max_drawdown_limit:
            return False

        return True

    def _calculate_sharpe_score(self, result: BacktestResult) -> float:
        """計算 Sharpe Ratio 分數"""
        sharpe_normalized = np.clip(
            result.sharpe_ratio / self.config.sharpe_max,
            -1, 1
        )
        return sharpe_normalized * self.config.sharpe_weight

    def _calculate_return_score(self, result: BacktestResult) -> float:
        """計算報酬率分數"""
        return_normalized = np.clip(
            result.total_return / self.config.return_max,
            -1, 1
        )
        return return_normalized * self.config.return_weight

    def _calculate_drawdown_penalty(self, result: BacktestResult) -> float:
        """計算回撤懲罰"""
        drawdown = np.clip(abs(result.max_drawdown), 0, 1)
        return drawdown * self.config.drawdown_weight

    def _calculate_win_rate_score(self, result: BacktestResult) -> float:
        """計算勝率分數"""
        return result.win_rate * self.config.win_rate_weight

    def _calculate_trade_count_score(self, result: BacktestResult) -> float:
        """計算交易數量分數"""
        trades = result.total_trades

        if trades < self.config.min_trades:
            trade_factor = -0.5  # 交易太少
        else:
            # 正規化到 0-1（上限 100 筆為滿分）
            trade_factor = min(1.0, trades / 100)

        return trade_factor * self.config.trade_count_weight

    def _calculate_complexity_penalty(self, individual: 'gp.PrimitiveTree') -> float:
        """計算複雜度懲罰"""
        from .constraints import calculate_complexity_penalty
        return calculate_complexity_penalty(
            individual,
            alpha=self.config.complexity_penalty,
            depth_weight=0.5
        )


# ============================================================================
# DEAP 適應度類型建立
# ============================================================================

def create_fitness_type():
    """
    建立 DEAP 適應度類型（最大化單目標）

    建立兩個類別:
        - FitnessMax: 適應度類別（weights=(1.0,) 表示最大化）
        - Individual: GP 個體類別（繼承 PrimitiveTree）

    Note:
        只需在程式開始時呼叫一次。
        重複呼叫會報錯（DEAP 限制）。

    使用範例:
        create_fitness_type()

        # 之後可使用
        toolbox.register("individual", tools.initIterate, creator.Individual, ...)
    """
    if not DEAP_AVAILABLE:
        raise ImportError("需要安裝 DEAP: pip install deap")

    # 檢查是否已建立（避免重複）
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def create_multi_objective_fitness(objectives: Tuple[float, ...]):
    """
    建立多目標適應度類型

    Args:
        objectives: 目標權重 tuple
                   例如 (1.0, -1.0) 表示最大化第一個、最小化第二個

    使用範例:
        # 最大化 Sharpe，最小化回撤
        create_multi_objective_fitness((1.0, -1.0))

    Note:
        多目標優化需要使用 NSGA-II 等演算法。
        單目標請使用 create_fitness_type()。
    """
    if not DEAP_AVAILABLE:
        raise ImportError("需要安裝 DEAP: pip install deap")

    # 檢查是否已建立
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=objectives)

    if not hasattr(creator, "IndividualMulti"):
        creator.create(
            "IndividualMulti",
            gp.PrimitiveTree,
            fitness=creator.FitnessMulti
        )
