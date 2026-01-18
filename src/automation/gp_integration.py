"""
GP 整合模組

整合 GPLoop 到 UltimateLoop 的資料契約和核心元件。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from ..strategies.base import BaseStrategy


# ============================================================================
# 資料契約
# ============================================================================

@dataclass
class GPExplorationRequest:
    """GP 探索請求

    定義 GP 演化所需的參數和配置。

    Attributes:
        symbol: 交易標的 (e.g., 'BTCUSDT', 'ETHUSDT')
        timeframe: 時間週期 (e.g., '4h', '1d')
        population_size: 種群大小（每代個體數量）
        generations: 演化代數（總共運行多少代）
        top_n: 返回最佳策略數量（取前 N 個最優策略）
        fitness_weights: 適應度權重 (sharpe, return, drawdown)
            - sharpe: Sharpe Ratio 權重（預設 1.0）
            - return: 總收益率權重（預設 0.5）
            - drawdown: 最大回撤權重（預設 -0.3，負數表示懲罰）

    Example:
        >>> request = GPExplorationRequest(
        ...     symbol='BTCUSDT',
        ...     timeframe='4h',
        ...     population_size=100,
        ...     generations=50
        ... )
    """
    symbol: str
    timeframe: str = '4h'
    population_size: int = 50
    generations: int = 30
    top_n: int = 3
    fitness_weights: Tuple[float, float, float] = (1.0, 0.5, -0.3)


@dataclass
class DynamicStrategyInfo:
    """動態策略資訊

    儲存 GP 演化生成的策略詳細資訊。

    Attributes:
        name: 策略名稱 (e.g., 'gp_evolved_001', 'gp_gen_05_rank_1')
        strategy_class: 策略類別（可實例化的 BaseStrategy 子類）
        expression: 原始表達式（GP 演化的邏輯樹字串表示）
        fitness: 適應度分數（越高越好）
        generation: 產生的代數（0-based，如 generation=5 表示第 6 代）
        created_at: 建立時間（UTC）
        metadata: 額外元資料
            - parent_ids: 父代 ID 列表（用於追溯演化路徑）
            - mutation_type: 變異類型（crossover/mutation/reproduction）
            - backtest_stats: 回測統計（Sharpe、Return、MaxDD 等）

    Example:
        >>> info = DynamicStrategyInfo(
        ...     name='gp_evolved_001',
        ...     strategy_class=GPEvolvedStrategy001,
        ...     expression='and(gt(rsi(14), 50), lt(rsi(14), 70))',
        ...     fitness=2.35,
        ...     generation=10,
        ...     created_at=datetime.utcnow(),
        ...     metadata={
        ...         'parent_ids': ['gp_gen_09_005', 'gp_gen_09_012'],
        ...         'mutation_type': 'crossover',
        ...         'backtest_stats': {'sharpe': 2.35, 'return': 125.3}
        ...     }
        ... )
    """
    name: str
    strategy_class: Type['BaseStrategy']
    expression: str
    fitness: float
    generation: int
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GPExplorationResult:
    """GP 探索結果

    封裝 GP 演化執行的完整結果。

    Attributes:
        success: 是否成功（True=成功完成，False=執行失敗）
        strategies: 生成的策略列表（按適應度排序，從高到低）
        evolution_stats: 演化統計
            - best_fitness_per_gen: 每代最佳適應度列表
            - avg_fitness_per_gen: 每代平均適應度列表
            - diversity_per_gen: 每代多樣性指標（表達式唯一性）
            - total_evaluations: 總評估次數
        elapsed_time: 執行時間（秒）
        error: 錯誤訊息（如果 success=False）

    Example:
        >>> result = GPExplorationResult(
        ...     success=True,
        ...     strategies=[strategy1, strategy2, strategy3],
        ...     evolution_stats={
        ...         'best_fitness_per_gen': [1.2, 1.5, 1.8, 2.1, 2.3],
        ...         'avg_fitness_per_gen': [0.8, 1.0, 1.2, 1.4, 1.5],
        ...         'diversity_per_gen': [0.95, 0.90, 0.85, 0.80, 0.75],
        ...         'total_evaluations': 250
        ...     },
        ...     elapsed_time=125.5,
        ...     error=None
        ... )
    """
    success: bool
    strategies: List[DynamicStrategyInfo]
    evolution_stats: Dict[str, Any]
    elapsed_time: float
    error: Optional[str] = None


# ============================================================================
# GPStrategyAdapter - 將 GP 個體轉換為策略類別
# ============================================================================

class GPStrategyAdapter:
    """
    GP 個體轉策略適配器

    將 GP 演化的表達式樹動態轉換為可註冊的策略類別。

    使用範例:
        >>> from src.gp.converter import ExpressionConverter
        >>> from src.gp.primitives import PrimitiveSetFactory
        >>>
        >>> # 準備轉換器
        >>> factory = PrimitiveSetFactory()
        >>> pset = factory.create_standard_set()
        >>> converter = ExpressionConverter(pset)
        >>>
        >>> # 建立適配器
        >>> adapter = GPStrategyAdapter(converter)
        >>>
        >>> # 轉換 GP 個體為策略類別
        >>> strategy_class = adapter.create_strategy_class(
        ...     individual=best_individual,
        ...     strategy_name='gp_evolved_001',
        ...     fitness=2.35,
        ...     generation=10
        ... )
        >>>
        >>> # 實例化策略
        >>> strategy = strategy_class()
        >>> signals = strategy.generate_signals(data)
    """

    def __init__(self, converter):
        """
        初始化適配器

        Args:
            converter: ExpressionConverter 實例（來自 src.gp.converter）
        """
        self.converter = converter

    def create_strategy_class(
        self,
        individual,
        strategy_name: str,
        fitness: float,
        generation: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Type['BaseStrategy']:
        """
        動態建立策略類別

        Args:
            individual: DEAP GP 個體（表達式樹）
            strategy_name: 策略名稱（e.g., 'gp_evolved_001'）
            fitness: 適應度分數
            generation: 演化代數
            metadata: 額外元資料（可選）

        Returns:
            Type[BaseStrategy]: 可實例化的策略類別

        Raises:
            RuntimeError: 如果轉換失敗

        Example:
            >>> strategy_class = adapter.create_strategy_class(
            ...     individual=best,
            ...     strategy_name='gp_evolved_rsi_001',
            ...     fitness=1.85,
            ...     generation=5
            ... )
            >>> isinstance(strategy_class, type)
            True
            >>> issubclass(strategy_class, BaseStrategy)
            True
        """
        # 延遲導入（避免循環依賴）
        from ..strategies.gp.evolved_strategy import EvolvedStrategy

        # 1. 編譯表達式為可呼叫函數
        try:
            signal_func = self.converter.compile(individual)
        except Exception as e:
            raise RuntimeError(
                f"Failed to compile GP expression: {e}"
            ) from e

        # 2. 取得表達式字串（用於顯示和除錯）
        try:
            expression_str = self.converter.to_python(individual)
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert GP expression to Python: {e}"
            ) from e

        # 3. 取得當前時間
        evolved_at = datetime.utcnow().isoformat() + 'Z'

        # 4. 建立類別名稱（CamelCase）
        class_name = self._to_class_name(strategy_name)

        # 5. 動態建立策略類別
        # 使用 type() 建立新類別，繼承 EvolvedStrategy
        strategy_class = type(
            class_name,
            (EvolvedStrategy,),
            {
                # 類別屬性
                'name': strategy_name,
                'version': '1.0',
                'description': f'GP evolved strategy with fitness {fitness:.4f}',

                # 演化元資料
                'expression': expression_str,
                'fitness_score': fitness,
                'generation': generation,
                'evolved_at': evolved_at,

                # 自定義 __init__
                '__init__': self._make_init(signal_func),

                # 參數（GP 策略無參數，表達式已固定）
                'params': {},
                'param_space': {},
            }
        )

        return strategy_class

    def _make_init(self, signal_func):
        """
        建立 __init__ 方法（閉包捕獲 signal_func）

        Args:
            signal_func: 編譯後的訊號函數

        Returns:
            Callable: __init__ 方法
        """
        def __init__(self, **kwargs):
            """初始化演化策略"""
            # 呼叫父類別 __init__
            from ..strategies.gp.evolved_strategy import EvolvedStrategy
            EvolvedStrategy.__init__(self, signal_func=signal_func, **kwargs)

        return __init__

    def _to_class_name(self, strategy_name: str) -> str:
        """
        將策略名稱轉為類別名稱（CamelCase）

        Args:
            strategy_name: 策略名稱（e.g., 'gp_evolved_rsi_001'）

        Returns:
            str: 類別名稱（e.g., 'GpEvolvedRsi001'）

        Example:
            >>> adapter._to_class_name('gp_evolved_001')
            'GpEvolved001'
            >>> adapter._to_class_name('rsi_ma_cross')
            'RsiMaCross'
        """
        parts = strategy_name.split('_')
        class_name = ''.join(word.capitalize() for word in parts)
        return class_name


# ============================================================================
# GPExplorer - 封裝 GPLoop 為可重用元件
# ============================================================================

class GPExplorer:
    """
    GP 探索器

    封裝 GPLoop 的調用邏輯，提供簡化的介面給 UltimateLoop 使用。

    使用範例:
        >>> from src.automation.gp_integration import (
        ...     GPExplorer,
        ...     GPExplorationRequest,
        ... )
        >>>
        >>> # 建立探索器
        >>> explorer = GPExplorer()
        >>>
        >>> # 建立請求
        >>> request = GPExplorationRequest(
        ...     symbol='BTCUSDT',
        ...     timeframe='4h',
        ...     population_size=50,
        ...     generations=30,
        ...     top_n=3
        ... )
        >>>
        >>> # 執行探索（需要提供資料）
        >>> result = explorer.explore(request, data=ohlcv_df)
        >>>
        >>> # 檢查結果
        >>> if result.success:
        ...     for strategy_info in result.strategies:
        ...         print(f"Found: {strategy_info.name}")
        ...         print(f"  Fitness: {strategy_info.fitness:.4f}")
        ...         print(f"  Expression: {strategy_info.expression}")
        ... else:
        ...     print(f"Exploration failed: {result.error}")

    實作細節:
        1. 接收 GPExplorationRequest（定義演化參數）
        2. 使用 GPLoop 執行 GP 演化
        3. 使用 GPStrategyAdapter 轉換最佳個體為策略類別
        4. 返回 GPExplorationResult（包含策略列表）
    """

    def __init__(
        self,
        converter: Optional[Any] = None,
        timeout: Optional[float] = None
    ):
        """
        初始化 GP 探索器

        Args:
            converter: ExpressionConverter 實例（可選，自動建立）
            timeout: 執行超時（秒，可選）
        """
        self.converter = converter
        self.timeout = timeout

    def explore(
        self,
        request: GPExplorationRequest,
        data: Any
    ) -> GPExplorationResult:
        """
        執行 GP 探索

        Args:
            request: GP 探索請求（定義演化參數）
            data: OHLCV DataFrame（市場資料）

        Returns:
            GPExplorationResult: 探索結果
                - success=True: 成功，strategies 包含演化策略
                - success=False: 失敗，error 包含錯誤訊息

        Raises:
            不會拋出異常，所有錯誤都封裝在 GPExplorationResult 中

        Example:
            >>> request = GPExplorationRequest(
            ...     symbol='BTCUSDT',
            ...     population_size=100,
            ...     generations=50,
            ...     top_n=5
            ... )
            >>> result = explorer.explore(request, data=ohlcv_df)
            >>> assert result.success
            >>> assert len(result.strategies) <= 5
        """
        import logging
        import time

        logger = logging.getLogger(__name__)

        start_time = time.time()

        try:
            # 延遲導入（避免循環依賴）
            from ..automation.gp_loop import GPLoop, GPLoopConfig
            from ..gp.primitives import PrimitiveSetFactory
            from ..gp.converter import ExpressionConverter

            # 1. 建立 GPLoop 配置
            loop_config = GPLoopConfig(
                symbol=request.symbol,
                timeframe=request.timeframe,
                population_size=request.population_size,
                generations=request.generations,
                generate_top_n=request.top_n,
                record_to_learning=False,  # 由 UltimateLoop 統一管理學習記錄
            )

            logger.info(
                f"開始 GP 探索: {request.symbol} {request.timeframe} "
                f"(pop={request.population_size}, gen={request.generations})"
            )

            # 2. 執行 GP 演化
            with GPLoop(loop_config) as loop:
                # 注入外部資料（避免重複下載）
                loop._data = data
                loop._validate_data(data)

                # 執行演化
                evolution_result = loop.run()

                # 3. 建立表達式轉換器（如果尚未提供）
                if self.converter is None:
                    factory = PrimitiveSetFactory()
                    pset = factory.create_standard_set()
                    self.converter = ExpressionConverter(pset)

                # 4. 建立策略適配器
                adapter = GPStrategyAdapter(self.converter)

                # 5. 轉換最佳個體為策略類別
                strategies = []
                hof = evolution_result.hall_of_fame[:request.top_n]

                for rank, individual in enumerate(hof, start=1):
                    # 策略名稱
                    strategy_name = (
                        f"gp_gen_{evolution_result.generations_run:02d}_"
                        f"rank_{rank}"
                    )

                    # 適應度
                    fitness = individual.fitness.values[0]

                    # 元資料
                    metadata = {
                        'symbol': request.symbol,
                        'timeframe': request.timeframe,
                        'rank': rank,
                        'population_size': request.population_size,
                        'generations': request.generations,
                    }

                    # 動態建立策略類別
                    try:
                        strategy_class = adapter.create_strategy_class(
                            individual=individual,
                            strategy_name=strategy_name,
                            fitness=fitness,
                            generation=evolution_result.generations_run,
                            metadata=metadata
                        )

                        # 取得表達式字串
                        expression_str = self.converter.to_python(individual)

                        # 建立策略資訊
                        strategy_info = DynamicStrategyInfo(
                            name=strategy_name,
                            strategy_class=strategy_class,
                            expression=expression_str,
                            fitness=fitness,
                            generation=evolution_result.generations_run,
                            created_at=datetime.utcnow(),
                            metadata=metadata
                        )

                        strategies.append(strategy_info)

                        logger.info(
                            f"轉換策略 {rank}/{len(hof)}: "
                            f"{strategy_name} (fitness={fitness:.4f})"
                        )

                    except Exception as e:
                        logger.error(
                            f"轉換策略 {strategy_name} 失敗: {e}",
                            exc_info=True
                        )
                        # 繼續處理下一個

                # 6. 建立演化統計
                evolution_stats = {
                    'best_fitness_per_gen': evolution_result.fitness_history,
                    'avg_fitness_per_gen': evolution_result.avg_fitness_history,
                    'diversity_per_gen': self._calculate_diversity(
                        evolution_result.fitness_history,
                        evolution_result.avg_fitness_history
                    ),
                    'total_evaluations': (
                        request.population_size * evolution_result.generations_run
                    ),
                    'stopped_early': evolution_result.stopped_early,
                }

                elapsed_time = time.time() - start_time

                logger.info(
                    f"GP 探索完成: 找到 {len(strategies)} 個策略 "
                    f"(耗時 {elapsed_time:.2f}s)"
                )

                # 7. 返回成功結果
                return GPExplorationResult(
                    success=True,
                    strategies=strategies,
                    evolution_stats=evolution_stats,
                    elapsed_time=elapsed_time,
                    error=None
                )

        except Exception as e:
            # 捕捉所有異常，返回失敗結果
            elapsed_time = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"

            logger.error(
                f"GP 探索失敗: {error_msg}",
                exc_info=True
            )

            return GPExplorationResult(
                success=False,
                strategies=[],
                evolution_stats={},
                elapsed_time=elapsed_time,
                error=error_msg
            )

    def _calculate_diversity(
        self,
        best_fitness: List[float],
        avg_fitness: List[float]
    ) -> List[float]:
        """
        計算每代的多樣性指標（簡化版）

        使用 best_fitness 與 avg_fitness 的差距作為多樣性代理指標。
        差距越大 → 多樣性越高（適應度分佈分散）
        差距越小 → 多樣性越低（趨同）

        Args:
            best_fitness: 每代最佳適應度列表
            avg_fitness: 每代平均適應度列表

        Returns:
            List[float]: 每代多樣性分數（0.0-1.0，歸一化）

        Note:
            這是一個簡化的多樣性指標。
            更精確的計算需要訪問整個種群的表達式樹。
        """
        if not best_fitness or not avg_fitness:
            return []

        # 計算差距
        diversity_raw = [
            best - avg
            for best, avg in zip(best_fitness, avg_fitness)
        ]

        # 歸一化到 [0, 1]
        max_diversity = max(diversity_raw) if diversity_raw else 1.0
        if max_diversity == 0:
            return [0.5] * len(diversity_raw)  # 無差異時返回中性值

        diversity_normalized = [
            d / max_diversity
            for d in diversity_raw
        ]

        return diversity_normalized
