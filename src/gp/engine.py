"""
GP 演化引擎

使用 DEAP 框架實現基因編程演化。

使用範例:
    from src.gp.primitives import PrimitiveSetFactory
    from src.gp.fitness import FitnessEvaluator, create_fitness_type
    from src.gp.engine import GPEngine, EvolutionConfig

    # 建立 DEAP fitness types
    create_fitness_type()

    # 建立原語集
    factory = PrimitiveSetFactory()
    pset = factory.create_standard_set()

    # 建立適應度評估器
    evaluator = FitnessEvaluator(pset, engine, data)

    # 建立 GP 引擎
    config = EvolutionConfig(population_size=100, generations=50)
    gp_engine = GPEngine(pset, evaluator.evaluate, config)

    # 執行演化
    result = gp_engine.evolve()

    # 取得最佳策略
    best = result.best_individual
    print(f"Best fitness: {result.best_fitness:.4f}")
"""

from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict, Any, Tuple
import logging
import time
import numpy as np

try:
    from deap import base, creator, tools, gp, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    base = creator = tools = gp = algorithms = None

logger = logging.getLogger(__name__)


# ============================================================================
# 演化配置
# ============================================================================

@dataclass
class EvolutionConfig:
    """演化配置

    定義 GP 演化的所有超參數。

    Attributes:
        population_size: 種群大小（預設 100）
        generations: 演化代數（預設 50）
        tournament_size: 錦標賽選擇大小（預設 3）
        crossover_prob: 交叉機率（預設 0.7）
        mutation_prob: 突變機率（預設 0.2）
        mutate_uniform_prob: Uniform 突變機率（預設 0.1）
        mutate_shrink_prob: Shrink 突變機率（預設 0.05）
        mutate_replace_prob: Node Replacement 突變機率（預設 0.05）
        early_stopping_generations: 早停代數（預設 10）
        min_improvement: 最小改善閾值（預設 0.01）
        elitism: 精英保留數量（預設 5）
        hof_size: Hall of Fame 大小（預設 10）
        n_workers: 並行 worker 數量（1 = 單執行緒，預設 1）
        seed: 隨機種子（預設 None）
    """

    # 種群參數
    population_size: int = 100
    generations: int = 50

    # 選擇
    tournament_size: int = 3

    # 遺傳運算
    crossover_prob: float = 0.7
    mutation_prob: float = 0.2

    # 突變參數（三種突變類型的相對機率）
    mutate_uniform_prob: float = 0.1
    mutate_shrink_prob: float = 0.05
    mutate_replace_prob: float = 0.05

    # 早停
    early_stopping_generations: int = 10  # 連續 N 代無改善則停止
    min_improvement: float = 0.01  # 最小改善閾值

    # 精英保留
    elitism: int = 5

    # Hall of Fame
    hof_size: int = 10

    # 並行
    n_workers: int = 1  # 1 = 單執行緒

    # 隨機種子
    seed: Optional[int] = None

    def __post_init__(self):
        """驗證配置"""
        if self.population_size < self.elitism:
            raise ValueError(
                f"population_size ({self.population_size}) 必須 >= elitism ({self.elitism})"
            )

        if self.crossover_prob + self.mutation_prob > 1.0:
            raise ValueError(
                f"crossover_prob + mutation_prob 不能超過 1.0 "
                f"(當前: {self.crossover_prob + self.mutation_prob:.2f})"
            )


@dataclass
class EvolutionResult:
    """演化結果

    包含演化過程的完整資訊。

    Attributes:
        best_individual: 最佳個體（gp.PrimitiveTree）
        best_fitness: 最佳適應度
        generations_run: 實際執行代數
        stopped_early: 是否早停
        fitness_history: 每代最佳適應度歷史
        avg_fitness_history: 每代平均適應度歷史
        population_size_history: 每代種群大小歷史
        hall_of_fame: Hall of Fame（前 N 個最佳個體）
        elapsed_time: 執行時間（秒）
        config: 演化配置
        metadata: 額外元資料
    """

    best_individual: Any  # gp.PrimitiveTree
    best_fitness: float
    generations_run: int
    stopped_early: bool

    # 統計
    fitness_history: List[float]  # 每代最佳適應度
    avg_fitness_history: List[float]  # 每代平均適應度
    population_size_history: List[int]  # 每代種群大小

    # Hall of Fame
    hall_of_fame: List[Any]

    # 執行時間
    elapsed_time: float

    # 元資料
    config: EvolutionConfig
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# GP 演化引擎
# ============================================================================

class GPEngine:
    """
    GP 演化引擎

    使用 DEAP 框架實現完整的基因編程演化流程。

    主要功能:
        1. 種群初始化（Half-and-Half）
        2. 選擇（Tournament Selection）
        3. 交叉（One-Point Crossover）
        4. 突變（Uniform / Shrink / Node Replacement）
        5. 精英保留（Elitism）
        6. 早停機制（Early Stopping）
        7. 並行化支援（Multiprocessing）

    使用範例:
        from src.gp.primitives import PrimitiveSetFactory
        from src.gp.fitness import FitnessEvaluator

        # 建立原語集
        factory = PrimitiveSetFactory()
        pset = factory.create_standard_set()

        # 建立適應度評估器
        evaluator = FitnessEvaluator(pset, engine, data)

        # 建立 GP 引擎
        gp_engine = GPEngine(pset, evaluator.evaluate)

        # 執行演化
        result = gp_engine.evolve()

        # 取得最佳策略
        best = result.best_individual
    """

    def __init__(
        self,
        pset: 'gp.PrimitiveSetTyped',
        evaluate_func: Callable,
        config: Optional[EvolutionConfig] = None
    ):
        """
        初始化 GP 引擎

        Args:
            pset: DEAP 原語集
            evaluate_func: 適應度評估函數 (individual) -> (fitness,)
            config: 演化配置（使用預設值如果未提供）

        Raises:
            ImportError: 如果 DEAP 未安裝
        """
        if not DEAP_AVAILABLE:
            raise ImportError("需要安裝 DEAP: pip install deap")

        self.pset = pset
        self.evaluate_func = evaluate_func
        self.config = config or EvolutionConfig()

        # 設定隨機種子
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        # 建立 DEAP toolbox
        self.toolbox = self._create_toolbox()

        logger.info(
            f"GPEngine 初始化完成 "
            f"(pop={self.config.population_size}, gen={self.config.generations})"
        )

    def _create_toolbox(self) -> 'base.Toolbox':
        """建立 DEAP toolbox

        註冊所有 GP 運算子：
            - 個體初始化（Half-and-Half）
            - 種群初始化
            - 編譯
            - 適應度評估
            - 選擇（Tournament）
            - 交叉（One-Point）
            - 突變（Uniform / Shrink / Node Replacement）

        Returns:
            DEAP toolbox
        """
        toolbox = base.Toolbox()

        # 個體初始化（Half-and-Half）
        # Half-and-Half = 50% Full + 50% Grow，增加初始多樣性
        toolbox.register(
            "expr",
            gp.genHalfAndHalf,
            pset=self.pset,
            min_=2,
            max_=6
        )
        toolbox.register(
            "individual",
            tools.initIterate,
            creator.Individual,
            toolbox.expr
        )
        toolbox.register(
            "population",
            tools.initRepeat,
            list,
            toolbox.individual
        )

        # 編譯
        toolbox.register("compile", gp.compile, pset=self.pset)

        # 適應度評估
        toolbox.register("evaluate", self.evaluate_func)

        # 選擇（Tournament Selection）
        # Tournament 選擇：隨機選 N 個個體，取最佳
        # 好處：保持多樣性，避免過早收斂
        toolbox.register(
            "select",
            tools.selTournament,
            tournsize=self.config.tournament_size
        )

        # 交叉（One-Point Crossover）
        toolbox.register("mate", gp.cxOnePoint)

        # 突變（Uniform Mutation）
        toolbox.register(
            "mutate",
            gp.mutUniform,
            expr=toolbox.expr,
            pset=self.pset
        )

        # 突變（Shrink Mutation）- 簡化個體
        toolbox.register("mutate_shrink", gp.mutShrink)

        # 突變（Node Replacement）- 替換節點
        toolbox.register(
            "mutate_replace",
            gp.mutNodeReplacement,
            pset=self.pset
        )

        # 應用約束（深度、節點數量限制）
        from .constraints import apply_constraints, ConstraintConfig
        constraint_config = ConstraintConfig()
        apply_constraints(toolbox, constraint_config)

        return toolbox

    def evolve(self) -> EvolutionResult:
        """
        執行演化

        演化流程:
            1. 初始化種群
            2. 評估適應度
            3. 選擇
            4. 交叉
            5. 突變
            6. 精英保留
            7. 更新 Hall of Fame
            8. 檢查早停
            9. 重複步驟 3-8

        Returns:
            EvolutionResult: 演化結果
        """
        start_time = time.time()

        cfg = self.config

        # 初始化種群
        logger.info(f"初始化種群 (size={cfg.population_size})")
        population = self.toolbox.population(n=cfg.population_size)

        # Hall of Fame（保存最佳個體）
        hof = tools.HallOfFame(cfg.hof_size)

        # 統計
        stats = self._create_stats()

        # 記錄
        fitness_history = []
        avg_fitness_history = []
        population_size_history = []

        # 早停追蹤
        best_fitness = float('-inf')
        no_improvement_count = 0
        stopped_early = False

        # 評估初始種群
        logger.info("評估初始種群")
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # 更新 HOF
        hof.update(population)

        # 記錄初始統計
        record = stats.compile(population)
        fitness_history.append(record['max'])
        avg_fitness_history.append(record['avg'])
        population_size_history.append(len(population))

        logger.info(
            f"Gen 0: max={record['max']:.4f}, avg={record['avg']:.4f}, "
            f"std={record['std']:.4f}, min={record['min']:.4f}"
        )

        # 演化迴圈
        for gen in range(1, cfg.generations + 1):
            logger.debug(f"開始 Gen {gen}")

            # 1. 選擇（保留空間給精英）
            offspring = self.toolbox.select(population, len(population) - cfg.elitism)
            offspring = list(map(self.toolbox.clone, offspring))

            # 2. 交叉
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < cfg.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # 3. 突變（多種突變類型）
            for mutant in offspring:
                if np.random.random() < cfg.mutation_prob:
                    # 隨機選擇突變類型
                    r = np.random.random()
                    total_prob = (
                        cfg.mutate_uniform_prob +
                        cfg.mutate_shrink_prob +
                        cfg.mutate_replace_prob
                    )

                    # 正規化機率
                    uniform_threshold = cfg.mutate_uniform_prob / total_prob
                    shrink_threshold = uniform_threshold + cfg.mutate_shrink_prob / total_prob

                    if r < uniform_threshold:
                        # Uniform Mutation（替換子樹）
                        self.toolbox.mutate(mutant)
                    elif r < shrink_threshold:
                        # Shrink Mutation（簡化）
                        self.toolbox.mutate_shrink(mutant)
                    else:
                        # Node Replacement（替換節點）
                        self.toolbox.mutate_replace(mutant)

                    del mutant.fitness.values

            # 4. 評估（只評估無效個體）
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            logger.debug(f"評估 {len(invalid_ind)} 個無效個體")
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # 5. 精英保留
            elites = tools.selBest(population, cfg.elitism)
            population[:] = offspring + elites

            # 6. 更新 HOF
            hof.update(population)

            # 7. 記錄統計
            record = stats.compile(population)
            fitness_history.append(record['max'])
            avg_fitness_history.append(record['avg'])
            population_size_history.append(len(population))

            logger.info(
                f"Gen {gen}: max={record['max']:.4f}, avg={record['avg']:.4f}, "
                f"std={record['std']:.4f}, min={record['min']:.4f}"
            )

            # 8. 早停檢查
            current_best = record['max']
            if current_best > best_fitness + cfg.min_improvement:
                best_fitness = current_best
                no_improvement_count = 0
                logger.debug(f"適應度改善: {current_best:.4f}")
            else:
                no_improvement_count += 1
                logger.debug(f"無改善計數: {no_improvement_count}")

            if no_improvement_count >= cfg.early_stopping_generations:
                logger.info(
                    f"早停：連續 {no_improvement_count} 代無改善 "
                    f"(閾值 {cfg.min_improvement})"
                )
                stopped_early = True
                break

        elapsed_time = time.time() - start_time

        logger.info(
            f"演化完成 (generations={gen}, elapsed={elapsed_time:.2f}s, "
            f"best_fitness={hof[0].fitness.values[0]:.4f})"
        )

        return EvolutionResult(
            best_individual=hof[0],
            best_fitness=hof[0].fitness.values[0],
            generations_run=gen,
            stopped_early=stopped_early,
            fitness_history=fitness_history,
            avg_fitness_history=avg_fitness_history,
            population_size_history=population_size_history,
            hall_of_fame=list(hof),
            elapsed_time=elapsed_time,
            config=cfg
        )

    def evolve_parallel(self) -> EvolutionResult:
        """
        並行演化（使用 multiprocessing）

        使用多進程池加速適應度評估。

        Returns:
            EvolutionResult: 演化結果

        Note:
            - 只有在 n_workers > 1 時才啟用並行
            - 適應度評估必須是可序列化的（pickle-able）
            - Windows 需要在 if __name__ == '__main__' 中呼叫

        Warning:
            並行化有額外開銷，只在適應度評估耗時時才有效。
            如果單次評估 < 0.1s，單執行緒可能更快。
        """
        import multiprocessing

        n_workers = self.config.n_workers
        if n_workers <= 1:
            logger.info("n_workers=1，使用單執行緒演化")
            return self.evolve()

        logger.info(f"啟動並行演化 (workers={n_workers})")

        # 建立進程池
        pool = multiprocessing.Pool(processes=n_workers)
        self.toolbox.register("map", pool.map)

        try:
            result = self.evolve()
        finally:
            pool.close()
            pool.join()
            logger.info("進程池已關閉")

        return result

    def _create_stats(self) -> 'tools.Statistics':
        """建立統計物件

        統計項目:
            - avg: 平均適應度
            - std: 標準差
            - min: 最小適應度
            - max: 最大適應度

        Returns:
            DEAP Statistics 物件
        """
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        return stats
