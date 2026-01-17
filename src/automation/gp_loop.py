"""
GP 演化循環

整合 GP 演化到現有的 BacktestLoop 系統。

使用範例:
    config = GPLoopConfig(
        symbol='ETHUSDT',
        population_size=100,
        generations=50
    )

    with GPLoop(config) as loop:
        result = loop.run()

        # 取得最佳策略
        best_strategy = result.best_strategy

        # 生成策略檔案
        loop.generate_strategies()
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging
import time
from datetime import datetime

from ..gp.engine import GPEngine, EvolutionConfig, EvolutionResult
from ..gp.primitives import PrimitiveSetFactory
from ..gp.fitness import FitnessEvaluator, FitnessConfig, create_fitness_type
from ..gp.converter import ExpressionConverter, StrategyGenerator
from ..strategies.registry import StrategyRegistry
from ..backtester.engine import BacktestEngine, BacktestConfig
from ..data import DataFetcher

logger = logging.getLogger(__name__)


# ============================================================================
# 配置
# ============================================================================

@dataclass
class GPLoopConfig:
    """GP 演化循環配置

    Attributes:
        symbol: 交易標的（預設 'BTCUSDT'）
        timeframe: 時間框架（預設 '4h'）
        population_size: 種群大小（預設 50）
        generations: 演化代數（預設 30）
        early_stopping: 早停代數（預設 10）
        primitive_set: 原語集類型（預設 'standard'）
        generate_top_n: 生成前 N 個最佳策略（預設 5）
        output_dir: 輸出目錄（預設 None，自動生成）
        record_to_learning: 是否記錄到學習系統（預設 True）
        fitness_config: 適應度配置（預設 None，使用預設值）
        initial_capital: 初始資金（預設 10000.0）
        leverage: 槓桿倍數（預設 10.0）
        maker_fee: Maker 手續費（預設 0.0002）
        taker_fee: Taker 手續費（預設 0.0004）
        min_data_points: 最少資料點數（預設 100）
    """

    # 資料設定
    symbol: str = 'BTCUSDT'
    timeframe: str = '4h'

    # 演化參數
    population_size: int = 50
    generations: int = 30
    early_stopping: int = 10

    # 原語集
    primitive_set: str = 'standard'  # 'standard', 'minimal', 'custom'

    # 策略生成
    generate_top_n: int = 5  # 生成前 N 個最佳策略
    output_dir: Optional[Path] = None

    # 學習整合
    record_to_learning: bool = True

    # 適應度配置
    fitness_config: Optional[FitnessConfig] = None

    # 回測配置
    initial_capital: float = 10000.0
    leverage: float = 10.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0004

    # 資料驗證
    min_data_points: int = 100


# ============================================================================
# GP 演化循環
# ============================================================================

class GPLoop:
    """
    GP 演化循環

    使用範例:
        config = GPLoopConfig(
            symbol='ETHUSDT',
            population_size=100,
            generations=50
        )

        with GPLoop(config) as loop:
            result = loop.run()

            # 取得最佳策略
            best_strategy = result.best_strategy

            # 生成策略檔案
            loop.generate_strategies()
    """

    def __init__(self, config: GPLoopConfig):
        """
        初始化 GP 演化循環

        Args:
            config: GPLoopConfig 配置物件
        """
        self.config = config

        # 內部元件（延遲初始化）
        self._engine: Optional[GPEngine] = None
        self._pset: Optional[Any] = None  # PrimitiveSetTyped
        self._evaluator: Optional[FitnessEvaluator] = None
        self._backtest_engine: Optional[BacktestEngine] = None
        self._data_fetcher: Optional[DataFetcher] = None
        self._converter: Optional[ExpressionConverter] = None

        # 執行結果
        self._result: Optional[EvolutionResult] = None
        self._data: Optional[Any] = None  # OHLCV DataFrame

    def __enter__(self) -> 'GPLoop':
        """Context Manager 入口"""
        self._setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager 出口"""
        self._cleanup()
        return False  # 不抑制異常

    def _setup(self):
        """初始化 GP 引擎"""
        logger.info("初始化 GPLoop 元件...")

        start_time = time.time()

        # 1. 建立適應度類型（全域建立一次）
        create_fitness_type()

        # 2. 建立原語集
        logger.info(f"建立原語集: {self.config.primitive_set}")
        factory = PrimitiveSetFactory()

        if self.config.primitive_set == 'standard':
            self._pset = factory.create_standard_set()
        elif self.config.primitive_set == 'minimal':
            self._pset = factory.create_minimal_set()
        else:
            raise ValueError(f"Unknown primitive set: {self.config.primitive_set}")

        # 3. 載入市場資料
        logger.info(f"載入市場資料: {self.config.symbol} {self.config.timeframe}")
        self._data_fetcher = DataFetcher()
        self._data = self._data_fetcher.fetch_ohlcv(
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            limit=5000
        )

        # 驗證資料
        self._validate_data(self._data)

        logger.info(f"載入 {len(self._data)} 筆資料")

        # 4. 建立回測引擎
        backtest_config = BacktestConfig(
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            start_date=self._data.index[0].to_pydatetime(),
            end_date=self._data.index[-1].to_pydatetime(),
            initial_capital=self.config.initial_capital,
            leverage=self.config.leverage,
            maker_fee=self.config.maker_fee,
            taker_fee=self.config.taker_fee,
            use_polars=True,
        )

        self._backtest_engine = BacktestEngine(backtest_config)
        self._backtest_engine.load_data(self._data)

        # 5. 建立適應度評估器
        fitness_config = self.config.fitness_config or FitnessConfig()

        self._evaluator = FitnessEvaluator(
            pset=self._pset,
            backtest_engine=self._backtest_engine,
            data=self._data,
            config=fitness_config
        )

        # 6. 建立 GP 引擎
        evolution_config = EvolutionConfig(
            population_size=self.config.population_size,
            generations=self.config.generations,
            early_stopping_generations=self.config.early_stopping
        )

        self._engine = GPEngine(
            pset=self._pset,
            evaluate_func=self._evaluator.evaluate,
            config=evolution_config
        )

        # 7. 建立表達式轉換器
        self._converter = ExpressionConverter(self._pset)

        elapsed = time.time() - start_time
        logger.info(f"GPLoop 初始化完成（耗時 {elapsed:.2f}s）")

    def _validate_data(self, data: Any):
        """
        驗證資料品質

        Args:
            data: OHLCV DataFrame

        Raises:
            ValueError: 資料不足、包含 NaN、價格異常、時間順序錯誤
        """
        # 檢查資料點數
        if len(data) < self.config.min_data_points:
            raise ValueError(
                f"資料不足: {self.config.symbol} {self.config.timeframe} "
                f"只有 {len(data)} 筆（需要至少 {self.config.min_data_points} 筆）"
            )

        # 檢查 NaN
        if data.isnull().any().any():
            raise ValueError(
                f"資料包含 NaN: {self.config.symbol} {self.config.timeframe}"
            )

        # 檢查價格範圍（合理性檢查）
        for col in ['open', 'high', 'low', 'close']:
            if (data[col] <= 0).any():
                raise ValueError(
                    f"資料包含非正價格: {col} 欄位有 <= 0 的值"
                )

        # 檢查時間順序
        if not data.index.is_monotonic_increasing:
            raise ValueError(
                f"資料時間順序錯誤: {self.config.symbol} {self.config.timeframe}"
            )

        # 檢查 OHLC 邏輯
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        if invalid_ohlc.any():
            raise ValueError(
                f"資料包含無效 OHLC: high < low 或其他邏輯錯誤"
            )

    def _cleanup(self):
        """清理資源"""
        logger.info("清理 GPLoop 資源...")

        # 清理引擎
        self._engine = None
        self._evaluator = None
        self._backtest_engine = None
        self._data_fetcher = None
        self._converter = None

        logger.info("GPLoop 清理完成")

    def run(self) -> EvolutionResult:
        """
        執行 GP 演化

        Returns:
            EvolutionResult: 演化結果

        Raises:
            RuntimeError: 如果 GPLoop 未初始化
        """
        if self._engine is None:
            raise RuntimeError("GPLoop 未初始化，請使用 'with GPLoop(config) as loop:'")

        logger.info(
            f"開始 GP 演化（pop={self.config.population_size}, "
            f"gen={self.config.generations}）"
        )

        # 執行演化
        self._result = self._engine.evolve()

        logger.info(
            f"演化完成（generations={self._result.generations_run}, "
            f"best_fitness={self._result.best_fitness:.4f}）"
        )

        # 記錄到學習系統
        if self.config.record_to_learning:
            self._record_to_learning()

        return self._result

    def generate_strategies(self, top_n: Optional[int] = None) -> List[Path]:
        """
        生成前 N 個最佳策略檔案

        Args:
            top_n: 生成策略數量（預設使用 config.generate_top_n）

        Returns:
            List[Path]: 生成的策略檔案路徑列表

        Raises:
            RuntimeError: 如果尚未執行演化
        """
        if self._result is None:
            raise RuntimeError("尚未執行演化，請先呼叫 run()")

        n = top_n or self.config.generate_top_n

        # 取得前 N 個最佳個體
        hof = self._result.hall_of_fame[:n]

        logger.info(f"生成前 {len(hof)} 個最佳策略...")

        # 建立生成器
        generator = StrategyGenerator(self._converter)

        # 生成策略檔案
        generated_files = []

        for i, individual in enumerate(hof):
            # 策略名稱
            strategy_name = f"gp_evolved_{self.config.symbol.lower()}_{i+1:03d}"

            # 適應度
            fitness = individual.fitness.values[0]

            # 元資料
            metadata = {
                'generation': self._result.generations_run,
                'population_size': self.config.population_size,
                'symbol': self.config.symbol,
                'timeframe': self.config.timeframe,
                'rank': i + 1
            }

            # 生成檔案
            try:
                file_path = generator.generate(
                    individual=individual,
                    strategy_name=strategy_name,
                    fitness=fitness,
                    metadata=metadata,
                    output_dir=self.config.output_dir
                )

                generated_files.append(file_path)
                logger.info(f"生成策略 {i+1}/{len(hof)}: {file_path.name}")

            except Exception as e:
                logger.error(f"生成策略 {strategy_name} 失敗: {e}", exc_info=True)

        logger.info(f"成功生成 {len(generated_files)} 個策略檔案")

        return generated_files

    def _record_to_learning(self):
        """記錄演化實驗到學習系統"""
        try:
            from ..gp.learning import GPLearningIntegration

            integrator = GPLearningIntegration()

            # 記錄演化實驗
            metadata = {
                'symbol': self.config.symbol,
                'timeframe': self.config.timeframe,
                'primitive_set': self.config.primitive_set,
            }

            exp_id = integrator.record_evolution(
                result=self._result,
                metadata=metadata
            )

            logger.info(f"已記錄演化實驗: {exp_id}")

            # 記錄到洞察（如果結果優秀）
            integrator.record_to_insights(self._result)

        except Exception as e:
            logger.error(f"記錄到學習系統失敗: {e}", exc_info=True)

    @property
    def best_individual(self):
        """取得最佳個體"""
        if self._result is None:
            return None
        return self._result.best_individual

    @property
    def best_fitness(self) -> Optional[float]:
        """取得最佳適應度"""
        if self._result is None:
            return None
        return self._result.best_fitness


# ============================================================================
# 便利函數
# ============================================================================

def run_gp_evolution(
    symbol: str = 'BTCUSDT',
    timeframe: str = '4h',
    population_size: int = 50,
    generations: int = 30,
    **kwargs
) -> EvolutionResult:
    """
    快速執行 GP 演化

    Args:
        symbol: 交易標的
        timeframe: 時間框架
        population_size: 種群大小
        generations: 演化代數
        **kwargs: 其他配置參數

    Returns:
        EvolutionResult: 演化結果

    Example:
        result = run_gp_evolution(
            symbol='ETHUSDT',
            population_size=100,
            generations=50
        )
        print(f"Best fitness: {result.best_fitness:.4f}")
    """
    config = GPLoopConfig(
        symbol=symbol,
        timeframe=timeframe,
        population_size=population_size,
        generations=generations,
        **kwargs
    )

    with GPLoop(config) as loop:
        result = loop.run()
        loop.generate_strategies()

    return result


# 公開 API
__all__ = [
    'GPLoopConfig',
    'GPLoop',
    'run_gp_evolution',
]
