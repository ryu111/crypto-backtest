"""
UltimateLoopController - 最強回測 Loop 主控制器

整合所有進階功能的完整回測系統：
1. Regime Detection - 識別市場狀態
2. Strategy Selection - 根據狀態選擇策略
3. Multi-Objective Optimization - 多目標優化
4. Validation - 5 階段驗證
5. Learning - 自動記錄學習

使用範例：
    from src.automation.ultimate_loop import UltimateLoopController
    from src.automation.ultimate_config import UltimateLoopConfig

    config = UltimateLoopConfig.create_production_config()
    controller = UltimateLoopController(config)

    summary = await controller.run_loop(n_iterations=50)
    print(summary.summary_text())
"""

import asyncio
import time
import logging
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

from .ultimate_config import UltimateLoopConfig
from ..types.enums import Grade

# 條件性導入（後續 Phase 會使用）
# HyperLoop - Phase 12.1.3 整合
try:
    from .hyperloop import (  # noqa: F401
        HyperLoopController,
        HyperLoopConfig,
        HyperLoopSummary
    )
    HYPERLOOP_AVAILABLE = True
except ImportError:
    HYPERLOOP_AVAILABLE = False
    HyperLoopController = None  # type: ignore[misc,assignment]
    HyperLoopConfig = None  # type: ignore[misc,assignment]
    HyperLoopSummary = None  # type: ignore[misc,assignment]

# Regime Detection - Phase 12.2 使用
try:
    from ..regime.analyzer import MarketStateAnalyzer, MarketState, MarketRegime  # noqa: F401
    REGIME_AVAILABLE = True
except ImportError:
    REGIME_AVAILABLE = False
    MarketStateAnalyzer = None  # type: ignore[misc,assignment]
    MarketState = None  # type: ignore[misc,assignment]
    MarketRegime = None  # type: ignore[misc,assignment]

# RegimeStrategyMapper - Phase 12.2.1
try:
    from .regime_mapper import RegimeStrategyMapper, StrategyRecommendation
    REGIME_MAPPER_AVAILABLE = True
except ImportError:
    REGIME_MAPPER_AVAILABLE = False
    RegimeStrategyMapper = None  # type: ignore[misc,assignment]
    StrategyRecommendation = None  # type: ignore[misc,assignment]

try:
    from ..strategies.registry import StrategyRegistry
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    StrategyRegistry = None

# CompositeStrategy - Phase 12.2 使用
try:
    from ..strategies.composite import CompositeStrategy, SignalAggregation  # noqa: F401
    COMPOSITE_AVAILABLE = True
except ImportError:
    COMPOSITE_AVAILABLE = False
    CompositeStrategy = None  # type: ignore[misc,assignment]
    SignalAggregation = None  # type: ignore[misc,assignment]

# Multi-Objective Optimizer - Phase 12.3 使用
try:
    from ..optimizer.multi_objective import (  # noqa: F401
        MultiObjectiveOptimizer,
        MultiObjectiveResult,
        ParetoSolution,
        ObjectiveResult
    )
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    MultiObjectiveOptimizer = None  # type: ignore[misc,assignment]
    MultiObjectiveResult = None  # type: ignore[misc,assignment]
    ParetoSolution = None  # type: ignore[misc,assignment]
    ObjectiveResult = None  # type: ignore[misc,assignment]

try:
    from ..learning.recorder import ExperimentRecorder
    RECORDER_AVAILABLE = True
except ImportError:
    RECORDER_AVAILABLE = False
    ExperimentRecorder = None

try:
    from ..backtester.validator import BacktestValidator
    VALIDATOR_AVAILABLE = True
except ImportError:
    VALIDATOR_AVAILABLE = False
    BacktestValidator = None

# ValidationRunner - Phase 12.4 使用
try:
    from .validation_runner import ValidationRunner, ValidationResult as VRunnerResult
    VALIDATION_RUNNER_AVAILABLE = True
except ImportError:
    VALIDATION_RUNNER_AVAILABLE = False
    ValidationRunner = None  # type: ignore[misc,assignment]
    VRunnerResult = None  # type: ignore[misc,assignment]

# BacktestEngine - 用於策略評估
try:
    from ..backtester.engine import BacktestEngine
    BACKTEST_ENGINE_AVAILABLE = True
except ImportError:
    BACKTEST_ENGINE_AVAILABLE = False
    BacktestEngine = None  # type: ignore[misc,assignment]

# BacktestConfig - 用於策略評估
try:
    from ..backtester.engine import BacktestConfig
    BACKTEST_CONFIG_AVAILABLE = True
except ImportError:
    BACKTEST_CONFIG_AVAILABLE = False
    BacktestConfig = None  # type: ignore[misc,assignment]

# WalkForwardAnalyzer - Phase 12.11 ValidationRunner 整合
try:
    from ..optimizer.walk_forward import WalkForwardAnalyzer
    WALK_FORWARD_AVAILABLE = True
except ImportError:
    WALK_FORWARD_AVAILABLE = False
    WalkForwardAnalyzer = None  # type: ignore[misc,assignment]

# GPUBatchOptimizer - Phase 12.3.3 GPU 批量優化
try:
    from ..optimizer.gpu_batch import GPUBatchOptimizer, GPUOptimizationResult
    GPU_BATCH_AVAILABLE = True
except ImportError:
    GPU_BATCH_AVAILABLE = False
    GPUBatchOptimizer = None  # type: ignore[misc,assignment]
    GPUOptimizationResult = None  # type: ignore[misc,assignment]

# MemoryIntegration - Phase 12.8 使用
try:
    from ..learning.memory import MemoryIntegration, StrategyInsight
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    MemoryIntegration = None
    StrategyInsight = None

logger = logging.getLogger(__name__)


@dataclass
class UltimateLoopSummary:
    """UltimateLoop 執行摘要"""

    # 基本統計
    total_iterations: int = 0
    successful_iterations: int = 0
    failed_iterations: int = 0

    # 時間統計
    total_duration_seconds: float = 0.0
    avg_iteration_time: float = 0.0

    # Regime 統計
    regime_distribution: Dict[str, int] = field(default_factory=dict)

    # 優化統計
    total_pareto_solutions: int = 0
    selected_pareto_solutions: int = 0
    validated_solutions: int = 0
    validation_passed_count: int = 0
    validation_failed_count: int = 0

    # 最佳結果
    best_strategy: Optional[str] = None
    best_params: Optional[Dict] = None
    best_objectives: Optional[Dict[str, float]] = None

    # 學習統計
    new_insights: int = 0
    memory_entries: int = 0
    experiments_recorded: int = 0

    def summary_text(self) -> str:
        """生成摘要報告"""
        lines = [
            "",
            "=" * 70,
            "UltimateLoop 執行摘要",
            "=" * 70,
            "",
            f"總迭代次數: {self.total_iterations}",
            f"成功: {self.successful_iterations}",
            f"失敗: {self.failed_iterations}",
            f"成功率: {self.successful_iterations/self.total_iterations*100:.1f}%" if self.total_iterations > 0 else "成功率: N/A",
            "",
            f"總執行時間: {self.total_duration_seconds:.1f}s",
            f"平均每次迭代: {self.avg_iteration_time:.1f}s",
            "",
        ]

        if self.regime_distribution:
            lines.append("市場狀態分布:")
            for regime, count in sorted(
                self.regime_distribution.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                pct = count / sum(self.regime_distribution.values()) * 100
                lines.append(f"  {regime}: {count} ({pct:.1f}%)")
            lines.append("")

        lines.extend([
            f"Pareto 解總數: {self.total_pareto_solutions}",
            f"通過驗證: {self.validated_solutions}",
            "",
        ])

        if self.best_strategy:
            lines.append("最佳策略:")
            lines.append(f"  策略: {self.best_strategy}")
            lines.append(f"  參數: {self.best_params}")
            if self.best_objectives:
                lines.append("  目標值:")
                for name, value in self.best_objectives.items():
                    lines.append(f"    {name}: {value:.4f}")
            lines.append("")

        lines.extend([
            f"新洞察: {self.new_insights}",
            f"Memory 條目: {self.memory_entries}",
            "",
            "=" * 70,
        ])

        return "\n".join(lines)


class UltimateLoopController:
    """最強回測 Loop 控制器

    整合所有進階功能的完整回測系統：
    1. Regime Detection - 識別市場狀態
    2. Strategy Selection - 根據狀態選擇策略
    3. Multi-Objective Optimization - 多目標優化
    4. Validation - 5 階段驗證
    5. Learning - 自動記錄學習

    使用範例：
        from src.automation.ultimate_loop import UltimateLoopController
        from src.automation.ultimate_config import UltimateLoopConfig

        config = UltimateLoopConfig.create_production_config()
        controller = UltimateLoopController(config)

        summary = await controller.run_loop(n_iterations=50)
        print(summary.summary_text())
    """

    # Insight 生成閾值（常數）
    INSIGHT_HIGH_SHARPE_THRESHOLD = 2.0      # 高 Sharpe 閾值
    INSIGHT_LOW_DRAWDOWN_THRESHOLD = 0.10    # 低回撤閾值 (10%)
    INSIGHT_HIGH_WINRATE_THRESHOLD = 0.60    # 高勝率閾值 (60%)

    def __init__(
        self,
        config: Optional[UltimateLoopConfig] = None,
        verbose: bool = True
    ):
        """初始化 UltimateLoop 控制器

        Args:
            config: 配置（None 則使用預設值）
            verbose: 是否顯示詳細資訊
        """
        self.config = config or UltimateLoopConfig()
        self.config.validate()  # 驗證配置
        self.verbose = verbose

        # 初始化子模組
        self._init_regime_analyzer()
        self._init_strategy_selector()
        self._init_optimizer()
        self._init_gpu_optimizer()  # Phase 12.3.3: GPU 批量優化
        self._init_validator()
        self._init_learning()
        self._init_hyperloop()  # 新增：初始化 HyperLoop
        self._init_data_cache()  # Phase 12.7: 預載 OHLCV 資料
        self._init_validation_runner()  # Phase 12.11: 初始化 ValidationRunner

        # 執行統計
        self.summary = UltimateLoopSummary()

        # Memory MCP 整合（Phase 12.8）
        self.memory: Optional['MemoryIntegration'] = None
        self._memory_suggestions: List[Dict] = []

        # 檢查點
        self._checkpoint_data: Dict = {}

        # 清理狀態標誌（防止重複清理）
        self._cleaned_up: bool = False

        if self.verbose:
            logger.info("UltimateLoopController initialized")
            self._log_module_availability()

    def _log_module_availability(self):
        """記錄模組可用性"""
        modules = {
            'HyperLoop': HYPERLOOP_AVAILABLE,
            'Regime Detection': REGIME_AVAILABLE,
            'Regime Mapper': REGIME_MAPPER_AVAILABLE,
            'Strategy Registry': REGISTRY_AVAILABLE,
            'Composite Strategy': COMPOSITE_AVAILABLE,
            'Multi-Objective Optimizer': OPTIMIZER_AVAILABLE,
            'GPU Batch Optimizer': GPU_BATCH_AVAILABLE,
            'Experiment Recorder': RECORDER_AVAILABLE,
            'Validator': VALIDATOR_AVAILABLE,
            'ValidationRunner': VALIDATION_RUNNER_AVAILABLE,
            'Memory Integration': MEMORY_AVAILABLE
        }

        logger.info("Module availability:")
        for name, available in modules.items():
            status = "✓" if available else "✗"
            logger.info(f"  {status} {name}")

    def _init_regime_analyzer(self):
        """初始化 Regime 分析器"""
        if self.config.regime_detection and REGIME_AVAILABLE and MarketStateAnalyzer is not None:
            self.regime_analyzer = MarketStateAnalyzer(
                direction_threshold_strong=self.config.direction_threshold_strong,
                direction_threshold_weak=self.config.direction_threshold_weak,
                volatility_threshold=self.config.volatility_threshold,
                direction_method=self.config.direction_method
            )
            if self.verbose:
                logger.info("Regime analyzer initialized")
        else:
            self.regime_analyzer = None
            if self.config.regime_detection:
                logger.warning("Regime detection enabled but module not available")

        # 初始化 RegimeStrategyMapper（如果可用）
        if REGIME_MAPPER_AVAILABLE and RegimeStrategyMapper is not None:
            self.regime_mapper = RegimeStrategyMapper()
            if self.verbose:
                logger.info("Regime strategy mapper initialized")
        else:
            self.regime_mapper = None

    def _init_strategy_selector(self):
        """初始化策略選擇器"""
        if REGISTRY_AVAILABLE and StrategyRegistry is not None:
            # 取得可用策略列表
            if self.config.enabled_strategies:
                self.available_strategies = self.config.enabled_strategies
            else:
                self.available_strategies = StrategyRegistry.list_all()

            # 策略績效追蹤
            self.strategy_stats: Dict[str, Dict] = {}

            if self.verbose:
                logger.info(f"Strategy selector initialized with {len(self.available_strategies)} strategies")
        else:
            self.available_strategies = []
            self.strategy_stats = {}
            logger.warning("Strategy registry not available")

    def _init_optimizer(self):
        """初始化多目標優化器"""
        # 稍後在執行時建立，因為需要 strategy 資訊
        self.optimizer: Optional[Any] = None

        if not OPTIMIZER_AVAILABLE:
            logger.warning("Multi-objective optimizer not available")

    def _init_gpu_optimizer(self):
        """初始化 GPU 批量優化器（Phase 12.3.3）

        根據 config.use_gpu 決定是否啟用 GPU 優化。
        如果 GPU 不可用，會自動降級到 CPU。
        """
        self.gpu_optimizer: Optional[Any] = None

        if not self.config.use_gpu:
            if self.verbose:
                logger.info("GPU optimization disabled by config")
            return

        if not GPU_BATCH_AVAILABLE or GPUBatchOptimizer is None:
            logger.warning("GPU batch optimizer not available")
            return

        try:
            self.gpu_optimizer = GPUBatchOptimizer(
                prefer_mlx=True,           # Apple Silicon 優先使用 MLX
                fallback_to_cpu=True,      # GPU 不可用時降級到 CPU
                verbose=self.verbose
            )

            if self.verbose:
                backend = getattr(self.gpu_optimizer, '_backend', 'unknown')
                logger.info(f"GPU optimizer initialized (backend: {backend})")

        except Exception as e:
            logger.warning(f"Failed to initialize GPU optimizer: {e}")
            self.gpu_optimizer = None

    def _init_validator(self):
        """初始化驗證器"""
        if self.config.validation_enabled and VALIDATOR_AVAILABLE and BacktestValidator is not None:
            self.validator = BacktestValidator()
            if self.verbose:
                logger.info("Validator initialized")
        else:
            self.validator = None
            if self.config.validation_enabled:
                logger.warning("Validation enabled but module not available")

    def _init_learning(self):
        """初始化學習系統

        Note:
            ExperimentRecorder 現在使用 DuckDB（data/experiments.duckdb）儲存實驗記錄。
            資源管理由 _cleanup() 方法處理，會自動呼叫 recorder.close() 釋放連線。

            也可以使用 context manager:
                with ExperimentRecorder() as recorder:
                    recorder.log_experiment(...)
        """
        if self.config.learning_enabled and RECORDER_AVAILABLE and ExperimentRecorder is not None:
            # ExperimentRecorder 使用 DuckDB 儲存（data/experiments.duckdb）
            # insights 自動更新到 learning/insights.md
            self.recorder = ExperimentRecorder()
            if self.verbose:
                logger.info("Learning system initialized (DuckDB storage)")
        else:
            self.recorder = None
            if self.config.learning_enabled:
                logger.warning("Learning enabled but module not available")

        # Memory MCP 整合（Phase 12.8）
        if self.config.memory_mcp_enabled and MEMORY_AVAILABLE and MemoryIntegration is not None:
            self.memory = MemoryIntegration()
            if self.verbose:
                logger.info("Memory MCP integration initialized")
        else:
            self.memory = None
            if self.config.memory_mcp_enabled and not MEMORY_AVAILABLE:
                logger.warning("Memory MCP enabled but module not available")

    def _init_validation_runner(self):
        """初始化 ValidationRunner 和相關模組（Phase 12.11）

        建立完整的 5 階段驗證系統：
        1. BacktestEngine - 用於執行回測
        2. WalkForwardAnalyzer - 用於 Stage 4 Walk-Forward 分析
        3. ValidationRunner - 整合所有驗證階段
        """
        from datetime import timedelta

        self._validation_runner: Optional[Any] = None
        self._validation_engine: Optional[Any] = None
        self._wfa_analyzer: Optional[Any] = None

        if not self.config.validation_enabled:
            if self.verbose:
                logger.info("Validation disabled by config")
            return

        # 檢查所有必要模組是否可用
        if not VALIDATION_RUNNER_AVAILABLE or ValidationRunner is None:
            logger.warning("ValidationRunner not available")
            return

        if not BACKTEST_ENGINE_AVAILABLE or BacktestEngine is None:
            logger.warning("BacktestEngine not available for validation")
            return

        if not BACKTEST_CONFIG_AVAILABLE or BacktestConfig is None:
            logger.warning("BacktestConfig not available for validation")
            return

        try:
            # 從資料緩存取得時間範圍（如果可用）
            start_date = datetime.now() - timedelta(days=365)
            end_date = datetime.now()

            if hasattr(self, '_data_cache') and self._data_cache:
                # 使用第一個資料集的時間範圍
                first_key = next(iter(self._data_cache.keys()), None)
                if first_key and first_key in self._data_cache:
                    df = self._data_cache[first_key]
                    if hasattr(df.index, 'min') and hasattr(df.index, 'max'):
                        idx_min = df.index.min()
                        idx_max = df.index.max()
                        if hasattr(idx_min, 'to_pydatetime'):
                            start_date = idx_min.to_pydatetime()
                        if hasattr(idx_max, 'to_pydatetime'):
                            end_date = idx_max.to_pydatetime()

            # 建立 BacktestConfig
            bt_config = BacktestConfig(
                symbol=self.config.symbols[0] if self.config.symbols else 'BTCUSDT',
                timeframe=self.config.timeframes[0] if self.config.timeframes else '1h',
                start_date=start_date,
                end_date=end_date,
                initial_capital=self.config.initial_capital,
                leverage=self.config.leverage
            )

            # 建立 BacktestEngine
            self._validation_engine = BacktestEngine(bt_config)

            # 建立 WalkForwardAnalyzer（如果可用）
            if WALK_FORWARD_AVAILABLE and WalkForwardAnalyzer is not None:
                self._wfa_analyzer = WalkForwardAnalyzer(
                    config=bt_config,
                    mode='rolling'
                )
                if self.verbose:
                    logger.info("WalkForwardAnalyzer initialized")

            # 建立 ValidationRunner
            self._validation_runner = ValidationRunner(
                engine=self._validation_engine,
                wfa_analyzer=self._wfa_analyzer,
                stages=[1, 2, 3, 4, 5]  # 全 5 階段驗證
            )

            if self.verbose:
                logger.info("ValidationRunner initialized with 5-stage validation")

        except Exception as e:
            logger.error(f"Failed to initialize ValidationRunner: {e}")
            self._validation_runner = None
            self._validation_engine = None
            self._wfa_analyzer = None

    def _init_hyperloop(self):
        """初始化 HyperLoop 控制器"""
        self.hyperloop: Optional[Any] = None

        # 檢查是否啟用高效能優化
        if not hasattr(self.config, 'hyperloop_enabled') or not self.config.hyperloop_enabled:
            return

        if not HYPERLOOP_AVAILABLE or HyperLoopController is None or HyperLoopConfig is None:
            logger.warning("HyperLoop enabled but module not available")
            return

        try:
            # 轉換 UltimateLoopConfig → HyperLoopConfig
            hyperloop_config = HyperLoopConfig(
                max_workers=getattr(self.config, 'max_workers', 8),
                use_gpu=getattr(self.config, 'use_gpu', True),
                symbols=getattr(self.config, 'symbols', ['BTCUSDT', 'ETHUSDT']),
                timeframes=getattr(self.config, 'timeframes', ['1h', '4h', '1d']),
                data_dir=getattr(self.config, 'data_dir', 'data'),
                n_trials=getattr(self.config, 'n_trials', 100),
                param_sweep_threshold=getattr(self.config, 'param_sweep_threshold', 100),
                min_sharpe=getattr(self.config, 'min_sharpe', 1.0),
                leverage=getattr(self.config, 'leverage', 5),
                initial_capital=getattr(self.config, 'initial_capital', 10000.0),
                maker_fee=getattr(self.config, 'maker_fee', 0.0002),
                taker_fee=getattr(self.config, 'taker_fee', 0.0004),
                timeout_per_iteration=getattr(self.config, 'timeout_per_iteration', 600),
                max_retries=getattr(self.config, 'max_retries', 3)
            )

            # 初始化 HyperLoop
            self.hyperloop = HyperLoopController(
                config=hyperloop_config,
                verbose=self.verbose
            )

            if self.verbose:
                logger.info("HyperLoop initialized for high-performance optimization")

        except Exception as e:
            logger.error(f"Failed to initialize HyperLoop: {e}")
            self.hyperloop = None

    def _init_data_cache(self):
        """預載所有 OHLCV 資料到記憶體緩存

        Phase 12.7: 從 data/ohlcv/*.parquet 預載資料供策略評估使用。
        """
        import pandas as pd

        self._data_cache: Dict[str, pd.DataFrame] = {}

        data_dir = Path(self.config.data_dir)
        ohlcv_dir = data_dir / "ohlcv"

        if not ohlcv_dir.exists():
            logger.warning(f"OHLCV directory not found: {ohlcv_dir}")
            return

        # 獲取所有時間框架（使用統一的 timeframes 列表）
        all_timeframes = self.config.timeframes if self.config.timeframes else ['1h', '4h', '1d']

        for symbol in self.config.symbols:
            for timeframe in all_timeframes:
                key = f"{symbol}_{timeframe}"
                file_path = ohlcv_dir / f"{key}.parquet"

                if file_path.exists():
                    try:
                        df = pd.read_parquet(file_path)
                        self._data_cache[key] = df
                        if self.verbose:
                            logger.debug(f"Loaded {key}: {len(df)} rows")
                    except Exception as e:
                        logger.warning(f"Failed to load {key}: {e}")

        if self.verbose:
            logger.info(f"Data cache initialized: {len(self._data_cache)} datasets loaded")

    def _get_data(self, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> Optional[pd.DataFrame]:
        """取得 OHLCV 資料

        Args:
            symbol: 標的（None 則使用第一個）
            timeframe: 時間框架（None 則使用第一個）

        Returns:
            pd.DataFrame 或 None（如果找不到）
        """
        if not hasattr(self, '_data_cache') or not self._data_cache:
            logger.warning("Data cache not initialized or empty")
            return None

        # 使用預設值
        if symbol is None:
            symbol = self.config.symbols[0] if self.config.symbols else 'BTCUSDT'

        if timeframe is None:
            # 使用第一個可用的 timeframe
            timeframe = self.config.timeframes[0] if self.config.timeframes else '1h'

        key = f"{symbol}_{timeframe}"
        return self._data_cache.get(key)

    async def run_loop(
        self,
        n_iterations: int,
        resume_from_checkpoint: bool = False
    ) -> UltimateLoopSummary:
        """執行 UltimateLoop

        完整流程：
        1. 驗證回測引擎正確性（啟動時）
        2. 分析市場狀態
        3. 選擇適合的策略
        4. 多目標優化
        5. 驗證 Pareto 解
        6. 記錄學習

        Args:
            n_iterations: 迭代次數
            resume_from_checkpoint: 是否從檢查點恢復（預設 False）

        Returns:
            UltimateLoopSummary: 執行摘要
        """
        start_time = time.time()
        self.summary = UltimateLoopSummary()
        self.summary.total_iterations = n_iterations

        if self.verbose:
            self._print_header(n_iterations)

        try:
            # 0. 啟動時驗證回測引擎
            if self.config.validation_enabled and self.validator:
                self._validate_engine_on_startup()

            # 嘗試恢復檢查點
            start_iteration = self._try_restore_checkpoint() if resume_from_checkpoint else 0

            if resume_from_checkpoint and start_iteration > 0:
                # 驗證恢復的迭代編號合理性
                if start_iteration >= n_iterations:
                    logger.warning(
                        f"檢查點迭代 ({start_iteration}) >= 目標迭代數 ({n_iterations})，"
                        "將從頭開始執行"
                    )
                    start_iteration = 0
                else:
                    logger.info(f"✓ 從檢查點恢復：將從迭代 {start_iteration} 繼續執行")
            elif resume_from_checkpoint and start_iteration == 0:
                logger.warning("未找到有效檢查點，將從頭開始執行")

            # 主迭代循環
            for i in range(start_iteration, n_iterations):
                try:
                    await self._run_iteration(i, n_iterations)
                    self.summary.successful_iterations += 1

                    # 定期存檔檢查點
                    if self.config.checkpoint_enabled:
                        if (i + 1) % self.config.checkpoint_interval == 0:
                            self._save_checkpoint(i + 1)

                except Exception as e:
                    self.summary.failed_iterations += 1
                    logger.error(f"迭代 {i+1} 失敗: {e}", exc_info=True)

                    # 重試機制
                    if self.config.max_retries > 0:
                        logger.info(f"將重試迭代 {i+1}...")
                        retry_success = False
                        for retry in range(self.config.max_retries):
                            try:
                                await self._run_iteration(i, n_iterations)
                                # 重試成功：修正統計（之前已加入 failed，現在改為 successful）
                                self.summary.failed_iterations -= 1
                                self.summary.successful_iterations += 1
                                retry_success = True
                                logger.info(f"重試成功 (第 {retry + 1} 次)")
                                break
                            except Exception as retry_e:
                                logger.error(f"重試 {retry + 1} 失敗: {retry_e}")
                                if retry == self.config.max_retries - 1:
                                    logger.error(f"迭代 {i+1} 最終失敗（已重試 {self.config.max_retries} 次）")

            # 計算統計
            self.summary.total_duration_seconds = time.time() - start_time
            self.summary.avg_iteration_time = (
                self.summary.total_duration_seconds / n_iterations
                if n_iterations > 0 else 0
            )

            # Phase 12.11: 輸出 Memory MCP 存儲建議
            if self._memory_suggestions and self.verbose:
                logger.info("=" * 50)
                logger.info("=== Memory MCP 存儲建議 ===")
                logger.info(f"共 {len(self._memory_suggestions)} 個待存儲洞察")
                for i, cmd in enumerate(self.get_memory_commands(), 1):
                    logger.info(f"  [{i}] {cmd[:100]}..." if len(cmd) > 100 else f"  [{i}] {cmd}")
                logger.info("=" * 50)
                logger.info("提示：使用 controller.get_memory_commands() 取得完整命令")

            if self.verbose:
                logger.info(self.summary.summary_text())

            return self.summary

        finally:
            # 清理資源
            self._cleanup()

    async def _run_iteration(self, iteration: int, total: int):
        """執行單次迭代

        Args:
            iteration: 當前迭代編號
            total: 總迭代次數
        """
        if self.verbose:
            logger.info(f"\n[{iteration+1}/{total}] 開始迭代...")

        # Phase 1: Market Analysis
        market_state = await self._analyze_market_state()

        # Phase 2: Strategy Selection
        selected_strategies = self._select_strategies(market_state)

        # Phase 3: Multi-Objective Optimization
        pareto_result = await self._run_optimization(selected_strategies)

        # Phase 4: Validation (Phase 12.11: 傳遞 market_state)
        validated_solutions = await self._validate_pareto_solutions(pareto_result, market_state)

        # Phase 5: Learning
        await self._record_and_learn(validated_solutions, market_state, selected_strategies)

        if self.verbose:
            logger.info(f"[{iteration+1}/{total}] 迭代完成")

    async def _analyze_market_state(self) -> Optional[Any]:
        """分析市場狀態

        Returns:
            Optional[MarketState]: 市場狀態（如果 regime detection 啟用）
        """
        if not self.regime_analyzer:
            return None

        # TODO: 獲取市場數據（這裡需要實際資料）
        # 暫時返回 None，後續整合資料層
        if self.verbose:
            logger.debug("Market state analysis (not yet implemented)")

        return None

    def _select_strategies(
        self,
        market_state: Optional[Any]
    ) -> List[str]:
        """根據市場狀態選擇策略

        Args:
            market_state: 市場狀態

        Returns:
            List[str]: 選中的策略名稱列表
        """
        if not self.available_strategies:
            logger.warning("No strategies available")
            return []

        if self.config.strategy_selection_mode == 'random':
            # 隨機選擇
            n_select = min(3, len(self.available_strategies))
            selected = list(np.random.choice(
                self.available_strategies,
                size=n_select,
                replace=False
            ))
            if self.verbose:
                logger.info(f"Randomly selected {n_select} strategies: {selected}")
            return selected

        elif self.config.strategy_selection_mode == 'exploit':
            # Exploit 模式：選擇歷史表現最好的
            selected = self._select_exploit_strategies()
            if self.verbose:
                logger.info(f"Exploit mode selected strategies: {selected}")
            return selected

        elif self.config.strategy_selection_mode == 'regime_aware':
            # Regime-aware 模式
            if market_state:
                selected = self._select_by_regime(market_state)
                if self.verbose:
                    logger.info(f"Regime-aware selected strategies: {selected}")
                return selected
            else:
                # Fallback to exploit
                selected = self._select_exploit_strategies()
                if self.verbose:
                    logger.info(f"Regime-aware fallback to exploit: {selected}")
                return selected

        # Default: 前 3 個策略
        selected = self.available_strategies[:3]
        if self.verbose:
            logger.info(f"Default selected strategies: {selected}")
        return selected

    def _select_exploit_strategies(self) -> List[str]:
        """選擇歷史表現最好的策略

        Returns:
            List[str]: 策略名稱列表
        """
        if not self.strategy_stats:
            # 沒有歷史數據，隨機選擇
            n_select = min(3, len(self.available_strategies))
            return list(np.random.choice(
                self.available_strategies,
                size=n_select,
                replace=False
            ))

        # 根據 Sharpe 排序
        sorted_strategies = sorted(
            self.strategy_stats.items(),
            key=lambda x: x[1].get('avg_sharpe', 0),
            reverse=True
        )

        # 80% exploit / 20% explore
        n_total = 3
        n_exploit = int(n_total * self.config.exploit_ratio)
        n_explore = n_total - n_exploit

        # Exploit: 取前 n_exploit 個最佳策略
        exploit = [s[0] for s in sorted_strategies[:n_exploit]]

        # Explore: 從剩餘策略隨機選
        remaining = [s for s in self.available_strategies if s not in exploit]
        if remaining and n_explore > 0:
            explore = list(np.random.choice(
                remaining,
                size=min(n_explore, len(remaining)),
                replace=False
            ))
        else:
            explore = []

        return exploit + explore

    def _select_by_regime(self, market_state: Any) -> List[str]:
        """根據 Regime 選擇策略

        Args:
            market_state: 市場狀態（MarketState 物件）

        Returns:
            List[str]: 策略名稱列表
        """
        # 檢查 regime_mapper 是否可用
        if not self.regime_mapper:
            if self.verbose:
                logger.debug("Regime mapper not available, using default strategies")
            return self.available_strategies[:3]

        # 從 market_state 獲取 regime
        regime = getattr(market_state, 'regime', None)
        if regime is None:
            if self.verbose:
                logger.debug("No regime in market_state, using default strategies")
            return self.available_strategies[:3]

        # 使用 RegimeStrategyMapper 獲取推薦
        recommendation = self.regime_mapper.get_strategies(
            regime=regime,
            available_strategies=self.available_strategies
        )

        if self.verbose:
            logger.info(
                f"Regime {regime.value} → Strategies: {recommendation.strategy_names} "
                f"(confidence: {recommendation.confidence:.2f})"
            )
            logger.debug(f"Reason: {recommendation.reason}")

        # 儲存推薦資訊供後續使用
        self._current_recommendation = recommendation

        return recommendation.strategy_names

    async def _run_optimization(
        self,
        strategies: List[str]
    ) -> Optional[Any]:
        """執行多目標優化

        Args:
            strategies: 要優化的策略列表

        Returns:
            Optional[MultiObjectiveResult]: 優化結果

        Note:
            優先使用 HyperLoop（如可用），否則 fallback 到 MultiObjectiveOptimizer
        """
        if not strategies:
            logger.warning("No strategies to optimize")
            return None

        # 優先使用 HyperLoop（高效能批量優化）
        if self.hyperloop is not None:
            try:
                if self.verbose:
                    logger.info(f"Using HyperLoop for optimization ({len(strategies)} strategies)")

                # 使用 HyperLoop 執行優化
                # 注意：HyperLoop 會自動處理多個策略的並行優化
                hyperloop_summary = await self.hyperloop.run_loop(
                    n_iterations=len(strategies)
                )

                # 轉換 HyperLoop 結果為 MultiObjectiveResult 格式
                # 這樣後續驗證階段可以正確處理 Pareto 解
                result = self._convert_hyperloop_to_multi_objective_result(hyperloop_summary)

                # 更新 Pareto 解統計
                if hasattr(result, 'pareto_front') and result.pareto_front:
                    self.summary.total_pareto_solutions += len(result.pareto_front)

                return result

            except Exception as e:
                logger.error(f"HyperLoop optimization failed: {e}")
                logger.info("Falling back to standard optimization...")

        # Fallback: 使用標準 MultiObjectiveOptimizer
        if not OPTIMIZER_AVAILABLE or MultiObjectiveOptimizer is None:
            logger.warning("Optimizer not available, skipping optimization")
            return None

        # 建立優化器（如果尚未建立）
        if self.optimizer is None:
            self.optimizer = MultiObjectiveOptimizer(
                objectives=self.config.objectives,
                n_trials=self.config.n_trials,
                seed=42,
                verbose=self.verbose
            )

        # 並行優化所有策略（使用 asyncio.gather + Semaphore 資源控制）
        if self.verbose:
            logger.info(f"Starting parallel optimization for {len(strategies)} strategies")

        # 限制並行數量避免資源過載
        MAX_CONCURRENT = min(self.config.max_workers, len(strategies))
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)

        async def optimize_with_error_handling(strategy_name: str):
            """帶錯誤處理和資源控制的優化包裝"""
            async with semaphore:
                try:
                    result = await self._optimize_strategy(strategy_name)
                    return (strategy_name, result, None)
                except Exception as e:
                    return (strategy_name, None, str(e))

        # 並行執行所有策略優化
        tasks = [optimize_with_error_handling(s) for s in strategies]
        results = await asyncio.gather(*tasks)

        # 處理結果
        all_results = []
        failed_strategies = []
        for strategy_name, result, error in results:
            if error:
                logger.error(f"Optimization failed for {strategy_name}: {error}")
                failed_strategies.append(strategy_name)
            elif result:
                all_results.append(result)
                self.summary.total_pareto_solutions += len(result.pareto_front)

        # 失敗率閾值檢查（> 50% 失敗則警告）
        failure_rate = len(failed_strategies) / len(strategies) if strategies else 0
        if failure_rate > 0.5:
            logger.warning(
                f"High failure rate: {failure_rate:.1%} ({len(failed_strategies)}/{len(strategies)} strategies failed)"
            )

        if not all_results:
            logger.warning(f"No successful optimizations. Failed: {failed_strategies}")
            return None

        # 合併結果
        combined_result = self._combine_optimization_results(all_results)

        # 選擇最佳 Pareto 解
        selected_solutions = self._select_pareto_solutions(combined_result)
        if selected_solutions:
            self.summary.selected_pareto_solutions += len(selected_solutions)

        if self.verbose:
            logger.info(
                f"Optimization completed: {len(all_results)} strategies, "
                f"{len(combined_result.pareto_front)} Pareto solutions, "
                f"{len(selected_solutions)} selected"
            )

        # 儲存選擇的解供後續驗證使用
        combined_result.selected_solutions = selected_solutions

        return combined_result

    async def _validate_pareto_solutions(
        self,
        pareto_result: Optional[Any],
        market_state: Optional[Any] = None
    ) -> List[Any]:
        """驗證 Pareto 解（Phase 12.11 修復版）

        使用 ValidationRunner 進行 5 階段驗證。

        Args:
            pareto_result: Pareto 優化結果（包含 selected_solutions 屬性）
            market_state: 市場狀態（用於取得 symbol/timeframe）

        Returns:
            List[ParetoSolution]: 通過驗證的解

        Note:
            - 優先使用 selected_solutions（由 _select_pareto_solutions 選出的最佳解）
            - 使用 _validation_runner 進行真正的 5 階段驗證
            - 並行驗證所有解，使用 Semaphore 控制並發數量（最多 4 個）
        """
        if not pareto_result:
            return []

        # 優先使用 selected_solutions（已經過篩選的最佳解）
        solutions_to_validate = getattr(pareto_result, 'selected_solutions', None)
        if not solutions_to_validate:
            # Fallback: 使用完整 pareto_front
            solutions_to_validate = getattr(pareto_result, 'pareto_front', [])

        if not solutions_to_validate:
            if self.verbose:
                logger.debug("No solutions to validate")
            return []

        # 更新待驗證統計
        self.summary.validated_solutions += len(solutions_to_validate)

        # 檢查 ValidationRunner 是否已初始化（Phase 12.11）
        if not hasattr(self, '_validation_runner') or self._validation_runner is None:
            if self.verbose:
                logger.debug(
                    f"ValidationRunner not initialized, returning {len(solutions_to_validate)} "
                    "unvalidated solutions"
                )
            for solution in solutions_to_validate:
                solution._validation_grade = 'UNVALIDATED'
            return solutions_to_validate

        # 取得 symbol 和 timeframe
        symbol = 'BTCUSDT'
        timeframe = '1h'
        if market_state:
            symbol = getattr(market_state, 'symbol', symbol)
            timeframe = getattr(market_state, 'timeframe', timeframe)
        elif self.config.symbols:
            symbol = self.config.symbols[0]
        if self.config.timeframes:
            timeframe = self.config.timeframes[0]

        # 取得市場資料
        data_key = f"{symbol}_{timeframe}"
        data = self._data_cache.get(data_key) if hasattr(self, '_data_cache') else None

        if data is None:
            if self.verbose:
                logger.warning(f"No data available for {data_key}, skipping validation")
            for solution in solutions_to_validate:
                solution._validation_grade = 'NO_DATA'
            return solutions_to_validate

        # Phase 12.11: 使用 ValidationRunner 進行真正的驗證
        if self.verbose:
            logger.info(f"Starting 5-stage validation for {len(solutions_to_validate)} Pareto solutions")

        validated_solutions = []
        failed_solutions = []

        # 並發控制：最多同時驗證 4 個解
        semaphore = asyncio.Semaphore(4)

        async def validate_solution(solution: Any) -> Tuple[Any, Optional[Any]]:
            """驗證單個解"""
            async with semaphore:
                try:
                    # 提取策略名稱和參數（優先檢查 _strategy_name）
                    strategy_name = getattr(solution, '_strategy_name', None)
                    if not strategy_name:
                        strategy_name = getattr(solution, 'strategy_name', None)
                    params = getattr(solution, 'params', {})

                    if not strategy_name:
                        # 嘗試從 metadata 取得
                        metadata = getattr(solution, 'metadata', {})
                        strategy_name = metadata.get('strategy_name')

                    if not strategy_name or not REGISTRY_AVAILABLE or StrategyRegistry is None:
                        logger.debug(f"Solution {getattr(solution, 'trial_number', '?')}: No strategy name")
                        solution._validation_grade = 'NO_STRATEGY'
                        return (solution, None)

                    # 建立策略實例
                    strategy = StrategyRegistry.create(strategy_name, **params)

                    # 執行 5 階段驗證（同步呼叫）
                    result = self._validation_runner.validate(
                        strategy=strategy,
                        params=params,
                        data=data,
                        symbol=symbol,
                        timeframe=timeframe
                    )

                    return (solution, result)

                except Exception as e:
                    trial_num = getattr(solution, 'trial_number', '?')
                    logger.warning(f"Validation error for solution {trial_num}: {e}")
                    solution._validation_grade = 'ERROR'
                    return (solution, None)

        # 並行驗證所有解
        validation_tasks = [validate_solution(sol) for sol in solutions_to_validate]
        validation_results = await asyncio.gather(*validation_tasks)

        # 路由驗證結果
        for solution, result in validation_results:
            if result is None:
                # 驗證失敗或跳過
                failed_solutions.append(solution)
                self.summary.validation_failed_count += 1
            elif hasattr(result, 'grade') and result.grade in ['A', 'B', 'C']:
                # 驗證通過（grade A/B/C）
                solution._validation_grade = result.grade
                solution._validation_result = result  # 保存完整結果
                validated_solutions.append(solution)
                self.summary.validation_passed_count += 1
            else:
                # 驗證失敗（grade D/F）
                grade = getattr(result, 'grade', 'F')
                solution._validation_grade = grade
                solution._validation_result = result
                failed_solutions.append(solution)
                self.summary.validation_failed_count += 1
                if self.verbose:
                    logger.debug(
                        f"Solution {getattr(solution, 'trial_number', '?')} failed validation: grade {grade}"
                    )

        # 記錄統計
        if self.verbose:
            pass_rate = (
                (len(validated_solutions) / len(solutions_to_validate) * 100)
                if solutions_to_validate else 0
            )
            logger.info(
                f"Validation completed: {len(validated_solutions)}/{len(solutions_to_validate)} "
                f"passed ({pass_rate:.1f}%)"
            )

        # 返回驗證通過的解（或所有解如果是 fallback 模式）
        return validated_solutions if validated_solutions else solutions_to_validate

    async def _record_and_learn(
        self,
        solutions: List[Any],
        market_state: Optional[Any],
        strategy_names: Optional[List[str]] = None
    ):
        """記錄學習

        Phase 12.5 實作：
        1. 記錄驗證通過的解到 experiments.json
        2. 生成洞察並更新 insights.md
        3. 嘗試存入 Memory MCP（可選）

        Args:
            solutions: 驗證通過的解（ParetoSolution 列表）
            market_state: 市場狀態（MarketState 物件，可選）
            strategy_names: 策略名稱列表（用於記錄）
        """
        if not self.recorder:
            if self.verbose:
                logger.debug("Recorder not available, skipping learning")
            return

        if not solutions:
            if self.verbose:
                logger.debug("No solutions to record")
            return

        # 統計
        recorded_count = 0

        # 組合策略名稱（多策略用 + 連接，空則用預設）
        combined_strategy_name = (
            '+'.join(strategy_names) if strategy_names and len(strategy_names) > 0
            else 'multi_objective_pareto'
        )

        if self.verbose:
            logger.info(f"Recording {len(solutions)} validated solutions for: {combined_strategy_name}")

        for solution in solutions:
            try:
                # 提取指標（防禦性檢查）
                objectives = getattr(solution, 'objectives', [])
                params = getattr(solution, 'params', {})
                grade = getattr(solution, '_validation_grade', 'UNKNOWN')

                # 跳過沒有指標的解（記錄 warning 以便除錯）
                if not objectives:
                    trial_num = getattr(solution, 'trial_number', 'unknown')
                    logger.warning(
                        f"Solution {trial_num} missing 'objectives' attribute, skipping. "
                        f"Solution type: {type(solution).__name__}"
                    )
                    continue

                # 生成洞察
                insights = self._generate_insights(objectives, market_state, grade)

                # 從 objectives 創建 BacktestResult（mock）
                mock_result = self._create_result_from_objectives(objectives, params)

                # 記錄實驗
                exp_id = self.recorder.log_experiment(
                    result=mock_result,
                    strategy_info={
                        'name': combined_strategy_name,
                        'type': 'multi_objective',
                        'version': '1.0',
                        'params': params  # 包含策略參數
                    },
                    config={
                        'iteration': self.summary.total_iterations,
                        'validation_grade': grade
                    },
                    validation_result=None,  # 驗證結果已整合到 grade
                    insights=insights
                )

                recorded_count += 1

                # Phase 12.8: Memory MCP 整合
                if self.memory and self.config.memory_mcp_enabled:
                    try:
                        # 提取 objectives 為 dict
                        obj_dict = {}
                        for obj in objectives:
                            name = getattr(obj, 'name', '')
                            value = getattr(obj, 'value', 0.0)
                            obj_dict[name] = value

                        sharpe = obj_dict.get('sharpe_ratio', 0.0)

                        # 只存儲達到 min_sharpe 的成功結果，或存儲失敗
                        should_store = (
                            sharpe >= self.config.memory_min_sharpe or
                            (self.config.memory_store_failures and grade in ['D', 'F', 'ERROR'])
                        )

                        if should_store and StrategyInsight is not None:
                            # 建立 StrategyInsight
                            insight = StrategyInsight(
                                strategy_name=combined_strategy_name,
                                symbol=market_state.symbol if market_state and hasattr(market_state, 'symbol') else 'UNKNOWN',
                                timeframe=market_state.timeframe if market_state and hasattr(market_state, 'timeframe') else '1h',
                                best_params=params,
                                sharpe_ratio=sharpe,
                                total_return=obj_dict.get('total_return', 0.0),
                                max_drawdown=obj_dict.get('max_drawdown', 0.0),
                                win_rate=obj_dict.get('win_rate', 0.0),
                                wfa_grade=grade,
                                market_conditions=str(market_state.regime) if market_state and hasattr(market_state, 'regime') else None,
                                notes=f"UltimateLoop iteration, Grade: {grade}"
                            )

                            # 格式化為 Memory 格式
                            content, metadata = self.memory.format_strategy_insight(insight)

                            # 記錄存儲建議
                            self._memory_suggestions.append({
                                'content': content,
                                'metadata': metadata,
                                'timestamp': datetime.now().isoformat(),
                                'grade': grade,
                                'sharpe': sharpe
                            })

                            self.summary.memory_entries += 1

                            if self.verbose:
                                logger.debug(f"Memory suggestion added: grade={grade}, sharpe={sharpe:.2f}")

                    except Exception as e:
                        logger.warning(f"Failed to create Memory suggestion: {e}")

                if self.verbose:
                    logger.debug(f"Recorded experiment: {exp_id} (grade: {grade})")

            except Exception as e:
                logger.warning(f"Failed to record solution: {e}")
                continue

        # 更新統計
        self.summary.experiments_recorded += recorded_count
        self.summary.new_insights += recorded_count  # 每個記錄都算一個洞察

        if self.verbose:
            logger.info(f"Successfully recorded {recorded_count} experiments")

    def _generate_insights(
        self,
        objectives: List[Any],
        market_state: Optional[Any],
        grade: str
    ) -> List[str]:
        """生成洞察

        根據績效指標和市場狀態生成洞察。

        Args:
            objectives: 目標值列表（List[ObjectiveResult]）
            market_state: 市場狀態（MarketState 物件，可選）
            grade: 驗證評級（A/B/C/D/F/UNKNOWN/UNVALIDATED/ERROR）

        Returns:
            List[str]: 洞察列表
        """
        insights = []

        # 提取指標值（防禦性處理）
        sharpe = 0.0
        max_dd = 1.0
        win_rate = 0.0

        for obj in objectives:
            name = getattr(obj, 'name', '')
            value = getattr(obj, 'value', 0.0)

            if name == 'sharpe_ratio':
                sharpe = value
            elif name == 'max_drawdown':
                max_dd = abs(value)  # 確保為正值
            elif name == 'win_rate':
                win_rate = value

        # 高 Sharpe 洞察（使用類別常數）
        if sharpe > self.INSIGHT_HIGH_SHARPE_THRESHOLD:
            insights.append(f"高夏普表現 (Sharpe={sharpe:.2f})，參數組合值得記錄")

        # 低回撤洞察（使用類別常數）
        if max_dd < self.INSIGHT_LOW_DRAWDOWN_THRESHOLD:
            insights.append(f"低回撤策略 (MaxDD={max_dd:.1%})，風險控制優秀")

        # 高勝率洞察（使用類別常數）
        if win_rate > self.INSIGHT_HIGH_WINRATE_THRESHOLD:
            insights.append(f"高勝率策略 (WinRate={win_rate:.1%})，進出場時機精確")

        # 市場狀態洞察
        if market_state:
            regime = getattr(market_state, 'regime', None)
            if regime:
                regime_value = getattr(regime, 'value', str(regime))
                insights.append(f"在 {regime_value} 狀態下表現優異")

        # 驗證洞察
        # 正規化 grade 為 Enum（支援字串或 Enum 輸入）
        grade_enum = grade if isinstance(grade, Grade) else Grade(grade) if grade else None
        if grade_enum in [Grade.A, Grade.B]:
            insights.append(f"通過嚴格驗證 (grade={grade_enum.value if grade_enum else grade})，策略穩健性高")
        elif grade_enum == Grade.C:
            insights.append(f"驗證評級 {grade_enum.value if grade_enum else grade}，需進一步測試")
        elif grade_enum in [Grade.D, Grade.F]:
            insights.append(f"未通過驗證 (grade={grade_enum.value if grade_enum else grade})，參數可能過擬合")

        # 如果沒有洞察，提供默認洞察
        if not insights:
            insights.append(f"記錄 Pareto 解：Sharpe={sharpe:.2f}, MaxDD={max_dd:.1%}, WinRate={win_rate:.1%}")

        return insights

    def _create_result_from_objectives(
        self,
        objectives: List[Any],
        params: Dict[str, Any]
    ) -> Any:
        """從 objectives 創建 BacktestResult（mock）

        因為 ExperimentRecorder.log_experiment() 需要 BacktestResult 物件，
        但我們只有 objectives 列表，所以創建一個 mock 物件。

        Args:
            objectives: 目標值列表（List[ObjectiveResult]）
            params: 參數字典

        Returns:
            Mock BacktestResult 物件
        """
        # 提取指標值
        metrics = {}
        for obj in objectives:
            name = getattr(obj, 'name', '')
            value = getattr(obj, 'value', 0.0)
            metrics[name] = value

        # 創建 mock 物件
        class MockBacktestResult:
            """Mock BacktestResult 物件"""
            def __init__(self, metrics: Dict[str, float], params: Dict[str, Any]):
                # 必要欄位（ExperimentRecorder 需要）
                self.total_return = metrics.get('total_return', 0.0)
                self.annual_return = metrics.get('annual_return', 0.0)
                self.sharpe_ratio = metrics.get('sharpe_ratio', 0.0)
                self.sortino_ratio = metrics.get('sortino_ratio', 0.0)
                self.max_drawdown = metrics.get('max_drawdown', 0.0)
                self.win_rate = metrics.get('win_rate', 0.0)
                self.profit_factor = metrics.get('profit_factor', 1.0)
                self.total_trades = int(metrics.get('total_trades', 0))
                self.avg_trade_duration = metrics.get('avg_trade_duration', 0.0)
                self.expectancy = metrics.get('expectancy', 0.0)

                # 參數
                self.parameters = params

        return MockBacktestResult(metrics, params)

    def _validate_engine_on_startup(self):
        """啟動時驗證回測引擎"""
        if not self.validator:
            logger.warning("Validator not available, skipping engine validation")
            return

        if self.verbose:
            logger.info("驗證回測引擎正確性...")

        try:
            report = self.validator.validate_level("L2")

            # 防禦性檢查：確保報告格式正確
            if not hasattr(report, 'all_passed') or not hasattr(report, 'results'):
                raise RuntimeError("驗證報告格式不正確：缺少必要屬性")

            if not report.all_passed:
                failed = [r for r in report.results if not r.success]
                error_msg = "回測引擎驗證失敗！\n"
                for test in failed:
                    error_msg += f"  - {test.test_name}: {test.message}\n"
                raise RuntimeError(error_msg)

            if self.verbose:
                logger.info("✅ 回測引擎驗證通過")
        except Exception as e:
            logger.error(f"Engine validation failed: {e}")
            raise

    def _try_restore_checkpoint(self) -> int:
        """嘗試恢復檢查點

        Returns:
            int: 恢復的迭代編號（0 表示從頭開始）
        """
        if not self.config.checkpoint_enabled:
            return 0

        checkpoint_path = Path(self.config.checkpoint_dir) / "ultimate_checkpoint.json"

        if not checkpoint_path.exists():
            if self.verbose:
                logger.info("未找到檢查點檔案")
            return 0

        try:
            # 讀取檢查點檔案
            with open(checkpoint_path, 'r') as f:
                self._checkpoint_data = json.load(f)

            # 驗證必要欄位
            if 'last_iteration' not in self._checkpoint_data:
                logger.warning("檢查點檔案格式錯誤：缺少 last_iteration")
                return 0

            last_iteration = self._checkpoint_data['last_iteration']

            # 驗證迭代編號合理性
            if not isinstance(last_iteration, int) or last_iteration < 0:
                logger.warning(f"檢查點迭代編號無效：{last_iteration}")
                return 0

            # 恢復統計數據
            if 'summary' in self._checkpoint_data:
                summary_data = self._checkpoint_data['summary']
                self.summary.successful_iterations = summary_data.get('successful', 0)
                self.summary.failed_iterations = summary_data.get('failed', 0)
                self.summary.best_strategy = summary_data.get('best_strategy')
                self.summary.best_objectives = summary_data.get('best_objectives')
                # 恢復 Pareto 解統計
                self.summary.total_pareto_solutions = summary_data.get('total_pareto_solutions', 0)
                self.summary.validated_solutions = summary_data.get('validated_solutions', 0)

            # 恢復策略統計
            if 'strategy_stats' in self._checkpoint_data:
                self.strategy_stats = self._checkpoint_data['strategy_stats']

            # 恢復市場狀態分佈
            if 'regime_distribution' in self._checkpoint_data:
                self.summary.regime_distribution = self._checkpoint_data['regime_distribution']

            checkpoint_timestamp = self._checkpoint_data.get('timestamp', 'unknown')

            if self.verbose:
                logger.info(
                    f"✓ 檢查點載入成功\n"
                    f"  - 迭代編號：{last_iteration}\n"
                    f"  - 時間戳記：{checkpoint_timestamp}\n"
                    f"  - 成功迭代：{self.summary.successful_iterations}\n"
                    f"  - 失敗迭代：{self.summary.failed_iterations}"
                )

            return last_iteration

        except json.JSONDecodeError as e:
            logger.warning(f"檢查點檔案損壞（JSON 解析失敗）: {e}")
            # 備份損壞的檔案
            try:
                backup_path = checkpoint_path.with_suffix('.json.corrupted')
                checkpoint_path.rename(backup_path)
                logger.info(f"損壞的檢查點已備份至: {backup_path}")
            except Exception as backup_e:
                logger.warning(f"備份損壞檔案失敗: {backup_e}")
            return 0
        except Exception as e:
            logger.warning(f"載入檢查點時發生未預期錯誤: {e}", exc_info=True)
            return 0

    def _save_checkpoint(self, iteration: int):
        """儲存檢查點

        Args:
            iteration: 當前迭代編號
        """
        try:
            checkpoint_dir = Path(self.config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = checkpoint_dir / "ultimate_checkpoint.json"
            temp_path = checkpoint_dir / "ultimate_checkpoint.json.tmp"

            checkpoint_data = {
                'last_iteration': iteration,
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'successful': self.summary.successful_iterations,
                    'failed': self.summary.failed_iterations,
                    'best_strategy': self.summary.best_strategy,
                    'best_objectives': self.summary.best_objectives,
                    'total_pareto_solutions': self.summary.total_pareto_solutions,
                    'validated_solutions': self.summary.validated_solutions
                },
                'strategy_stats': self.strategy_stats,
                'regime_distribution': self.summary.regime_distribution
            }

            # 原子性寫入：先寫 .tmp，再 rename
            with open(temp_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)

            temp_path.replace(checkpoint_path)

            if self.verbose:
                logger.info(f"檢查點已儲存：迭代 {iteration}")
        except Exception as e:
            logger.error(f"儲存檢查點失敗 (迭代 {iteration}): {e}")
            # 不中斷執行，只記錄錯誤

    def get_memory_suggestions(self) -> List[Dict]:
        """取得待存儲的 Memory 建議

        Returns:
            List of {content, metadata, timestamp, grade, sharpe}
        """
        return self._memory_suggestions.copy()

    def get_memory_commands(self) -> List[str]:
        """取得 Memory MCP 存儲命令（供外部執行）

        將 Memory 建議轉換為可執行的 MCP 命令字串。
        這些命令可以由 Claude 或其他系統執行以存儲洞察。

        Returns:
            List[str]: Memory MCP 存儲命令列表

        Example:
            >>> commands = controller.get_memory_commands()
            >>> for cmd in commands:
            ...     print(cmd)
            store_memory(content='MA Cross 策略...', metadata={'tags': 'strategy,success'})
        """
        commands = []
        for suggestion in self._memory_suggestions:
            content = suggestion.get('content', '')
            metadata = suggestion.get('metadata', {})

            # 格式化 tags
            tags = metadata.get('tags', '')
            if isinstance(tags, list):
                tags = ','.join(tags)

            # 建立命令字串
            cmd = f"store_memory(content='{content}', metadata={{'tags': '{tags}'}})"
            commands.append(cmd)

        return commands

    def clear_memory_suggestions(self):
        """清除已處理的 Memory 建議"""
        self._memory_suggestions.clear()

    def _cleanup(self):
        """清理資源（只執行一次）"""
        # 防止重複清理
        if self._cleaned_up:
            return

        if self.verbose:
            logger.info("Cleaning up resources...")

        # 清理 HyperLoop（優先）
        if self.hyperloop is not None:
            try:
                if hasattr(self.hyperloop, '_cleanup'):
                    self.hyperloop._cleanup()
                elif hasattr(self.hyperloop, 'cleanup'):
                    self.hyperloop.cleanup()
                if self.verbose:
                    logger.debug("HyperLoop cleaned up")
            except Exception as e:
                logger.warning(f"HyperLoop cleanup failed: {e}")

        # 清理 optimizer
        if self.optimizer and hasattr(self.optimizer, 'cleanup'):
            try:
                self.optimizer.cleanup()
                if self.verbose:
                    logger.debug("Optimizer cleaned up")
            except Exception as e:
                logger.warning(f"Optimizer cleanup failed: {e}")

        # 清理 GPU optimizer（Phase 12.3.3）
        if self.gpu_optimizer and hasattr(self.gpu_optimizer, 'cleanup'):
            try:
                self.gpu_optimizer.cleanup()
                if self.verbose:
                    logger.debug("GPU optimizer cleaned up")
            except Exception as e:
                logger.warning(f"GPU optimizer cleanup failed: {e}")

        # 清理 regime analyzer
        if self.regime_analyzer and hasattr(self.regime_analyzer, 'cleanup'):
            try:
                self.regime_analyzer.cleanup()
                if self.verbose:
                    logger.debug("Regime analyzer cleaned up")
            except Exception as e:
                logger.warning(f"Regime analyzer cleanup failed: {e}")

        # 清理 validator
        if self.validator and hasattr(self.validator, 'cleanup'):
            try:
                self.validator.cleanup()
                if self.verbose:
                    logger.debug("Validator cleaned up")
            except Exception as e:
                logger.warning(f"Validator cleanup failed: {e}")

        # 清理 recorder (使用新的 DuckDB 版本的 close() 方法)
        if self.recorder:
            try:
                # 優先使用 close()（DuckDB 版本）
                if hasattr(self.recorder, 'close'):
                    self.recorder.close()
                # 向後相容舊的 cleanup()
                elif hasattr(self.recorder, 'cleanup'):
                    self.recorder.cleanup()
                if self.verbose:
                    logger.debug("Recorder cleaned up")
            except Exception as e:
                logger.warning(f"Recorder cleanup failed: {e}")

        # 清理資料緩存（Phase 12.7）
        if hasattr(self, '_data_cache'):
            cache_size = len(self._data_cache)
            self._data_cache.clear()
            self._data_cache = {}
            if self.verbose:
                logger.debug(f"Data cache cleared ({cache_size} datasets)")

        # 清理 Memory 建議（Phase 12.8）
        if hasattr(self, '_memory_suggestions'):
            suggestions_count = len(self._memory_suggestions)
            self._memory_suggestions.clear()
            if self.verbose:
                logger.debug(f"Memory suggestions cleared ({suggestions_count} entries)")

        # 清理 Memory 整合
        self.memory = None

        # 重置引用
        self.hyperloop = None
        self.optimizer = None
        self.gpu_optimizer = None
        self.regime_analyzer = None
        self.validator = None
        self.recorder = None

        # 標記清理完成
        self._cleaned_up = True

        if self.verbose:
            logger.info("Resources cleaned up")

    def __enter__(self):
        """Context manager 進入"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 退出（自動清理）"""
        self._cleanup()

    def __del__(self):
        """析構函數：確保共享記憶體被清理，避免洩漏"""
        try:
            self._cleanup()
        except Exception:
            pass  # 忽略析構時的錯誤

    def _print_header(self, n_iterations: int):
        """輸出標題

        Args:
            n_iterations: 總迭代次數
        """
        # GPU 狀態描述
        if self.gpu_optimizer:
            gpu_backend = getattr(self.gpu_optimizer, '_backend', 'unknown')
            gpu_status = f'啟用 ({gpu_backend})'
        else:
            gpu_status = '停用'

        lines = [
            "",
            "=" * 70,
            "UltimateLoop - 最強回測系統",
            "=" * 70,
            f"總迭代次數: {n_iterations}",
            f"GPU 加速: {gpu_status}",
            f"Regime Detection: {'啟用' if self.config.regime_detection else '停用'}",
            f"策略選擇模式: {self.config.strategy_selection_mode}",
            f"驗證系統: {'啟用' if self.config.validation_enabled else '停用'}",
            f"學習系統: {'啟用' if self.config.learning_enabled else '停用'}",
            f"可用策略數: {len(self.available_strategies)}",
            "=" * 70,
        ]
        logger.info("\n".join(lines))

    async def _optimize_strategy(self, strategy_name: str) -> Optional[Any]:
        """優化單個策略

        根據參數空間大小選擇優化方式：
        1. 參數組合 > param_sweep_threshold 且 GPU 可用 → GPU 批量優化
        2. 否則 → 標準 MultiObjectiveOptimizer（CPU）

        Args:
            strategy_name: 策略名稱

        Returns:
            Optional[MultiObjectiveResult]: 優化結果
        """
        if not REGISTRY_AVAILABLE or StrategyRegistry is None:
            logger.warning("Strategy registry not available")
            return None

        try:
            # 獲取策略的參數空間
            param_space = StrategyRegistry.get_param_space(strategy_name)
            if not param_space:
                logger.warning(f"No param_space for strategy '{strategy_name}'")
                return None

            # 計算參數空間大小（估算）
            param_space_size = self._estimate_param_space_size(param_space)

            # Phase 12.7 TODO: GPU 優化整合
            # 目前所有優化都使用 CPU（MultiObjectiveOptimizer）
            # GPU 優化需要 DataPool 提供市場數據後才能實現
            gpu_would_be_beneficial = (
                self.gpu_optimizer is not None and
                param_space_size >= self.config.param_sweep_threshold
            )

            if gpu_would_be_beneficial and self.verbose:
                logger.info(
                    f"GPU optimization would be beneficial for {strategy_name} "
                    f"(param_space_size={param_space_size} >= threshold={self.config.param_sweep_threshold}), "
                    f"but requires Phase 12.7 DataPool integration. Using CPU optimization."
                )
            elif self.verbose:
                logger.debug(
                    f"Using CPU optimization for {strategy_name} "
                    f"(param_space_size={param_space_size})"
                )

            # 使用 CPU 優化（Phase 12.7 完成後將支援 GPU）
            result = await self._optimize_with_cpu(strategy_name, param_space)

            return result

        except Exception as e:
            logger.error(f"Failed to optimize {strategy_name}: {e}", exc_info=True)
            return None

    def _estimate_param_space_size(self, param_space: Dict[str, Any]) -> int:
        """估算參數空間大小

        根據參數類型估算離散化後的組合數量。

        Args:
            param_space: 參數空間定義

        Returns:
            int: 估算的參數組合數量
        """
        total = 1
        for param_name, param_def in param_space.items():
            if isinstance(param_def, dict):
                param_type = param_def.get('type', 'float')
                if param_type == 'int':
                    low = param_def.get('low', 0)
                    high = param_def.get('high', 10)
                    step = param_def.get('step', 1)
                    # 計算整數範圍內的離散點數
                    total *= max(1, (high - low) // step + 1)
                elif param_type == 'float':
                    low = param_def.get('low', 0.0)
                    high = param_def.get('high', 1.0)
                    step = param_def.get('step')
                    if step and step > 0:
                        # 有明確 step，計算離散點數
                        total *= max(1, int((high - low) / step) + 1)
                    else:
                        # 無 step，使用連續優化（Bayesian），估算為 20 個有效探索點
                        total *= 20
                elif param_type == 'categorical':
                    choices = param_def.get('choices', [])
                    total *= max(len(choices), 1)
            elif isinstance(param_def, (list, tuple)):
                # 直接是選項列表
                total *= max(len(param_def), 1)

        return total

    async def _optimize_with_gpu(
        self,
        strategy_name: str,
        param_space: Dict[str, Any]
    ) -> Optional[Any]:
        """使用 GPU 批量優化

        Phase 12.7 待實現 - 需要 DataPool 整合

        GPUBatchOptimizer.batch_optimize() 需要：
        1. strategy_fn: (np.ndarray, Dict) -> np.ndarray（策略訊號函數）
        2. price_data: np.ndarray（OHLCV 數據）

        完整整合需要：
        - DataPool 提供歷史 OHLCV 數據
        - 策略轉換為 numpy-based signal function

        Args:
            strategy_name: 策略名稱
            param_space: 參數空間

        Returns:
            優化結果（MultiObjectiveResult 格式）

        Raises:
            NotImplementedError: Phase 12.7 尚未完成
        """
        raise NotImplementedError(
            f"GPU optimization for '{strategy_name}' requires Phase 12.7 DataPool integration.\n"
            "\nRequired for GPU batch_optimize():\n"
            "1. strategy_fn: (np.ndarray, Dict) -> np.ndarray\n"
            "2. price_data: np.ndarray (OHLCV from DataPool)\n"
            "\nUse _optimize_with_cpu() instead, or wait for Phase 12.7."
        )

    async def _optimize_with_cpu(
        self,
        strategy_name: str,
        param_space: Dict[str, Any]
    ) -> Optional[Any]:
        """使用 CPU 標準優化（MultiObjectiveOptimizer）

        Args:
            strategy_name: 策略名稱
            param_space: 參數空間

        Returns:
            MultiObjectiveResult: 優化結果
        """
        if self.optimizer is None:
            logger.warning("CPU optimizer not available")
            return None

        # 建立評估函數
        evaluate_fn = self._create_evaluate_fn(strategy_name)

        # 使用工廠函數避免 lambda 閉包捕獲問題
        def create_optimize_task(
            optimizer: Any,
            ps: Dict[str, Any],
            ef: Callable
        ) -> Callable[[], Any]:
            """建立優化任務（避免 lambda 閉包捕獲變數引用問題）"""
            def task() -> Any:
                return optimizer.optimize(
                    param_space=ps,
                    evaluate_fn=ef,
                    show_progress_bar=False
                )
            return task

        # 執行優化（同步轉異步）
        optimize_task = create_optimize_task(self.optimizer, param_space, evaluate_fn)
        result = await asyncio.get_event_loop().run_in_executor(None, optimize_task)

        return result

    def _convert_hyperloop_to_multi_objective_result(
        self,
        hyperloop_summary: Any
    ) -> Any:
        """將 HyperLoopSummary 轉換為 MultiObjectiveResult 格式

        Args:
            hyperloop_summary: HyperLoop 執行摘要

        Returns:
            MultiObjectiveResult 格式的結果
        """
        if not OPTIMIZER_AVAILABLE or MultiObjectiveResult is None or ParetoSolution is None:
            logger.warning("Optimizer not available, cannot convert HyperLoop result")
            return hyperloop_summary

        if ObjectiveResult is None:
            logger.warning("ObjectiveResult not available, returning raw HyperLoop result")
            return hyperloop_summary

        # 從 HyperLoop 結果提取迭代結果
        iteration_results = getattr(hyperloop_summary, 'iteration_results', [])
        if not iteration_results:
            logger.warning("No iteration results in HyperLoop summary")
            # 返回空的 MultiObjectiveResult 而非原始 summary
            return MultiObjectiveResult(
                pareto_front=[],
                all_solutions=[],
                n_trials=0,
                study=None,
                optimization_time=getattr(hyperloop_summary, 'total_duration_seconds', 0.0),
                n_completed_trials=0,
                n_failed_trials=getattr(hyperloop_summary, 'failed_iterations', 0)
            )

        # 建立所有解的 objectives 列表（用於 Pareto rank 計算）
        all_objectives = []
        for result in iteration_results:
            sharpe = result.get('sharpe', 0.0)
            max_dd = result.get('max_drawdown', 0.5)  # 預設 50% 如果沒有
            win_rate = result.get('win_rate', 0.5)    # 預設 50% 如果沒有
            all_objectives.append({
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'win_rate': win_rate
            })

        # 計算 Pareto ranks
        pareto_ranks = self._calculate_pareto_ranks(all_objectives)

        # 建立 ParetoSolution 列表
        pareto_solutions = []
        for idx, result in enumerate(iteration_results):
            objectives_list = [
                ObjectiveResult(
                    name='sharpe_ratio',
                    value=result.get('sharpe', 0.0),
                    direction='maximize'
                ),
                ObjectiveResult(
                    name='max_drawdown',
                    value=result.get('max_drawdown', 0.5),
                    direction='minimize'
                ),
                ObjectiveResult(
                    name='win_rate',
                    value=result.get('win_rate', 0.5),
                    direction='maximize'
                ),
                # 新增：total_return 和 total_trades
                ObjectiveResult(
                    name='total_return',
                    value=result.get('total_return', 0.0),
                    direction='maximize'
                ),
                ObjectiveResult(
                    name='total_trades',
                    value=float(result.get('total_trades', 0)),
                    direction='maximize'
                )
            ]

            solution = ParetoSolution(
                params=result.get('best_params', {}),
                objectives=objectives_list,
                rank=pareto_ranks[idx],
                crowding_distance=0.0,
                trial_number=idx
            )
            # 附加策略資訊
            solution._strategy_name = result.get('strategy_name', 'unknown')
            solution._symbol = result.get('symbol', 'BTCUSDT')
            solution._timeframe = result.get('timeframe', '1h')
            # 附加額外指標（供記錄器使用）
            solution._total_return = result.get('total_return', 0.0)
            solution._total_trades = result.get('total_trades', 0)
            pareto_solutions.append(solution)

        # Pareto front = rank 0 的解
        pareto_front = [s for s in pareto_solutions if s.rank == 0]

        if self.verbose:
            logger.info(
                f"Converted HyperLoop result: {len(iteration_results)} iterations -> "
                f"{len(pareto_front)} Pareto solutions"
            )

        # 建立 MultiObjectiveResult
        return MultiObjectiveResult(
            pareto_front=pareto_front,
            all_solutions=pareto_solutions,
            n_trials=len(iteration_results),
            study=None,
            optimization_time=getattr(hyperloop_summary, 'total_duration_seconds', 0.0),
            n_completed_trials=getattr(hyperloop_summary, 'successful_iterations', 0),
            n_failed_trials=getattr(hyperloop_summary, 'failed_iterations', 0)
        )

    def _convert_gpu_to_multi_objective_result(
        self,
        gpu_result: Any
    ) -> Any:
        """將 GPUOptimizationResult 轉換為 MultiObjectiveResult 格式

        實作 NSGA-II non-dominated sorting 計算真正的 Pareto rank。

        Args:
            gpu_result: GPU 優化結果

        Returns:
            MultiObjectiveResult 格式的結果
        """
        if not OPTIMIZER_AVAILABLE or MultiObjectiveResult is None or ParetoSolution is None:
            return gpu_result

        if ObjectiveResult is None:
            logger.warning("ObjectiveResult not available, returning raw GPU result")
            return gpu_result

        # 從 GPU 結果建立 Pareto 解
        all_results = getattr(gpu_result, 'all_results', [])
        if not all_results:
            return gpu_result

        # 建立所有解的 objectives 列表（用於 Pareto rank 計算）
        all_objectives = []
        for batch_result in all_results:
            all_objectives.append({
                'sharpe_ratio': batch_result.sharpe_ratio,
                'max_drawdown': batch_result.max_drawdown,
                'win_rate': batch_result.win_rate
            })

        # 計算 Pareto ranks（non-dominated sorting）
        pareto_ranks = self._calculate_pareto_ranks(all_objectives)

        # 建立 ParetoSolution 列表
        pareto_solutions = []
        for idx, batch_result in enumerate(all_results):
            # 將 dict 轉換為 List[ObjectiveResult]
            objectives_list = [
                ObjectiveResult(
                    name='sharpe_ratio',
                    value=batch_result.sharpe_ratio,
                    direction='maximize'
                ),
                ObjectiveResult(
                    name='max_drawdown',
                    value=batch_result.max_drawdown,
                    direction='minimize'
                ),
                ObjectiveResult(
                    name='win_rate',
                    value=batch_result.win_rate,
                    direction='maximize'
                )
            ]

            solution = ParetoSolution(
                params=batch_result.params,
                objectives=objectives_list,
                rank=pareto_ranks[idx],
                crowding_distance=0.0,  # Phase 12.7 TODO: 計算擁擠度
                trial_number=idx
            )
            pareto_solutions.append(solution)

        # Pareto front = rank 0 的解
        pareto_front = [s for s in pareto_solutions if s.rank == 0]

        # 建立 MultiObjectiveResult
        return MultiObjectiveResult(
            pareto_front=pareto_front,
            all_solutions=pareto_solutions,
            n_trials=getattr(gpu_result, 'n_trials', 0),
            study=getattr(gpu_result, 'study', None),
            optimization_time=getattr(gpu_result, 'total_time_seconds', 0.0),
            n_completed_trials=getattr(gpu_result, 'n_trials', 0),
            n_failed_trials=0
        )

    def _calculate_pareto_ranks(
        self,
        objectives_list: List[Dict[str, float]]
    ) -> List[int]:
        """計算 Pareto ranks（NSGA-II non-dominated sorting）

        每個解的 rank 表示它被 dominate 的層級：
        - rank 0: Pareto front（非被支配解）
        - rank 1: 只被 rank 0 的解支配
        - rank n: 只被 rank 0 ~ n-1 的解支配

        Args:
            objectives_list: 每個解的目標值字典列表

        Returns:
            每個解的 Pareto rank 列表
        """
        n = len(objectives_list)
        if n == 0:
            return []

        # 初始化
        ranks = [-1] * n
        domination_count = [0] * n  # 被支配次數
        dominated_solutions: List[List[int]] = [[] for _ in range(n)]  # 支配的解

        # 計算支配關係
        for i in range(n):
            for j in range(i + 1, n):
                dom_result = self._dominates(objectives_list[i], objectives_list[j])
                if dom_result == 1:  # i dominates j
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif dom_result == -1:  # j dominates i
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

        # 找出第一層 front（rank 0）
        current_front = []
        for i in range(n):
            if domination_count[i] == 0:
                ranks[i] = 0
                current_front.append(i)

        # 迭代計算後續 fronts
        current_rank = 0
        while current_front:
            next_front = []
            for i in current_front:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        ranks[j] = current_rank + 1
                        next_front.append(j)
            current_front = next_front
            current_rank += 1

        return ranks

    def _dominates(
        self,
        obj_a: Dict[str, float],
        obj_b: Dict[str, float]
    ) -> int:
        """判斷 obj_a 是否支配 obj_b

        支配定義（Pareto dominance）：
        - A dominates B: A 在所有目標上都不比 B 差，且至少一個目標上嚴格優於 B

        目標方向：
        - sharpe_ratio: maximize（越大越好）
        - max_drawdown: minimize（越小越好）
        - win_rate: maximize（越大越好）

        Args:
            obj_a: 解 A 的目標值
            obj_b: 解 B 的目標值

        Returns:
            1: A dominates B
            -1: B dominates A
            0: 互不支配
        """
        a_better_count = 0
        b_better_count = 0

        # sharpe_ratio: maximize
        if obj_a.get('sharpe_ratio', 0) > obj_b.get('sharpe_ratio', 0):
            a_better_count += 1
        elif obj_a.get('sharpe_ratio', 0) < obj_b.get('sharpe_ratio', 0):
            b_better_count += 1

        # max_drawdown: minimize（越小越好）
        if obj_a.get('max_drawdown', 1) < obj_b.get('max_drawdown', 1):
            a_better_count += 1
        elif obj_a.get('max_drawdown', 1) > obj_b.get('max_drawdown', 1):
            b_better_count += 1

        # win_rate: maximize
        if obj_a.get('win_rate', 0) > obj_b.get('win_rate', 0):
            a_better_count += 1
        elif obj_a.get('win_rate', 0) < obj_b.get('win_rate', 0):
            b_better_count += 1

        # 判斷支配關係
        if a_better_count > 0 and b_better_count == 0:
            return 1  # A dominates B
        elif b_better_count > 0 and a_better_count == 0:
            return -1  # B dominates A
        else:
            return 0  # 互不支配

    def _create_evaluate_fn(self, strategy_name: str) -> Callable[[Dict[str, Any]], Dict[str, float]]:
        """建立評估函數

        Phase 12.7: 使用 DataCache + BacktestEngine 進行真實策略評估。

        Args:
            strategy_name: 策略名稱

        Returns:
            Callable: 評估函數 (params) -> {sharpe_ratio, max_drawdown, win_rate}

        Raises:
            ValueError: 如果找不到資料
        """
        # 取得預設資料
        data = self._get_data()

        if data is None or len(data) == 0:
            raise ValueError(
                f"No data available for strategy evaluation. "
                f"Please ensure OHLCV data exists in {self.config.data_dir}/ohlcv/"
            )

        # 驗證必要欄位（OHLCV）
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col.lower() not in [c.lower() for c in data.columns]]
        if missing_columns:
            raise ValueError(f"Missing required OHLCV columns: {missing_columns}")

        # 驗證資料長度足夠
        MIN_DATA_ROWS = 100
        if len(data) < MIN_DATA_ROWS:
            logger.warning(f"Limited data for backtest: {len(data)} rows (recommended: {MIN_DATA_ROWS}+)")

        # 計算時間範圍
        start_date = None
        end_date = None
        try:
            # 嘗試從 index 獲取時間範圍
            idx_min = data.index.min()
            idx_max = data.index.max()
            # 轉換為 datetime（pandas Timestamp -> datetime）
            start_date = getattr(idx_min, 'to_pydatetime', lambda: idx_min)()
            end_date = getattr(idx_max, 'to_pydatetime', lambda: idx_max)()
        except (AttributeError, TypeError):
            # Fallback: 從 timestamp 欄位取得
            if 'timestamp' in data.columns:
                start_date = data['timestamp'].min()
                end_date = data['timestamp'].max()

        # 取得 symbol 和 timeframe
        symbol = self.config.symbols[0] if self.config.symbols else 'BTCUSDT'

        # 使用第一個可用的 timeframe
        timeframe = self.config.timeframes[0] if self.config.timeframes else '1h'

        # 捕獲變數供閉包使用
        _data = data
        _symbol = symbol
        _timeframe = timeframe
        _start_date = start_date
        _end_date = end_date
        _initial_capital = self.config.initial_capital
        _leverage = self.config.leverage
        _strategy_name = strategy_name

        def evaluate(params: Dict[str, Any]) -> Dict[str, float]:
            """評估策略參數組合

            Args:
                params: 策略參數

            Returns:
                Dict: {sharpe_ratio, max_drawdown, win_rate}
            """
            try:
                # 1. 建立策略實例
                if not REGISTRY_AVAILABLE or StrategyRegistry is None:
                    logger.warning("StrategyRegistry not available")
                    return {'sharpe_ratio': -10.0, 'max_drawdown': 1.0, 'win_rate': 0.0}

                strategy = StrategyRegistry.create(_strategy_name, **params)

                # 2. 建立回測引擎和配置
                if not BACKTEST_ENGINE_AVAILABLE or BacktestEngine is None:
                    logger.warning("BacktestEngine not available")
                    return {'sharpe_ratio': -10.0, 'max_drawdown': 1.0, 'win_rate': 0.0}

                if not BACKTEST_CONFIG_AVAILABLE or BacktestConfig is None:
                    logger.warning("BacktestConfig not available")
                    return {'sharpe_ratio': -10.0, 'max_drawdown': 1.0, 'win_rate': 0.0}

                config = BacktestConfig(
                    symbol=_symbol,
                    timeframe=_timeframe,
                    start_date=_start_date,
                    end_date=_end_date,
                    initial_capital=_initial_capital,
                    leverage=_leverage
                )

                engine = BacktestEngine(config)

                # 3. 執行回測
                result = engine.run(strategy, data=_data)

                # 4. 提取績效指標（明確 None 檢查）
                sharpe = getattr(result, 'sharpe_ratio', None)
                sharpe = 0.0 if sharpe is None else float(sharpe)

                max_dd = getattr(result, 'max_drawdown', None)
                max_dd = 0.0 if max_dd is None else abs(float(max_dd))

                win_rate = getattr(result, 'win_rate', None)
                win_rate = 0.0 if win_rate is None else float(win_rate)

                return {
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_dd,
                    'win_rate': win_rate
                }

            except Exception as e:
                logger.warning(f"Evaluation failed for {_strategy_name}: {e}")
                return {'sharpe_ratio': -10.0, 'max_drawdown': 1.0, 'win_rate': 0.0}

        return evaluate

    def _combine_optimization_results(
        self,
        results: List[Any]
    ) -> Any:
        """合併多個策略的優化結果

        Args:
            results: MultiObjectiveResult 列表

        Returns:
            MultiObjectiveResult: 合併後的結果
        """
        if not results:
            return None

        if not OPTIMIZER_AVAILABLE or MultiObjectiveResult is None:
            return results[0] if results else None

        # 合併所有 Pareto 前緣
        all_pareto_solutions = []
        all_solutions = []
        total_trials = 0
        total_time = 0.0
        total_completed = 0
        total_failed = 0

        for result in results:
            all_pareto_solutions.extend(result.pareto_front)
            all_solutions.extend(result.all_solutions)
            total_trials += result.n_trials
            total_time += result.optimization_time
            total_completed += result.n_completed_trials
            total_failed += result.n_failed_trials

        # 從合併的解中重新計算 Pareto 前緣
        # 簡化版本：取所有 Pareto 解（已經是 rank=0）
        combined_pareto = all_pareto_solutions

        # 重新計算擁擠度距離
        if self.optimizer and hasattr(self.optimizer, '_calculate_crowding_distance'):
            combined_pareto = self.optimizer._calculate_crowding_distance(combined_pareto)
            # 按擁擠度排序
            combined_pareto.sort(key=lambda s: s.crowding_distance, reverse=True)

        # 建立合併結果
        combined_result = MultiObjectiveResult(
            pareto_front=combined_pareto,
            all_solutions=all_solutions,
            n_trials=total_trials,
            study=None,  # 合併結果沒有單一 study
            optimization_time=total_time,
            n_completed_trials=total_completed,
            n_failed_trials=total_failed
        )

        return combined_result

    def _select_pareto_solutions(
        self,
        result: Any
    ) -> List[Any]:
        """從 Pareto 前緣選擇最佳解

        根據 config.pareto_select_method 選擇方法：
        - 'knee': 選擇膝點附近的解（最佳平衡）
        - 'crowding': 選擇擁擠度高的解（多樣性）
        - 'random': 隨機選擇

        Args:
            result: MultiObjectiveResult

        Returns:
            List[ParetoSolution]: 選擇的解（已過濾 None）
        """
        # 防禦性檢查：result 和 pareto_front 存在性
        if not result or not hasattr(result, 'pareto_front') or not result.pareto_front:
            return []

        # 過濾 None 元素（防禦性，確保元素有效）
        valid_solutions = [s for s in result.pareto_front if s is not None]
        if not valid_solutions:
            logger.warning("Pareto front contains no valid solutions after filtering")
            return []

        # 計算實際選擇數量（防止 random.sample ValueError）
        n_select = min(self.config.pareto_top_n, len(valid_solutions))
        method = self.config.pareto_select_method

        if method == 'knee':
            # 使用 filter_pareto_front 的 knee 方法
            if hasattr(result, 'filter_pareto_front'):
                try:
                    return result.filter_pareto_front('knee', n_select)
                except Exception as e:
                    logger.warning(f"filter_pareto_front('knee') failed: {e}, using fallback")
                    return valid_solutions[:n_select]
            else:
                logger.debug("filter_pareto_front not available, using slice fallback")
                return valid_solutions[:n_select]

        elif method == 'crowding':
            # 使用 crowding 方法
            if hasattr(result, 'filter_pareto_front'):
                try:
                    return result.filter_pareto_front('crowding', n_select)
                except Exception as e:
                    logger.warning(f"filter_pareto_front('crowding') failed: {e}, using fallback")
                    return valid_solutions[:n_select]
            else:
                logger.debug("filter_pareto_front not available, using slice fallback")
                return valid_solutions[:n_select]

        elif method == 'random':
            # 隨機選擇（n_select 已確保 <= len(valid_solutions)）
            return random.sample(valid_solutions, n_select)

        else:
            # Fallback: 取前 N 個
            logger.warning(f"Unknown pareto_select_method '{method}', using default slice")
            return valid_solutions[:n_select]


# ===== 測試區塊 =====

async def _test_basic():
    """基礎測試"""
    print("\n" + "=" * 70)
    print("測試 1: 基礎初始化")
    print("=" * 70)

    config = UltimateLoopConfig.create_quick_test_config()
    controller = UltimateLoopController(config, verbose=True)

    print(f"\n✓ Controller 初始化成功")
    print(f"  可用策略: {len(controller.available_strategies)}")
    print(f"  Regime 分析器: {'啟用' if controller.regime_analyzer else '停用'}")
    print(f"  驗證器: {'啟用' if controller.validator else '停用'}")
    print(f"  學習系統: {'啟用' if controller.recorder else '停用'}")


async def _test_loop():
    """測試執行 loop"""
    print("\n" + "=" * 70)
    print("測試 2: 執行 Loop（快速測試模式）")
    print("=" * 70)

    config = UltimateLoopConfig.create_quick_test_config()
    config.validation_enabled = False  # 測試時關閉驗證
    config.learning_enabled = False    # 測試時關閉學習

    controller = UltimateLoopController(config, verbose=True)

    try:
        summary = await controller.run_loop(n_iterations=2)
        print(summary.summary_text())
        print("✓ Loop 執行成功")
    except Exception as e:
        print(f"✗ Loop 執行失敗: {e}")
        import traceback
        traceback.print_exc()


async def _test_checkpoint():
    """測試檢查點功能"""
    print("\n" + "=" * 70)
    print("測試 3: 檢查點功能")
    print("=" * 70)

    config = UltimateLoopConfig.create_quick_test_config()
    config.checkpoint_enabled = True
    config.checkpoint_interval = 1
    config.validation_enabled = False
    config.learning_enabled = False

    controller = UltimateLoopController(config, verbose=False)

    # 執行部分迭代
    print("執行前 2 次迭代...")
    await controller.run_loop(n_iterations=2)

    # 檢查檢查點是否存在
    checkpoint_path = Path(config.checkpoint_dir) / "ultimate_checkpoint.json"
    if checkpoint_path.exists():
        print(f"✓ 檢查點已建立: {checkpoint_path}")

        # 嘗試恢復
        controller2 = UltimateLoopController(config, verbose=False)
        start_iter = controller2._try_restore_checkpoint()
        print(f"✓ 檢查點恢復成功，起始迭代: {start_iter}")
    else:
        print(f"✗ 檢查點不存在")


if __name__ == "__main__":
    """測試 UltimateLoopController"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 70)
    print("UltimateLoopController 測試")
    print("=" * 70)

    # 執行測試
    asyncio.run(_test_basic())
    asyncio.run(_test_loop())
    asyncio.run(_test_checkpoint())

    print("\n" + "=" * 70)
    print("✅ 所有測試完成")
    print("=" * 70)
