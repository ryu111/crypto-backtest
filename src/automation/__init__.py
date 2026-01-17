"""
Automation 模組

AI Loop 執行控制器和自動化工具。
包含遺傳算法特徵工程、自動策略生成、執行調度等功能。
"""

from .loop import (
    LoopController,
    LoopMode,
    LoopState,
    IterationResult,
    IterationStatus,
    create_loop_controller
)

from .feature_engineering import (
    Feature,
    FeatureSet,
    BaseFeatureGenerator,
    RandomFeatureGenerator,
    GeneticFeatureEngineer,
    FeatureSelector,
    AutoStrategyGenerator,
    create_feature_engineer,
    quick_feature_evolution
)

from .scheduler import (
    ExecutorType,
    TaskType,
    BacktestTask,
    ExecutionPlan,
    ExecutionScheduler,
    create_scheduler,
    schedule_strategy_optimization
)

from .hyperloop import (
    HyperLoopConfig,
    HyperLoopSummary,
    HyperLoopController,
    IterationTask,
    create_hyperloop,
    run_hyperloop
)

from .loop_config import (
    BacktestLoopConfig,
    SelectionMode,
    ValidationStage,
    IterationSummary,
    LoopResult,
    create_default_config,
    create_quick_config,
    create_production_config
)

from .validation_runner import (
    ValidationRunner,
    ValidationResult,
    StageResult
)

from .backtest_loop import (
    BacktestLoop,
    run_backtest_loop,
    quick_optimize,
    validate_strategy
)

from .gp_loop import (
    GPLoop,
    GPLoopConfig,
    run_gp_evolution
)

__all__ = [
    # Loop 控制
    'LoopController',
    'LoopMode',
    'LoopState',
    'IterationResult',
    'IterationStatus',
    'create_loop_controller',

    # 特徵工程
    'Feature',
    'FeatureSet',
    'BaseFeatureGenerator',
    'RandomFeatureGenerator',
    'GeneticFeatureEngineer',
    'FeatureSelector',
    'AutoStrategyGenerator',
    'create_feature_engineer',
    'quick_feature_evolution',

    # 執行調度
    'ExecutorType',
    'TaskType',
    'BacktestTask',
    'ExecutionPlan',
    'ExecutionScheduler',
    'create_scheduler',
    'schedule_strategy_optimization',

    # HyperLoop 高效能並行
    'HyperLoopConfig',
    'HyperLoopSummary',
    'HyperLoopController',
    'IterationTask',
    'create_hyperloop',
    'run_hyperloop',

    # BacktestLoop 配置與結果
    'BacktestLoopConfig',
    'SelectionMode',
    'ValidationStage',
    'IterationSummary',
    'LoopResult',
    'create_default_config',
    'create_quick_config',
    'create_production_config',

    # ValidationRunner 驗證執行器
    'ValidationRunner',
    'ValidationResult',
    'StageResult',

    # BacktestLoop 使用者 API
    'BacktestLoop',
    'run_backtest_loop',
    'quick_optimize',
    'validate_strategy',

    # GP 演化循環
    'GPLoop',
    'GPLoopConfig',
    'run_gp_evolution',
]
