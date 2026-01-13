"""
優化器模組

提供策略參數優化和驗證工具。
"""

# 無 VectorBT 依賴的模組（安全導入）
from .portfolio import (
    PortfolioOptimizer,
    PortfolioWeights
)
from .gpu_batch import (
    GPUBatchOptimizer,
    GPUBatchResult,
    GPUOptimizationResult,
    gpu_optimize_strategy
)

__all__ = [
    'PortfolioOptimizer',
    'PortfolioWeights',
    'GPUBatchOptimizer',
    'GPUBatchResult',
    'GPUOptimizationResult',
    'gpu_optimize_strategy',
]

# 有 VectorBT 依賴的模組（延遲導入）
try:
    from .multi_objective import (
        MultiObjectiveOptimizer,
        MultiObjectiveResult,
        ParetoSolution,
        ObjectiveResult,
        optimize_multi_objective
    )
    from .walk_forward import (
        WalkForwardAnalyzer,
        WFAResult,
        WindowResult
    )
    from .bayesian import (
        BayesianOptimizer,
        OptimizationResult,
        optimize_strategy
    )
    __all__.extend([
        'MultiObjectiveOptimizer',
        'MultiObjectiveResult',
        'ParetoSolution',
        'ObjectiveResult',
        'optimize_multi_objective',
        'WalkForwardAnalyzer',
        'WFAResult',
        'WindowResult',
        'BayesianOptimizer',
        'OptimizationResult',
        'optimize_strategy',
    ])
except ImportError:
    # VectorBT 不可用時跳過
    MultiObjectiveOptimizer = None  # type: ignore[assignment,misc]
    MultiObjectiveResult = None  # type: ignore[assignment,misc]
    ParetoSolution = None  # type: ignore[assignment,misc]
    ObjectiveResult = None  # type: ignore[assignment,misc]
    optimize_multi_objective = None  # type: ignore[assignment,misc]
    WalkForwardAnalyzer = None  # type: ignore[assignment,misc]
    WFAResult = None  # type: ignore[assignment,misc]
    WindowResult = None  # type: ignore[assignment,misc]
    BayesianOptimizer = None  # type: ignore[assignment,misc]
    OptimizationResult = None  # type: ignore[assignment,misc]
    optimize_strategy = None  # type: ignore[assignment,misc]
