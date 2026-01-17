"""
Genetic Programming (GP) 策略演化模組

使用 DEAP 框架實現自動策略生成。

模組結構：
- primitives: GP 原語（指標、比較、邏輯、數學）
- fitness: 適應度函數（與回測引擎整合）
- constraints: 約束函數（深度限制、膨脹控制）
- engine: GP 演化引擎
- converter: 表達式轉換器（轉為 Python 程式碼和策略檔案）
- learning: 學習系統整合
"""

from .primitives import (
    PrimitiveSetFactory,
    Price, Indicator, Signal, Number,
    rsi, ma, ema, atr, macd, bb_upper, bb_lower,
    gt, lt, cross_above, cross_below,
    and_, or_, not_,
    add, sub, mul, protected_div, protected_log,
)

from .fitness import (
    FitnessEvaluator,
    FitnessConfig,
    create_fitness_type,
    create_multi_objective_fitness,
)

from .constraints import (
    ConstraintConfig,
    apply_constraints,
    limit_depth,
    limit_size,
    calculate_complexity_penalty,
    validate_individual,
    count_constants,
)

from .engine import (
    GPEngine,
    EvolutionConfig,
    EvolutionResult,
)

from .converter import (
    ExpressionConverter,
    StrategyGenerator,
)

from .learning import (
    GPLearningIntegration,
)

__all__ = [
    # Primitives
    "PrimitiveSetFactory",
    "Price", "Indicator", "Signal", "Number",
    "rsi", "ma", "ema", "atr", "macd", "bb_upper", "bb_lower",
    "gt", "lt", "cross_above", "cross_below",
    "and_", "or_", "not_",
    "add", "sub", "mul", "protected_div", "protected_log",

    # Fitness
    "FitnessEvaluator",
    "FitnessConfig",
    "create_fitness_type",
    "create_multi_objective_fitness",

    # Constraints
    "ConstraintConfig",
    "apply_constraints",
    "limit_depth",
    "limit_size",
    "calculate_complexity_penalty",
    "validate_individual",
    "count_constants",

    # Engine
    "GPEngine",
    "EvolutionConfig",
    "EvolutionResult",

    # Converter
    "ExpressionConverter",
    "StrategyGenerator",

    # Learning
    "GPLearningIntegration",
]
