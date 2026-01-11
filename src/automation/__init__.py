"""
Automation 模組

AI Loop 執行控制器和自動化工具。
包含遺傳算法特徵工程、自動策略生成等功能。
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
    'quick_feature_evolution'
]
