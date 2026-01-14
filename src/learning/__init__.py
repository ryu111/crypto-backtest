"""
AI Learning System

跨專案知識存儲與檢索系統，使用 Memory MCP 服務。
實驗記錄與洞察萃取系統。
"""

# 先 import recorder（無依賴）
from .recorder import ExperimentRecorder

# 向後相容：Experiment 已移到 src.types.ExperimentRecord
from src.types import ExperimentRecord as Experiment

# 可選：import memory（有外部依賴）
try:
    from .memory import (
        MemoryIntegration,
        StrategyInsight,
        MarketInsight,
        TradingLesson,
        MemoryTags,
        create_memory_integration,
        store_successful_experiment,
        store_failed_experiment,
        retrieve_best_params_guide
    )

    __all__ = [
        # Experiment Recorder
        'ExperimentRecorder',
        'Experiment',

        # Memory MCP
        'MemoryIntegration',
        'StrategyInsight',
        'MarketInsight',
        'TradingLesson',
        'MemoryTags',
        'create_memory_integration',
        'store_successful_experiment',
        'store_failed_experiment',
        'retrieve_best_params_guide',
    ]

except ImportError:
    # 如果 memory.py 有問題，至少 recorder 可以用
    __all__ = [
        'ExperimentRecorder',
        'Experiment',
    ]
