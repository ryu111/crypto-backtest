"""
Benchmark Framework - 效能測試工具

提供高精度計時、記憶體追蹤、基準測試套件等功能。

Components:
- BenchmarkTimer: 高精度計時器
- MemoryTracker: 記憶體使用追蹤
- BenchmarkSuite: 基準測試套件
- BenchmarkReport: 測試報告
- DataFrameRunner: DataFrame 操作效能測試
- EngineRunner: 回測引擎效能測試
- GPURunner: GPU 加速效能測試
"""

from .framework import (
    BenchmarkTimer,
    MemoryTracker,
    TimingResult,
    MemoryResult,
    BenchmarkSuite,
    BenchmarkReport,
)

from .runners import (
    DataFrameRunner,
    EngineRunner,
    GPURunner,
    generate_ohlcv_data,
    run_all_benchmarks,
)

__all__ = [
    # Framework
    "BenchmarkTimer",
    "MemoryTracker",
    "TimingResult",
    "MemoryResult",
    "BenchmarkSuite",
    "BenchmarkReport",
    # Runners
    "DataFrameRunner",
    "EngineRunner",
    "GPURunner",
    "generate_ohlcv_data",
    "run_all_benchmarks",
]
