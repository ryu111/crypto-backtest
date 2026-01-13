"""
Benchmark Framework - Core Implementation

高精度效能測試框架，支援時間和記憶體測量。
"""

import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any, Optional, Generator
import json
import statistics
from contextlib import contextmanager


@dataclass
class TimingResult:
    """計時結果"""

    name: str
    iterations: int
    warmup: int
    mean: float  # 平均時間（秒）
    min: float   # 最小時間（秒）
    max: float   # 最大時間（秒）
    std: float   # 標準差（秒）
    total: float # 總時間（秒）

    def __str__(self) -> str:
        return (
            f"{self.name}: "
            f"mean={self.mean*1000:.2f}ms, "
            f"min={self.min*1000:.2f}ms, "
            f"max={self.max*1000:.2f}ms, "
            f"std={self.std*1000:.2f}ms "
            f"({self.iterations} iterations)"
        )

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "name": self.name,
            "iterations": self.iterations,
            "warmup": self.warmup,
            "mean_ms": self.mean * 1000,
            "min_ms": self.min * 1000,
            "max_ms": self.max * 1000,
            "std_ms": self.std * 1000,
            "total_s": self.total,
        }


@dataclass
class MemoryResult:
    """記憶體測量結果"""

    name: str
    peak_mb: float      # 峰值記憶體（MB）
    net_mb: float       # 淨增記憶體（MB）
    allocations: int    # 分配次數

    def __str__(self) -> str:
        return (
            f"{self.name}: "
            f"peak={self.peak_mb:.2f}MB, "
            f"net={self.net_mb:.2f}MB, "
            f"allocs={self.allocations}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "name": self.name,
            "peak_mb": self.peak_mb,
            "net_mb": self.net_mb,
            "allocations": self.allocations,
        }


class BenchmarkTimer:
    """
    高精度計時器

    使用 time.perf_counter() 進行高精度計時。
    支援 warmup 和多次迭代測量。

    Example:
        timer = BenchmarkTimer(warmup=3, iterations=10)
        result = timer.measure("my_function", my_function, arg1, arg2)
    """

    def __init__(self, warmup: int = 3, iterations: int = 10):
        """
        Args:
            warmup: Warmup 次數（預熱，不計入結果）
            iterations: 測量次數
        """
        if warmup < 0:
            raise ValueError("warmup must be >= 0")
        if iterations < 1:
            raise ValueError("iterations must be >= 1")

        self.warmup = warmup
        self.iterations = iterations

    def measure(
        self,
        name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> TimingResult:
        """
        測量函數執行時間

        Args:
            name: 測試名稱
            func: 要測量的函數
            *args: 函數的位置參數
            **kwargs: 函數的關鍵字參數

        Returns:
            TimingResult: 計時結果
        """
        # Warmup
        for _ in range(self.warmup):
            func(*args, **kwargs)

        # Measure
        times: List[float] = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)

        # Calculate statistics
        mean_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        total_time = sum(times)

        return TimingResult(
            name=name,
            iterations=self.iterations,
            warmup=self.warmup,
            mean=mean_time,
            min=min_time,
            max=max_time,
            std=std_time,
            total=total_time,
        )

    @contextmanager
    def time(self, name: str) -> Generator[None, None, None]:
        """
        Context manager for timing a single code block

        Note: This measures a single execution, not multiple iterations.
        For multiple iterations with statistics, use measure() instead.

        Example:
            timer = BenchmarkTimer()
            with timer.time("my_block"):
                # code to measure
                pass
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            elapsed = end - start
            print(f"{name}: {elapsed*1000:.2f}ms")


class MemoryTracker:
    """
    記憶體使用追蹤器

    使用 tracemalloc 追蹤記憶體使用。
    測量峰值記憶體和淨增記憶體。

    Example:
        tracker = MemoryTracker()
        result = tracker.measure("my_function", my_function, arg1, arg2)
    """

    def measure(
        self,
        name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> MemoryResult:
        """
        測量函數記憶體使用

        Args:
            name: 測試名稱
            func: 要測量的函數
            *args: 函數的位置參數
            **kwargs: 函數的關鍵字參數

        Returns:
            MemoryResult: 記憶體測量結果
        """
        # 檢查是否已經在追蹤（避免重複啟動）
        was_tracing = tracemalloc.is_tracing()

        if not was_tracing:
            tracemalloc.start()

        # Get baseline
        baseline_current, baseline_peak = tracemalloc.get_traced_memory()

        try:
            # Run function
            func(*args, **kwargs)

            # Get final memory
            final_current, final_peak = tracemalloc.get_traced_memory()

            # Get allocation count
            stats = tracemalloc.take_snapshot().statistics('lineno')
            allocations = sum(stat.count for stat in stats)
        finally:
            # 確保一定會 stop（只有當我們啟動的時候才停止）
            if not was_tracing:
                tracemalloc.stop()

        # Calculate results (convert bytes to MB)
        # 峰值差異可能為負（baseline 較高時），使用 max(0, ...) 確保非負
        peak_mb = max(0, (final_peak - baseline_peak) / (1024 * 1024))
        net_mb = (final_current - baseline_current) / (1024 * 1024)

        return MemoryResult(
            name=name,
            peak_mb=peak_mb,
            net_mb=net_mb,
            allocations=allocations,
        )

    @contextmanager
    def track(self, name: str) -> Generator[None, None, None]:
        """
        Context manager for tracking memory

        Example:
            tracker = MemoryTracker()
            with tracker.track("my_block"):
                # code to measure
                pass
        """
        # 檢查是否已經在追蹤（避免重複啟動）
        was_tracing = tracemalloc.is_tracing()

        if not was_tracing:
            tracemalloc.start()

        baseline_current, baseline_peak = tracemalloc.get_traced_memory()

        try:
            yield
        finally:
            final_current, final_peak = tracemalloc.get_traced_memory()
            stats = tracemalloc.take_snapshot().statistics('lineno')
            allocations = sum(stat.count for stat in stats)

            if not was_tracing:
                tracemalloc.stop()

            peak_mb = max(0, (final_peak - baseline_peak) / (1024 * 1024))
            net_mb = (final_current - baseline_current) / (1024 * 1024)
            print(f"{name}: peak={peak_mb:.2f}MB, net={net_mb:.2f}MB, allocs={allocations}")


@dataclass
class BenchmarkReport:
    """
    基準測試報告

    包含所有測試結果，提供摘要和比較功能。
    """

    timing_results: List[TimingResult] = field(default_factory=list)
    memory_results: List[MemoryResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_timing(self, result: TimingResult) -> None:
        """添加計時結果"""
        self.timing_results.append(result)

    def add_memory(self, result: MemoryResult) -> None:
        """添加記憶體結果"""
        self.memory_results.append(result)

    def summary(self) -> str:
        """
        生成摘要報告

        Returns:
            str: 文字格式的摘要
        """
        lines = ["Benchmark Summary", "=" * 60]

        if self.timing_results:
            lines.append("\nTiming Results:")
            lines.append("-" * 60)
            for result in self.timing_results:
                lines.append(str(result))

        if self.memory_results:
            lines.append("\nMemory Results:")
            lines.append("-" * 60)
            for result in self.memory_results:
                lines.append(str(result))

        return "\n".join(lines)

    def to_markdown(self) -> str:
        """
        生成 Markdown 格式報告

        Returns:
            str: Markdown 格式的報告
        """
        lines = ["# Benchmark Report", ""]

        # Metadata
        if self.metadata:
            lines.append("## Metadata")
            for key, value in self.metadata.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        # Timing results
        if self.timing_results:
            lines.append("## Timing Results")
            lines.append("")
            lines.append("| Name | Mean (ms) | Min (ms) | Max (ms) | Std (ms) | Iterations |")
            lines.append("|------|-----------|----------|----------|----------|------------|")
            for result in self.timing_results:
                lines.append(
                    f"| {result.name} "
                    f"| {result.mean*1000:.2f} "
                    f"| {result.min*1000:.2f} "
                    f"| {result.max*1000:.2f} "
                    f"| {result.std*1000:.2f} "
                    f"| {result.iterations} |"
                )
            lines.append("")

        # Memory results
        if self.memory_results:
            lines.append("## Memory Results")
            lines.append("")
            lines.append("| Name | Peak (MB) | Net (MB) | Allocations |")
            lines.append("|------|-----------|----------|-------------|")
            for result in self.memory_results:
                lines.append(
                    f"| {result.name} "
                    f"| {result.peak_mb:.2f} "
                    f"| {result.net_mb:.2f} "
                    f"| {result.allocations} |"
                )
            lines.append("")

        # Speedup comparison (if multiple timing results)
        if len(self.timing_results) > 1:
            lines.append("## Speedup Analysis")
            lines.append("")
            baseline = self.timing_results[0]
            lines.append(f"Baseline: {baseline.name} ({baseline.mean*1000:.2f}ms)")
            lines.append("")
            lines.append("| Implementation | Mean (ms) | Speedup |")
            lines.append("|----------------|-----------|---------|")
            for result in self.timing_results:
                speedup = baseline.mean / result.mean if result.mean > 0 else float('inf')
                lines.append(
                    f"| {result.name} "
                    f"| {result.mean*1000:.2f} "
                    f"| {speedup:.2f}x |"
                )
            lines.append("")

        return "\n".join(lines)

    def to_json(self, indent: int = 2) -> str:
        """
        生成 JSON 格式報告

        Args:
            indent: JSON 縮排空格數

        Returns:
            str: JSON 格式的報告
        """
        data = {
            "metadata": self.metadata,
            "timing": [r.to_dict() for r in self.timing_results],
            "memory": [r.to_dict() for r in self.memory_results],
        }

        # Add speedup analysis
        if len(self.timing_results) > 1:
            baseline = self.timing_results[0]
            speedups = []
            for result in self.timing_results:
                speedup = baseline.mean / result.mean if result.mean > 0 else float('inf')
                speedups.append({
                    "name": result.name,
                    "speedup": speedup,
                    "baseline": baseline.name,
                })
            data["speedup_analysis"] = speedups

        return json.dumps(data, indent=indent)

    def compare(self, baseline_name: str, target_name: str) -> Optional[float]:
        """
        比較兩個實作的加速比

        Args:
            baseline_name: 基準實作名稱
            target_name: 目標實作名稱

        Returns:
            float: 加速比（target 相對於 baseline）
            None: 如果找不到對應的結果，或無法計算有意義的加速比
        """
        baseline = next((r for r in self.timing_results if r.name == baseline_name), None)
        target = next((r for r in self.timing_results if r.name == target_name), None)

        if baseline is None or target is None:
            return None

        # 無法計算有意義的加速比（避免 division by zero）
        if baseline.mean == 0 or target.mean == 0:
            return None

        return baseline.mean / target.mean


class BenchmarkSuite:
    """
    基準測試套件

    收集多個基準測試，統一執行並生成報告。

    Example:
        suite = BenchmarkSuite(warmup=3, iterations=10)
        suite.add_benchmark("test1", func1, arg1, arg2)
        suite.add_benchmark("test2", func2, arg3, arg4)
        report = suite.run()
        print(report.to_markdown())
    """

    def __init__(
        self,
        warmup: int = 3,
        iterations: int = 10,
        track_memory: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            warmup: Warmup 次數
            iterations: 測量次數
            track_memory: 是否追蹤記憶體使用
            metadata: 報告的元數據
        """
        self.timer = BenchmarkTimer(warmup=warmup, iterations=iterations)
        self.memory_tracker = MemoryTracker() if track_memory else None
        self.benchmarks: List[Dict[str, Any]] = []
        self.metadata = metadata or {}

    def add_benchmark(
        self,
        name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> None:
        """
        添加基準測試

        Args:
            name: 測試名稱
            func: 要測試的函數
            *args: 函數的位置參數
            **kwargs: 函數的關鍵字參數
        """
        self.benchmarks.append({
            "name": name,
            "func": func,
            "args": args,
            "kwargs": kwargs,
        })

    def run(self, verbose: bool = True) -> BenchmarkReport:
        """
        執行所有基準測試

        Args:
            verbose: 是否顯示進度

        Returns:
            BenchmarkReport: 測試報告
        """
        report = BenchmarkReport(metadata=self.metadata)

        for i, benchmark in enumerate(self.benchmarks, 1):
            name = benchmark["name"]
            func = benchmark["func"]
            args = benchmark["args"]
            kwargs = benchmark["kwargs"]

            if verbose:
                print(f"Running benchmark {i}/{len(self.benchmarks)}: {name}")

            # Timing
            timing_result = self.timer.measure(name, func, *args, **kwargs)
            report.add_timing(timing_result)

            if verbose:
                print(f"  {timing_result}")

            # Memory (if enabled)
            if self.memory_tracker:
                memory_result = self.memory_tracker.measure(name, func, *args, **kwargs)
                report.add_memory(memory_result)

                if verbose:
                    print(f"  {memory_result}")

        return report
