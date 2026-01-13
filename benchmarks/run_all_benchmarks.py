#!/usr/bin/env python3
"""
整合基準測試執行腳本

提供命令行介面來執行所有基準測試，支援：
- DataFrame 操作效能（Pandas vs Polars）
- 回測引擎效能
- GPU 批量優化效能

使用方式：
    python benchmarks/run_all_benchmarks.py --quick
    python benchmarks/run_all_benchmarks.py --data-sizes 10000 50000
    python benchmarks/run_all_benchmarks.py --skip-gpu
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Callable
from datetime import datetime

# 添加專案根目錄到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark.runners import (
    DataFrameRunner,
    EngineRunner,
    GPURunner,
    BenchmarkReport
)


# ============================================================================
# 常數
# ============================================================================

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
SEPARATOR_WIDTH = 70


# ============================================================================
# 進度顯示
# ============================================================================

class ProgressTracker:
    """進度追蹤器"""

    def __init__(self, total_tests: int, verbose: bool = False):
        """
        初始化進度追蹤器

        Args:
            total_tests: 總測試數量
            verbose: 是否詳細輸出
        """
        self.total_tests = total_tests
        self.verbose = verbose
        self.current = 0
        self.start_time = time.time()
        self.test_times: List[float] = []
        self._test_start: float = 0.0  # 初始化避免 AttributeError

    def start_test(self, test_name: str):
        """開始測試"""
        self.current += 1
        percentage = (self.current / self.total_tests) * 100

        elapsed = time.time() - self.start_time
        if self.current > 1 and self.test_times:
            avg_time = sum(self.test_times) / len(self.test_times)
            remaining_tests = self.total_tests - self.current
            eta_seconds = avg_time * remaining_tests
            eta_str = format_time(eta_seconds)
        else:
            eta_str = "calculating..."

        print(f"\n[{self.current}/{self.total_tests}] ({percentage:.1f}%) {test_name}")
        print(f"  Elapsed: {format_time(elapsed)} | ETA: {eta_str}")

        self._test_start = time.time()

    def end_test(self, success: bool = True):
        """
        結束測試

        Args:
            success: 測試是否成功，失敗時不計入 ETA 計算
        """
        if self._test_start == 0.0:
            return  # 防禦性檢查

        test_time = time.time() - self._test_start

        # 只有成功的測試才計入 ETA 計算
        if success:
            self.test_times.append(test_time)

        if self.verbose:
            status = "✓" if success else "✗"
            print(f"  {status} Completed in {format_time(test_time)}")


def format_time(seconds: float) -> str:
    """
    格式化時間

    Args:
        seconds: 秒數

    Returns:
        格式化的時間字串
    """
    if seconds < SECONDS_PER_MINUTE:
        return f"{seconds:.1f}s"
    elif seconds < SECONDS_PER_HOUR:
        minutes = int(seconds // SECONDS_PER_MINUTE)
        secs = int(seconds % SECONDS_PER_MINUTE)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // SECONDS_PER_HOUR)
        minutes = int((seconds % SECONDS_PER_HOUR) // SECONDS_PER_MINUTE)
        return f"{hours}h {minutes}m"


# ============================================================================
# 測試執行器
# ============================================================================

class BenchmarkExecutor:
    """基準測試執行器"""

    def __init__(
        self,
        data_sizes: List[int],
        batch_sizes: List[int],
        output_dir: Path,
        skip_gpu: bool = False,
        verbose: bool = False
    ):
        """
        初始化執行器

        Args:
            data_sizes: 資料大小列表
            batch_sizes: GPU 批次大小列表
            output_dir: 報告輸出目錄
            skip_gpu: 是否跳過 GPU 測試
            verbose: 詳細輸出
        """
        self.data_sizes = data_sizes
        self.batch_sizes = batch_sizes
        self.output_dir = output_dir
        self.skip_gpu = skip_gpu
        self.verbose = verbose

        self.reports: Dict[str, BenchmarkReport] = {}
        self.errors: Dict[str, Exception] = {}

        # 建立輸出目錄
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, BenchmarkReport]:
        """
        執行所有測試

        Returns:
            測試報告字典
        """
        # 計算總測試數
        total_tests = self._count_tests()

        print("=" * SEPARATOR_WIDTH)
        print("🚀 合約交易回測系統 - 效能基準測試")
        print("=" * SEPARATOR_WIDTH)
        print(f"資料大小: {self.data_sizes}")
        print(f"批次大小: {self.batch_sizes}")
        print(f"輸出目錄: {self.output_dir}")
        print(f"總測試數: {total_tests}")
        print("=" * SEPARATOR_WIDTH)

        tracker = ProgressTracker(total_tests, verbose=self.verbose)

        # 1. DataFrame 操作測試
        self._run_dataframe_tests(tracker)

        # 2. 回測引擎測試
        self._run_engine_tests(tracker)

        # 3. GPU 測試
        if not self.skip_gpu:
            self._run_gpu_tests(tracker)

        # 儲存報告
        self._save_reports()

        # 顯示總結
        self._print_summary()

        return self.reports

    def _count_tests(self) -> int:
        """計算總測試數"""
        count = 0

        # DataFrame 測試：3 種操作
        count += 3

        # 回測引擎測試：1 種
        count += 1

        # GPU 測試（如果啟用）
        if not self.skip_gpu:
            count += 1

        return count

    def _run_single_test(
        self,
        tracker: ProgressTracker,
        test_key: str,
        test_name: str,
        test_func: Callable,
        *args
    ):
        """
        執行單一測試

        Args:
            tracker: 進度追蹤器
            test_key: 報告字典的 key
            test_name: 顯示名稱
            test_func: 測試函數
            *args: 傳入測試函數的參數
        """
        tracker.start_test(test_name)
        try:
            self.reports[test_key] = test_func(*args)
            tracker.end_test(success=True)
            if self.verbose:
                print(self.reports[test_key].summary())
        except Exception as e:
            self.errors[test_key] = e
            print(f"  ❌ Error: {e}")
            if self.verbose:
                traceback.print_exc()
            tracker.end_test(success=False)

    def _run_dataframe_tests(self, tracker: ProgressTracker):
        """執行 DataFrame 操作測試"""
        print("\n" + "=" * SEPARATOR_WIDTH)
        print("📊 DataFrame 操作效能測試")
        print("=" * SEPARATOR_WIDTH)

        df_runner = DataFrameRunner()

        # Rolling Mean
        self._run_single_test(
            tracker, "rolling_mean", "Rolling Mean",
            df_runner.benchmark_rolling_mean, self.data_sizes
        )

        # Where
        self._run_single_test(
            tracker, "where", "Where (Conditional Selection)",
            df_runner.benchmark_where, self.data_sizes
        )

        # EWM
        self._run_single_test(
            tracker, "ewm", "EWM (Exponential Weighted Mean)",
            df_runner.benchmark_ewm, self.data_sizes
        )

    def _run_engine_tests(self, tracker: ProgressTracker):
        """執行回測引擎測試"""
        print("\n" + "=" * SEPARATOR_WIDTH)
        print("⚙️ 回測引擎效能測試")
        print("=" * SEPARATOR_WIDTH)

        engine_runner = EngineRunner()

        self._run_single_test(
            tracker, "backtest", "Backtest Engine",
            engine_runner.benchmark_backtest, self.data_sizes
        )

    def _run_gpu_tests(self, tracker: ProgressTracker):
        """執行 GPU 測試"""
        print("\n" + "=" * SEPARATOR_WIDTH)
        print("🎮 GPU 批量優化效能測試")
        print("=" * SEPARATOR_WIDTH)

        gpu_runner = GPURunner()
        print(f"可用後端: {gpu_runner.available_backends}")

        if len(gpu_runner.available_backends) > 1:  # 不只 CPU
            self._run_single_test(
                tracker, "gpu_batch", "GPU Batch Optimization",
                gpu_runner.benchmark_batch_optimization, self.batch_sizes
            )
        else:
            print("⚠️ 無 GPU 後端可用，跳過 GPU 測試")

    def _save_reports(self):
        """儲存報告"""
        print("\n" + "=" * SEPARATOR_WIDTH)
        print("💾 儲存報告")
        print("=" * SEPARATOR_WIDTH)

        for name, report in self.reports.items():
            # Markdown
            md_file = self.output_dir / f"{name}_report.md"
            try:
                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(report.to_markdown())
                print(f"  ✓ {md_file}")
            except Exception as e:
                print(f"  ❌ Failed to save {md_file}: {e}")

            # JSON
            json_file = self.output_dir / f"{name}_report.json"
            try:
                with open(json_file, 'w', encoding='utf-8') as f:
                    f.write(report.to_json())
                print(f"  ✓ {json_file}")
            except Exception as e:
                print(f"  ❌ Failed to save {json_file}: {e}")

        # 儲存總結報告
        self._save_summary_report()

    def _save_summary_report(self):
        """儲存總結報告"""
        summary_file = self.output_dir / "summary.md"

        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("# 基準測試總結報告\n\n")
                f.write(f"**執行時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("## 配置\n\n")
                f.write(f"- **資料大小**: {self.data_sizes}\n")
                f.write(f"- **批次大小**: {self.batch_sizes}\n")
                f.write(f"- **跳過 GPU**: {self.skip_gpu}\n\n")

                f.write("## 測試結果\n\n")
                f.write(f"- **成功**: {len(self.reports)}\n")
                f.write(f"- **失敗**: {len(self.errors)}\n\n")

                if self.errors:
                    f.write("## 錯誤\n\n")
                    for name, error in self.errors.items():
                        f.write(f"- **{name}**: {error}\n")
                    f.write("\n")

                f.write("## 報告檔案\n\n")
                for name in self.reports.keys():
                    f.write(f"- [{name}_report.md](./{name}_report.md)\n")
                    f.write(f"- [{name}_report.json](./{name}_report.json)\n")

            print(f"  ✓ {summary_file}")

        except Exception as e:
            print(f"  ❌ Failed to save summary: {e}")

    def _print_summary(self):
        """顯示總結"""
        print("\n" + "=" * SEPARATOR_WIDTH)
        print("📋 測試總結")
        print("=" * SEPARATOR_WIDTH)
        print(f"成功: {len(self.reports)}")
        print(f"失敗: {len(self.errors)}")

        if self.errors:
            print("\n失敗的測試:")
            for name, error in self.errors.items():
                print(f"  ❌ {name}: {error}")

        print(f"\n✅ 報告已儲存至 {self.output_dir}")
        print("=" * SEPARATOR_WIDTH)


# ============================================================================
# 命令行介面
# ============================================================================

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(
        description="合約交易回測系統效能基準測試",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 快速測試（小數據量）
  python benchmarks/run_all_benchmarks.py --quick

  # 自訂資料大小
  python benchmarks/run_all_benchmarks.py --data-sizes 10000 50000 100000

  # 跳過 GPU 測試
  python benchmarks/run_all_benchmarks.py --skip-gpu

  # 詳細輸出
  python benchmarks/run_all_benchmarks.py --verbose
        """
    )

    parser.add_argument(
        '--data-sizes',
        type=int,
        nargs='+',
        default=[10000, 50000, 100000],
        help='資料大小列表 (預設: 10000 50000 100000)'
    )

    parser.add_argument(
        '--batch-sizes',
        type=int,
        nargs='+',
        default=[10, 50, 100],
        help='GPU 批次大小列表 (預設: 10 50 100)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('benchmark_results'),
        help='報告輸出目錄 (預設: benchmark_results)'
    )

    parser.add_argument(
        '--skip-gpu',
        action='store_true',
        help='跳過 GPU 測試'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='快速測試模式（小數據量）'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='詳細輸出'
    )

    return parser.parse_args()


def main():
    """主函數"""
    args = parse_args()

    # 快速模式
    if args.quick:
        args.data_sizes = [1000, 5000]
        args.batch_sizes = [10, 20]

    # 建立執行器
    executor = BenchmarkExecutor(
        data_sizes=args.data_sizes,
        batch_sizes=args.batch_sizes,
        output_dir=args.output_dir,
        skip_gpu=args.skip_gpu,
        verbose=args.verbose
    )

    # 執行測試
    start_time = time.time()
    _ = executor.run()  # 報告已儲存到檔案
    total_time = time.time() - start_time

    print(f"\n⏱️ 總執行時間: {format_time(total_time)}")

    # 返回 exit code
    exit_code = 0 if len(executor.errors) == 0 else 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
