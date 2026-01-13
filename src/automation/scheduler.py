"""
執行調度器

智能選擇最優執行路徑：
- GPU 批量處理（大量參數掃描）
- CPU 多進程池（多策略並行）
- CPU 單進程（簡單任務）
- 順序執行（有依賴的任務）
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
from enum import Enum

logger = logging.getLogger(__name__)


class ExecutorType(Enum):
    """執行器類型"""
    GPU_BATCH = "gpu_batch"      # GPU 批量（參數掃描）
    CPU_POOL = "cpu_pool"        # CPU 多進程池
    CPU_SINGLE = "cpu_single"    # CPU 單進程
    SEQUENTIAL = "sequential"    # 順序執行（有依賴）


class TaskType(Enum):
    """任務類型"""
    PARAM_SWEEP = "param_sweep"           # 參數掃描
    MULTI_STRATEGY = "multi_strategy"     # 多策略測試
    WALK_FORWARD = "walk_forward"         # Walk-Forward 分析
    MONTE_CARLO = "monte_carlo"           # Monte Carlo 模擬
    SINGLE_BACKTEST = "single_backtest"   # 單次回測
    VALIDATION = "validation"             # 5 階段驗證


@dataclass
class BacktestTask:
    """回測任務定義"""
    task_id: str
    task_type: TaskType
    strategy_name: str

    # 參數相關
    param_space: Optional[Dict] = None
    param_count: int = 0

    # 資料相關
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    data_key: Optional[str] = None

    # 執行設定
    priority: int = 0                     # 優先級（越高越先執行）
    timeout_seconds: int = 300            # 超時時間
    retry_count: int = 0                  # 重試次數

    # 依賴
    depends_on: List[str] = field(default_factory=list)


@dataclass
class ExecutionPlan:
    """執行計劃"""
    executor: ExecutorType

    # CPU 設定
    n_workers: int = 1

    # GPU 設定
    batch_size: int = 50
    use_gpu: bool = False

    # 資源估算
    estimated_memory_mb: int = 0
    estimated_time_seconds: int = 0

    # 排程
    priority: int = 0

    def to_dict(self) -> Dict:
        """轉換為字典"""
        return {
            'executor': self.executor.value,
            'n_workers': self.n_workers,
            'batch_size': self.batch_size,
            'use_gpu': self.use_gpu,
            'estimated_memory_mb': self.estimated_memory_mb,
            'estimated_time_seconds': self.estimated_time_seconds,
            'priority': self.priority
        }


class ExecutionScheduler:
    """智能執行調度器

    根據任務特性和系統資源，選擇最優執行路徑：
    - GPU 批量：大量參數掃描（>100 組）
    - CPU 多進程：多策略並行
    - CPU 單進程：簡單任務
    - 順序執行：有依賴的任務（Walk-Forward）
    """

    # 資源估算常數
    MEMORY_PER_BACKTEST_MB = 50      # 每次回測平均記憶體
    TIME_PER_BACKTEST_SECONDS = 2    # 每次回測平均時間
    GPU_SPEEDUP_FACTOR = 5           # GPU 加速倍數

    # 調度閾值
    GPU_THRESHOLD = 100              # 參數數量超過此值使用 GPU
    PARALLEL_THRESHOLD = 5           # 任務數量超過此值使用多進程

    def __init__(
        self,
        max_cpu_workers: Optional[int] = None,
        gpu_available: Optional[bool] = None,
        max_memory_gb: float = 32.0,
        verbose: bool = False
    ):
        """初始化調度器

        Args:
            max_cpu_workers: 最大 CPU 工作進程數（預設自動檢測）
            gpu_available: GPU 是否可用（預設自動檢測）
            max_memory_gb: 可用記憶體上限
            verbose: 是否顯示調度決策
        """
        self.max_cpu_workers = max_cpu_workers or self._detect_cpu_count()
        self.gpu_available = gpu_available if gpu_available is not None else self._detect_gpu()
        self.max_memory_gb = max_memory_gb
        self.verbose = verbose

        # 資源追蹤
        self._active_workers = 0
        self._gpu_in_use = False

        if self.verbose:
            logger.info(
                f"Scheduler initialized: CPU={self.max_cpu_workers}, "
                f"GPU={self.gpu_available}, Memory={self.max_memory_gb}GB"
            )

    def _detect_cpu_count(self) -> int:
        """檢測可用 CPU 核心數

        保留 2 核心給系統使用
        """
        cpu_count = os.cpu_count() or 4
        return max(1, cpu_count - 2)

    def _detect_gpu(self) -> bool:
        """檢測 GPU 是否可用

        嘗試檢測 CUDA、MPS（Apple Silicon）等 GPU 環境
        """
        # 檢測 PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                return True
        except ImportError:
            pass

        # 檢測 Apple Silicon MPS
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return True
        except (ImportError, AttributeError):
            pass

        # 檢測 TensorFlow GPU
        try:
            import tensorflow as tf
            if len(tf.config.list_physical_devices('GPU')) > 0:
                return True
        except (ImportError, AttributeError):
            pass

        return False

    def schedule(self, task: BacktestTask) -> ExecutionPlan:
        """為任務選擇執行計劃

        Args:
            task: 回測任務

        Returns:
            ExecutionPlan
        """
        # 1. 根據任務類型和參數數量選擇執行器
        executor = self._select_executor(task)

        # 2. 計算資源配置
        n_workers = self._get_optimal_workers(task, executor)
        batch_size = self._get_optimal_batch_size(task, executor)
        use_gpu = self._should_use_gpu(task)

        # 3. 估算資源需求
        memory_mb, time_seconds = self._estimate_resources(task, executor)

        # 4. 建立執行計劃
        plan = ExecutionPlan(
            executor=executor,
            n_workers=n_workers,
            batch_size=batch_size,
            use_gpu=use_gpu,
            estimated_memory_mb=memory_mb,
            estimated_time_seconds=time_seconds,
            priority=task.priority
        )

        if self.verbose:
            logger.info(
                f"Scheduled task '{task.task_id}': {executor.value}, "
                f"workers={n_workers}, GPU={use_gpu}, "
                f"~{memory_mb}MB, ~{time_seconds}s"
            )

        return plan

    def schedule_batch(self, tasks: List[BacktestTask]) -> Dict[str, ExecutionPlan]:
        """批量排程多個任務

        Args:
            tasks: 任務列表

        Returns:
            {task_id: ExecutionPlan}
        """
        # 按優先級排序
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)

        # 為每個任務建立計劃
        plans = {}
        for task in sorted_tasks:
            plans[task.task_id] = self.schedule(task)

        if self.verbose:
            logger.info(f"Batch scheduled {len(tasks)} tasks")

        return plans

    def _select_executor(self, task: BacktestTask) -> ExecutorType:
        """選擇執行器類型

        決策邏輯：
        1. 有依賴關係的任務 → SEQUENTIAL
        2. GPU 可用且參數量大 → GPU_BATCH
        3. 多任務或參數量中等 → CPU_POOL
        4. 單次簡單任務 → CPU_SINGLE
        """
        # 有依賴關係必須順序執行
        if task.depends_on:
            return ExecutorType.SEQUENTIAL

        # Walk-Forward 和 Validation 需要順序執行
        if task.task_type in (TaskType.WALK_FORWARD, TaskType.VALIDATION):
            return ExecutorType.SEQUENTIAL

        # GPU 適合大量參數掃描
        if self._should_use_gpu(task):
            return ExecutorType.GPU_BATCH

        # 多策略或中量參數使用多進程
        if task.task_type == TaskType.MULTI_STRATEGY:
            return ExecutorType.CPU_POOL

        if task.task_type == TaskType.PARAM_SWEEP and task.param_count >= self.PARALLEL_THRESHOLD:
            return ExecutorType.CPU_POOL

        # Monte Carlo 使用多進程
        if task.task_type == TaskType.MONTE_CARLO:
            return ExecutorType.CPU_POOL

        # 單次回測用單進程
        return ExecutorType.CPU_SINGLE

    def _should_use_gpu(self, task: BacktestTask) -> bool:
        """判斷是否應該使用 GPU

        GPU 適合：
        - GPU 可用
        - 參數掃描任務
        - 參數數量 > GPU_THRESHOLD
        """
        if not self.gpu_available:
            return False

        if task.task_type != TaskType.PARAM_SWEEP:
            return False

        if task.param_count < self.GPU_THRESHOLD:
            return False

        return True

    def _get_optimal_workers(self, task: BacktestTask, executor: ExecutorType) -> int:
        """計算最佳 worker 數量

        考慮因素：
        - 任務類型
        - 系統 CPU 核心數
        - 參數數量
        """
        if executor in (ExecutorType.CPU_SINGLE, ExecutorType.SEQUENTIAL):
            return 1

        if executor == ExecutorType.GPU_BATCH:
            # GPU 模式只需要少量 worker 準備資料
            return min(4, self.max_cpu_workers)

        # CPU_POOL: 根據任務量決定
        if task.task_type == TaskType.PARAM_SWEEP:
            # 參數掃描：worker 數 = min(param_count, max_workers)
            return min(task.param_count, self.max_cpu_workers)

        # 其他多任務：使用所有可用 worker
        return self.max_cpu_workers

    def _get_optimal_batch_size(self, task: BacktestTask, executor: ExecutorType) -> int:
        """計算最佳批次大小

        主要用於 GPU 批量處理
        """
        if executor != ExecutorType.GPU_BATCH:
            return 1

        # GPU 批次大小根據記憶體限制
        # 保守估計：每個回測 50MB
        max_batch_by_memory = int((self.max_memory_gb * 1024) / self.MEMORY_PER_BACKTEST_MB)

        # 限制在合理範圍內
        batch_size = min(max_batch_by_memory, 100)
        batch_size = max(batch_size, 10)

        return batch_size

    def _estimate_resources(
        self,
        task: BacktestTask,
        executor: ExecutorType
    ) -> tuple[int, int]:
        """估算資源需求

        Returns:
            (memory_mb, time_seconds)
        """
        # 估算回測次數
        if task.task_type == TaskType.PARAM_SWEEP:
            n_backtests = task.param_count
        elif task.task_type == TaskType.MULTI_STRATEGY:
            # 假設有 10 個策略
            n_backtests = 10
        elif task.task_type == TaskType.MONTE_CARLO:
            # 假設 100 次 Monte Carlo
            n_backtests = 100
        elif task.task_type == TaskType.VALIDATION:
            # 5 階段驗證
            n_backtests = 5
        elif task.task_type == TaskType.WALK_FORWARD:
            # 假設 10 個時間窗口
            n_backtests = 10
        else:
            n_backtests = 1

        # 記憶體估算
        if executor == ExecutorType.GPU_BATCH:
            # GPU 需要一次載入整個批次
            batch_size = self._get_optimal_batch_size(task, executor)
            memory_mb = batch_size * self.MEMORY_PER_BACKTEST_MB
        elif executor == ExecutorType.CPU_POOL:
            # 多進程需要為每個 worker 預留記憶體
            n_workers = self._get_optimal_workers(task, executor)
            memory_mb = n_workers * self.MEMORY_PER_BACKTEST_MB
        else:
            # 單進程或順序執行
            memory_mb = self.MEMORY_PER_BACKTEST_MB

        # 時間估算
        if executor == ExecutorType.GPU_BATCH:
            # GPU 加速
            time_seconds = int(
                n_backtests * self.TIME_PER_BACKTEST_SECONDS / self.GPU_SPEEDUP_FACTOR
            )
        elif executor == ExecutorType.CPU_POOL:
            # 並行加速
            n_workers = self._get_optimal_workers(task, executor)
            time_seconds = int(
                n_backtests * self.TIME_PER_BACKTEST_SECONDS / n_workers
            )
        else:
            # 順序執行
            time_seconds = n_backtests * self.TIME_PER_BACKTEST_SECONDS

        return memory_mb, time_seconds

    def get_system_status(self) -> Dict:
        """取得系統資源狀態"""
        return {
            'cpu_count': os.cpu_count(),
            'max_workers': self.max_cpu_workers,
            'active_workers': self._active_workers,
            'gpu_available': self.gpu_available,
            'gpu_in_use': self._gpu_in_use,
            'max_memory_gb': self.max_memory_gb
        }

    def acquire_resources(self, plan: ExecutionPlan) -> bool:
        """取得執行資源（用於資源管理）

        Args:
            plan: 執行計劃

        Returns:
            是否成功取得資源
        """
        # 檢查 GPU 資源
        if plan.use_gpu and self._gpu_in_use:
            if self.verbose:
                logger.warning("GPU already in use, cannot acquire")
            return False

        # 檢查 CPU 資源
        available_workers = self.max_cpu_workers - self._active_workers
        if plan.n_workers > available_workers:
            if self.verbose:
                logger.warning(
                    f"Not enough workers: need {plan.n_workers}, "
                    f"available {available_workers}"
                )
            return False

        # 取得資源
        self._active_workers += plan.n_workers
        if plan.use_gpu:
            self._gpu_in_use = True

        if self.verbose:
            logger.info(
                f"Acquired resources: {plan.n_workers} workers, "
                f"GPU={plan.use_gpu}"
            )

        return True

    def release_resources(self, plan: ExecutionPlan):
        """釋放執行資源

        Args:
            plan: 執行計劃
        """
        self._active_workers = max(0, self._active_workers - plan.n_workers)
        if plan.use_gpu:
            self._gpu_in_use = False

        if self.verbose:
            logger.info(
                f"Released resources: {plan.n_workers} workers, "
                f"GPU={plan.use_gpu}"
            )


# 便利函數
def create_scheduler(verbose: bool = False) -> ExecutionScheduler:
    """建立調度器

    Args:
        verbose: 是否顯示調度日誌

    Returns:
        ExecutionScheduler 實例
    """
    return ExecutionScheduler(verbose=verbose)


def schedule_strategy_optimization(
    strategy_name: str,
    param_space: Dict,
    symbol: str = "BTCUSDT",
    timeframe: str = "1h"
) -> ExecutionPlan:
    """快速排程策略優化任務

    Args:
        strategy_name: 策略名稱
        param_space: 參數空間定義
        symbol: 交易對
        timeframe: 時間週期

    Returns:
        ExecutionPlan
    """
    scheduler = ExecutionScheduler()

    # 估算參數組合數
    param_count = _estimate_param_combinations(param_space)

    task = BacktestTask(
        task_id=f"{strategy_name}_{symbol}_{timeframe}",
        task_type=TaskType.PARAM_SWEEP,
        strategy_name=strategy_name,
        param_space=param_space,
        param_count=param_count,
        symbol=symbol,
        timeframe=timeframe
    )

    return scheduler.schedule(task)


def _estimate_param_combinations(param_space: Dict) -> int:
    """估算參數組合數量

    Args:
        param_space: 參數空間定義
            格式: {param_name: {'type': 'int'|'float', 'low': x, 'high': y}}

    Returns:
        估算的組合數
    """
    total = 1
    for key, spec in param_space.items():
        if spec.get('type') == 'int':
            # 整數參數：high - low + 1
            total *= (spec['high'] - spec['low'] + 1)
        else:
            # 浮點數參數：假設 10 個採樣點
            total *= 10

    return total


# 測試程式碼
if __name__ == "__main__":
    # 設定 logging
    logging.basicConfig(level=logging.INFO)

    print("=== ExecutionScheduler Test ===\n")

    # 建立調度器
    scheduler = ExecutionScheduler(verbose=True)

    # 顯示系統狀態
    print("\n1. System Status:")
    status = scheduler.get_system_status()
    for key, value in status.items():
        print(f"   {key}: {value}")

    # 測試不同任務類型的排程
    print("\n2. Scheduling Tests:")

    # 小量參數掃描
    task1 = BacktestTask(
        task_id="small_param_sweep",
        task_type=TaskType.PARAM_SWEEP,
        strategy_name="ma_cross",
        param_count=10
    )
    plan1 = scheduler.schedule(task1)
    print(f"\n   Small param sweep: {plan1.to_dict()}")

    # 大量參數掃描
    task2 = BacktestTask(
        task_id="large_param_sweep",
        task_type=TaskType.PARAM_SWEEP,
        strategy_name="ma_cross",
        param_count=500
    )
    plan2 = scheduler.schedule(task2)
    print(f"\n   Large param sweep: {plan2.to_dict()}")

    # Walk-Forward
    task3 = BacktestTask(
        task_id="walk_forward",
        task_type=TaskType.WALK_FORWARD,
        strategy_name="rsi",
        depends_on=["task1"]
    )
    plan3 = scheduler.schedule(task3)
    print(f"\n   Walk-Forward: {plan3.to_dict()}")

    # 批量排程
    print("\n3. Batch Scheduling:")
    tasks = [task1, task2, task3]
    plans = scheduler.schedule_batch(tasks)
    for task_id, plan in plans.items():
        print(f"   {task_id}: {plan.executor.value}")

    # 測試資源管理
    print("\n4. Resource Management:")
    if scheduler.acquire_resources(plan1):
        print("   ✓ Acquired resources for plan1")
        scheduler.release_resources(plan1)
        print("   ✓ Released resources for plan1")

    # 測試便利函數
    print("\n5. Quick Schedule:")
    param_space = {
        'window': {'type': 'int', 'low': 10, 'high': 50},
        'threshold': {'type': 'float', 'low': 0.1, 'high': 0.9}
    }
    plan = schedule_strategy_optimization(
        strategy_name="rsi",
        param_space=param_space,
        symbol="BTCUSDT",
        timeframe="4h"
    )
    print(f"   Quick plan: {plan.to_dict()}")

    print("\n=== Test Complete ===")
