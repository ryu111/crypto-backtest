"""
Loop 控制器

AI Loop 的執行控制器，管理迭代循環、狀態持久化、中斷恢復。
支援多種執行模式和進度報告。
"""

import json
import signal
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Literal
from enum import Enum

import pandas as pd


class LoopMode(Enum):
    """Loop 執行模式"""
    CONTINUOUS = "continuous"       # 持續執行直到手動停止
    N_ITERATIONS = "n_iterations"   # 執行指定次數
    TIME_BASED = "time_based"       # 執行指定時間
    UNTIL_TARGET = "until_target"   # 執行直到達到目標


class IterationStatus(Enum):
    """迭代狀態"""
    SUCCESS = "success"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class IterationResult:
    """單次迭代結果"""

    iteration: int
    timestamp: datetime
    status: IterationStatus

    # 績效指標
    sharpe_ratio: float
    total_return: float
    max_drawdown: float

    # 策略資訊
    strategy_name: str
    best_params: Dict[str, Any]

    # 實驗 ID
    experiment_id: Optional[str] = None

    # 錯誤資訊（如果失敗）
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['status'] = self.status.value
        return data


@dataclass
class LoopState:
    """Loop 狀態"""

    # 基本資訊
    started_at: datetime
    paused_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None

    # 執行模式
    mode: str = LoopMode.CONTINUOUS.value
    target: Optional[int] = None  # N_ITERATIONS 的目標次數或 UNTIL_TARGET 的目標 Sharpe
    time_limit_minutes: Optional[int] = None  # TIME_BASED 的時間限制

    # 進度
    current_iteration: int = 0
    completed_iterations: int = 0
    successful_iterations: int = 0
    failed_iterations: int = 0

    # 最佳結果
    best_sharpe: float = float('-inf')
    best_strategy: str = ""
    best_experiment_id: str = ""
    best_params: Dict[str, Any] = field(default_factory=dict)

    # 迭代歷史
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)

    # 狀態標記
    is_paused: bool = False
    is_stopped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典（JSON 序列化）"""
        data = asdict(self)
        data['started_at'] = self.started_at.isoformat()
        if self.paused_at:
            data['paused_at'] = self.paused_at.isoformat()
        if self.stopped_at:
            data['stopped_at'] = self.stopped_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LoopState':
        """從字典建立"""
        data = data.copy()
        data['started_at'] = datetime.fromisoformat(data['started_at'])
        if data.get('paused_at'):
            data['paused_at'] = datetime.fromisoformat(data['paused_at'])
        if data.get('stopped_at'):
            data['stopped_at'] = datetime.fromisoformat(data['stopped_at'])
        return cls(**data)


class LoopController:
    """
    Loop 控制器

    管理 AI Loop 的執行、暫停、恢復、狀態保存等功能。
    支援多種執行模式和回調機制。

    使用範例:
        # 建立控制器
        controller = LoopController(
            iteration_callback=run_single_iteration
        )

        # 啟動 Loop（執行 100 次）
        controller.start(
            mode=LoopMode.N_ITERATIONS,
            target=100
        )

        # 或持續執行直到達到目標 Sharpe
        controller.start(
            mode=LoopMode.UNTIL_TARGET,
            target=3.0  # Sharpe >= 3.0
        )

        # 暫停
        controller.pause()

        # 恢復
        controller.resume()

        # 取得進度
        progress = controller.get_progress()
        print(progress)
    """

    def __init__(
        self,
        iteration_callback: Callable[[], IterationResult],
        state_file: Optional[Path] = None,
        auto_save: bool = True,
        callbacks: Optional[Dict[str, Callable]] = None
    ):
        """
        初始化 Loop 控制器

        Args:
            iteration_callback: 單次迭代執行函數，返回 IterationResult
            state_file: 狀態檔案路徑（預設: learning/loop_state.json）
            auto_save: 是否每次迭代後自動保存狀態
            callbacks: 回調函數字典
                {
                    'on_iteration_start': callable,
                    'on_iteration_end': callable,
                    'on_success': callable,
                    'on_failure': callable,
                    'on_new_best': callable,
                    'on_loop_end': callable
                }
        """
        self.iteration_callback = iteration_callback
        self.auto_save = auto_save

        # 確定狀態檔案路徑
        if state_file is None:
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            self.state_file = project_root / 'learning' / 'loop_state.json'
        else:
            self.state_file = state_file

        # 確保目錄存在
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # 狀態
        self.state: Optional[LoopState] = None

        # 回調函數
        self.callbacks = callbacks or {}

        # 信號處理（優雅停止）
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """設定信號處理器（SIGINT, SIGTERM）"""
        def signal_handler(signum, frame):
            print("\n收到停止信號，正在優雅停止 Loop...")
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def start(
        self,
        mode: LoopMode = LoopMode.CONTINUOUS,
        target: Optional[int] = None,
        time_limit_minutes: Optional[int] = None,
        resume: bool = False
    ):
        """
        啟動 Loop

        Args:
            mode: 執行模式
            target: 目標值（依模式而定）
                - N_ITERATIONS: 執行次數
                - UNTIL_TARGET: 目標 Sharpe Ratio
            time_limit_minutes: 時間限制（分鐘）
            resume: 是否從上次中斷處恢復
        """
        # 恢復或建立新狀態
        if resume and self.state_file.exists():
            print("從上次中斷處恢復...")
            self.state = self.load_state()
            self.state.is_paused = False
            self.state.is_stopped = False
        else:
            print(f"啟動新的 Loop（模式: {mode.value}）")
            self.state = LoopState(
                started_at=datetime.now(),
                mode=mode.value,
                target=target,
                time_limit_minutes=time_limit_minutes
            )

        # 驗證配置
        self._validate_config(mode, target, time_limit_minutes)

        # 執行主循環
        self._run_loop()

    def _validate_config(
        self,
        mode: LoopMode,
        target: Optional[int],
        time_limit_minutes: Optional[int]
    ):
        """驗證配置"""
        if mode == LoopMode.N_ITERATIONS and target is None:
            raise ValueError("N_ITERATIONS 模式需要提供 target（執行次數）")

        if mode == LoopMode.UNTIL_TARGET and target is None:
            raise ValueError("UNTIL_TARGET 模式需要提供 target（目標 Sharpe）")

        if mode == LoopMode.TIME_BASED and time_limit_minutes is None:
            raise ValueError("TIME_BASED 模式需要提供 time_limit_minutes")

    def _run_loop(self):
        """執行主循環"""
        mode = LoopMode(self.state.mode)

        print(f"\n{'='*60}")
        print(f"Loop 開始執行")
        print(f"模式: {mode.value}")
        if self.state.target:
            print(f"目標: {self.state.target}")
        if self.state.time_limit_minutes:
            print(f"時間限制: {self.state.time_limit_minutes} 分鐘")
        print(f"{'='*60}\n")

        # 執行循環
        while not self.state.is_stopped:
            # 檢查是否應該停止
            if self._should_stop():
                print("\n達到停止條件")
                break

            # 檢查暫停
            while self.state.is_paused and not self.state.is_stopped:
                time.sleep(1)

            if self.state.is_stopped:
                break

            # 執行單次迭代
            self.state.current_iteration += 1
            self._run_iteration()

            # 自動保存
            if self.auto_save:
                self.save_state()

        # Loop 結束
        self._on_loop_end()

    def _should_stop(self) -> bool:
        """判斷是否應該停止 Loop"""
        mode = LoopMode(self.state.mode)

        if mode == LoopMode.CONTINUOUS:
            return False  # 需手動停止

        elif mode == LoopMode.N_ITERATIONS:
            return self.state.completed_iterations >= self.state.target

        elif mode == LoopMode.TIME_BASED:
            elapsed = datetime.now() - self.state.started_at
            time_limit = timedelta(minutes=self.state.time_limit_minutes)
            return elapsed >= time_limit

        elif mode == LoopMode.UNTIL_TARGET:
            return self.state.best_sharpe >= self.state.target

        return False

    def _run_iteration(self):
        """執行單次迭代"""
        iteration_num = self.state.current_iteration

        print(f"\n{'─'*60}")
        print(f"迭代 #{iteration_num}")
        print(f"{'─'*60}")

        # 回調：迭代開始
        self._trigger_callback('on_iteration_start', iteration_num)

        try:
            # 執行迭代
            result = self.iteration_callback()

            # 記錄結果
            self._record_result(result)

            # 檢查是否為最佳結果
            if result.sharpe_ratio > self.state.best_sharpe:
                self._update_best_result(result)

            # 回調：成功
            self._trigger_callback('on_success', result)

        except Exception as e:
            # 記錄失敗
            error_result = IterationResult(
                iteration=iteration_num,
                timestamp=datetime.now(),
                status=IterationStatus.FAILED,
                sharpe_ratio=float('-inf'),
                total_return=0.0,
                max_drawdown=0.0,
                strategy_name="unknown",
                best_params={},
                error=str(e)
            )
            self._record_result(error_result)

            # 回調：失敗
            self._trigger_callback('on_failure', e)

            print(f"❌ 迭代失敗: {e}")

        # 回調：迭代結束
        self._trigger_callback('on_iteration_end', iteration_num)

    def _record_result(self, result: IterationResult):
        """記錄迭代結果"""
        self.state.completed_iterations += 1

        if result.status == IterationStatus.SUCCESS:
            self.state.successful_iterations += 1
        else:
            self.state.failed_iterations += 1

        # 添加到歷史
        self.state.iteration_history.append(result.to_dict())

        # 顯示結果
        if result.status == IterationStatus.SUCCESS:
            print(f"✓ Sharpe: {result.sharpe_ratio:.4f}")
            print(f"  Return: {result.total_return:.2%}")
            print(f"  Drawdown: {result.max_drawdown:.2%}")
            print(f"  Strategy: {result.strategy_name}")
        else:
            print(f"✗ 失敗: {result.error}")

    def _update_best_result(self, result: IterationResult):
        """更新最佳結果"""
        old_best = self.state.best_sharpe

        self.state.best_sharpe = result.sharpe_ratio
        self.state.best_strategy = result.strategy_name
        self.state.best_experiment_id = result.experiment_id or ""
        self.state.best_params = result.best_params

        print(f"\n🎉 新的最佳結果！")
        print(f"  Sharpe: {old_best:.4f} → {result.sharpe_ratio:.4f}")
        print(f"  Strategy: {result.strategy_name}")

        # 回調：新最佳
        self._trigger_callback('on_new_best', result)

    def _trigger_callback(self, name: str, *args, **kwargs):
        """觸發回調函數"""
        if name in self.callbacks:
            try:
                self.callbacks[name](*args, **kwargs)
            except Exception as e:
                print(f"回調 {name} 執行失敗: {e}")

    def _on_loop_end(self):
        """Loop 結束處理"""
        self.state.stopped_at = datetime.now()
        self.state.is_stopped = True

        # 保存最終狀態
        self.save_state()

        # 顯示摘要
        summary = self.get_summary()
        print(f"\n{summary}")

        # 回調：Loop 結束
        self._trigger_callback('on_loop_end', self.state)

    def pause(self):
        """暫停 Loop"""
        if self.state and not self.state.is_paused:
            self.state.is_paused = True
            self.state.paused_at = datetime.now()
            self.save_state()
            print("Loop 已暫停")

    def resume(self):
        """恢復 Loop"""
        if self.state and self.state.is_paused:
            self.state.is_paused = False
            self.state.paused_at = None
            print("Loop 已恢復")

    def stop(self):
        """停止 Loop"""
        if self.state:
            self.state.is_stopped = True
            print("Loop 已停止")

    def save_state(self):
        """保存狀態到檔案"""
        if self.state:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state.to_dict(), f, indent=2, ensure_ascii=False)

    def load_state(self) -> LoopState:
        """從檔案載入狀態"""
        with open(self.state_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return LoopState.from_dict(data)

    def get_progress(self) -> Dict[str, Any]:
        """
        取得進度資訊

        Returns:
            {
                'current_iteration': int,
                'completed_iterations': int,
                'successful_iterations': int,
                'failed_iterations': int,
                'success_rate': float,
                'best_sharpe': float,
                'best_strategy': str,
                'elapsed_time': str,
                'estimated_remaining': str (如適用)
            }
        """
        if not self.state:
            return {}

        elapsed = datetime.now() - self.state.started_at
        elapsed_str = str(elapsed).split('.')[0]  # 移除微秒

        success_rate = (
            self.state.successful_iterations / self.state.completed_iterations
            if self.state.completed_iterations > 0 else 0.0
        )

        progress = {
            'current_iteration': self.state.current_iteration,
            'completed_iterations': self.state.completed_iterations,
            'successful_iterations': self.state.successful_iterations,
            'failed_iterations': self.state.failed_iterations,
            'success_rate': success_rate,
            'best_sharpe': self.state.best_sharpe,
            'best_strategy': self.state.best_strategy,
            'elapsed_time': elapsed_str
        }

        # 估算剩餘時間（僅 N_ITERATIONS 模式）
        mode = LoopMode(self.state.mode)
        if mode == LoopMode.N_ITERATIONS and self.state.completed_iterations > 0:
            avg_time_per_iter = elapsed / self.state.completed_iterations
            remaining_iters = self.state.target - self.state.completed_iterations
            estimated_remaining = avg_time_per_iter * remaining_iters
            progress['estimated_remaining'] = str(estimated_remaining).split('.')[0]

        return progress

    def get_summary(self) -> str:
        """
        產生摘要報告

        Returns:
            摘要字串
        """
        if not self.state:
            return "尚未啟動 Loop"

        elapsed = datetime.now() - self.state.started_at
        elapsed_str = str(elapsed).split('.')[0]

        success_rate = (
            self.state.successful_iterations / self.state.completed_iterations
            if self.state.completed_iterations > 0 else 0.0
        )

        return f"""
{'='*60}
Loop 執行摘要
{'='*60}
執行時間: {elapsed_str}
完成迭代: {self.state.completed_iterations}
成功: {self.state.successful_iterations} ({success_rate:.1%})
失敗: {self.state.failed_iterations}

最佳結果
{'-'*60}
Sharpe Ratio: {self.state.best_sharpe:.4f}
策略: {self.state.best_strategy}
實驗 ID: {self.state.best_experiment_id}
參數: {self.state.best_params}
{'='*60}
"""

    def get_iteration_history(self) -> pd.DataFrame:
        """
        取得迭代歷史 DataFrame

        Returns:
            包含所有迭代結果的 DataFrame
        """
        if not self.state or not self.state.iteration_history:
            return pd.DataFrame()

        df = pd.DataFrame(self.state.iteration_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def clear_state(self):
        """清除狀態檔案"""
        if self.state_file.exists():
            self.state_file.unlink()
            print(f"狀態檔案已清除: {self.state_file}")


# 便利函數

def create_loop_controller(
    iteration_callback: Callable[[], IterationResult],
    auto_save: bool = True,
    callbacks: Optional[Dict[str, Callable]] = None
) -> LoopController:
    """
    建立 Loop 控制器

    Args:
        iteration_callback: 單次迭代執行函數
        auto_save: 是否自動保存狀態
        callbacks: 回調函數字典

    Returns:
        LoopController 實例
    """
    return LoopController(
        iteration_callback=iteration_callback,
        auto_save=auto_save,
        callbacks=callbacks
    )
