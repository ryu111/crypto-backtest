"""
Loop 狀態持久化模組

實現回測 Loop 的狀態保存與恢復，支援中斷續跑功能。

參考：
- .claude/skills/AI自動化/SKILL.md

使用範例：
    persistence = LoopStatePersistence()

    # 保存狀態
    persistence.save_state(controller)

    # 恢復狀態
    state = persistence.load_state()
    if state:
        controller.restore_from_state(state)

    # 清除狀態（完成後）
    persistence.clear_state()
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .ultimate_loop import UltimateLoopController

logger = logging.getLogger(__name__)


@dataclass
class LoopState:
    """Loop 狀態資料結構

    保存 UltimateLoop 的完整狀態，用於中斷續跑。
    """

    # 基本資訊
    iteration: int  # 當前迭代數
    total_iterations: int  # 目標迭代數
    timestamp: str  # 保存時間 (ISO format)
    version: str = "1.0"  # 狀態格式版本

    # 執行統計
    successful_iterations: int = 0
    failed_iterations: int = 0

    # 策略統計
    completed_strategies: List[str] = field(default_factory=list)
    strategy_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # 最佳結果快照
    best_strategy: Optional[str] = None
    best_params: Optional[Dict[str, Any]] = None
    best_objectives: Optional[Dict[str, float]] = None

    # Pareto 解統計
    total_pareto_solutions: int = 0
    validated_solutions: int = 0

    # 市場狀態分布
    regime_distribution: Dict[str, int] = field(default_factory=dict)

    # GP 策略統計
    gp_strategies_generated: int = 0
    gp_strategies_validated: int = 0

    # 學習統計
    new_insights: int = 0
    memory_entries: int = 0
    experiments_recorded: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LoopState':
        """從字典建立狀態物件"""
        # 處理舊版本兼容
        version = data.get('version', '1.0')

        return cls(
            iteration=data.get('iteration', 0),
            total_iterations=data.get('total_iterations', 0),
            timestamp=data.get('timestamp', ''),
            version=version,
            successful_iterations=data.get('successful_iterations', 0),
            failed_iterations=data.get('failed_iterations', 0),
            completed_strategies=data.get('completed_strategies', []),
            strategy_stats=data.get('strategy_stats', {}),
            best_strategy=data.get('best_strategy'),
            best_params=data.get('best_params'),
            best_objectives=data.get('best_objectives'),
            total_pareto_solutions=data.get('total_pareto_solutions', 0),
            validated_solutions=data.get('validated_solutions', 0),
            regime_distribution=data.get('regime_distribution', {}),
            gp_strategies_generated=data.get('gp_strategies_generated', 0),
            gp_strategies_validated=data.get('gp_strategies_validated', 0),
            new_insights=data.get('new_insights', 0),
            memory_entries=data.get('memory_entries', 0),
            experiments_recorded=data.get('experiments_recorded', 0)
        )

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return asdict(self)

    def progress_percent(self) -> float:
        """計算完成百分比"""
        if self.total_iterations == 0:
            return 0.0
        return (self.iteration / self.total_iterations) * 100

    def summary(self) -> str:
        """生成狀態摘要"""
        lines = [
            f"迭代進度: {self.iteration}/{self.total_iterations} ({self.progress_percent():.1f}%)",
            f"成功: {self.successful_iterations}, 失敗: {self.failed_iterations}",
            f"保存時間: {self.timestamp}"
        ]

        if self.best_strategy:
            lines.append(f"最佳策略: {self.best_strategy}")
            if self.best_objectives:
                sharpe = self.best_objectives.get('sharpe', 0)
                lines.append(f"最佳 Sharpe: {sharpe:.4f}")

        return "\n".join(lines)


class LoopStatePersistence:
    """Loop 狀態持久化管理器

    負責保存和恢復 UltimateLoop 的執行狀態。

    特性：
    - 原子性寫入（先寫 .tmp 再 rename）
    - 版本控制（支援舊版本狀態檔案）
    - 自動備份上一次狀態

    使用範例：
        persistence = LoopStatePersistence()

        # 保存（每次迭代後）
        persistence.save_state(controller)

        # 恢復（啟動時）
        state = persistence.load_state()
        if state:
            print(f"從迭代 {state.iteration} 恢復")

        # 清除（完成後）
        persistence.clear_state()
    """

    DEFAULT_STATE_DIR = "checkpoints"
    STATE_FILE_NAME = "loop_state.json"
    BACKUP_FILE_NAME = "loop_state.backup.json"

    def __init__(
        self,
        state_dir: Optional[str] = None,
        auto_backup: bool = True
    ):
        """初始化狀態持久化管理器

        Args:
            state_dir: 狀態檔案目錄（預設 checkpoints/）
            auto_backup: 是否自動備份上一次狀態
        """
        self.state_dir = Path(state_dir) if state_dir else Path(self.DEFAULT_STATE_DIR)
        self.auto_backup = auto_backup

        # 確保目錄存在
        self.state_dir.mkdir(parents=True, exist_ok=True)

    @property
    def state_path(self) -> Path:
        """狀態檔案路徑"""
        return self.state_dir / self.STATE_FILE_NAME

    @property
    def backup_path(self) -> Path:
        """備份檔案路徑"""
        return self.state_dir / self.BACKUP_FILE_NAME

    @property
    def temp_path(self) -> Path:
        """臨時檔案路徑"""
        return self.state_dir / f"{self.STATE_FILE_NAME}.tmp"

    def save_state(self, controller: 'UltimateLoopController') -> bool:
        """保存當前狀態

        從 UltimateLoopController 提取狀態並保存。

        Args:
            controller: UltimateLoopController 實例

        Returns:
            bool: 是否保存成功
        """
        try:
            # 建立備份（如果啟用）
            if self.auto_backup and self.state_path.exists():
                self._create_backup()

            # 從 controller 提取狀態
            state = self._extract_state_from_controller(controller)

            # 原子性寫入
            self._atomic_write(state)

            logger.info(f"狀態已保存：迭代 {state.iteration}")
            return True

        except Exception as e:
            logger.error(f"保存狀態失敗: {e}")
            return False

    def save_state_direct(self, state: LoopState) -> bool:
        """直接保存狀態物件

        Args:
            state: LoopState 物件

        Returns:
            bool: 是否保存成功
        """
        try:
            if self.auto_backup and self.state_path.exists():
                self._create_backup()

            self._atomic_write(state)
            logger.info(f"狀態已保存：迭代 {state.iteration}")
            return True

        except Exception as e:
            logger.error(f"保存狀態失敗: {e}")
            return False

    def load_state(self) -> Optional[LoopState]:
        """載入狀態

        Returns:
            LoopState: 狀態物件，如果不存在或讀取失敗則返回 None
        """
        if not self.state_path.exists():
            logger.info("未找到狀態檔案")
            return None

        try:
            with open(self.state_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            state = LoopState.from_dict(data)

            # 驗證狀態有效性
            if not self._validate_state(state):
                logger.warning("狀態檔案格式無效")
                return None

            logger.info(f"已載入狀態：{state.summary()}")
            return state

        except json.JSONDecodeError as e:
            logger.error(f"狀態檔案 JSON 格式錯誤: {e}")
            return self._try_load_backup()

        except Exception as e:
            logger.error(f"載入狀態失敗: {e}")
            return self._try_load_backup()

    def clear_state(self) -> bool:
        """清除狀態檔案

        Loop 完成後應調用此方法清除狀態。

        Returns:
            bool: 是否清除成功
        """
        try:
            files_cleared = []

            if self.state_path.exists():
                self.state_path.unlink()
                files_cleared.append(str(self.state_path))

            if self.backup_path.exists():
                self.backup_path.unlink()
                files_cleared.append(str(self.backup_path))

            if self.temp_path.exists():
                self.temp_path.unlink()
                files_cleared.append(str(self.temp_path))

            if files_cleared:
                logger.info(f"已清除狀態檔案: {', '.join(files_cleared)}")
            else:
                logger.info("無狀態檔案需要清除")

            return True

        except Exception as e:
            logger.error(f"清除狀態失敗: {e}")
            return False

    def has_state(self) -> bool:
        """檢查是否有可恢復的狀態"""
        return self.state_path.exists()

    def get_state_info(self) -> Optional[Dict[str, Any]]:
        """取得狀態資訊（不載入完整狀態）

        Returns:
            Dict: 狀態基本資訊（iteration, timestamp, progress）
        """
        state = self.load_state()
        if not state:
            return None

        return {
            'iteration': state.iteration,
            'total_iterations': state.total_iterations,
            'progress_percent': state.progress_percent(),
            'timestamp': state.timestamp,
            'best_strategy': state.best_strategy
        }

    def _extract_state_from_controller(
        self,
        controller: 'UltimateLoopController'
    ) -> LoopState:
        """從 controller 提取狀態"""
        summary = controller.summary

        # 收集已完成的策略
        completed_strategies = []
        if hasattr(controller, 'strategy_stats'):
            completed_strategies = list(controller.strategy_stats.keys())

        return LoopState(
            iteration=summary.successful_iterations + summary.failed_iterations,
            total_iterations=summary.total_iterations,
            timestamp=datetime.now().isoformat(),
            successful_iterations=summary.successful_iterations,
            failed_iterations=summary.failed_iterations,
            completed_strategies=completed_strategies,
            strategy_stats=getattr(controller, 'strategy_stats', {}),
            best_strategy=summary.best_strategy,
            best_params=summary.best_params,
            best_objectives=summary.best_objectives,
            total_pareto_solutions=summary.total_pareto_solutions,
            validated_solutions=summary.validated_solutions,
            regime_distribution=summary.regime_distribution,
            gp_strategies_generated=summary.gp_strategies_generated,
            gp_strategies_validated=summary.gp_strategies_validated,
            new_insights=summary.new_insights,
            memory_entries=summary.memory_entries,
            experiments_recorded=summary.experiments_recorded
        )

    def _atomic_write(self, state: LoopState) -> None:
        """原子性寫入狀態檔案"""
        # 寫入臨時檔案
        with open(self.temp_path, 'w', encoding='utf-8') as f:
            json.dump(state.to_dict(), f, indent=2, ensure_ascii=False, default=str)

        # 原子性 rename
        self.temp_path.replace(self.state_path)

    def _create_backup(self) -> None:
        """建立備份"""
        try:
            if self.state_path.exists():
                # 複製到備份位置
                with open(self.state_path, 'r', encoding='utf-8') as src:
                    data = src.read()
                with open(self.backup_path, 'w', encoding='utf-8') as dst:
                    dst.write(data)
        except Exception as e:
            logger.warning(f"建立備份失敗: {e}")

    def _try_load_backup(self) -> Optional[LoopState]:
        """嘗試從備份載入"""
        if not self.backup_path.exists():
            return None

        try:
            logger.info("嘗試從備份恢復...")
            with open(self.backup_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            state = LoopState.from_dict(data)

            if self._validate_state(state):
                logger.info("從備份恢復成功")
                return state

        except Exception as e:
            logger.error(f"從備份恢復失敗: {e}")

        return None

    def _validate_state(self, state: LoopState) -> bool:
        """驗證狀態有效性"""
        # 必須有有效的迭代數
        if state.iteration < 0:
            return False

        # 必須有時間戳
        if not state.timestamp:
            return False

        # 迭代數不能超過總數
        if state.total_iterations > 0 and state.iteration > state.total_iterations:
            return False

        return True


def restore_controller_state(
    controller: 'UltimateLoopController',
    state: LoopState
) -> None:
    """恢復 controller 狀態

    將 LoopState 的內容恢復到 UltimateLoopController。

    Args:
        controller: 要恢復的 controller
        state: 狀態物件
    """
    # 恢復 summary
    controller.summary.total_iterations = state.total_iterations
    controller.summary.successful_iterations = state.successful_iterations
    controller.summary.failed_iterations = state.failed_iterations
    controller.summary.best_strategy = state.best_strategy
    controller.summary.best_params = state.best_params
    controller.summary.best_objectives = state.best_objectives
    controller.summary.total_pareto_solutions = state.total_pareto_solutions
    controller.summary.validated_solutions = state.validated_solutions
    controller.summary.regime_distribution = state.regime_distribution
    controller.summary.gp_strategies_generated = state.gp_strategies_generated
    controller.summary.gp_strategies_validated = state.gp_strategies_validated
    controller.summary.new_insights = state.new_insights
    controller.summary.memory_entries = state.memory_entries
    controller.summary.experiments_recorded = state.experiments_recorded

    # 恢復策略統計
    if hasattr(controller, 'strategy_stats'):
        controller.strategy_stats = state.strategy_stats

    logger.info(f"已恢復 controller 狀態：迭代 {state.iteration}")
