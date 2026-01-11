"""
Loop 控制器單元測試
"""

import sys
from pathlib import Path
from datetime import datetime
import pytest
import time

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.automation.loop import (
    LoopController,
    LoopMode,
    LoopState,
    IterationResult,
    IterationStatus,
    create_loop_controller
)
import numpy as np


class TestIterationResult:
    """測試 IterationResult"""

    def test_create_success_result(self):
        """測試建立成功結果"""
        result = IterationResult(
            iteration=1,
            timestamp=datetime.now(),
            status=IterationStatus.SUCCESS,
            sharpe_ratio=1.85,
            total_return=0.45,
            max_drawdown=-0.12,
            strategy_name="MA Cross",
            best_params={'fast': 10, 'slow': 30},
            experiment_id="exp_001"
        )

        assert result.iteration == 1
        assert result.status == IterationStatus.SUCCESS
        assert result.sharpe_ratio == 1.85
        assert result.strategy_name == "MA Cross"

    def test_create_failed_result(self):
        """測試建立失敗結果"""
        result = IterationResult(
            iteration=2,
            timestamp=datetime.now(),
            status=IterationStatus.FAILED,
            sharpe_ratio=float('-inf'),
            total_return=0.0,
            max_drawdown=0.0,
            strategy_name="unknown",
            best_params={},
            error="Optimization failed"
        )

        assert result.status == IterationStatus.FAILED
        assert result.error == "Optimization failed"

    def test_to_dict(self):
        """測試轉換為字典"""
        result = IterationResult(
            iteration=1,
            timestamp=datetime.now(),
            status=IterationStatus.SUCCESS,
            sharpe_ratio=1.85,
            total_return=0.45,
            max_drawdown=-0.12,
            strategy_name="Test",
            best_params={}
        )

        data = result.to_dict()

        assert 'iteration' in data
        assert 'sharpe_ratio' in data
        assert 'status' in data
        assert data['status'] == 'success'


class TestLoopState:
    """測試 LoopState"""

    def test_create_state(self):
        """測試建立狀態"""
        state = LoopState(
            started_at=datetime.now(),
            mode=LoopMode.N_ITERATIONS.value,
            target=100
        )

        assert state.mode == LoopMode.N_ITERATIONS.value
        assert state.target == 100
        assert state.current_iteration == 0

    def test_state_serialization(self):
        """測試狀態序列化"""
        state = LoopState(
            started_at=datetime.now(),
            mode=LoopMode.CONTINUOUS.value,
            current_iteration=5,
            best_sharpe=1.85
        )

        # 轉為字典
        data = state.to_dict()
        assert isinstance(data['started_at'], str)

        # 從字典恢復
        restored = LoopState.from_dict(data)
        assert restored.mode == state.mode
        assert restored.current_iteration == state.current_iteration
        assert restored.best_sharpe == state.best_sharpe


class TestLoopController:
    """測試 LoopController"""

    @pytest.fixture
    def simple_callback(self):
        """簡單的迭代回調"""
        iteration_count = {'count': 0}

        def callback() -> IterationResult:
            iteration_count['count'] += 1
            sharpe = np.random.uniform(0.5, 2.5)

            return IterationResult(
                iteration=iteration_count['count'],
                timestamp=datetime.now(),
                status=IterationStatus.SUCCESS,
                sharpe_ratio=sharpe,
                total_return=0.3,
                max_drawdown=-0.1,
                strategy_name="Test Strategy",
                best_params={}
            )

        return callback

    @pytest.fixture
    def temp_state_file(self, tmp_path):
        """臨時狀態檔案"""
        return tmp_path / "test_loop_state.json"

    def test_controller_creation(self, simple_callback, temp_state_file):
        """測試建立控制器"""
        controller = LoopController(
            iteration_callback=simple_callback,
            state_file=temp_state_file
        )

        assert controller.iteration_callback == simple_callback
        assert controller.state_file == temp_state_file

    def test_n_iterations_mode(self, simple_callback, temp_state_file):
        """測試 N_ITERATIONS 模式"""
        controller = LoopController(
            iteration_callback=simple_callback,
            state_file=temp_state_file,
            auto_save=False
        )

        controller.start(
            mode=LoopMode.N_ITERATIONS,
            target=3
        )

        assert controller.state.completed_iterations == 3
        assert controller.state.successful_iterations == 3
        assert controller.state.is_stopped is True

    def test_until_target_mode(self, temp_state_file):
        """測試 UNTIL_TARGET 模式"""
        iteration_count = {'count': 0}

        def improving_callback() -> IterationResult:
            iteration_count['count'] += 1
            # Sharpe 逐步提升
            sharpe = 0.5 + (iteration_count['count'] * 0.4)

            return IterationResult(
                iteration=iteration_count['count'],
                timestamp=datetime.now(),
                status=IterationStatus.SUCCESS,
                sharpe_ratio=sharpe,
                total_return=0.3,
                max_drawdown=-0.1,
                strategy_name="Improving",
                best_params={}
            )

        controller = LoopController(
            iteration_callback=improving_callback,
            state_file=temp_state_file,
            auto_save=False
        )

        controller.start(
            mode=LoopMode.UNTIL_TARGET,
            target=2.0
        )

        assert controller.state.best_sharpe >= 2.0
        assert controller.state.is_stopped is True

    def test_callbacks(self, simple_callback, temp_state_file):
        """測試回調函數"""
        callback_tracker = {
            'on_iteration_start': 0,
            'on_success': 0,
            'on_new_best': 0,
            'on_loop_end': 0
        }

        def on_iteration_start(iteration_num):
            callback_tracker['on_iteration_start'] += 1

        def on_success(result):
            callback_tracker['on_success'] += 1

        def on_new_best(result):
            callback_tracker['on_new_best'] += 1

        def on_loop_end(state):
            callback_tracker['on_loop_end'] += 1

        callbacks = {
            'on_iteration_start': on_iteration_start,
            'on_success': on_success,
            'on_new_best': on_new_best,
            'on_loop_end': on_loop_end
        }

        controller = LoopController(
            iteration_callback=simple_callback,
            state_file=temp_state_file,
            auto_save=False,
            callbacks=callbacks
        )

        controller.start(
            mode=LoopMode.N_ITERATIONS,
            target=3
        )

        assert callback_tracker['on_iteration_start'] == 3
        assert callback_tracker['on_success'] == 3
        assert callback_tracker['on_loop_end'] == 1
        # on_new_best 可能被觸發多次（取決於隨機結果）
        assert callback_tracker['on_new_best'] >= 1

    def test_state_save_load(self, simple_callback, temp_state_file):
        """測試狀態保存和載入"""
        controller = LoopController(
            iteration_callback=simple_callback,
            state_file=temp_state_file,
            auto_save=True
        )

        # 執行一些迭代
        controller.start(
            mode=LoopMode.N_ITERATIONS,
            target=2
        )

        # 檢查檔案存在
        assert temp_state_file.exists()

        # 載入狀態
        loaded_state = controller.load_state()

        assert loaded_state.completed_iterations == 2
        assert loaded_state.mode == LoopMode.N_ITERATIONS.value

    def test_get_progress(self, simple_callback, temp_state_file):
        """測試取得進度"""
        controller = LoopController(
            iteration_callback=simple_callback,
            state_file=temp_state_file,
            auto_save=False
        )

        controller.start(
            mode=LoopMode.N_ITERATIONS,
            target=3
        )

        progress = controller.get_progress()

        assert 'completed_iterations' in progress
        assert 'successful_iterations' in progress
        assert 'best_sharpe' in progress
        assert progress['completed_iterations'] == 3

    def test_get_summary(self, simple_callback, temp_state_file):
        """測試取得摘要"""
        controller = LoopController(
            iteration_callback=simple_callback,
            state_file=temp_state_file,
            auto_save=False
        )

        controller.start(
            mode=LoopMode.N_ITERATIONS,
            target=2
        )

        summary = controller.get_summary()

        assert isinstance(summary, str)
        assert 'Loop 執行摘要' in summary
        assert 'Sharpe Ratio' in summary

    def test_get_iteration_history(self, simple_callback, temp_state_file):
        """測試取得迭代歷史"""
        controller = LoopController(
            iteration_callback=simple_callback,
            state_file=temp_state_file,
            auto_save=False
        )

        controller.start(
            mode=LoopMode.N_ITERATIONS,
            target=3
        )

        history_df = controller.get_iteration_history()

        assert len(history_df) == 3
        assert 'iteration' in history_df.columns
        assert 'sharpe_ratio' in history_df.columns
        assert 'status' in history_df.columns

    def test_clear_state(self, simple_callback, temp_state_file):
        """測試清除狀態"""
        controller = LoopController(
            iteration_callback=simple_callback,
            state_file=temp_state_file,
            auto_save=True
        )

        controller.start(
            mode=LoopMode.N_ITERATIONS,
            target=1
        )

        assert temp_state_file.exists()

        controller.clear_state()

        assert not temp_state_file.exists()


class TestConvenienceFunctions:
    """測試便利函數"""

    def test_create_loop_controller(self):
        """測試建立控制器便利函數"""

        def dummy_callback() -> IterationResult:
            return IterationResult(
                iteration=1,
                timestamp=datetime.now(),
                status=IterationStatus.SUCCESS,
                sharpe_ratio=1.0,
                total_return=0.1,
                max_drawdown=-0.05,
                strategy_name="Test",
                best_params={}
            )

        controller = create_loop_controller(
            iteration_callback=dummy_callback,
            auto_save=False
        )

        assert isinstance(controller, LoopController)
        assert controller.auto_save is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
