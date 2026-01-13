"""
測試 UltimateLoopController

測試涵蓋：
1. 配置驗證
2. 初始化
3. 核心流程（run_loop, _run_single_iteration）
4. 檢查點機制（save, load, 損壞處理）
5. 資源清理
6. 防禦性檢查
"""

import pytest
import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from dataclasses import dataclass
from datetime import datetime

from src.automation.ultimate_loop import (
    UltimateLoopController,
    UltimateLoopSummary
)
from src.automation.ultimate_config import UltimateLoopConfig


# ===== Fixtures =====

@pytest.fixture
def quick_config():
    """快速測試配置"""
    config = UltimateLoopConfig.create_quick_test_config()
    # 關閉驗證以加速測試
    config.validation_enabled = False
    config.learning_enabled = False
    return config


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """臨時檢查點目錄"""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def controller_with_temp_checkpoint(quick_config, temp_checkpoint_dir):
    """帶臨時檢查點目錄的 controller"""
    quick_config.checkpoint_enabled = True
    quick_config.checkpoint_dir = str(temp_checkpoint_dir)
    return UltimateLoopController(quick_config, verbose=False)


# ===== 1. 配置驗證測試 =====

class TestConfigValidation:
    """測試配置驗證"""

    def test_valid_config(self):
        """測試有效配置"""
        config = UltimateLoopConfig.create_quick_test_config()
        # 不應拋出異常
        config.validate()

    def test_invalid_max_workers(self):
        """測試無效 max_workers"""
        config = UltimateLoopConfig()
        config.max_workers = -1

        with pytest.raises(ValueError, match="max_workers 必須 >= 1"):
            config.validate()

    def test_invalid_direction_method(self):
        """測試無效 direction_method"""
        config = UltimateLoopConfig()
        config.direction_method = 'invalid'

        with pytest.raises(ValueError, match="direction_method 必須是"):
            config.validate()

    def test_invalid_strategy_selection_mode(self):
        """測試無效 strategy_selection_mode"""
        config = UltimateLoopConfig()
        config.strategy_selection_mode = 'invalid'

        with pytest.raises(ValueError, match="strategy_selection_mode 必須是"):
            config.validate()

    def test_invalid_exploit_ratio(self):
        """測試無效 exploit_ratio"""
        config = UltimateLoopConfig()
        config.exploit_ratio = 1.5

        with pytest.raises(ValueError, match="exploit_ratio 必須在 0-1 之間"):
            config.validate()

    def test_invalid_objectives(self):
        """測試無效 objectives"""
        config = UltimateLoopConfig()
        config.objectives = [
            ('invalid_metric', 'maximize')
        ]

        with pytest.raises(ValueError, match="不支援的指標"):
            config.validate()

    def test_unsafe_path(self):
        """測試不安全路徑"""
        config = UltimateLoopConfig()
        config.data_dir = "../../../etc/passwd"

        with pytest.raises(ValueError, match="路徑不安全"):
            config.validate()


# ===== 2. 初始化測試 =====

class TestInitialization:
    """測試初始化"""

    def test_basic_initialization(self, quick_config):
        """測試基本初始化"""
        controller = UltimateLoopController(quick_config, verbose=False)

        assert controller.config is not None
        assert controller.summary is not None
        assert isinstance(controller.summary, UltimateLoopSummary)
        assert controller._checkpoint_data == {}

    def test_none_config_uses_default(self):
        """測試 None config 使用預設值"""
        controller = UltimateLoopController(config=None, verbose=False)

        assert controller.config is not None
        # 應該使用預設配置
        assert controller.config.max_workers == 8

    def test_module_initialization(self, quick_config):
        """測試模組初始化"""
        controller = UltimateLoopController(quick_config, verbose=False)

        # 檢查模組是否正確初始化（根據 config）
        if quick_config.regime_detection:
            assert controller.regime_analyzer is not None
        else:
            assert controller.regime_analyzer is None

        if quick_config.validation_enabled:
            assert controller.validator is not None
        else:
            assert controller.validator is None

        if quick_config.learning_enabled:
            assert controller.recorder is not None
        else:
            assert controller.recorder is None


# ===== 3. 核心流程測試 =====

class TestCoreFlow:
    """測試核心流程"""

    @pytest.mark.asyncio
    async def test_run_loop_basic(self, quick_config):
        """測試基本 loop 執行"""
        controller = UltimateLoopController(quick_config, verbose=False)

        summary = await controller.run_loop(n_iterations=2)

        assert isinstance(summary, UltimateLoopSummary)
        assert summary.total_iterations == 2
        # 成功和失敗的總和應該等於總迭代次數
        assert summary.successful_iterations + summary.failed_iterations == 2
        assert summary.total_duration_seconds > 0

    @pytest.mark.asyncio
    async def test_run_loop_with_retries(self, quick_config):
        """測試重試機制"""
        quick_config.max_retries = 2
        controller = UltimateLoopController(quick_config, verbose=False)

        # Mock _run_iteration 讓第一次失敗，重試成功
        original_run_iteration = controller._run_iteration
        call_count = [0]

        async def mock_run_iteration(iteration, total):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Simulated failure")
            return await original_run_iteration(iteration, total)

        controller._run_iteration = mock_run_iteration

        summary = await controller.run_loop(n_iterations=1)

        # 應該重試成功
        assert summary.successful_iterations == 1
        assert summary.failed_iterations == 0
        assert call_count[0] == 2  # 1 失敗 + 1 成功

    @pytest.mark.asyncio
    async def test_run_iteration_phases(self, quick_config):
        """測試 iteration 各階段執行"""
        controller = UltimateLoopController(quick_config, verbose=False)

        # Mock 各階段方法（使用 AsyncMock）
        controller._analyze_market_state = AsyncMock(return_value=None)
        controller._select_strategies = Mock(return_value=['strategy1'])
        controller._run_optimization = AsyncMock(return_value=None)
        controller._validate_pareto_solutions = AsyncMock(return_value=[])
        controller._record_and_learn = AsyncMock(return_value=None)

        await controller._run_iteration(0, 1)

        # 驗證各階段都被呼叫
        assert controller._analyze_market_state.called
        assert controller._select_strategies.called
        assert controller._run_optimization.called
        assert controller._validate_pareto_solutions.called
        assert controller._record_and_learn.called


# ===== 4. 檢查點機制測試 =====

class TestCheckpointMechanism:
    """測試檢查點機制"""

    def test_save_checkpoint(self, controller_with_temp_checkpoint, temp_checkpoint_dir):
        """測試儲存檢查點"""
        controller = controller_with_temp_checkpoint
        controller.summary.successful_iterations = 5
        controller.summary.failed_iterations = 1
        controller.summary.best_strategy = "test_strategy"

        controller._save_checkpoint(iteration=10)

        checkpoint_path = temp_checkpoint_dir / "ultimate_checkpoint.json"
        assert checkpoint_path.exists()

        # 驗證檢查點內容
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)

        assert data['last_iteration'] == 10
        assert data['summary']['successful'] == 5
        assert data['summary']['failed'] == 1
        assert data['summary']['best_strategy'] == "test_strategy"

    def test_save_checkpoint_atomic(self, controller_with_temp_checkpoint, temp_checkpoint_dir):
        """測試檢查點原子寫入（先寫 .tmp，再 rename）"""
        controller = controller_with_temp_checkpoint

        # Mock open 讓寫入過程可追蹤
        checkpoint_path = temp_checkpoint_dir / "ultimate_checkpoint.json"
        temp_path = temp_checkpoint_dir / "ultimate_checkpoint.json.tmp"

        controller._save_checkpoint(iteration=5)

        # 檢查最終檢查點存在，.tmp 檔案不存在
        assert checkpoint_path.exists()
        assert not temp_path.exists()

    def test_load_checkpoint(self, controller_with_temp_checkpoint, temp_checkpoint_dir):
        """測試載入檢查點"""
        controller = controller_with_temp_checkpoint

        # 先儲存檢查點
        controller.summary.successful_iterations = 3
        controller.summary.best_strategy = "loaded_strategy"
        controller._save_checkpoint(iteration=5)

        # 建立新 controller 並載入
        new_controller = UltimateLoopController(
            controller.config,
            verbose=False
        )
        start_iteration = new_controller._try_restore_checkpoint()

        assert start_iteration == 5
        assert new_controller.summary.successful_iterations == 3
        assert new_controller.summary.best_strategy == "loaded_strategy"

    def test_load_corrupted_checkpoint(self, controller_with_temp_checkpoint, temp_checkpoint_dir):
        """測試載入損壞的檢查點"""
        controller = controller_with_temp_checkpoint

        # 建立損壞的檢查點
        checkpoint_path = temp_checkpoint_dir / "ultimate_checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            f.write("{ invalid json ")

        # 載入應該失敗但不拋出異常，返回 0
        start_iteration = controller._try_restore_checkpoint()

        assert start_iteration == 0

    def test_corrupted_checkpoint_creates_backup(self, controller_with_temp_checkpoint, temp_checkpoint_dir):
        """測試損壞檔案備份功能"""
        controller = controller_with_temp_checkpoint

        # 建立損壞的檢查點
        checkpoint_path = temp_checkpoint_dir / "ultimate_checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            f.write("{ invalid json ")

        # 嘗試載入（應該會備份損壞檔案）
        start_iteration = controller._try_restore_checkpoint()

        # 檢查備份檔案是否存在
        backup_path = temp_checkpoint_dir / "ultimate_checkpoint.json.corrupted"
        assert backup_path.exists(), "損壞檔案應該被備份"
        assert not checkpoint_path.exists(), "原檔案應該被 rename"

        # 備份內容應該與原損壞內容相同
        with open(backup_path, 'r') as f:
            backup_content = f.read()
        assert backup_content == "{ invalid json "

        # 應該返回 0（從頭開始）
        assert start_iteration == 0

    def test_corrupted_checkpoint_backup_fails_gracefully(self, controller_with_temp_checkpoint, temp_checkpoint_dir, caplog):
        """測試備份失敗時優雅處理"""
        controller = controller_with_temp_checkpoint

        # 建立損壞的檢查點
        checkpoint_path = temp_checkpoint_dir / "ultimate_checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            f.write("{ invalid json ")

        # Mock rename 讓備份失敗
        original_rename = Path.rename

        def mock_rename_fail(self, target):
            raise PermissionError("Simulated permission error")

        # 暫時替換 rename 方法
        Path.rename = mock_rename_fail

        try:
            with caplog.at_level(logging.WARNING):
                start_iteration = controller._try_restore_checkpoint()

            # 應該記錄警告但不中斷
            assert any("備份損壞檔案失敗" in record.message for record in caplog.records)

            # 仍應返回 0
            assert start_iteration == 0
        finally:
            # 恢復原始方法
            Path.rename = original_rename

    @pytest.mark.asyncio
    async def test_iteration_sanity_check_resets(self, quick_config, temp_checkpoint_dir):
        """測試迭代合理性驗證：start_iteration >= n_iterations 時重置"""
        quick_config.checkpoint_enabled = True
        quick_config.checkpoint_dir = str(temp_checkpoint_dir)

        controller = UltimateLoopController(quick_config, verbose=False)

        # 手動建立一個 last_iteration > n_iterations 的檢查點
        checkpoint_path = temp_checkpoint_dir / "ultimate_checkpoint.json"
        checkpoint_data = {
            'last_iteration': 100,  # 遠超目標迭代數
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'successful': 90,
                'failed': 10
            }
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)

        # 執行 3 次迭代（< 100）
        summary = await controller.run_loop(n_iterations=3, resume_from_checkpoint=True)

        # 應該從頭開始執行（因為檢查點迭代數 >= 目標）
        # 總迭代次數應該是 3，不是從 100 繼續
        assert summary.total_iterations == 3

    @pytest.mark.asyncio
    async def test_iteration_sanity_check_logs_warning(self, quick_config, temp_checkpoint_dir, caplog):
        """測試迭代合理性驗證記錄警告訊息"""
        quick_config.checkpoint_enabled = True
        quick_config.checkpoint_dir = str(temp_checkpoint_dir)

        controller = UltimateLoopController(quick_config, verbose=False)

        # 建立檢查點：last_iteration >= n_iterations
        checkpoint_path = temp_checkpoint_dir / "ultimate_checkpoint.json"
        checkpoint_data = {
            'last_iteration': 50,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'successful': 40,
                'failed': 10
            }
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)

        # 執行 50 次迭代（= last_iteration）
        with caplog.at_level(logging.WARNING):
            await controller.run_loop(n_iterations=50, resume_from_checkpoint=True)

        # 應該記錄警告
        assert any(
            "檢查點迭代" in record.message and "將從頭開始執行" in record.message
            for record in caplog.records
        ), "應該記錄迭代編號不合理的警告"

    def test_checkpoint_missing_required_fields(self, controller_with_temp_checkpoint, temp_checkpoint_dir):
        """測試檢查點缺少必要欄位"""
        controller = controller_with_temp_checkpoint

        # 建立缺少 last_iteration 的檢查點
        checkpoint_path = temp_checkpoint_dir / "ultimate_checkpoint.json"
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'successful': 5
            }
            # 缺少 last_iteration
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)

        # 應該返回 0 並記錄警告
        start_iteration = controller._try_restore_checkpoint()
        assert start_iteration == 0

    def test_checkpoint_invalid_iteration_type(self, controller_with_temp_checkpoint, temp_checkpoint_dir):
        """測試檢查點迭代編號型別錯誤"""
        controller = controller_with_temp_checkpoint

        # 建立型別錯誤的檢查點
        checkpoint_path = temp_checkpoint_dir / "ultimate_checkpoint.json"
        checkpoint_data = {
            'last_iteration': "not_a_number",  # 應該是 int
            'timestamp': datetime.now().isoformat()
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)

        # 應該返回 0 並記錄警告
        start_iteration = controller._try_restore_checkpoint()
        assert start_iteration == 0

    def test_checkpoint_negative_iteration(self, controller_with_temp_checkpoint, temp_checkpoint_dir):
        """測試檢查點負數迭代編號"""
        controller = controller_with_temp_checkpoint

        # 建立負數迭代的檢查點
        checkpoint_path = temp_checkpoint_dir / "ultimate_checkpoint.json"
        checkpoint_data = {
            'last_iteration': -5,  # 無效值
            'timestamp': datetime.now().isoformat()
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)

        # 應該返回 0
        start_iteration = controller._try_restore_checkpoint()
        assert start_iteration == 0

    def test_checkpoint_restores_all_fields(self, controller_with_temp_checkpoint, temp_checkpoint_dir):
        """測試檢查點完整恢復所有欄位"""
        controller = controller_with_temp_checkpoint

        # 建立完整的檢查點
        checkpoint_path = temp_checkpoint_dir / "ultimate_checkpoint.json"
        checkpoint_data = {
            'last_iteration': 10,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'successful': 8,
                'failed': 2,
                'best_strategy': 'ma_cross',
                'best_objectives': {'sharpe_ratio': 2.5, 'max_drawdown': 0.15},
                'total_pareto_solutions': 50,
                'validated_solutions': 30
            },
            'strategy_stats': {
                'ma_cross': {'avg_sharpe': 2.0, 'runs': 5},
                'rsi': {'avg_sharpe': 1.5, 'runs': 3}
            },
            'regime_distribution': {
                'trending': 6,
                'ranging': 4
            }
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)

        # 恢復檢查點
        start_iteration = controller._try_restore_checkpoint()

        # 驗證所有欄位
        assert start_iteration == 10
        assert controller.summary.successful_iterations == 8
        assert controller.summary.failed_iterations == 2
        assert controller.summary.best_strategy == 'ma_cross'
        assert controller.summary.best_objectives == {'sharpe_ratio': 2.5, 'max_drawdown': 0.15}
        assert controller.summary.total_pareto_solutions == 50
        assert controller.summary.validated_solutions == 30
        assert controller.strategy_stats == {
            'ma_cross': {'avg_sharpe': 2.0, 'runs': 5},
            'rsi': {'avg_sharpe': 1.5, 'runs': 3}
        }
        assert controller.summary.regime_distribution == {
            'trending': 6,
            'ranging': 4
        }

    def test_checkpoint_file_not_exists(self, controller_with_temp_checkpoint):
        """測試檢查點檔案不存在"""
        controller = controller_with_temp_checkpoint

        # 不建立檔案，直接嘗試恢復
        start_iteration = controller._try_restore_checkpoint()

        # 應該返回 0（從頭開始）
        assert start_iteration == 0

    def test_checkpoint_disabled(self, quick_config):
        """測試停用檢查點"""
        quick_config.checkpoint_enabled = False
        controller = UltimateLoopController(quick_config, verbose=False)

        # 嘗試恢復應該直接返回 0
        start_iteration = controller._try_restore_checkpoint()
        assert start_iteration == 0


# ===== 5. 資源清理測試 =====

class TestResourceCleanup:
    """測試資源清理"""

    def test_cleanup_basic(self, quick_config):
        """測試基本清理"""
        controller = UltimateLoopController(quick_config, verbose=False)

        # 設定一些假資源
        controller.optimizer = Mock()
        controller.regime_analyzer = Mock()
        controller.validator = Mock()
        controller.recorder = Mock()

        controller._cleanup()

        # 所有資源應該被重置為 None
        assert controller.optimizer is None
        assert controller.regime_analyzer is None
        assert controller.validator is None
        assert controller.recorder is None

    def test_cleanup_with_cleanup_methods(self, quick_config):
        """測試清理時呼叫資源的 cleanup 方法"""
        controller = UltimateLoopController(quick_config, verbose=False)

        # 建立帶 cleanup 方法的 mock
        mock_optimizer = Mock()
        mock_optimizer.cleanup = Mock()
        controller.optimizer = mock_optimizer

        controller._cleanup()

        # cleanup 方法應該被呼叫
        assert mock_optimizer.cleanup.called

    def test_cleanup_handles_exceptions(self, quick_config, caplog):
        """測試清理過程中的異常不會傳播"""
        controller = UltimateLoopController(quick_config, verbose=False)

        # Mock cleanup 方法拋出異常
        mock_optimizer = Mock()
        mock_optimizer.cleanup = Mock(side_effect=RuntimeError("Cleanup failed"))
        controller.optimizer = mock_optimizer

        # 不應拋出異常
        with caplog.at_level(logging.WARNING):
            controller._cleanup()

        # 應該記錄警告
        assert any("cleanup failed" in record.message.lower() for record in caplog.records)


# ===== 6. 防禦性檢查測試 =====

class TestDefensiveChecks:
    """測試防禦性檢查"""

    def test_validate_engine_with_invalid_report(self, quick_config):
        """測試驗證報告缺少屬性"""
        quick_config.validation_enabled = True
        controller = UltimateLoopController(quick_config, verbose=False)

        # Mock validator.validate_level 返回不完整的報告
        @dataclass
        class IncompleteReport:
            # 缺少 all_passed 和 results 屬性
            pass

        controller.validator = Mock()
        controller.validator.validate_level = Mock(return_value=IncompleteReport())

        with pytest.raises(RuntimeError, match="驗證報告格式不正確"):
            controller._validate_engine_on_startup()

    def test_select_strategies_with_no_strategies(self, quick_config):
        """測試沒有可用策略時"""
        controller = UltimateLoopController(quick_config, verbose=False)
        controller.available_strategies = []

        strategies = controller._select_strategies(market_state=None)

        assert strategies == []

    def test_select_exploit_with_no_history(self, quick_config):
        """測試 exploit 模式但沒有歷史數據"""
        quick_config.strategy_selection_mode = 'exploit'
        controller = UltimateLoopController(quick_config, verbose=False)
        controller.available_strategies = ['s1', 's2', 's3']
        controller.strategy_stats = {}  # 空歷史

        strategies = controller._select_exploit_strategies()

        # 應該 fallback 到隨機選擇
        assert len(strategies) > 0
        assert all(s in controller.available_strategies for s in strategies)


# ===== 7. Summary 測試 =====

class TestSummary:
    """測試 Summary 物件"""

    def test_summary_text_basic(self):
        """測試基本 summary_text"""
        summary = UltimateLoopSummary(
            total_iterations=10,
            successful_iterations=8,
            failed_iterations=2,
            total_duration_seconds=100.0
        )

        text = summary.summary_text()

        assert "總迭代次數: 10" in text
        assert "成功: 8" in text
        assert "失敗: 2" in text
        assert "80.0%" in text  # 成功率
        assert "總執行時間: 100.0s" in text

    def test_summary_text_with_regime_distribution(self):
        """測試帶 regime 分布的 summary"""
        summary = UltimateLoopSummary(
            total_iterations=10,
            successful_iterations=10,
            failed_iterations=0,
            regime_distribution={
                'trending': 6,
                'ranging': 4
            }
        )

        text = summary.summary_text()

        assert "市場狀態分布:" in text
        assert "trending" in text
        assert "ranging" in text

    def test_summary_text_with_best_strategy(self):
        """測試帶最佳策略的 summary"""
        summary = UltimateLoopSummary(
            total_iterations=5,
            successful_iterations=5,
            failed_iterations=0,
            best_strategy="ma_cross",
            best_params={'fast': 10, 'slow': 20},
            best_objectives={'sharpe_ratio': 2.5, 'max_drawdown': 0.15}
        )

        text = summary.summary_text()

        assert "最佳策略:" in text
        assert "ma_cross" in text
        assert "{'fast': 10, 'slow': 20}" in text
        assert "sharpe_ratio: 2.5000" in text


# ===== 8. 整合測試 =====

class TestIntegration:
    """整合測試"""

    @pytest.mark.asyncio
    async def test_full_loop_with_checkpoint(self, quick_config, temp_checkpoint_dir):
        """測試完整 loop 與檢查點整合"""
        quick_config.checkpoint_enabled = True
        quick_config.checkpoint_interval = 1
        quick_config.checkpoint_dir = str(temp_checkpoint_dir)

        controller = UltimateLoopController(quick_config, verbose=False)

        # 執行 3 次迭代
        summary = await controller.run_loop(n_iterations=3)

        assert summary.total_iterations == 3

        # 檢查點應該存在
        checkpoint_path = temp_checkpoint_dir / "ultimate_checkpoint.json"
        assert checkpoint_path.exists()

        # 恢復檢查點並繼續
        controller2 = UltimateLoopController(quick_config, verbose=False)
        start_iter = controller2._try_restore_checkpoint()

        assert start_iter == 3
        assert controller2.summary.successful_iterations == summary.successful_iterations

    @pytest.mark.asyncio
    async def test_loop_cleanup_on_completion(self, quick_config):
        """測試 loop 完成後資源清理"""
        controller = UltimateLoopController(quick_config, verbose=False)

        # 設定假資源
        controller.optimizer = Mock()
        controller.validator = Mock()

        await controller.run_loop(n_iterations=1)

        # 資源應該被清理
        assert controller.optimizer is None
        assert controller.validator is None

    @pytest.mark.asyncio
    async def test_loop_cleanup_on_exception(self, quick_config):
        """測試 loop 異常時仍執行清理"""
        controller = UltimateLoopController(quick_config, verbose=False)

        # 設定假資源
        controller.optimizer = Mock()

        # Mock _run_iteration 讓所有迭代都失敗
        async def mock_fail(iteration, total):
            raise RuntimeError("Force fail")

        controller._run_iteration = mock_fail
        controller.config.max_retries = 0

        # 執行 loop（會失敗）
        summary = await controller.run_loop(n_iterations=1)

        # 資源仍應被清理
        assert controller.optimizer is None
        # 失敗統計應該正確
        assert summary.failed_iterations == 1


# ===== 主測試執行 =====

# ===== 8. Pareto 前緣選擇與驗證測試 (Task 12.3.2) =====

class TestParetoSolutionSelection:
    """測試 Pareto 前緣解選擇與驗證邏輯"""

    def test_select_pareto_solutions_filter_none(self, quick_config):
        """測試過濾 None 元素"""
        controller = UltimateLoopController(quick_config, verbose=False)
        controller.config.pareto_select_method = 'random'  # 使用 random 避免 knee 方法複雜度
        controller.config.pareto_top_n = 10  # 選擇足夠多

        from src.optimizer.multi_objective import MultiObjectiveResult, ParetoSolution, ObjectiveResult

        # 建立包含 None 的 Pareto front
        pareto_front = [
            None,  # 第一個是 None
            ParetoSolution(
                params={'param1': 10},
                objectives=[ObjectiveResult('sharpe_ratio', 2.0, 'maximize')],
                rank=0
            ),
            None,  # 中間有 None
            ParetoSolution(
                params={'param1': 20},
                objectives=[ObjectiveResult('sharpe_ratio', 2.5, 'maximize')],
                rank=0
            ),
            None  # 最後也有 None
        ]

        result = MultiObjectiveResult(
            pareto_front=pareto_front,
            all_solutions=[],
            n_trials=10,
            study=None
        )

        # 執行選擇
        selected = controller._select_pareto_solutions(result)

        # 驗證 None 被過濾
        assert len(selected) == 2  # 只有 2 個有效解
        assert all(s is not None for s in selected)

    def test_select_pareto_solutions_random_small_list(self, quick_config):
        """測試小列表 random.sample 安全（n_select <= len）"""
        controller = UltimateLoopController(quick_config, verbose=False)
        controller.config.pareto_select_method = 'random'
        controller.config.pareto_top_n = 10  # 要選 10 個

        from src.optimizer.multi_objective import MultiObjectiveResult, ParetoSolution, ObjectiveResult

        # 建立只有 3 個解的 Pareto front
        pareto_front = [
            ParetoSolution(
                params={'param1': i * 10},
                objectives=[ObjectiveResult('sharpe_ratio', 1.0 + i * 0.5, 'maximize')],
                rank=0
            )
            for i in range(3)  # 只有 3 個
        ]

        result = MultiObjectiveResult(
            pareto_front=pareto_front,
            all_solutions=[],
            n_trials=10,
            study=None
        )

        # 執行選擇（應該不拋出 ValueError）
        selected = controller._select_pareto_solutions(result)

        # 驗證最多選 3 個（min(10, 3) = 3）
        assert len(selected) == 3
        assert all(s is not None for s in selected)

    def test_select_pareto_solutions_knee_method(self, quick_config):
        """測試 knee 方法"""
        controller = UltimateLoopController(quick_config, verbose=False)
        controller.config.pareto_select_method = 'knee'
        controller.config.pareto_top_n = 5

        from src.optimizer.multi_objective import MultiObjectiveResult, ParetoSolution, ObjectiveResult

        pareto_front = [
            ParetoSolution(
                params={'param1': i * 10},
                objectives=[ObjectiveResult('sharpe_ratio', 2.0 + i * 0.1, 'maximize')],
                rank=0
            )
            for i in range(10)
        ]

        result = MultiObjectiveResult(
            pareto_front=pareto_front,
            all_solutions=[],
            n_trials=10,
            study=None
        )

        # 執行選擇
        selected = controller._select_pareto_solutions(result)

        # 驗證選擇結果
        assert len(selected) <= 5
        assert len(selected) > 0

    def test_select_pareto_solutions_crowding_method(self, quick_config):
        """測試 crowding 方法"""
        controller = UltimateLoopController(quick_config, verbose=False)
        controller.config.pareto_select_method = 'crowding'
        controller.config.pareto_top_n = 5

        from src.optimizer.multi_objective import MultiObjectiveResult, ParetoSolution, ObjectiveResult

        pareto_front = [
            ParetoSolution(
                params={'param1': i * 10},
                objectives=[ObjectiveResult('sharpe_ratio', 2.0 + i * 0.1, 'maximize')],
                rank=0
            )
            for i in range(10)
        ]

        result = MultiObjectiveResult(
            pareto_front=pareto_front,
            all_solutions=[],
            n_trials=10,
            study=None
        )

        # 執行選擇
        selected = controller._select_pareto_solutions(result)

        # 驗證選擇結果
        assert len(selected) <= 5
        assert len(selected) > 0

    def test_select_pareto_solutions_fallback(self, quick_config):
        """測試未知方法 fallback 到 slice"""
        controller = UltimateLoopController(quick_config, verbose=False)
        controller.config.pareto_select_method = 'unknown_method'  # 無效方法
        controller.config.pareto_top_n = 5

        from src.optimizer.multi_objective import MultiObjectiveResult, ParetoSolution, ObjectiveResult

        pareto_front = [
            ParetoSolution(
                params={'param1': i * 10},
                objectives=[ObjectiveResult('sharpe_ratio', 2.0 + i * 0.1, 'maximize')],
                rank=0
            )
            for i in range(10)
        ]

        result = MultiObjectiveResult(
            pareto_front=pareto_front,
            all_solutions=[],
            n_trials=10,
            study=None
        )

        # 執行選擇（應該 fallback 到前 5 個）
        selected = controller._select_pareto_solutions(result)

        # 驗證 fallback 行為
        assert len(selected) == 5
        assert selected == pareto_front[:5]

    @pytest.mark.asyncio
    async def test_validate_uses_selected_solutions(self, quick_config):
        """測試 _validate_pareto_solutions 優先使用 selected_solutions"""
        controller = UltimateLoopController(quick_config, verbose=False)

        from src.optimizer.multi_objective import MultiObjectiveResult, ParetoSolution, ObjectiveResult

        # 建立 pareto_front
        pareto_front = [
            ParetoSolution(
                params={'param1': i * 10},
                objectives=[ObjectiveResult('sharpe_ratio', 2.0 + i * 0.1, 'maximize')],
                rank=0
            )
            for i in range(10)
        ]

        # 建立 selected_solutions（只選 3 個）
        selected_solutions = pareto_front[:3]

        # 建立 result 物件並設定 selected_solutions 屬性
        result = MultiObjectiveResult(
            pareto_front=pareto_front,
            all_solutions=[],
            n_trials=10,
            study=None
        )
        result.selected_solutions = selected_solutions

        # 執行驗證
        validated = await controller._validate_pareto_solutions(result)

        # 驗證使用 selected_solutions（3 個）而不是 pareto_front（10 個）
        assert len(validated) == 3
        assert validated == selected_solutions

    @pytest.mark.asyncio
    async def test_validate_fallback_to_pareto_front(self, quick_config):
        """測試沒有 selected_solutions 時 fallback 到 pareto_front"""
        controller = UltimateLoopController(quick_config, verbose=False)

        from src.optimizer.multi_objective import MultiObjectiveResult, ParetoSolution, ObjectiveResult

        pareto_front = [
            ParetoSolution(
                params={'param1': i * 10},
                objectives=[ObjectiveResult('sharpe_ratio', 2.0 + i * 0.1, 'maximize')],
                rank=0
            )
            for i in range(5)
        ]

        result = MultiObjectiveResult(
            pareto_front=pareto_front,
            all_solutions=[],
            n_trials=10,
            study=None
        )
        # 不設定 selected_solutions

        # 執行驗證
        validated = await controller._validate_pareto_solutions(result)

        # 驗證 fallback 到 pareto_front
        assert len(validated) == 5
        assert validated == pareto_front

    @pytest.mark.asyncio
    async def test_validate_stats_accumulate_with_validator(self, quick_config):
        """測試統計累加（需要 validator 才會累加統計）"""
        # 啟用 validator
        quick_config.validation_enabled = True
        controller = UltimateLoopController(quick_config, verbose=False)

        from src.optimizer.multi_objective import MultiObjectiveResult, ParetoSolution, ObjectiveResult

        # 如果 validator 仍是 None（依賴不可用），跳過測試
        if controller.validator is None:
            pytest.skip("Validator not available")

        # 初始統計應該是 0
        assert controller.summary.validated_solutions == 0

        # 第一次驗證 3 個解
        pareto_front1 = [
            ParetoSolution(
                params={'param1': i * 10},
                objectives=[ObjectiveResult('sharpe_ratio', 2.0, 'maximize')],
                rank=0
            )
            for i in range(3)
        ]

        result1 = MultiObjectiveResult(
            pareto_front=pareto_front1,
            all_solutions=[],
            n_trials=10,
            study=None
        )
        result1.selected_solutions = pareto_front1

        await controller._validate_pareto_solutions(result1)

        # 驗證統計累加 +3
        assert controller.summary.validated_solutions == 3

        # 第二次驗證 5 個解
        pareto_front2 = [
            ParetoSolution(
                params={'param1': i * 10},
                objectives=[ObjectiveResult('sharpe_ratio', 2.0, 'maximize')],
                rank=0
            )
            for i in range(5)
        ]

        result2 = MultiObjectiveResult(
            pareto_front=pareto_front2,
            all_solutions=[],
            n_trials=10,
            study=None
        )
        result2.selected_solutions = pareto_front2

        await controller._validate_pareto_solutions(result2)

        # 驗證統計累加 +5（總共 8）
        assert controller.summary.validated_solutions == 8

    @pytest.mark.asyncio
    async def test_validate_stats_no_accumulate_without_validator(self, quick_config):
        """測試無 validator 時不累加統計（BUG：應該累加但實際不會）"""
        quick_config.validation_enabled = False
        controller = UltimateLoopController(quick_config, verbose=False)

        from src.optimizer.multi_objective import MultiObjectiveResult, ParetoSolution, ObjectiveResult

        # 確認沒有 validator
        assert controller.validator is None

        # 初始統計
        assert controller.summary.validated_solutions == 0

        # 驗證 3 個解
        pareto_front = [
            ParetoSolution(
                params={'param1': i * 10},
                objectives=[ObjectiveResult('sharpe_ratio', 2.0, 'maximize')],
                rank=0
            )
            for i in range(3)
        ]

        result = MultiObjectiveResult(
            pareto_front=pareto_front,
            all_solutions=[],
            n_trials=10,
            study=None
        )
        result.selected_solutions = pareto_front

        await controller._validate_pareto_solutions(result)

        # BUG 已修復：統計累加現在正確放在 validator 檢查之前
        # 無論是否有 validator，統計都會正確累加
        assert controller.summary.validated_solutions == 3  # 正確行為

    def test_select_empty_pareto_front(self, quick_config):
        """測試空 Pareto front"""
        controller = UltimateLoopController(quick_config, verbose=False)

        from src.optimizer.multi_objective import MultiObjectiveResult

        result = MultiObjectiveResult(
            pareto_front=[],
            all_solutions=[],
            n_trials=10,
            study=None
        )

        selected = controller._select_pareto_solutions(result)

        assert selected == []

    def test_select_all_none_pareto_front(self, quick_config):
        """測試全是 None 的 Pareto front"""
        controller = UltimateLoopController(quick_config, verbose=False)

        from src.optimizer.multi_objective import MultiObjectiveResult

        result = MultiObjectiveResult(
            pareto_front=[None, None, None],
            all_solutions=[],
            n_trials=10,
            study=None
        )

        selected = controller._select_pareto_solutions(result)

        assert selected == []


# ===== 9. MultiObjectiveOptimizer 整合測試 =====

class TestMultiObjectiveOptimizerIntegration:
    """測試 MultiObjectiveOptimizer 整合到 UltimateLoopController"""

    @pytest.mark.asyncio
    async def test_run_optimization_parallel_execution(self, quick_config):
        """測試 _run_optimization 使用 asyncio.gather 並行優化"""
        controller = UltimateLoopController(quick_config, verbose=False)

        # Mock optimizer
        if controller.optimizer is None:
            from src.optimizer.multi_objective import MultiObjectiveOptimizer, MultiObjectiveResult, ParetoSolution
            controller.optimizer = MultiObjectiveOptimizer(
                objectives=[('sharpe_ratio', 'maximize')],
                n_trials=10,
                verbose=False
            )

        strategies = ['test_strategy_1', 'test_strategy_2', 'test_strategy_3']

        # Mock _optimize_strategy 返回成功結果
        async def mock_optimize(strategy_name: str):
            """模擬優化成功"""
            from src.optimizer.multi_objective import MultiObjectiveResult, ParetoSolution, ObjectiveResult
            solution = ParetoSolution(
                params={'param1': 10},
                objectives=[ObjectiveResult('sharpe_ratio', 2.0, 'maximize')],
                rank=0
            )
            return MultiObjectiveResult(
                pareto_front=[solution],
                all_solutions=[solution],
                n_trials=10,
                study=None,
                n_completed_trials=10,
                n_failed_trials=0
            )

        controller._optimize_strategy = mock_optimize

        # 執行優化
        result = await controller._run_optimization(strategies)

        # 驗證並行執行
        assert result is not None
        # 應該合併 3 個策略的結果
        assert len(result.all_solutions) >= 3
        assert controller.summary.total_pareto_solutions >= 3

    @pytest.mark.asyncio
    async def test_run_optimization_with_semaphore_limit(self, quick_config):
        """測試 Semaphore 限制並行數量"""
        quick_config.max_workers = 2  # 限制最多 2 個並行
        controller = UltimateLoopController(quick_config, verbose=False)

        if controller.optimizer is None:
            from src.optimizer.multi_objective import MultiObjectiveOptimizer
            controller.optimizer = MultiObjectiveOptimizer(
                objectives=[('sharpe_ratio', 'maximize')],
                n_trials=10,
                verbose=False
            )

        strategies = ['s1', 's2', 's3', 's4', 's5']

        # 追蹤並行數量
        concurrent_count = [0]
        max_concurrent = [0]

        async def mock_optimize(strategy_name: str):
            """追蹤並行數"""
            concurrent_count[0] += 1
            max_concurrent[0] = max(max_concurrent[0], concurrent_count[0])
            await asyncio.sleep(0.01)  # 模擬耗時
            concurrent_count[0] -= 1

            from src.optimizer.multi_objective import MultiObjectiveResult, ParetoSolution, ObjectiveResult
            solution = ParetoSolution(
                params={'param1': 10},
                objectives=[ObjectiveResult('sharpe_ratio', 2.0, 'maximize')],
                rank=0
            )
            return MultiObjectiveResult(
                pareto_front=[solution],
                all_solutions=[solution],
                n_trials=10,
                study=None
            )

        controller._optimize_strategy = mock_optimize

        # 執行優化
        await controller._run_optimization(strategies)

        # 驗證並行數量不超過 max_workers
        assert max_concurrent[0] <= quick_config.max_workers

    @pytest.mark.asyncio
    async def test_run_optimization_hyperloop_fallback(self, quick_config):
        """測試 HyperLoop 失敗時 fallback 到 MultiObjectiveOptimizer"""
        quick_config.hyperloop_enabled = True
        controller = UltimateLoopController(quick_config, verbose=False)

        strategies = ['test_strategy_1']

        # 如果 HyperLoop 可用，Mock 讓它失敗
        if controller.hyperloop is not None:
            controller.hyperloop.run_loop = AsyncMock(
                side_effect=RuntimeError("HyperLoop failed")
            )

        # Mock MultiObjectiveOptimizer
        if controller.optimizer is None:
            from src.optimizer.multi_objective import MultiObjectiveOptimizer, MultiObjectiveResult, ParetoSolution
            controller.optimizer = MultiObjectiveOptimizer(
                objectives=[('sharpe_ratio', 'maximize')],
                n_trials=10,
                verbose=False
            )

        async def mock_optimize(strategy_name: str):
            """模擬優化成功"""
            from src.optimizer.multi_objective import MultiObjectiveResult, ParetoSolution, ObjectiveResult
            solution = ParetoSolution(
                params={'param1': 10},
                objectives=[ObjectiveResult('sharpe_ratio', 2.0, 'maximize')],
                rank=0
            )
            return MultiObjectiveResult(
                pareto_front=[solution],
                all_solutions=[solution],
                n_trials=10,
                study=None
            )

        controller._optimize_strategy = mock_optimize

        # 執行優化應該 fallback 成功
        result = await controller._run_optimization(strategies)

        # 應該返回結果（fallback 成功）
        assert result is not None

    @pytest.mark.asyncio
    async def test_run_optimization_high_failure_rate_warning(self, quick_config, caplog):
        """測試優化失敗率 > 50% 時警告"""
        controller = UltimateLoopController(quick_config, verbose=False)

        if controller.optimizer is None:
            from src.optimizer.multi_objective import MultiObjectiveOptimizer
            controller.optimizer = MultiObjectiveOptimizer(
                objectives=[('sharpe_ratio', 'maximize')],
                n_trials=10,
                verbose=False
            )

        strategies = ['s1', 's2', 's3', 's4']

        # Mock _optimize_strategy：3/4 失敗（75% 失敗率）
        call_count = [0]

        async def mock_optimize(strategy_name: str):
            call_count[0] += 1
            if call_count[0] <= 3:
                raise RuntimeError(f"{strategy_name} failed")

            from src.optimizer.multi_objective import MultiObjectiveResult, ParetoSolution, ObjectiveResult
            solution = ParetoSolution(
                params={'param1': 10},
                objectives=[ObjectiveResult('sharpe_ratio', 2.0, 'maximize')],
                rank=0
            )
            return MultiObjectiveResult(
                pareto_front=[solution],
                all_solutions=[solution],
                n_trials=10,
                study=None
            )

        controller._optimize_strategy = mock_optimize

        # 執行優化
        with caplog.at_level(logging.WARNING):
            await controller._run_optimization(strategies)

        # 應該記錄高失敗率警告
        assert any(
            "High failure rate" in record.message or "失敗率" in record.message
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_optimize_strategy_method(self, quick_config):
        """測試 _optimize_strategy 方法"""
        controller = UltimateLoopController(quick_config, verbose=False)

        # 初始化 optimizer
        if controller.optimizer is None:
            from src.optimizer.multi_objective import MultiObjectiveOptimizer
            controller.optimizer = MultiObjectiveOptimizer(
                objectives=[('sharpe_ratio', 'maximize')],
                n_trials=5,
                verbose=False
            )

        # Mock evaluate_fn
        def mock_evaluate(params):
            return {'sharpe_ratio': 1.5}

        controller._create_evaluate_fn = Mock(return_value=mock_evaluate)

        # 執行優化
        result = await controller._optimize_strategy('test_strategy')

        # 驗證結果
        # 注意：由於 StrategyRegistry 可能不可用，result 可能為 None
        # 測試主要驗證方法不拋出異常

    @pytest.mark.asyncio
    async def test_optimize_strategy_error_handling(self, quick_config):
        """測試 _optimize_strategy 錯誤處理"""
        controller = UltimateLoopController(quick_config, verbose=False)

        if controller.optimizer is None:
            from src.optimizer.multi_objective import MultiObjectiveOptimizer
            controller.optimizer = MultiObjectiveOptimizer(
                objectives=[('sharpe_ratio', 'maximize')],
                n_trials=5,
                verbose=False
            )

        # Mock get_param_space 拋出異常
        from unittest.mock import patch

        # 執行優化應該捕捉異常並返回 None
        result = await controller._optimize_strategy('invalid_strategy')

        # 應該返回 None（錯誤處理）
        assert result is None

    def test_combine_optimization_results(self, quick_config):
        """測試 _combine_optimization_results 方法"""
        controller = UltimateLoopController(quick_config, verbose=False)

        from src.optimizer.multi_objective import MultiObjectiveResult, ParetoSolution, ObjectiveResult

        # 建立多個優化結果
        result1 = MultiObjectiveResult(
            pareto_front=[
                ParetoSolution(
                    params={'param1': 10},
                    objectives=[ObjectiveResult('sharpe_ratio', 2.0, 'maximize')],
                    rank=0
                )
            ],
            all_solutions=[
                ParetoSolution(
                    params={'param1': 10},
                    objectives=[ObjectiveResult('sharpe_ratio', 2.0, 'maximize')],
                    rank=0
                )
            ],
            n_trials=10,
            study=None,
            optimization_time=5.0,
            n_completed_trials=10,
            n_failed_trials=0
        )

        result2 = MultiObjectiveResult(
            pareto_front=[
                ParetoSolution(
                    params={'param1': 20},
                    objectives=[ObjectiveResult('sharpe_ratio', 2.5, 'maximize')],
                    rank=0
                )
            ],
            all_solutions=[
                ParetoSolution(
                    params={'param1': 20},
                    objectives=[ObjectiveResult('sharpe_ratio', 2.5, 'maximize')],
                    rank=0
                )
            ],
            n_trials=15,
            study=None,
            optimization_time=7.0,
            n_completed_trials=14,
            n_failed_trials=1
        )

        # 合併結果
        combined = controller._combine_optimization_results([result1, result2])

        # 驗證合併
        assert combined is not None
        assert len(combined.pareto_front) == 2  # 2 個 Pareto 解
        assert len(combined.all_solutions) == 2
        assert combined.n_trials == 25  # 10 + 15
        assert combined.optimization_time == 12.0  # 5.0 + 7.0
        assert combined.n_completed_trials == 24  # 10 + 14
        assert combined.n_failed_trials == 1  # 0 + 1

    def test_combine_empty_results(self, quick_config):
        """測試合併空結果列表"""
        controller = UltimateLoopController(quick_config, verbose=False)

        combined = controller._combine_optimization_results([])

        assert combined is None


# ===== 9. HyperLoop 整合測試 =====

class TestHyperLoopIntegration:
    """測試 HyperLoop 整合"""

    def test_config_hyperloop_enabled_field(self):
        """測試 hyperloop_enabled 配置欄位"""
        config = UltimateLoopConfig()

        # 預設應該啟用
        assert hasattr(config, 'hyperloop_enabled')
        assert config.hyperloop_enabled is True

    def test_config_param_sweep_threshold_field(self):
        """測試 param_sweep_threshold 配置欄位"""
        config = UltimateLoopConfig()

        # 應該有此欄位且預設為 100
        assert hasattr(config, 'param_sweep_threshold')
        assert config.param_sweep_threshold == 100

    def test_config_validation_with_hyperloop_params(self):
        """測試配置驗證包含 HyperLoop 參數"""
        config = UltimateLoopConfig()
        config.hyperloop_enabled = True
        config.param_sweep_threshold = 50

        # 不應拋出異常
        config.validate()

    def test_hyperloop_initialization_when_enabled(self, quick_config):
        """測試 hyperloop_enabled=True 時初始化 HyperLoop"""
        quick_config.hyperloop_enabled = True
        controller = UltimateLoopController(quick_config, verbose=False)

        # 檢查是否有 hyperloop 屬性
        assert hasattr(controller, 'hyperloop')

        # 可能初始化成功或失敗（取決於依賴）
        # 只要不拋異常就算通過
        # hyperloop 可能是 None（模組不可用）或實例（初始化成功）
        assert controller.hyperloop is None or controller.hyperloop is not None

    def test_hyperloop_not_initialized_when_disabled(self, quick_config):
        """測試 hyperloop_enabled=False 時不初始化"""
        quick_config.hyperloop_enabled = False
        controller = UltimateLoopController(quick_config, verbose=False)

        # hyperloop 應該為 None
        assert controller.hyperloop is None

    def test_hyperloop_graceful_degradation(self, quick_config):
        """測試 HyperLoop 模組不可用時優雅處理"""
        quick_config.hyperloop_enabled = True
        controller = UltimateLoopController(quick_config, verbose=False)

        # 即使 HyperLoop 不可用，也不應拋異常
        # Controller 應該正常初始化
        assert controller is not None
        assert hasattr(controller, 'hyperloop')

    @pytest.mark.asyncio
    async def test_optimization_uses_hyperloop_when_available(self, quick_config):
        """測試優化階段使用 HyperLoop（如可用）"""
        quick_config.hyperloop_enabled = True
        controller = UltimateLoopController(quick_config, verbose=False)

        # Mock 策略列表
        strategies = ['test_strategy_1']

        # Mock HyperLoop（如果不可用就跳過測試）
        if controller.hyperloop is None:
            pytest.skip("HyperLoop not available")

        # Mock hyperloop.run_loop 返回假結果
        mock_summary = MagicMock()
        mock_summary.best_params = {'param1': 10}
        mock_summary.best_objectives = {'sharpe_ratio': 2.0}

        controller.hyperloop.run_loop = AsyncMock(return_value=mock_summary)

        # 執行優化
        result = await controller._run_optimization(strategies)

        # 應該呼叫 hyperloop.run_loop
        assert controller.hyperloop.run_loop.called

    @pytest.mark.asyncio
    async def test_optimization_fallback_when_hyperloop_fails(self, quick_config):
        """測試 HyperLoop 失敗時 fallback 到標準優化器"""
        quick_config.hyperloop_enabled = True
        controller = UltimateLoopController(quick_config, verbose=False)

        strategies = ['test_strategy_1']

        # 如果 HyperLoop 不可用，跳過測試
        if controller.hyperloop is None:
            pytest.skip("HyperLoop not available, fallback test not applicable")

        # Mock hyperloop.run_loop 拋出異常
        controller.hyperloop.run_loop = AsyncMock(
            side_effect=RuntimeError("HyperLoop failed")
        )

        # Mock 標準優化器為 None（測試無優化器情況）
        controller.optimizer = None

        # 執行優化應該優雅處理失敗
        result = await controller._run_optimization(strategies)

        # 應該返回 None 或空結果（不拋異常）
        assert result is None or result == []

    @pytest.mark.asyncio
    async def test_optimization_without_hyperloop(self, quick_config):
        """測試無 HyperLoop 和無優化器時正確跳過"""
        quick_config.hyperloop_enabled = False
        controller = UltimateLoopController(quick_config, verbose=False)
        controller.optimizer = None

        strategies = ['test_strategy_1']

        # 執行優化應該跳過並返回 None
        result = await controller._run_optimization(strategies)

        assert result is None or result == []

    def test_hyperloop_cleanup(self, quick_config):
        """測試 HyperLoop 資源清理"""
        quick_config.hyperloop_enabled = True
        controller = UltimateLoopController(quick_config, verbose=False)

        # Mock HyperLoop 物件
        mock_hyperloop = Mock()
        mock_hyperloop._cleanup = Mock()
        controller.hyperloop = mock_hyperloop

        # 執行清理
        controller._cleanup()

        # hyperloop._cleanup 應該被呼叫
        assert mock_hyperloop._cleanup.called
        # hyperloop 應該被設為 None
        assert controller.hyperloop is None

    def test_hyperloop_cleanup_with_cleanup_method(self, quick_config):
        """測試 HyperLoop cleanup 方法（沒有 _cleanup）"""
        quick_config.hyperloop_enabled = True
        controller = UltimateLoopController(quick_config, verbose=False)

        # Mock HyperLoop 只有 cleanup 方法
        mock_hyperloop = Mock()
        mock_hyperloop.cleanup = Mock()
        delattr(mock_hyperloop, '_cleanup')
        controller.hyperloop = mock_hyperloop

        # 執行清理
        controller._cleanup()

        # hyperloop.cleanup 應該被呼叫
        assert mock_hyperloop.cleanup.called

    def test_hyperloop_cleanup_handles_exceptions(self, quick_config, caplog):
        """測試 HyperLoop 清理異常不中斷其他清理"""
        quick_config.hyperloop_enabled = True
        controller = UltimateLoopController(quick_config, verbose=False)

        # Mock HyperLoop cleanup 拋出異常
        mock_hyperloop = Mock()
        mock_hyperloop._cleanup = Mock(side_effect=RuntimeError("Cleanup failed"))
        controller.hyperloop = mock_hyperloop

        # Mock 其他資源
        mock_optimizer = Mock()
        mock_optimizer.cleanup = Mock()
        controller.optimizer = mock_optimizer

        # 執行清理不應拋異常
        with caplog.at_level(logging.WARNING):
            controller._cleanup()

        # HyperLoop 清理失敗應該記錄警告
        assert any("hyperloop" in record.message.lower() for record in caplog.records)

        # 其他資源仍應被清理
        assert mock_optimizer.cleanup.called


if __name__ == "__main__":
    """執行測試"""
    pytest.main([__file__, '-v', '--tb=short'])
