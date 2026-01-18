"""
測試 UltimateLoop 的 GP 策略整合 (Phase 13)

驗證：
1. UltimateLoopSummary 包含 GP 統計欄位
2. _is_gp_strategy 正確識別 GP 策略
3. _get_gp_metadata_batch 正確提取元資料
4. summary_text() 顯示 GP 統計
"""

import pytest
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import Dict, Any, List

from src.automation.ultimate_loop import (
    UltimateLoopSummary,
    UltimateLoopController
)
from src.automation.ultimate_config import UltimateLoopConfig


class TestUltimateLoopSummaryGP:
    """測試 UltimateLoopSummary 的 GP 欄位"""

    def test_summary_has_gp_fields(self):
        """驗證 GP 統計欄位存在"""
        summary = UltimateLoopSummary()

        assert hasattr(summary, 'gp_strategies_generated')
        assert hasattr(summary, 'gp_strategies_validated')

    def test_default_values(self):
        """驗證預設值"""
        summary = UltimateLoopSummary()

        assert summary.gp_strategies_generated == 0
        assert summary.gp_strategies_validated == 0

    def test_summary_text_with_gp_stats(self):
        """驗證 summary_text 包含 GP 統計"""
        summary = UltimateLoopSummary()
        summary.gp_strategies_generated = 15
        summary.gp_strategies_validated = 10

        text = summary.summary_text()

        assert "GP 策略統計:" in text
        assert "生成數量: 15" in text
        assert "通過驗證: 10" in text
        assert "驗證率: 66.7%" in text

    def test_summary_text_without_gp_stats(self):
        """驗證沒有 GP 統計時不顯示該區塊"""
        summary = UltimateLoopSummary()
        summary.gp_strategies_generated = 0
        summary.gp_strategies_validated = 0

        text = summary.summary_text()

        assert "GP 策略統計:" not in text


class TestGPStrategyIdentification:
    """測試 GP 策略識別"""

    def setup_method(self):
        """建立測試用 controller"""
        config = UltimateLoopConfig(
            symbols=['BTCUSDT'],
            timeframes=['4h'],
            regime_detection=False,
            validation_enabled=False,
            learning_enabled=False  # 禁用學習系統避免數據庫鎖
        )
        self.controller = UltimateLoopController(config, verbose=False)

    def test_identify_gp_by_prefix(self):
        """測試通過名稱前綴識別 GP 策略"""
        # GP 策略（大寫 GP_）
        assert self.controller._is_gp_strategy('GP_evolved_001') is True
        assert self.controller._is_gp_strategy('GP_gen_05_rank_1') is True

        # GP 策略（小寫 gp_）
        assert self.controller._is_gp_strategy('gp_evolved_002') is True

    def test_identify_non_gp_strategies(self):
        """測試非 GP 策略不被識別"""
        assert self.controller._is_gp_strategy('ma_cross') is False
        assert self.controller._is_gp_strategy('rsi_strategy') is False
        assert self.controller._is_gp_strategy('bollinger_bands') is False


class TestGPMetadataExtraction:
    """測試 GP 元資料提取"""

    def setup_method(self):
        """建立測試用 controller"""
        config = UltimateLoopConfig(
            symbols=['BTCUSDT'],
            timeframes=['4h'],
            regime_detection=False,
            validation_enabled=False,
            learning_enabled=False  # 禁用學習系統避免數據庫鎖
        )
        self.controller = UltimateLoopController(config, verbose=False)

    def test_empty_strategy_list(self):
        """測試空策略列表"""
        result = self.controller._get_gp_metadata_batch([])
        assert result == {}

    def test_metadata_structure(self):
        """測試元資料結構正確"""
        # 沒有實際註冊的策略，應該返回空 dict
        result = self.controller._get_gp_metadata_batch(['GP_test'])

        # 應該返回空 dict（因為沒有註冊策略）
        assert isinstance(result, dict)


class TestGPLearningIntegration:
    """測試 GP 策略學習記錄整合"""

    def setup_method(self):
        """建立測試用 controller"""
        config = UltimateLoopConfig(
            symbols=['BTCUSDT'],
            timeframes=['4h'],
            regime_detection=False,
            validation_enabled=False,
            learning_enabled=False  # 禁用學習系統避免數據庫鎖
        )
        self.controller = UltimateLoopController(config, verbose=False)

    @pytest.mark.asyncio
    async def test_record_and_learn_with_gp_strategies(self):
        """測試記錄 GP 策略"""
        # Mock recorder
        self.controller.recorder = Mock()
        self.controller.recorder.log_experiment = Mock(return_value='exp_001')

        # 建立 mock solutions
        @dataclass
        class MockObjective:
            name: str
            value: float

        @dataclass
        class MockSolution:
            objectives: List[MockObjective]
            params: Dict[str, Any]
            _validation_grade: str = 'A'

        solutions = [
            MockSolution(
                objectives=[
                    MockObjective('sharpe_ratio', 2.5),
                    MockObjective('total_return', 0.35),
                    MockObjective('max_drawdown', -0.12)
                ],
                params={'fast': 10, 'slow': 30},
                _validation_grade='A'
            )
        ]

        # GP 策略名稱
        strategy_names = ['GP_evolved_001']

        # 執行記錄
        await self.controller._record_and_learn(
            solutions=solutions,
            market_state=None,
            strategy_names=strategy_names
        )

        # 驗證統計更新
        assert self.controller.summary.gp_strategies_generated == 1
        assert self.controller.summary.gp_strategies_validated == 1
        assert self.controller.summary.experiments_recorded == 1

        # 驗證 recorder 被調用
        assert self.controller.recorder.log_experiment.called

        # 檢查傳入的 strategy_info
        call_args = self.controller.recorder.log_experiment.call_args
        strategy_info = call_args.kwargs['strategy_info']

        # 應該包含 source = 'gp_explorer'
        assert strategy_info.get('source') == 'gp_explorer'

    @pytest.mark.asyncio
    async def test_record_and_learn_with_mixed_strategies(self):
        """測試混合 GP 和一般策略"""
        # Mock recorder
        self.controller.recorder = Mock()
        self.controller.recorder.log_experiment = Mock(return_value='exp_002')

        @dataclass
        class MockObjective:
            name: str
            value: float

        @dataclass
        class MockSolution:
            objectives: List[MockObjective]
            params: Dict[str, Any]
            _validation_grade: str = 'A'

        solutions = [
            MockSolution(
                objectives=[
                    MockObjective('sharpe_ratio', 1.8),
                    MockObjective('total_return', 0.25)
                ],
                params={'period': 14},
                _validation_grade='B'
            )
        ]

        # 混合策略：1 個 GP + 1 個一般策略
        strategy_names = ['GP_evolved_002', 'ma_cross']

        await self.controller._record_and_learn(
            solutions=solutions,
            market_state=None,
            strategy_names=strategy_names
        )

        # 應該只計算 GP 策略
        assert self.controller.summary.gp_strategies_generated == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
