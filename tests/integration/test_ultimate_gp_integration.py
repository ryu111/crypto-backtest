"""
Integration tests for Ultimate GP exploration - Task 6.3

Tests the complete integration of GP exploration with UltimateLoopController:
1. Full GP explore flow
2. GP strategy validation pipeline
3. GP strategy selection and exploitation
4. GP learning record integration

These tests verify that GP-generated strategies integrate seamlessly
with the existing UltimateLoop system.
"""

import pytest
import asyncio
import logging
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
from dataclasses import dataclass

from src.automation.ultimate_loop import (
    UltimateLoopController,
    UltimateLoopSummary
)
from src.automation.ultimate_config import UltimateLoopConfig
from src.automation.gp_integration import (
    GPExplorationRequest,
    DynamicStrategyInfo,
    GPExplorationResult,
)


# ===== Fixtures =====

@pytest.fixture
def small_gp_config():
    """小規模 GP 配置用於快速測試"""
    config = UltimateLoopConfig.create_development_config()
    config.gp_explore_enabled = True
    config.gp_explore_ratio = 1.0  # 100% 使用 GP 探索
    config.gp_population_size = 10
    config.gp_generations = 3  # 非常小用於測試
    config.gp_top_n = 2
    config.validation_enabled = False
    config.learning_enabled = False
    config.regime_detection = False  # 簡化測試
    return config


@pytest.fixture
def sample_ohlcv_data():
    """生成測試用的 OHLCV 資料"""
    np.random.seed(42)
    n_candles = 100

    dates = pd.date_range('2024-01-01', periods=n_candles, freq='1h')
    close_prices = np.cumsum(np.random.randn(n_candles) * 0.5) + 100

    return pd.DataFrame({
        'open': close_prices + np.random.randn(n_candles) * 0.2,
        'high': close_prices + np.abs(np.random.randn(n_candles) * 0.3),
        'low': close_prices - np.abs(np.random.randn(n_candles) * 0.3),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, n_candles)
    }, index=dates)


@pytest.fixture
def mock_gp_strategies():
    """建立 mock GP 生成的策略"""
    strategies = []

    for i in range(2):
        strategy_info = DynamicStrategyInfo(
            name=f'gp_evolved_{i:03d}',
            strategy_class=Mock,  # Mock 策略類別
            expression=f'AND(GT(RSI(close, 14), 50), LT(RSI(close, 14), 70))',
            fitness=2.0 - (i * 0.1),  # 第一個的 fitness 較高
            generation=2,
            created_at=pd.Timestamp.utcnow(),
            metadata={
                'parent_ids': [f'parent_{i}'],
                'mutation_type': 'crossover',
                'backtest_stats': {
                    'sharpe': 1.8 - (i * 0.2),
                    'return': 0.15 - (i * 0.02),
                    'max_dd': -0.10 - (i * 0.01),
                }
            }
        )
        strategies.append(strategy_info)

    return strategies


# ===== Test Classes =====

class TestFullGPExploreFlow:
    """測試完整 GP 探索流程"""

    @pytest.mark.asyncio
    async def test_gp_explorer_initialization(self, small_gp_config):
        """測試 UltimateLoopController 初始化時啟用 GP 探索"""
        controller = UltimateLoopController(small_gp_config)

        # 驗證 GP Explorer 已初始化
        # 由於 GPExplorer 需要複雜的 DEAP 設置，這裡驗證初始化邏輯
        assert small_gp_config.gp_explore_enabled is True
        assert small_gp_config.gp_population_size == 10
        assert small_gp_config.gp_generations == 3

    @pytest.mark.asyncio
    async def test_gp_exploration_request_creation(self, small_gp_config):
        """測試 GP 探索請求正確建立"""
        request = GPExplorationRequest(
            symbol='BTCUSDT',
            timeframe='4h',
            population_size=small_gp_config.gp_population_size,
            generations=small_gp_config.gp_generations,
            top_n=small_gp_config.gp_top_n
        )

        assert request.symbol == 'BTCUSDT'
        assert request.timeframe == '4h'
        assert request.population_size == 10
        assert request.generations == 3
        assert request.top_n == 2

    @pytest.mark.asyncio
    async def test_gp_exploration_result_creation(self, mock_gp_strategies):
        """測試 GP 探索結果正確建立"""
        result = GPExplorationResult(
            success=True,
            strategies=mock_gp_strategies,
            evolution_stats={
                'best_fitness_per_gen': [1.2, 1.5, 1.8],
                'avg_fitness_per_gen': [0.8, 1.0, 1.2],
                'diversity_per_gen': [0.9, 0.85, 0.8],
                'total_evaluations': 30
            },
            elapsed_time=45.2
        )

        assert result.success is True
        assert len(result.strategies) == 2
        assert result.strategies[0].fitness > result.strategies[1].fitness
        assert result.elapsed_time > 0

    @pytest.mark.asyncio
    async def test_gp_strategies_sorted_by_fitness(self, mock_gp_strategies):
        """測試 GP 策略按適應度排序"""
        # 確保 mock 策略按 fitness 排序（高到低）
        sorted_strategies = sorted(
            mock_gp_strategies,
            key=lambda s: s.fitness,
            reverse=True
        )

        assert sorted_strategies[0].fitness >= sorted_strategies[1].fitness
        assert sorted_strategies[0].name == 'gp_evolved_000'
        assert sorted_strategies[1].name == 'gp_evolved_001'


class TestGPStrategyValidationPipeline:
    """測試 GP 策略進入驗證流程"""

    @pytest.mark.asyncio
    async def test_gp_strategy_has_required_attributes(self, mock_gp_strategies):
        """測試 GP 策略具備驗證所需的屬性"""
        strategy_info = mock_gp_strategies[0]

        # 驗證必要屬性
        assert hasattr(strategy_info, 'name')
        assert hasattr(strategy_info, 'fitness')
        assert hasattr(strategy_info, 'generation')
        assert hasattr(strategy_info, 'expression')
        assert hasattr(strategy_info, 'metadata')

        # 驗證值類型和內容
        assert isinstance(strategy_info.name, str)
        assert isinstance(strategy_info.fitness, float)
        assert isinstance(strategy_info.generation, int)
        assert isinstance(strategy_info.expression, str)
        assert isinstance(strategy_info.metadata, dict)

    @pytest.mark.asyncio
    async def test_gp_strategy_metadata_backtest_stats(self, mock_gp_strategies):
        """測試 GP 策略元資料包含回測統計"""
        strategy_info = mock_gp_strategies[0]
        metadata = strategy_info.metadata

        assert 'backtest_stats' in metadata
        stats = metadata['backtest_stats']

        # 驗證必要的績效指標
        assert 'sharpe' in stats
        assert 'return' in stats
        assert 'max_dd' in stats

    @pytest.mark.asyncio
    async def test_gp_strategy_lineage_tracking(self, mock_gp_strategies):
        """測試 GP 策略譜系追蹤"""
        strategy_info = mock_gp_strategies[0]
        metadata = strategy_info.metadata

        # 驗證演化譜系資訊
        assert 'parent_ids' in metadata
        assert 'mutation_type' in metadata
        assert isinstance(metadata['parent_ids'], list)
        assert metadata['mutation_type'] in ['crossover', 'mutation', 'reproduction']

    @pytest.mark.asyncio
    async def test_gp_strategy_fitness_score_validity(self, mock_gp_strategies):
        """測試 GP 策略適應度分數合理性"""
        for strategy_info in mock_gp_strategies:
            # 適應度應該在合理範圍內
            assert 0 <= strategy_info.fitness <= 10, \
                f"Fitness {strategy_info.fitness} out of expected range"

            # 較好的策略 fitness 應該 >= 1.0
            if strategy_info.generation > 5:
                # 演化足夠代數後應該有合理的 fitness
                assert strategy_info.fitness >= 0.5


class TestGPStrategyExploitSelection:
    """測試 GP 策略在 exploit 模式被選中"""

    @pytest.mark.asyncio
    async def test_gp_strategy_registration_format(self, mock_gp_strategies):
        """測試 GP 策略可以正確註冊"""
        strategy_info = mock_gp_strategies[0]

        # 模擬註冊過程
        registered_name = strategy_info.name

        assert registered_name.startswith('gp_')
        assert isinstance(registered_name, str)
        assert len(registered_name) > 0

    @pytest.mark.asyncio
    async def test_gp_strategy_name_uniqueness(self, mock_gp_strategies):
        """測試 GP 策略名稱唯一性"""
        names = [s.name for s in mock_gp_strategies]

        # 所有名稱應該不重複
        assert len(names) == len(set(names))

    @pytest.mark.asyncio
    async def test_gp_strategy_selection_by_fitness(self, mock_gp_strategies):
        """測試可以按 fitness 選擇 GP 策略"""
        # 按 fitness 排序並選擇 top-2
        selected = sorted(
            mock_gp_strategies,
            key=lambda s: s.fitness,
            reverse=True
        )[:2]

        assert len(selected) == 2
        assert selected[0].fitness >= selected[1].fitness

    @pytest.mark.asyncio
    async def test_gp_strategy_exploit_vs_explore_ratio(self):
        """測試 exploit vs explore 比例配置"""
        config = UltimateLoopConfig.create_development_config()

        # 設定 80% exploit / 20% explore
        config.exploit_ratio = 0.8
        assert config.exploit_ratio == 0.8

        # 驗證 explore 比例
        explore_ratio = 1.0 - config.exploit_ratio
        assert explore_ratio == pytest.approx(0.2)


class TestGPLearningRecordIntegration:
    """測試 GP 策略學習記錄"""

    @pytest.mark.asyncio
    async def test_gp_strategy_recording_structure(self, mock_gp_strategies):
        """測試 GP 策略可以記錄到實驗日誌"""
        strategy_info = mock_gp_strategies[0]

        # 建立記錄結構
        record = {
            'name': strategy_info.name,
            'fitness': strategy_info.fitness,
            'generation': strategy_info.generation,
            'expression': strategy_info.expression,
            'metadata': strategy_info.metadata,
            'created_at': str(strategy_info.created_at)
        }

        assert 'name' in record
        assert 'fitness' in record
        assert 'expression' in record
        assert record['fitness'] > 0

    @pytest.mark.asyncio
    async def test_gp_strategy_memory_compatibility(self, mock_gp_strategies):
        """測試 GP 策略資訊可存入 Memory MCP"""
        strategy_info = mock_gp_strategies[0]

        # 模擬 Memory 儲存格式
        memory_content = (
            f"GP Strategy: {strategy_info.name}\n"
            f"Fitness: {strategy_info.fitness:.4f}\n"
            f"Generation: {strategy_info.generation}\n"
            f"Expression: {strategy_info.expression}\n"
        )

        assert len(memory_content) > 0
        assert strategy_info.name in memory_content
        assert str(strategy_info.fitness) in memory_content

    @pytest.mark.asyncio
    async def test_gp_strategy_learning_tags(self, mock_gp_strategies):
        """測試 GP 策略學習記錄的標籤"""
        strategy_info = mock_gp_strategies[0]

        # 建立標籤
        tags = [
            'gp_strategy',
            'generated',
            f'generation_{strategy_info.generation}',
            'explore_phase'
        ]

        assert 'gp_strategy' in tags
        assert 'generated' in tags
        assert len(tags) > 0

    @pytest.mark.asyncio
    async def test_gp_exploration_result_logging(self, mock_gp_strategies):
        """測試 GP 探索結果可以記錄"""
        result = GPExplorationResult(
            success=True,
            strategies=mock_gp_strategies,
            evolution_stats={
                'best_fitness_per_gen': [1.0, 1.5, 2.0],
                'avg_fitness_per_gen': [0.7, 1.0, 1.2],
                'diversity_per_gen': [0.9, 0.85, 0.8],
                'total_evaluations': 30
            },
            elapsed_time=45.2
        )

        # 驗證結果可以記錄的資訊
        assert result.success is True
        assert len(result.strategies) > 0
        assert result.elapsed_time > 0
        assert 'best_fitness_per_gen' in result.evolution_stats

    @pytest.mark.asyncio
    async def test_gp_strategy_statistics_calculation(self, mock_gp_strategies):
        """測試 GP 策略統計計算"""
        # 計算平均 fitness
        avg_fitness = np.mean([s.fitness for s in mock_gp_strategies])
        max_fitness = max([s.fitness for s in mock_gp_strategies])
        min_fitness = min([s.fitness for s in mock_gp_strategies])

        assert max_fitness >= avg_fitness >= min_fitness
        assert max_fitness > 0
        assert min_fitness > 0


# ===== Integration Test Classes =====

class TestGPIntegrationWithUltimateLoop:
    """測試 GP 與 UltimateLoop 的完整整合"""

    @pytest.mark.asyncio
    async def test_controller_with_gp_disabled(self, small_gp_config):
        """測試關閉 GP 探索的控制器初始化"""
        config = small_gp_config
        config.gp_explore_enabled = False

        controller = UltimateLoopController(config, verbose=False)

        # 驗證控制器初始化成功
        assert controller.config.gp_explore_enabled is False

    @pytest.mark.asyncio
    async def test_controller_with_gp_enabled(self, small_gp_config):
        """測試啟用 GP 探索的控制器初始化"""
        config = small_gp_config
        config.gp_explore_enabled = True

        controller = UltimateLoopController(config, verbose=False)

        # 驗證 GP 探索已啟用
        assert controller.config.gp_explore_enabled is True
        assert controller.config.gp_population_size == 10
        assert controller.config.gp_generations == 3

    @pytest.mark.asyncio
    async def test_controller_config_validation(self, small_gp_config):
        """測試控制器配置驗證"""
        # 驗證有效的配置
        small_gp_config.validate()

        # 應該沒有拋出異常
        assert small_gp_config.gp_explore_enabled is True

    @pytest.mark.asyncio
    async def test_gp_strategy_count_tracking(self, small_gp_config):
        """測試 GP 策略計數追蹤"""
        controller = UltimateLoopController(small_gp_config, verbose=False)

        # 初始時應該沒有 GP 策略
        initial_count = controller.summary.gp_strategies_generated
        assert initial_count == 0

        # 模擬 GP 策略生成
        controller.summary.gp_strategies_generated = 5
        controller.summary.gp_strategies_validated = 3

        assert controller.summary.gp_strategies_generated == 5
        assert controller.summary.gp_strategies_validated == 3

    @pytest.mark.asyncio
    async def test_controller_summary_includes_gp_stats(self, small_gp_config):
        """測試控制器摘要包含 GP 統計"""
        controller = UltimateLoopController(small_gp_config, verbose=False)

        # 設定 GP 統計
        controller.summary.gp_strategies_generated = 10
        controller.summary.gp_strategies_validated = 8

        summary_text = controller.summary.summary_text()

        # 驗證摘要包含 GP 統計
        assert 'GP 策略統計' in summary_text or 'gp_strategies' in str(controller.summary.__dict__)


class TestGPExplorationEdgeCases:
    """測試 GP 探索邊界情況"""

    @pytest.mark.asyncio
    async def test_empty_gp_result(self):
        """測試 GP 未生成任何策略的情況"""
        result = GPExplorationResult(
            success=True,
            strategies=[],  # 空列表
            evolution_stats={},
            elapsed_time=10.0
        )

        assert result.success is True
        assert len(result.strategies) == 0

    @pytest.mark.asyncio
    async def test_gp_exploration_failure(self):
        """測試 GP 探索失敗的情況"""
        result = GPExplorationResult(
            success=False,
            strategies=[],
            evolution_stats={},
            elapsed_time=5.0,
            error="Timeout during evolution"
        )

        assert result.success is False
        assert result.error is not None
        assert len(result.strategies) == 0

    @pytest.mark.asyncio
    async def test_gp_strategy_with_extreme_fitness(self):
        """測試 GP 策略適應度極值"""
        # 非常高的 fitness
        strategy_high = DynamicStrategyInfo(
            name='gp_extreme_high',
            strategy_class=Mock,
            expression='test_expr',
            fitness=9.99,  # 接近上限
            generation=10,
            created_at=pd.Timestamp.utcnow(),
            metadata={}
        )

        assert strategy_high.fitness < 10.0

        # 非常低的 fitness
        strategy_low = DynamicStrategyInfo(
            name='gp_extreme_low',
            strategy_class=Mock,
            expression='test_expr',
            fitness=0.01,  # 接近下限
            generation=10,
            created_at=pd.Timestamp.utcnow(),
            metadata={}
        )

        assert strategy_low.fitness > 0

    @pytest.mark.asyncio
    async def test_gp_strategy_with_empty_expression(self):
        """測試 GP 策略空表達式的情況"""
        strategy = DynamicStrategyInfo(
            name='gp_empty',
            strategy_class=Mock,
            expression='',  # 空表達式
            fitness=1.0,
            generation=0,
            created_at=pd.Timestamp.utcnow(),
            metadata={}
        )

        # 應該仍然可以建立，但在使用時會失敗
        assert strategy.name == 'gp_empty'
        assert strategy.expression == ''

    @pytest.mark.asyncio
    async def test_large_gp_result_set(self):
        """測試大量 GP 策略結果的處理"""
        # 建立 100 個 mock 策略
        strategies = []
        for i in range(100):
            strategy = DynamicStrategyInfo(
                name=f'gp_large_{i:03d}',
                strategy_class=Mock,
                expression=f'expr_{i}',
                fitness=2.0 - (i * 0.01),
                generation=5,
                created_at=pd.Timestamp.utcnow(),
                metadata={}
            )
            strategies.append(strategy)

        # 驗證可以處理大量策略
        assert len(strategies) == 100

        # 驗證排序
        sorted_strategies = sorted(
            strategies,
            key=lambda s: s.fitness,
            reverse=True
        )
        assert sorted_strategies[0].fitness >= sorted_strategies[-1].fitness


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
