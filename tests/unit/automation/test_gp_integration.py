"""
Unit tests for gp_integration.py

Testing data contracts and core components:
- GPExplorationRequest
- DynamicStrategyInfo
- GPExplorationResult
"""

import pytest
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any

from src.automation.gp_integration import (
    GPExplorationRequest,
    DynamicStrategyInfo,
    GPExplorationResult,
    GPStrategyAdapter,
)


# Mock strategy class for testing
@dataclass
class MockStrategy:
    """Mock strategy class for testing"""
    name: str = "mock_strategy"


class TestGPExplorationRequestImport:
    """Test: 確認模組可正常導入"""

    def test_module_imports_successfully(self):
        """應該能成功導入模組"""
        # 如果執行到這裡，表示導入成功
        assert GPExplorationRequest is not None
        assert DynamicStrategyInfo is not None
        assert GPExplorationResult is not None


class TestGPExplorationRequest:
    """Test: GPExplorationRequest 資料契約"""

    def test_default_values(self):
        """應該有正確的預設值"""
        request = GPExplorationRequest(
            symbol='BTCUSDT',
            timeframe='4h'
        )

        assert request.symbol == 'BTCUSDT'
        assert request.timeframe == '4h'
        assert request.population_size == 50
        assert request.generations == 30
        assert request.top_n == 3
        assert request.fitness_weights == (1.0, 0.5, -0.3)

    def test_custom_values(self):
        """應該能使用自訂值"""
        request = GPExplorationRequest(
            symbol='ETHUSDT',
            timeframe='1d',
            population_size=100,
            generations=50,
            top_n=5,
            fitness_weights=(1.5, 0.8, -0.5)
        )

        assert request.symbol == 'ETHUSDT'
        assert request.timeframe == '1d'
        assert request.population_size == 100
        assert request.generations == 50
        assert request.top_n == 5
        assert request.fitness_weights == (1.5, 0.8, -0.5)

    def test_minimal_initialization(self):
        """應該支援最小初始化（只指定必需欄位）"""
        request = GPExplorationRequest(symbol='BTCUSDT')

        assert request.symbol == 'BTCUSDT'
        assert request.timeframe == '4h'  # 預設值

    def test_fitness_weights_format(self):
        """健身度權重應該是三元組"""
        request = GPExplorationRequest(
            symbol='BTCUSDT',
            fitness_weights=(2.0, 1.0, -1.0)
        )

        assert len(request.fitness_weights) == 3
        assert isinstance(request.fitness_weights, tuple)
        assert request.fitness_weights[0] == 2.0  # sharpe weight
        assert request.fitness_weights[1] == 1.0  # return weight
        assert request.fitness_weights[2] == -1.0  # drawdown weight

    def test_population_and_generations_positive(self):
        """種群大小和代數應該是正整數"""
        request = GPExplorationRequest(
            symbol='BTCUSDT',
            population_size=100,
            generations=50
        )

        assert request.population_size > 0
        assert request.generations > 0
        assert isinstance(request.population_size, int)
        assert isinstance(request.generations, int)


class TestDynamicStrategyInfo:
    """Test: DynamicStrategyInfo 資料契約"""

    def test_basic_initialization(self):
        """應該能初始化基本欄位"""
        now = datetime.utcnow()
        info = DynamicStrategyInfo(
            name='gp_evolved_001',
            strategy_class=MockStrategy,
            expression='and(gt(rsi(14), 50), lt(rsi(14), 70))',
            fitness=2.35,
            generation=10,
            created_at=now
        )

        assert info.name == 'gp_evolved_001'
        assert info.strategy_class == MockStrategy
        assert info.expression == 'and(gt(rsi(14), 50), lt(rsi(14), 70))'
        assert info.fitness == 2.35
        assert info.generation == 10
        assert info.created_at == now

    def test_metadata_default_empty_dict(self):
        """metadata 應該預設為空 dict"""
        now = datetime.utcnow()
        info = DynamicStrategyInfo(
            name='gp_evolved_001',
            strategy_class=MockStrategy,
            expression='test_expr',
            fitness=1.0,
            generation=0,
            created_at=now
        )

        assert info.metadata == {}
        assert isinstance(info.metadata, dict)

    def test_metadata_custom_values(self):
        """應該能設定自訂 metadata"""
        now = datetime.utcnow()
        metadata = {
            'parent_ids': ['gp_gen_09_005', 'gp_gen_09_012'],
            'mutation_type': 'crossover',
            'backtest_stats': {
                'sharpe': 2.35,
                'return': 125.3,
                'max_drawdown': 15.2
            }
        }

        info = DynamicStrategyInfo(
            name='gp_evolved_001',
            strategy_class=MockStrategy,
            expression='test_expr',
            fitness=2.35,
            generation=10,
            created_at=now,
            metadata=metadata
        )

        assert info.metadata == metadata
        assert info.metadata['mutation_type'] == 'crossover'
        assert info.metadata['backtest_stats']['sharpe'] == 2.35

    def test_generation_zero_based(self):
        """generation 應該是 0-based"""
        now = datetime.utcnow()
        for gen in [0, 5, 10, 49]:
            info = DynamicStrategyInfo(
                name=f'gp_gen_{gen}',
                strategy_class=MockStrategy,
                expression='test',
                fitness=1.0,
                generation=gen,
                created_at=now
            )
            assert info.generation == gen
            assert isinstance(info.generation, int)

    def test_fitness_score_numeric(self):
        """fitness 應該是數值型別"""
        now = datetime.utcnow()
        for fitness in [0.0, 1.5, 2.35, 10.0]:
            info = DynamicStrategyInfo(
                name='test',
                strategy_class=MockStrategy,
                expression='test',
                fitness=fitness,
                generation=0,
                created_at=now
            )
            assert info.fitness == fitness
            assert isinstance(info.fitness, (int, float))

    def test_created_at_is_datetime(self):
        """created_at 應該是 datetime 物件"""
        now = datetime.utcnow()
        info = DynamicStrategyInfo(
            name='test',
            strategy_class=MockStrategy,
            expression='test',
            fitness=1.0,
            generation=0,
            created_at=now
        )

        assert isinstance(info.created_at, datetime)
        assert info.created_at == now


class TestGPExplorationResult:
    """Test: GPExplorationResult 資料契約"""

    def test_success_true_scenario(self):
        """success=True 時的情景"""
        now = datetime.utcnow()
        strategy1 = DynamicStrategyInfo(
            name='strat1',
            strategy_class=MockStrategy,
            expression='expr1',
            fitness=2.5,
            generation=10,
            created_at=now
        )
        strategy2 = DynamicStrategyInfo(
            name='strat2',
            strategy_class=MockStrategy,
            expression='expr2',
            fitness=2.3,
            generation=10,
            created_at=now
        )

        result = GPExplorationResult(
            success=True,
            strategies=[strategy1, strategy2],
            evolution_stats={
                'best_fitness_per_gen': [1.2, 1.5, 1.8, 2.1, 2.3, 2.5],
                'avg_fitness_per_gen': [0.8, 1.0, 1.2, 1.4, 1.5, 1.6],
                'diversity_per_gen': [0.95, 0.90, 0.85, 0.80, 0.75, 0.70],
                'total_evaluations': 250
            },
            elapsed_time=125.5,
            error=None
        )

        assert result.success is True
        assert len(result.strategies) == 2
        assert result.strategies[0].fitness == 2.5
        assert result.elapsed_time == 125.5
        assert result.error is None

    def test_success_false_scenario(self):
        """success=False 時的情景"""
        result = GPExplorationResult(
            success=False,
            strategies=[],
            evolution_stats={},
            elapsed_time=0.0,
            error="Insufficient data for backtest"
        )

        assert result.success is False
        assert result.strategies == []
        assert result.error == "Insufficient data for backtest"

    def test_strategies_empty_list(self):
        """strategies 應該能是空列表"""
        result = GPExplorationResult(
            success=False,
            strategies=[],
            evolution_stats={},
            elapsed_time=0.0,
            error="Failed"
        )

        assert isinstance(result.strategies, list)
        assert len(result.strategies) == 0

    def test_strategies_ordering(self):
        """strategies 應該按適應度排序（高到低）"""
        now = datetime.utcnow()
        strategy1 = DynamicStrategyInfo(
            name='low',
            strategy_class=MockStrategy,
            expression='expr1',
            fitness=1.0,
            generation=0,
            created_at=now
        )
        strategy2 = DynamicStrategyInfo(
            name='high',
            strategy_class=MockStrategy,
            expression='expr2',
            fitness=3.0,
            generation=0,
            created_at=now
        )
        strategy3 = DynamicStrategyInfo(
            name='mid',
            strategy_class=MockStrategy,
            expression='expr3',
            fitness=2.0,
            generation=0,
            created_at=now
        )

        result = GPExplorationResult(
            success=True,
            strategies=[strategy2, strategy3, strategy1],
            evolution_stats={},
            elapsed_time=1.0
        )

        assert result.strategies[0].fitness == 3.0
        assert result.strategies[1].fitness == 2.0
        assert result.strategies[2].fitness == 1.0

    def test_evolution_stats_structure(self):
        """evolution_stats 應該包含正確的鍵"""
        stats = {
            'best_fitness_per_gen': [1.0, 1.5, 2.0],
            'avg_fitness_per_gen': [0.5, 0.8, 1.0],
            'diversity_per_gen': [0.9, 0.85, 0.8],
            'total_evaluations': 150
        }

        result = GPExplorationResult(
            success=True,
            strategies=[],
            evolution_stats=stats,
            elapsed_time=10.0
        )

        assert 'best_fitness_per_gen' in result.evolution_stats
        assert 'avg_fitness_per_gen' in result.evolution_stats
        assert 'diversity_per_gen' in result.evolution_stats
        assert 'total_evaluations' in result.evolution_stats

    def test_elapsed_time_numeric(self):
        """elapsed_time 應該是數值"""
        result = GPExplorationResult(
            success=True,
            strategies=[],
            evolution_stats={},
            elapsed_time=125.5
        )

        assert isinstance(result.elapsed_time, (int, float))
        assert result.elapsed_time > 0

    def test_error_field_optional(self):
        """error 欄位應該是可選的"""
        result = GPExplorationResult(
            success=True,
            strategies=[],
            evolution_stats={},
            elapsed_time=1.0
        )

        assert result.error is None


class TestDataContractIntegration:
    """Test: 資料契約整合場景"""

    def test_request_to_result_workflow(self):
        """完整的 request → result 流程"""
        request = GPExplorationRequest(
            symbol='BTCUSDT',
            timeframe='4h',
            population_size=100,
            generations=50
        )

        assert request.symbol == 'BTCUSDT'

        now = datetime.utcnow()
        strategies = [
            DynamicStrategyInfo(
                name=f'gp_strategy_{i}',
                strategy_class=MockStrategy,
                expression=f'expr_{i}',
                fitness=2.0 - i * 0.1,
                generation=49,
                created_at=now,
                metadata={'parent_ids': [f'parent_{i}']}
            )
            for i in range(3)
        ]

        result = GPExplorationResult(
            success=True,
            strategies=strategies,
            evolution_stats={
                'best_fitness_per_gen': list(range(1, 51)),
                'avg_fitness_per_gen': list(range(1, 51)),
                'diversity_per_gen': [0.95 - i * 0.01 for i in range(50)],
                'total_evaluations': 100 * 50
            },
            elapsed_time=300.0
        )

        assert result.success is True
        assert len(result.strategies) == 3
        assert result.strategies[0].fitness == 2.0
        assert all(s.generation == 49 for s in result.strategies)

    def test_multiple_symbols_compatibility(self):
        """應該支援多個交易標的"""
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

        for symbol in symbols:
            request = GPExplorationRequest(symbol=symbol)
            assert request.symbol == symbol

    def test_multiple_timeframes_compatibility(self):
        """應該支援多個時間框架"""
        timeframes = ['1h', '4h', '1d', '1w']

        for tf in timeframes:
            request = GPExplorationRequest(
                symbol='BTCUSDT',
                timeframe=tf
            )
            assert request.timeframe == tf


class TestDataContractEdgeCases:
    """Test: 邊界情況"""

    def test_very_large_population_size(self):
        """應該支援很大的種群大小"""
        request = GPExplorationRequest(
            symbol='BTCUSDT',
            population_size=10000,
            generations=100
        )
        assert request.population_size == 10000
        assert request.generations == 100

    def test_minimum_viable_configuration(self):
        """應該能用最小配置"""
        request = GPExplorationRequest(
            symbol='BTC',
            timeframe='1h',
            population_size=10,
            generations=1,
            top_n=1
        )
        assert request.population_size == 10
        assert request.generations == 1
        assert request.top_n == 1

    def test_zero_fitness_score(self):
        """應該支援 fitness 為零"""
        now = datetime.utcnow()
        info = DynamicStrategyInfo(
            name='zero_fitness',
            strategy_class=MockStrategy,
            expression='test',
            fitness=0.0,
            generation=0,
            created_at=now
        )
        assert info.fitness == 0.0

    def test_negative_fitness_score(self):
        """應該支援負 fitness（表示虧損）"""
        now = datetime.utcnow()
        info = DynamicStrategyInfo(
            name='negative_fitness',
            strategy_class=MockStrategy,
            expression='bad_expr',
            fitness=-0.5,
            generation=0,
            created_at=now
        )
        assert info.fitness == -0.5

    def test_very_long_expression(self):
        """應該支援長表達式"""
        now = datetime.utcnow()
        long_expr = '(' * 100 + 'test' + ')' * 100
        info = DynamicStrategyInfo(
            name='long_expr',
            strategy_class=MockStrategy,
            expression=long_expr,
            fitness=1.0,
            generation=0,
            created_at=now
        )
        assert len(info.expression) > 100

    def test_very_old_generation(self):
        """應該支援很舊的代數"""
        now = datetime.utcnow()
        info = DynamicStrategyInfo(
            name='old_gen',
            strategy_class=MockStrategy,
            expression='test',
            fitness=1.0,
            generation=9999,
            created_at=now
        )
        assert info.generation == 9999

    def test_empty_strategies_list_success(self):
        """即使 strategies 為空，success=True 也應該有效"""
        result = GPExplorationResult(
            success=True,
            strategies=[],
            evolution_stats={},
            elapsed_time=0.0
        )
        assert result.success is True
        assert len(result.strategies) == 0


class TestGPStrategyAdapter:
    """Test: GPStrategyAdapter 類別"""

    @pytest.fixture
    def mock_converter(self):
        """建立 mock converter"""
        class MockConverter:
            def compile(self, individual):
                """Mock compile method"""
                # 返回一個簡單的訊號函數
                def signal_func(close, high, low):
                    # 簡單策略：收盤價 > 均值
                    import numpy as np
                    mean_price = np.mean(close)
                    return close > mean_price
                return signal_func

            def to_python(self, individual):
                """Mock to_python method"""
                return "gt(close, ma(close, 20))"

        return MockConverter()

    @pytest.fixture
    def mock_individual(self):
        """建立 mock GP individual"""
        class MockIndividual:
            """Mock DEAP individual"""
            def __str__(self):
                return "GT(close, MA(close, 20))"

        return MockIndividual()

    def test_adapter_initialization(self, mock_converter):
        """應該能正確初始化"""
        adapter = GPStrategyAdapter(mock_converter)
        assert adapter.converter is not None

    def test_create_strategy_class(self, mock_converter, mock_individual):
        """應該能建立策略類別"""
        adapter = GPStrategyAdapter(mock_converter)

        strategy_class = adapter.create_strategy_class(
            individual=mock_individual,
            strategy_name='gp_test_001',
            fitness=1.85,
            generation=5
        )

        # 驗證返回的是類別
        assert isinstance(strategy_class, type)
        assert strategy_class.name == 'gp_test_001'
        assert strategy_class.fitness_score == 1.85
        assert strategy_class.generation == 5

    def test_strategy_class_attributes(self, mock_converter, mock_individual):
        """策略類別應該有正確的屬性"""
        adapter = GPStrategyAdapter(mock_converter)

        strategy_class = adapter.create_strategy_class(
            individual=mock_individual,
            strategy_name='gp_rsi_001',
            fitness=2.5,
            generation=10
        )

        assert strategy_class.name == 'gp_rsi_001'
        assert strategy_class.version == '1.0'
        assert 'fitness 2.5' in strategy_class.description.lower()
        assert strategy_class.expression == "gt(close, ma(close, 20))"
        assert strategy_class.fitness_score == 2.5
        assert strategy_class.generation == 10
        assert strategy_class.evolved_at.endswith('Z')  # ISO 格式

    def test_strategy_can_be_instantiated(self, mock_converter, mock_individual):
        """生成的策略類別應該可以實例化"""
        adapter = GPStrategyAdapter(mock_converter)

        strategy_class = adapter.create_strategy_class(
            individual=mock_individual,
            strategy_name='gp_test_002',
            fitness=1.5,
            generation=3
        )

        # 實例化策略
        strategy = strategy_class()

        # 驗證實例
        assert strategy.name == 'gp_test_002'
        assert strategy.fitness_score == 1.5
        assert strategy.generation == 3

    def test_strategy_has_signal_func(self, mock_converter, mock_individual):
        """策略實例應該有訊號函數"""
        adapter = GPStrategyAdapter(mock_converter)

        strategy_class = adapter.create_strategy_class(
            individual=mock_individual,
            strategy_name='gp_test_003',
            fitness=1.0,
            generation=0
        )

        strategy = strategy_class()

        # 驗證訊號函數存在且可呼叫
        assert hasattr(strategy, '_signal_func')
        assert callable(strategy._signal_func)

    def test_to_class_name_conversion(self, mock_converter):
        """應該正確轉換策略名稱為類別名稱"""
        adapter = GPStrategyAdapter(mock_converter)

        # 測試各種命名格式
        assert adapter._to_class_name('gp_evolved_001') == 'GpEvolved001'
        assert adapter._to_class_name('rsi_ma_cross') == 'RsiMaCross'
        assert adapter._to_class_name('simple_strategy') == 'SimpleStrategy'
        assert adapter._to_class_name('test') == 'Test'

    def test_compile_error_handling(self, mock_individual):
        """應該處理編譯錯誤"""
        class FailingConverter:
            def compile(self, individual):
                raise ValueError("Compilation failed")

            def to_python(self, individual):
                return "test_expr"

        adapter = GPStrategyAdapter(FailingConverter())

        with pytest.raises(RuntimeError) as exc_info:
            adapter.create_strategy_class(
                individual=mock_individual,
                strategy_name='test',
                fitness=1.0,
                generation=0
            )

        assert "Failed to compile GP expression" in str(exc_info.value)

    def test_to_python_error_handling(self, mock_individual):
        """應該處理轉換錯誤"""
        class FailingConverter:
            def compile(self, individual):
                return lambda c, h, l: c > 0

            def to_python(self, individual):
                raise ValueError("Conversion failed")

        adapter = GPStrategyAdapter(FailingConverter())

        with pytest.raises(RuntimeError) as exc_info:
            adapter.create_strategy_class(
                individual=mock_individual,
                strategy_name='test',
                fitness=1.0,
                generation=0
            )

        assert "Failed to convert GP expression to Python" in str(exc_info.value)

    def test_metadata_is_preserved(self, mock_converter, mock_individual):
        """應該保留元資料（雖然目前未使用）"""
        adapter = GPStrategyAdapter(mock_converter)

        metadata = {
            'parent_ids': ['parent1', 'parent2'],
            'mutation_type': 'crossover'
        }

        # 雖然 metadata 參數目前未在 create_strategy_class 中使用，
        # 但應該能接受而不報錯
        strategy_class = adapter.create_strategy_class(
            individual=mock_individual,
            strategy_name='test',
            fitness=1.0,
            generation=0,
            metadata=metadata
        )

        assert strategy_class is not None

    def test_params_are_empty(self, mock_converter, mock_individual):
        """GP 策略應該沒有參數（表達式已固定）"""
        adapter = GPStrategyAdapter(mock_converter)

        strategy_class = adapter.create_strategy_class(
            individual=mock_individual,
            strategy_name='test',
            fitness=1.0,
            generation=0
        )

        assert strategy_class.params == {}
        assert strategy_class.param_space == {}


class TestGPExplorer:
    """Test: GPExplorer 類別"""

    @pytest.fixture
    def mock_converter(self):
        """建立 mock converter"""
        class MockConverter:
            def compile(self, individual):
                """Mock compile method"""
                def signal_func(close, high, low):
                    import numpy as np
                    mean_price = np.mean(close)
                    return close > mean_price
                return signal_func

            def to_python(self, individual):
                """Mock to_python method"""
                return "gt(close, ma(close, 20))"

        return MockConverter()

    @pytest.fixture
    def mock_gp_loop(self):
        """建立 mock GPLoop"""
        from unittest.mock import MagicMock

        # 模擬 GPLoopResult
        class MockGPLoopResult:
            def __init__(self):
                self.hall_of_fame = []
                self.generations_run = 5
                self.fitness_history = [1.0, 1.5, 2.0, 2.2, 2.5]
                self.avg_fitness_history = [0.5, 0.8, 1.0, 1.2, 1.4]
                self.stopped_early = False

        # 建立具有 top_n 個體的 hall_of_fame
        result = MockGPLoopResult()

        # 建立 mock individuals
        for i in range(3):
            individual = MagicMock()
            individual.fitness = MagicMock()
            individual.fitness.values = (2.5 - i * 0.1,)  # 递减的適應度
            result.hall_of_fame.append(individual)

        return result

    def test_explorer_initialization(self, mock_converter):
        """應該能正確初始化 GPExplorer"""
        from src.automation.gp_integration import GPExplorer

        explorer = GPExplorer(converter=mock_converter, timeout=300.0)
        assert explorer.converter is not None
        assert explorer.timeout == 300.0

    def test_explorer_initialization_with_defaults(self):
        """應該支援預設初始化"""
        from src.automation.gp_integration import GPExplorer

        explorer = GPExplorer()
        assert explorer.converter is None
        assert explorer.timeout is None

    @pytest.mark.parametrize("timeout", [None, 60.0, 300.0, 3600.0])
    def test_explorer_timeout_configuration(self, mock_converter, timeout):
        """應該支援不同的超時設定"""
        from src.automation.gp_integration import GPExplorer

        explorer = GPExplorer(converter=mock_converter, timeout=timeout)
        assert explorer.timeout == timeout

    def test_explore_success_path(self, mock_converter, mock_gp_loop):
        """應該成功執行探索"""
        from src.automation.gp_integration import (
            GPExplorer,
            GPExplorationRequest
        )
        from unittest.mock import patch, MagicMock
        import pandas as pd
        import numpy as np

        # 建立 mock 資料
        data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
        })

        explorer = GPExplorer(converter=mock_converter)
        request = GPExplorationRequest(
            symbol='BTCUSDT',
            timeframe='4h',
            population_size=50,
            generations=5,
            top_n=3
        )

        # Mock GPLoop 和相關依賴（所有都是延遲導入的）
        with patch('src.automation.gp_loop.GPLoop') as mock_gp_loop_class, \
             patch('src.gp.primitives.PrimitiveSetFactory'), \
             patch('src.gp.converter.ExpressionConverter'):

            # 設定 mock GPLoop
            mock_loop_instance = MagicMock()
            mock_loop_instance.__enter__ = MagicMock(return_value=mock_loop_instance)
            mock_loop_instance.__exit__ = MagicMock(return_value=None)
            mock_loop_instance.run.return_value = mock_gp_loop
            mock_gp_loop_class.return_value = mock_loop_instance

            # 執行探索
            result = explorer.explore(request, data)

            # 驗證結果
            assert result.success is True
            assert len(result.strategies) > 0
            assert result.elapsed_time >= 0
            assert result.error is None

    def test_explore_returns_correct_strategy_count(self, mock_converter, mock_gp_loop):
        """應該返回正確數量的策略"""
        from src.automation.gp_integration import (
            GPExplorer,
            GPExplorationRequest
        )
        from unittest.mock import patch, MagicMock
        import pandas as pd
        import numpy as np

        data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
        })

        explorer = GPExplorer(converter=mock_converter)

        # 測試 top_n=2
        request = GPExplorationRequest(
            symbol='BTCUSDT',
            population_size=50,
            generations=5,
            top_n=2
        )

        with patch('src.automation.gp_loop.GPLoop') as mock_gp_loop_class, \
             patch('src.gp.primitives.PrimitiveSetFactory'), \
             patch('src.gp.converter.ExpressionConverter'):

            mock_loop_instance = MagicMock()
            mock_loop_instance.__enter__ = MagicMock(return_value=mock_loop_instance)
            mock_loop_instance.__exit__ = MagicMock(return_value=None)
            mock_loop_instance.run.return_value = mock_gp_loop
            mock_gp_loop_class.return_value = mock_loop_instance

            result = explorer.explore(request, data)

            # top_n=2，所以最多返回 2 個策略
            assert len(result.strategies) <= 2

    def test_explore_strategy_info_completeness(self, mock_converter, mock_gp_loop):
        """返回的策略資訊應該完整"""
        from src.automation.gp_integration import (
            GPExplorer,
            GPExplorationRequest,
            DynamicStrategyInfo
        )
        from unittest.mock import patch, MagicMock
        import pandas as pd
        import numpy as np

        data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
        })

        explorer = GPExplorer(converter=mock_converter)
        request = GPExplorationRequest(
            symbol='BTCUSDT',
            timeframe='4h',
            population_size=50,
            generations=5,
            top_n=1
        )

        with patch('src.automation.gp_loop.GPLoop') as mock_gp_loop_class, \
             patch('src.gp.primitives.PrimitiveSetFactory'), \
             patch('src.gp.converter.ExpressionConverter'):

            mock_loop_instance = MagicMock()
            mock_loop_instance.__enter__ = MagicMock(return_value=mock_loop_instance)
            mock_loop_instance.__exit__ = MagicMock(return_value=None)
            mock_loop_instance.run.return_value = mock_gp_loop
            mock_gp_loop_class.return_value = mock_loop_instance

            result = explorer.explore(request, data)

            assert result.success is True
            assert len(result.strategies) > 0

            strategy_info = result.strategies[0]
            assert isinstance(strategy_info, DynamicStrategyInfo)
            assert strategy_info.name is not None
            assert strategy_info.strategy_class is not None
            assert strategy_info.expression is not None
            assert isinstance(strategy_info.fitness, (int, float))
            assert isinstance(strategy_info.generation, int)
            assert strategy_info.created_at is not None
            assert isinstance(strategy_info.metadata, dict)

    def test_explore_evolution_stats_present(self, mock_converter, mock_gp_loop):
        """結果應該包含演化統計"""
        from src.automation.gp_integration import (
            GPExplorer,
            GPExplorationRequest
        )
        from unittest.mock import patch, MagicMock
        import pandas as pd
        import numpy as np

        data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
        })

        explorer = GPExplorer(converter=mock_converter)
        request = GPExplorationRequest(
            symbol='BTCUSDT',
            population_size=50,
            generations=5
        )

        with patch('src.automation.gp_loop.GPLoop') as mock_gp_loop_class, \
             patch('src.gp.primitives.PrimitiveSetFactory'), \
             patch('src.gp.converter.ExpressionConverter'):

            mock_loop_instance = MagicMock()
            mock_loop_instance.__enter__ = MagicMock(return_value=mock_loop_instance)
            mock_loop_instance.__exit__ = MagicMock(return_value=None)
            mock_loop_instance.run.return_value = mock_gp_loop
            mock_gp_loop_class.return_value = mock_loop_instance

            result = explorer.explore(request, data)

            assert result.success is True
            assert 'best_fitness_per_gen' in result.evolution_stats
            assert 'avg_fitness_per_gen' in result.evolution_stats
            assert 'diversity_per_gen' in result.evolution_stats
            assert 'total_evaluations' in result.evolution_stats
            assert 'stopped_early' in result.evolution_stats

    def test_explore_invalid_request_error_handling(self, mock_converter):
        """應該正確處理無效輸入"""
        from src.automation.gp_integration import (
            GPExplorer,
            GPExplorationRequest
        )
        import pandas as pd

        explorer = GPExplorer(converter=mock_converter)
        request = GPExplorationRequest(symbol='BTCUSDT')

        # 傳入無效資料（None）
        result = explorer.explore(request, None)

        # 應該返回失敗結果而不是拋出異常
        assert result.success is False
        assert result.error is not None
        assert len(result.strategies) == 0

    def test_explore_empty_data_error_handling(self, mock_converter):
        """應該正確處理空資料"""
        from src.automation.gp_integration import (
            GPExplorer,
            GPExplorationRequest
        )
        import pandas as pd

        explorer = GPExplorer(converter=mock_converter)
        request = GPExplorationRequest(symbol='BTCUSDT')

        # 傳入空 DataFrame
        empty_data = pd.DataFrame()
        result = explorer.explore(request, empty_data)

        # 應該返回失敗結果
        assert result.success is False
        assert result.error is not None

    def test_explore_never_throws_exception(self, mock_converter):
        """explore() 不應該拋出異常，所有錯誤都應該在結果中返回"""
        from src.automation.gp_integration import (
            GPExplorer,
            GPExplorationRequest
        )
        from unittest.mock import patch, MagicMock
        import pandas as pd

        explorer = GPExplorer(converter=mock_converter)
        request = GPExplorationRequest(symbol='BTCUSDT')

        # 模擬 GPLoop 拋出異常
        with patch('src.automation.gp_loop.GPLoop') as mock_gp_loop_class:
            mock_loop_instance = MagicMock()
            mock_loop_instance.__enter__ = MagicMock(
                side_effect=Exception("GPLoop initialization failed")
            )
            mock_loop_instance.__exit__ = MagicMock(return_value=None)
            mock_gp_loop_class.return_value = mock_loop_instance

            # 不應該拋出異常
            try:
                result = explorer.explore(request, pd.DataFrame())
                # 應該返回失敗結果
                assert result.success is False
                assert result.error is not None
            except Exception as e:
                pytest.fail(f"explore() should not throw exception: {e}")

    def test_calculate_diversity_with_valid_data(self, mock_converter):
        """應該正確計算多樣性指標"""
        from src.automation.gp_integration import GPExplorer

        explorer = GPExplorer(converter=mock_converter)

        best_fitness = [1.0, 1.5, 2.0, 2.2, 2.5]
        avg_fitness = [0.5, 0.8, 1.0, 1.2, 1.4]

        diversity = explorer._calculate_diversity(best_fitness, avg_fitness)

        assert len(diversity) == len(best_fitness)
        assert all(0 <= d <= 1 for d in diversity)

    def test_calculate_diversity_edge_cases(self, mock_converter):
        """應該正確處理邊界情況"""
        from src.automation.gp_integration import GPExplorer

        explorer = GPExplorer(converter=mock_converter)

        # 空列表
        diversity = explorer._calculate_diversity([], [])
        assert diversity == []

        # 單一元素
        diversity = explorer._calculate_diversity([1.0], [0.5])
        assert len(diversity) == 1

        # 差距為零（所有值相同）
        diversity = explorer._calculate_diversity([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
        assert all(d == 0.5 for d in diversity)  # 返回中性值

    @pytest.mark.parametrize("top_n", [1, 3, 5, 10])
    def test_explore_respects_top_n_parameter(self, mock_converter, mock_gp_loop, top_n):
        """應該尊重 top_n 參數"""
        from src.automation.gp_integration import (
            GPExplorer,
            GPExplorationRequest
        )
        from unittest.mock import patch, MagicMock
        import pandas as pd
        import numpy as np

        data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
        })

        explorer = GPExplorer(converter=mock_converter)
        request = GPExplorationRequest(
            symbol='BTCUSDT',
            population_size=50,
            generations=5,
            top_n=top_n
        )

        with patch('src.automation.gp_loop.GPLoop') as mock_gp_loop_class, \
             patch('src.gp.primitives.PrimitiveSetFactory'), \
             patch('src.gp.converter.ExpressionConverter'):

            mock_loop_instance = MagicMock()
            mock_loop_instance.__enter__ = MagicMock(return_value=mock_loop_instance)
            mock_loop_instance.__exit__ = MagicMock(return_value=None)
            mock_loop_instance.run.return_value = mock_gp_loop
            mock_gp_loop_class.return_value = mock_loop_instance

            result = explorer.explore(request, data)

            # 返回的策略數應該 <= top_n
            assert len(result.strategies) <= top_n


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
