"""
Phase 6 Integration 測試

測試以下元件的整合：
1. Task 6.1: Strategy Registry GP 擴展
2. Task 6.2: GP Loop 自動化
3. Task 6.3: Learning System 整合

測試覆蓋：
- 單元測試：個別函數和方法
- 整合測試：元件間的協作
- 邊界測試：邊界條件和錯誤處理
- 回歸測試：確保現有功能不受影響
"""

import pytest
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import json
from pandas import Series, DataFrame

from src.strategies.registry import StrategyRegistry
from src.strategies.base import BaseStrategy
from src.automation.gp_loop import GPLoop, GPLoopConfig, run_gp_evolution
from src.gp.learning import GPLearningIntegration, INSIGHT_FITNESS_THRESHOLD

logger = logging.getLogger(__name__)


# ============================================================================
# 測試 Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_registry():
    """清理策略註冊表"""
    yield
    StrategyRegistry.clear()


@pytest.fixture
def sample_strategy():
    """建立樣本策略"""

    @StrategyRegistry.register('test_strategy')
    class TestStrategy(BaseStrategy):
        name = "test_strategy"
        description = "Test strategy"
        strategy_type = "trend"
        version = "1.0"
        params = {"period": 20}
        param_space = {
            'period': {'type': 'int', 'low': 5, 'high': 50},
            'threshold': {'type': 'float', 'low': 0.1, 'high': 0.9}
        }

        def __init__(self, period: int = 20, threshold: float = 0.5):
            self.period = period
            self.threshold = threshold

        def calculate_indicators(self, data: DataFrame) -> Dict[str, Series]:
            return {}

        def generate_signals(self, data):
            return None, None, None, None

    return TestStrategy


@pytest.fixture
def gp_strategy():
    """建立 GP 策略"""

    class GPTestStrategy(BaseStrategy):
        name = "gp_test_001"
        description = "GP generated test strategy"
        strategy_type = "gp_evolved"
        version = "1.0"
        params = {}
        param_space = {}

        def calculate_indicators(self, data: DataFrame) -> Dict[str, Series]:
            return {}

        def generate_signals(self, data):
            return None, None, None, None

    return GPTestStrategy


@pytest.fixture
def mock_evolution_result():
    """建立模擬的演化結果"""
    result = Mock()
    result.best_fitness = 1.85
    result.best_individual = Mock()
    result.best_individual.__str__ = Mock(return_value="(+ (* x y) (* z w))")
    result.generations_run = 30
    result.stopped_early = False
    result.elapsed_time = 120.5
    result.hall_of_fame = [
        Mock(fitness=Mock(values=(1.85,))),
        Mock(fitness=Mock(values=(1.75,))),
        Mock(fitness=Mock(values=(1.65,))),
    ]
    result.fitness_history = [0.5, 0.8, 1.0, 1.2, 1.5, 1.85]
    result.config = Mock()
    result.config.population_size = 50
    return result


# ============================================================================
# Task 6.1: Strategy Registry GP 擴展
# ============================================================================

class TestStrategyRegistryGPExtension:
    """測試 Strategy Registry 的 GP 擴展功能"""

    def test_register_gp_strategy_basic(self, gp_strategy):
        """測試基本的 GP 策略註冊"""
        metadata = {
            'fitness': 1.85,
            'generation': 30,
            'expression': '(+ (* x y) (* z w))'
        }

        StrategyRegistry.register_gp_strategy(
            'gp_test_001',
            gp_strategy,
            metadata=metadata
        )

        assert StrategyRegistry.exists('gp_test_001')
        assert StrategyRegistry.get('gp_test_001') == gp_strategy

    def test_register_gp_strategy_with_metadata(self, gp_strategy):
        """測試 GP 策略註冊並驗證元資料儲存"""
        metadata = {
            'fitness': 1.85,
            'generation': 30,
            'expression': '(+ (* x y) (* z w))',
            'evolved_at': '2024-01-18T10:00:00'
        }

        StrategyRegistry.register_gp_strategy(
            'gp_evolved_btc_001',
            gp_strategy,
            metadata=metadata
        )

        retrieved_metadata = StrategyRegistry.get_strategy_metadata('gp_evolved_btc_001')
        assert retrieved_metadata is not None
        assert retrieved_metadata['fitness'] == 1.85
        assert retrieved_metadata['generation'] == 30

    def test_get_strategy_metadata_nonexistent(self):
        """測試取得不存在的策略元資料"""
        metadata = StrategyRegistry.get_strategy_metadata('nonexistent')
        assert metadata is None

    def test_list_gp_strategies_by_metadata(self, gp_strategy):
        """測試透過元資料標識 GP 策略"""
        StrategyRegistry.register_gp_strategy(
            'gp_001',
            gp_strategy,
            metadata={'fitness': 1.5}
        )

        StrategyRegistry.register_gp_strategy(
            'gp_002',
            gp_strategy,
            metadata={'fitness': 1.8}
        )

        # 註冊一個普通策略（使用裝飾器）
        @StrategyRegistry.register('normal')
        class NormalStrategy(BaseStrategy):
            params = {}
            param_space = {}

            def calculate_indicators(self, data: DataFrame) -> Dict[str, Series]:
                return {}

            def generate_signals(self, data):
                return None, None, None, None

        gp_list = StrategyRegistry.list_gp_strategies()

        assert 'gp_001' in gp_list
        assert 'gp_002' in gp_list
        assert 'normal' not in gp_list

    def test_register_gp_strategy_invalid_class(self):
        """測試註冊非 BaseStrategy 類別應該失敗"""

        class InvalidStrategy:
            pass

        with pytest.raises(TypeError):
            StrategyRegistry.register_gp_strategy(
                'invalid',
                InvalidStrategy
            )

    def test_register_gp_strategy_duplicate_name(self, gp_strategy):
        """測試重複註冊相同名稱應該失敗"""
        StrategyRegistry.register_gp_strategy(
            'gp_dup',
            gp_strategy,
            metadata={'fitness': 1.0}
        )

        with pytest.raises(ValueError):
            StrategyRegistry.register_gp_strategy(
                'gp_dup',
                gp_strategy
            )

    def test_store_and_retrieve_gp_metadata(self, gp_strategy):
        """測試 GP 元資料的儲存和檢索"""
        metadata_dict = {
            'fitness': 2.1,
            'generation': 50,
            'expression': '(/ (+ x y) (* z 2))',
            'evolved_at': '2024-01-18',
            'symbol': 'BTCUSDT'
        }

        StrategyRegistry.register_gp_strategy(
            'gp_test_meta',
            gp_strategy,
            metadata=metadata_dict
        )

        retrieved = StrategyRegistry.get_strategy_metadata('gp_test_meta')

        assert retrieved is not None
        for key, value in metadata_dict.items():
            assert retrieved[key] == value


# ============================================================================
# Task 6.2: GP Loop 自動化
# ============================================================================

class TestGPLoopConfiguration:
    """測試 GPLoopConfig 配置"""

    def test_config_defaults(self):
        """測試預設配置值"""
        config = GPLoopConfig()

        assert config.symbol == 'BTCUSDT'
        assert config.timeframe == '4h'
        assert config.population_size == 50
        assert config.generations == 30
        assert config.early_stopping == 10
        assert config.generate_top_n == 5
        assert config.record_to_learning is True
        assert config.initial_capital == 10000.0
        assert config.leverage == 10.0
        assert config.min_data_points == 100

    def test_config_custom_values(self):
        """測試自訂配置值"""
        config = GPLoopConfig(
            symbol='ETHUSDT',
            timeframe='1h',
            population_size=100,
            generations=50,
            initial_capital=50000.0,
            leverage=20.0
        )

        assert config.symbol == 'ETHUSDT'
        assert config.timeframe == '1h'
        assert config.population_size == 100
        assert config.generations == 50
        assert config.initial_capital == 50000.0
        assert config.leverage == 20.0


class TestGPLoopDataValidation:
    """測試 GPLoop 資料驗證"""

    def test_validate_data_insufficient_data_points(self):
        """測試資料點數不足應該拋出異常"""
        import pandas as pd

        config = GPLoopConfig(symbol='BTCUSDT', min_data_points=100)
        loop = GPLoop(config)

        data = pd.DataFrame({
            'open': [100.0] * 50,
            'high': [101.0] * 50,
            'low': [99.0] * 50,
            'close': [100.5] * 50,
            'volume': [1000.0] * 50
        })

        with pytest.raises(ValueError, match="資料不足"):
            loop._validate_data(data)

    def test_validate_data_with_nan(self):
        """測試包含 NaN 的資料應該拋出異常"""
        import pandas as pd
        import numpy as np

        config = GPLoopConfig(symbol='BTCUSDT', min_data_points=100)
        loop = GPLoop(config)

        data = pd.DataFrame({
            'open': [100.0] * 100,
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'close': [100.5] * 100,
            'volume': [1000.0] * 100
        })
        data.loc[50, 'close'] = np.nan

        with pytest.raises(ValueError, match="資料包含 NaN"):
            loop._validate_data(data)

    def test_validate_data_with_non_positive_prices(self):
        """測試包含非正價格的資料應該拋出異常"""
        import pandas as pd

        config = GPLoopConfig(symbol='BTCUSDT', min_data_points=100)
        loop = GPLoop(config)

        data = pd.DataFrame({
            'open': [100.0] * 100,
            'high': [101.0] * 100,
            'low': [-1.0] * 100,
            'close': [100.5] * 100,
            'volume': [1000.0] * 100
        })

        with pytest.raises(ValueError, match="資料包含非正價格"):
            loop._validate_data(data)

    def test_validate_data_invalid_ohlc(self):
        """測試 OHLC 邏輯錯誤的資料應該拋出異常"""
        import pandas as pd

        config = GPLoopConfig(symbol='BTCUSDT', min_data_points=100)
        loop = GPLoop(config)

        data = pd.DataFrame({
            'open': [100.0] * 100,
            'high': [50.0] * 100,
            'low': [99.0] * 100,
            'close': [100.5] * 100,
            'volume': [1000.0] * 100
        })

        with pytest.raises(ValueError, match="資料包含無效 OHLC"):
            loop._validate_data(data)

    def test_validate_data_non_monotonic_time(self):
        """測試時間順序錯誤的資料應該拋出異常"""
        import pandas as pd

        config = GPLoopConfig(symbol='BTCUSDT', min_data_points=100)
        loop = GPLoop(config)

        data = pd.DataFrame({
            'open': [100.0] * 100,
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'close': [100.5] * 100,
            'volume': [1000.0] * 100
        })
        data = data.iloc[::-1]

        with pytest.raises(ValueError, match="資料時間順序錯誤"):
            loop._validate_data(data)

    def test_validate_data_valid(self):
        """測試有效資料不應拋出異常"""
        import pandas as pd

        config = GPLoopConfig(symbol='BTCUSDT', min_data_points=100)
        loop = GPLoop(config)

        dates = pd.date_range('2024-01-01', periods=100, freq='4h')
        data = pd.DataFrame({
            'open': [100.0 + i * 0.1 for i in range(100)],
            'high': [101.0 + i * 0.1 for i in range(100)],
            'low': [99.0 + i * 0.1 for i in range(100)],
            'close': [100.5 + i * 0.1 for i in range(100)],
            'volume': [1000.0] * 100
        }, index=dates)

        loop._validate_data(data)


# ============================================================================
# Task 6.3: Learning System 整合
# ============================================================================

class TestGPLearningIntegration:
    """測試 GP 學習系統整合"""

    def test_learning_integration_initialization(self):
        """測試初始化"""
        integrator = GPLearningIntegration()
        assert integrator.recorder is None

    def test_learning_integration_with_recorder(self):
        """測試帶有 Recorder 的初始化"""
        mock_recorder = Mock()
        integrator = GPLearningIntegration(recorder=mock_recorder)
        assert integrator.recorder == mock_recorder

    def test_record_evolution_basic(self, mock_evolution_result):
        """測試基本演化記錄"""
        integrator = GPLearningIntegration()

        exp_id = integrator.record_evolution(
            result=mock_evolution_result,
            metadata={'symbol': 'BTCUSDT', 'timeframe': '4h'}
        )

        assert exp_id is not None
        assert 'exp' in exp_id
        assert 'gp' in exp_id
        assert 'btc' in exp_id

    def test_record_evolution_without_metadata(self, mock_evolution_result):
        """測試沒有元資料的演化記錄"""
        integrator = GPLearningIntegration()

        exp_id = integrator.record_evolution(result=mock_evolution_result)

        assert exp_id is not None
        assert 'exp' in exp_id

    def test_prepare_experiment_data(self, mock_evolution_result):
        """測試實驗資料準備"""
        integrator = GPLearningIntegration()

        metadata = {'symbol': 'ETHUSDT', 'timeframe': '1h'}
        data = integrator._prepare_experiment_data(mock_evolution_result, metadata)

        assert data['type'] == 'gp_evolution'
        assert data['best_fitness'] == 1.85
        assert data['generations_run'] == 30
        assert data['symbol'] == 'ETHUSDT'
        assert data['timeframe'] == '1h'
        assert 'timestamp' in data

    def test_generate_experiment_id_format(self):
        """測試實驗 ID 格式"""
        integrator = GPLearningIntegration()

        exp_id = integrator._generate_experiment_id({'symbol': 'BTCUSDT'})

        parts = exp_id.split('_')
        assert len(parts) >= 4
        assert parts[0] == 'exp'
        assert parts[1] == 'gp'
        assert parts[4] == 'btc'

    def test_record_to_insights_high_fitness(self, mock_evolution_result, tmp_path):
        """測試記錄優秀適應度到 insights"""
        mock_evolution_result.best_fitness = 1.5

        integrator = GPLearningIntegration()

        with patch.object(integrator, '_get_insights_file') as mock_get_file:
            insights_file = tmp_path / 'insights.md'
            insights_file.write_text('# 策略洞察\n', encoding='utf-8')
            mock_get_file.return_value = insights_file

            integrator.record_to_insights(mock_evolution_result)

            content = insights_file.read_text(encoding='utf-8')
            assert 'GP 演化洞察' in content

    def test_record_to_insights_low_fitness_skipped(self, mock_evolution_result):
        """測試低適應度不記錄到 insights"""
        mock_evolution_result.best_fitness = 0.5

        integrator = GPLearningIntegration()

        with patch.object(integrator, '_get_insights_file') as mock_get_file:
            integrator.record_to_insights(mock_evolution_result)

            mock_get_file.assert_not_called()

    def test_format_insight_content(self, mock_evolution_result):
        """測試洞察格式化內容"""
        integrator = GPLearningIntegration()

        insight = integrator._format_insight(mock_evolution_result, 'gp_evolution')

        assert 'GP 演化洞察' in insight
        assert '30' in insight
        assert 'Hall of Fame' in insight


# ============================================================================
# 整合測試
# ============================================================================

class TestPhase6Integration:
    """Phase 6 整合測試"""

    def test_registry_and_learning_integration(self, gp_strategy):
        """測試 Registry 和 Learning 的整合"""
        metadata = {
            'fitness': 1.8,
            'generation': 25,
            'symbol': 'BTCUSDT'
        }

        StrategyRegistry.register_gp_strategy(
            'gp_integrated_test',
            gp_strategy,
            metadata=metadata
        )

        assert StrategyRegistry.exists('gp_integrated_test')

        gp_list = StrategyRegistry.list_gp_strategies()
        assert 'gp_integrated_test' in gp_list

        retrieved_meta = StrategyRegistry.get_strategy_metadata('gp_integrated_test')
        assert retrieved_meta['fitness'] == 1.8

    def test_strategy_info_retrieval(self, sample_strategy):
        """測試策略資訊檢索"""
        info = StrategyRegistry.get_info('test_strategy')

        assert info['name'] == 'test_strategy'
        assert info['type'] == 'trend'
        assert info['version'] == '1.0'
        assert 'param_space' in info

    def test_strategy_creation(self, sample_strategy):
        """測試策略建立"""
        strategy = StrategyRegistry.create(
            'test_strategy',
            period=10,
            threshold=0.7
        )

        assert strategy.period == 10
        assert strategy.threshold == 0.7


# ============================================================================
# 邊界測試
# ============================================================================

class TestEdgeCases:
    """邊界情況測試"""

    def test_get_nonexistent_strategy(self):
        """測試取得不存在的策略"""
        with pytest.raises(KeyError):
            StrategyRegistry.get('nonexistent_strategy')

    def test_unregister_nonexistent(self):
        """測試移除不存在的策略"""
        with pytest.raises(KeyError):
            StrategyRegistry.unregister('nonexistent')

    def test_param_space_validation_missing_type(self):
        """測試缺少 type 的參數空間驗證"""

        @StrategyRegistry.register('bad_params')
        class BadParamStrategy(BaseStrategy):
            params = {}
            param_space = {
                'period': {'low': 5, 'high': 50}
            }

            def calculate_indicators(self, data: DataFrame) -> Dict[str, Series]:
                return {}

            def generate_signals(self, data):
                pass

        assert not StrategyRegistry.validate_param_space('bad_params')

    def test_param_space_validation_invalid_range(self):
        """測試無效範圍的參數空間驗證"""

        @StrategyRegistry.register('bad_range')
        class BadRangeStrategy(BaseStrategy):
            params = {}
            param_space = {
                'period': {
                    'type': 'int',
                    'low': 50,
                    'high': 5
                }
            }

            def calculate_indicators(self, data: DataFrame) -> Dict[str, Series]:
                return {}

            def generate_signals(self, data):
                pass

        assert not StrategyRegistry.validate_param_space('bad_range')

    def test_strategy_count_zero(self):
        """測試空的策略計數"""
        assert StrategyRegistry.get_strategy_count() == 0


# ============================================================================
# 回歸測試
# ============================================================================

class TestRegression:
    """回歸測試 - 確保現有功能不受影響"""

    def test_basic_strategy_registration_still_works(self):
        """測試基本策略註冊仍然有效"""

        @StrategyRegistry.register('legacy_strategy')
        class LegacyStrategy(BaseStrategy):
            params = {}
            param_space = {}

            def calculate_indicators(self, data: DataFrame) -> Dict[str, Series]:
                return {}

            def generate_signals(self, data):
                pass

        assert StrategyRegistry.exists('legacy_strategy')
        assert StrategyRegistry.get('legacy_strategy') == LegacyStrategy

    def test_list_all_strategies_works(self, sample_strategy):
        """測試列出所有策略仍然有效"""
        strategies = StrategyRegistry.list_all()
        assert 'test_strategy' in strategies
        assert isinstance(strategies, list)

    def test_list_by_type_works(self, sample_strategy):
        """測試按類型列出策略仍然有效"""
        trend_strategies = StrategyRegistry.list_by_type('trend')
        assert 'test_strategy' in trend_strategies

    def test_create_strategy_works(self, sample_strategy):
        """測試建立策略實例仍然有效"""
        strategy = StrategyRegistry.create('test_strategy', period=15, threshold=0.6)
        assert strategy.period == 15
        assert strategy.threshold == 0.6


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
