"""
測試 RegimeStrategyMapper 整合到 UltimateLoopController

測試涵蓋：
1. Regime Mapper 初始化
2. _select_by_regime 方法
3. 模組可用性日誌
4. 整合測試
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

# 測試前先檢查模組是否可用
from src.automation.ultimate_loop import (
    UltimateLoopController,
    REGIME_MAPPER_AVAILABLE,
    RegimeStrategyMapper,
    StrategyRecommendation
)
from src.automation.ultimate_config import UltimateLoopConfig

# 如果 regime 模組可用，導入相關類型
try:
    from src.regime.analyzer import MarketRegime, MarketState
    REGIME_AVAILABLE = True
except ImportError:
    REGIME_AVAILABLE = False
    MarketRegime = None
    MarketState = None


# ===== Fixtures =====

@pytest.fixture
def quick_config():
    """快速測試配置"""
    config = UltimateLoopConfig.create_quick_test_config()
    config.validation_enabled = False
    config.learning_enabled = False
    config.regime_detection = True  # 啟用 regime detection
    config.strategy_selection_mode = 'regime_aware'
    return config


# ===== 1. Regime Mapper 初始化測試 =====

class TestRegimeMapperInitialization:
    """測試 Regime Mapper 初始化"""

    def test_regime_mapper_initialized_when_available(self, quick_config):
        """測試 RegimeMapper 可用時正確初始化"""
        if not REGIME_MAPPER_AVAILABLE:
            pytest.skip("RegimeStrategyMapper not available")

        controller = UltimateLoopController(quick_config, verbose=False)

        # 應該有 regime_mapper 屬性
        assert hasattr(controller, 'regime_mapper')

        # 應該初始化為 RegimeStrategyMapper 實例
        assert controller.regime_mapper is not None
        assert isinstance(controller.regime_mapper, RegimeStrategyMapper)

    def test_regime_mapper_none_when_module_unavailable(self, quick_config):
        """測試模組不可用時 regime_mapper 為 None"""
        if REGIME_MAPPER_AVAILABLE:
            # 模擬模組不可用
            with patch('src.automation.ultimate_loop.REGIME_MAPPER_AVAILABLE', False):
                controller = UltimateLoopController(quick_config, verbose=False)
                assert controller.regime_mapper is None
        else:
            # 實際不可用情況
            controller = UltimateLoopController(quick_config, verbose=False)
            assert controller.regime_mapper is None

    def test_regime_mapper_initialization_logged(self, quick_config, caplog):
        """測試初始化時記錄日誌"""
        if not REGIME_MAPPER_AVAILABLE:
            pytest.skip("RegimeStrategyMapper not available")

        with caplog.at_level(logging.INFO):
            controller = UltimateLoopController(quick_config, verbose=True)

        # 應該記錄初始化訊息
        assert any(
            "Regime strategy mapper initialized" in record.message
            for record in caplog.records
        )


# ===== 2. _select_by_regime 方法測試 =====

class TestSelectByRegime:
    """測試 _select_by_regime 方法"""

    @pytest.mark.skipif(
        not REGIME_MAPPER_AVAILABLE or not REGIME_AVAILABLE,
        reason="Regime modules not available"
    )
    def test_select_by_regime_calls_mapper(self, quick_config):
        """測試 regime_mapper 可用時正確調用"""
        controller = UltimateLoopController(quick_config, verbose=False)
        controller.available_strategies = ['ma_cross', 'rsi', 'supertrend']

        # Mock market_state
        @dataclass
        class MockMarketState:
            regime: MarketRegime = MarketRegime.STRONG_BULL_LOW_VOL

        market_state = MockMarketState()

        # Mock regime_mapper.get_strategies
        mock_recommendation = StrategyRecommendation(
            strategy_names=['ma_cross', 'supertrend'],
            weights=[0.6, 0.4],
            reason='測試推薦',
            confidence=0.85
        )

        controller.regime_mapper.get_strategies = Mock(return_value=mock_recommendation)

        # 執行
        result = controller._select_by_regime(market_state)

        # 驗證呼叫
        controller.regime_mapper.get_strategies.assert_called_once_with(
            regime=MarketRegime.STRONG_BULL_LOW_VOL,
            available_strategies=['ma_cross', 'rsi', 'supertrend']
        )

        # 驗證回傳
        assert result == ['ma_cross', 'supertrend']

    def test_select_by_regime_fallback_when_mapper_unavailable(self, quick_config):
        """測試 regime_mapper 不可用時 fallback"""
        controller = UltimateLoopController(quick_config, verbose=False)
        controller.available_strategies = ['s1', 's2', 's3', 's4']
        controller.regime_mapper = None  # 模擬不可用

        @dataclass
        class MockMarketState:
            regime = 'trending'

        market_state = MockMarketState()

        result = controller._select_by_regime(market_state)

        # 應該 fallback 到前 3 個策略
        assert result == ['s1', 's2', 's3']

    @pytest.mark.skipif(
        not REGIME_MAPPER_AVAILABLE,
        reason="RegimeStrategyMapper not available"
    )
    def test_select_by_regime_fallback_when_no_regime_in_state(self, quick_config):
        """測試 market_state 無 regime 時 fallback"""
        controller = UltimateLoopController(quick_config, verbose=False)
        controller.available_strategies = ['s1', 's2', 's3', 's4']

        # market_state 沒有 regime 屬性
        @dataclass
        class MockMarketState:
            volatility: float = 0.5

        market_state = MockMarketState()

        result = controller._select_by_regime(market_state)

        # 應該 fallback
        assert result == ['s1', 's2', 's3']

    @pytest.mark.skipif(
        not REGIME_MAPPER_AVAILABLE or not REGIME_AVAILABLE,
        reason="Regime modules not available"
    )
    def test_select_by_regime_stores_recommendation(self, quick_config):
        """測試推薦結果被儲存到 _current_recommendation"""
        controller = UltimateLoopController(quick_config, verbose=False)
        controller.available_strategies = ['ma_cross', 'rsi']

        @dataclass
        class MockMarketState:
            regime: MarketRegime = MarketRegime.NEUTRAL_LOW_VOL

        market_state = MockMarketState()

        # Mock recommendation
        mock_recommendation = StrategyRecommendation(
            strategy_names=['rsi'],
            weights=[1.0],
            reason='中性低波動',
            confidence=0.80
        )

        controller.regime_mapper.get_strategies = Mock(return_value=mock_recommendation)

        # 執行
        controller._select_by_regime(market_state)

        # 驗證 _current_recommendation 被設定
        assert hasattr(controller, '_current_recommendation')
        assert controller._current_recommendation == mock_recommendation


# ===== 3. 模組可用性日誌測試 =====

class TestModuleAvailabilityLogging:
    """測試模組可用性顯示"""

    def test_module_availability_logs_regime_mapper(self, quick_config, caplog):
        """測試 Regime Mapper 狀態顯示在日誌中"""
        with caplog.at_level(logging.INFO):
            controller = UltimateLoopController(quick_config, verbose=True)

        # 查找 module availability 相關日誌
        log_messages = [record.message for record in caplog.records]

        # 應該有 "Module availability:" 訊息
        assert any("Module availability:" in msg for msg in log_messages)

        # 應該有 Regime Mapper 狀態
        assert any("Regime Mapper" in msg for msg in log_messages)

    def test_regime_mapper_status_correct(self, quick_config, caplog):
        """測試 Regime Mapper 狀態標記正確"""
        with caplog.at_level(logging.INFO):
            controller = UltimateLoopController(quick_config, verbose=True)

        log_messages = [record.message for record in caplog.records]

        # 根據實際可用性檢查標記
        if REGIME_MAPPER_AVAILABLE:
            # 應該有 "✓ Regime Mapper"
            assert any("✓" in msg and "Regime Mapper" in msg for msg in log_messages)
        else:
            # 應該有 "✗ Regime Mapper"
            assert any("✗" in msg and "Regime Mapper" in msg for msg in log_messages)


# ===== 4. 整合測試 =====

class TestRegimeMapperIntegration:
    """測試 Regime-aware 策略選擇完整流程"""

    @pytest.mark.skipif(
        not REGIME_MAPPER_AVAILABLE or not REGIME_AVAILABLE,
        reason="Regime modules not available"
    )
    def test_regime_aware_selection_in_select_strategies(self, quick_config):
        """測試 _select_strategies 使用 regime_aware 模式"""
        quick_config.strategy_selection_mode = 'regime_aware'

        controller = UltimateLoopController(quick_config, verbose=False)
        controller.available_strategies = ['ma_cross', 'rsi', 'supertrend']

        # Mock market_state
        @dataclass
        class MockMarketState:
            regime: MarketRegime = MarketRegime.STRONG_BULL_LOW_VOL

        market_state = MockMarketState()

        # Mock _select_by_regime
        controller._select_by_regime = Mock(return_value=['ma_cross', 'supertrend'])

        # 執行
        result = controller._select_strategies(market_state)

        # 驗證 _select_by_regime 被呼叫
        controller._select_by_regime.assert_called_once_with(market_state)

        # 驗證回傳
        assert result == ['ma_cross', 'supertrend']

    @pytest.mark.skipif(
        not REGIME_MAPPER_AVAILABLE or not REGIME_AVAILABLE,
        reason="Regime modules not available"
    )
    def test_regime_aware_fallback_when_no_market_state(self, quick_config):
        """測試 regime_aware 模式但無 market_state 時 fallback"""
        quick_config.strategy_selection_mode = 'regime_aware'

        controller = UltimateLoopController(quick_config, verbose=False)
        controller.available_strategies = ['s1', 's2', 's3']
        controller.strategy_stats = {}

        # market_state = None
        result = controller._select_strategies(market_state=None)

        # 應該 fallback 到 exploit 模式
        # 因為無 strategy_stats，會隨機選擇
        assert len(result) > 0
        assert all(s in controller.available_strategies for s in result)

    @pytest.mark.skipif(
        not REGIME_MAPPER_AVAILABLE or not REGIME_AVAILABLE,
        reason="Regime modules not available"
    )
    def test_full_regime_aware_workflow(self, quick_config):
        """測試完整的 regime-aware 工作流"""
        controller = UltimateLoopController(quick_config, verbose=False)
        controller.available_strategies = ['ma_cross', 'rsi', 'macd', 'supertrend']

        # 建立 market_state
        @dataclass
        class MockMarketState:
            regime: MarketRegime = MarketRegime.STRONG_BULL_LOW_VOL

        market_state = MockMarketState()

        # 執行 _select_strategies（使用 regime_aware 模式）
        result = controller._select_strategies(market_state)

        # 應該回傳趨勢策略（根據 DEFAULT_MAPPING）
        # STRONG_BULL_LOW_VOL → primary: ['trend']
        trend_strategies = ['ma_cross', 'supertrend']  # 這兩個是趨勢策略
        assert any(s in result for s in trend_strategies), \
            f"Expected trend strategies in {result}"

    def test_graceful_degradation_when_modules_unavailable(self, quick_config):
        """測試模組不可用時優雅處理"""
        # 即使模組不可用，controller 也應該正常初始化
        controller = UltimateLoopController(quick_config, verbose=False)

        assert controller is not None
        assert hasattr(controller, 'regime_mapper')

        # _select_strategies 應該仍能運作（fallback 到其他模式）
        controller.available_strategies = ['s1', 's2', 's3']
        result = controller._select_strategies(market_state=None)

        assert len(result) > 0


# ===== 5. Fallback 行為測試 =====

class TestFallbackBehavior:
    """測試各種 fallback 情境"""

    def test_fallback_to_default_when_no_strategies(self, quick_config):
        """測試無可用策略時的行為"""
        controller = UltimateLoopController(quick_config, verbose=False)
        controller.available_strategies = []

        result = controller._select_strategies(market_state=None)

        # 應該回傳空列表
        assert result == []

    @pytest.mark.skipif(
        not REGIME_MAPPER_AVAILABLE or not REGIME_AVAILABLE,
        reason="Regime modules not available"
    )
    def test_fallback_when_mapper_returns_empty(self, quick_config):
        """測試 mapper 回傳空列表時的行為"""
        controller = UltimateLoopController(quick_config, verbose=False)
        controller.available_strategies = ['s1', 's2']

        @dataclass
        class MockMarketState:
            regime: MarketRegime = MarketRegime.NEUTRAL_LOW_VOL

        market_state = MockMarketState()

        # Mock mapper 回傳空列表
        controller.regime_mapper.get_strategies = Mock(
            return_value=StrategyRecommendation(
                strategy_names=[],
                weights=[],
                reason='無可用策略',
                confidence=0.0
            )
        )

        result = controller._select_by_regime(market_state)

        # 應該回傳空列表（由 mapper 的 fallback 處理）
        assert result == []


# ===== 主測試執行 =====

if __name__ == "__main__":
    """執行測試"""
    pytest.main([__file__, '-v', '--tb=short'])
