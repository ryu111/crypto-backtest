"""測試 RegimeStrategyMapper - 市場狀態與策略映射"""

import pytest
from src.automation.regime_mapper import RegimeStrategyMapper, StrategyRecommendation
from src.regime.analyzer import MarketRegime


class TestRegimeStrategyMapperBasic:
    """基本功能測試"""

    def test_init_default_mapping(self):
        """測試使用預設映射初始化"""
        mapper = RegimeStrategyMapper()
        assert mapper.mapping is not None
        assert len(mapper.mapping) == len(MarketRegime)

    def test_init_custom_mapping(self):
        """測試使用自訂映射初始化"""
        custom = {
            MarketRegime.STRONG_BULL_HIGH_VOL: {
                'primary': ['custom_strategy'],
                'secondary': [],
                'reason': 'Custom reason',
                'confidence': 0.95,
                'weights': [1.0]
            }
        }
        mapper = RegimeStrategyMapper(custom_mapping=custom)

        # Custom mapping 應覆蓋預設值
        config = mapper.get_mapping_info(MarketRegime.STRONG_BULL_HIGH_VOL)
        assert config['primary_types'] == ['custom_strategy']
        assert config['confidence'] == 0.95

    def test_get_strategies_all_regimes(self):
        """測試所有 MarketRegime 都能返回有效結果"""
        mapper = RegimeStrategyMapper()
        available = ['ma_cross', 'supertrend', 'rsi', 'macd', 'stochastic']

        for regime in MarketRegime:
            recommendation = mapper.get_strategies(regime, available)

            # 基本驗證
            assert isinstance(recommendation, StrategyRecommendation)
            assert isinstance(recommendation.strategy_names, list)
            assert isinstance(recommendation.weights, list)
            assert isinstance(recommendation.reason, str)
            assert 0.0 <= recommendation.confidence <= 1.0

            # 權重與策略數量一致
            if recommendation.strategy_names:
                assert len(recommendation.weights) == len(recommendation.strategy_names)

    def test_get_strategies_empty_available(self):
        """測試空的 available_strategies"""
        mapper = RegimeStrategyMapper()
        recommendation = mapper.get_strategies(
            MarketRegime.STRONG_BULL_HIGH_VOL,
            available_strategies=[]
        )

        # 應返回空列表的 fallback
        assert recommendation.strategy_names == []
        assert recommendation.weights == []
        assert recommendation.confidence == 0.0
        assert '無可用策略' in recommendation.reason


class TestStrictValidation:
    """strict_validation 參數測試"""

    def test_strict_validation_false_logs_warning(self):
        """strict_validation=False（預設）：無效配置只記錄警告"""
        invalid_mapping = {
            MarketRegime.STRONG_BULL_HIGH_VOL: {
                'primary': ['trend'],
                # 缺少 'weights' 欄位
                'reason': 'Test',
                'confidence': 0.8
            }
        }

        # 應該成功初始化（不拋出錯誤）
        mapper = RegimeStrategyMapper(
            custom_mapping=invalid_mapping,
            strict_validation=False
        )

        # 驗證應報告問題
        validation = mapper.validate_mapping()
        assert not validation['valid']
        assert len(validation['issues']) > 0

    def test_strict_validation_true_raises_error(self):
        """strict_validation=True：無效配置拋出 ValueError"""
        invalid_mapping = {
            MarketRegime.STRONG_BULL_HIGH_VOL: {
                'primary': ['trend'],
                # 缺少 'weights' 欄位
                'reason': 'Test',
                'confidence': 0.8
            }
        }

        # 應該拋出 ValueError
        with pytest.raises(ValueError, match="Invalid mapping configuration"):
            RegimeStrategyMapper(
                custom_mapping=invalid_mapping,
                strict_validation=True
            )

    def test_strict_validation_weights_sum_not_one(self):
        """測試權重總和不為 1.0 時的驗證"""
        invalid_mapping = {
            MarketRegime.STRONG_BULL_HIGH_VOL: {
                'primary': ['trend', 'momentum'],
                'secondary': [],
                'reason': 'Test',
                'confidence': 0.8,
                'weights': [0.5, 0.4]  # 總和 0.9，不是 1.0
            }
        }

        # strict_validation=True 應拋出錯誤
        with pytest.raises(ValueError, match="Invalid mapping configuration"):
            RegimeStrategyMapper(
                custom_mapping=invalid_mapping,
                strict_validation=True
            )

        # strict_validation=False 應成功但記錄問題
        mapper = RegimeStrategyMapper(
            custom_mapping=invalid_mapping,
            strict_validation=False
        )
        validation = mapper.validate_mapping()
        assert not validation['valid']
        assert any('Weights sum' in issue for issue in validation['issues'])


class TestWeightIndexAlignment:
    """權重索引對齊測試"""

    def test_weight_idx_alignment_missing_primary_strategy(self):
        """
        測試當某個 primary 策略類型不存在時，權重索引仍對齊

        Scenario:
            primary=['trend', 'momentum'], weights=[0.6, 0.4]
            但 available_strategies 中無 trend 策略
            → momentum 應使用 weights[1] = 0.4（而非 weights[0]）
        """
        custom_mapping = {
            MarketRegime.STRONG_BULL_HIGH_VOL: {
                'primary': ['trend', 'momentum'],
                'secondary': [],
                'reason': 'Test alignment',
                'confidence': 0.8,
                'weights': [0.6, 0.4]
            }
        }

        mapper = RegimeStrategyMapper(
            custom_mapping=custom_mapping,
            strict_validation=False
        )

        # available_strategies 中只有 momentum 策略（無 trend）
        available = ['rsi', 'macd']  # 都是 momentum 類型

        recommendation = mapper.get_strategies(
            MarketRegime.STRONG_BULL_HIGH_VOL,
            available_strategies=available
        )

        # 應該推薦 momentum 策略
        assert len(recommendation.strategy_names) > 0
        assert all(s in available for s in recommendation.strategy_names)

        # 權重應該歸一化（總和為 1.0）
        assert abs(sum(recommendation.weights) - 1.0) < 0.01

    def test_weight_idx_alignment_missing_secondary_strategy(self):
        """
        測試當某個 secondary 策略類型不存在時，權重索引仍對齊
        """
        custom_mapping = {
            MarketRegime.NEUTRAL_HIGH_VOL: {
                'primary': ['momentum'],
                'secondary': ['mean_reversion', 'statistical_arbitrage'],
                'reason': 'Test secondary alignment',
                'confidence': 0.75,
                'weights': [0.5, 0.3, 0.2]
            }
        }

        mapper = RegimeStrategyMapper(
            custom_mapping=custom_mapping,
            strict_validation=False
        )

        # available_strategies 中無 mean_reversion 策略
        available = ['rsi']  # momentum 類型

        recommendation = mapper.get_strategies(
            MarketRegime.NEUTRAL_HIGH_VOL,
            available_strategies=available
        )

        # 應該只推薦 momentum 策略（mean_reversion 和 statistical_arbitrage 不存在）
        assert len(recommendation.strategy_names) > 0
        assert all(s in available for s in recommendation.strategy_names)

        # 權重應歸一化
        assert abs(sum(recommendation.weights) - 1.0) < 0.01

    def test_weight_idx_all_strategy_types_missing(self):
        """測試所有策略類型都不存在時，返回 fallback"""
        mapper = RegimeStrategyMapper()

        # available_strategies 中沒有任何策略（空列表）
        available = []

        recommendation = mapper.get_strategies(
            MarketRegime.STRONG_BULL_HIGH_VOL,
            available_strategies=available
        )

        # 應該是空的 fallback
        assert recommendation.strategy_names == []
        assert recommendation.weights == []
        assert recommendation.confidence == 0.0
        assert '無可用策略' in recommendation.reason


class TestWeightSumValidation:
    """權重總和驗證測試"""

    def test_weights_normalize_to_one(self):
        """測試權重歸一化到 1.0"""
        mapper = RegimeStrategyMapper()
        available = ['ma_cross', 'supertrend', 'rsi', 'macd']

        for regime in MarketRegime:
            recommendation = mapper.get_strategies(regime, available)

            if recommendation.strategy_names:
                # 權重總和應為 1.0（允許浮點誤差）
                weight_sum = sum(recommendation.weights)
                assert abs(weight_sum - 1.0) < 0.01, \
                    f"{regime.value}: 權重總和 {weight_sum} 不等於 1.0"

    def test_weights_count_matches_strategies_count(self):
        """測試權重數量與策略數量一致"""
        mapper = RegimeStrategyMapper()
        available = ['ma_cross', 'supertrend', 'rsi', 'macd']

        for regime in MarketRegime:
            recommendation = mapper.get_strategies(regime, available)

            assert len(recommendation.weights) == len(recommendation.strategy_names), \
                f"{regime.value}: 權重數量與策略數量不一致"


class TestEdgeCases:
    """邊界情況測試"""

    def test_none_available_strategies(self):
        """測試 available_strategies=None（應使用 StrategyRegistry.list_all()）"""
        mapper = RegimeStrategyMapper()

        # None 應觸發從 StrategyRegistry 獲取所有策略
        recommendation = mapper.get_strategies(
            MarketRegime.STRONG_BULL_HIGH_VOL,
            available_strategies=None
        )

        # 應該返回有效結果（假設 StrategyRegistry 有策略）
        assert isinstance(recommendation, StrategyRecommendation)
        # 可能有策略，也可能沒有（取決於 StrategyRegistry 狀態）

    def test_unknown_regime_fallback(self):
        """測試未知 regime 時的 fallback"""
        mapper = RegimeStrategyMapper()

        # 清空 mapping 模擬未知 regime
        mapper.mapping.clear()

        available = ['ma_cross', 'rsi']
        recommendation = mapper.get_strategies(
            MarketRegime.STRONG_BULL_HIGH_VOL,
            available_strategies=available
        )

        # 應該使用 fallback
        assert 'Fallback' in recommendation.reason
        assert recommendation.confidence == 0.3
        assert set(recommendation.strategy_names) == set(available)

    def test_get_mapping_info_existing_regime(self):
        """測試 get_mapping_info 返回正確的配置資訊"""
        mapper = RegimeStrategyMapper()

        info = mapper.get_mapping_info(MarketRegime.STRONG_BULL_HIGH_VOL)

        assert 'regime' in info
        assert 'primary_types' in info
        assert 'secondary_types' in info
        assert 'reason' in info
        assert 'confidence' in info
        assert 'weights' in info
        assert info['regime'] == MarketRegime.STRONG_BULL_HIGH_VOL.value

    def test_get_mapping_info_missing_regime(self):
        """測試 get_mapping_info 處理不存在的 regime"""
        mapper = RegimeStrategyMapper()
        mapper.mapping.clear()

        info = mapper.get_mapping_info(MarketRegime.STRONG_BULL_HIGH_VOL)

        assert 'error' in info
        assert info['error'] == 'No mapping found'

    def test_get_all_mappings(self):
        """測試 get_all_mappings 返回所有映射"""
        mapper = RegimeStrategyMapper()

        all_mappings = mapper.get_all_mappings()

        # 應該包含所有 MarketRegime
        assert len(all_mappings) == len(MarketRegime)

        for regime in MarketRegime:
            assert regime.value in all_mappings


class TestValidateMapping:
    """validate_mapping 方法測試"""

    def test_validate_default_mapping(self):
        """測試預設映射應該是有效的"""
        mapper = RegimeStrategyMapper()

        validation = mapper.validate_mapping()

        assert validation['valid'], \
            f"預設映射無效: {validation['issues']}"
        assert validation['total_regimes'] == len(MarketRegime)
        assert validation['mapped_regimes'] == len(MarketRegime)
        assert len(validation['issues']) == 0

    def test_validate_incomplete_mapping(self):
        """測試不完整的映射（缺少某些 regime）"""
        # 只映射一個 regime
        incomplete_mapping = {
            MarketRegime.STRONG_BULL_HIGH_VOL: {
                'primary': ['trend'],
                'secondary': [],
                'reason': 'Test',
                'confidence': 0.8,
                'weights': [1.0]
            }
        }

        mapper = RegimeStrategyMapper(
            custom_mapping=incomplete_mapping,
            strict_validation=False
        )

        validation = mapper.validate_mapping()

        # 應該報告缺少其他 regime 的映射
        # 但由於我們使用 update，預設映射仍存在
        # 所以應該是 valid 的
        assert validation['valid']

    def test_validate_missing_required_fields(self):
        """測試缺少必要欄位的映射"""
        invalid_mapping = {
            MarketRegime.STRONG_BULL_HIGH_VOL: {
                'primary': ['trend'],
                # 缺少 weights, reason, confidence
            }
        }

        mapper = RegimeStrategyMapper(
            custom_mapping=invalid_mapping,
            strict_validation=False
        )

        validation = mapper.validate_mapping()

        assert not validation['valid']
        assert len(validation['issues']) > 0
        assert any('Missing' in issue for issue in validation['issues'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
