"""Regime Strategy Mapper - 市場狀態與策略映射

將市場狀態（MarketRegime）映射到合適的策略列表。
提供可自訂的映射規則，並動態從 StrategyRegistry 獲取可用策略。
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
from src.regime.analyzer import MarketRegime
from src.strategies import StrategyRegistry

logger = logging.getLogger(__name__)


@dataclass
class StrategyRecommendation:
    """策略推薦結果"""
    strategy_names: List[str]       # 推薦的策略列表
    weights: List[float]            # 每個策略的權重（相對重要性）
    reason: str                     # 推薦原因說明
    confidence: float               # 信心度 (0-1)

    def to_dict(self) -> Dict:
        """轉換為字典格式"""
        return {
            'strategy_names': self.strategy_names,
            'weights': self.weights,
            'reason': self.reason,
            'confidence': self.confidence
        }


class RegimeStrategyMapper:
    """
    Regime → Strategy 映射器

    根據市場狀態（MarketRegime）推薦合適的策略組合。

    使用場景：
    1. AI Loop 自動選擇策略
    2. 多策略組合動態調整
    3. 市場狀態過濾（只在適合環境執行策略）

    設計原則：
    - 趨勢策略適合強方向 + 低波動
    - 動量策略適合弱方向 + 高波動（捕捉反轉）
    - 均值回歸策略適合中性 + 低波動（區間震盪）
    - 波動率策略適合中性 + 高波動（捕捉突破）
    - 資金費率策略不依賴市場狀態（始終可用）
    """

    # 預設映射規則（可被 custom_mapping 覆蓋）
    DEFAULT_MAPPING = {
        # 強勢上漲
        MarketRegime.STRONG_BULL_HIGH_VOL: {
            'primary': ['trend'],       # 主要推薦：趨勢跟隨
            'secondary': ['momentum'],  # 次要推薦：動量突破
            'reason': '強勢上漲高波動：趨勢明確，適合趨勢跟隨',
            'confidence': 0.85,
            'weights': [0.7, 0.3]
        },
        MarketRegime.STRONG_BULL_LOW_VOL: {
            'primary': ['trend'],
            'secondary': [],
            'reason': '強勢上漲低波動：趨勢穩健，專注趨勢策略',
            'confidence': 0.90,
            'weights': [1.0]
        },

        # 弱勢上漲
        MarketRegime.WEAK_BULL_HIGH_VOL: {
            'primary': ['momentum'],
            'secondary': ['mean_reversion'],
            'reason': '弱勢上漲高波動：可能反轉，動量與均值回歸並用',
            'confidence': 0.70,
            'weights': [0.6, 0.4]
        },
        MarketRegime.WEAK_BULL_LOW_VOL: {
            'primary': ['trend', 'mean_reversion'],
            'secondary': [],
            'reason': '弱勢上漲低波動：趨勢與區間震盪並存',
            'confidence': 0.65,
            'weights': [0.5, 0.5]
        },

        # 中性
        MarketRegime.NEUTRAL_HIGH_VOL: {
            'primary': ['momentum'],
            'secondary': ['statistical_arbitrage'],
            'reason': '中性高波動：捕捉突破與統計套利機會',
            'confidence': 0.75,
            'weights': [0.7, 0.3]
        },
        MarketRegime.NEUTRAL_LOW_VOL: {
            'primary': ['mean_reversion'],
            'secondary': ['statistical_arbitrage'],
            'reason': '中性低波動：區間震盪，適合均值回歸',
            'confidence': 0.80,
            'weights': [0.7, 0.3]
        },

        # 弱勢下跌
        MarketRegime.WEAK_BEAR_HIGH_VOL: {
            'primary': ['momentum'],
            'secondary': ['mean_reversion'],
            'reason': '弱勢下跌高波動：可能反轉，動量與均值回歸並用',
            'confidence': 0.70,
            'weights': [0.6, 0.4]
        },
        MarketRegime.WEAK_BEAR_LOW_VOL: {
            'primary': ['trend', 'mean_reversion'],
            'secondary': [],
            'reason': '弱勢下跌低波動：趨勢與區間震盪並存',
            'confidence': 0.65,
            'weights': [0.5, 0.5]
        },

        # 強勢下跌
        MarketRegime.STRONG_BEAR_HIGH_VOL: {
            'primary': ['trend'],
            'secondary': ['momentum'],
            'reason': '強勢下跌高波動：趨勢明確，適合趨勢跟隨（反向）',
            'confidence': 0.85,
            'weights': [0.7, 0.3]
        },
        MarketRegime.STRONG_BEAR_LOW_VOL: {
            'primary': ['trend'],
            'secondary': [],
            'reason': '強勢下跌低波動：趨勢穩健，專注趨勢策略',
            'confidence': 0.90,
            'weights': [1.0]
        },
    }

    def __init__(
        self,
        custom_mapping: Optional[Dict] = None,
        strict_validation: bool = False
    ):
        """
        初始化映射器

        Args:
            custom_mapping: 自訂映射規則，格式同 DEFAULT_MAPPING
                {
                    MarketRegime.STRONG_BULL_HIGH_VOL: {
                        'primary': ['trend'],
                        'secondary': ['momentum'],
                        'reason': '...',
                        'confidence': 0.85,
                        'weights': [0.7, 0.3]
                    }
                }
            strict_validation: 嚴格驗證模式
                - True: 驗證失敗時拋出 ValueError（適合測試環境）
                - False: 驗證失敗僅記錄警告（預設，適合生產環境）

        Note:
            weights 陣列對應 primary + secondary 類型的索引位置，
            即使某類型無可用策略，該權重仍會被跳過（保持索引對齊）。
        """
        self.mapping = self.DEFAULT_MAPPING.copy()
        if custom_mapping:
            self.mapping.update(custom_mapping)

        # 初始化時驗證映射完整性
        validation = self.validate_mapping()
        if not validation['valid']:
            if strict_validation:
                raise ValueError(f"Invalid mapping configuration: {validation['issues']}")
            else:
                logger.warning(f"Mapping configuration has issues: {validation['issues']}")

    def get_strategies(
        self,
        regime: MarketRegime,
        available_strategies: Optional[List[str]] = None
    ) -> StrategyRecommendation:
        """
        根據市場狀態獲取推薦策略

        Args:
            regime: 當前市場狀態（MarketRegime 枚舉）
            available_strategies: 可用的策略列表（策略名稱），
                                 如果為 None，則從 StrategyRegistry 獲取所有策略

        Returns:
            StrategyRecommendation: 推薦結果

        Example:
            >>> mapper = RegimeStrategyMapper()
            >>> recommendation = mapper.get_strategies(
            ...     MarketRegime.STRONG_BULL_LOW_VOL,
            ...     available_strategies=['ma_cross', 'rsi', 'supertrend']
            ... )
            >>> print(recommendation.strategy_names)
            ['ma_cross', 'supertrend']  # 趨勢策略
        """
        # 獲取可用策略列表
        if available_strategies is None:
            available_strategies = StrategyRegistry.list_all()

        # 獲取映射配置（防禦性檢查）
        config = self.mapping.get(regime)
        if config is None:
            # Fallback：如果映射不存在，返回所有可用策略
            return self._fallback_recommendation(available_strategies)

        # 獲取推薦的策略類型
        primary_types = config.get('primary', [])
        secondary_types = config.get('secondary', [])
        config_weights = config.get('weights', [])

        # 從 StrategyRegistry 獲取每個類型的策略
        recommended_strategies = []
        weights = []

        # 處理 primary 策略（使用獨立的權重索引計數器）
        #
        # weight_idx 行為說明：
        # - 即使 strategy_type 無對應策略，weight_idx 也會遞增
        # - 確保與 config['weights'] 索引對齊
        # - Example: primary=['trend', 'momentum'], weights=[0.6, 0.4]
        #   - 如果 trend 無策略：跳過但 weight_idx 仍從 0→1
        #   - momentum 使用 weights[1] = 0.4
        weight_idx = 0
        for strategy_type in primary_types:
            strategies = self._get_strategies_by_type(strategy_type, available_strategies)
            if not strategies:
                weight_idx += 1  # 即使跳過也要遞增索引，保持對齊
                continue
            recommended_strategies.extend(strategies)
            # 為每個策略分配權重（primary 權重較高）
            if weight_idx < len(config_weights):
                weight_per_strategy = config_weights[weight_idx] / len(strategies)
            else:
                weight_per_strategy = 0.5 / len(strategies)  # 預設權重
            weights.extend([weight_per_strategy] * len(strategies))
            weight_idx += 1

        # 處理 secondary 策略（繼續使用相同的權重索引）
        for strategy_type in secondary_types:
            strategies = self._get_strategies_by_type(strategy_type, available_strategies)
            if not strategies:
                weight_idx += 1  # 即使跳過也要遞增索引，保持對齊
                continue
            recommended_strategies.extend(strategies)
            # 為每個策略分配權重（secondary 權重較低）
            if weight_idx < len(config_weights):
                weight_per_strategy = config_weights[weight_idx] / len(strategies)
            else:
                weight_per_strategy = 0.1 / len(strategies)  # 預設小權重
            weights.extend([weight_per_strategy] * len(strategies))
            weight_idx += 1

        # 防禦性檢查：如果沒有推薦策略，使用 fallback
        if not recommended_strategies:
            return self._fallback_recommendation(available_strategies)

        # 標準化權重（確保總和為 1）
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(recommended_strategies)] * len(recommended_strategies)

        # 防禦性檢查：確保 weights 和 strategies 長度一致
        if len(weights) != len(recommended_strategies):
            logger.warning(
                f"權重與策略長度不匹配 ({len(weights)} vs {len(recommended_strategies)})，"
                "使用均等權重"
            )
            weights = [1.0 / len(recommended_strategies)] * len(recommended_strategies)

        return StrategyRecommendation(
            strategy_names=recommended_strategies,
            weights=weights,
            reason=config.get('reason', '策略推薦'),
            confidence=config.get('confidence', 0.5)
        )

    def _get_strategies_by_type(
        self,
        strategy_type: str,
        available_strategies: List[str]
    ) -> List[str]:
        """
        根據策略類型獲取可用策略

        Args:
            strategy_type: 策略類型（trend, momentum, mean_reversion, etc.）
            available_strategies: 可用策略列表

        Returns:
            符合類型的策略名稱列表
        """
        # 從 StrategyRegistry 獲取該類型的所有策略
        type_strategies = StrategyRegistry.list_by_type(strategy_type)

        # 過濾出可用的策略
        filtered = [s for s in type_strategies if s in available_strategies]

        return filtered

    def _fallback_recommendation(
        self,
        available_strategies: List[str]
    ) -> StrategyRecommendation:
        """
        Fallback 推薦策略（當映射失敗或無策略時）

        Args:
            available_strategies: 可用策略列表

        Returns:
            StrategyRecommendation: 包含所有可用策略的均等權重推薦
        """
        if not available_strategies:
            return StrategyRecommendation(
                strategy_names=[],
                weights=[],
                reason='無可用策略',
                confidence=0.0
            )

        # 均等權重分配
        n = len(available_strategies)
        equal_weight = 1.0 / n

        return StrategyRecommendation(
            strategy_names=available_strategies,
            weights=[equal_weight] * n,
            reason='Fallback：未找到對應映射，使用所有可用策略',
            confidence=0.3
        )

    def get_mapping_info(self, regime: MarketRegime) -> Dict:
        """
        獲取指定市場狀態的映射配置資訊

        Args:
            regime: 市場狀態

        Returns:
            dict: 映射配置
        """
        config = self.mapping.get(regime)
        if config is None:
            return {
                'regime': regime.value,
                'error': 'No mapping found'
            }

        return {
            'regime': regime.value,
            'primary_types': config.get('primary', []),
            'secondary_types': config.get('secondary', []),
            'reason': config.get('reason', ''),
            'confidence': config.get('confidence', 0.5),
            'weights': config.get('weights', [])
        }

    def get_all_mappings(self) -> Dict[str, Dict]:
        """
        獲取所有市場狀態的映射配置

        Returns:
            dict: {regime_name: mapping_info}
        """
        return {
            regime.value: self.get_mapping_info(regime)
            for regime in MarketRegime
        }

    def validate_mapping(self) -> Dict:
        """
        驗證映射配置的完整性

        Returns:
            dict: 驗證結果
        """
        issues = []

        # 檢查是否所有 MarketRegime 都有映射
        for regime in MarketRegime:
            if regime not in self.mapping:
                issues.append(f'Missing mapping for {regime.value}')

        # 檢查每個映射的配置完整性
        for regime, config in self.mapping.items():
            if 'primary' not in config:
                issues.append(f'{regime.value}: Missing "primary" field')
            if 'weights' not in config:
                issues.append(f'{regime.value}: Missing "weights" field')
            if 'reason' not in config:
                issues.append(f'{regime.value}: Missing "reason" field')
            if 'confidence' not in config:
                issues.append(f'{regime.value}: Missing "confidence" field')

            # 檢查權重長度是否與策略類型數量匹配
            primary_count = len(config.get('primary', []))
            secondary_count = len(config.get('secondary', []))
            total_types = primary_count + secondary_count
            weights_count = len(config.get('weights', []))

            if total_types > 0 and weights_count != total_types:
                issues.append(
                    f'{regime.value}: Weights count ({weights_count}) '
                    f'!= strategy types count ({total_types})'
                )

            # 檢查權重總和是否接近 1.0（允許 0.01 的誤差）
            weights_list = config.get('weights', [])
            if weights_list and abs(sum(weights_list) - 1.0) > 0.01:
                issues.append(
                    f'{regime.value}: Weights sum ({sum(weights_list):.3f}) != 1.0'
                )

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'total_regimes': len(MarketRegime),
            'mapped_regimes': len(self.mapping)
        }
