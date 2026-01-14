"""
經驗教訓偵測器

自動判斷回測結果是否值得記錄到 insights.md 和 Memory MCP。

職責：
- 偵測值得記錄的實驗結果（優異表現、失敗教訓、風險事件等）
- 生成結構化洞察資料（StrategyInsight / TradingLesson）
- 整合 Memory MCP 和 insights.md 更新

使用範例:
    from src.learning.lesson_detector import LessonDetector
    from src.learning.memory import MemoryIntegration
    from src.learning.insights import InsightsManager

    memory = MemoryIntegration()
    insights_manager = InsightsManager(insights_file)
    detector = LessonDetector(memory, insights_manager)

    # 分析回測結果
    analysis = detector.analyze(
        result=backtest_result,
        validation=validation_result,
        strategy_info={'name': 'ma_cross', 'type': 'trend'},
        config={'symbol': 'BTCUSDT', 'timeframe': '4h'},
        expected_sharpe=1.5
    )

    if analysis:
        print(f"偵測到 {analysis['reason']}")
        # 自動存儲到 Memory MCP（需外部呼叫 MCP）
        # mcp__memory-service__store_memory(
        #     content=analysis['memory_content'],
        #     metadata=analysis['memory_metadata']
        # )
"""

import logging
from typing import Dict, Any, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from src.types.results import BacktestResult, ValidationResult

from .memory import (
    MemoryIntegration,
    StrategyInsight,
    TradingLesson
)
from .insights import InsightsManager
from ..types.enums import LessonType

logger = logging.getLogger(__name__)

# 偵測閾值常數（來自 CLAUDE.md）
EXCEPTIONAL_SHARPE = 2.0
POOR_SHARPE = 0.5
HIGH_OVERFIT_PROB = 0.3
HIGH_DRAWDOWN = 0.25
HIGH_ROBUSTNESS_VARIANCE = 0.5


class LessonDetector:
    """
    經驗教訓偵測器

    自動判斷回測結果是否值得記錄，並生成結構化洞察資料。

    偵測規則（來自 CLAUDE.md）:
        - exceptional_performance: Sharpe > 2.0
        - unexpected_poor_performance: Sharpe < 0.5 且預期 > 1.0
        - overfit_warning: 過擬合機率 > 0.3
        - risk_event: MaxDD > 0.25
        - parameter_sensitivity: 穩健性差異 > 0.5
    """

    def __init__(
        self,
        memory: MemoryIntegration,
        insights_manager: InsightsManager
    ):
        """
        初始化偵測器

        Args:
            memory: MemoryIntegration 實例
            insights_manager: InsightsManager 實例
        """
        self.memory = memory
        self.insights_manager = insights_manager

    def analyze(
        self,
        result: 'BacktestResult',
        validation: Optional['ValidationResult'],
        strategy_info: Dict[str, Any],
        config: Dict[str, Any],
        expected_sharpe: Optional[float] = None,
        robustness_variance: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        分析回測結果，判斷是否值得記錄

        Args:
            result: BacktestResult 物件
            validation: ValidationResult 物件（可選）
            strategy_info: 策略資訊 {'name': str, 'type': str, 'version': str}
            config: 配置 {'symbol': str, 'timeframe': str, ...}
            expected_sharpe: 預期 Sharpe（用於判斷是否異常低）
            robustness_variance: 穩健性測試的變異數（用於判斷參數敏感度）

        Returns:
            分析結果字典，如果不值得記錄則返回 None
            {
                'should_record': bool,
                'lesson_type': LessonType,
                'reason': str,
                'insight': StrategyInsight | TradingLesson,
                'memory_content': str,
                'memory_metadata': dict
            }
        """
        # 提取績效指標
        sharpe = result.metrics.sharpe_ratio
        max_dd = abs(result.metrics.max_drawdown)  # 確保為正值

        # 過擬合機率（從 ValidationResult 或直接提供）
        overfit_prob = None
        if validation and validation.overfit_probability is not None:
            overfit_prob = validation.overfit_probability

        # 逐一檢查偵測規則
        lesson_type, reason = self._detect_lesson_type(
            sharpe=sharpe,
            max_dd=max_dd,
            overfit_prob=overfit_prob,
            expected_sharpe=expected_sharpe,
            robustness_variance=robustness_variance
        )

        # 如果沒有值得記錄的洞察
        if lesson_type is None:
            return None

        # 根據類型生成洞察
        if lesson_type == LessonType.EXCEPTIONAL:
            insight = self._create_success_insight(
                result, validation, strategy_info, config
            )
        else:
            insight = self._create_failure_lesson(
                result, validation, strategy_info, config, lesson_type, reason
            )

        # 格式化為 Memory MCP 存儲格式
        memory_content, memory_metadata = self._format_for_memory(insight)

        return {
            'should_record': True,
            'lesson_type': lesson_type,
            'reason': reason,
            'insight': insight,
            'memory_content': memory_content,
            'memory_metadata': memory_metadata
        }

    def _detect_lesson_type(
        self,
        sharpe: float,
        max_dd: float,
        overfit_prob: Optional[float],
        expected_sharpe: Optional[float],
        robustness_variance: Optional[float]
    ) -> tuple[Optional[LessonType], str]:
        """
        偵測洞察類型

        Args:
            sharpe: Sharpe Ratio
            max_dd: 最大回撤（絕對值）
            overfit_prob: 過擬合機率（0-1）
            expected_sharpe: 預期 Sharpe
            robustness_variance: 穩健性變異數

        Returns:
            (lesson_type, reason) 或 (None, '') 如果不值得記錄
        """
        # 1. Exceptional Performance
        if sharpe > EXCEPTIONAL_SHARPE:
            return LessonType.EXCEPTIONAL, f'Sharpe {sharpe:.2f} 超越 {EXCEPTIONAL_SHARPE}'

        # 2. Unexpected Poor Performance
        if expected_sharpe and sharpe < POOR_SHARPE and expected_sharpe > 1.0:
            return (
                LessonType.POOR,
                f'Sharpe {sharpe:.2f} 遠低於預期 {expected_sharpe:.2f}'
            )

        # 3. Overfit Warning
        if overfit_prob and overfit_prob > HIGH_OVERFIT_PROB:
            return (
                LessonType.OVERFIT,
                f'過擬合機率 {overfit_prob:.1%} 超過閾值 {HIGH_OVERFIT_PROB:.0%}'
            )

        # 4. Risk Event
        if max_dd > HIGH_DRAWDOWN:
            return (
                LessonType.RISK,
                f'最大回撤 {max_dd:.1%} 超過閾值 {HIGH_DRAWDOWN:.0%}'
            )

        # 5. Parameter Sensitivity
        if robustness_variance and robustness_variance > HIGH_ROBUSTNESS_VARIANCE:
            return (
                LessonType.SENSITIVITY,
                f'參數敏感度 {robustness_variance:.2f} 超過閾值 {HIGH_ROBUSTNESS_VARIANCE}'
            )

        # 無值得記錄的洞察
        return None, ''

    def _create_success_insight(
        self,
        result: 'BacktestResult',
        validation: Optional['ValidationResult'],
        strategy_info: Dict[str, Any],
        config: Dict[str, Any]
    ) -> StrategyInsight:
        """
        建立成功洞察（StrategyInsight）

        Args:
            result: BacktestResult
            validation: ValidationResult（可選）
            strategy_info: 策略資訊
            config: 配置

        Returns:
            StrategyInsight 物件
        """
        # 提取績效
        metrics = result.metrics

        # 提取驗證結果
        wfa_efficiency = None
        wfa_grade = None
        if validation:
            wfa_efficiency = validation.efficiency
            wfa_grade = validation.grade

        # 提取參數（從 strategy_info 或 result）
        params = strategy_info.get('params', {})
        if hasattr(result, 'params'):
            params = result.params

        insight = StrategyInsight(
            strategy_name=strategy_info.get('name', ''),
            symbol=config.get('symbol', ''),
            timeframe=config.get('timeframe', ''),
            best_params=params,
            sharpe_ratio=metrics.sharpe_ratio,
            total_return=metrics.total_return,
            max_drawdown=abs(metrics.max_drawdown),
            win_rate=metrics.win_rate,
            wfa_efficiency=wfa_efficiency,
            wfa_grade=wfa_grade,
            market_conditions=None,  # TODO: 從市場數據推斷
            notes=f'自動偵測：優異表現 (Sharpe {metrics.sharpe_ratio:.2f})'
        )

        return insight

    def _create_failure_lesson(
        self,
        result: 'BacktestResult',
        validation: Optional['ValidationResult'],
        strategy_info: Dict[str, Any],
        config: Dict[str, Any],
        lesson_type: LessonType,
        reason: str
    ) -> TradingLesson:
        """
        建立失敗教訓（TradingLesson）

        Args:
            result: BacktestResult
            validation: ValidationResult（可選）
            strategy_info: 策略資訊
            config: 配置
            lesson_type: 洞察類型
            reason: 原因描述

        Returns:
            TradingLesson 物件
        """
        # 映射 lesson_type → failure_type
        failure_type_map = {
            LessonType.POOR: 'market_change',
            LessonType.OVERFIT: 'overfitting',
            LessonType.RISK: 'poor_validation',
            LessonType.SENSITIVITY: 'parameter_instability'
        }

        failure_type = failure_type_map.get(lesson_type, 'poor_validation')

        # 生成症狀描述
        symptoms = self._generate_symptoms(result, validation, lesson_type)

        # 生成預防建議
        prevention = self._generate_prevention(lesson_type)

        # 提取失敗參數
        failed_params = strategy_info.get('params', {})
        if hasattr(result, 'params'):
            failed_params = result.params

        lesson = TradingLesson(
            strategy_name=strategy_info.get('name', ''),
            symbol=config.get('symbol', ''),
            timeframe=config.get('timeframe', ''),
            failure_type=failure_type,
            description=reason,
            symptoms=symptoms,
            prevention=prevention,
            failed_params=failed_params
        )

        return lesson

    def _generate_symptoms(
        self,
        result: 'BacktestResult',
        validation: Optional['ValidationResult'],
        lesson_type: LessonType
    ) -> str:
        """生成症狀描述"""
        metrics = result.metrics

        if lesson_type == LessonType.POOR:
            return (
                f'Sharpe {metrics.sharpe_ratio:.2f}, '
                f'Return {metrics.total_return:.1%}, '
                f'遠低於歷史表現'
            )

        if lesson_type == LessonType.OVERFIT:
            overfit_prob = validation.overfit_probability if validation else 0
            return (
                f'樣本內 Sharpe {metrics.sharpe_ratio:.2f} 良好，'
                f'但過擬合機率高達 {overfit_prob:.1%}'
            )

        if lesson_type == LessonType.RISK:
            return (
                f'最大回撤 {abs(metrics.max_drawdown):.1%}，'
                f'超過可接受風險範圍'
            )

        if lesson_type == LessonType.SENSITIVITY:
            return (
                f'參數稍微調整後績效劇烈變化，'
                f'顯示策略不穩定'
            )

        return '未知症狀'

    def _generate_prevention(self, lesson_type: LessonType) -> str:
        """生成預防建議"""
        prevention_map = {
            LessonType.POOR: (
                '定期重新驗證策略，監控市場狀態變化，'
                '設定績效閾值自動停止交易'
            ),
            LessonType.OVERFIT: (
                '增加樣本外驗證時間，使用 Walk-Forward 分析，'
                '簡化策略邏輯減少參數數量'
            ),
            LessonType.RISK: (
                '降低槓桿倍數，收緊止損設定，'
                '增加部位管理規則'
            ),
            LessonType.SENSITIVITY: (
                '選擇對參數不敏感的穩健策略，'
                '使用參數範圍優化而非單點優化'
            )
        }

        return prevention_map.get(
            lesson_type,
            '進行更嚴格的驗證測試'
        )

    def _format_for_memory(
        self,
        insight: Union[StrategyInsight, TradingLesson]
    ) -> tuple[str, Dict[str, Any]]:
        """
        格式化洞察為 Memory MCP 存儲格式

        Args:
            insight: StrategyInsight 或 TradingLesson

        Returns:
            (content, metadata) 適合 mcp__memory-service__store_memory
        """
        if isinstance(insight, StrategyInsight):
            return self.memory.format_strategy_insight(insight)
        else:
            return self.memory.format_trading_lesson(insight)

    def record_to_insights_md(
        self,
        insight: Union[StrategyInsight, TradingLesson],
        total_experiments: int
    ):
        """
        將洞察記錄到 insights.md

        Args:
            insight: StrategyInsight 或 TradingLesson
            total_experiments: 總實驗數
        """
        # 轉換為 InsightsManager 需要的格式（舊格式 Experiment）
        legacy_exp = self._to_legacy_experiment(insight)

        # 更新 insights.md
        self.insights_manager.update(legacy_exp, total_experiments)

    def _to_legacy_experiment(
        self,
        insight: Union[StrategyInsight, TradingLesson]
    ) -> Any:
        """
        轉換洞察為 InsightsManager 需要的舊格式 Experiment

        Args:
            insight: StrategyInsight 或 TradingLesson

        Returns:
            舊格式 Experiment 物件（duck typing）
        """
        from types import SimpleNamespace

        # 推斷策略類型（從策略名稱）
        def infer_strategy_type(strategy_name: str) -> str:
            name_lower = strategy_name.lower()
            if 'ma' in name_lower or 'moving average' in name_lower or 'cross' in name_lower:
                return 'trend'
            elif 'rsi' in name_lower or 'momentum' in name_lower:
                return 'momentum'
            elif 'bollinger' in name_lower or 'mean' in name_lower or 'reversion' in name_lower:
                return 'mean_reversion'
            elif 'breakout' in name_lower or 'donchian' in name_lower:
                return 'breakout'
            else:
                return 'trend'  # 預設

        # StrategyInsight
        if isinstance(insight, StrategyInsight):
            strategy_type = infer_strategy_type(insight.strategy_name)

            # 建立支援 property 存取的類別
            class LegacyExperiment:
                def __init__(self, data):
                    self._data = data
                    # 複製所有屬性
                    for key, value in data.items():
                        setattr(self, key, value)

                @property
                def strategy_type(self):
                    return self._data['strategy']['type']

                @property
                def strategy_name(self):
                    return self._data['strategy']['name']

                @property
                def symbol(self):
                    return self._data['config'].get('symbol', '')

                @property
                def sharpe_ratio(self):
                    return self._data['results']['sharpe_ratio']

                @property
                def total_return(self):
                    return self._data['results']['total_return']

                @property
                def params(self):
                    return self._data.get('parameters', {})

                @property
                def grade(self):
                    return self._data['validation']['grade']

            data = {
                'id': f'insight_{insight.strategy_name}_{insight.created_at}',
                'timestamp': insight.created_at,
                'strategy': {
                    'name': insight.strategy_name,
                    'type': strategy_type
                },
                'config': {
                    'symbol': insight.symbol,
                    'timeframe': insight.timeframe
                },
                'parameters': insight.best_params,
                'results': {
                    'sharpe_ratio': insight.sharpe_ratio,
                    'total_return': insight.total_return,
                    'max_drawdown': insight.max_drawdown,
                    'win_rate': insight.win_rate
                },
                'validation': {
                    'grade': insight.wfa_grade or 'A',
                    'stages_passed': [1, 2, 3, 4, 5] if insight.wfa_grade in ['A', 'B'] else [1, 2]
                },
                'insights': [insight.notes] if insight.notes else []
            }

            return LegacyExperiment(data)

        # TradingLesson
        else:
            strategy_type = infer_strategy_type(insight.strategy_name)

            class LegacyExperiment:
                def __init__(self, data):
                    self._data = data
                    for key, value in data.items():
                        setattr(self, key, value)

                @property
                def strategy_type(self):
                    return self._data['strategy']['type']

                @property
                def strategy_name(self):
                    return self._data['strategy']['name']

                @property
                def symbol(self):
                    return self._data['config'].get('symbol', '')

                @property
                def sharpe_ratio(self):
                    return self._data['results']['sharpe_ratio']

                @property
                def total_return(self):
                    return self._data['results']['total_return']

                @property
                def params(self):
                    return self._data.get('parameters', {})

                @property
                def grade(self):
                    return self._data['validation']['grade']

            data = {
                'id': f'lesson_{insight.strategy_name}_{insight.created_at}',
                'timestamp': insight.created_at,
                'strategy': {
                    'name': insight.strategy_name,
                    'type': strategy_type
                },
                'config': {
                    'symbol': insight.symbol,
                    'timeframe': insight.timeframe
                },
                'parameters': insight.failed_params or {},
                'results': {
                    'sharpe_ratio': 0.0,  # 失敗案例無績效
                    'total_return': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0
                },
                'validation': {
                    'grade': 'F',
                    'stages_passed': []
                },
                'insights': [insight.description, insight.prevention]
            }

            return LegacyExperiment(data)


# 便利函數

def create_lesson_detector(
    insights_file_path: str
) -> LessonDetector:
    """
    建立 LessonDetector 實例（便利函數）

    Args:
        insights_file_path: insights.md 檔案路徑

    Returns:
        LessonDetector 實例
    """
    from pathlib import Path

    memory = MemoryIntegration()
    insights_manager = InsightsManager(Path(insights_file_path))

    return LessonDetector(memory, insights_manager)
