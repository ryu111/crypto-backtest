"""
LessonDetector 系統測試

測試項目：
1. LessonDetector 偵測邏輯
   - exceptional_performance: Sharpe > 2.0
   - unexpected_poor_performance: Sharpe < 0.5 且預期 > 1.0
   - risk_event: MaxDD > 25%
   - overfit_warning: 過擬合機率 > 30%
   - parameter_sensitivity: 穩健性差異 > 0.5

2. Memory MCP 格式化
   - 標籤格式正確（strategy,success,策略名）
   - content 包含完整資訊

3. InsightsManager 整合
   - insights.md 正確更新

執行方式：
    pytest tests/test_lesson_detector.py -v
"""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from src.learning.lesson_detector import (
    LessonDetector,
    create_lesson_detector
)
from src.learning.memory import (
    MemoryIntegration,
    StrategyInsight,
    TradingLesson,
    MemoryTags
)
from src.learning.insights import InsightsManager
from src.types.results import (
    BacktestResult,
    ValidationResult,
    PerformanceMetrics
)


# ========== 測試資料工廠 ==========

def create_test_backtest_result(
    sharpe: float = 1.5,
    total_return: float = 0.30,
    max_drawdown: float = -0.15,
    win_rate: float = 0.55
) -> BacktestResult:
    """建立測試用 BacktestResult"""
    metrics = PerformanceMetrics(
        sharpe_ratio=sharpe,
        total_return=total_return,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        profit_factor=1.5,
        total_trades=100
    )
    return BacktestResult(metrics=metrics)


def create_test_validation_result(
    grade: str = 'B',
    stages_passed: list = None,
    efficiency: float = None,
    overfit_prob: float = None
) -> ValidationResult:
    """建立測試用 ValidationResult"""
    if stages_passed is None:
        stages_passed = [1, 2, 3, 4]

    return ValidationResult(
        grade=grade,
        stages_passed=stages_passed,
        efficiency=efficiency,
        overfit_probability=overfit_prob
    )


# ========== 測試 LessonDetector 偵測邏輯 ==========

class TestLessonDetection:
    """測試偵測規則"""

    def setup_method(self):
        """每個測試前執行"""
        self.memory = MemoryIntegration()
        # 使用臨時檔案
        self.temp_dir = tempfile.mkdtemp()
        self.insights_file = Path(self.temp_dir) / "insights.md"
        self.insights_manager = InsightsManager(self.insights_file)
        self.detector = LessonDetector(self.memory, self.insights_manager)

    def teardown_method(self):
        """每個測試後清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_exceptional_performance_detection(self):
        """測試優異表現偵測（Sharpe > 2.0）"""
        result = create_test_backtest_result(sharpe=2.3)
        validation = create_test_validation_result(grade='A')

        analysis = self.detector.analyze(
            result=result,
            validation=validation,
            strategy_info={'name': 'MA Cross', 'type': 'trend'},
            config={'symbol': 'BTCUSDT', 'timeframe': '4h'}
        )

        # 驗證偵測結果
        assert analysis is not None, "應該偵測到優異表現"
        assert analysis['should_record'] is True
        assert analysis['lesson_type'] == 'exceptional_performance'
        assert 'Sharpe' in analysis['reason']
        assert '2.3' in analysis['reason']

        # 驗證 insight 類型
        assert isinstance(analysis['insight'], StrategyInsight)

    def test_unexpected_poor_performance_detection(self):
        """測試異常低表現偵測（Sharpe < 0.5 且預期 > 1.0）"""
        result = create_test_backtest_result(sharpe=0.3)
        validation = create_test_validation_result(grade='D')

        analysis = self.detector.analyze(
            result=result,
            validation=validation,
            strategy_info={'name': 'MA Cross', 'type': 'trend'},
            config={'symbol': 'BTCUSDT', 'timeframe': '4h'},
            expected_sharpe=1.5  # 預期是 1.5，實際只有 0.3
        )

        # 驗證偵測結果
        assert analysis is not None
        assert analysis['lesson_type'] == 'unexpected_poor_performance'
        assert '0.3' in analysis['reason']
        assert '1.5' in analysis['reason']

        # 驗證 lesson 類型
        assert isinstance(analysis['insight'], TradingLesson)

    def test_risk_event_detection(self):
        """測試風險事件偵測（MaxDD > 25%）"""
        result = create_test_backtest_result(
            sharpe=1.2,
            max_drawdown=-0.30  # 30% 回撤
        )
        validation = create_test_validation_result(grade='C')

        analysis = self.detector.analyze(
            result=result,
            validation=validation,
            strategy_info={'name': 'Breakout', 'type': 'breakout'},
            config={'symbol': 'ETHUSDT', 'timeframe': '1h'}
        )

        # 驗證偵測結果
        assert analysis is not None
        assert analysis['lesson_type'] == 'risk_event'
        assert '30' in analysis['reason'] or '0.30' in analysis['reason']

        # 驗證 lesson 類型
        assert isinstance(analysis['insight'], TradingLesson)

    def test_overfit_warning_detection(self):
        """測試過擬合警告偵測（過擬合機率 > 30%）"""
        result = create_test_backtest_result(sharpe=1.8)
        validation = create_test_validation_result(
            grade='C',
            overfit_prob=0.45  # 45% 過擬合機率
        )

        analysis = self.detector.analyze(
            result=result,
            validation=validation,
            strategy_info={'name': 'RSI', 'type': 'momentum'},
            config={'symbol': 'BTCUSDT', 'timeframe': '15m'}
        )

        # 驗證偵測結果
        assert analysis is not None
        assert analysis['lesson_type'] == 'overfit_warning'
        # 支援多種格式：45%, 45.0%, 0.45
        assert any(s in analysis['reason'] for s in ['45%', '45.0%', '0.45'])

    def test_parameter_sensitivity_detection(self):
        """測試參數敏感度偵測（穩健性差異 > 0.5）"""
        result = create_test_backtest_result(sharpe=1.5)
        validation = create_test_validation_result(grade='B')

        analysis = self.detector.analyze(
            result=result,
            validation=validation,
            strategy_info={'name': 'MA Cross', 'type': 'trend'},
            config={'symbol': 'BTCUSDT', 'timeframe': '4h'},
            robustness_variance=0.75  # 穩健性變異數很高
        )

        # 驗證偵測結果
        assert analysis is not None
        assert analysis['lesson_type'] == 'parameter_sensitivity'
        assert '0.75' in analysis['reason']

    def test_no_detection_for_normal_performance(self):
        """測試正常表現不觸發偵測"""
        result = create_test_backtest_result(
            sharpe=1.5,  # 正常範圍
            max_drawdown=-0.12  # 正常範圍
        )
        validation = create_test_validation_result(
            grade='B',
            overfit_prob=0.15  # 低過擬合機率
        )

        analysis = self.detector.analyze(
            result=result,
            validation=validation,
            strategy_info={'name': 'MA Cross', 'type': 'trend'},
            config={'symbol': 'BTCUSDT', 'timeframe': '4h'},
            expected_sharpe=1.3,  # 符合預期
            robustness_variance=0.2  # 穩健性良好
        )

        # 驗證不應觸發偵測
        assert analysis is None, "正常表現不應觸發偵測"

    def test_detection_priority_exceptional_first(self):
        """測試偵測優先順序（優異表現優先）"""
        # 同時滿足多個條件：優異表現 + 高回撤
        result = create_test_backtest_result(
            sharpe=2.5,  # 觸發 exceptional_performance
            max_drawdown=-0.30  # 也觸發 risk_event
        )
        validation = create_test_validation_result(grade='A')

        analysis = self.detector.analyze(
            result=result,
            validation=validation,
            strategy_info={'name': 'Aggressive', 'type': 'momentum'},
            config={'symbol': 'BTCUSDT', 'timeframe': '1h'}
        )

        # 應優先偵測優異表現
        assert analysis['lesson_type'] == 'exceptional_performance'


# ========== 測試 Memory MCP 格式化 ==========

class TestMemoryFormatting:
    """測試 Memory MCP 格式化"""

    def setup_method(self):
        self.memory = MemoryIntegration()
        self.temp_dir = tempfile.mkdtemp()
        self.insights_file = Path(self.temp_dir) / "insights.md"
        self.insights_manager = InsightsManager(self.insights_file)
        self.detector = LessonDetector(self.memory, self.insights_manager)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_success_insight_memory_format(self):
        """測試成功洞察的 Memory MCP 格式"""
        result = create_test_backtest_result(sharpe=2.2)
        validation = create_test_validation_result(
            grade='A',
            efficiency=0.85
        )

        analysis = self.detector.analyze(
            result=result,
            validation=validation,
            strategy_info={
                'name': 'MA Cross',
                'type': 'trend',
                'params': {'fast': 10, 'slow': 30}
            },
            config={'symbol': 'BTCUSDT', 'timeframe': '4h'}
        )

        content = analysis['memory_content']
        metadata = analysis['memory_metadata']

        # 驗證 content 包含關鍵資訊
        assert 'MA Cross' in content
        assert 'BTCUSDT' in content
        assert '4h' in content
        assert '2.2' in content or '2.20' in content
        assert 'fast 10' in content or 'fast: 10' in content

        # 驗證 metadata 格式
        assert metadata['type'] == 'trading-insight'
        assert 'tags' in metadata

        # 驗證標籤格式
        tags = metadata['tags'].split(',')
        assert MemoryTags.ASSET_CRYPTO in tags
        assert MemoryTags.ASSET_BTC in tags
        assert '4h' in tags
        assert MemoryTags.STATUS_VALIDATED in tags
        assert MemoryTags.STRATEGY_MA_CROSS in tags
        assert MemoryTags.STRATEGY_TREND in tags

    def test_failure_lesson_memory_format(self):
        """測試失敗教訓的 Memory MCP 格式"""
        result = create_test_backtest_result(
            sharpe=0.3,
            max_drawdown=-0.35
        )
        validation = create_test_validation_result(
            grade='F',
            stages_passed=[1],
            overfit_prob=0.5
        )

        analysis = self.detector.analyze(
            result=result,
            validation=validation,
            strategy_info={
                'name': 'RSI Extreme',
                'type': 'momentum',
                'params': {'rsi_period': 7}
            },
            config={'symbol': 'ETHUSDT', 'timeframe': '15m'},
            expected_sharpe=1.5
        )

        content = analysis['memory_content']
        metadata = analysis['memory_metadata']

        # 驗證 content
        assert 'RSI Extreme' in content
        assert 'ETHUSDT' in content
        assert '教訓' in content or 'LESSON' in content.upper()

        # 驗證 metadata
        assert metadata['type'] == 'trading-lesson'

        # 驗證標籤
        tags = metadata['tags'].split(',')
        assert MemoryTags.ASSET_CRYPTO in tags
        assert MemoryTags.ASSET_ETH in tags
        assert MemoryTags.STATUS_FAILED in tags
        assert MemoryTags.STRATEGY_MOMENTUM in tags


# ========== 測試 InsightsManager 整合 ==========

class TestInsightsManagerIntegration:
    """測試 InsightsManager 整合"""

    def setup_method(self):
        self.memory = MemoryIntegration()
        self.temp_dir = tempfile.mkdtemp()
        self.insights_file = Path(self.temp_dir) / "insights.md"
        self.insights_manager = InsightsManager(self.insights_file)
        self.detector = LessonDetector(self.memory, self.insights_manager)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_record_success_to_insights_md(self):
        """測試記錄成功洞察到 insights.md"""
        result = create_test_backtest_result(sharpe=2.1)
        validation = create_test_validation_result(grade='A')

        analysis = self.detector.analyze(
            result=result,
            validation=validation,
            strategy_info={'name': 'Trend Follow', 'type': 'trend'},
            config={'symbol': 'BTCUSDT', 'timeframe': '1d'}
        )

        # 記錄到 insights.md
        self.detector.record_to_insights_md(
            insight=analysis['insight'],
            total_experiments=42
        )

        # 驗證檔案已更新
        content = self.insights_file.read_text(encoding='utf-8')

        # 驗證標題更新
        assert '總實驗數：42' in content

        # 驗證策略洞察已記錄
        assert 'Trend Follow' in content
        assert 'BTCUSDT' in content
        assert '2.1' in content or '2.10' in content

    def test_record_failure_to_insights_md(self):
        """測試記錄失敗教訓到 insights.md"""
        result = create_test_backtest_result(sharpe=0.4, max_drawdown=-0.30)
        validation = create_test_validation_result(
            grade='D',
            stages_passed=[1, 2]
        )

        analysis = self.detector.analyze(
            result=result,
            validation=validation,
            strategy_info={'name': 'Risky Strategy', 'type': 'momentum'},
            config={'symbol': 'ETHUSDT', 'timeframe': '5m'},
            expected_sharpe=1.5
        )

        # 記錄到 insights.md
        self.detector.record_to_insights_md(
            insight=analysis['insight'],
            total_experiments=10
        )

        # 驗證失敗教訓已記錄
        content = self.insights_file.read_text(encoding='utf-8')
        assert 'Risky Strategy' in content or '失敗' in content or '教訓' in content


# ========== 測試便利函數 ==========

class TestConvenienceFunctions:
    """測試便利函數"""

    def test_create_lesson_detector(self):
        """測試 create_lesson_detector"""
        with tempfile.TemporaryDirectory() as temp_dir:
            insights_file = str(Path(temp_dir) / "insights.md")

            detector = create_lesson_detector(insights_file)

            assert isinstance(detector, LessonDetector)
            assert isinstance(detector.memory, MemoryIntegration)
            assert isinstance(detector.insights_manager, InsightsManager)


# ========== 邊界測試 ==========

class TestEdgeCases:
    """邊界情況測試"""

    def setup_method(self):
        self.memory = MemoryIntegration()
        self.temp_dir = tempfile.mkdtemp()
        self.insights_file = Path(self.temp_dir) / "insights.md"
        self.insights_manager = InsightsManager(self.insights_file)
        self.detector = LessonDetector(self.memory, self.insights_manager)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sharpe_exactly_at_threshold(self):
        """測試 Sharpe 剛好在閾值上"""
        # Sharpe 剛好 2.0（邊界值）
        result = create_test_backtest_result(sharpe=2.0)
        validation = create_test_validation_result(grade='A')

        analysis = self.detector.analyze(
            result=result,
            validation=validation,
            strategy_info={'name': 'Test', 'type': 'trend'},
            config={'symbol': 'BTCUSDT', 'timeframe': '4h'}
        )

        # 閾值是 > 2.0，所以 2.0 不應觸發
        assert analysis is None

    def test_sharpe_just_above_threshold(self):
        """測試 Sharpe 剛好超過閾值"""
        result = create_test_backtest_result(sharpe=2.01)
        validation = create_test_validation_result(grade='A')

        analysis = self.detector.analyze(
            result=result,
            validation=validation,
            strategy_info={'name': 'Test', 'type': 'trend'},
            config={'symbol': 'BTCUSDT', 'timeframe': '4h'}
        )

        # 應該觸發
        assert analysis is not None
        assert analysis['lesson_type'] == 'exceptional_performance'

    def test_maxdd_exactly_at_threshold(self):
        """測試 MaxDD 剛好在閾值上"""
        # MaxDD 剛好 -0.25（25%）
        result = create_test_backtest_result(
            sharpe=1.5,
            max_drawdown=-0.25
        )
        validation = create_test_validation_result(grade='B')

        analysis = self.detector.analyze(
            result=result,
            validation=validation,
            strategy_info={'name': 'Test', 'type': 'trend'},
            config={'symbol': 'BTCUSDT', 'timeframe': '4h'}
        )

        # 閾值是 > 0.25（絕對值），所以 0.25 不應觸發
        assert analysis is None

    def test_maxdd_just_above_threshold(self):
        """測試 MaxDD 剛好超過閾值"""
        result = create_test_backtest_result(
            sharpe=1.5,
            max_drawdown=-0.251
        )
        validation = create_test_validation_result(grade='B')

        analysis = self.detector.analyze(
            result=result,
            validation=validation,
            strategy_info={'name': 'Test', 'type': 'trend'},
            config={'symbol': 'BTCUSDT', 'timeframe': '4h'}
        )

        # 應該觸發
        assert analysis is not None
        assert analysis['lesson_type'] == 'risk_event'

    def test_missing_validation_result(self):
        """測試缺少 ValidationResult"""
        result = create_test_backtest_result(sharpe=2.3)

        analysis = self.detector.analyze(
            result=result,
            validation=None,  # 沒有驗證結果
            strategy_info={'name': 'Test', 'type': 'trend'},
            config={'symbol': 'BTCUSDT', 'timeframe': '4h'}
        )

        # 仍應能偵測優異表現
        assert analysis is not None
        assert analysis['lesson_type'] == 'exceptional_performance'

    def test_positive_max_drawdown_value(self):
        """測試正值 MaxDD（確保取絕對值）"""
        # 有些系統可能返回正值的 MaxDD
        result = create_test_backtest_result(
            sharpe=1.5,
            max_drawdown=0.30  # 正值（應該取絕對值）
        )
        validation = create_test_validation_result(grade='B')

        analysis = self.detector.analyze(
            result=result,
            validation=validation,
            strategy_info={'name': 'Test', 'type': 'trend'},
            config={'symbol': 'BTCUSDT', 'timeframe': '4h'}
        )

        # 應該觸發風險事件（因為 abs(0.30) > 0.25）
        assert analysis is not None
        assert analysis['lesson_type'] == 'risk_event'


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
