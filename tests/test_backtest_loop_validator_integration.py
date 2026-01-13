"""
測試 BacktestLoop 與 BacktestValidator 的整合

驗證 Phase 8.5：BacktestLoop._validate_engine_on_startup() 方法
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.automation.backtest_loop import BacktestLoop
from src.automation.loop_config import BacktestLoopConfig
from src.backtester.validator import BacktestValidator, ValidationReport, ValidationResult


class TestBacktestLoopValidatorIntegration:
    """測試 BacktestLoop 啟動時驗證整合"""

    def test_validate_method_exists(self):
        """測試 1：確認 _validate_engine_on_startup 方法存在"""
        # Arrange
        config = BacktestLoopConfig(
            strategies=['ma_cross'],
            symbols=['BTCUSDT'],
            n_iterations=1
        )
        loop = BacktestLoop(config)

        # Act & Assert
        assert hasattr(loop, '_validate_engine_on_startup'), \
            "_validate_engine_on_startup 方法不存在"
        assert callable(loop._validate_engine_on_startup), \
            "_validate_engine_on_startup 不是可呼叫方法"

    @patch('src.backtester.validator.BacktestValidator')
    def test_validation_pass_no_exception(self, MockValidator):
        """測試 2：正常路徑 - 當驗證通過時，不應拋出異常"""
        # Arrange
        config = BacktestLoopConfig(
            strategies=['ma_cross'],
            symbols=['BTCUSDT'],
            n_iterations=1
        )

        # Mock ValidationReport（驗證通過）
        mock_report = Mock()
        mock_report.all_passed = True

        mock_validator = MockValidator.return_value
        mock_validator.validate_level.return_value = mock_report

        # Act & Assert（使用 context manager 會觸發 _setup）
        try:
            with BacktestLoop(config) as loop:
                # 如果沒有拋出異常，測試通過
                pass
        except RuntimeError:
            pytest.fail("驗證通過時不應拋出 RuntimeError")

        # 驗證 validate_level 被正確呼叫
        mock_validator.validate_level.assert_called_once_with("L2")

    @patch('src.backtester.validator.BacktestValidator')
    def test_validation_fail_raises_runtime_error(self, MockValidator):
        """測試 3：失敗路徑 - 當驗證失敗時，應拋出 RuntimeError"""
        # Arrange
        config = BacktestLoopConfig(
            strategies=['ma_cross'],
            symbols=['BTCUSDT'],
            n_iterations=1
        )

        # Mock ValidationResult（失敗的測試）
        failed_result = ValidationResult(
            success=False,
            level="L2",
            test_name="validate_sharpe_calculation",
            message="Sharpe 計算錯誤"
        )

        # Mock ValidationReport（驗證失敗）
        mock_report = Mock()
        mock_report.all_passed = False
        mock_report.results = [failed_result]

        mock_validator = MockValidator.return_value
        mock_validator.validate_level.return_value = mock_report

        # Act & Assert
        with pytest.raises(RuntimeError) as exc_info:
            with BacktestLoop(config) as loop:
                pass

        # 驗證異常訊息
        error_msg = str(exc_info.value)
        assert "❌ 回測引擎驗證失敗" in error_msg, "錯誤訊息應包含失敗標題"
        assert "validate_sharpe_calculation" in error_msg, "錯誤訊息應包含失敗測試名稱"

    @patch('src.backtester.validator.BacktestValidator')
    def test_error_message_contains_details(self, MockValidator):
        """測試 4：錯誤訊息應包含詳細資訊"""
        # Arrange
        config = BacktestLoopConfig(
            strategies=['ma_cross'],
            symbols=['BTCUSDT'],
            n_iterations=1
        )

        # Mock 多個失敗測試
        failed_tests = [
            ValidationResult(
                success=False,
                level="L2",
                test_name="validate_sharpe_calculation",
                message="Sharpe 計算錯誤：差異過大"
            ),
            ValidationResult(
                success=False,
                level="L2",
                test_name="validate_maxdd_calculation",
                message="MaxDD 計算錯誤：預期 0.15，實際 0.20"
            )
        ]

        mock_report = Mock()
        mock_report.all_passed = False
        mock_report.results = failed_tests

        mock_validator = MockValidator.return_value
        mock_validator.validate_level.return_value = mock_report

        # Act
        with pytest.raises(RuntimeError) as exc_info:
            with BacktestLoop(config) as loop:
                pass

        # Assert
        error_msg = str(exc_info.value)
        assert "validate_sharpe_calculation" in error_msg, "錯誤訊息應包含第一個失敗測試"
        assert "validate_maxdd_calculation" in error_msg, "錯誤訊息應包含第二個失敗測試"
        assert "Sharpe 計算錯誤" in error_msg, "錯誤訊息應包含詳細說明"
        assert "MaxDD 計算錯誤" in error_msg, "錯誤訊息應包含詳細說明"

    @patch('src.backtester.validator.BacktestValidator')
    def test_setup_calls_validate(self, MockValidator):
        """測試 5：_setup() 確實呼叫驗證方法"""
        # Arrange
        config = BacktestLoopConfig(
            strategies=['ma_cross'],
            symbols=['BTCUSDT'],
            n_iterations=1
        )

        mock_report = Mock()
        mock_report.all_passed = True

        mock_validator = MockValidator.return_value
        mock_validator.validate_level.return_value = mock_report

        # Act
        with BacktestLoop(config) as loop:
            pass

        # Assert
        # 驗證 BacktestValidator 被實例化
        MockValidator.assert_called()

        # 驗證 validate_level 被呼叫（L2 層級）
        mock_validator.validate_level.assert_called_once_with("L2")

    @patch('src.backtester.validator.BacktestValidator')
    def test_validation_logs_success(self, MockValidator, caplog):
        """測試 6：驗證通過時應記錄 INFO 日誌"""
        # Arrange
        config = BacktestLoopConfig(
            strategies=['ma_cross'],
            symbols=['BTCUSDT'],
            n_iterations=1
        )

        mock_report = Mock()
        mock_report.all_passed = True

        mock_validator = MockValidator.return_value
        mock_validator.validate_level.return_value = mock_report

        # Act
        import logging
        with caplog.at_level(logging.INFO):
            with BacktestLoop(config) as loop:
                pass

        # Assert
        assert "✅ 回測引擎驗證通過" in caplog.text, "應記錄驗證通過訊息"
        assert "L2 數值正確性" in caplog.text, "應說明驗證層級"

    @patch('src.backtester.validator.BacktestValidator')
    def test_validation_logs_failure(self, MockValidator, caplog):
        """測試 7：驗證失敗時應記錄 ERROR 日誌"""
        # Arrange
        config = BacktestLoopConfig(
            strategies=['ma_cross'],
            symbols=['BTCUSDT'],
            n_iterations=1
        )

        failed_result = ValidationResult(
            success=False,
            level="L2",
            test_name="validate_sharpe_calculation",
            message="計算錯誤"
        )

        mock_report = Mock()
        mock_report.all_passed = False
        mock_report.results = [failed_result]

        mock_validator = MockValidator.return_value
        mock_validator.validate_level.return_value = mock_report

        # Act
        import logging
        with caplog.at_level(logging.ERROR):
            try:
                with BacktestLoop(config) as loop:
                    pass
            except RuntimeError:
                pass  # 預期會拋出

        # Assert
        assert "❌ 回測引擎驗證失敗" in caplog.text, "應記錄驗證失敗訊息"


class TestBacktestValidatorReport:
    """測試 ValidationReport 的 all_passed 屬性"""

    def test_validation_report_has_all_passed_property(self):
        """測試 ValidationReport 有 all_passed 屬性"""
        # Arrange
        report = ValidationReport()

        # Act & Assert
        # 檢查 all_passed 是否存在（可能是屬性或方法）
        has_property = hasattr(report, 'all_passed')
        has_method = hasattr(report, 'all_passed') and callable(getattr(report, 'all_passed'))

        # 兩者之一應該存在
        assert has_property or has_method, \
            "ValidationReport 應該有 all_passed 屬性或方法"

    def test_all_passed_returns_true_when_no_failures(self):
        """測試當所有測試通過時，all_passed 為 True"""
        # Arrange
        report = ValidationReport()
        report.add(ValidationResult(
            success=True,
            level="L2",
            test_name="test1",
            message="通過"
        ))
        report.add(ValidationResult(
            success=True,
            level="L2",
            test_name="test2",
            message="通過"
        ))

        # Act
        result = report.all_passed if isinstance(report.all_passed, bool) else report.all_passed()

        # Assert
        assert result is True, "所有測試通過時 all_passed 應為 True"

    def test_all_passed_returns_false_when_has_failures(self):
        """測試當有測試失敗時，all_passed 為 False"""
        # Arrange
        report = ValidationReport()
        report.add(ValidationResult(
            success=True,
            level="L2",
            test_name="test1",
            message="通過"
        ))
        report.add(ValidationResult(
            success=False,
            level="L2",
            test_name="test2",
            message="失敗"
        ))

        # Act
        result = report.all_passed if isinstance(report.all_passed, bool) else report.all_passed()

        # Assert
        assert result is False, "有測試失敗時 all_passed 應為 False"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
