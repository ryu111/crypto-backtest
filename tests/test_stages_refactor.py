"""
測試 stages.py 的 Extract Method 重構

測試重點：
1. _create_stage_result 方法的邏輯正確性
2. 所有 checks 通過時 passed=True, score=100
3. 部分 checks 失敗時 passed=False, score 正確計算
4. message 正確生成
5. details 正確包含 checks
"""

import pytest
from src.validator.stages import StageValidator, StageResult


class TestCreateStageResult:
    """測試 _create_stage_result 方法"""

    @pytest.fixture
    def validator(self):
        """建立驗證器實例"""
        return StageValidator()

    def test_all_checks_pass(self, validator):
        """測試所有檢查通過時的結果"""
        checks = {
            'check1': True,
            'check2': True,
            'check3': True,
        }
        thresholds = {'threshold1': 0.5}
        details = {'metric1': 1.5}
        pass_message = "所有檢查通過"
        fail_message_prefix = "部分檢查失敗"

        result = validator._create_stage_result(
            checks=checks,
            thresholds=thresholds,
            details=details,
            pass_message=pass_message,
            fail_message_prefix=fail_message_prefix
        )

        # 驗證
        assert isinstance(result, StageResult)
        assert result.passed is True, "所有 checks 通過時 passed 應為 True"
        assert result.score == 100.0, f"所有 checks 通過時 score 應為 100，實際為 {result.score}"
        assert result.message == pass_message, f"通過時應使用 pass_message，實際為 {result.message}"
        assert 'checks' in result.details, "details 應包含 checks"
        assert result.details['checks'] == checks, "details['checks'] 內容應與輸入一致"
        assert result.details['metric1'] == 1.5, "原始 details 內容應保留"
        assert result.threshold == thresholds, "threshold 應與輸入一致"

    def test_partial_checks_fail(self, validator):
        """測試部分檢查失敗時的結果"""
        checks = {
            'check1': True,
            'check2': False,
            'check3': True,
            'check4': False,
        }
        thresholds = {'threshold1': 0.5}
        details = {'metric1': 1.5}
        pass_message = "所有檢查通過"
        fail_message_prefix = "未通過檢查"

        result = validator._create_stage_result(
            checks=checks,
            thresholds=thresholds,
            details=details,
            pass_message=pass_message,
            fail_message_prefix=fail_message_prefix
        )

        # 驗證
        assert result.passed is False, "有 checks 失敗時 passed 應為 False"

        # Score = 2/4 * 100 = 50.0
        expected_score = 50.0
        assert result.score == expected_score, f"score 應為 {expected_score}，實際為 {result.score}"

        # Message 應包含失敗的 checks
        assert result.message.startswith(fail_message_prefix), "失敗時應使用 fail_message_prefix"
        assert 'check2' in result.message, "message 應包含失敗的 check2"
        assert 'check4' in result.message, "message 應包含失敗的 check4"
        assert 'check1' not in result.message or 'check3' not in result.message, "message 不應包含通過的 checks"

        # Details 應包含 checks
        assert 'checks' in result.details
        assert result.details['checks'] == checks
        assert result.details['metric1'] == 1.5

    def test_all_checks_fail(self, validator):
        """測試所有檢查失敗時的結果"""
        checks = {
            'check1': False,
            'check2': False,
            'check3': False,
        }
        thresholds = {'threshold1': 0.5}
        details = {'metric1': 1.5}
        pass_message = "所有檢查通過"
        fail_message_prefix = "未通過檢查"

        result = validator._create_stage_result(
            checks=checks,
            thresholds=thresholds,
            details=details,
            pass_message=pass_message,
            fail_message_prefix=fail_message_prefix
        )

        # 驗證
        assert result.passed is False
        assert result.score == 0.0, f"所有 checks 失敗時 score 應為 0，實際為 {result.score}"
        assert result.message.startswith(fail_message_prefix)
        # 所有 checks 都應該在 message 中
        assert 'check1' in result.message
        assert 'check2' in result.message
        assert 'check3' in result.message

    def test_single_check_pass(self, validator):
        """測試單一檢查通過的結果"""
        checks = {
            'single_check': True,
        }
        thresholds = {}
        details = {}
        pass_message = "檢查通過"
        fail_message_prefix = "檢查失敗"

        result = validator._create_stage_result(
            checks=checks,
            thresholds=thresholds,
            details=details,
            pass_message=pass_message,
            fail_message_prefix=fail_message_prefix
        )

        assert result.passed is True
        assert result.score == 100.0
        assert result.message == pass_message

    def test_single_check_fail(self, validator):
        """測試單一檢查失敗的結果"""
        checks = {
            'single_check': False,
        }
        thresholds = {}
        details = {}
        pass_message = "檢查通過"
        fail_message_prefix = "檢查失敗"

        result = validator._create_stage_result(
            checks=checks,
            thresholds=thresholds,
            details=details,
            pass_message=pass_message,
            fail_message_prefix=fail_message_prefix
        )

        assert result.passed is False
        assert result.score == 0.0
        assert 'single_check' in result.message

    def test_score_calculation_various_ratios(self, validator):
        """測試不同比例下的 score 計算"""
        test_cases = [
            # (通過數, 總數, 期望 score)
            (5, 5, 100.0),
            (4, 5, 80.0),
            (3, 5, 60.0),
            (2, 5, 40.0),
            (1, 5, 20.0),
            (0, 5, 0.0),
            (3, 4, 75.0),
            (2, 3, 66.66666666666667),
        ]

        for passed, total, expected_score in test_cases:
            checks = {f'check{i}': i < passed for i in range(total)}

            result = validator._create_stage_result(
                checks=checks,
                thresholds={},
                details={},
                pass_message="通過",
                fail_message_prefix="失敗"
            )

            assert abs(result.score - expected_score) < 0.01, \
                f"當 {passed}/{total} 通過時，score 應為 {expected_score}，實際為 {result.score}"

    def test_details_preservation(self, validator):
        """測試原始 details 是否正確保留"""
        checks = {'check1': True}
        thresholds = {'t1': 0.5, 't2': 1.0}
        original_details = {
            'metric1': 100,
            'metric2': 'test',
            'nested': {'key': 'value'},
            'list': [1, 2, 3],
        }

        result = validator._create_stage_result(
            checks=checks,
            thresholds=thresholds,
            details=original_details.copy(),
            pass_message="通過",
            fail_message_prefix="失敗"
        )

        # 原始 details 應保留
        assert result.details['metric1'] == 100
        assert result.details['metric2'] == 'test'
        assert result.details['nested'] == {'key': 'value'}
        assert result.details['list'] == [1, 2, 3]

        # checks 應該加入
        assert 'checks' in result.details
        assert result.details['checks'] == checks

    def test_empty_checks_edge_case(self, validator):
        """測試空 checks 的邊界情況"""
        # 注意：空 checks 在實際使用中不應該出現，但測試完整性
        checks = {}

        # 這會導致 ZeroDivisionError，我們預期會拋出錯誤
        with pytest.raises(ZeroDivisionError):
            validator._create_stage_result(
                checks=checks,
                thresholds={},
                details={},
                pass_message="通過",
                fail_message_prefix="失敗"
            )


class TestStageMethodsUsingCreateStageResult:
    """測試各個 stage 方法是否正確使用 _create_stage_result"""

    @pytest.fixture
    def validator(self):
        """建立驗證器實例"""
        return StageValidator()

    def test_stage1_uses_create_stage_result(self, validator):
        """驗證 stage1 正確使用 _create_stage_result"""
        import pandas as pd
        import numpy as np
        from src.backtester.engine import BacktestResult

        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        equity = pd.Series(np.linspace(10000, 15000, 100), index=dates)
        returns = equity.pct_change().fillna(0)

        trades = pd.DataFrame({
            'Entry Timestamp': dates[:50],
            'Exit Timestamp': dates[1:51],
            'PnL': np.random.randn(50) * 100 + 50,
            'Return Pct': np.random.randn(50) * 0.02 + 0.01,
            'Size': [1.0] * 50,
            'Avg Entry Price': [50000.0] * 50,
        })

        result = BacktestResult(
            total_return=0.5,
            annual_return=0.3,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=1.2,
            max_drawdown=-0.15,
            max_drawdown_duration=10,
            volatility=0.2,
            total_trades=50,
            win_rate=0.6,
            profit_factor=1.8,
            avg_win=80.0,
            avg_loss=-40.0,
            avg_trade_duration=24.0,
            expectancy=10.0,
            recovery_factor=3.0,
            ulcer_index=5.0,
            equity_curve=equity,
            trades=trades,
            daily_returns=returns,
        )

        stage_result = validator.stage1_basic_backtest(result)

        # 驗證 StageResult 結構
        assert isinstance(stage_result, StageResult)
        assert hasattr(stage_result, 'passed')
        assert hasattr(stage_result, 'score')
        assert hasattr(stage_result, 'details')
        assert hasattr(stage_result, 'message')
        assert hasattr(stage_result, 'threshold')

        # 驗證 details 包含 checks
        assert 'checks' in stage_result.details, "stage1 應使用 _create_stage_result，details 應包含 checks"

        # 驗證 checks 包含所有必要的檢查項目
        checks = stage_result.details['checks']
        expected_checks = ['total_return', 'total_trades', 'sharpe_ratio', 'max_drawdown', 'profit_factor']
        for check in expected_checks:
            assert check in checks, f"checks 應包含 {check}"

    def test_stage2_uses_create_stage_result(self, validator):
        """驗證 stage2 正確使用 _create_stage_result"""
        import pandas as pd
        import numpy as np

        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.02 + 0.001)

        stage_result = validator.stage2_statistical_tests(returns)

        assert isinstance(stage_result, StageResult)
        assert 'checks' in stage_result.details, "stage2 應使用 _create_stage_result"

        checks = stage_result.details['checks']
        expected_checks = ['t_test', 'sharpe_ci', 'skewness']
        for check in expected_checks:
            assert check in checks, f"checks 應包含 {check}"

    def test_stage5_uses_create_stage_result(self, validator):
        """驗證 stage5 正確使用 _create_stage_result（足夠交易數）"""
        import pandas as pd
        import numpy as np

        trades = pd.DataFrame({
            'Return Pct': np.random.randn(50) * 0.02 + 0.01,
        })

        stage_result = validator.stage5_monte_carlo(trades)

        assert isinstance(stage_result, StageResult)

        # 只有在交易數足夠時才有 checks
        if stage_result.passed or 'error' not in stage_result.details:
            assert 'checks' in stage_result.details, "stage5 應使用 _create_stage_result"

            checks = stage_result.details['checks']
            expected_checks = ['p5_positive', 'p1_acceptable', 'median_vs_original']
            for check in expected_checks:
                assert check in checks, f"checks 應包含 {check}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
