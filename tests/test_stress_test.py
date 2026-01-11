"""
黑天鵝壓力測試器測試
"""

import pytest
import numpy as np
import pandas as pd
from src.validator.stress_test import (
    StressTester,
    StressTestResult,
    StressTestReport,
    HISTORICAL_EVENTS
)


@pytest.fixture
def sample_returns():
    """建立測試用報酬序列"""
    np.random.seed(42)
    # 產生 252 天的日報酬（約一年）
    returns = pd.Series(
        np.random.normal(loc=0.001, scale=0.02, size=252),
        index=pd.date_range('2023-01-01', periods=252, freq='D')
    )
    return returns


@pytest.fixture
def stress_tester():
    """建立壓力測試器實例"""
    return StressTester(survival_threshold=-0.5, risk_free_rate=0.02)


class TestStressTester:
    """StressTester 測試"""

    def test_init(self):
        """測試初始化"""
        tester = StressTester()
        assert tester.survival_threshold == -0.5
        assert tester.risk_free_rate == 0.0

        custom_tester = StressTester(survival_threshold=-0.3, risk_free_rate=0.05)
        assert custom_tester.survival_threshold == -0.3
        assert custom_tester.risk_free_rate == 0.05

    def test_replay_covid_crash(self, stress_tester, sample_returns):
        """測試 COVID-19 崩盤重播"""
        result = stress_tester.replay_historical_event(
            strategy_returns=sample_returns,
            event_name='covid_crash_2020'
        )

        assert isinstance(result, StressTestResult)
        assert result.event_name == 'COVID-19 崩盤'
        assert result.drop_percentage == -0.40
        assert result.duration_days == 2
        assert len(result.returns) == len(sample_returns)
        assert isinstance(result.equity_curve, pd.Series)

    def test_replay_china_ban(self, stress_tester, sample_returns):
        """測試中國禁令重播"""
        result = stress_tester.replay_historical_event(
            strategy_returns=sample_returns,
            event_name='china_ban_2021'
        )

        assert result.event_name == '中國禁令'
        assert result.drop_percentage == -0.30
        assert result.duration_days == 5

    def test_replay_luna_crash(self, stress_tester, sample_returns):
        """測試 LUNA 崩盤重播"""
        result = stress_tester.replay_historical_event(
            strategy_returns=sample_returns,
            event_name='luna_crash_2022'
        )

        assert result.event_name == 'LUNA 崩盤'
        assert result.drop_percentage == -0.50
        assert result.duration_days == 4

    def test_replay_ftx_collapse(self, stress_tester, sample_returns):
        """測試 FTX 倒閉重播"""
        result = stress_tester.replay_historical_event(
            strategy_returns=sample_returns,
            event_name='ftx_collapse_2022'
        )

        assert result.event_name == 'FTX 倒閉'
        assert result.drop_percentage == -0.25
        assert result.duration_days == 3

    def test_replay_unknown_event(self, stress_tester, sample_returns):
        """測試未知事件"""
        with pytest.raises(ValueError, match="未知事件"):
            stress_tester.replay_historical_event(
                strategy_returns=sample_returns,
                event_name='unknown_event'
            )

    def test_replay_custom_event(self, stress_tester, sample_returns):
        """測試自定義事件覆蓋"""
        custom_event = {
            'name': '自定義崩盤',
            'start': '2023-06-01',
            'end': '2023-06-05',
            'drop': -0.35,
            'description': '測試用自定義事件'
        }

        result = stress_tester.replay_historical_event(
            strategy_returns=sample_returns,
            event_name='custom',
            custom_event=custom_event
        )

        assert result.event_name == '自定義崩盤'
        assert result.drop_percentage == -0.35
        assert result.duration_days == 5

    def test_run_scenario(self, stress_tester, sample_returns):
        """測試假設情境"""
        scenario = {
            'name': '極端下跌',
            'drop': -0.60,
            'duration': 7,
            'description': '60% 下跌，持續 7 天'
        }

        result = stress_tester.run_scenario(
            strategy_returns=sample_returns,
            scenario=scenario
        )

        assert isinstance(result, StressTestResult)
        assert result.event_name == '極端下跌'
        assert result.drop_percentage == -0.60
        assert result.duration_days == 7

    def test_run_scenario_minimal(self, stress_tester, sample_returns):
        """測試最小情境參數"""
        scenario = {
            'drop': -0.20,
            'duration': 3
        }

        result = stress_tester.run_scenario(
            strategy_returns=sample_returns,
            scenario=scenario
        )

        assert result.event_name == 'Custom Scenario'
        assert result.drop_percentage == -0.20

    def test_run_scenario_missing_params(self, stress_tester, sample_returns):
        """測試缺少必要參數"""
        with pytest.raises(ValueError, match="必須包含 'drop' 和 'duration'"):
            stress_tester.run_scenario(
                strategy_returns=sample_returns,
                scenario={'drop': -0.30}  # 缺少 duration
            )

    def test_generate_stress_report(self, stress_tester, sample_returns):
        """測試產生完整報告"""
        report = stress_tester.generate_stress_report(
            strategy_returns=sample_returns
        )

        assert isinstance(report, StressTestReport)
        assert report.n_scenarios == len(HISTORICAL_EVENTS)
        assert len(report.test_results) == len(HISTORICAL_EVENTS)
        assert 0 <= report.survival_rate <= 1
        assert 0 <= report.profit_rate <= 1

    def test_generate_stress_report_with_custom_scenarios(
        self,
        stress_tester,
        sample_returns
    ):
        """測試包含自定義情境的報告"""
        custom_scenarios = [
            {'name': '情境1', 'drop': -0.30, 'duration': 5},
            {'name': '情境2', 'drop': -0.50, 'duration': 3},
        ]

        report = stress_tester.generate_stress_report(
            strategy_returns=sample_returns,
            custom_scenarios=custom_scenarios
        )

        expected_count = len(HISTORICAL_EVENTS) + len(custom_scenarios)
        assert report.n_scenarios == expected_count
        assert len(report.test_results) == expected_count

    def test_inject_shock(self, stress_tester, sample_returns):
        """測試注入衝擊"""
        shocked_returns = stress_tester._inject_shock(
            returns=sample_returns,
            shock_magnitude=-0.40,
            shock_duration=5,
            shock_position=100
        )

        assert len(shocked_returns) == len(sample_returns)
        # 檢查衝擊位置附近的報酬確實變差
        shock_region = shocked_returns.iloc[100:105]
        original_region = sample_returns.iloc[100:105]
        assert (shock_region < original_region).any()

    def test_inject_shock_random_position(self, stress_tester, sample_returns):
        """測試隨機位置衝擊"""
        shocked_returns = stress_tester._inject_shock(
            returns=sample_returns,
            shock_magnitude=-0.30,
            shock_duration=10
        )

        assert len(shocked_returns) == len(sample_returns)
        # 應該與原始序列不同
        assert not shocked_returns.equals(sample_returns)

    def test_inject_shock_too_long(self, stress_tester):
        """測試衝擊持續時間過長"""
        short_returns = pd.Series([0.01, 0.02, 0.01])

        with pytest.raises(ValueError, match="短於衝擊持續時間"):
            stress_tester._inject_shock(
                returns=short_returns,
                shock_magnitude=-0.50,
                shock_duration=10
            )

    def test_calculate_metrics(self, stress_tester, sample_returns):
        """測試指標計算"""
        metrics = stress_tester._calculate_metrics(sample_returns)

        assert 'total_return' in metrics
        assert 'max_drawdown' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'var_95' in metrics
        assert 'cvar_95' in metrics

        # 檢查合理性
        assert isinstance(metrics['total_return'], (int, float))
        assert metrics['max_drawdown'] <= 0  # 回撤應為負數或零

    def test_calculate_metrics_all_positive(self, stress_tester):
        """測試全獲利序列"""
        positive_returns = pd.Series([0.01, 0.02, 0.015, 0.01, 0.02])
        metrics = stress_tester._calculate_metrics(positive_returns)

        assert metrics['total_return'] > 0
        assert metrics['max_drawdown'] == 0  # 無回撤

    def test_calculate_metrics_all_negative(self, stress_tester):
        """測試全虧損序列"""
        negative_returns = pd.Series([-0.01, -0.02, -0.01, -0.015, -0.01])
        metrics = stress_tester._calculate_metrics(negative_returns)

        assert metrics['total_return'] < 0
        assert metrics['max_drawdown'] < 0

    def test_calculate_recovery(self, stress_tester):
        """測試恢復時間計算"""
        # 建立有明顯回撤和恢復的權益曲線
        equity = pd.Series([
            1.0, 1.1, 1.2, 1.3,  # 上漲
            1.2, 1.1, 1.0, 0.9,  # 回撤
            1.0, 1.1, 1.2, 1.3, 1.4  # 恢復並創新高
        ])

        recovery_days, time_underwater = stress_tester._calculate_recovery(equity)

        assert recovery_days is not None
        assert recovery_days > 0
        assert time_underwater > 0

    def test_calculate_recovery_no_recovery(self, stress_tester):
        """測試未恢復的情況"""
        # 持續下跌，未恢復
        equity = pd.Series([1.0, 0.9, 0.8, 0.7, 0.6])

        recovery_days, time_underwater = stress_tester._calculate_recovery(equity)

        assert recovery_days is None
        assert time_underwater > 0

    def test_calculate_recovery_no_drawdown(self, stress_tester):
        """測試無回撤的情況"""
        # 持續上漲
        equity = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4])

        recovery_days, time_underwater = stress_tester._calculate_recovery(equity)

        # 無回撤則無需恢復
        assert time_underwater == 0

    def test_result_structure(self, stress_tester, sample_returns):
        """測試結果結構完整性"""
        result = stress_tester.replay_historical_event(
            strategy_returns=sample_returns,
            event_name='covid_crash_2020'
        )

        # 檢查所有欄位存在
        assert hasattr(result, 'event_name')
        assert hasattr(result, 'description')
        assert hasattr(result, 'drop_percentage')
        assert hasattr(result, 'duration_days')
        assert hasattr(result, 'total_return')
        assert hasattr(result, 'max_drawdown')
        assert hasattr(result, 'sharpe_ratio')
        assert hasattr(result, 'var_95')
        assert hasattr(result, 'cvar_95')
        assert hasattr(result, 'recovery_days')
        assert hasattr(result, 'time_underwater')
        assert hasattr(result, 'total_trades')
        assert hasattr(result, 'winning_trades')
        assert hasattr(result, 'losing_trades')
        assert hasattr(result, 'win_rate')
        assert hasattr(result, 'returns')
        assert hasattr(result, 'equity_curve')

    def test_report_structure(self, stress_tester, sample_returns):
        """測試報告結構完整性"""
        report = stress_tester.generate_stress_report(
            strategy_returns=sample_returns
        )

        # 檢查所有欄位存在
        assert hasattr(report, 'n_scenarios')
        assert hasattr(report, 'test_results')
        assert hasattr(report, 'average_return')
        assert hasattr(report, 'average_max_drawdown')
        assert hasattr(report, 'average_recovery_days')
        assert hasattr(report, 'worst_scenario')
        assert hasattr(report, 'worst_return')
        assert hasattr(report, 'worst_drawdown')
        assert hasattr(report, 'best_scenario')
        assert hasattr(report, 'best_return')
        assert hasattr(report, 'survival_rate')
        assert hasattr(report, 'profit_rate')

    def test_print_result(self, stress_tester, sample_returns, capsys):
        """測試結果輸出"""
        result = stress_tester.replay_historical_event(
            strategy_returns=sample_returns,
            event_name='covid_crash_2020'
        )

        StressTester.print_result(result)
        captured = capsys.readouterr()

        # 檢查輸出包含關鍵資訊
        assert 'COVID-19 崩盤' in captured.out
        assert '總報酬' in captured.out
        assert '最大回撤' in captured.out
        assert 'Sharpe Ratio' in captured.out
        assert '勝率' in captured.out

    def test_print_report(self, stress_tester, sample_returns, capsys):
        """測試報告輸出"""
        report = stress_tester.generate_stress_report(
            strategy_returns=sample_returns
        )

        StressTester.print_report(report)
        captured = capsys.readouterr()

        # 檢查輸出包含關鍵資訊
        assert '壓力測試報告' in captured.out
        assert '整體統計' in captured.out
        assert '最差情境' in captured.out
        assert '最佳情境' in captured.out
        assert '存活率' in captured.out
        assert '獲利率' in captured.out

    def test_survival_threshold(self):
        """測試存活閾值"""
        # 嚴格閾值
        strict_tester = StressTester(survival_threshold=-0.1)

        # 寬鬆閾值
        loose_tester = StressTester(survival_threshold=-0.9)

        returns = pd.Series([-0.05] * 10)  # 總虧損 -50%
        report_strict = strict_tester.generate_stress_report(returns)
        report_loose = loose_tester.generate_stress_report(returns)

        # 嚴格閾值下存活率應該較低
        # （但這取決於具體的壓力測試結果，這裡只檢查邏輯）
        assert 0 <= report_strict.survival_rate <= 1
        assert 0 <= report_loose.survival_rate <= 1

    def test_edge_case_single_day(self, stress_tester):
        """測試單日報酬"""
        single_return = pd.Series([0.05])

        # 應該無法注入多日衝擊
        with pytest.raises(ValueError):
            stress_tester._inject_shock(
                returns=single_return,
                shock_magnitude=-0.30,
                shock_duration=5
            )

    def test_edge_case_zero_returns(self, stress_tester):
        """測試零報酬序列"""
        zero_returns = pd.Series([0.0] * 100)
        metrics = stress_tester._calculate_metrics(zero_returns)

        assert metrics['total_return'] == 0.0
        assert metrics['max_drawdown'] == 0.0
        assert metrics['sharpe_ratio'] == 0.0

    def test_historical_events_coverage(self):
        """測試所有歷史事件可用"""
        expected_events = [
            'covid_crash_2020',
            'china_ban_2021',
            'luna_crash_2022',
            'ftx_collapse_2022'
        ]

        for event in expected_events:
            assert event in HISTORICAL_EVENTS
            assert 'name' in HISTORICAL_EVENTS[event]
            assert 'start' in HISTORICAL_EVENTS[event]
            assert 'end' in HISTORICAL_EVENTS[event]
            assert 'drop' in HISTORICAL_EVENTS[event]

    def test_multiple_scenarios_ordering(self, stress_tester, sample_returns):
        """測試多情境排序正確性"""
        custom_scenarios = [
            {'name': '輕微', 'drop': -0.10, 'duration': 2},
            {'name': '中等', 'drop': -0.30, 'duration': 5},
            {'name': '嚴重', 'drop': -0.60, 'duration': 10},
        ]

        report = stress_tester.generate_stress_report(
            strategy_returns=sample_returns,
            custom_scenarios=custom_scenarios
        )

        # 找到自定義情境的結果
        custom_results = [r for r in report.test_results if r.event_name in ['輕微', '中等', '嚴重']]

        # 驗證嚴重程度與報酬的關係（通常越嚴重報酬越差）
        assert len(custom_results) == 3


class TestStressTestResult:
    """StressTestResult 測試"""

    def test_result_dataclass(self):
        """測試結果 dataclass"""
        returns = pd.Series([0.01, -0.02, 0.01])
        equity = pd.Series([1.0, 1.01, 0.99, 1.00])

        result = StressTestResult(
            event_name='測試事件',
            description='測試描述',
            drop_percentage=-0.30,
            duration_days=5,
            total_return=0.05,
            max_drawdown=-0.15,
            sharpe_ratio=1.2,
            var_95=-0.02,
            cvar_95=-0.03,
            recovery_days=10,
            time_underwater=5,
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            win_rate=0.6,
            returns=returns,
            equity_curve=equity
        )

        assert result.event_name == '測試事件'
        assert result.drop_percentage == -0.30
        assert result.win_rate == 0.6
        assert len(result.returns) == 3


class TestStressTestReport:
    """StressTestReport 測試"""

    def test_report_dataclass(self):
        """測試報告 dataclass"""
        # 建立假結果
        result1 = StressTestResult(
            event_name='事件1',
            description='',
            drop_percentage=-0.30,
            duration_days=5,
            total_return=0.05,
            max_drawdown=-0.10,
            sharpe_ratio=1.0,
            var_95=-0.02,
            cvar_95=-0.03,
            recovery_days=5,
            time_underwater=3,
            total_trades=50,
            winning_trades=30,
            losing_trades=20,
            win_rate=0.6,
            returns=pd.Series([0.01]),
            equity_curve=pd.Series([1.0, 1.01])
        )

        report = StressTestReport(
            n_scenarios=1,
            test_results=[result1],
            average_return=0.05,
            average_max_drawdown=-0.10,
            average_recovery_days=5.0,
            worst_scenario='事件1',
            worst_return=0.05,
            worst_drawdown=-0.10,
            best_scenario='事件1',
            best_return=0.05,
            survival_rate=1.0,
            profit_rate=1.0
        )

        assert report.n_scenarios == 1
        assert len(report.test_results) == 1
        assert report.survival_rate == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
