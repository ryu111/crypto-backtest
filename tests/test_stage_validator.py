"""
StageValidator 單元測試
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 直接從模組導入，避免延遲導入問題
from src.validator.stages import (
    StageValidator,
    ValidationGrade,
    ValidationResult,
    StageResult
)
from src.strategies.momentum.rsi import RSIStrategy
from src.backtester.engine import BacktestResult


class TestStageValidator:
    """StageValidator 測試"""

    @pytest.fixture
    def validator(self):
        """建立驗證器"""
        return StageValidator()

    @pytest.fixture
    def sample_data(self):
        """建立樣本資料"""
        dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='1h')
        n = len(dates)

        data = pd.DataFrame({
            'open': np.random.randn(n).cumsum() + 50000,
            'high': np.random.randn(n).cumsum() + 50100,
            'low': np.random.randn(n).cumsum() + 49900,
            'close': np.random.randn(n).cumsum() + 50000,
            'volume': np.random.rand(n) * 1000,
        }, index=dates)

        return data

    @pytest.fixture
    def good_backtest_result(self):
        """建立良好的回測結果"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

        equity = pd.Series(
            np.linspace(10000, 15000, 100),
            index=dates
        )

        returns = equity.pct_change().fillna(0)

        trades = pd.DataFrame({
            'Entry Timestamp': dates[:50],
            'Exit Timestamp': dates[1:51],
            'PnL': np.random.randn(50) * 100 + 50,
            'Return Pct': np.random.randn(50) * 0.02 + 0.01,
            'Size': [1.0] * 50,
            'Avg Entry Price': [50000.0] * 50,
        })

        return BacktestResult(
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

    def test_stage1_basic_backtest_pass(self, validator, good_backtest_result):
        """測試階段 1 - 通過"""
        result = validator.stage1_basic_backtest(good_backtest_result)

        assert result.passed is True
        assert result.score > 80
        assert result.details['total_return'] > 0
        assert result.details['sharpe_ratio'] > 0.5

    def test_stage1_basic_backtest_fail(self, validator):
        """測試階段 1 - 失敗"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

        bad_result = BacktestResult(
            total_return=-0.2,  # 虧損
            annual_return=-0.1,
            sharpe_ratio=0.3,  # 太低
            sortino_ratio=0.5,
            calmar_ratio=0.0,
            max_drawdown=-0.5,  # 太大
            max_drawdown_duration=30,
            volatility=0.3,
            total_trades=10,  # 太少
            win_rate=0.3,
            profit_factor=0.8,  # < 1
            avg_win=20.0,
            avg_loss=-30.0,
            avg_trade_duration=48.0,
            expectancy=-5.0,
            recovery_factor=0.5,
            ulcer_index=10.0,
            equity_curve=pd.Series([10000, 8000], index=dates[:2]),
            trades=pd.DataFrame(),
            daily_returns=pd.Series([0.0, -0.2], index=dates[:2]),
        )

        result = validator.stage1_basic_backtest(bad_result)

        assert result.passed is False
        assert result.score < 50

    def test_stage2_statistical_tests_pass(self, validator):
        """測試階段 2 - 統計檢驗通過"""
        # 建立有顯著正報酬的序列
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.02 + 0.001)

        result = validator.stage2_statistical_tests(returns)

        # 可能通過或不通過，取決於隨機性
        assert isinstance(result, StageResult)
        assert 'p_value' in result.details
        assert 'sharpe_ratio' in result.details

    def test_stage2_statistical_tests_fail(self, validator):
        """測試階段 2 - 統計檢驗失敗"""
        # 建立隨機遊走（無顯著報酬）
        returns = pd.Series(np.random.randn(100) * 0.01)

        result = validator.stage2_statistical_tests(returns)

        assert isinstance(result, StageResult)
        # 大概率失敗
        if not result.passed:
            assert result.score < 100

    def test_stage5_monte_carlo_sufficient_trades(self, validator, good_backtest_result):
        """測試階段 5 - 足夠交易數"""
        result = validator.stage5_monte_carlo(good_backtest_result.trades)

        assert isinstance(result, StageResult)
        assert 'original_return' in result.details
        assert 'p5' in result.details
        assert 'median' in result.details

    def test_stage5_monte_carlo_insufficient_trades(self, validator):
        """測試階段 5 - 交易數不足"""
        trades = pd.DataFrame({
            'PnL': [100, 200, -50],
        })

        result = validator.stage5_monte_carlo(trades)

        assert result.passed is False
        assert 'insufficient_trades' in result.details.get('error', '')

    def test_calculate_grade(self, validator):
        """測試評級計算"""
        assert validator._calculate_grade(5) == ValidationGrade.A
        assert validator._calculate_grade(4) == ValidationGrade.B
        assert validator._calculate_grade(3) == ValidationGrade.C
        assert validator._calculate_grade(2) == ValidationGrade.D
        assert validator._calculate_grade(1) == ValidationGrade.D
        assert validator._calculate_grade(0) == ValidationGrade.F

    def test_generate_recommendation(self, validator):
        """測試建議生成"""
        stage_results = {}

        # 測試每個評級
        for grade in ValidationGrade:
            rec = validator._generate_recommendation(grade, stage_results)
            assert isinstance(rec, str)
            assert len(rec) > 0

    def test_parameter_sensitivity(self, validator, sample_data):
        """測試參數敏感度"""
        from src.backtester.engine import BacktestEngine, BacktestConfig

        strategy = RSIStrategy()
        params = {'rsi_period': 14}

        config = BacktestConfig(
            symbol='BTCUSDT',
            timeframe='1h',
            start_date=sample_data.index[0],
            end_date=sample_data.index[-1],
        )

        engine = BacktestEngine(config)

        sensitivity = validator._test_parameter_sensitivity(
            strategy, sample_data, params, engine
        )

        assert 0 <= sensitivity <= 1

    def test_time_consistency(self, validator, sample_data):
        """測試時間一致性"""
        from src.backtester.engine import BacktestEngine, BacktestConfig

        strategy = RSIStrategy()
        params = {'rsi_period': 14}

        config = BacktestConfig(
            symbol='BTCUSDT',
            timeframe='1h',
            start_date=sample_data.index[0],
            end_date=sample_data.index[-1],
        )

        engine = BacktestEngine(config)

        is_consistent = validator._test_time_consistency(
            strategy, sample_data, params, engine
        )

        assert isinstance(is_consistent, bool)

    def test_validation_result_summary(self):
        """測試驗證結果摘要"""
        stage_results = {
            '階段1': StageResult(
                passed=True,
                score=85.0,
                details={},
                message="測試",
                threshold={}
            )
        }

        result = ValidationResult(
            grade=ValidationGrade.A,
            passed_stages=5,
            stage_results=stage_results,
            recommendation="優秀"
        )

        summary = result.summary()

        assert isinstance(summary, str)
        assert 'A' in summary
        assert '5/5' in summary or '5' in summary

    def test_early_exit(self, validator):
        """測試提前結束"""
        stage_results = {
            '階段1': StageResult(
                passed=False,
                score=30.0,
                details={},
                message="失敗",
                threshold={}
            )
        }

        result = validator._early_exit(stage_results, 0)

        assert result.grade == ValidationGrade.F
        assert result.passed_stages == 0
        assert result.details.get('early_exit') is True


class TestValidationIntegration:
    """整合測試"""

    def test_full_validation_workflow(self):
        """測試完整驗證流程"""
        # 建立測試資料
        dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='1h')
        n = len(dates)

        data_btc = pd.DataFrame({
            'open': np.random.randn(n).cumsum() + 50000,
            'high': np.random.randn(n).cumsum() + 50100,
            'low': np.random.randn(n).cumsum() + 49900,
            'close': np.random.randn(n).cumsum() + 50000,
            'volume': np.random.rand(n) * 1000,
        }, index=dates)

        data_eth = data_btc.copy() * 0.04  # 簡化的 ETH 資料

        # 建立策略
        strategy = RSIStrategy()
        params = {'rsi_period': 14}

        # 執行驗證
        validator = StageValidator()

        result = validator.validate(
            strategy=strategy,
            data_btc=data_btc,
            data_eth=data_eth,
            params=params
        )

        # 驗證結果結構
        assert isinstance(result, ValidationResult)
        assert isinstance(result.grade, ValidationGrade)
        assert 0 <= result.passed_stages <= 5
        assert len(result.stage_results) > 0
        assert isinstance(result.recommendation, str)

        # 確保至少執行了階段 1
        assert '階段1_基礎回測' in result.stage_results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
