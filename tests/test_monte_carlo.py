"""
Monte Carlo 模擬器測試
"""

import pytest
import numpy as np
import pandas as pd
from src.validator import MonteCarloSimulator, MonteCarloResult


@pytest.fixture
def sample_trades():
    """建立測試用交易記錄"""
    np.random.seed(42)
    pnl = np.random.normal(loc=10, scale=50, size=100)
    return pd.DataFrame({'pnl': pnl})


@pytest.fixture
def simulator():
    """建立模擬器實例"""
    return MonteCarloSimulator(seed=42)


class TestMonteCarloSimulator:
    """Monte Carlo 模擬器測試"""

    def test_init(self):
        """測試初始化"""
        sim = MonteCarloSimulator()
        assert sim.seed is None

        sim_with_seed = MonteCarloSimulator(seed=42)
        assert sim_with_seed.seed == 42

    def test_shuffle_simulation(self, simulator, sample_trades):
        """測試交易順序隨機化模擬"""
        result = simulator.simulate(
            trades=sample_trades,
            n_simulations=1000,
            method='shuffle'
        )

        assert isinstance(result, MonteCarloResult)
        assert result.n_simulations == 1000
        assert result.method == 'shuffle'
        assert len(result.simulated_returns) == 1000

        # 驗證總報酬一致性（shuffle 不改變總和）
        original_sum = sample_trades['pnl'].sum()
        assert all(
            abs(r - original_sum) < 1e-10
            for r in result.simulated_returns
        )

    def test_bootstrap_simulation(self, simulator, sample_trades):
        """測試 Bootstrap 模擬"""
        result = simulator.simulate(
            trades=sample_trades,
            n_simulations=1000,
            method='bootstrap'
        )

        assert isinstance(result, MonteCarloResult)
        assert result.n_simulations == 1000
        assert result.method == 'bootstrap'
        assert len(result.simulated_returns) == 1000

    def test_block_bootstrap_simulation(self, simulator, sample_trades):
        """測試區塊 Bootstrap 模擬"""
        result = simulator.simulate(
            trades=sample_trades,
            n_simulations=1000,
            method='block_bootstrap',
            block_size=5
        )

        assert isinstance(result, MonteCarloResult)
        assert result.n_simulations == 1000
        assert result.method == 'block_bootstrap'
        assert len(result.simulated_returns) == 1000

    def test_invalid_method(self, simulator, sample_trades):
        """測試無效方法"""
        with pytest.raises(ValueError, match="不支援的模擬方法"):
            simulator.simulate(
                trades=sample_trades,
                n_simulations=100,
                method='invalid_method'
            )

    def test_empty_trades(self, simulator):
        """測試空交易記錄"""
        empty_trades = pd.DataFrame({'pnl': []})
        with pytest.raises(ValueError, match="交易記錄為空"):
            simulator.simulate(empty_trades, n_simulations=100)

    def test_missing_pnl_column(self, simulator):
        """測試缺少 pnl 欄位"""
        invalid_trades = pd.DataFrame({'price': [100, 110, 105]})
        with pytest.raises(ValueError, match="必須包含 'pnl' 欄位"):
            simulator.simulate(invalid_trades, n_simulations=100)

    def test_result_statistics(self, simulator, sample_trades):
        """測試結果統計"""
        result = simulator.simulate(
            trades=sample_trades,
            n_simulations=1000,
            method='bootstrap'
        )

        # 檢查統計值合理性
        assert result.mean is not None
        assert result.std > 0
        assert result.median is not None

        # 檢查百分位數順序
        assert result.percentile_1 <= result.percentile_5
        assert result.percentile_5 <= result.percentile_25
        assert result.percentile_25 <= result.percentile_75
        assert result.percentile_75 <= result.percentile_95
        assert result.percentile_95 <= result.percentile_99

    def test_var_cvar(self, simulator, sample_trades):
        """測試 VaR 和 CVaR"""
        result = simulator.simulate(
            trades=sample_trades,
            n_simulations=1000,
            method='bootstrap'
        )

        # CVaR 應該小於等於 VaR（都是負值時）
        assert result.cvar_95 <= result.var_95

    def test_probabilities(self, simulator, sample_trades):
        """測試機率計算"""
        result = simulator.simulate(
            trades=sample_trades,
            n_simulations=1000,
            method='bootstrap'
        )

        # 機率應在 0-1 之間
        assert 0 <= result.probability_profitable <= 1
        assert 0 <= result.probability_beat_original <= 1

    def test_calculate_statistics(self, simulator):
        """測試統計計算"""
        simulated_returns = np.array([100, 200, 150, 180, 120])
        stats = simulator.calculate_statistics(simulated_returns)

        assert stats['mean'] == 150.0
        assert stats['median'] == 150.0
        assert 'percentile_5' in stats
        assert 'percentile_95' in stats

    def test_calculate_var(self, simulator):
        """測試 VaR 計算"""
        returns = np.array([-10, -5, 0, 5, 10])
        var_95 = simulator.calculate_var(returns, confidence=0.95)

        # 95% VaR 應該在最小值附近
        assert var_95 <= -5

    def test_calculate_cvar(self, simulator):
        """測試 CVaR 計算"""
        returns = np.array([-10, -8, -5, 0, 5, 10])
        cvar_95 = simulator.calculate_cvar(returns, confidence=0.95)

        # CVaR 應該是尾部的平均
        var_95 = simulator.calculate_var(returns, confidence=0.95)
        assert cvar_95 <= var_95

    def test_generate_equity_paths(self, simulator, sample_trades):
        """測試權益曲線路徑產生"""
        equity_paths, original_path = simulator.generate_equity_paths(
            trades=sample_trades,
            n_simulations=100,
            method='bootstrap'
        )

        # 檢查形狀
        assert equity_paths.shape[0] == 100
        assert equity_paths.shape[1] == len(sample_trades) + 1
        assert len(original_path) == len(sample_trades) + 1

        # 檢查起始點
        assert all(equity_paths[:, 0] == 0)
        assert original_path[0] == 0

        # 檢查最終值
        assert original_path[-1] == pytest.approx(sample_trades['pnl'].sum())

    def test_reproducibility(self):
        """測試結果可重現性"""
        trades = pd.DataFrame({'pnl': np.random.randn(50)})

        sim1 = MonteCarloSimulator(seed=123)
        result1 = sim1.simulate(trades, n_simulations=100, method='bootstrap')

        sim2 = MonteCarloSimulator(seed=123)
        result2 = sim2.simulate(trades, n_simulations=100, method='bootstrap')

        # 相同種子應產生相同結果
        np.testing.assert_array_almost_equal(
            result1.simulated_returns,
            result2.simulated_returns
        )

    def test_print_result(self, simulator, sample_trades, capsys):
        """測試結果輸出"""
        result = simulator.simulate(
            trades=sample_trades,
            n_simulations=100,
            method='shuffle'
        )

        simulator.print_result(result)
        captured = capsys.readouterr()

        # 檢查輸出包含關鍵資訊
        assert 'Monte Carlo 模擬結果' in captured.out
        assert '模擬次數' in captured.out
        assert '平均報酬' in captured.out
        assert 'VaR' in captured.out

    def test_small_block_size(self, simulator, sample_trades):
        """測試小區塊大小"""
        # 當交易數少於區塊大小時，應降級為普通 bootstrap
        small_trades = sample_trades.head(3)

        result = simulator.simulate(
            trades=small_trades,
            n_simulations=100,
            method='block_bootstrap',
            block_size=10
        )

        assert isinstance(result, MonteCarloResult)
        assert len(result.simulated_returns) == 100

    def test_single_trade(self, simulator):
        """測試單筆交易"""
        single_trade = pd.DataFrame({'pnl': [100]})

        result = simulator.simulate(
            trades=single_trade,
            n_simulations=100,
            method='shuffle'
        )

        # 所有模擬結果應該相同
        assert all(r == 100 for r in result.simulated_returns)
        assert result.std == 0

    def test_all_positive_trades(self, simulator):
        """測試全獲利交易"""
        positive_trades = pd.DataFrame({'pnl': [10, 20, 30, 40, 50]})

        result = simulator.simulate(
            trades=positive_trades,
            n_simulations=1000,
            method='bootstrap'
        )

        # 獲利機率應該是 100%
        assert result.probability_profitable == 1.0
        assert result.mean > 0

    def test_all_negative_trades(self, simulator):
        """測試全虧損交易"""
        negative_trades = pd.DataFrame({'pnl': [-10, -20, -30, -40, -50]})

        result = simulator.simulate(
            trades=negative_trades,
            n_simulations=1000,
            method='bootstrap'
        )

        # 獲利機率應該是 0%
        assert result.probability_profitable == 0.0
        assert result.mean < 0


class TestMonteCarloResult:
    """MonteCarloResult 測試"""

    def test_result_dataclass(self):
        """測試結果 dataclass"""
        simulated_returns = np.array([100, 200, 300])

        result = MonteCarloResult(
            n_simulations=3,
            method='test',
            mean=200.0,
            std=100.0,
            median=200.0,
            percentile_1=100.0,
            percentile_5=110.0,
            percentile_25=150.0,
            percentile_75=250.0,
            percentile_95=290.0,
            percentile_99=300.0,
            var_95=110.0,
            cvar_95=100.0,
            original_return=200.0,
            probability_profitable=0.8,
            probability_beat_original=0.5,
            simulated_returns=simulated_returns
        )

        assert result.n_simulations == 3
        assert result.method == 'test'
        assert result.mean == 200.0
        assert len(result.simulated_returns) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
