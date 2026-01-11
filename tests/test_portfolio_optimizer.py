"""
測試策略組合優化器

驗證各種優化方法的正確性和穩定性。
"""

import pytest
import pandas as pd
import numpy as np

from src.optimizer.portfolio import PortfolioOptimizer, PortfolioWeights


@pytest.fixture
def sample_returns():
    """建立樣本策略回報資料"""
    np.random.seed(42)

    # 模擬 3 個策略的日回報（252 天）
    n_days = 252
    returns_data = {
        'strategy_a': np.random.normal(0.0005, 0.01, n_days),  # 低報酬低波動
        'strategy_b': np.random.normal(0.001, 0.015, n_days),  # 中報酬中波動
        'strategy_c': np.random.normal(0.0015, 0.02, n_days)   # 高報酬高波動
    }

    # 添加一些相關性
    df = pd.DataFrame(returns_data)
    df['strategy_b'] = df['strategy_b'] + 0.3 * df['strategy_a']
    df['strategy_c'] = df['strategy_c'] + 0.2 * df['strategy_a']

    return df


@pytest.fixture
def optimizer(sample_returns):
    """建立優化器實例"""
    return PortfolioOptimizer(
        returns=sample_returns,
        risk_free_rate=0.0,
        frequency=252,
        use_ledoit_wolf=False  # 測試中使用標準協方差
    )


class TestPortfolioOptimizer:
    """組合優化器測試"""

    def test_initialization(self, optimizer, sample_returns):
        """測試初始化"""
        assert optimizer.n_assets == 3
        assert len(optimizer.strategy_names) == 3
        assert optimizer.frequency == 252
        assert optimizer.risk_free_rate == 0.0

        # 檢查協方差矩陣形狀
        assert optimizer.cov_matrix.shape == (3, 3)

        # 檢查平均回報形狀
        assert len(optimizer.mean_returns) == 3

    def test_equal_weight_portfolio(self, optimizer):
        """測試等權重組合"""
        result = optimizer.equal_weight_portfolio()

        assert isinstance(result, PortfolioWeights)
        assert result.optimization_success is True

        # 檢查權重總和為 1
        weights_sum = sum(result.weights.values())
        assert abs(weights_sum - 1.0) < 1e-6

        # 檢查每個權重約為 1/3
        for weight in result.weights.values():
            assert abs(weight - 1.0/3.0) < 1e-6

    def test_inverse_volatility_portfolio(self, optimizer):
        """測試反波動率加權組合"""
        result = optimizer.inverse_volatility_portfolio()

        assert isinstance(result, PortfolioWeights)
        assert result.optimization_success is True

        # 檢查權重總和為 1
        weights_sum = sum(result.weights.values())
        assert abs(weights_sum - 1.0) < 1e-6

        # 檢查權重都是正數
        for weight in result.weights.values():
            assert weight > 0

        # 策略 A（低波動）應該權重最高
        assert result.weights['strategy_a'] > result.weights['strategy_b']
        assert result.weights['strategy_a'] > result.weights['strategy_c']

    def test_max_sharpe_optimize(self, optimizer):
        """測試最大化 Sharpe Ratio 優化"""
        result = optimizer.max_sharpe_optimize()

        assert isinstance(result, PortfolioWeights)

        # 檢查權重總和為 1
        weights_sum = sum(result.weights.values())
        assert abs(weights_sum - 1.0) < 1e-4

        # 檢查權重在 [0, 1] 範圍內（不允許空頭）
        for weight in result.weights.values():
            assert 0.0 <= weight <= 1.0

        # 檢查 Sharpe Ratio 是有限值
        assert np.isfinite(result.sharpe_ratio)

    def test_max_sharpe_with_constraints(self, optimizer):
        """測試帶約束的最大化 Sharpe Ratio"""
        result = optimizer.max_sharpe_optimize(
            max_weight=0.5,  # 每個策略最多 50%
            min_weight=0.1   # 每個策略至少 10%
        )

        # 檢查所有權重在約束範圍內
        for weight in result.weights.values():
            assert 0.1 <= weight <= 0.5

        # 檢查權重總和為 1
        weights_sum = sum(result.weights.values())
        assert abs(weights_sum - 1.0) < 1e-4

    def test_mean_variance_optimize_min_variance(self, optimizer):
        """測試最小變異數組合"""
        result = optimizer.mean_variance_optimize()

        assert isinstance(result, PortfolioWeights)

        # 最小變異數組合應該有較低的波動率
        equal_weight = optimizer.equal_weight_portfolio()
        assert result.expected_volatility <= equal_weight.expected_volatility

    def test_mean_variance_optimize_target_return(self, optimizer):
        """測試目標報酬優化"""
        target_return = 0.15  # 15% 年化報酬

        result = optimizer.mean_variance_optimize(
            target_return=target_return
        )

        # 檢查實際報酬接近目標
        assert abs(result.expected_return - target_return) < 0.01

    def test_risk_parity_optimize(self, optimizer):
        """測試風險平價優化"""
        result = optimizer.risk_parity_optimize()

        assert isinstance(result, PortfolioWeights)

        # 檢查權重總和為 1
        weights_sum = sum(result.weights.values())
        assert abs(weights_sum - 1.0) < 1e-4

        # 檢查權重都是正數
        for weight in result.weights.values():
            assert weight > 0

        # 風險平價應該讓低波動策略權重較高
        # 但不像反波動率那麼極端
        assert result.weights['strategy_a'] > 0.2

    def test_efficient_frontier(self, optimizer):
        """測試效率前緣計算"""
        frontier = optimizer.efficient_frontier(n_points=10)

        # 應該有接近 10 個點（可能有些優化失敗）
        assert len(frontier) >= 5

        # 檢查前緣按風險排序
        volatilities = [p.expected_volatility for p in frontier]
        assert volatilities == sorted(volatilities)

        # 檢查所有點的權重總和為 1
        for portfolio in frontier:
            weights_sum = sum(portfolio.weights.values())
            assert abs(weights_sum - 1.0) < 1e-4

    def test_portfolio_weights_to_dict(self, optimizer):
        """測試 PortfolioWeights 轉字典"""
        result = optimizer.equal_weight_portfolio()
        data = result.to_dict()

        assert 'weights' in data
        assert 'expected_return' in data
        assert 'expected_volatility' in data
        assert 'sharpe_ratio' in data
        assert 'optimization_success' in data

    def test_portfolio_weights_summary(self, optimizer):
        """測試 PortfolioWeights 摘要"""
        result = optimizer.equal_weight_portfolio()
        summary = result.summary()

        assert isinstance(summary, str)
        assert 'strategy_a' in summary
        assert 'Sharpe Ratio' in summary
        assert '成功' in summary

    def test_correlation_matrix(self, optimizer):
        """測試相關性矩陣"""
        corr_matrix = optimizer.get_correlation_matrix()

        # 檢查形狀
        assert corr_matrix.shape == (3, 3)

        # 檢查對角線為 1
        for i in range(3):
            assert abs(corr_matrix.iloc[i, i] - 1.0) < 1e-6

        # 檢查對稱性
        assert np.allclose(corr_matrix, corr_matrix.T)

    def test_allow_short(self, optimizer):
        """測試允許空頭"""
        result = optimizer.max_sharpe_optimize(
            allow_short=True,
            max_weight=1.0,
            min_weight=-0.5  # 允許最多 -50% 空頭
        )

        # 權重總和仍應為 1
        weights_sum = sum(result.weights.values())
        assert abs(weights_sum - 1.0) < 1e-4

        # 可能有負權重（但測試資料不一定會產生）
        weights_array = np.array(list(result.weights.values()))
        assert np.all(weights_array >= -0.5)
        assert np.all(weights_array <= 1.0)

    def test_ledoit_wolf_covariance(self, sample_returns):
        """測試 Ledoit-Wolf 協方差估計"""
        try:
            optimizer_lw = PortfolioOptimizer(
                returns=sample_returns,
                use_ledoit_wolf=True
            )

            # 應該成功建立
            assert optimizer_lw.cov_matrix is not None
            assert optimizer_lw.cov_matrix.shape == (3, 3)

        except ImportError:
            pytest.skip("sklearn 未安裝，跳過 Ledoit-Wolf 測試")

    def test_invalid_data(self):
        """測試無效資料處理"""
        # 只有 1 個策略（應該失敗）
        with pytest.raises(ValueError, match="至少需要 2 個策略"):
            PortfolioOptimizer(
                returns=pd.DataFrame({'strategy_a': [0.01, 0.02, 0.03]})
            )

    def test_nan_handling(self):
        """測試 NaN 處理"""
        returns_with_nan = pd.DataFrame({
            'strategy_a': [0.01, np.nan, 0.02],
            'strategy_b': [0.015, 0.02, np.nan],
            'strategy_c': [0.008, 0.012, 0.005]
        })

        # 應該發出警告但仍能建立
        with pytest.warns(UserWarning, match="回報資料包含 NaN"):
            optimizer = PortfolioOptimizer(returns=returns_with_nan)
            assert optimizer is not None


class TestEdgeCases:
    """邊界情況測試"""

    def test_identical_strategies(self):
        """測試完全相同的策略"""
        returns = pd.DataFrame({
            'strategy_a': [0.01, 0.02, -0.01, 0.015],
            'strategy_b': [0.01, 0.02, -0.01, 0.015],  # 完全相同
        })

        optimizer = PortfolioOptimizer(returns=returns)
        result = optimizer.max_sharpe_optimize()

        # 權重應該相等（或接近）
        weights = list(result.weights.values())
        assert abs(weights[0] - weights[1]) < 0.1

    def test_zero_volatility_strategy(self):
        """測試零波動策略"""
        returns = pd.DataFrame({
            'strategy_a': [0.01] * 100,  # 常數回報
            'strategy_b': np.random.normal(0.01, 0.02, 100)
        })

        # 應該能處理但可能發出警告
        optimizer = PortfolioOptimizer(returns=returns)
        result = optimizer.max_sharpe_optimize()

        # 零波動策略應該獲得較高權重
        assert result.weights['strategy_a'] > 0.3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
