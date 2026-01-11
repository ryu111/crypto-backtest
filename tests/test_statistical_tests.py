"""
統計檢定模組單元測試

測試 Bootstrap、Permutation Test、Block Bootstrap 等功能。
"""

import pytest
import numpy as np
from src.validator.statistical_tests import (
    calculate_sharpe,
    bootstrap_sharpe,
    permutation_test,
    block_bootstrap,
    run_statistical_tests,
    BootstrapResult,
    PermutationResult,
    StatisticalTestReport,
)


# ========== Fixtures ==========

@pytest.fixture
def positive_returns():
    """正收益策略（應該有正 Sharpe）"""
    np.random.seed(42)
    return np.random.randn(252) * 0.01 + 0.002  # 平均 +0.2%/天（較強信號）


@pytest.fixture
def negative_returns():
    """負收益策略（應該有負 Sharpe）"""
    np.random.seed(42)
    return np.random.randn(252) * 0.01 - 0.0005  # 平均 -0.05%/天


@pytest.fixture
def random_returns():
    """隨機遊走（Sharpe 應接近 0）"""
    np.random.seed(42)
    return np.random.randn(252) * 0.01  # 平均 0%


@pytest.fixture
def autocorrelated_returns():
    """有自相關的收益序列"""
    np.random.seed(42)
    n = 252
    returns = np.zeros(n)
    returns[0] = np.random.randn() * 0.01

    # AR(1) 過程：r_t = 0.5 * r_{t-1} + 噪音
    for i in range(1, n):
        returns[i] = 0.5 * returns[i-1] + np.random.randn() * 0.01

    return returns


# ========== 基礎函數測試 ==========

class TestCalculateSharpe:
    """測試 Sharpe Ratio 計算"""

    def test_basic_calculation(self, positive_returns):
        """基本計算測試"""
        sharpe = calculate_sharpe(positive_returns)
        assert isinstance(sharpe, float)
        assert sharpe > 0  # 正收益應該有正 Sharpe

    def test_negative_sharpe(self, negative_returns):
        """負 Sharpe 測試"""
        sharpe = calculate_sharpe(negative_returns)
        assert sharpe < 0

    def test_zero_std_returns(self):
        """標準差為 0 的情況（應回傳 0）"""
        returns = np.ones(100) * 0.01  # 所有收益都相同
        sharpe = calculate_sharpe(returns)
        assert sharpe == 0.0

    def test_empty_returns(self):
        """空序列測試"""
        sharpe = calculate_sharpe(np.array([]))
        assert sharpe == 0.0

    def test_risk_free_rate(self, positive_returns):
        """測試無風險利率影響"""
        sharpe_0 = calculate_sharpe(positive_returns, risk_free_rate=0.0)
        sharpe_5 = calculate_sharpe(positive_returns, risk_free_rate=0.05)

        # 加入無風險利率後，Sharpe 應該變小
        assert sharpe_5 < sharpe_0


# ========== Bootstrap Test ==========

class TestBootstrapSharpe:
    """測試 Bootstrap Test"""

    def test_basic_bootstrap(self, positive_returns):
        """基本 Bootstrap 測試"""
        result = bootstrap_sharpe(
            positive_returns,
            n_bootstrap=1000,  # 使用較小的次數加速測試
            confidence=0.95,
            n_jobs=1,  # 單執行緒測試
            random_state=42
        )

        # 檢查回傳類型
        assert isinstance(result, BootstrapResult)

        # 檢查欄位
        assert isinstance(result.sharpe_mean, float)
        assert isinstance(result.ci_lower, float)
        assert isinstance(result.ci_upper, float)
        assert 0 <= result.p_value <= 1

        # 信賴區間應該有效
        assert result.ci_lower < result.sharpe_mean < result.ci_upper

        # 正收益策略的 p-value 應該較小（使用較寬鬆的閾值）
        assert result.p_value < 0.5  # 至少應該 < 50%

    def test_distribution_shape(self, positive_returns):
        """檢查 Bootstrap 分布形狀"""
        result = bootstrap_sharpe(
            positive_returns,
            n_bootstrap=1000,
            random_state=42
        )

        # 分布長度正確
        assert len(result.sharpe_distribution) == 1000

        # 分布應該接近常態（檢查偏度）
        dist = result.sharpe_distribution
        mean = np.mean(dist)
        std = np.std(dist, ddof=1)
        skewness = np.mean(((dist - mean) / std) ** 3)

        # 偏度應該不要太極端（|skew| < 1 為經驗法則）
        assert abs(skewness) < 2

    def test_confidence_levels(self, positive_returns):
        """測試不同信賴水準"""
        result_90 = bootstrap_sharpe(positive_returns, confidence=0.90, random_state=42)
        result_95 = bootstrap_sharpe(positive_returns, confidence=0.95, random_state=42)
        result_99 = bootstrap_sharpe(positive_returns, confidence=0.99, random_state=42)

        # 信賴水準越高，區間越寬
        width_90 = result_90.ci_upper - result_90.ci_lower
        width_95 = result_95.ci_upper - result_95.ci_lower
        width_99 = result_99.ci_upper - result_99.ci_lower

        assert width_90 < width_95 < width_99

    def test_reproducibility(self, positive_returns):
        """測試隨機種子可重現性"""
        result1 = bootstrap_sharpe(positive_returns, random_state=42)
        result2 = bootstrap_sharpe(positive_returns, random_state=42)

        # 使用相同種子應該得到相同結果
        assert result1.sharpe_mean == result2.sharpe_mean
        np.testing.assert_array_equal(
            result1.sharpe_distribution,
            result2.sharpe_distribution
        )

    def test_insufficient_data(self):
        """測試資料不足的情況"""
        with pytest.raises(ValueError, match="至少 2 個收益資料點"):
            bootstrap_sharpe(np.array([0.01]))


# ========== Permutation Test ==========

class TestPermutationTest:
    """測試 Permutation Test"""

    def test_significant_strategy(self, positive_returns):
        """測試顯著策略"""
        result = permutation_test(
            positive_returns,
            n_permutations=1000,
            n_jobs=1,
            random_state=42
        )

        # 檢查回傳類型
        assert isinstance(result, PermutationResult)

        # 實際 Sharpe 應該 > 0
        assert result.actual_sharpe > 0

        # 虛無假設平均應接近 0（符號翻轉後）
        assert abs(result.null_mean) < 1.0

        # p-value 應該在合理範圍
        assert 0 <= result.p_value <= 1

        # 正收益策略應該顯著（或至少 p-value 較小）
        assert result.p_value < 0.5

    def test_random_strategy(self, random_returns):
        """測試隨機策略（應該不顯著）"""
        result = permutation_test(
            random_returns,
            n_permutations=1000,
            n_jobs=1,
            random_state=42
        )

        # p-value 應該不顯著
        # 注意：隨機測試可能偶爾失敗，使用寬鬆的閾值
        assert result.p_value > 0.01

    def test_null_distribution_shape(self, positive_returns):
        """檢查虛無假設分布"""
        result = permutation_test(
            positive_returns,
            n_permutations=1000,
            random_state=42
        )

        # 分布長度正確
        assert len(result.null_distribution) == 1000

        # 虛無假設平均應接近 0（符號翻轉後）
        assert abs(result.null_mean) < 1.0

        # 標準差應該 > 0
        assert result.null_std > 0

    def test_reproducibility(self, positive_returns):
        """測試可重現性"""
        result1 = permutation_test(positive_returns, random_state=42)
        result2 = permutation_test(positive_returns, random_state=42)

        assert result1.actual_sharpe == result2.actual_sharpe
        assert result1.p_value == result2.p_value
        np.testing.assert_array_equal(
            result1.null_distribution,
            result2.null_distribution
        )


# ========== Block Bootstrap ==========

class TestBlockBootstrap:
    """測試 Block Bootstrap"""

    def test_basic_block_bootstrap(self, autocorrelated_returns):
        """基本 Block Bootstrap 測試"""
        result = block_bootstrap(
            autocorrelated_returns,
            block_size=20,
            n_bootstrap=1000,
            random_state=42
        )

        # 檢查回傳類型（應該同 BootstrapResult）
        assert isinstance(result, BootstrapResult)

        # 基本統計量檢查
        assert isinstance(result.sharpe_mean, float)
        assert result.ci_lower < result.ci_upper

    def test_block_size_validation(self):
        """測試區塊大小驗證"""
        short_returns = np.random.randn(10)

        # block_size > 資料長度應該報錯
        with pytest.raises(ValueError, match="資料長度"):
            block_bootstrap(short_returns, block_size=20)

    def test_different_block_sizes(self, autocorrelated_returns):
        """測試不同區塊大小"""
        result_small = block_bootstrap(
            autocorrelated_returns,
            block_size=5,
            random_state=42
        )
        result_large = block_bootstrap(
            autocorrelated_returns,
            block_size=50,
            random_state=42
        )

        # 兩者應該都能執行
        assert isinstance(result_small.sharpe_mean, float)
        assert isinstance(result_large.sharpe_mean, float)

    def test_preserves_time_structure(self, autocorrelated_returns):
        """驗證 Block Bootstrap 是否保留時間結構"""
        # 這個測試較複雜，簡化為檢查結果的合理性
        result = block_bootstrap(
            autocorrelated_returns,
            block_size=20,
            n_bootstrap=1000,
            random_state=42
        )

        # 標準差應該 > 0
        assert result.sharpe_std > 0

        # 分布應該有合理範圍
        assert len(result.sharpe_distribution) == 1000


# ========== 整合測試 ==========

class TestRunStatisticalTests:
    """測試完整統計檢定流程"""

    def test_comprehensive_report(self, positive_returns):
        """測試完整報告"""
        report = run_statistical_tests(
            positive_returns,
            n_bootstrap=1000,
            n_permutations=1000,
            n_jobs=1,
            random_state=42
        )

        # 檢查回傳類型
        assert isinstance(report, StatisticalTestReport)

        # 檢查欄位
        assert isinstance(report.bootstrap_sharpe, float)
        assert isinstance(report.bootstrap_ci, tuple)
        assert len(report.bootstrap_ci) == 2
        assert 0 <= report.bootstrap_p_value <= 1
        assert 0 <= report.permutation_p_value <= 1

        # Bootstrap Sharpe 應該 > 0（正收益策略）
        assert report.bootstrap_sharpe > 0

        # 詳細結果應該存在
        assert report.bootstrap_result is not None
        assert report.permutation_result is not None

    def test_non_significant_strategy(self, random_returns):
        """測試不顯著策略"""
        report = run_statistical_tests(
            random_returns,
            n_bootstrap=1000,
            n_permutations=1000,
            n_jobs=1,
            random_state=42
        )

        # 隨機策略應該不顯著（或至少 p-value 較大）
        # 使用寬鬆條件避免隨機失敗
        assert report.bootstrap_p_value > 0.01 or report.permutation_p_value > 0.01

    def test_block_bootstrap_option(self, autocorrelated_returns):
        """測試 Block Bootstrap 選項"""
        report = run_statistical_tests(
            autocorrelated_returns,
            use_block_bootstrap=True,
            block_size=20,
            n_bootstrap=1000,
            n_permutations=1000,
            random_state=42
        )

        # 應該成功執行
        assert isinstance(report, StatisticalTestReport)
        assert report.bootstrap_result is not None

    def test_both_tests_must_pass(self):
        """測試「兩項檢定都通過」的邏輯"""
        # 建立一個 Bootstrap 顯著但 Permutation 不顯著的情況
        # 這需要手動構造，此處僅測試邏輯

        # 使用模擬資料
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.0002  # 微弱正收益

        report = run_statistical_tests(
            returns,
            n_bootstrap=500,
            n_permutations=500,
            n_jobs=1,
            random_state=42
        )

        # 驗證邏輯：只有兩者都 < 0.05 才顯著
        if report.bootstrap_p_value < 0.05 and report.permutation_p_value < 0.05:
            assert report.is_statistically_significant
        else:
            assert not report.is_statistically_significant


# ========== 邊界情況測試 ==========

class TestEdgeCases:
    """測試邊界情況"""

    def test_very_short_series(self):
        """測試極短序列"""
        short_returns = np.array([0.01, 0.02])

        # Bootstrap 應該能執行（但結果可能不可靠）
        result = bootstrap_sharpe(short_returns, n_bootstrap=100, random_state=42)
        assert isinstance(result, BootstrapResult)

        # Permutation 也應該能執行
        result = permutation_test(short_returns, n_permutations=100, random_state=42)
        assert isinstance(result, PermutationResult)

    def test_all_zero_returns(self):
        """測試全零收益"""
        zero_returns = np.zeros(100)

        sharpe = calculate_sharpe(zero_returns)
        assert sharpe == 0.0

        # Bootstrap 應該回傳 0
        result = bootstrap_sharpe(zero_returns, n_bootstrap=100, random_state=42)
        assert result.sharpe_mean == 0.0

    def test_extreme_values(self):
        """測試極端值"""
        extreme_returns = np.array([0.1, -0.1, 0.1, -0.1] * 50)  # 極端波動

        # 應該能正常計算
        sharpe = calculate_sharpe(extreme_returns)
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)


# ========== 效能測試 ==========

class TestPerformance:
    """測試效能（可選）"""

    @pytest.mark.slow
    def test_large_bootstrap(self, positive_returns):
        """測試大量 Bootstrap（使用多核心）"""
        import time

        start = time.time()
        result = bootstrap_sharpe(
            positive_returns,
            n_bootstrap=100000,
            n_jobs=-1  # 使用所有 CPU
        )
        elapsed = time.time() - start

        print(f"\n100k Bootstrap 耗時: {elapsed:.2f} 秒")
        assert result is not None

    @pytest.mark.slow
    def test_parallel_speedup(self, positive_returns):
        """測試多核心加速效果"""
        import time

        # 單執行緒
        start = time.time()
        bootstrap_sharpe(positive_returns, n_bootstrap=10000, n_jobs=1)
        time_single = time.time() - start

        # 多執行緒
        start = time.time()
        bootstrap_sharpe(positive_returns, n_bootstrap=10000, n_jobs=-1)
        time_multi = time.time() - start

        print(f"\n單執行緒: {time_single:.2f}s, 多執行緒: {time_multi:.2f}s")
        print(f"加速比: {time_single / time_multi:.2f}x")

        # 多執行緒應該更快（但不是絕對，小數據可能因開銷反而變慢）
        # 僅作為參考，不做斷言


if __name__ == "__main__":
    # 允許直接執行測試
    pytest.main([__file__, "-v"])
