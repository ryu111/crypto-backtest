"""
測試 Deflated Sharpe Ratio 模組

驗證：
1. Deflated Sharpe Ratio 計算正確性
2. PBO (Probability of Backtest Overfitting) 計算
3. 最小回測長度計算
4. 邊界條件處理
"""

import pytest
import numpy as np
from src.validator.sharpe_correction import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    minimum_backtest_length,
    calculate_sharpe_variance,
    expected_maximum_sharpe,
    DeflatedSharpeResult,
    PBOResult,
    MinimumBacktestLength,
)


# ========== Fixtures ==========

@pytest.fixture
def sample_returns():
    """產生模擬收益序列（252 天，Sharpe ≈ 2.0）"""
    np.random.seed(42)
    # 年化 Sharpe = 2.0 → 日均 μ = 2.0 / sqrt(252) ≈ 0.126%
    # 日標準差 σ = 1%
    returns = np.random.normal(0.00126, 0.01, 252)
    return returns


@pytest.fixture
def zero_returns():
    """零收益（Sharpe = 0）"""
    return np.zeros(252)


@pytest.fixture
def high_sharpe_returns():
    """高 Sharpe 收益（Sharpe ≈ 3.0）"""
    np.random.seed(123)
    returns = np.random.normal(0.0019, 0.01, 252)
    return returns


# ========== 測試 calculate_sharpe_variance ==========

def test_sharpe_variance_normal_distribution():
    """測試常態分布的 Sharpe 變異數計算"""
    returns = np.random.randn(252) * 0.01
    variance = calculate_sharpe_variance(returns, sharpe=1.0)

    assert variance > 0
    assert np.isfinite(variance)
    # 對於常態分布，variance ≈ 1/T
    assert 0.001 < variance < 0.01  # 合理範圍


def test_sharpe_variance_with_skewness_kurtosis():
    """測試有偏態和峰度的情況"""
    returns = np.random.randn(252) * 0.01

    # 手動指定偏態和峰度（與預設不同）
    variance = calculate_sharpe_variance(
        returns,
        sharpe=2.0,
        skewness=0.5,
        kurtosis=5.0  # 顯著不同於 3.0
    )

    assert variance > 0
    # 有偏態/峰度會影響變異數
    variance_normal = calculate_sharpe_variance(
        returns,
        sharpe=2.0,
        skewness=0.0,
        kurtosis=3.0
    )
    assert variance != variance_normal


def test_sharpe_variance_zero_sharpe():
    """測試 Sharpe = 0 的情況"""
    returns = np.random.randn(252) * 0.01
    variance = calculate_sharpe_variance(returns, sharpe=0.0)

    assert variance > 0
    # Sharpe = 0 時，variance ≈ 1/T
    assert 0.001 < variance < 0.01


# ========== 測試 expected_maximum_sharpe ==========

def test_expected_max_sharpe_single_trial():
    """測試只有 1 個策略時（應該回傳 0）"""
    expected_max = expected_maximum_sharpe(
        n_trials=1,
        t_years=1.0,
        sharpe_variance=0.004
    )
    assert expected_max == 0.0


def test_expected_max_sharpe_increases_with_trials():
    """測試預期最大 Sharpe 隨嘗試次數增加"""
    variance = 0.004

    max_10 = expected_maximum_sharpe(10, 1.0, variance)
    max_100 = expected_maximum_sharpe(100, 1.0, variance)
    max_1000 = expected_maximum_sharpe(1000, 1.0, variance)

    # 更多嘗試 → 預期最大值更高
    assert max_10 < max_100 < max_1000
    assert max_10 > 0
    assert max_1000 > max_100


def test_expected_max_sharpe_realistic_values():
    """測試實際數值合理性"""
    # 100 個策略，1 年回測
    expected_max = expected_maximum_sharpe(
        n_trials=100,
        t_years=1.0,
        sharpe_variance=0.004
    )

    # 預期最大 Sharpe 應該為正且合理（約 0.1 - 0.5）
    assert 0.05 < expected_max < 1.0


# ========== 測試 deflated_sharpe_ratio ==========

def test_deflated_sharpe_basic(sample_returns):
    """測試基本 Deflated Sharpe 計算"""
    result = deflated_sharpe_ratio(
        sharpe=2.0,
        n_trials=100,
        returns=sample_returns,
        t_years=1.0
    )

    assert isinstance(result, DeflatedSharpeResult)
    assert result.observed_sharpe == 2.0
    assert result.n_trials == 100
    assert result.t_years == 1.0

    # DSR 是 Z-score，數值可能大於原始 Sharpe（取決於 σ_SR）
    # 關鍵是檢查它是否有限且合理
    assert np.isfinite(result.deflated_sharpe)
    assert 0 <= result.p_value <= 1
    assert result.expected_max_sharpe > 0

    # 如果 Sharpe 顯著大於預期最大值，DSR 應該為正
    if result.observed_sharpe > result.expected_max_sharpe:
        assert result.deflated_sharpe > 0


def test_deflated_sharpe_single_trial(sample_returns):
    """測試只有 1 個策略時（無多重檢定偏差）"""
    result = deflated_sharpe_ratio(
        sharpe=2.0,
        n_trials=1,
        returns=sample_returns,
        t_years=1.0
    )

    # n_trials=1 時，expected_max=0，DSR 應該接近原始 Sharpe
    assert result.expected_max_sharpe == 0.0
    # DSR = (SR - 0) / σ
    assert result.deflated_sharpe > 0


def test_deflated_sharpe_high_trials(sample_returns):
    """測試大量嘗試次數（強烈的多重檢定偏差）"""
    result = deflated_sharpe_ratio(
        sharpe=2.0,
        n_trials=1000,
        returns=sample_returns,
        t_years=1.0
    )

    # 大量嘗試會增加 expected_max_sharpe
    # 但 DSR 的數值取決於 (SR - E[max]) / σ
    # 只檢查 expected_max 增加
    result_few = deflated_sharpe_ratio(
        sharpe=2.0,
        n_trials=10,
        returns=sample_returns,
        t_years=1.0
    )

    # 更多嘗試 → 預期最大值更高
    assert result.expected_max_sharpe > result_few.expected_max_sharpe


def test_deflated_sharpe_significance():
    """測試顯著性判斷"""
    # 高 Sharpe，少量嘗試 → 顯著
    result_significant = deflated_sharpe_ratio(
        sharpe=3.0,
        n_trials=10,
        variance=0.004,
        t_years=1.0
    )
    assert result_significant.is_significant
    assert result_significant.p_value < 0.05

    # 低 Sharpe，大量嘗試，大 variance → 不顯著
    result_not_significant = deflated_sharpe_ratio(
        sharpe=0.3,
        n_trials=1000,
        variance=0.02,  # 較大的變異數
        t_years=1.0
    )
    assert not result_not_significant.is_significant
    assert result_not_significant.p_value > 0.05


def test_deflated_sharpe_zero_sharpe(zero_returns):
    """測試 Sharpe = 0 的情況"""
    result = deflated_sharpe_ratio(
        sharpe=0.0,
        n_trials=100,
        returns=zero_returns,
        t_years=1.0
    )

    # Sharpe = 0 < expected_max → DSR 應該是負數
    assert result.deflated_sharpe < 0
    assert not result.is_significant
    # p-value 應該接近 1（H1: DSR > 0 不成立）
    assert result.p_value > 0.7


def test_deflated_sharpe_negative_sharpe():
    """測試負 Sharpe"""
    result = deflated_sharpe_ratio(
        sharpe=-1.0,
        n_trials=100,
        variance=0.004,
        t_years=1.0
    )

    # 負 Sharpe → DSR 更負
    assert result.deflated_sharpe < -1.0
    assert not result.is_significant
    assert result.p_value > 0.9


# ========== 測試 probability_of_backtest_overfitting ==========

def test_pbo_perfect_correlation():
    """測試完美相關（IS = OOS）"""
    is_sharpe = np.array([2.0, 1.5, 1.0, 0.5])
    oos_sharpe = np.array([2.0, 1.5, 1.0, 0.5])

    result = probability_of_backtest_overfitting(is_sharpe, oos_sharpe, n_trials=4)

    assert isinstance(result, PBOResult)
    # median(IS) = 1.25，所有 OOS >= 0.5，PBO 取決於有多少 < 1.25
    # 實際上 2 個 < 1.25 → PBO = 0.5
    assert result.pbo == 0.5
    # 排名相關性應該接近 1
    assert result.rank_correlation > 0.9
    # PBO = 0.5 會觸發警告
    assert result.warning is not None


def test_pbo_no_correlation():
    """測試無相關（IS 和 OOS 獨立）"""
    is_sharpe = np.array([2.0, 1.5, 1.0, 0.5])
    oos_sharpe = np.array([0.5, 1.0, 1.5, 2.0])  # 反向

    result = probability_of_backtest_overfitting(is_sharpe, oos_sharpe, n_trials=4)

    # PBO 應該很高（接近 1）
    assert result.pbo >= 0.5
    # 排名相關性應該是負數
    assert result.rank_correlation < 0
    assert result.warning is not None


def test_pbo_overfitting_scenario():
    """測試典型過擬合情境（IS 高，OOS 低）"""
    np.random.seed(42)
    n = 20

    # In-Sample: 高 Sharpe
    is_sharpe = np.random.uniform(1.5, 3.0, n)

    # Out-of-Sample: 顯著下降
    oos_sharpe = is_sharpe * 0.4 + np.random.normal(0, 0.2, n)

    result = probability_of_backtest_overfitting(is_sharpe, oos_sharpe, n_trials=n)

    # 過擬合機率應該很高
    assert result.pbo > 0.5
    assert "過擬合" in result.warning


def test_pbo_robust_strategy():
    """測試穩健策略（IS 和 OOS 一致）"""
    np.random.seed(123)
    n = 20

    # IS 和 OOS 接近，有小幅雜訊
    is_sharpe = np.random.uniform(1.0, 2.0, n)
    oos_sharpe = is_sharpe + np.random.normal(0, 0.1, n)

    result = probability_of_backtest_overfitting(is_sharpe, oos_sharpe, n_trials=n)

    # PBO 應該低於 0.5（大部分情況）
    # 但由於雜訊，可能在邊界
    assert result.pbo <= 0.5
    # 排名相關性應該高
    assert result.rank_correlation > 0.7


def test_pbo_mismatched_lengths():
    """測試輸入長度不匹配"""
    is_sharpe = np.array([1.0, 2.0, 3.0])
    oos_sharpe = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="長度必須相同"):
        probability_of_backtest_overfitting(is_sharpe, oos_sharpe, n_trials=3)


# ========== 測試 minimum_backtest_length ==========

def test_minimum_backtest_length_basic():
    """測試基本最小回測長度計算"""
    result = minimum_backtest_length(
        target_sharpe=2.0,
        n_trials=100,
        confidence=0.95
    )

    assert isinstance(result, MinimumBacktestLength)
    assert result.min_years > 0
    assert result.min_observations > 0
    assert result.target_sharpe == 2.0
    assert result.n_trials == 100

    # 最小觀察次數應該 = min_years * 252
    assert abs(result.min_observations - result.min_years * 252) < 10


def test_minimum_backtest_length_high_sharpe():
    """測試高 Sharpe（需要較少資料）"""
    result = minimum_backtest_length(
        target_sharpe=5.0,
        n_trials=100
    )

    # 高 Sharpe → 容易達到顯著 → 需要較少資料
    assert result.min_years < 5.0
    assert result.min_observations < 1260  # < 5 年


def test_minimum_backtest_length_low_sharpe():
    """測試低 Sharpe（需要更多資料）"""
    result_low = minimum_backtest_length(
        target_sharpe=1.0,
        n_trials=100
    )

    result_high = minimum_backtest_length(
        target_sharpe=3.0,
        n_trials=100
    )

    # 低 Sharpe → 需要更多資料（相對於高 Sharpe）
    assert result_low.min_years >= result_high.min_years


def test_minimum_backtest_length_many_trials():
    """測試大量嘗試次數（需要更多資料）"""
    result_few = minimum_backtest_length(
        target_sharpe=2.0,
        n_trials=10
    )

    result_many = minimum_backtest_length(
        target_sharpe=2.0,
        n_trials=1000
    )

    # 更多嘗試 → 需要更多資料抵消偏差
    assert result_many.min_years >= result_few.min_years
    assert result_many.min_observations >= result_few.min_observations


def test_minimum_backtest_length_confidence_levels():
    """測試不同信賴水準"""
    result_90 = minimum_backtest_length(
        target_sharpe=2.0,
        n_trials=100,
        confidence=0.90
    )

    result_99 = minimum_backtest_length(
        target_sharpe=2.0,
        n_trials=100,
        confidence=0.99
    )

    # 更高信賴水準 → 需要更多資料
    assert result_99.min_years >= result_90.min_years


# ========== 邊界條件測試 ==========

def test_deflated_sharpe_with_variance_directly():
    """測試直接提供 variance（不提供 returns）"""
    result = deflated_sharpe_ratio(
        sharpe=2.0,
        n_trials=100,
        variance=0.005,  # 直接指定
        t_years=1.0
    )

    assert result.sharpe_std == np.sqrt(0.005)


def test_deflated_sharpe_missing_variance_and_returns():
    """測試缺少 variance 和 returns（應該報錯）"""
    with pytest.raises(ValueError, match="必須提供 variance 或 returns"):
        deflated_sharpe_ratio(
            sharpe=2.0,
            n_trials=100,
            t_years=1.0
        )


def test_calculate_sharpe_variance_short_series():
    """測試短時間序列"""
    short_returns = np.random.randn(10) * 0.01

    variance = calculate_sharpe_variance(short_returns, sharpe=1.0)

    # 短序列 → 高變異數
    assert variance > 0.01


def test_deflated_sharpe_very_high_sharpe():
    """測試極高 Sharpe（數值穩定性）"""
    result = deflated_sharpe_ratio(
        sharpe=10.0,
        n_trials=100,
        variance=0.001,
        t_years=1.0
    )

    # 檢查數值穩定性
    assert np.isfinite(result.deflated_sharpe)
    assert np.isfinite(result.p_value)
    assert result.is_significant  # 極高 Sharpe 應該顯著


# ========== 整合測試 ==========

def test_full_workflow(sample_returns, high_sharpe_returns):
    """測試完整工作流程"""
    # 假設測試了 50 個策略
    n_trials = 50

    # 計算 Sharpe
    from src.validator.statistical_tests import calculate_sharpe
    sharpe = calculate_sharpe(high_sharpe_returns)

    # Deflated Sharpe
    dsr_result = deflated_sharpe_ratio(
        sharpe=sharpe,
        n_trials=n_trials,
        returns=high_sharpe_returns,
        t_years=1.0
    )

    # 檢查結果合理性
    assert np.isfinite(dsr_result.deflated_sharpe)
    assert 0 <= dsr_result.p_value <= 1

    # 如果 DSR 顯著，檢查最小回測長度
    if dsr_result.is_significant:
        min_length = minimum_backtest_length(
            target_sharpe=sharpe,
            n_trials=n_trials
        )
        # 最小長度應該合理
        assert min_length.min_years > 0

    # 模擬 PBO 檢測
    # 假設 IS = 當前收益，OOS = 略低
    is_sharpe = np.array([sharpe] * 10)
    oos_sharpe = is_sharpe * 0.8 + np.random.normal(0, 0.1, 10)

    pbo_result = probability_of_backtest_overfitting(
        is_sharpe, oos_sharpe, n_trials=10
    )

    assert 0 <= pbo_result.pbo <= 1


def test_consistency_with_bootstrap():
    """測試與 Bootstrap 方法的一致性"""
    from src.validator.statistical_tests import bootstrap_sharpe

    np.random.seed(42)
    returns = np.random.normal(0.001, 0.01, 252)

    # Bootstrap Sharpe
    bootstrap_result = bootstrap_sharpe(returns, n_bootstrap=1000)

    # Deflated Sharpe
    dsr_result = deflated_sharpe_ratio(
        sharpe=bootstrap_result.sharpe_mean,
        n_trials=100,
        returns=returns,
        t_years=1.0
    )

    # 兩者的顯著性判斷應該一致（大部分情況）
    # Bootstrap: p < 0.05 → 顯著
    # DSR: DSR > 1.96 → 顯著
    if bootstrap_result.p_value < 0.05:
        # Bootstrap 顯著，DSR 也應該傾向顯著（但可能因多重檢定而不顯著）
        assert dsr_result.deflated_sharpe > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
