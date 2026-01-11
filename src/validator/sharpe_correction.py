"""
Deflated Sharpe Ratio 模組

實作 Bailey-López de Prado (2014) 的 Deflated Sharpe Ratio，
用於校正多重檢定偏差（Multiple Testing Bias）。

當測試多個策略時，即使策略本身無預測能力，也有機率產生高 Sharpe Ratio。
Deflated Sharpe Ratio 考慮嘗試次數，調整 Sharpe 的統計顯著性。

主要功能：
1. Deflated Sharpe Ratio - 校正後的 Sharpe Ratio
2. Probability of Backtest Overfitting (PBO) - 過擬合機率
3. 最小回測長度計算 - 需要多少資料才能達到統計顯著

參考文獻：
    Bailey, D. H., & López de Prado, M. (2014).
    "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality"
    Journal of Portfolio Management, 40(5), 94-107.

使用範例：
    >>> returns = np.array([0.01, -0.02, 0.03, ...])
    >>> dsr = deflated_sharpe_ratio(
    ...     sharpe=2.5,
    ...     n_trials=100,        # 測試了 100 個策略
    ...     variance=0.1,
    ...     t_years=1
    ... )
    >>> print(f"Deflated SR: {dsr:.2f}")
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import stats


# ========== 資料類別 ==========

@dataclass
class DeflatedSharpeResult:
    """Deflated Sharpe Ratio 計算結果"""
    observed_sharpe: float          # 觀察到的 Sharpe Ratio
    deflated_sharpe: float          # 校正後的 Deflated Sharpe Ratio
    expected_max_sharpe: float      # 預期的最大 Sharpe（多重檢定下）
    sharpe_std: float               # Sharpe 的標準誤
    p_value: float                  # 顯著性 p 值
    is_significant: bool            # 是否顯著（p < 0.05）
    n_trials: int                   # 嘗試的策略數量
    t_years: float                  # 回測年數
    confidence_level: float = 0.95  # 信賴水準


@dataclass
class PBOResult:
    """Probability of Backtest Overfitting 結果"""
    pbo: float                      # 過擬合機率（0-1）
    n_trials: int                   # 嘗試的策略數量
    rank_correlation: float         # In-sample vs Out-of-sample 排名相關性
    warning: Optional[str] = None   # 警告訊息


@dataclass
class MinimumBacktestLength:
    """最小回測長度計算結果"""
    min_years: float                # 最小回測年數
    min_observations: int           # 最小觀察次數（假設日資料）
    target_sharpe: float            # 目標 Sharpe Ratio
    n_trials: int                   # 嘗試的策略數量
    confidence: float               # 信賴水準


# ========== 核心計算函數 ==========

def calculate_sharpe_variance(
    returns: np.ndarray,
    sharpe: Optional[float] = None,
    skewness: Optional[float] = None,
    kurtosis: Optional[float] = None
) -> float:
    """
    計算 Sharpe Ratio 的變異數

    考慮收益分布的高階動差（skewness, kurtosis）。

    Args:
        returns: 收益序列（用於計算 skewness 和 kurtosis）
        sharpe: Sharpe Ratio（如果未提供，從 returns 計算）
        skewness: 收益偏態（如果未提供，從 returns 計算）
        kurtosis: 收益峰度（如果未提供，從 returns 計算）

    Returns:
        Sharpe Ratio 的變異數

    Notes:
        公式來自 Bailey-López de Prado (2014) Eq. 5:
        Var[SR] = (1 + SR²/2 - skew*SR + (kurt-1)*SR²/4) / T
    """
    if sharpe is None:
        # 從 returns 計算 Sharpe
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0.0

    # 如果 returns 可用，計算實際的 skewness 和 kurtosis
    if returns is not None and len(returns) > 0:
        if skewness is None:
            skewness = stats.skew(returns, bias=False)
        if kurtosis is None:
            kurtosis = stats.kurtosis(returns, bias=False, fisher=False)  # Excess kurtosis + 3
    else:
        # 預設值：常態分布
        if skewness is None:
            skewness = 0.0
        if kurtosis is None:
            kurtosis = 3.0

    T = len(returns) if returns is not None else 252  # 假設 1 年

    # Bailey-López de Prado (2014) 公式
    variance = (
        1.0
        + sharpe**2 / 2.0
        - skewness * sharpe
        + (kurtosis - 1) * sharpe**2 / 4.0
    ) / T

    return max(variance, 1e-10)  # 防止負值或零


def expected_maximum_sharpe(
    n_trials: int,
    t_years: float,
    sharpe_variance: float,
    gamma: float = 0.5772156649  # Euler-Mascheroni constant
) -> float:
    """
    計算多重檢定下的預期最大 Sharpe Ratio

    當測試 N 個策略時，即使它們都是隨機的，最大 Sharpe 也會比單一策略高。

    Args:
        n_trials: 嘗試的策略數量
        t_years: 回測年數
        sharpe_variance: Sharpe 的變異數
        gamma: Euler-Mascheroni 常數（約 0.5772）

    Returns:
        預期的最大 Sharpe Ratio（E[max SR]）

    Notes:
        公式來自 Bailey-López de Prado (2014) Eq. 7:
        E[max SR] = (1 - γ) * Φ⁻¹(1 - 1/N) + γ * Φ⁻¹(1 - 1/(N*e))
        其中 Φ⁻¹ 是標準常態的逆 CDF
    """
    if n_trials <= 1:
        return 0.0

    # 標準誤
    sharpe_std = np.sqrt(sharpe_variance)

    # Bailey-López de Prado (2014) 公式
    # E[max SR] = (1-γ) * Z(1 - 1/N) + γ * Z(1 - 1/(N*e))
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))

    expected_max = ((1 - gamma) * z1 + gamma * z2) * sharpe_std

    return expected_max


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    variance: Optional[float] = None,
    returns: Optional[np.ndarray] = None,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    t_years: float = 1.0,
    confidence: float = 0.95
) -> DeflatedSharpeResult:
    """
    計算 Deflated Sharpe Ratio (DSR)

    DSR 校正多重檢定偏差，考慮嘗試了多少個策略。

    Args:
        sharpe: 觀察到的 Sharpe Ratio
        n_trials: 嘗試的策略數量（包含未通過的）
        variance: Sharpe 的變異數（如果未提供，從 returns 計算）
        returns: 收益序列（用於計算 variance, skewness, kurtosis）
        skewness: 收益偏態（預設 0）
        kurtosis: 收益峰度（預設 3）
        t_years: 回測年數（預設 1）
        confidence: 信賴水準（預設 0.95）

    Returns:
        DeflatedSharpeResult 包含：
            - deflated_sharpe: 校正後的 DSR
            - p_value: 顯著性檢定 p 值
            - expected_max_sharpe: 預期的最大 Sharpe

    Example:
        >>> # 測試了 100 個策略，最佳的 Sharpe = 2.5
        >>> result = deflated_sharpe_ratio(
        ...     sharpe=2.5,
        ...     n_trials=100,
        ...     variance=0.1,
        ...     t_years=1
        ... )
        >>> print(f"Deflated SR: {result.deflated_sharpe:.2f}")
        >>> print(f"p-value: {result.p_value:.4f}")
        >>> print(f"Significant: {result.is_significant}")

    Notes:
        公式來自 Bailey-López de Prado (2014) Eq. 8:
        DSR = (SR_observed - E[max SR]) / σ_SR

        解讀：
        - DSR > 0: 優於隨機策略
        - DSR > 1.96: 顯著（95% 信賴水準）
        - DSR < 0: 可能是運氣，不是真實技能
    """
    # 計算 Sharpe 的變異數
    if variance is None:
        if returns is None:
            raise ValueError("必須提供 variance 或 returns")
        variance = calculate_sharpe_variance(
            returns, sharpe, skewness, kurtosis
        )

    # 計算預期的最大 Sharpe（多重檢定偏差）
    expected_max = expected_maximum_sharpe(
        n_trials=n_trials,
        t_years=t_years,
        sharpe_variance=variance
    )

    # 計算 Deflated Sharpe Ratio
    sharpe_std = np.sqrt(variance)
    deflated_sr = (sharpe - expected_max) / sharpe_std if sharpe_std > 0 else 0.0

    # 計算 p-value（單尾檢定：H1: DSR > 0）
    p_value = 1 - stats.norm.cdf(deflated_sr)

    # 判斷顯著性
    critical_value = stats.norm.ppf(confidence)
    is_significant = deflated_sr > critical_value

    return DeflatedSharpeResult(
        observed_sharpe=sharpe,
        deflated_sharpe=deflated_sr,
        expected_max_sharpe=expected_max,
        sharpe_std=sharpe_std,
        p_value=p_value,
        is_significant=is_significant,
        n_trials=n_trials,
        t_years=t_years,
        confidence_level=confidence
    )


def probability_of_backtest_overfitting(
    is_sharpe: np.ndarray,
    oos_sharpe: np.ndarray,
    n_trials: int
) -> PBOResult:
    """
    計算 Probability of Backtest Overfitting (PBO)

    PBO 衡量策略在樣本外表現不如樣本內的機率。

    Args:
        is_sharpe: In-Sample Sharpe Ratios（N 個策略）
        oos_sharpe: Out-of-Sample Sharpe Ratios（對應的 N 個策略）
        n_trials: 嘗試的策略數量

    Returns:
        PBOResult 包含：
            - pbo: 過擬合機率（0-1）
            - rank_correlation: IS vs OOS 排名相關性

    Example:
        >>> # 100 個策略的 IS/OOS Sharpe
        >>> is_sharpe = np.array([2.5, 2.3, 1.8, ...])   # 訓練期
        >>> oos_sharpe = np.array([1.2, 0.5, 1.5, ...])  # 測試期
        >>> result = probability_of_backtest_overfitting(
        ...     is_sharpe, oos_sharpe, n_trials=100
        ... )
        >>> if result.pbo > 0.5:
        ...     print(f"警告：過擬合機率高 ({result.pbo:.1%})")

    Notes:
        PBO 解讀：
        - PBO < 0.3: 低風險
        - 0.3 ≤ PBO < 0.5: 中等風險
        - PBO ≥ 0.5: 高風險，可能過擬合
    """
    is_sharpe = np.asarray(is_sharpe)
    oos_sharpe = np.asarray(oos_sharpe)

    if len(is_sharpe) != len(oos_sharpe):
        raise ValueError("IS 和 OOS Sharpe 長度必須相同")

    # 計算排名相關性（Spearman）
    rank_corr, _ = stats.spearmanr(is_sharpe, oos_sharpe)

    # PBO = P(OOS Sharpe < median(IS Sharpe))
    median_is = np.median(is_sharpe)
    pbo = np.mean(oos_sharpe < median_is)

    # 產生警告
    warning = None
    if pbo >= 0.5:
        warning = "高過擬合風險：OOS 表現顯著差於 IS"
    elif pbo >= 0.3:
        warning = "中等過擬合風險：建議增加 OOS 驗證"

    return PBOResult(
        pbo=pbo,
        n_trials=n_trials,
        rank_correlation=rank_corr,
        warning=warning
    )


def minimum_backtest_length(
    target_sharpe: float,
    n_trials: int,
    confidence: float = 0.95,
    skewness: float = 0.0,
    kurtosis: float = 3.0
) -> MinimumBacktestLength:
    """
    計算達到統計顯著所需的最小回測長度

    Args:
        target_sharpe: 目標 Sharpe Ratio
        n_trials: 嘗試的策略數量
        confidence: 信賴水準（預設 0.95）
        skewness: 預期的收益偏態
        kurtosis: 預期的收益峰度

    Returns:
        MinimumBacktestLength 包含：
            - min_years: 最小回測年數
            - min_observations: 最小觀察次數（假設日資料）

    Example:
        >>> # 要達到 Sharpe = 2.0，測試 100 個策略
        >>> result = minimum_backtest_length(
        ...     target_sharpe=2.0,
        ...     n_trials=100
        ... )
        >>> print(f"需要至少 {result.min_years:.1f} 年資料")
        >>> print(f"約 {result.min_observations} 天")

    Notes:
        這個計算假設：
        1. 日資料（252 交易日/年）
        2. 使用 Deflated Sharpe Ratio 框架
        3. 目標顯著水準 95%
    """
    critical_value = stats.norm.ppf(confidence)

    # Bailey-López de Prado (2014) 反推公式
    # 需要解方程：DSR = critical_value
    # 其中 DSR = (SR - E[max SR]) / σ_SR

    # 簡化計算：假設 variance ≈ 1/T（常態分布）
    # 迭代求解 T

    # 初始猜測：1 年
    t_years = 1.0
    for iteration in range(200):  # 迭代優化
        T = t_years * 252  # 交易日數

        variance = (
            1.0
            + target_sharpe**2 / 2.0
            - skewness * target_sharpe
            + (kurtosis - 1) * target_sharpe**2 / 4.0
        ) / T

        expected_max = expected_maximum_sharpe(
            n_trials=n_trials,
            t_years=t_years,
            sharpe_variance=variance
        )

        sharpe_std = np.sqrt(variance)

        # 避免除零
        if sharpe_std < 1e-10:
            break

        dsr = (target_sharpe - expected_max) / sharpe_std

        # 調整 t_years（更激進的步長）
        if dsr < critical_value:
            # 需要更多資料
            step = max(0.1, (critical_value - dsr) * 0.1)
            t_years *= (1 + step)
        else:
            # 已經達到目標，可以提早終止
            if iteration > 10:  # 至少迭代 10 次確保穩定
                break
            t_years *= 0.99  # 微調

    min_observations = int(t_years * 252)

    return MinimumBacktestLength(
        min_years=t_years,
        min_observations=min_observations,
        target_sharpe=target_sharpe,
        n_trials=n_trials,
        confidence=confidence
    )


# ========== 輔助函數 ==========

def print_deflated_sharpe_report(result: DeflatedSharpeResult) -> None:
    """
    美化輸出 Deflated Sharpe Ratio 報告

    Example:
        >>> result = deflated_sharpe_ratio(sharpe=2.5, n_trials=100, variance=0.1)
        >>> print_deflated_sharpe_report(result)
    """
    print("=" * 70)
    print("Deflated Sharpe Ratio 報告".center(70))
    print("=" * 70)

    print(f"\n[輸入參數]")
    print(f"  嘗試的策略數量: {result.n_trials}")
    print(f"  回測年數: {result.t_years:.1f}")
    print(f"  信賴水準: {result.confidence_level * 100:.0f}%")

    print(f"\n[Sharpe Ratio]")
    print(f"  觀察到的 Sharpe: {result.observed_sharpe:.3f}")
    print(f"  預期最大 Sharpe (多重檢定): {result.expected_max_sharpe:.3f}")
    print(f"  Sharpe 標準誤: {result.sharpe_std:.3f}")

    print(f"\n[Deflated Sharpe Ratio]")
    print(f"  DSR: {result.deflated_sharpe:.3f}")
    print(f"  p-value: {result.p_value:.4f}")

    print("\n" + "=" * 70)
    if result.is_significant:
        print("結論: 策略具有統計顯著性 ✓".center(70))
        print(f"（DSR > {stats.norm.ppf(result.confidence_level):.2f}，"
              f"p < {1 - result.confidence_level:.2f}）".center(70))
    else:
        print("結論: 策略缺乏統計顯著性 ✗".center(70))
        print("可能是多重檢定下的偽陽性".center(70))
    print("=" * 70)


def print_pbo_report(result: PBOResult) -> None:
    """
    美化輸出 PBO 報告

    Example:
        >>> result = probability_of_backtest_overfitting(is_sharpe, oos_sharpe, n_trials=100)
        >>> print_pbo_report(result)
    """
    print("=" * 70)
    print("Probability of Backtest Overfitting (PBO) 報告".center(70))
    print("=" * 70)

    print(f"\n[過擬合指標]")
    print(f"  PBO: {result.pbo:.1%}")
    print(f"  排名相關性 (IS vs OOS): {result.rank_correlation:.3f}")
    print(f"  嘗試的策略數量: {result.n_trials}")

    print(f"\n[風險評估]")
    if result.pbo < 0.3:
        risk_level = "低風險 ✓"
    elif result.pbo < 0.5:
        risk_level = "中等風險 ⚠"
    else:
        risk_level = "高風險 ✗"
    print(f"  風險等級: {risk_level}")

    if result.warning:
        print(f"  警告: {result.warning}")

    print("\n" + "=" * 70)
    print("建議：".center(70))
    if result.pbo >= 0.5:
        print("- 增加 Out-of-Sample 驗證期".center(70))
        print("- 考慮 Walk-Forward 分析".center(70))
        print("- 減少參數優化次數".center(70))
    elif result.pbo >= 0.3:
        print("- 建議使用多個 OOS 期間驗證".center(70))
    else:
        print("策略穩健性良好".center(70))
    print("=" * 70)
