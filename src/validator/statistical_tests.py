"""
進階統計檢定模組

提供 Bootstrap Test、Permutation Test 等統計方法，用於驗證策略績效的顯著性。

主要功能：
1. Bootstrap Test - 計算 Sharpe Ratio 信賴區間
2. Permutation Test - 檢定策略績效顯著性
3. Block Bootstrap - 保留時間序列相關性的抽樣
4. 統計檢定報告 - 整合所有檢定結果

使用範例：
    >>> returns = np.array([0.01, -0.02, 0.03, ...])
    >>> result = bootstrap_sharpe(returns, n_bootstrap=10000)
    >>> print(f"Sharpe: {result.sharpe_mean:.2f} "
    ...       f"({result.ci_lower:.2f}, {result.ci_upper:.2f})")
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial


# ========== 資料類別 ==========

@dataclass
class BootstrapResult:
    """Bootstrap Test 結果"""
    sharpe_mean: float           # Bootstrap 平均 Sharpe
    sharpe_std: float            # Bootstrap 標準差
    ci_lower: float              # 信賴區間下界
    ci_upper: float              # 信賴區間上界
    p_value: float               # H0: Sharpe <= 0 的 p 值
    confidence: float            # 信賴水準（例如 0.95）
    n_bootstrap: int             # Bootstrap 重複次數
    sharpe_distribution: np.ndarray  # 完整的 Sharpe 分布


@dataclass
class PermutationResult:
    """Permutation Test 結果"""
    actual_sharpe: float         # 實際策略的 Sharpe
    null_mean: float             # 虛無假設分布的平均
    null_std: float              # 虛無假設分布的標準差
    p_value: float               # 單尾 p 值
    is_significant: bool         # p < 0.05?
    n_permutations: int          # 置換次數
    null_distribution: np.ndarray  # 虛無假設的 Sharpe 分布


@dataclass
class StatisticalTestReport:
    """完整統計檢定報告"""
    # Bootstrap 結果
    bootstrap_sharpe: float
    bootstrap_ci: Tuple[float, float]
    bootstrap_p_value: float

    # Permutation Test 結果
    permutation_p_value: float

    # 綜合判斷
    is_statistically_significant: bool  # 兩項檢定都顯著
    confidence_level: float

    # 詳細結果（可選）
    bootstrap_result: Optional[BootstrapResult] = None
    permutation_result: Optional[PermutationResult] = None


# ========== 核心統計函數 ==========

def calculate_sharpe(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    計算年化 Sharpe Ratio

    Args:
        returns: 收益序列（日收益或其他週期）
        risk_free_rate: 無風險利率（年化）

    Returns:
        年化 Sharpe Ratio

    Notes:
        - 假設 returns 為日收益，使用 sqrt(252) 年化
        - 如果標準差為 0，回傳 0（避免除零）
    """
    if len(returns) == 0:
        return 0.0

    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)  # 使用樣本標準差

    # 處理零標準差（所有收益相同）
    if std_return == 0 or np.isclose(std_return, 0, atol=1e-10):
        return 0.0

    # 年化（假設日收益）
    sharpe = (mean_return - risk_free_rate / 252) / std_return * np.sqrt(252)

    # 防止數值溢出
    if not np.isfinite(sharpe):
        return 0.0

    return sharpe


def _bootstrap_worker(
    returns: np.ndarray,
    seed: int,
    risk_free_rate: float
) -> float:
    """Bootstrap 工作函數（用於多進程）"""
    rng = np.random.RandomState(seed)
    sample = rng.choice(returns, size=len(returns), replace=True)
    return calculate_sharpe(sample, risk_free_rate)


def _permutation_worker(
    returns: np.ndarray,
    seed: int,
    risk_free_rate: float
) -> float:
    """Permutation 工作函數（用於多進程）"""
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(returns)
    return calculate_sharpe(shuffled, risk_free_rate)


# ========== Bootstrap Test ==========

def bootstrap_sharpe(
    returns: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    risk_free_rate: float = 0.0,
    n_jobs: int = -1,
    random_state: Optional[int] = None
) -> BootstrapResult:
    """
    計算 Sharpe Ratio 的 Bootstrap 信賴區間

    透過重複抽樣估計 Sharpe Ratio 的抽樣分布，並計算信賴區間。

    Args:
        returns: 收益序列（1D numpy array）
        n_bootstrap: Bootstrap 重複次數（預設 10000）
        confidence: 信賴水準（預設 0.95）
        risk_free_rate: 無風險利率（年化，預設 0.0）
        n_jobs: 並行工作數（-1 表示使用所有 CPU）
        random_state: 隨機種子

    Returns:
        BootstrapResult 包含：
            - sharpe_mean: Bootstrap 平均 Sharpe
            - ci_lower, ci_upper: 信賴區間
            - p_value: H0: Sharpe <= 0 的單尾 p 值

    Example:
        >>> returns = np.random.randn(252) * 0.01  # 模擬一年的日收益
        >>> result = bootstrap_sharpe(returns, n_bootstrap=10000)
        >>> print(f"Sharpe: {result.sharpe_mean:.2f} "
        ...       f"95% CI: ({result.ci_lower:.2f}, {result.ci_upper:.2f})")
        >>> print(f"p-value: {result.p_value:.4f}")
    """
    returns = np.asarray(returns).flatten()

    if len(returns) < 2:
        raise ValueError("需要至少 2 個收益資料點")

    # 設定隨機種子
    if random_state is None:
        random_state = np.random.randint(0, 2**31 - 1)

    # 決定使用的 CPU 數量
    if n_jobs == -1:
        import os
        n_jobs = os.cpu_count() or 1
    else:
        n_jobs = max(1, n_jobs)

    # 產生種子序列（確保可重現性）
    rng = np.random.RandomState(random_state)
    seeds = rng.randint(0, 2**31 - 1, size=n_bootstrap)

    # 多進程 Bootstrap
    worker = partial(_bootstrap_worker, returns, risk_free_rate=risk_free_rate)

    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            sharpe_dist = np.array(list(executor.map(worker, seeds)))
    else:
        sharpe_dist = np.array([worker(seed) for seed in seeds])

    # 計算統計量
    sharpe_mean = float(np.mean(sharpe_dist))
    sharpe_std = float(np.std(sharpe_dist, ddof=1))

    # 信賴區間（百分位法）
    alpha = 1 - confidence
    ci_lower = float(np.percentile(sharpe_dist, 100 * alpha / 2))
    ci_upper = float(np.percentile(sharpe_dist, 100 * (1 - alpha / 2)))

    # p-value: H0: Sharpe <= 0
    p_value = float(np.mean(sharpe_dist <= 0))

    return BootstrapResult(
        sharpe_mean=sharpe_mean,
        sharpe_std=sharpe_std,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        sharpe_distribution=sharpe_dist
    )


# ========== Permutation Test ==========

def permutation_test(
    returns: np.ndarray,
    n_permutations: int = 10000,
    risk_free_rate: float = 0.0,
    n_jobs: int = -1,
    random_state: Optional[int] = None
) -> PermutationResult:
    """
    Permutation Test 檢定策略績效顯著性

    **注意**：標準 Permutation Test 對 Sharpe Ratio 無效，因為打亂順序不改變均值和標準差。
    此實作改為檢定「平均收益是否顯著大於 0」，透過隨機翻轉收益符號。

    Args:
        returns: 收益序列（1D numpy array）
        n_permutations: 置換次數（預設 10000）
        risk_free_rate: 無風險利率（年化，預設 0.0）
        n_jobs: 並行工作數（-1 表示使用所有 CPU）
        random_state: 隨機種子

    Returns:
        PermutationResult 包含：
            - actual_sharpe: 實際策略的 Sharpe
            - p_value: 單尾 p 值（H1: mean > 0）
            - is_significant: p < 0.05?

    Example:
        >>> returns = strategy.calculate_returns()
        >>> result = permutation_test(returns, n_permutations=10000)
        >>> if result.is_significant:
        ...     print(f"策略顯著優於隨機 (p={result.p_value:.4f})")
    """
    returns = np.asarray(returns).flatten()

    if len(returns) < 2:
        raise ValueError("需要至少 2 個收益資料點")

    # 計算實際 Sharpe
    actual_sharpe = calculate_sharpe(returns, risk_free_rate)

    # 設定隨機種子
    if random_state is None:
        random_state = np.random.randint(0, 2**31 - 1)

    rng = np.random.RandomState(random_state)

    # Permutation Test: 隨機翻轉收益符號
    # 虛無假設 H0: 收益的符號是隨機的（無預測能力）
    null_dist = np.zeros(n_permutations)

    for i in range(n_permutations):
        # 隨機翻轉符號
        signs = rng.choice([-1, 1], size=len(returns))
        permuted_returns = returns * signs
        null_dist[i] = calculate_sharpe(permuted_returns, risk_free_rate)

    # 計算 p-value（單尾檢定：H1: actual > null）
    p_value = float(np.mean(null_dist >= actual_sharpe))

    # 判斷顯著性（alpha = 0.05）
    is_significant = bool(p_value < 0.05)

    return PermutationResult(
        actual_sharpe=actual_sharpe,
        null_mean=float(np.mean(null_dist)),
        null_std=float(np.std(null_dist, ddof=1)),
        p_value=p_value,
        is_significant=is_significant,
        n_permutations=n_permutations,
        null_distribution=null_dist
    )


# ========== Block Bootstrap ==========

def block_bootstrap(
    returns: np.ndarray,
    block_size: int = 20,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    risk_free_rate: float = 0.0,
    random_state: Optional[int] = None
) -> BootstrapResult:
    """
    Block Bootstrap - 保留時間序列相關性的抽樣

    金融時間序列通常有自相關性（autocorrelation），標準 Bootstrap
    會破壞這種結構。Block Bootstrap 以連續區塊為單位抽樣，保留時間依賴性。

    Args:
        returns: 收益序列（1D numpy array）
        block_size: 區塊大小（預設 20，約 1 個月）
        n_bootstrap: Bootstrap 重複次數（預設 10000）
        confidence: 信賴水準（預設 0.95）
        risk_free_rate: 無風險利率（年化，預設 0.0）
        random_state: 隨機種子

    Returns:
        BootstrapResult（同 bootstrap_sharpe）

    Notes:
        - block_size 建議為自相關函數首次降至 0 的 lag
        - 對於日資料，通常設為 10-30 天

    Example:
        >>> returns = strategy.calculate_returns()
        >>> # 使用 20 天區塊（約 1 個月）
        >>> result = block_bootstrap(returns, block_size=20)
    """
    returns = np.asarray(returns).flatten()
    n = len(returns)

    if n < block_size:
        raise ValueError(f"資料長度 ({n}) 必須 >= block_size ({block_size})")

    # 設定隨機種子
    rng = np.random.RandomState(random_state)

    sharpe_dist = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # 計算需要多少個區塊
        n_blocks = int(np.ceil(n / block_size))

        # 隨機選擇區塊起點
        block_starts = rng.randint(0, n - block_size + 1, size=n_blocks)

        # 組合區塊
        blocks = [returns[start:start + block_size] for start in block_starts]
        sample = np.concatenate(blocks)[:n]  # 截斷到原始長度

        # 計算 Sharpe
        sharpe_dist[i] = calculate_sharpe(sample, risk_free_rate)

    # 計算統計量（同 bootstrap_sharpe）
    sharpe_mean = float(np.mean(sharpe_dist))
    sharpe_std = float(np.std(sharpe_dist, ddof=1))

    alpha = 1 - confidence
    ci_lower = float(np.percentile(sharpe_dist, 100 * alpha / 2))
    ci_upper = float(np.percentile(sharpe_dist, 100 * (1 - alpha / 2)))

    p_value = float(np.mean(sharpe_dist <= 0))

    return BootstrapResult(
        sharpe_mean=sharpe_mean,
        sharpe_std=sharpe_std,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        sharpe_distribution=sharpe_dist
    )


# ========== 整合報告 ==========

def run_statistical_tests(
    returns: np.ndarray,
    n_bootstrap: int = 10000,
    n_permutations: int = 10000,
    confidence: float = 0.95,
    use_block_bootstrap: bool = False,
    block_size: int = 20,
    risk_free_rate: float = 0.0,
    n_jobs: int = -1,
    random_state: Optional[int] = None
) -> StatisticalTestReport:
    """
    執行完整的統計檢定流程

    同時執行 Bootstrap Test 和 Permutation Test，產生綜合報告。

    Args:
        returns: 收益序列
        n_bootstrap: Bootstrap 重複次數
        n_permutations: Permutation 次數
        confidence: 信賴水準
        use_block_bootstrap: 是否使用 Block Bootstrap（預設 False）
        block_size: Block Bootstrap 區塊大小（僅在 use_block_bootstrap=True 時使用）
        risk_free_rate: 無風險利率（年化）
        n_jobs: 並行工作數
        random_state: 隨機種子

    Returns:
        StatisticalTestReport 包含：
            - bootstrap_sharpe, bootstrap_ci, bootstrap_p_value
            - permutation_p_value
            - is_statistically_significant（兩項檢定都通過）

    Example:
        >>> returns = strategy.calculate_returns()
        >>> report = run_statistical_tests(returns)
        >>> if report.is_statistically_significant:
        ...     print("策略通過統計檢定 ✓")
        ...     print(f"Bootstrap Sharpe: {report.bootstrap_sharpe:.2f} "
        ...           f"{report.bootstrap_ci}")
        ...     print(f"Permutation p-value: {report.permutation_p_value:.4f}")
    """
    # 1. Bootstrap Test
    if use_block_bootstrap:
        bootstrap_result = block_bootstrap(
            returns,
            block_size=block_size,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            risk_free_rate=risk_free_rate,
            random_state=random_state
        )
    else:
        bootstrap_result = bootstrap_sharpe(
            returns,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            risk_free_rate=risk_free_rate,
            n_jobs=n_jobs,
            random_state=random_state
        )

    # 2. Permutation Test
    permutation_result = permutation_test(
        returns,
        n_permutations=n_permutations,
        risk_free_rate=risk_free_rate,
        n_jobs=n_jobs,
        random_state=random_state
    )

    # 3. 綜合判斷（兩項檢定都顯著）
    bootstrap_significant = bootstrap_result.p_value < 0.05
    permutation_significant = permutation_result.is_significant
    is_significant = bootstrap_significant and permutation_significant

    return StatisticalTestReport(
        bootstrap_sharpe=bootstrap_result.sharpe_mean,
        bootstrap_ci=(bootstrap_result.ci_lower, bootstrap_result.ci_upper),
        bootstrap_p_value=bootstrap_result.p_value,
        permutation_p_value=permutation_result.p_value,
        is_statistically_significant=is_significant,
        confidence_level=confidence,
        bootstrap_result=bootstrap_result,
        permutation_result=permutation_result
    )


# ========== 輔助函數 ==========

def print_test_report(report: StatisticalTestReport) -> None:
    """
    美化輸出統計檢定報告

    Example:
        >>> report = run_statistical_tests(returns)
        >>> print_test_report(report)
    """
    print("=" * 60)
    print("統計檢定報告".center(60))
    print("=" * 60)

    # Bootstrap 結果
    print("\n[Bootstrap Test]")
    print(f"  Sharpe Ratio: {report.bootstrap_sharpe:.3f}")
    print(f"  {int(report.confidence_level * 100)}% 信賴區間: "
          f"({report.bootstrap_ci[0]:.3f}, {report.bootstrap_ci[1]:.3f})")
    print(f"  p-value (H0: Sharpe ≤ 0): {report.bootstrap_p_value:.4f}")

    # Permutation 結果
    print("\n[Permutation Test]")
    if report.permutation_result:
        print(f"  實際 Sharpe: {report.permutation_result.actual_sharpe:.3f}")
        print(f"  虛無假設平均: {report.permutation_result.null_mean:.3f}")
        print(f"  p-value (H1: Sharpe > 隨機): {report.permutation_p_value:.4f}")
    else:
        print(f"  p-value: {report.permutation_p_value:.4f}")

    # 綜合結論
    print("\n" + "=" * 60)
    if report.is_statistically_significant:
        print("結論: 策略具有統計顯著性 ✓".center(60))
    else:
        print("結論: 策略缺乏統計顯著性 ✗".center(60))
    print("=" * 60)
