"""
策略驗證模組

提供 5 階段驗證系統和相關驗證工具。
"""

# Monte Carlo 模擬（無額外依賴）
from .monte_carlo import (
    MonteCarloSimulator,
    MonteCarloResult
)

# 統計檢定（無額外依賴）
from .statistical_tests import (
    BootstrapResult,
    PermutationResult,
    StatisticalTestReport,
    bootstrap_sharpe,
    permutation_test,
    block_bootstrap,
    run_statistical_tests,
    print_test_report,
    calculate_sharpe,
)

# Deflated Sharpe Ratio（無額外依賴）
from .sharpe_correction import (
    DeflatedSharpeResult,
    PBOResult,
    MinimumBacktestLength,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    minimum_backtest_length,
    calculate_sharpe_variance,
    expected_maximum_sharpe,
    print_deflated_sharpe_report,
    print_pbo_report,
)

# Walk-Forward 分析和 Combinatorial Purged CV（無額外依賴）
from .walk_forward import (
    PurgedKFold,
    CombinatorialPurgedCV,
    PurgedFoldResult,
    CombinatorialCVResult,
    combinatorial_purged_cv
)

# 黑天鵝壓力測試（無額外依賴）
from .stress_test import (
    StressTester,
    StressTestResult,
    StressTestReport,
    HISTORICAL_EVENTS,
)

# 延遲導入 stages（避免 vectorbt 依賴問題）
def _import_stages():
    """延遲導入 5 階段驗證系統"""
    from .stages import (
        StageValidator,
        ValidationResult,
        ValidationGrade,
        StageResult,
    )
    return StageValidator, ValidationResult, ValidationGrade, StageResult


# 方便的 getter
def get_stage_validator():
    """獲取 StageValidator（延遲導入）"""
    StageValidator, _, _, _ = _import_stages()
    return StageValidator


__all__ = [
    # Monte Carlo 模擬
    'MonteCarloSimulator',
    'MonteCarloResult',
    # 統計檢定
    'BootstrapResult',
    'PermutationResult',
    'StatisticalTestReport',
    'bootstrap_sharpe',
    'permutation_test',
    'block_bootstrap',
    'run_statistical_tests',
    'print_test_report',
    'calculate_sharpe',
    # Deflated Sharpe Ratio
    'DeflatedSharpeResult',
    'PBOResult',
    'MinimumBacktestLength',
    'deflated_sharpe_ratio',
    'probability_of_backtest_overfitting',
    'minimum_backtest_length',
    'calculate_sharpe_variance',
    'expected_maximum_sharpe',
    'print_deflated_sharpe_report',
    'print_pbo_report',
    # Walk-Forward 和 Combinatorial Purged CV
    'PurgedKFold',
    'CombinatorialPurgedCV',
    'PurgedFoldResult',
    'CombinatorialCVResult',
    'combinatorial_purged_cv',
    # 黑天鵝壓力測試
    'StressTester',
    'StressTestResult',
    'StressTestReport',
    'HISTORICAL_EVENTS',
    # 延遲導入 helper
    'get_stage_validator',
]

# 嘗試導入 stages（如果依賴可用）
try:
    from .stages import (
        StageValidator,
        ValidationResult,
        ValidationGrade,
        StageResult,
    )
    __all__.extend([
        'StageValidator',
        'ValidationResult',
        'ValidationGrade',
        'StageResult',
    ])
except ImportError:
    # 如果 vectorbt 等依賴不可用，只提供 Monte Carlo
    pass
