"""
UI 輔助模組

整合 helpers 和 data_loader 的所有功能。
"""

# 從 helpers.py 導入（原 utils.py 的功能）
from .helpers import (
    load_experiments,
    calculate_summary_stats,
    get_latest_experiments,
    format_percentage,
    format_sharpe,
    grade_color,
    format_timestamp,
    get_data_source_status,
    render_sidebar_navigation,
    render_page_header,
)

# 從 data_loader.py 導入
from .data_loader import (
    load_experiment_data,
    load_equity_curve,
    load_daily_returns,
    load_trades,
    calculate_monthly_returns,
    get_all_experiments,
    get_best_experiments,
)

__all__ = [
    # helpers
    'load_experiments',
    'calculate_summary_stats',
    'get_latest_experiments',
    'format_percentage',
    'format_sharpe',
    'grade_color',
    'format_timestamp',
    'get_data_source_status',
    'render_sidebar_navigation',
    'render_page_header',
    # data_loader
    'load_experiment_data',
    'load_equity_curve',
    'load_daily_returns',
    'load_trades',
    'calculate_monthly_returns',
    'get_all_experiments',
    'get_best_experiments',
]
