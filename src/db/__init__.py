"""
DuckDB 資料層模組

提供實驗記錄和策略統計的資料庫操作介面。

使用範例:
    from src.db import Repository, QueryFilters

    # 初始化資料庫
    repo = Repository("data/experiments.duckdb")

    # 查詢實驗
    filters = QueryFilters(strategy_name="ma_cross", min_sharpe=1.5)
    experiments = repo.query_experiments(filters)

    # 獲取最佳結果
    top_10 = repo.get_best_experiments(metric="sharpe_ratio", n=10)
"""

from .repository import Repository, QueryFilters

__all__ = [
    'Repository',
    'QueryFilters',
]
