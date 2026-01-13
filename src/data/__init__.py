"""
資料模組

提供資料獲取、清洗、驗證、共享功能
"""

from .fetcher import DataFetcher
from .cleaner import DataCleaner, DataQualityReport, GapInfo, GapFillStrategy
from .shared_pool import SharedDataPool, SharedDataInfo, create_shared_pool, attach_to_pool

__all__ = [
    'DataFetcher',
    'DataCleaner',
    'DataQualityReport',
    'GapInfo',
    'GapFillStrategy',
    'SharedDataPool',
    'SharedDataInfo',
    'create_shared_pool',
    'attach_to_pool'
]
