"""
趨勢策略模組

提供基於趨勢跟隨的交易策略。

可用策略：
- MACrossStrategy: 雙均線交叉策略
- SupertrendStrategy: Supertrend 指標策略
- DonchianStrategy: Donchian Channel 突破策略
"""

from .ma_cross import MACrossStrategy
from .supertrend import SupertrendStrategy
from .donchian import DonchianStrategy

__all__ = [
    'MACrossStrategy',
    'SupertrendStrategy',
    'DonchianStrategy',
]
