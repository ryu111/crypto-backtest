"""信號過濾器模組

提供多層過濾器管道，用於過濾假信號。

過濾器類型：
- BaseSignalFilter: 過濾器基礎類別
- SignalStrengthFilter: 信號強度過濾器
- ConfirmationFilter: 多指標確認過濾器
- TimeFilter: 時間過濾器
- VolumeFilter: 成交量過濾器
- FilterPipeline: 過濾管道（串聯多個過濾器）

使用範例：
    # 建立預設過濾管道
    pipeline = FilterPipeline.create_default()

    # 執行過濾
    filtered_long, long_exit, filtered_short, short_exit = pipeline.process(
        data, long_entry, long_exit, short_entry, short_exit
    )

    # 查看統計
    print(pipeline.get_summary())

    # 自定義管道
    pipeline = FilterPipeline()
    pipeline.add_filter(TimeFilter())
    pipeline.add_filter(SignalStrengthFilter(min_rsi_distance=10.0))

    # 使用工廠函數建立過濾器
    filter = create_filter('strength', {
        'min_rsi_distance': 5.0,
        'min_macd_strength': 0.002
    })
"""

from typing import Dict, Type, Optional
from .base_filter import BaseSignalFilter
from .strength_filter import SignalStrengthFilter
from .confirmation_filter import ConfirmationFilter
from .time_filter import TimeFilter
from .volume_filter import VolumeFilter
from .pipeline import FilterPipeline


# 過濾器註冊表
FILTER_REGISTRY: Dict[str, Type[BaseSignalFilter]] = {
    'strength': SignalStrengthFilter,
    'confirmation': ConfirmationFilter,
    'time': TimeFilter,
    'volume': VolumeFilter,
}


def create_filter(name: str, config: Optional[Dict] = None) -> BaseSignalFilter:
    """工廠函數建立過濾器

    Args:
        name: 過濾器名稱（'strength', 'confirmation', 'time', 'volume'）
        config: 過濾器配置參數

    Returns:
        建立的過濾器實例

    Raises:
        ValueError: 如果過濾器名稱不存在

    Examples:
        >>> # 建立強度過濾器
        >>> filter = create_filter('strength', {
        ...     'min_rsi_distance': 5.0,
        ...     'min_macd_strength': 0.002
        ... })

        >>> # 建立時間過濾器
        >>> filter = create_filter('time', {
        ...     'avoid_funding_hours': [0, 8, 16],
        ...     'funding_buffer_minutes': 30
        ... })
    """
    if name not in FILTER_REGISTRY:
        available = ', '.join(FILTER_REGISTRY.keys())
        raise ValueError(f"未知的過濾器: {name}. 可用的過濾器: {available}")

    filter_class = FILTER_REGISTRY[name]
    config = config or {}

    return filter_class(**config)


def list_available_filters() -> Dict[str, str]:
    """列出所有可用的過濾器

    Returns:
        過濾器名稱到描述的映射
    """
    return {
        'strength': '信號強度過濾器 - 只保留強信號',
        'confirmation': '確認過濾器 - 要求多指標確認',
        'time': '時間過濾器 - 避開特定時段',
        'volume': '成交量過濾器 - 只在高成交量時交易',
    }


__all__ = [
    # 基礎類別
    'BaseSignalFilter',

    # 具體過濾器
    'SignalStrengthFilter',
    'ConfirmationFilter',
    'TimeFilter',
    'VolumeFilter',

    # 過濾管道
    'FilterPipeline',

    # 工廠函數
    'create_filter',
    'list_available_filters',

    # 過濾器註冊表
    'FILTER_REGISTRY',
]
