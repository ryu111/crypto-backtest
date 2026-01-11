"""
策略註冊表

提供策略的註冊、查詢、實例化功能。
"""

from typing import Dict, Type, List, Optional
from .base import BaseStrategy


class StrategyRegistry:
    """
    策略註冊表

    提供裝飾器註冊策略，並支援查詢與實例化。

    Example:
        @StrategyRegistry.register('my_strategy')
        class MyStrategy(BaseStrategy):
            ...

        # 取得策略
        strategy_class = StrategyRegistry.get('my_strategy')
        strategy = strategy_class(param1=10)

        # 列出所有策略
        all_strategies = StrategyRegistry.list_all()
    """

    _strategies: Dict[str, Type[BaseStrategy]] = {}

    @classmethod
    def register(cls, name: str):
        """
        策略註冊裝飾器

        Args:
            name: 策略唯一識別名稱

        Returns:
            decorator: 裝飾器函數

        Example:
            @StrategyRegistry.register('ma_cross')
            class MACrossStrategy(BaseStrategy):
                ...
        """
        def decorator(strategy_class: Type[BaseStrategy]):
            if not issubclass(strategy_class, BaseStrategy):
                raise TypeError(
                    f"{strategy_class.__name__} must inherit from BaseStrategy"
                )

            if name in cls._strategies:
                raise ValueError(
                    f"Strategy '{name}' is already registered. "
                    f"Use a different name or unregister first."
                )

            cls._strategies[name] = strategy_class

            # 設定策略名稱
            if not hasattr(strategy_class, 'name') or strategy_class.name == "base_strategy":
                strategy_class.name = name

            return strategy_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseStrategy]:
        """
        取得已註冊策略

        Args:
            name: 策略名稱

        Returns:
            Type[BaseStrategy]: 策略類別

        Raises:
            KeyError: 策略不存在
        """
        if name not in cls._strategies:
            raise KeyError(
                f"Strategy '{name}' not found. "
                f"Available: {list(cls._strategies.keys())}"
            )
        return cls._strategies[name]

    @classmethod
    def list_all(cls) -> List[str]:
        """
        列出所有已註冊策略名稱

        Returns:
            List[str]: 策略名稱列表
        """
        return list(cls._strategies.keys())

    @classmethod
    def list_by_type(cls, strategy_type: str) -> List[str]:
        """
        列出指定類型的策略

        Args:
            strategy_type: 策略類型 (trend, momentum, mean_reversion, etc.)

        Returns:
            List[str]: 符合類型的策略名稱列表
        """
        return [
            name for name, strategy_class in cls._strategies.items()
            if strategy_class.strategy_type == strategy_type
        ]

    @classmethod
    def get_param_space(cls, name: str) -> Dict:
        """
        取得策略的參數優化空間

        Args:
            name: 策略名稱

        Returns:
            dict: 參數空間定義

        Example:
            {
                'fast_period': {'type': 'int', 'low': 5, 'high': 20},
                'slow_period': {'type': 'int', 'low': 20, 'high': 100},
                'stop_loss_atr': {'type': 'float', 'low': 1.0, 'high': 3.0}
            }
        """
        strategy_class = cls.get(name)
        return strategy_class.param_space

    @classmethod
    def get_info(cls, name: str) -> Dict:
        """
        取得策略詳細資訊

        Args:
            name: 策略名稱

        Returns:
            dict: 策略資訊
        """
        strategy_class = cls.get(name)
        return {
            'name': strategy_class.name,
            'type': strategy_class.strategy_type,
            'version': strategy_class.version,
            'description': strategy_class.description,
            'params': strategy_class.params,
            'param_space': strategy_class.param_space
        }

    @classmethod
    def get_all_info(cls) -> Dict[str, Dict]:
        """
        取得所有策略資訊

        Returns:
            dict: {策略名稱: 策略資訊}
        """
        return {
            name: cls.get_info(name)
            for name in cls.list_all()
        }

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseStrategy:
        """
        實例化策略

        Args:
            name: 策略名稱
            **kwargs: 策略參數

        Returns:
            BaseStrategy: 策略實例

        Example:
            strategy = StrategyRegistry.create(
                'ma_cross',
                fast_period=10,
                slow_period=30
            )
        """
        strategy_class = cls.get(name)
        return strategy_class(**kwargs)

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        移除已註冊策略

        Args:
            name: 策略名稱

        Raises:
            KeyError: 策略不存在
        """
        if name not in cls._strategies:
            raise KeyError(f"Strategy '{name}' not found")
        del cls._strategies[name]

    @classmethod
    def clear(cls) -> None:
        """清空所有已註冊策略"""
        cls._strategies.clear()

    @classmethod
    def exists(cls, name: str) -> bool:
        """
        檢查策略是否存在

        Args:
            name: 策略名稱

        Returns:
            bool: 策略是否已註冊
        """
        return name in cls._strategies

    @classmethod
    def validate_param_space(cls, name: str) -> bool:
        """
        驗證參數空間定義是否有效

        Args:
            name: 策略名稱

        Returns:
            bool: 參數空間是否有效
        """
        param_space = cls.get_param_space(name)

        if not param_space:
            return True  # 空參數空間視為有效

        # 驗證每個參數定義
        for param_name, param_config in param_space.items():
            if not isinstance(param_config, dict):
                return False

            # 必須有 type 欄位
            if 'type' not in param_config:
                return False

            param_type = param_config['type']

            # 驗證數值型參數
            if param_type in ['int', 'float']:
                if 'low' not in param_config or 'high' not in param_config:
                    return False
                if param_config['low'] >= param_config['high']:
                    return False

            # 驗證類別型參數
            elif param_type == 'categorical':
                if 'choices' not in param_config:
                    return False
                if not param_config['choices']:
                    return False

        return True

    @classmethod
    def get_strategy_count(cls) -> int:
        """
        取得已註冊策略總數

        Returns:
            int: 策略數量
        """
        return len(cls._strategies)

    @classmethod
    def get_type_counts(cls) -> Dict[str, int]:
        """
        取得各類型策略數量

        Returns:
            dict: {策略類型: 數量}
        """
        type_counts = {}
        for strategy_class in cls._strategies.values():
            strategy_type = strategy_class.strategy_type
            type_counts[strategy_type] = type_counts.get(strategy_type, 0) + 1
        return type_counts


# 便利函數

def register_strategy(name: str):
    """
    策略註冊裝飾器（便利函數）

    這是 StrategyRegistry.register 的別名。

    Example:
        from src.strategies import register_strategy

        @register_strategy('my_strategy')
        class MyStrategy(BaseStrategy):
            ...
    """
    return StrategyRegistry.register(name)


def get_strategy(name: str) -> Type[BaseStrategy]:
    """
    取得策略類別（便利函數）

    Args:
        name: 策略名稱

    Returns:
        Type[BaseStrategy]: 策略類別
    """
    return StrategyRegistry.get(name)


def list_strategies() -> List[str]:
    """
    列出所有策略（便利函數）

    Returns:
        List[str]: 策略名稱列表
    """
    return StrategyRegistry.list_all()


def create_strategy(name: str, **kwargs) -> BaseStrategy:
    """
    建立策略實例（便利函數）

    Args:
        name: 策略名稱
        **kwargs: 策略參數

    Returns:
        BaseStrategy: 策略實例
    """
    return StrategyRegistry.create(name, **kwargs)
