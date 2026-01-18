"""
策略註冊表

提供策略的註冊、查詢、實例化功能。
"""

from typing import Dict, Type, List, Optional, Any, TYPE_CHECKING, ClassVar
from .base import BaseStrategy

if TYPE_CHECKING:
    from ..automation.gp_integration import DynamicStrategyInfo


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

    _strategies: ClassVar[Dict[str, Type[BaseStrategy]]] = {}
    _dynamic_strategies: ClassVar[Dict[str, 'DynamicStrategyInfo']] = {}

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
            # 使用共用方法註冊
            cls._validate_and_register(name, strategy_class)
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
        # 先查找靜態註冊
        if name in cls._strategies:
            return cls._strategies[name]

        # 再查找動態註冊
        if name in cls._dynamic_strategies:
            return cls._dynamic_strategies[name].strategy_class

        # 策略不存在
        all_strategies = list(cls._strategies.keys()) + list(cls._dynamic_strategies.keys())
        raise KeyError(
            f"Strategy '{name}' not found. "
            f"Available: {all_strategies}"
        )

    @classmethod
    def list_all(cls, include_dynamic: bool = True) -> List[str]:
        """
        列出所有已註冊策略名稱

        Args:
            include_dynamic: 是否包含動態註冊的策略（預設 True）

        Returns:
            List[str]: 策略名稱列表
        """
        strategies = list(cls._strategies.keys())
        if include_dynamic:
            strategies.extend(cls._dynamic_strategies.keys())
        return strategies

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

    # ========== GP 策略支援 ==========

    @classmethod
    def _validate_and_register(
        cls,
        name: str,
        strategy_class: Type[BaseStrategy]
    ) -> None:
        """
        驗證並註冊策略（共用方法）

        Args:
            name: 策略名稱
            strategy_class: 策略類別

        Raises:
            TypeError: 如果不是 BaseStrategy 子類別
            ValueError: 如果策略名稱已存在
        """
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

    @classmethod
    def register_gp_strategy(
        cls,
        name: str,
        strategy_class: Type[BaseStrategy],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        註冊 GP 演化的策略

        Args:
            name: 策略名稱
            strategy_class: 策略類別
            metadata: 演化元資料（適應度、代數等）
                - fitness: 適應度分數
                - generation: 演化代數
                - expression: GP 表達式
                - evolved_at: 演化時間

        Raises:
            TypeError: 如果不是 BaseStrategy 子類別
            ValueError: 如果策略名稱已存在

        Example:
            >>> StrategyRegistry.register_gp_strategy(
            ...     'gp_evolved_001',
            ...     EvolvedStrategy001,
            ...     metadata={'fitness': 1.85, 'generation': 30}
            ... )
        """
        # 使用共用方法註冊
        cls._validate_and_register(name, strategy_class)

        # 儲存 GP 元資料（使用私有屬性）
        if metadata:
            cls._store_gp_metadata(name, metadata)

    @classmethod
    def _store_gp_metadata(cls, name: str, metadata: Dict[str, Any]) -> None:
        """
        儲存 GP 策略元資料

        Args:
            name: 策略名稱
            metadata: 元資料字典
        """
        if not hasattr(cls, '_gp_metadata'):
            cls._gp_metadata: Dict[str, Dict[str, Any]] = {}

        cls._gp_metadata[name] = metadata

    @classmethod
    def list_gp_strategies(cls) -> List[str]:
        """
        列出所有 GP 演化的策略

        Returns:
            List[str]: GP 策略名稱列表

        Note:
            識別方式：
            1. 檢查是否有 GP 元資料
            2. 策略類型為 'gp_evolved'
            3. 策略名稱前綴為 'gp_' 或 'evolved_'
        """
        gp_strategies = []

        for name, strategy_class in cls._strategies.items():
            # 方法 1: 檢查元資料
            if hasattr(cls, '_gp_metadata') and name in cls._gp_metadata:
                gp_strategies.append(name)
                continue

            # 方法 2: 檢查策略類型
            if hasattr(strategy_class, 'strategy_type') and \
               strategy_class.strategy_type == 'gp_evolved':
                gp_strategies.append(name)
                continue

            # 方法 3: 檢查名稱前綴
            if name.startswith('gp_') or name.startswith('evolved_'):
                gp_strategies.append(name)

        return gp_strategies

    @classmethod
    def get_strategy_metadata(cls, name: str) -> Optional[Dict[str, Any]]:
        """
        取得策略元資料

        Args:
            name: 策略名稱

        Returns:
            Optional[Dict]: 元資料，如果無則返回 None
                - fitness: 適應度分數
                - generation: 演化代數
                - expression: GP 表達式
                - evolved_at: 演化時間

        Example:
            >>> metadata = StrategyRegistry.get_strategy_metadata('gp_evolved_001')
            >>> print(metadata['fitness'])  # 1.85
        """
        if not hasattr(cls, '_gp_metadata'):
            return None

        return cls._gp_metadata.get(name)

    # ========== 動態策略註冊 ==========

    @classmethod
    def register_dynamic(
        cls,
        info: 'DynamicStrategyInfo',
        replace_if_exists: bool = False
    ) -> bool:
        """
        動態註冊策略

        Args:
            info: 動態策略資訊
            replace_if_exists: 是否覆蓋已存在的同名策略（預設 False）

        Returns:
            bool: 是否成功註冊

        Raises:
            ValueError: 策略名稱已存在且不允許覆蓋
            TypeError: strategy_class 不是 BaseStrategy 子類

        Example:
            >>> from automation.gp_integration import DynamicStrategyInfo
            >>> info = DynamicStrategyInfo(
            ...     name='gp_evolved_001',
            ...     strategy_class=GPEvolvedStrategy,
            ...     expression='...',
            ...     fitness=2.35,
            ...     generation=10,
            ...     created_at=datetime.utcnow()
            ... )
            >>> StrategyRegistry.register_dynamic(info)
            True
        """
        # 驗證 strategy_class
        if not issubclass(info.strategy_class, BaseStrategy):
            raise TypeError(
                f"{info.strategy_class.__name__} must inherit from BaseStrategy"
            )

        # 檢查名稱衝突
        if not replace_if_exists:
            if info.name in cls._strategies:
                raise ValueError(
                    f"Strategy '{info.name}' already exists in static registry. "
                    f"Use replace_if_exists=True to override."
                )
            if info.name in cls._dynamic_strategies:
                raise ValueError(
                    f"Dynamic strategy '{info.name}' already exists. "
                    f"Use replace_if_exists=True to override."
                )

        # 註冊動態策略
        cls._dynamic_strategies[info.name] = info

        # 設定策略名稱
        if not hasattr(info.strategy_class, 'name') or info.strategy_class.name == "base_strategy":
            info.strategy_class.name = info.name

        return True

    @classmethod
    def unregister_dynamic(cls, name: str) -> bool:
        """
        移除動態註冊的策略

        Args:
            name: 策略名稱

        Returns:
            bool: 是否成功移除（True 表示成功，False 表示策略不存在）

        Example:
            >>> StrategyRegistry.unregister_dynamic('gp_evolved_001')
            True
        """
        if name in cls._dynamic_strategies:
            del cls._dynamic_strategies[name]
            return True
        return False

    @classmethod
    def list_dynamic(cls) -> List['DynamicStrategyInfo']:
        """
        列出所有動態註冊的策略

        Returns:
            List[DynamicStrategyInfo]: 動態策略資訊列表

        Example:
            >>> strategies = StrategyRegistry.list_dynamic()
            >>> for info in strategies:
            ...     print(f"{info.name}: fitness={info.fitness}")
        """
        return list(cls._dynamic_strategies.values())

    @classmethod
    def clear_dynamic(cls) -> int:
        """
        清除所有動態註冊的策略

        Returns:
            int: 清除的策略數量

        Example:
            >>> count = StrategyRegistry.clear_dynamic()
            >>> print(f"Cleared {count} dynamic strategies")
        """
        count = len(cls._dynamic_strategies)
        cls._dynamic_strategies.clear()
        return count

    @classmethod
    def is_dynamic(cls, name: str) -> bool:
        """
        檢查策略是否為動態註冊

        Args:
            name: 策略名稱

        Returns:
            bool: 是否為動態策略

        Example:
            >>> StrategyRegistry.is_dynamic('gp_evolved_001')
            True
            >>> StrategyRegistry.is_dynamic('ma_cross')
            False
        """
        return name in cls._dynamic_strategies

    @classmethod
    def get_dynamic_info(cls, name: str) -> Optional['DynamicStrategyInfo']:
        """
        取得動態策略的完整資訊

        Args:
            name: 策略名稱

        Returns:
            Optional[DynamicStrategyInfo]: 動態策略資訊，如果不是動態策略則返回 None

        Example:
            >>> info = StrategyRegistry.get_dynamic_info('gp_evolved_001')
            >>> if info:
            ...     print(f"Fitness: {info.fitness}, Generation: {info.generation}")
        """
        return cls._dynamic_strategies.get(name)


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
