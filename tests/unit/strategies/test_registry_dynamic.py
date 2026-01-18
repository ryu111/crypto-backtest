"""
測試 StrategyRegistry 動態策略註冊功能

測試項目：
1. register_dynamic() - 動態策略註冊
2. unregister_dynamic() - 移除動態策略
3. list_dynamic() - 列出動態策略
4. clear_dynamic() - 清除所有動態策略
5. is_dynamic() - 檢查是否為動態策略
6. get_dynamic_info() - 取得動態策略資訊
7. get() 整合測試 - 能查到動態策略
8. list_all() 整合測試 - 包含/不包含動態策略
"""

import pytest
from datetime import datetime
from typing import Dict, Any
from dataclasses import dataclass, field

from src.strategies.registry import StrategyRegistry
from src.strategies.base import BaseStrategy


# ===== Test Fixtures and Helpers =====

@dataclass
class MockDynamicStrategyInfo:
    """模擬 DynamicStrategyInfo，避免循環依賴"""
    name: str
    strategy_class: type
    expression: str = "test_expression"
    fitness: float = 1.85
    generation: int = 10
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TestStrategy(BaseStrategy):
    """測試用策略"""
    name = "test_strategy"
    strategy_type = "test"
    version = "1.0"
    description = "Test strategy"
    params = {"param1": 10}
    param_space = {"param1": {"type": "int", "low": 1, "high": 100}}

    def calculate_indicators(self, data):
        return {}

    def generate_signals(self, indicators):
        return {}

    def validate_params(self):
        return True


class AnotherTestStrategy(BaseStrategy):
    """另一個測試策略"""
    name = "another_test_strategy"
    strategy_type = "test"
    version = "1.0"
    description = "Another test strategy"
    params = {"param2": 20}
    param_space = {"param2": {"type": "int", "low": 1, "high": 200}}

    def calculate_indicators(self, data):
        return {}

    def generate_signals(self, indicators):
        return {}

    def validate_params(self):
        return True


@pytest.fixture(autouse=True)
def cleanup_registry():
    """在每個測試前後清理動態策略註冊表和臨時靜態策略"""
    # 記錄原始的靜態策略
    original_static = StrategyRegistry._strategies.copy()
    
    # 清除動態策略
    StrategyRegistry.clear_dynamic()
    
    yield
    
    # 清除動態策略
    StrategyRegistry.clear_dynamic()
    
    # 移除測試期間添加的臨時靜態策略
    temp_strategies = [
        "static_001", "static_002", "test_static", "test_001"
    ]
    for name in temp_strategies:
        if name in StrategyRegistry._strategies:
            del StrategyRegistry._strategies[name]


@pytest.fixture
def dynamic_strategy_info():
    """建立測試用的動態策略資訊"""
    return MockDynamicStrategyInfo(
        name="gp_evolved_001",
        strategy_class=TestStrategy,
        expression="and(gt(rsi(14), 50), lt(rsi(14), 70))",
        fitness=2.35,
        generation=10,
        metadata={"parent_ids": ["gp_gen_09_005"], "mutation_type": "crossover"}
    )


# ===== Test register_dynamic() =====

class TestRegisterDynamic:
    """測試 register_dynamic() 方法"""

    def test_register_new_dynamic_strategy(self, dynamic_strategy_info):
        """成功註冊新動態策略"""
        result = StrategyRegistry.register_dynamic(dynamic_strategy_info)

        assert result is True
        assert StrategyRegistry.is_dynamic("gp_evolved_001")
        assert StrategyRegistry.get_dynamic_info("gp_evolved_001") == dynamic_strategy_info

    def test_register_invalid_strategy_class(self):
        """註冊非 BaseStrategy 子類應該拋出 TypeError"""
        class InvalidStrategy:
            pass

        info = MockDynamicStrategyInfo(
            name="invalid_001",
            strategy_class=InvalidStrategy
        )

        with pytest.raises(TypeError, match="must inherit from BaseStrategy"):
            StrategyRegistry.register_dynamic(info)

    def test_register_duplicate_without_replace(self, dynamic_strategy_info):
        """重複註冊同名策略應該拋出 ValueError"""
        # 第一次註冊成功
        StrategyRegistry.register_dynamic(dynamic_strategy_info)

        # 第二次不允許覆蓋應該失敗
        with pytest.raises(ValueError, match="already exists"):
            StrategyRegistry.register_dynamic(dynamic_strategy_info, replace_if_exists=False)

    def test_register_duplicate_with_replace(self, dynamic_strategy_info):
        """使用 replace_if_exists=True 應該覆蓋成功"""
        # 第一次註冊
        StrategyRegistry.register_dynamic(dynamic_strategy_info)

        # 修改策略資訊
        updated_info = MockDynamicStrategyInfo(
            name="gp_evolved_001",
            strategy_class=AnotherTestStrategy,
            fitness=2.50,
            generation=15
        )

        # 第二次註冊應該成功
        result = StrategyRegistry.register_dynamic(updated_info, replace_if_exists=True)

        assert result is True
        retrieved = StrategyRegistry.get_dynamic_info("gp_evolved_001")
        assert retrieved.fitness == 2.50
        assert retrieved.generation == 15

    def test_register_conflict_with_static_strategy(self):
        """與靜態策略衝突（沒有 replace_if_exists）"""
        # 先手動註冊靜態策略
        StrategyRegistry._strategies["test_static"] = TestStrategy

        info = MockDynamicStrategyInfo(
            name="test_static",
            strategy_class=AnotherTestStrategy
        )

        # 應該拋出 ValueError
        with pytest.raises(ValueError, match="already exists in static registry"):
            StrategyRegistry.register_dynamic(info)

    def test_register_conflict_with_static_strategy_with_replace(self):
        """與靜態策略衝突但使用 replace_if_exists=True"""
        # 先手動註冊靜態策略
        StrategyRegistry._strategies["test_static"] = TestStrategy

        info = MockDynamicStrategyInfo(
            name="test_static",
            strategy_class=AnotherTestStrategy
        )

        # 應該成功註冊到動態註冊表
        result = StrategyRegistry.register_dynamic(info, replace_if_exists=True)

        assert result is True
        assert StrategyRegistry.is_dynamic("test_static")

    def test_register_sets_strategy_name(self, dynamic_strategy_info):
        """註冊時應設定策略類別的 name"""
        # 創建沒有名稱的策略類
        class UnnamedStrategy(BaseStrategy):
            name = "base_strategy"  # 預設值
            strategy_type = "test"
            def calculate_indicators(self, data): return {}
            def generate_signals(self, indicators): return {}
            def validate_params(self): return True

        info = MockDynamicStrategyInfo(
            name="gp_evolved_002",
            strategy_class=UnnamedStrategy
        )

        StrategyRegistry.register_dynamic(info)

        # 名稱應該被設定為註冊的名稱
        assert info.strategy_class.name == "gp_evolved_002"

    def test_register_multiple_strategies(self):
        """成功註冊多個動態策略"""
        info1 = MockDynamicStrategyInfo(
            name="gp_001",
            strategy_class=TestStrategy,
            fitness=1.8
        )
        info2 = MockDynamicStrategyInfo(
            name="gp_002",
            strategy_class=AnotherTestStrategy,
            fitness=2.1
        )

        StrategyRegistry.register_dynamic(info1)
        StrategyRegistry.register_dynamic(info2)

        assert len(StrategyRegistry.list_dynamic()) == 2
        assert StrategyRegistry.is_dynamic("gp_001")
        assert StrategyRegistry.is_dynamic("gp_002")


# ===== Test unregister_dynamic() =====

class TestUnregisterDynamic:
    """測試 unregister_dynamic() 方法"""

    def test_unregister_existing_strategy(self, dynamic_strategy_info):
        """成功移除已存在的動態策略"""
        StrategyRegistry.register_dynamic(dynamic_strategy_info)

        result = StrategyRegistry.unregister_dynamic("gp_evolved_001")

        assert result is True
        assert not StrategyRegistry.is_dynamic("gp_evolved_001")

    def test_unregister_nonexistent_strategy(self):
        """移除不存在的策略應返回 False"""
        result = StrategyRegistry.unregister_dynamic("nonexistent_strategy")

        assert result is False

    def test_unregister_after_clear(self, dynamic_strategy_info):
        """清除後再嘗試移除應返回 False"""
        StrategyRegistry.register_dynamic(dynamic_strategy_info)
        StrategyRegistry.clear_dynamic()

        result = StrategyRegistry.unregister_dynamic("gp_evolved_001")

        assert result is False

    def test_unregister_does_not_affect_static(self):
        """移除動態策略不應影響靜態策略"""
        # 註冊靜態策略
        StrategyRegistry._strategies["static_001"] = TestStrategy

        # 嘗試移除動態策略（不存在）
        result = StrategyRegistry.unregister_dynamic("static_001")

        assert result is False
        # 靜態策略應該還存在
        assert StrategyRegistry.exists("static_001")


# ===== Test list_dynamic() =====

class TestListDynamic:
    """測試 list_dynamic() 方法"""

    def test_list_empty_dynamic_strategies(self):
        """空列表時應返回空列表"""
        result = StrategyRegistry.list_dynamic()

        assert isinstance(result, list)
        assert len(result) == 0

    def test_list_single_dynamic_strategy(self, dynamic_strategy_info):
        """列出單個動態策略"""
        StrategyRegistry.register_dynamic(dynamic_strategy_info)

        result = StrategyRegistry.list_dynamic()

        assert len(result) == 1
        assert result[0] == dynamic_strategy_info

    def test_list_multiple_dynamic_strategies(self):
        """列出多個動態策略"""
        info1 = MockDynamicStrategyInfo(
            name="gp_001",
            strategy_class=TestStrategy
        )
        info2 = MockDynamicStrategyInfo(
            name="gp_002",
            strategy_class=AnotherTestStrategy
        )
        info3 = MockDynamicStrategyInfo(
            name="gp_003",
            strategy_class=TestStrategy
        )

        StrategyRegistry.register_dynamic(info1)
        StrategyRegistry.register_dynamic(info2)
        StrategyRegistry.register_dynamic(info3)

        result = StrategyRegistry.list_dynamic()

        assert len(result) == 3
        names = [info.name for info in result]
        assert "gp_001" in names
        assert "gp_002" in names
        assert "gp_003" in names

    def test_list_returns_info_objects(self, dynamic_strategy_info):
        """列表應返回 DynamicStrategyInfo 物件"""
        StrategyRegistry.register_dynamic(dynamic_strategy_info)

        result = StrategyRegistry.list_dynamic()

        assert len(result) == 1
        info = result[0]
        assert hasattr(info, 'name')
        assert hasattr(info, 'strategy_class')
        assert hasattr(info, 'fitness')
        assert hasattr(info, 'generation')


# ===== Test clear_dynamic() =====

class TestClearDynamic:
    """測試 clear_dynamic() 方法"""

    def test_clear_empty_registry(self):
        """清除空的動態註冊表應返回 0"""
        count = StrategyRegistry.clear_dynamic()

        assert count == 0

    def test_clear_single_strategy(self, dynamic_strategy_info):
        """清除單個策略應返回 1"""
        StrategyRegistry.register_dynamic(dynamic_strategy_info)

        count = StrategyRegistry.clear_dynamic()

        assert count == 1
        assert len(StrategyRegistry.list_dynamic()) == 0

    def test_clear_multiple_strategies(self):
        """清除多個策略應返回正確的數量"""
        for i in range(5):
            info = MockDynamicStrategyInfo(
                name=f"gp_{i:03d}",
                strategy_class=TestStrategy
            )
            StrategyRegistry.register_dynamic(info)

        count = StrategyRegistry.clear_dynamic()

        assert count == 5
        assert len(StrategyRegistry.list_dynamic()) == 0

    def test_clear_does_not_affect_static(self):
        """清除動態策略不應影響靜態策略"""
        # 註冊靜態策略
        StrategyRegistry._strategies["static_001"] = TestStrategy

        # 註冊動態策略並清除
        info = MockDynamicStrategyInfo(
            name="gp_001",
            strategy_class=AnotherTestStrategy
        )
        StrategyRegistry.register_dynamic(info)
        StrategyRegistry.clear_dynamic()

        # 靜態策略應該還存在
        assert StrategyRegistry.exists("static_001")
        assert not StrategyRegistry.is_dynamic("static_001")


# ===== Test is_dynamic() =====

class TestIsDynamic:
    """測試 is_dynamic() 方法"""

    def test_is_dynamic_returns_true(self, dynamic_strategy_info):
        """已註冊的動態策略應返回 True"""
        StrategyRegistry.register_dynamic(dynamic_strategy_info)

        result = StrategyRegistry.is_dynamic("gp_evolved_001")

        assert result is True

    def test_is_dynamic_returns_false_for_unregistered(self):
        """未註冊的策略應返回 False"""
        result = StrategyRegistry.is_dynamic("nonexistent_strategy")

        assert result is False

    def test_is_dynamic_returns_false_for_static(self):
        """靜態策略應返回 False"""
        StrategyRegistry._strategies["static_001"] = TestStrategy

        result = StrategyRegistry.is_dynamic("static_001")

        assert result is False

    def test_is_dynamic_after_unregister(self, dynamic_strategy_info):
        """移除後應返回 False"""
        StrategyRegistry.register_dynamic(dynamic_strategy_info)
        StrategyRegistry.unregister_dynamic("gp_evolved_001")

        result = StrategyRegistry.is_dynamic("gp_evolved_001")

        assert result is False


# ===== Test get_dynamic_info() =====

class TestGetDynamicInfo:
    """測試 get_dynamic_info() 方法"""

    def test_get_existing_dynamic_info(self, dynamic_strategy_info):
        """取得已註冊動態策略的資訊"""
        StrategyRegistry.register_dynamic(dynamic_strategy_info)

        result = StrategyRegistry.get_dynamic_info("gp_evolved_001")

        assert result is not None
        assert result == dynamic_strategy_info
        assert result.fitness == 2.35
        assert result.generation == 10

    def test_get_nonexistent_dynamic_info(self):
        """取得不存在的動態策略應返回 None"""
        result = StrategyRegistry.get_dynamic_info("nonexistent_strategy")

        assert result is None

    def test_get_static_strategy_returns_none(self):
        """取得靜態策略應返回 None"""
        StrategyRegistry._strategies["static_001"] = TestStrategy

        result = StrategyRegistry.get_dynamic_info("static_001")

        assert result is None

    def test_get_info_preserves_metadata(self, dynamic_strategy_info):
        """取得的資訊應保留所有元資料"""
        StrategyRegistry.register_dynamic(dynamic_strategy_info)

        result = StrategyRegistry.get_dynamic_info("gp_evolved_001")

        assert result.metadata == dynamic_strategy_info.metadata
        assert result.expression == dynamic_strategy_info.expression


# ===== Integration Tests =====

class TestGetIntegration:
    """測試 get() 方法與動態策略的整合"""

    def test_get_dynamic_strategy(self, dynamic_strategy_info):
        """能通過 get() 取到動態策略"""
        StrategyRegistry.register_dynamic(dynamic_strategy_info)

        strategy_class = StrategyRegistry.get("gp_evolved_001")

        assert strategy_class == TestStrategy

    def test_get_prefers_dynamic_over_static(self, dynamic_strategy_info):
        """get() 查詢時，靜態優先於動態"""
        # 註冊靜態策略
        StrategyRegistry._strategies["test_001"] = TestStrategy

        # 嘗試註冊同名動態策略（應該會失敗，因為靜態已存在）
        dynamic_info = MockDynamicStrategyInfo(
            name="test_001",
            strategy_class=AnotherTestStrategy
        )

        # 不允許覆蓋
        with pytest.raises(ValueError):
            StrategyRegistry.register_dynamic(dynamic_info, replace_if_exists=False)

        # get() 應該查到靜態策略
        strategy_class = StrategyRegistry.get("test_001")
        assert strategy_class == TestStrategy


class TestListAllIntegration:
    """測試 list_all() 方法與動態策略的整合"""

    def test_list_all_includes_dynamic_by_default(self):
        """list_all() 預設應包含動態策略"""
        # 註冊靜態策略
        StrategyRegistry._strategies["static_001"] = TestStrategy

        # 註冊動態策略
        info = MockDynamicStrategyInfo(
            name="gp_001",
            strategy_class=AnotherTestStrategy
        )
        StrategyRegistry.register_dynamic(info)

        result = StrategyRegistry.list_all()

        assert "static_001" in result
        assert "gp_001" in result

    def test_list_all_excludes_dynamic_when_requested(self):
        """list_all(include_dynamic=False) 不應包含動態策略"""
        # 註冊靜態策略
        StrategyRegistry._strategies["static_001"] = TestStrategy
        StrategyRegistry._strategies["static_002"] = AnotherTestStrategy

        # 註冊動態策略
        info = MockDynamicStrategyInfo(
            name="gp_001",
            strategy_class=TestStrategy
        )
        StrategyRegistry.register_dynamic(info)

        result = StrategyRegistry.list_all(include_dynamic=False)

        assert "static_001" in result
        assert "static_002" in result
        assert "gp_001" not in result

    def test_list_all_with_multiple_dynamic(self):
        """list_all() 應包含所有動態策略"""
        # 清除靜態策略以確保只有我們的動態策略
        StrategyRegistry._strategies.clear()
        
        for i in range(3):
            info = MockDynamicStrategyInfo(
                name=f"gp_{i:03d}",
                strategy_class=TestStrategy
            )
            StrategyRegistry.register_dynamic(info)

        result = StrategyRegistry.list_all()

        assert "gp_000" in result
        assert "gp_001" in result
        assert "gp_002" in result
        # 應該只有動態策略
        dynamic_count = sum(1 for name in result if name.startswith("gp_"))
        assert dynamic_count == 3


# ===== Edge Cases and Error Handling =====

class TestEdgeCases:
    """邊界情況和錯誤處理"""

    def test_register_with_empty_name(self):
        """使用空字符串作為名稱"""
        info = MockDynamicStrategyInfo(
            name="",
            strategy_class=TestStrategy
        )

        # 應該可以註冊（名稱有效）
        result = StrategyRegistry.register_dynamic(info)
        assert result is True

    def test_register_with_special_characters_in_name(self):
        """名稱包含特殊字符"""
        info = MockDynamicStrategyInfo(
            name="gp-evolved_001@v1.0",
            strategy_class=TestStrategy
        )

        result = StrategyRegistry.register_dynamic(info)

        assert result is True
        assert StrategyRegistry.is_dynamic("gp-evolved_001@v1.0")

    def test_unregister_then_register_same_name(self, dynamic_strategy_info):
        """移除後重新註冊相同名稱"""
        StrategyRegistry.register_dynamic(dynamic_strategy_info)
        StrategyRegistry.unregister_dynamic("gp_evolved_001")

        # 應該可以重新註冊
        result = StrategyRegistry.register_dynamic(dynamic_strategy_info)

        assert result is True
        assert StrategyRegistry.is_dynamic("gp_evolved_001")

    def test_get_after_clear(self, dynamic_strategy_info):
        """清除後嘗試取得應拋出 KeyError"""
        StrategyRegistry.register_dynamic(dynamic_strategy_info)
        StrategyRegistry.clear_dynamic()

        with pytest.raises(KeyError):
            StrategyRegistry.get("gp_evolved_001")

    def test_multiple_metadata_fields(self):
        """包含複雜元資料的策略"""
        metadata = {
            "parent_ids": ["gp_gen_09_005", "gp_gen_09_012"],
            "mutation_type": "crossover",
            "backtest_stats": {
                "sharpe": 2.35,
                "return": 125.3,
                "max_dd": -15.2
            },
            "nested": {
                "level": 2,
                "data": [1, 2, 3]
            }
        }

        info = MockDynamicStrategyInfo(
            name="gp_complex",
            strategy_class=TestStrategy,
            metadata=metadata
        )

        StrategyRegistry.register_dynamic(info)
        retrieved = StrategyRegistry.get_dynamic_info("gp_complex")

        assert retrieved.metadata == metadata


# ===== Performance and Scale Tests =====

class TestScalability:
    """可擴展性測試"""

    def test_register_many_strategies(self):
        """註冊大量策略"""
        count = 100
        for i in range(count):
            info = MockDynamicStrategyInfo(
                name=f"gp_{i:05d}",
                strategy_class=TestStrategy,
                fitness=1.0 + i * 0.01
            )
            StrategyRegistry.register_dynamic(info)

        assert len(StrategyRegistry.list_dynamic()) == count

    def test_list_dynamic_performance(self):
        """列出大量策略的性能"""
        count = 100
        for i in range(count):
            info = MockDynamicStrategyInfo(
                name=f"gp_{i:05d}",
                strategy_class=TestStrategy
            )
            StrategyRegistry.register_dynamic(info)

        # 應該快速返回
        result = StrategyRegistry.list_dynamic()

        assert len(result) == count

    def test_is_dynamic_performance(self):
        """查詢大量策略中的動態狀態"""
        count = 100
        for i in range(count):
            info = MockDynamicStrategyInfo(
                name=f"gp_{i:05d}",
                strategy_class=TestStrategy
            )
            StrategyRegistry.register_dynamic(info)

        # 查詢應該快速
        assert StrategyRegistry.is_dynamic("gp_00050")
        assert not StrategyRegistry.is_dynamic("nonexistent")
