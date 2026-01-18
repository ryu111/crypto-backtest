"""
測試 UltimateLoopConfig 的 GP 探索功能

測試範圍：
1. GP 參數的預設值
2. GP 參數驗證
3. Factory methods 的 GP 配置
4. to_dict() 包含 GP 參數
"""

import pytest
from src.automation.ultimate_config import UltimateLoopConfig


class TestUltimateLoopConfigGP:
    """測試 UltimateLoopConfig 的 GP 探索配置"""

    def test_gp_default_values(self):
        """測試 GP 參數的預設值"""
        config = UltimateLoopConfig()

        # 驗證所有 GP 參數都有預設值
        assert hasattr(config, 'gp_explore_enabled')
        assert hasattr(config, 'gp_explore_ratio')
        assert hasattr(config, 'gp_population_size')
        assert hasattr(config, 'gp_generations')
        assert hasattr(config, 'gp_top_n')

        # 驗證預設值合理
        assert config.gp_explore_enabled is True
        assert 0 <= config.gp_explore_ratio <= 1
        assert config.gp_population_size > 0
        assert config.gp_generations > 0
        assert config.gp_top_n > 0

        # 預設配置應該通過驗證
        assert config.validate() is True

    def test_gp_explore_ratio_validation(self):
        """測試 gp_explore_ratio 驗證（必須在 0-1 之間）"""
        # 測試無效值：< 0
        with pytest.raises(ValueError, match="gp_explore_ratio 必須在 0-1 之間"):
            config = UltimateLoopConfig(gp_explore_ratio=-0.1)
            config.validate()

        # 測試無效值：> 1
        with pytest.raises(ValueError, match="gp_explore_ratio 必須在 0-1 之間"):
            config = UltimateLoopConfig(gp_explore_ratio=1.5)
            config.validate()

        # 測試邊界值：0 和 1 應該有效
        config_zero = UltimateLoopConfig(gp_explore_ratio=0.0)
        assert config_zero.validate() is True

        config_one = UltimateLoopConfig(gp_explore_ratio=1.0)
        assert config_one.validate() is True

    def test_gp_population_size_validation(self):
        """測試 gp_population_size 驗證（必須 > 0）"""
        # 測試無效值：0
        with pytest.raises(ValueError, match="gp_population_size 必須 >= 1"):
            config = UltimateLoopConfig(gp_population_size=0)
            config.validate()

        # 測試無效值：負數
        with pytest.raises(ValueError, match="gp_population_size 必須 >= 1"):
            config = UltimateLoopConfig(gp_population_size=-10)
            config.validate()

        # 測試有效值
        config = UltimateLoopConfig(gp_population_size=50)
        assert config.validate() is True

    def test_gp_generations_validation(self):
        """測試 gp_generations 驗證（必須 > 0）"""
        # 測試無效值：0
        with pytest.raises(ValueError, match="gp_generations 必須 >= 1"):
            config = UltimateLoopConfig(gp_generations=0)
            config.validate()

        # 測試無效值：負數
        with pytest.raises(ValueError, match="gp_generations 必須 >= 1"):
            config = UltimateLoopConfig(gp_generations=-5)
            config.validate()

        # 測試有效值
        config = UltimateLoopConfig(gp_generations=30)
        assert config.validate() is True

    def test_gp_top_n_validation(self):
        """測試 gp_top_n 驗證（必須 > 0）"""
        # 測試無效值：0
        with pytest.raises(ValueError, match="gp_top_n 必須 >= 1"):
            config = UltimateLoopConfig(gp_top_n=0)
            config.validate()

        # 測試無效值：負數
        with pytest.raises(ValueError, match="gp_top_n 必須 >= 1"):
            config = UltimateLoopConfig(gp_top_n=-3)
            config.validate()

        # 測試有效值
        config = UltimateLoopConfig(gp_top_n=3)
        assert config.validate() is True

    def test_production_config_gp_settings(self):
        """測試生產環境配置的 GP 設定"""
        config = UltimateLoopConfig.create_production_config()

        # 生產環境應該啟用 GP
        assert config.gp_explore_enabled is True

        # 驗證生產環境的 GP 配置
        assert config.gp_explore_ratio == 0.15  # 保守（15%）
        assert config.gp_population_size == 100  # 大族群
        assert config.gp_generations == 50       # 多代數
        assert config.gp_top_n == 5              # 產生 5 個策略

        # 配置應該通過驗證
        assert config.validate() is True

    def test_development_config_gp_settings(self):
        """測試開發環境配置的 GP 設定"""
        config = UltimateLoopConfig.create_development_config()

        # 開發環境應該啟用 GP
        assert config.gp_explore_enabled is True

        # 驗證開發環境的 GP 配置
        assert config.gp_explore_ratio == 0.2   # 標準（20%）
        assert config.gp_population_size == 50  # 中等族群
        assert config.gp_generations == 30      # 標準代數
        assert config.gp_top_n == 3             # 產生 3 個策略

        # 配置應該通過驗證
        assert config.validate() is True

    def test_quick_test_config_gp_settings(self):
        """測試快速測試配置的 GP 設定"""
        config = UltimateLoopConfig.create_quick_test_config()

        # 快速測試應該關閉 GP（提升速度）
        assert config.gp_explore_enabled is False

        # 驗證快速測試的 GP 配置（即使關閉，參數也應有效）
        assert config.gp_explore_ratio == 0.1  # 小比例
        assert config.gp_population_size == 20  # 小族群
        assert config.gp_generations == 10      # 少代數
        assert config.gp_top_n == 1             # 只產生 1 個

        # 配置應該通過驗證
        assert config.validate() is True

    def test_high_performance_config_gp_settings(self):
        """測試高效能配置的 GP 設定"""
        config = UltimateLoopConfig.create_high_performance_config()

        # 高效能應該啟用 GP
        assert config.gp_explore_enabled is True

        # 驗證高效能的 GP 配置
        assert config.gp_explore_ratio == 0.25   # 更高比例（25%）
        assert config.gp_population_size == 100  # 大族群
        assert config.gp_generations == 50       # 充分演化
        assert config.gp_top_n == 5              # 產生 5 個策略

        # 配置應該通過驗證
        assert config.validate() is True

    def test_to_dict_includes_gp_params(self):
        """測試 to_dict() 包含所有 GP 參數"""
        config = UltimateLoopConfig()
        config_dict = config.to_dict()

        # 驗證所有 GP 參數都在字典中
        assert 'gp_explore_enabled' in config_dict
        assert 'gp_explore_ratio' in config_dict
        assert 'gp_population_size' in config_dict
        assert 'gp_generations' in config_dict
        assert 'gp_top_n' in config_dict

        # 驗證值正確
        assert config_dict['gp_explore_enabled'] == config.gp_explore_enabled
        assert config_dict['gp_explore_ratio'] == config.gp_explore_ratio
        assert config_dict['gp_population_size'] == config.gp_population_size
        assert config_dict['gp_generations'] == config.gp_generations
        assert config_dict['gp_top_n'] == config.gp_top_n

    def test_custom_gp_config(self):
        """測試自訂 GP 配置"""
        config = UltimateLoopConfig(
            gp_explore_enabled=True,
            gp_explore_ratio=0.3,
            gp_population_size=200,
            gp_generations=100,
            gp_top_n=10
        )

        # 驗證自訂值
        assert config.gp_explore_enabled is True
        assert config.gp_explore_ratio == 0.3
        assert config.gp_population_size == 200
        assert config.gp_generations == 100
        assert config.gp_top_n == 10

        # 配置應該通過驗證
        assert config.validate() is True

    def test_multiple_invalid_gp_params(self):
        """測試多個無效 GP 參數會全部報錯"""
        with pytest.raises(ValueError) as exc_info:
            config = UltimateLoopConfig(
                gp_explore_ratio=1.5,      # 無效：> 1
                gp_population_size=-10,    # 無效：< 1
                gp_generations=0,          # 無效：< 1
                gp_top_n=-5                # 無效：< 1
            )
            config.validate()

        error_msg = str(exc_info.value)

        # 驗證所有錯誤都被報告
        assert "gp_explore_ratio 必須在 0-1 之間" in error_msg
        assert "gp_population_size 必須 >= 1" in error_msg
        assert "gp_generations 必須 >= 1" in error_msg
        assert "gp_top_n 必須 >= 1" in error_msg

    def test_gp_disabled_with_valid_params(self):
        """測試 GP 關閉但參數仍有效"""
        config = UltimateLoopConfig(
            gp_explore_enabled=False,  # 關閉 GP
            gp_explore_ratio=0.5,
            gp_population_size=50,
            gp_generations=30,
            gp_top_n=3
        )

        # 即使 GP 關閉，參數仍應有效
        assert config.gp_explore_enabled is False
        assert config.validate() is True


if __name__ == "__main__":
    # 運行測試
    pytest.main([__file__, "-v", "--tb=short"])
