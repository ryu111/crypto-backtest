"""
測試 adaptive_leverage.py

測試策略：
1. Unit Tests（70%）：測試每個調整方法的邏輯
2. Integration Tests（20%）：測試整體計算流程
3. Edge Cases（10%）：測試邊界情況和異常處理
"""

import pytest
import numpy as np
from src.risk.adaptive_leverage import (
    AdaptiveLeverageConfig,
    AdaptiveLeverageController,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def default_config():
    """預設配置"""
    return AdaptiveLeverageConfig()


@pytest.fixture
def custom_config():
    """自定義配置（用於測試配置變更）"""
    return AdaptiveLeverageConfig(
        base_leverage=3,
        min_leverage=1,
        max_leverage=5,
        low_vol_threshold=0.005,
        high_vol_threshold=0.025,
    )


@pytest.fixture
def controller(default_config):
    """預設控制器"""
    return AdaptiveLeverageController(default_config)


@pytest.fixture
def custom_controller(custom_config):
    """自定義控制器"""
    return AdaptiveLeverageController(custom_config)


# ============================================================================
# 1. Configuration Tests（配置測試）
# ============================================================================

class TestAdaptiveLeverageConfig:
    """測試配置類別"""

    def test_default_config(self):
        """測試預設配置值"""
        config = AdaptiveLeverageConfig()

        assert config.base_leverage == 5
        assert config.min_leverage == 1
        assert config.max_leverage == 10
        assert config.volatility_mode is True
        assert config.drawdown_mode is True
        assert config.performance_mode is True

    def test_custom_config(self):
        """測試自定義配置"""
        config = AdaptiveLeverageConfig(
            base_leverage=3,
            max_leverage=7,
            low_vol_threshold=0.015,
        )

        assert config.base_leverage == 3
        assert config.max_leverage == 7
        assert config.low_vol_threshold == 0.015

    def test_drawdown_reduction_dict(self):
        """測試回撤降槓字典"""
        config = AdaptiveLeverageConfig()

        assert 0.05 in config.dd_leverage_reduction
        assert config.dd_leverage_reduction[0.05] == 0.8
        assert config.dd_leverage_reduction[0.10] == 0.5


# ============================================================================
# 2. Initialization Tests（初始化測試）
# ============================================================================

class TestControllerInitialization:
    """測試控制器初始化"""

    def test_init_with_default_config(self):
        """測試預設配置初始化"""
        controller = AdaptiveLeverageController()

        assert controller.config.base_leverage == 5
        assert controller._current_streak == 0
        assert controller._recent_trades == []
        assert controller._smoothed_leverage is None

    def test_init_with_custom_config(self, custom_config):
        """測試自定義配置初始化"""
        controller = AdaptiveLeverageController(custom_config)

        assert controller.config.base_leverage == 3
        assert controller.config.max_leverage == 5

    def test_properties_after_init(self, controller):
        """測試初始化後的屬性"""
        assert controller.recent_win_rate == 0.5  # 預設 50%
        assert controller.current_streak == 0


# ============================================================================
# 3. Calculate Leverage Tests（槓桿計算測試）
# ============================================================================

class TestCalculateLeverage:
    """測試槓桿計算主函數"""

    def test_normal_conditions(self, controller):
        """測試正常條件下的計算"""
        leverage = controller.calculate_leverage(
            current_volatility=0.02,
            current_drawdown=0.03,
            recent_win_rate=0.55
        )

        assert isinstance(leverage, int)
        assert 1 <= leverage <= 10

    def test_respects_min_leverage(self, controller):
        """測試最小槓桿限制"""
        # 極端條件：高波動 + 高回撤 → 應該降到最小
        leverage = controller.calculate_leverage(
            current_volatility=0.10,  # 極高波動
            current_drawdown=0.20,    # 極高回撤
        )

        assert leverage >= controller.config.min_leverage

    def test_respects_max_leverage(self, controller):
        """測試最大槓桿限制"""
        # 理想條件：低波動 + 無回撤 + 連勝
        for _ in range(10):
            controller.update_streak(True)  # 10 連勝

        leverage = controller.calculate_leverage(
            current_volatility=0.005,  # 極低波動
            current_drawdown=0.0,
            recent_win_rate=0.90
        )

        assert leverage <= controller.config.max_leverage

    def test_parameter_validation_volatility(self, controller):
        """測試波動率參數驗證"""
        with pytest.raises(ValueError, match="current_volatility 必須 >= 0"):
            controller.calculate_leverage(
                current_volatility=-0.01,
                current_drawdown=0.0
            )

    def test_parameter_validation_drawdown(self, controller):
        """測試回撤參數驗證"""
        with pytest.raises(ValueError, match="current_drawdown 必須介於 0 和 1 之間"):
            controller.calculate_leverage(
                current_volatility=0.02,
                current_drawdown=1.5
            )

    def test_parameter_validation_win_rate(self, controller):
        """測試勝率參數驗證"""
        with pytest.raises(ValueError, match="recent_win_rate 必須介於 0 和 1 之間"):
            controller.calculate_leverage(
                current_volatility=0.02,
                current_drawdown=0.05,
                recent_win_rate=1.5
            )


# ============================================================================
# 4. Volatility Adjustment Tests（波動度調整測試）
# ============================================================================

class TestVolatilityAdjustment:
    """測試波動度調整邏輯"""

    def test_low_volatility_increases_leverage(self, controller):
        """測試低波動提高槓桿"""
        # 低波動：應該提高槓桿
        low_vol_leverage = controller.calculate_leverage(
            current_volatility=0.005,  # 低於 1% 閾值
            current_drawdown=0.0
        )

        # 正常波動
        normal_leverage = controller.calculate_leverage(
            current_volatility=0.02,
            current_drawdown=0.0
        )

        assert low_vol_leverage > normal_leverage

    def test_high_volatility_decreases_leverage(self, controller):
        """測試高波動降低槓桿"""
        # 高波動：應該降低槓桿
        high_vol_leverage = controller.calculate_leverage(
            current_volatility=0.05,  # 高於 3% 閾值
            current_drawdown=0.0
        )

        # 正常波動
        normal_leverage = controller.calculate_leverage(
            current_volatility=0.02,
            current_drawdown=0.0
        )

        assert high_vol_leverage < normal_leverage

    def test_volatility_mode_disabled(self, controller):
        """測試波動度調整關閉"""
        controller.config.volatility_mode = False

        low_vol = controller.calculate_leverage(
            current_volatility=0.005,
            current_drawdown=0.0
        )

        high_vol = controller.calculate_leverage(
            current_volatility=0.05,
            current_drawdown=0.0
        )

        # 關閉後，波動度不應影響槓桿
        assert low_vol == high_vol


# ============================================================================
# 5. Drawdown Adjustment Tests（回撤調整測試）
# ============================================================================

class TestDrawdownAdjustment:
    """測試回撤調整邏輯"""

    def test_drawdown_reduces_leverage(self, controller):
        """測試回撤降低槓桿"""
        no_dd = controller.calculate_leverage(
            current_volatility=0.02,
            current_drawdown=0.0
        )

        with_dd = controller.calculate_leverage(
            current_volatility=0.02,
            current_drawdown=0.10  # 10% 回撤
        )

        assert with_dd < no_dd

    def test_drawdown_thresholds(self, controller):
        """測試回撤閾值階梯"""
        # 重置避免平滑影響
        controller.reset()

        dd_5 = controller.calculate_leverage(
            current_volatility=0.02,
            current_drawdown=0.05
        )

        controller.reset()
        dd_10 = controller.calculate_leverage(
            current_volatility=0.02,
            current_drawdown=0.10
        )

        controller.reset()
        dd_15 = controller.calculate_leverage(
            current_volatility=0.02,
            current_drawdown=0.15
        )

        # 回撤越大，槓桿越低
        assert dd_5 > dd_10 > dd_15

    def test_drawdown_mode_disabled(self, controller):
        """測試回撤調整關閉"""
        controller.config.drawdown_mode = False

        no_dd = controller.calculate_leverage(
            current_volatility=0.02,
            current_drawdown=0.0
        )

        with_dd = controller.calculate_leverage(
            current_volatility=0.02,
            current_drawdown=0.15
        )

        # 關閉後，回撤不應影響槓桿
        assert no_dd == with_dd


# ============================================================================
# 6. Performance Adjustment Tests（表現調整測試）
# ============================================================================

class TestPerformanceAdjustment:
    """測試表現調整邏輯"""

    def test_winning_streak_increases_leverage(self, controller):
        """測試連勝提高槓桿"""
        # 無連勝
        baseline = controller.calculate_leverage(
            current_volatility=0.02,
            current_drawdown=0.0
        )

        # 建立連勝
        for _ in range(5):
            controller.update_streak(True)

        with_streak = controller.calculate_leverage(
            current_volatility=0.02,
            current_drawdown=0.0
        )

        assert with_streak > baseline

    def test_losing_streak_decreases_leverage(self, controller):
        """測試連虧降低槓桿"""
        # 無連虧
        baseline = controller.calculate_leverage(
            current_volatility=0.02,
            current_drawdown=0.0
        )

        # 建立連虧
        for _ in range(5):
            controller.update_streak(False)

        with_losses = controller.calculate_leverage(
            current_volatility=0.02,
            current_drawdown=0.0
        )

        assert with_losses < baseline

    def test_performance_mode_disabled(self, controller):
        """測試表現調整關閉"""
        controller.config.performance_mode = False

        # 建立連勝
        for _ in range(5):
            controller.update_streak(True)

        with_streak = controller.calculate_leverage(
            current_volatility=0.02,
            current_drawdown=0.0
        )

        # 重置
        controller.reset()

        no_streak = controller.calculate_leverage(
            current_volatility=0.02,
            current_drawdown=0.0
        )

        # 關閉後，連勝不應影響槓桿
        assert with_streak == no_streak


# ============================================================================
# 7. Streak Management Tests（連勝連虧管理測試）
# ============================================================================

class TestStreakManagement:
    """測試連勝連虧狀態管理"""

    def test_winning_streak_accumulation(self, controller):
        """測試連勝累積"""
        assert controller.current_streak == 0

        controller.update_streak(True)
        assert controller.current_streak == 1

        controller.update_streak(True)
        assert controller.current_streak == 2

        controller.update_streak(True)
        assert controller.current_streak == 3

    def test_losing_streak_accumulation(self, controller):
        """測試連虧累積"""
        controller.update_streak(False)
        assert controller.current_streak == -1

        controller.update_streak(False)
        assert controller.current_streak == -2

    def test_streak_reset_on_opposite_result(self, controller):
        """測試連勝/連虧中斷"""
        # 建立連勝
        for _ in range(3):
            controller.update_streak(True)
        assert controller.current_streak == 3

        # 一次虧損應該重置為 -1
        controller.update_streak(False)
        assert controller.current_streak == -1

        # 再虧損應該繼續累積
        controller.update_streak(False)
        assert controller.current_streak == -2

    def test_recent_trades_limit(self, controller):
        """測試最近交易記錄限制"""
        # 添加 30 筆交易
        for i in range(30):
            controller.update_streak(i % 2 == 0)

        # 應該只保留最近 20 筆
        assert len(controller._recent_trades) == 20

    def test_recent_win_rate_calculation(self, controller):
        """測試勝率計算"""
        # 10 筆交易：6 勝 4 虧
        for i in range(10):
            controller.update_streak(i < 6)

        assert controller.recent_win_rate == 0.6


# ============================================================================
# 8. Smoothing Tests（平滑機制測試）
# ============================================================================

class TestLeverageSmoothing:
    """測試槓桿平滑機制"""

    def test_first_calculation_no_smoothing(self, controller):
        """測試首次計算無平滑"""
        assert controller._smoothed_leverage is None

        leverage = controller.calculate_leverage(
            current_volatility=0.02,
            current_drawdown=0.0
        )

        # 首次應該直接設定
        assert controller._smoothed_leverage is not None

    def test_smoothing_reduces_volatility(self, controller):
        """測試平滑減少波動"""
        # 第一次計算（低波動 → 高槓桿）
        lev1 = controller.calculate_leverage(
            current_volatility=0.005,
            current_drawdown=0.0
        )

        # 第二次計算（高波動 → 低槓桿）
        lev2 = controller.calculate_leverage(
            current_volatility=0.05,
            current_drawdown=0.0
        )

        # 因為平滑，第二次不應該跟第一次差太多
        # （如果沒有平滑，差距會很大）
        assert abs(lev2 - lev1) < abs(10 - 2)  # 平滑後變化應該小於極端差異


# ============================================================================
# 9. Reporting Tests（報告測試）
# ============================================================================

class TestLeverageReporting:
    """測試報告功能"""

    def test_report_with_no_history(self, controller):
        """測試無歷史記錄時的報告"""
        report = controller.get_leverage_report()

        assert report['total_adjustments'] == 0
        assert report['avg_leverage'] == controller.config.base_leverage
        assert report['current_streak'] == 0

    def test_report_with_history(self, controller):
        """測試有歷史記錄的報告"""
        # 執行幾次計算
        for i in range(10):
            controller.calculate_leverage(
                current_volatility=0.02,
                current_drawdown=0.0
            )

        report = controller.get_leverage_report()

        assert report['total_adjustments'] == 10
        assert 'avg_leverage' in report
        assert 'min_leverage' in report
        assert 'max_leverage' in report
        assert 'std_leverage' in report
        assert len(report['recent_history']) == 5  # 最近 5 筆

    def test_report_statistics(self, controller):
        """測試報告統計資訊正確性"""
        # 執行計算
        leverages = []
        for i in range(20):
            lev = controller.calculate_leverage(
                current_volatility=0.01 + i * 0.001,
                current_drawdown=0.0
            )
            leverages.append(lev)

        report = controller.get_leverage_report()

        # 驗證統計
        assert report['min_leverage'] == min(leverages)
        assert report['max_leverage'] == max(leverages)
        assert abs(report['avg_leverage'] - np.mean(leverages)) < 0.01


# ============================================================================
# 10. Reset Tests（重置測試）
# ============================================================================

class TestControllerReset:
    """測試控制器重置"""

    def test_reset_clears_state(self, controller):
        """測試重置清空狀態"""
        # 建立一些狀態
        for _ in range(5):
            controller.update_streak(True)

        controller.calculate_leverage(
            current_volatility=0.02,
            current_drawdown=0.05
        )

        # 重置
        controller.reset()

        # 驗證狀態已清空
        assert controller._current_streak == 0
        assert controller._recent_trades == []
        assert controller._smoothed_leverage is None
        assert controller._adjustment_history == []

    def test_reset_preserves_config(self, custom_controller):
        """測試重置保留配置"""
        original_base = custom_controller.config.base_leverage

        custom_controller.reset()

        # 配置應該保留
        assert custom_controller.config.base_leverage == original_base


# ============================================================================
# 11. Edge Cases（邊界情況測試）
# ============================================================================

class TestEdgeCases:
    """測試邊界情況"""

    def test_zero_volatility(self, controller):
        """測試零波動"""
        leverage = controller.calculate_leverage(
            current_volatility=0.0,
            current_drawdown=0.0
        )

        assert isinstance(leverage, int)
        assert leverage >= controller.config.min_leverage

    def test_extreme_volatility(self, controller):
        """測試極端波動"""
        leverage = controller.calculate_leverage(
            current_volatility=0.5,  # 50% 日波動（極端）
            current_drawdown=0.0
        )

        # 應該接近最低（因為平滑可能不會立即到最低）
        assert leverage <= controller.config.min_leverage + 1

    def test_max_drawdown(self, controller):
        """測試最大回撤"""
        leverage = controller.calculate_leverage(
            current_volatility=0.02,
            current_drawdown=0.99  # 99% 回撤
        )

        # 應該降到最低
        assert leverage == controller.config.min_leverage

    def test_extreme_winning_streak(self, controller):
        """測試極端連勝"""
        # 100 連勝
        for _ in range(100):
            controller.update_streak(True)

        leverage = controller.calculate_leverage(
            current_volatility=0.005,  # 低波動
            current_drawdown=0.0
        )

        # 即使連勝，也不應超過最大槓桿
        assert leverage <= controller.config.max_leverage

    def test_extreme_losing_streak(self, controller):
        """測試極端連虧"""
        # 100 連虧
        for _ in range(100):
            controller.update_streak(False)

        leverage = controller.calculate_leverage(
            current_volatility=0.02,
            current_drawdown=0.0
        )

        # 極端連虧應該接近最小槓桿
        assert leverage <= controller.config.base_leverage


# ============================================================================
# 12. Integration Tests（整合測試）
# ============================================================================

class TestIntegration:
    """整合測試：測試真實使用場景"""

    def test_realistic_trading_scenario(self, controller):
        """測試真實交易場景"""
        # 模擬一系列交易
        scenarios = [
            # (volatility, drawdown, trade_result)
            (0.02, 0.0, True),    # 正常交易，獲利
            (0.015, 0.0, True),   # 低波動，獲利
            (0.025, 0.02, False), # 正常波動，虧損
            (0.03, 0.05, False),  # 高波動，虧損，回撤開始
            (0.04, 0.08, False),  # 高波動，虧損，回撤加深
            (0.025, 0.10, True),  # 波動降低，獲利，回撤止跌
            (0.02, 0.08, True),   # 正常，獲利，回撤恢復
            (0.015, 0.05, True),  # 低波動，獲利
        ]

        leverages = []
        for vol, dd, won in scenarios:
            lev = controller.calculate_leverage(vol, dd)
            leverages.append(lev)
            controller.update_streak(won)

        # 驗證槓桿在合理範圍
        assert all(controller.config.min_leverage <= lev <= controller.config.max_leverage
                   for lev in leverages)

        # 驗證回撤時槓桿降低
        assert leverages[4] < leverages[0]  # 高回撤時應該更低

        # 驗證恢復時槓桿提高（至少不低於低谷）
        assert leverages[-1] >= leverages[4]  # 恢復後應該提高或持平

    def test_backtesting_workflow(self, controller):
        """測試回測工作流"""
        # 模擬回測：100 次交易
        np.random.seed(42)

        for i in range(100):
            # 隨機市場狀態
            vol = np.random.uniform(0.01, 0.04)
            dd = max(0, np.random.normal(0.05, 0.03))
            dd = min(dd, 0.3)  # 限制最大 30%

            # 計算槓桿
            leverage = controller.calculate_leverage(vol, dd)

            # 模擬交易結果
            won = np.random.random() > 0.45  # 55% 勝率
            controller.update_streak(won)

        # 驗證統計
        report = controller.get_leverage_report()

        assert report['total_adjustments'] == 100
        assert report['total_trades'] == 20  # 保留最近 20 筆
        assert 0 <= report['recent_win_rate'] <= 1

    def test_multiple_resets(self, controller):
        """測試多次重置（回測多策略）"""
        for strategy in range(5):
            # 執行一些交易
            for _ in range(10):
                controller.calculate_leverage(0.02, 0.03)
                controller.update_streak(True)

            # 驗證有狀態
            assert len(controller._adjustment_history) > 0

            # 重置準備下一個策略
            controller.reset()

            # 驗證已清空
            assert len(controller._adjustment_history) == 0


# ============================================================================
# 13. Property Tests（屬性測試）
# ============================================================================

class TestProperties:
    """測試 property 方法"""

    def test_recent_win_rate_empty(self, controller):
        """測試無交易時的勝率"""
        assert controller.recent_win_rate == 0.5  # 預設 50%

    def test_recent_win_rate_all_wins(self, controller):
        """測試全勝勝率"""
        for _ in range(10):
            controller.update_streak(True)

        assert controller.recent_win_rate == 1.0

    def test_recent_win_rate_all_losses(self, controller):
        """測試全虧勝率"""
        for _ in range(10):
            controller.update_streak(False)

        assert controller.recent_win_rate == 0.0

    def test_current_streak_property(self, controller):
        """測試 current_streak 屬性"""
        for i in range(5):
            controller.update_streak(True)
            assert controller.current_streak == i + 1


# ============================================================================
# 14. Repr Tests（表示測試）
# ============================================================================

class TestRepr:
    """測試 __repr__ 方法"""

    def test_repr_format(self, controller):
        """測試 __repr__ 格式"""
        repr_str = repr(controller)

        assert 'AdaptiveLeverageController' in repr_str
        assert 'base=' in repr_str
        assert 'range=' in repr_str
        assert 'streak=' in repr_str
        assert 'win_rate=' in repr_str

    def test_repr_values(self, controller):
        """測試 __repr__ 數值正確"""
        # 建立一些狀態
        for _ in range(3):
            controller.update_streak(True)

        repr_str = repr(controller)

        assert 'base=5x' in repr_str
        assert 'range=[1, 10]' in repr_str
        assert 'streak=3' in repr_str


# ============================================================================
# 15. Performance Tests（效能測試）
# ============================================================================

class TestPerformance:
    """測試效能（確保計算夠快）"""

    def test_bulk_calculations(self, controller):
        """測試批量計算"""
        import time

        start = time.time()

        for i in range(1000):
            controller.calculate_leverage(
                current_volatility=0.01 + i * 0.00001,
                current_drawdown=0.0
            )

        elapsed = time.time() - start

        # 1000 次計算應該在 1 秒內完成
        assert elapsed < 1.0
