"""
Unit Tests for Position Sizing Module

測試 Kelly Criterion 及其變體的正確性。
"""

import pytest
import logging
from src.risk.position_sizing import (
    kelly_criterion,
    KellyPositionSizer,
    PositionSizeResult,
)


class TestKellyCriterion:
    """測試 kelly_criterion 函數"""

    def test_basic_calculation(self):
        """測試基本計算"""
        # 55% 勝率，盈虧比 1.5
        result = kelly_criterion(0.55, 1.5)
        # f* = 0.55 - (0.45 / 1.5) = 0.55 - 0.3 = 0.25
        assert result == pytest.approx(0.25, abs=1e-6)

    def test_even_odds(self):
        """測試 50/50 勝率，盈虧比 1:1 的情況"""
        result = kelly_criterion(0.5, 1.0)
        # f* = 0.5 - (0.5 / 1.0) = 0
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_high_win_rate(self):
        """測試高勝率情況"""
        # 70% 勝率，盈虧比 2.0
        result = kelly_criterion(0.7, 2.0)
        # f* = 0.7 - (0.3 / 2.0) = 0.7 - 0.15 = 0.55
        assert result == pytest.approx(0.55, abs=1e-6)

    def test_negative_expectancy(self):
        """測試負期望值（應返回 0）"""
        # 40% 勝率，盈虧比 1.0
        result = kelly_criterion(0.4, 1.0)
        # f* = 0.4 - (0.6 / 1.0) = -0.2 -> 限制為 0
        assert result == 0.0

    def test_low_win_rate_high_ratio(self):
        """測試低勝率但高盈虧比"""
        # 30% 勝率，盈虧比 3.0
        result = kelly_criterion(0.3, 3.0)
        # f* = 0.3 - (0.7 / 3.0) = 0.3 - 0.2333 = 0.0667
        assert result == pytest.approx(0.0667, abs=1e-4)

    def test_extreme_high_kelly(self):
        """測試極端高 Kelly 值（應限制為 1.0）"""
        # 極端情況：90% 勝率，盈虧比 10.0
        result = kelly_criterion(0.9, 10.0)
        # f* = 0.9 - (0.1 / 10.0) = 0.89
        # 不會超過 1.0，所以應該接近 0.89
        assert result == pytest.approx(0.89, abs=1e-6)

    def test_invalid_win_rate_negative(self):
        """測試無效的勝率（負數）"""
        with pytest.raises(ValueError, match="win_rate 必須介於 0 和 1 之間"):
            kelly_criterion(-0.1, 1.5)

    def test_invalid_win_rate_too_high(self):
        """測試無效的勝率（超過 1）"""
        with pytest.raises(ValueError, match="win_rate 必須介於 0 和 1 之間"):
            kelly_criterion(1.5, 1.5)

    def test_invalid_win_loss_ratio_zero(self):
        """測試無效的盈虧比（0）"""
        with pytest.raises(ValueError, match="win_loss_ratio 必須大於 0"):
            kelly_criterion(0.55, 0)

    def test_invalid_win_loss_ratio_negative(self):
        """測試無效的盈虧比（負數）"""
        with pytest.raises(ValueError, match="win_loss_ratio 必須大於 0"):
            kelly_criterion(0.55, -1.5)


class TestKellyPositionSizer:
    """測試 KellyPositionSizer 類別"""

    def test_initialization_defaults(self):
        """測試預設初始化"""
        sizer = KellyPositionSizer()
        assert sizer.kelly_fraction == 0.5
        assert sizer.kelly_type == "Half Kelly"
        assert sizer.max_position_fraction == 0.25
        assert sizer.min_win_rate == 0.4
        assert sizer.min_win_loss_ratio == 1.0

    def test_initialization_full_kelly(self):
        """測試 Full Kelly 初始化"""
        sizer = KellyPositionSizer(kelly_fraction=1.0)
        assert sizer.kelly_fraction == 1.0
        assert sizer.kelly_type == "Full Kelly"

    def test_initialization_quarter_kelly(self):
        """測試 Quarter Kelly 初始化"""
        sizer = KellyPositionSizer(kelly_fraction=0.25)
        assert sizer.kelly_fraction == 0.25
        assert sizer.kelly_type == "Quarter Kelly"

    def test_initialization_custom_kelly(self):
        """測試自訂 Kelly 乘數"""
        sizer = KellyPositionSizer(kelly_fraction=0.7)
        assert sizer.kelly_fraction == 0.7
        assert sizer.kelly_type == "0.7x Kelly"

    def test_initialization_invalid_kelly_fraction(self):
        """測試無效的 Kelly 乘數"""
        with pytest.raises(ValueError, match="kelly_fraction 必須介於 0 和 1 之間"):
            KellyPositionSizer(kelly_fraction=1.5)

        with pytest.raises(ValueError, match="kelly_fraction 必須介於 0 和 1 之間"):
            KellyPositionSizer(kelly_fraction=0)

    def test_calculate_position_size_basic(self):
        """測試基本部位計算"""
        sizer = KellyPositionSizer(kelly_fraction=1.0, max_position_fraction=1.0)

        result = sizer.calculate_position_size(
            capital=10000,
            win_rate=0.55,
            avg_win=150,
            avg_loss=100,
            enforce_min_requirements=False
        )

        # 盈虧比 = 150 / 100 = 1.5
        # Kelly = 0.55 - (0.45 / 1.5) = 0.25
        # Full Kelly, 無限制
        assert result.optimal_fraction == pytest.approx(0.25, abs=1e-6)
        assert result.position_size == pytest.approx(2500, abs=1e-2)
        assert result.kelly_type == "Full Kelly"
        assert result.win_rate == 0.55
        assert result.win_loss_ratio == pytest.approx(1.5, abs=1e-6)

    def test_calculate_position_size_half_kelly(self):
        """測試 Half Kelly"""
        sizer = KellyPositionSizer(kelly_fraction=0.5, max_position_fraction=1.0)

        result = sizer.calculate_position_size(
            capital=10000,
            win_rate=0.55,
            avg_win=150,
            avg_loss=100,
            enforce_min_requirements=False
        )

        # Kelly = 0.25 * 0.5 = 0.125
        assert result.optimal_fraction == pytest.approx(0.125, abs=1e-6)
        assert result.position_size == pytest.approx(1250, abs=1e-2)
        assert result.kelly_type == "Half Kelly"

    def test_calculate_position_size_with_max_limit(self):
        """測試最大部位限制"""
        sizer = KellyPositionSizer(
            kelly_fraction=1.0,
            max_position_fraction=0.1  # 最大 10%
        )

        result = sizer.calculate_position_size(
            capital=10000,
            win_rate=0.7,
            avg_win=200,
            avg_loss=100,
            enforce_min_requirements=False
        )

        # Kelly = 0.7 - (0.3 / 2.0) = 0.55
        # 但受限於 max_position_fraction = 0.1
        assert result.optimal_fraction == pytest.approx(0.1, abs=1e-6)
        assert result.position_size == pytest.approx(1000, abs=1e-2)

    def test_calculate_position_size_below_min_win_rate(self):
        """測試低於最低勝率要求"""
        sizer = KellyPositionSizer(min_win_rate=0.5)

        result = sizer.calculate_position_size(
            capital=10000,
            win_rate=0.45,  # 低於最低要求
            avg_win=150,
            avg_loss=100,
            enforce_min_requirements=True
        )

        # 應返回 0
        assert result.optimal_fraction == 0.0
        assert result.position_size == 0.0

    def test_calculate_position_size_below_min_ratio(self):
        """測試低於最低盈虧比要求"""
        sizer = KellyPositionSizer(min_win_loss_ratio=1.5)

        result = sizer.calculate_position_size(
            capital=10000,
            win_rate=0.55,
            avg_win=120,
            avg_loss=100,  # 盈虧比 = 1.2，低於 1.5
            enforce_min_requirements=True
        )

        # 應返回 0
        assert result.optimal_fraction == 0.0
        assert result.position_size == 0.0

    def test_calculate_position_size_ignore_min_requirements(self):
        """測試忽略最低要求"""
        sizer = KellyPositionSizer(
            kelly_fraction=1.0,
            max_position_fraction=1.0,
            min_win_rate=0.5,
            min_win_loss_ratio=1.5
        )

        result = sizer.calculate_position_size(
            capital=10000,
            win_rate=0.45,  # 低於最低要求
            avg_win=120,
            avg_loss=100,  # 盈虧比 1.2，低於 1.5
            enforce_min_requirements=False  # 忽略最低要求
        )

        # 仍然計算（即使不符合要求）
        # Kelly = 0.45 - (0.55 / 1.2) = 0.45 - 0.458 = -0.008 -> 0
        assert result.optimal_fraction == 0.0

    def test_calculate_position_size_invalid_capital(self):
        """測試無效的資金"""
        sizer = KellyPositionSizer()

        with pytest.raises(ValueError, match="capital 必須大於 0"):
            sizer.calculate_position_size(
                capital=0,
                win_rate=0.55,
                avg_win=150,
                avg_loss=100
            )

    def test_calculate_position_size_invalid_avg_win(self):
        """測試無效的平均獲利"""
        sizer = KellyPositionSizer()

        with pytest.raises(ValueError, match="avg_win 必須大於 0"):
            sizer.calculate_position_size(
                capital=10000,
                win_rate=0.55,
                avg_win=0,
                avg_loss=100
            )

    def test_calculate_position_size_invalid_avg_loss(self):
        """測試無效的平均虧損"""
        sizer = KellyPositionSizer()

        with pytest.raises(ValueError, match="avg_loss 必須大於 0"):
            sizer.calculate_position_size(
                capital=10000,
                win_rate=0.55,
                avg_win=150,
                avg_loss=0
            )

    def test_calculate_from_trades_basic(self):
        """測試從交易記錄計算"""
        sizer = KellyPositionSizer(
            kelly_fraction=1.0,
            max_position_fraction=1.0
        )

        winning_trades = [150, 200, 100]  # avg = 150
        losing_trades = [100, 100]        # avg = 100

        result = sizer.calculate_from_trades(
            capital=10000,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            enforce_min_requirements=False
        )

        # 勝率 = 3 / 5 = 0.6
        # 盈虧比 = 150 / 100 = 1.5
        # Kelly = 0.6 - (0.4 / 1.5) = 0.6 - 0.267 = 0.333
        assert result.win_rate == pytest.approx(0.6, abs=1e-6)
        assert result.win_loss_ratio == pytest.approx(1.5, abs=1e-6)
        assert result.optimal_fraction == pytest.approx(0.333, abs=1e-3)
        assert result.position_size == pytest.approx(3333.33, abs=1)

    def test_calculate_from_trades_no_losses(self):
        """測試沒有虧損交易的情況"""
        sizer = KellyPositionSizer(max_position_fraction=0.2)

        winning_trades = [150, 200, 100]
        losing_trades = []

        result = sizer.calculate_from_trades(
            capital=10000,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            enforce_min_requirements=False
        )

        # 應使用保守估計（最大 10%）
        assert result.optimal_fraction == pytest.approx(0.1, abs=1e-6)
        assert result.position_size == pytest.approx(1000, abs=1e-2)
        assert result.win_rate == 1.0

    def test_calculate_from_trades_no_wins(self):
        """測試沒有獲利交易的情況"""
        sizer = KellyPositionSizer()

        winning_trades = []
        losing_trades = [100, 100]

        result = sizer.calculate_from_trades(
            capital=10000,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            enforce_min_requirements=False
        )

        # 應返回 0
        assert result.optimal_fraction == 0.0
        assert result.position_size == 0.0
        assert result.win_rate == 0.0

    def test_calculate_from_trades_empty_list(self):
        """測試空交易記錄"""
        sizer = KellyPositionSizer()

        with pytest.raises(ValueError, match="至少需要一筆交易記錄"):
            sizer.calculate_from_trades(
                capital=10000,
                winning_trades=[],
                losing_trades=[]
            )

    def test_adjust_kelly_fraction(self):
        """測試動態調整 Kelly 乘數"""
        sizer = KellyPositionSizer(kelly_fraction=0.5)
        assert sizer.kelly_type == "Half Kelly"

        sizer.adjust_kelly_fraction(0.25)
        assert sizer.kelly_fraction == 0.25
        assert sizer.kelly_type == "Quarter Kelly"

        sizer.adjust_kelly_fraction(1.0)
        assert sizer.kelly_fraction == 1.0
        assert sizer.kelly_type == "Full Kelly"

    def test_adjust_kelly_fraction_invalid(self):
        """測試無效的 Kelly 乘數調整"""
        sizer = KellyPositionSizer()

        with pytest.raises(ValueError, match="new_fraction 必須介於 0 和 1 之間"):
            sizer.adjust_kelly_fraction(1.5)


class TestPositionSizeResult:
    """測試 PositionSizeResult 資料類別"""

    def test_str_representation(self):
        """測試字串表示"""
        result = PositionSizeResult(
            optimal_fraction=0.25,
            position_size=2500.0,
            kelly_type="Half Kelly",
            win_rate=0.55,
            win_loss_ratio=1.5
        )

        str_repr = str(result)
        assert "Half Kelly" in str_repr
        assert "0.2500" in str_repr  # optimal_fraction
        assert "25.00%" in str_repr
        assert "$2,500.00" in str_repr
        assert "win_rate=0.5500" in str_repr
        assert "win_loss_ratio=1.5000" in str_repr


class TestIntegrationScenarios:
    """整合測試場景"""

    def test_conservative_trading_scenario(self):
        """測試保守交易場景"""
        # 使用 Quarter Kelly，嚴格限制
        sizer = KellyPositionSizer(
            kelly_fraction=0.25,
            max_position_fraction=0.1,
            min_win_rate=0.5,
            min_win_loss_ratio=1.5
        )

        # 符合最低要求的策略
        result = sizer.calculate_position_size(
            capital=100000,
            win_rate=0.55,
            avg_win=300,
            avg_loss=200,  # 盈虧比 1.5
            enforce_min_requirements=True
        )

        # Kelly = 0.55 - (0.45 / 1.5) = 0.25
        # Quarter Kelly = 0.0625
        # 受限於 max = 0.1
        assert result.optimal_fraction == pytest.approx(0.0625, abs=1e-4)
        assert result.position_size == pytest.approx(6250, abs=1)

    def test_aggressive_trading_scenario(self):
        """測試激進交易場景"""
        # 使用 Full Kelly，寬鬆限制
        sizer = KellyPositionSizer(
            kelly_fraction=1.0,
            max_position_fraction=0.5,
            min_win_rate=0.4,
            min_win_loss_ratio=1.0
        )

        # 高勝率高盈虧比策略
        result = sizer.calculate_position_size(
            capital=100000,
            win_rate=0.65,
            avg_win=400,
            avg_loss=200,  # 盈虧比 2.0
            enforce_min_requirements=True
        )

        # Kelly = 0.65 - (0.35 / 2.0) = 0.65 - 0.175 = 0.475
        # Full Kelly, 受限於 max = 0.5
        assert result.optimal_fraction == pytest.approx(0.475, abs=1e-3)
        assert result.position_size == pytest.approx(47500, abs=1)

    def test_real_world_trading_history(self):
        """測試真實交易歷史場景"""
        sizer = KellyPositionSizer(kelly_fraction=0.5)

        # 模擬真實交易記錄
        winning_trades = [
            250, 180, 300, 220, 150,  # 5 筆獲利
            280, 190, 240, 160, 210   # 5 筆獲利
        ]  # avg = 218

        losing_trades = [
            120, 100, 130, 110, 105,  # 5 筆虧損
        ]  # avg = 113

        result = sizer.calculate_from_trades(
            capital=50000,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            enforce_min_requirements=False
        )

        # 勝率 = 10 / 15 = 0.667
        # 盈虧比 = 218 / 113 = 1.929
        # Kelly = 0.667 - (0.333 / 1.929) ≈ 0.494
        # Half Kelly = 0.247
        assert result.win_rate == pytest.approx(0.667, abs=1e-3)
        assert result.win_loss_ratio == pytest.approx(1.929, abs=1e-3)
        assert result.optimal_fraction == pytest.approx(0.247, abs=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
