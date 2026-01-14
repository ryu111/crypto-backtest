"""
測試永續合約計算模組
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.backtester.perpetual import (
    PerpetualCalculator,
    PerpetualPosition,
    PerpetualRiskMonitor
)


class TestPerpetualCalculator:
    """測試 PerpetualCalculator 類別"""

    @pytest.fixture
    def calc(self):
        """建立計算器實例"""
        return PerpetualCalculator(
            maintenance_margin_rate=0.005,
            funding_interval_hours=8
        )

    # ===== 資金費率測試 =====

    def test_calculate_funding_cost_long(self, calc):
        """測試做多的資金費率成本計算"""
        # 做多，正費率 → 支付
        cost = calc.calculate_funding_cost(
            position_value=10000,
            funding_rate=0.0001,
            direction=1
        )
        assert cost == pytest.approx(1.0)

    def test_calculate_funding_cost_short(self, calc):
        """測試做空的資金費率成本計算"""
        # 做空，正費率 → 收取
        cost = calc.calculate_funding_cost(
            position_value=10000,
            funding_rate=0.0001,
            direction=-1
        )
        assert cost == -1.0

    def test_annualized_funding_rate(self, calc):
        """測試年化資金費率計算"""
        # 每 8 小時結算，每天 3 次，一年 1095 次
        annualized = calc.annualized_funding_rate(0.0001)
        assert pytest.approx(annualized, rel=1e-3) == 0.1095

    # ===== 保證金測試 =====

    def test_calculate_initial_margin(self, calc):
        """測試初始保證金計算"""
        # 1 BTC @ $50,000，10x 槓桿
        margin = calc.calculate_initial_margin(
            position_size=1,
            entry_price=50000,
            leverage=10
        )
        assert margin == pytest.approx(5000.0)

    def test_calculate_margin_ratio(self, calc):
        """測試保證金率計算"""
        ratio = calc.calculate_margin_ratio(
            equity=5500,
            position_value=50000
        )
        assert pytest.approx(ratio, rel=1e-3) == 0.11

    def test_calculate_available_margin(self, calc):
        """測試可用保證金計算"""
        available = calc.calculate_available_margin(
            total_equity=10000,
            used_margin=5000
        )
        assert available == 5000

    # ===== 強平測試 =====

    def test_liquidation_price_long(self, calc):
        """測試做多強平價格計算"""
        liq_price = calc.calculate_liquidation_price(
            entry_price=50000,
            leverage=10,
            direction=1
        )
        # 強平價 = 50000 * (1 - 0.1 + 0.005) = 45250
        assert liq_price == pytest.approx(45250.0)

    def test_liquidation_price_short(self, calc):
        """測試做空強平價格計算"""
        liq_price = calc.calculate_liquidation_price(
            entry_price=50000,
            leverage=10,
            direction=-1
        )
        # 強平價 = 50000 * (1 + 0.1 - 0.005) = 54750
        assert liq_price == pytest.approx(54750.0)

    def test_check_liquidation_long_safe(self, calc):
        """測試做多未爆倉情況"""
        is_liquidated = calc.check_liquidation(
            current_price=48000,
            entry_price=50000,
            leverage=10,
            direction=1
        )
        assert is_liquidated is False

    def test_check_liquidation_long_liquidated(self, calc):
        """測試做多已爆倉情況"""
        is_liquidated = calc.check_liquidation(
            current_price=45000,
            entry_price=50000,
            leverage=10,
            direction=1
        )
        assert is_liquidated is True

    def test_check_liquidation_short_safe(self, calc):
        """測試做空未爆倉情況"""
        is_liquidated = calc.check_liquidation(
            current_price=52000,
            entry_price=50000,
            leverage=10,
            direction=-1
        )
        assert is_liquidated is False

    def test_check_liquidation_short_liquidated(self, calc):
        """測試做空已爆倉情況"""
        is_liquidated = calc.check_liquidation(
            current_price=55000,
            entry_price=50000,
            leverage=10,
            direction=-1
        )
        assert is_liquidated is True

    def test_calculate_liquidation_distance(self, calc):
        """測試強平距離計算"""
        distance_pct, distance_price = calc.calculate_liquidation_distance(
            current_price=48000,
            entry_price=50000,
            leverage=10,
            direction=1
        )
        # 強平價 45250，當前 48000，距離 -2750
        assert distance_price == pytest.approx(-2750, rel=1e-2)
        # 距離百分比約 -5.73%
        assert distance_pct == pytest.approx(-5.73, rel=1e-2)

    # ===== 盈虧測試 =====

    def test_unrealized_pnl_long_profit(self, calc):
        """測試做多浮盈計算"""
        pnl = calc.calculate_unrealized_pnl(
            entry_price=50000,
            mark_price=52000,
            size=1,
            direction=1
        )
        assert pnl == pytest.approx(2000.0)

    def test_unrealized_pnl_long_loss(self, calc):
        """測試做多浮虧計算"""
        pnl = calc.calculate_unrealized_pnl(
            entry_price=50000,
            mark_price=48000,
            size=1,
            direction=1
        )
        assert pnl == -2000.0

    def test_unrealized_pnl_short_profit(self, calc):
        """測試做空浮盈計算"""
        pnl = calc.calculate_unrealized_pnl(
            entry_price=50000,
            mark_price=48000,
            size=1,
            direction=-1
        )
        assert pnl == pytest.approx(2000.0)

    def test_unrealized_pnl_short_loss(self, calc):
        """測試做空浮虧計算"""
        pnl = calc.calculate_unrealized_pnl(
            entry_price=50000,
            mark_price=52000,
            size=1,
            direction=-1
        )
        assert pnl == -2000.0

    def test_pnl_percentage(self, calc):
        """測試盈虧百分比計算"""
        pnl_pct = calc.calculate_pnl_percentage(
            pnl=500,
            margin=5000
        )
        assert pnl_pct == pytest.approx(10.0)

    # ===== Mark Price 和基差測試 =====

    def test_calculate_basis(self, calc):
        """測試基差計算"""
        basis_abs, basis_pct = calc.calculate_basis(
            perp_price=50500,
            spot_price=50000
        )
        assert basis_abs == pytest.approx(500.0)
        assert basis_pct == pytest.approx(1.0)

    def test_calculate_basis_discount(self, calc):
        """測試負基差計算"""
        basis_abs, basis_pct = calc.calculate_basis(
            perp_price=49500,
            spot_price=50000
        )
        assert basis_abs == -500.0
        assert basis_pct == -1.0

    # ===== 風險指標測試 =====

    def test_effective_leverage(self, calc):
        """測試有效槓桿計算"""
        eff_lev = calc.calculate_effective_leverage(
            position_value=50000,
            equity=5500
        )
        assert pytest.approx(eff_lev, rel=1e-2) == 9.09

    def test_bankruptcy_price_long(self, calc):
        """測試做多破產價格計算"""
        bankruptcy_price = calc.calculate_bankruptcy_price(
            entry_price=50000,
            leverage=10,
            direction=1
        )
        # 破產價 = 50000 * (1 - 0.1) = 45000
        assert bankruptcy_price == pytest.approx(45000.0)

    def test_bankruptcy_price_short(self, calc):
        """測試做空破產價格計算"""
        bankruptcy_price = calc.calculate_bankruptcy_price(
            entry_price=50000,
            leverage=10,
            direction=-1
        )
        # 破產價 = 50000 * (1 + 0.1) = 55000
        assert bankruptcy_price == pytest.approx(55000.0)

    def test_estimate_max_position_size(self, calc):
        """測試最大倉位估算"""
        max_size = calc.estimate_max_position_size(
            available_capital=10000,
            price=50000,
            leverage=10,
            fee_rate=0.0004
        )
        # 扣除費用後約可開 2 BTC
        assert max_size == pytest.approx(1.999, rel=1e-2)


class TestPerpetualPosition:
    """測試 PerpetualPosition 類別"""

    def test_position_creation_long(self):
        """測試建立做多倉位"""
        position = PerpetualPosition(
            entry_price=50000,
            size=1.0,
            leverage=10,
            entry_time=datetime.now(),
            margin=5000
        )

        assert position.is_long is True
        assert position.is_short is False
        assert position.direction == 1
        assert position.notional_value == 50000

    def test_position_creation_short(self):
        """測試建立做空倉位"""
        position = PerpetualPosition(
            entry_price=50000,
            size=-1.0,
            leverage=10,
            entry_time=datetime.now(),
            margin=5000
        )

        assert position.is_long is False
        assert position.is_short is True
        assert position.direction == -1
        assert position.notional_value == 50000


class TestPerpetualRiskMonitor:
    """測試 PerpetualRiskMonitor 類別"""

    @pytest.fixture
    def monitor(self):
        """建立風險監控器實例"""
        return PerpetualRiskMonitor(
            warning_threshold=0.02,
            critical_threshold=0.01
        )

    @pytest.fixture
    def position(self):
        """建立測試倉位"""
        return PerpetualPosition(
            entry_price=50000,
            size=1.0,
            leverage=10,
            entry_time=datetime.now(),
            margin=5000
        )

    def test_risk_level_safe(self, monitor, position):
        """測試安全風險等級"""
        risk_level = monitor.assess_risk_level(
            position=position,
            current_price=48000
        )
        assert risk_level == "safe"

    def test_risk_level_warning(self, monitor, position):
        """測試警告風險等級"""
        # 強平價 45250，當前 45500，距離約 0.55%
        risk_level = monitor.assess_risk_level(
            position=position,
            current_price=46000  # 距離強平約 1.6%
        )
        assert risk_level == "warning"

    def test_risk_level_critical(self, monitor, position):
        """測試危急風險等級"""
        # 強平價 45250，當前價接近
        risk_level = monitor.assess_risk_level(
            position=position,
            current_price=45700  # 距離強平約 1%
        )
        assert risk_level == "critical"

    def test_risk_level_liquidated(self, monitor, position):
        """測試已爆倉風險等級"""
        risk_level = monitor.assess_risk_level(
            position=position,
            current_price=45000  # 低於強平價
        )
        assert risk_level == "liquidated"

    def test_generate_risk_report(self, monitor, position):
        """測試風險報告生成"""
        report = monitor.generate_risk_report(
            position=position,
            current_price=48000
        )

        assert 'risk_level' in report
        assert 'liquidation_price' in report
        assert 'distance_to_liquidation_pct' in report
        assert 'margin_ratio' in report
        assert 'unrealized_pnl' in report

        assert report['liquidation_price'] == pytest.approx(45250.0)
        assert report['unrealized_pnl'] == pytest.approx(-2000.0)
        assert report['current_price'] == 48000
        assert report['entry_price'] == 50000


class TestFundingRateIntegration:
    """測試資金費率整合"""

    def test_apply_funding_to_equity(self):
        """測試將資金費率應用到權益曲線"""
        calc = PerpetualCalculator()

        # 建立測試資料
        dates = pd.date_range('2024-01-01', periods=24, freq='H')

        equity_curve = pd.Series(10000, index=dates)
        positions = pd.Series(1, index=dates)  # 做多
        position_sizes = pd.Series(10000, index=dates)

        # 資金費率在 8:00 和 16:00 結算
        funding_times = [dates[8], dates[16]]
        funding_rates = pd.DataFrame({
            'timestamp': funding_times,
            'rate': [0.0001, 0.0001]
        })

        # 應用資金費率
        adjusted_equity = calc.apply_funding_to_equity(
            equity_curve,
            positions,
            funding_rates,
            position_sizes
        )

        # 做多支付 2 次，每次 1 USDT
        assert adjusted_equity.iloc[-1] == pytest.approx(9998.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
