"""
測試 engine.py 的 Extract Method 重構

測試 _calculate_trade_statistics 方法的各種情況。
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.backtester.engine import BacktestEngine, BacktestConfig


class TestCalculateTradeStatistics:
    """測試 _calculate_trade_statistics 方法"""

    @pytest.fixture
    def engine(self):
        """建立測試用的 BacktestEngine"""
        config = BacktestConfig(
            symbol='BTCUSDT',
            timeframe='1h',
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            initial_capital=10000,
            leverage=1
        )
        return BacktestEngine(config)

    @pytest.fixture
    def empty_trades(self):
        """空交易列表"""
        return pd.DataFrame(columns=['PnL', 'Entry Timestamp', 'Exit Timestamp'])

    @pytest.fixture
    def all_winning_trades(self):
        """全部獲利的交易"""
        base_time = datetime(2024, 1, 1)
        return pd.DataFrame({
            'PnL': [100.0, 200.0, 150.0],
            'Entry Timestamp': [base_time + timedelta(hours=i*10) for i in range(3)],
            'Exit Timestamp': [base_time + timedelta(hours=i*10+5) for i in range(3)]
        })

    @pytest.fixture
    def all_losing_trades(self):
        """全部虧損的交易"""
        base_time = datetime(2024, 1, 1)
        return pd.DataFrame({
            'PnL': [-50.0, -100.0, -75.0],
            'Entry Timestamp': [base_time + timedelta(hours=i*10) for i in range(3)],
            'Exit Timestamp': [base_time + timedelta(hours=i*10+5) for i in range(3)]
        })

    @pytest.fixture
    def mixed_trades(self):
        """混合獲利與虧損的交易"""
        base_time = datetime(2024, 1, 1)
        return pd.DataFrame({
            'PnL': [100.0, -50.0, 200.0, -100.0, 150.0],
            'Entry Timestamp': [base_time + timedelta(hours=i*10) for i in range(5)],
            'Exit Timestamp': [base_time + timedelta(hours=i*10+8) for i in range(5)]
        })

    def test_empty_trades(self, engine, empty_trades):
        """測試空交易列表"""
        stats = engine._calculate_trade_statistics(empty_trades)

        assert stats['total_trades'] == 0
        assert stats['win_rate'] == 0.0
        assert stats['avg_win'] == 0.0
        assert stats['avg_loss'] == 0.0
        assert stats['profit_factor'] == 0.0
        assert stats['expectancy'] == 0.0
        assert stats['avg_duration'] == 0.0

    def test_all_winning_trades(self, engine, all_winning_trades):
        """測試全部獲利交易"""
        stats = engine._calculate_trade_statistics(all_winning_trades)

        assert stats['total_trades'] == 3
        assert stats['win_rate'] == 1.0  # 100% 勝率
        assert stats['avg_win'] == pytest.approx(150.0)  # (100 + 200 + 150) / 3
        assert stats['avg_loss'] == 0.0  # 沒有虧損交易
        assert stats['profit_factor'] == 0.0  # 沒有虧損，定義為 0
        assert stats['expectancy'] == pytest.approx(150.0)  # avg_win * 1.0
        assert stats['avg_duration'] == pytest.approx(5.0)  # 5 小時

    def test_all_losing_trades(self, engine, all_losing_trades):
        """測試全部虧損交易"""
        stats = engine._calculate_trade_statistics(all_losing_trades)

        assert stats['total_trades'] == 3
        assert stats['win_rate'] == 0.0  # 0% 勝率
        assert stats['avg_win'] == 0.0  # 沒有獲利交易
        assert stats['avg_loss'] == pytest.approx(-75.0)  # (-50 + -100 + -75) / 3
        assert stats['profit_factor'] == 0.0  # 沒有獲利，定義為 0
        assert stats['expectancy'] == pytest.approx(-75.0)  # avg_loss * 1.0
        assert stats['avg_duration'] == pytest.approx(5.0)  # 5 小時

    def test_mixed_trades(self, engine, mixed_trades):
        """測試混合交易（包含獲利和虧損）"""
        stats = engine._calculate_trade_statistics(mixed_trades)

        # 基本統計
        assert stats['total_trades'] == 5
        assert stats['win_rate'] == pytest.approx(0.6)  # 3 勝 / 5 = 60%

        # 平均獲利/虧損
        # 獲利: [100, 200, 150] → avg = 150
        # 虧損: [-50, -100] → avg = -75
        assert stats['avg_win'] == pytest.approx(150.0)
        assert stats['avg_loss'] == pytest.approx(-75.0)

        # 獲利因子
        # total_wins = 100 + 200 + 150 = 450
        # total_losses = 50 + 100 = 150
        # profit_factor = 450 / 150 = 3.0
        assert stats['profit_factor'] == pytest.approx(3.0)

        # 期望值
        # expectancy = avg_win * win_rate + avg_loss * (1 - win_rate)
        # = 150 * 0.6 + (-75) * 0.4
        # = 90 - 30 = 60
        assert stats['expectancy'] == pytest.approx(60.0)

        # 平均持倉時間
        assert stats['avg_duration'] == pytest.approx(8.0)  # 8 小時

    def test_single_trade_win(self, engine):
        """測試單筆獲利交易"""
        base_time = datetime(2024, 1, 1)
        single_trade = pd.DataFrame({
            'PnL': [250.0],
            'Entry Timestamp': [base_time],
            'Exit Timestamp': [base_time + timedelta(hours=10)]
        })

        stats = engine._calculate_trade_statistics(single_trade)

        assert stats['total_trades'] == 1
        assert stats['win_rate'] == 1.0
        assert stats['avg_win'] == pytest.approx(250.0)
        assert stats['avg_loss'] == 0.0
        assert stats['profit_factor'] == 0.0
        assert stats['expectancy'] == pytest.approx(250.0)
        assert stats['avg_duration'] == pytest.approx(10.0)

    def test_single_trade_loss(self, engine):
        """測試單筆虧損交易"""
        base_time = datetime(2024, 1, 1)
        single_trade = pd.DataFrame({
            'PnL': [-150.0],
            'Entry Timestamp': [base_time],
            'Exit Timestamp': [base_time + timedelta(hours=6)]
        })

        stats = engine._calculate_trade_statistics(single_trade)

        assert stats['total_trades'] == 1
        assert stats['win_rate'] == 0.0
        assert stats['avg_win'] == 0.0
        assert stats['avg_loss'] == pytest.approx(-150.0)
        assert stats['profit_factor'] == 0.0
        assert stats['expectancy'] == pytest.approx(-150.0)
        assert stats['avg_duration'] == pytest.approx(6.0)

    def test_zero_pnl_trades(self, engine):
        """測試損益為零的交易（平手）"""
        base_time = datetime(2024, 1, 1)
        zero_trades = pd.DataFrame({
            'PnL': [0.0, 0.0, 0.0],
            'Entry Timestamp': [base_time + timedelta(hours=i*5) for i in range(3)],
            'Exit Timestamp': [base_time + timedelta(hours=i*5+3) for i in range(3)]
        })

        stats = engine._calculate_trade_statistics(zero_trades)

        assert stats['total_trades'] == 3
        assert stats['win_rate'] == 0.0  # 沒有獲利交易
        assert stats['avg_win'] == 0.0
        assert stats['avg_loss'] == 0.0
        assert stats['profit_factor'] == 0.0
        assert stats['expectancy'] == 0.0
        assert stats['avg_duration'] == pytest.approx(3.0)

    def test_extreme_values(self, engine):
        """測試極端數值"""
        base_time = datetime(2024, 1, 1)
        extreme_trades = pd.DataFrame({
            'PnL': [10000.0, -9000.0, 500.0, -100.0],
            'Entry Timestamp': [base_time + timedelta(hours=i*20) for i in range(4)],
            'Exit Timestamp': [base_time + timedelta(hours=i*20+15) for i in range(4)]
        })

        stats = engine._calculate_trade_statistics(extreme_trades)

        assert stats['total_trades'] == 4
        assert stats['win_rate'] == pytest.approx(0.5)  # 2 勝 / 4 = 50%

        # 獲利: [10000, 500] → avg = 5250
        # 虧損: [-9000, -100] → avg = -4550
        assert stats['avg_win'] == pytest.approx(5250.0)
        assert stats['avg_loss'] == pytest.approx(-4550.0)

        # profit_factor = 10500 / 9100 ≈ 1.154
        assert stats['profit_factor'] == pytest.approx(1.154, abs=0.01)

        # expectancy = 5250 * 0.5 + (-4550) * 0.5 = 350
        assert stats['expectancy'] == pytest.approx(350.0)

        assert stats['avg_duration'] == pytest.approx(15.0)

    def test_varying_durations(self, engine):
        """測試不同持倉時間"""
        base_time = datetime(2024, 1, 1)
        varying_duration_trades = pd.DataFrame({
            'PnL': [100.0, -50.0, 200.0],
            'Entry Timestamp': [
                base_time,
                base_time + timedelta(hours=10),
                base_time + timedelta(hours=30)
            ],
            'Exit Timestamp': [
                base_time + timedelta(hours=2),   # 2 小時
                base_time + timedelta(hours=20),  # 10 小時
                base_time + timedelta(hours=50)   # 20 小時
            ]
        })

        stats = engine._calculate_trade_statistics(varying_duration_trades)

        # 平均持倉時間 = (2 + 10 + 20) / 3 ≈ 10.67 小時
        assert stats['avg_duration'] == pytest.approx(10.67, abs=0.01)


class TestIntegration:
    """整合測試：確保 _calculate_trade_statistics 在完整流程中正常工作"""

    def test_method_integration(self):
        """測試方法在 _calculate_metrics 中的整合"""
        # 這個測試確保重構後方法簽名和返回值正確
        config = BacktestConfig(
            symbol='BTCUSDT',
            timeframe='1h',
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            initial_capital=10000,
            leverage=1
        )
        engine = BacktestEngine(config)

        # 建立模擬交易資料
        base_time = datetime(2024, 1, 1)
        trades_df = pd.DataFrame({
            'PnL': [100.0, -50.0, 200.0],
            'Entry Timestamp': [base_time + timedelta(hours=i*10) for i in range(3)],
            'Exit Timestamp': [base_time + timedelta(hours=i*10+5) for i in range(3)]
        })

        # 呼叫方法
        stats = engine._calculate_trade_statistics(trades_df)

        # 確認回傳的字典包含所有必要欄位
        required_keys = [
            'total_trades', 'win_rate', 'avg_win', 'avg_loss',
            'profit_factor', 'expectancy', 'avg_duration'
        ]
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"

        # 確認值的類型正確
        assert isinstance(stats['total_trades'], int)
        assert isinstance(stats['win_rate'], float)
        assert isinstance(stats['avg_win'], (float, np.floating))
        assert isinstance(stats['avg_loss'], (float, np.floating))
        assert isinstance(stats['profit_factor'], (float, np.floating))
        assert isinstance(stats['expectancy'], (float, np.floating))
        assert isinstance(stats['avg_duration'], (float, np.floating))


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
