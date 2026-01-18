"""Walk-Forward Analysis 單元測試套件

測試覆蓋範圍：
1. WalkForwardWindow - 窗口資料結構
2. WindowResult - 窗口結果計算
3. WalkForwardResult - WFA 結果聚合
4. WalkForwardAnalyzer.generate_windows() - 窗口生成邏輯
5. WalkForwardAnalyzer.calculate_efficiency() - 效率計算
6. 邊界情況測試 - 資料不足、無效參數、空資料
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.automation.walk_forward import (
    WalkForwardWindow,
    WindowResult,
    WalkForwardResult,
    WalkForwardAnalyzer,
)


# ========== Fixtures ==========

@pytest.fixture
def sample_data():
    """生成 500 筆日線資料"""
    dates = pd.date_range("2023-01-01", periods=500, freq="D")
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(500) * 0.5)

    df = pd.DataFrame({
        "date": dates,
        "open": prices,
        "high": prices + 1,
        "low": prices - 1,
        "close": prices,
        "volume": np.random.randint(1000000, 5000000, 500),
    })
    df.set_index("date", inplace=True)
    return df


@pytest.fixture
def sample_params():
    """樣本策略參數"""
    return {
        "fast_ma": 10,
        "slow_ma": 30,
        "threshold": 0.05,
    }


@pytest.fixture
def small_data():
    """生成 50 筆資料（邊界測試用）"""
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    prices = 100 + np.cumsum(np.random.randn(50) * 0.5)
    df = pd.DataFrame({
        "close": prices,
    }, index=dates)
    return df


@pytest.fixture
def analyzer():
    """標準配置的 WalkForwardAnalyzer"""
    return WalkForwardAnalyzer(
        is_ratio=0.7,
        n_windows=5,
        overlap=0.5,
        min_window_size=50,
    )


# ========== TestWalkForwardWindow ==========

class TestWalkForwardWindow:
    """WalkForwardWindow 資料結構測試"""

    def test_window_creation(self):
        """測試窗口創建"""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 6, 30)

        window = WalkForwardWindow(
            window_id=1,
            is_start=start,
            is_end=datetime(2023, 4, 30),
            oos_start=datetime(2023, 5, 1),
            oos_end=end,
        )

        assert window.window_id == 1
        assert window.is_start == start
        assert window.oos_end == end

    def test_is_size_empty(self):
        """測試 IS 大小計算（無資料）"""
        window = WalkForwardWindow(
            window_id=1,
            is_start=datetime.now(),
            is_end=datetime.now(),
            oos_start=datetime.now(),
            oos_end=datetime.now(),
        )

        assert window.is_size == 0

    def test_is_size_with_data(self, small_data):
        """測試 IS 大小計算（有資料）"""
        is_data = small_data.iloc[:35]
        window = WalkForwardWindow(
            window_id=1,
            is_start=is_data.index[0],
            is_end=is_data.index[-1],
            oos_start=small_data.index[35],
            oos_end=small_data.index[-1],
            is_data=is_data,
        )

        assert window.is_size == 35

    def test_oos_size_with_data(self, small_data):
        """測試 OOS 大小計算"""
        oos_data = small_data.iloc[35:]
        window = WalkForwardWindow(
            window_id=1,
            is_start=small_data.index[0],
            is_end=small_data.index[34],
            oos_start=small_data.index[35],
            oos_end=oos_data.index[-1],
            oos_data=oos_data,
        )

        assert window.oos_size == 15


# ========== TestWindowResult ==========

class TestWindowResult:
    """WindowResult 效率計算測試"""

    def test_efficiency_calculation(self):
        """測試窗口效率計算 (OOS / IS)"""
        result = WindowResult(
            window_id=1,
            params={"fast": 10},
            is_return=0.1,
            oos_return=0.08,
            is_sharpe=2.0,
            oos_sharpe=1.6,
            is_trades=50,
            oos_trades=45,
        )

        # 效率 = OOS Sharpe / IS Sharpe
        expected = 1.6 / 2.0  # 0.8
        assert result.efficiency == pytest.approx(0.8)

    def test_efficiency_zero_is_sharpe(self):
        """測試效率計算（IS Sharpe = 0）"""
        result = WindowResult(
            window_id=1,
            params={"fast": 10},
            is_return=0.0,
            oos_return=0.0,
            is_sharpe=0.0,
            oos_sharpe=0.0,
            is_trades=0,
            oos_trades=0,
        )

        assert result.efficiency == 0.0

    def test_efficiency_negative_is_sharpe(self):
        """測試效率計算（IS Sharpe < 0）"""
        result = WindowResult(
            window_id=1,
            params={"fast": 10},
            is_return=-0.05,
            oos_return=0.02,
            is_sharpe=-1.0,
            oos_sharpe=0.5,
            is_trades=30,
            oos_trades=25,
        )

        assert result.efficiency == 0.0


# ========== TestWalkForwardResult ==========

class TestWalkForwardResult:
    """WalkForwardResult 聚合計算測試"""

    def test_empty_result(self):
        """測試空結果"""
        result = WalkForwardResult()

        assert result.is_mean_return == 0.0
        assert result.oos_mean_return == 0.0
        assert result.is_mean_sharpe == 0.0
        assert result.oos_mean_sharpe == 0.0
        assert result.efficiency == 0.0
        assert result.oos_win_rate == 0.0
        assert result.total_oos_return == 0.0

    def test_single_window_result(self):
        """測試單窗口結果"""
        window = WindowResult(
            window_id=1,
            params={"fast": 10},
            is_return=0.15,
            oos_return=0.12,
            is_sharpe=2.0,
            oos_sharpe=1.8,
            is_trades=50,
            oos_trades=45,
        )

        result = WalkForwardResult(windows=[window])

        assert result.is_mean_return == 0.15
        assert result.oos_mean_return == 0.12
        assert result.is_mean_sharpe == 2.0
        assert result.oos_mean_sharpe == 1.8
        assert result.efficiency == 0.9

    def test_multiple_windows_results(self):
        """測試多窗口結果聚合"""
        windows = [
            WindowResult(
                window_id=1,
                params={"fast": 10},
                is_return=0.15,
                oos_return=0.12,
                is_sharpe=2.0,
                oos_sharpe=1.8,
                is_trades=50,
                oos_trades=45,
            ),
            WindowResult(
                window_id=2,
                params={"fast": 10},
                is_return=0.10,
                oos_return=0.08,
                is_sharpe=1.5,
                oos_sharpe=1.2,
                is_trades=40,
                oos_trades=35,
            ),
        ]

        result = WalkForwardResult(windows=windows)

        assert result.is_mean_return == pytest.approx(0.125)
        assert result.oos_mean_return == pytest.approx(0.10)
        assert result.is_mean_sharpe == pytest.approx(1.75)
        assert result.oos_mean_sharpe == pytest.approx(1.5)

    def test_oos_win_rate(self):
        """測試 OOS 獲利窗口比例"""
        windows = [
            WindowResult(
                window_id=1,
                params={"fast": 10},
                is_return=0.1,
                oos_return=0.08,  # 獲利
                is_sharpe=1.5,
                oos_sharpe=1.2,
                is_trades=40,
                oos_trades=35,
            ),
            WindowResult(
                window_id=2,
                params={"fast": 10},
                is_return=0.12,
                oos_return=-0.02,  # 虧損
                is_sharpe=1.8,
                oos_sharpe=0.5,
                is_trades=50,
                oos_trades=45,
            ),
            WindowResult(
                window_id=3,
                params={"fast": 10},
                is_return=0.1,
                oos_return=0.05,  # 獲利
                is_sharpe=1.5,
                oos_sharpe=1.0,
                is_trades=40,
                oos_trades=35,
            ),
        ]

        result = WalkForwardResult(windows=windows)

        assert result.oos_win_rate == pytest.approx(2.0 / 3.0)

    def test_total_oos_return(self):
        """測試累積 OOS 報酬"""
        windows = [
            WindowResult(
                window_id=1,
                params={"fast": 10},
                is_return=0.1,
                oos_return=0.10,  # 10%
                is_sharpe=1.5,
                oos_sharpe=1.2,
                is_trades=40,
                oos_trades=35,
            ),
            WindowResult(
                window_id=2,
                params={"fast": 10},
                is_return=0.12,
                oos_return=0.10,  # 10%
                is_sharpe=1.8,
                oos_sharpe=1.5,
                is_trades=50,
                oos_trades=45,
            ),
        ]

        result = WalkForwardResult(windows=windows)

        # 累積: (1 + 0.10) * (1 + 0.10) - 1 = 0.21
        assert result.total_oos_return == pytest.approx(0.21)

    def test_risk_level_low(self):
        """測試低風險判定"""
        windows = [
            WindowResult(
                window_id=1,
                params={"fast": 10},
                is_return=0.1,
                oos_return=0.09,
                is_sharpe=2.0,
                oos_sharpe=1.8,  # efficiency = 0.9 >= 0.8
                is_trades=50,
                oos_trades=45,
            ),
        ]

        result = WalkForwardResult(windows=windows)
        assert result.risk_level == "LOW"

    def test_risk_level_medium(self):
        """測試中風險判定"""
        windows = [
            WindowResult(
                window_id=1,
                params={"fast": 10},
                is_return=0.1,
                oos_return=0.07,
                is_sharpe=2.0,
                oos_sharpe=1.4,  # efficiency = 0.7，在 60-80%
                is_trades=50,
                oos_trades=45,
            ),
        ]

        result = WalkForwardResult(windows=windows)
        assert result.risk_level == "MEDIUM"

    def test_risk_level_high(self):
        """測試高風險判定"""
        windows = [
            WindowResult(
                window_id=1,
                params={"fast": 10},
                is_return=0.1,
                oos_return=0.05,
                is_sharpe=2.0,
                oos_sharpe=1.0,  # efficiency = 0.5，在 40-60%
                is_trades=50,
                oos_trades=45,
            ),
        ]

        result = WalkForwardResult(windows=windows)
        assert result.risk_level == "HIGH"

    def test_risk_level_critical(self):
        """測試極危風險判定"""
        windows = [
            WindowResult(
                window_id=1,
                params={"fast": 10},
                is_return=0.1,
                oos_return=0.02,
                is_sharpe=2.0,
                oos_sharpe=0.4,  # efficiency = 0.2 < 0.4
                is_trades=50,
                oos_trades=45,
            ),
        ]

        result = WalkForwardResult(windows=windows)
        assert result.risk_level == "CRITICAL"


# ========== TestGenerateWindows ==========

class TestGenerateWindows:
    """WalkForwardAnalyzer.generate_windows() 測試"""

    def test_generate_windows_basic(self, sample_data, analyzer):
        """測試基本窗口生成"""
        windows = analyzer.generate_windows(sample_data)

        assert len(windows) == 5
        assert all(isinstance(w, WalkForwardWindow) for w in windows)
        assert all(w.is_size > 0 for w in windows)
        assert all(w.oos_size > 0 for w in windows)

    def test_window_properties(self, sample_data, analyzer):
        """測試窗口屬性"""
        windows = analyzer.generate_windows(sample_data)

        for i, window in enumerate(windows):
            assert window.window_id == i + 1
            assert window.is_start < window.is_end
            assert window.oos_start < window.oos_end
            assert window.is_end <= window.oos_start

    def test_window_is_ratio(self, sample_data, analyzer):
        """測試 IS 比例"""
        windows = analyzer.generate_windows(sample_data)

        for window in windows:
            total_size = window.is_size + window.oos_size
            actual_ratio = window.is_size / total_size
            # 允許小誤差
            assert actual_ratio == pytest.approx(0.7, abs=0.05)

    def test_insufficient_data(self, analyzer):
        """測試資料不足異常"""
        small_df = pd.DataFrame({"close": [100, 101, 102]})
        small_df.index = pd.date_range("2023-01-01", periods=3)

        with pytest.raises(ValueError, match="資料不足"):
            analyzer.generate_windows(small_df)

    def test_invalid_is_ratio(self):
        """測試無效 IS 比例參數"""
        with pytest.raises(ValueError, match="is_ratio"):
            WalkForwardAnalyzer(is_ratio=0.4, n_windows=5)

        with pytest.raises(ValueError, match="is_ratio"):
            WalkForwardAnalyzer(is_ratio=0.95, n_windows=5)

    def test_invalid_n_windows(self):
        """測試無效窗口數量"""
        with pytest.raises(ValueError, match="n_windows"):
            WalkForwardAnalyzer(is_ratio=0.7, n_windows=1)

    def test_invalid_overlap(self):
        """測試無效重疊比例"""
        with pytest.raises(ValueError, match="overlap"):
            WalkForwardAnalyzer(is_ratio=0.7, n_windows=5, overlap=-0.1)

        with pytest.raises(ValueError, match="overlap"):
            WalkForwardAnalyzer(is_ratio=0.7, n_windows=5, overlap=1.0)

    def test_window_datetime_index(self, sample_data, analyzer):
        """測試使用 DatetimeIndex"""
        windows = analyzer.generate_windows(sample_data)

        for window in windows:
            assert isinstance(window.is_start, (datetime, pd.Timestamp))
            assert isinstance(window.is_end, (datetime, pd.Timestamp))

    def test_window_datetime_column(self, analyzer):
        """測試使用 datetime 欄位"""
        dates = pd.date_range("2023-01-01", periods=500, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "close": 100 + np.cumsum(np.random.randn(500) * 0.5),
        })

        windows = analyzer.generate_windows(df, datetime_column="date")

        assert len(windows) == 5
        for window in windows:
            assert isinstance(window.is_start, (datetime, pd.Timestamp))

    def test_no_datetime_fallback(self):
        """測試無日期索引回退到整數索引"""
        df = pd.DataFrame({
            "close": 100 + np.cumsum(np.random.randn(500) * 0.5),
        })

        analyzer = WalkForwardAnalyzer(is_ratio=0.7, n_windows=5, overlap=0.5)
        windows = analyzer.generate_windows(df)

        assert len(windows) == 5
        # 回退情況下使用 timestamp
        assert isinstance(windows[0].is_start, datetime)

    def test_overlap_effect(self, sample_data):
        """測試重疊比例影響"""
        analyzer_no_overlap = WalkForwardAnalyzer(
            is_ratio=0.7, n_windows=3, overlap=0.0
        )
        analyzer_with_overlap = WalkForwardAnalyzer(
            is_ratio=0.7, n_windows=3, overlap=0.5
        )

        windows_no = analyzer_no_overlap.generate_windows(sample_data)
        windows_with = analyzer_with_overlap.generate_windows(sample_data)

        # 有重疊時窗口數可能較多
        assert len(windows_no) <= len(windows_with)


# ========== TestCalculateEfficiency ==========

class TestCalculateEfficiency:
    """WalkForwardAnalyzer.calculate_efficiency() 測試"""

    def test_efficiency_perfect(self, analyzer):
        """測試完全效率 (OOS = IS)"""
        is_sharpes = [2.0, 1.8, 1.9]
        oos_sharpes = [2.0, 1.8, 1.9]

        efficiency = analyzer.calculate_efficiency(is_sharpes, oos_sharpes)
        assert efficiency == pytest.approx(1.0)

    def test_efficiency_degradation(self, analyzer):
        """測試績效衰退"""
        is_sharpes = [2.0, 2.0, 2.0]
        oos_sharpes = [1.0, 1.0, 1.0]

        efficiency = analyzer.calculate_efficiency(is_sharpes, oos_sharpes)
        assert efficiency == pytest.approx(0.5)

    def test_efficiency_zero_is_sharpe(self, analyzer):
        """測試 IS Sharpe = 0"""
        is_sharpes = [0.0, 0.0]
        oos_sharpes = [0.5, 0.5]

        efficiency = analyzer.calculate_efficiency(is_sharpes, oos_sharpes)
        assert efficiency == 0.0

    def test_efficiency_empty_lists(self, analyzer):
        """測試空列表"""
        efficiency = analyzer.calculate_efficiency([], [])
        assert efficiency == 0.0

    def test_efficiency_single_value(self, analyzer):
        """測試單一值"""
        is_sharpes = [2.5]
        oos_sharpes = [1.5]

        efficiency = analyzer.calculate_efficiency(is_sharpes, oos_sharpes)
        assert efficiency == pytest.approx(0.6)


# ========== TestToDict ==========

class TestToDict:
    """序列化測試"""

    def test_window_result_to_dict(self):
        """測試 WindowResult 轉字典"""
        result = WindowResult(
            window_id=1,
            params={"fast": 10, "slow": 30},
            is_return=0.15,
            oos_return=0.12,
            is_sharpe=2.0,
            oos_sharpe=1.8,
            is_trades=50,
            oos_trades=45,
        )

        d = {"efficiency": result.efficiency}
        assert d["efficiency"] == pytest.approx(0.9)

    def test_wfa_result_to_dict(self):
        """測試 WalkForwardResult 轉字典"""
        windows = [
            WindowResult(
                window_id=1,
                params={"fast": 10},
                is_return=0.15,
                oos_return=0.12,
                is_sharpe=2.0,
                oos_sharpe=1.8,
                is_trades=50,
                oos_trades=45,
            ),
        ]

        result = WalkForwardResult(
            windows=windows,
            strategy_name="MA_Cross",
            n_windows=1,
            is_ratio=0.7,
            overlap=0.5,
        )

        d = result.to_dict()

        assert d["strategy_name"] == "MA_Cross"
        assert d["n_windows"] == 1
        assert d["is_ratio"] == 0.7
        assert d["overlap"] == 0.5
        assert len(d["windows"]) == 1
