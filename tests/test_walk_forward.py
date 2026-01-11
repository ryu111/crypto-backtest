"""
Walk-Forward 分析器測試
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.optimizer import WalkForwardAnalyzer, WFAResult, WindowResult
from src.backtester.engine import BacktestConfig
from src.strategies.base import BaseStrategy


# === Fixtures ===

@pytest.fixture
def sample_data():
    """生成測試資料"""
    dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='1h')
    np.random.seed(42)

    # 生成帶趨勢的價格
    trend = np.linspace(40000, 42000, len(dates))
    noise = np.random.normal(0, 100, len(dates))
    close = trend + noise

    data = pd.DataFrame({
        'open': close * 0.999,
        'high': close * 1.001,
        'low': close * 0.998,
        'close': close,
        'volume': np.random.uniform(100, 200, len(dates))
    }, index=dates)

    return data


@pytest.fixture
def backtest_config():
    """回測配置"""
    return BacktestConfig(
        symbol='BTCUSDT',
        timeframe='1h',
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 6, 30),
        initial_capital=10000,
        leverage=1
    )


@pytest.fixture
def simple_strategy():
    """簡單測試策略"""

    class SimpleStrategy(BaseStrategy):
        name = "simple_test"
        strategy_type = "trend"
        params = {'threshold': 0.01}

        def calculate_indicators(self, data):
            return {'sma': data['close'].rolling(20).mean()}

        def generate_signals(self, data):
            sma = data['close'].rolling(20).mean()
            long_entry = data['close'] > sma * 1.01
            long_exit = data['close'] < sma * 0.99
            short_entry = pd.Series(False, index=data.index)
            short_exit = pd.Series(False, index=data.index)
            return long_entry, long_exit, short_entry, short_exit

    return SimpleStrategy()


# === 基礎測試 ===

def test_analyzer_initialization(backtest_config):
    """測試分析器初始化"""
    analyzer = WalkForwardAnalyzer(
        config=backtest_config,
        mode='rolling',
        optimize_metric='sharpe_ratio'
    )

    assert analyzer.mode == 'rolling'
    assert analyzer.optimize_metric == 'sharpe_ratio'
    assert analyzer.config == backtest_config


def test_invalid_mode(backtest_config):
    """測試無效模式"""
    with pytest.raises(ValueError, match="不支援的模式"):
        WalkForwardAnalyzer(
            config=backtest_config,
            mode='invalid_mode'
        )


# === 窗口切分測試 ===

def test_split_windows_rolling(backtest_config, sample_data):
    """測試滾動窗口切分"""
    analyzer = WalkForwardAnalyzer(config=backtest_config, mode='rolling')
    windows = analyzer._split_windows(sample_data, n_windows=3, is_ratio=0.7)

    assert len(windows) > 0
    assert len(windows) <= 3

    # 檢查每個窗口
    for is_data, oos_data in windows:
        assert len(is_data) > 0
        assert len(oos_data) > 0
        assert is_data.index[-1] < oos_data.index[0]  # IS 在 OOS 之前


def test_split_windows_expanding(backtest_config, sample_data):
    """測試擴展窗口切分"""
    analyzer = WalkForwardAnalyzer(config=backtest_config, mode='expanding')
    windows = analyzer._split_windows(sample_data, n_windows=3, is_ratio=0.7)

    assert len(windows) > 0

    # 檢查 IS 窗口逐步增大
    is_lengths = [len(is_data) for is_data, _ in windows]
    for i in range(len(is_lengths) - 1):
        assert is_lengths[i] < is_lengths[i + 1]


def test_split_windows_anchored(backtest_config, sample_data):
    """測試錨定窗口切分"""
    analyzer = WalkForwardAnalyzer(config=backtest_config, mode='anchored')
    windows = analyzer._split_windows(sample_data, n_windows=3, is_ratio=0.7)

    assert len(windows) > 0

    # 檢查起點相同
    is_starts = [is_data.index[0] for is_data, _ in windows]
    assert all(start == is_starts[0] for start in is_starts)


def test_invalid_is_ratio(backtest_config, sample_data, simple_strategy):
    """測試無效的 IS 比例"""
    analyzer = WalkForwardAnalyzer(config=backtest_config)

    with pytest.raises(ValueError, match="is_ratio 必須在 0 到 1 之間"):
        analyzer.analyze(
            strategy=simple_strategy,
            data=sample_data,
            param_grid={'threshold': [0.01]},
            n_windows=2,
            is_ratio=1.5
        )


# === 優化測試 ===

def test_optimize_window(backtest_config, sample_data, simple_strategy):
    """測試窗口優化"""
    analyzer = WalkForwardAnalyzer(config=backtest_config)

    param_grid = {'threshold': [0.005, 0.01, 0.02]}

    best_params, is_result = analyzer._optimize_window(
        strategy=simple_strategy,
        is_data=sample_data,
        param_grid=param_grid,
        verbose=False
    )

    assert best_params is not None
    assert 'threshold' in best_params
    assert best_params['threshold'] in param_grid['threshold']
    assert is_result.total_return is not None


def test_test_window(backtest_config, sample_data, simple_strategy):
    """測試 OOS 窗口測試"""
    analyzer = WalkForwardAnalyzer(config=backtest_config)

    params = {'threshold': 0.01}
    oos_result = analyzer._test_window(
        strategy=simple_strategy,
        oos_data=sample_data,
        params=params
    )

    assert oos_result is not None
    assert hasattr(oos_result, 'total_return')
    assert hasattr(oos_result, 'sharpe_ratio')


# === 完整分析測試 ===

def test_full_analysis(backtest_config, sample_data, simple_strategy):
    """測試完整 WFA 分析"""
    analyzer = WalkForwardAnalyzer(
        config=backtest_config,
        mode='rolling',
        optimize_metric='sharpe_ratio'
    )

    param_grid = {'threshold': [0.005, 0.01, 0.02]}

    result = analyzer.analyze(
        strategy=simple_strategy,
        data=sample_data,
        param_grid=param_grid,
        n_windows=3,
        is_ratio=0.7,
        min_trades=1,  # 降低門檻以通過測試
        verbose=False
    )

    # 檢查結果結構
    assert isinstance(result, WFAResult)
    assert len(result.windows) > 0
    assert result.efficiency is not None
    assert result.consistency >= 0 and result.consistency <= 1


# === WindowResult 測試 ===

def test_window_result():
    """測試 WindowResult 資料類別"""
    window = WindowResult(
        window_id=1,
        is_start=datetime(2023, 1, 1),
        is_end=datetime(2023, 3, 1),
        oos_start=datetime(2023, 3, 1),
        oos_end=datetime(2023, 4, 1),
        is_return=0.15,
        oos_return=0.12,
        is_sharpe=1.5,
        oos_sharpe=1.2,
        best_params={'threshold': 0.01}
    )

    result_dict = window.to_dict()
    assert result_dict['window_id'] == 1
    assert result_dict['is_return'] == 0.15
    assert result_dict['oos_return'] == 0.12


# === WFAResult 測試 ===

def test_wfa_result():
    """測試 WFAResult 資料類別"""
    windows = [
        WindowResult(
            window_id=1,
            is_start=datetime(2023, 1, 1),
            is_end=datetime(2023, 3, 1),
            oos_start=datetime(2023, 3, 1),
            oos_end=datetime(2023, 4, 1),
            is_return=0.15,
            oos_return=0.12,
            is_sharpe=1.5,
            oos_sharpe=1.2,
            best_params={}
        ),
        WindowResult(
            window_id=2,
            is_start=datetime(2023, 2, 1),
            is_end=datetime(2023, 4, 1),
            oos_start=datetime(2023, 4, 1),
            oos_end=datetime(2023, 5, 1),
            is_return=0.20,
            oos_return=0.18,
            is_sharpe=1.8,
            oos_sharpe=1.6,
            best_params={}
        )
    ]

    result = WFAResult(
        windows=windows,
        efficiency=0.85,
        oos_returns=[0.12, 0.18],
        is_returns=[0.15, 0.20],
        oos_sharpes=[1.2, 1.6],
        is_sharpes=[1.5, 1.8],
        consistency=1.0
    )

    # 檢查自動計算的統計指標
    assert result.oos_mean_return == pytest.approx((0.12 + 0.18) / 2)
    assert result.oos_min_return == 0.12
    assert result.oos_max_return == 0.18

    # 檢查 to_dict
    result_dict = result.to_dict()
    assert result_dict['efficiency'] == 0.85
    assert result_dict['n_windows'] == 2

    # 檢查 summary
    summary = result.summary()
    assert 'Walk-Forward' in summary
    assert '0.85' in summary or '85' in summary  # 效率


# === 衰退分析測試 ===

def test_analyze_degradation(backtest_config):
    """測試效能衰退分析"""
    analyzer = WalkForwardAnalyzer(config=backtest_config)

    windows = [
        WindowResult(
            window_id=1,
            is_start=datetime(2023, 1, 1),
            is_end=datetime(2023, 3, 1),
            oos_start=datetime(2023, 3, 1),
            oos_end=datetime(2023, 4, 1),
            is_return=0.20,
            oos_return=0.15,
            is_sharpe=1.5,
            oos_sharpe=1.2,
            is_max_dd=-0.10,
            oos_max_dd=-0.15,
            best_params={}
        )
    ]

    result = WFAResult(
        windows=windows,
        efficiency=0.75,
        oos_returns=[0.15],
        is_returns=[0.20],
        oos_sharpes=[1.2],
        is_sharpes=[1.5],
        consistency=1.0
    )

    degradation = analyzer.analyze_degradation(result)

    assert 'avg_return_degradation' in degradation
    assert 'avg_sharpe_degradation' in degradation
    assert 'avg_max_dd_increase' in degradation


# === 邊界條件測試 ===

def test_empty_data(backtest_config, simple_strategy):
    """測試空資料"""
    analyzer = WalkForwardAnalyzer(config=backtest_config)
    empty_data = pd.DataFrame()

    with pytest.raises(Exception):
        analyzer.analyze(
            strategy=simple_strategy,
            data=empty_data,
            param_grid={'threshold': [0.01]},
            n_windows=2,
            is_ratio=0.7
        )


def test_insufficient_data(backtest_config, simple_strategy):
    """測試資料不足"""
    analyzer = WalkForwardAnalyzer(config=backtest_config)

    # 只有 10 筆資料
    dates = pd.date_range(start='2023-01-01', periods=10, freq='1h')
    small_data = pd.DataFrame({
        'open': [40000] * 10,
        'high': [40100] * 10,
        'low': [39900] * 10,
        'close': [40000] * 10,
        'volume': [100] * 10
    }, index=dates)

    with pytest.raises(Exception):
        analyzer.analyze(
            strategy=simple_strategy,
            data=small_data,
            param_grid={'threshold': [0.01]},
            n_windows=5,
            is_ratio=0.7
        )


def test_single_param_combination(backtest_config, sample_data, simple_strategy):
    """測試單一參數組合"""
    analyzer = WalkForwardAnalyzer(config=backtest_config)

    param_grid = {'threshold': [0.01]}  # 只有一種組合

    result = analyzer.analyze(
        strategy=simple_strategy,
        data=sample_data,
        param_grid=param_grid,
        n_windows=2,
        is_ratio=0.7,
        min_trades=1,
        verbose=False
    )

    assert result is not None
    assert len(result.windows) > 0


# === 整合測試 ===

@pytest.mark.integration
def test_multiple_modes_comparison(backtest_config, sample_data, simple_strategy):
    """測試比較不同窗口模式"""
    modes = ['rolling', 'expanding', 'anchored']
    param_grid = {'threshold': [0.01, 0.02]}

    results = {}

    for mode in modes:
        analyzer = WalkForwardAnalyzer(
            config=backtest_config,
            mode=mode
        )

        try:
            result = analyzer.analyze(
                strategy=simple_strategy,
                data=sample_data,
                param_grid=param_grid,
                n_windows=2,
                is_ratio=0.7,
                min_trades=1,
                verbose=False
            )
            results[mode] = result
        except Exception as e:
            pytest.fail(f"{mode} 模式失敗: {e}")

    # 確保所有模式都成功
    assert len(results) == 3

    # 比較結果
    for mode, result in results.items():
        assert result.efficiency is not None
        assert 0 <= result.consistency <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
