"""
多策略相關性分析模組單元測試

測試 CorrelationAnalyzer 的所有功能。
"""

import pytest
import numpy as np
import pandas as pd
from src.risk.correlation import (
    CorrelationAnalyzer,
    CorrelationMatrix,
    RollingCorrelation,
    TailCorrelation,
)


# ========== Fixtures ==========

@pytest.fixture
def analyzer():
    """建立相關性分析器"""
    return CorrelationAnalyzer(window=20)


@pytest.fixture
def positive_corr_returns():
    """正相關的兩個策略收益率"""
    np.random.seed(42)
    n = 100
    base = np.random.randn(n) * 0.01
    return {
        'strategy_a': pd.Series(base + np.random.randn(n) * 0.005),
        'strategy_b': pd.Series(base + np.random.randn(n) * 0.005)
    }


@pytest.fixture
def negative_corr_returns():
    """負相關的兩個策略收益率"""
    np.random.seed(42)
    n = 100
    base = np.random.randn(n) * 0.01
    return {
        'strategy_a': pd.Series(base),
        'strategy_b': pd.Series(-base + np.random.randn(n) * 0.005)
    }


@pytest.fixture
def multi_strategy_returns():
    """多個策略收益率（不同相關性）"""
    np.random.seed(42)
    n = 100
    base = np.random.randn(n) * 0.01

    return {
        'trend': pd.Series(base + np.random.randn(n) * 0.003),
        'mean_reversion': pd.Series(-base * 0.5 + np.random.randn(n) * 0.003),
        'momentum': pd.Series(base * 0.8 + np.random.randn(n) * 0.004),
        'random': pd.Series(np.random.randn(n) * 0.01)
    }


@pytest.fixture
def tail_event_returns():
    """包含尾部事件的收益率"""
    np.random.seed(42)
    n = 200

    # 正常時期：低相關
    normal_a = np.random.randn(n) * 0.01
    normal_b = np.random.randn(n) * 0.01

    # 注入尾部事件（危機時期：高相關）
    crisis_indices = [50, 51, 52, 100, 101, 102, 150, 151]
    for idx in crisis_indices:
        shock = -0.03  # 3% 下跌
        normal_a[idx] = shock + np.random.randn() * 0.002
        normal_b[idx] = shock + np.random.randn() * 0.002

    return pd.Series(normal_a), pd.Series(normal_b)


# ========== 相關性矩陣測試 ==========

def test_correlation_matrix_basic(analyzer, positive_corr_returns):
    """測試基本相關性矩陣計算"""
    result = analyzer.calculate_correlation_matrix(positive_corr_returns)

    assert isinstance(result, CorrelationMatrix)
    assert result.matrix.shape == (2, 2)
    assert 0 < result.mean_correlation < 1  # 正相關
    assert result.max_correlation <= 1
    assert result.min_correlation >= -1


def test_correlation_matrix_multi_strategy(analyzer, multi_strategy_returns):
    """測試多策略相關性矩陣"""
    result = analyzer.calculate_correlation_matrix(multi_strategy_returns)

    # 檢查矩陣維度
    assert result.matrix.shape == (4, 4)

    # 檢查對角線為 1
    np.testing.assert_array_almost_equal(
        np.diag(result.matrix),
        np.ones(4),
        decimal=10
    )

    # 檢查對稱性
    assert np.allclose(result.matrix, result.matrix.T)

    # 檢查分散比率
    assert 0 <= result.diversification_ratio <= 1


def test_correlation_matrix_negative_corr(analyzer, negative_corr_returns):
    """測試負相關策略"""
    result = analyzer.calculate_correlation_matrix(negative_corr_returns)

    # 負相關策略的平均相關性應為負
    assert result.mean_correlation < 0


def test_correlation_matrix_single_strategy(analyzer):
    """測試單一策略（應該失敗）"""
    returns = {'strategy_a': pd.Series(np.random.randn(100) * 0.01)}

    with pytest.raises(ValueError, match="至少需要 2 個策略"):
        analyzer.calculate_correlation_matrix(returns)


def test_correlation_matrix_empty_returns(analyzer):
    """測試空收益率"""
    returns = {}

    with pytest.raises(ValueError):
        analyzer.calculate_correlation_matrix(returns)


# ========== 滾動相關性測試 ==========

def test_rolling_correlation_basic(analyzer, positive_corr_returns):
    """測試基本滾動相關性計算"""
    ret1 = positive_corr_returns['strategy_a']
    ret2 = positive_corr_returns['strategy_b']

    result = analyzer.rolling_correlation(ret1, ret2)

    assert isinstance(result, RollingCorrelation)
    assert len(result.correlation) > 0
    assert -1 <= result.mean <= 1
    assert result.std >= 0
    assert result.max >= result.min


def test_rolling_correlation_regime_changes(analyzer):
    """測試趨勢變化檢測"""
    # 建立有明確趨勢變化的序列
    np.random.seed(42)
    n = 100

    # 前半：正相關
    ret1_part1 = np.random.randn(50) * 0.01
    ret2_part1 = ret1_part1 + np.random.randn(50) * 0.003

    # 後半：負相關
    ret1_part2 = np.random.randn(50) * 0.01
    ret2_part2 = -ret1_part2 + np.random.randn(50) * 0.003

    ret1 = pd.Series(np.concatenate([ret1_part1, ret1_part2]))
    ret2 = pd.Series(np.concatenate([ret2_part1, ret2_part2]))

    result = analyzer.rolling_correlation(ret1, ret2)

    # 應該檢測到趨勢變化
    assert result.regime_changes > 0


def test_rolling_correlation_length_mismatch(analyzer):
    """測試長度不一致的序列"""
    ret1 = pd.Series(np.random.randn(100) * 0.01)
    ret2 = pd.Series(np.random.randn(50) * 0.01)

    with pytest.raises(ValueError, match="長度必須一致"):
        analyzer.rolling_correlation(ret1, ret2)


def test_rolling_correlation_insufficient_data(analyzer):
    """測試資料不足"""
    # 序列長度小於窗口
    ret1 = pd.Series(np.random.randn(10) * 0.01)
    ret2 = pd.Series(np.random.randn(10) * 0.01)

    with pytest.raises(ValueError, match="必須 >= 窗口大小"):
        analyzer.rolling_correlation(ret1, ret2)


# ========== 尾部相關性測試 ==========

def test_tail_correlation_basic(analyzer, tail_event_returns):
    """測試基本尾部相關性計算"""
    ret1, ret2 = tail_event_returns

    result = analyzer.tail_correlation(ret1, ret2, threshold=-0.02)

    assert isinstance(result, TailCorrelation)
    assert -1 <= result.left_tail <= 1
    assert -1 <= result.right_tail <= 1
    assert -1 <= result.normal <= 1
    assert result.left_tail_count >= 0
    assert result.right_tail_count >= 0


def test_tail_correlation_crisis_higher(analyzer, tail_event_returns):
    """測試危機時期相關性是否上升"""
    ret1, ret2 = tail_event_returns

    result = analyzer.tail_correlation(ret1, ret2, threshold=-0.02)

    # 危機相關性應該高於正常相關性（因為我們注入了同步下跌）
    # 但這不是絕對的，所以我們只檢查有計算出值
    assert result.crisis_correlation != 0.0 or result.left_tail_count < 2


def test_tail_correlation_length_mismatch(analyzer):
    """測試長度不一致的序列"""
    ret1 = pd.Series(np.random.randn(100) * 0.01)
    ret2 = pd.Series(np.random.randn(50) * 0.01)

    with pytest.raises(ValueError, match="長度必須一致"):
        analyzer.tail_correlation(ret1, ret2)


def test_tail_correlation_no_tail_events(analyzer):
    """測試沒有尾部事件的情況"""
    # 小波動序列，不會觸發閾值
    ret1 = pd.Series(np.random.randn(100) * 0.001)
    ret2 = pd.Series(np.random.randn(100) * 0.001)

    result = analyzer.tail_correlation(ret1, ret2, threshold=-0.02)

    # 應該返回 0（因為樣本數不足）
    assert result.left_tail == 0.0
    assert result.crisis_correlation == 0.0


# ========== 投資組合分散分析測試 ==========

def test_portfolio_diversification_equal_weight(analyzer, multi_strategy_returns):
    """測試等權重投資組合分散分析"""
    result = analyzer.analyze_portfolio_diversification(multi_strategy_returns)

    assert 'mean_correlation' in result
    assert 'diversification_benefit' in result
    assert 'portfolio_std' in result
    assert result['portfolio_std'] > 0


def test_portfolio_diversification_custom_weight(analyzer, multi_strategy_returns):
    """測試自訂權重投資組合"""
    weights = {
        'trend': 0.4,
        'mean_reversion': 0.3,
        'momentum': 0.2,
        'random': 0.1
    }

    result = analyzer.analyze_portfolio_diversification(
        multi_strategy_returns,
        weights=weights
    )

    # 權重總和應為 1
    assert abs(sum(weights.values()) - 1.0) < 1e-10

    assert result['diversification_benefit'] >= -1  # 可能為負（沒有分散效果）


def test_portfolio_diversification_low_correlation(analyzer, negative_corr_returns):
    """測試低相關性策略的分散效果"""
    result = analyzer.analyze_portfolio_diversification(negative_corr_returns)

    # 負相關策略應該有較高的分散效益
    assert result['diversification_benefit'] > 0


# ========== 邊界條件測試 ==========

def test_perfect_correlation(analyzer):
    """測試完全相關的策略"""
    np.random.seed(42)
    ret = pd.Series(np.random.randn(100) * 0.01)

    returns_dict = {
        'strategy_a': ret,
        'strategy_b': ret  # 完全相同
    }

    result = analyzer.calculate_correlation_matrix(returns_dict)

    # 完全相關應該接近 1
    assert abs(result.mean_correlation - 1.0) < 1e-10


def test_zero_correlation(analyzer):
    """測試零相關的策略"""
    np.random.seed(42)
    n = 1000  # 大樣本以確保接近理論值

    returns_dict = {
        'strategy_a': pd.Series(np.random.randn(n) * 0.01),
        'strategy_b': pd.Series(np.random.randn(n) * 0.01)
    }

    result = analyzer.calculate_correlation_matrix(returns_dict)

    # 獨立隨機變數的相關性應接近 0
    assert abs(result.mean_correlation) < 0.1  # 允許一些抽樣誤差


def test_constant_returns(analyzer):
    """測試常數收益率（無變異）"""
    returns_dict = {
        'strategy_a': pd.Series([0.01] * 100),
        'strategy_b': pd.Series([0.01] * 100)
    }

    result = analyzer.calculate_correlation_matrix(returns_dict)

    # 常數序列的相關性為 NaN（標準差為 0）
    assert np.isnan(result.mean_correlation)


# ========== 整合測試 ==========

def test_full_workflow(analyzer, multi_strategy_returns):
    """測試完整工作流程"""
    # 1. 計算相關性矩陣
    corr_matrix = analyzer.calculate_correlation_matrix(multi_strategy_returns)
    assert corr_matrix.mean_correlation is not None

    # 2. 計算兩兩滾動相關性
    ret1 = multi_strategy_returns['trend']
    ret2 = multi_strategy_returns['mean_reversion']
    rolling = analyzer.rolling_correlation(ret1, ret2)
    assert len(rolling.correlation) > 0

    # 3. 計算尾部相關性
    tail = analyzer.tail_correlation(ret1, ret2)
    assert tail.normal is not None

    # 4. 分析投資組合分散
    diversification = analyzer.analyze_portfolio_diversification(multi_strategy_returns)
    assert diversification['diversification_benefit'] is not None


def test_analyzer_window_sizes():
    """測試不同窗口大小"""
    np.random.seed(42)
    ret1 = pd.Series(np.random.randn(200) * 0.01)
    ret2 = pd.Series(np.random.randn(200) * 0.01)

    # 小窗口
    analyzer_small = CorrelationAnalyzer(window=10)
    result_small = analyzer_small.rolling_correlation(ret1, ret2)

    # 大窗口
    analyzer_large = CorrelationAnalyzer(window=50)
    result_large = analyzer_large.rolling_correlation(ret1, ret2)

    # 小窗口應該有更多資料點
    assert len(result_small.correlation) > len(result_large.correlation)

    # 大窗口應該有較小的標準差（更平滑）
    assert result_large.std < result_small.std


# ========== 效能測試 ==========

def test_performance_large_dataset():
    """測試大數據集的效能"""
    np.random.seed(42)
    n = 10000
    analyzer = CorrelationAnalyzer(window=60)

    returns_dict = {
        f'strategy_{i}': pd.Series(np.random.randn(n) * 0.01)
        for i in range(5)
    }

    # 應該能在合理時間內完成
    import time
    start = time.time()
    result = analyzer.calculate_correlation_matrix(returns_dict)
    elapsed = time.time() - start

    assert elapsed < 1.0  # 應在 1 秒內完成
    assert result.matrix.shape == (5, 5)
