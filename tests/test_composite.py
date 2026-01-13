"""
測試組合策略（Composite Strategy）

測試項目：
1. 基本功能測試
2. 訊號聚合測試
3. 權重優化測試
4. 動態再平衡測試
5. 邊界條件測試
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from pandas import Series, DataFrame

from src.strategies.composite import (
    CompositeStrategy,
    SignalAggregation,
    RebalanceTrigger
)
from src.strategies.base import BaseStrategy


# ============================================================
# Mock Strategies for Testing
# ============================================================

class MockStrategy(BaseStrategy):
    """簡單的 Mock 策略用於測試"""

    name = "mock_strategy"
    strategy_type = "mock"
    version = "1.0"
    description = "Mock strategy for testing"

    def __init__(self, signal_pattern: str = "all_false", strategy_name: Optional[str] = None, **kwargs):
        """
        初始化 Mock 策略

        Args:
            signal_pattern: 訊號模式
                - 'all_true': 全部 True
                - 'all_false': 全部 False
                - 'alternating': 交替 True/False
                - 'first_half': 前半 True，後半 False
            strategy_name: 策略名稱（用於區分多個實例）
        """
        self.signal_pattern = signal_pattern
        if strategy_name:
            self.name = strategy_name
        super().__init__(**kwargs)

    def calculate_indicators(self, data: DataFrame) -> Dict[str, Series]:
        """回傳假指標"""
        return {"mock_indicator": pd.Series(0.0, index=data.index)}

    def generate_signals(self, data: DataFrame) -> Tuple[Series, Series, Series, Series]:
        """根據 signal_pattern 產生訊號"""
        n = len(data)

        if self.signal_pattern == "all_true":
            pattern = pd.Series(True, index=data.index)
        elif self.signal_pattern == "all_false":
            pattern = pd.Series(False, index=data.index)
        elif self.signal_pattern == "alternating":
            pattern = pd.Series([i % 2 == 0 for i in range(n)], index=data.index)
        elif self.signal_pattern == "first_half":
            pattern = pd.Series([i < n // 2 for i in range(n)], index=data.index)
        else:
            pattern = pd.Series(False, index=data.index)

        # 回傳四個相同的訊號（簡化）
        return pattern, pattern, pattern, pattern


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_data():
    """產生測試用 OHLCV 資料"""
    n = 100
    dates = pd.date_range('2024-01-01', periods=n, freq='1h')
    data = pd.DataFrame({
        'open': np.random.uniform(100, 110, n),
        'high': np.random.uniform(110, 120, n),
        'low': np.random.uniform(90, 100, n),
        'close': np.random.uniform(100, 110, n),
        'volume': np.random.uniform(1000, 2000, n)
    }, index=dates)
    return data


@pytest.fixture
def mock_strategies():
    """產生測試用 Mock 策略"""
    s1 = MockStrategy(signal_pattern="all_true", strategy_name="strategy_1")
    s2 = MockStrategy(signal_pattern="all_false", strategy_name="strategy_2")
    s3 = MockStrategy(signal_pattern="alternating", strategy_name="strategy_3")
    return [s1, s2, s3]


@pytest.fixture
def strategy_returns():
    """產生測試用策略回報資料"""
    n = 100
    dates = pd.date_range('2024-01-01', periods=n, freq='1D')
    returns = pd.DataFrame({
        'strategy_1': np.random.normal(0.001, 0.02, n),
        'strategy_2': np.random.normal(0.0005, 0.015, n),
        'strategy_3': np.random.normal(0.0008, 0.018, n)
    }, index=dates)
    return returns


# ============================================================
# 1. 基本功能測試
# ============================================================

def test_add_strategy(mock_strategies):
    """測試添加策略"""
    # 初始化時需要至少一個策略（避免 validate_params 失敗）
    composite = CompositeStrategy(strategies=[mock_strategies[0]])

    # 驗證初始狀態
    assert len(composite.strategies) == 1
    assert "strategy_1" in composite.strategies

    # 添加第二個策略
    composite.add_strategy(mock_strategies[1])
    assert len(composite.strategies) == 2
    assert "strategy_2" in composite.strategies


def test_add_strategy_type_check():
    """測試類型檢查：非 BaseStrategy 應拋出 TypeError"""
    # 需要先初始化一個有效的策略
    s1 = MockStrategy(strategy_name="s1")
    composite = CompositeStrategy(strategies=[s1])

    with pytest.raises(TypeError, match="strategy 必須是 BaseStrategy 實例"):
        composite.add_strategy("not_a_strategy")


def test_remove_strategy(mock_strategies):
    """測試移除策略"""
    composite = CompositeStrategy(strategies=mock_strategies[:2])

    # 移除策略
    composite.remove_strategy("strategy_1")
    assert len(composite.strategies) == 1
    assert "strategy_1" not in composite.strategies
    assert "strategy_2" in composite.strategies

    # 權重應該重新正規化
    assert abs(sum(composite.weights.values()) - 1.0) < 1e-6


def test_equal_weights(mock_strategies):
    """測試等權重計算"""
    composite = CompositeStrategy(strategies=mock_strategies)

    # 所有權重應相等
    expected_weight = 1.0 / len(mock_strategies)
    for weight in composite.weights.values():
        assert abs(weight - expected_weight) < 1e-6

    # 總和應為 1
    assert abs(sum(composite.weights.values()) - 1.0) < 1e-6


def test_weight_normalization():
    """測試權重正規化"""
    s1 = MockStrategy(strategy_name="s1")
    s2 = MockStrategy(strategy_name="s2")

    # 傳入未正規化的權重
    composite = CompositeStrategy(
        strategies=[s1, s2],
        weights={'s1': 3.0, 's2': 1.0}
    )

    # 應該自動正規化
    assert abs(composite.weights['s1'] - 0.75) < 1e-6
    assert abs(composite.weights['s2'] - 0.25) < 1e-6
    assert abs(sum(composite.weights.values()) - 1.0) < 1e-6


# ============================================================
# 2. 訊號聚合測試
# ============================================================

def test_aggregate_weighted(sample_data):
    """測試加權平均聚合"""
    # 建立策略：2 個 all_true, 1 個 all_false
    s1 = MockStrategy(signal_pattern="all_true", strategy_name="s1")
    s2 = MockStrategy(signal_pattern="all_true", strategy_name="s2")
    s3 = MockStrategy(signal_pattern="all_false", strategy_name="s3")

    composite = CompositeStrategy(
        strategies=[s1, s2, s3],
        aggregation=SignalAggregation.WEIGHTED,
        weighted_threshold=0.5  # 超過 50% 為 True
    )

    # 等權重：(1 + 1 + 0) / 3 = 0.667 > 0.5 → True
    le, lx, se, sx = composite.generate_signals(sample_data)

    assert le.all()  # 應該全 True
    assert lx.all()
    assert se.all()
    assert sx.all()


def test_aggregate_voting(sample_data):
    """測試多數決聚合"""
    # 建立策略：2 個 all_true, 1 個 all_false
    s1 = MockStrategy(signal_pattern="all_true", strategy_name="s1")
    s2 = MockStrategy(signal_pattern="all_true", strategy_name="s2")
    s3 = MockStrategy(signal_pattern="all_false", strategy_name="s3")

    composite = CompositeStrategy(
        strategies=[s1, s2, s3],
        aggregation=SignalAggregation.VOTING
    )

    # 多數決：2/3 > 50% → True
    le, lx, se, sx = composite.generate_signals(sample_data)

    assert le.all()  # 應該全 True
    assert lx.all()


def test_aggregate_ranked(sample_data):
    """測試排名選擇聚合"""
    # 建立策略：權重不同
    s1 = MockStrategy(signal_pattern="all_true", strategy_name="s1")
    s2 = MockStrategy(signal_pattern="all_false", strategy_name="s2")
    s3 = MockStrategy(signal_pattern="all_false", strategy_name="s3")

    composite = CompositeStrategy(
        strategies=[s1, s2, s3],
        weights={'s1': 0.6, 's2': 0.3, 's3': 0.1},  # s1 權重最高
        aggregation=SignalAggregation.RANKED,
        ranked_top_n=1  # 只聽權重最高的策略
    )

    # 只聽 s1（all_true）
    le, lx, se, sx = composite.generate_signals(sample_data)

    assert le.all()  # 應該全 True
    assert lx.all()


def test_aggregate_unanimous(sample_data):
    """測試全體一致聚合"""
    # 測試 1：全部 all_true → 應該 True
    s1 = MockStrategy(signal_pattern="all_true", strategy_name="s1")
    s2 = MockStrategy(signal_pattern="all_true", strategy_name="s2")

    composite = CompositeStrategy(
        strategies=[s1, s2],
        aggregation=SignalAggregation.UNANIMOUS
    )

    le, lx, se, sx = composite.generate_signals(sample_data)
    assert le.all()  # 全體一致 → True

    # 測試 2：有一個 all_false → 應該 False
    s3 = MockStrategy(signal_pattern="all_false", strategy_name="s3")
    composite.add_strategy(s3)

    le, lx, se, sx = composite.generate_signals(sample_data)
    assert not le.any()  # 不一致 → False


def test_empty_strategies(sample_data):
    """測試空策略列表情況（直接測試 _create_empty_signals）"""
    # 由於 CompositeStrategy 不允許空策略初始化，
    # 我們改為測試 generate_signals 在移除所有策略後的行為
    s1 = MockStrategy(strategy_name="s1")
    composite = CompositeStrategy(strategies=[s1])

    # 移除所有策略（此時 validate_params 會失敗，但仍可調用 generate_signals）
    composite.remove_strategy("s1")

    # generate_signals 應該檢測到空策略並回傳全 False
    le, lx, se, sx = composite.generate_signals(sample_data)

    # 應該回傳全 False
    assert not le.any()
    assert not lx.any()
    assert not se.any()
    assert not sx.any()


# ============================================================
# 3. 權重優化測試
# ============================================================

def test_optimize_weights_max_sharpe(mock_strategies, strategy_returns):
    """測試最大化 Sharpe 優化"""
    composite = CompositeStrategy(strategies=mock_strategies)

    # 執行優化
    result = composite.optimize_weights(strategy_returns, method='max_sharpe')

    # 檢查結果
    assert result.optimization_success or result.optimization_message  # 可能成功或失敗
    assert abs(sum(composite.weights.values()) - 1.0) < 1e-6  # 權重總和為 1
    assert all(w >= 0 for w in composite.weights.values())  # 權重非負


def test_optimize_weights_risk_parity(mock_strategies, strategy_returns):
    """測試風險平價優化"""
    composite = CompositeStrategy(strategies=mock_strategies)

    # 執行優化
    result = composite.optimize_weights(strategy_returns, method='risk_parity')

    # 檢查結果
    assert abs(sum(composite.weights.values()) - 1.0) < 1e-6
    assert all(w >= 0 for w in composite.weights.values())


def test_optimize_weights_failure_fallback(mock_strategies):
    """測試優化失敗降級到等權重"""
    composite = CompositeStrategy(strategies=mock_strategies)

    # 使用無效的 returns（全 NaN）
    n = 10
    invalid_returns = pd.DataFrame({
        'strategy_1': [np.nan] * n,
        'strategy_2': [np.nan] * n,
        'strategy_3': [np.nan] * n
    })

    # 執行優化（應該失敗並降級）
    result = composite.optimize_weights(invalid_returns, method='max_sharpe')

    # 應該降級到等權重
    assert not result.optimization_success
    expected_weight = 1.0 / len(mock_strategies)
    for weight in composite.weights.values():
        assert abs(weight - expected_weight) < 1e-6


# ============================================================
# 4. 動態再平衡測試
# ============================================================

def test_rebalance_periodic(mock_strategies, strategy_returns):
    """測試固定週期再平衡"""
    composite = CompositeStrategy(
        strategies=mock_strategies,
        rebalance_trigger=RebalanceTrigger.PERIODIC,
        rebalance_period=5
    )

    # 前 4 次不應該觸發
    for _ in range(4):
        assert not composite.rebalance(returns=strategy_returns)

    # 第 5 次應該觸發
    assert composite.rebalance(returns=strategy_returns)


def test_rebalance_drift(mock_strategies, strategy_returns):
    """測試權重漂移觸發"""
    composite = CompositeStrategy(
        strategies=mock_strategies,
        weights={'strategy_1': 0.5, 'strategy_2': 0.3, 'strategy_3': 0.2},
        rebalance_trigger=RebalanceTrigger.DRIFT,
        drift_threshold=0.1
    )

    # 第一次呼叫會設定 _initial_weights
    composite.rebalance(returns=strategy_returns)

    # 模擬權重漂移
    composite.weights['strategy_1'] = 0.7  # 偏離 0.2 > threshold (0.7 - 0.5 = 0.2)
    composite.weights['strategy_2'] = 0.2
    composite.weights['strategy_3'] = 0.1

    # 應該觸發再平衡
    assert composite.rebalance(returns=strategy_returns)


def test_rebalance_missing_returns(mock_strategies):
    """測試缺少 returns 資料"""
    composite = CompositeStrategy(
        strategies=mock_strategies,
        rebalance_trigger=RebalanceTrigger.PERIODIC,
        rebalance_period=1
    )

    # 缺少 returns 應該回傳 False（並記錄錯誤）
    assert not composite.rebalance(returns=None)


def test_trigger_regime_rebalance(mock_strategies, strategy_returns):
    """測試手動觸發 REGIME_CHANGE"""
    composite = CompositeStrategy(
        strategies=mock_strategies,
        rebalance_trigger=RebalanceTrigger.REGIME_CHANGE
    )

    # 手動觸發
    assert composite.trigger_regime_rebalance(strategy_returns)


# ============================================================
# 5. 邊界條件測試
# ============================================================

def test_ranked_top_n_exceeds_strategies(sample_data):
    """測試 ranked_top_n 超過策略數"""
    s1 = MockStrategy(signal_pattern="all_true", strategy_name="s1")
    s2 = MockStrategy(signal_pattern="all_false", strategy_name="s2")

    composite = CompositeStrategy(
        strategies=[s1, s2],
        aggregation=SignalAggregation.RANKED,
        ranked_top_n=10  # 超過實際策略數（2）
    )

    # 應該自動限制為 2
    le, lx, se, sx = composite.generate_signals(sample_data)

    # 兩個策略 OR 運算：True OR False = True
    assert le.all()


def test_voting_single_strategy(sample_data):
    """測試單策略投票"""
    s1 = MockStrategy(signal_pattern="all_true", strategy_name="s1")

    composite = CompositeStrategy(
        strategies=[s1],
        aggregation=SignalAggregation.VOTING
    )

    # 單策略投票：1 > 0.5 → True
    le, lx, se, sx = composite.generate_signals(sample_data)
    assert le.all()


def test_validate_params(mock_strategies):
    """測試參數驗證"""
    # 測試 1：正常情況
    composite = CompositeStrategy(strategies=mock_strategies)
    assert composite.validate_params()

    # 測試 2：權重不匹配
    composite.weights['strategy_1'] = 0.5
    composite.weights['strategy_2'] = 0.3
    # strategy_3 缺少權重（已被刪除）
    del composite.weights['strategy_3']
    assert not composite.validate_params()


# ============================================================
# 6. 額外測試：calculate_indicators
# ============================================================

def test_calculate_indicators(sample_data):
    """測試 calculate_indicators 彙整"""
    s1 = MockStrategy(strategy_name="s1")
    s2 = MockStrategy(strategy_name="s2")

    composite = CompositeStrategy(strategies=[s1, s2])

    indicators = composite.calculate_indicators(sample_data)

    # 應該有兩個指標（帶前綴）
    assert 's1.mock_indicator' in indicators
    assert 's2.mock_indicator' in indicators
    assert len(indicators) == 2


# ============================================================
# 7. 邊界測試：alternating 模式
# ============================================================

def test_alternating_pattern_voting(sample_data):
    """測試 alternating 模式的投票聚合"""
    # 3 個策略：2 個 alternating，1 個 all_false
    s1 = MockStrategy(signal_pattern="alternating", strategy_name="s1")
    s2 = MockStrategy(signal_pattern="alternating", strategy_name="s2")
    s3 = MockStrategy(signal_pattern="all_false", strategy_name="s3")

    composite = CompositeStrategy(
        strategies=[s1, s2, s3],
        aggregation=SignalAggregation.VOTING
    )

    le, lx, se, sx = composite.generate_signals(sample_data)

    # 在偶數索引處：2 個 True，1 個 False → 多數決 True
    # 在奇數索引處：2 個 False，1 個 False → 多數決 False
    expected = pd.Series([i % 2 == 0 for i in range(len(sample_data))], index=sample_data.index)
    pd.testing.assert_series_equal(le, expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
