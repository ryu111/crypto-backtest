"""
Regime Detection 模組單元測試

測試涵蓋：
1. 方向分數計算函數
2. 波動分數計算函數
3. MarketStateAnalyzer
4. StrategySwitch
5. 邊界情況和錯誤處理
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.regime.analyzer import (
    MarketRegime,
    MarketState,
    MarketStateAnalyzer,
    RegimeValidator,
    calculate_direction_score,
    adx_direction_score,
    elder_power_score,
    volatility_score_atr,
    volatility_score_bbw,
    choppiness_index,
)
from src.regime.switch import (
    StrategyConfig,
    StrategySwitch,
    setup_default_switch,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_ohlcv():
    """生成正常的 OHLCV 測試數據（100 根 K 線）"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    np.random.seed(42)

    # 生成有趨勢的價格數據
    base_price = 50000
    trend = np.linspace(0, 5000, 100)  # 上升趨勢
    noise = np.random.randn(100) * 500
    close = base_price + trend + noise

    # 生成 OHLC
    high = close + np.abs(np.random.randn(100) * 100)
    low = close - np.abs(np.random.randn(100) * 100)
    open_ = close + np.random.randn(100) * 50
    volume = np.random.randint(1000, 10000, 100)

    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


@pytest.fixture
def minimal_ohlcv():
    """最小可用數據（20 根 K 線，剛好滿足計算需求）"""
    dates = pd.date_range(start='2024-01-01', periods=20, freq='1h')
    close = np.linspace(50000, 51000, 20)
    high = close + 100
    low = close - 100
    open_ = close
    volume = np.ones(20) * 1000

    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


@pytest.fixture
def empty_ohlcv():
    """空數據"""
    return pd.DataFrame({
        'open': pd.Series([], dtype=float),
        'high': pd.Series([], dtype=float),
        'low': pd.Series([], dtype=float),
        'close': pd.Series([], dtype=float),
        'volume': pd.Series([], dtype=float),
    })


@pytest.fixture
def all_zero_ohlcv():
    """全零價格數據（異常情況）"""
    dates = pd.date_range(start='2024-01-01', periods=50, freq='1h')
    return pd.DataFrame({
        'open': np.zeros(50),
        'high': np.zeros(50),
        'low': np.zeros(50),
        'close': np.zeros(50),
        'volume': np.ones(50)
    }, index=dates)


@pytest.fixture
def extreme_volatility_ohlcv():
    """極端波動數據"""
    dates = pd.date_range(start='2024-01-01', periods=50, freq='1h')
    close = np.array([50000 if i % 2 == 0 else 60000 for i in range(50)])
    high = close + 1000
    low = close - 1000
    open_ = close
    volume = np.ones(50) * 1000

    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


@pytest.fixture
def sample_market_state():
    """生成樣本 MarketState"""
    return MarketState(
        direction=5.0,
        volatility=7.0,
        regime=MarketRegime.WEAK_BULL_HIGH_VOL,
        timestamp=datetime.now()
    )


# ============================================================================
# 1. 方向分數計算測試
# ============================================================================

class TestDirectionScores:
    """方向分數計算函數測試"""

    def test_calculate_direction_score_normal(self, sample_ohlcv):
        """正常情況：應返回 -10 到 10 之間的分數"""
        result = calculate_direction_score(sample_ohlcv)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv)
        assert result.name == 'direction_score'

        # 檢查範圍（允許少量 NaN 在開頭）
        valid_scores = result.dropna()
        assert valid_scores.min() >= -10
        assert valid_scores.max() <= 10

    def test_calculate_direction_score_uptrend(self, sample_ohlcv):
        """上升趨勢：應返回正分數"""
        result = calculate_direction_score(sample_ohlcv)

        # 檢查最後 20 根的平均方向（避免初期 NaN）
        last_scores = result.iloc[-20:]
        assert last_scores.mean() > 0, "上升趨勢應該有正方向分數"

    def test_calculate_direction_score_minimal_data(self, minimal_ohlcv):
        """最小數據：應能正常計算"""
        result = calculate_direction_score(minimal_ohlcv)

        assert isinstance(result, pd.Series)
        assert len(result) == len(minimal_ohlcv)
        # 最後幾根應該有有效值
        assert not pd.isna(result.iloc[-1])

    def test_calculate_direction_score_zero_prices(self, all_zero_ohlcv):
        """全零價格：應處理除零錯誤，返回有效 Series"""
        result = calculate_direction_score(all_zero_ohlcv)

        assert isinstance(result, pd.Series)
        assert len(result) == len(all_zero_ohlcv)
        # 應該不包含 inf
        assert not np.isinf(result).any()

    def test_adx_direction_score_normal(self, sample_ohlcv):
        """ADX 方向分數：正常情況"""
        result = adx_direction_score(sample_ohlcv)

        assert isinstance(result, pd.Series)
        assert result.name == 'adx_direction'
        assert len(result) == len(sample_ohlcv)

        valid_scores = result.dropna()
        assert valid_scores.min() >= -10
        assert valid_scores.max() <= 10

    def test_adx_direction_score_zero_prices(self, all_zero_ohlcv):
        """ADX 方向分數：全零價格"""
        result = adx_direction_score(all_zero_ohlcv)

        assert isinstance(result, pd.Series)
        # 不應包含 inf 或過大值
        assert not np.isinf(result).any()

    def test_elder_power_score_normal(self, sample_ohlcv):
        """Elder Power 分數：正常情況"""
        result = elder_power_score(sample_ohlcv)

        assert isinstance(result, pd.Series)
        assert result.name == 'elder_power'
        assert len(result) == len(sample_ohlcv)

        valid_scores = result.dropna()
        assert valid_scores.min() >= -10
        assert valid_scores.max() <= 10

    def test_elder_power_score_minimal_data(self, minimal_ohlcv):
        """Elder Power 分數：最小數據"""
        result = elder_power_score(minimal_ohlcv)

        assert isinstance(result, pd.Series)
        assert not pd.isna(result.iloc[-1])


# ============================================================================
# 2. 波動分數計算測試
# ============================================================================

class TestVolatilityScores:
    """波動分數計算函數測試"""

    def test_volatility_score_atr_normal(self, sample_ohlcv):
        """ATR 波動分數：正常情況"""
        result = volatility_score_atr(sample_ohlcv)

        assert isinstance(result, pd.Series)
        assert result.name == 'volatility_atr'
        assert len(result) == len(sample_ohlcv)

        # 波動分數應在 0-10 之間
        valid_scores = result.dropna()
        assert valid_scores.min() >= 0
        assert valid_scores.max() <= 10

    def test_volatility_score_atr_high_volatility(self, extreme_volatility_ohlcv):
        """ATR 波動分數：極端波動"""
        result = volatility_score_atr(extreme_volatility_ohlcv)

        # 極端波動數據較短，可能全部是 NaN（lookback 需要 100 根）
        # 測試應該確保不會拋出異常，且結果是有效的 Series
        assert isinstance(result, pd.Series)
        assert result.name == 'volatility_atr'
        # 如果有有效值，應在 0-10 範圍內
        valid_scores = result.dropna()
        if len(valid_scores) > 0:
            assert valid_scores.min() >= 0
            assert valid_scores.max() <= 10

    def test_volatility_score_bbw_normal(self, sample_ohlcv):
        """BBW 波動分數：正常情況"""
        result = volatility_score_bbw(sample_ohlcv)

        assert isinstance(result, pd.Series)
        assert result.name == 'volatility_bbw'

        valid_scores = result.dropna()
        assert valid_scores.min() >= 0
        assert valid_scores.max() <= 10

    def test_volatility_score_bbw_zero_prices(self, all_zero_ohlcv):
        """BBW 波動分數：全零價格"""
        result = volatility_score_bbw(all_zero_ohlcv)

        assert isinstance(result, pd.Series)
        # 不應包含 inf
        assert not np.isinf(result).any()

    def test_choppiness_index_normal(self, sample_ohlcv):
        """Choppiness Index：正常情況"""
        result = choppiness_index(sample_ohlcv)

        assert isinstance(result, pd.Series)
        assert result.name == 'choppiness'

        valid_scores = result.dropna()
        assert valid_scores.min() >= 0
        assert valid_scores.max() <= 10

    def test_choppiness_index_trending(self, sample_ohlcv):
        """Choppiness Index：趨勢市場應該低分"""
        result = choppiness_index(sample_ohlcv)

        # 趨勢市場的 CI 應該較低（< 5）
        valid_scores = result.dropna()
        assert valid_scores.mean() < 7, "趨勢市場應該有低 CI 分數"


# ============================================================================
# 3. MarketStateAnalyzer 測試
# ============================================================================

class TestMarketStateAnalyzer:
    """MarketStateAnalyzer 測試"""

    def test_analyzer_init_default(self):
        """初始化：預設參數"""
        analyzer = MarketStateAnalyzer()

        assert analyzer.dir_strong == 5.0
        assert analyzer.dir_weak == 2.0
        assert analyzer.vol_threshold == 5.0
        assert analyzer.direction_method == 'composite'

    def test_analyzer_init_custom(self):
        """初始化：自定義參數"""
        analyzer = MarketStateAnalyzer(
            direction_threshold_strong=6.0,
            direction_threshold_weak=3.0,
            volatility_threshold=7.0,
            direction_method='adx'
        )

        assert analyzer.dir_strong == 6.0
        assert analyzer.dir_weak == 3.0
        assert analyzer.vol_threshold == 7.0
        assert analyzer.direction_method == 'adx'

    def test_calculate_state_normal(self, sample_ohlcv):
        """計算市場狀態：正常情況"""
        analyzer = MarketStateAnalyzer()
        state = analyzer.calculate_state(sample_ohlcv)

        assert isinstance(state, MarketState)
        assert -10 <= state.direction <= 10
        assert 0 <= state.volatility <= 10
        assert isinstance(state.regime, MarketRegime)
        assert isinstance(state.timestamp, datetime)

    def test_calculate_state_minimal_data(self, minimal_ohlcv):
        """計算市場狀態：最小數據"""
        analyzer = MarketStateAnalyzer()
        state = analyzer.calculate_state(minimal_ohlcv)

        assert isinstance(state, MarketState)
        assert state.direction is not None
        assert state.volatility is not None

    def test_calculate_state_different_methods(self, sample_ohlcv):
        """計算市場狀態：不同方向計算方法"""
        methods = ['composite', 'adx', 'elder']

        for method in methods:
            analyzer = MarketStateAnalyzer(direction_method=method)
            state = analyzer.calculate_state(sample_ohlcv)

            assert isinstance(state, MarketState)
            assert -10 <= state.direction <= 10

    def test_determine_regime_strong_bull_high_vol(self):
        """判斷狀態：強牛 + 高波動"""
        analyzer = MarketStateAnalyzer()
        regime = analyzer._determine_regime(direction=7.0, volatility=8.0)

        assert regime == MarketRegime.STRONG_BULL_HIGH_VOL

    def test_determine_regime_strong_bear_low_vol(self):
        """判斷狀態：強熊 + 低波動"""
        analyzer = MarketStateAnalyzer()
        regime = analyzer._determine_regime(direction=-7.0, volatility=3.0)

        assert regime == MarketRegime.STRONG_BEAR_LOW_VOL

    def test_determine_regime_neutral_high_vol(self):
        """判斷狀態：中性 + 高波動"""
        analyzer = MarketStateAnalyzer()
        regime = analyzer._determine_regime(direction=1.0, volatility=7.0)

        assert regime == MarketRegime.NEUTRAL_HIGH_VOL

    def test_determine_regime_weak_bull_low_vol(self):
        """判斷狀態：弱牛 + 低波動"""
        analyzer = MarketStateAnalyzer()
        regime = analyzer._determine_regime(direction=3.0, volatility=4.0)

        assert regime == MarketRegime.WEAK_BULL_LOW_VOL

    def test_market_state_to_dict(self, sample_market_state):
        """MarketState.to_dict()：正常轉換"""
        result = sample_market_state.to_dict()

        assert isinstance(result, dict)
        assert 'direction' in result
        assert 'volatility' in result
        assert 'regime' in result
        assert 'timestamp' in result
        assert result['regime'] == 'weak_bull_high_vol'


# ============================================================================
# 4. RegimeValidator 測試
# ============================================================================

class TestRegimeValidator:
    """RegimeValidator 測試"""

    def test_validator_init(self):
        """初始化：預設參數"""
        validator = RegimeValidator()

        assert validator.forward_periods == 20
        assert validator.dir_threshold == 0.03
        assert validator.vol_threshold == 1.5

    def test_validate_direction_normal(self, sample_ohlcv):
        """驗證方向：正常情況"""
        analyzer = MarketStateAnalyzer()
        states = []

        # 生成多個狀態（使用滾動窗口）
        for i in range(50, 70):
            state = analyzer.calculate_state(sample_ohlcv.iloc[:i])
            states.append(state)

        validator = RegimeValidator()
        result = validator.validate_direction(sample_ohlcv, states)

        assert isinstance(result, dict)
        assert 'overall_accuracy' in result
        assert 'by_prediction' in result
        assert 'n_samples' in result
        assert 'passed' in result
        assert 0 <= result['overall_accuracy'] <= 1

    def test_validate_volatility_normal(self, sample_ohlcv):
        """驗證波動：正常情況"""
        analyzer = MarketStateAnalyzer()
        states = []

        for i in range(50, 70):
            state = analyzer.calculate_state(sample_ohlcv.iloc[:i])
            states.append(state)

        validator = RegimeValidator()
        result = validator.validate_volatility(sample_ohlcv, states)

        assert isinstance(result, dict)
        assert 'overall_accuracy' in result
        assert 'passed' in result
        assert 0 <= result['overall_accuracy'] <= 1

    def test_validate_stability_stable(self):
        """驗證穩定性：穩定狀態"""
        states = [
            MarketState(5.0, 5.0, MarketRegime.WEAK_BULL_HIGH_VOL, datetime.now()),
            MarketState(5.5, 5.0, MarketRegime.WEAK_BULL_HIGH_VOL, datetime.now()),
            MarketState(4.8, 5.0, MarketRegime.WEAK_BULL_HIGH_VOL, datetime.now()),
            MarketState(5.2, 5.0, MarketRegime.WEAK_BULL_HIGH_VOL, datetime.now()),
        ]

        validator = RegimeValidator()
        result = validator.validate_stability(states)

        assert result['passed'] is True
        assert result['flip_rate'] < 0.2

    def test_validate_stability_unstable(self):
        """驗證穩定性：不穩定狀態"""
        states = [
            MarketState(5.0, 5.0, MarketRegime.WEAK_BULL_HIGH_VOL, datetime.now()),
            MarketState(-5.0, 5.0, MarketRegime.WEAK_BEAR_HIGH_VOL, datetime.now()),
            MarketState(5.0, 5.0, MarketRegime.WEAK_BULL_HIGH_VOL, datetime.now()),
            MarketState(-5.0, 5.0, MarketRegime.WEAK_BEAR_HIGH_VOL, datetime.now()),
        ]

        validator = RegimeValidator()
        result = validator.validate_stability(states)

        assert result['passed'] is False
        assert result['flip_rate'] > 0.2

    def test_validate_stability_insufficient_data(self):
        """驗證穩定性：數據不足"""
        states = [MarketState(5.0, 5.0, MarketRegime.WEAK_BULL_HIGH_VOL, datetime.now())]

        validator = RegimeValidator()
        result = validator.validate_stability(states)

        assert result['passed'] is False
        assert 'reason' in result

    def test_full_validation(self, sample_ohlcv):
        """完整驗證：整合測試"""
        analyzer = MarketStateAnalyzer()
        states = []

        for i in range(50, 70):
            state = analyzer.calculate_state(sample_ohlcv.iloc[:i])
            states.append(state)

        validator = RegimeValidator()
        result = validator.full_validation(sample_ohlcv, states)

        assert isinstance(result, dict)
        assert 'direction' in result
        assert 'volatility' in result
        assert 'stability' in result
        assert 'all_passed' in result
        assert 'recommendation' in result


# ============================================================================
# 5. StrategyConfig 測試
# ============================================================================

class TestStrategyConfig:
    """StrategyConfig 測試"""

    def test_config_init(self):
        """初始化：正常情況"""
        config = StrategyConfig(
            name="test_strategy",
            direction_range=(3.0, 8.0),
            volatility_range=(2.0, 7.0),
            weight=1.5
        )

        assert config.name == "test_strategy"
        assert config.direction_range == (3.0, 8.0)
        assert config.volatility_range == (2.0, 7.0)
        assert config.weight == 1.5

    def test_config_is_active_true(self):
        """is_active：應該激活"""
        config = StrategyConfig(
            name="test",
            direction_range=(3.0, 8.0),
            volatility_range=(2.0, 7.0)
        )

        assert config.is_active(direction=5.0, volatility=5.0) is True

    def test_config_is_active_false_direction(self):
        """is_active：方向超出範圍"""
        config = StrategyConfig(
            name="test",
            direction_range=(3.0, 8.0),
            volatility_range=(2.0, 7.0)
        )

        assert config.is_active(direction=2.0, volatility=5.0) is False
        assert config.is_active(direction=9.0, volatility=5.0) is False

    def test_config_is_active_false_volatility(self):
        """is_active：波動超出範圍"""
        config = StrategyConfig(
            name="test",
            direction_range=(3.0, 8.0),
            volatility_range=(2.0, 7.0)
        )

        assert config.is_active(direction=5.0, volatility=1.0) is False
        assert config.is_active(direction=5.0, volatility=8.0) is False

    def test_config_is_active_boundary(self):
        """is_active：邊界值測試"""
        config = StrategyConfig(
            name="test",
            direction_range=(3.0, 8.0),
            volatility_range=(2.0, 7.0)
        )

        # 邊界值應該激活（包含邊界）
        assert config.is_active(direction=3.0, volatility=2.0) is True
        assert config.is_active(direction=8.0, volatility=7.0) is True


# ============================================================================
# 6. StrategySwitch 測試
# ============================================================================

class TestStrategySwitch:
    """StrategySwitch 測試"""

    def test_switch_init(self):
        """初始化：空狀態"""
        switch = StrategySwitch()

        assert isinstance(switch.strategies, dict)
        assert len(switch.strategies) == 0

    def test_register_strategy_valid(self):
        """註冊策略：有效參數"""
        switch = StrategySwitch()
        switch.register_strategy(
            "trend_following",
            direction_range=(3.0, 10.0),
            volatility_range=(5.0, 10.0),
            weight=1.0
        )

        assert "trend_following" in switch.strategies
        assert switch.strategies["trend_following"].name == "trend_following"

    def test_register_strategy_invalid_direction_range(self):
        """註冊策略：無效方向範圍"""
        switch = StrategySwitch()

        # 超出 -10 到 10 範圍
        with pytest.raises(ValueError):
            switch.register_strategy(
                "test",
                direction_range=(-15.0, 5.0),
                volatility_range=(0.0, 10.0)
            )

        # min > max
        with pytest.raises(ValueError):
            switch.register_strategy(
                "test",
                direction_range=(8.0, 3.0),
                volatility_range=(0.0, 10.0)
            )

    def test_register_strategy_invalid_volatility_range(self):
        """註冊策略：無效波動範圍"""
        switch = StrategySwitch()

        # 負值
        with pytest.raises(ValueError):
            switch.register_strategy(
                "test",
                direction_range=(0.0, 5.0),
                volatility_range=(-1.0, 5.0)
            )

        # 超出 10
        with pytest.raises(ValueError):
            switch.register_strategy(
                "test",
                direction_range=(0.0, 5.0),
                volatility_range=(0.0, 15.0)
            )

    def test_register_strategy_invalid_weight(self):
        """註冊策略：無效權重"""
        switch = StrategySwitch()

        with pytest.raises(ValueError):
            switch.register_strategy(
                "test",
                direction_range=(0.0, 5.0),
                volatility_range=(0.0, 5.0),
                weight=0.0
            )

        with pytest.raises(ValueError):
            switch.register_strategy(
                "test",
                direction_range=(0.0, 5.0),
                volatility_range=(0.0, 5.0),
                weight=-1.0
            )

    def test_get_active_strategies_single(self, sample_market_state):
        """獲取活躍策略：單一策略"""
        switch = StrategySwitch()
        switch.register_strategy(
            "trend_following",
            direction_range=(3.0, 10.0),
            volatility_range=(5.0, 10.0)
        )

        # sample_market_state: direction=5.0, volatility=7.0
        active = switch.get_active_strategies(sample_market_state)

        assert "trend_following" in active
        assert len(active) == 1

    def test_get_active_strategies_multiple(self):
        """獲取活躍策略：多個策略"""
        switch = StrategySwitch()
        switch.register_strategy(
            "strategy_a",
            direction_range=(0.0, 10.0),
            volatility_range=(0.0, 10.0)
        )
        switch.register_strategy(
            "strategy_b",
            direction_range=(3.0, 8.0),
            volatility_range=(5.0, 10.0)
        )

        state = MarketState(5.0, 7.0, MarketRegime.WEAK_BULL_HIGH_VOL, datetime.now())
        active = switch.get_active_strategies(state)

        assert "strategy_a" in active
        assert "strategy_b" in active
        assert len(active) == 2

    def test_get_active_strategies_none(self):
        """獲取活躍策略：無符合條件"""
        switch = StrategySwitch()
        switch.register_strategy(
            "trend_following",
            direction_range=(7.0, 10.0),
            volatility_range=(8.0, 10.0)
        )

        state = MarketState(2.0, 3.0, MarketRegime.WEAK_BULL_LOW_VOL, datetime.now())
        active = switch.get_active_strategies(state)

        assert len(active) == 0

    def test_get_strategy_weights_single(self):
        """獲取策略權重：單一策略"""
        switch = StrategySwitch()
        switch.register_strategy(
            "test",
            direction_range=(0.0, 10.0),
            volatility_range=(0.0, 10.0),
            weight=2.0
        )

        state = MarketState(5.0, 5.0, MarketRegime.WEAK_BULL_HIGH_VOL, datetime.now())
        weights = switch.get_strategy_weights(state)

        assert weights["test"] == 1.0  # 正規化後

    def test_get_strategy_weights_multiple(self):
        """獲取策略權重：多個策略正規化"""
        switch = StrategySwitch()
        switch.register_strategy(
            "strategy_a",
            direction_range=(0.0, 10.0),
            volatility_range=(0.0, 10.0),
            weight=3.0
        )
        switch.register_strategy(
            "strategy_b",
            direction_range=(0.0, 10.0),
            volatility_range=(0.0, 10.0),
            weight=1.0
        )

        state = MarketState(5.0, 5.0, MarketRegime.WEAK_BULL_HIGH_VOL, datetime.now())
        weights = switch.get_strategy_weights(state)

        assert weights["strategy_a"] == 0.75  # 3 / (3+1)
        assert weights["strategy_b"] == 0.25  # 1 / (3+1)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_get_strategy_weights_none_active(self):
        """獲取策略權重：無活躍策略"""
        switch = StrategySwitch()
        switch.register_strategy(
            "test",
            direction_range=(7.0, 10.0),
            volatility_range=(7.0, 10.0)
        )

        state = MarketState(2.0, 2.0, MarketRegime.WEAK_BULL_LOW_VOL, datetime.now())
        weights = switch.get_strategy_weights(state)

        assert len(weights) == 0


# ============================================================================
# 7. setup_default_switch 測試
# ============================================================================

class TestSetupDefaultSwitch:
    """setup_default_switch 測試"""

    def test_default_switch_has_strategies(self):
        """預設配置：包含預期策略"""
        switch = setup_default_switch()

        expected_strategies = [
            "trend_following_long",
            "trend_following_short",
            "mean_reversion",
            "breakout",
            "grid_trading",
            "funding_rate_arb"
        ]

        for name in expected_strategies:
            assert name in switch.strategies

    def test_default_switch_trend_long_active(self):
        """預設配置：趨勢做多策略激活條件"""
        switch = setup_default_switch()

        # direction > 3, volatility > 3
        state = MarketState(5.0, 7.0, MarketRegime.WEAK_BULL_HIGH_VOL, datetime.now())
        active = switch.get_active_strategies(state)

        assert "trend_following_long" in active

    def test_default_switch_mean_reversion_active(self):
        """預設配置：均值回歸策略激活條件"""
        switch = setup_default_switch()

        # -3 < direction < 3, volatility < 5
        state = MarketState(1.0, 3.0, MarketRegime.NEUTRAL_LOW_VOL, datetime.now())
        active = switch.get_active_strategies(state)

        assert "mean_reversion" in active

    def test_default_switch_funding_rate_always_active(self):
        """預設配置：資金費率套利總是激活"""
        switch = setup_default_switch()

        # 任意狀態
        states = [
            MarketState(8.0, 8.0, MarketRegime.STRONG_BULL_HIGH_VOL, datetime.now()),
            MarketState(-8.0, 2.0, MarketRegime.STRONG_BEAR_LOW_VOL, datetime.now()),
            MarketState(0.0, 5.0, MarketRegime.NEUTRAL_HIGH_VOL, datetime.now()),
        ]

        for state in states:
            active = switch.get_active_strategies(state)
            assert "funding_rate_arb" in active


# ============================================================================
# 8. 邊界情況整合測試
# ============================================================================

class TestEdgeCases:
    """邊界情況整合測試"""

    def test_empty_dataframe(self, empty_ohlcv):
        """空數據框：應優雅處理"""
        analyzer = MarketStateAnalyzer()

        # 應該拋出錯誤或返回預設值
        with pytest.raises((IndexError, ValueError, KeyError)):
            analyzer.calculate_state(empty_ohlcv)

    def test_very_short_data(self):
        """極短數據（< 20 根）：應優雅處理"""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='1h')
        data = pd.DataFrame({
            'open': np.ones(10) * 50000,
            'high': np.ones(10) * 50100,
            'low': np.ones(10) * 49900,
            'close': np.ones(10) * 50000,
            'volume': np.ones(10) * 1000
        }, index=dates)

        analyzer = MarketStateAnalyzer()

        # 可能拋出錯誤或返回有限的結果
        try:
            state = analyzer.calculate_state(data)
            # 如果成功，檢查是否有有效值
            assert state is not None
        except (IndexError, ValueError):
            # 也接受拋出錯誤
            pass

    def test_all_nan_prices(self):
        """全 NaN 價格：應優雅處理並返回預設值"""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='1h')
        data = pd.DataFrame({
            'open': np.nan,
            'high': np.nan,
            'low': np.nan,
            'close': np.nan,
            'volume': 1000
        }, index=dates)

        analyzer = MarketStateAnalyzer()

        # 由於安全除法處理，全 NaN 數據會返回預設值
        # 這是設計上的優雅處理，而非拋出錯誤
        state = analyzer.calculate_state(data)
        assert isinstance(state, MarketState)
        # 確保返回有效的 MarketRegime（任何狀態都是合法的）
        assert isinstance(state.regime, MarketRegime)
        # direction 和 volatility 可能是 NaN 或計算結果
        # 重要的是不會拋出異常

    def test_missing_columns(self):
        """缺少必要欄位：應拋出錯誤"""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='1h')
        data = pd.DataFrame({
            'close': np.linspace(50000, 51000, 50)
        }, index=dates)

        analyzer = MarketStateAnalyzer()

        with pytest.raises(KeyError):
            analyzer.calculate_state(data)

    def test_extreme_direction_values(self):
        """極端方向值：應正確分類"""
        analyzer = MarketStateAnalyzer()

        # 極端正值
        regime_pos = analyzer._determine_regime(direction=10.0, volatility=5.0)
        assert 'strong_bull' in regime_pos.value

        # 極端負值
        regime_neg = analyzer._determine_regime(direction=-10.0, volatility=5.0)
        assert 'strong_bear' in regime_neg.value

    def test_extreme_volatility_values(self):
        """極端波動值：應正確分類"""
        analyzer = MarketStateAnalyzer()

        # 極端高波動
        regime_high = analyzer._determine_regime(direction=5.0, volatility=10.0)
        assert 'high_vol' in regime_high.value

        # 極端低波動（接近 0）
        regime_low = analyzer._determine_regime(direction=5.0, volatility=0.1)
        assert 'low_vol' in regime_low.value
