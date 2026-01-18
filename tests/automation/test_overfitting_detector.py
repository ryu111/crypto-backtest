"""過擬合偵測器單元測試套件

測試覆蓋範圍：
1. OverfitMetrics - 過擬合指標與風險等級
2. OverfitDetector.calculate_pbo() - PBO 計算
3. OverfitDetector.check_trade_count() - 交易筆數檢查
4. OverfitMetrics.overall_risk - 綜合風險邏輯
5. 邊界情況測試 - 無效輸入、極端值
"""

import pytest
import numpy as np
from typing import Tuple

from src.automation.overfitting_detector import (
    OverfitMetrics,
    PBOResult,
    OverfitDetector,
)


# ========== Fixtures ==========

@pytest.fixture
def detector():
    """標準配置的 OverfitDetector"""
    return OverfitDetector(
        min_trades=30,
        max_pbo=0.5,
        max_is_oos_ratio=2.0,
        max_param_sensitivity=0.3,
    )


@pytest.fixture
def detector_strict():
    """嚴格配置的 OverfitDetector"""
    return OverfitDetector(
        min_trades=50,
        max_pbo=0.3,
        max_is_oos_ratio=1.5,
        max_param_sensitivity=0.2,
    )


@pytest.fixture
def sample_returns_matrix():
    """生成樣本報酬矩陣 (100 週期, 20 個試驗)"""
    np.random.seed(42)
    # 每列代表一個參數試驗，每列是該試驗的每日報酬
    returns = np.random.randn(100, 20) * 0.01  # 平均每日報酬±1%
    return returns


@pytest.fixture
def good_returns_matrix():
    """好的報酬矩陣（低過擬合）"""
    np.random.seed(42)
    # 所有試驗都有相似的穩定表現
    returns = np.random.randn(100, 10) * 0.01 + 0.001  # 平均每日+0.1%
    return returns


@pytest.fixture
def overfit_returns_matrix():
    """過擬合報酬矩陣（高過擬合）"""
    np.random.seed(42)
    # 部分試驗在 IS 期間表現很好，但 OOS 差
    returns = np.random.randn(100, 10) * 0.02
    # 在前 50 期（IS），給某些試驗加上額外利潤
    returns[:50, 0:3] += 0.02  # 這些試驗在 IS 很好，但 OOS 會表現差
    return returns


# ========== TestOverfitMetrics ==========

class TestOverfitMetrics:
    """OverfitMetrics 指標與風險邏輯測試"""

    def test_metrics_creation(self):
        """測試指標建立"""
        metrics = OverfitMetrics(
            pbo=0.3,
            is_oos_ratio=1.2,
            param_sensitivity=0.15,
            trade_count=100,
            degradation=0.1,
        )

        assert metrics.pbo == 0.3
        assert metrics.is_oos_ratio == 1.2
        assert metrics.param_sensitivity == 0.15
        assert metrics.trade_count == 100

    def test_overall_risk_low(self):
        """測試低風險判定"""
        metrics = OverfitMetrics(
            pbo=0.2,  # < 0.25 = 低
            is_oos_ratio=1.3,  # < 1.5 = 低
            param_sensitivity=0.10,  # < 0.2 = 低
            trade_count=100,  # > 50 = 低
            degradation=0.05,
        )

        assert metrics.overall_risk == "LOW"

    def test_overall_risk_medium_one_medium_risk(self):
        """測試中風險判定（1 個中風險不夠）"""
        # 只有 1 個中風險，應該是 LOW
        metrics = OverfitMetrics(
            pbo=0.2,  # 低
            is_oos_ratio=1.7,  # 1.5-2.0 = 中風險
            param_sensitivity=0.10,  # 低
            trade_count=100,
            degradation=0.05,
        )

        # 只有 1 個 MEDIUM，不足以達到 MEDIUM 級別
        assert metrics.overall_risk == "LOW"

    def test_overall_risk_medium_two_medium(self):
        """測試中風險判定（2 個中風險）"""
        metrics = OverfitMetrics(
            pbo=0.2,  # 低
            is_oos_ratio=1.7,  # 1.5-2.0 = 中風險
            param_sensitivity=0.25,  # 0.2-0.3 = 中風險
            trade_count=100,  # 低
            degradation=0.05,
        )

        # 2 個 MEDIUM 達到 MEDIUM 級別
        assert metrics.overall_risk == "MEDIUM"

    def test_overall_risk_medium_one_high_risk(self):
        """測試中風險判定（1 個高風險）"""
        metrics = OverfitMetrics(
            pbo=0.60,  # > 0.5 = 高風險
            is_oos_ratio=1.3,  # 低
            param_sensitivity=0.10,  # 低
            trade_count=100,
            degradation=0.2,
        )

        # 1 個 HIGH 達到 MEDIUM 級別
        assert metrics.overall_risk == "MEDIUM"

    def test_overall_risk_high_two_high(self):
        """測試高風險判定（2 個高風險）"""
        metrics = OverfitMetrics(
            pbo=0.60,  # > 0.5 = 高風險
            is_oos_ratio=2.5,  # > 2.0 = 高風險
            param_sensitivity=0.15,  # 低
            trade_count=100,
            degradation=0.2,
        )

        # 2 個 HIGH 達到 HIGH 級別
        assert metrics.overall_risk == "HIGH"

    def test_overall_risk_high_one_high_two_medium(self):
        """測試高風險判定（1 個高 + 其他）"""
        metrics = OverfitMetrics(
            pbo=0.60,  # > 0.5 = 高風險
            is_oos_ratio=1.7,  # 1.5-2.0 = 中風險
            param_sensitivity=0.25,  # 0.2-0.3 = 中風險
            trade_count=100,
            degradation=0.1,
        )

        # 1 個 HIGH + 2 個 MEDIUM，但邏輯是：1 個 HIGH 或 2 個 MEDIUM 就算 MEDIUM
        # 所以應該是 MEDIUM
        assert metrics.overall_risk == "MEDIUM"

    def test_recommendation_low(self):
        """測試推薦（低風險）"""
        metrics = OverfitMetrics(
            pbo=0.1,
            is_oos_ratio=1.2,
            param_sensitivity=0.10,
            trade_count=100,
        )

        assert "可靠" in metrics.recommendation
        assert "下一步驗證" in metrics.recommendation

    def test_recommendation_medium(self):
        """測試推薦（中風險）"""
        metrics = OverfitMetrics(
            pbo=0.3,
            is_oos_ratio=1.7,
            param_sensitivity=0.25,  # 兩個 MEDIUM 指標
            trade_count=100,
        )

        assert "謹慎" in metrics.recommendation

    def test_recommendation_high(self):
        """測試推薦（高風險）"""
        metrics = OverfitMetrics(
            pbo=0.7,
            is_oos_ratio=2.5,  # 兩個 HIGH 指標
            param_sensitivity=0.35,
            trade_count=20,
        )

        assert "重新設計" in metrics.recommendation

    def test_warnings_pbo(self):
        """測試警告訊息（PBO）"""
        metrics = OverfitMetrics(
            pbo=0.60,
            is_oos_ratio=1.2,
            param_sensitivity=0.10,
            trade_count=100,
        )

        warnings = metrics.warnings
        assert any("PBO" in w and "超過 50%" in w for w in warnings)

    def test_warnings_is_oos_ratio(self):
        """測試警告訊息（IS/OOS 比）"""
        metrics = OverfitMetrics(
            pbo=0.1,
            is_oos_ratio=2.5,
            param_sensitivity=0.10,
            trade_count=100,
        )

        warnings = metrics.warnings
        assert any("IS/OOS 比" in w and "超過 2.0" in w for w in warnings)

    def test_warnings_trade_count(self):
        """測試警告訊息（交易筆數）"""
        metrics = OverfitMetrics(
            pbo=0.1,
            is_oos_ratio=1.2,
            param_sensitivity=0.10,
            trade_count=20,
        )

        warnings = metrics.warnings
        assert any("交易筆數" in w and "< 30" in w for w in warnings)

    def test_warnings_param_sensitivity(self):
        """測試警告訊息（參數敏感度）"""
        metrics = OverfitMetrics(
            pbo=0.1,
            is_oos_ratio=1.2,
            param_sensitivity=0.40,
            trade_count=100,
        )

        warnings = metrics.warnings
        assert any("參數敏感度" in w and "> 30%" in w for w in warnings)

    def test_to_dict(self):
        """測試轉字典"""
        metrics = OverfitMetrics(
            pbo=0.3,
            is_oos_ratio=1.5,
            param_sensitivity=0.15,
            trade_count=75,
            degradation=0.1,
        )

        d = metrics.to_dict()

        assert d["pbo"] == 0.3
        assert d["is_oos_ratio"] == 1.5
        assert d["param_sensitivity"] == 0.15
        assert d["trade_count"] == 75
        assert d["degradation"] == 0.1
        assert "overall_risk" in d
        assert "recommendation" in d
        assert "warnings" in d


# ========== TestCalculatePBO ==========

class TestCalculatePBO:
    """OverfitDetector.calculate_pbo() 測試"""

    def test_pbo_calculation_shape(self, detector, sample_returns_matrix):
        """測試 PBO 計算（形狀驗證）"""
        result = detector.calculate_pbo(sample_returns_matrix, n_splits=8)

        assert isinstance(result, PBOResult)
        assert 0 <= result.pbo <= 1
        assert result.degradation >= 0
        assert result.n_combinations > 0

    def test_pbo_insufficient_data(self, detector):
        """測試 PBO（資料不足）"""
        small_matrix = np.random.randn(5, 10)  # 太少時間點
        result = detector.calculate_pbo(small_matrix, n_splits=8)

        # 應回傳中性值
        assert result.pbo == 0.5

    def test_pbo_empty_matrix(self, detector):
        """測試 PBO（空矩陣）"""
        empty_matrix = np.array([]).reshape(0, 0)
        result = detector.calculate_pbo(empty_matrix, n_splits=8)

        assert result.pbo == 0.5

    def test_pbo_degradation(self, detector, sample_returns_matrix):
        """測試 PBO 績效衰退計算"""
        result = detector.calculate_pbo(sample_returns_matrix, n_splits=8)

        assert 0 <= result.degradation <= 1

    def test_pbo_different_metrics(self, detector, sample_returns_matrix):
        """測試不同績效指標"""
        # Sharpe（預設）
        result_sharpe = detector.calculate_pbo(
            sample_returns_matrix, n_splits=8, metric="sharpe"
        )
        assert isinstance(result_sharpe.pbo, float)

        # Return
        result_return = detector.calculate_pbo(
            sample_returns_matrix, n_splits=8, metric="return"
        )
        assert isinstance(result_return.pbo, float)

        # Omega
        result_omega = detector.calculate_pbo(
            sample_returns_matrix, n_splits=8, metric="omega"
        )
        assert isinstance(result_omega.pbo, float)

    def test_pbo_consistent_with_same_data(self, detector, sample_returns_matrix):
        """測試 PBO 一致性（相同資料）"""
        result1 = detector.calculate_pbo(sample_returns_matrix, n_splits=8)
        result2 = detector.calculate_pbo(sample_returns_matrix, n_splits=8)

        # 應該得到相同結果（因為沒有隨機性）
        assert result1.pbo == result2.pbo


# ========== TestCalculateIsOosRatio ==========

class TestCalculateIsOosRatio:
    """IS/OOS 比計算測試"""

    def test_is_oos_ratio_normal(self, detector):
        """測試正常情況"""
        ratio = detector.calculate_is_oos_ratio(is_sharpe=2.0, oos_sharpe=1.5)
        assert ratio == pytest.approx(2.0 / 1.5)

    def test_is_oos_ratio_perfect(self, detector):
        """測試完全效率"""
        ratio = detector.calculate_is_oos_ratio(is_sharpe=2.0, oos_sharpe=2.0)
        assert ratio == pytest.approx(1.0)

    def test_is_oos_ratio_zero_oos(self, detector):
        """測試 OOS Sharpe = 0"""
        ratio = detector.calculate_is_oos_ratio(is_sharpe=2.0, oos_sharpe=0.0)
        assert ratio == float("inf")

    def test_is_oos_ratio_both_zero(self, detector):
        """測試兩者都為 0"""
        ratio = detector.calculate_is_oos_ratio(is_sharpe=0.0, oos_sharpe=0.0)
        assert ratio == 1.0

    def test_is_oos_ratio_negative_is(self, detector):
        """測試 IS Sharpe < 0（負值計算）"""
        ratio = detector.calculate_is_oos_ratio(is_sharpe=-1.0, oos_sharpe=0.5)
        # 負值除以正值得到負數
        assert ratio == pytest.approx(-2.0)


# ========== TestCalculateParamSensitivity ==========

class TestCalculateParamSensitivity:
    """參數敏感度計算測試"""

    def test_param_sensitivity_stable(self, detector):
        """測試穩定參數"""
        # 所有 Sharpe 都相近
        sharpe_matrix = np.array([[2.0, 2.1, 1.9], [2.05, 2.0, 2.1]])

        sensitivity, is_sensitive = detector.calculate_param_sensitivity(sharpe_matrix)

        assert sensitivity < 0.3
        assert not is_sensitive

    def test_param_sensitivity_sensitive(self, detector):
        """測試敏感參數"""
        # Sharpe 變化很大
        sharpe_matrix = np.array([[2.0, 0.5, 3.5], [0.3, 2.2, 1.5]])

        sensitivity, is_sensitive = detector.calculate_param_sensitivity(sharpe_matrix)

        assert is_sensitive

    def test_param_sensitivity_single_column(self, detector):
        """測試單欄矩陣"""
        sharpe_matrix = np.array([[2.0], [1.9], [2.1]])

        sensitivity, is_sensitive = detector.calculate_param_sensitivity(sharpe_matrix)

        assert sensitivity >= 0

    def test_param_sensitivity_single_cell(self, detector):
        """測試單一值"""
        sharpe_matrix = np.array([[2.0]])

        sensitivity, is_sensitive = detector.calculate_param_sensitivity(sharpe_matrix)

        assert sensitivity == 0.0
        assert not is_sensitive

    def test_param_sensitivity_empty(self, detector):
        """測試空矩陣"""
        sharpe_matrix = np.array([]).reshape(0, 0)

        sensitivity, is_sensitive = detector.calculate_param_sensitivity(sharpe_matrix)

        assert sensitivity == 0.0
        assert not is_sensitive


# ========== TestCheckTradeCount ==========

class TestCheckTradeCount:
    """交易筆數檢查測試"""

    def test_check_trade_count_sufficient(self, detector):
        """測試充足交易筆數"""
        passed, message = detector.check_trade_count(100)

        assert passed
        assert "統計有效" in message

    def test_check_trade_count_low(self, detector):
        """測試交易筆數偏低（但足夠）"""
        passed, message = detector.check_trade_count(40)

        assert passed
        assert "信心較低" in message

    def test_check_trade_count_insufficient(self, detector):
        """測試交易筆數不足"""
        passed, message = detector.check_trade_count(20)

        assert not passed
        assert "統計無效" in message

    def test_check_trade_count_zero(self, detector):
        """測試零交易"""
        passed, message = detector.check_trade_count(0)

        assert not passed

    def test_check_trade_count_custom_threshold(self):
        """測試自訂閾值"""
        detector = OverfitDetector(min_trades=50)
        passed, message = detector.check_trade_count(45)

        assert not passed

        passed, message = detector.check_trade_count(55)
        assert passed


# ========== TestAssess ==========

class TestAssess:
    """綜合評估測試"""

    def test_assess_low_risk(self, detector):
        """測試低風險評估"""
        metrics = detector.assess(
            is_sharpe=2.0,
            oos_sharpe=1.9,
            trade_count=100,
            param_sensitivity=0.10,
        )

        assert metrics.pbo == 0.0  # 無矩陣時
        assert metrics.is_oos_ratio == pytest.approx(2.0 / 1.9)
        assert metrics.param_sensitivity == 0.10
        assert metrics.trade_count == 100

    def test_assess_with_returns_matrix(self, detector, sample_returns_matrix):
        """測試使用報酬矩陣的評估"""
        metrics = detector.assess(
            is_sharpe=2.0,
            oos_sharpe=1.5,
            trade_count=75,
            param_sensitivity=0.15,
            returns_matrix=sample_returns_matrix,
        )

        assert 0 <= metrics.pbo <= 1
        assert metrics.degradation >= 0


# ========== TestShouldRejectStrategy ==========

class TestShouldRejectStrategy:
    """策略拒絕邏輯測試"""

    def test_should_reject_pbo(self, detector):
        """測試 PBO 拒絕"""
        metrics = OverfitMetrics(
            pbo=0.6,  # > 0.5
            is_oos_ratio=1.2,
            param_sensitivity=0.10,
            trade_count=100,
        )

        should_reject, reason = detector.should_reject_strategy(metrics)

        assert should_reject
        assert "PBO" in reason

    def test_should_reject_is_oos_ratio(self, detector):
        """測試 IS/OOS 比拒絕"""
        metrics = OverfitMetrics(
            pbo=0.2,
            is_oos_ratio=2.5,  # > 2.0
            param_sensitivity=0.10,
            trade_count=100,
        )

        should_reject, reason = detector.should_reject_strategy(metrics)

        assert should_reject
        assert "IS/OOS" in reason

    def test_should_reject_trade_count(self, detector):
        """測試交易筆數拒絕"""
        metrics = OverfitMetrics(
            pbo=0.2,
            is_oos_ratio=1.2,
            param_sensitivity=0.10,
            trade_count=20,  # < 30
        )

        should_reject, reason = detector.should_reject_strategy(metrics)

        assert should_reject
        assert "交易筆數" in reason

    def test_should_reject_param_sensitivity(self, detector):
        """測試參數敏感度拒絕"""
        metrics = OverfitMetrics(
            pbo=0.2,
            is_oos_ratio=1.2,
            param_sensitivity=0.40,  # > 0.3
            trade_count=100,
        )

        should_reject, reason = detector.should_reject_strategy(metrics)

        assert should_reject
        assert "參數敏感度" in reason

    def test_should_not_reject_good_metrics(self, detector):
        """測試良好指標不拒絕"""
        metrics = OverfitMetrics(
            pbo=0.2,
            is_oos_ratio=1.3,
            param_sensitivity=0.15,
            trade_count=100,
        )

        should_reject, reason = detector.should_reject_strategy(metrics)

        assert not should_reject
        assert "通過" in reason

    def test_should_reject_strict_detector(self, detector_strict):
        """測試嚴格配置"""
        metrics = OverfitMetrics(
            pbo=0.35,  # > 0.3（嚴格閾值）
            is_oos_ratio=1.2,
            param_sensitivity=0.15,
            trade_count=60,
        )

        should_reject, reason = detector_strict.should_reject_strategy(metrics)

        assert should_reject
        assert "PBO" in reason


# ========== TestEdgeCases ==========

class TestEdgeCases:
    """邊界情況測試"""

    def test_all_indicators_at_threshold(self, detector):
        """測試所有指標在閾值邊界"""
        metrics = OverfitMetrics(
            pbo=0.5,  # exactly at max_pbo
            is_oos_ratio=2.0,  # exactly at max_is_oos_ratio
            param_sensitivity=0.3,  # exactly at max_param_sensitivity
            trade_count=30,  # exactly at min_trades
        )

        should_reject, _ = detector.should_reject_strategy(metrics)

        # 應視為通過（因為是 > 而非 >=）
        assert not should_reject

    def test_extreme_pbo(self, detector):
        """測試極端 PBO 值"""
        metrics_high = OverfitMetrics(
            pbo=0.99,
            is_oos_ratio=1.0,
            param_sensitivity=0.0,
            trade_count=1000,
        )

        should_reject, _ = detector.should_reject_strategy(metrics_high)
        assert should_reject

        metrics_low = OverfitMetrics(
            pbo=0.01,
            is_oos_ratio=1.0,
            param_sensitivity=0.0,
            trade_count=1000,
        )

        should_reject, _ = detector.should_reject_strategy(metrics_low)
        assert not should_reject

    def test_very_high_trade_count(self, detector):
        """測試非常高的交易筆數"""
        passed, message = detector.check_trade_count(10000)

        assert passed
        assert "有效" in message
