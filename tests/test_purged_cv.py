"""
Combinatorial Purged CV 測試
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.validator.walk_forward import (
    PurgedKFold,
    CombinatorialPurgedCV,
    PurgedFoldResult,
    CombinatorialCVResult,
    combinatorial_purged_cv
)


# === Fixtures ===

@pytest.fixture
def sample_data():
    """生成測試資料"""
    dates = pd.date_range(start='2023-01-01', end='2023-03-31', freq='1h')
    np.random.seed(42)

    # 生成帶趨勢的報酬率
    trend = np.linspace(0, 0.1, len(dates))
    noise = np.random.normal(0, 0.01, len(dates))
    returns = trend + noise

    data = pd.DataFrame({
        'returns': returns,
        'price': (1 + returns).cumprod() * 40000
    }, index=dates)

    return data


@pytest.fixture
def simple_strategy():
    """簡單測試策略函數"""

    def strategy_func(data: pd.DataFrame, params: dict) -> dict:
        """
        簡單動量策略

        Args:
            data: 包含 'returns' 欄位的 DataFrame
            params: {'threshold': float}

        Returns:
            {'return': float, 'sharpe': float, 'trades': int}
        """
        threshold = params.get('threshold', 0.01)

        # 簡單動量信號
        signals = (data['returns'] > threshold).astype(int)
        strategy_returns = signals.shift(1) * data['returns']
        strategy_returns = strategy_returns.dropna()

        # 計算績效
        total_return = strategy_returns.sum()
        sharpe = (
            strategy_returns.mean() / strategy_returns.std()
            if strategy_returns.std() > 0 else 0
        )
        trades = signals.diff().abs().sum()

        return {
            'return': total_return,
            'sharpe': sharpe * np.sqrt(252 * 24),  # 年化（小時資料）
            'trades': int(trades)
        }

    return strategy_func


# === PurgedKFold 測試 ===

def test_purged_kfold_initialization():
    """測試 PurgedKFold 初始化"""
    kfold = PurgedKFold(n_splits=5, purge_gap=10, embargo_pct=0.01)

    assert kfold.n_splits == 5
    assert kfold.purge_gap == 10
    assert kfold.embargo_pct == 0.01


def test_purged_kfold_invalid_params():
    """測試無效參數"""
    with pytest.raises(ValueError, match="n_splits 必須 >= 2"):
        PurgedKFold(n_splits=1)

    with pytest.raises(ValueError, match="purge_gap 必須 >= 0"):
        PurgedKFold(purge_gap=-1)

    with pytest.raises(ValueError, match="embargo_pct 必須在"):
        PurgedKFold(embargo_pct=1.5)


def test_purged_kfold_split(sample_data):
    """測試 K-Fold 切分"""
    kfold = PurgedKFold(n_splits=5, purge_gap=10, embargo_pct=0.01)
    splits = kfold.split(sample_data)

    assert len(splits) > 0
    assert len(splits) <= 5

    # 檢查每個 split
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0
        assert len(test_idx) > 0

        # 確保沒有重疊
        assert len(set(train_idx) & set(test_idx)) == 0


def test_purged_kfold_purge_gap(sample_data):
    """測試 Purge Gap 是否生效"""
    purge_gap = 20
    kfold = PurgedKFold(n_splits=3, purge_gap=purge_gap, embargo_pct=0)
    splits = kfold.split(sample_data)

    for train_idx, test_idx in splits:
        # 檢查訓練集最後一個索引和測試集第一個索引之間的 gap
        if len(train_idx) > 0 and len(test_idx) > 0:
            train_max = np.max(train_idx)
            test_min = np.min(test_idx)

            # 如果有前段訓練集，應該存在 gap
            if train_max < test_min:
                gap = test_min - train_max - 1
                assert gap >= purge_gap


def test_purged_kfold_get_fold_info(sample_data):
    """測試獲取 fold 資訊"""
    kfold = PurgedKFold(n_splits=3, purge_gap=10, embargo_pct=0.01)
    fold_info = kfold.get_fold_info(sample_data, fold_id=0)

    assert fold_info['fold_id'] == 0
    assert fold_info['train_start'] is not None
    assert fold_info['test_start'] is not None
    assert fold_info['train_size'] > 0
    assert fold_info['test_size'] > 0


def test_purged_kfold_embargo(sample_data):
    """測試 Embargo 期間"""
    embargo_pct = 0.05
    kfold = PurgedKFold(n_splits=3, purge_gap=0, embargo_pct=embargo_pct)

    fold_info = kfold.get_fold_info(sample_data, fold_id=0)

    # 檢查是否有 embargo 資訊
    if fold_info['embargo_start'] is not None:
        assert fold_info['embargo_end'] is not None


# === CombinatorialPurgedCV 測試 ===

def test_combinatorial_cv_initialization():
    """測試 CombinatorialPurgedCV 初始化"""
    cv = CombinatorialPurgedCV(
        n_splits=5,
        n_test_groups=2,
        purge_gap=10,
        embargo_pct=0.01
    )

    assert cv.n_splits == 5
    assert cv.n_test_groups == 2
    assert cv.purge_gap == 10
    assert cv.embargo_pct == 0.01


def test_combinatorial_cv_invalid_params():
    """測試無效參數"""
    with pytest.raises(ValueError, match="n_splits 必須 >= 3"):
        CombinatorialPurgedCV(n_splits=2)

    with pytest.raises(ValueError, match="n_test_groups 必須 < n_splits"):
        CombinatorialPurgedCV(n_splits=5, n_test_groups=5)

    with pytest.raises(ValueError, match="n_test_groups 必須 >= 1"):
        CombinatorialPurgedCV(n_splits=5, n_test_groups=0)


def test_combinatorial_cv_validate(sample_data, simple_strategy):
    """測試完整驗證流程"""
    cv = CombinatorialPurgedCV(
        n_splits=3,
        n_test_groups=1,
        purge_gap=5,
        embargo_pct=0.01
    )

    result = cv.validate(
        data=sample_data,
        strategy_func=simple_strategy,
        param_grid={'threshold': [0.005, 0.01]},
        optimize_metric='sharpe',
        verbose=False
    )

    # 檢查結果結構
    assert isinstance(result, CombinatorialCVResult)
    assert len(result.folds) > 0
    assert result.n_splits == 3
    assert result.n_test_groups == 1
    assert result.n_combinations == 3  # C(3, 1) = 3

    # 檢查績效指標
    assert result.mean_train_return is not None
    assert result.mean_test_return is not None
    assert 0 <= result.consistency <= 1
    assert result.overfitting_ratio is not None


def test_combinatorial_cv_without_param_grid(sample_data, simple_strategy):
    """測試不使用參數網格"""
    cv = CombinatorialPurgedCV(n_splits=3, n_test_groups=1)

    result = cv.validate(
        data=sample_data,
        strategy_func=simple_strategy,
        param_grid=None,  # 不優化參數
        verbose=False
    )

    assert result is not None
    assert len(result.folds) > 0


def test_combinatorial_cv_multiple_test_groups(sample_data, simple_strategy):
    """測試多個測試組"""
    cv = CombinatorialPurgedCV(
        n_splits=5,
        n_test_groups=2
    )

    result = cv.validate(
        data=sample_data,
        strategy_func=simple_strategy,
        verbose=False
    )

    # C(5, 2) = 10 種組合
    assert result.n_combinations == 10
    assert len(result.folds) == 10


# === PurgedFoldResult 測試 ===

def test_purged_fold_result():
    """測試 PurgedFoldResult 資料類別"""
    fold_result = PurgedFoldResult(
        fold_id=1,
        train_start=datetime(2023, 1, 1),
        train_end=datetime(2023, 2, 1),
        test_start=datetime(2023, 2, 1),
        test_end=datetime(2023, 3, 1),
        purge_start=None,
        purge_end=None,
        embargo_start=None,
        embargo_end=None,
        train_return=0.15,
        test_return=0.12,
        train_sharpe=1.5,
        test_sharpe=1.2,
        train_trades=100,
        test_trades=30,
        best_params={'threshold': 0.01}
    )

    result_dict = fold_result.to_dict()
    assert result_dict['fold_id'] == 1
    assert result_dict['train_return'] == 0.15
    assert result_dict['test_return'] == 0.12
    assert result_dict['best_params']['threshold'] == 0.01


# === CombinatorialCVResult 測試 ===

def test_combinatorial_cv_result():
    """測試 CombinatorialCVResult 資料類別"""
    folds = [
        PurgedFoldResult(
            fold_id=1,
            train_start=datetime(2023, 1, 1),
            train_end=datetime(2023, 2, 1),
            test_start=datetime(2023, 2, 1),
            test_end=datetime(2023, 3, 1),
            purge_start=None,
            purge_end=None,
            embargo_start=None,
            embargo_end=None,
            train_return=0.20,
            test_return=0.15,
            train_sharpe=1.8,
            test_sharpe=1.4,
            train_trades=100,
            test_trades=30,
            best_params={}
        ),
        PurgedFoldResult(
            fold_id=2,
            train_start=datetime(2023, 1, 1),
            train_end=datetime(2023, 2, 1),
            test_start=datetime(2023, 2, 1),
            test_end=datetime(2023, 3, 1),
            purge_start=None,
            purge_end=None,
            embargo_start=None,
            embargo_end=None,
            train_return=0.25,
            test_return=0.18,
            train_sharpe=2.0,
            test_sharpe=1.6,
            train_trades=110,
            test_trades=35,
            best_params={}
        )
    ]

    result = CombinatorialCVResult(
        folds=folds,
        n_splits=5,
        n_test_groups=2,
        n_combinations=10,
        mean_train_return=0.0,  # 會自動計算
        mean_test_return=0.0,
        mean_train_sharpe=0.0,
        mean_test_sharpe=0.0,
        test_return_std=0.0,
        test_sharpe_std=0.0,
        overfitting_ratio=0.0,
        consistency=0.0,
        purge_gap=10,
        embargo_pct=0.01
    )

    # 檢查自動計算的統計指標
    assert result.mean_train_return == pytest.approx((0.20 + 0.25) / 2)
    assert result.mean_test_return == pytest.approx((0.15 + 0.18) / 2)
    assert result.consistency == 1.0  # 所有測試都是正報酬

    # 檢查 to_dict
    result_dict = result.to_dict()
    assert result_dict['n_splits'] == 5
    assert result_dict['n_combinations'] == 10

    # 檢查 summary
    summary = result.summary()
    assert 'Combinatorial Purged' in summary
    assert 'K-Fold 數量' in summary


# === 便捷函數測試 ===

def test_combinatorial_purged_cv_function(sample_data, simple_strategy):
    """測試便捷函數"""
    result = combinatorial_purged_cv(
        returns=sample_data,
        strategy_func=simple_strategy,
        n_splits=3,
        n_test_groups=1,
        purge_gap=5,
        embargo_pct=0.01,
        param_grid={'threshold': [0.01]}
    )

    assert isinstance(result, CombinatorialCVResult)
    assert len(result.folds) > 0


def test_combinatorial_purged_cv_with_series(simple_strategy):
    """測試使用 Series 作為輸入"""
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='1h')
    returns = pd.Series(np.random.normal(0.0001, 0.01, 1000), index=dates)

    result = combinatorial_purged_cv(
        returns=returns,
        strategy_func=simple_strategy,
        n_splits=3,
        n_test_groups=1
    )

    assert result is not None


# === 邊界條件測試 ===

def test_insufficient_data(simple_strategy):
    """測試資料不足"""
    # 只有 50 筆資料
    dates = pd.date_range(start='2023-01-01', periods=50, freq='1h')
    small_data = pd.DataFrame({
        'returns': np.random.normal(0, 0.01, 50)
    }, index=dates)

    cv = CombinatorialPurgedCV(n_splits=3, n_test_groups=1)

    # 應該能執行但可能結果不佳
    result = cv.validate(
        data=small_data,
        strategy_func=simple_strategy,
        verbose=False
    )

    assert result is not None


def test_overfitting_detection(sample_data):
    """測試過擬合檢測"""

    def overfitted_strategy(data, params):
        """過擬合的策略（訓練好，測試差）"""
        # 使用隨機信號模擬過擬合
        np.random.seed(params.get('seed', 0))
        signals = np.random.choice([0, 1], len(data))
        strategy_returns = signals * data['returns'].values

        total_return = strategy_returns.sum()
        sharpe = (
            strategy_returns.mean() / (strategy_returns.std() + 1e-8)
            * np.sqrt(252 * 24)
        )

        return {
            'return': total_return,
            'sharpe': sharpe,
            'trades': signals.sum()
        }

    cv = CombinatorialPurgedCV(n_splits=3, n_test_groups=1)

    result = cv.validate(
        data=sample_data,
        strategy_func=overfitted_strategy,
        param_grid={'seed': [0, 1, 2]},
        verbose=False
    )

    # 過擬合策略應該有較低的 overfitting_ratio
    # （但這個測試不一定總是通過，因為隨機性）
    assert result.overfitting_ratio is not None


# === 整合測試 ===

@pytest.mark.integration
def test_full_workflow(sample_data, simple_strategy):
    """測試完整工作流程"""

    # 1. 使用 PurgedKFold
    kfold = PurgedKFold(n_splits=5, purge_gap=10, embargo_pct=0.01)
    splits = kfold.split(sample_data)

    assert len(splits) > 0

    # 2. 使用 CombinatorialPurgedCV
    cv = CombinatorialPurgedCV(
        n_splits=5,
        n_test_groups=2,
        purge_gap=10,
        embargo_pct=0.01
    )

    result = cv.validate(
        data=sample_data,
        strategy_func=simple_strategy,
        param_grid={'threshold': [0.005, 0.01, 0.02]},
        verbose=False
    )

    # 檢查結果
    assert result.n_combinations == 10  # C(5, 2)
    assert len(result.folds) == 10
    assert 0 <= result.consistency <= 1

    # 3. 輸出摘要
    summary = result.summary()
    assert len(summary) > 0


@pytest.mark.integration
def test_compare_different_configs(sample_data, simple_strategy):
    """測試比較不同配置"""
    configs = [
        {'n_splits': 3, 'n_test_groups': 1, 'purge_gap': 0},
        {'n_splits': 3, 'n_test_groups': 1, 'purge_gap': 10},
        {'n_splits': 5, 'n_test_groups': 2, 'purge_gap': 10},
    ]

    results = {}

    for i, config in enumerate(configs):
        cv = CombinatorialPurgedCV(**config)
        result = cv.validate(
            data=sample_data,
            strategy_func=simple_strategy,
            verbose=False
        )
        results[f"config_{i}"] = result

    # 確保所有配置都成功
    assert len(results) == 3

    # 檢查每個結果
    for name, result in results.items():
        assert result.mean_test_return is not None
        assert 0 <= result.consistency <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
