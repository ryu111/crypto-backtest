"""
Walk-Forward 分析器（含 Combinatorial Purged CV）

防止過擬合的核心驗證工具：
1. Combinatorial Purged Cross-Validation
2. Embargo Period
3. Purged K-Fold
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Callable
import pandas as pd
import numpy as np
from itertools import combinations


@dataclass
class PurgedFoldResult:
    """單一 Purged Fold 結果"""

    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    purge_start: Optional[datetime]
    purge_end: Optional[datetime]
    embargo_start: Optional[datetime]
    embargo_end: Optional[datetime]

    # 績效指標
    train_return: float
    test_return: float
    train_sharpe: float
    test_sharpe: float
    train_trades: int
    test_trades: int

    # 參數
    best_params: Dict[str, Any]

    def to_dict(self) -> Dict:
        """轉為字典"""
        return {
            'fold_id': self.fold_id,
            'train_start': self.train_start,
            'train_end': self.train_end,
            'test_start': self.test_start,
            'test_end': self.test_end,
            'purge_start': self.purge_start,
            'purge_end': self.purge_end,
            'embargo_start': self.embargo_start,
            'embargo_end': self.embargo_end,
            'train_return': self.train_return,
            'test_return': self.test_return,
            'train_sharpe': self.train_sharpe,
            'test_sharpe': self.test_sharpe,
            'train_trades': self.train_trades,
            'test_trades': self.test_trades,
            'best_params': self.best_params
        }


@dataclass
class CombinatorialCVResult:
    """Combinatorial Purged CV 結果"""

    folds: List[PurgedFoldResult]
    n_splits: int
    n_test_groups: int
    n_combinations: int

    # 整體績效
    mean_train_return: float
    mean_test_return: float
    mean_train_sharpe: float
    mean_test_sharpe: float

    # 穩健性指標
    test_return_std: float
    test_sharpe_std: float
    overfitting_ratio: float  # mean_test / mean_train
    consistency: float  # 勝率

    # 配置
    purge_gap: int
    embargo_pct: float

    def __post_init__(self):
        """計算統計指標"""
        test_returns = [f.test_return for f in self.folds]
        train_returns = [f.train_return for f in self.folds]
        test_sharpes = [f.test_sharpe for f in self.folds]
        train_sharpes = [f.train_sharpe for f in self.folds]

        self.mean_test_return = np.mean(test_returns)
        self.mean_train_return = np.mean(train_returns)
        self.mean_test_sharpe = np.mean(test_sharpes)
        self.mean_train_sharpe = np.mean(train_sharpes)

        self.test_return_std = np.std(test_returns)
        self.test_sharpe_std = np.std(test_sharpes)

        # 過擬合比例（越接近 1 越好）
        self.overfitting_ratio = (
            self.mean_test_return / self.mean_train_return
            if self.mean_train_return != 0 else 0.0
        )

        # 一致性（測試集勝率）
        self.consistency = sum(1 for r in test_returns if r > 0) / len(test_returns)

    def to_dict(self) -> Dict:
        """轉為字典"""
        return {
            'n_splits': self.n_splits,
            'n_test_groups': self.n_test_groups,
            'n_combinations': self.n_combinations,
            'mean_train_return': self.mean_train_return,
            'mean_test_return': self.mean_test_return,
            'mean_train_sharpe': self.mean_train_sharpe,
            'mean_test_sharpe': self.mean_test_sharpe,
            'test_return_std': self.test_return_std,
            'test_sharpe_std': self.test_sharpe_std,
            'overfitting_ratio': self.overfitting_ratio,
            'consistency': self.consistency,
            'purge_gap': self.purge_gap,
            'embargo_pct': self.embargo_pct,
            'folds': [f.to_dict() for f in self.folds]
        }

    def summary(self) -> str:
        """產生摘要報告"""
        return f"""
Combinatorial Purged Cross-Validation 結果
{'='*70}
配置
{'-'*70}
K-Fold 數量: {self.n_splits}
測試組數量: {self.n_test_groups}
組合總數: {self.n_combinations}
Purge Gap: {self.purge_gap}
Embargo %: {self.embargo_pct:.2%}

整體績效
{'-'*70}
訓練集平均報酬: {self.mean_train_return:.2%}
測試集平均報酬: {self.mean_test_return:.2%} ± {self.test_return_std:.2%}
訓練集平均夏普: {self.mean_train_sharpe:.2f}
測試集平均夏普: {self.mean_test_sharpe:.2f} ± {self.test_sharpe_std:.2f}

穩健性指標
{'-'*70}
過擬合比例: {self.overfitting_ratio:.2%} (越接近 100% 越好)
測試集勝率: {self.consistency:.2%}

Fold 詳情
{'-'*70}
{'ID':<4} {'訓練報酬':<12} {'測試報酬':<12} {'訓練夏普':<12} {'測試夏普':<12}
{'-'*70}
""" + "\n".join([
    f"{f.fold_id:<4} {f.train_return:<12.2%} {f.test_return:<12.2%} "
    f"{f.train_sharpe:<12.2f} {f.test_sharpe:<12.2f}"
    for f in self.folds
])


class PurgedKFold:
    """
    Purged K-Fold Cross-Validation

    防止資訊洩漏的交叉驗證方法，適合時間序列資料。

    關鍵特性：
    1. **Purge Gap**: 訓練結束和測試開始之間的緩衝期，避免 look-ahead bias
    2. **Embargo Period**: 測試結束後的禁運期，防止資訊反向洩漏
    3. **時序保持**: 不打亂資料順序

    使用範例：
        purged_kfold = PurgedKFold(
            n_splits=5,
            purge_gap=10,
            embargo_pct=0.01
        )

        for fold_id, (train_idx, test_idx) in enumerate(purged_kfold.split(data)):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            # ... 訓練和測試

    參考文獻：
        Advances in Financial Machine Learning, Marcos López de Prado
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 0,
        embargo_pct: float = 0.01
    ):
        """
        初始化 Purged K-Fold

        Args:
            n_splits: K-Fold 數量
            purge_gap: 訓練結束和測試開始之間的 gap（避免 look-ahead bias）
            embargo_pct: embargo 期間佔測試集的比例
        """
        if n_splits < 2:
            raise ValueError("n_splits 必須 >= 2")
        if purge_gap < 0:
            raise ValueError("purge_gap 必須 >= 0")
        if not 0 <= embargo_pct < 1:
            raise ValueError("embargo_pct 必須在 [0, 1) 之間")

        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

    def split(
        self,
        data: pd.DataFrame
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        產生 Purged K-Fold 索引

        Args:
            data: 時間序列資料（需要有 DatetimeIndex）

        Returns:
            List of (train_indices, test_indices)
        """
        n_samples = len(data)
        fold_size = n_samples // self.n_splits

        splits = []

        for i in range(self.n_splits):
            # 計算測試集範圍
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples

            # 計算 embargo 期間
            embargo_size = int((test_end - test_start) * self.embargo_pct)
            embargo_end = min(test_end + embargo_size, n_samples)

            # 測試集索引（不包含 embargo）
            test_indices = np.arange(test_start, test_end)

            # 訓練集：所有非測試集索引
            train_indices = []

            # 訓練集前段（測試集之前）
            if test_start > 0:
                # 扣除 purge gap
                train_before_end = max(0, test_start - self.purge_gap)
                train_indices.extend(np.arange(0, train_before_end))

            # 訓練集後段（測試集之後，扣除 embargo）
            if embargo_end < n_samples:
                train_indices.extend(np.arange(embargo_end, n_samples))

            train_indices = np.array(train_indices)

            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))

        return splits

    def get_fold_info(
        self,
        data: pd.DataFrame,
        fold_id: int
    ) -> Dict[str, Any]:
        """
        獲取特定 fold 的詳細資訊

        Args:
            data: 資料
            fold_id: Fold ID

        Returns:
            包含 fold 資訊的字典
        """
        splits = self.split(data)

        if fold_id >= len(splits):
            raise ValueError(f"fold_id {fold_id} 超出範圍（總共 {len(splits)} 個 folds）")

        train_idx, test_idx = splits[fold_id]

        # 計算各階段時間
        test_start = data.index[test_idx[0]]
        test_end = data.index[test_idx[-1]]

        if len(train_idx) > 0:
            train_start = data.index[train_idx[0]]
            train_end = data.index[train_idx[-1]]

            # 找出 purge 區間（如果存在）
            purge_start = None
            purge_end = None
            if self.purge_gap > 0 and train_idx[-1] + self.purge_gap < test_idx[0]:
                purge_start = data.index[train_idx[-1] + 1]
                purge_end = data.index[test_idx[0] - 1]
        else:
            train_start = None
            train_end = None
            purge_start = None
            purge_end = None

        # 找出 embargo 區間（如果存在）
        embargo_start = None
        embargo_end = None
        if self.embargo_pct > 0:
            embargo_size = int(len(test_idx) * self.embargo_pct)
            if test_idx[-1] + embargo_size < len(data):
                embargo_start = data.index[test_idx[-1] + 1]
                embargo_end = data.index[min(test_idx[-1] + embargo_size, len(data) - 1)]

        return {
            'fold_id': fold_id,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'purge_start': purge_start,
            'purge_end': purge_end,
            'embargo_start': embargo_start,
            'embargo_end': embargo_end,
            'train_size': len(train_idx),
            'test_size': len(test_idx)
        }


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation

    結合 Purged K-Fold 和組合驗證，提供更全面的過擬合檢測。

    工作原理：
    1. 將資料分成 N 個 fold
    2. 每次選擇 k 個 fold 作為測試集
    3. 剩餘 fold 作為訓練集
    4. 所有可能的組合都會被測試
    5. 每個 fold 之間加入 purge gap 和 embargo

    優勢：
    - 更多樣的測試組合 (C(N, k) 種)
    - 每個樣本都有機會被測試多次
    - 降低單一測試集的偶然性

    使用範例：
        cv = CombinatorialPurgedCV(
            n_splits=5,
            n_test_groups=2,
            purge_gap=10,
            embargo_pct=0.01
        )

        result = cv.validate(
            data=market_data,
            strategy_func=my_strategy,
            param_grid={'period': [10, 20, 30]}
        )

        print(result.summary())

    參考文獻：
        Advances in Financial Machine Learning, Chapter 7
        Marcos López de Prado
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_groups: int = 2,
        purge_gap: int = 0,
        embargo_pct: float = 0.01
    ):
        """
        初始化 Combinatorial Purged CV

        Args:
            n_splits: K-Fold 數量
            n_test_groups: 每次用作測試集的 fold 數量
            purge_gap: 訓練/測試之間的 gap
            embargo_pct: embargo 期間佔比
        """
        if n_splits < 3:
            raise ValueError("n_splits 必須 >= 3")
        if n_test_groups >= n_splits:
            raise ValueError("n_test_groups 必須 < n_splits")
        if n_test_groups < 1:
            raise ValueError("n_test_groups 必須 >= 1")

        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

        self.purged_kfold = PurgedKFold(
            n_splits=n_splits,
            purge_gap=purge_gap,
            embargo_pct=embargo_pct
        )

    def validate(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        param_grid: Optional[Dict[str, List]] = None,
        optimize_metric: str = 'sharpe_ratio',
        verbose: bool = True
    ) -> CombinatorialCVResult:
        """
        執行 Combinatorial Purged CV

        Args:
            data: 時間序列資料
            strategy_func: 策略函數 (data, params) -> (returns, sharpe, trades)
            param_grid: 參數網格（可選）
            optimize_metric: 優化指標
            verbose: 是否輸出進度

        Returns:
            CombinatorialCVResult
        """
        # 產生所有組合
        fold_combinations = list(combinations(range(self.n_splits), self.n_test_groups))
        n_combinations = len(fold_combinations)

        if verbose:
            print(f"開始 Combinatorial Purged CV")
            print(f"  K-Fold: {self.n_splits}")
            print(f"  測試組數: {self.n_test_groups}")
            print(f"  組合總數: {n_combinations}")
            print(f"  Purge Gap: {self.purge_gap}")
            print(f"  Embargo: {self.embargo_pct:.2%}")
            print()

        # 獲取所有 fold splits
        all_splits = self.purged_kfold.split(data)

        fold_results = []

        for combo_id, test_fold_ids in enumerate(fold_combinations, 1):
            if verbose:
                print(f"處理組合 {combo_id}/{n_combinations}: 測試 folds {test_fold_ids}")

            # 組合測試集索引
            test_indices = []
            for fold_id in test_fold_ids:
                _, test_idx = all_splits[fold_id]
                test_indices.extend(test_idx)
            test_indices = np.array(sorted(set(test_indices)))

            # 組合訓練集索引（剩餘所有 fold）
            train_indices = []
            for fold_id in range(self.n_splits):
                if fold_id not in test_fold_ids:
                    train_idx, _ = all_splits[fold_id]
                    train_indices.extend(train_idx)
            train_indices = np.array(sorted(set(train_indices)))

            # 提取資料
            train_data = data.iloc[train_indices]
            test_data = data.iloc[test_indices]

            # 1. 訓練階段（參數優化）
            if param_grid:
                best_params, train_metrics = self._optimize_params(
                    train_data, strategy_func, param_grid, optimize_metric
                )
            else:
                best_params = {}
                train_metrics = strategy_func(train_data, best_params)

            # 2. 測試階段
            test_metrics = strategy_func(test_data, best_params)

            # 3. 記錄結果
            fold_result = PurgedFoldResult(
                fold_id=combo_id,
                train_start=train_data.index[0],
                train_end=train_data.index[-1],
                test_start=test_data.index[0],
                test_end=test_data.index[-1],
                purge_start=None,  # 組合模式下不適用
                purge_end=None,
                embargo_start=None,
                embargo_end=None,
                train_return=train_metrics['return'],
                test_return=test_metrics['return'],
                train_sharpe=train_metrics['sharpe'],
                test_sharpe=test_metrics['sharpe'],
                train_trades=train_metrics['trades'],
                test_trades=test_metrics['trades'],
                best_params=best_params
            )

            fold_results.append(fold_result)

            if verbose:
                print(f"  訓練: 報酬={train_metrics['return']:.2%}, 夏普={train_metrics['sharpe']:.2f}")
                print(f"  測試: 報酬={test_metrics['return']:.2%}, 夏普={test_metrics['sharpe']:.2f}")
                print()

        # 整合結果
        result = CombinatorialCVResult(
            folds=fold_results,
            n_splits=self.n_splits,
            n_test_groups=self.n_test_groups,
            n_combinations=n_combinations,
            mean_train_return=0.0,  # 會在 __post_init__ 計算
            mean_test_return=0.0,
            mean_train_sharpe=0.0,
            mean_test_sharpe=0.0,
            test_return_std=0.0,
            test_sharpe_std=0.0,
            overfitting_ratio=0.0,
            consistency=0.0,
            purge_gap=self.purge_gap,
            embargo_pct=self.embargo_pct
        )

        return result

    def _optimize_params(
        self,
        train_data: pd.DataFrame,
        strategy_func: Callable,
        param_grid: Dict[str, List],
        optimize_metric: str
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        網格搜尋參數優化

        Args:
            train_data: 訓練資料
            strategy_func: 策略函數
            param_grid: 參數網格
            optimize_metric: 優化指標

        Returns:
            (best_params, best_metrics)
        """
        from itertools import product

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        best_metric = float('-inf')
        best_params = None
        best_metrics = None

        for values in product(*param_values):
            params = dict(zip(param_names, values))

            try:
                metrics = strategy_func(train_data, params)
                current_metric = metrics.get(optimize_metric, metrics.get('sharpe', 0))

                if current_metric > best_metric:
                    best_metric = current_metric
                    best_params = params.copy()
                    best_metrics = metrics
            except Exception:
                continue

        if best_params is None:
            raise ValueError("參數優化失敗：沒有任何參數組合產生有效結果")

        return best_params, best_metrics


def combinatorial_purged_cv(
    returns: pd.Series,
    strategy_func: Callable,
    n_splits: int = 5,
    n_test_groups: int = 2,
    purge_gap: int = 0,
    embargo_pct: float = 0.01,
    param_grid: Optional[Dict[str, List]] = None
) -> CombinatorialCVResult:
    """
    便捷函數：執行 Combinatorial Purged CV

    這是一個簡化的介面，適合快速驗證策略。

    Args:
        returns: 報酬序列（DatetimeIndex）
        strategy_func: 策略函數 (data, params) -> dict
            必須返回 {'return': float, 'sharpe': float, 'trades': int}
        n_splits: K-Fold 數量
        n_test_groups: 測試組數量
        purge_gap: Purge gap
        embargo_pct: Embargo 百分比
        param_grid: 參數網格

    Returns:
        CombinatorialCVResult

    範例：
        def my_strategy(data, params):
            # ... 策略邏輯
            return {
                'return': total_return,
                'sharpe': sharpe_ratio,
                'trades': num_trades
            }

        result = combinatorial_purged_cv(
            returns=my_returns,
            strategy_func=my_strategy,
            n_splits=5,
            n_test_groups=2
        )
    """
    # 將 Series 轉為 DataFrame（如果需要）
    if isinstance(returns, pd.Series):
        data = returns.to_frame(name='returns')
    else:
        data = returns

    cv = CombinatorialPurgedCV(
        n_splits=n_splits,
        n_test_groups=n_test_groups,
        purge_gap=purge_gap,
        embargo_pct=embargo_pct
    )

    return cv.validate(
        data=data,
        strategy_func=strategy_func,
        param_grid=param_grid
    )
