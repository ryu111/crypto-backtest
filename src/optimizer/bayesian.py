"""
貝葉斯參數優化器

使用 Optuna 進行智慧參數優化,目標為最大化 Sharpe Ratio。
支援並行優化、參數空間自動採樣、優化歷史追蹤。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
import warnings
import logging
from datetime import datetime

import pandas as pd
import numpy as np

try:
    import optuna
    from optuna.trial import Trial
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    raise ImportError(
        "Optuna 未安裝。請執行: pip install optuna"
    )

from ..backtester.engine import BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """優化結果"""

    best_params: Dict[str, Any]
    best_value: float  # Best Sharpe Ratio
    n_trials: int
    history: List[Dict]  # Trial history
    study: optuna.Study

    # 最佳回測結果
    best_backtest_result: Optional[BacktestResult] = None

    # 統計資訊
    optimization_time: float = 0.0  # 秒
    n_completed_trials: int = 0
    n_failed_trials: int = 0
    n_pruned_trials: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """轉為字典"""
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials': self.n_trials,
            'n_completed_trials': self.n_completed_trials,
            'n_failed_trials': self.n_failed_trials,
            'n_pruned_trials': self.n_pruned_trials,
            'optimization_time': self.optimization_time,
            'best_backtest_metrics': (
                self.best_backtest_result.to_dict()
                if self.best_backtest_result else None
            )
        }

    def summary(self) -> str:
        """產生摘要報告"""
        return f"""
優化結果摘要
{'='*60}
最佳 Sharpe Ratio: {self.best_value:.4f}
最佳參數: {self.best_params}

統計資訊
{'-'*60}
總試驗次數: {self.n_trials}
完成: {self.n_completed_trials}
失敗: {self.n_failed_trials}
剪枝: {self.n_pruned_trials}
優化時間: {self.optimization_time:.2f} 秒

{self.best_backtest_result.summary() if self.best_backtest_result else ''}
"""

    def plot_optimization_history(self, save_path: Optional[str] = None):
        """繪製優化歷史"""
        try:
            from optuna.visualization import plot_optimization_history
            fig = plot_optimization_history(self.study)
            if save_path:
                fig.write_html(save_path)
            return fig
        except ImportError:
            warnings.warn("需要安裝 plotly: pip install plotly")

    def plot_param_importances(self, save_path: Optional[str] = None):
        """繪製參數重要性"""
        try:
            from optuna.visualization import plot_param_importances
            fig = plot_param_importances(self.study)
            if save_path:
                fig.write_html(save_path)
            return fig
        except ImportError:
            warnings.warn("需要安裝 plotly: pip install plotly")


class BayesianOptimizer:
    """
    貝葉斯優化器

    使用 Optuna 的 TPE (Tree-structured Parzen Estimator) 進行
    參數優化，目標為最大化 Sharpe Ratio。

    使用範例：
        optimizer = BayesianOptimizer(
            engine=backtest_engine,
            n_trials=100,
            n_jobs=4
        )

        result = optimizer.optimize(
            strategy=my_strategy,
            data=market_data,
            metric='sharpe_ratio'
        )

        print(result.summary())
        result.plot_optimization_history('optimization_history.html')
    """

    def __init__(
        self,
        engine: BacktestEngine,
        n_trials: int = 100,
        n_jobs: int = 1,
        timeout: Optional[float] = None,
        seed: Optional[int] = None,
        verbose: bool = True
    ):
        """
        初始化優化器

        Args:
            engine: BacktestEngine 實例
            n_trials: 試驗次數（預設 100）
            n_jobs: 並行工作數（預設 1，-1 為使用所有 CPU）
            timeout: 最大優化時間（秒），None 表示無限制
            seed: 隨機種子（可重現性）
            verbose: 是否顯示進度
        """
        self.engine = engine
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.timeout = timeout
        self.seed = seed
        self.verbose = verbose

        # 優化器狀態
        self._study: Optional[optuna.Study] = None
        self._best_result: Optional[BacktestResult] = None
        self._optimization_start_time: Optional[datetime] = None

    def optimize(
        self,
        strategy: Any,
        data: pd.DataFrame,
        metric: str = 'sharpe_ratio',
        direction: str = 'maximize',
        param_space: Optional[Dict[str, Dict]] = None,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        show_progress_bar: bool = True
    ) -> OptimizationResult:
        """
        執行參數優化

        Args:
            strategy: 策略物件（需有 param_space 屬性）
            data: 市場資料 (OHLCV DataFrame)
            metric: 優化目標指標（預設 'sharpe_ratio'）
            direction: 'maximize' 或 'minimize'（預設 'maximize'）
            param_space: 參數空間定義（若未提供則使用 strategy.param_space）
            study_name: 研究名稱（用於儲存/載入）
            storage: 儲存後端 URL（如 'sqlite:///optuna.db'）
            show_progress_bar: 是否顯示進度條

        Returns:
            OptimizationResult 物件

        參數空間格式範例:
            {
                'fast_period': {'type': 'int', 'low': 5, 'high': 20},
                'slow_period': {'type': 'int', 'low': 20, 'high': 50},
                'stop_loss_atr': {'type': 'float', 'low': 1.0, 'high': 3.0, 'step': 0.1},
                'use_filter': {'type': 'categorical', 'choices': [True, False]}
            }
        """
        # 使用策略的 param_space 或自訂的
        param_space = param_space or getattr(strategy, 'param_space', {})

        if not param_space:
            raise ValueError(
                "策略沒有定義 param_space，請提供 param_space 參數"
            )

        # 驗證 metric
        valid_metrics = [
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'total_return', 'annual_return', 'max_drawdown',
            'profit_factor', 'win_rate', 'expectancy'
        ]
        if metric not in valid_metrics:
            raise ValueError(
                f"無效的 metric: {metric}，有效值為 {valid_metrics}"
            )

        # 建立 study
        sampler = TPESampler(seed=self.seed)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)

        self._study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            storage=storage,
            load_if_exists=True
        )

        # 目標函數（使用閉包捕獲參數）
        def objective(trial: Trial) -> float:
            return self._objective(
                trial=trial,
                strategy=strategy,
                data=data,
                param_space=param_space,
                metric=metric
            )

        # 執行優化
        self._optimization_start_time = datetime.now()

        # 設定 logging
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        self._study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=show_progress_bar
        )

        optimization_time = (
            datetime.now() - self._optimization_start_time
        ).total_seconds()

        # 收集結果
        best_params = self._study.best_params
        best_value = self._study.best_value

        # 使用最佳參數執行最終回測
        final_result = self.engine.run(
            strategy=strategy,
            params=best_params,
            data=data
        )

        # 整理試驗歷史
        history = []
        for trial in self._study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    'number': trial.number,
                    'params': trial.params,
                    'value': trial.value,
                    'datetime': trial.datetime_start
                })

        # 統計資訊
        n_completed = len([
            t for t in self._study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ])
        n_failed = len([
            t for t in self._study.trials
            if t.state == optuna.trial.TrialState.FAIL
        ])
        n_pruned = len([
            t for t in self._study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        ])

        return OptimizationResult(
            best_params=best_params,
            best_value=best_value,
            n_trials=self.n_trials,
            history=history,
            study=self._study,
            best_backtest_result=final_result,
            optimization_time=optimization_time,
            n_completed_trials=n_completed,
            n_failed_trials=n_failed,
            n_pruned_trials=n_pruned
        )

    def _objective(
        self,
        trial: Trial,
        strategy: Any,
        data: pd.DataFrame,
        param_space: Dict[str, Dict],
        metric: str
    ) -> float:
        """
        Optuna 目標函數

        Args:
            trial: Optuna Trial 物件
            strategy: 策略物件
            data: 市場資料
            param_space: 參數空間定義
            metric: 優化目標指標

        Returns:
            目標指標值
        """
        # 從 trial 採樣參數
        params = self._sample_params(trial, param_space)

        try:
            # 執行回測
            result = self.engine.run(
                strategy=strategy,
                params=params,
                data=data
            )

            # 取得目標指標
            metric_value = getattr(result, metric)

            # 處理無效值
            if not np.isfinite(metric_value):
                raise optuna.TrialPruned(f"Invalid metric value: {metric_value}")

            return metric_value

        except Exception as e:
            trial.set_user_attr('error', str(e))
            logger.warning(f"Trial {trial.number} 失敗: {e}")
            raise optuna.TrialPruned()

    def _sample_params(
        self,
        trial: Trial,
        param_space: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """
        從參數空間採樣參數

        Args:
            trial: Optuna Trial 物件
            param_space: 參數空間定義

        Returns:
            採樣的參數字典

        參數空間格式:
            {
                'param_name': {
                    'type': 'int' | 'float' | 'categorical',
                    'low': number (for int/float),
                    'high': number (for int/float),
                    'step': number (optional, for int/float),
                    'log': bool (optional, for int/float),
                    'choices': list (for categorical)
                }
            }
        """
        params = {}

        for param_name, config in param_space.items():
            param_type = config['type']

            if param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    low=config['low'],
                    high=config['high'],
                    step=config.get('step', 1),
                    log=config.get('log', False)
                )

            elif param_type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    low=config['low'],
                    high=config['high'],
                    step=config.get('step'),
                    log=config.get('log', False)
                )

            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    choices=config['choices']
                )

            else:
                raise ValueError(
                    f"不支援的參數類型: {param_type}，"
                    f"有效類型為 'int', 'float', 'categorical'"
                )

        return params

    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """取得最佳參數"""
        if self._study is None:
            return None
        return self._study.best_params

    def get_best_value(self) -> Optional[float]:
        """取得最佳目標值"""
        if self._study is None:
            return None
        return self._study.best_value

    def get_optimization_history(self) -> Optional[pd.DataFrame]:
        """
        取得優化歷史

        Returns:
            包含所有試驗資訊的 DataFrame
        """
        if self._study is None:
            return None

        trials_df = self._study.trials_dataframe()
        return trials_df

    def get_param_importances(self) -> Optional[Dict[str, float]]:
        """
        計算參數重要性

        Returns:
            {param_name: importance_score}
        """
        if self._study is None or len(self._study.trials) < 10:
            return None

        try:
            from optuna.importance import get_param_importances
            importances = get_param_importances(self._study)
            return importances
        except Exception as e:
            warnings.warn(f"無法計算參數重要性: {e}")
            return None

    def save_study(self, filepath: str):
        """
        儲存 study 到檔案

        Args:
            filepath: 儲存路徑 (joblib 格式)

        Note:
            建議使用 Optuna 的內建儲存後端（如 SQLite）
            而非檔案序列化，以獲得更好的可靠性：
            optimizer.optimize(..., storage='sqlite:///optuna.db')
        """
        if self._study is None:
            raise ValueError("尚未執行優化，無法儲存")

        try:
            import joblib
            joblib.dump(self._study, filepath)
        except ImportError:
            warnings.warn(
                "需要安裝 joblib: pip install joblib\n"
                "建議使用 Optuna 的 storage 參數（如 'sqlite:///optuna.db'）"
            )
            raise

    @classmethod
    def load_study(cls, filepath: str) -> optuna.Study:
        """
        從檔案載入 study

        Args:
            filepath: 檔案路徑

        Returns:
            Optuna Study 物件

        Note:
            建議使用 Optuna 的內建儲存後端（如 SQLite）
            而非檔案序列化，以獲得更好的可靠性。
        """
        try:
            import joblib
            study = joblib.load(filepath)
            return study
        except ImportError:
            warnings.warn(
                "需要安裝 joblib: pip install joblib\n"
                "建議使用 Optuna 的 storage 參數（如 'sqlite:///optuna.db'）"
            )
            raise


# 便利函數

def optimize_strategy(
    strategy: Any,
    data: pd.DataFrame,
    engine: BacktestEngine,
    n_trials: int = 100,
    metric: str = 'sharpe_ratio',
    n_jobs: int = 1,
    verbose: bool = True
) -> OptimizationResult:
    """
    便利函數：快速優化策略

    Args:
        strategy: 策略物件
        data: 市場資料
        engine: BacktestEngine 實例
        n_trials: 試驗次數
        metric: 優化目標指標
        n_jobs: 並行工作數
        verbose: 是否顯示進度

    Returns:
        OptimizationResult 物件
    """
    optimizer = BayesianOptimizer(
        engine=engine,
        n_trials=n_trials,
        n_jobs=n_jobs,
        verbose=verbose
    )

    result = optimizer.optimize(
        strategy=strategy,
        data=data,
        metric=metric
    )

    return result
