"""
多目標優化器

使用 NSGA-II 演算法進行多目標策略優化。
支援多個目標函數（Sharpe、Sortino、Max Drawdown、Win Rate）的同時優化。
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
import warnings
import logging
from datetime import datetime

import pandas as pd
import numpy as np

OPTUNA_AVAILABLE = False
try:
    import optuna
    from optuna.trial import Trial
    from optuna.samplers import NSGAIISampler
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None  # type: ignore[assignment]
    Trial = None  # type: ignore[assignment,misc]
    NSGAIISampler = None  # type: ignore[assignment,misc]

from ..backtester.engine import BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class ObjectiveResult:
    """單一目標結果"""
    name: str
    value: float
    direction: Literal['maximize', 'minimize']

    def __repr__(self) -> str:
        return f"{self.name}={self.value:.4f} ({self.direction})"


@dataclass
class ParetoSolution:
    """Pareto 最優解"""
    params: Dict[str, Any]
    objectives: List[ObjectiveResult]
    rank: int = 0
    crowding_distance: float = 0.0
    trial_number: int = 0

    def get_objective_value(self, name: str) -> Optional[float]:
        """取得指定目標的值"""
        for obj in self.objectives:
            if obj.name == name:
                return obj.value
        return None

    def to_dict(self) -> Dict[str, Any]:
        """轉為字典"""
        return {
            'trial_number': self.trial_number,
            'params': self.params,
            'objectives': {obj.name: obj.value for obj in self.objectives},
            'rank': self.rank,
            'crowding_distance': self.crowding_distance
        }

    def __repr__(self) -> str:
        obj_str = ", ".join([str(obj) for obj in self.objectives])
        return f"ParetoSolution(rank={self.rank}, {obj_str})"


@dataclass
class MultiObjectiveResult:
    """多目標優化結果"""
    pareto_front: List[ParetoSolution]
    all_solutions: List[ParetoSolution]
    n_trials: int
    study: 'optuna.Study'
    optimization_time: float = 0.0
    n_completed_trials: int = 0
    n_failed_trials: int = 0

    def get_best_solution(
        self,
        weights: Optional[Dict[str, float]] = None
    ) -> Optional[ParetoSolution]:
        """
        從 Pareto 前緣選擇最佳解

        Args:
            weights: 各目標權重（預設均等），格式 {'sharpe': 0.5, 'max_dd': 0.5}

        Returns:
            加權分數最高的解（考慮目標方向）
        """
        if not self.pareto_front:
            return None

        if weights is None:
            # 均等權重
            weights = {
                obj.name: 1.0 / len(self.pareto_front[0].objectives)
                for obj in self.pareto_front[0].objectives
            }

        # 標準化並加權
        best_solution = None
        best_score = float('-inf')

        # 計算每個目標的範圍（用於標準化）
        obj_ranges = {}
        for obj in self.pareto_front[0].objectives:
            values = [s.get_objective_value(obj.name) for s in self.pareto_front]
            values = [v for v in values if v is not None]
            if values:
                obj_ranges[obj.name] = {
                    'min': min(values),
                    'max': max(values),
                    'direction': obj.direction
                }

        for solution in self.pareto_front:
            score = 0.0
            for obj in solution.objectives:
                weight = weights.get(obj.name, 0.0)
                obj_range = obj_ranges.get(obj.name)

                if obj_range:
                    value_range = obj_range['max'] - obj_range['min']
                    if value_range > 0:
                        # 標準化到 [0, 1]
                        normalized = (obj.value - obj_range['min']) / value_range

                        # 根據方向調整
                        if obj.direction == 'minimize':
                            normalized = 1.0 - normalized

                        score += weight * normalized

            if score > best_score:
                best_score = score
                best_solution = solution

        return best_solution

    def summary(self) -> str:
        """產生摘要報告"""
        lines = [
            "多目標優化結果摘要",
            "=" * 60,
            f"Pareto 前緣解數量: {len(self.pareto_front)}",
            f"總試驗次數: {self.n_trials}",
            f"完成: {self.n_completed_trials}",
            f"失敗: {self.n_failed_trials}",
            f"優化時間: {self.optimization_time:.2f} 秒",
            "",
            "Pareto 前緣（前 10 個解）:",
            "-" * 60
        ]

        for i, solution in enumerate(self.pareto_front[:10], 1):
            lines.append(f"\n解 {i} (Rank={solution.rank}, CD={solution.crowding_distance:.4f}):")
            for obj in solution.objectives:
                lines.append(f"  {obj.name}: {obj.value:.4f} ({obj.direction})")
            lines.append(f"  參數: {solution.params}")

        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """轉為 DataFrame（Pareto 前緣）"""
        records = []
        for solution in self.pareto_front:
            record = {
                'trial_number': solution.trial_number,
                'rank': solution.rank,
                'crowding_distance': solution.crowding_distance,
                **solution.params,
                **{obj.name: obj.value for obj in solution.objectives}
            }
            records.append(record)
        return pd.DataFrame(records)

    def plot_pareto_front_2d(
        self,
        obj_x: str,
        obj_y: str,
        save_path: Optional[str] = None
    ):
        """
        繪製 2D Pareto 前緣

        Args:
            obj_x: X 軸目標名稱
            obj_y: Y 軸目標名稱
            save_path: 儲存路徑（HTML）
        """
        try:
            import plotly.graph_objects as go

            x_values = [s.get_objective_value(obj_x) for s in self.pareto_front]
            y_values = [s.get_objective_value(obj_y) for s in self.pareto_front]

            fig = go.Figure(data=go.Scatter(
                x=x_values,
                y=y_values,
                mode='markers',
                marker=dict(
                    size=10,
                    color=list(range(len(self.pareto_front))),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="解編號")
                ),
                text=[f"Trial {s.trial_number}" for s in self.pareto_front],
                hovertemplate='<b>%{text}</b><br>' +
                              f'{obj_x}: %{{x:.4f}}<br>' +
                              f'{obj_y}: %{{y:.4f}}<extra></extra>'
            ))

            fig.update_layout(
                title='Pareto Front (2D)',
                xaxis_title=obj_x,
                yaxis_title=obj_y,
                hovermode='closest'
            )

            if save_path:
                fig.write_html(save_path)
            return fig

        except ImportError:
            warnings.warn("需要安裝 plotly: pip install plotly")
            return None


class MultiObjectiveOptimizer:
    """
    多目標優化器（NSGA-II）

    使用 Optuna 的 NSGA-II 演算法進行多目標優化。
    支援同時優化多個目標（如 Sharpe Ratio、Max Drawdown 等）。

    使用範例：
        optimizer = MultiObjectiveOptimizer(
            objectives=[
                ('sharpe_ratio', 'maximize'),
                ('max_drawdown', 'minimize'),
                ('sortino_ratio', 'maximize')
            ],
            n_trials=200,
            seed=42
        )

        result = optimizer.optimize(
            param_space=param_space,
            evaluate_fn=evaluate_function
        )

        # 取得最佳平衡解
        best = result.get_best_solution(
            weights={'sharpe_ratio': 0.4, 'max_drawdown': 0.3, 'sortino_ratio': 0.3}
        )

        print(result.summary())
        result.plot_pareto_front_2d('sharpe_ratio', 'max_drawdown', 'pareto.html')
    """

    def __init__(
        self,
        objectives: List[Tuple[str, Literal['maximize', 'minimize']]],
        n_trials: int = 100,
        seed: Optional[int] = None,
        verbose: bool = True,
        population_size: Optional[int] = None,
        mutation_prob: Optional[float] = None,
        crossover_prob: Optional[float] = None
    ):
        """
        初始化多目標優化器

        Args:
            objectives: 目標函數列表，格式 [('sharpe', 'maximize'), ('max_dd', 'minimize')]
            n_trials: 優化試驗次數（預設 100）
            seed: 隨機種子（可重現性）
            verbose: 是否顯示進度
            population_size: NSGA-II 族群大小（預設自動）
            mutation_prob: 突變機率（預設 None，自動計算）
            crossover_prob: 交叉機率（預設 0.9）
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna 未安裝。請執行: pip install optuna"
            )

        if not objectives:
            raise ValueError("至少需要提供一個目標函數")

        self.objectives = objectives
        self.n_trials = n_trials
        self.seed = seed
        self.verbose = verbose

        # NSGA-II 參數
        self.population_size = population_size
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob

        # 內部狀態
        self._study: Optional[Any] = None  # optuna.Study when available
        self._optimization_start_time: Optional[datetime] = None

    def optimize(
        self,
        param_space: Dict[str, Dict],
        evaluate_fn: Callable[[Dict], Dict[str, float]],
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        show_progress_bar: bool = True
    ) -> MultiObjectiveResult:
        """
        執行多目標優化

        Args:
            param_space: 參數空間定義（與 BayesianOptimizer 相同格式）
            evaluate_fn: 評估函數，接收參數字典，回傳目標值字典
                        例如: {'sharpe_ratio': 1.5, 'max_drawdown': 0.15}
            study_name: 研究名稱（用於儲存/載入）
            storage: 儲存後端 URL（如 'sqlite:///optuna.db'）
            show_progress_bar: 是否顯示進度條

        Returns:
            MultiObjectiveResult 物件

        參數空間格式範例:
            {
                'fast_period': {'type': 'int', 'low': 5, 'high': 20},
                'slow_period': {'type': 'int', 'low': 20, 'high': 50},
                'stop_loss_atr': {'type': 'float', 'low': 1.0, 'high': 3.0, 'step': 0.1}
            }
        """
        if not param_space:
            raise ValueError("必須提供參數空間定義")

        if not OPTUNA_AVAILABLE or optuna is None or NSGAIISampler is None:
            raise ImportError("Optuna is required for multi-objective optimization. Install with: pip install optuna")

        # 建立 NSGA-II sampler
        sampler_kwargs: Dict[str, Any] = {'seed': self.seed}
        if self.population_size is not None:
            sampler_kwargs['population_size'] = self.population_size
        if self.mutation_prob is not None:
            sampler_kwargs['mutation_prob'] = self.mutation_prob
        if self.crossover_prob is not None:
            sampler_kwargs['crossover_prob'] = self.crossover_prob

        sampler = NSGAIISampler(**sampler_kwargs)

        # 建立 study（多目標）
        directions = [obj[1] for obj in self.objectives]
        self._study = optuna.create_study(
            directions=directions,
            sampler=sampler,
            study_name=study_name,
            storage=storage,
            load_if_exists=True
        )

        # 目標函數
        def objective(trial: Trial) -> Tuple[float, ...]:
            return self._objective(
                trial=trial,
                param_space=param_space,
                evaluate_fn=evaluate_fn
            )

        # 執行優化
        self._optimization_start_time = datetime.now()

        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        self._study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=show_progress_bar
        )

        optimization_time = (
            datetime.now() - self._optimization_start_time
        ).total_seconds()

        # 收集結果
        pareto_front = self.get_pareto_front()
        all_solutions = self._extract_all_solutions()

        # 統計資訊
        n_completed = len([
            t for t in self._study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ])
        n_failed = len([
            t for t in self._study.trials
            if t.state == optuna.trial.TrialState.FAIL
        ])

        return MultiObjectiveResult(
            pareto_front=pareto_front,
            all_solutions=all_solutions,
            n_trials=self.n_trials,
            study=self._study,
            optimization_time=optimization_time,
            n_completed_trials=n_completed,
            n_failed_trials=n_failed
        )

    def _objective(
        self,
        trial: Trial,
        param_space: Dict[str, Dict],
        evaluate_fn: Callable[[Dict], Dict[str, float]]
    ) -> Tuple[float, ...]:
        """
        Optuna 目標函數

        Args:
            trial: Optuna Trial 物件
            param_space: 參數空間定義
            evaluate_fn: 評估函數

        Returns:
            各目標值的 tuple
        """
        # 從 trial 採樣參數
        params = self._sample_params(trial, param_space)

        try:
            # 執行評估函數
            results = evaluate_fn(params)

            # 驗證結果
            if not isinstance(results, dict):
                raise ValueError(f"evaluate_fn 必須回傳 dict，而非 {type(results)}")

            # 提取各目標值
            objective_values = []
            for obj_name, obj_direction in self.objectives:
                if obj_name not in results:
                    raise ValueError(f"evaluate_fn 結果缺少目標 '{obj_name}'")

                value = results[obj_name]
                if not np.isfinite(value):
                    raise optuna.TrialPruned(f"Invalid value for {obj_name}: {value}")

                objective_values.append(value)

            return tuple(objective_values)

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
        從參數空間採樣參數（與 BayesianOptimizer 相同）

        Args:
            trial: Optuna Trial 物件
            param_space: 參數空間定義

        Returns:
            採樣的參數字典
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

    def get_pareto_front(self) -> List[ParetoSolution]:
        """
        取得 Pareto 前緣

        Returns:
            Pareto 最優解列表（已排序並計算擁擠度距離）
        """
        if self._study is None:
            return []

        # 取得 Pareto 前緣的試驗
        pareto_trials = [
            t for t in self._study.best_trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if not pareto_trials:
            return []

        # 轉為 ParetoSolution
        solutions = []
        for trial in pareto_trials:
            objectives = [
                ObjectiveResult(
                    name=obj_name,
                    value=trial.values[i],
                    direction=obj_direction
                )
                for i, (obj_name, obj_direction) in enumerate(self.objectives)
            ]

            solution = ParetoSolution(
                params=trial.params,
                objectives=objectives,
                rank=0,  # Pareto 前緣的 rank 都是 0
                trial_number=trial.number
            )
            solutions.append(solution)

        # 計算擁擠度距離
        solutions = self._calculate_crowding_distance(solutions)

        # 按擁擠度距離排序（由大到小）
        solutions.sort(key=lambda s: s.crowding_distance, reverse=True)

        return solutions

    def _calculate_crowding_distance(
        self,
        solutions: List[ParetoSolution]
    ) -> List[ParetoSolution]:
        """
        計算擁擠度距離（Crowding Distance）

        擁擠度距離衡量解在目標空間中的分散程度。
        邊界解（每個目標的最大/最小值）被賦予無限大距離。

        Args:
            solutions: ParetoSolution 列表

        Returns:
            更新 crowding_distance 後的列表
        """
        if len(solutions) <= 2:
            # 少於 3 個解，全部設為無限大
            for solution in solutions:
                solution.crowding_distance = float('inf')
            return solutions

        n_objectives = len(self.objectives)
        n_solutions = len(solutions)

        # 初始化距離為 0
        for solution in solutions:
            solution.crowding_distance = 0.0

        # 對每個目標計算距離
        for obj_idx in range(n_objectives):
            # 按此目標排序
            solutions.sort(key=lambda s: s.objectives[obj_idx].value)

            # 邊界解設為無限大
            solutions[0].crowding_distance = float('inf')
            solutions[-1].crowding_distance = float('inf')

            # 計算目標值範圍
            obj_min = solutions[0].objectives[obj_idx].value
            obj_max = solutions[-1].objectives[obj_idx].value
            obj_range = obj_max - obj_min

            if obj_range == 0:
                continue  # 避免除以零

            # 計算中間解的距離
            for i in range(1, n_solutions - 1):
                if solutions[i].crowding_distance == float('inf'):
                    continue

                prev_value = solutions[i - 1].objectives[obj_idx].value
                next_value = solutions[i + 1].objectives[obj_idx].value

                solutions[i].crowding_distance += (next_value - prev_value) / obj_range

        return solutions

    def _extract_all_solutions(self) -> List[ParetoSolution]:
        """提取所有完成的試驗為 ParetoSolution"""
        if self._study is None:
            return []

        solutions = []
        for trial in self._study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue

            objectives = [
                ObjectiveResult(
                    name=obj_name,
                    value=trial.values[i],
                    direction=obj_direction
                )
                for i, (obj_name, obj_direction) in enumerate(self.objectives)
            ]

            solution = ParetoSolution(
                params=trial.params,
                objectives=objectives,
                trial_number=trial.number
            )
            solutions.append(solution)

        return solutions

    def get_optimization_history(self) -> Optional[pd.DataFrame]:
        """
        取得優化歷史

        Returns:
            包含所有試驗資訊的 DataFrame
        """
        if self._study is None:
            return None

        return self._study.trials_dataframe()


# 便利函數

def optimize_multi_objective(
    param_space: Dict[str, Dict],
    evaluate_fn: Callable[[Dict], Dict[str, float]],
    objectives: List[Tuple[str, Literal['maximize', 'minimize']]],
    n_trials: int = 100,
    seed: Optional[int] = None,
    verbose: bool = True
) -> MultiObjectiveResult:
    """
    便利函數：快速多目標優化

    Args:
        param_space: 參數空間定義
        evaluate_fn: 評估函數
        objectives: 目標列表
        n_trials: 試驗次數
        seed: 隨機種子
        verbose: 是否顯示進度

    Returns:
        MultiObjectiveResult 物件

    範例:
        result = optimize_multi_objective(
            param_space={
                'param1': {'type': 'int', 'low': 1, 'high': 10},
                'param2': {'type': 'float', 'low': 0.1, 'high': 1.0}
            },
            evaluate_fn=lambda params: {
                'sharpe': calculate_sharpe(params),
                'max_dd': calculate_max_dd(params)
            },
            objectives=[('sharpe', 'maximize'), ('max_dd', 'minimize')],
            n_trials=200
        )
    """
    optimizer = MultiObjectiveOptimizer(
        objectives=objectives,
        n_trials=n_trials,
        seed=seed,
        verbose=verbose
    )

    return optimizer.optimize(
        param_space=param_space,
        evaluate_fn=evaluate_fn
    )
