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

# Optional regime detection import
REGIME_AVAILABLE = False
MarketStateAnalyzer: Any = None

def _load_regime_module() -> bool:
    """Lazy load regime module to avoid circular imports"""
    global REGIME_AVAILABLE, MarketStateAnalyzer
    if not REGIME_AVAILABLE:
        try:
            from ..regime import MarketStateAnalyzer as _MSA
            MarketStateAnalyzer = _MSA
            REGIME_AVAILABLE = True
        except ImportError:
            pass
    return REGIME_AVAILABLE

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
    study: Any  # optuna.Study when available
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

    def plot_pareto_front_3d(
        self,
        obj_x: str,
        obj_y: str,
        obj_z: str,
        save_path: Optional[str] = None
    ):
        """
        繪製 3D Pareto 前緣

        Args:
            obj_x: X 軸目標名稱
            obj_y: Y 軸目標名稱
            obj_z: Z 軸目標名稱
            save_path: 儲存路徑（HTML）

        Returns:
            plotly Figure 物件（如果 plotly 可用）
        """
        try:
            import plotly.graph_objects as go

            x_values = [s.get_objective_value(obj_x) for s in self.pareto_front]
            y_values = [s.get_objective_value(obj_y) for s in self.pareto_front]
            z_values = [s.get_objective_value(obj_z) for s in self.pareto_front]

            fig = go.Figure(data=[go.Scatter3d(
                x=x_values,
                y=y_values,
                z=z_values,
                mode='markers',
                marker=dict(
                    size=6,
                    color=list(range(len(self.pareto_front))),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="解編號")
                ),
                text=[f"Trial {s.trial_number}" for s in self.pareto_front],
                hovertemplate='<b>%{text}</b><br>' +
                              f'{obj_x}: %{{x:.4f}}<br>' +
                              f'{obj_y}: %{{y:.4f}}<br>' +
                              f'{obj_z}: %{{z:.4f}}<extra></extra>'
            )])

            fig.update_layout(
                title='Pareto Front (3D)',
                scene=dict(
                    xaxis_title=obj_x,
                    yaxis_title=obj_y,
                    zaxis_title=obj_z
                ),
                hovermode='closest'
            )

            if save_path:
                fig.write_html(save_path)
            return fig

        except ImportError:
            warnings.warn("需要安裝 plotly: pip install plotly")
            return None

    def plot_parallel_coordinates(
        self,
        save_path: Optional[str] = None,
        max_params: int = 10
    ):
        """
        繪製平行座標圖，顯示所有目標和參數的關係

        Args:
            save_path: 儲存路徑（HTML）
            max_params: 最多顯示的參數數量（避免過於擁擠）

        Returns:
            plotly Figure 物件（如果 plotly 可用）
        """
        try:
            import plotly.graph_objects as go

            if not self.pareto_front:
                warnings.warn("Pareto 前沿為空，無法繪製")
                return None

            # 收集所有維度資料
            dimensions = []

            # 1. 目標維度
            for obj in self.pareto_front[0].objectives:
                values = [s.get_objective_value(obj.name) for s in self.pareto_front]
                # Filter out None values for min/max
                valid_values = [v for v in values if v is not None]
                if not valid_values:
                    continue
                dimensions.append(dict(
                    label=f"{obj.name}",
                    values=values,
                    range=[min(valid_values), max(valid_values)]
                ))

            # 2. 參數維度（限制數量）
            param_names = list(self.pareto_front[0].params.keys())[:max_params]
            for param_name in param_names:
                values = [s.params[param_name] for s in self.pareto_front]
                dimensions.append(dict(
                    label=param_name,
                    values=values,
                    range=[min(values), max(values)]
                ))

            # 建立平行座標圖
            fig = go.Figure(data=go.Parcoords(
                line=dict(
                    color=list(range(len(self.pareto_front))),
                    colorscale='Viridis',
                    showscale=True,
                    cmin=0,
                    cmax=len(self.pareto_front) - 1,
                    colorbar=dict(title="解編號")
                ),
                dimensions=dimensions
            ))

            fig.update_layout(
                title='Pareto Front - Parallel Coordinates',
                height=600
            )

            if save_path:
                fig.write_html(save_path)
            return fig

        except ImportError:
            warnings.warn("需要安裝 plotly: pip install plotly")
            return None

    def find_knee_point(self) -> Optional[ParetoSolution]:
        """
        找出 Pareto 前沿的膝點（最佳平衡點）

        使用幾何方法：找到離理想點和最差點連線最遠的點。
        這個點通常代表目標之間的最佳 trade-off。

        Returns:
            膝點解，如果 Pareto 前沿為空或只有一個點則返回 None

        Note:
            此方法假設所有目標已經標準化到相似的尺度。
            對於目標尺度差異很大的情況，建議先進行標準化。
        """
        if len(self.pareto_front) <= 1:
            return None

        # 標準化所有目標到 [0, 1]
        n_objectives = len(self.pareto_front[0].objectives)
        obj_ranges = {}

        for obj_idx in range(n_objectives):
            obj_name = self.pareto_front[0].objectives[obj_idx].name
            values = [s.objectives[obj_idx].value for s in self.pareto_front]
            obj_ranges[obj_name] = {
                'min': min(values),
                'max': max(values)
            }

        # 標準化並考慮目標方向
        normalized_solutions = []
        for solution in self.pareto_front:
            normalized_values = []
            for obj in solution.objectives:
                obj_range = obj_ranges[obj.name]
                value_range = obj_range['max'] - obj_range['min']

                if value_range > 0:
                    normalized = (obj.value - obj_range['min']) / value_range
                else:
                    normalized = 0.5  # 所有值相同時置中

                # maximize 目標：值越大越好 → 保持 [0, 1]
                # minimize 目標：值越小越好 → 反轉為 [1, 0]
                if obj.direction == 'minimize':
                    normalized = 1.0 - normalized

                normalized_values.append(normalized)

            normalized_solutions.append(normalized_values)

        # 找到理想點（所有目標的最大值）和最差點（所有目標的最小值）
        ideal_point = [max(vals[i] for vals in normalized_solutions) for i in range(n_objectives)]
        nadir_point = [min(vals[i] for vals in normalized_solutions) for i in range(n_objectives)]

        # 計算每個解到理想-最差連線的距離
        max_distance = -1.0
        knee_idx = 0

        for idx, norm_values in enumerate(normalized_solutions):
            # 計算點到線段的距離（使用叉積）
            distance = self._point_to_line_distance(
                point=np.array(norm_values),
                line_start=np.array(nadir_point),
                line_end=np.array(ideal_point)
            )

            if distance > max_distance:
                max_distance = distance
                knee_idx = idx

        return self.pareto_front[knee_idx]

    @staticmethod
    def _point_to_line_distance(
        point: np.ndarray,
        line_start: np.ndarray,
        line_end: np.ndarray
    ) -> float:
        """
        計算點到線段的距離

        Args:
            point: 點座標
            line_start: 線段起點
            line_end: 線段終點

        Returns:
            距離值
        """
        # 線段向量
        line_vec = line_end - line_start
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-10:
            # 線段退化為點
            return float(np.linalg.norm(point - line_start))

        # 點到線段起點的向量
        point_vec = point - line_start

        # 投影長度
        t = np.dot(point_vec, line_vec) / (line_len ** 2)
        t = np.clip(t, 0.0, 1.0)  # 限制在線段範圍內

        # 最近點
        closest_point = line_start + t * line_vec

        # 距離
        return float(np.linalg.norm(point - closest_point))

    def filter_pareto_front(
        self,
        method: Literal['crowding', 'knee', 'extreme', 'uniform'] = 'crowding',
        n_select: int = 10
    ) -> List[ParetoSolution]:
        """
        從 Pareto 前沿篩選代表性解

        Args:
            method: 篩選方法
                - 'crowding': 依擁擠度選擇（多樣性優先）
                - 'knee': 選擇膝點附近的解
                - 'extreme': 選擇各目標的極值解
                - 'uniform': 均勻選擇（沿 Pareto 前沿）
            n_select: 選擇數量

        Returns:
            篩選後的解列表

        Examples:
            >>> # 選擇最多樣化的 10 個解
            >>> diverse_solutions = result.filter_pareto_front('crowding', 10)
            >>>
            >>> # 選擇膝點附近的解
            >>> balanced_solutions = result.filter_pareto_front('knee', 5)
            >>>
            >>> # 選擇極值解（適合展示 trade-off）
            >>> extreme_solutions = result.filter_pareto_front('extreme', 6)
        """
        if not self.pareto_front:
            return []

        if n_select >= len(self.pareto_front):
            return self.pareto_front.copy()

        if method == 'crowding':
            # 已按擁擠度排序，直接取前 n 個
            return self.pareto_front[:n_select]

        elif method == 'knee':
            # 找到膝點
            knee_solution = self.find_knee_point()
            if knee_solution is None:
                return self.pareto_front[:n_select]

            # 計算所有解到膝點的距離（在標準化空間）
            knee_idx = self.pareto_front.index(knee_solution)
            distances = []

            for idx, solution in enumerate(self.pareto_front):
                # 簡化：使用目標值的歐式距離
                dist = sum(
                    (s_obj.value - k_obj.value) ** 2
                    for s_obj, k_obj in zip(solution.objectives, knee_solution.objectives)
                ) ** 0.5
                distances.append((idx, dist))

            # 選擇距離膝點最近的 n 個解
            distances.sort(key=lambda x: x[1])
            selected_indices = [idx for idx, _ in distances[:n_select]]
            return [self.pareto_front[idx] for idx in selected_indices]

        elif method == 'extreme':
            # 選擇每個目標的極值解
            selected = []
            n_objectives = len(self.pareto_front[0].objectives)

            for obj_idx in range(n_objectives):
                obj_name = self.pareto_front[0].objectives[obj_idx].name
                obj_direction = self.pareto_front[0].objectives[obj_idx].direction

                # 按該目標排序（處理 Optional[float]）
                sorted_solutions = sorted(
                    self.pareto_front,
                    key=lambda s: s.get_objective_value(obj_name) or 0.0,
                    reverse=(obj_direction == 'maximize')
                )

                # 添加最佳解
                if sorted_solutions[0] not in selected:
                    selected.append(sorted_solutions[0])

                # 添加最差解（展示 trade-off）
                if len(selected) < n_select and sorted_solutions[-1] not in selected:
                    selected.append(sorted_solutions[-1])

            # 如果還需要更多解，用擁擠度填充
            if len(selected) < n_select:
                for solution in self.pareto_front:
                    if solution not in selected:
                        selected.append(solution)
                        if len(selected) >= n_select:
                            break

            return selected[:n_select]

        elif method == 'uniform':
            # 均勻選擇（基於擁擠度排序後的索引）
            # 若 n_select > 可用解數，返回所有可用解
            if n_select >= len(self.pareto_front):
                return self.pareto_front.copy()

            # 使用 numpy linspace 確保均勻分佈且無重複
            import numpy as np
            indices = np.linspace(0, len(self.pareto_front) - 1, n_select).astype(int)
            # 去除可能的重複索引（當 pareto_front 很小時）
            indices = sorted(set(indices))
            return [self.pareto_front[idx] for idx in indices]

        else:
            raise ValueError(
                f"不支援的篩選方法: {method}，"
                f"有效方法為 'crowding', 'knee', 'extreme', 'uniform'"
            )


class MultiObjectiveOptimizer:
    """
    多目標優化器（NSGA-II）

    使用 Optuna 的 NSGA-II 演算法進行多目標優化。
    支援同時優化多個目標（如 Sharpe Ratio、Max Drawdown 等）。

    功能特性：
        - 多目標優化（NSGA-II 演算法）
        - 約束條件支援
        - 暖啟動（warm start）
        - Market regime 感知（可選）
        - 膝點檢測（knee point detection）
        - Pareto 前沿篩選（多種方法）
        - 豐富的視覺化（2D、3D、平行座標）

    基本使用範例：
        >>> optimizer = MultiObjectiveOptimizer(
        ...     objectives=[
        ...         ('sharpe_ratio', 'maximize'),
        ...         ('max_drawdown', 'minimize'),
        ...         ('sortino_ratio', 'maximize')
        ...     ],
        ...     n_trials=200,
        ...     seed=42
        ... )
        >>>
        >>> result = optimizer.optimize(
        ...     param_space=param_space,
        ...     evaluate_fn=evaluate_function
        ... )
        >>>
        >>> # 取得最佳平衡解
        >>> best = result.get_best_solution(
        ...     weights={'sharpe_ratio': 0.4, 'max_drawdown': 0.3, 'sortino_ratio': 0.3}
        ... )

    進階功能範例：
        >>> # 1. 約束條件
        >>> def leverage_constraint(params):
        ...     # 槓桿不得超過 10x
        ...     return max(0, params.get('leverage', 1) - 10)
        >>>
        >>> optimizer = MultiObjectiveOptimizer(
        ...     objectives=[('sharpe', 'maximize'), ('max_dd', 'minimize')],
        ...     constraints=[leverage_constraint],
        ...     n_trials=100
        ... )
        >>>
        >>> # 2. 暖啟動
        >>> optimizer.warm_start([
        ...     {'fast_period': 10, 'slow_period': 30},
        ...     {'fast_period': 12, 'slow_period': 26}
        ... ])
        >>> result = optimizer.optimize(param_space, evaluate_fn)
        >>>
        >>> # 3. 膝點檢測
        >>> knee = result.find_knee_point()
        >>> print(f"最佳平衡解: {knee}")
        >>>
        >>> # 4. Pareto 前沿篩選
        >>> diverse_solutions = result.filter_pareto_front('crowding', n_select=10)
        >>> balanced_solutions = result.filter_pareto_front('knee', n_select=5)
        >>> extreme_solutions = result.filter_pareto_front('extreme', n_select=6)
        >>>
        >>> # 5. 視覺化
        >>> result.plot_pareto_front_2d('sharpe', 'max_dd', 'pareto_2d.html')
        >>> result.plot_pareto_front_3d('sharpe', 'max_dd', 'win_rate', 'pareto_3d.html')
        >>> result.plot_parallel_coordinates('parallel.html')
    """

    def __init__(
        self,
        objectives: List[Tuple[str, Literal['maximize', 'minimize']]],
        n_trials: int = 100,
        seed: Optional[int] = None,
        verbose: bool = True,
        population_size: Optional[int] = None,
        mutation_prob: Optional[float] = None,
        crossover_prob: Optional[float] = None,
        constraints: Optional[List[Callable[[Dict], float]]] = None,
        regime_aware: bool = False,
        regime_analyzer: Optional[Any] = None  # MarketStateAnalyzer when regime module available
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
            constraints: 約束條件列表，每個函數接收參數字典，
                        回傳正值表示違反約束（值越大違反越嚴重），
                        0 或負值表示滿足約束
            regime_aware: 是否啟用市場狀態感知（需要提供 regime_analyzer）
            regime_analyzer: 市場狀態分析器（用於 regime-aware 優化）
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna 未安裝。請執行: pip install optuna"
            )

        if not objectives:
            raise ValueError("至少需要提供一個目標函數")

        if regime_aware and not REGIME_AVAILABLE:
            raise ImportError(
                "Regime-aware 優化需要 regime_detection 模組。"
            )

        if regime_aware and regime_analyzer is None:
            raise ValueError("regime_aware=True 時必須提供 regime_analyzer")

        self.objectives = objectives
        self.n_trials = n_trials
        self.seed = seed
        self.verbose = verbose

        # NSGA-II 參數
        self.population_size = population_size
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob

        # 約束條件
        self.constraints = constraints or []

        # Regime 感知
        self.regime_aware = regime_aware
        self.regime_analyzer = regime_analyzer

        # 內部狀態
        self._study: Optional[Any] = None  # optuna.Study when available
        self._optimization_start_time: Optional[datetime] = None
        self._initial_solutions: List[Dict[str, Any]] = []

    def warm_start(self, initial_solutions: List[Dict[str, Any]]) -> None:
        """
        用已知的優良解初始化搜索（暖啟動）

        Args:
            initial_solutions: 初始解列表，每個解為參數字典

        Examples:
            >>> # 使用歷史最佳參數作為起點
            >>> optimizer.warm_start([
            ...     {'fast_period': 10, 'slow_period': 30},
            ...     {'fast_period': 12, 'slow_period': 26}
            ... ])
            >>> result = optimizer.optimize(param_space, evaluate_fn)

        Note:
            暖啟動可以加速收斂，但可能導致過早收斂到局部最優。
            建議與足夠的 n_trials 結合使用，以保持探索性。
        """
        self._initial_solutions = initial_solutions.copy()
        logger.info(f"暖啟動：載入 {len(initial_solutions)} 個初始解")

    def _check_constraints(self, params: Dict) -> Tuple[bool, float]:
        """
        檢查參數是否滿足所有約束條件

        Args:
            params: 參數字典

        Returns:
            (是否滿足約束, 違反程度總和)
            - 滿足所有約束時返回 (True, 0.0)
            - 違反約束時返回 (False, violation_sum)
        """
        if not self.constraints:
            return True, 0.0

        total_violation = 0.0

        for constraint in self.constraints:
            try:
                violation = constraint(params)
                if violation > 0:
                    total_violation += violation
            except Exception as e:
                logger.warning(f"約束檢查失敗: {e}")
                # 約束檢查失敗視為違反
                return False, float('inf')

        is_satisfied = total_violation <= 0
        return is_satisfied, max(0.0, total_violation)

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
        def objective(trial: Any) -> Tuple[float, ...]:
            return self._objective(
                trial=trial,
                param_space=param_space,
                evaluate_fn=evaluate_fn
            )

        # 暖啟動：先評估初始解
        if self._initial_solutions:
            logger.info(f"暖啟動：評估 {len(self._initial_solutions)} 個初始解")
            for init_params in self._initial_solutions:
                try:
                    # 檢查約束條件
                    is_satisfied, violation = self._check_constraints(init_params)
                    if not is_satisfied:
                        logger.debug(f"暖啟動解違反約束（violation={violation:.4f}），跳過")
                        continue

                    trial = self._study.ask()
                    # 手動設定參數
                    for param_name, param_value in init_params.items():
                        trial.suggest_categorical(f"_fixed_{param_name}", [param_value])
                        trial.set_user_attr(param_name, param_value)

                    # 評估
                    result = evaluate_fn(init_params)
                    objective_values = tuple(result[obj[0]] for obj in self.objectives)

                    self._study.tell(trial, objective_values)
                except Exception as e:
                    logger.warning(f"暖啟動解評估失敗: {e}")

        # 執行優化
        self._optimization_start_time = datetime.now()

        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # 計算剩餘試驗次數（扣除暖啟動）
        remaining_trials = self.n_trials - len(self._initial_solutions)
        if remaining_trials > 0:
            self._study.optimize(
                objective,
                n_trials=remaining_trials,
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
        trial: Any,  # optuna.Trial when available
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

        if optuna is None:
            raise ImportError("Optuna is required")

        try:
            # 檢查約束條件
            is_satisfied, violation = self._check_constraints(params)
            if not is_satisfied:
                trial.set_user_attr('constraint_violation', violation)
                logger.debug(f"Trial {trial.number} 違反約束: {violation}")
                raise optuna.TrialPruned(f"Constraint violation: {violation}")

            # 執行評估函數
            results = evaluate_fn(params)

            # 驗證結果
            if not isinstance(results, dict):
                raise ValueError(f"evaluate_fn 必須回傳 dict，而非 {type(results)}")

            # Regime 感知調整（如果啟用）
            if self.regime_aware and self.regime_analyzer is not None:
                results = self._evaluate_with_regime(params, results)

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

    def _evaluate_with_regime(
        self,
        params: Dict[str, Any],
        results: Dict[str, float]
    ) -> Dict[str, float]:
        """
        根據市場狀態動態調整評估結果

        Args:
            params: 參數字典
            results: 原始評估結果

        Returns:
            調整後的結果字典

        Note:
            此方法可根據市場狀態（趨勢/震盪/高波動等）對目標值進行加權調整。
            例如：在震盪市場中提高 win_rate 的重要性，在趨勢市場中提高 sharpe 的重要性。
        """
        if self.regime_analyzer is None:
            return results

        # 目前簡化版本：只記錄 regime 資訊，不做調整
        # 未來可擴展為根據 regime 動態調整目標值權重

        # 取得當前市場狀態（需要從 regime_analyzer 獲取）
        # current_regime = self.regime_analyzer.get_current_regime()

        # 根據 regime 調整（範例）
        # if current_regime == 'trending':
        #     results['sharpe_ratio'] *= 1.1  # 趨勢市場提高 sharpe 重要性
        # elif current_regime == 'ranging':
        #     results['win_rate'] *= 1.1      # 震盪市場提高勝率重要性

        # 目前保持原值返回
        return results

    def _sample_params(
        self,
        trial: Any,  # optuna.Trial when available
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
        if optuna is None:
            return []
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
        if self._study is None or optuna is None:
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
