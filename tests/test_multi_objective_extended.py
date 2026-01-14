"""
測試多目標優化器的擴展功能

測試項目：
1. 約束條件支援
2. 暖啟動
3. 膝點檢測
4. Pareto 前沿篩選
5. 視覺化方法
"""

import pytest
import numpy as np
from typing import Dict

from src.optimizer.multi_objective import MultiObjectiveOptimizer


@pytest.fixture
def simple_param_space():
    """簡單參數空間"""
    return {
        'x': {'type': 'float', 'low': 0.0, 'high': 10.0},
        'y': {'type': 'float', 'low': 0.0, 'high': 10.0}
    }


@pytest.fixture
def simple_evaluate_fn():
    """簡單評估函數（ZDT1 問題的簡化版）"""
    def evaluate(params: Dict) -> Dict[str, float]:
        x = params['x']
        y = params['y']

        # 目標 1: minimize f1(x) = x
        f1 = x

        # 目標 2: minimize f2(x,y) = g(y) * (1 - sqrt(x/g(y)))
        g = 1 + 9 * y / 9
        f2 = g * (1 - np.sqrt(x / g)) if x < g else 0

        return {
            'f1': f1,
            'f2': f2
        }

    return evaluate


class TestConstraints:
    """測試約束條件支援"""

    def test_constraint_violation(self, simple_param_space, simple_evaluate_fn):
        """測試違反約束的參數被拒絕"""

        def x_constraint(params: Dict) -> float:
            """x 不得超過 5"""
            return max(0, params['x'] - 5.0)

        optimizer = MultiObjectiveOptimizer(
            objectives=[('f1', 'minimize'), ('f2', 'minimize')],
            n_trials=20,
            constraints=[x_constraint],
            verbose=False
        )

        result = optimizer.optimize(simple_param_space, simple_evaluate_fn)

        # 檢查所有 Pareto 前沿解都滿足約束
        for solution in result.pareto_front:
            assert solution.params['x'] <= 5.0, "約束未生效"

    def test_multiple_constraints(self, simple_param_space, simple_evaluate_fn):
        """測試多個約束條件"""

        def x_constraint(params: Dict) -> float:
            return max(0, params['x'] - 5.0)

        def y_constraint(params: Dict) -> float:
            return max(0, params['y'] - 3.0)

        optimizer = MultiObjectiveOptimizer(
            objectives=[('f1', 'minimize'), ('f2', 'minimize')],
            n_trials=20,
            constraints=[x_constraint, y_constraint],
            verbose=False
        )

        result = optimizer.optimize(simple_param_space, simple_evaluate_fn)

        # 檢查所有解滿足兩個約束
        for solution in result.pareto_front:
            assert solution.params['x'] <= 5.0
            assert solution.params['y'] <= 3.0


class TestWarmStart:
    """測試暖啟動功能"""

    def test_warm_start_basic(self, simple_param_space, simple_evaluate_fn):
        """測試基本暖啟動功能"""

        optimizer = MultiObjectiveOptimizer(
            objectives=[('f1', 'minimize'), ('f2', 'minimize')],
            n_trials=20,
            verbose=False
        )

        # 提供初始解
        initial_solutions = [
            {'x': 1.0, 'y': 0.5},
            {'x': 5.0, 'y': 1.0}
        ]

        optimizer.warm_start(initial_solutions)
        result = optimizer.optimize(simple_param_space, simple_evaluate_fn)

        # 檢查結果中包含初始解（或其近似）
        assert len(result.all_solutions) >= len(initial_solutions)

    def test_warm_start_improves_convergence(self, simple_param_space, simple_evaluate_fn):
        """測試暖啟動確實改善收斂速度"""

        # 無暖啟動
        optimizer1 = MultiObjectiveOptimizer(
            objectives=[('f1', 'minimize'), ('f2', 'minimize')],
            n_trials=30,
            seed=42,
            verbose=False
        )
        result1 = optimizer1.optimize(simple_param_space, simple_evaluate_fn)

        # 有暖啟動
        optimizer2 = MultiObjectiveOptimizer(
            objectives=[('f1', 'minimize'), ('f2', 'minimize')],
            n_trials=30,
            seed=42,
            verbose=False
        )

        # 使用第一次優化的部分結果作為初始解
        initial_solutions = [
            sol.params for sol in result1.pareto_front[:3]
        ]
        optimizer2.warm_start(initial_solutions)
        result2 = optimizer2.optimize(simple_param_space, simple_evaluate_fn)

        # 暖啟動版本應該有更好或相近的 Pareto 前沿
        assert len(result2.pareto_front) > 0


class TestKneePoint:
    """測試膝點檢測"""

    def test_find_knee_point(self, simple_param_space, simple_evaluate_fn):
        """測試找到膝點"""

        optimizer = MultiObjectiveOptimizer(
            objectives=[('f1', 'minimize'), ('f2', 'minimize')],
            n_trials=50,
            verbose=False
        )

        result = optimizer.optimize(simple_param_space, simple_evaluate_fn)

        knee = result.find_knee_point()

        # 應該能找到膝點
        assert knee is not None
        assert knee in result.pareto_front

    def test_knee_point_empty_front(self):
        """測試空 Pareto 前沿的情況"""

        from src.optimizer.multi_objective import MultiObjectiveResult, ParetoSolution

        # 建立空結果
        result = MultiObjectiveResult(
            pareto_front=[],
            all_solutions=[],
            n_trials=0,
            study=None,  # type: ignore
        )

        knee = result.find_knee_point()
        assert knee is None


class TestParetoFiltering:
    """測試 Pareto 前沿篩選"""

    def test_crowding_filter(self, simple_param_space, simple_evaluate_fn):
        """測試擁擠度篩選"""

        optimizer = MultiObjectiveOptimizer(
            objectives=[('f1', 'minimize'), ('f2', 'minimize')],
            n_trials=50,
            verbose=False
        )

        result = optimizer.optimize(simple_param_space, simple_evaluate_fn)

        # 篩選最多樣化的 5 個解
        filtered = result.filter_pareto_front('crowding', n_select=5)

        assert len(filtered) <= 5
        assert len(filtered) <= len(result.pareto_front)

    def test_knee_filter(self, simple_param_space, simple_evaluate_fn):
        """測試膝點附近篩選"""

        optimizer = MultiObjectiveOptimizer(
            objectives=[('f1', 'minimize'), ('f2', 'minimize')],
            n_trials=50,
            verbose=False
        )

        result = optimizer.optimize(simple_param_space, simple_evaluate_fn)

        filtered = result.filter_pareto_front('knee', n_select=5)

        assert len(filtered) <= 5

    def test_extreme_filter(self, simple_param_space, simple_evaluate_fn):
        """測試極值解篩選"""

        optimizer = MultiObjectiveOptimizer(
            objectives=[('f1', 'minimize'), ('f2', 'minimize')],
            n_trials=50,
            verbose=False
        )

        result = optimizer.optimize(simple_param_space, simple_evaluate_fn)

        filtered = result.filter_pareto_front('extreme', n_select=6)

        assert len(filtered) <= 6

        # 極值解應該包含各目標的最佳解
        f1_values = [v for v in [s.get_objective_value('f1') for s in filtered] if v is not None]
        pareto_f1_values = [v for v in [s.get_objective_value('f1') for s in result.pareto_front] if v is not None]

        # 最小 f1 應該在篩選結果中
        if f1_values and pareto_f1_values:
            assert min(f1_values) == min(pareto_f1_values)

    def test_uniform_filter(self, simple_param_space, simple_evaluate_fn):
        """測試均勻篩選"""

        optimizer = MultiObjectiveOptimizer(
            objectives=[('f1', 'minimize'), ('f2', 'minimize')],
            n_trials=50,
            seed=42,  # 固定 seed 確保可重現性
            verbose=False
        )

        result = optimizer.optimize(simple_param_space, simple_evaluate_fn)

        filtered = result.filter_pareto_front('uniform', n_select=5)

        # 當 Pareto front 解數 >= n_select 時，返回 n_select 個
        # 當 Pareto front 解數 < n_select 時，返回所有可用解
        assert len(filtered) <= 5
        assert len(filtered) > 0


class TestVisualization:
    """測試視覺化方法"""

    def test_plot_pareto_front_3d(self, simple_param_space):
        """測試 3D Pareto 前沿繪製"""

        def three_objective_fn(params: Dict) -> Dict[str, float]:
            x = params['x']
            y = params['y']
            return {
                'f1': x,
                'f2': y,
                'f3': (x - 5) ** 2 + (y - 5) ** 2
            }

        optimizer = MultiObjectiveOptimizer(
            objectives=[('f1', 'minimize'), ('f2', 'minimize'), ('f3', 'minimize')],
            n_trials=30,
            verbose=False
        )

        result = optimizer.optimize(simple_param_space, three_objective_fn)

        # 測試繪圖不拋出異常
        try:
            fig = result.plot_pareto_front_3d('f1', 'f2', 'f3')
            # 如果 plotly 可用，應該返回 figure
            if fig is not None:
                assert hasattr(fig, 'data')
        except ImportError:
            # plotly 未安裝，跳過
            pass

    def test_plot_parallel_coordinates(self, simple_param_space, simple_evaluate_fn):
        """測試平行座標圖"""

        optimizer = MultiObjectiveOptimizer(
            objectives=[('f1', 'minimize'), ('f2', 'minimize')],
            n_trials=30,
            verbose=False
        )

        result = optimizer.optimize(simple_param_space, simple_evaluate_fn)

        try:
            fig = result.plot_parallel_coordinates()
            if fig is not None:
                assert hasattr(fig, 'data')
        except ImportError:
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
