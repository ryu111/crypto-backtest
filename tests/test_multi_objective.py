"""
多目標優化器測試

測試 NSGA-II 多目標優化的各項功能。
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict

from src.optimizer.multi_objective import (
    MultiObjectiveOptimizer,
    MultiObjectiveResult,
    ParetoSolution,
    ObjectiveResult,
    optimize_multi_objective,
    OPTUNA_AVAILABLE
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_param_space():
    """簡單參數空間"""
    return {
        'x': {'type': 'float', 'low': 0.0, 'high': 5.0},
        'y': {'type': 'float', 'low': 0.0, 'high': 5.0}
    }


@pytest.fixture
def complex_param_space():
    """複雜參數空間（包含不同類型）"""
    return {
        'fast_period': {'type': 'int', 'low': 5, 'high': 20},
        'slow_period': {'type': 'int', 'low': 20, 'high': 50},
        'stop_loss': {'type': 'float', 'low': 0.01, 'high': 0.05, 'step': 0.005},
        'take_profit': {'type': 'float', 'low': 0.02, 'high': 0.1},
        'use_filter': {'type': 'categorical', 'choices': [True, False]}
    }


@pytest.fixture
def two_objective_evaluator():
    """雙目標評估函數（最小化距離原點的距離）"""
    def evaluate(params: Dict) -> Dict[str, float]:
        x = params['x']
        y = params['y']
        return {
            'f1': x**2,           # 最小化 x^2
            'f2': (y - 5)**2      # 最小化 (y-5)^2
        }
    return evaluate


@pytest.fixture
def three_objective_evaluator():
    """三目標評估函數"""
    def evaluate(params: Dict) -> Dict[str, float]:
        x = params['x']
        y = params['y']
        return {
            'sharpe': -((x - 2.5)**2 + (y - 2.5)**2),  # 最大化（負平方距離）
            'max_dd': (x - 2.5)**2,                     # 最小化
            'win_rate': -abs(x - y)                     # 最大化（負差異）
        }
    return evaluate


# ============================================================================
# 1. 基本功能測試
# ============================================================================

@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
def test_optimizer_initialization():
    """測試優化器初始化"""
    optimizer = MultiObjectiveOptimizer(
        objectives=[('f1', 'minimize'), ('f2', 'minimize')],
        n_trials=10,
        seed=42
    )

    assert optimizer.n_trials == 10
    assert optimizer.seed == 42
    assert len(optimizer.objectives) == 2
    assert optimizer.objectives[0] == ('f1', 'minimize')
    assert optimizer.objectives[1] == ('f2', 'minimize')


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
def test_optimizer_requires_objectives():
    """測試必須提供目標函數"""
    with pytest.raises(ValueError, match="至少需要提供一個目標函數"):
        MultiObjectiveOptimizer(objectives=[], n_trials=10)


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
def test_simple_optimization(simple_param_space, two_objective_evaluator):
    """測試簡單雙目標優化"""
    optimizer = MultiObjectiveOptimizer(
        objectives=[('f1', 'minimize'), ('f2', 'minimize')],
        n_trials=30,
        seed=42,
        verbose=False
    )

    result = optimizer.optimize(
        param_space=simple_param_space,
        evaluate_fn=two_objective_evaluator,
        show_progress_bar=False
    )

    # 驗證結果
    assert isinstance(result, MultiObjectiveResult)
    assert result.n_trials == 30
    assert result.n_completed_trials > 0
    assert len(result.pareto_front) > 0
    assert len(result.all_solutions) == result.n_completed_trials


# ============================================================================
# 2. Pareto 前緣測試
# ============================================================================

@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
def test_pareto_front_properties(simple_param_space, two_objective_evaluator):
    """測試 Pareto 前緣的性質"""
    optimizer = MultiObjectiveOptimizer(
        objectives=[('f1', 'minimize'), ('f2', 'minimize')],
        n_trials=50,
        seed=42,
        verbose=False
    )

    result = optimizer.optimize(
        param_space=simple_param_space,
        evaluate_fn=two_objective_evaluator,
        show_progress_bar=False
    )

    # Pareto 前緣應該有多個解
    assert len(result.pareto_front) > 1

    # 所有 Pareto 解的 rank 應該是 0
    for solution in result.pareto_front:
        assert solution.rank == 0

    # 驗證非支配性（任意兩個解之間）
    for i, sol1 in enumerate(result.pareto_front):
        for sol2 in result.pareto_front[i+1:]:
            # sol1 不應嚴格支配 sol2（至少有一個目標更差）
            f1_1 = sol1.get_objective_value('f1')
            f2_1 = sol1.get_objective_value('f2')
            f1_2 = sol2.get_objective_value('f1')
            f2_2 = sol2.get_objective_value('f2')

            # 檢查非嚴格支配
            assert not (f1_1 < f1_2 and f2_1 < f2_2)
            assert not (f1_2 < f1_1 and f2_2 < f2_1)


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
def test_crowding_distance_calculation(simple_param_space, two_objective_evaluator):
    """測試擁擠度距離計算"""
    optimizer = MultiObjectiveOptimizer(
        objectives=[('f1', 'minimize'), ('f2', 'minimize')],
        n_trials=50,
        seed=42,
        verbose=False
    )

    result = optimizer.optimize(
        param_space=simple_param_space,
        evaluate_fn=two_objective_evaluator,
        show_progress_bar=False
    )

    # 所有解都應該有擁擠度距離
    for solution in result.pareto_front:
        assert solution.crowding_distance >= 0.0

    # 如果有超過 2 個解，邊界解應該有無限大距離
    if len(result.pareto_front) > 2:
        # 第一個和最後一個（已按 crowding_distance 排序）
        assert result.pareto_front[0].crowding_distance == float('inf') or \
               result.pareto_front[-1].crowding_distance == float('inf')


# ============================================================================
# 3. 多目標測試（3 個目標）
# ============================================================================

@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
def test_three_objective_optimization(simple_param_space, three_objective_evaluator):
    """測試三目標優化"""
    optimizer = MultiObjectiveOptimizer(
        objectives=[
            ('sharpe', 'maximize'),
            ('max_dd', 'minimize'),
            ('win_rate', 'maximize')
        ],
        n_trials=50,
        seed=42,
        verbose=False
    )

    result = optimizer.optimize(
        param_space=simple_param_space,
        evaluate_fn=three_objective_evaluator,
        show_progress_bar=False
    )

    # 驗證結果
    assert len(result.pareto_front) > 0

    # 每個解應該有 3 個目標
    for solution in result.pareto_front:
        assert len(solution.objectives) == 3
        assert solution.get_objective_value('sharpe') is not None
        assert solution.get_objective_value('max_dd') is not None
        assert solution.get_objective_value('win_rate') is not None


# ============================================================================
# 4. 最佳解選擇測試
# ============================================================================

@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
def test_get_best_solution_equal_weights(simple_param_space, two_objective_evaluator):
    """測試取得最佳解（均等權重）"""
    optimizer = MultiObjectiveOptimizer(
        objectives=[('f1', 'minimize'), ('f2', 'minimize')],
        n_trials=50,
        seed=42,
        verbose=False
    )

    result = optimizer.optimize(
        param_space=simple_param_space,
        evaluate_fn=two_objective_evaluator,
        show_progress_bar=False
    )

    # 取得最佳解（均等權重）
    best = result.get_best_solution()

    assert best is not None
    assert isinstance(best, ParetoSolution)
    assert best in result.pareto_front


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
def test_get_best_solution_custom_weights(simple_param_space, two_objective_evaluator):
    """測試取得最佳解（自訂權重）"""
    optimizer = MultiObjectiveOptimizer(
        objectives=[('f1', 'minimize'), ('f2', 'minimize')],
        n_trials=50,
        seed=42,
        verbose=False
    )

    result = optimizer.optimize(
        param_space=simple_param_space,
        evaluate_fn=two_objective_evaluator,
        show_progress_bar=False
    )

    # 偏重 f1
    best_f1 = result.get_best_solution(weights={'f1': 0.8, 'f2': 0.2})
    # 偏重 f2
    best_f2 = result.get_best_solution(weights={'f1': 0.2, 'f2': 0.8})

    assert best_f1 is not None
    assert best_f2 is not None

    # 不同權重應該可能得到不同解（但不保證）
    # 至少應該都在 Pareto 前緣上
    assert best_f1 in result.pareto_front
    assert best_f2 in result.pareto_front


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
def test_get_best_solution_empty_pareto():
    """測試空 Pareto 前緣"""
    result = MultiObjectiveResult(
        pareto_front=[],
        all_solutions=[],
        n_trials=0,
        study=None,  # type: ignore
        optimization_time=0.0
    )

    best = result.get_best_solution()
    assert best is None


# ============================================================================
# 5. 資料結構測試
# ============================================================================

def test_objective_result():
    """測試 ObjectiveResult"""
    obj = ObjectiveResult(name='sharpe', value=1.5, direction='maximize')

    assert obj.name == 'sharpe'
    assert obj.value == 1.5
    assert obj.direction == 'maximize'
    assert 'sharpe=1.5' in str(obj)


def test_pareto_solution():
    """測試 ParetoSolution"""
    solution = ParetoSolution(
        params={'x': 1.0, 'y': 2.0},
        objectives=[
            ObjectiveResult('f1', 0.5, 'minimize'),
            ObjectiveResult('f2', 1.0, 'minimize')
        ],
        rank=0,
        crowding_distance=2.5,
        trial_number=10
    )

    assert solution.get_objective_value('f1') == 0.5
    assert solution.get_objective_value('f2') == 1.0
    assert solution.get_objective_value('f3') is None

    # 測試 to_dict
    d = solution.to_dict()
    assert d['params'] == {'x': 1.0, 'y': 2.0}
    assert d['objectives'] == {'f1': 0.5, 'f2': 1.0}
    assert d['rank'] == 0
    assert d['crowding_distance'] == 2.5


# ============================================================================
# 6. 輸出格式測試
# ============================================================================

@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
def test_result_to_dataframe(simple_param_space, two_objective_evaluator):
    """測試結果轉為 DataFrame"""
    optimizer = MultiObjectiveOptimizer(
        objectives=[('f1', 'minimize'), ('f2', 'minimize')],
        n_trials=30,
        seed=42,
        verbose=False
    )

    result = optimizer.optimize(
        param_space=simple_param_space,
        evaluate_fn=two_objective_evaluator,
        show_progress_bar=False
    )

    df = result.to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(result.pareto_front)
    assert 'x' in df.columns
    assert 'y' in df.columns
    assert 'f1' in df.columns
    assert 'f2' in df.columns
    assert 'rank' in df.columns
    assert 'crowding_distance' in df.columns


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
def test_result_summary(simple_param_space, two_objective_evaluator):
    """測試結果摘要"""
    optimizer = MultiObjectiveOptimizer(
        objectives=[('f1', 'minimize'), ('f2', 'minimize')],
        n_trials=20,
        seed=42,
        verbose=False
    )

    result = optimizer.optimize(
        param_space=simple_param_space,
        evaluate_fn=two_objective_evaluator,
        show_progress_bar=False
    )

    summary = result.summary()

    assert isinstance(summary, str)
    assert 'Pareto 前緣' in summary
    assert 'f1' in summary
    assert 'f2' in summary


# ============================================================================
# 7. 錯誤處理測試
# ============================================================================

@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
def test_invalid_param_space():
    """測試無效參數空間"""
    optimizer = MultiObjectiveOptimizer(
        objectives=[('f1', 'minimize')],
        n_trials=10
    )

    with pytest.raises(ValueError, match="必須提供參數空間定義"):
        optimizer.optimize(
            param_space={},
            evaluate_fn=lambda p: {'f1': 0.0}
        )


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
def test_evaluator_returns_non_dict(simple_param_space):
    """測試評估函數回傳非字典"""
    optimizer = MultiObjectiveOptimizer(
        objectives=[('f1', 'minimize')],
        n_trials=5,
        verbose=False
    )

    def bad_evaluator(params):
        return 0.5  # 應該回傳 dict

    # 應該被剪枝，但不會崩潰
    result = optimizer.optimize(
        param_space=simple_param_space,
        evaluate_fn=bad_evaluator,
        show_progress_bar=False
    )

    # 所有試驗應該失敗
    assert result.n_completed_trials == 0


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
def test_evaluator_missing_objective(simple_param_space):
    """測試評估函數缺少目標"""
    optimizer = MultiObjectiveOptimizer(
        objectives=[('f1', 'minimize'), ('f2', 'minimize')],
        n_trials=5,
        verbose=False
    )

    def incomplete_evaluator(params):
        return {'f1': 0.5}  # 缺少 f2

    result = optimizer.optimize(
        param_space=simple_param_space,
        evaluate_fn=incomplete_evaluator,
        show_progress_bar=False
    )

    # 所有試驗應該失敗
    assert result.n_completed_trials == 0


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
def test_evaluator_returns_nan(simple_param_space):
    """測試評估函數回傳 NaN"""
    optimizer = MultiObjectiveOptimizer(
        objectives=[('f1', 'minimize')],
        n_trials=5,
        verbose=False
    )

    def nan_evaluator(params):
        return {'f1': float('nan')}

    result = optimizer.optimize(
        param_space=simple_param_space,
        evaluate_fn=nan_evaluator,
        show_progress_bar=False
    )

    # 所有試驗應該被剪枝
    assert result.n_completed_trials == 0


# ============================================================================
# 8. 便利函數測試
# ============================================================================

@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
def test_optimize_multi_objective_convenience(simple_param_space, two_objective_evaluator):
    """測試便利函數"""
    result = optimize_multi_objective(
        param_space=simple_param_space,
        evaluate_fn=two_objective_evaluator,
        objectives=[('f1', 'minimize'), ('f2', 'minimize')],
        n_trials=20,
        seed=42,
        verbose=False
    )

    assert isinstance(result, MultiObjectiveResult)
    assert result.n_trials == 20
    assert len(result.pareto_front) > 0


# ============================================================================
# 9. 複雜參數空間測試
# ============================================================================

@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
def test_complex_param_space_optimization(complex_param_space):
    """測試複雜參數空間"""
    def complex_evaluator(params):
        # 簡化的策略評估
        fast = params['fast_period']
        slow = params['slow_period']
        sl = params['stop_loss']
        tp = params['take_profit']
        use_filter = params['use_filter']

        # 虛擬指標（實際應使用回測結果）
        sharpe = -(abs(slow - fast - 20) / 20.0) + (tp / sl)
        max_dd = sl * 10
        win_rate = tp / (sl + tp)

        if use_filter:
            sharpe *= 1.1
            win_rate *= 1.05

        return {
            'sharpe': sharpe,
            'max_dd': max_dd,
            'win_rate': win_rate
        }

    optimizer = MultiObjectiveOptimizer(
        objectives=[
            ('sharpe', 'maximize'),
            ('max_dd', 'minimize'),
            ('win_rate', 'maximize')
        ],
        n_trials=30,
        seed=42,
        verbose=False
    )

    result = optimizer.optimize(
        param_space=complex_param_space,
        evaluate_fn=complex_evaluator,
        show_progress_bar=False
    )

    # 驗證參數類型正確
    for solution in result.pareto_front:
        assert isinstance(solution.params['fast_period'], int)
        assert isinstance(solution.params['slow_period'], int)
        assert isinstance(solution.params['stop_loss'], float)
        assert isinstance(solution.params['take_profit'], float)
        assert isinstance(solution.params['use_filter'], bool)


# ============================================================================
# 10. 性能測試
# ============================================================================

@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
def test_optimization_time_recorded(simple_param_space, two_objective_evaluator):
    """測試優化時間記錄"""
    optimizer = MultiObjectiveOptimizer(
        objectives=[('f1', 'minimize'), ('f2', 'minimize')],
        n_trials=10,
        seed=42,
        verbose=False
    )

    result = optimizer.optimize(
        param_space=simple_param_space,
        evaluate_fn=two_objective_evaluator,
        show_progress_bar=False
    )

    # 優化時間應該 > 0
    assert result.optimization_time > 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
