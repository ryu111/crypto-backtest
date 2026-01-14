"""
src/types/ 邊界測試和整合測試

測試範圍：
1. 邊界值處理
2. 空值和 None 值處理
3. 與真實 experiments.json 格式相容性
4. 參數約束邊界情況
5. datetime 序列化/反序列化
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import datetime, timezone
from src.types import (
    BacktestResult,
    ValidationResult,
    ExperimentRecord,
    PerformanceMetrics,
    ParamSpace,
    StrategyStats,
)


def test_performance_metrics_none_filtering():
    """測試 PerformanceMetrics 過濾 None 值"""
    print("測試 PerformanceMetrics None 值過濾...")

    metrics = PerformanceMetrics(
        sharpe_ratio=1.5,
        total_return=0.3,
        max_drawdown=0.15,
        win_rate=0.55,
        profit_factor=1.8,
        total_trades=100,
        sortino_ratio=None,  # 明確設為 None
        avg_win=None,
        max_consecutive_wins=None,
    )

    data = metrics.to_dict()

    # None 值應該被過濾
    assert 'sortino_ratio' not in data
    assert 'avg_win' not in data
    assert 'max_consecutive_wins' not in data

    # 有值的欄位應該存在
    assert 'sharpe_ratio' in data
    assert data['sharpe_ratio'] == 1.5

    print("  ✅ None 值過濾正確")


def test_performance_metrics_unknown_fields():
    """測試 PerformanceMetrics from_dict 忽略未知欄位"""
    print("測試 PerformanceMetrics 忽略未知欄位...")

    data = {
        'sharpe_ratio': 1.5,
        'total_return': 0.3,
        'max_drawdown': 0.15,
        'win_rate': 0.55,
        'profit_factor': 1.8,
        'total_trades': 100,
        'unknown_field_1': 'should be ignored',
        'unknown_field_2': 999,
    }

    # 不應該拋出錯誤
    metrics = PerformanceMetrics.from_dict(data)
    assert metrics.sharpe_ratio == 1.5

    print("  ✅ 未知欄位被正確忽略")


def test_experiment_record_empty_dicts():
    """測試 ExperimentRecord 處理空字典"""
    print("測試 ExperimentRecord 空字典處理...")

    record = ExperimentRecord(
        id='exp_empty',
        timestamp=datetime.now(),
        strategy={},
        config={},
        results={},
        validation={},
    )

    # property 存取器應該回傳預設值
    assert record.sharpe_ratio == 0.0
    assert record.total_return == 0.0
    assert record.max_drawdown == 0.0
    assert record.grade == 'F'
    assert record.strategy_name == ''
    assert record.strategy_type == ''
    assert record.symbol == ''
    assert record.params == {}

    print("  ✅ 空字典處理正確")


def test_datetime_timezone_handling():
    """測試帶時區的 datetime"""
    print("測試 datetime 時區處理...")

    # UTC 時區
    now_utc = datetime.now(timezone.utc)
    record = ExperimentRecord(
        id='exp_tz',
        timestamp=now_utc,
        strategy={},
        config={},
        results={},
        validation={},
    )

    # 序列化
    data = record.to_dict()
    assert isinstance(data['timestamp'], str)
    assert 'T' in data['timestamp']

    # 反序列化
    restored = ExperimentRecord.from_dict(data)
    assert isinstance(restored.timestamp, datetime)

    print("  ✅ 時區處理正確")


def test_datetime_microsecond_precision():
    """測試 datetime 微秒精度"""
    print("測試 datetime 微秒精度...")

    now = datetime(2024, 1, 15, 10, 30, 45, 123456)
    record = ExperimentRecord(
        id='exp_micro',
        timestamp=now,
        strategy={},
        config={},
        results={},
        validation={},
    )

    # 往返轉換
    data = record.to_dict()
    restored = ExperimentRecord.from_dict(data)

    # ISO 格式應保留微秒
    assert restored.timestamp == now

    print("  ✅ 微秒精度保留正確")


def test_param_space_int_boundary():
    """測試 ParamSpace 整數邊界"""
    print("測試 ParamSpace 整數邊界...")

    space = ParamSpace(params={'value': (1, 1, 'int')})  # min == max

    # 採樣應該總是返回 1
    for _ in range(10):
        params = space.sample_random()
        assert params['value'] == 1

    print("  ✅ 整數邊界處理正確")


def test_param_space_float_precision():
    """測試 ParamSpace 浮點數精度"""
    print("測試 ParamSpace 浮點數精度...")

    space = ParamSpace(params={'threshold': (0.0001, 0.0002, 'float')})

    for _ in range(10):
        params = space.sample_random()
        assert 0.0001 <= params['threshold'] <= 0.0002
        assert isinstance(params['threshold'], float)

    print("  ✅ 浮點數精度正確")


def test_param_space_log_scale():
    """測試 ParamSpace 對數尺度"""
    print("測試 ParamSpace 對數尺度...")

    space = ParamSpace(params={'lr': (0.0001, 0.1, 'log')})

    # 採樣多次，檢查範圍
    samples = []
    for _ in range(100):
        params = space.sample_random()
        samples.append(params['lr'])
        assert 0.0001 <= params['lr'] <= 0.1

    # 對數尺度應該讓小值出現更頻繁
    small_values = sum(1 for x in samples if x < 0.01)
    assert small_values > 10  # 至少有 10% 的樣本 < 0.01

    print("  ✅ 對數尺度正確")


def test_param_space_impossible_constraints():
    """測試 ParamSpace 無法滿足的約束"""
    print("測試 ParamSpace 無法滿足的約束...")

    space = ParamSpace(
        params={'value': (10, 20, 'int')},
        constraints=[lambda p: p['value'] > 100]  # 永遠無法滿足
    )

    try:
        space.sample_random(max_retries=5)
        assert False, "應該拋出 ValueError"
    except ValueError as e:
        assert '無法在' in str(e)
        assert '次嘗試內滿足約束條件' in str(e)

    print("  ✅ 無法滿足的約束正確處理")


def test_param_space_multiple_constraints():
    """測試 ParamSpace 多個約束"""
    print("測試 ParamSpace 多個約束...")

    space = ParamSpace(
        params={
            'a': (1, 100, 'int'),
            'b': (1, 100, 'int'),
            'c': (1, 100, 'int'),
        },
        constraints=[
            lambda p: p['a'] < p['b'],
            lambda p: p['b'] < p['c'],
            lambda p: p['a'] + p['b'] + p['c'] <= 150,
        ]
    )

    for _ in range(20):
        params = space.sample_random()
        assert params['a'] < params['b']
        assert params['b'] < params['c']
        assert params['a'] + params['b'] + params['c'] <= 150

    print("  ✅ 多個約束正確處理")


def test_strategy_stats_zero_attempts():
    """測試 StrategyStats 零次嘗試"""
    print("測試 StrategyStats 零次嘗試...")

    stats = StrategyStats(name='test')

    # 避免除以零
    assert stats.success_rate == 0.0

    # UCB 應該回傳無限大（未嘗試的策略優先）
    ucb = stats.calculate_ucb(total_attempts=100)
    assert ucb == float('inf')

    print("  ✅ 零次嘗試處理正確")


def test_strategy_stats_incremental_average():
    """測試 StrategyStats 增量平均值更新"""
    print("測試 StrategyStats 增量平均值更新...")

    stats = StrategyStats(name='test')

    # 第一次更新
    stats.update_from_experiment(1.0, True, {'a': 1})
    assert stats.avg_sharpe == 1.0

    # 第二次更新
    stats.update_from_experiment(2.0, True, {'a': 2})
    assert stats.avg_sharpe == 1.5  # (1.0 + 2.0) / 2

    # 第三次更新
    stats.update_from_experiment(3.0, False, {'a': 3})
    assert stats.avg_sharpe == 2.0  # (1.0 + 2.0 + 3.0) / 3

    # 最佳/最差 Sharpe
    assert stats.best_sharpe == 3.0
    assert stats.worst_sharpe == 1.0

    print("  ✅ 增量平均值更新正確")


def test_strategy_stats_datetime_tracking():
    """測試 StrategyStats datetime 追蹤"""
    print("測試 StrategyStats datetime 追蹤...")

    stats = StrategyStats(name='test')

    # 第一次更新
    stats.update_from_experiment(1.0, True, {})
    first_attempt = stats.first_attempt
    last_attempt1 = stats.last_attempt

    assert first_attempt is not None
    assert last_attempt1 is not None

    # 第二次更新
    import time
    time.sleep(0.01)  # 確保時間不同
    stats.update_from_experiment(2.0, True, {})
    last_attempt2 = stats.last_attempt

    # first_attempt 不變，last_attempt 更新
    assert stats.first_attempt == first_attempt
    assert stats.last_attempt > last_attempt1

    print("  ✅ datetime 追蹤正確")


def test_real_experiments_json_compatibility():
    """測試與真實 experiments.json 格式相容性"""
    print("測試與真實 experiments.json 格式相容性...")

    json_path = Path(__file__).parent.parent / 'learning' / 'experiments.json'

    if not json_path.exists():
        print("  ⚠️  experiments.json 不存在，跳過測試")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # 測試讀取所有實驗
    success_count = 0
    for exp_data in data['experiments'][:10]:  # 只測試前 10 筆
        try:
            record = ExperimentRecord.from_dict(exp_data)

            # 驗證必要欄位
            assert record.id
            assert record.timestamp
            assert isinstance(record.sharpe_ratio, (int, float))
            assert isinstance(record.grade, str)

            success_count += 1
        except Exception as e:
            print(f"    ⚠️  實驗 {exp_data.get('id', 'unknown')} 解析失敗: {e}")

    print(f"  ✅ 成功解析 {success_count} 筆實驗記錄")


def test_real_experiments_json_roundtrip():
    """測試與真實資料的往返轉換"""
    print("測試真實資料往返轉換...")

    json_path = Path(__file__).parent.parent / 'learning' / 'experiments.json'

    if not json_path.exists():
        print("  ⚠️  experiments.json 不存在，跳過測試")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    if not data['experiments']:
        print("  ⚠️  experiments.json 為空，跳過測試")
        return

    # 取第一筆實驗
    original_data = data['experiments'][0]

    # 載入 → 轉回字典
    record = ExperimentRecord.from_dict(original_data)
    restored_data = record.to_dict()

    # 驗證關鍵欄位一致
    assert restored_data['id'] == original_data['id']
    assert restored_data['strategy'] == original_data['strategy']
    assert restored_data['config'] == original_data['config']

    # results 應該相同（忽略順序）
    for key in original_data['results']:
        if key in restored_data['results']:
            assert abs(restored_data['results'][key] - original_data['results'][key]) < 1e-6

    print("  ✅ 真實資料往返轉換正確")


def test_validation_result_all_grades():
    """測試 ValidationResult 所有評級"""
    print("測試 ValidationResult 所有評級...")

    # A/B 為通過
    assert ValidationResult(grade='A', stages_passed=[1,2,3,4,5]).is_passing is True
    assert ValidationResult(grade='B', stages_passed=[1,2,3]).is_passing is True

    # C/D/F 為不通過
    assert ValidationResult(grade='C', stages_passed=[1]).is_passing is False
    assert ValidationResult(grade='D', stages_passed=[]).is_passing is False
    assert ValidationResult(grade='F', stages_passed=[]).is_passing is False

    print("  ✅ 所有評級處理正確")


def test_backtest_result_flattening():
    """測試 BacktestResult 扁平化"""
    print("測試 BacktestResult 扁平化...")

    metrics = PerformanceMetrics(
        sharpe_ratio=1.5,
        total_return=0.3,
        max_drawdown=0.15,
        win_rate=0.55,
        profit_factor=1.8,
        total_trades=100,
        sortino_ratio=2.0,
    )

    result = BacktestResult(metrics=metrics, execution_time=2.5)
    data = result.to_dict()

    # 應該展開為扁平結構（與 experiments.json 一致）
    assert 'sharpe_ratio' in data
    assert 'sortino_ratio' in data
    assert 'execution_time' in data
    assert 'metrics' not in data  # 不應該有嵌套的 metrics

    print("  ✅ 扁平化正確")


def run_all_tests():
    """執行所有邊界和整合測試"""
    print("\n" + "=" * 60)
    print("src/types/ 邊界測試和整合測試")
    print("=" * 60 + "\n")

    tests = [
        test_performance_metrics_none_filtering,
        test_performance_metrics_unknown_fields,
        test_experiment_record_empty_dicts,
        test_datetime_timezone_handling,
        test_datetime_microsecond_precision,
        test_param_space_int_boundary,
        test_param_space_float_precision,
        test_param_space_log_scale,
        test_param_space_impossible_constraints,
        test_param_space_multiple_constraints,
        test_strategy_stats_zero_attempts,
        test_strategy_stats_incremental_average,
        test_strategy_stats_datetime_tracking,
        test_real_experiments_json_compatibility,
        test_real_experiments_json_roundtrip,
        test_validation_result_all_grades,
        test_backtest_result_flattening,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {test.__name__} 失敗: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__} 錯誤: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"邊界測試完成: {passed} 通過, {failed} 失敗")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
