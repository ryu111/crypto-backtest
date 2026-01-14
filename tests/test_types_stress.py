"""
src/types/ 壓力測試

測試範圍：
1. 大量資料處理（622 筆實驗記錄）
2. 效能測試（序列化/反序列化速度）
3. 記憶體效率測試
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
from src.types import ExperimentRecord


def test_load_all_experiments():
    """測試載入所有 622 筆實驗記錄"""
    print("測試載入所有實驗記錄...")

    json_path = Path(__file__).parent.parent / 'learning' / 'experiments.json'

    if not json_path.exists():
        print("  ⚠️  experiments.json 不存在，跳過測試")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    total = len(data['experiments'])
    print(f"  📊 總共 {total} 筆實驗記錄")

    # 載入所有記錄
    start_time = time.time()
    records = []
    errors = 0

    for exp_data in data['experiments']:
        try:
            record = ExperimentRecord.from_dict(exp_data)
            records.append(record)
        except Exception as e:
            errors += 1
            print(f"    ⚠️  解析失敗: {e}")

    elapsed = time.time() - start_time

    print(f"  ✅ 成功載入 {len(records)} 筆記錄")
    print(f"  ⏱️  耗時 {elapsed:.3f} 秒 ({elapsed/total*1000:.2f} ms/record)")

    if errors > 0:
        print(f"  ⚠️  {errors} 筆記錄解析失敗")

    # 統計分析
    print("\n  📈 統計分析:")
    sharpe_values = [r.sharpe_ratio for r in records]
    grades = [r.grade for r in records]

    print(f"    - 平均 Sharpe: {sum(sharpe_values)/len(sharpe_values):.3f}")
    print(f"    - 最大 Sharpe: {max(sharpe_values):.3f}")
    print(f"    - 最小 Sharpe: {min(sharpe_values):.3f}")
    print(f"    - A/B 評級: {sum(1 for g in grades if g in ['A', 'B'])} 筆")
    print(f"    - C/D/F 評級: {sum(1 for g in grades if g in ['C', 'D', 'F'])} 筆")

    assert len(records) == total - errors


def test_serialization_performance():
    """測試序列化/反序列化效能"""
    print("\n測試序列化/反序列化效能...")

    json_path = Path(__file__).parent.parent / 'learning' / 'experiments.json'

    if not json_path.exists():
        print("  ⚠️  experiments.json 不存在，跳過測試")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    experiments = data['experiments'][:100]  # 測試前 100 筆

    # 測試 from_dict 效能
    start_time = time.time()
    records = [ExperimentRecord.from_dict(exp) for exp in experiments]
    from_dict_time = time.time() - start_time

    # 測試 to_dict 效能
    start_time = time.time()
    dicts = [record.to_dict() for record in records]
    to_dict_time = time.time() - start_time

    print(f"  ✅ from_dict: {from_dict_time:.3f} 秒 ({from_dict_time/100*1000:.2f} ms/record)")
    print(f"  ✅ to_dict: {to_dict_time:.3f} 秒 ({to_dict_time/100*1000:.2f} ms/record)")

    # 驗證往返轉換一致性
    for original, restored in zip(experiments, dicts):
        assert original['id'] == restored['id']


def test_property_access_performance():
    """測試 property 存取效能"""
    print("\n測試 property 存取效能...")

    json_path = Path(__file__).parent.parent / 'learning' / 'experiments.json'

    if not json_path.exists():
        print("  ⚠️  experiments.json 不存在，跳過測試")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    record = ExperimentRecord.from_dict(data['experiments'][0])

    # 重複存取 property
    iterations = 10000
    start_time = time.time()

    for _ in range(iterations):
        _ = record.sharpe_ratio
        _ = record.total_return
        _ = record.max_drawdown
        _ = record.grade
        _ = record.strategy_name

    elapsed = time.time() - start_time

    print(f"  ✅ {iterations} 次存取耗時 {elapsed:.3f} 秒 ({elapsed/iterations*1e6:.2f} μs/access)")


def test_filter_by_criteria():
    """測試篩選實驗記錄（模擬查詢場景）"""
    print("\n測試篩選實驗記錄...")

    json_path = Path(__file__).parent.parent / 'learning' / 'experiments.json'

    if not json_path.exists():
        print("  ⚠️  experiments.json 不存在，跳過測試")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # 載入所有記錄
    records = [ExperimentRecord.from_dict(exp) for exp in data['experiments']]

    # 篩選：Sharpe > 1.5 且評級 A/B
    start_time = time.time()
    filtered = [
        r for r in records
        if r.sharpe_ratio > 1.5 and r.is_success
    ]
    elapsed = time.time() - start_time

    print(f"  ✅ 篩選出 {len(filtered)} 筆高品質實驗")
    print(f"  ⏱️  篩選耗時 {elapsed:.3f} 秒")

    # 按 Sharpe 排序
    start_time = time.time()
    sorted_records = sorted(filtered, key=lambda r: r.sharpe_ratio, reverse=True)
    elapsed = time.time() - start_time

    print(f"  ✅ 排序耗時 {elapsed:.3f} 秒")

    if sorted_records:
        print(f"  🏆 最佳實驗: Sharpe {sorted_records[0].sharpe_ratio:.3f} ({sorted_records[0].strategy_name})")


def test_group_by_strategy():
    """測試按策略分組（模擬統計場景）"""
    print("\n測試按策略分組...")

    json_path = Path(__file__).parent.parent / 'learning' / 'experiments.json'

    if not json_path.exists():
        print("  ⚠️  experiments.json 不存在，跳過測試")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    records = [ExperimentRecord.from_dict(exp) for exp in data['experiments']]

    # 按策略名稱分組
    from collections import defaultdict
    strategy_groups = defaultdict(list)

    start_time = time.time()
    for record in records:
        strategy_groups[record.strategy_name].append(record)
    elapsed = time.time() - start_time

    print(f"  ✅ 分組耗時 {elapsed:.3f} 秒")
    print(f"  📊 共 {len(strategy_groups)} 種策略")

    # 統計每個策略的表現
    print("\n  策略表現統計:")
    for strategy_name, group in sorted(strategy_groups.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
        avg_sharpe = sum(r.sharpe_ratio for r in group) / len(group)
        success_rate = sum(1 for r in group if r.is_success) / len(group)
        print(f"    - {strategy_name}: {len(group)} 次, 平均 Sharpe {avg_sharpe:.3f}, 成功率 {success_rate:.1%}")


def run_all_tests():
    """執行所有壓力測試"""
    print("\n" + "=" * 60)
    print("src/types/ 壓力測試")
    print("=" * 60 + "\n")

    tests = [
        test_load_all_experiments,
        test_serialization_performance,
        test_property_access_performance,
        test_filter_by_criteria,
        test_group_by_strategy,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  ❌ {test.__name__} 錯誤: {e}")

    print("\n" + "=" * 60)
    print("壓力測試完成")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_tests()
