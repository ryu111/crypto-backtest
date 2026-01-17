"""
VariationTracker 单元测试套件

测试覆盖范围：
1. TestComputeHash - hash 计算与参数标准化
2. TestIsSimilarParams - 参数相似度检测（5% 阈值）
3. TestUpdateFromExperiment - Grade → Status 状态转换
4. TestGetFinalBacktestList - 最终回测清单过滤与排序
5. TestVariationRecord - 数据序列化与反序列化
6. TestRegisterVariation - 变化注册与去重
"""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.automation.variation_tracker import (
    VariationTracker,
    VariationRecord,
    VariationStatus
)


# ========== Fixtures ==========

@pytest.fixture
def tmp_variation_tracker(tmp_path):
    """创建临时 VariationTracker 实例（使用 tmp_path 避免影响真实文件）"""
    json_path = tmp_path / "variations.json"
    md_path = tmp_path / "variations.md"

    tracker = VariationTracker(
        json_path=json_path,
        md_path=md_path,
        float_precision=2,
        similarity_threshold=0.05
    )

    return tracker


@pytest.fixture
def sample_params():
    """样本参数字典"""
    return {
        "fast": 10.12345,
        "slow": 30.98765,
        "threshold": 0.05,
        "stop_loss": -0.02
    }


@pytest.fixture
def sample_metrics():
    """样本性能指标"""
    return {
        "sharpe_ratio": 2.1,
        "total_return": 0.35,
        "max_drawdown": 0.12,
        "win_rate": 0.58,
        "total_trades": 150
    }


# ========== TestComputeHash ==========

class TestComputeHash:
    """Hash 计算测试"""

    def test_same_params_same_hash(self, tmp_variation_tracker, sample_params):
        """同一参数应产生相同 hash"""
        tracker = tmp_variation_tracker

        hash1 = tracker.compute_hash("ma_cross", sample_params)
        hash2 = tracker.compute_hash("ma_cross", sample_params)

        assert hash1 == hash2
        assert hash1.startswith("var_")
        assert len(hash1) == len("var_") + tracker.HASH_LENGTH

    def test_different_params_different_hash(self, tmp_variation_tracker):
        """不同参数应产生不同 hash"""
        tracker = tmp_variation_tracker

        params1 = {"fast": 10, "slow": 30}
        params2 = {"fast": 10, "slow": 31}

        hash1 = tracker.compute_hash("ma_cross", params1)
        hash2 = tracker.compute_hash("ma_cross", params2)

        assert hash1 != hash2

    def test_float_precision_rounding(self, tmp_variation_tracker):
        """浮点数应正确捨入（2位小数）"""
        tracker = tmp_variation_tracker

        # 浮点数差异不足 0.01，应视为相同
        params1 = {"price": 10.123}
        params2 = {"price": 10.124}

        hash1 = tracker.compute_hash("strategy", params1)
        hash2 = tracker.compute_hash("strategy", params2)

        # 都应该四舍五入到 10.12，所以 hash 相同
        assert hash1 == hash2

    def test_float_precision_difference(self, tmp_variation_tracker):
        """浮点数差异 >= 0.01，应产生不同 hash"""
        tracker = tmp_variation_tracker

        params1 = {"price": 10.12}
        params2 = {"price": 10.13}

        hash1 = tracker.compute_hash("strategy", params1)
        hash2 = tracker.compute_hash("strategy", params2)

        assert hash1 != hash2

    def test_param_order_independent(self, tmp_variation_tracker):
        """参数顺序不影响 hash"""
        tracker = tmp_variation_tracker

        params1 = {"fast": 10, "slow": 30, "threshold": 0.05}
        params2 = {"threshold": 0.05, "slow": 30, "fast": 10}

        hash1 = tracker.compute_hash("ma_cross", params1)
        hash2 = tracker.compute_hash("ma_cross", params2)

        assert hash1 == hash2

    def test_different_strategy_different_hash(self, tmp_variation_tracker):
        """不同策略名称应产生不同 hash"""
        tracker = tmp_variation_tracker

        params = {"fast": 10, "slow": 30}

        hash1 = tracker.compute_hash("ma_cross", params)
        hash2 = tracker.compute_hash("rsi_strategy", params)

        assert hash1 != hash2

    def test_hash_format(self, tmp_variation_tracker):
        """Hash 格式应为 var_<16个十六进制数字>"""
        tracker = tmp_variation_tracker

        hash_result = tracker.compute_hash("test", {"param": 1})

        assert hash_result.startswith("var_")
        assert len(hash_result) == 4 + 16  # "var_" + 16 chars
        # 验证后面是十六进制
        assert all(c in "0123456789abcdef" for c in hash_result[4:])


# ========== TestIsSimilarParams ==========

class TestIsSimilarParams:
    """参数相似度检测测试（5% 阈值）"""

    def test_identical_params_similar(self, tmp_variation_tracker):
        """完全相同的参数应视为相似"""
        tracker = tmp_variation_tracker

        params = {"fast": 10, "slow": 30, "threshold": 0.05}

        assert tracker._is_similar_params(params, params) is True

    def test_threshold_boundary_below_threshold(self, tmp_variation_tracker):
        """相对差异 4.9% < 5% 应视为相似"""
        tracker = tmp_variation_tracker
        threshold = tracker.similarity_threshold  # 0.05 = 5%
        # 使用简单的数值，确保相对差异 < 5%
        params1 = {"value": 100.0}
        params2 = {"value": 95.0}  # 5% 差异

        # 验证相对差异
        rel_diff = abs(100.0 - 95.0) / 95.0  # 0.0526 = 5.26%，略超过 5%
        # 改用 96 得到 4.17% 差异
        params2 = {"value": 96.0}
        rel_diff = abs(100.0 - 96.0) / 96.0  # 0.0417 = 4.17% < 5%
        assert rel_diff < 0.05
        assert tracker._is_similar_params(params1, params2) is True

    def test_threshold_boundary_at_threshold(self, tmp_variation_tracker):
        """相对差异恰好 5% 时的边界情况"""
        tracker = tmp_variation_tracker

        base = 100.0
        # 恰好 5% 差异
        params1 = {"value": base}
        params2 = {"value": base * (1 - 0.05)}

        # 边界值不应相同（> 而不是 >=）
        assert tracker._is_similar_params(params1, params2) is False

    def test_threshold_boundary_above_threshold(self, tmp_variation_tracker):
        """相对差异 5.1% > 5% 应视为不相似"""
        tracker = tmp_variation_tracker

        base = 100.0
        # 差异 5.1%
        params1 = {"value": base}
        params2 = {"value": base * (1 - 0.051)}

        assert tracker._is_similar_params(params1, params2) is False

    def test_zero_value_handling(self, tmp_variation_tracker):
        """零值处理（避免除以零）"""
        tracker = tmp_variation_tracker

        # 一个为零，另一个非零
        params1 = {"value": 0.0}
        params2 = {"value": 1.0}

        # 应该视为不相似
        result = tracker._is_similar_params(params1, params2)
        assert result is False

    def test_both_zero_similar(self, tmp_variation_tracker):
        """两个都是零应视为相似"""
        tracker = tmp_variation_tracker

        params1 = {"value": 0.0}
        params2 = {"value": 0.0}

        assert tracker._is_similar_params(params1, params2) is True

    def test_very_small_values_near_zero(self, tmp_variation_tracker):
        """极小值（接近 ZERO_THRESHOLD）的处理"""
        tracker = tmp_variation_tracker
        zero_threshold = tracker.ZERO_THRESHOLD  # 1e-10

        # 一个在阈值以下，一个为零
        params1 = {"value": zero_threshold / 2}
        params2 = {"value": 0.0}

        # 应该视为相似
        assert tracker._is_similar_params(params1, params2) is True

    def test_different_keys_not_similar(self, tmp_variation_tracker):
        """参数键不同应视为不相似"""
        tracker = tmp_variation_tracker

        params1 = {"fast": 10, "slow": 30}
        params2 = {"fast": 10, "medium": 20}

        assert tracker._is_similar_params(params1, params2) is False

    def test_non_numeric_params_exact_match(self, tmp_variation_tracker):
        """非数值参数必须完全相等"""
        tracker = tmp_variation_tracker

        params1 = {"mode": "strict", "value": 10}
        params2 = {"mode": "strict", "value": 10}

        assert tracker._is_similar_params(params1, params2) is True

    def test_non_numeric_params_different_not_similar(self, tmp_variation_tracker):
        """非数值参数不同应视为不相似"""
        tracker = tmp_variation_tracker

        params1 = {"mode": "strict", "value": 10}
        params2 = {"mode": "loose", "value": 10}

        assert tracker._is_similar_params(params1, params2) is False

    def test_mixed_params_similarity(self, tmp_variation_tracker):
        """混合参数的相似度检测"""
        tracker = tmp_variation_tracker

        # 数值参数在 5% 内，非数值参数相同
        params1 = {
            "fast": 10,
            "mode": "strict"
        }
        params2 = {
            "fast": 10.4,  # 4% 差异
            "mode": "strict"
        }

        assert tracker._is_similar_params(params1, params2) is True


# ========== TestUpdateFromExperiment ==========

class TestUpdateFromExperiment:
    """从实验结果更新状态测试"""

    def test_grade_a_to_passed(self, tmp_variation_tracker, sample_metrics):
        """Grade A 应转换为 PASSED"""
        tracker = tmp_variation_tracker

        # 注册变化
        var_hash = tracker.register_variation(
            "ma_cross", "trend", {"fast": 10, "slow": 30}
        )

        # 更新为 Grade A
        tracker.update_from_experiment(
            var_hash, "exp_001", "A", sample_metrics
        )

        record = tracker.variations[var_hash]
        assert record.status == VariationStatus.PASSED
        assert record.grade == "A"
        assert record.experiment_id == "exp_001"

    def test_grade_b_to_optimizable(self, tmp_variation_tracker, sample_metrics):
        """Grade B 应转换为 OPTIMIZABLE"""
        tracker = tmp_variation_tracker

        var_hash = tracker.register_variation(
            "ma_cross", "trend", {"fast": 10, "slow": 30}
        )

        tracker.update_from_experiment(
            var_hash, "exp_002", "B", sample_metrics
        )

        record = tracker.variations[var_hash]
        assert record.status == VariationStatus.OPTIMIZABLE
        assert record.grade == "B"

    def test_grade_c_to_failed(self, tmp_variation_tracker, sample_metrics):
        """Grade C 应转换为 FAILED"""
        tracker = tmp_variation_tracker

        var_hash = tracker.register_variation(
            "ma_cross", "trend", {"fast": 10, "slow": 30}
        )

        tracker.update_from_experiment(
            var_hash, "exp_003", "C", sample_metrics
        )

        record = tracker.variations[var_hash]
        assert record.status == VariationStatus.FAILED
        assert record.grade == "C"

    def test_grade_d_to_failed(self, tmp_variation_tracker, sample_metrics):
        """Grade D 应转换为 FAILED"""
        tracker = tmp_variation_tracker

        var_hash = tracker.register_variation(
            "ma_cross", "trend", {"fast": 10, "slow": 30}
        )

        tracker.update_from_experiment(
            var_hash, "exp_004", "D", sample_metrics
        )

        assert tracker.variations[var_hash].status == VariationStatus.FAILED

    def test_grade_f_to_failed(self, tmp_variation_tracker, sample_metrics):
        """Grade F 应转换为 FAILED"""
        tracker = tmp_variation_tracker

        var_hash = tracker.register_variation(
            "ma_cross", "trend", {"fast": 10, "slow": 30}
        )

        tracker.update_from_experiment(
            var_hash, "exp_005", "F", sample_metrics
        )

        assert tracker.variations[var_hash].status == VariationStatus.FAILED

    def test_update_sets_tested_at(self, tmp_variation_tracker, sample_metrics):
        """更新应设置 tested_at 时间戳"""
        tracker = tmp_variation_tracker

        var_hash = tracker.register_variation(
            "ma_cross", "trend", {"fast": 10, "slow": 30}
        )

        before = datetime.now()
        tracker.update_from_experiment(
            var_hash, "exp_006", "A", sample_metrics
        )
        after = datetime.now()

        record = tracker.variations[var_hash]
        assert record.tested_at is not None
        assert before <= record.tested_at <= after

    def test_update_with_failure_reason(self, tmp_variation_tracker, sample_metrics):
        """FAILED 状态可以设置失败原因"""
        tracker = tmp_variation_tracker

        var_hash = tracker.register_variation(
            "ma_cross", "trend", {"fast": 10, "slow": 30}
        )

        tracker.update_from_experiment(
            var_hash, "exp_007", "F", sample_metrics,
            failure_reason="Strategy underperforms on BTC 4h"
        )

        record = tracker.variations[var_hash]
        assert record.failure_reason == "Strategy underperforms on BTC 4h"

    def test_update_nonexistent_variation(self, tmp_variation_tracker, sample_metrics):
        """更新不存在的变化应记录警告（不应抛异常）"""
        tracker = tmp_variation_tracker

        # 应该不会抛异常
        tracker.update_from_experiment(
            "var_nonexistent", "exp_008", "A", sample_metrics
        )

        # 变化应仍不存在
        assert "var_nonexistent" not in tracker.variations


# ========== TestGetFinalBacktestList ==========

class TestGetFinalBacktestList:
    """最终回测清单过滤与排序测试"""

    def test_filter_by_status_only_passed_optimizable(self, tmp_variation_tracker):
        """只返回 PASSED 和 OPTIMIZABLE 状态的变化"""
        tracker = tmp_variation_tracker

        # 创建各种状态的变化
        passed_hash = tracker.register_variation("s1", "trend", {"p": 1})
        optimizable_hash = tracker.register_variation("s2", "trend", {"p": 2})
        failed_hash = tracker.register_variation("s3", "trend", {"p": 3})
        untested_hash = tracker.register_variation("s4", "trend", {"p": 4})

        # 更新状态和指标
        metrics = {"sharpe_ratio": 2.0, "total_return": 0.3, "max_drawdown": 0.1}

        tracker.update_from_experiment(passed_hash, "e1", "A", metrics)
        tracker.update_from_experiment(optimizable_hash, "e2", "B", metrics)
        tracker.update_from_experiment(failed_hash, "e3", "C", metrics)
        # untested_hash 不更新

        result = tracker.get_final_backtest_list()

        assert len(result) == 2
        assert passed_hash in [r.variation_hash for r in result]
        assert optimizable_hash in [r.variation_hash for r in result]

    def test_filter_by_sharpe_min_threshold(self, tmp_variation_tracker):
        """Sharpe < min_sharpe 的变化应被过滤"""
        tracker = tmp_variation_tracker

        # Sharpe 2.5
        high_sharpe_hash = tracker.register_variation("s1", "trend", {"p": 1})
        # Sharpe 1.2
        low_sharpe_hash = tracker.register_variation("s2", "trend", {"p": 2})

        metrics_high = {"sharpe_ratio": 2.5, "total_return": 0.3}
        metrics_low = {"sharpe_ratio": 1.2, "total_return": 0.1}

        tracker.update_from_experiment(high_sharpe_hash, "e1", "A", metrics_high)
        tracker.update_from_experiment(low_sharpe_hash, "e2", "A", metrics_low)

        # min_sharpe = 1.5
        result = tracker.get_final_backtest_list(min_sharpe=1.5)

        assert len(result) == 1
        assert result[0].variation_hash == high_sharpe_hash

    def test_sort_by_sharpe_descending(self, tmp_variation_tracker):
        """结果应按 Sharpe 降序排列"""
        tracker = tmp_variation_tracker

        hashes = []
        sharpes = [2.5, 1.8, 2.0, 1.6]

        for i, sharpe in enumerate(sharpes):
            h = tracker.register_variation(f"s{i}", "trend", {"p": i})
            hashes.append(h)
            metrics = {"sharpe_ratio": sharpe, "total_return": 0.1}
            tracker.update_from_experiment(h, f"e{i}", "A", metrics)

        result = tracker.get_final_backtest_list(min_sharpe=1.5)

        # 应为 [2.5, 2.0, 1.8, 1.6] 降序
        result_sharpes = [r.metrics.get("sharpe_ratio") for r in result]
        assert result_sharpes == sorted(result_sharpes, reverse=True)

    def test_empty_result(self, tmp_variation_tracker):
        """没有符合条件的变化时返回空列表"""
        tracker = tmp_variation_tracker

        # 创建只有 FAILED 状态的变化
        failed_hash = tracker.register_variation("s1", "trend", {"p": 1})
        metrics = {"sharpe_ratio": 1.0}
        tracker.update_from_experiment(failed_hash, "e1", "C", metrics)

        result = tracker.get_final_backtest_list(min_sharpe=1.5)

        assert len(result) == 0

    def test_limit_results(self, tmp_variation_tracker):
        """limit 参数应限制返回数量"""
        tracker = tmp_variation_tracker

        # 创建 5 个 PASSED 变化
        for i in range(5):
            h = tracker.register_variation(f"s{i}", "trend", {"p": i})
            metrics = {"sharpe_ratio": 2.0 + i * 0.1}
            tracker.update_from_experiment(h, f"e{i}", "A", metrics)

        result = tracker.get_final_backtest_list(limit=3)

        assert len(result) == 3

    def test_missing_metrics_excluded(self, tmp_variation_tracker):
        """没有 metrics 的变化应被排除"""
        tracker = tmp_variation_tracker

        # 直接修改变化状态为 PASSED（不设置 metrics）
        var_hash = tracker.register_variation("s1", "trend", {"p": 1})
        record = tracker.variations[var_hash]
        record.status = VariationStatus.PASSED
        record.grade = "A"

        result = tracker.get_final_backtest_list()

        assert len(result) == 0


# ========== TestVariationRecord ==========

class TestVariationRecord:
    """变化记录的序列化与反序列化测试"""

    def test_record_to_dict(self):
        """VariationRecord 应正确序列化为字典"""
        record = VariationRecord(
            variation_hash="var_abc123",
            strategy_name="ma_cross",
            strategy_type="trend",
            params={"fast": 10, "slow": 30},
            status=VariationStatus.PASSED,
            grade="A",
            metrics={"sharpe_ratio": 2.0},
            tested_at=datetime(2024, 1, 15, 10, 30, 0)
        )

        data = record.to_dict()

        assert data["variation_hash"] == "var_abc123"
        assert data["strategy_name"] == "ma_cross"
        assert data["status"] == "PASSED"  # Enum 转换为字符串
        assert isinstance(data["tested_at"], str)  # datetime 转换为 ISO 格式

    def test_record_from_dict(self):
        """从字典反序列化应正确恢复 VariationRecord"""
        data = {
            "variation_hash": "var_abc123",
            "strategy_name": "ma_cross",
            "strategy_type": "trend",
            "params": {"fast": 10, "slow": 30},
            "status": "PASSED",
            "grade": "A",
            "metrics": {"sharpe_ratio": 2.0},
            "tested_at": "2024-01-15T10:30:00",
            "registered_at": "2024-01-14T10:30:00"
        }

        record = VariationRecord.from_dict(data)

        assert record.variation_hash == "var_abc123"
        assert record.status == VariationStatus.PASSED
        assert isinstance(record.tested_at, datetime)
        assert isinstance(record.registered_at, datetime)

    def test_record_roundtrip_serialization(self):
        """序列化 -> 反序列化 应恢复原始数据"""
        original = VariationRecord(
            variation_hash="var_test",
            strategy_name="rsi_strategy",
            strategy_type="momentum",
            params={"period": 14, "threshold": 30.5},
            status=VariationStatus.OPTIMIZABLE,
            grade="B",
            metrics={"sharpe_ratio": 1.8, "total_return": 0.25},
            tags=["btc", "4h"]
        )

        # 序列化 -> 反序列化
        data = original.to_dict()
        recovered = VariationRecord.from_dict(data)

        # 验证关键字段
        assert recovered.variation_hash == original.variation_hash
        assert recovered.status == original.status
        assert recovered.metrics == original.metrics
        assert recovered.tags == original.tags


# ========== TestRegisterVariation ==========

class TestRegisterVariation:
    """变化注册与去重测试"""

    def test_register_new_variation(self, tmp_variation_tracker):
        """注册新变化应成功"""
        tracker = tmp_variation_tracker

        var_hash = tracker.register_variation(
            "ma_cross", "trend",
            {"fast": 10, "slow": 30}
        )

        assert var_hash in tracker.variations
        assert tracker.variations[var_hash].status == VariationStatus.UNTESTED

    def test_register_duplicate_returns_same_hash(self, tmp_variation_tracker):
        """注册相同参数应返回相同的 hash"""
        tracker = tmp_variation_tracker

        params = {"fast": 10, "slow": 30}

        hash1 = tracker.register_variation("ma_cross", "trend", params)
        hash2 = tracker.register_variation("ma_cross", "trend", params)

        assert hash1 == hash2
        assert len(tracker.variations) == 1

    def test_register_with_tags(self, tmp_variation_tracker):
        """注册变化时可以添加标签"""
        tracker = tmp_variation_tracker

        var_hash = tracker.register_variation(
            "ma_cross", "trend",
            {"fast": 10, "slow": 30},
            tags=["btc", "4h"]
        )

        record = tracker.variations[var_hash]
        assert record.tags == ["btc", "4h"]

    def test_registered_at_timestamp(self, tmp_variation_tracker):
        """注册时应设置 registered_at 时间戳"""
        tracker = tmp_variation_tracker

        before = datetime.now()
        var_hash = tracker.register_variation(
            "ma_cross", "trend", {"fast": 10}
        )
        after = datetime.now()

        record = tracker.variations[var_hash]
        assert record.registered_at is not None
        assert before <= record.registered_at <= after


# ========== TestQueryMethods ==========

class TestQueryMethods:
    """查询方法测试"""

    def test_is_tested_untested_variation(self, tmp_variation_tracker):
        """未测试的变化应返回 False"""
        tracker = tmp_variation_tracker

        var_hash = tracker.register_variation("ma_cross", "trend", {"p": 1})

        assert tracker.is_tested(var_hash) is False

    def test_is_tested_tested_variation(self, tmp_variation_tracker):
        """已测试的变化应返回 True"""
        tracker = tmp_variation_tracker

        var_hash = tracker.register_variation("ma_cross", "trend", {"p": 1})
        metrics = {"sharpe_ratio": 2.0}
        tracker.update_from_experiment(var_hash, "e1", "A", metrics)

        assert tracker.is_tested(var_hash) is True

    def test_get_status(self, tmp_variation_tracker):
        """get_status 应返回变化的当前状态"""
        tracker = tmp_variation_tracker

        var_hash = tracker.register_variation("ma_cross", "trend", {"p": 1})

        assert tracker.get_status(var_hash) == VariationStatus.UNTESTED

        metrics = {"sharpe_ratio": 2.0}
        tracker.update_from_experiment(var_hash, "e1", "A", metrics)

        assert tracker.get_status(var_hash) == VariationStatus.PASSED

    def test_get_untested_variations_filter_by_strategy(self, tmp_variation_tracker):
        """get_untested_variations 应支持按策略名称过滤"""
        tracker = tmp_variation_tracker

        # 创建不同策略的未测试变化
        ma_hash = tracker.register_variation("ma_cross", "trend", {"p": 1})
        rsi_hash = tracker.register_variation("rsi_strategy", "momentum", {"p": 2})

        results = tracker.get_untested_variations(strategy_name="ma_cross")

        assert len(results) == 1
        assert results[0].variation_hash == ma_hash

    def test_find_similar_variations(self, tmp_variation_tracker):
        """find_similar_variations 应找到相似的已测试变化"""
        tracker = tmp_variation_tracker

        # 注册变化 1
        hash1 = tracker.register_variation("ma_cross", "trend", {"fast": 10})
        metrics = {"sharpe_ratio": 2.0}
        tracker.update_from_experiment(hash1, "e1", "A", metrics)

        # 查询相似的变化（4% 差异）
        similar_params = {"fast": 10.4}
        results = tracker.find_similar_variations(similar_params, "ma_cross")

        assert len(results) == 1
        assert results[0].variation_hash == hash1


# ========== TestPersistence ==========

class TestPersistence:
    """持久化与 I/O 测试"""

    def test_save_loads_json(self, tmp_variation_tracker):
        """保存的 JSON 文件应能正确载入"""
        tracker = tmp_variation_tracker

        # 注册并更新变化
        var_hash = tracker.register_variation(
            "ma_cross", "trend", {"fast": 10}
        )
        metrics = {"sharpe_ratio": 2.1}
        tracker.update_from_experiment(var_hash, "e1", "A", metrics)

        # 验证 JSON 文件存在
        assert tracker.json_path.exists()

        # 创建新 tracker，从文件载入
        new_tracker = VariationTracker(
            json_path=tracker.json_path,
            md_path=tracker.md_path
        )

        assert len(new_tracker.variations) == 1
        assert var_hash in new_tracker.variations
        assert new_tracker.variations[var_hash].status == VariationStatus.PASSED

    def test_save_generates_markdown(self, tmp_variation_tracker):
        """保存时应生成 Markdown 报告"""
        tracker = tmp_variation_tracker

        var_hash = tracker.register_variation(
            "ma_cross", "trend", {"fast": 10}
        )
        metrics = {"sharpe_ratio": 2.1}
        tracker.update_from_experiment(var_hash, "e1", "A", metrics)

        # 验证 Markdown 文件存在
        assert tracker.md_path.exists()

        # 验证内容包含必要部分
        content = tracker.md_path.read_text(encoding="utf-8")
        assert "策略变化追踪报告" in content or "策略變化追蹤報告" in content

    def test_json_contains_statistics(self, tmp_variation_tracker):
        """JSON 应包含统计信息"""
        tracker = tmp_variation_tracker

        var_hash = tracker.register_variation(
            "ma_cross", "trend", {"fast": 10}
        )
        metrics = {"sharpe_ratio": 2.1}
        tracker.update_from_experiment(var_hash, "e1", "A", metrics)

        # 读取 JSON 文件
        with open(tracker.json_path) as f:
            data = json.load(f)

        assert "statistics" in data
        assert "version" in data
        assert data["statistics"]["total"] == 1


# ========== TestEdgeCases ==========

class TestEdgeCases:
    """边界与特殊情况测试"""

    def test_empty_params(self, tmp_variation_tracker):
        """空参数字典应处理正确"""
        tracker = tmp_variation_tracker

        hash1 = tracker.compute_hash("strategy", {})
        hash2 = tracker.compute_hash("strategy", {})

        assert hash1 == hash2

    def test_very_large_param_values(self, tmp_variation_tracker):
        """大数值参数应处理正确"""
        tracker = tmp_variation_tracker

        params = {"max_iterations": 1000000, "threshold": 0.000001}
        var_hash = tracker.register_variation(
            "big_numbers", "momentum", params
        )

        assert var_hash in tracker.variations

    def test_negative_params(self, tmp_variation_tracker):
        """负数参数应处理正确"""
        tracker = tmp_variation_tracker

        params = {"stop_loss": -0.02, "atr_multiplier": -1.5}
        var_hash = tracker.register_variation(
            "negative_params", "risk", params
        )

        assert var_hash in tracker.variations

    def test_string_params(self, tmp_variation_tracker):
        """字符串参数应处理正确"""
        tracker = tmp_variation_tracker

        params = {
            "timeframe": "4h",
            "symbol": "BTC/USDT",
            "mode": "aggressive"
        }
        var_hash = tracker.register_variation(
            "string_params", "trend", params
        )

        assert var_hash in tracker.variations

    def test_boolean_params(self, tmp_variation_tracker):
        """布尔参数应处理正确"""
        tracker = tmp_variation_tracker

        params = {
            "use_leverage": True,
            "hedge_enabled": False
        }
        var_hash = tracker.register_variation(
            "bool_params", "trend", params
        )

        assert var_hash in tracker.variations

    def test_mixed_numeric_types(self, tmp_variation_tracker):
        """混合 int 和 float 参数可以正常处理"""
        tracker = tmp_variation_tracker

        params = {
            "fast_period": 10,  # int
            "slow_period": 30.5  # float
        }

        # 应该能正常生成 hash（不会抛异常）
        hash1 = tracker.compute_hash("ma_cross", params)

        assert hash1.startswith("var_")
        assert len(hash1) == 4 + tracker.HASH_LENGTH

        # 注册变化也应该成功
        var_hash = tracker.register_variation("ma_cross", "trend", params)
        assert var_hash in tracker.variations

    def test_none_value_in_params(self, tmp_variation_tracker):
        """None 值参数应处理正确"""
        tracker = tmp_variation_tracker

        params = {"threshold": 0.05, "backup": None}
        var_hash = tracker.register_variation(
            "none_params", "trend", params
        )

        assert var_hash in tracker.variations

    def test_unicode_strategy_name(self, tmp_variation_tracker):
        """Unicode 策略名称应处理正确"""
        tracker = tmp_variation_tracker

        var_hash = tracker.register_variation(
            "策略_ma_cross", "trend",
            {"fast": 10}
        )

        assert var_hash in tracker.variations
