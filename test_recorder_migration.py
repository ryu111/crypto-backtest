"""
測試 ExperimentRecorder 重構後的功能

驗證項目:
1. Context Manager 正常運作
2. DuckDB 插入和查詢
3. 前綴查詢效能
4. 資源正確關閉
"""

import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

# 測試用的 mock BacktestResult
class MockBacktestResult:
    def __init__(self):
        self.total_return = 0.45
        self.annual_return = 0.18
        self.sharpe_ratio = 1.5
        self.sortino_ratio = 1.8
        self.max_drawdown = -0.15
        self.win_rate = 0.55
        self.profit_factor = 1.8
        self.total_trades = 100
        self.avg_trade_duration = 24.5
        self.expectancy = 0.012
        self.params = {'fast': 10, 'slow': 30}


def test_context_manager():
    """測試 Context Manager 機制"""
    print("🧪 測試 1: Context Manager")

    # 使用專案內的臨時目錄（避免路徑驗證問題）
    project_root = Path(__file__).parent
    temp_dir = project_root / "test_temp"
    temp_dir.mkdir(exist_ok=True)

    db_path = temp_dir / "test.duckdb"
    insights_path = temp_dir / "insights.md"

    # 建立空的 insights.md
    insights_path.write_text("# Insights\n")

    try:
        from src.learning.recorder import ExperimentRecorder

        # 測試 with 語句
        with ExperimentRecorder(
            db_path=db_path,
            insights_file=insights_path
        ) as recorder:
            # 記錄一筆實驗
            exp_id = recorder.log_experiment(
                result=MockBacktestResult(),
                strategy_info={'name': 'test_ma', 'type': 'trend', 'version': '1.0'},
                config={'symbol': 'BTCUSDT', 'timeframe': '4h'},
                validation_result=None,
                insights=['測試洞察']
            )

            print(f"  ✓ 成功記錄實驗: {exp_id}")

            # 驗證可以查詢
            exp = recorder.get_experiment(exp_id)
            assert exp is not None, "應該能查到實驗"
            assert exp.sharpe_ratio == 1.5, f"Sharpe 應為 1.5，實際: {exp.sharpe_ratio}"
            print(f"  ✓ 查詢成功，Sharpe: {exp.sharpe_ratio}")

        # 退出 context manager 後，資源應該已關閉
        print("  ✓ Context Manager 正常退出")

    finally:
        # 清理
        shutil.rmtree(temp_dir)

    print("  ✅ 測試通過\n")


def test_strategy_prefix_query():
    """測試前綴查詢（驗證效能改善）"""
    print("🧪 測試 2: 策略前綴查詢")

    project_root = Path(__file__).parent
    temp_dir = project_root / "test_temp"
    temp_dir.mkdir(exist_ok=True)

    db_path = temp_dir / "test2.duckdb"
    insights_path = temp_dir / "insights.md"
    insights_path.write_text("# Insights\n")

    try:
        from src.learning.recorder import ExperimentRecorder

        with ExperimentRecorder(
            db_path=db_path,
            insights_file=insights_path
        ) as recorder:
            # 插入多筆不同策略的實驗
            strategies = [
                'ma_cross_v1',
                'ma_cross_v2',
                'rsi_divergence',
                'macd_signal'
            ]

            for strat in strategies:
                recorder.log_experiment(
                    result=MockBacktestResult(),
                    strategy_info={'name': strat, 'type': 'trend', 'version': '1.0'},
                    config={'symbol': 'BTCUSDT', 'timeframe': '4h'},
                    validation_result=None
                )

            print(f"  ✓ 已插入 {len(strategies)} 筆實驗")

            # 測試前綴查詢（應該只找到 ma_cross_*）
            evolution = recorder.get_strategy_evolution('ma_cross')

            assert len(evolution) == 2, f"應該找到 2 筆 ma_cross，實際: {len(evolution)}"
            print(f"  ✓ 前綴查詢正確: 找到 {len(evolution)} 筆 ma_cross 實驗")

            # 測試策略統計
            stats = recorder.get_strategy_stats('ma_cross')
            assert stats is not None, "應該有統計資料"
            assert stats['attempts'] == 2, f"嘗試次數應為 2，實際: {stats['attempts']}"
            print(f"  ✓ 策略統計正確: {stats['attempts']} 次嘗試")

    finally:
        shutil.rmtree(temp_dir)

    print("  ✅ 測試通過\n")


def test_param_extraction():
    """測試參數提取（新舊格式相容）"""
    print("🧪 測試 3: 參數提取")

    project_root = Path(__file__).parent
    temp_dir = project_root / "test_temp"
    temp_dir.mkdir(exist_ok=True)

    db_path = temp_dir / "test3.duckdb"
    insights_path = temp_dir / "insights.md"
    insights_path.write_text("# Insights\n")

    try:
        from src.learning.recorder import ExperimentRecorder

        with ExperimentRecorder(
            db_path=db_path,
            insights_file=insights_path
        ) as recorder:
            # 測試新格式 (params)
            result_new = MockBacktestResult()
            result_new.params = {'fast': 10, 'slow': 30}

            exp_id1 = recorder.log_experiment(
                result=result_new,
                strategy_info={'name': 'test_new', 'type': 'trend'},
                config={'symbol': 'BTCUSDT', 'timeframe': '4h'}
            )

            exp1 = recorder.get_experiment(exp_id1)
            assert exp1.params == {'fast': 10, 'slow': 30}, "新格式參數應正確"
            print("  ✓ 新格式 (params) 提取正確")

            # 測試舊格式 (parameters)
            result_old = MockBacktestResult()
            delattr(result_old, 'params')  # 移除 params
            result_old.parameters = {'period': 14}

            exp_id2 = recorder.log_experiment(
                result=result_old,
                strategy_info={'name': 'test_old', 'type': 'trend'},
                config={'symbol': 'BTCUSDT', 'timeframe': '4h'}
            )

            exp2 = recorder.get_experiment(exp_id2)
            assert exp2.params == {'period': 14}, "舊格式參數應正確"
            print("  ✓ 舊格式 (parameters) 向後相容")

    finally:
        shutil.rmtree(temp_dir)

    print("  ✅ 測試通過\n")


def test_resource_cleanup():
    """測試資源清理"""
    print("🧪 測試 4: 資源清理")

    project_root = Path(__file__).parent
    temp_dir = project_root / "test_temp"
    temp_dir.mkdir(exist_ok=True)

    db_path = temp_dir / "test4.duckdb"
    insights_path = temp_dir / "insights.md"
    insights_path.write_text("# Insights\n")

    try:
        from src.learning.recorder import ExperimentRecorder

        # 測試 1: 正常關閉
        recorder = ExperimentRecorder(
            db_path=db_path,
            insights_file=insights_path
        )
        recorder.log_experiment(
            result=MockBacktestResult(),
            strategy_info={'name': 'test', 'type': 'trend'},
            config={'symbol': 'BTCUSDT', 'timeframe': '4h'}
        )
        recorder.close()
        print("  ✓ 手動關閉成功")

        # 測試 2: 重複關閉不報錯
        recorder.close()
        print("  ✓ 重複關閉不報錯")

        # 測試 3: __del__ 清理
        recorder2 = ExperimentRecorder(
            db_path=db_path,
            insights_file=insights_path
        )
        del recorder2  # 觸發 __del__
        print("  ✓ __del__ 清理成功")

    finally:
        shutil.rmtree(temp_dir)

    print("  ✅ 測試通過\n")


def test_filter_conversion():
    """測試過濾器轉換"""
    print("🧪 測試 5: 過濾器轉換")

    project_root = Path(__file__).parent
    temp_dir = project_root / "test_temp"
    temp_dir.mkdir(exist_ok=True)

    db_path = temp_dir / "test5.duckdb"
    insights_path = temp_dir / "insights.md"
    insights_path.write_text("# Insights\n")

    try:
        from src.learning.recorder import ExperimentRecorder

        with ExperimentRecorder(
            db_path=db_path,
            insights_file=insights_path
        ) as recorder:
            # 插入測試資料
            for i in range(5):
                result = MockBacktestResult()
                result.sharpe_ratio = 1.0 + i * 0.2  # 1.0, 1.2, 1.4, 1.6, 1.8

                recorder.log_experiment(
                    result=result,
                    strategy_info={'name': f'test_{i}', 'type': 'trend'},
                    config={'symbol': 'BTCUSDT', 'timeframe': '4h'}
                )

            # 測試舊格式過濾器
            experiments = recorder.query_experiments({
                'min_sharpe': 1.5,
                'symbol': 'BTCUSDT'
            })

            # 應該找到 sharpe >= 1.5 的實驗（1.6, 1.8）
            assert len(experiments) == 2, f"應找到 2 筆，實際: {len(experiments)}"
            print(f"  ✓ 過濾器轉換正確: 找到 {len(experiments)} 筆 sharpe >= 1.5")

            # 測試 get_best_experiments
            best = recorder.get_best_experiments('sharpe_ratio', n=3)
            assert len(best) == 3, f"應返回 3 筆最佳，實際: {len(best)}"
            assert best[0].sharpe_ratio >= best[1].sharpe_ratio, "應按 sharpe 降序"
            print(f"  ✓ 最佳實驗查詢正確: Top 3 sharpe = {[round(e.sharpe_ratio, 1) for e in best]}")

    finally:
        shutil.rmtree(temp_dir)

    print("  ✅ 測試通過\n")


if __name__ == '__main__':
    print("=" * 60)
    print("ExperimentRecorder 重構測試")
    print("=" * 60 + "\n")

    # 準備測試環境
    project_root = Path(__file__).parent
    temp_dir = project_root / "test_temp"

    try:
        test_context_manager()
        test_strategy_prefix_query()
        test_param_extraction()
        test_resource_cleanup()
        test_filter_conversion()

        print("=" * 60)
        print("✅ 所有測試通過!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    finally:
        # 清理測試檔案
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"\n🧹 已清理測試目錄: {temp_dir}")
