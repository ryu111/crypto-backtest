"""
DuckDB 整合測試

測試項目：
1. 端到端測試：使用 ExperimentRecorder 記錄實驗並查詢
2. 組件整合測試：Repository + ExperimentRecord + InsightsManager
3. 效能測試：插入和查詢效能
4. 驗證檢查清單：遷移後資料一致性
"""

import time
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

import pytest

from src.learning.recorder import ExperimentRecorder
from src.learning.insights import InsightsManager
from src.db.repository import Repository, QueryFilters
from src.types import ExperimentRecord


class TestEndToEnd:
    """端到端測試：完整的記錄和查詢流程"""

    @pytest.fixture
    def temp_dirs(self):
        """建立臨時目錄"""
        # 使用專案根目錄下的臨時目錄（避免 _validate_path 錯誤）
        project_root = Path(__file__).parent.parent
        test_dir = project_root / 'data' / 'test_e2e'
        test_dir.mkdir(parents=True, exist_ok=True)

        db_path = test_dir / 'test_experiments.duckdb'
        insights_file = test_dir / 'test_insights.md'

        yield db_path, insights_file, test_dir

        # 清理
        if db_path.exists():
            db_path.unlink()
        if insights_file.exists():
            insights_file.unlink()
        if test_dir.exists() and not any(test_dir.iterdir()):
            test_dir.rmdir()

    def test_record_and_query_experiment(self, temp_dirs):
        """測試：記錄實驗並查詢"""
        db_path, insights_file, _ = temp_dirs

        # 建立 mock BacktestResult
        class MockBacktestResult:
            def __init__(self):
                self.total_return = 0.35
                self.annual_return = 0.52
                self.sharpe_ratio = 1.8
                self.sortino_ratio = 2.1
                self.max_drawdown = 0.12
                self.win_rate = 0.62
                self.profit_factor = 2.3
                self.total_trades = 150
                self.avg_trade_duration = 4.2
                self.expectancy = 0.0023
                self.params = {'fast': 10, 'slow': 30}

        # 建立 mock ValidationResult
        class MockStageResult:
            def __init__(self, passed, score, message):
                self.passed = passed
                self.score = score
                self.message = message
                self.details = {}

        class MockValidationResult:
            def __init__(self):
                self.grade = MockGrade('B')
                self.passed_stages = [1, 2, 3]
                self.stage_results = {
                    '階段1_基礎績效': MockStageResult(True, 0.85, 'Passed'),
                    '階段2_統計檢驗': MockStageResult(True, 0.78, 'Passed'),
                    '階段3_穩健性': MockStageResult(True, 0.72, 'Passed'),
                }

        class MockGrade:
            def __init__(self, value):
                self.value = value

        # 使用 ExperimentRecorder
        with ExperimentRecorder(db_path=db_path, insights_file=insights_file) as recorder:
            # 記錄實驗
            exp_id = recorder.log_experiment(
                result=MockBacktestResult(),
                strategy_info={'name': 'ma_cross', 'type': 'trend', 'version': '1.0'},
                config={'symbol': 'BTCUSDT', 'timeframe': '4h'},
                validation_result=MockValidationResult(),
                insights=['MA Cross 在 BTC 4h 表現優異']
            )

            assert exp_id is not None, "實驗 ID 應該生成"
            assert exp_id.startswith('exp_'), "實驗 ID 格式錯誤"

            # 查詢實驗
            retrieved = recorder.get_experiment(exp_id)
            assert retrieved is not None, "應該能查詢到實驗"
            assert retrieved.id == exp_id, "實驗 ID 不匹配"
            assert retrieved.sharpe_ratio == 1.8, "Sharpe ratio 不匹配"
            assert retrieved.strategy_name == 'ma_cross', "策略名稱不匹配"
            assert retrieved.grade == 'B', "評級不匹配"

        # 驗證 insights.md 已更新
        assert insights_file.exists(), "insights.md 應該存在"
        content = insights_file.read_text(encoding='utf-8')
        assert 'ma_cross' in content, "insights.md 應包含策略名稱"
        assert '1.8' in content or '1.80' in content, "insights.md 應包含 Sharpe ratio"

    def test_query_with_filters(self, temp_dirs):
        """測試：使用 filters 查詢實驗"""
        db_path, insights_file, _ = temp_dirs

        class MockBacktestResult:
            def __init__(self, sharpe):
                self.total_return = 0.20
                self.annual_return = 0.30
                self.sharpe_ratio = sharpe
                self.sortino_ratio = sharpe + 0.2
                self.max_drawdown = 0.10
                self.win_rate = 0.55
                self.profit_factor = 1.8
                self.total_trades = 100
                self.avg_trade_duration = 3.0
                self.expectancy = 0.002
                self.params = {}

        # 插入多個實驗
        with ExperimentRecorder(db_path=db_path, insights_file=insights_file) as recorder:
            for i, sharpe in enumerate([0.8, 1.5, 2.1, 1.2]):
                recorder.log_experiment(
                    result=MockBacktestResult(sharpe),
                    strategy_info={'name': f'strategy_{i}', 'type': 'trend'},
                    config={'symbol': 'BTCUSDT', 'timeframe': '4h'},
                )

            # 查詢 Sharpe >= 1.5 的實驗
            results = recorder.query_experiments({'min_sharpe': 1.5})
            assert len(results) == 2, f"應該有 2 個實驗符合條件，實際: {len(results)}"

            sharpe_values = [r.sharpe_ratio for r in results]
            assert all(s >= 1.5 for s in sharpe_values), "所有結果的 Sharpe 應 >= 1.5"


class TestComponentIntegration:
    """組件整合測試"""

    @pytest.fixture
    def temp_dirs(self):
        """建立臨時目錄"""
        project_root = Path(__file__).parent.parent
        test_dir = project_root / 'data' / 'test_components'
        test_dir.mkdir(parents=True, exist_ok=True)

        db_path = test_dir / 'test_experiments.duckdb'
        insights_file = test_dir / 'test_insights.md'

        yield db_path, insights_file, test_dir

        # 清理
        if db_path.exists():
            db_path.unlink()
        if insights_file.exists():
            insights_file.unlink()
        if test_dir.exists() and not any(test_dir.iterdir()):
            test_dir.rmdir()

    def test_repository_experiment_record_integration(self, temp_dirs):
        """測試：Repository + ExperimentRecord 整合"""
        db_path, _, _ = temp_dirs

        # 建立測試實驗
        experiment = ExperimentRecord(
            id='test_001',
            timestamp=datetime.now(),
            strategy={'name': 'test_strategy', 'type': 'trend', 'params': {'x': 1}},
            config={'symbol': 'BTCUSDT', 'timeframe': '1h'},
            results={
                'sharpe_ratio': 1.5,
                'total_return': 0.25,
                'max_drawdown': 0.10,
            },
            validation={'grade': 'B', 'stages_passed': [1, 2, 3]},
            status='completed',
        )

        # 使用 Repository
        with Repository(str(db_path)) as repo:
            repo.insert_experiment(experiment)

            # 查詢
            retrieved = repo.get_experiment('test_001')
            assert retrieved is not None, "應該能查詢到實驗"
            assert retrieved.id == 'test_001'
            assert retrieved.sharpe_ratio == 1.5
            assert retrieved.params == {'x': 1}, "參數應該正確解析"

    def test_insights_manager_integration(self, temp_dirs):
        """測試：InsightsManager + ExperimentRecord 整合"""
        _, insights_file, _ = temp_dirs

        # 建立測試實驗
        experiment = ExperimentRecord(
            id='test_002',
            timestamp=datetime.now(),
            strategy={'name': 'rsi_strategy', 'type': 'momentum', 'params': {'period': 14}},
            config={'symbol': 'ETHUSDT', 'timeframe': '1h'},
            results={
                'sharpe_ratio': 2.0,
                'total_return': 0.40,
                'max_drawdown': 0.08,
            },
            validation={'grade': 'A', 'stages_passed': [1, 2, 3, 4]},
            status='completed',
            insights=['RSI 策略在震盪市場表現優異'],
        )

        # 使用 InsightsManager
        manager = InsightsManager(insights_file)
        manager.update(experiment, total_experiments=1)

        # 驗證
        content = insights_file.read_text(encoding='utf-8')
        assert 'rsi_strategy' in content, "應包含策略名稱"
        assert '總實驗數：1' in content, "應更新總實驗數"
        assert 'RSI 策略在震盪市場表現優異' in content, "應包含洞察"


class TestPerformance:
    """效能測試"""

    @pytest.fixture
    def temp_dirs(self):
        """建立臨時目錄"""
        project_root = Path(__file__).parent.parent
        test_dir = project_root / 'data' / 'test_performance'
        test_dir.mkdir(parents=True, exist_ok=True)

        db_path = test_dir / 'test_experiments.duckdb'
        insights_file = test_dir / 'test_insights.md'

        yield db_path, insights_file, test_dir

        # 清理
        if db_path.exists():
            db_path.unlink()
        if insights_file.exists():
            insights_file.unlink()
        if test_dir.exists() and not any(test_dir.iterdir()):
            test_dir.rmdir()

    def test_insert_100_experiments_performance(self, temp_dirs):
        """測試：插入 100 筆實驗的效能"""
        db_path, insights_file, _ = temp_dirs

        class MockBacktestResult:
            def __init__(self, idx):
                self.total_return = 0.20 + idx * 0.01
                self.annual_return = 0.30
                self.sharpe_ratio = 1.0 + idx * 0.1
                self.sortino_ratio = 1.2
                self.max_drawdown = 0.10
                self.win_rate = 0.55
                self.profit_factor = 1.8
                self.total_trades = 100
                self.avg_trade_duration = 3.0
                self.expectancy = 0.002
                self.params = {'param': idx}

        start_time = time.time()

        with ExperimentRecorder(db_path=db_path, insights_file=insights_file) as recorder:
            for i in range(100):
                recorder.log_experiment(
                    result=MockBacktestResult(i),
                    strategy_info={'name': f'strategy_{i}', 'type': 'trend'},
                    config={'symbol': 'BTCUSDT', 'timeframe': '4h'},
                )

        elapsed = time.time() - start_time
        print(f"\n插入 100 筆實驗耗時: {elapsed:.2f}s")
        assert elapsed < 30.0, f"插入 100 筆實驗應在 30 秒內完成（實際: {elapsed:.2f}s）"

    def test_query_performance(self, temp_dirs):
        """測試：查詢效能"""
        db_path, insights_file, _ = temp_dirs

        # 先插入資料
        with Repository(str(db_path)) as repo:
            for i in range(100):
                exp = ExperimentRecord(
                    id=f'test_{i:03d}',
                    timestamp=datetime.now() - timedelta(days=i),
                    strategy={'name': f'strategy_{i % 10}', 'type': 'trend', 'params': {}},
                    config={'symbol': 'BTCUSDT', 'timeframe': '4h'},
                    results={
                        'sharpe_ratio': 1.0 + i * 0.01,
                        'total_return': 0.20,
                        'max_drawdown': 0.10,
                    },
                    validation={'grade': 'B', 'stages_passed': [1, 2]},
                    status='completed',
                )
                repo.insert_experiment(exp)

        # 測試聚合查詢效能
        with Repository(str(db_path)) as repo:
            start = time.time()
            results = repo.get_best_experiments(metric='sharpe_ratio', n=10)
            elapsed = (time.time() - start) * 1000  # 轉換為 ms

            print(f"\n聚合查詢 (top 10) 耗時: {elapsed:.2f}ms")
            assert elapsed < 100, f"聚合查詢應在 100ms 內完成（實際: {elapsed:.2f}ms）"
            assert len(results) == 10, "應返回 10 筆結果"

        # 測試單筆查詢效能
        with Repository(str(db_path)) as repo:
            start = time.time()
            result = repo.get_experiment('test_050')
            elapsed = (time.time() - start) * 1000  # 轉換為 ms

            print(f"單筆查詢耗時: {elapsed:.2f}ms")
            assert elapsed < 10, f"單筆查詢應在 10ms 內完成（實際: {elapsed:.2f}ms）"
            assert result is not None, "應查詢到實驗"


class TestMigrationValidation:
    """遷移驗證測試"""

    @pytest.fixture
    def temp_dirs(self):
        """建立臨時目錄"""
        project_root = Path(__file__).parent.parent
        test_dir = project_root / 'data' / 'test_migration'
        test_dir.mkdir(parents=True, exist_ok=True)

        db_path = test_dir / 'test_experiments.duckdb'
        json_file = test_dir / 'experiments.json'
        insights_file = test_dir / 'test_insights.md'

        yield db_path, json_file, insights_file, test_dir

        # 清理
        if db_path.exists():
            db_path.unlink()
        if json_file.exists():
            json_file.unlink()
        if insights_file.exists():
            insights_file.unlink()
        if test_dir.exists() and not any(test_dir.iterdir()):
            test_dir.rmdir()

    def test_data_count_consistency(self, temp_dirs):
        """測試：遷移後資料筆數一致"""
        db_path, json_file, insights_file, _ = temp_dirs

        # 建立測試 JSON 檔案
        import json

        test_data = {
            'experiments': [
                {
                    'id': f'exp_{i:03d}',
                    'timestamp': datetime.now().isoformat(),
                    'strategy': {'name': f'strategy_{i}', 'type': 'trend', 'params': {}},
                    'config': {'symbol': 'BTCUSDT', 'timeframe': '4h'},
                    'results': {'sharpe_ratio': 1.0, 'total_return': 0.2, 'max_drawdown': 0.1},
                    'validation': {'grade': 'B', 'stages_passed': [1, 2]},
                    'status': 'completed',
                }
                for i in range(50)
            ]
        }

        json_file.write_text(json.dumps(test_data), encoding='utf-8')

        # 執行遷移
        with ExperimentRecorder(
            db_path=db_path,
            insights_file=insights_file,
            experiments_file=json_file
        ) as recorder:
            # 檢查遷移結果
            all_experiments = recorder.query_experiments({})
            assert len(all_experiments) == 50, f"遷移後應有 50 筆實驗，實際: {len(all_experiments)}"

            # 驗證 JSON 已備份
            backup_path = json_file.with_suffix('.json.migrated')
            assert backup_path.exists(), "應建立 JSON 備份檔案"

    def test_export_to_json(self, temp_dirs):
        """測試：匯出到 JSON"""
        db_path, _, insights_file, test_dir = temp_dirs

        # 插入資料
        with ExperimentRecorder(db_path=db_path, insights_file=insights_file) as recorder:
            for i in range(10):
                exp = ExperimentRecord(
                    id=f'exp_{i:03d}',
                    timestamp=datetime.now(),
                    strategy={'name': f'strategy_{i}', 'type': 'trend', 'params': {}},
                    config={'symbol': 'BTCUSDT', 'timeframe': '4h'},
                    results={'sharpe_ratio': 1.0, 'total_return': 0.2, 'max_drawdown': 0.1},
                    validation={'grade': 'B', 'stages_passed': [1, 2]},
                    status='completed',
                )
                recorder.repo.insert_experiment(exp)

            # 匯出
            output_file = test_dir / 'export.json'
            result_path = recorder.export_to_json(output_file)

            assert result_path.exists(), "應建立匯出檔案"

            # 驗證內容
            import json
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            assert len(data['experiments']) == 10, "應匯出 10 筆實驗"
            assert data['metadata']['total_experiments'] == 10, "metadata 應正確"

        # 清理
        if output_file.exists():
            output_file.unlink()


def main():
    """執行測試"""
    pytest.main([__file__, '-v', '-s'])


if __name__ == '__main__':
    main()
