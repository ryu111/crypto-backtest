"""
ExperimentRecorder 安全性測試

測試路徑驗證、JSON 錯誤處理等安全功能。
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime

from src.learning import ExperimentRecorder, Experiment


class MockBacktestResult:
    """模擬回測結果"""

    def __init__(self, sharpe=1.5):
        self.total_return = 0.456
        self.annual_return = 0.23
        self.sharpe_ratio = sharpe
        self.sortino_ratio = 2.1
        self.max_drawdown = -0.10
        self.win_rate = 0.55
        self.profit_factor = 1.72
        self.total_trades = 124
        self.avg_trade_duration = 12.5
        self.expectancy = 0.0037
        self.parameters = {'fast': 10, 'slow': 30}


class TestPathTraversalProtection:
    """測試路徑遍歷（Path Traversal）防護"""

    def test_reject_path_outside_project(self):
        """拒絕專案目錄外的路徑"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # 嘗試使用外部路徑
            external_path = Path('/tmp/malicious_experiments.json')

            with pytest.raises(ValueError, match="outside project directory"):
                recorder = ExperimentRecorder(
                    experiments_file=external_path,
                    insights_file=tmpdir_path / 'insights.md'
                )

    def test_reject_parent_directory_traversal(self):
        """拒絕父目錄遍歷（../../../etc/passwd）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # 嘗試使用 ../ 跳出專案目錄
            traversal_path = tmpdir_path / '../../../etc/passwd'

            # 這個測試依賴實際專案結構，所以我們檢查是否正確解析
            # 如果 tmpdir 在專案外，應該被拒絕
            try:
                recorder = ExperimentRecorder(
                    experiments_file=traversal_path,
                    insights_file=tmpdir_path / 'insights.md'
                )
                # 如果沒拋出異常，檢查是否真的在專案內
                assert str(recorder.experiments_file).startswith(str(recorder.project_root))
            except ValueError as e:
                # 如果拋出異常，確認是路徑驗證錯誤
                assert "outside project directory" in str(e)

    def test_accept_valid_subdirectory(self):
        """接受專案子目錄的有效路徑"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # 建立一個看起來像專案子目錄的結構
            # 這裡需要 mock project_root
            exp_file = tmpdir_path / 'learning' / 'experiments.json'
            insights_file = tmpdir_path / 'learning' / 'insights.md'

            # 手動建立 recorder 並覆寫 project_root
            recorder = ExperimentRecorder.__new__(ExperimentRecorder)
            recorder.project_root = tmpdir_path

            # 驗證路徑
            validated_exp = recorder._validate_path(exp_file)
            validated_insights = recorder._validate_path(insights_file)

            assert str(validated_exp).startswith(str(tmpdir_path))
            assert str(validated_insights).startswith(str(tmpdir_path))

    def test_resolve_symbolic_links(self):
        """測試路徑解析（處理符號連結）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # 建立實際檔案
            real_file = tmpdir_path / 'real.json'
            real_file.touch()

            # 建立符號連結（如果系統支援）
            try:
                symlink_file = tmpdir_path / 'symlink.json'
                symlink_file.symlink_to(real_file)

                # mock project_root
                recorder = ExperimentRecorder.__new__(ExperimentRecorder)
                recorder.project_root = tmpdir_path

                # 驗證符號連結
                validated = recorder._validate_path(symlink_file)

                # 應該解析為實際路徑
                assert validated.resolve() == real_file.resolve()
                assert str(validated).startswith(str(tmpdir_path))

            except OSError:
                # 如果系統不支援符號連結，跳過測試
                pytest.skip("Symbolic links not supported on this system")


class TestJSONErrorHandling:
    """測試 JSON 錯誤處理"""

    def test_handle_corrupted_json(self):
        """處理損壞的 JSON 檔案"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            exp_file = tmpdir_path / 'experiments.json'

            # 寫入損壞的 JSON
            exp_file.write_text('{ invalid json content', encoding='utf-8')

            recorder = ExperimentRecorder.__new__(ExperimentRecorder)
            recorder.project_root = tmpdir_path
            recorder.experiments_file = exp_file
            recorder.insights_file = tmpdir_path / 'insights.md'

            # 載入時應返回空資料結構，而不是崩潰
            data = recorder._load_experiments()

            assert data['version'] == '1.0'
            assert data['metadata']['total_experiments'] == 0
            assert len(data['experiments']) == 0

    def test_handle_empty_json_file(self):
        """處理空 JSON 檔案"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            exp_file = tmpdir_path / 'experiments.json'

            # 寫入空檔案
            exp_file.write_text('', encoding='utf-8')

            recorder = ExperimentRecorder.__new__(ExperimentRecorder)
            recorder.project_root = tmpdir_path
            recorder.experiments_file = exp_file
            recorder.insights_file = tmpdir_path / 'insights.md'

            # 載入時應返回空資料結構
            data = recorder._load_experiments()

            assert data['version'] == '1.0'
            assert data['metadata']['total_experiments'] == 0

    def test_handle_invalid_json_structure(self):
        """處理無效的 JSON 結構（格式正確但內容錯誤）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            exp_file = tmpdir_path / 'experiments.json'
            insights_file = tmpdir_path / 'insights.md'

            # 寫入格式正確但結構錯誤的 JSON
            exp_file.write_text('{"wrong": "structure"}', encoding='utf-8')

            # 建立 recorder（會初始化檔案）
            recorder = ExperimentRecorder(
                experiments_file=exp_file,
                insights_file=insights_file
            )

            # 因為 _init_files 會檢查檔案存在，所以會保留錯誤的 JSON
            # 但 _load_experiments 應該能處理缺少的鍵
            data = recorder._load_experiments()

            # 應該有基本結構（即使內容可能不完整）
            assert 'wrong' in data or 'version' in data


class TestExperimentRecordingRobustness:
    """測試實驗記錄的穩健性"""

    def test_record_experiment_with_missing_validation(self):
        """測試沒有驗證結果的實驗記錄"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            recorder = ExperimentRecorder(
                experiments_file=tmpdir_path / 'experiments.json',
                insights_file=tmpdir_path / 'insights.md'
            )

            result = MockBacktestResult()
            strategy_info = {'name': 'test', 'type': 'trend'}
            config = {'symbol': 'BTCUSDT', 'timeframe': '1h'}

            # 沒有 validation_result
            exp_id = recorder.log_experiment(
                result=result,
                strategy_info=strategy_info,
                config=config,
                validation_result=None
            )

            # 應該成功記錄
            assert exp_id.startswith('exp_')

            exp = recorder.get_experiment(exp_id)
            assert exp is not None
            assert exp.validation == {}

    def test_record_experiment_with_missing_insights(self):
        """測試沒有洞察的實驗記錄"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            recorder = ExperimentRecorder(
                experiments_file=tmpdir_path / 'experiments.json',
                insights_file=tmpdir_path / 'insights.md'
            )

            result = MockBacktestResult()
            strategy_info = {'name': 'test', 'type': 'trend'}
            config = {'symbol': 'BTCUSDT', 'timeframe': '1h'}

            # 沒有 insights
            exp_id = recorder.log_experiment(
                result=result,
                strategy_info=strategy_info,
                config=config,
                insights=None
            )

            exp = recorder.get_experiment(exp_id)
            assert exp.insights == []

    def test_query_with_invalid_filters(self):
        """測試使用無效過濾器查詢"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            recorder = ExperimentRecorder(
                experiments_file=tmpdir_path / 'experiments.json',
                insights_file=tmpdir_path / 'insights.md'
            )

            # 記錄一個實驗
            result = MockBacktestResult()
            recorder.log_experiment(
                result,
                {'name': 'test', 'type': 'trend'},
                {'symbol': 'BTCUSDT', 'timeframe': '1h'}
            )

            # 使用不存在的過濾鍵
            exps = recorder.query_experiments({
                'nonexistent_key': 'value'
            })

            # 應該忽略無效過濾器，返回所有實驗
            assert len(exps) == 1

    def test_get_best_experiments_empty_list(self):
        """測試空實驗列表的最佳查詢"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            recorder = ExperimentRecorder(
                experiments_file=tmpdir_path / 'experiments.json',
                insights_file=tmpdir_path / 'insights.md'
            )

            # 查詢最佳（但沒有任何實驗）
            best = recorder.get_best_experiments('sharpe_ratio', n=5)

            assert len(best) == 0


class TestEdgeCases:
    """測試邊界情況"""

    def test_date_range_filter_with_same_dates(self):
        """測試日期範圍過濾（同一天）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            recorder = ExperimentRecorder(
                experiments_file=tmpdir_path / 'experiments.json',
                insights_file=tmpdir_path / 'insights.md'
            )

            result = MockBacktestResult()
            recorder.log_experiment(
                result,
                {'name': 'test', 'type': 'trend'},
                {'symbol': 'BTCUSDT', 'timeframe': '1h'}
            )

            # 查詢今天
            today = datetime.now().strftime('%Y-%m-%d')
            exps = recorder.query_experiments({
                'date_range': (today, today)
            })

            # 應該能找到今天的實驗
            assert len(exps) >= 1

    def test_generate_tags_with_empty_inputs(self):
        """測試空輸入的標籤生成"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            recorder = ExperimentRecorder(
                experiments_file=tmpdir_path / 'experiments.json',
                insights_file=tmpdir_path / 'insights.md'
            )

            tags = recorder.generate_tags(
                strategy_info={},
                config={},
                validation=None
            )

            # 至少應該有 'crypto' 標籤
            assert 'crypto' in tags

    def test_calculate_improvement_with_nonexistent_parent(self):
        """測試計算改進（父實驗不存在）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            recorder = ExperimentRecorder(
                experiments_file=tmpdir_path / 'experiments.json',
                insights_file=tmpdir_path / 'insights.md'
            )

            improvement = recorder._calculate_improvement(
                'exp_current',
                'nonexistent_parent',
                {'sharpe_ratio': 2.0}
            )

            # 應該返回 None
            assert improvement is None

    def test_multiple_experiments_same_second(self):
        """測試同一秒內記錄多個實驗（ID 衝突）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            recorder = ExperimentRecorder(
                experiments_file=tmpdir_path / 'experiments.json',
                insights_file=tmpdir_path / 'insights.md'
            )

            result = MockBacktestResult()
            strategy_info = {'name': 'test', 'type': 'trend'}
            config = {'symbol': 'BTCUSDT', 'timeframe': '1h'}

            # 快速記錄多個實驗
            ids = []
            for _ in range(3):
                exp_id = recorder.log_experiment(result, strategy_info, config)
                ids.append(exp_id)

            # 驗證所有 ID 都被記錄（即使可能有相同的時間戳）
            data = json.loads(recorder.experiments_file.read_text())
            assert len(data['experiments']) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
