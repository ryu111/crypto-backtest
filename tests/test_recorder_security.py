"""
ExperimentRecorder 安全性測試

測試路徑驗證、資料庫錯誤處理等安全功能（DuckDB 版本）。
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
import shutil

from src.learning import ExperimentRecorder


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
        # 使用外部路徑
        external_path = Path('/tmp/malicious_experiments.duckdb')

        with pytest.raises(ValueError, match="outside project directory"):
            recorder = ExperimentRecorder(
                db_path=external_path,
                insights_file=Path('/tmp/insights.md')
            )

    def test_reject_parent_directory_traversal(self):
        """拒絕父目錄遍歷（../../../etc/passwd）"""
        # 取得專案根目錄
        project_root = Path(__file__).parent.parent

        # 嘗試使用 ../ 跳出專案目錄
        # 建立一個會解析到專案外的路徑
        traversal_path = project_root / '../../../etc/passwd'

        # 應該被拒絕
        with pytest.raises(ValueError, match="outside project directory"):
            recorder = ExperimentRecorder(
                db_path=traversal_path,
                insights_file=project_root / 'learning' / 'insights.md'
            )

    def test_accept_valid_subdirectory(self):
        """接受專案子目錄的有效路徑"""
        # 取得專案根目錄
        project_root = Path(__file__).parent.parent
        temp_dir = project_root / '.test_temp' / f'test_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 建立專案子目錄的路徑
            db_path = temp_dir / 'test.duckdb'
            insights_file = temp_dir / 'insights.md'

            # 應該成功建立（在專案內）
            recorder = ExperimentRecorder(
                db_path=db_path,
                insights_file=insights_file
            )

            # 驗證路徑在專案內
            assert str(recorder.db_path.resolve()).startswith(str(recorder.project_root))
            assert str(recorder.insights_manager.insights_file.resolve()).startswith(str(recorder.project_root))

            recorder.close()
        finally:
            # 清理
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def test_resolve_symbolic_links(self):
        """測試路徑解析（處理符號連結）"""
        project_root = Path(__file__).parent.parent
        temp_dir = project_root / '.test_temp' / f'test_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 建立實際路徑（不是檔案，讓 DuckDB 自己建立資料庫）
            real_file = temp_dir / 'real.duckdb'

            # 建立符號連結（如果系統支援）
            try:
                symlink_file = temp_dir / 'symlink.duckdb'
                symlink_file.symlink_to(real_file)

                # 建立 recorder 使用符號連結
                # ExperimentRecorder 會透過 DuckDB 建立實際的資料庫檔案
                recorder = ExperimentRecorder(
                    db_path=symlink_file,
                    insights_file=temp_dir / 'insights.md'
                )

                # 驗證路徑已解析
                # db_path 應該已經被 resolve 過
                assert recorder.db_path.resolve() == real_file.resolve()
                assert str(recorder.db_path.resolve()).startswith(str(recorder.project_root))

                # 驗證資料庫可以正常運作
                assert recorder.db_path.exists()

                recorder.close()

            except OSError:
                # 如果系統不支援符號連結，跳過測試
                pytest.skip("Symbolic links not supported on this system")
        finally:
            # 清理
            if temp_dir.exists():
                shutil.rmtree(temp_dir)


class TestDatabaseErrorHandling:
    """測試資料庫錯誤處理"""

    def test_handle_corrupted_database(self):
        """處理損壞的資料庫檔案"""
        project_root = Path(__file__).parent.parent
        temp_dir = project_root / '.test_temp' / f'test_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            db_path = temp_dir / 'corrupted.duckdb'

            # 寫入無效內容（假裝是損壞的資料庫）
            db_path.write_text('This is not a valid DuckDB file', encoding='utf-8')

            # 嘗試建立 recorder
            # DuckDB 應該能處理並重建資料庫
            try:
                recorder = ExperimentRecorder(
                    db_path=db_path,
                    insights_file=temp_dir / 'insights.md'
                )

                # 驗證資料庫已初始化
                count = recorder.repo.conn.execute(
                    "SELECT COUNT(*) FROM experiments"
                ).fetchone()[0]
                assert count == 0

                recorder.close()
            except Exception as e:
                # 如果 DuckDB 無法處理，至少不應該導致系統崩潰
                # 應該有明確的錯誤訊息
                assert "duckdb" in str(e).lower() or "database" in str(e).lower()
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def test_handle_missing_table(self):
        """處理缺少表的資料庫"""
        project_root = Path(__file__).parent.parent
        temp_dir = project_root / '.test_temp' / f'test_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            db_path = temp_dir / 'missing_table.duckdb'

            # 建立空資料庫（沒有表）
            import duckdb
            conn = duckdb.connect(str(db_path))
            conn.close()

            # ExperimentRecorder 應該能自動建立缺少的表
            recorder = ExperimentRecorder(
                db_path=db_path,
                insights_file=temp_dir / 'insights.md'
            )

            # 驗證表已建立
            count = recorder.repo.conn.execute(
                "SELECT COUNT(*) FROM experiments"
            ).fetchone()[0]
            assert count == 0

            recorder.close()
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def test_database_connection_recovery(self):
        """測試資料庫連接恢復"""
        project_root = Path(__file__).parent.parent
        temp_dir = project_root / '.test_temp' / f'test_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            db_path = temp_dir / 'test.duckdb'

            recorder = ExperimentRecorder(
                db_path=db_path,
                insights_file=temp_dir / 'insights.md'
            )

            # 記錄一個實驗
            result = MockBacktestResult()
            exp_id = recorder.log_experiment(
                result,
                {'name': 'test', 'type': 'trend'},
                {'symbol': 'BTCUSDT', 'timeframe': '1h'}
            )

            # 關閉並重新開啟
            recorder.close()

            recorder2 = ExperimentRecorder(
                db_path=db_path,
                insights_file=temp_dir / 'insights.md'
            )

            # 應該能讀取之前的實驗
            exp = recorder2.get_experiment(exp_id)
            assert exp is not None
            assert exp.id == exp_id

            recorder2.close()
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)


class TestExperimentRecordingRobustness:
    """測試實驗記錄的穩健性"""

    def test_record_experiment_with_missing_validation(self):
        """測試沒有驗證結果的實驗記錄"""
        project_root = Path(__file__).parent.parent
        temp_dir = project_root / '.test_temp' / f'test_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            recorder = ExperimentRecorder(
                db_path=temp_dir / 'test.duckdb',
                insights_file=temp_dir / 'insights.md'
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
            # 驗證欄位應該為空字典或預設值
            assert isinstance(exp.validation, dict)
            assert exp.grade == 'F'  # 沒有驗證結果時的預設值（未驗證 = F）

            recorder.close()
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def test_record_experiment_with_missing_insights(self):
        """測試沒有洞察的實驗記錄"""
        project_root = Path(__file__).parent.parent
        temp_dir = project_root / '.test_temp' / f'test_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            recorder = ExperimentRecorder(
                db_path=temp_dir / 'test.duckdb',
                insights_file=temp_dir / 'insights.md'
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

            recorder.close()
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def test_query_with_invalid_filters(self):
        """測試使用無效過濾器查詢"""
        project_root = Path(__file__).parent.parent
        temp_dir = project_root / '.test_temp' / f'test_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            recorder = ExperimentRecorder(
                db_path=temp_dir / 'test.duckdb',
                insights_file=temp_dir / 'insights.md'
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

            recorder.close()
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def test_get_best_experiments_empty_list(self):
        """測試空實驗列表的最佳查詢"""
        project_root = Path(__file__).parent.parent
        temp_dir = project_root / '.test_temp' / f'test_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            recorder = ExperimentRecorder(
                db_path=temp_dir / 'test.duckdb',
                insights_file=temp_dir / 'insights.md'
            )

            # 查詢最佳（但沒有任何實驗）
            best = recorder.get_best_experiments('sharpe_ratio', n=5)

            assert len(best) == 0

            recorder.close()
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)


class TestEdgeCases:
    """測試邊界情況"""

    def test_date_range_filter_with_same_dates(self):
        """測試日期範圍過濾（同一天）"""
        project_root = Path(__file__).parent.parent
        temp_dir = project_root / '.test_temp' / f'test_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            recorder = ExperimentRecorder(
                db_path=temp_dir / 'test.duckdb',
                insights_file=temp_dir / 'insights.md'
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

            recorder.close()
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def test_generate_tags_with_empty_inputs(self):
        """測試空輸入的標籤生成"""
        project_root = Path(__file__).parent.parent
        temp_dir = project_root / '.test_temp' / f'test_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            recorder = ExperimentRecorder(
                db_path=temp_dir / 'test.duckdb',
                insights_file=temp_dir / 'insights.md'
            )

            tags = recorder.generate_tags(
                strategy_info={},
                config={},
                validation=None
            )

            # 至少應該有 'crypto' 標籤
            assert 'crypto' in tags

            recorder.close()
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def test_multiple_experiments_same_second(self):
        """測試同一秒內記錄多個實驗（ID 唯一性）"""
        project_root = Path(__file__).parent.parent
        temp_dir = project_root / '.test_temp' / f'test_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            recorder = ExperimentRecorder(
                db_path=temp_dir / 'test.duckdb',
                insights_file=temp_dir / 'insights.md'
            )

            result = MockBacktestResult()
            config = {'symbol': 'BTCUSDT', 'timeframe': '1h'}

            # 快速記錄多個實驗，使用不同的策略名稱確保 ID 不同
            ids = []
            for i in range(3):
                strategy_info = {'name': f'test_{i}', 'type': 'trend'}
                exp_id = recorder.log_experiment(result, strategy_info, config)
                ids.append(exp_id)

            # 驗證所有 ID 都不同
            assert len(set(ids)) == 3

            # 驗證所有實驗都被記錄到資料庫
            count = recorder.repo.conn.execute(
                "SELECT COUNT(*) FROM experiments"
            ).fetchone()[0]
            assert count == 3

            recorder.close()
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
