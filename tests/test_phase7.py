"""Phase 7 單元測試"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# === DataFrameOps 測試 ===

class TestSeriesOps:
    """SeriesOps 測試"""

    def test_rolling_mean_pandas(self):
        """測試 Pandas rolling mean"""
        from src.strategies.utils import DataFrameOps

        df = pd.DataFrame({'close': [1.0, 2.0, 3.0, 4.0, 5.0]})
        ops = DataFrameOps(df)
        result = ops['close'].rolling_mean(3)

        assert result is not None
        assert len(result.to_pandas()) == 5
        # 第3個值應該是 (1+2+3)/3 = 2.0
        assert abs(result.to_pandas().iloc[2] - 2.0) < 0.01

    def test_ewm_mean_pandas(self):
        """測試 Pandas EWM"""
        from src.strategies.utils import DataFrameOps

        df = pd.DataFrame({'close': [1.0, 2.0, 3.0, 4.0, 5.0]})
        ops = DataFrameOps(df)
        result = ops['close'].ewm_mean(3)

        assert result is not None
        assert len(result.to_pandas()) == 5

    def test_comparison_operators(self):
        """測試比較運算子"""
        from src.strategies.utils import DataFrameOps

        df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [3, 3, 3, 3, 3]})
        ops = DataFrameOps(df)

        gt = ops['a'] > ops['b']  # [F, F, F, T, T]
        lt = ops['a'] < ops['b']  # [T, T, F, F, F]

        assert gt.to_pandas().iloc[3] == True
        assert lt.to_pandas().iloc[0] == True

    def test_logical_operators(self):
        """測試邏輯運算子"""
        from src.strategies.utils import DataFrameOps

        df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [3, 3, 3, 3, 3]})
        ops = DataFrameOps(df)

        cond1 = ops['a'] > 2  # [F, F, T, T, T]
        cond2 = ops['a'] < 5  # [T, T, T, T, F]
        combined = cond1 & cond2  # [F, F, T, T, F]

        assert combined.to_pandas().iloc[2] == True
        assert combined.to_pandas().iloc[4] == False

    def test_repr(self):
        """測試 __repr__"""
        from src.strategies.utils import DataFrameOps

        df = pd.DataFrame({'a': [1, 2, 3]})
        ops = DataFrameOps(df)
        series_ops = ops['a']

        repr_str = repr(series_ops)
        assert 'SeriesOps' in repr_str
        assert 'pandas' in repr_str

    def test_shift_diff(self):
        """測試 shift 和 diff"""
        from src.strategies.utils import DataFrameOps

        df = pd.DataFrame({'close': [1.0, 2.0, 4.0, 7.0, 11.0]})
        ops = DataFrameOps(df)

        shifted = ops['close'].shift(1)
        diffed = ops['close'].diff()

        # shift(1) 後第一個值應該是 NaN
        assert pd.isna(shifted.to_pandas().iloc[0])
        # diff() 第二個值應該是 2-1 = 1
        assert abs(diffed.to_pandas().iloc[1] - 1.0) < 0.01

class TestDataFrameOps:
    """DataFrameOps 測試"""

    def test_column_access(self):
        """測試欄位存取"""
        from src.strategies.utils import DataFrameOps

        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ops = DataFrameOps(df)

        assert 'a' in ops.columns
        assert 'b' in ops.columns
        assert len(ops) == 3

    def test_to_pandas(self):
        """測試轉換為 Pandas"""
        from src.strategies.utils import DataFrameOps

        df = pd.DataFrame({'a': [1, 2, 3]})
        ops = DataFrameOps(df)

        result = ops.to_pandas()
        assert isinstance(result, pd.DataFrame)

# === ExperimentRecorder 測試 ===

class TestExperimentRecorder:
    """ExperimentRecorder API 測試"""

    def test_get_strategy_stats_no_experiments(self):
        """測試無實驗記錄時返回 None"""
        from src.learning.recorder import ExperimentRecorder
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # 建立臨時 experiments.json
            exp_file = Path(tmpdir) / 'learning' / 'experiments.json'
            exp_file.parent.mkdir(parents=True, exist_ok=True)
            exp_file.write_text('{"version": "1.0", "metadata": {"total_experiments": 0}, "experiments": []}')

            insights_file = Path(tmpdir) / 'learning' / 'insights.md'
            insights_file.write_text('# Insights')

            # 暫時修改專案根目錄（這是個 hack，用於測試）
            recorder = ExperimentRecorder.__new__(ExperimentRecorder)
            recorder.project_root = Path(tmpdir)
            recorder.experiments_file = exp_file

            # 建立 storage 和 insights_manager mock
            from src.learning.storage import TimeSeriesStorage
            from src.learning.insights import InsightsManager
            recorder.storage = TimeSeriesStorage(Path(tmpdir))
            recorder.insights_manager = InsightsManager(insights_file)

            result = recorder.get_strategy_stats('nonexistent')
            assert result is None

    def test_passing_grades_constant(self):
        """測試 PASSING_GRADES 常數存在"""
        from src.learning.recorder import PASSING_GRADES

        assert 'A' in PASSING_GRADES
        assert 'B' in PASSING_GRADES
        assert 'C' not in PASSING_GRADES


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
