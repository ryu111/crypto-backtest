"""
測試 data_loader 模組
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# 添加專案路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ui.utils.data_loader import (
    load_experiment_data,
    load_equity_curve,
    load_daily_returns,
    calculate_monthly_returns,
    validate_experiment_id,
    _get_recorder
)


class TestValidateExperimentId:
    """測試實驗 ID 驗證"""

    def test_valid_id(self):
        assert validate_experiment_id('exp_20260111_120000') is True
        assert validate_experiment_id('exp_20250101_000000') is True

    def test_invalid_id(self):
        assert validate_experiment_id('invalid_id') is False
        assert validate_experiment_id('exp_2026_120000') is False
        assert validate_experiment_id('exp_20260111') is False
        assert validate_experiment_id('') is False


class TestCalculateMonthlyReturns:
    """測試月度報酬計算"""

    def test_calculate_monthly_returns_normal(self):
        # 建立測試數據：2026-01 兩天，2026-02 一天
        dates = pd.to_datetime(['2026-01-15', '2026-01-20', '2026-02-10'])
        daily_returns = pd.Series([0.01, 0.02, -0.01], index=dates)

        result = calculate_monthly_returns(daily_returns)

        # 檢查結構
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['year', 'month', 'return']
        assert len(result) == 2  # 兩個月

        # 檢查 2026-01 計算正確
        jan_row = result[result['month'] == 1].iloc[0]
        assert jan_row['year'] == 2026
        # (1.01 * 1.02 - 1) * 100 = 3.02%
        assert abs(jan_row['return'] - 3.02) < 0.01

        # 檢查 2026-02 計算正確
        feb_row = result[result['month'] == 2].iloc[0]
        assert feb_row['year'] == 2026
        # (0.99 - 1) * 100 = -1%
        assert abs(feb_row['return'] - (-1.0)) < 0.01

    def test_calculate_monthly_returns_empty(self):
        # 空數據
        daily_returns = pd.Series([], dtype=float)
        result = calculate_monthly_returns(daily_returns)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ['year', 'month', 'return']

    def test_calculate_monthly_returns_none(self):
        # None 輸入
        result = calculate_monthly_returns(None)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_calculate_monthly_returns_non_datetime_index(self):
        # 非日期索引會自動轉換
        daily_returns = pd.Series(
            [0.01, 0.02],
            index=['2026-01-15', '2026-01-20']
        )

        result = calculate_monthly_returns(daily_returns)

        assert len(result) == 1
        assert result.iloc[0]['year'] == 2026
        assert result.iloc[0]['month'] == 1


class TestRecorderIntegration:
    """測試與 ExperimentRecorder 的整合"""

    def test_recorder_singleton(self):
        # 多次呼叫應該返回同一個實例
        recorder1 = _get_recorder()
        recorder2 = _get_recorder()

        assert recorder1 is recorder2


class TestLoadFunctions:
    """測試載入函數（需要實際實驗數據）"""

    def test_load_experiment_data_nonexistent(self):
        # 不存在的實驗應該返回 None
        result = load_experiment_data('exp_99999999_999999')
        assert result is None

    def test_load_equity_curve_nonexistent(self):
        # 不存在的實驗應該返回 None
        result = load_equity_curve('exp_99999999_999999')
        assert result is None

    def test_load_daily_returns_nonexistent(self):
        # 不存在的實驗應該返回 None
        result = load_daily_returns('exp_99999999_999999')
        assert result is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
