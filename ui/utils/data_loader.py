"""
數據載入輔助模組

提供 Streamlit UI 使用的數據載入函數，整合 ExperimentRecorder。
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import streamlit as st

# 導入 ExperimentRecorder
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.learning.recorder import ExperimentRecorder, Experiment


# 全域單例 recorder
_recorder: Optional[ExperimentRecorder] = None


def _get_recorder() -> ExperimentRecorder:
    """取得全域 recorder 實例（單例模式）"""
    global _recorder
    if _recorder is None:
        _recorder = ExperimentRecorder()
    return _recorder


@st.cache_data(ttl=60)
def load_experiment_data(exp_id: str) -> Optional[Experiment]:
    """
    載入完整實驗數據

    Args:
        exp_id: 實驗 ID（例如：exp_20260111_120000）

    Returns:
        Experiment 物件，如果不存在則返回 None

    範例:
        >>> exp = load_experiment_data('exp_20260111_120000')
        >>> if exp:
        ...     st.write(f"Sharpe Ratio: {exp.results['sharpe_ratio']}")
    """
    recorder = _get_recorder()
    return recorder.get_experiment(exp_id)


@st.cache_data(ttl=60)
def load_equity_curve(exp_id: str) -> Optional[pd.Series]:
    """
    載入權益曲線

    Args:
        exp_id: 實驗 ID

    Returns:
        pd.Series: 權益曲線（index 為日期），如果不存在則返回 None

    範例:
        >>> equity = load_equity_curve('exp_20260111_120000')
        >>> if equity is not None:
        ...     st.line_chart(equity)
    """
    recorder = _get_recorder()
    return recorder.load_equity_curve(exp_id)


@st.cache_data(ttl=60)
def load_daily_returns(exp_id: str) -> Optional[pd.Series]:
    """
    載入日報酬率

    Args:
        exp_id: 實驗 ID

    Returns:
        pd.Series: 日報酬率（index 為日期），如果不存在則返回 None

    範例:
        >>> returns = load_daily_returns('exp_20260111_120000')
        >>> if returns is not None:
        ...     monthly = calculate_monthly_returns(returns)
        ...     st.dataframe(monthly)
    """
    recorder = _get_recorder()
    return recorder.load_daily_returns(exp_id)


@st.cache_data(ttl=60)
def calculate_monthly_returns(daily_returns: pd.Series) -> pd.DataFrame:
    """
    計算月度報酬

    Args:
        daily_returns: 日報酬率 Series（index 為日期）

    Returns:
        pd.DataFrame: 月度報酬表
            - year: 年份
            - month: 月份
            - return: 報酬率（%）

    範例:
        >>> daily_returns = load_daily_returns('exp_20260111_120000')
        >>> if daily_returns is not None:
        ...     monthly = calculate_monthly_returns(daily_returns)
        ...     st.dataframe(monthly)

    計算邏輯:
        monthly_return = (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
    """
    if daily_returns is None or len(daily_returns) == 0:
        return pd.DataFrame(columns=['year', 'month', 'return'])

    # 確保 index 為 DatetimeIndex
    if not isinstance(daily_returns.index, pd.DatetimeIndex):
        daily_returns.index = pd.to_datetime(daily_returns.index)

    # 按月聚合：複利累積
    monthly = daily_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)

    # 轉換為百分比
    monthly_pct = monthly * 100

    # 建立 DataFrame
    df = pd.DataFrame({
        'year': monthly_pct.index.year,
        'month': monthly_pct.index.month,
        'return': monthly_pct.values
    })

    return df


@st.cache_data(ttl=60)
def load_trades(exp_id: str) -> Optional[pd.DataFrame]:
    """
    載入交易記錄

    Args:
        exp_id: 實驗 ID

    Returns:
        pd.DataFrame: 交易記錄，如果不存在則返回 None

    範例:
        >>> trades = load_trades('exp_20260111_120000')
        >>> if trades is not None:
        ...     st.dataframe(trades)
    """
    recorder = _get_recorder()
    return recorder.load_trades(exp_id)


@st.cache_data(ttl=300)
def get_all_experiments() -> list[Experiment]:
    """
    取得所有實驗列表（快取 5 分鐘）

    Returns:
        List[Experiment]: 所有實驗列表

    範例:
        >>> experiments = get_all_experiments()
        >>> for exp in experiments:
        ...     st.write(f"{exp.id}: Sharpe {exp.results['sharpe_ratio']:.2f}")
    """
    recorder = _get_recorder()
    return recorder.query_experiments()


@st.cache_data(ttl=300)
def get_best_experiments(metric: str = 'sharpe_ratio', n: int = 10) -> list[Experiment]:
    """
    取得最佳 N 個實驗

    Args:
        metric: 排序指標（sharpe_ratio, total_return, profit_factor 等）
        n: 取得數量

    Returns:
        List[Experiment]: 最佳實驗列表

    範例:
        >>> best = get_best_experiments('sharpe_ratio', n=5)
        >>> for exp in best:
        ...     st.write(f"{exp.strategy['name']}: {exp.results['sharpe_ratio']:.2f}")
    """
    recorder = _get_recorder()
    return recorder.get_best_experiments(metric=metric, n=n)


# ===== 錯誤處理輔助函數 =====

def validate_experiment_id(exp_id: str) -> bool:
    """
    驗證實驗 ID 格式

    Args:
        exp_id: 實驗 ID

    Returns:
        bool: 格式是否正確

    範例:
        >>> validate_experiment_id('exp_20260111_120000')
        True
        >>> validate_experiment_id('invalid_id')
        False
    """
    import re
    pattern = r'^exp_\d{8}_\d{6}$'
    return bool(re.match(pattern, exp_id))


def handle_missing_data(exp_id: str, data_type: str = 'experiment') -> None:
    """
    處理數據不存在的情況（顯示 Streamlit 錯誤訊息）

    Args:
        exp_id: 實驗 ID
        data_type: 數據類型（experiment, equity_curve, daily_returns, trades）

    範例:
        >>> data = load_experiment_data('exp_20260111_120000')
        >>> if data is None:
        ...     handle_missing_data('exp_20260111_120000', 'experiment')
    """
    type_names = {
        'experiment': '實驗數據',
        'equity_curve': '權益曲線',
        'daily_returns': '日報酬率',
        'trades': '交易記錄'
    }

    name = type_names.get(data_type, data_type)
    st.error(f"❌ 找不到{name}：{exp_id}")
    st.info(f"請確認實驗 ID 是否正確，或該實驗是否已完成回測。")
