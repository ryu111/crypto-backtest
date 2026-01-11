"""
時間序列資料存取

負責 equity_curve、daily_returns、trades 等時間序列資料的存取。
"""

import logging
from pathlib import Path
from typing import Optional, Any
import pandas as pd


logger = logging.getLogger(__name__)


class TimeSeriesStorage:
    """
    時間序列資料存取器

    職責：
    - 儲存回測產生的時間序列資料（equity_curve, daily_returns, trades）
    - 載入時間序列資料供後續分析使用

    儲存結構:
        results/{exp_id}/
            ├── equity_curve.csv    (index=date, columns=['equity'])
            ├── daily_returns.csv   (index=date, columns=['return'])
            └── trades.csv          (交易記錄)
    """

    def __init__(self, project_root: Path):
        """
        初始化 Storage

        Args:
            project_root: 專案根目錄
        """
        self.project_root = project_root
        self.results_dir = project_root / 'results'
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save(self, exp_id: str, result: Any):
        """
        儲存時間序列資料到獨立 CSV 檔案

        Args:
            exp_id: 實驗 ID
            result: BacktestResult 物件
        """
        exp_dir = self.results_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # 1. 儲存 equity_curve
        if hasattr(result, 'equity_curve') and result.equity_curve is not None:
            equity_df = pd.DataFrame({
                'equity': result.equity_curve
            })
            equity_df.index.name = 'date'
            equity_df.to_csv(exp_dir / 'equity_curve.csv')

        # 2. 儲存 daily_returns
        if hasattr(result, 'daily_returns') and result.daily_returns is not None:
            returns_df = pd.DataFrame({
                'return': result.daily_returns
            })
            returns_df.index.name = 'date'
            returns_df.to_csv(exp_dir / 'daily_returns.csv')

        # 3. 儲存 trades
        if hasattr(result, 'trades') and result.trades is not None and len(result.trades) > 0:
            result.trades.to_csv(exp_dir / 'trades.csv', index=False)

    def load_equity_curve(self, exp_id: str) -> Optional[pd.Series]:
        """
        載入實驗的權益曲線

        Args:
            exp_id: 實驗 ID

        Returns:
            pd.Series: 權益曲線（index 為日期），如果不存在則返回 None
        """
        equity_file = self.results_dir / exp_id / 'equity_curve.csv'

        if not equity_file.exists():
            return None

        try:
            df = pd.read_csv(equity_file, index_col='date', parse_dates=True)
            return df['equity']
        except Exception as e:
            logger.error(f"載入 equity_curve 失敗 ({exp_id}): {e}")
            return None

    def load_daily_returns(self, exp_id: str) -> Optional[pd.Series]:
        """
        載入實驗的每日收益率

        Args:
            exp_id: 實驗 ID

        Returns:
            pd.Series: 每日收益率（index 為日期），如果不存在則返回 None
        """
        returns_file = self.results_dir / exp_id / 'daily_returns.csv'

        if not returns_file.exists():
            return None

        try:
            df = pd.read_csv(returns_file, index_col='date', parse_dates=True)
            return df['return']
        except Exception as e:
            logger.error(f"載入 daily_returns 失敗 ({exp_id}): {e}")
            return None

    def load_trades(self, exp_id: str) -> Optional[pd.DataFrame]:
        """
        載入實驗的交易記錄

        Args:
            exp_id: 實驗 ID

        Returns:
            pd.DataFrame: 交易記錄，如果不存在則返回 None
        """
        trades_file = self.results_dir / exp_id / 'trades.csv'

        if not trades_file.exists():
            return None

        try:
            return pd.read_csv(trades_file)
        except Exception as e:
            logger.error(f"載入 trades 失敗 ({exp_id}): {e}")
            return None
