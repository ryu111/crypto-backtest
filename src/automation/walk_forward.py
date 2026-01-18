"""Walk-Forward Analysis 模組

實作滾動窗口分析，避免過擬合。

參考：/.claude/skills/參數優化/references/walk-forward.md
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class WFARiskLevel(str, Enum):
    """Walk-Forward 風險等級"""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class WFAThresholds:
    """Walk-Forward 門檻常數"""

    # IS 比例範圍
    IS_RATIO_MIN = 0.5
    IS_RATIO_MAX = 0.9

    # 最小窗口數
    MIN_WINDOWS = 2

    # 效率門檻
    EFFICIENCY_EXCELLENT = 0.8  # 優秀
    EFFICIENCY_GOOD = 0.6  # 良好
    EFFICIENCY_FAIR = 0.4  # 一般
    EFFICIENCY_THRESHOLD = 0.5  # 50% 閾值

    # 最小窗口資料筆數
    MIN_SPLIT_SIZE = 10


@dataclass
class WalkForwardWindow:
    """Walk-Forward 單一窗口"""

    window_id: int
    is_start: datetime
    is_end: datetime
    oos_start: datetime
    oos_end: datetime
    is_data: Optional[pd.DataFrame] = None
    oos_data: Optional[pd.DataFrame] = None

    @property
    def is_size(self) -> int:
        """In-Sample 資料筆數"""
        return len(self.is_data) if self.is_data is not None else 0

    @property
    def oos_size(self) -> int:
        """Out-of-Sample 資料筆數"""
        return len(self.oos_data) if self.oos_data is not None else 0


@dataclass
class WindowResult:
    """單一窗口的優化結果"""

    window_id: int
    params: Dict[str, Any]
    is_return: float
    oos_return: float
    is_sharpe: float
    oos_sharpe: float
    is_trades: int
    oos_trades: int
    is_max_drawdown: float = 0.0
    oos_max_drawdown: float = 0.0

    @property
    def efficiency(self) -> float:
        """窗口效率 (OOS / IS)"""
        if self.is_sharpe > 0:
            return self.oos_sharpe / self.is_sharpe
        return 0.0


@dataclass
class WalkForwardResult:
    """Walk-Forward Analysis 結果"""

    windows: List[WindowResult] = field(default_factory=list)
    strategy_name: str = ""
    n_windows: int = 0
    is_ratio: float = 0.7
    overlap: float = 0.5

    @property
    def is_mean_return(self) -> float:
        """IS 平均報酬"""
        if not self.windows:
            return 0.0
        return np.mean([w.is_return for w in self.windows])

    @property
    def oos_mean_return(self) -> float:
        """OOS 平均報酬"""
        if not self.windows:
            return 0.0
        return np.mean([w.oos_return for w in self.windows])

    @property
    def is_mean_sharpe(self) -> float:
        """IS 平均 Sharpe"""
        if not self.windows:
            return 0.0
        return np.mean([w.is_sharpe for w in self.windows])

    @property
    def oos_mean_sharpe(self) -> float:
        """OOS 平均 Sharpe"""
        if not self.windows:
            return 0.0
        return np.mean([w.oos_sharpe for w in self.windows])

    @property
    def efficiency(self) -> float:
        """WFA 效率 (OOS / IS)

        判斷標準：
        - > 80%: 優秀，策略穩健
        - 60-80%: 良好，可接受
        - 40-60%: 一般，需謹慎
        - 20-40%: 差，高度過擬合嫌疑
        - < 20%: 極差，策略無效
        """
        if self.is_mean_sharpe > 0:
            return self.oos_mean_sharpe / self.is_mean_sharpe
        return 0.0

    @property
    def oos_win_rate(self) -> float:
        """OOS 獲利窗口比例"""
        if not self.windows:
            return 0.0
        wins = sum(1 for w in self.windows if w.oos_return > 0)
        return wins / len(self.windows)

    @property
    def consistency(self) -> float:
        """一致性（獲利窗口比例）- oos_win_rate 的別名"""
        return self.oos_win_rate

    @property
    def oos_std_return(self) -> float:
        """OOS 報酬標準差"""
        if not self.windows:
            return 0.0
        returns = [w.oos_return for w in self.windows]
        return float(np.std(returns))

    @property
    def total_oos_return(self) -> float:
        """累積 OOS 報酬"""
        if not self.windows:
            return 0.0
        cumulative = 1.0
        for w in self.windows:
            cumulative *= (1 + w.oos_return)
        return cumulative - 1

    @property
    def risk_level(self) -> WFARiskLevel:
        """過擬合風險等級"""
        eff = self.efficiency
        T = WFAThresholds
        if eff >= T.EFFICIENCY_EXCELLENT:
            return WFARiskLevel.LOW
        elif eff >= T.EFFICIENCY_GOOD:
            return WFARiskLevel.MEDIUM
        elif eff >= T.EFFICIENCY_FAIR:
            return WFARiskLevel.HIGH
        else:
            return WFARiskLevel.CRITICAL

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "strategy_name": self.strategy_name,
            "n_windows": self.n_windows,
            "is_ratio": self.is_ratio,
            "overlap": self.overlap,
            "is_mean_return": self.is_mean_return,
            "oos_mean_return": self.oos_mean_return,
            "is_mean_sharpe": self.is_mean_sharpe,
            "oos_mean_sharpe": self.oos_mean_sharpe,
            "efficiency": self.efficiency,
            "oos_win_rate": self.oos_win_rate,
            "total_oos_return": self.total_oos_return,
            "risk_level": self.risk_level.value,
            "windows": [
                {
                    "window_id": w.window_id,
                    "params": w.params,
                    "is_return": w.is_return,
                    "oos_return": w.oos_return,
                    "is_sharpe": w.is_sharpe,
                    "oos_sharpe": w.oos_sharpe,
                    "efficiency": w.efficiency,
                }
                for w in self.windows
            ],
        }


class WalkForwardAnalyzer:
    """Walk-Forward Analysis 分析器

    將歷史資料分成多個滾動窗口，每個窗口內：
    1. In-Sample (IS)：優化參數
    2. Out-of-Sample (OOS)：測試參數

    這樣可以：
    - 模擬真實交易情境
    - 參數必須在未見資料上有效
    - 自動偵測過擬合
    """

    def __init__(
        self,
        is_ratio: float = 0.7,
        n_windows: int = 5,
        overlap: float = 0.5,
        min_window_size: int = 100,
    ):
        """初始化分析器

        Args:
            is_ratio: In-Sample 比例（0.7 = 70%）
            n_windows: 窗口數量
            overlap: 窗口重疊比例（0.5 = 50%）
            min_window_size: 最小窗口大小（資料筆數）
        """
        T = WFAThresholds
        if not T.IS_RATIO_MIN <= is_ratio <= T.IS_RATIO_MAX:
            raise ValueError(f"is_ratio 應在 {T.IS_RATIO_MIN}-{T.IS_RATIO_MAX} 之間")
        if n_windows < T.MIN_WINDOWS:
            raise ValueError(f"n_windows 至少為 {T.MIN_WINDOWS}")
        if not 0 <= overlap < 1:
            raise ValueError("overlap 應在 0-1 之間")

        self.is_ratio = is_ratio
        self.n_windows = n_windows
        self.overlap = overlap
        self.min_window_size = min_window_size

    def generate_windows(
        self,
        data: pd.DataFrame,
        datetime_column: Optional[str] = None,
    ) -> List[WalkForwardWindow]:
        """生成滾動窗口

        Args:
            data: 完整歷史資料（需有時間索引或 datetime 欄位）
            datetime_column: 日期時間欄位名稱（如果不是索引）

        Returns:
            List[WalkForwardWindow]: 滾動窗口列表
        """
        if len(data) < self.min_window_size * self.n_windows:
            raise ValueError(
                f"資料不足：需要至少 {self.min_window_size * self.n_windows} 筆，"
                f"但只有 {len(data)} 筆"
            )

        total_len = len(data)

        # 計算窗口大小和步進
        # 公式：total_len = window_size + (n_windows - 1) * step_size
        # 其中 step_size = window_size * (1 - overlap)
        # 解方程式得：window_size = total_len / (1 + (n_windows - 1) * (1 - overlap))
        window_size = int(total_len / (1 + (self.n_windows - 1) * (1 - self.overlap)))
        step_size = int(window_size * (1 - self.overlap))
        is_size = int(window_size * self.is_ratio)
        oos_size = window_size - is_size

        # 確保最小窗口大小
        if window_size < self.min_window_size:
            raise ValueError(
                f"計算出的窗口大小 {window_size} 小於最小要求 {self.min_window_size}"
            )

        windows = []

        for i in range(self.n_windows):
            start_idx = i * step_size
            is_end_idx = start_idx + is_size
            oos_end_idx = start_idx + window_size

            # 確保不超出資料範圍
            if oos_end_idx > total_len:
                break

            # 切分資料
            is_data = data.iloc[start_idx:is_end_idx].copy()
            oos_data = data.iloc[is_end_idx:oos_end_idx].copy()

            # 取得時間範圍
            if datetime_column:
                is_start = is_data[datetime_column].iloc[0]
                is_end = is_data[datetime_column].iloc[-1]
                oos_start = oos_data[datetime_column].iloc[0]
                oos_end = oos_data[datetime_column].iloc[-1]
            elif isinstance(data.index, pd.DatetimeIndex):
                is_start = is_data.index[0]
                is_end = is_data.index[-1]
                oos_start = oos_data.index[0]
                oos_end = oos_data.index[-1]
            else:
                # 使用索引值作為時間標記
                is_start = datetime.fromtimestamp(start_idx)
                is_end = datetime.fromtimestamp(is_end_idx)
                oos_start = datetime.fromtimestamp(is_end_idx)
                oos_end = datetime.fromtimestamp(oos_end_idx)

            window = WalkForwardWindow(
                window_id=i + 1,
                is_start=is_start,
                is_end=is_end,
                oos_start=oos_start,
                oos_end=oos_end,
                is_data=is_data,
                oos_data=oos_data,
            )
            windows.append(window)

        return windows

    def calculate_efficiency(
        self,
        is_sharpes: List[float],
        oos_sharpes: List[float],
    ) -> float:
        """計算 WFA 效率

        Args:
            is_sharpes: 各窗口 IS Sharpe 列表
            oos_sharpes: 各窗口 OOS Sharpe 列表

        Returns:
            float: WFA 效率 (0-1+)
        """
        is_mean = np.mean(is_sharpes) if is_sharpes else 0
        oos_mean = np.mean(oos_sharpes) if oos_sharpes else 0

        if is_mean > 0:
            return oos_mean / is_mean
        return 0.0

    async def analyze(
        self,
        data: pd.DataFrame,
        strategy_name: str,
        optimize_func: Callable[[pd.DataFrame], Tuple[Dict[str, Any], float]],
        evaluate_func: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
        datetime_column: Optional[str] = None,
    ) -> WalkForwardResult:
        """執行 Walk-Forward Analysis

        Args:
            data: 完整歷史資料
            strategy_name: 策略名稱
            optimize_func: 優化函數，輸入 IS 資料，回傳 (最佳參數, IS Sharpe)
            evaluate_func: 評估函數，輸入資料和參數，回傳績效指標字典
            datetime_column: 日期時間欄位名稱

        Returns:
            WalkForwardResult: 分析結果
        """
        windows = self.generate_windows(data, datetime_column)

        result = WalkForwardResult(
            strategy_name=strategy_name,
            n_windows=len(windows),
            is_ratio=self.is_ratio,
            overlap=self.overlap,
        )

        for window in windows:
            # 在 IS 優化參數
            best_params, is_sharpe = await self._run_async_or_sync(
                optimize_func, window.is_data
            )

            # 在 IS 評估績效
            is_metrics = await self._run_async_or_sync(
                evaluate_func, window.is_data, best_params
            )

            # 在 OOS 評估績效
            oos_metrics = await self._run_async_or_sync(
                evaluate_func, window.oos_data, best_params
            )

            window_result = WindowResult(
                window_id=window.window_id,
                params=best_params,
                is_return=is_metrics.get("total_return", 0),
                oos_return=oos_metrics.get("total_return", 0),
                is_sharpe=is_metrics.get("sharpe_ratio", is_sharpe),
                oos_sharpe=oos_metrics.get("sharpe_ratio", 0),
                is_trades=is_metrics.get("total_trades", 0),
                oos_trades=oos_metrics.get("total_trades", 0),
                is_max_drawdown=is_metrics.get("max_drawdown", 0),
                oos_max_drawdown=oos_metrics.get("max_drawdown", 0),
            )
            result.windows.append(window_result)

        return result

    async def _run_async_or_sync(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """執行同步或非同步函數"""
        import asyncio
        import inspect

        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    def get_recommended_params(
        self,
        result: WalkForwardResult,
    ) -> Dict[str, Any]:
        """取得推薦參數

        選項：
        1. 使用最後一個窗口的參數（最新）
        2. 使用效率最高窗口的參數
        3. 使用所有窗口參數的平均值（數值參數）

        預設使用效率最高的窗口參數。
        """
        if not result.windows:
            return {}

        # 找效率最高的窗口
        best_window = max(result.windows, key=lambda w: w.efficiency)
        return best_window.params

    def plot_results(self, result: WalkForwardResult) -> None:
        """繪製 WFA 結果圖表

        需要 matplotlib，如果沒有則跳過。
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        windows = result.windows
        n = len(windows)
        if n == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. IS vs OOS 報酬
        ax1 = axes[0, 0]
        x = range(1, n + 1)
        ax1.bar(
            [i - 0.2 for i in x],
            [w.is_return * 100 for w in windows],
            0.4,
            label="In-Sample",
            alpha=0.7,
            color="#00f5ff",
        )
        ax1.bar(
            [i + 0.2 for i in x],
            [w.oos_return * 100 for w in windows],
            0.4,
            label="Out-of-Sample",
            alpha=0.7,
            color="#bf00ff",
        )
        ax1.set_xlabel("Window")
        ax1.set_ylabel("Return (%)")
        ax1.legend()
        ax1.set_title("IS vs OOS Returns by Window")
        ax1.axhline(y=0, color="white", linestyle="-", linewidth=0.5)

        # 2. 累積 OOS 權益曲線
        ax2 = axes[0, 1]
        oos_returns = [w.oos_return for w in windows]
        cumulative = np.cumprod([1 + r for r in oos_returns])
        ax2.plot(x, cumulative, marker="o", color="#00ff88")
        ax2.set_xlabel("Window")
        ax2.set_ylabel("Cumulative Return")
        ax2.set_title("Cumulative OOS Equity Curve")

        # 3. 效率趨勢
        ax3 = axes[1, 0]
        efficiencies = [w.efficiency for w in windows]
        colors = ["#00ff88" if e >= 0.5 else "#ff0066" for e in efficiencies]
        ax3.bar(x, efficiencies, color=colors)
        ax3.axhline(y=0.5, color="yellow", linestyle="--", label="50% threshold")
        ax3.set_xlabel("Window")
        ax3.set_ylabel("Efficiency (OOS/IS)")
        ax3.set_title("Window Efficiency")
        ax3.legend()

        # 4. IS vs OOS Sharpe
        ax4 = axes[1, 1]
        ax4.scatter(
            [w.is_sharpe for w in windows],
            [w.oos_sharpe for w in windows],
            c=range(n),
            cmap="viridis",
            s=100,
        )
        max_sharpe = max(
            max(w.is_sharpe for w in windows), max(w.oos_sharpe for w in windows)
        )
        ax4.plot([0, max_sharpe], [0, max_sharpe], "r--", label="Perfect efficiency")
        ax4.set_xlabel("IS Sharpe")
        ax4.set_ylabel("OOS Sharpe")
        ax4.set_title("IS vs OOS Sharpe")
        ax4.legend()

        plt.tight_layout()
        plt.savefig("walk_forward_analysis.png", dpi=150, facecolor="#0a0a0f")
        plt.close()
