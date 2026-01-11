"""
Walk-Forward 分析器

防止過擬合的核心驗證工具。
透過滾動窗口的 IS/OOS 測試，驗證策略在未見資料上的穩健性。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import optimize

from ..backtester.engine import BacktestEngine, BacktestConfig, BacktestResult
from ..strategies.base import BaseStrategy


@dataclass
class WindowResult:
    """單一窗口結果"""

    window_id: int
    is_start: datetime
    is_end: datetime
    oos_start: datetime
    oos_end: datetime
    is_return: float
    oos_return: float
    is_sharpe: float
    oos_sharpe: float
    best_params: Dict[str, Any]

    # 額外資訊
    is_trades: int = 0
    oos_trades: int = 0
    is_max_dd: float = 0.0
    oos_max_dd: float = 0.0
    optimization_time: float = 0.0

    def to_dict(self) -> Dict:
        """轉為字典"""
        return {
            'window_id': self.window_id,
            'is_start': self.is_start,
            'is_end': self.is_end,
            'oos_start': self.oos_start,
            'oos_end': self.oos_end,
            'is_return': self.is_return,
            'oos_return': self.oos_return,
            'is_sharpe': self.is_sharpe,
            'oos_sharpe': self.oos_sharpe,
            'best_params': self.best_params,
            'is_trades': self.is_trades,
            'oos_trades': self.oos_trades,
            'is_max_dd': self.is_max_dd,
            'oos_max_dd': self.oos_max_dd,
            'optimization_time': self.optimization_time
        }


@dataclass
class WFAResult:
    """Walk-Forward 分析結果"""

    windows: List[WindowResult]
    efficiency: float  # avg(OOS_return) / avg(IS_return)
    oos_returns: List[float]
    is_returns: List[float]
    oos_sharpes: List[float]
    is_sharpes: List[float]
    consistency: float  # OOS 勝率

    # 統計指標
    oos_mean_return: float = 0.0
    oos_std_return: float = 0.0
    oos_mean_sharpe: float = 0.0
    oos_min_return: float = 0.0
    oos_max_return: float = 0.0

    # 元數據
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """計算統計指標"""
        if self.oos_returns:
            self.oos_mean_return = np.mean(self.oos_returns)
            self.oos_std_return = np.std(self.oos_returns)
            self.oos_min_return = np.min(self.oos_returns)
            self.oos_max_return = np.max(self.oos_returns)

        if self.oos_sharpes:
            self.oos_mean_sharpe = np.mean(self.oos_sharpes)

    def to_dict(self) -> Dict:
        """轉為字典"""
        return {
            'efficiency': self.efficiency,
            'consistency': self.consistency,
            'oos_mean_return': self.oos_mean_return,
            'oos_std_return': self.oos_std_return,
            'oos_mean_sharpe': self.oos_mean_sharpe,
            'oos_min_return': self.oos_min_return,
            'oos_max_return': self.oos_max_return,
            'n_windows': len(self.windows),
            'windows': [w.to_dict() for w in self.windows],
            **self.metadata
        }

    def summary(self) -> str:
        """產生摘要報告"""
        return f"""
Walk-Forward 分析結果
{'='*60}
窗口數量: {len(self.windows)}
WFA 效率: {self.efficiency:.2%} (OOS/IS 報酬比)
OOS 勝率: {self.consistency:.2%}

OOS 績效統計
{'-'*60}
平均報酬: {self.oos_mean_return:.2%} ± {self.oos_std_return:.2%}
報酬範圍: [{self.oos_min_return:.2%}, {self.oos_max_return:.2%}]
平均夏普: {self.oos_mean_sharpe:.2f}

窗口詳情
{'-'*60}
{'ID':<4} {'IS報酬':<10} {'OOS報酬':<10} {'IS夏普':<10} {'OOS夏普':<10}
{'-'*60}
""" + "\n".join([
    f"{w.window_id:<4} {w.is_return:<10.2%} {w.oos_return:<10.2%} "
    f"{w.is_sharpe:<10.2f} {w.oos_sharpe:<10.2f}"
    for w in self.windows
])


class WalkForwardAnalyzer:
    """
    Walk-Forward 分析器

    實作滾動窗口的樣本內/樣本外測試，檢測策略過擬合並驗證穩健性。

    工作流程:
    1. 將資料切分為多個窗口
    2. 每個窗口：
       - 在 IS 期間優化參數
       - 用最佳參數在 OOS 期間測試
    3. 計算 WFA Efficiency = avg(OOS) / avg(IS)

    使用範例:
        analyzer = WalkForwardAnalyzer(
            config=backtest_config,
            mode='rolling'
        )

        result = analyzer.analyze(
            strategy=my_strategy,
            data=market_data,
            param_grid={'period': [10, 20, 30]},
            n_windows=5,
            is_ratio=0.7
        )

        print(result.summary())
    """

    def __init__(
        self,
        config: BacktestConfig,
        mode: str = 'rolling',
        optimize_metric: str = 'sharpe_ratio'
    ):
        """
        初始化分析器

        Args:
            config: 回測配置
            mode: 窗口模式 ('rolling' | 'expanding' | 'anchored')
            optimize_metric: 優化目標指標
        """
        self.config = config
        self.mode = mode
        self.optimize_metric = optimize_metric
        self.engine = BacktestEngine(config)

        if mode not in ['rolling', 'expanding', 'anchored']:
            raise ValueError(f"不支援的模式: {mode}")

    def analyze(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        param_grid: Dict[str, List],
        n_windows: int = 5,
        is_ratio: float = 0.7,
        min_trades: int = 10,
        verbose: bool = True
    ) -> WFAResult:
        """
        執行 Walk-Forward 分析

        Args:
            strategy: 策略物件
            data: 完整市場資料
            param_grid: 參數網格 {'param': [val1, val2, ...]}
            n_windows: 窗口數量
            is_ratio: IS 佔比 (0-1)
            min_trades: 最小交易次數門檻
            verbose: 是否輸出進度

        Returns:
            WFA 分析結果
        """
        if not 0 < is_ratio < 1:
            raise ValueError("is_ratio 必須在 0 到 1 之間")

        # 切分窗口
        windows_data = self._split_windows(data, n_windows, is_ratio)

        # 執行每個窗口
        window_results = []
        for i, (is_data, oos_data) in enumerate(windows_data, 1):
            if verbose:
                print(f"\n處理窗口 {i}/{n_windows}...")
                print(f"  IS:  {is_data.index[0]} ~ {is_data.index[-1]}")
                print(f"  OOS: {oos_data.index[0]} ~ {oos_data.index[-1]}")

            # 1. 優化 IS 窗口
            import time
            start_time = time.time()
            best_params, is_result = self._optimize_window(
                strategy, is_data, param_grid, verbose
            )
            opt_time = time.time() - start_time

            # 檢查 IS 交易次數
            if is_result.total_trades < min_trades:
                if verbose:
                    print(f"  ⚠️  IS 交易次數不足 ({is_result.total_trades} < {min_trades}), 跳過此窗口")
                continue

            # 2. 測試 OOS 窗口
            oos_result = self._test_window(strategy, oos_data, best_params)

            # 檢查 OOS 交易次數
            if oos_result.total_trades < min_trades:
                if verbose:
                    print(f"  ⚠️  OOS 交易次數不足 ({oos_result.total_trades} < {min_trades}), 跳過此窗口")
                continue

            # 3. 儲存結果
            window_result = WindowResult(
                window_id=i,
                is_start=is_data.index[0],
                is_end=is_data.index[-1],
                oos_start=oos_data.index[0],
                oos_end=oos_data.index[-1],
                is_return=is_result.total_return,
                oos_return=oos_result.total_return,
                is_sharpe=is_result.sharpe_ratio,
                oos_sharpe=oos_result.sharpe_ratio,
                best_params=best_params,
                is_trades=is_result.total_trades,
                oos_trades=oos_result.total_trades,
                is_max_dd=is_result.max_drawdown,
                oos_max_dd=oos_result.max_drawdown,
                optimization_time=opt_time
            )

            window_results.append(window_result)

            if verbose:
                print(f"  ✓ IS:  報酬={is_result.total_return:.2%}, 夏普={is_result.sharpe_ratio:.2f}")
                print(f"  ✓ OOS: 報酬={oos_result.total_return:.2%}, 夏普={oos_result.sharpe_ratio:.2f}")

        if not window_results:
            raise ValueError("沒有任何窗口產生有效結果，請檢查資料和參數")

        # 計算整體指標
        wfa_result = self._calculate_efficiency(window_results)

        # 添加元數據
        wfa_result.metadata = {
            'mode': self.mode,
            'n_windows': n_windows,
            'is_ratio': is_ratio,
            'optimize_metric': self.optimize_metric,
            'param_grid': param_grid,
            'strategy_name': strategy.name
        }

        return wfa_result

    def _split_windows(
        self,
        data: pd.DataFrame,
        n_windows: int,
        is_ratio: float
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        切分 IS/OOS 窗口

        窗口模式：
        - rolling: 固定大小窗口滾動
        - expanding: IS 窗口逐步擴大
        - anchored: IS 起點固定，終點逐步後移

        Args:
            data: 完整資料
            n_windows: 窗口數量
            is_ratio: IS 佔窗口比例

        Returns:
            [(is_data_1, oos_data_1), (is_data_2, oos_data_2), ...]
        """
        total_len = len(data)
        windows = []

        if self.mode == 'rolling':
            # 滾動窗口：固定大小
            # |----IS----|--OOS--|
            #      |----IS----|--OOS--|
            #           |----IS----|--OOS--|

            window_size = total_len // n_windows
            is_size = int(window_size * is_ratio)
            oos_size = window_size - is_size

            for i in range(n_windows):
                start = i * window_size
                is_end = start + is_size
                oos_end = min(start + window_size, total_len)

                if oos_end - is_end < oos_size // 2:
                    # OOS 資料不足，跳過
                    break

                is_data = data.iloc[start:is_end]
                oos_data = data.iloc[is_end:oos_end]

                windows.append((is_data, oos_data))

        elif self.mode == 'expanding':
            # 擴展窗口：IS 逐步增大
            # |----IS----|--OOS--|
            # |-------IS--------|--OOS--|
            # |------------IS------------|--OOS--|

            oos_size = total_len // (n_windows * 2)  # 固定 OOS 大小
            is_start = 0

            for i in range(n_windows):
                is_end = total_len - (n_windows - i) * oos_size
                oos_end = is_end + oos_size

                if is_end <= is_start or oos_end > total_len:
                    break

                is_data = data.iloc[is_start:is_end]
                oos_data = data.iloc[is_end:oos_end]

                windows.append((is_data, oos_data))

        elif self.mode == 'anchored':
            # 錨定窗口：起點固定
            # |----IS----|--OOS--|
            # |-------IS--------|--OOS--|
            # |------------IS------------|--OOS--|

            is_start = 0
            oos_size = total_len // (n_windows * 3)

            for i in range(1, n_windows + 1):
                is_end = total_len - (n_windows - i + 1) * oos_size
                oos_end = is_end + oos_size

                if is_end <= is_start or oos_end > total_len:
                    break

                is_data = data.iloc[is_start:is_end]
                oos_data = data.iloc[is_end:oos_end]

                windows.append((is_data, oos_data))

        return windows

    def _optimize_window(
        self,
        strategy: BaseStrategy,
        is_data: pd.DataFrame,
        param_grid: Dict[str, List],
        verbose: bool = False
    ) -> Tuple[Dict[str, Any], BacktestResult]:
        """
        優化單一 IS 窗口

        使用網格搜尋找出最佳參數組合。

        Args:
            strategy: 策略物件
            is_data: IS 資料
            param_grid: 參數網格
            verbose: 是否輸出進度

        Returns:
            (最佳參數, IS 結果)
        """
        from itertools import product

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        best_metric = float('-inf')
        best_params = None
        best_result = None

        total_combinations = np.prod([len(v) for v in param_values])

        if verbose:
            print(f"  優化中... ({total_combinations} 種組合)")

        for i, values in enumerate(product(*param_values), 1):
            params = dict(zip(param_names, values))

            try:
                # 執行回測
                result = self.engine.run(strategy, params, is_data)
                current_metric = getattr(result, self.optimize_metric)

                # 更新最佳結果
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_params = params.copy()
                    best_result = result

                if verbose and i % max(1, total_combinations // 10) == 0:
                    progress = i / total_combinations * 100
                    print(f"    進度: {progress:.0f}% - 當前最佳 {self.optimize_metric}: {best_metric:.2f}")

            except Exception as e:
                if verbose:
                    print(f"    ⚠️  參數組合 {params} 失敗: {e}")
                continue

        if best_params is None:
            raise ValueError("優化失敗：沒有任何參數組合產生有效結果")

        return best_params, best_result

    def _test_window(
        self,
        strategy: BaseStrategy,
        oos_data: pd.DataFrame,
        params: Dict[str, Any]
    ) -> BacktestResult:
        """
        測試 OOS 窗口

        使用 IS 優化得到的參數在 OOS 資料上測試。

        Args:
            strategy: 策略物件
            oos_data: OOS 資料
            params: IS 最佳參數

        Returns:
            OOS 回測結果
        """
        return self.engine.run(strategy, params, oos_data)

    def _calculate_efficiency(
        self,
        window_results: List[WindowResult]
    ) -> WFAResult:
        """
        計算 WFA 效率

        Efficiency = avg(OOS_return) / avg(IS_return)
        Consistency = OOS 勝率 (報酬 > 0 的比例)

        Args:
            window_results: 所有窗口結果

        Returns:
            WFA 結果
        """
        oos_returns = [w.oos_return for w in window_results]
        is_returns = [w.is_return for w in window_results]
        oos_sharpes = [w.oos_sharpe for w in window_results]
        is_sharpes = [w.is_sharpe for w in window_results]

        # 計算平均
        avg_oos = np.mean(oos_returns)
        avg_is = np.mean(is_returns)

        # 計算效率
        efficiency = avg_oos / avg_is if avg_is != 0 else 0.0

        # 計算一致性（OOS 勝率）
        winning_windows = sum(1 for r in oos_returns if r > 0)
        consistency = winning_windows / len(oos_returns) if oos_returns else 0.0

        return WFAResult(
            windows=window_results,
            efficiency=efficiency,
            oos_returns=oos_returns,
            is_returns=is_returns,
            oos_sharpes=oos_sharpes,
            is_sharpes=is_sharpes,
            consistency=consistency
        )

    def analyze_degradation(
        self,
        wfa_result: WFAResult
    ) -> Dict[str, float]:
        """
        分析效能衰退

        計算 IS 到 OOS 的各項指標衰退程度。

        Args:
            wfa_result: WFA 結果

        Returns:
            衰退指標字典
        """
        degradations = {
            'return_degradation': [],
            'sharpe_degradation': [],
            'max_dd_increase': []
        }

        for w in wfa_result.windows:
            # 報酬衰退
            if w.is_return != 0:
                ret_deg = (w.is_return - w.oos_return) / abs(w.is_return)
                degradations['return_degradation'].append(ret_deg)

            # 夏普衰退
            if w.is_sharpe != 0:
                sharpe_deg = (w.is_sharpe - w.oos_sharpe) / abs(w.is_sharpe)
                degradations['sharpe_degradation'].append(sharpe_deg)

            # 最大回撤增加
            dd_increase = w.oos_max_dd - w.is_max_dd
            degradations['max_dd_increase'].append(dd_increase)

        # 計算平均衰退
        result = {}
        for key, values in degradations.items():
            if values:
                result[f'avg_{key}'] = np.mean(values)
                result[f'std_{key}'] = np.std(values)
                result[f'max_{key}'] = np.max(values)

        return result

    def plot_results(
        self,
        wfa_result: WFAResult,
        save_path: Optional[str] = None
    ):
        """
        繪製 WFA 結果圖表

        Args:
            wfa_result: WFA 結果
            save_path: 儲存路徑（可選）
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("需要安裝 matplotlib: pip install matplotlib")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Walk-Forward Analysis Results', fontsize=16)

        windows_id = [w.window_id for w in wfa_result.windows]

        # 1. IS vs OOS 報酬
        ax1 = axes[0, 0]
        ax1.plot(windows_id, wfa_result.is_returns, 'o-', label='IS Return', color='blue')
        ax1.plot(windows_id, wfa_result.oos_returns, 's-', label='OOS Return', color='red')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title('Returns: IS vs OOS')
        ax1.set_xlabel('Window')
        ax1.set_ylabel('Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. IS vs OOS 夏普
        ax2 = axes[0, 1]
        ax2.plot(windows_id, wfa_result.is_sharpes, 'o-', label='IS Sharpe', color='blue')
        ax2.plot(windows_id, wfa_result.oos_sharpes, 's-', label='OOS Sharpe', color='red')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_title('Sharpe Ratio: IS vs OOS')
        ax2.set_xlabel('Window')
        ax2.set_ylabel('Sharpe')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. OOS 報酬分佈
        ax3 = axes[1, 0]
        ax3.hist(wfa_result.oos_returns, bins=20, alpha=0.7, color='red', edgecolor='black')
        ax3.axvline(x=wfa_result.oos_mean_return, color='darkred', linestyle='--',
                   label=f'Mean: {wfa_result.oos_mean_return:.2%}')
        ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_title('OOS Returns Distribution')
        ax3.set_xlabel('Return')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 效率指標
        ax4 = axes[1, 1]
        metrics_names = ['Efficiency', 'Consistency', 'OOS Win%']
        metrics_values = [
            wfa_result.efficiency,
            wfa_result.consistency,
            sum(1 for r in wfa_result.oos_returns if r > 0) / len(wfa_result.oos_returns)
        ]
        ax4.bar(metrics_names, metrics_values, color=['green', 'blue', 'orange'])
        ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax4.set_title('WFA Metrics')
        ax4.set_ylabel('Value')
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"圖表已儲存至: {save_path}")
        else:
            plt.show()
