"""
Monte Carlo 模擬器

用於評估策略在不同隨機情境下的表現。
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class MonteCarloResult:
    """Monte Carlo 模擬結果"""

    n_simulations: int
    method: str

    # 分布統計
    mean: float
    std: float
    median: float

    # 百分位數
    percentile_1: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    percentile_99: float

    # 風險指標
    var_95: float
    cvar_95: float

    # 原始 vs 模擬
    original_return: float
    probability_profitable: float  # P(return > 0)
    probability_beat_original: float  # P(return > original)

    # 完整分布
    simulated_returns: np.ndarray


class MonteCarloSimulator:
    """
    Monte Carlo 模擬器

    實作多種 Monte Carlo 模擬方法，用於評估策略在不同
    隨機情境下的表現。
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: 隨機種子（用於可重現性）
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def simulate(
        self,
        trades: pd.DataFrame,
        n_simulations: int = 1000,
        method: str = 'shuffle',
        block_size: int = 5
    ) -> MonteCarloResult:
        """
        執行 Monte Carlo 模擬

        Args:
            trades: 交易記錄 DataFrame（需包含 'pnl' 欄位）
            n_simulations: 模擬次數
            method: 模擬方法
                - 'shuffle': 交易順序隨機化
                - 'bootstrap': Bootstrap 有放回抽樣
                - 'block_bootstrap': 區塊 Bootstrap
            block_size: 區塊大小（僅用於 block_bootstrap）

        Returns:
            MonteCarloResult 物件
        """
        if len(trades) == 0:
            raise ValueError("交易記錄為空")

        if 'pnl' not in trades.columns:
            raise ValueError("交易記錄必須包含 'pnl' 欄位")

        # 計算原始報酬
        original_return = trades['pnl'].sum()

        # 根據方法選擇模擬函數
        if method == 'shuffle':
            simulated_returns = self._shuffle_simulation(trades, n_simulations)
        elif method == 'bootstrap':
            simulated_returns = self._bootstrap_simulation(trades, n_simulations)
        elif method == 'block_bootstrap':
            simulated_returns = self._block_bootstrap(trades, n_simulations, block_size)
        else:
            raise ValueError(f"不支援的模擬方法: {method}")

        # 計算統計指標
        stats = self.calculate_statistics(simulated_returns)

        # 計算風險指標
        var_95 = self.calculate_var(simulated_returns, confidence=0.95)
        cvar_95 = self.calculate_cvar(simulated_returns, confidence=0.95)

        # 計算機率
        prob_profitable = np.mean(simulated_returns > 0)
        prob_beat_original = np.mean(simulated_returns > original_return)

        return MonteCarloResult(
            n_simulations=n_simulations,
            method=method,
            mean=stats['mean'],
            std=stats['std'],
            median=stats['median'],
            percentile_1=stats['percentile_1'],
            percentile_5=stats['percentile_5'],
            percentile_25=stats['percentile_25'],
            percentile_75=stats['percentile_75'],
            percentile_95=stats['percentile_95'],
            percentile_99=stats['percentile_99'],
            var_95=var_95,
            cvar_95=cvar_95,
            original_return=original_return,
            probability_profitable=prob_profitable,
            probability_beat_original=prob_beat_original,
            simulated_returns=simulated_returns
        )

    def _shuffle_simulation(
        self,
        trades: pd.DataFrame,
        n_simulations: int
    ) -> np.ndarray:
        """
        交易順序隨機化模擬

        保持原始交易，只打亂順序，評估策略是否依賴特定順序。

        Args:
            trades: 交易記錄
            n_simulations: 模擬次數

        Returns:
            模擬報酬陣列
        """
        pnl = trades['pnl'].values
        simulated_returns = np.zeros(n_simulations)

        for i in range(n_simulations):
            # 隨機打亂交易順序
            shuffled_pnl = np.random.permutation(pnl)
            simulated_returns[i] = shuffled_pnl.sum()

        return simulated_returns

    def _bootstrap_simulation(
        self,
        trades: pd.DataFrame,
        n_simulations: int
    ) -> np.ndarray:
        """
        Bootstrap 有放回抽樣模擬

        從交易中有放回抽樣，產生更多樣的模擬路徑。

        Args:
            trades: 交易記錄
            n_simulations: 模擬次數

        Returns:
            模擬報酬陣列
        """
        pnl = trades['pnl'].values
        n_trades = len(pnl)
        simulated_returns = np.zeros(n_simulations)

        for i in range(n_simulations):
            # 有放回抽樣
            sampled_pnl = np.random.choice(pnl, size=n_trades, replace=True)
            simulated_returns[i] = sampled_pnl.sum()

        return simulated_returns

    def _block_bootstrap(
        self,
        trades: pd.DataFrame,
        n_simulations: int,
        block_size: int = 5
    ) -> np.ndarray:
        """
        區塊 Bootstrap 模擬

        保持時間相關性，適合有序列相關的策略。
        將交易分成區塊，然後對區塊進行有放回抽樣。

        Args:
            trades: 交易記錄
            n_simulations: 模擬次數
            block_size: 區塊大小

        Returns:
            模擬報酬陣列
        """
        pnl = trades['pnl'].values
        n_trades = len(pnl)

        # 將交易分成區塊
        n_blocks = n_trades // block_size
        if n_blocks == 0:
            # 交易數少於區塊大小，降級為普通 bootstrap
            return self._bootstrap_simulation(trades, n_simulations)

        blocks = []
        for i in range(n_blocks):
            start_idx = i * block_size
            end_idx = start_idx + block_size
            blocks.append(pnl[start_idx:end_idx])

        # 處理剩餘交易
        remainder = n_trades % block_size
        if remainder > 0:
            blocks.append(pnl[-remainder:])

        blocks = np.array(blocks, dtype=object)
        simulated_returns = np.zeros(n_simulations)

        for i in range(n_simulations):
            # 對區塊進行有放回抽樣
            sampled_blocks = np.random.choice(
                len(blocks),
                size=n_blocks,
                replace=True
            )

            # 組合抽樣的區塊
            sampled_pnl = []
            for block_idx in sampled_blocks:
                sampled_pnl.extend(blocks[block_idx])

            simulated_returns[i] = np.sum(sampled_pnl)

        return simulated_returns

    def calculate_statistics(self, simulated_returns: np.ndarray) -> dict:
        """
        計算模擬報酬的統計指標

        Args:
            simulated_returns: 模擬報酬陣列

        Returns:
            統計指標字典
        """
        return {
            'mean': np.mean(simulated_returns),
            'std': np.std(simulated_returns),
            'median': np.median(simulated_returns),
            'percentile_1': np.percentile(simulated_returns, 1),
            'percentile_5': np.percentile(simulated_returns, 5),
            'percentile_25': np.percentile(simulated_returns, 25),
            'percentile_75': np.percentile(simulated_returns, 75),
            'percentile_95': np.percentile(simulated_returns, 95),
            'percentile_99': np.percentile(simulated_returns, 99),
        }

    def calculate_var(
        self,
        simulated_returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """
        計算風險價值（Value at Risk）

        在給定信心水準下，預期的最大損失。

        Args:
            simulated_returns: 模擬報酬陣列
            confidence: 信心水準（0-1）

        Returns:
            VaR 值（負數表示損失）
        """
        percentile = (1 - confidence) * 100
        var = np.percentile(simulated_returns, percentile)
        return var

    def calculate_cvar(
        self,
        simulated_returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """
        計算條件風險價值（Conditional VaR / Expected Shortfall）

        超過 VaR 的平均損失。

        Args:
            simulated_returns: 模擬報酬陣列
            confidence: 信心水準（0-1）

        Returns:
            CVaR 值
        """
        var = self.calculate_var(simulated_returns, confidence)
        cvar = simulated_returns[simulated_returns <= var].mean()
        return cvar

    def plot_distribution(
        self,
        result: MonteCarloResult,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> None:
        """
        繪製模擬報酬分布圖

        Args:
            result: MonteCarloResult 物件
            figsize: 圖表大小
            save_path: 儲存路徑（None 則不儲存）
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 左圖：直方圖 + KDE
        ax1.hist(
            result.simulated_returns,
            bins=50,
            density=True,
            alpha=0.7,
            color='skyblue',
            edgecolor='black'
        )

        # 標記原始報酬
        ax1.axvline(
            result.original_return,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'原始報酬: {result.original_return:.2f}'
        )

        # 標記平均值
        ax1.axvline(
            result.mean,
            color='green',
            linestyle='--',
            linewidth=2,
            label=f'平均: {result.mean:.2f}'
        )

        # 標記 VaR
        ax1.axvline(
            result.var_95,
            color='orange',
            linestyle='--',
            linewidth=2,
            label=f'VaR (95%): {result.var_95:.2f}'
        )

        ax1.set_xlabel('報酬')
        ax1.set_ylabel('密度')
        ax1.set_title(f'Monte Carlo 模擬分布\n({result.method}, n={result.n_simulations})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 右圖：箱型圖
        bp = ax2.boxplot(
            result.simulated_returns,
            vert=True,
            patch_artist=True,
            widths=0.5
        )

        # 設定箱型圖顏色
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][0].set_alpha(0.7)

        # 標記原始報酬
        ax2.axhline(
            result.original_return,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'原始報酬: {result.original_return:.2f}'
        )

        ax2.set_ylabel('報酬')
        ax2.set_title('報酬分布（箱型圖）')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_paths(
        self,
        equity_paths: np.ndarray,
        n_paths_to_plot: int = 100,
        original_path: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> None:
        """
        繪製權益曲線路徑

        Args:
            equity_paths: 權益曲線路徑陣列 (n_simulations, n_points)
            n_paths_to_plot: 要繪製的路徑數量
            original_path: 原始權益曲線
            figsize: 圖表大小
            save_path: 儲存路徑（None 則不儲存）
        """
        fig, ax = plt.subplots(figsize=figsize)

        # 隨機選擇要繪製的路徑
        n_simulations = equity_paths.shape[0]
        n_to_plot = min(n_paths_to_plot, n_simulations)

        if n_to_plot < n_simulations:
            indices = np.random.choice(n_simulations, n_to_plot, replace=False)
            paths_to_plot = equity_paths[indices]
        else:
            paths_to_plot = equity_paths

        # 繪製模擬路徑
        for path in paths_to_plot:
            ax.plot(path, color='gray', alpha=0.1, linewidth=0.5)

        # 繪製平均路徑
        mean_path = equity_paths.mean(axis=0)
        ax.plot(
            mean_path,
            color='blue',
            linewidth=2,
            label='平均路徑'
        )

        # 繪製百分位數區間
        p5 = np.percentile(equity_paths, 5, axis=0)
        p95 = np.percentile(equity_paths, 95, axis=0)

        ax.fill_between(
            range(len(mean_path)),
            p5,
            p95,
            color='blue',
            alpha=0.2,
            label='90% 信賴區間'
        )

        # 繪製原始路徑
        if original_path is not None:
            ax.plot(
                original_path,
                color='red',
                linewidth=2,
                label='原始路徑'
            )

        ax.set_xlabel('交易數')
        ax.set_ylabel('累積報酬')
        ax.set_title('Monte Carlo 權益曲線模擬')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def generate_equity_paths(
        self,
        trades: pd.DataFrame,
        n_simulations: int = 1000,
        method: str = 'shuffle',
        block_size: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        產生權益曲線路徑

        Args:
            trades: 交易記錄
            n_simulations: 模擬次數
            method: 模擬方法
            block_size: 區塊大小（僅用於 block_bootstrap）

        Returns:
            (equity_paths, original_path)
            - equity_paths: 模擬權益曲線 (n_simulations, n_trades+1)
            - original_path: 原始權益曲線
        """
        pnl = trades['pnl'].values
        n_trades = len(pnl)

        # 計算原始權益曲線
        original_path = np.concatenate([[0], np.cumsum(pnl)])

        # 產生模擬權益曲線
        equity_paths = np.zeros((n_simulations, n_trades + 1))

        for i in range(n_simulations):
            # 根據方法產生模擬 PnL
            if method == 'shuffle':
                simulated_pnl = np.random.permutation(pnl)
            elif method == 'bootstrap':
                simulated_pnl = np.random.choice(pnl, size=n_trades, replace=True)
            elif method == 'block_bootstrap':
                # 簡化版區塊 bootstrap
                n_blocks = max(1, n_trades // block_size)
                blocks = []
                for j in range(n_blocks):
                    start_idx = j * block_size
                    end_idx = min(start_idx + block_size, n_trades)
                    blocks.append(pnl[start_idx:end_idx])

                sampled_pnl = []
                for _ in range(n_blocks):
                    block = blocks[np.random.randint(len(blocks))]
                    sampled_pnl.extend(block)

                simulated_pnl = np.array(sampled_pnl[:n_trades])
            else:
                raise ValueError(f"不支援的模擬方法: {method}")

            # 計算權益曲線
            equity_paths[i] = np.concatenate([[0], np.cumsum(simulated_pnl)])

        return equity_paths, original_path

    @staticmethod
    def print_result(result: MonteCarloResult) -> None:
        """
        格式化輸出 Monte Carlo 結果

        Args:
            result: MonteCarloResult 物件
        """
        print("\nMonte Carlo 模擬結果")
        print("=" * 60)
        print(f"模擬次數: {result.n_simulations}")
        print(f"模擬方法: {result.method}")
        print()

        print("分布統計")
        print("-" * 60)
        print(f"平均報酬:     {result.mean:>12.2f}")
        print(f"標準差:       {result.std:>12.2f}")
        print(f"中位數:       {result.median:>12.2f}")
        print()

        print("百分位數")
        print("-" * 60)
        print(f"1%:          {result.percentile_1:>12.2f}")
        print(f"5%:          {result.percentile_5:>12.2f}")
        print(f"25%:         {result.percentile_25:>12.2f}")
        print(f"75%:         {result.percentile_75:>12.2f}")
        print(f"95%:         {result.percentile_95:>12.2f}")
        print(f"99%:         {result.percentile_99:>12.2f}")
        print()

        print("風險指標")
        print("-" * 60)
        print(f"VaR (95%):   {result.var_95:>12.2f}")
        print(f"CVaR (95%):  {result.cvar_95:>12.2f}")
        print()

        print("績效比較")
        print("-" * 60)
        print(f"原始報酬:     {result.original_return:>12.2f}")
        print(f"獲利機率:     {result.probability_profitable:>11.2%}")
        print(f"超越原始機率: {result.probability_beat_original:>11.2%}")
        print()
