"""
策略組合優化器

提供多種組合優化方法：
1. Mean-Variance 優化（Markowitz）
2. 風險平價（Risk Parity）配置
3. 最大化 Sharpe Ratio 的權重優化
4. 效率前緣計算

支援各種約束條件（最大/最小權重、多空限制等）。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings
import logging

import pandas as pd
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.stats import gmean

logger = logging.getLogger(__name__)


@dataclass
class PortfolioWeights:
    """組合權重與績效指標"""

    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float

    # 可選的額外資訊
    optimization_success: bool = True
    optimization_message: str = ""

    def to_dict(self) -> Dict:
        """轉為字典格式"""
        return {
            'weights': self.weights,
            'expected_return': self.expected_return,
            'expected_volatility': self.expected_volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'optimization_success': self.optimization_success,
            'optimization_message': self.optimization_message
        }

    def summary(self) -> str:
        """產生摘要報告"""
        weights_str = '\n'.join(
            f"  {name}: {weight:.4f}" for name, weight in self.weights.items()
        )
        return f"""
組合權重配置
{'='*60}
{weights_str}

預期績效
{'-'*60}
預期年化報酬: {self.expected_return*100:.2f}%
預期年化波動: {self.expected_volatility*100:.2f}%
Sharpe Ratio: {self.sharpe_ratio:.4f}

優化狀態: {'成功' if self.optimization_success else '失敗'}
{self.optimization_message if self.optimization_message else ''}
"""


class PortfolioOptimizer:
    """
    策略組合優化器

    使用 Modern Portfolio Theory (MPT) 進行組合優化。
    支援多種優化方法和約束條件。

    使用範例：
        # 準備策略回報資料
        returns = pd.DataFrame({
            'strategy_a': [0.01, 0.02, -0.01, ...],
            'strategy_b': [0.015, -0.005, 0.02, ...],
            'strategy_c': [0.008, 0.012, 0.005, ...]
        })

        # 建立優化器
        optimizer = PortfolioOptimizer(
            returns=returns,
            risk_free_rate=0.0
        )

        # 最大化 Sharpe Ratio
        weights = optimizer.max_sharpe_optimize()
        print(weights.summary())

        # 風險平價配置
        weights = optimizer.risk_parity_optimize()

        # 計算效率前緣
        frontier = optimizer.efficient_frontier(n_points=50)
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.0,
        frequency: int = 252,
        use_ledoit_wolf: bool = True
    ):
        """
        初始化組合優化器

        Args:
            returns: 各策略回報 DataFrame (columns=策略名稱, index=日期)
            risk_free_rate: 無風險利率（年化）
            frequency: 年化頻率（252=每日, 52=每週, 12=每月）
            use_ledoit_wolf: 是否使用 Ledoit-Wolf 協方差矩陣估計（防止過擬合）
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.frequency = frequency
        self.strategy_names = list(returns.columns)
        self.n_assets = len(self.strategy_names)

        # 計算預期報酬（年化）
        self.mean_returns = returns.mean() * frequency

        # 計算協方差矩陣（年化）
        if use_ledoit_wolf:
            self.cov_matrix = self._ledoit_wolf_cov()
        else:
            self.cov_matrix = returns.cov() * frequency

        # 驗證資料
        self._validate_data()

    def _validate_data(self):
        """驗證輸入資料"""
        if self.n_assets < 2:
            raise ValueError("至少需要 2 個策略進行組合優化")

        if self.returns.isnull().any().any():
            warnings.warn("回報資料包含 NaN，將自動填補為 0")
            self.returns = self.returns.fillna(0)

        # 檢查協方差矩陣是否正定
        eigenvalues = np.linalg.eigvals(self.cov_matrix.values)
        if not np.all(eigenvalues > 0):
            warnings.warn(
                "協方差矩陣不是正定矩陣，可能導致優化問題。"
                "建議啟用 Ledoit-Wolf 收縮估計。"
            )

    def _ledoit_wolf_cov(self) -> pd.DataFrame:
        """
        使用 Ledoit-Wolf 收縮估計計算協方差矩陣

        這個方法可以減少估計誤差，特別是在樣本數較少時。
        """
        try:
            from sklearn.covariance import LedoitWolf

            lw = LedoitWolf()
            lw.fit(self.returns)
            cov_matrix = pd.DataFrame(
                lw.covariance_ * self.frequency,
                index=self.strategy_names,
                columns=self.strategy_names
            )
            return cov_matrix
        except ImportError:
            warnings.warn(
                "sklearn 未安裝，使用標準協方差估計。"
                "建議安裝: pip install scikit-learn"
            )
            return self.returns.cov() * self.frequency

    def _portfolio_performance(
        self,
        weights: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        計算組合績效

        Args:
            weights: 權重陣列

        Returns:
            (expected_return, volatility, sharpe_ratio)
        """
        # 預期報酬
        portfolio_return = np.dot(weights, self.mean_returns.values)

        # 波動率（標準差）
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(self.cov_matrix.values, weights))
        )

        # Sharpe Ratio
        sharpe_ratio = (
            (portfolio_return - self.risk_free_rate) / portfolio_volatility
            if portfolio_volatility > 0 else 0.0
        )

        return portfolio_return, portfolio_volatility, sharpe_ratio

    def _negative_sharpe(self, weights: np.ndarray) -> float:
        """目標函數：負 Sharpe Ratio（用於最小化）"""
        _, _, sharpe = self._portfolio_performance(weights)
        return -sharpe

    def _portfolio_variance(self, weights: np.ndarray) -> float:
        """目標函數：組合變異數（用於風險最小化）"""
        return np.dot(weights.T, np.dot(self.cov_matrix.values, weights))

    def _weights_to_dict(self, weights: np.ndarray) -> Dict[str, float]:
        """將權重陣列轉為字典"""
        return {
            name: float(weight)
            for name, weight in zip(self.strategy_names, weights)
        }

    def _create_portfolio_weights(
        self,
        weights: np.ndarray,
        success: bool,
        message: str = ""
    ) -> PortfolioWeights:
        """建立 PortfolioWeights 物件"""
        ret, vol, sharpe = self._portfolio_performance(weights)

        return PortfolioWeights(
            weights=self._weights_to_dict(weights),
            expected_return=float(ret),
            expected_volatility=float(vol),
            sharpe_ratio=float(sharpe),
            optimization_success=success,
            optimization_message=message
        )

    def mean_variance_optimize(
        self,
        target_return: Optional[float] = None,
        target_risk: Optional[float] = None,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
        allow_short: bool = False
    ) -> PortfolioWeights:
        """
        Mean-Variance 優化（Markowitz）

        Args:
            target_return: 目標報酬率（年化）。若指定，則最小化風險
            target_risk: 目標風險（年化標準差）。若指定，則最大化報酬
            max_weight: 單一策略最大權重
            min_weight: 單一策略最小權重
            allow_short: 是否允許空頭（負權重）

        Returns:
            PortfolioWeights 物件

        Note:
            - 若同時指定 target_return 和 target_risk，優先使用 target_return
            - 若都未指定，則最小化風險（minimum variance portfolio）
        """
        # 初始權重：等權重
        initial_weights = np.array([1.0 / self.n_assets] * self.n_assets)

        # 權重約束
        if allow_short:
            bounds = Bounds(
                lb=[-max_weight if min_weight < 0 else min_weight] * self.n_assets,
                ub=[max_weight] * self.n_assets
            )
        else:
            bounds = Bounds(
                lb=[max(0.0, min_weight)] * self.n_assets,
                ub=[max_weight] * self.n_assets
            )

        # 線性約束：權重總和為 1
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]

        # 目標報酬約束
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.dot(w, self.mean_returns.values) - target_return
            })
            objective = self._portfolio_variance  # 最小化風險

        # 目標風險約束
        elif target_risk is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.sqrt(self._portfolio_variance(w)) - target_risk
            })
            # 最大化報酬 = 最小化負報酬
            objective = lambda w: -np.dot(w, self.mean_returns.values)

        # 無約束：最小化風險
        else:
            objective = self._portfolio_variance

        # 執行優化
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        if not result.success:
            logger.warning(f"Mean-Variance 優化失敗: {result.message}")

        return self._create_portfolio_weights(
            result.x,
            result.success,
            result.message
        )

    def max_sharpe_optimize(
        self,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
        allow_short: bool = False,
        target_weights: Optional[Dict[str, float]] = None
    ) -> PortfolioWeights:
        """
        最大化 Sharpe Ratio 的權重優化

        Args:
            max_weight: 單一策略最大權重
            min_weight: 單一策略最小權重
            allow_short: 是否允許空頭（負權重）
            target_weights: 目標權重（可用於添加追蹤誤差約束）

        Returns:
            PortfolioWeights 物件
        """
        # 初始權重
        initial_weights = np.array([1.0 / self.n_assets] * self.n_assets)

        # 權重約束
        if allow_short:
            bounds = Bounds(
                lb=[-max_weight if min_weight < 0 else min_weight] * self.n_assets,
                ub=[max_weight] * self.n_assets
            )
        else:
            bounds = Bounds(
                lb=[max(0.0, min_weight)] * self.n_assets,
                ub=[max_weight] * self.n_assets
            )

        # 線性約束：權重總和為 1
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]

        # 執行優化
        result = minimize(
            self._negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        if not result.success:
            logger.warning(f"Sharpe Ratio 優化失敗: {result.message}")

        return self._create_portfolio_weights(
            result.x,
            result.success,
            result.message
        )

    def risk_parity_optimize(
        self,
        max_weight: float = 1.0,
        min_weight: float = 0.0
    ) -> PortfolioWeights:
        """
        風險平價（Risk Parity）配置

        使每個策略對組合風險的貢獻相等。

        Args:
            max_weight: 單一策略最大權重
            min_weight: 單一策略最小權重

        Returns:
            PortfolioWeights 物件

        Note:
            風險貢獻 = w_i * (Cov * w)_i / portfolio_variance
            目標：所有風險貢獻相等
        """
        def risk_parity_objective(weights: np.ndarray) -> float:
            """
            目標函數：最小化風險貢獻的變異數

            當所有風險貢獻相等時，變異數為 0
            """
            portfolio_variance = self._portfolio_variance(weights)

            if portfolio_variance < 1e-10:
                return 1e10  # 避免除以零

            # 計算每個資產的邊際風險貢獻
            marginal_contrib = np.dot(self.cov_matrix.values, weights)

            # 計算風險貢獻
            risk_contrib = weights * marginal_contrib / np.sqrt(portfolio_variance)

            # 目標：最小化風險貢獻的標準差
            return np.std(risk_contrib)

        # 初始權重：等權重
        initial_weights = np.array([1.0 / self.n_assets] * self.n_assets)

        # 權重約束（不允許空頭）
        bounds = Bounds(
            lb=[max(0.0, min_weight)] * self.n_assets,
            ub=[max_weight] * self.n_assets
        )

        # 線性約束：權重總和為 1
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]

        # 執行優化
        result = minimize(
            risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        if not result.success:
            logger.warning(f"Risk Parity 優化失敗: {result.message}")

        return self._create_portfolio_weights(
            result.x,
            result.success,
            result.message
        )

    def efficient_frontier(
        self,
        n_points: int = 50,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
        allow_short: bool = False
    ) -> List[PortfolioWeights]:
        """
        計算效率前緣（Efficient Frontier）

        Args:
            n_points: 前緣上的點數
            max_weight: 單一策略最大權重
            min_weight: 單一策略最小權重
            allow_short: 是否允許空頭

        Returns:
            PortfolioWeights 列表（按風險從小到大排序）
        """
        # 計算最小和最大可能報酬
        min_variance_portfolio = self.mean_variance_optimize(
            max_weight=max_weight,
            min_weight=min_weight,
            allow_short=allow_short
        )

        max_return_portfolio = self.mean_variance_optimize(
            target_risk=999.0,  # 大數值，實際上會達到最大報酬
            max_weight=max_weight,
            min_weight=min_weight,
            allow_short=allow_short
        )

        min_return = min_variance_portfolio.expected_return
        max_return = max(
            max_return_portfolio.expected_return,
            self.mean_returns.max()
        )

        # 在最小和最大報酬之間均勻取樣
        target_returns = np.linspace(min_return, max_return, n_points)

        frontier = []
        for target_return in target_returns:
            try:
                portfolio = self.mean_variance_optimize(
                    target_return=target_return,
                    max_weight=max_weight,
                    min_weight=min_weight,
                    allow_short=allow_short
                )
                if portfolio.optimization_success:
                    frontier.append(portfolio)
            except Exception as e:
                logger.debug(f"計算前緣點失敗 (target_return={target_return}): {e}")
                continue

        # 按風險排序
        frontier.sort(key=lambda p: p.expected_volatility)

        return frontier

    def equal_weight_portfolio(self) -> PortfolioWeights:
        """
        等權重組合（基準）

        Returns:
            PortfolioWeights 物件
        """
        weights = np.array([1.0 / self.n_assets] * self.n_assets)
        return self._create_portfolio_weights(
            weights,
            success=True,
            message="Equal weight portfolio"
        )

    def inverse_volatility_portfolio(self) -> PortfolioWeights:
        """
        反波動率加權組合

        權重與波動率成反比，波動率低的策略權重高。

        Returns:
            PortfolioWeights 物件
        """
        # 計算各策略的波動率
        volatilities = np.sqrt(np.diag(self.cov_matrix.values))

        # 反波動率權重（未正規化）
        inv_vol_weights = 1.0 / volatilities

        # 正規化
        weights = inv_vol_weights / inv_vol_weights.sum()

        return self._create_portfolio_weights(
            weights,
            success=True,
            message="Inverse volatility portfolio"
        )

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        取得策略間的相關性矩陣

        Returns:
            相關性矩陣 DataFrame
        """
        return self.returns.corr()

    def plot_efficient_frontier(
        self,
        frontier: List[PortfolioWeights],
        save_path: Optional[str] = None,
        show_assets: bool = True
    ):
        """
        繪製效率前緣

        Args:
            frontier: 效率前緣點列表
            save_path: 儲存路徑（可選）
            show_assets: 是否顯示個別資產
        """
        try:
            import matplotlib.pyplot as plt

            # 提取風險與報酬
            risks = [p.expected_volatility for p in frontier]
            returns = [p.expected_return for p in frontier]
            sharpes = [p.sharpe_ratio for p in frontier]

            # 建立圖表
            fig, ax = plt.subplots(figsize=(10, 6))

            # 繪製效率前緣（顏色依 Sharpe Ratio）
            scatter = ax.scatter(
                risks, returns, c=sharpes,
                cmap='viridis', s=50, alpha=0.6
            )

            # 個別資產
            if show_assets:
                asset_vols = np.sqrt(np.diag(self.cov_matrix.values))
                asset_rets = self.mean_returns.values
                ax.scatter(
                    asset_vols, asset_rets,
                    marker='x', s=100, c='red', label='Individual Strategies'
                )

                # 標註資產名稱
                for i, name in enumerate(self.strategy_names):
                    ax.annotate(
                        name,
                        (asset_vols[i], asset_rets[i]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8
                    )

            # 最大 Sharpe Ratio 點
            max_sharpe_idx = np.argmax(sharpes)
            ax.scatter(
                risks[max_sharpe_idx],
                returns[max_sharpe_idx],
                marker='*', s=500, c='gold',
                edgecolors='black', label='Max Sharpe Ratio'
            )

            # 設定標籤
            ax.set_xlabel('Risk (Volatility)')
            ax.set_ylabel('Expected Return')
            ax.set_title('Efficient Frontier')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Color bar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Sharpe Ratio')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"效率前緣圖表已儲存至: {save_path}")

            return fig

        except ImportError:
            warnings.warn("需要安裝 matplotlib: pip install matplotlib")
            return None
