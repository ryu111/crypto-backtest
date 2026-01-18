"""
多標的驗證模組

在多個交易標的上測試同一策略，評估策略的跨標的穩定性。

參考：
- .claude/skills/策略驗證/SKILL.md

使用範例：
    validator = MultiAssetValidator()

    result = validator.validate_across_assets(
        strategy=my_strategy,
        params={'period': 14},
        assets={
            'BTC': btc_data,
            'ETH': eth_data
        }
    )

    print(result.summary())
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class AssetResult:
    """單一標的測試結果"""
    asset: str
    n_bars: int

    # 績效指標
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int

    # 是否通過
    is_profitable: bool

    def to_dict(self) -> dict:
        """轉為字典"""
        return {
            'asset': self.asset,
            'n_bars': self.n_bars,
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'n_trades': self.n_trades,
            'is_profitable': self.is_profitable
        }


@dataclass
class CorrelationAnalysis:
    """標的間相關性分析"""
    correlation_matrix: pd.DataFrame
    avg_correlation: float
    high_correlation_pairs: List[tuple]  # (asset1, asset2, correlation)

    def to_dict(self) -> dict:
        """轉為字典"""
        return {
            'avg_correlation': self.avg_correlation,
            'high_correlation_pairs': self.high_correlation_pairs,
            'correlation_matrix': self.correlation_matrix.to_dict() if self.correlation_matrix is not None else {}
        }


@dataclass
class MultiAssetResult:
    """多標的驗證結果"""
    passed: bool
    cross_asset_score: float  # 0-1, 跨標的一致性分數
    asset_results: Dict[str, AssetResult]

    # 統計數據
    n_assets: int
    profitable_assets: int
    all_profitable: bool

    # Sharpe 統計
    sharpe_mean: float
    sharpe_std: float
    sharpe_diff: float  # 最大 Sharpe 差異

    # 相關性分析
    correlation_analysis: Optional[CorrelationAnalysis] = None

    # 詳細資訊
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """生成摘要報告"""
        status = "PASS" if self.passed else "FAIL"
        lines = [
            "",
            "=" * 60,
            f"多標的驗證結果: {status}",
            "=" * 60,
            "",
            f"跨標的一致性分數: {self.cross_asset_score:.1%}",
            f"測試標的數: {self.n_assets}",
            f"獲利標的: {self.profitable_assets}/{self.n_assets}",
            f"全部獲利: {'是' if self.all_profitable else '否'}",
            "",
            "Sharpe 統計:",
            f"  平均: {self.sharpe_mean:.2f}",
            f"  標準差: {self.sharpe_std:.2f}",
            f"  最大差異: {self.sharpe_diff:.2f}",
            "",
            "各標的詳情:",
            "-" * 60
        ]

        for asset, result in self.asset_results.items():
            status_icon = "" if result.is_profitable else ""
            lines.append(
                f"  {asset}: "
                f"Return: {result.total_return:+.2%} | "
                f"Sharpe: {result.sharpe_ratio:.2f} | "
                f"MaxDD: {result.max_drawdown:.2%} | "
                f"Trades: {result.n_trades} {status_icon}"
            )

        # 相關性分析
        if self.correlation_analysis:
            lines.extend([
                "",
                "相關性分析:",
                f"  平均相關性: {self.correlation_analysis.avg_correlation:.2f}",
            ])
            if self.correlation_analysis.high_correlation_pairs:
                lines.append("  高相關性標的對:")
                for a1, a2, corr in self.correlation_analysis.high_correlation_pairs:
                    lines.append(f"    {a1}-{a2}: {corr:.2f}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """轉為字典"""
        return {
            'passed': self.passed,
            'cross_asset_score': self.cross_asset_score,
            'n_assets': self.n_assets,
            'profitable_assets': self.profitable_assets,
            'all_profitable': self.all_profitable,
            'sharpe_mean': self.sharpe_mean,
            'sharpe_std': self.sharpe_std,
            'sharpe_diff': self.sharpe_diff,
            'asset_results': {k: v.to_dict() for k, v in self.asset_results.items()},
            'correlation_analysis': self.correlation_analysis.to_dict() if self.correlation_analysis else None
        }


class MultiAssetValidator:
    """
    多標的驗證器

    在多個交易標的上測試同一策略，評估跨標的穩定性。

    驗證標準：
    - 兩個標的都獲利
    - Sharpe 差異 < 1.0
    - 提供相關性分析

    使用範例：
        validator = MultiAssetValidator()

        result = validator.validate_across_assets(
            strategy=my_strategy,
            params={'period': 14},
            assets={
                'BTC': btc_data,
                'ETH': eth_data
            }
        )

        if result.passed:
            print("策略通過多標的驗證")
    """

    # 預設驗證標準
    MAX_SHARPE_DIFF = 1.0           # 最大 Sharpe 差異
    HIGH_CORRELATION_THRESHOLD = 0.8  # 高相關性閾值

    def __init__(
        self,
        max_sharpe_diff: float = None,
        high_correlation_threshold: float = None,
        require_all_profitable: bool = True
    ):
        """初始化多標的驗證器

        Args:
            max_sharpe_diff: 最大允許的 Sharpe 差異
            high_correlation_threshold: 高相關性閾值
            require_all_profitable: 是否要求所有標的都獲利
        """
        self.max_sharpe_diff = max_sharpe_diff or self.MAX_SHARPE_DIFF
        self.high_correlation_threshold = high_correlation_threshold or self.HIGH_CORRELATION_THRESHOLD
        self.require_all_profitable = require_all_profitable

    def validate_across_assets(
        self,
        strategy: Any,
        params: Dict[str, Any],
        assets: Dict[str, pd.DataFrame],
        backtest_func: Optional[Callable] = None
    ) -> MultiAssetResult:
        """在多個標的上執行驗證

        Args:
            strategy: 策略物件
            params: 策略參數
            assets: 標的資料字典 {'BTC': btc_data, 'ETH': eth_data}
            backtest_func: 自訂回測函數（可選）

        Returns:
            MultiAssetResult: 驗證結果
        """
        if len(assets) < 2:
            raise ValueError("至少需要 2 個標的進行多標的驗證")

        logger.info(f"開始多標的驗證: {list(assets.keys())}")

        # 1. 測試每個標的
        asset_results = {}
        returns_series = {}

        for asset, data in assets.items():
            try:
                if backtest_func:
                    backtest_result = backtest_func(strategy, data, params)
                else:
                    backtest_result = self._default_backtest(strategy, data, params)

                result = AssetResult(
                    asset=asset,
                    n_bars=len(data),
                    total_return=backtest_result.get('total_return', 0.0),
                    sharpe_ratio=backtest_result.get('sharpe_ratio', 0.0),
                    max_drawdown=backtest_result.get('max_drawdown', 0.0),
                    win_rate=backtest_result.get('win_rate', 0.0),
                    n_trades=backtest_result.get('n_trades', 0),
                    is_profitable=backtest_result.get('total_return', 0.0) > 0
                )
                asset_results[asset] = result

                # 收集報酬序列（用於相關性分析）
                if 'returns' in backtest_result:
                    returns_series[asset] = backtest_result['returns']

            except Exception as e:
                logger.error(f"測試標的 {asset} 失敗: {e}")
                asset_results[asset] = AssetResult(
                    asset=asset,
                    n_bars=len(data),
                    total_return=-1.0,
                    sharpe_ratio=-999.0,
                    max_drawdown=1.0,
                    win_rate=0.0,
                    n_trades=0,
                    is_profitable=False
                )

        # 2. 計算統計數據
        sharpes = [r.sharpe_ratio for r in asset_results.values() if r.sharpe_ratio > -900]
        profitable_count = sum(1 for r in asset_results.values() if r.is_profitable)

        sharpe_mean = np.mean(sharpes) if sharpes else 0.0
        sharpe_std = np.std(sharpes) if len(sharpes) > 1 else 0.0
        sharpe_diff = max(sharpes) - min(sharpes) if sharpes else 0.0

        # 3. 相關性分析
        correlation_analysis = None
        if len(returns_series) >= 2:
            correlation_analysis = self._analyze_correlation(returns_series)

        # 4. 計算跨標的一致性分數
        cross_asset_score = self.calculate_cross_asset_score(asset_results)

        # 5. 判斷是否通過
        all_profitable = all(r.is_profitable for r in asset_results.values())

        passed = (
            sharpe_diff <= self.max_sharpe_diff and
            (not self.require_all_profitable or all_profitable)
        )

        result = MultiAssetResult(
            passed=passed,
            cross_asset_score=cross_asset_score,
            asset_results=asset_results,
            n_assets=len(assets),
            profitable_assets=profitable_count,
            all_profitable=all_profitable,
            sharpe_mean=sharpe_mean,
            sharpe_std=sharpe_std,
            sharpe_diff=sharpe_diff,
            correlation_analysis=correlation_analysis,
            details={
                'thresholds': {
                    'max_sharpe_diff': self.max_sharpe_diff,
                    'require_all_profitable': self.require_all_profitable
                }
            }
        )

        status = "PASS" if passed else "FAIL"
        logger.info(f"多標的驗證完成: {status}, 一致性分數: {cross_asset_score:.2%}")

        return result

    def _default_backtest(
        self,
        strategy: Any,
        data: pd.DataFrame,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """預設回測函數"""
        try:
            from ..backtester.engine import BacktestEngine, BacktestConfig

            # 設定策略參數
            if hasattr(strategy, 'set_params'):
                strategy.set_params(**params)
            elif hasattr(strategy, 'params'):
                for k, v in params.items():
                    if hasattr(strategy.params, k):
                        setattr(strategy.params, k, v)

            # 執行回測
            config = BacktestConfig(
                initial_capital=100000,
                leverage=1
            )
            engine = BacktestEngine(config)
            result = engine.run(strategy, data)

            return {
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'n_trades': result.n_trades,
                'returns': getattr(result, 'returns', None)
            }

        except Exception as e:
            logger.warning(f"預設回測失敗: {e}，返回空結果")
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'n_trades': 0
            }

    def _analyze_correlation(
        self,
        returns_series: Dict[str, pd.Series]
    ) -> CorrelationAnalysis:
        """分析標的間的相關性

        Args:
            returns_series: 各標的報酬序列

        Returns:
            CorrelationAnalysis: 相關性分析結果
        """
        # 建立 DataFrame
        returns_df = pd.DataFrame(returns_series)

        # 計算相關性矩陣
        corr_matrix = returns_df.corr()

        # 計算平均相關性（排除對角線）
        n = len(corr_matrix)
        if n > 1:
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            avg_corr = corr_matrix.values[mask].mean()
        else:
            avg_corr = 0.0

        # 找出高相關性標的對
        high_corr_pairs = []
        assets = list(returns_series.keys())
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                corr = corr_matrix.loc[assets[i], assets[j]]
                if abs(corr) >= self.high_correlation_threshold:
                    high_corr_pairs.append((assets[i], assets[j], corr))

        return CorrelationAnalysis(
            correlation_matrix=corr_matrix,
            avg_correlation=avg_corr,
            high_correlation_pairs=high_corr_pairs
        )

    def calculate_cross_asset_score(
        self,
        asset_results: Dict[str, AssetResult]
    ) -> float:
        """計算跨標的一致性分數

        基於三個指標計算綜合分數：
        1. 獲利一致性 (40%)
        2. Sharpe 穩定性 (40%)
        3. 回撤控制 (20%)

        Args:
            asset_results: 各標的測試結果

        Returns:
            float: 0-1 之間的一致性分數
        """
        if not asset_results:
            return 0.0

        results = list(asset_results.values())
        sharpes = [r.sharpe_ratio for r in results if r.sharpe_ratio > -900]
        drawdowns = [r.max_drawdown for r in results]

        # 1. 獲利一致性分數 (0-1)
        profitable_ratio = sum(1 for r in results if r.is_profitable) / len(results)
        profit_score = profitable_ratio

        # 2. Sharpe 穩定性分數 (0-1)
        if len(sharpes) > 1:
            sharpe_diff = max(sharpes) - min(sharpes)
            # 差異越小越好，上限為 max_sharpe_diff
            stability_score = max(0, 1 - min(sharpe_diff / self.max_sharpe_diff, 1))
        else:
            stability_score = 0.5

        # 3. 回撤控制分數 (0-1)
        max_dd = max(drawdowns) if drawdowns else 0.0
        # 回撤 < 10% 滿分，> 30% 零分
        if max_dd <= 0.10:
            dd_score = 1.0
        elif max_dd >= 0.30:
            dd_score = 0.0
        else:
            dd_score = 1 - (max_dd - 0.10) / 0.20

        # 加權平均
        cross_asset_score = (
            0.40 * profit_score +
            0.40 * stability_score +
            0.20 * dd_score
        )

        return cross_asset_score


def validate_multi_asset(
    strategy: Any,
    params: Dict[str, Any],
    assets: Dict[str, pd.DataFrame],
    backtest_func: Optional[Callable] = None
) -> MultiAssetResult:
    """便捷函數：執行多標的驗證

    Args:
        strategy: 策略物件
        params: 策略參數
        assets: 標的資料字典
        backtest_func: 自訂回測函數

    Returns:
        MultiAssetResult: 驗證結果
    """
    validator = MultiAssetValidator()
    return validator.validate_across_assets(strategy, params, assets, backtest_func)
