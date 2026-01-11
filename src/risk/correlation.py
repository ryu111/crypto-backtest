"""
多策略相關性分析模組

提供策略間相關性計算、滾動相關性、極端情況相關性分析。
用於評估多策略組合的分散效果。
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CorrelationMatrix:
    """相關性矩陣結果"""

    matrix: pd.DataFrame
    """相關性矩陣"""

    mean_correlation: float
    """平均相關性"""

    max_correlation: float
    """最大相關性"""

    min_correlation: float
    """最小相關性"""

    diversification_ratio: float
    """分散比率（越低越好）"""


@dataclass
class RollingCorrelation:
    """滾動相關性結果"""

    correlation: pd.Series
    """時間序列相關性"""

    mean: float
    """平均相關性"""

    std: float
    """相關性標準差"""

    max: float
    """最大相關性"""

    min: float
    """最小相關性"""

    regime_changes: int
    """相關性趨勢變化次數"""


@dataclass
class TailCorrelation:
    """尾部相關性結果"""

    left_tail: float
    """左尾相關性（下跌時）"""

    right_tail: float
    """右尾相關性（上漲時）"""

    normal: float
    """正常時期相關性"""

    crisis_correlation: float
    """危機相關性（極端下跌）"""

    left_tail_count: int
    """左尾樣本數"""

    right_tail_count: int
    """右尾樣本數"""


class CorrelationAnalyzer:
    """
    多策略相關性分析器

    分析多個策略收益率之間的相關性，包括：
    - 靜態相關性矩陣
    - 滾動相關性（時間變化）
    - 尾部相關性（極端情況）
    """

    def __init__(self, window: int = 60):
        """
        Args:
            window: 滾動窗口大小（天數）
        """
        self.window = window

    def calculate_correlation_matrix(
        self,
        returns_dict: Dict[str, pd.Series]
    ) -> CorrelationMatrix:
        """
        計算策略間相關性矩陣

        Args:
            returns_dict: 策略名稱到收益率序列的字典
                         例如: {"MA": series1, "RSI": series2}

        Returns:
            CorrelationMatrix 物件

        Raises:
            ValueError: 如果輸入少於 2 個策略
        """
        if len(returns_dict) < 2:
            raise ValueError("至少需要 2 個策略才能計算相關性")

        # 建立 DataFrame
        df = pd.DataFrame(returns_dict)

        # 計算相關性矩陣
        corr_matrix = df.corr()

        # 提取上三角（不含對角線）的相關性
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper_triangle = corr_matrix.where(mask)

        # 統計指標
        correlations = upper_triangle.values[mask]
        mean_corr = np.mean(correlations)
        max_corr = np.max(correlations)
        min_corr = np.min(correlations)

        # 計算分散比率
        # 分散比率 = 平均相關性 / 1.0
        # 越接近 0 表示越分散，越接近 1 表示越集中
        diversification_ratio = (mean_corr + 1) / 2  # 標準化到 [0, 1]

        return CorrelationMatrix(
            matrix=corr_matrix,
            mean_correlation=mean_corr,
            max_correlation=max_corr,
            min_correlation=min_corr,
            diversification_ratio=diversification_ratio
        )

    def rolling_correlation(
        self,
        returns1: pd.Series,
        returns2: pd.Series
    ) -> RollingCorrelation:
        """
        計算滾動相關性

        觀察兩個策略的相關性如何隨時間變化。

        Args:
            returns1: 策略 1 的收益率序列
            returns2: 策略 2 的收益率序列

        Returns:
            RollingCorrelation 物件

        Raises:
            ValueError: 如果序列長度不一致或小於窗口大小
        """
        if len(returns1) != len(returns2):
            raise ValueError("兩個收益率序列長度必須一致")

        if len(returns1) < self.window:
            raise ValueError(f"序列長度 ({len(returns1)}) 必須 >= 窗口大小 ({self.window})")

        # 計算滾動相關性
        rolling_corr = returns1.rolling(window=self.window).corr(returns2)

        # 移除 NaN
        valid_corr = rolling_corr.dropna()

        if len(valid_corr) == 0:
            raise ValueError("沒有足夠的資料計算滾動相關性")

        # 統計指標
        mean = valid_corr.mean()
        std = valid_corr.std()
        max_val = valid_corr.max()
        min_val = valid_corr.min()

        # 計算趨勢變化次數（符號改變）
        regime_changes = self._count_regime_changes(valid_corr)

        return RollingCorrelation(
            correlation=valid_corr,
            mean=mean,
            std=std,
            max=max_val,
            min=min_val,
            regime_changes=regime_changes
        )

    def tail_correlation(
        self,
        returns1: pd.Series,
        returns2: pd.Series,
        threshold: float = -0.02  # 2% 下跌
    ) -> TailCorrelation:
        """
        計算尾部相關性（極端情況）

        分析在極端市場環境下（大漲/大跌），策略的相關性是否增加。
        這對風險管理非常重要，因為危機時期相關性通常會上升。

        Args:
            returns1: 策略 1 的收益率序列
            returns2: 策略 2 的收益率序列
            threshold: 極端事件閾值（負值表示下跌）

        Returns:
            TailCorrelation 物件

        Raises:
            ValueError: 如果序列長度不一致
        """
        if len(returns1) != len(returns2):
            raise ValueError("兩個收益率序列長度必須一致")

        # 建立 DataFrame 方便處理
        df = pd.DataFrame({
            'ret1': returns1,
            'ret2': returns2
        })

        # 計算正常時期相關性（作為基準）
        normal_corr = df['ret1'].corr(df['ret2'])

        # 左尾：ret1 < threshold（下跌）
        left_tail_mask = df['ret1'] < threshold
        left_tail_df = df[left_tail_mask]

        if len(left_tail_df) >= 2:
            left_tail_corr = left_tail_df['ret1'].corr(left_tail_df['ret2'])
        else:
            left_tail_corr = np.nan

        # 右尾：ret1 > -threshold（上漲）
        right_tail_mask = df['ret1'] > -threshold
        right_tail_df = df[right_tail_mask]

        if len(right_tail_df) >= 2:
            right_tail_corr = right_tail_df['ret1'].corr(right_tail_df['ret2'])
        else:
            right_tail_corr = np.nan

        # 危機相關性：更極端的閾值（3%）
        crisis_threshold = threshold * 1.5
        crisis_mask = df['ret1'] < crisis_threshold
        crisis_df = df[crisis_mask]

        if len(crisis_df) >= 2:
            crisis_corr = crisis_df['ret1'].corr(crisis_df['ret2'])
        else:
            crisis_corr = np.nan

        return TailCorrelation(
            left_tail=left_tail_corr if not np.isnan(left_tail_corr) else 0.0,
            right_tail=right_tail_corr if not np.isnan(right_tail_corr) else 0.0,
            normal=normal_corr,
            crisis_correlation=crisis_corr if not np.isnan(crisis_corr) else 0.0,
            left_tail_count=len(left_tail_df),
            right_tail_count=len(right_tail_df)
        )

    def _count_regime_changes(self, series: pd.Series) -> int:
        """
        計算序列中趨勢變化的次數

        Args:
            series: 數值序列

        Returns:
            趨勢變化次數
        """
        if len(series) < 2:
            return 0

        # 計算差分
        diff = series.diff()

        # 計算符號改變次數
        sign_changes = (np.sign(diff[1:].values) != np.sign(diff[:-1].values)).sum()

        return int(sign_changes)

    def analyze_portfolio_diversification(
        self,
        returns_dict: Dict[str, pd.Series],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        分析投資組合的分散效果

        Args:
            returns_dict: 策略名稱到收益率序列的字典
            weights: 策略權重（None 則等權重）

        Returns:
            包含分散效果指標的字典
        """
        # 計算相關性矩陣
        corr_result = self.calculate_correlation_matrix(returns_dict)

        # 如果沒有指定權重，使用等權重
        if weights is None:
            n = len(returns_dict)
            weights = {name: 1.0 / n for name in returns_dict.keys()}

        # 計算組合收益率
        df = pd.DataFrame(returns_dict)
        weight_array = np.array([weights[name] for name in df.columns])
        portfolio_returns = (df * weight_array).sum(axis=1)

        # 計算組合標準差
        portfolio_std = portfolio_returns.std()

        # 計算加權平均個別標準差
        individual_stds = df.std()
        weighted_avg_std = (individual_stds * weight_array).sum()

        # 分散比率 = 組合標準差 / 加權平均個別標準差
        # 值越小表示分散效果越好
        diversification_benefit = 1 - (portfolio_std / weighted_avg_std)

        return {
            'mean_correlation': corr_result.mean_correlation,
            'max_correlation': corr_result.max_correlation,
            'min_correlation': corr_result.min_correlation,
            'portfolio_std': portfolio_std,
            'weighted_avg_std': weighted_avg_std,
            'diversification_benefit': diversification_benefit,
            'diversification_ratio': corr_result.diversification_ratio
        }
