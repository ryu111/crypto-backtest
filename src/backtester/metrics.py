"""
績效指標計算器

計算完整的回測績效指標。
"""

import numpy as np
import pandas as pd
from typing import Optional


class MetricsCalculator:
    """
    績效指標計算器

    提供各種回測績效指標的計算方法。
    """

    def __init__(self, risk_free_rate: float = 0.0):
        """
        Args:
            risk_free_rate: 無風險利率（年化）
        """
        self.risk_free_rate = risk_free_rate

    def calculate_sharpe(
        self,
        returns: pd.Series,
        risk_free_rate: Optional[float] = None,
        periods: int = 252
    ) -> float:
        """
        計算夏普比率（Sharpe Ratio）

        衡量每單位風險的超額報酬。

        Args:
            returns: 日報酬率序列
            risk_free_rate: 無風險利率（年化），None 則使用預設值
            periods: 年化週期數（252 for daily, 52 for weekly）

        Returns:
            夏普比率
        """
        if len(returns) == 0:
            return 0.0

        rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate

        excess_returns = returns - rf_rate / periods

        if excess_returns.std() == 0:
            return 0.0

        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(periods)
        return sharpe

    def calculate_sortino(
        self,
        returns: pd.Series,
        target_return: float = 0.0,
        periods: int = 252
    ) -> float:
        """
        計算索提諾比率（Sortino Ratio）

        只考慮下行風險的夏普比率改良版。

        Args:
            returns: 日報酬率序列
            target_return: 目標報酬率
            periods: 年化週期數

        Returns:
            索提諾比率
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - target_return / periods
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(periods)
        return sortino

    def calculate_calmar(
        self,
        annual_return: float,
        max_drawdown: float
    ) -> float:
        """
        計算卡爾馬比率（Calmar Ratio）

        年化報酬率 / 最大回撤。

        Args:
            annual_return: 年化報酬率
            max_drawdown: 最大回撤（正數）

        Returns:
            卡爾馬比率
        """
        if max_drawdown == 0:
            return 0.0

        return annual_return / abs(max_drawdown)

    def calculate_omega(
        self,
        returns: pd.Series,
        threshold: float = 0.0
    ) -> float:
        """
        計算 Omega 比率

        衡量收益與損失的比率。

        Args:
            returns: 報酬率序列
            threshold: 門檻報酬率

        Returns:
            Omega 比率
        """
        if len(returns) == 0:
            return 0.0

        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]

        if losses.sum() == 0:
            return float('inf') if gains.sum() > 0 else 0.0

        omega = gains.sum() / losses.sum()
        return omega

    def calculate_ulcer_index(
        self,
        equity_curve: pd.Series,
        periods: int = 14
    ) -> float:
        """
        計算潰瘍指數（Ulcer Index）

        衡量回撤的深度和持續時間。

        Args:
            equity_curve: 權益曲線
            periods: 計算週期

        Returns:
            潰瘍指數
        """
        if len(equity_curve) < 2:
            return 0.0

        # 計算每日相對於歷史最高點的回撤百分比
        running_max = equity_curve.expanding().max()
        drawdown_pct = ((equity_curve - running_max) / running_max) * 100

        # 計算平方和的平方根
        ulcer = np.sqrt((drawdown_pct ** 2).rolling(periods).mean().iloc[-1])
        return ulcer

    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """
        計算風險價值（Value at Risk）

        在給定信心水準下，預期的最大損失。

        Args:
            returns: 報酬率序列
            confidence_level: 信心水準（0-1）

        Returns:
            VaR 值（負數表示損失）
        """
        if len(returns) == 0:
            return 0.0

        var = np.percentile(returns, (1 - confidence_level) * 100)
        return var

    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """
        計算條件風險價值（Conditional VaR / Expected Shortfall）

        超過 VaR 的平均損失。

        Args:
            returns: 報酬率序列
            confidence_level: 信心水準（0-1）

        Returns:
            CVaR 值
        """
        if len(returns) == 0:
            return 0.0

        var = self.calculate_var(returns, confidence_level)
        cvar = returns[returns <= var].mean()
        return cvar

    def calculate_max_dd_duration(
        self,
        drawdown_series: pd.Series
    ) -> int:
        """
        計算最大回撤持續時間

        Args:
            drawdown_series: 回撤序列

        Returns:
            最大回撤持續天數
        """
        if len(drawdown_series) == 0:
            return 0

        # 找出非零回撤區間
        is_drawdown = drawdown_series != 0
        drawdown_periods = is_drawdown.ne(is_drawdown.shift()).cumsum()

        # 計算每個回撤期間的長度
        dd_durations = drawdown_periods[is_drawdown].value_counts()

        if len(dd_durations) == 0:
            return 0

        max_duration = dd_durations.max()
        return int(max_duration)

    def calculate_information_ratio(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        periods: int = 252
    ) -> float:
        """
        計算資訊比率（Information Ratio）

        衡量相對於基準的超額報酬穩定性。

        Args:
            returns: 策略報酬率
            benchmark_returns: 基準報酬率
            periods: 年化週期數

        Returns:
            資訊比率
        """
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0

        active_returns = returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(periods)

        if tracking_error == 0:
            return 0.0

        ir = (active_returns.mean() * periods) / tracking_error
        return ir

    def calculate_tail_ratio(
        self,
        returns: pd.Series,
        percentile: float = 5.0
    ) -> float:
        """
        計算尾部比率（Tail Ratio）

        衡量右尾（獲利）與左尾（虧損）的比率。

        Args:
            returns: 報酬率序列
            percentile: 百分位數（預設 5%）

        Returns:
            尾部比率（>1 表示獲利尾部較大）
        """
        if len(returns) == 0:
            return 0.0

        right_tail = np.percentile(returns, 100 - percentile)
        left_tail = abs(np.percentile(returns, percentile))

        if left_tail == 0:
            return 0.0

        tail_ratio = right_tail / left_tail
        return tail_ratio

    def calculate_stability(
        self,
        equity_curve: pd.Series
    ) -> float:
        """
        計算權益曲線穩定性

        使用 R² 衡量權益曲線與線性趨勢的擬合度。

        Args:
            equity_curve: 權益曲線

        Returns:
            穩定性係數（0-1，越接近 1 越穩定）
        """
        if len(equity_curve) < 2:
            return 0.0

        x = np.arange(len(equity_curve))
        y = equity_curve.values

        # 線性回歸
        coeffs = np.polyfit(x, y, 1)
        p = np.poly1d(coeffs)
        y_pred = p(x)

        # 計算 R²
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 0.0

        r_squared = 1 - (ss_res / ss_tot)
        return max(0.0, r_squared)  # 確保非負

    def calculate_all(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        annual_return: float,
        max_drawdown: float,
        benchmark_returns: Optional[pd.Series] = None
    ) -> dict:
        """
        計算所有指標

        Args:
            returns: 日報酬率序列
            equity_curve: 權益曲線
            annual_return: 年化報酬率
            max_drawdown: 最大回撤
            benchmark_returns: 基準報酬率（可選）

        Returns:
            包含所有指標的字典
        """
        metrics = {
            'sharpe_ratio': self.calculate_sharpe(returns),
            'sortino_ratio': self.calculate_sortino(returns),
            'calmar_ratio': self.calculate_calmar(annual_return, max_drawdown),
            'omega_ratio': self.calculate_omega(returns),
            'ulcer_index': self.calculate_ulcer_index(equity_curve),
            'var_95': self.calculate_var(returns, 0.95),
            'cvar_95': self.calculate_cvar(returns, 0.95),
            'tail_ratio': self.calculate_tail_ratio(returns),
            'stability': self.calculate_stability(equity_curve),
        }

        if benchmark_returns is not None:
            metrics['information_ratio'] = self.calculate_information_ratio(
                returns, benchmark_returns
            )

        return metrics

    @staticmethod
    def print_metrics(metrics: dict) -> None:
        """
        格式化輸出指標

        Args:
            metrics: 指標字典
        """
        print("\n績效指標")
        print("=" * 50)

        for name, value in metrics.items():
            # 格式化名稱
            display_name = name.replace('_', ' ').title()

            # 格式化數值
            if isinstance(value, float):
                if 'ratio' in name.lower():
                    print(f"{display_name:.<30} {value:>8.2f}")
                elif 'var' in name.lower() or 'cvar' in name.lower():
                    print(f"{display_name:.<30} {value:>8.4f}")
                else:
                    print(f"{display_name:.<30} {value:>8.2f}")
            else:
                print(f"{display_name:.<30} {value}")
