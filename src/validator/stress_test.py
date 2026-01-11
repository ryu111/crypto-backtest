"""
黑天鵝壓力測試器

用於測試策略在極端市場事件下的表現。
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta


# 預設歷史黑天鵝事件
HISTORICAL_EVENTS = {
    'covid_crash_2020': {
        'name': 'COVID-19 崩盤',
        'start': '2020-03-12',
        'end': '2020-03-13',
        'drop': -0.40,
        'description': 'COVID-19 疫情爆發引發的市場崩盤'
    },
    'china_ban_2021': {
        'name': '中國禁令',
        'start': '2021-05-19',
        'end': '2021-05-23',
        'drop': -0.30,
        'description': '中國政府宣布加密貨幣交易禁令'
    },
    'luna_crash_2022': {
        'name': 'LUNA 崩盤',
        'start': '2022-05-09',
        'end': '2022-05-12',
        'drop': -0.50,
        'description': 'Terra/LUNA 生態系統崩潰'
    },
    'ftx_collapse_2022': {
        'name': 'FTX 倒閉',
        'start': '2022-11-08',
        'end': '2022-11-10',
        'drop': -0.25,
        'description': 'FTX 交易所破產危機'
    },
}


@dataclass
class StressTestResult:
    """單一壓力測試結果"""

    event_name: str
    description: str

    # 事件參數
    drop_percentage: float
    duration_days: int

    # 績效指標
    total_return: float
    max_drawdown: float
    sharpe_ratio: float

    # 風險指標
    var_95: float
    cvar_95: float

    # 恢復指標
    recovery_days: Optional[int]
    time_underwater: int

    # 交易統計
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # 模擬報酬序列
    returns: pd.Series
    equity_curve: pd.Series


@dataclass
class StressTestReport:
    """完整壓力測試報告"""

    n_scenarios: int
    test_results: List[StressTestResult]

    # 整體統計
    average_return: float
    average_max_drawdown: float
    average_recovery_days: float

    # 最差情境
    worst_scenario: str
    worst_return: float
    worst_drawdown: float

    # 最佳情境
    best_scenario: str
    best_return: float

    # 通過率
    survival_rate: float  # 不爆倉的比例
    profit_rate: float    # 獲利的比例


class StressTester:
    """
    黑天鵝壓力測試器

    實作多種壓力測試方法，用於評估策略在極端市場
    事件下的表現和穩健性。
    """

    def __init__(
        self,
        survival_threshold: float = -0.5,
        risk_free_rate: float = 0.0
    ):
        """
        Args:
            survival_threshold: 爆倉閾值（例如 -0.5 表示虧損 50% 視為爆倉）
            risk_free_rate: 無風險利率（年化）
        """
        self.survival_threshold = survival_threshold
        self.risk_free_rate = risk_free_rate

    def replay_historical_event(
        self,
        strategy_returns: pd.Series,
        event_name: str,
        custom_event: Optional[Dict] = None
    ) -> StressTestResult:
        """
        重播歷史黑天鵝事件，計算策略表現

        Args:
            strategy_returns: 策略報酬率序列（日報酬率）
            event_name: 事件名稱（HISTORICAL_EVENTS 中的 key）
            custom_event: 自定義事件參數（覆蓋預設）

        Returns:
            StressTestResult 物件
        """
        # 取得事件參數
        if custom_event:
            event = custom_event
        elif event_name in HISTORICAL_EVENTS:
            event = HISTORICAL_EVENTS[event_name]
        else:
            raise ValueError(
                f"未知事件: {event_name}. "
                f"可用事件: {list(HISTORICAL_EVENTS.keys())}"
            )

        # 解析事件參數
        name = event.get('name', event_name)
        description = event.get('description', '')
        drop = event['drop']
        start_date = pd.to_datetime(event['start'])
        end_date = pd.to_datetime(event['end'])
        duration = (end_date - start_date).days + 1

        # 在策略報酬中注入黑天鵝事件
        stressed_returns = self._inject_shock(
            returns=strategy_returns,
            shock_magnitude=drop,
            shock_duration=duration
        )

        # 計算績效指標
        metrics = self._calculate_metrics(stressed_returns)

        # 計算恢復時間
        equity_curve = (1 + stressed_returns).cumprod()
        recovery_days, time_underwater = self._calculate_recovery(equity_curve)

        # 計算交易統計
        winning_trades = (stressed_returns > 0).sum()
        losing_trades = (stressed_returns < 0).sum()
        total_trades = len(stressed_returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        return StressTestResult(
            event_name=name,
            description=description,
            drop_percentage=drop,
            duration_days=duration,
            total_return=metrics['total_return'],
            max_drawdown=metrics['max_drawdown'],
            sharpe_ratio=metrics['sharpe_ratio'],
            var_95=metrics['var_95'],
            cvar_95=metrics['cvar_95'],
            recovery_days=recovery_days,
            time_underwater=time_underwater,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            returns=stressed_returns,
            equity_curve=equity_curve
        )

    def run_scenario(
        self,
        strategy_returns: pd.Series,
        scenario: Dict
    ) -> StressTestResult:
        """
        執行假設情境測試

        Args:
            strategy_returns: 策略報酬率序列
            scenario: 情境參數字典
                {
                    'name': str,           # 情境名稱
                    'drop': float,         # 下跌幅度（負數）
                    'duration': int,       # 持續天數
                    'description': str     # 描述（可選）
                }

        Returns:
            StressTestResult 物件
        """
        # 驗證輸入
        if 'drop' not in scenario or 'duration' not in scenario:
            raise ValueError("scenario 必須包含 'drop' 和 'duration'")

        name = scenario.get('name', 'Custom Scenario')
        description = scenario.get('description', f"下跌 {scenario['drop']:.1%}，持續 {scenario['duration']} 天")
        drop = scenario['drop']
        duration = scenario['duration']

        # 注入衝擊
        stressed_returns = self._inject_shock(
            returns=strategy_returns,
            shock_magnitude=drop,
            shock_duration=duration
        )

        # 計算績效指標
        metrics = self._calculate_metrics(stressed_returns)

        # 計算恢復時間
        equity_curve = (1 + stressed_returns).cumprod()
        recovery_days, time_underwater = self._calculate_recovery(equity_curve)

        # 計算交易統計
        winning_trades = (stressed_returns > 0).sum()
        losing_trades = (stressed_returns < 0).sum()
        total_trades = len(stressed_returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        return StressTestResult(
            event_name=name,
            description=description,
            drop_percentage=drop,
            duration_days=duration,
            total_return=metrics['total_return'],
            max_drawdown=metrics['max_drawdown'],
            sharpe_ratio=metrics['sharpe_ratio'],
            var_95=metrics['var_95'],
            cvar_95=metrics['cvar_95'],
            recovery_days=recovery_days,
            time_underwater=time_underwater,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            returns=stressed_returns,
            equity_curve=equity_curve
        )

    def generate_stress_report(
        self,
        strategy_returns: pd.Series,
        custom_scenarios: Optional[List[Dict]] = None
    ) -> StressTestReport:
        """
        產生完整壓力測試報告

        測試所有預設歷史事件 + 自定義情境

        Args:
            strategy_returns: 策略報酬率序列
            custom_scenarios: 自定義情境列表（可選）

        Returns:
            StressTestReport 物件
        """
        test_results = []

        # 1. 測試所有歷史事件
        for event_name in HISTORICAL_EVENTS:
            result = self.replay_historical_event(
                strategy_returns=strategy_returns,
                event_name=event_name
            )
            test_results.append(result)

        # 2. 測試自定義情境
        if custom_scenarios:
            for scenario in custom_scenarios:
                result = self.run_scenario(
                    strategy_returns=strategy_returns,
                    scenario=scenario
                )
                test_results.append(result)

        # 3. 計算整體統計
        n_scenarios = len(test_results)

        # 平均指標
        average_return = np.mean([r.total_return for r in test_results])
        average_max_drawdown = np.mean([r.max_drawdown for r in test_results])

        # 計算平均恢復天數（排除未恢復的情境）
        recovery_days = [r.recovery_days for r in test_results if r.recovery_days is not None]
        average_recovery_days = np.mean(recovery_days) if recovery_days else float('inf')

        # 找出最差情境
        worst_idx = np.argmin([r.total_return for r in test_results])
        worst_result = test_results[worst_idx]

        # 找出最佳情境
        best_idx = np.argmax([r.total_return for r in test_results])
        best_result = test_results[best_idx]

        # 計算通過率
        survival_count = sum(
            1 for r in test_results
            if r.total_return > self.survival_threshold
        )
        survival_rate = survival_count / n_scenarios

        profit_count = sum(1 for r in test_results if r.total_return > 0)
        profit_rate = profit_count / n_scenarios

        return StressTestReport(
            n_scenarios=n_scenarios,
            test_results=test_results,
            average_return=average_return,
            average_max_drawdown=average_max_drawdown,
            average_recovery_days=average_recovery_days,
            worst_scenario=worst_result.event_name,
            worst_return=worst_result.total_return,
            worst_drawdown=worst_result.max_drawdown,
            best_scenario=best_result.event_name,
            best_return=best_result.total_return,
            survival_rate=survival_rate,
            profit_rate=profit_rate
        )

    def _inject_shock(
        self,
        returns: pd.Series,
        shock_magnitude: float,
        shock_duration: int,
        shock_position: Optional[int] = None
    ) -> pd.Series:
        """
        在報酬序列中注入衝擊

        Args:
            returns: 原始報酬序列
            shock_magnitude: 衝擊幅度（負數表示下跌）
            shock_duration: 衝擊持續天數
            shock_position: 衝擊位置（None 則隨機選擇）

        Returns:
            注入衝擊後的報酬序列
        """
        stressed_returns = returns.copy()

        # 確保有足夠長度
        if len(stressed_returns) < shock_duration:
            raise ValueError(
                f"報酬序列長度 ({len(stressed_returns)}) "
                f"短於衝擊持續時間 ({shock_duration})"
            )

        # 決定衝擊位置
        if shock_position is None:
            # 隨機選擇位置（避免太靠近開頭或結尾）
            max_position = len(stressed_returns) - shock_duration
            shock_position = np.random.randint(
                max(1, len(stressed_returns) // 4),  # 至少在第 1/4 之後
                max_position
            )

        # 將衝擊均勻分布在持續期間
        daily_shock = shock_magnitude / shock_duration

        for i in range(shock_duration):
            idx = shock_position + i
            if idx < len(stressed_returns):
                stressed_returns.iloc[idx] += daily_shock

        return stressed_returns

    def _calculate_metrics(self, returns: pd.Series) -> Dict:
        """
        計算績效指標

        Args:
            returns: 報酬率序列

        Returns:
            指標字典
        """
        # 總報酬
        total_return = (1 + returns).prod() - 1

        # 權益曲線
        equity_curve = (1 + returns).cumprod()

        # 最大回撤
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()

        # Sharpe Ratio
        if returns.std() > 0:
            sharpe_ratio = (
                (returns.mean() - self.risk_free_rate / 252) /
                returns.std() * np.sqrt(252)
            )
        else:
            sharpe_ratio = 0.0

        # VaR 和 CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()

        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
        }

    def _calculate_recovery(
        self,
        equity_curve: pd.Series
    ) -> tuple[Optional[int], int]:
        """
        計算恢復時間

        Args:
            equity_curve: 權益曲線

        Returns:
            (recovery_days, time_underwater)
            - recovery_days: 從最大回撤恢復到新高的天數（None 表示未恢復）
            - time_underwater: 總水下時間（天數）
        """
        # 找到最大回撤點
        running_max = equity_curve.expanding().max()
        drawdown = equity_curve - running_max

        # 找到最低點
        lowest_point_idx = drawdown.idxmin()

        # 在最低點之後找到恢復點（權益超過之前的最高點）
        max_before_lowest = running_max.loc[lowest_point_idx]
        recovery_idx = None

        for idx in equity_curve.index[equity_curve.index > lowest_point_idx]:
            if equity_curve.loc[idx] >= max_before_lowest:
                recovery_idx = idx
                break

        # 計算恢復天數
        if recovery_idx is not None:
            recovery_days = equity_curve.index.get_loc(recovery_idx) - equity_curve.index.get_loc(lowest_point_idx)
        else:
            recovery_days = None

        # 計算總水下時間
        is_underwater = equity_curve < running_max
        time_underwater = is_underwater.sum()

        return recovery_days, time_underwater

    @staticmethod
    def print_result(result: StressTestResult) -> None:
        """
        格式化輸出單一測試結果

        Args:
            result: StressTestResult 物件
        """
        print(f"\n{'=' * 70}")
        print(f"壓力測試: {result.event_name}")
        print(f"{'=' * 70}")
        print(f"描述: {result.description}")
        print(f"衝擊幅度: {result.drop_percentage:.2%}")
        print(f"持續天數: {result.duration_days}")
        print()

        print("績效指標")
        print("-" * 70)
        print(f"總報酬:       {result.total_return:>12.2%}")
        print(f"最大回撤:     {result.max_drawdown:>12.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:>12.2f}")
        print()

        print("風險指標")
        print("-" * 70)
        print(f"VaR (95%):    {result.var_95:>12.4f}")
        print(f"CVaR (95%):   {result.cvar_95:>12.4f}")
        print()

        print("恢復指標")
        print("-" * 70)
        if result.recovery_days is not None:
            print(f"恢復天數:     {result.recovery_days:>12}")
        else:
            print(f"恢復天數:     {'未恢復':>12}")
        print(f"水下時間:     {result.time_underwater:>12} 天")
        print()

        print("交易統計")
        print("-" * 70)
        print(f"總交易數:     {result.total_trades:>12}")
        print(f"獲利交易:     {result.winning_trades:>12}")
        print(f"虧損交易:     {result.losing_trades:>12}")
        print(f"勝率:         {result.win_rate:>11.2%}")
        print()

    @staticmethod
    def print_report(report: StressTestReport) -> None:
        """
        格式化輸出完整壓力測試報告

        Args:
            report: StressTestReport 物件
        """
        print("\n" + "=" * 70)
        print("壓力測試報告")
        print("=" * 70)
        print(f"測試情境數: {report.n_scenarios}")
        print()

        print("整體統計")
        print("-" * 70)
        print(f"平均報酬:         {report.average_return:>12.2%}")
        print(f"平均最大回撤:     {report.average_max_drawdown:>12.2%}")
        if report.average_recovery_days != float('inf'):
            print(f"平均恢復天數:     {report.average_recovery_days:>12.1f}")
        else:
            print(f"平均恢復天數:     {'未恢復':>12}")
        print()

        print("最差情境")
        print("-" * 70)
        print(f"情境:             {report.worst_scenario}")
        print(f"報酬:             {report.worst_return:>12.2%}")
        print(f"最大回撤:         {report.worst_drawdown:>12.2%}")
        print()

        print("最佳情境")
        print("-" * 70)
        print(f"情境:             {report.best_scenario}")
        print(f"報酬:             {report.best_return:>12.2%}")
        print()

        print("通過率")
        print("-" * 70)
        print(f"存活率:           {report.survival_rate:>11.2%}")
        print(f"獲利率:           {report.profit_rate:>11.2%}")
        print()

        print("詳細結果")
        print("-" * 70)
        for i, result in enumerate(report.test_results, 1):
            status = "✓" if result.total_return > 0 else "✗"
            print(
                f"{i:2}. {status} {result.event_name:25} "
                f"報酬: {result.total_return:>8.2%}  "
                f"回撤: {result.max_drawdown:>8.2%}"
            )
        print()
