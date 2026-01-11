"""
5 階段策略驗證系統

依序執行 5 個驗證階段,確保策略真實有效。
參考：.claude/skills/策略驗證/SKILL.md
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from scipy.stats import t as t_dist
from enum import Enum

from ..backtester.engine import BacktestEngine, BacktestConfig, BacktestResult

logger = logging.getLogger(__name__)


class ValidationGrade(Enum):
    """驗證評級"""
    A = "A"  # 優秀：通過所有 5 階段
    B = "B"  # 良好：通過階段 1-4
    C = "C"  # 及格：通過階段 1-3
    D = "D"  # 不及格：未通過階段 3
    F = "F"  # 失敗：未通過階段 1


@dataclass
class StageResult:
    """單一階段驗證結果"""
    passed: bool
    score: float  # 0-100
    details: Dict[str, Any]
    message: str
    threshold: Dict[str, float]  # 門檻值

    def __repr__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status} (Score: {self.score:.1f}/100) - {self.message}"


@dataclass
class ValidationResult:
    """完整驗證結果"""
    grade: ValidationGrade
    passed_stages: int  # 通過幾個階段
    stage_results: Dict[str, StageResult]
    recommendation: str
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """產生摘要報告"""
        lines = [
            "\n" + "="*60,
            "策略驗證結果",
            "="*60,
            f"最終評級: {self.grade.value}",
            f"通過階段: {self.passed_stages}/5",
            "",
            "各階段結果:",
            "-"*60,
        ]

        for stage_name, result in self.stage_results.items():
            lines.append(f"{stage_name}: {result}")

        lines.extend([
            "",
            "建議:",
            "-"*60,
            self.recommendation,
            "="*60,
        ])

        return "\n".join(lines)


class StageValidator:
    """
    5 階段策略驗證器

    階段 1：基礎回測 - 驗證基本獲利能力
    階段 2：統計檢驗 - 確認非隨機結果
    階段 3：穩健性測試 - 驗證參數/時間/標的穩健性
    階段 4：Walk-Forward 分析 - 驗證樣本外表現
    階段 5：Monte Carlo 模擬 - 評估風險分布

    使用範例:
        validator = StageValidator()
        result = validator.validate(
            strategy=my_strategy,
            data_btc=btc_data,
            data_eth=eth_data,
            params={'period': 14}
        )
        print(result.summary())
    """

    def __init__(self):
        self.thresholds = self._load_thresholds()

    def _load_thresholds(self) -> Dict:
        """載入各階段門檻值"""
        return {
            'stage1': {
                'total_return': 0.0,
                'total_trades': 30,
                'sharpe_ratio': 0.5,
                'max_drawdown': 0.3,
                'profit_factor': 1.0,
            },
            'stage2': {
                't_test_p': 0.05,
                'sharpe_ci_excludes_zero': True,
                'skewness_abs': 2.0,
            },
            'stage3': {
                'param_sensitivity': 0.3,  # 30%
                'time_consistency': True,  # 前後半期皆獲利
                'asset_consistency': True,  # BTC/ETH 皆獲利
            },
            'stage4': {
                'wfa_efficiency': 0.5,  # 50%
                'oos_win_rate': 0.5,
                'max_single_window_dd': 0.1,  # -10%
            },
            'stage5': {
                'percentile_5th': 0.0,
                'percentile_1st': -0.3,
                'median_vs_original': 0.5,  # Median > Original * 50%
            },
        }

    def validate(
        self,
        strategy: Any,
        data_btc: pd.DataFrame,
        data_eth: pd.DataFrame,
        params: Optional[Dict] = None,
        config: Optional[BacktestConfig] = None
    ) -> ValidationResult:
        """
        執行完整 5 階段驗證

        Args:
            strategy: 策略物件
            data_btc: BTC 市場資料
            data_eth: ETH 市場資料
            params: 策略參數
            config: 回測配置（可選）

        Returns:
            ValidationResult: 驗證結果
        """
        stage_results = {}
        passed_stages = 0

        # 建立預設配置
        if config is None:
            config = self._create_default_config(data_btc)

        # 階段 1：基礎回測
        print("\n執行階段 1：基礎回測...")
        engine = BacktestEngine(config)
        base_result = engine.run(strategy, params, data_btc)

        stage1 = self.stage1_basic_backtest(base_result)
        stage_results['階段1_基礎回測'] = stage1

        if not stage1.passed:
            return self._early_exit(stage_results, 0)

        passed_stages = 1

        # 階段 2：統計檢驗
        print("執行階段 2：統計檢驗...")
        stage2 = self.stage2_statistical_tests(base_result.daily_returns)
        stage_results['階段2_統計檢驗'] = stage2

        if not stage2.passed:
            return self._early_exit(stage_results, 1)

        passed_stages = 2

        # 階段 3：穩健性測試
        print("執行階段 3：穩健性測試...")
        stage3 = self.stage3_robustness_tests(
            strategy, data_btc, data_eth, params, config
        )
        stage_results['階段3_穩健性'] = stage3

        if not stage3.passed:
            return self._early_exit(stage_results, 2)

        passed_stages = 3

        # 階段 4：Walk-Forward 分析
        print("執行階段 4：Walk-Forward 分析...")
        stage4 = self.stage4_walk_forward(strategy, data_btc, params, config)
        stage_results['階段4_WalkForward'] = stage4

        if not stage4.passed:
            return self._early_exit(stage_results, 3)

        passed_stages = 4

        # 階段 5：Monte Carlo 模擬
        print("執行階段 5：Monte Carlo 模擬...")
        stage5 = self.stage5_monte_carlo(base_result.trades)
        stage_results['階段5_MonteCarlo'] = stage5

        if stage5.passed:
            passed_stages = 5

        # 計算最終評級
        grade = self._calculate_grade(passed_stages)
        recommendation = self._generate_recommendation(grade, stage_results)

        return ValidationResult(
            grade=grade,
            passed_stages=passed_stages,
            stage_results=stage_results,
            recommendation=recommendation,
            details={
                'base_result': base_result.to_dict(),
                'params': params or {},
            }
        )

    def stage1_basic_backtest(self, result: BacktestResult) -> StageResult:
        """
        階段 1：基礎回測驗證

        門檻：
        - total_return > 0
        - total_trades >= 30
        - sharpe_ratio > 0.5
        - max_drawdown < 30%
        - profit_factor > 1.0
        """
        thresholds = self.thresholds['stage1']
        checks = {
            'total_return': result.total_return > thresholds['total_return'],
            'total_trades': result.total_trades >= thresholds['total_trades'],
            'sharpe_ratio': result.sharpe_ratio > thresholds['sharpe_ratio'],
            'max_drawdown': abs(result.max_drawdown) < thresholds['max_drawdown'],
            'profit_factor': result.profit_factor > thresholds['profit_factor'],
        }

        passed = all(checks.values())
        score = sum(checks.values()) / len(checks) * 100

        details = {
            'total_return': result.total_return,
            'total_trades': result.total_trades,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'profit_factor': result.profit_factor,
            'checks': checks,
        }

        if passed:
            message = "基礎績效符合要求"
        else:
            failed_checks = [k for k, v in checks.items() if not v]
            message = f"未通過檢查: {', '.join(failed_checks)}"

        return StageResult(
            passed=passed,
            score=score,
            details=details,
            message=message,
            threshold=thresholds
        )

    def stage2_statistical_tests(self, returns: pd.Series) -> StageResult:
        """
        階段 2：統計檢驗

        檢驗：
        - t-test p < 0.05 (拒絕平均報酬為 0 的假設)
        - Sharpe 95% CI 不包含 0
        - 偏態 |skew| < 2 (避免極端分布)
        """
        thresholds = self.thresholds['stage2']

        # 1. t-test: 檢驗平均報酬是否顯著異於 0
        t_stat, p_value = stats.ttest_1samp(returns.dropna(), 0)
        t_test_pass = p_value < thresholds['t_test_p'] and t_stat > 0

        # 2. Sharpe Ratio 95% 信賴區間
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        n = len(returns)
        se_sharpe = np.sqrt((1 + 0.5 * sharpe**2) / n)
        ci_lower = sharpe - 1.96 * se_sharpe
        ci_upper = sharpe + 1.96 * se_sharpe
        ci_excludes_zero = ci_lower > 0

        # 3. 偏態檢驗
        skewness = stats.skew(returns.dropna())
        skew_pass = abs(skewness) < thresholds['skewness_abs']

        checks = {
            't_test': t_test_pass,
            'sharpe_ci': ci_excludes_zero,
            'skewness': skew_pass,
        }

        passed = all(checks.values())
        score = sum(checks.values()) / len(checks) * 100

        details = {
            't_statistic': t_stat,
            'p_value': p_value,
            'sharpe_ratio': sharpe,
            'sharpe_ci': (ci_lower, ci_upper),
            'skewness': skewness,
            'checks': checks,
        }

        if passed:
            message = "統計檢驗通過，非隨機結果"
        else:
            failed = [k for k, v in checks.items() if not v]
            message = f"統計檢驗未通過: {', '.join(failed)}"

        return StageResult(
            passed=passed,
            score=score,
            details=details,
            message=message,
            threshold=thresholds
        )

    def stage3_robustness_tests(
        self,
        strategy: Any,
        data_btc: pd.DataFrame,
        data_eth: pd.DataFrame,
        params: Optional[Dict],
        config: BacktestConfig
    ) -> StageResult:
        """
        階段 3：穩健性測試

        測試：
        1. 參數敏感度 < 30%
        2. 時間一致性（前後半期皆獲利）
        3. 標的一致性（BTC/ETH 皆獲利）
        """
        thresholds = self.thresholds['stage3']
        engine = BacktestEngine(config)

        # 1. 參數敏感度測試
        param_sensitivity = self._test_parameter_sensitivity(
            strategy, data_btc, params, engine
        )
        param_pass = param_sensitivity < thresholds['param_sensitivity']

        # 2. 時間一致性
        time_consistency = self._test_time_consistency(
            strategy, data_btc, params, engine
        )
        time_pass = time_consistency

        # 3. 標的一致性
        asset_consistency = self._test_asset_consistency(
            strategy, data_btc, data_eth, params, engine
        )
        asset_pass = asset_consistency

        checks = {
            'parameter_sensitivity': param_pass,
            'time_consistency': time_pass,
            'asset_consistency': asset_pass,
        }

        passed = all(checks.values())
        score = sum(checks.values()) / len(checks) * 100

        details = {
            'param_sensitivity_pct': param_sensitivity * 100,
            'time_consistent': time_consistency,
            'asset_consistent': asset_consistency,
            'checks': checks,
        }

        if passed:
            message = "穩健性測試通過"
        else:
            failed = [k for k, v in checks.items() if not v]
            message = f"穩健性不足: {', '.join(failed)}"

        return StageResult(
            passed=passed,
            score=score,
            details=details,
            message=message,
            threshold=thresholds
        )

    def _test_parameter_sensitivity(
        self,
        strategy: Any,
        data: pd.DataFrame,
        params: Optional[Dict],
        engine: BacktestEngine
    ) -> float:
        """測試參數敏感度"""
        if not params:
            return 0.0

        # 取第一個數值型參數測試
        test_param = None
        for key, value in params.items():
            if isinstance(value, (int, float)) and value > 0:
                test_param = key
                break

        if not test_param:
            return 0.0

        base_value = params[test_param]
        base_result = engine.run(strategy, params, data)
        base_return = base_result.total_return

        # 測試 ±20% 變化
        variations = [0.8, 0.9, 1.1, 1.2]
        returns = []

        for mult in variations:
            test_params = params.copy()
            new_value = base_value * mult

            # 確保整數型參數保持整數
            if isinstance(base_value, int):
                new_value = int(new_value)

            test_params[test_param] = new_value

            try:
                result = engine.run(strategy, test_params, data)
                returns.append(result.total_return)
            except Exception as e:
                logger.debug(f"參數敏感度測試失敗: {e}")
                continue

        if not returns:
            return 1.0  # 無法測試，視為敏感

        # 計算報酬率變異係數
        returns_array = np.array(returns + [base_return])
        cv = np.std(returns_array) / (abs(np.mean(returns_array)) + 1e-10)

        return cv

    def _test_time_consistency(
        self,
        strategy: Any,
        data: pd.DataFrame,
        params: Optional[Dict],
        engine: BacktestEngine
    ) -> bool:
        """測試時間一致性（前後半期）"""
        mid_point = len(data) // 2

        # 前半期
        data_first = data.iloc[:mid_point]
        result_first = engine.run(strategy, params, data_first)

        # 後半期
        data_second = data.iloc[mid_point:]
        result_second = engine.run(strategy, params, data_second)

        # 兩期皆獲利
        return (
            result_first.total_return > 0 and
            result_second.total_return > 0
        )

    def _test_asset_consistency(
        self,
        strategy: Any,
        data_btc: pd.DataFrame,
        data_eth: pd.DataFrame,
        params: Optional[Dict],
        engine: BacktestEngine
    ) -> bool:
        """測試標的一致性（BTC/ETH）"""
        result_btc = engine.run(strategy, params, data_btc)
        result_eth = engine.run(strategy, params, data_eth)

        # 兩個標的皆獲利
        return (
            result_btc.total_return > 0 and
            result_eth.total_return > 0
        )

    def stage4_walk_forward(
        self,
        strategy: Any,
        data: pd.DataFrame,
        params: Optional[Dict],
        config: BacktestConfig
    ) -> StageResult:
        """
        階段 4：Walk-Forward 分析

        門檻：
        - WFA Efficiency >= 50%
        - OOS 勝率 > 50%
        - 無單窗口 > -10%
        """
        thresholds = self.thresholds['stage4']

        # 執行 WFA（6 窗口，訓練:測試 = 3:1）
        wfa_result = self._perform_walk_forward(
            strategy, data, params, config, n_windows=6, train_ratio=0.75
        )

        efficiency = wfa_result['efficiency']
        oos_win_rate = wfa_result['oos_win_rate']
        max_oos_dd = wfa_result['max_oos_dd']

        checks = {
            'efficiency': efficiency >= thresholds['wfa_efficiency'],
            'oos_win_rate': oos_win_rate > thresholds['oos_win_rate'],
            'max_oos_dd': max_oos_dd > -thresholds['max_single_window_dd'],
        }

        passed = all(checks.values())
        score = sum(checks.values()) / len(checks) * 100

        details = {
            'efficiency': efficiency,
            'oos_win_rate': oos_win_rate,
            'max_oos_dd': max_oos_dd,
            'oos_returns': wfa_result['oos_returns'],
            'checks': checks,
        }

        if passed:
            message = "Walk-Forward 驗證通過"
        else:
            failed = [k for k, v in checks.items() if not v]
            message = f"Walk-Forward 未通過: {', '.join(failed)}"

        return StageResult(
            passed=passed,
            score=score,
            details=details,
            message=message,
            threshold=thresholds
        )

    def _perform_walk_forward(
        self,
        strategy: Any,
        data: pd.DataFrame,
        params: Optional[Dict],
        config: BacktestConfig,
        n_windows: int = 6,
        train_ratio: float = 0.75
    ) -> Dict:
        """執行 Walk-Forward 分析"""
        window_size = len(data) // n_windows
        train_size = int(window_size * train_ratio)
        test_size = window_size - train_size

        oos_returns = []
        is_returns = []

        for i in range(n_windows):
            start_idx = i * window_size

            # 訓練期
            train_start = start_idx
            train_end = start_idx + train_size

            # 測試期
            test_start = train_end
            test_end = min(train_end + test_size, len(data))

            if test_end - test_start < 10:  # 測試期太短
                continue

            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]

            # 訓練期回測
            engine = BacktestEngine(config)
            is_result = engine.run(strategy, params, train_data)
            is_returns.append(is_result.total_return)

            # 測試期回測（使用相同參數）
            oos_result = engine.run(strategy, params, test_data)
            oos_returns.append(oos_result.total_return)

        if not oos_returns:
            return {
                'efficiency': 0.0,
                'oos_win_rate': 0.0,
                'max_oos_dd': -1.0,
                'oos_returns': [],
            }

        # WFA Efficiency = OOS 總報酬 / IS 總報酬
        oos_total = np.prod([1 + r for r in oos_returns]) - 1
        is_total = np.prod([1 + r for r in is_returns]) - 1
        efficiency = oos_total / is_total if is_total > 0 else 0

        # OOS 勝率
        oos_win_rate = sum(1 for r in oos_returns if r > 0) / len(oos_returns)

        # 最大單窗口回撤
        max_oos_dd = min(oos_returns)

        return {
            'efficiency': efficiency,
            'oos_win_rate': oos_win_rate,
            'max_oos_dd': max_oos_dd,
            'oos_returns': oos_returns,
        }

    def stage5_monte_carlo(
        self,
        trades: pd.DataFrame,
        n_simulations: int = 1000
    ) -> StageResult:
        """
        階段 5：Monte Carlo 模擬

        門檻：
        - 5th percentile > 0
        - 1st percentile > -30%
        - Median > Original × 50%
        """
        thresholds = self.thresholds['stage5']

        if len(trades) < 30:
            return StageResult(
                passed=False,
                score=0.0,
                details={'error': 'insufficient_trades'},
                message="交易數不足，無法進行 Monte Carlo 模擬",
                threshold=thresholds
            )

        # 提取交易報酬
        trade_returns = trades['Return Pct'].values if 'Return Pct' in trades.columns else trades['PnL'].values / 10000
        original_return = np.prod(1 + trade_returns) - 1

        # Monte Carlo 模擬（隨機重排交易順序）
        simulated_returns = []

        for _ in range(n_simulations):
            shuffled = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
            sim_return = np.prod(1 + shuffled) - 1
            simulated_returns.append(sim_return)

        simulated_returns = np.array(simulated_returns)

        # 計算百分位數
        p1 = np.percentile(simulated_returns, 1)
        p5 = np.percentile(simulated_returns, 5)
        median = np.median(simulated_returns)

        checks = {
            'p5_positive': p5 > thresholds['percentile_5th'],
            'p1_acceptable': p1 > thresholds['percentile_1st'],
            'median_vs_original': median > original_return * thresholds['median_vs_original'],
        }

        passed = all(checks.values())
        score = sum(checks.values()) / len(checks) * 100

        details = {
            'original_return': original_return,
            'p1': p1,
            'p5': p5,
            'median': median,
            'p95': np.percentile(simulated_returns, 95),
            'checks': checks,
        }

        if passed:
            message = "Monte Carlo 模擬通過，風險可控"
        else:
            failed = [k for k, v in checks.items() if not v]
            message = f"Monte Carlo 風險過高: {', '.join(failed)}"

        return StageResult(
            passed=passed,
            score=score,
            details=details,
            message=message,
            threshold=thresholds
        )

    def _calculate_grade(self, passed_stages: int) -> ValidationGrade:
        """計算最終評級"""
        if passed_stages == 5:
            return ValidationGrade.A
        elif passed_stages == 4:
            return ValidationGrade.B
        elif passed_stages == 3:
            return ValidationGrade.C
        elif passed_stages >= 1:
            return ValidationGrade.D
        else:
            return ValidationGrade.F

    def _generate_recommendation(
        self,
        grade: ValidationGrade,
        stage_results: Dict[str, StageResult]
    ) -> str:
        """產生建議"""
        recommendations = {
            ValidationGrade.A: (
                "優秀！策略通過所有驗證階段。\n"
                "建議：\n"
                "- 可以進入實盤測試（小倉位）\n"
                "- 持續監控實盤表現\n"
                "- 定期重新驗證"
            ),
            ValidationGrade.B: (
                "良好！策略通過前 4 階段驗證。\n"
                "建議：\n"
                "- Monte Carlo 風險較高，建議降低倉位\n"
                "- 可考慮加入止損機制\n"
                "- 謹慎進入實盤測試"
            ),
            ValidationGrade.C: (
                "及格！策略通過基礎驗證，但需改進。\n"
                "建議：\n"
                "- Walk-Forward 表現不佳，可能過擬合\n"
                "- 優化參數，提高穩健性\n"
                "- 延長測試期，觀察長期表現"
            ),
            ValidationGrade.D: (
                "不及格！策略穩健性不足。\n"
                "建議：\n"
                "- 檢查參數敏感度過高問題\n"
                "- 測試更多時間段和標的\n"
                "- 重新設計策略邏輯"
            ),
            ValidationGrade.F: (
                "失敗！策略無法通過基礎驗證。\n"
                "建議：\n"
                "- 基礎績效不達標，策略邏輯有問題\n"
                "- 重新設計策略\n"
                "- 不建議實盤"
            ),
        }

        return recommendations[grade]

    def _early_exit(
        self,
        stage_results: Dict[str, StageResult],
        passed_stages: int
    ) -> ValidationResult:
        """提前結束驗證"""
        grade = self._calculate_grade(passed_stages)
        recommendation = self._generate_recommendation(grade, stage_results)

        return ValidationResult(
            grade=grade,
            passed_stages=passed_stages,
            stage_results=stage_results,
            recommendation=recommendation,
            details={'early_exit': True}
        )

    def _create_default_config(self, data: pd.DataFrame) -> BacktestConfig:
        """建立預設回測配置"""
        return BacktestConfig(
            symbol='BTCUSDT',
            timeframe='1h',
            start_date=data.index[0],
            end_date=data.index[-1],
            initial_capital=10000.0,
            leverage=1,
            maker_fee=0.0002,
            taker_fee=0.0004,
            slippage=0.0001,
        )
