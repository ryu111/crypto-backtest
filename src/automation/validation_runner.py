"""
驗證執行器

整合現有驗證器，執行 5 階段策略驗證流程。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
import logging

from ..backtester.engine import BacktestEngine, BacktestResult
from ..optimizer.walk_forward import WalkForwardAnalyzer, WFAResult
from ..strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """單一驗證階段結果"""

    stage: int
    name: str
    passed: bool
    score: float  # 0.0 - 1.0
    details: Dict[str, Any]
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        """轉為字典"""
        return {
            'stage': self.stage,
            'name': self.name,
            'passed': self.passed,
            'score': self.score,
            'details': self.details,
            'error': self.error
        }


@dataclass
class ValidationResult:
    """完整驗證結果"""

    strategy_name: str
    params: Dict[str, Any]
    symbol: str
    timeframe: str

    # 回測結果
    sharpe_ratio: float
    total_return: float
    max_drawdown: float

    # 驗證結果
    stages: List[StageResult]
    grade: str  # A/B/C/D/F
    passed: bool

    # Walk-Forward 結果（如果執行）
    wf_sharpe: Optional[float] = None
    wf_efficiency: Optional[float] = None

    # Monte Carlo 結果（如果執行）
    mc_p5_sharpe: Optional[float] = None
    overfit_probability: Optional[float] = None

    # 額外資訊
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """JSON 序列化"""
        return {
            'strategy_name': self.strategy_name,
            'params': self.params,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'sharpe_ratio': self.sharpe_ratio,
            'total_return': self.total_return,
            'max_drawdown': self.max_drawdown,
            'grade': self.grade,
            'passed': self.passed,
            'wf_sharpe': self.wf_sharpe,
            'wf_efficiency': self.wf_efficiency,
            'mc_p5_sharpe': self.mc_p5_sharpe,
            'overfit_probability': self.overfit_probability,
            'stages': [s.to_dict() for s in self.stages],
            **self.metadata
        }

    def summary(self) -> str:
        """產生摘要報告"""
        stages_status = "\n".join([
            f"  Stage {s.stage}: {s.name} - {'✓ PASS' if s.passed else '✗ FAIL'} (Score: {s.score:.2f})"
            for s in self.stages
        ])

        return f"""
驗證結果摘要
{'='*60}
策略: {self.strategy_name}
參數: {self.params}
標的: {self.symbol} ({self.timeframe})

整體評級: {self.grade}
驗證通過: {'✓ YES' if self.passed else '✗ NO'}

績效指標
{'-'*60}
夏普比率: {self.sharpe_ratio:.2f}
總報酬率: {self.total_return:.2%}
最大回撤: {self.max_drawdown:.2%}

Walk-Forward 驗證
{'-'*60}
WF 夏普: {self.wf_sharpe:.2f if self.wf_sharpe else 'N/A'}
WF 效率: {self.wf_efficiency:.2% if self.wf_efficiency else 'N/A'}

Monte Carlo 驗證
{'-'*60}
MC P5 夏普: {self.mc_p5_sharpe:.2f if self.mc_p5_sharpe else 'N/A'}
過擬合機率: {self.overfit_probability:.2% if self.overfit_probability else 'N/A'}

階段結果
{'-'*60}
{stages_status}
"""


class ValidationRunner:
    """
    驗證執行器

    整合現有的 WalkForwardAnalyzer 和回測引擎，執行 5 階段驗證。

    驗證階段：
    1. 基礎回測 - 檢查邏輯正確性
    2. 統計檢定 - Sharpe > 0 的 t-test
    3. 穩健性測試 - 參數 ±10%
    4. Walk-Forward 驗證 - 防止過擬合
    5. Monte Carlo 模擬 - 隨機性檢驗

    使用範例：
        runner = ValidationRunner(
            engine=backtest_engine,
            stages=[1, 2, 3, 4, 5]
        )

        result = runner.validate(
            strategy=my_strategy,
            params={'period': 20},
            data=market_data,
            symbol='BTCUSDT',
            timeframe='1h'
        )

        print(result.summary())
        print(f"Grade: {result.grade}")
    """

    def __init__(
        self,
        engine: BacktestEngine,
        wfa_analyzer: Optional[WalkForwardAnalyzer] = None,
        stages: List[int] = [4, 5]  # 預設執行 WF + MC
    ):
        """
        初始化驗證執行器

        Args:
            engine: 回測引擎
            wfa_analyzer: Walk-Forward 分析器（可選）
            stages: 要執行的驗證階段列表
        """
        self.engine = engine
        self.wfa_analyzer = wfa_analyzer
        self.stages_to_run = stages

        # 驗證階段配置
        self.stage_configs = {
            1: {'name': 'Basic Backtest', 'method': self._stage_1_basic},
            2: {'name': 'Statistical Test', 'method': self._stage_2_statistical},
            3: {'name': 'Stability Test', 'method': self._stage_3_stability},
            4: {'name': 'Walk-Forward', 'method': self._stage_4_walk_forward},
            5: {'name': 'Monte Carlo', 'method': self._stage_5_monte_carlo}
        }

    def validate(
        self,
        strategy: BaseStrategy,
        params: Dict[str, Any],
        data: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> ValidationResult:
        """
        執行驗證流程

        Args:
            strategy: 策略物件
            params: 策略參數
            data: 市場資料
            symbol: 標的代碼
            timeframe: 時間週期

        Returns:
            ValidationResult: 驗證結果
        """
        logger.info(f"開始驗證策略: {strategy.name} with params: {params}")

        # 執行基礎回測（必須）
        base_result = self.engine.run(strategy, params, data)

        # 執行各階段驗證
        stage_results = []
        for stage_num in sorted(self.stages_to_run):
            if stage_num not in self.stage_configs:
                logger.warning(f"未知驗證階段: {stage_num}")
                continue

            config = self.stage_configs[stage_num]
            logger.info(f"執行 Stage {stage_num}: {config['name']}")

            try:
                stage_result = config['method'](
                    strategy, params, data, base_result
                )
                stage_results.append(stage_result)
            except Exception as e:
                logger.error(f"Stage {stage_num} 失敗: {e}", exc_info=True)
                stage_results.append(StageResult(
                    stage=stage_num,
                    name=config['name'],
                    passed=False,
                    score=0.0,
                    details={},
                    error=str(e)
                ))

        # 計算評級
        grade = self._calculate_grade(stage_results)
        passed = grade in ['A', 'B', 'C']

        # 提取特定指標
        wf_sharpe = None
        wf_efficiency = None
        mc_p5_sharpe = None
        overfit_probability = None

        for sr in stage_results:
            if sr.stage == 4 and sr.passed:
                wf_sharpe = sr.details.get('oos_mean_sharpe')
                wf_efficiency = sr.details.get('efficiency')
            elif sr.stage == 5 and sr.passed:
                mc_p5_sharpe = sr.details.get('p5_sharpe')
                overfit_probability = sr.details.get('overfit_probability')

        return ValidationResult(
            strategy_name=strategy.name,
            params=params,
            symbol=symbol,
            timeframe=timeframe,
            sharpe_ratio=base_result.sharpe_ratio,
            total_return=base_result.total_return,
            max_drawdown=base_result.max_drawdown,
            stages=stage_results,
            grade=grade,
            passed=passed,
            wf_sharpe=wf_sharpe,
            wf_efficiency=wf_efficiency,
            mc_p5_sharpe=mc_p5_sharpe,
            overfit_probability=overfit_probability,
            metadata={
                'total_trades': base_result.total_trades,
                'win_rate': base_result.win_rate,
                'volatility': base_result.volatility
            }
        )

    def _stage_1_basic(
        self,
        strategy: BaseStrategy,
        params: Dict[str, Any],
        data: pd.DataFrame,
        base_result: BacktestResult
    ) -> StageResult:
        """
        Stage 1: 基礎回測 - 檢查邏輯正確性

        檢查項目：
        - 是否有交易產生
        - 是否有明顯錯誤（如無限回撤）
        - 基本統計指標是否合理

        Args:
            strategy: 策略物件
            params: 策略參數
            data: 市場資料
            base_result: 基礎回測結果

        Returns:
            StageResult: 驗證結果
        """
        passed = True
        score = 1.0
        details = {}

        # 檢查交易數量
        if base_result.total_trades < 10:
            passed = False
            score *= 0.5
            details['error'] = f"交易次數不足: {base_result.total_trades}"

        # 檢查回撤合理性
        if base_result.max_drawdown > 0.9:  # 90% 回撤視為異常
            passed = False
            score *= 0.5
            details['error'] = f"回撤過大: {base_result.max_drawdown:.2%}"

        # 檢查夏普比率是否為有限值
        if not np.isfinite(base_result.sharpe_ratio):
            passed = False
            score = 0.0
            details['error'] = "夏普比率無效"

        details.update({
            'total_trades': base_result.total_trades,
            'sharpe_ratio': base_result.sharpe_ratio,
            'max_drawdown': base_result.max_drawdown,
            'total_return': base_result.total_return
        })

        return StageResult(
            stage=1,
            name='Basic Backtest',
            passed=passed,
            score=score,
            details=details
        )

    def _stage_2_statistical(
        self,
        strategy: BaseStrategy,
        params: Dict[str, Any],
        data: pd.DataFrame,
        base_result: BacktestResult
    ) -> StageResult:
        """
        Stage 2: 統計檢定 - Sharpe > 0 的 t-test

        檢查策略報酬是否顯著大於 0

        Args:
            strategy: 策略物件
            params: 策略參數
            data: 市場資料
            base_result: 基礎回測結果

        Returns:
            StageResult: 驗證結果
        """
        returns = base_result.daily_returns.dropna()

        if len(returns) < 30:
            return StageResult(
                stage=2,
                name='Statistical Test',
                passed=False,
                score=0.0,
                details={'error': '樣本數不足 (< 30)'},
                error='樣本數不足'
            )

        # t-test: H0: mean(returns) <= 0
        ttest_result = stats.ttest_1samp(returns, 0, alternative='greater')
        t_stat: float = float(ttest_result[0])  # type: ignore[arg-type]
        p_value: float = float(ttest_result[1])  # type: ignore[arg-type]

        # 顯著性水準 α = 0.05
        passed = bool(p_value < 0.05 and float(returns.mean()) > 0)

        # Score 基於 p-value (越小越好)
        score = float(max(0.0, 1.0 - p_value)) if passed else 0.0

        details = {
            't_statistic': t_stat,
            'p_value': p_value,
            'mean_return': float(returns.mean()),
            'std_return': float(returns.std()),
            'n_samples': len(returns),
            'significant': passed
        }

        return StageResult(
            stage=2,
            name='Statistical Test',
            passed=passed,
            score=score,
            details=details
        )

    def _stage_3_stability(
        self,
        strategy: BaseStrategy,
        params: Dict[str, Any],
        data: pd.DataFrame,
        base_result: BacktestResult
    ) -> StageResult:
        """
        Stage 3: 穩健性測試 - 參數 ±10%

        測試參數微調對績效的影響

        Args:
            strategy: 策略物件
            params: 策略參數
            data: 市場資料
            base_result: 基礎回測結果

        Returns:
            StageResult: 驗證結果
        """
        base_sharpe = base_result.sharpe_ratio

        if not np.isfinite(base_sharpe):
            return StageResult(
                stage=3,
                name='Stability Test',
                passed=False,
                score=0.0,
                details={'error': 'Base Sharpe 無效'},
                error='Base Sharpe 無效'
            )

        # 對每個數值參數進行 ±10% 測試
        param_variations = []
        sharpe_variations = []

        for param_name, param_value in params.items():
            # 只測試數值參數
            if not isinstance(param_value, (int, float)):
                continue

            # 避免參數為 0
            if param_value == 0:
                continue

            # ±10% 變化
            for factor in [0.9, 1.1]:
                new_value = param_value * factor

                # 整數參數取整
                if isinstance(param_value, int):
                    new_value = int(round(new_value))
                    # 避免變成相同值
                    if new_value == param_value:
                        continue

                # 建立新參數組合
                varied_params = params.copy()
                varied_params[param_name] = new_value

                try:
                    result = self.engine.run(strategy, varied_params, data)
                    param_variations.append({
                        'param': param_name,
                        'factor': factor,
                        'value': new_value
                    })
                    sharpe_variations.append(result.sharpe_ratio)
                except Exception as e:
                    logger.warning(f"參數變化測試失敗 {param_name}={new_value}: {e}")
                    continue

        if not sharpe_variations:
            return StageResult(
                stage=3,
                name='Stability Test',
                passed=False,
                score=0.0,
                details={'error': '無法進行參數變化測試'},
                error='無法進行參數變化測試'
            )

        # 計算穩健性指標
        sharpe_variations = np.array(sharpe_variations)
        sharpe_std = np.std(sharpe_variations)
        sharpe_min = np.min(sharpe_variations)

        # 穩健性得分：變異越小越好
        # Score = 1.0 - (std / mean) 且 min > 0.5 * base
        stability_coef = sharpe_std / abs(base_sharpe) if base_sharpe != 0 else float('inf')
        min_ratio = sharpe_min / base_sharpe if base_sharpe > 0 else 0.0

        score = max(0.0, 1.0 - stability_coef) * 0.7 + min(1.0, min_ratio) * 0.3
        passed = stability_coef < 0.5 and min_ratio > 0.5

        details = {
            'base_sharpe': float(base_sharpe),
            'sharpe_variations': sharpe_variations.tolist(),
            'sharpe_std': float(sharpe_std),
            'sharpe_min': float(sharpe_min),
            'sharpe_max': float(np.max(sharpe_variations)),
            'stability_coefficient': float(stability_coef),
            'min_ratio': float(min_ratio),
            'n_variations': len(sharpe_variations)
        }

        return StageResult(
            stage=3,
            name='Stability Test',
            passed=bool(passed),
            score=float(score),
            details=details
        )

    def _stage_4_walk_forward(
        self,
        strategy: BaseStrategy,
        params: Dict[str, Any],
        data: pd.DataFrame,
        base_result: BacktestResult
    ) -> StageResult:
        """
        Stage 4: Walk-Forward 驗證

        使用 WalkForwardAnalyzer 執行滾動窗口驗證

        Args:
            strategy: 策略物件
            params: 策略參數
            data: 市場資料
            base_result: 基礎回測結果

        Returns:
            StageResult: 驗證結果
        """
        if self.wfa_analyzer is None:
            # 如果沒有提供 WFA，建立一個
            self.wfa_analyzer = WalkForwardAnalyzer(
                n_windows=5,
                is_ratio=0.7,
                overlap=0.5,
            )

        # 取得策略名稱
        strategy_name = getattr(strategy, 'name', strategy.__class__.__name__)

        # 建立優化函數（使用已知的最佳參數）
        def optimize_func(is_data: pd.DataFrame) -> tuple:
            """返回已知的最佳參數和 IS Sharpe"""
            # 在 IS 資料上執行回測以取得 Sharpe
            try:
                result = self.engine.run(strategy, params, is_data)
                is_sharpe = result.sharpe_ratio if result.sharpe_ratio else 0.0
            except Exception:
                is_sharpe = 0.0
            return params, is_sharpe

        # 建立評估函數
        def evaluate_func(eval_data: pd.DataFrame, eval_params: Dict[str, Any]) -> Dict[str, float]:
            """評估策略在給定資料上的績效"""
            try:
                result = self.engine.run(strategy, eval_params, eval_data)
                return {
                    'sharpe': result.sharpe_ratio if result.sharpe_ratio else 0.0,
                    'return': result.total_return if result.total_return else 0.0,
                    'max_drawdown': result.max_drawdown if result.max_drawdown else 0.0,
                    'win_rate': result.win_rate if result.win_rate else 0.0,
                }
            except Exception:
                return {'sharpe': 0.0, 'return': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0}

        try:
            # 使用新的 API 呼叫 WFA
            import asyncio

            async def run_wfa():
                return await self.wfa_analyzer.analyze(
                    data=data,
                    strategy_name=strategy_name,
                    optimize_func=optimize_func,
                    evaluate_func=evaluate_func,
                )

            # 檢查是否已有 event loop 在運行
            try:
                loop = asyncio.get_running_loop()
                # 如果有 running loop，使用 nest_asyncio 或創建新線程
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, run_wfa())
                    wfa_result = future.result(timeout=60)
            except RuntimeError:
                # 沒有 running loop，可以直接使用 asyncio.run
                wfa_result = asyncio.run(run_wfa())

            # 評估 WFA 結果
            efficiency = wfa_result.efficiency
            oos_sharpe = wfa_result.oos_mean_sharpe
            consistency = wfa_result.consistency

            # 通過條件：
            # - Efficiency > 0.5 (OOS 報酬至少是 IS 的 50%)
            # - OOS Sharpe > 0.3
            # - Consistency > 0.4 (40% 窗口獲利)
            passed = (
                efficiency > 0.5 and
                oos_sharpe > 0.3 and
                consistency > 0.4
            )

            # Score 基於三個指標的加權平均
            score = (
                min(1.0, efficiency / 1.0) * 0.4 +
                min(1.0, oos_sharpe / 1.0) * 0.4 +
                min(1.0, consistency / 0.8) * 0.2
            )

            details = {
                'efficiency': float(efficiency),
                'oos_mean_sharpe': float(oos_sharpe),
                'consistency': float(consistency),
                'oos_mean_return': float(wfa_result.oos_mean_return),
                'oos_std_return': float(wfa_result.oos_std_return),
                'n_windows': len(wfa_result.windows)
            }

            return StageResult(
                stage=4,
                name='Walk-Forward',
                passed=passed,
                score=score,
                details=details
            )

        except Exception as e:
            logger.error(f"Walk-Forward 驗證失敗: {e}", exc_info=True)
            return StageResult(
                stage=4,
                name='Walk-Forward',
                passed=False,
                score=0.0,
                details={},
                error=str(e)
            )

    def _stage_5_monte_carlo(
        self,
        strategy: BaseStrategy,
        params: Dict[str, Any],
        data: pd.DataFrame,
        base_result: BacktestResult
    ) -> StageResult:
        """
        Stage 5: Monte Carlo 模擬

        隨機打亂交易順序，測試結果是否穩定

        Args:
            strategy: 策略物件
            params: 策略參數
            data: 市場資料
            base_result: 基礎回測結果

        Returns:
            StageResult: 驗證結果
        """
        base_sharpe = base_result.sharpe_ratio
        trades = base_result.trades

        if len(trades) < 30:
            return StageResult(
                stage=5,
                name='Monte Carlo',
                passed=False,
                score=0.0,
                details={'error': '交易次數不足 (< 30)'},
                error='交易次數不足'
            )

        # 執行 1000 次 Monte Carlo 模擬
        n_simulations = 1000
        simulated_sharpes = []

        trade_pnls = np.asarray(trades['PnL'].values)

        for _ in range(n_simulations):
            # 隨機打亂交易順序
            shuffled_pnls = np.random.permutation(trade_pnls)

            # 計算權益曲線
            equity = np.cumsum(shuffled_pnls) + self.engine.config.initial_capital
            returns = np.diff(equity) / equity[:-1]

            # 計算夏普比率
            if len(returns) > 0 and returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(252)
                simulated_sharpes.append(sharpe)

        simulated_sharpes = np.array(simulated_sharpes)

        # 計算百分位數
        p5 = np.percentile(simulated_sharpes, 5)
        p50 = np.percentile(simulated_sharpes, 50)
        p95 = np.percentile(simulated_sharpes, 95)

        # 過擬合機率：實際夏普超過 95 百分位的機率
        overfit_prob = float(np.mean(simulated_sharpes < base_sharpe))

        # 通過條件：
        # - P5 > 0.3 (5% 最差情況仍有正夏普)
        # - 過擬合機率 < 0.7
        passed = bool(float(p5) > 0.3 and overfit_prob < 0.7)

        # Score 基於 P5 和過擬合機率
        score = float(min(1.0, float(p5) / 0.5) * 0.6 + (1.0 - overfit_prob) * 0.4)

        details = {
            'base_sharpe': float(base_sharpe),
            'p5_sharpe': float(p5),
            'p50_sharpe': float(p50),
            'p95_sharpe': float(p95),
            'overfit_probability': float(overfit_prob),
            'n_simulations': n_simulations,
            'n_trades': len(trade_pnls)
        }

        return StageResult(
            stage=5,
            name='Monte Carlo',
            passed=bool(passed),
            score=float(score),
            details=details
        )

    def _calculate_grade(self, results: List[StageResult]) -> str:
        """
        計算評級 A/B/C/D/F

        評級邏輯：
        - A: WF Sharpe > 1.0 且 MC P5 > 0.5
        - B: WF Sharpe > 0.5 且 MC P5 > 0.3
        - C: WF Sharpe > 0.3 且通過基礎驗證
        - D: 通過基礎驗證但 WF 不理想
        - F: 未通過基礎驗證

        Args:
            results: 所有階段結果

        Returns:
            str: 評級 (A/B/C/D/F)
        """
        # 提取關鍵指標
        wf_sharpe = None
        mc_p5 = None
        basic_passed = False

        for sr in results:
            if sr.stage == 1:
                basic_passed = sr.passed
            elif sr.stage == 4 and sr.passed:
                wf_sharpe = sr.details.get('oos_mean_sharpe')
            elif sr.stage == 5 and sr.passed:
                mc_p5 = sr.details.get('p5_sharpe')

        # 未通過基礎驗證
        if not basic_passed:
            return 'F'

        # 沒有執行 WF/MC
        if wf_sharpe is None and mc_p5 is None:
            # 只執行基礎驗證
            avg_score = np.mean([sr.score for sr in results])
            if avg_score > 0.8:
                return 'C'
            elif avg_score > 0.6:
                return 'D'
            else:
                return 'F'

        # 執行了 WF/MC
        if wf_sharpe is not None and mc_p5 is not None:
            if wf_sharpe > 1.0 and mc_p5 > 0.5:
                return 'A'
            elif wf_sharpe > 0.5 and mc_p5 > 0.3:
                return 'B'
            elif wf_sharpe > 0.3:
                return 'C'
            else:
                return 'D'
        elif wf_sharpe is not None:
            if wf_sharpe > 1.0:
                return 'A'
            elif wf_sharpe > 0.5:
                return 'B'
            elif wf_sharpe > 0.3:
                return 'C'
            else:
                return 'D'
        elif mc_p5 is not None:
            if mc_p5 > 0.5:
                return 'B'
            elif mc_p5 > 0.3:
                return 'C'
            else:
                return 'D'

        return 'D'
