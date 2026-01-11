"""
AI Loop 主協調器

協調策略選擇、參數優化、驗證、記錄的完整自動化流程。
這是整個 AI Loop 的核心控制器。
"""

import time
import random
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np

from ..strategies.registry import StrategyRegistry
from ..optimizer.bayesian import BayesianOptimizer, OptimizationResult
from ..validator.stages import StageValidator, ValidationResult, ValidationGrade
from ..learning.recorder import ExperimentRecorder
from ..learning.memory import MemoryIntegration, StrategyInsight, TradingLesson
from ..backtester.engine import BacktestEngine, BacktestConfig

logger = logging.getLogger(__name__)


@dataclass
class IterationResult:
    """單次迭代結果"""

    iteration: int
    strategy_name: str
    best_params: Dict[str, Any]
    best_sharpe: float
    validation_grade: str
    passed_stages: int
    recorded: bool
    duration: float  # 秒
    error: Optional[str] = None

    def __repr__(self) -> str:
        if self.error:
            return f"Iteration {self.iteration}: ERROR - {self.error}"

        status = "✓ RECORDED" if self.recorded else "✗ NOT RECORDED"
        return (
            f"Iteration {self.iteration}: {self.strategy_name} | "
            f"Sharpe {self.best_sharpe:.2f} | Grade {self.validation_grade} | "
            f"{status} | {self.duration:.1f}s"
        )


@dataclass
class LoopSummary:
    """Loop 摘要統計"""

    total_iterations: int
    successful_iterations: int
    failed_iterations: int
    recorded_experiments: int

    best_strategy: Optional[str] = None
    best_sharpe: float = 0.0
    best_grade: str = "F"

    total_duration: float = 0.0  # 秒
    avg_iteration_time: float = 0.0

    iteration_results: List[IterationResult] = field(default_factory=list)

    def summary_text(self) -> str:
        """產生摘要報告"""
        success_rate = (
            self.successful_iterations / self.total_iterations * 100
            if self.total_iterations > 0 else 0
        )

        record_rate = (
            self.recorded_experiments / self.successful_iterations * 100
            if self.successful_iterations > 0 else 0
        )

        lines = [
            "\n" + "="*60,
            "AI Loop 執行摘要",
            "="*60,
            f"總迭代次數: {self.total_iterations}",
            f"成功: {self.successful_iterations} ({success_rate:.1f}%)",
            f"失敗: {self.failed_iterations}",
            f"記錄實驗數: {self.recorded_experiments} ({record_rate:.1f}%)",
            "",
            "最佳結果:",
            "-"*60,
            f"策略: {self.best_strategy or 'N/A'}",
            f"Sharpe Ratio: {self.best_sharpe:.2f}",
            f"驗證等級: {self.best_grade}",
            "",
            "效能統計:",
            "-"*60,
            f"總執行時間: {self.total_duration/60:.1f} 分鐘",
            f"平均每次迭代: {self.avg_iteration_time:.1f} 秒",
            "="*60,
        ]

        return "\n".join(lines)


class Orchestrator:
    """
    AI Loop 主協調器

    協調完整的 AI 學習流程：
    1. 策略選擇 (80% 利用 / 20% 探索)
    2. 參數生成 (基於歷史最佳)
    3. 貝葉斯優化
    4. 5 階段驗證
    5. 價值判斷
    6. 記錄 (experiments.json + insights.md + Memory MCP)

    使用範例:
        orchestrator = Orchestrator(config={
            'n_trials': 50,
            'min_sharpe': 1.0,
            'min_stages': 3,
            'max_overfit': 0.5,
            'symbols': ['BTCUSDT', 'ETHUSDT'],
            'timeframes': ['4h'],
            'leverage': 5
        })

        # 執行單次迭代
        result = orchestrator.run_iteration(data_btc, data_eth)

        # 執行多次迭代
        summary = orchestrator.run_loop(n_iterations=10, data_btc, data_eth)
        print(summary.summary_text())
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        seed: Optional[int] = None
    ):
        """
        初始化協調器

        Args:
            config: 配置字典
                {
                    'n_trials': 50,           # Optuna 試驗次數
                    'min_sharpe': 1.0,        # 最低 Sharpe
                    'min_stages': 3,          # 最低通過階段
                    'max_overfit': 0.5,       # 最大過擬合率
                    'symbols': ['BTCUSDT', 'ETHUSDT'],
                    'timeframes': ['4h'],
                    'leverage': 5,
                    'initial_capital': 10000,
                    'maker_fee': 0.0002,
                    'taker_fee': 0.0004
                }
            verbose: 是否顯示詳細訊息
            seed: 隨機數種子（用於可重現性）
        """
        self.config = self._load_default_config()
        if config:
            self.config.update(config)

        self.verbose = verbose

        # 初始化隨機數生成器
        self._rng = random.Random(seed)

        # 初始化各模組
        self.validator = StageValidator()
        self.recorder = ExperimentRecorder()
        self.memory = MemoryIntegration()

        # 迭代統計
        self.iteration_count = 0
        self.recorded_count = 0

        # Loop 摘要
        self.loop_summary = LoopSummary(
            total_iterations=0,
            successful_iterations=0,
            failed_iterations=0,
            recorded_experiments=0
        )

    def _load_default_config(self) -> Dict[str, Any]:
        """載入預設配置"""
        return {
            'n_trials': 50,
            'min_sharpe': 1.0,
            'min_stages': 3,
            'max_overfit': 0.5,
            'symbols': ['BTCUSDT', 'ETHUSDT'],
            'timeframes': ['4h'],
            'leverage': 5,
            'initial_capital': 10000.0,
            'maker_fee': 0.0002,
            'taker_fee': 0.0004,
            'slippage': 0.0001,
        }

    def run_iteration(
        self,
        data_btc: pd.DataFrame,
        data_eth: pd.DataFrame
    ) -> IterationResult:
        """
        執行單次迭代

        流程:
        1. 策略選擇 (StrategySelector)
        2. 參數生成 (ParameterGenerator)
        3. 貝葉斯優化 (BayesianOptimizer)
        4. 5 階段驗證 (StageValidator)
        5. 價值判斷 (should_record)
        6. 記錄 (ExperimentRecorder + Memory MCP)

        Args:
            data_btc: BTC 市場資料
            data_eth: ETH 市場資料

        Returns:
            IterationResult: 迭代結果
        """
        start_time = time.time()
        self.iteration_count += 1

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"開始 Iteration {self.iteration_count}")
            print(f"{'='*60}")

        try:
            # 1. 策略選擇
            strategy_name = self._select_strategy()
            if self.verbose:
                print(f"\n[1/6] 策略選擇: {strategy_name}")

            # 2. 參數空間生成
            param_space = self._generate_param_space(strategy_name)
            if self.verbose:
                print(f"[2/6] 參數空間: {list(param_space.keys())}")

            # 3. 貝葉斯優化
            if self.verbose:
                print(f"[3/6] 貝葉斯優化 ({self.config['n_trials']} trials)...")

            opt_result = self._optimize(
                strategy_name,
                param_space,
                data_btc
            )

            if self.verbose:
                print(f"      最佳 Sharpe: {opt_result.best_value:.2f}")
                print(f"      最佳參數: {opt_result.best_params}")

            # 4. 5 階段驗證
            if self.verbose:
                print(f"[4/6] 5 階段驗證...")

            validation_result = self._validate(
                strategy_name,
                opt_result.best_params,
                data_btc,
                data_eth
            )

            if self.verbose:
                print(f"      驗證等級: {validation_result.grade.value}")
                print(f"      通過階段: {validation_result.passed_stages}/5")

            # 5. 價值判斷
            should_record = self._should_record(validation_result)

            if self.verbose:
                print(f"[5/6] 價值判斷: {'✓ 通過' if should_record else '✗ 不通過'}")

            # 6. 記錄
            recorded = False
            if should_record:
                if self.verbose:
                    print(f"[6/6] 記錄實驗...")

                self._record(
                    strategy_name,
                    opt_result,
                    validation_result
                )
                recorded = True
                self.recorded_count += 1
            else:
                if self.verbose:
                    print(f"[6/6] 跳過記錄（未達標準）")

            # 更新 Loop 摘要
            duration = time.time() - start_time
            self._update_loop_summary(
                strategy_name,
                opt_result,
                validation_result,
                recorded,
                duration
            )

            # 建立結果
            result = IterationResult(
                iteration=self.iteration_count,
                strategy_name=strategy_name,
                best_params=opt_result.best_params,
                best_sharpe=opt_result.best_value,
                validation_grade=validation_result.grade.value,
                passed_stages=validation_result.passed_stages,
                recorded=recorded,
                duration=duration
            )

            if self.verbose:
                print(f"\n{result}")
                print(f"{'='*60}\n")

            return result

        except Exception as e:
            duration = time.time() - start_time
            self.loop_summary.failed_iterations += 1

            error_msg = f"{type(e).__name__}: {str(e)}"

            if self.verbose:
                print(f"\n✗ 迭代失敗: {error_msg}")
                print(f"{'='*60}\n")

            return IterationResult(
                iteration=self.iteration_count,
                strategy_name="unknown",
                best_params={},
                best_sharpe=0.0,
                validation_grade="F",
                passed_stages=0,
                recorded=False,
                duration=duration,
                error=error_msg
            )

    def run_loop(
        self,
        n_iterations: int,
        data_btc: pd.DataFrame,
        data_eth: pd.DataFrame
    ) -> LoopSummary:
        """
        執行多次迭代

        Args:
            n_iterations: 迭代次數
            data_btc: BTC 市場資料
            data_eth: ETH 市場資料

        Returns:
            LoopSummary: Loop 摘要
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"開始 AI Loop (共 {n_iterations} 次迭代)")
            print(f"{'='*60}")

        loop_start_time = time.time()

        for i in range(n_iterations):
            result = self.run_iteration(data_btc, data_eth)
            self.loop_summary.iteration_results.append(result)

            # 每 5 次顯示進度
            if self.verbose and (i + 1) % 5 == 0:
                print(f"\n進度: {i+1}/{n_iterations} 完成")
                print(f"記錄數: {self.recorded_count}")

        # 計算統計
        self.loop_summary.total_duration = time.time() - loop_start_time
        self.loop_summary.avg_iteration_time = (
            self.loop_summary.total_duration / n_iterations
        )

        if self.verbose:
            print(self.loop_summary.summary_text())

        return self.loop_summary

    # ===== 私有方法 =====

    def _select_strategy(self) -> str:
        """
        策略選擇：80% 利用 / 20% 探索

        利用：選擇歷史最佳策略
        探索：隨機選擇策略

        Returns:
            str: 策略名稱
        """
        # 取得所有已註冊策略
        all_strategies = StrategyRegistry.list_all()

        if not all_strategies:
            raise ValueError("沒有已註冊的策略")

        # 80% 機率利用歷史最佳
        if self._rng.random() < 0.8:
            # 查詢歷史最佳策略
            best_experiments = self.recorder.get_best_experiments(
                metric='sharpe_ratio',
                n=5,
                filters={'grade': ['A', 'B']}
            )

            if best_experiments:
                # 從 top 5 中隨機選一個
                exp = self._rng.choice(best_experiments)
                strategy_name = exp.strategy['name']

                if self.verbose:
                    print(f"      [利用] 選擇歷史最佳: {strategy_name}")

                return strategy_name

        # 20% 機率探索新策略
        strategy_name = self._rng.choice(all_strategies)

        if self.verbose:
            print(f"      [探索] 隨機選擇: {strategy_name}")

        return strategy_name

    def _generate_param_space(
        self,
        strategy_name: str
    ) -> Dict[str, Dict]:
        """
        參數空間生成：基於歷史最佳 ±30% 範圍

        Args:
            strategy_name: 策略名稱

        Returns:
            dict: 參數空間定義
        """
        # 取得策略的預設參數空間
        base_param_space = StrategyRegistry.get_param_space(strategy_name)

        # 查詢歷史最佳參數
        best_experiments = self.recorder.query_experiments(
            filters={
                'strategy_type': StrategyRegistry.get(strategy_name).strategy_type,
                'grade': ['A', 'B']
            }
        )

        if not best_experiments:
            # 沒有歷史記錄，使用預設空間
            return base_param_space

        # 找到最佳實驗
        best_exp = max(
            best_experiments,
            key=lambda e: e.results.get('sharpe_ratio', 0)
        )

        best_params = best_exp.parameters

        # 根據最佳參數調整空間（±30%）
        adjusted_space = {}

        for param_name, param_config in base_param_space.items():
            param_type = param_config['type']

            if param_name not in best_params:
                # 沒有歷史數據，使用原始空間
                adjusted_space[param_name] = param_config
                continue

            best_value = best_params[param_name]

            if param_type == 'int':
                # 整數型：±30%，但不超過原始範圍
                margin = max(1, int(best_value * 0.3))
                new_low = max(param_config['low'], best_value - margin)
                new_high = min(param_config['high'], best_value + margin)

                adjusted_space[param_name] = {
                    'type': 'int',
                    'low': new_low,
                    'high': new_high
                }

            elif param_type == 'float':
                # 浮點型：±30%，但不超過原始範圍
                margin = best_value * 0.3
                new_low = max(param_config['low'], best_value - margin)
                new_high = min(param_config['high'], best_value + margin)

                adjusted_space[param_name] = {
                    'type': 'float',
                    'low': new_low,
                    'high': new_high,
                    'step': param_config.get('step')
                }

            else:
                # 類別型：保持原樣
                adjusted_space[param_name] = param_config

        return adjusted_space

    def _optimize(
        self,
        strategy_name: str,
        param_space: Dict[str, Dict],
        data: pd.DataFrame
    ) -> OptimizationResult:
        """
        執行貝葉斯優化

        Args:
            strategy_name: 策略名稱
            param_space: 參數空間
            data: 市場資料

        Returns:
            OptimizationResult: 優化結果
        """
        # 建立策略實例
        strategy_class = StrategyRegistry.get(strategy_name)
        strategy = strategy_class()

        # 建立回測配置
        config = BacktestConfig(
            symbol=self.config['symbols'][0],
            timeframe=self.config['timeframes'][0],
            start_date=data.index[0],
            end_date=data.index[-1],
            initial_capital=self.config['initial_capital'],
            leverage=self.config['leverage'],
            maker_fee=self.config['maker_fee'],
            taker_fee=self.config['taker_fee'],
            slippage=self.config['slippage']
        )

        # 建立回測引擎
        engine = BacktestEngine(config)

        # 建立優化器
        optimizer = BayesianOptimizer(
            engine=engine,
            n_trials=self.config['n_trials'],
            n_jobs=1,
            verbose=False
        )

        # 執行優化
        result = optimizer.optimize(
            strategy=strategy,
            data=data,
            param_space=param_space,
            metric='sharpe_ratio',
            direction='maximize',
            show_progress_bar=self.verbose
        )

        return result

    def _validate(
        self,
        strategy_name: str,
        params: Dict[str, Any],
        data_btc: pd.DataFrame,
        data_eth: pd.DataFrame
    ) -> ValidationResult:
        """
        執行 5 階段驗證

        Args:
            strategy_name: 策略名稱
            params: 策略參數
            data_btc: BTC 資料
            data_eth: ETH 資料

        Returns:
            ValidationResult: 驗證結果
        """
        # 建立策略實例
        strategy_class = StrategyRegistry.get(strategy_name)
        strategy = strategy_class()

        # 建立回測配置
        config = BacktestConfig(
            symbol='BTCUSDT',
            timeframe=self.config['timeframes'][0],
            start_date=data_btc.index[0],
            end_date=data_btc.index[-1],
            initial_capital=self.config['initial_capital'],
            leverage=self.config['leverage'],
            maker_fee=self.config['maker_fee'],
            taker_fee=self.config['taker_fee'],
            slippage=self.config['slippage']
        )

        # 執行驗證
        result = self.validator.validate(
            strategy=strategy,
            data_btc=data_btc,
            data_eth=data_eth,
            params=params,
            config=config
        )

        return result

    def _should_record(
        self,
        validation_result: ValidationResult
    ) -> bool:
        """
        價值判斷：決定是否記錄

        標準：
        1. passed_stages >= min_stages (預設 3)
        2. Sharpe > min_sharpe (預設 1.0)
        3. 過擬合率 < max_overfit (預設 50%)

        Args:
            validation_result: 驗證結果

        Returns:
            bool: 是否記錄
        """
        # 1. 通過階段數
        if validation_result.passed_stages < self.config['min_stages']:
            return False

        # 2. Sharpe Ratio（從驗證結果的 details 中取得）
        base_result = validation_result.details.get('base_result', {})
        sharpe = base_result.get('sharpe_ratio', 0)

        if sharpe < self.config['min_sharpe']:
            return False

        # 3. 過擬合率（WFA Efficiency）
        if '階段4_WalkForward' in validation_result.stage_results:
            wfa_stage = validation_result.stage_results['階段4_WalkForward']
            efficiency = wfa_stage.details.get('efficiency', 0)
            overfit_rate = 1 - efficiency

            if overfit_rate > self.config['max_overfit']:
                return False

        return True

    def _record(
        self,
        strategy_name: str,
        opt_result: OptimizationResult,
        validation_result: ValidationResult
    ):
        """
        記錄實驗結果

        記錄到：
        1. experiments.json (ExperimentRecorder)
        2. insights.md (ExperimentRecorder)
        3. Memory MCP (MemoryIntegration)

        Args:
            strategy_name: 策略名稱
            opt_result: 優化結果
            validation_result: 驗證結果
        """
        # 提取策略資訊
        strategy_class = StrategyRegistry.get(strategy_name)
        strategy_info = {
            'name': strategy_name,
            'type': strategy_class.strategy_type,
            'version': strategy_class.version
        }

        # 提取配置
        config = {
            'symbol': self.config['symbols'][0],
            'timeframe': self.config['timeframes'][0],
            'leverage': self.config['leverage'],
            'initial_capital': self.config['initial_capital']
        }

        # 1. 記錄到 experiments.json
        exp_id = self.recorder.log_experiment(
            result=opt_result.best_backtest_result,
            strategy_info=strategy_info,
            config=config,
            validation_result=validation_result,
            insights=[
                f"Iteration {self.iteration_count}",
                f"Grade {validation_result.grade.value}",
                f"Passed {validation_result.passed_stages}/5 stages"
            ]
        )

        if self.verbose:
            print(f"      記錄實驗: {exp_id}")

        # 2. 記錄到 Memory MCP（印出指引）
        self._record_to_memory(
            strategy_name,
            opt_result,
            validation_result
        )

    def _record_to_memory(
        self,
        strategy_name: str,
        opt_result: OptimizationResult,
        validation_result: ValidationResult
    ):
        """
        記錄到 Memory MCP（印出指引）

        Args:
            strategy_name: 策略名稱
            opt_result: 優化結果
            validation_result: 驗證結果
        """
        # 建立洞察
        insight = StrategyInsight(
            strategy_name=strategy_name,
            symbol=self.config['symbols'][0],
            timeframe=self.config['timeframes'][0],
            best_params=opt_result.best_params,
            sharpe_ratio=opt_result.best_value,
            total_return=opt_result.best_backtest_result.total_return,
            max_drawdown=opt_result.best_backtest_result.max_drawdown,
            win_rate=opt_result.best_backtest_result.win_rate,
            wfa_efficiency=(
                validation_result.stage_results['階段4_WalkForward'].details.get('efficiency', 0)
                if '階段4_WalkForward' in validation_result.stage_results else None
            ),
            wfa_grade=validation_result.grade.value,
            market_conditions="AI Loop 優化結果",
            notes=f"Iteration {self.iteration_count}"
        )

        # 格式化為 Memory 存儲格式
        content, metadata = self.memory.format_strategy_insight(insight)

        # 印出指引（供 Claude 手動呼叫 MCP）
        if self.verbose:
            print("\n" + "="*60)
            print("請使用 Memory MCP 存儲以下內容:")
            print("="*60)
            print(f"Content:\n{content}\n")
            print(f"Metadata: {metadata}\n")
            print("="*60 + "\n")

    def _update_loop_summary(
        self,
        strategy_name: str,
        opt_result: OptimizationResult,
        validation_result: ValidationResult,
        recorded: bool,
        duration: float
    ):
        """
        更新 Loop 摘要

        Args:
            strategy_name: 策略名稱
            opt_result: 優化結果
            validation_result: 驗證結果
            recorded: 是否記錄
            duration: 執行時間
        """
        self.loop_summary.total_iterations += 1
        self.loop_summary.successful_iterations += 1

        if recorded:
            self.loop_summary.recorded_experiments += 1

        # 更新最佳結果
        if opt_result.best_value > self.loop_summary.best_sharpe:
            self.loop_summary.best_strategy = strategy_name
            self.loop_summary.best_sharpe = opt_result.best_value
            self.loop_summary.best_grade = validation_result.grade.value

    def get_loop_summary(self) -> LoopSummary:
        """
        取得 Loop 摘要

        Returns:
            LoopSummary: Loop 摘要
        """
        return self.loop_summary


# 便利函數

def create_orchestrator(
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Orchestrator:
    """
    建立 Orchestrator 實例

    Args:
        config: 配置字典
        verbose: 是否顯示詳細訊息

    Returns:
        Orchestrator: 協調器實例
    """
    return Orchestrator(config=config, verbose=verbose)


def run_ai_loop(
    n_iterations: int,
    data_btc: pd.DataFrame,
    data_eth: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> LoopSummary:
    """
    便利函數：執行 AI Loop

    Args:
        n_iterations: 迭代次數
        data_btc: BTC 市場資料
        data_eth: ETH 市場資料
        config: 配置字典
        verbose: 是否顯示詳細訊息

    Returns:
        LoopSummary: Loop 摘要
    """
    orchestrator = create_orchestrator(config, verbose)
    summary = orchestrator.run_loop(n_iterations, data_btc, data_eth)
    return summary
