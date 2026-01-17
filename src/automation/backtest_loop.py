"""
BacktestLoop - 使用者導向的回測循環 API

提供簡單易用的接口進行自動化回測優化。
整合 LoopRunner、StrategySelector、BacktestEngine 等元件。

參考：.claude/skills/AI自動化/SKILL.md
"""

import time
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any

from .loop_config import (
    BacktestLoopConfig,
    LoopResult,
    IterationSummary,
    create_default_config,
    create_quick_config,
    create_production_config,
)
from .selector import StrategySelector
from ..strategies.registry import StrategyRegistry
from ..learning.recorder import ExperimentRecorder
from ..backtester.engine import BacktestEngine, BacktestConfig

logger = logging.getLogger(__name__)


class BacktestLoop:
    """
    使用者導向的回測循環系統

    提供簡單的 Context Manager API，自動管理資源，支援暫停/恢復/停止。

    範例:
        config = BacktestLoopConfig(
            strategies=['ma_cross', 'rsi'],
            symbols=['BTCUSDT'],
            n_iterations=100
        )

        with BacktestLoop(config) as loop:
            result = loop.run()
            print(result.summary())

    進階範例:
        def progress_callback(iteration, total, summary):
            print(f"[{iteration}/{total}] {summary.strategy_name}: Sharpe={summary.sharpe_ratio:.2f}")

        with BacktestLoop(config) as loop:
            result = loop.run(progress_callback=progress_callback)

            # 暫停/恢復
            loop.pause()
            time.sleep(5)
            loop.resume()

            # 停止
            if some_condition:
                loop.stop()
    """

    def __init__(self, config: BacktestLoopConfig):
        """
        初始化回測循環

        Args:
            config: BacktestLoopConfig 配置物件
        """
        self.config = config
        config.validate()  # 驗證配置有效性

        # 內部元件（延遲初始化）
        self._engine: Optional[BacktestEngine] = None
        self._selector: Optional[StrategySelector] = None
        self._recorder: Optional[ExperimentRecorder] = None
        self._variation_tracker = None  # VariationTracker（延遲初始化）

        # 執行狀態
        self._is_running = False
        self._is_paused = False
        self._current_iteration = 0
        self._start_time: Optional[datetime] = None

        # 結果收集
        self._results: List[IterationSummary] = []
        self._best_result: Optional[IterationSummary] = None

    def __enter__(self) -> 'BacktestLoop':
        """Context Manager 入口"""
        self._setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager 出口"""
        self._cleanup()
        return False  # 不抑制異常

    def _setup(self):
        """初始化引擎和元件"""
        logger.info("初始化 BacktestLoop 元件...")

        # 🆕 啟動時驗證回測引擎正確性
        self._validate_engine_on_startup()

        # 初始化 VariationTracker（追蹤策略變化，避免重複測試）
        try:
            from .variation_tracker import VariationTracker
            self._variation_tracker = VariationTracker()
        except Exception as e:
            logger.warning(f"VariationTracker 初始化失敗: {e}")
            logger.warning("變化追蹤功能已禁用，將使用隨機採樣")
            self._variation_tracker = None

        # 初始化 ExperimentRecorder
        self._recorder = ExperimentRecorder()

        # 初始化 StrategySelector
        self._selector = StrategySelector(
            strategy_registry=StrategyRegistry,
            experiment_recorder=self._recorder,
            config={
                'epsilon': self.config.epsilon,
                'ucb_c': self.config.ucb_c,
            }
        )

        # 初始化 BacktestEngine
        backtest_config = BacktestConfig(
            symbol=self.config.symbols[0],  # 預設使用第一個標的
            timeframe=self.config.timeframes[0],  # 預設使用第一個時間框架
            start_date=datetime(2020, 1, 1),  # 預設範圍（可後續配置）
            end_date=datetime.now(),
            initial_capital=self.config.initial_capital,
            leverage=self.config.leverage,
            maker_fee=self.config.maker_fee,
            taker_fee=self.config.taker_fee,
            use_polars=True,  # 使用 Polars（策略不支援時會自動 fallback 到 Pandas）
        )

        self._engine = BacktestEngine(backtest_config)

        # 初始化 DataFetcher
        from ..data import DataFetcher
        self._data_fetcher = DataFetcher()

        logger.info("BacktestLoop 初始化完成")

    def _cleanup(self):
        """清理資源"""
        logger.info("清理 BacktestLoop 資源...")

        # 清理引擎
        self._engine = None
        self._selector = None
        self._recorder = None

        logger.info("BacktestLoop 清理完成")

    def _validate_engine_on_startup(self):
        """
        啟動時驗證回測引擎正確性

        在 AI Loop 開始前執行一次驗證，確保：
        - 數值計算正確（Sharpe、MaxDD、Return）
        - 如果驗證失敗，立即停止並報告錯誤

        Raises:
            RuntimeError: 驗證失敗時拋出，包含詳細錯誤訊息
        """
        from ..backtester.validator import BacktestValidator

        logger.info("🔍 驗證回測引擎正確性...")

        validator = BacktestValidator()
        report = validator.validate_level("L2")  # 只驗證數值正確性

        if not report.all_passed:
            failed_tests = [r for r in report.results if not r.success]
            error_msg = "❌ 回測引擎驗證失敗！\n"
            error_msg += "=" * 50 + "\n"
            for test in failed_tests:
                error_msg += f"  ✗ {test.test_name}: {test.message}\n"
            error_msg += "=" * 50 + "\n"
            error_msg += "請修復上述問題後重新啟動。\n"

            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info("✅ 回測引擎驗證通過（L2 數值正確性）")

    def run(self, progress_callback: Optional[Callable[[int, int, IterationSummary], None]] = None) -> LoopResult:
        """
        執行所有迭代

        Args:
            progress_callback: 進度回調函數 (iteration, total, summary)
                範例: lambda i, total, s: print(f"{i}/{total}: {s.strategy_name}")

        Returns:
            LoopResult: 完整執行結果
        """
        self._is_running = True
        self._start_time = datetime.now()
        self._results = []

        logger.info(f"開始執行 {self.config.n_iterations} 次迭代...")

        for i in range(self.config.n_iterations):
            # 檢查停止信號
            if not self._is_running:
                logger.info(f"收到停止信號，已完成 {i} 次迭代")
                break

            # 檢查暫停信號
            while self._is_paused:
                time.sleep(0.1)

            self._current_iteration = i + 1

            try:
                # 執行單次迭代
                result = self._run_iteration(i + 1)
                self._results.append(result)

                # 更新最佳結果
                if self._best_result is None or result.sharpe_ratio > self._best_result.sharpe_ratio:
                    self._best_result = result

                # 進度回調
                if progress_callback:
                    progress_callback(i + 1, self.config.n_iterations, result)

            except Exception as e:
                logger.error(f"迭代 {i + 1} 失敗: {e}", exc_info=True)

                # 記錄失敗結果
                failed_result = IterationSummary(
                    iteration=i + 1,
                    strategy_name="unknown",
                    symbol="unknown",
                    timeframe="unknown",
                    best_params={},
                    sharpe_ratio=0.0,
                    total_return=0.0,
                    max_drawdown=1.0,
                    validation_grade='F',
                    duration_seconds=0.0,
                    timestamp=datetime.now(),
                    passed=False,
                    error=str(e)
                )
                self._results.append(failed_result)

        # 生成最終結果
        loop_result = self._create_loop_result()

        logger.info(f"執行完成！通過率: {loop_result.pass_rate * 100:.1f}%")

        return loop_result

    def _run_iteration(self, iteration: int) -> IterationSummary:
        """
        執行單次迭代

        Args:
            iteration: 迭代編號

        Returns:
            IterationSummary: 迭代摘要
        """
        iteration_start = time.time()

        # 確保元件已初始化
        if self._selector is None:
            raise RuntimeError("StrategySelector not initialized. Use 'with BacktestLoop(config) as loop:'")
        if self._engine is None:
            raise RuntimeError("BacktestEngine not initialized. Use 'with BacktestLoop(config) as loop:'")

        # 1. 選擇策略
        strategy_name = self._selector.select(method=self.config.selection_mode)
        logger.info(f"[{iteration}] 選擇策略: {strategy_name}")

        # 2. 選擇標的和時間框架
        import random
        symbol = random.choice(self.config.symbols)
        timeframe = random.choice(self.config.timeframes)

        # 3. 生成參數（使用 VariationTracker 避免重複測試）
        strategy_class = StrategyRegistry.get(strategy_name)
        if not hasattr(strategy_class, 'param_space'):
            raise AttributeError(f"Strategy {strategy_name} missing param_space attribute")
        param_space = strategy_class.param_space
        strategy_type = getattr(strategy_class, 'strategy_type', 'unknown')

        # 使用 _sample_unique_params 確保不重複測試
        params, variation_hash = self._sample_unique_params(
            strategy_name=strategy_name,
            strategy_type=strategy_type,
            param_space=param_space
        )

        # 4. 獲取市場資料
        data = self._data_fetcher.fetch_ohlcv(symbol, timeframe, limit=5000)
        if len(data) < 100:
            raise ValueError(f"資料不足: {symbol} {timeframe} 只有 {len(data)} 筆")

        # 更新引擎配置
        self._engine.config.symbol = symbol
        self._engine.config.timeframe = timeframe
        self._engine.config.start_date = data.index[0].to_pydatetime()
        self._engine.config.end_date = data.index[-1].to_pydatetime()

        # 載入資料到引擎
        self._engine.load_data(data)

        # 5. 建立策略實例並執行回測
        from ..strategies import create_strategy
        strategy = create_strategy(strategy_name, **params)

        # 執行回測
        backtest_result = self._engine.run(strategy)

        # 提取結果
        sharpe_ratio = float(backtest_result.sharpe_ratio)
        total_return = float(backtest_result.total_return)
        max_drawdown = float(backtest_result.max_drawdown)

        # 6. 執行驗證
        from .validation_runner import ValidationRunner
        validator = ValidationRunner(
            engine=self._engine,
            stages=self.config.validation_stages
        )
        validation_result = validator.validate(
            strategy=strategy,
            params=params,
            data=data,
            symbol=symbol,
            timeframe=timeframe
        )

        passed = validation_result.passed
        grade = validation_result.grade
        wf_sharpe = validation_result.wf_sharpe
        mc_p5 = validation_result.mc_p5_sharpe

        iteration_duration = time.time() - iteration_start

        # 建立摘要
        summary = IterationSummary(
            iteration=iteration,
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            best_params=params,
            sharpe_ratio=sharpe_ratio,
            total_return=total_return,
            max_drawdown=max_drawdown,
            validation_grade=grade,
            duration_seconds=iteration_duration,
            timestamp=datetime.now(),
            wf_sharpe=wf_sharpe,
            mc_p5=mc_p5,
            passed=passed,
        )

        # 更新變化追蹤器狀態
        if self._variation_tracker is not None:
            self._variation_tracker.update_from_experiment(
                variation_hash=variation_hash,
                experiment_id=f"iter_{iteration}_{strategy_name}_{symbol}",
                grade=grade,
                metrics={
                    'sharpe_ratio': sharpe_ratio,
                    'total_return': total_return,
                    'max_drawdown': max_drawdown,
                },
                validation={
                    'passed': passed,
                    'wf_sharpe': wf_sharpe,
                    'mc_p5_sharpe': mc_p5,
                }
            )

        # 更新選擇器統計
        self._selector.update_stats(strategy_name, {
            'passed': passed,
            'sharpe_ratio': sharpe_ratio,
            'params': params
        })

        return summary

    def _sample_params(self, param_space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        從參數空間採樣參數

        Args:
            param_space: 參數空間定義

        Returns:
            參數字典
        """
        import random

        params = {}
        for param_name, param_config in param_space.items():
            param_type = param_config['type']

            if param_type == 'int':
                params[param_name] = random.randint(param_config['low'], param_config['high'])
            elif param_type == 'float':
                params[param_name] = random.uniform(param_config['low'], param_config['high'])
            elif param_type == 'categorical':
                params[param_name] = random.choice(param_config['choices'])

        return params

    def _sample_unique_params(
        self,
        strategy_name: str,
        strategy_type: str,
        param_space: Dict[str, Dict[str, Any]],
        max_retries: int = 10
    ) -> tuple:
        """
        採樣未測試的參數組合

        策略:
        1. 優先使用未測試的登記變化
        2. 否則隨機生成，並檢查是否已測試
        3. 超過重試次數則強制使用（可能重複）

        Args:
            strategy_name: 策略名稱
            strategy_type: 策略類型
            param_space: 參數空間
            max_retries: 最大重試次數

        Returns:
            tuple: (params, variation_hash)
        """
        if self._variation_tracker is None:
            # 沒有追蹤器，直接隨機採樣（仍生成臨時 hash 保持一致性）
            params = self._sample_params(param_space)
            import hashlib
            temp_hash = hashlib.sha256(
                f"{strategy_name}:{sorted(params.items())}".encode()
            ).hexdigest()[:16]
            return params, f"var_{temp_hash}"

        # 1. 檢查是否有未測試的登記變化
        untested = self._variation_tracker.get_untested_variations(strategy_name=strategy_name)
        if untested:
            # 優先使用未測試變化（按註冊時間）
            variation = untested[0]
            logger.info(f"使用未測試變化: {variation.variation_hash[:12]}...")
            return variation.params, variation.variation_hash

        # 2. 隨機生成參數，檢查重複
        for attempt in range(max_retries):
            params = self._sample_params(param_space)
            variation_hash = self._variation_tracker.compute_hash(strategy_name, params)

            # 檢查是否已測試
            if not self._variation_tracker.is_tested(variation_hash):
                # 註冊新變化
                self._variation_tracker.register_variation(
                    strategy_name=strategy_name,
                    strategy_type=strategy_type,
                    params=params,
                    tags=['auto_generated']
                )
                logger.debug(f"生成新變化: {variation_hash[:12]}... (嘗試 {attempt + 1})")
                return params, variation_hash

            # 檢查相似變化
            similar = self._variation_tracker.find_similar_variations(
                params=params,
                strategy_name=strategy_name
            )
            if similar:
                logger.debug(
                    f"變化 {variation_hash[:12]}... 與已測試變化相似，重新採樣 "
                    f"(嘗試 {attempt + 1})"
                )
            else:
                logger.debug(
                    f"變化 {variation_hash[:12]}... 已測試，重新採樣 "
                    f"(嘗試 {attempt + 1})"
                )

        # 3. 超過重試次數，強制使用（記錄警告）
        logger.warning(
            f"超過 {max_retries} 次重試仍重複，強制使用 "
            f"(變化: {variation_hash[:12]}...)"
        )

        # 仍需註冊（避免狀態不一致）
        self._variation_tracker.register_variation(
            strategy_name=strategy_name,
            strategy_type=strategy_type,
            params=params,
            tags=['auto_generated', 'forced_retry']
        )

        return params, variation_hash

    def _create_loop_result(self) -> LoopResult:
        """建立最終結果"""
        total_duration = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0

        # 分離通過和失敗的策略
        passed_strategies = [r for r in self._results if r.passed]
        failed_strategies = [r for r in self._results if not r.passed]

        # 排序通過的策略（按 Sharpe）
        best_strategies = sorted(passed_strategies, key=lambda r: r.sharpe_ratio, reverse=True)

        # 計算統計
        avg_sharpe = sum(r.sharpe_ratio for r in passed_strategies) / len(passed_strategies) if passed_strategies else 0.0
        best_sharpe = max((r.sharpe_ratio for r in passed_strategies), default=0.0)

        # 計算 WF Sharpe（如果執行階段 4）
        wf_sharpes = [r.wf_sharpe for r in passed_strategies if r.wf_sharpe is not None]
        avg_wf_sharpe = sum(wf_sharpes) / len(wf_sharpes) if wf_sharpes else 0.0

        # 通過率
        pass_rate = len(passed_strategies) / len(self._results) if self._results else 0.0

        # 策略統計
        strategy_counts: Dict[str, int] = {}
        strategy_wins: Dict[str, int] = {}

        for result in self._results:
            name = result.strategy_name
            strategy_counts[name] = strategy_counts.get(name, 0) + 1
            if result.passed:
                strategy_wins[name] = strategy_wins.get(name, 0) + 1

        strategy_win_rates = {
            name: strategy_wins.get(name, 0) / count
            for name, count in strategy_counts.items()
        }

        # 提取實驗 ID
        experiment_ids = [r.experiment_id for r in self._results if r.experiment_id]

        return LoopResult(
            iterations_completed=len(self._results),
            total_iterations=self.config.n_iterations,
            best_strategies=best_strategies[:10],  # 前 10 名
            failed_strategies=failed_strategies,
            experiment_ids=experiment_ids,
            duration_seconds=total_duration,
            avg_sharpe=avg_sharpe,
            best_sharpe=best_sharpe,
            avg_wf_sharpe=avg_wf_sharpe,
            pass_rate=pass_rate,
            strategy_counts=strategy_counts,
            strategy_win_rates=strategy_win_rates,
        )

    def pause(self):
        """暫停執行"""
        self._is_paused = True
        logger.info("BacktestLoop 已暫停")

    def resume(self):
        """恢復執行"""
        self._is_paused = False
        logger.info("BacktestLoop 已恢復")

    def stop(self):
        """停止執行"""
        self._is_running = False
        logger.info("BacktestLoop 已停止")

    @property
    def is_running(self) -> bool:
        """是否正在執行"""
        return self._is_running

    @property
    def is_paused(self) -> bool:
        """是否已暫停"""
        return self._is_paused

    @property
    def current_iteration(self) -> int:
        """當前迭代次數"""
        return self._current_iteration

    @property
    def best_result(self) -> Optional[IterationSummary]:
        """目前最佳結果"""
        return self._best_result


# ===== 便利函數 =====

def run_backtest_loop(
    strategies: List[str],
    symbols: List[str],
    n_iterations: int = 100,
    **kwargs
) -> LoopResult:
    """
    快速執行回測循環

    Args:
        strategies: 策略列表
        symbols: 標的列表
        n_iterations: 迭代次數
        **kwargs: 其他配置參數

    Returns:
        LoopResult: 執行結果

    範例:
        result = run_backtest_loop(
            strategies=['ma_cross', 'rsi'],
            symbols=['BTCUSDT'],
            n_iterations=50,
            use_gpu=True
        )
        print(result.summary())
    """
    config = BacktestLoopConfig(
        strategies=strategies,
        symbols=symbols,
        n_iterations=n_iterations,
        **kwargs
    )

    with BacktestLoop(config) as loop:
        return loop.run()


def quick_optimize(
    strategy: str,
    symbol: str = 'BTCUSDT',
    timeframe: str = '1h',
    n_trials: int = 50
) -> IterationSummary:
    """
    快速優化單一策略

    Args:
        strategy: 策略名稱
        symbol: 交易標的
        timeframe: 時間框架
        n_trials: 優化試驗次數

    Returns:
        IterationSummary: 最佳結果

    範例:
        result = quick_optimize('ma_cross', symbol='BTCUSDT', n_trials=30)
        print(f"最佳 Sharpe: {result.sharpe_ratio:.2f}")
        print(f"最佳參數: {result.best_params}")
    """
    config = create_quick_config(
        strategies=[strategy],
        n_iterations=n_trials,
        use_gpu=False
    )
    config.symbols = [symbol]
    config.timeframes = [timeframe]
    config.selection_mode = 'single'  # 單一策略模式

    with BacktestLoop(config) as loop:
        result = loop.run()

        if result.best_strategies:
            return result.best_strategies[0]
        else:
            raise ValueError(f"策略 {strategy} 優化失敗")


def validate_strategy(
    strategy: str,
    params: Dict[str, Any],
    symbol: str = 'BTCUSDT',
    timeframe: str = '1h'
) -> Dict[str, Any]:
    """
    驗證策略（不優化，使用給定參數）

    ⚠️ 警告：此函數目前是佔位符，使用隨機假數據。
    正式驗證邏輯在 BacktestLoop._run_iteration() 中使用 ValidationRunner 實現。

    Args:
        strategy: 策略名稱
        params: 策略參數
        symbol: 交易標的
        timeframe: 時間框架

    Returns:
        Dict: 驗證結果
            {
                'passed': bool,
                'grade': str,
                'sharpe_ratio': float,
                'max_drawdown': float,
                'validation_details': {...}
            }

    範例:
        result = validate_strategy(
            'ma_cross',
            params={'fast_period': 10, 'slow_period': 30},
            symbol='BTCUSDT'
        )
        print(f"驗證{'通過' if result['passed'] else '失敗'}")
    """
    # ⚠️ 佔位符實作 - 正式邏輯使用 ValidationRunner
    import warnings
    warnings.warn(
        "validate_strategy() 使用假數據。請使用 BacktestLoop + ValidationRunner。",
        DeprecationWarning
    )
    import numpy as np

    sharpe = np.random.uniform(0.5, 2.5)
    max_dd = np.random.uniform(0.05, 0.30)

    passed = sharpe > 1.0 and max_dd < 0.30
    grade = 'A' if sharpe > 2.0 else 'B' if sharpe > 1.5 else 'C'

    return {
        'passed': passed,
        'grade': grade,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'params': params,
        'validation_details': {
            'basic': {'passed': True},
            'statistical': {'passed': True},
            'stability': {'passed': True},
            'walk_forward': {'passed': passed, 'wf_sharpe': sharpe * 0.9},
            'monte_carlo': {'passed': passed, 'mc_p5': sharpe * 0.8},
        }
    }


# ===== 範例使用 =====

if __name__ == '__main__':
    # 設定日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 範例 1: 基本使用
    print("=" * 70)
    print("範例 1: 基本使用")
    print("=" * 70)

    config = create_quick_config(
        strategies=['ma_cross', 'rsi'],
        n_iterations=10,
        use_gpu=False
    )

    with BacktestLoop(config) as loop:
        result = loop.run()
        print(result.summary())

    # 範例 2: 進度回調
    print("\n" + "=" * 70)
    print("範例 2: 進度回調")
    print("=" * 70)

    def progress(i, total, summary):
        status = "✓" if summary.passed else "✗"
        print(
            f"[{i}/{total}] {status} {summary.strategy_name} @ {summary.symbol} {summary.timeframe} "
            f"| Sharpe: {summary.sharpe_ratio:.2f} | Return: {summary.total_return:.2%}"
        )

    config = create_quick_config(strategies=['ma_cross'], n_iterations=5)

    with BacktestLoop(config) as loop:
        result = loop.run(progress_callback=progress)

    # 範例 3: 便利函數
    print("\n" + "=" * 70)
    print("範例 3: 便利函數")
    print("=" * 70)

    result = run_backtest_loop(
        strategies=['ma_cross', 'rsi'],
        symbols=['BTCUSDT'],
        n_iterations=10
    )
    print(result.summary())

    # 範例 4: 快速優化
    print("\n" + "=" * 70)
    print("範例 4: 快速優化")
    print("=" * 70)

    best = quick_optimize('ma_cross', n_trials=10)
    print(f"最佳 Sharpe: {best.sharpe_ratio:.2f}")
    print(f"最佳參數: {best.best_params}")

    # 範例 5: 驗證策略
    print("\n" + "=" * 70)
    print("範例 5: 驗證策略")
    print("=" * 70)

    validation_result = validate_strategy(
        'ma_cross',
        params={'fast_period': 10, 'slow_period': 30}
    )
    print(f"驗證{'通過' if validation_result['passed'] else '失敗'}")
    print(f"等級: {validation_result['grade']}")
    print(f"Sharpe: {validation_result['sharpe_ratio']:.2f}")
