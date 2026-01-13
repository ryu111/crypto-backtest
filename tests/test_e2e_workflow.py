"""
E2E 完整回測流程測試

驗證整個回測系統的端到端功能：
1. 策略載入與實例化
2. 資料準備
3. 單一策略回測
4. 參數優化（Optuna）
5. Regime Detection
6. Composite Strategy
7. 學習系統記錄

執行方式:
    pytest tests/test_e2e_workflow.py -v --tb=short
    python tests/test_e2e_workflow.py  # 直接執行
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

# 確保專案根目錄在 path 中
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class E2ETestResult:
    """E2E 測試結果收集器"""

    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}
        self.errors: list = []
        self.warnings: list = []

    def record(self, test_name: str, success: bool, message: str = "", details: Any = None):
        """記錄測試結果"""
        self.results[test_name] = {
            "success": success,
            "message": message,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {message}")
        if not success and details:
            print(f"   詳情: {details}")

    def add_error(self, test_name: str, error: Exception):
        """記錄錯誤"""
        self.errors.append({
            "test": test_name,
            "error": str(error),
            "type": type(error).__name__
        })
        print(f"❌ {test_name}: {type(error).__name__} - {error}")

    def add_warning(self, message: str):
        """記錄警告"""
        self.warnings.append(message)
        print(f"⚠️  {message}")

    def summary(self) -> str:
        """產生測試摘要"""
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r["success"])
        failed = total - passed

        lines = [
            "",
            "=" * 60,
            "E2E 測試報告",
            "=" * 60,
            f"總測試數: {total}",
            f"通過: {passed}",
            f"失敗: {failed}",
            f"錯誤數: {len(self.errors)}",
            f"警告數: {len(self.warnings)}",
            ""
        ]

        if self.errors:
            lines.append("錯誤清單:")
            for err in self.errors:
                lines.append(f"  - [{err['test']}] {err['type']}: {err['error']}")
            lines.append("")

        if self.warnings:
            lines.append("警告清單:")
            for warn in self.warnings:
                lines.append(f"  - {warn}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)


def generate_sample_data(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    產生模擬 OHLCV 資料

    Args:
        n_bars: K 線數量
        seed: 隨機種子

    Returns:
        模擬的 OHLCV DataFrame
    """
    np.random.seed(seed)

    # 時間索引
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=n_bars),
        periods=n_bars,
        freq='4h'
    )

    # 模擬價格走勢（帶趨勢的隨機漫步）
    base_price = 50000
    returns = np.random.normal(0.0002, 0.02, n_bars)  # 微正漂移
    prices = base_price * np.cumprod(1 + returns)

    # 產生 OHLCV
    data = pd.DataFrame(index=dates)
    data['close'] = prices

    # 產生 high/low（基於 close 的隨機偏移）
    volatility = np.abs(np.random.normal(0, 0.01, n_bars))
    data['high'] = data['close'] * (1 + volatility)
    data['low'] = data['close'] * (1 - volatility)

    # open 使用前一根 close (使用 .loc 避免 ChainedAssignment 警告)
    data['open'] = data['close'].shift(1)
    data.loc[data.index[0], 'open'] = base_price

    # 成交量
    data['volume'] = np.random.uniform(100, 1000, n_bars) * 1e6

    # 確保 OHLC 邏輯正確
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    return data


def run_e2e_tests() -> E2ETestResult:
    """執行完整 E2E 測試流程"""

    result = E2ETestResult()
    print("\n" + "=" * 60)
    print("🚀 開始 E2E 回測流程測試")
    print("=" * 60 + "\n")

    # 共享變數
    sample_data = generate_sample_data(500)
    engine = None
    backtest_result = None
    RSIStrategy = None
    MACrossStrategy = None

    # ========================================
    # 測試 1: 策略模組載入
    # ========================================
    print("\n📦 測試 1: 策略模組載入")
    print("-" * 40)

    try:
        from src.strategies import (
            BaseStrategy,
            list_strategies,
            get_strategy,
            create_strategy,
        )
        from src.strategies.momentum.rsi import RSIStrategy as _RSIStrategy
        from src.strategies.trend.ma_cross import MACrossStrategy as _MACrossStrategy
        from src.strategies.composite import CompositeStrategy, SignalAggregation

        RSIStrategy = _RSIStrategy
        MACrossStrategy = _MACrossStrategy

        strategies = list_strategies()
        result.record(
            "1.1 策略註冊表",
            len(strategies) >= 12,
            f"找到 {len(strategies)} 個策略",
            strategies
        )

        # 檢查所有策略可實例化
        failed_strategies = []
        for name in strategies:
            try:
                strategy_class = get_strategy(name)
                strategy = strategy_class()
                if not isinstance(strategy, BaseStrategy):
                    failed_strategies.append(f"{name}: 不是 BaseStrategy")
            except Exception as e:
                failed_strategies.append(f"{name}: {e}")

        result.record(
            "1.2 策略實例化",
            len(failed_strategies) == 0,
            f"{len(strategies) - len(failed_strategies)}/{len(strategies)} 策略可正常實例化",
            failed_strategies if failed_strategies else None
        )

    except Exception as e:
        result.add_error("1.x 策略模組載入", e)
        return result  # 無法繼續

    # ========================================
    # 測試 2: 回測引擎
    # ========================================
    print("\n⚙️ 測試 2: 回測引擎")
    print("-" * 40)

    try:
        from src.backtester.engine import BacktestEngine, BacktestConfig

        config = BacktestConfig(
            symbol='BTCUSDT',
            timeframe='4h',
            start_date=datetime.now() - timedelta(days=60),
            end_date=datetime.now(),
            initial_capital=10000,
            leverage=3,
            maker_fee=0.0002,
            taker_fee=0.0004,
            use_polars=False  # 使用 Pandas 避免 Polars 轉換問題
        )

        engine = BacktestEngine(config)
        result.record("2.1 引擎初始化", True, "BacktestEngine 建立成功")

        # 載入資料
        engine.load_data(sample_data)
        result.record("2.2 資料載入", True, f"載入 {len(sample_data)} 筆資料")

        # 執行回測（明確使用 Pandas 模式）
        strategy = RSIStrategy(trend_filter=False)  # 簡化：不使用趨勢過濾
        backtest_result = engine.run(strategy)

        if backtest_result is not None:
            result.record(
                "2.3 回測執行",
                True,
                f"Sharpe={backtest_result.sharpe_ratio:.4f}, Return={backtest_result.total_return:.2%}"
            )

            # 驗證結果完整性
            required_attrs = ['total_return', 'sharpe_ratio', 'max_drawdown', 'total_trades']
            missing = [attr for attr in required_attrs if not hasattr(backtest_result, attr)]
            result.record(
                "2.4 結果完整性",
                len(missing) == 0,
                f"所有必要指標存在" if not missing else f"缺少: {missing}"
            )
        else:
            result.record("2.3 回測執行", False, "回測結果為 None")

    except Exception as e:
        result.add_error("2.x 回測引擎", e)

    # ========================================
    # 測試 3: 參數優化（簡化版）
    # ========================================
    print("\n🔧 測試 3: 參數優化 (Optuna)")
    print("-" * 40)

    try:
        if engine is None:
            result.add_warning("引擎未初始化，跳過優化測試")
        else:
            from src.optimizer.bayesian import BayesianOptimizer

            # 使用較少的 trials 快速測試
            optimizer = BayesianOptimizer(
                engine=engine,
                n_trials=5,  # 快速測試
                n_jobs=1,
                seed=42,
                verbose=False
            )

            result.record("3.1 優化器初始化", True, "BayesianOptimizer 建立成功")

            # 執行優化
            opt_result = optimizer.optimize(
                strategy=RSIStrategy(trend_filter=False),
                data=sample_data,
                metric='sharpe_ratio'
            )

            if opt_result and opt_result.best_params:
                result.record(
                    "3.2 優化執行",
                    True,
                    f"最佳 Sharpe={opt_result.best_value:.4f}, 參數={opt_result.best_params}"
                )
            else:
                result.record("3.2 優化執行", False, "優化結果為空")

    except ImportError as e:
        result.add_warning(f"Optuna 未安裝，跳過優化測試: {e}")
    except Exception as e:
        result.add_error("3.x 參數優化", e)

    # ========================================
    # 測試 4: Regime Detection
    # ========================================
    print("\n📊 測試 4: Regime Detection")
    print("-" * 40)

    try:
        from src.regime.analyzer import (
            calculate_direction_score,
            volatility_score_atr,  # 正確的函數名
            MarketStateAnalyzer,
            MarketRegime
        )

        # 計算方向分數
        direction = calculate_direction_score(sample_data)
        result.record(
            "4.1 方向分數計算",
            direction is not None and len(direction) > 0,
            f"計算完成，範圍: [{direction.min():.2f}, {direction.max():.2f}]"
        )

        # 計算波動度分數（使用正確的函數名）
        volatility = volatility_score_atr(sample_data)
        result.record(
            "4.2 波動度分數計算",
            volatility is not None and len(volatility) > 0,
            f"計算完成，範圍: [{volatility.min():.2f}, {volatility.max():.2f}]"
        )

        # 市場狀態分析器（使用正確的方法名 calculate_state）
        analyzer = MarketStateAnalyzer()
        state = analyzer.calculate_state(sample_data)

        result.record(
            "4.3 市場狀態分析",
            state is not None,
            f"當前狀態: {state.regime.value if state else 'N/A'}"
        )

        # 驗證所有 regime 類型
        all_regimes = list(MarketRegime)
        result.record(
            "4.4 Regime 枚舉",
            len(all_regimes) == 10,
            f"共 {len(all_regimes)} 種市場狀態"
        )

    except ImportError as e:
        result.add_warning(f"Regime 模組未完全安裝: {e}")
    except Exception as e:
        result.add_error("4.x Regime Detection", e)

    # ========================================
    # 測試 5: Composite Strategy
    # ========================================
    print("\n🔗 測試 5: Composite Strategy")
    print("-" * 40)

    try:
        from src.strategies.composite import CompositeStrategy, SignalAggregation

        # 建立子策略（必須先建立才能傳入 CompositeStrategy）
        rsi_strategy = RSIStrategy(trend_filter=False)
        ma_strategy = MACrossStrategy()

        # 建立組合策略時傳入策略列表（使用等權重，讓系統自動計算）
        composite = CompositeStrategy(
            strategies=[rsi_strategy, ma_strategy],
            aggregation=SignalAggregation.WEIGHTED,
            weighted_threshold=0.5
        )
        # 權重會自動設為等權重 (0.5, 0.5)

        result.record(
            "5.1 組合策略建立",
            len(composite.strategies) == 2,
            f"包含 {len(composite.strategies)} 個子策略"
        )

        # 產生訊號
        signals = composite.generate_signals(sample_data)
        long_entry, long_exit, short_entry, short_exit = signals

        result.record(
            "5.2 訊號聚合",
            len(long_entry) == len(sample_data),
            f"產生 {long_entry.sum()} 個多頭進場訊號"
        )

        # 測試不同聚合模式（每個模式都傳入策略列表）
        aggregation_ok = True
        for mode in SignalAggregation:
            try:
                test_strategies = [RSIStrategy(trend_filter=False), MACrossStrategy()]
                test_composite = CompositeStrategy(
                    strategies=test_strategies,
                    aggregation=mode
                )
                _ = test_composite.generate_signals(sample_data)
            except Exception as e:
                result.add_warning(f"聚合模式 {mode.value} 失敗: {e}")
                aggregation_ok = False

        result.record(
            "5.3 所有聚合模式",
            aggregation_ok,
            f"測試了 {len(SignalAggregation)} 種聚合模式"
        )

        # 使用組合策略回測
        if engine:
            composite_result = engine.run(composite)
            if composite_result:
                result.record(
                    "5.4 組合策略回測",
                    True,
                    f"Sharpe={composite_result.sharpe_ratio:.4f}"
                )
            else:
                result.record("5.4 組合策略回測", False, "回測結果為 None")
        else:
            result.add_warning("引擎未初始化，跳過組合策略回測")

    except Exception as e:
        result.add_error("5.x Composite Strategy", e)

    # ========================================
    # 測試 6: 學習系統
    # ========================================
    print("\n📚 測試 6: 學習系統")
    print("-" * 40)

    try:
        from src.learning.recorder import ExperimentRecorder

        # 使用專案內的測試目錄（避免 _validate_path 拋出「路徑不在專案目錄內」錯誤）
        test_tmp_dir = PROJECT_ROOT / 'tests' / 'tmp'
        test_tmp_dir.mkdir(parents=True, exist_ok=True)

        experiments_file = test_tmp_dir / 'e2e_test_experiments.json'
        insights_file = test_tmp_dir / 'e2e_test_insights.md'

        try:
            recorder = ExperimentRecorder(
                experiments_file=experiments_file,
                insights_file=insights_file
            )

            result.record("6.1 記錄器初始化", True, "ExperimentRecorder 建立成功")

            # 記錄實驗
            if backtest_result:
                exp_id = recorder.log_experiment(
                    result=backtest_result,
                    strategy_info={'name': 'rsi_test', 'type': 'momentum', 'version': '1.0'},
                    config={'symbol': 'BTCUSDT', 'timeframe': '4h', 'capital': 10000}
                )

                result.record(
                    "6.2 實驗記錄",
                    exp_id is not None,
                    f"記錄 ID: {exp_id}"
                )

                # 查詢實驗
                exp = recorder.get_experiment(exp_id)
                result.record(
                    "6.3 實驗查詢",
                    exp is not None and exp.id == exp_id,
                    f"成功查詢 {exp_id}"
                )

                # 查詢最佳實驗
                best_exps = recorder.get_best_experiments('sharpe_ratio', n=5)
                result.record(
                    "6.4 最佳實驗查詢",
                    len(best_exps) >= 1,
                    f"找到 {len(best_exps)} 個實驗"
                )
            else:
                result.add_warning("無回測結果，跳過實驗記錄測試")

        finally:
            # 清理測試檔案
            if experiments_file.exists():
                experiments_file.unlink()
            if insights_file.exists():
                insights_file.unlink()

    except Exception as e:
        result.add_error("6.x 學習系統", e)

    # ========================================
    # 測試 7: 多目標優化（快速檢查）
    # ========================================
    print("\n🎯 測試 7: 多目標優化 (NSGA-II)")
    print("-" * 40)

    try:
        if engine is None:
            result.add_warning("引擎未初始化，跳過多目標優化測試")
        else:
            from src.optimizer.multi_objective import MultiObjectiveOptimizer

            optimizer = MultiObjectiveOptimizer(
                objectives=[
                    ('sharpe_ratio', 'maximize'),
                    ('max_drawdown', 'minimize')
                ],
                n_trials=3,  # 快速測試
                seed=42,
                verbose=False
            )

            result.record("7.1 NSGA-II 初始化", True, "MultiObjectiveOptimizer 建立成功")

            # 定義評估函數
            def evaluate_fn(params: Dict) -> Dict[str, float]:
                strategy = RSIStrategy(**params, trend_filter=False)
                bt_result = engine.run(strategy)
                return {
                    'sharpe_ratio': bt_result.sharpe_ratio if bt_result else 0,
                    'max_drawdown': bt_result.max_drawdown if bt_result else 1
                }

            # 定義參數空間（排除 trend_filter）
            param_space = {
                'rsi_period': {'type': 'int', 'low': 7, 'high': 28},
                'oversold': {'type': 'int', 'low': 20, 'high': 40},
                'overbought': {'type': 'int', 'low': 60, 'high': 80},
            }

            mo_result = optimizer.optimize(
                param_space=param_space,
                evaluate_fn=evaluate_fn
            )

            if mo_result and mo_result.pareto_front:
                result.record(
                    "7.2 Pareto 前沿",
                    len(mo_result.pareto_front) > 0,
                    f"找到 {len(mo_result.pareto_front)} 個 Pareto 最優解"
                )
            else:
                result.record("7.2 Pareto 前沿", False, "無 Pareto 解")

    except ImportError as e:
        result.add_warning(f"多目標優化模組未安裝: {e}")
    except Exception as e:
        result.add_error("7.x 多目標優化", e)

    # ========================================
    # 測試 8: 驗證系統
    # ========================================
    print("\n✅ 測試 8: 回測驗證系統")
    print("-" * 40)

    try:
        from src.backtester.validator import BacktestValidator

        validator = BacktestValidator()
        result.record("8.1 驗證器初始化", True, "BacktestValidator 建立成功")

        # 執行 L2 驗證（數值正確性）
        report = validator.validate_level("L2")

        if report:
            result.record(
                "8.2 L2 數值驗證",
                report.all_passed,
                f"通過: {sum(1 for r in report.results if r.success)}/{len(report.results)}"
            )
        else:
            result.record("8.2 L2 數值驗證", False, "驗證報告為 None")

    except ImportError as e:
        result.add_warning(f"驗證模組未安裝: {e}")
    except Exception as e:
        result.add_error("8.x 驗證系統", e)

    # ========================================
    # 輸出測試摘要
    # ========================================
    print(result.summary())

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="E2E 回測流程測試")
    parser.add_argument("--quick", action="store_true", help="快速模式（減少優化試驗數）")
    args = parser.parse_args()

    result = run_e2e_tests()

    # 設定退出碼
    if result.errors or any(not r["success"] for r in result.results.values()):
        sys.exit(1)
    else:
        print("\n🎉 所有 E2E 測試通過！")
        sys.exit(0)
