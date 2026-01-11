#!/usr/bin/env python3
"""
AI Loop 啟動腳本

執行持續的策略優化循環，支援多種模式和命令行參數。

使用範例:
    # 執行 100 次迭代
    python scripts/run_loop.py --mode n_iterations --target 100

    # 持續執行直到 Sharpe >= 3.0
    python scripts/run_loop.py --mode until_target --target 3.0

    # 執行 2 小時
    python scripts/run_loop.py --mode time_based --time 120

    # 從上次中斷處恢復
    python scripts/run_loop.py --resume

    # 清除狀態並重新開始
    python scripts/run_loop.py --clear
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.automation.loop import (
    LoopController,
    LoopMode,
    IterationResult,
    IterationStatus
)
from src.backtester.engine import BacktestEngine, BacktestConfig
from src.optimizer.bayesian import BayesianOptimizer
from src.learning import ExperimentRecorder
from src.strategies.trend.ma_cross import MovingAverageCross
from src.data.fetcher import DataFetcher

import pandas as pd
import numpy as np


def create_iteration_callback(
    strategy_class,
    data: pd.DataFrame,
    config: BacktestConfig,
    n_trials: int = 50
):
    """
    建立迭代回調函數

    Args:
        strategy_class: 策略類別
        data: 市場資料
        config: 回測配置
        n_trials: 每次迭代的優化試驗次數

    Returns:
        迭代回調函數
    """
    # 建立引擎和記錄器
    engine = BacktestEngine(config)
    recorder = ExperimentRecorder()

    def iteration_callback() -> IterationResult:
        """單次迭代執行"""
        print(f"\n執行策略優化（{n_trials} trials）...")

        # 建立策略實例
        strategy = strategy_class()

        # 執行優化
        optimizer = BayesianOptimizer(
            engine=engine,
            n_trials=n_trials,
            n_jobs=1,
            verbose=False
        )

        try:
            opt_result = optimizer.optimize(
                strategy=strategy,
                data=data,
                metric='sharpe_ratio',
                show_progress_bar=False
            )

            # 記錄實驗
            strategy_info = {
                'name': strategy.name,
                'type': 'trend',
                'version': '1.0'
            }

            config_dict = {
                'symbol': config.symbol,
                'timeframe': config.timeframe,
                'initial_capital': config.initial_capital,
                'leverage': config.leverage
            }

            exp_id = recorder.log_experiment(
                result=opt_result.best_backtest_result,
                strategy_info=strategy_info,
                config=config_dict
            )

            # 建立迭代結果
            result = IterationResult(
                iteration=0,  # 會被 controller 覆蓋
                timestamp=datetime.now(),
                status=IterationStatus.SUCCESS,
                sharpe_ratio=opt_result.best_backtest_result.sharpe_ratio,
                total_return=opt_result.best_backtest_result.total_return,
                max_drawdown=opt_result.best_backtest_result.max_drawdown,
                strategy_name=strategy.name,
                best_params=opt_result.best_params,
                experiment_id=exp_id
            )

            return result

        except Exception as e:
            print(f"優化失敗: {e}")
            raise

    return iteration_callback


def create_callbacks():
    """建立回調函數"""

    def on_iteration_start(iteration_num):
        """迭代開始"""
        print(f"\n⏳ 迭代 #{iteration_num} 開始...")

    def on_success(result: IterationResult):
        """迭代成功"""
        print(f"✅ 迭代成功")

    def on_failure(error: Exception):
        """迭代失敗"""
        print(f"❌ 迭代失敗: {error}")

    def on_new_best(result: IterationResult):
        """發現新的最佳結果"""
        print(f"🏆 發現更佳策略！")
        print(f"   Sharpe: {result.sharpe_ratio:.4f}")
        print(f"   參數: {result.best_params}")

    def on_loop_end(state):
        """Loop 結束"""
        print("\n🏁 Loop 已結束")
        print(f"總迭代: {state.completed_iterations}")
        print(f"最佳 Sharpe: {state.best_sharpe:.4f}")

    return {
        'on_iteration_start': on_iteration_start,
        'on_success': on_success,
        'on_failure': on_failure,
        'on_new_best': on_new_best,
        'on_loop_end': on_loop_end
    }


def load_market_data(symbol: str, timeframe: str, days: int = 365) -> pd.DataFrame:
    """
    載入市場資料

    Args:
        symbol: 交易標的
        timeframe: 時間框架
        days: 回溯天數

    Returns:
        OHLCV DataFrame
    """
    print(f"載入 {symbol} {timeframe} 資料（最近 {days} 天）...")

    # 這裡應該整合真實的資料來源
    # 目前使用模擬資料
    end_date = datetime.now()
    start_date = end_date - pd.Timedelta(days=days)

    # 模擬資料（實際使用時替換為 DataFetcher）
    dates = pd.date_range(start=start_date, end=end_date, freq='1h')
    np.random.seed(42)

    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)

    data['high'] = data['open'] + np.abs(np.random.randn(len(dates)) * 0.3)
    data['low'] = data['open'] - np.abs(np.random.randn(len(dates)) * 0.3)
    data['close'] = data['open'] + np.random.randn(len(dates)) * 0.2

    print(f"✓ 資料已載入: {len(data)} 筆")
    return data


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='AI Loop 啟動腳本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Loop 模式
    parser.add_argument(
        '--mode',
        type=str,
        choices=['continuous', 'n_iterations', 'time_based', 'until_target'],
        default='continuous',
        help='執行模式（預設: continuous）'
    )

    parser.add_argument(
        '--target',
        type=float,
        help='目標值（n_iterations: 次數, until_target: Sharpe）'
    )

    parser.add_argument(
        '--time',
        type=int,
        help='時間限制（分鐘，time_based 模式）'
    )

    # 狀態管理
    parser.add_argument(
        '--resume',
        action='store_true',
        help='從上次中斷處恢復'
    )

    parser.add_argument(
        '--clear',
        action='store_true',
        help='清除狀態檔案並重新開始'
    )

    # 策略與資料
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTCUSDT',
        help='交易標的（預設: BTCUSDT）'
    )

    parser.add_argument(
        '--timeframe',
        type=str,
        default='1h',
        help='時間框架（預設: 1h）'
    )

    parser.add_argument(
        '--leverage',
        type=int,
        default=5,
        help='槓桿倍數（預設: 5）'
    )

    parser.add_argument(
        '--trials',
        type=int,
        default=50,
        help='每次迭代的優化試驗次數（預設: 50）'
    )

    args = parser.parse_args()

    # 顯示配置
    print("="*60)
    print("AI Loop 啟動腳本")
    print("="*60)
    print(f"模式: {args.mode}")
    if args.target:
        print(f"目標: {args.target}")
    if args.time:
        print(f"時間: {args.time} 分鐘")
    print(f"標的: {args.symbol}")
    print(f"時間框架: {args.timeframe}")
    print(f"槓桿: {args.leverage}x")
    print(f"每次優化試驗: {args.trials}")
    print("="*60)

    # 建立回測配置
    config = BacktestConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=datetime.now() - pd.Timedelta(days=365),
        end_date=datetime.now(),
        initial_capital=10000,
        leverage=args.leverage
    )

    # 載入市場資料
    data = load_market_data(args.symbol, args.timeframe)

    # 建立迭代回調
    iteration_callback = create_iteration_callback(
        strategy_class=MovingAverageCross,
        data=data,
        config=config,
        n_trials=args.trials
    )

    # 建立控制器
    controller = LoopController(
        iteration_callback=iteration_callback,
        auto_save=True,
        callbacks=create_callbacks()
    )

    # 清除狀態（如果指定）
    if args.clear:
        controller.clear_state()
        print("狀態已清除")
        return

    # 啟動 Loop
    mode = LoopMode[args.mode.upper()]

    try:
        controller.start(
            mode=mode,
            target=int(args.target) if args.target and mode == LoopMode.N_ITERATIONS else args.target,
            time_limit_minutes=args.time,
            resume=args.resume
        )
    except KeyboardInterrupt:
        print("\n收到中斷信號...")
        controller.stop()
    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()
        controller.stop()


if __name__ == '__main__':
    main()
