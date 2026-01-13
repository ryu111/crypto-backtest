#!/usr/bin/env python
"""
一鍵回測腳本 - 開始回測

使用方式：
    python scripts/run_backtest.py                    # 預設配置
    python scripts/run_backtest.py --quick            # 快速測試（5 次迭代）
    python scripts/run_backtest.py --production       # 生產配置（100 次迭代）
    python scripts/run_backtest.py --strategies ma_cross,rsi  # 指定策略
    python scripts/run_backtest.py --symbols BTCUSDT  # 指定標的
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# 添加專案根目錄到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.automation import (
    BacktestLoop,
    BacktestLoopConfig,
    create_default_config,
    create_quick_config,
    create_production_config,
)
from src.strategies import StrategyRegistry


def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description='一鍵回測腳本 - 舊策略優化 + 新策略搜尋',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python scripts/run_backtest.py                     # 預設配置
  python scripts/run_backtest.py --quick             # 快速測試
  python scripts/run_backtest.py --production        # 生產環境
  python scripts/run_backtest.py --iterations 50     # 自訂迭代次數
  python scripts/run_backtest.py --strategies trend_ma_cross,momentum_rsi
        """
    )

    # 模式選擇
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--quick', action='store_true',
        help='快速測試模式（5 次迭代，跳過驗證）'
    )
    mode_group.add_argument(
        '--production', action='store_true',
        help='生產模式（100 次迭代，完整驗證）'
    )

    # 自訂參數
    parser.add_argument(
        '--iterations', '-n', type=int, default=20,
        help='迭代次數（預設 20）'
    )
    parser.add_argument(
        '--strategies', '-s', type=str, default=None,
        help='策略清單，逗號分隔（預設：全部 12 個）'
    )
    parser.add_argument(
        '--symbols', type=str, default='BTCUSDT,ETHUSDT',
        help='交易對，逗號分隔（預設：BTCUSDT,ETHUSDT）'
    )
    parser.add_argument(
        '--timeframes', '-tf', type=str, default='1h,4h',
        help='時間框架，逗號分隔（預設：1h,4h）'
    )
    parser.add_argument(
        '--workers', '-w', type=int, default=8,
        help='並行 worker 數量（預設 8）'
    )
    parser.add_argument(
        '--gpu', action='store_true', default=True,
        help='啟用 GPU 加速（預設啟用）'
    )
    parser.add_argument(
        '--no-gpu', action='store_true',
        help='停用 GPU 加速'
    )
    parser.add_argument(
        '--selection', type=str, default='epsilon_greedy',
        choices=['epsilon_greedy', 'ucb', 'thompson_sampling', 'round_robin'],
        help='策略選擇模式（預設：epsilon_greedy）'
    )
    parser.add_argument(
        '--validation', type=str, default='4,5',
        help='驗證階段，逗號分隔（1-5，預設：4,5 = WFA + MC）'
    )
    parser.add_argument(
        '--min-sharpe', type=float, default=1.0,
        help='最低 Sharpe 閾值（預設 1.0）'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='詳細輸出'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='只顯示配置，不執行'
    )

    return parser.parse_args()


def get_strategies(strategy_arg: Optional[str]) -> List[str]:
    """獲取策略清單"""
    if strategy_arg:
        return [s.strip() for s in strategy_arg.split(',')]

    # 預設使用所有已註冊策略
    registry = StrategyRegistry()
    return list(registry._strategies.keys())


def create_config(args) -> BacktestLoopConfig:
    """根據參數建立配置"""

    # 模式選擇
    if args.quick:
        config = create_quick_config()
        print("📋 使用快速測試配置")
    elif args.production:
        config = create_production_config()
        print("📋 使用生產配置")
    else:
        config = create_default_config()
        print("📋 使用預設配置")

    # 覆蓋自訂參數
    strategies = get_strategies(args.strategies)
    symbols = [s.strip() for s in args.symbols.split(',')]
    timeframes = [tf.strip() for tf in args.timeframes.split(',')]
    validation_stages = [int(v.strip()) for v in args.validation.split(',')]

    # 建立最終配置
    config = BacktestLoopConfig(
        strategies=strategies,
        symbols=symbols,
        timeframes=timeframes,
        n_iterations=args.iterations if not args.quick and not args.production else config.n_iterations,
        selection_mode=args.selection,
        validation_stages=validation_stages,
        min_sharpe=args.min_sharpe,
        max_workers=args.workers,
        use_gpu=args.gpu and not args.no_gpu,
    )

    return config


def print_config(config: BacktestLoopConfig):
    """印出配置摘要"""
    print("\n" + "=" * 60)
    print("📊 回測配置")
    print("=" * 60)
    print(f"策略數量: {len(config.strategies)}")
    for s in config.strategies[:5]:
        print(f"  - {s}")
    if len(config.strategies) > 5:
        print(f"  ... 還有 {len(config.strategies) - 5} 個")
    print(f"交易對: {', '.join(config.symbols)}")
    print(f"時間框架: {', '.join(config.timeframes)}")
    print(f"迭代次數: {config.n_iterations}")
    print(f"選擇模式: {config.selection_mode}")
    print(f"驗證階段: {config.validation_stages}")
    print(f"最低 Sharpe: {config.min_sharpe}")
    print(f"Workers: {config.max_workers}")
    print(f"GPU: {'✅' if config.use_gpu else '❌'}")
    print("=" * 60 + "\n")


def print_progress(iteration: int, total: int, summary: dict):
    """印出進度"""
    pct = iteration / total * 100
    bar_len = 30
    filled = int(bar_len * iteration / total)
    bar = "█" * filled + "░" * (bar_len - filled)

    print(f"\r[{bar}] {pct:.1f}% ({iteration}/{total})", end="")

    if summary:
        sharpe = getattr(summary, 'sharpe_ratio', 0)
        grade = getattr(summary, 'validation_grade', '-')
        print(f" | Sharpe: {sharpe:.2f} | Grade: {grade}", end="")

    if iteration == total:
        print()  # 換行


def run_backtest(config: BacktestLoopConfig, verbose: bool = False):
    """執行回測"""
    print("\n🚀 開始回測...")
    print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    start_time = time.time()

    try:
        with BacktestLoop(config) as loop:
            # 設定進度回調
            if verbose:
                loop.on_iteration_end = lambda i, total, s: print_progress(i, total, s)

            result = loop.run()

    except KeyboardInterrupt:
        print("\n\n⚠️ 使用者中斷")
        return None
    except Exception as e:
        print(f"\n\n❌ 錯誤: {e}")
        raise

    elapsed = time.time() - start_time

    # 印出結果
    print("\n" + "=" * 60)
    print("📈 回測結果")
    print("=" * 60)
    print(result.summary())
    print(f"\n⏱️ 總耗時: {elapsed:.1f} 秒")
    print(f"完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    return result


def main():
    """主程式"""
    print("\n" + "=" * 60)
    print("🎯 合約交易回測系統")
    print("   舊策略優化 + 新策略搜尋")
    print("=" * 60)

    args = parse_args()
    config = create_config(args)
    print_config(config)

    if args.dry_run:
        print("🔍 Dry run 模式，不執行回測")
        return

    # 驗證配置
    errors = config.validate()
    if errors:
        print("❌ 配置驗證失敗:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    result = run_backtest(config, args.verbose)

    if result:
        # 記錄成功的迭代數
        successful = len(result.best_strategies)
        total = result.iterations_completed
        print(f"\n✅ 回測完成！成功: {successful}/{total}")

        # 最佳策略
        if result.best_strategies:
            best = result.best_strategies[0]
            print(f"\n🏆 最佳策略:")
            print(f"   策略: {best.strategy_name}")
            print(f"   標的: {best.symbol} {best.timeframe}")
            print(f"   Sharpe: {best.sharpe_ratio:.2f}")
            print(f"   報酬: {best.total_return:.1f}%")
            print(f"   評級: {best.validation_grade}")


if __name__ == "__main__":
    main()
