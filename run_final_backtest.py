#!/usr/bin/env python3
"""
最終回測執行腳本 - 高效能配置 (M4 Max)

配置：
- 12 核心並行
- 40GB 資料池
- 100 iterations × 50 trials = 5000 total
- 全功能啟用（信號放大、過濾、動態風控、自適應槓桿）
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# 設定專案路徑
sys.path.insert(0, str(Path(__file__).parent))

from src.automation.ultimate_loop import UltimateLoopController
from src.automation.ultimate_config import UltimateLoopConfig


async def run_final_backtest():
    """執行最終回測"""

    print("=" * 70)
    print("🚀 最終回測 - 高效能配置 (M4 Max)")
    print("=" * 70)
    print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 使用高效能配置
    config = UltimateLoopConfig.create_high_performance_config()

    # 顯示配置
    print("📋 配置摘要:")
    print(f"   Workers: {config.max_workers} 核心")
    print(f"   GPU: {'啟用' if config.use_gpu else '停用'}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Data Pool: {config.data_pool_max_gb} GB")
    print(f"   Iterations: {config.n_iterations}")
    print(f"   Trials/Iteration: {config.trials_per_iteration}")
    print(f"   Total Trials: {config.n_iterations * config.trials_per_iteration}")
    print()
    print("📊 交易優化功能:")
    print(f"   信號放大器: {'✓' if config.signal_amplification_enabled else '✗'}")
    print(f"   信號過濾: {'✓' if config.signal_filter_enabled else '✗'}")
    print(f"   動態風控: {'✓' if config.dynamic_risk_enabled else '✗'}")
    print(f"   自適應槓桿: {'✓' if config.adaptive_leverage_enabled else '✗'}")
    print()
    print("🔍 驗證設定:")
    print(f"   Min Stages: {config.min_stages}")
    print(f"   Min Sharpe: {config.min_sharpe}")
    print(f"   Max Overfit: {config.max_overfit}")
    print()
    print("=" * 70)
    print("開始執行...")
    print()

    # 建立控制器
    controller = UltimateLoopController(config, verbose=True)

    try:
        # 執行 loop
        summary = await controller.run_loop(n_iterations=config.n_iterations)

        # 輸出結果
        print()
        print("=" * 70)
        print("📈 執行完成")
        print("=" * 70)
        print(summary.summary_text())

        return summary

    except KeyboardInterrupt:
        print("\n⚠️ 用戶中斷執行")
        return None
    except Exception as e:
        print(f"\n❌ 執行錯誤: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 設定 logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )

    # 執行
    asyncio.run(run_final_backtest())
