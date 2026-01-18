#!/usr/bin/env python3
"""
執行完整回測 (UltimateLoop)

使用方式：
    python run_ultimate_loop.py 1000
    python run_ultimate_loop.py 100 --verbose
    python run_ultimate_loop.py 50 --resume
    python run_ultimate_loop.py 100 --monitor  # 啟動 Web Dashboard
"""

import asyncio
import sys
import argparse
import logging
from datetime import datetime

# 自動安裝進程守護
import src  # noqa: F401

from src.automation.ultimate_loop import UltimateLoopController
from src.automation.ultimate_config import UltimateLoopConfig


def setup_logging(verbose: bool = False):
    """設定日誌"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f'logs/ultimate_loop_{datetime.now():%Y%m%d_%H%M%S}.log',
                mode='w'
            )
        ]
    )


async def main():
    parser = argparse.ArgumentParser(description='執行完整回測 (UltimateLoop)')
    parser.add_argument('iterations', type=int, help='迭代次數')
    parser.add_argument('--verbose', '-v', action='store_true', help='詳細輸出')
    parser.add_argument('--resume', '-r', action='store_true', help='從檢查點恢復')
    parser.add_argument('--config', '-c', choices=['quick', 'dev', 'production', 'performance'],
                        default='performance', help='配置模式（預設 12 核心）')
    parser.add_argument('--monitor', '-m', action='store_true',
                        help='啟動 Web Dashboard 監控（http://localhost:8765）')
    parser.add_argument('--port', '-p', type=int, default=8765,
                        help='監控服務埠號（預設 8765）')

    args = parser.parse_args()

    # 確保 logs 目錄存在
    from pathlib import Path
    Path('logs').mkdir(exist_ok=True)

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # 建立配置
    if args.config == 'quick':
        config = UltimateLoopConfig.create_quick_test_config()
    elif args.config == 'production':
        config = UltimateLoopConfig.create_production_config()
    elif args.config == 'performance':
        config = UltimateLoopConfig.create_high_performance_config()
    else:
        config = UltimateLoopConfig.create_development_config()

    logger.info(f"=" * 60)
    logger.info(f"開始完整回測: {args.iterations} 次迭代")
    logger.info(f"配置模式: {args.config}")
    if args.monitor:
        logger.info(f"監控 Dashboard: http://localhost:{args.port}")
    logger.info(f"=" * 60)

    # 初始化監控服務（如果啟用）
    monitor = None
    monitor_task = None

    if args.monitor:
        try:
            from src.monitoring.monitor_service import MonitorService
            monitor = MonitorService(port=args.port)
            # 在背景啟動監控服務
            import webbrowser
            monitor_task = asyncio.create_task(monitor.start())
            # 等待服務啟動
            await asyncio.sleep(1)
            # 自動開啟瀏覽器
            webbrowser.open(f"http://localhost:{args.port}")
            logger.info(f"監控 Dashboard 已啟動: http://localhost:{args.port}")
        except ImportError as e:
            logger.warning(f"無法啟動監控服務: {e}")
            monitor = None

    # 建立控制器
    controller = UltimateLoopController(config, verbose=args.verbose, monitor=monitor)

    try:
        # 執行回測
        summary = await controller.run_loop(
            n_iterations=args.iterations,
            resume_from_checkpoint=args.resume
        )

        # 輸出結果
        print("\n" + "=" * 60)
        print("回測完成！")
        print("=" * 60)
        print(summary.summary_text())

        # 保存結果
        result_file = f'logs/ultimate_loop_result_{datetime.now():%Y%m%d_%H%M%S}.json'
        with open(result_file, 'w') as f:
            import json
            json.dump(summary.to_dict(), f, indent=2, default=str)
        print(f"\n結果已保存到: {result_file}")

        # 如果有監控，等待用戶按鍵後退出
        if args.monitor and monitor_task:
            print("\n監控 Dashboard 仍在運行中...")
            print("按 Ctrl+C 關閉監控服務")
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

        return 0

    except KeyboardInterrupt:
        logger.info("\n使用者中斷執行")
        return 1
    except Exception as e:
        logger.error(f"執行失敗: {e}", exc_info=True)
        return 2
    finally:
        # 清理監控服務
        if monitor_task and not monitor_task.done():
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
