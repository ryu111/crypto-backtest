"""
SharedDataPool 使用範例

展示如何在多進程回測中使用共享資料池實現零拷貝資料共享。
"""

import sys
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import multiprocessing as mp
from src.data import create_shared_pool, attach_to_pool


def worker_process(pool_name: str, worker_id: int):
    """Worker 進程函數（模擬回測 Worker）

    Args:
        pool_name: 共享資料池名稱
        worker_id: Worker ID
    """
    # 附加到主進程建立的共享記憶體
    pool = attach_to_pool(pool_name)

    print(f"Worker {worker_id} 已附加到共享資料池")
    print(f"可用資料鍵: {pool.list_keys()}")

    # 零拷貝存取資料
    btc_1h = pool.get("BTCUSDT_1h")
    eth_1h = pool.get("ETHUSDT_1h")

    print(f"Worker {worker_id} - BTC 1h shape: {btc_1h.shape}")
    print(f"Worker {worker_id} - ETH 1h shape: {eth_1h.shape}")

    # 模擬計算（存取共享資料）
    btc_mean = btc_1h[:, 3].mean()  # close 價格平均
    eth_mean = eth_1h[:, 3].mean()

    print(f"Worker {worker_id} - BTC 平均價格: {btc_mean:.2f}")
    print(f"Worker {worker_id} - ETH 平均價格: {eth_mean:.2f}")

    # 清理（只 close，不 unlink）
    pool.cleanup()

    return worker_id


def main():
    """主進程：建立共享資料池並啟動 Workers"""

    print("=" * 60)
    print("SharedDataPool 多進程使用範例")
    print("=" * 60)

    # 1. 建立並預載共享資料池
    print("\n[主進程] 建立共享資料池...")
    pool = create_shared_pool(
        data_dir="data",
        symbols=["BTCUSDT", "ETHUSDT"],
        timeframes=["1h"],
        pool_name="example_pool",
        include_funding=False
    )

    print(f"[主進程] 共享資料池已建立: {pool}")
    print(f"[主進程] 總大小: {pool.get_total_size_mb():.2f}MB")

    # 2. 啟動多個 Worker 進程
    print("\n[主進程] 啟動 4 個 Worker 進程...")

    with mp.Pool(processes=4) as worker_pool:
        # 每個 Worker 都會附加到同一個共享資料池
        results = worker_pool.starmap(
            worker_process,
            [("example_pool", i) for i in range(4)]
        )

    print(f"\n[主進程] 所有 Workers 完成: {results}")

    # 3. 清理共享資料池
    print("\n[主進程] 清理共享資料池...")
    pool.cleanup()

    print("\n" + "=" * 60)
    print("範例完成！")
    print("=" * 60)


if __name__ == "__main__":
    # 設定 multiprocessing 啟動方法
    mp.set_start_method('spawn', force=True)

    main()
