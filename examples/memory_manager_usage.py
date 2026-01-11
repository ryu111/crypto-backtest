"""
記憶體管理器使用範例

展示如何在回測系統中使用記憶體管理器來優化效能。
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from src.data.memory_manager import SharedMemoryPool, UnifiedMemoryManager


# 範例 1: 基本使用 - 預載入資料
def example_basic_usage():
    """基本使用：預載入多年資料"""
    print("=== 範例 1: 基本使用 ===\n")

    # 建立記憶體管理器（使用 48GB 快取）
    manager = UnifiedMemoryManager(max_cache_gb=48.0)

    # 模擬資料載入函數
    def load_ohlcv_data(symbol, timeframe, start_date, end_date):
        """載入 OHLCV 資料（這裡用隨機資料模擬）"""
        days = (end_date - start_date).days
        rows = days * 1440  # 假設 1 分鐘資料
        # 實際應該從 database 或 CSV 讀取
        return np.random.rand(rows, 6)  # [timestamp, open, high, low, close, volume]

    # 預載入 2 年資料
    symbols = ["BTCUSDT", "ETHUSDT"]
    timeframes = ["1m", "5m", "15m"]
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 1, 1)

    print(f"預載入資料: {symbols} x {timeframes}")
    print(f"時間範圍: {start_date.date()} ~ {end_date.date()}\n")

    preloaded = manager.preload_data(
        symbols=symbols,
        timeframes=timeframes,
        start_date=start_date,
        end_date=end_date,
        data_loader_fn=load_ohlcv_data,
    )

    # 顯示已載入的資料
    total_size_gb = sum(arr.nbytes for arr in preloaded.values()) / (1024**3)
    print(f"已載入 {len(preloaded)} 組資料")
    print(f"總大小: {total_size_gb:.2f} GB\n")

    # 取得記憶體統計
    stats = manager.get_stats()
    print(stats)


# 範例 2: 零拷貝資料共享
def example_zero_copy():
    """零拷貝資料共享（避免重複記憶體使用）"""
    print("\n=== 範例 2: 零拷貝資料共享 ===\n")

    manager = UnifiedMemoryManager(max_cache_gb=10.0)

    # 模擬載入大型資料
    large_data = np.random.rand(1000000, 6)  # 48 MB
    print(f"原始資料大小: {large_data.nbytes / 1024**2:.2f} MB")

    # 優化並快取
    optimized = manager.optimize_for_gpu(large_data)
    manager._cache["BTCUSDT_1m_2024"] = optimized

    # 取得共享陣列（零拷貝）
    shared_array = manager.get_shared_array("BTCUSDT_1m_2024")

    # 驗證零拷貝
    is_zero_copy = manager.verify_zero_copy("BTCUSDT_1m_2024")
    print(f"零拷貝驗證: {'✓ 成功' if is_zero_copy else '✗ 失敗'}")

    # 驗證記憶體共享
    shares_memory = np.shares_memory(optimized, shared_array)
    print(f"記憶體共享: {'✓ 是' if shares_memory else '✗ 否'}")

    # 修改 shared_array 會影響 optimized（證明是同一塊記憶體）
    original_value = shared_array[0, 0]
    shared_array[0, 0] = 999.0
    print(f"\n修改 shared_array[0,0] = 999.0")
    print(f"optimized[0,0] = {optimized[0, 0]} (應該也是 999.0)")
    shared_array[0, 0] = original_value  # 恢復


# 範例 3: Memory-mapped files（超大資料集）
def example_memory_mapped():
    """使用 memory-mapped files 處理超過 RAM 的資料"""
    print("\n=== 範例 3: Memory-mapped Files ===\n")

    manager = UnifiedMemoryManager()

    # 建立臨時檔案
    mmap_file = Path("/tmp/large_dataset.npy")

    # 建立超大資料集（只佔用少量 RAM）
    shape = (10000000, 6)  # 480 MB
    dtype = np.float64

    print(f"建立 memory-mapped 陣列: {shape}")
    print(f"理論大小: {np.prod(shape) * dtype(0).nbytes / 1024**2:.2f} MB")

    # 建立 memory-mapped 陣列
    mmap_array = manager.create_mmap_array(mmap_file, shape, dtype, mode="w+")

    # 寫入資料（分批寫入，避免一次性載入全部）
    batch_size = 100000
    for i in range(0, shape[0], batch_size):
        end = min(i + batch_size, shape[0])
        mmap_array[i:end] = np.random.rand(end - i, shape[1])

    print(f"✓ 資料寫入完成")
    print(f"實際 RAM 使用: 極少（資料儲存在硬碟）\n")

    # 讀取部分資料
    sample = mmap_array[0:1000]  # 只載入前 1000 行到 RAM
    print(f"讀取樣本: {sample.shape}")

    # 清理
    mmap_file.unlink()


# 範例 4: 跨進程共享記憶體（多進程回測）
def example_shared_memory_pool():
    """跨進程共享記憶體池（用於並行回測）"""
    print("\n=== 範例 4: 跨進程共享記憶體 ===\n")

    # 建立共享記憶體池
    pool = SharedMemoryPool(name="backtest_data", size_gb=1.0)

    try:
        pool.create()
        print("✓ 共享記憶體池已建立\n")

        # 主進程：載入資料到共享記憶體
        btc_data = np.random.rand(100000, 6)
        eth_data = np.random.rand(100000, 6)

        pool.put("BTCUSDT", btc_data, offset=0)
        pool.put("ETHUSDT", eth_data, offset=btc_data.nbytes)

        print(f"已放入資料:")
        print(f"  BTCUSDT: {btc_data.nbytes / 1024**2:.2f} MB")
        print(f"  ETHUSDT: {eth_data.nbytes / 1024**2:.2f} MB\n")

        # 子進程可以附加到相同的共享記憶體
        # 這裡模擬取得資料（實際使用時在子進程執行）
        retrieved_btc = pool.get("BTCUSDT")
        retrieved_eth = pool.get("ETHUSDT")

        # 驗證
        assert np.array_equal(retrieved_btc, btc_data)
        assert np.array_equal(retrieved_eth, eth_data)

        print("✓ 資料驗證成功")
        print("子進程可以零拷貝存取相同資料")

    finally:
        pool.close()
        pool.unlink()


# 範例 5: 效能比較（傳統 vs 零拷貝）
def example_performance_comparison():
    """效能比較：傳統複製 vs 零拷貝"""
    import time

    print("\n=== 範例 5: 效能比較 ===\n")

    manager = UnifiedMemoryManager()
    large_data = np.random.rand(5000000, 6)  # 240 MB

    # 優化並快取
    optimized = manager.optimize_for_gpu(large_data)
    manager._cache["test_data"] = optimized

    # 測試 1: 傳統複製
    iterations = 100
    start = time.time()
    for _ in range(iterations):
        copy = large_data.copy()  # 完整複製
    copy_time = time.time() - start

    # 測試 2: 零拷貝
    start = time.time()
    for _ in range(iterations):
        shared = manager.get_shared_array("test_data")  # 零拷貝
    zero_copy_time = time.time() - start

    print(f"資料大小: {large_data.nbytes / 1024**2:.2f} MB")
    print(f"迭代次數: {iterations}\n")

    print(f"傳統複製: {copy_time:.3f} 秒")
    print(f"零拷貝:   {zero_copy_time:.6f} 秒")
    print(f"\n加速倍數: {copy_time / zero_copy_time:.0f}x 🚀")


# 主程式
if __name__ == "__main__":
    # 執行所有範例
    example_basic_usage()
    example_zero_copy()
    example_memory_mapped()
    example_shared_memory_pool()
    example_performance_comparison()

    print("\n=== 所有範例完成 ===")
