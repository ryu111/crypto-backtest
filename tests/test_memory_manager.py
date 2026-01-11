"""
測試記憶體管理器

驗證：
1. 資料預載入
2. 零拷貝驗證
3. Memory-mapped files
4. 記憶體統計
5. 載入時間效能
"""

import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from src.data.memory_manager import SharedMemoryPool, UnifiedMemoryManager


@pytest.fixture
def memory_manager():
    """建立記憶體管理器"""
    return UnifiedMemoryManager(max_cache_gb=1.0)  # 測試用小容量


@pytest.fixture
def sample_data():
    """建立測試資料（10MB）"""
    # 模擬 OHLCV 資料：1 年 * 365 天 * 1440 分鐘/天 * 6 欄位
    rows = 365 * 1440
    return np.random.rand(rows, 6).astype(np.float64)


def dummy_data_loader(symbol: str, timeframe: str, start_date: datetime, end_date: datetime):
    """模擬資料載入函數"""
    days = (end_date - start_date).days
    rows = days * 1440  # 1 分鐘資料
    return np.random.rand(rows, 6).astype(np.float64)


class TestUnifiedMemoryManager:
    """測試統一記憶體管理器"""

    def test_preload_data(self, memory_manager):
        """測試預載入資料"""
        symbols = ["BTCUSDT", "ETHUSDT"]
        timeframes = ["1m"]
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 7)  # 7 天

        preloaded = memory_manager.preload_data(
            symbols=symbols,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            data_loader_fn=dummy_data_loader,
        )

        # 驗證資料已載入
        assert len(preloaded) == 2  # 2 個商品 * 1 個時間框架
        for key, data in preloaded.items():
            assert isinstance(data, np.ndarray)
            assert data.shape[1] == 6  # OHLCV + timestamp

    def test_zero_copy_sharing(self, memory_manager, sample_data):
        """測試零拷貝資料共享"""
        key = "test_data"

        # 優化並快取
        optimized = memory_manager.optimize_for_gpu(sample_data)
        memory_manager._cache[key] = optimized

        # 取得共享陣列
        shared = memory_manager.get_shared_array(key)

        # 驗證零拷貝
        assert shared is not None
        assert memory_manager.verify_zero_copy(key)
        assert np.shares_memory(optimized, shared)

        # 驗證修改會同步
        original_value = shared[0, 0]
        shared[0, 0] = 999.0
        assert optimized[0, 0] == 999.0
        shared[0, 0] = original_value  # 恢復

    def test_optimize_for_gpu(self, memory_manager):
        """測試 GPU 記憶體對齊優化"""
        # 建立非連續陣列
        non_contiguous = np.random.rand(1000, 6).T  # 轉置變成非連續

        assert not non_contiguous.flags["C_CONTIGUOUS"]

        # 優化
        optimized = memory_manager.optimize_for_gpu(non_contiguous)

        # 驗證
        assert optimized.flags["C_CONTIGUOUS"]
        assert optimized.ctypes.data % 16 == 0  # 16-byte 對齊

    def test_memory_stats(self, memory_manager, sample_data):
        """測試記憶體統計"""
        # 快取一些資料
        memory_manager._cache["data1"] = sample_data
        memory_manager._cache["data2"] = sample_data.copy()

        stats = memory_manager.get_stats()

        assert stats.total_gb > 0
        assert stats.available_gb > 0
        assert stats.cached_data_gb > 0
        assert stats.cached_data_gb < 1.0  # 應該遠小於 1GB

    def test_cache_eviction(self, memory_manager):
        """測試快取淘汰（當超過容量限制）"""
        # 建立大量資料直到超過限制
        data = np.random.rand(10000, 1000).astype(np.float64)  # ~80MB

        # 手動測試淘汰邏輯
        for i in range(15):  # 總共 ~1.2GB
            memory_manager._cache[f"data_{i}"] = data.copy()

            # 當超過限制時，手動觸發淘汰
            while memory_manager._get_cached_size() > memory_manager.max_cache_bytes:
                memory_manager._evict_oldest_cache()

        # 驗證快取大小小於限制
        cached_size_gb = memory_manager._get_cached_size() / (1024**3)
        assert cached_size_gb <= memory_manager.max_cache_gb

    def test_clear_cache(self, memory_manager, sample_data):
        """測試清除快取"""
        memory_manager._cache["BTCUSDT_1m"] = sample_data
        memory_manager._cache["ETHUSDT_1m"] = sample_data
        memory_manager._cache["BTCUSDT_5m"] = sample_data

        # 清除特定模式
        memory_manager.clear_cache(pattern="BTCUSDT")

        assert "BTCUSDT_1m" not in memory_manager._cache
        assert "BTCUSDT_5m" not in memory_manager._cache
        assert "ETHUSDT_1m" in memory_manager._cache

        # 清除全部
        memory_manager.clear_cache()
        assert len(memory_manager._cache) == 0


class TestMemoryMappedFiles:
    """測試 Memory-mapped files"""

    def test_create_mmap_array(self, memory_manager):
        """測試建立 memory-mapped 陣列"""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.npy"

            # 建立 memory-mapped 陣列
            shape = (1000, 6)
            dtype = np.float64
            mmap_array = memory_manager.create_mmap_array(file_path, shape, dtype, mode="w+")

            # 寫入資料
            mmap_array[:] = np.random.rand(*shape)

            # 驗證
            assert mmap_array.shape == shape
            assert mmap_array.dtype == dtype

            # 再次取得（應該從快取返回）
            mmap_array2 = memory_manager.create_mmap_array(file_path, shape, dtype, mode="r+")
            assert np.shares_memory(mmap_array, mmap_array2)


class TestSharedMemoryPool:
    """測試跨進程共享記憶體池"""

    def test_create_and_attach(self):
        """測試建立和附加共享記憶體"""
        pool = SharedMemoryPool(name="test_pool", size_gb=0.1)

        try:
            pool.create()

            # 放入資料
            data = np.random.rand(1000, 6)
            pool.put("test_data", data, offset=0)

            # 取得資料
            retrieved = pool.get("test_data")
            assert retrieved is not None
            assert np.array_equal(retrieved, data)

        finally:
            pool.close()
            pool.unlink()

    def test_context_manager(self):
        """測試 context manager 用法"""
        data = np.random.rand(500, 6)

        with SharedMemoryPool(name="test_ctx_pool", size_gb=0.05) as pool:
            pool.create()
            pool.put("data", data)
            retrieved = pool.get("data")
            assert np.array_equal(retrieved, data)

        # context manager 應該自動 close 和 unlink


class TestPerformance:
    """效能測試"""

    def test_preload_performance(self, memory_manager):
        """測試預載入效能（目標：10GB < 2 秒）"""
        # 模擬 10GB 資料（簡化版：100MB）
        symbols = ["BTCUSDT", "ETHUSDT"]
        timeframes = ["1m"]
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)  # 1 年

        start_time = time.time()

        preloaded = memory_manager.preload_data(
            symbols=symbols,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            data_loader_fn=dummy_data_loader,
        )

        elapsed = time.time() - start_time

        # 驗證
        assert len(preloaded) == 2
        total_size_mb = sum(arr.nbytes for arr in preloaded.values()) / (1024**2)

        print(f"\nPreloaded {total_size_mb:.2f} MB in {elapsed:.3f} seconds")
        print(f"Throughput: {total_size_mb / elapsed:.2f} MB/s")

        # 假設硬碟讀取速度 500 MB/s，預載入應該在合理時間內完成
        assert elapsed < 10  # 100MB 應該在 10 秒內完成

    def test_zero_copy_overhead(self, memory_manager, sample_data):
        """測試零拷貝的額外開銷（應該接近零）"""
        key = "perf_test"
        memory_manager._cache[key] = sample_data

        # 測試取得時間
        iterations = 10000
        start_time = time.time()

        for _ in range(iterations):
            _ = memory_manager.get_shared_array(key)

        elapsed = time.time() - start_time
        avg_time_us = (elapsed / iterations) * 1e6

        print(f"\nAverage get_shared_array time: {avg_time_us:.2f} μs")

        # 應該非常快（< 1 μs）
        assert avg_time_us < 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
