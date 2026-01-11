"""
統一記憶體管理器（Apple Silicon 優化）

功能：
1. 預載入多年資料到記憶體（64GB 足夠）
2. 零拷貝資料共享（CPU/GPU 共享記憶體）
3. Memory-mapped files for 超大資料集
4. 記憶體使用監控和優化
"""

import mmap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

import numpy as np
import psutil

if TYPE_CHECKING:
    pass  # 用於類型檢查的額外 imports

# Constants
BYTES_TO_GB = 1024**3
ALIGNMENT_BYTES = 16  # Apple GPU 優化：16-byte 對齊


@dataclass
class MemoryStats:
    """記憶體統計資訊"""

    total_gb: float
    used_gb: float
    available_gb: float
    cached_data_gb: float

    def __str__(self) -> str:
        return (
            f"Memory Stats:\n"
            f"  Total:     {self.total_gb:.2f} GB\n"
            f"  Used:      {self.used_gb:.2f} GB\n"
            f"  Available: {self.available_gb:.2f} GB\n"
            f"  Cached:    {self.cached_data_gb:.2f} GB"
        )


class UnifiedMemoryManager:
    """
    統一記憶體管理器（Apple Silicon 優化）

    Apple Silicon 使用統一記憶體架構，CPU/GPU 共享相同的實體記憶體，
    無需在 CPU 和 GPU 之間複製資料。

    Features:
    - 預載入資料到記憶體
    - 零拷貝資料共享
    - Memory-mapped files 支援
    - 記憶體對齊優化
    """

    def __init__(self, max_cache_gb: float = 48.0):
        """
        Args:
            max_cache_gb: 最大快取大小（GB），預設 48GB（保留 16GB 給系統）
        """
        self.max_cache_gb = max_cache_gb
        self.max_cache_bytes = int(max_cache_gb * BYTES_TO_GB)
        self._cache: Dict[str, np.ndarray] = {}
        self._mmap_files: Dict[str, mmap.mmap] = {}
        self._mmap_arrays: Dict[str, np.ndarray] = {}

    def preload_data(
        self,
        symbols: List[str],
        timeframes: List[str],
        start_date: datetime,
        end_date: datetime,
        data_loader_fn,
    ) -> Dict[str, np.ndarray]:
        """
        預載入資料到記憶體

        Args:
            symbols: 商品符號列表（如 ['BTCUSDT', 'ETHUSDT']）
            timeframes: 時間框架列表（如 ['1m', '5m', '1h']）
            start_date: 開始日期
            end_date: 結束日期
            data_loader_fn: 資料載入函數，簽名為 fn(symbol, timeframe, start, end) -> ndarray

        Returns:
            Dict[cache_key, np.ndarray]
        """
        preloaded = {}

        for symbol in symbols:
            for timeframe in timeframes:
                cache_key = f"{symbol}_{timeframe}_{start_date.date()}_{end_date.date()}"

                # 檢查是否已快取
                if cache_key in self._cache:
                    preloaded[cache_key] = self._cache[cache_key]
                    continue

                # 載入資料
                data = data_loader_fn(symbol, timeframe, start_date, end_date)

                # 優化記憶體布局
                data_optimized = self.optimize_for_gpu(data)

                # 檢查記憶體限制
                if self._get_cached_size() + data_optimized.nbytes > self.max_cache_bytes:
                    # 清除最舊的快取（LRU）
                    self._evict_oldest_cache()

                # 存入快取
                self._cache[cache_key] = data_optimized
                preloaded[cache_key] = data_optimized

        return preloaded

    def get_shared_array(self, key: str) -> Optional[np.ndarray]:
        """
        取得共享陣列（零拷貝）

        Args:
            key: 快取鍵值

        Returns:
            零拷貝的 numpy 陣列，或 None（如果不存在）
        """
        array = self._cache.get(key)
        if array is None:
            array = self._mmap_arrays.get(key)

        return array  # 返回 view，不是 copy

    def create_mmap_array(
        self, file_path: Path, shape: tuple, dtype: np.dtype,
        mode: Literal['r', 'r+', 'w+', 'c'] = "r+"
    ) -> np.ndarray:
        """
        建立 memory-mapped 陣列

        適用於超過可用 RAM 的資料集。

        Args:
            file_path: 檔案路徑
            shape: 陣列形狀
            dtype: 資料型別
            mode: 模式 ('r', 'r+', 'w+', 'c')

        Returns:
            Memory-mapped numpy 陣列
        """
        key = str(file_path)

        # 如果已存在，直接返回
        if key in self._mmap_arrays:
            return self._mmap_arrays[key]

        # 建立 memory-mapped 陣列
        mmap_array = np.memmap(file_path, dtype=dtype, mode=mode, shape=shape)

        # 快取引用
        self._mmap_arrays[key] = mmap_array

        return mmap_array

    def get_stats(self) -> MemoryStats:
        """
        取得記憶體統計

        Returns:
            MemoryStats 物件
        """
        vm = psutil.virtual_memory()

        total_gb = vm.total / BYTES_TO_GB
        used_gb = vm.used / BYTES_TO_GB
        available_gb = vm.available / BYTES_TO_GB
        cached_data_gb = self._get_cached_size() / BYTES_TO_GB

        return MemoryStats(
            total_gb=total_gb,
            used_gb=used_gb,
            available_gb=available_gb,
            cached_data_gb=cached_data_gb,
        )

    def optimize_for_gpu(self, data: np.ndarray) -> np.ndarray:
        """
        優化資料布局以利 GPU 存取

        Apple GPU 偏好：
        - 連續記憶體布局（C-contiguous）
        - 16-byte 對齊

        Args:
            data: 原始 numpy 陣列

        Returns:
            優化後的陣列（可能是 view 或 copy）
        """
        # 確保 C-contiguous（行優先）
        if not data.flags["C_CONTIGUOUS"]:
            data = np.ascontiguousarray(data)

        # 檢查 16-byte 對齊
        if data.ctypes.data % ALIGNMENT_BYTES != 0:
            # 建立對齊的副本
            aligned_data = np.empty_like(data, order="C")
            aligned_data[:] = data
            return aligned_data

        return data

    def clear_cache(self, pattern: Optional[str] = None) -> None:
        """
        清除快取

        Args:
            pattern: 可選的模式匹配（如 'BTCUSDT' 清除所有 BTC 相關資料）
        """
        if pattern is None:
            # 清除所有
            self._cache.clear()
        else:
            # 清除匹配的
            keys_to_remove = [key for key in self._cache if pattern in key]
            for key in keys_to_remove:
                del self._cache[key]

    def verify_zero_copy(self, key: str) -> bool:
        """
        驗證資料是否為零拷貝（shares memory）

        Args:
            key: 快取鍵值

        Returns:
            True if zero-copy
        """
        original = self._cache.get(key)
        if original is None:
            return False

        retrieved = self.get_shared_array(key)
        if retrieved is None:
            return False

        return np.shares_memory(original, retrieved)

    # Private methods

    def _get_cached_size(self) -> int:
        """取得已快取資料的總大小（bytes）"""
        return sum(arr.nbytes for arr in self._cache.values())

    def _evict_oldest_cache(self) -> None:
        """移除最舊的快取項目（簡單 FIFO 策略）"""
        if not self._cache:
            return

        # 取得第一個鍵值（Python 3.7+ dict 保持插入順序）
        oldest_key = next(iter(self._cache))
        del self._cache[oldest_key]


class SharedMemoryPool:
    """
    跨進程共享記憶體池

    用於多進程並行回測時共享資料，避免重複載入。
    """

    def __init__(self, name: str, size_gb: float):
        """
        Args:
            name: 共享記憶體名稱
            size_gb: 記憶體池大小（GB）
        """
        from multiprocessing import shared_memory

        self.name = name
        self.size_bytes = int(size_gb * BYTES_TO_GB)
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._metadata: Dict[str, dict] = {}  # key -> {offset, shape, dtype}

    def create(self) -> None:
        """建立共享記憶體"""
        from multiprocessing import shared_memory

        self._shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size_bytes)

    def attach(self) -> None:
        """附加到現有的共享記憶體"""
        from multiprocessing import shared_memory

        self._shm = shared_memory.SharedMemory(name=self.name)

    def put(self, key: str, data: np.ndarray, offset: int = 0) -> None:
        """
        放入資料（零拷貝）

        Args:
            key: 資料鍵值
            data: numpy 陣列
            offset: 記憶體偏移量（bytes）
        """
        if self._shm is None:
            raise RuntimeError("SharedMemory not initialized. Call create() or attach() first.")

        # 建立共享陣列
        shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=self._shm.buf, offset=offset)

        # 複製資料（這是必要的，因為共享記憶體需要初始化）
        shared_array[:] = data[:]

        # 記錄 metadata
        self._metadata[key] = {
            "offset": offset,
            "shape": data.shape,
            "dtype": data.dtype,
        }

    def get(self, key: str) -> Optional[np.ndarray]:
        """
        取得資料（零拷貝）

        Args:
            key: 資料鍵值

        Returns:
            零拷貝的 numpy 陣列
        """
        if self._shm is None:
            raise RuntimeError("SharedMemory not initialized. Call create() or attach() first.")

        metadata = self._metadata.get(key)
        if metadata is None:
            return None

        # 建立 view（零拷貝）
        shared_array = np.ndarray(
            metadata["shape"], dtype=metadata["dtype"], buffer=self._shm.buf, offset=metadata["offset"]
        )

        return shared_array

    def close(self) -> None:
        """關閉共享記憶體"""
        if self._shm is not None:
            self._shm.close()

    def unlink(self) -> None:
        """解除共享記憶體（刪除）"""
        if self._shm is not None:
            self._shm.unlink()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if exc_type is None:
            self.unlink()
