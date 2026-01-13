"""
共享資料池模組（跨進程零拷貝資料共享）

使用 multiprocessing.shared_memory 實現多進程間的高效資料共享。
所有 Worker 進程可以直接存取預載的資料，無需複製。

特性：
- 零拷貝資料共享（Zero-copy）
- 自動資料格式轉換（Parquet → float32 NumPy）
- 進程安全的並發讀取
- 自動資源管理（Context Manager）
- 詳細的錯誤處理與日誌
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import json

from multiprocessing import shared_memory
import numpy as np
import pandas as pd

# 設定 logger
logger = logging.getLogger(__name__)

# Registry 檔案位置（用於跨進程共享 metadata）
REGISTRY_DIR = Path("/tmp/shared_data_pool")
REGISTRY_DIR.mkdir(exist_ok=True)


@dataclass
class SharedDataInfo:
    """共享資料的元資訊"""
    key: str                     # 資料鍵名，如 "BTCUSDT_1h"
    shape: Tuple[int, ...]       # 資料形狀
    dtype: str                   # 資料類型（字串格式，如 'float32'）
    shm_name: str                # 共享記憶體名稱
    nbytes: int                  # 位元組大小
    columns: Optional[List[str]] # DataFrame 欄位名（如果適用）

    def __str__(self) -> str:
        size_mb = self.nbytes / (1024 ** 2)
        return (
            f"SharedDataInfo(key={self.key}, shape={self.shape}, "
            f"dtype={self.dtype}, size={size_mb:.2f}MB)"
        )

    def to_dict(self) -> Dict[str, Any]:
        """轉換為可 JSON 序列化的字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SharedDataInfo':
        """從字典建立實例"""
        # 將 shape 轉換為 tuple（JSON 會變成 list）
        data['shape'] = tuple(data['shape'])
        return cls(**data)


class SharedDataPool:
    """跨進程共享資料池

    使用 Python multiprocessing.shared_memory 實現零拷貝資料共享。
    所有 Worker 進程可以直接存取預載的資料，無需複製。

    使用範例：
        # 建立並預載資料
        with SharedDataPool() as pool:
            pool.preload_ohlcv("data", ["BTCUSDT"], ["1h", "4h"])

            # 在主進程中使用
            data = pool.get("BTCUSDT_1h")

        # 在子進程中附加
        pool = attach_to_pool()
        data = pool.get("BTCUSDT_1h")  # 零拷貝存取
        pool.cleanup()
    """

    def __init__(self, pool_name: str = "backtest_data"):
        """初始化共享資料池

        Args:
            pool_name: 共享記憶體池名稱（用於跨進程識別）
        """
        self.pool_name = pool_name
        self._registry: Dict[str, SharedDataInfo] = {}
        self._shm_blocks: Dict[str, shared_memory.SharedMemory] = {}
        self._is_creator = False  # 標記此實例是否為建立者
        self._registry_file = REGISTRY_DIR / f"{pool_name}_registry.json"

        # 如果是附加模式，嘗試載入現有 registry
        if self._registry_file.exists():
            self._load_registry()

        logger.info(f"初始化共享資料池: {pool_name}")

    def preload_ohlcv(
        self,
        data_dir: str,
        symbols: List[str],
        timeframes: List[str]
    ) -> Dict[str, int]:
        """預載所有 OHLCV 資料到共享記憶體

        從 Parquet 檔案讀取資料，轉換為 float32 NumPy 陣列，
        並放入共享記憶體供多進程存取。

        Args:
            data_dir: 資料目錄路徑（包含 ohlcv/ 子目錄）
            symbols: 標的列表，如 ['BTCUSDT', 'ETHUSDT']
            timeframes: 時間框架列表，如 ['5m', '15m', '1h', '4h', '1d']

        Returns:
            已載入資料的大小字典 {key: nbytes}

        Raises:
            FileNotFoundError: 如果資料檔案不存在
            MemoryError: 如果記憶體不足
        """
        data_path = Path(data_dir)
        ohlcv_dir = data_path / "ohlcv"

        if not ohlcv_dir.exists():
            raise FileNotFoundError(f"OHLCV 資料目錄不存在: {ohlcv_dir}")

        loaded_sizes = {}

        for symbol in symbols:
            for timeframe in timeframes:
                # 建立資料鍵名
                key = f"{symbol}_{timeframe}"

                # 尋找對應的 Parquet 檔案
                file_pattern = f"{symbol}_{timeframe}.parquet"
                file_path = ohlcv_dir / file_pattern

                if not file_path.exists():
                    logger.warning(f"找不到資料檔案: {file_path}，跳過")
                    continue

                try:
                    # 讀取 Parquet 並轉換為 NumPy
                    df = pd.read_parquet(file_path)

                    # 轉換為 float32 陣列（節省記憶體）
                    data = df.values.astype(np.float32)

                    # 放入共享記憶體
                    info = self.put(key, data, columns=df.columns.tolist())
                    loaded_sizes[key] = info.nbytes

                    logger.info(f"已載入 {key}: {info}")

                except Exception as e:
                    logger.error(f"載入 {file_path} 時發生錯誤: {e}")
                    continue

        total_mb = sum(loaded_sizes.values()) / (1024 ** 2)
        logger.info(f"預載完成，共 {len(loaded_sizes)} 個資料集，總大小 {total_mb:.2f}MB")

        return loaded_sizes

    def preload_funding_rates(
        self,
        data_dir: str,
        symbols: List[str]
    ) -> Dict[str, int]:
        """預載資金費率資料

        Args:
            data_dir: 資料目錄路徑（包含 funding/ 子目錄）
            symbols: 標的列表，如 ['BTCUSDT', 'ETHUSDT']

        Returns:
            已載入資料的大小字典 {key: nbytes}
        """
        data_path = Path(data_dir)
        funding_dir = data_path / "funding"

        if not funding_dir.exists():
            raise FileNotFoundError(f"資金費率資料目錄不存在: {funding_dir}")

        loaded_sizes = {}

        for symbol in symbols:
            key = f"{symbol}_funding"
            file_pattern = f"{symbol}_funding.parquet"
            file_path = funding_dir / file_pattern

            if not file_path.exists():
                logger.warning(f"找不到資金費率檔案: {file_path}，跳過")
                continue

            try:
                df = pd.read_parquet(file_path)
                data = df.values.astype(np.float32)

                info = self.put(key, data, columns=df.columns.tolist())
                loaded_sizes[key] = info.nbytes

                logger.info(f"已載入 {key}: {info}")

            except Exception as e:
                logger.error(f"載入 {file_path} 時發生錯誤: {e}")
                continue

        total_mb = sum(loaded_sizes.values()) / (1024 ** 2)
        logger.info(f"資金費率預載完成，共 {len(loaded_sizes)} 個資料集，總大小 {total_mb:.2f}MB")

        return loaded_sizes

    def put(
        self,
        key: str,
        data: np.ndarray,
        columns: Optional[List[str]] = None
    ) -> SharedDataInfo:
        """將資料放入共享記憶體

        Args:
            key: 資料鍵名（唯一識別碼）
            data: NumPy 陣列（建議使用 float32 以節省記憶體）
            columns: 可選的欄位名列表（用於還原 DataFrame）

        Returns:
            SharedDataInfo 物件

        Raises:
            ValueError: 如果鍵名已存在
            MemoryError: 如果記憶體不足
        """
        if key in self._registry:
            raise ValueError(f"鍵名已存在: {key}")

        # 確保資料是 C-contiguous（提升存取效能）
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)

        # 建立共享記憶體
        shm_name = f"{self.pool_name}_{key}"
        nbytes = data.nbytes

        try:
            shm = shared_memory.SharedMemory(
                name=shm_name,
                create=True,
                size=nbytes
            )
            self._is_creator = True

            # 建立 NumPy 陣列視圖並複製資料
            shared_array = np.ndarray(
                data.shape,
                dtype=data.dtype,
                buffer=shm.buf
            )
            shared_array[:] = data[:]

            # 記錄到 registry
            info = SharedDataInfo(
                key=key,
                shape=data.shape,
                dtype=str(data.dtype),  # 轉為字串
                shm_name=shm_name,
                nbytes=nbytes,
                columns=columns
            )

            self._registry[key] = info
            self._shm_blocks[key] = shm

            # 持久化 registry（供其他進程使用）
            self._save_registry()

            logger.debug(f"已建立共享記憶體: {info}")

            return info

        except Exception as e:
            logger.error(f"建立共享記憶體失敗: {e}")
            raise

    def get(self, key: str) -> np.ndarray:
        """從共享記憶體取得資料（零拷貝）

        Args:
            key: 資料鍵名

        Returns:
            NumPy 陣列視圖（直接指向共享記憶體，零拷貝）

        Raises:
            KeyError: 如果鍵名不存在
        """
        if key not in self._registry:
            raise KeyError(f"找不到資料: {key}")

        info = self._registry[key]

        # 如果已有引用，直接返回
        if key in self._shm_blocks:
            shm = self._shm_blocks[key]
        else:
            # 附加到現有的共享記憶體
            try:
                shm = shared_memory.SharedMemory(name=info.shm_name)
                self._shm_blocks[key] = shm
            except FileNotFoundError:
                raise RuntimeError(f"共享記憶體已被釋放: {info.shm_name}")

        # 建立 NumPy 陣列視圖（零拷貝）
        shared_array = np.ndarray(
            info.shape,
            dtype=np.dtype(info.dtype),  # 從字串轉回 dtype
            buffer=shm.buf
        )

        return shared_array

    def get_dataframe(
        self,
        key: str,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """取得 DataFrame 格式的資料

        Args:
            key: 資料鍵名
            columns: 可選的欄位名列表（如果未提供，使用儲存時的欄位名）

        Returns:
            pandas DataFrame（底層陣列仍是零拷貝視圖）
        """
        data = self.get(key)
        info = self._registry[key]

        # 使用儲存時的欄位名（如果有）
        if columns is None:
            columns = info.columns

        # 建立 DataFrame（view，不複製資料）
        df = pd.DataFrame(data)
        if columns is not None:
            df.columns = pd.Index(columns)  # type: ignore[assignment]

        return df

    def list_keys(self) -> List[str]:
        """列出所有已載入的資料鍵"""
        return list(self._registry.keys())

    def get_info(self, key: str) -> SharedDataInfo:
        """取得資料的詳細資訊

        Args:
            key: 資料鍵名

        Returns:
            SharedDataInfo 物件
        """
        if key not in self._registry:
            raise KeyError(f"找不到資料: {key}")

        return self._registry[key]

    def get_total_size(self) -> int:
        """取得共享記憶體總大小（bytes）"""
        return sum(info.nbytes for info in self._registry.values())

    def get_total_size_mb(self) -> float:
        """取得共享記憶體總大小（MB）"""
        return self.get_total_size() / (1024 ** 2)

    def detach(self):
        """分離共享記憶體引用（子進程使用）

        只關閉引用，不刪除共享記憶體。
        子進程應使用此方法，建立者使用 cleanup()。
        """
        for key, shm in self._shm_blocks.items():
            try:
                shm.close()
                logger.debug(f"已分離共享記憶體引用: {key}")
            except Exception as e:
                logger.warning(f"分離共享記憶體 {key} 時發生錯誤: {e}")

        self._shm_blocks.clear()
        logger.info("共享記憶體引用已分離")

    def cleanup(self):
        """清理所有共享記憶體

        注意：只有建立者應該 unlink，其他進程只需 close
        """
        for key, shm in self._shm_blocks.items():
            try:
                shm.close()

                # 只有建立者才 unlink（刪除共享記憶體）
                if self._is_creator:
                    shm.unlink()
                    logger.debug(f"已釋放共享記憶體: {key}")

            except Exception as e:
                logger.warning(f"清理共享記憶體 {key} 時發生錯誤: {e}")

        self._shm_blocks.clear()
        self._registry.clear()

        # 建立者負責刪除 registry 檔案
        if self._is_creator and self._registry_file.exists():
            try:
                self._registry_file.unlink()
                logger.debug(f"已刪除 registry 檔案: {self._registry_file}")
            except Exception as e:
                logger.warning(f"刪除 registry 檔案失敗: {e}")

        if self._is_creator:
            logger.info("共享記憶體已清理（建立者）")
        else:
            logger.info("共享記憶體已關閉（使用者）")

    def __enter__(self):
        """Context manager 進入"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 退出（自動清理）"""
        self.cleanup()

    def __repr__(self) -> str:
        return (
            f"SharedDataPool(name={self.pool_name}, "
            f"keys={len(self._registry)}, "
            f"total_size={self.get_total_size_mb():.2f}MB)"
        )

    def _save_registry(self):
        """持久化 registry 到檔案（供其他進程讀取）"""
        registry_data = {
            key: info.to_dict()
            for key, info in self._registry.items()
        }

        try:
            with open(self._registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)

            logger.debug(f"Registry 已儲存到 {self._registry_file}")

        except Exception as e:
            logger.error(f"儲存 registry 失敗: {e}")

    def _load_registry(self):
        """從檔案載入 registry（供子進程使用）"""
        try:
            with open(self._registry_file, 'r') as f:
                registry_data = json.load(f)

            self._registry = {
                key: SharedDataInfo.from_dict(data)
                for key, data in registry_data.items()
            }

            logger.debug(f"Registry 已從 {self._registry_file} 載入，共 {len(self._registry)} 個項目")

        except Exception as e:
            logger.error(f"載入 registry 失敗: {e}")
            self._registry = {}


# ============= 便利函數 =============

def create_shared_pool(
    data_dir: str = "data",
    symbols: List[str] = ["BTCUSDT", "ETHUSDT"],
    timeframes: List[str] = ["5m", "15m", "30m", "1h", "4h", "1d"],
    pool_name: str = "backtest_data",
    include_funding: bool = True
) -> SharedDataPool:
    """建立並預載共享資料池

    這是最常用的進入點，一次完成建立和預載。

    Args:
        data_dir: 資料目錄路徑
        symbols: 標的列表
        timeframes: 時間框架列表
        pool_name: 共享記憶體池名稱
        include_funding: 是否包含資金費率資料

    Returns:
        已預載資料的 SharedDataPool 實例

    使用範例：
        pool = create_shared_pool(
            symbols=["BTCUSDT"],
            timeframes=["1h", "4h"]
        )

        # 使用資料
        btc_1h = pool.get("BTCUSDT_1h")

        # 清理
        pool.cleanup()
    """
    pool = SharedDataPool(pool_name=pool_name)

    # 預載 OHLCV 資料
    pool.preload_ohlcv(data_dir, symbols, timeframes)

    # 預載資金費率
    if include_funding:
        pool.preload_funding_rates(data_dir, symbols)

    logger.info(f"共享資料池已建立: {pool}")

    return pool


def attach_to_pool(pool_name: str = "backtest_data") -> SharedDataPool:
    """附加到現有共享資料池（供子進程使用）

    子進程應使用此函數附加到主進程建立的共享記憶體。

    Args:
        pool_name: 共享記憶體池名稱（必須與建立者相同）

    Returns:
        SharedDataPool 實例（可存取已預載的資料）

    使用範例：
        # 在子進程中
        pool = attach_to_pool()
        data = pool.get("BTCUSDT_1h")  # 零拷貝存取
        pool.cleanup()  # 只 close，不 unlink
    """
    pool = SharedDataPool(pool_name=pool_name)
    pool._is_creator = False  # 標記為使用者，不是建立者

    logger.info(f"已附加到共享資料池: {pool_name}")

    return pool


# ============= 測試程式 =============

if __name__ == "__main__":
    """測試共享資料池的基本功能"""

    # 設定 logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("測試 1: 基本 put/get 功能")
    print("=" * 60)

    with SharedDataPool(pool_name="test_pool") as pool:
        # 建立測試資料
        test_data = np.random.randn(1000, 6).astype(np.float32)
        columns = ['open', 'high', 'low', 'close', 'volume', 'count']

        # 放入共享記憶體
        info = pool.put("test_data", test_data, columns=columns)
        print(f"已放入資料: {info}")

        # 取得資料（零拷貝）
        retrieved = pool.get("test_data")
        print(f"已取得資料: shape={retrieved.shape}, dtype={retrieved.dtype}")

        # 驗證零拷貝（第二次 get 應該與第一次共享記憶體）
        retrieved2 = pool.get("test_data")
        assert np.shares_memory(retrieved, retrieved2), "兩次 get 應該共享記憶體！"
        print("✓ 零拷貝驗證通過（多次 get 共享記憶體）")

        # 驗證資料一致性
        assert np.allclose(test_data, retrieved), "資料應該完全一致！"
        print("✓ 資料一致性驗證通過")

        # 測試 DataFrame 格式
        df = pool.get_dataframe("test_data")
        print(f"DataFrame shape: {df.shape}, columns: {list(df.columns)}")

        # 列出所有鍵
        print(f"所有鍵: {pool.list_keys()}")

        # 總大小
        print(f"總大小: {pool.get_total_size_mb():.2f}MB")

    print("\n✓ 測試 1 通過\n")

    # 測試 2：預載功能（需要實際資料檔案）
    print("=" * 60)
    print("測試 2: 預載功能（需要 data/ohlcv/ 目錄）")
    print("=" * 60)

    data_dir = Path("data")
    if (data_dir / "ohlcv").exists():
        try:
            pool = create_shared_pool(
                data_dir="data",
                symbols=["BTCUSDT"],
                timeframes=["1h"],
                include_funding=False
            )

            print(f"預載成功: {pool}")

            # 嘗試存取
            if "BTCUSDT_1h" in pool.list_keys():
                btc_data = pool.get("BTCUSDT_1h")
                print(f"BTCUSDT_1h shape: {btc_data.shape}")

            pool.cleanup()
            print("✓ 測試 2 通過")

        except Exception as e:
            print(f"測試 2 跳過（資料不存在或格式不符）: {e}")
    else:
        print("測試 2 跳過（data/ohlcv/ 目錄不存在）")

    print("\n" + "=" * 60)
    print("所有測試完成！")
    print("=" * 60)
