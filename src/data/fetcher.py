"""
資料獲取模組

使用 CCXT 從 Binance Futures 獲取歷史資料
支援 OHLCV 和資金費率的批量下載與增量更新
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Literal

import ccxt
import pandas as pd


class DataFetcher:
    """資料獲取器 - 從 Binance Futures 下載交易資料"""

    def __init__(
        self,
        exchange_name: str = 'binance',
        rate_limit: bool = True,
        verbose: bool = False
    ):
        """
        初始化資料獲取器

        Args:
            exchange_name: 交易所名稱，預設 'binance'
            rate_limit: 是否啟用速率限制，預設 True
            verbose: 是否顯示詳細資訊，預設 False
        """
        self.exchange = ccxt.binance({
            'enableRateLimit': rate_limit,
            'options': {'defaultType': 'future'},
            'verbose': verbose
        })
        self.verbose = verbose

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '4h',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        批量下載 OHLCV 資料

        Args:
            symbol: 交易對，如 'BTCUSDT'
            timeframe: 時間框架，如 '1h', '4h', '1d'
            start_date: 起始日期，預設為 2 年前
            end_date: 結束日期，預設為當前時間
            limit: 每次請求的最大資料量，預設 1000

        Returns:
            包含 OHLCV 資料的 DataFrame，時間戳為索引
        """
        # 標準化交易對格式
        normalized_symbol = self._normalize_symbol(symbol)

        # 設定預設日期範圍
        if start_date is None:
            start_date = datetime.now() - timedelta(days=730)
        if end_date is None:
            end_date = datetime.now()

        if self.verbose:
            print(f"開始下載 {normalized_symbol} {timeframe} 資料")
            print(f"時間範圍: {start_date} 到 {end_date}")

        all_data = []
        current = start_date

        while current < end_date:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=normalized_symbol,
                    timeframe=timeframe,
                    since=int(current.timestamp() * 1000),
                    limit=limit
                )

                if not ohlcv:
                    if self.verbose:
                        print(f"無法獲取 {current} 之後的資料，停止下載")
                    break

                all_data.extend(ohlcv)
                last_timestamp = datetime.fromtimestamp(ohlcv[-1][0] / 1000)

                if self.verbose:
                    print(f"已下載到 {last_timestamp}, 共 {len(all_data)} 筆")

                # 檢查是否已到達最新資料
                if last_timestamp <= current:
                    if self.verbose:
                        print("已到達最新資料")
                    break

                current = last_timestamp

                # 避免觸發 rate limit
                time.sleep(0.1)

            except Exception as e:
                print(f"下載資料時發生錯誤: {e}")
                break

        if not all_data:
            raise ValueError(f"未能獲取任何資料: {symbol} {timeframe}")

        # 轉換為 DataFrame
        df = pd.DataFrame(
            all_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # 移除重複和排序
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()

        if self.verbose:
            print(f"下載完成: {len(df)} 筆資料")
            print(f"時間範圍: {df.index.min()} 到 {df.index.max()}")

        return df

    def fetch_funding_rates(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        批量下載資金費率資料

        Args:
            symbol: 交易對，如 'BTCUSDT'
            start_date: 起始日期，預設為 2 年前
            end_date: 結束日期，預設為當前時間
            limit: 每次請求的最大資料量，預設 1000

        Returns:
            包含資金費率的 DataFrame，時間戳為索引
        """
        normalized_symbol = self._normalize_symbol(symbol)

        if start_date is None:
            start_date = datetime.now() - timedelta(days=730)
        if end_date is None:
            end_date = datetime.now()

        if self.verbose:
            print(f"開始下載 {normalized_symbol} 資金費率")
            print(f"時間範圍: {start_date} 到 {end_date}")

        all_rates = []
        current = start_date

        while current < end_date:
            try:
                rates = self.exchange.fetch_funding_rate_history(
                    symbol=normalized_symbol,
                    since=int(current.timestamp() * 1000),
                    limit=limit
                )

                if not rates:
                    if self.verbose:
                        print(f"無法獲取 {current} 之後的資金費率，停止下載")
                    break

                all_rates.extend(rates)
                last_timestamp = datetime.fromtimestamp(rates[-1]['timestamp'] / 1000)

                if self.verbose:
                    print(f"已下載到 {last_timestamp}, 共 {len(all_rates)} 筆")

                if last_timestamp <= current:
                    if self.verbose:
                        print("已到達最新資料")
                    break

                current = last_timestamp
                time.sleep(0.1)

            except Exception as e:
                print(f"下載資金費率時發生錯誤: {e}")
                break

        if not all_rates:
            raise ValueError(f"未能獲取任何資金費率資料: {symbol}")

        # 轉換為 DataFrame
        df = pd.DataFrame(all_rates)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['fundingRate']].rename(columns={'fundingRate': 'rate'})

        # 移除重複和排序
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()

        if self.verbose:
            print(f"下載完成: {len(df)} 筆資金費率")
            print(f"時間範圍: {df.index.min()} 到 {df.index.max()}")

        return df

    def save_to_parquet(
        self,
        df: pd.DataFrame,
        file_path: str,
        compression: str = 'snappy'
    ) -> None:
        """
        儲存資料為 Parquet 格式

        Args:
            df: 要儲存的 DataFrame
            file_path: 儲存路徑
            compression: 壓縮演算法，預設 'snappy'
        """
        # 確保目錄存在
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(file_path, compression=compression)

        if self.verbose:
            file_size = os.path.getsize(file_path) / 1024 / 1024
            print(f"已儲存到 {file_path} ({file_size:.2f} MB)")

    def update_data(
        self,
        symbol: str,
        timeframe: str,
        data_path: str,
        data_type: Literal['ohlcv', 'funding'] = 'ohlcv'
    ) -> pd.DataFrame:
        """
        增量更新資料（僅下載最新部分）

        Args:
            symbol: 交易對
            timeframe: 時間框架（僅 OHLCV 需要）
            data_path: 資料檔案路徑
            data_type: 資料類型，'ohlcv' 或 'funding'

        Returns:
            更新後的完整 DataFrame
        """
        # 讀取現有資料
        if os.path.exists(data_path):
            existing = pd.read_parquet(data_path)
            last_timestamp = existing.index.max()

            if self.verbose:
                print(f"現有資料最後時間: {last_timestamp}")
                print(f"準備下載增量資料...")

            # 從最後時間戳開始下載（往前推一點以避免遺漏）
            start_date = last_timestamp - timedelta(days=1)
        else:
            existing = pd.DataFrame()
            start_date = datetime.now() - timedelta(days=730)

            if self.verbose:
                print("無現有資料，將下載完整歷史")

        # 下載新資料
        if data_type == 'ohlcv':
            new_data = self.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=datetime.now()
            )
        elif data_type == 'funding':
            new_data = self.fetch_funding_rates(
                symbol=symbol,
                start_date=start_date,
                end_date=datetime.now()
            )
        else:
            raise ValueError(f"不支援的資料類型: {data_type}")

        # 合併資料
        if not existing.empty:
            combined = pd.concat([existing, new_data])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()

            new_count = len(combined) - len(existing)
            if self.verbose:
                print(f"新增 {new_count} 筆資料")
        else:
            combined = new_data

        # 儲存更新後的資料
        self.save_to_parquet(combined, data_path)

        return combined

    def _normalize_symbol(self, symbol: str) -> str:
        """
        標準化交易對格式為 CCXT 格式

        Args:
            symbol: 原始交易對格式，如 'BTCUSDT' 或 'BTC/USDT'

        Returns:
            CCXT 標準格式 'BTC/USDT'
        """
        if '/' in symbol:
            return symbol

        # 處理 USDT 永續合約
        if symbol.endswith('USDT'):
            base = symbol[:-4]
            return f"{base}/USDT"

        raise ValueError(f"無法解析交易對格式: {symbol}")

    def validate_data(self, df: pd.DataFrame) -> list[str]:
        """
        驗證 OHLCV 資料品質

        Args:
            df: OHLCV DataFrame

        Returns:
            問題列表，如果無問題則為空列表
        """
        issues = []

        # 1. 檢查缺失值
        if df.isnull().any().any():
            null_count = df.isnull().sum().sum()
            issues.append(f"缺失值: {null_count} 筆")

        # 2. 檢查時間連續性
        if len(df) > 1:
            gaps = df.index.to_series().diff()
            median_gap = gaps.median()
            large_gaps = (gaps > median_gap * 2).sum()
            if large_gaps > 0:
                issues.append(f"時間不連續: {large_gaps} 處")

        # 3. 檢查 OHLC 邏輯（僅針對 OHLCV 資料）
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_ohlc = (
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close']) |
                (df['high'] < df['low'])
            )
            if invalid_ohlc.any():
                issues.append(f"OHLC 邏輯錯誤: {invalid_ohlc.sum()} 筆")

        # 4. 檢查成交量（如果有）
        if 'volume' in df.columns:
            if (df['volume'] <= 0).any():
                issues.append(f"成交量異常: {(df['volume'] <= 0).sum()} 筆")

        # 5. 檢查重複時間戳
        if df.index.duplicated().any():
            issues.append(f"重複時間戳: {df.index.duplicated().sum()} 筆")

        return issues


# 快速使用範例
if __name__ == '__main__':
    # 建立 fetcher
    fetcher = DataFetcher(verbose=True)

    # 下載 BTCUSDT 4小時資料
    btc_data = fetcher.fetch_ohlcv(
        symbol='BTCUSDT',
        timeframe='4h',
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2025, 1, 1)
    )

    # 驗證資料品質
    issues = fetcher.validate_data(btc_data)
    if issues:
        print(f"資料品質問題: {issues}")
    else:
        print("資料品質良好")

    # 儲存資料
    fetcher.save_to_parquet(
        btc_data,
        'data/ohlcv/BTCUSDT_4h.parquet'
    )

    # 下載資金費率
    btc_funding = fetcher.fetch_funding_rates(
        symbol='BTCUSDT',
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2025, 1, 1)
    )

    fetcher.save_to_parquet(
        btc_funding,
        'data/funding_rates/BTCUSDT_funding.parquet'
    )
