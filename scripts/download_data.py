#!/usr/bin/env python3
"""
批量下載歷史資料腳本
從 Binance Futures 下載 OHLCV 和資金費率資料
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.fetcher import DataFetcher


def download_symbol_timeframe(
    symbol: str,
    timeframe: str,
    start_date: datetime,
    output_dir: Path,
    verbose: bool = True
) -> dict:
    """下載單一交易對和週期的資料"""
    fetcher = DataFetcher(verbose=verbose)

    result = {
        'symbol': symbol,
        'timeframe': timeframe,
        'status': 'pending',
        'rows': 0,
        'file_path': None,
        'error': None
    }

    try:
        # 下載 OHLCV
        ohlcv_path = output_dir / 'ohlcv' / f'{symbol}_{timeframe}.parquet'

        print(f"\n{'='*60}")
        print(f"下載 {symbol} {timeframe}")
        print(f"時間範圍: {start_date} 到 現在")
        print(f"{'='*60}")

        df = fetcher.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=datetime.now()
        )

        # 驗證資料
        issues = fetcher.validate_data(df)
        if issues:
            print(f"⚠️ 資料品質問題: {issues}")

        # 儲存
        fetcher.save_to_parquet(df, str(ohlcv_path))

        result['status'] = 'success'
        result['rows'] = len(df)
        result['file_path'] = str(ohlcv_path)
        result['date_range'] = f"{df.index.min()} - {df.index.max()}"

        print(f"✅ 完成: {len(df)} 筆資料")
        print(f"   時間: {df.index.min()} ~ {df.index.max()}")

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        print(f"❌ 錯誤: {e}")

    return result


def download_funding_rates(
    symbol: str,
    start_date: datetime,
    output_dir: Path,
    verbose: bool = True
) -> dict:
    """下載資金費率資料"""
    fetcher = DataFetcher(verbose=verbose)

    result = {
        'symbol': symbol,
        'type': 'funding_rates',
        'status': 'pending',
        'rows': 0,
        'file_path': None,
        'error': None
    }

    try:
        funding_path = output_dir / 'funding_rates' / f'{symbol}_funding.parquet'

        print(f"\n{'='*60}")
        print(f"下載 {symbol} 資金費率")
        print(f"時間範圍: {start_date} 到 現在")
        print(f"{'='*60}")

        df = fetcher.fetch_funding_rates(
            symbol=symbol,
            start_date=start_date,
            end_date=datetime.now()
        )

        fetcher.save_to_parquet(df, str(funding_path))

        result['status'] = 'success'
        result['rows'] = len(df)
        result['file_path'] = str(funding_path)
        result['date_range'] = f"{df.index.min()} - {df.index.max()}"

        print(f"✅ 完成: {len(df)} 筆資料")
        print(f"   時間: {df.index.min()} ~ {df.index.max()}")

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        print(f"❌ 錯誤: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description='下載歷史交易資料')
    parser.add_argument('--symbol', type=str, help='交易對 (BTCUSDT, ETHUSDT)')
    parser.add_argument('--timeframe', type=str, help='週期 (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)')
    parser.add_argument('--all', action='store_true', help='下載所有交易對和週期')
    parser.add_argument('--funding', action='store_true', help='下載資金費率')
    parser.add_argument('--start-year', type=int, default=2017, help='起始年份')
    parser.add_argument('--output', type=str, default='data', help='輸出目錄')

    args = parser.parse_args()

    output_dir = project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'ohlcv').mkdir(exist_ok=True)
    (output_dir / 'funding_rates').mkdir(exist_ok=True)

    start_date = datetime(args.start_year, 1, 1)

    symbols = ['BTCUSDT', 'ETHUSDT']
    timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']

    results = []

    if args.all:
        # 下載所有組合
        for symbol in symbols:
            for tf in timeframes:
                result = download_symbol_timeframe(symbol, tf, start_date, output_dir)
                results.append(result)

            # 下載資金費率
            result = download_funding_rates(symbol, start_date, output_dir)
            results.append(result)

    elif args.symbol and args.timeframe:
        # 下載指定的交易對和週期
        result = download_symbol_timeframe(args.symbol, args.timeframe, start_date, output_dir)
        results.append(result)

        if args.funding:
            result = download_funding_rates(args.symbol, start_date, output_dir)
            results.append(result)

    elif args.symbol and args.funding:
        # 只下載資金費率
        result = download_funding_rates(args.symbol, start_date, output_dir)
        results.append(result)

    else:
        parser.print_help()
        return

    # 輸出摘要
    print(f"\n{'='*60}")
    print("下載摘要")
    print(f"{'='*60}")

    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')
    total_rows = sum(r['rows'] for r in results)

    print(f"成功: {success_count}")
    print(f"失敗: {error_count}")
    print(f"總資料量: {total_rows:,} 筆")

    if error_count > 0:
        print("\n錯誤列表:")
        for r in results:
            if r['status'] == 'error':
                print(f"  - {r.get('symbol', '')} {r.get('timeframe', r.get('type', ''))}: {r['error']}")


if __name__ == '__main__':
    main()
