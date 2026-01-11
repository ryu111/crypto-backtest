#!/usr/bin/env python3
"""
Metal GPU 加速回測範例

展示如何使用 MetalBacktestEngine 進行批次回測與參數優化。
"""

import sys
from pathlib import Path

# 加入專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.backtester.metal_engine import MetalBacktestEngine


def generate_sample_data(length: int = 5000, seed: int = 42) -> np.ndarray:
    """生成模擬價格資料"""
    np.random.seed(seed)
    # 隨機遊走 + 趨勢
    trend = np.linspace(0, 50, length)
    noise = np.cumsum(np.random.randn(length) * 0.5)
    prices = 100 + trend + noise
    return prices.reshape(-1, 1)


def simple_sma_strategy(prices: np.ndarray, sma_period: int = 20) -> np.ndarray:
    """
    簡單 SMA 策略

    規則：價格 > SMA → 做多(1)，否則空手(0)
    """
    prices_1d = prices[:, 0]
    signals = np.zeros(len(prices))

    for i in range(sma_period, len(prices)):
        sma = np.mean(prices_1d[i - sma_period:i])
        signals[i] = 1.0 if prices_1d[i] > sma else 0.0

    return signals


def dual_sma_strategy(
    prices: np.ndarray,
    fast_period: int = 10,
    slow_period: int = 50
) -> np.ndarray:
    """
    雙均線策略

    規則：快線 > 慢線 → 做多(1)，否則空手(0)
    """
    engine = MetalBacktestEngine()  # 用於計算 SMA
    prices_1d = prices[:, 0]

    fast_sma = engine.gpu_sma(prices_1d, fast_period)
    slow_sma = engine.gpu_sma(prices_1d, slow_period)

    signals = np.zeros(len(prices))

    # 避免 NaN
    valid_idx = max(fast_period, slow_period)
    signals[valid_idx:] = (fast_sma[valid_idx:] > slow_sma[valid_idx:]).astype(float)

    return signals


def example_1_basic_backtest():
    """範例 1：基本批次回測"""
    print("=" * 60)
    print("範例 1：基本批次回測")
    print("=" * 60)

    # 初始化
    engine = MetalBacktestEngine(prefer_mlx=True)
    print(f"Backend: {engine.backend}")
    print(f"GPU Available: {engine.is_gpu_available()}\n")

    # 資料
    prices = generate_sample_data(1000)

    # 參數網格
    param_grid = [
        {"sma_period": 10},
        {"sma_period": 20},
        {"sma_period": 50},
    ]

    # 執行回測
    print(f"Testing {len(param_grid)} parameter combinations...")
    results = engine.batch_backtest(prices, param_grid, simple_sma_strategy)

    # 顯示結果
    print("\nResults:")
    print("-" * 60)
    for result in results:
        print(f"Period: {result.params['sma_period']:3d} | "
              f"Return: {result.total_return:8.4f} | "
              f"Sharpe: {result.sharpe_ratio:6.3f} | "
              f"MaxDD: {result.max_drawdown:6.3f} | "
              f"Time: {result.execution_time_ms:6.2f}ms")

    print("\n")


def example_2_parameter_optimization():
    """範例 2：參數優化"""
    print("=" * 60)
    print("範例 2：參數優化（雙均線策略）")
    print("=" * 60)

    # 初始化
    engine = MetalBacktestEngine(prefer_mlx=True)
    print(f"Backend: {engine.backend}\n")

    # 資料
    prices = generate_sample_data(5000)

    # 大參數網格
    param_grid = [
        {"fast_period": fast, "slow_period": slow}
        for fast in [5, 10, 15, 20]
        for slow in [30, 50, 70, 100]
        if fast < slow
    ]

    print(f"Testing {len(param_grid)} parameter combinations...")

    # 執行回測
    results = engine.batch_backtest(prices, param_grid, dual_sma_strategy)

    # 按 Sharpe Ratio 排序
    results.sort(key=lambda r: r.sharpe_ratio, reverse=True)

    # 顯示前 5 名
    print("\nTop 5 Results:")
    print("-" * 60)
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. Fast: {result.params['fast_period']:2d}, "
              f"Slow: {result.params['slow_period']:3d} | "
              f"Sharpe: {result.sharpe_ratio:6.3f} | "
              f"Return: {result.total_return:8.4f} | "
              f"MaxDD: {result.max_drawdown:6.3f}")

    print("\n")


def example_3_gpu_indicators():
    """範例 3：GPU 加速技術指標"""
    print("=" * 60)
    print("範例 3：GPU 加速技術指標")
    print("=" * 60)

    # 初始化
    engine = MetalBacktestEngine(prefer_mlx=True)
    print(f"Backend: {engine.backend}\n")

    # 簡單價格序列
    prices = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    period = 3

    # 計算 SMA
    sma = engine.gpu_sma(prices, period)
    print("SMA Calculation:")
    print(f"Prices: {prices}")
    print(f"SMA({period}): {sma}")
    print()

    # 計算 EMA
    ema = engine.gpu_ema(prices, period)
    print("EMA Calculation:")
    print(f"Prices: {prices}")
    print(f"EMA({period}): {ema}")
    print()

    # 比較 SMA 和 EMA
    print("Comparison (valid values only):")
    valid_idx = period - 1
    for i in range(valid_idx, len(prices)):
        print(f"  Index {i}: Price={prices[i]:.2f}, "
              f"SMA={sma[i]:.4f}, EMA={ema[i]:.4f}")

    print("\n")


def example_4_performance_comparison():
    """範例 4：效能對比（CPU vs GPU）"""
    print("=" * 60)
    print("範例 4：效能對比（CPU vs GPU）")
    print("=" * 60)

    # 資料
    prices = generate_sample_data(5000)

    # 大參數網格
    param_grid = [
        {"sma_period": period}
        for period in range(10, 100, 5)
    ]

    print(f"Testing {len(param_grid)} parameter combinations...\n")

    # GPU 執行
    engine_gpu = MetalBacktestEngine(prefer_mlx=True)
    if engine_gpu.is_gpu_available():
        import time

        start = time.perf_counter()
        results_gpu = engine_gpu.batch_backtest(prices, param_grid, simple_sma_strategy)
        gpu_time = (time.perf_counter() - start) * 1000

        print(f"GPU ({engine_gpu.backend}) Time: {gpu_time:.2f}ms")

        # CPU 執行
        engine_gpu.backend = "cpu"
        start = time.perf_counter()
        results_cpu = engine_gpu.batch_backtest(prices, param_grid, simple_sma_strategy)
        cpu_time = (time.perf_counter() - start) * 1000

        print(f"CPU Time: {cpu_time:.2f}ms")
        print(f"Speedup: {cpu_time / gpu_time:.2f}x")
    else:
        print("GPU not available, skipping comparison")

    print("\n")


def main():
    """執行所有範例"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "Metal GPU 加速回測範例" + " " * 26 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")

    example_1_basic_backtest()
    example_2_parameter_optimization()
    example_3_gpu_indicators()
    example_4_performance_comparison()

    print("✅ All examples completed!")


if __name__ == "__main__":
    main()
