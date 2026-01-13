"""
GPU 批量優化器使用範例

展示如何使用 GPUBatchOptimizer 進行高效參數優化。
"""

import sys
from pathlib import Path

# 添加專案根目錄到 Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.optimizer import GPUBatchOptimizer, gpu_optimize_strategy
import logging

logging.basicConfig(level=logging.INFO)


def example_1_basic_usage():
    """範例 1：基本使用"""
    print("\n" + "=" * 60)
    print("範例 1：GPU 批量優化基本使用")
    print("=" * 60)

    # 建立優化器
    optimizer = GPUBatchOptimizer(
        prefer_mlx=True,
        fallback_to_cpu=True,
        verbose=True
    )

    print(f"後端：{optimizer._backend}")
    print(f"GPU 可用：{optimizer.is_gpu_available()}")

    # 定義簡單策略函數（移動平均交叉）
    # 注意：Metal Engine 期待 strategy_fn(prices, **params)
    def ma_cross_strategy(price_data: np.ndarray, fast_period: int, slow_period: int) -> np.ndarray:
        """移動平均交叉策略"""
        prices = price_data[:, 3]  # 收盤價

        # 簡單移動平均
        fast_ma = np.convolve(prices, np.ones(fast_period)/fast_period, mode='same')
        slow_ma = np.convolve(prices, np.ones(slow_period)/slow_period, mode='same')

        # 生成信號
        signals = np.where(fast_ma > slow_ma, 1, -1)

        return signals.astype(np.float32)

    # 生成假資料（實際應使用真實市場資料）
    np.random.seed(42)
    n_bars = 1000
    price_data = np.random.randn(n_bars, 5).astype(np.float32)
    price_data[:, 3] = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)  # 收盤價

    # 定義參數空間
    param_space = {
        'fast_period': {'type': 'int', 'low': 5, 'high': 20},
        'slow_period': {'type': 'int', 'low': 20, 'high': 50}
    }

    # 執行優化
    result = optimizer.batch_optimize(
        strategy_fn=ma_cross_strategy,
        price_data=price_data,
        param_space=param_space,
        n_trials=50,
        batch_size=10,
        metric='sharpe_ratio',
        direction='maximize',
        seed=42
    )

    # 顯示結果
    print(result.summary())

    # 顯示前 5 名結果
    top_5 = sorted(result.all_results, key=lambda r: r.sharpe_ratio, reverse=True)[:5]
    print("\nTop 5 參數組合：")
    for i, r in enumerate(top_5, 1):
        print(f"{i}. Sharpe={r.sharpe_ratio:.4f}, Params={r.params}")


def example_2_strategy_wrapper():
    """範例 2：使用策略包裝器"""
    print("\n" + "=" * 60)
    print("範例 2：使用 gpu_optimize_strategy 便利函數")
    print("=" * 60)

    # 模擬策略類別
    class SimpleStrategy:
        def __init__(self):
            self.fast_period = 10
            self.slow_period = 30

            # 定義參數空間
            self.param_space = {
                'fast_period': {'type': 'int', 'low': 5, 'high': 20},
                'slow_period': {'type': 'int', 'low': 20, 'high': 50}
            }

        def generate_signals(self, df):
            """生成交易信號"""
            import pandas as pd

            prices = df['close'].values
            fast_ma = pd.Series(prices).rolling(self.fast_period).mean().values
            slow_ma = pd.Series(prices).rolling(self.slow_period).mean().values

            signals = np.where(fast_ma > slow_ma, 1, -1)
            return pd.Series(signals)

    # 生成測試資料
    import pandas as pd
    np.random.seed(42)
    n_bars = 1000

    df = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(n_bars) * 0.5),
        'high': 101 + np.cumsum(np.random.randn(n_bars) * 0.5),
        'low': 99 + np.cumsum(np.random.randn(n_bars) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(n_bars) * 0.5),
        'volume': np.random.randint(1000, 10000, n_bars)
    })

    # 轉為 numpy
    price_data = df.values.astype(np.float32)

    # 建立策略
    strategy = SimpleStrategy()

    # 使用便利函數優化
    result = gpu_optimize_strategy(
        strategy=strategy,
        data=price_data,
        n_trials=30,
        batch_size=10,
        metric='sharpe_ratio'
    )

    print(result.summary())


def example_3_custom_metric():
    """範例 3：自訂優化指標"""
    print("\n" + "=" * 60)
    print("範例 3：使用不同優化指標")
    print("=" * 60)

    optimizer = GPUBatchOptimizer(verbose=False)

    # 簡單策略
    def simple_strategy(price_data: np.ndarray, params: dict) -> np.ndarray:
        threshold = params['threshold']
        returns = np.diff(price_data[:, 3])
        signals = np.where(returns > threshold, 1, -1)
        signals = np.append(signals, 0)  # 對齊長度
        return signals.astype(np.float32)

    # 生成資料
    np.random.seed(42)
    price_data = np.random.randn(500, 5).astype(np.float32)
    price_data[:, 3] = 100 + np.cumsum(np.random.randn(500) * 0.5)

    param_space = {
        'threshold': {'type': 'float', 'low': 0.0, 'high': 1.0, 'step': 0.1}
    }

    # 優化不同指標
    metrics = ['sharpe_ratio', 'total_return', 'max_drawdown']

    for metric in metrics:
        direction = 'minimize' if metric == 'max_drawdown' else 'maximize'

        result = optimizer.batch_optimize(
            strategy_fn=simple_strategy,
            price_data=price_data,
            param_space=param_space,
            n_trials=20,
            batch_size=5,
            metric=metric,
            direction=direction,
            seed=42
        )

        print(f"\n優化指標：{metric}")
        print(f"  最佳參數：{result.best_params}")
        print(f"  最佳值：{getattr(result, f'best_{metric.split("_")[0]}'):.4f}")


if __name__ == "__main__":
    # 執行所有範例
    example_1_basic_usage()
    # example_2_strategy_wrapper()  # 需要 pandas
    # example_3_custom_metric()

    print("\n" + "=" * 60)
    print("✅ 所有範例完成")
    print("=" * 60)
