"""
貝葉斯優化器使用範例

展示如何使用 BayesianOptimizer 進行策略參數優化。
"""

import pandas as pd
from datetime import datetime

from src.backtester.engine import BacktestEngine, BacktestConfig
from src.optimizer import BayesianOptimizer, optimize_strategy
from src.strategies.base import BaseStrategy


# 1. 定義策略範例（需有 param_space）
class SimpleMAStrategy(BaseStrategy):
    """簡單移動平均策略"""

    name = "simple_ma"
    strategy_type = "trend"
    version = "1.0"
    description = "雙均線交叉策略"

    params = {
        'fast_period': 10,
        'slow_period': 30,
        'use_filter': False
    }

    # 定義參數空間（Optuna 優化用）
    param_space = {
        'fast_period': {
            'type': 'int',
            'low': 5,
            'high': 20,
            'step': 1
        },
        'slow_period': {
            'type': 'int',
            'low': 20,
            'high': 50,
            'step': 5
        },
        'use_filter': {
            'type': 'categorical',
            'choices': [True, False]
        }
    }

    def calculate_indicators(self, data):
        """計算指標"""
        fast = data['close'].rolling(self.params['fast_period']).mean()
        slow = data['close'].rolling(self.params['slow_period']).mean()
        return {'fast_ma': fast, 'slow_ma': slow}

    def generate_signals(self, data):
        """產生交易訊號"""
        indicators = self.calculate_indicators(data)
        fast_ma = indicators['fast_ma']
        slow_ma = indicators['slow_ma']

        # 金叉做多，死叉平倉
        long_entry = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        long_exit = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

        # 不做空
        short_entry = pd.Series(False, index=data.index)
        short_exit = pd.Series(False, index=data.index)

        return long_entry, long_exit, short_entry, short_exit


def main():
    """主程式"""

    # 2. 準備市場資料（這裡使用模擬資料）
    print("載入市場資料...")
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1H')
    data = pd.DataFrame({
        'open': 30000 + pd.Series(range(len(dates))) * 0.5,
        'high': 30100 + pd.Series(range(len(dates))) * 0.5,
        'low': 29900 + pd.Series(range(len(dates))) * 0.5,
        'close': 30000 + pd.Series(range(len(dates))) * 0.5,
        'volume': 100
    }, index=dates)

    # 3. 建立回測引擎
    print("建立回測引擎...")
    config = BacktestConfig(
        symbol='BTCUSDT',
        timeframe='1h',
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_capital=10000,
        leverage=2
    )
    engine = BacktestEngine(config)

    # 4. 建立策略實例
    strategy = SimpleMAStrategy()

    # 5a. 方法一：使用 BayesianOptimizer 類別（完整控制）
    print("\n" + "="*60)
    print("方法一：使用 BayesianOptimizer 類別")
    print("="*60)

    optimizer = BayesianOptimizer(
        engine=engine,
        n_trials=50,  # 試驗次數
        n_jobs=2,     # 並行工作數（使用 2 個 CPU）
        seed=42,      # 可重現性
        verbose=True
    )

    result = optimizer.optimize(
        strategy=strategy,
        data=data,
        metric='sharpe_ratio',  # 優化目標
        direction='maximize',
        show_progress_bar=True
    )

    # 顯示結果
    print(result.summary())

    # 取得優化歷史
    history_df = optimizer.get_optimization_history()
    print("\n優化歷史 (前 10 筆):")
    print(history_df.head(10))

    # 參數重要性
    importances = optimizer.get_param_importances()
    if importances:
        print("\n參數重要性:")
        for param, importance in importances.items():
            print(f"  {param}: {importance:.4f}")

    # 視覺化（需要 plotly）
    try:
        result.plot_optimization_history('optimization_history.html')
        result.plot_param_importances('param_importances.html')
        print("\n已儲存視覺化結果到 HTML 檔案")
    except Exception as e:
        print(f"\n無法產生視覺化: {e}")

    # 5b. 方法二：使用便利函數（快速優化）
    print("\n" + "="*60)
    print("方法二：使用便利函數 optimize_strategy")
    print("="*60)

    result2 = optimize_strategy(
        strategy=strategy,
        data=data,
        engine=engine,
        n_trials=30,
        metric='sortino_ratio',  # 使用 Sortino Ratio
        n_jobs=2,
        verbose=True
    )

    print(result2.summary())

    # 6. 使用 SQLite 儲存後端（推薦方式）
    print("\n" + "="*60)
    print("方法三：使用 SQLite 儲存後端")
    print("="*60)

    optimizer_db = BayesianOptimizer(
        engine=engine,
        n_trials=20,
        n_jobs=2,
        verbose=True
    )

    result3 = optimizer_db.optimize(
        strategy=strategy,
        data=data,
        metric='sharpe_ratio',
        study_name='simple_ma_optimization',
        storage='sqlite:///optuna_study.db'  # 使用 SQLite 儲存
    )

    print(result3.summary())
    print("\n已儲存 study 到 optuna_study.db")

    # 7. 載入已儲存的 study 並繼續優化
    print("\n" + "="*60)
    print("繼續優化（從資料庫載入）")
    print("="*60)

    optimizer_continue = BayesianOptimizer(
        engine=engine,
        n_trials=10,  # 額外 10 次試驗
        n_jobs=2,
        verbose=True
    )

    result4 = optimizer_continue.optimize(
        strategy=strategy,
        data=data,
        metric='sharpe_ratio',
        study_name='simple_ma_optimization',
        storage='sqlite:///optuna_study.db',  # 同一個資料庫
        show_progress_bar=True
    )

    print(f"\n現在共有 {result4.n_trials} 次試驗")
    print(result4.summary())


if __name__ == '__main__':
    main()
