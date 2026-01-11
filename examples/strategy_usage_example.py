"""
策略使用範例

展示如何使用策略基礎架構建立、註冊和使用交易策略。
"""

import sys
from pathlib import Path

# 將專案根目錄加入 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.strategies import (
    BaseStrategy,
    TrendStrategy,
    MomentumStrategy,
    register_strategy,
    list_strategies,
    create_strategy
)


# ============================================================
# 範例 1: 簡單趨勢策略
# ============================================================

@register_strategy('simple_ma_cross')
class SimpleMACross(TrendStrategy):
    """簡單均線交叉策略"""

    params = {
        'fast_period': 10,
        'slow_period': 30,
    }

    param_space = {
        'fast_period': {'type': 'int', 'low': 5, 'high': 20},
        'slow_period': {'type': 'int', 'low': 20, 'high': 100},
    }

    version = "1.0"
    description = "Simple moving average crossover"

    def calculate_indicators(self, data):
        close = data['close']
        return {
            'sma_fast': close.rolling(self.params['fast_period']).mean(),
            'sma_slow': close.rolling(self.params['slow_period']).mean()
        }

    def generate_signals(self, data):
        indicators = self.calculate_indicators(data)
        fast = indicators['sma_fast']
        slow = indicators['sma_slow']

        long_entry = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        long_exit = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        short_entry = long_exit.copy()
        short_exit = long_entry.copy()

        return long_entry, long_exit, short_entry, short_exit


# ============================================================
# 範例 2: RSI 動量策略
# ============================================================

@register_strategy('rsi_momentum')
class RSIMomentum(MomentumStrategy):
    """RSI 超買超賣策略"""

    params = {
        'rsi_period': 14,
        'oversold': 30,
        'overbought': 70,
    }

    param_space = {
        'rsi_period': {'type': 'int', 'low': 7, 'high': 21},
        'oversold': {'type': 'int', 'low': 20, 'high': 40},
        'overbought': {'type': 'int', 'low': 60, 'high': 80},
    }

    version = "1.0"
    description = "RSI overbought/oversold strategy"

    def calculate_indicators(self, data):
        rsi = self.calculate_rsi(data['close'], self.params['rsi_period'])
        return {'rsi': rsi}

    def generate_signals(self, data):
        indicators = self.calculate_indicators(data)
        rsi = indicators['rsi']

        long_entry = rsi < self.params['oversold']
        long_exit = rsi > 50
        short_entry = rsi > self.params['overbought']
        short_exit = rsi < 50

        return long_entry, long_exit, short_entry, short_exit


# ============================================================
# 範例 3: 帶過濾器的策略
# ============================================================

@register_strategy('filtered_breakout')
class FilteredBreakout(BaseStrategy):
    """帶趨勢過濾的突破策略"""

    strategy_type = "momentum"

    params = {
        'bb_period': 20,
        'bb_std': 2.0,
        'trend_period': 200,
        'volume_mult': 1.5,
    }

    param_space = {
        'bb_period': {'type': 'int', 'low': 10, 'high': 30},
        'bb_std': {'type': 'float', 'low': 1.5, 'high': 3.0},
        'volume_mult': {'type': 'float', 'low': 1.0, 'high': 2.0},
    }

    version = "1.0"
    description = "Bollinger Band breakout with trend filter"

    def calculate_indicators(self, data):
        close = data['close']
        volume = data['volume']

        # 布林帶
        ma = close.rolling(self.params['bb_period']).mean()
        std = close.rolling(self.params['bb_period']).std()
        upper = ma + self.params['bb_std'] * std
        lower = ma - self.params['bb_std'] * std

        # 趨勢過濾
        trend_ma = close.rolling(self.params['trend_period']).mean()

        # 成交量
        vol_ma = volume.rolling(self.params['bb_period']).mean()

        return {
            'bb_upper': upper,
            'bb_lower': lower,
            'bb_middle': ma,
            'trend_ma': trend_ma,
            'volume_ma': vol_ma
        }

    def generate_signals(self, data):
        indicators = self.calculate_indicators(data)
        close = data['close']
        volume = data['volume']

        # 趨勢過濾
        uptrend = close > indicators['trend_ma']
        downtrend = close < indicators['trend_ma']

        # 成交量確認
        high_volume = volume > indicators['volume_ma'] * self.params['volume_mult']

        # 突破訊號（只在趨勢方向交易）
        long_entry = (
            (close > indicators['bb_upper']) &
            high_volume &
            uptrend
        )
        long_exit = close < indicators['bb_middle']

        short_entry = (
            (close < indicators['bb_lower']) &
            high_volume &
            downtrend
        )
        short_exit = close > indicators['bb_middle']

        return long_entry, long_exit, short_entry, short_exit


# ============================================================
# 使用範例
# ============================================================

def demo_basic_usage():
    """基本使用示範"""
    print("=== 基本使用 ===\n")

    # 建立模擬資料
    np.random.seed(42)
    data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })

    # 建立策略
    strategy = create_strategy('simple_ma_cross', fast_period=12, slow_period=26)

    # 產生訊號
    long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(data)

    print(f"策略: {strategy}")
    print(f"多單進場: {long_entry.sum()} 次")
    print(f"多單出場: {long_exit.sum()} 次\n")


def demo_list_strategies():
    """列出所有策略"""
    print("=== 已註冊策略 ===\n")

    from src.strategies import StrategyRegistry

    strategies = list_strategies()
    for name in strategies:
        info = StrategyRegistry.get_info(name)
        print(f"{name}:")
        print(f"  類型: {info['type']}")
        print(f"  版本: {info['version']}")
        print(f"  描述: {info['description']}")
        print()


def demo_param_optimization():
    """參數優化範例"""
    print("=== 參數優化空間 ===\n")

    from src.strategies import StrategyRegistry

    for strategy_name in list_strategies():
        param_space = StrategyRegistry.get_param_space(strategy_name)
        if param_space:
            print(f"{strategy_name}:")
            for param, config in param_space.items():
                print(f"  {param}: {config}")
            print()


def demo_multiple_strategies():
    """比較多個策略"""
    print("=== 策略比較 ===\n")

    # 建立資料
    np.random.seed(42)
    data = pd.DataFrame({
        'open': np.random.randn(200).cumsum() + 100,
        'high': np.random.randn(200).cumsum() + 101,
        'low': np.random.randn(200).cumsum() + 99,
        'close': np.random.randn(200).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 200)
    })

    # 測試多個策略
    strategies = ['simple_ma_cross', 'rsi_momentum', 'filtered_breakout']

    for strategy_name in strategies:
        strategy = create_strategy(strategy_name)
        long_entry, _, _, _ = strategy.generate_signals(data)
        print(f"{strategy_name}: {long_entry.sum()} 次進場")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("策略使用範例")
    print("="*50 + "\n")

    demo_basic_usage()
    demo_list_strategies()
    demo_param_optimization()
    demo_multiple_strategies()

    print("="*50)
    print("✅ 範例執行完成")
    print("="*50 + "\n")
