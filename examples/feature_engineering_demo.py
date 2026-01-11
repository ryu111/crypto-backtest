"""
自動特徵工程 Demo

展示如何使用遺傳算法自動生成和優化技術指標組合。

執行方式:
    python examples/feature_engineering_demo.py
"""

import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.automation.feature_engineering import (
    Feature,
    FeatureSet,
    RandomFeatureGenerator,
    GeneticFeatureEngineer,
    FeatureSelector,
    AutoStrategyGenerator,
    create_feature_engineer,
    quick_feature_evolution
)


# ============================================================================
# 1. 生成模擬市場資料
# ============================================================================

def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """生成模擬市場資料"""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=n_samples),
        periods=n_samples,
        freq='1h'
    )

    # 生成隨機價格（帶趨勢）
    trend = np.linspace(40000, 50000, n_samples)
    noise = np.random.randn(n_samples) * 500
    close = trend + noise

    data = pd.DataFrame({
        'timestamp': dates,
        'open': close + np.random.randn(n_samples) * 100,
        'high': close + np.abs(np.random.randn(n_samples)) * 200,
        'low': close - np.abs(np.random.randn(n_samples)) * 200,
        'close': close,
        'volume': np.random.uniform(1000, 5000, n_samples)
    })

    # 確保 high >= close >= low
    data['high'] = data[['high', 'close']].max(axis=1)
    data['low'] = data[['low', 'close']].min(axis=1)

    return data


# ============================================================================
# 2. Demo 1: 隨機特徵生成
# ============================================================================

def demo_random_features():
    """Demo: 隨機特徵生成"""
    print("\n" + "="*60)
    print("Demo 1: 隨機特徵生成")
    print("="*60)

    generator = RandomFeatureGenerator(
        base_indicators=['SMA', 'EMA', 'RSI', 'MACD', 'ATR'],
        operators=['+', '-', '*', '/'],
        price_cols=['close', 'high', 'low', 'volume'],
        max_complexity=2,
        seed=42
    )

    features = generator.generate(n_features=10)

    print(f"\n生成 {len(features)} 個特徵:\n")
    for i, feature in enumerate(features, 1):
        print(f"{i:2d}. {feature.name:20s} = {feature.expression}")


# ============================================================================
# 3. Demo 2: 遺傳算法特徵演化
# ============================================================================

def demo_genetic_evolution():
    """Demo: 遺傳算法特徵演化"""
    print("\n" + "="*60)
    print("Demo 2: 遺傳算法特徵演化")
    print("="*60)

    # 生成資料
    data = generate_sample_data(n_samples=500)

    # 定義適應度函數（簡化版：隨機評分）
    def simple_fitness_function(feature_set: FeatureSet, data: pd.DataFrame) -> float:
        """
        簡化的適應度函數
        實際應用中應該是回測 Sharpe Ratio
        """
        # 這裡用隨機分數模擬
        # 實際應該用回測引擎評估策略績效
        base_score = np.random.uniform(0.5, 2.0)

        # 給予較少特徵的獎勵（避免過擬合）
        complexity_penalty = len(feature_set.features) * 0.01

        return base_score - complexity_penalty

    # 建立遺傳工程器
    engineer = GeneticFeatureEngineer(
        base_indicators=['SMA', 'EMA', 'RSI', 'MACD'],
        operators=['+', '-', '*', '/'],
        population_size=20,
        generations=10,
        mutation_rate=0.2,
        crossover_rate=0.7,
        features_per_set=5,
        seed=42
    )

    # 執行演化
    best_features = engineer.evolve(
        data=data,
        fitness_function=simple_fitness_function,
        verbose=True
    )

    print(f"\n最佳特徵集合 (Fitness: {best_features.fitness_score:.4f}):\n")
    for i, feature in enumerate(best_features.features, 1):
        print(f"{i}. {feature.name:20s} = {feature.expression}")

    # 演化歷史
    history_df = engineer.get_evolution_history()
    print("\n演化歷史:")
    print(history_df[['generation', 'best_fitness', 'avg_fitness']])


# ============================================================================
# 4. Demo 3: 特徵選擇
# ============================================================================

def demo_feature_selection():
    """Demo: 特徵選擇"""
    print("\n" + "="*60)
    print("Demo 3: 特徵選擇")
    print("="*60)

    # 建立特徵集合（帶重要性）
    features = [
        Feature(name='feature_1', expression='SMA(close, 20)', importance=0.8),
        Feature(name='feature_2', expression='EMA(close, 50)', importance=0.6),
        Feature(name='feature_3', expression='RSI(close, 14)', importance=0.9),
        Feature(name='feature_4', expression='MACD(close, 12, 26, 9)', importance=0.5),
        Feature(name='feature_5', expression='ATR(close, 14)', importance=0.7),
        Feature(name='feature_6', expression='SMA(volume, 20)', importance=0.3),
        Feature(name='feature_7', expression='BB(close, 20, 2.0)', importance=0.4),
    ]

    selector = FeatureSelector()

    # 選擇前 5 個特徵
    selected = selector.select_features(
        features=features,
        max_features=5,
        correlation_threshold=0.8
    )

    print("\n原始特徵:")
    for f in features:
        print(f"  {f.name}: {f.expression:30s} (importance: {f.importance:.2f})")

    print("\n選擇的前 5 個特徵:")
    for f in selected:
        print(f"  {f.name}: {f.expression:30s} (importance: {f.importance:.2f})")


# ============================================================================
# 5. Demo 4: 自動策略生成
# ============================================================================

def demo_auto_strategy_generation():
    """Demo: 自動策略生成"""
    print("\n" + "="*60)
    print("Demo 4: 自動策略生成")
    print("="*60)

    # 建立特徵集合
    features = FeatureSet(
        features=[
            Feature(name='trend_strength', expression='SMA(close, 20) - SMA(close, 50)', importance=0.9),
            Feature(name='momentum', expression='RSI(close, 14)', importance=0.8),
            Feature(name='volatility', expression='ATR(close, 14)', importance=0.7),
            Feature(name='volume_ratio', expression='volume / SMA(volume, 20)', importance=0.6),
            Feature(name='price_position', expression='(close - low) / (high - low)', importance=0.5),
        ],
        fitness_score=1.85
    )

    # 生成策略
    generator = AutoStrategyGenerator()

    strategy = generator.generate_strategy(
        features=features,
        entry_rules=3,
        exit_rules=2,
        rule_type='mixed'
    )

    print("\n生成的策略規則:\n")
    print("多單進場條件:")
    for rule in strategy['entry_long']:
        print(f"  - {rule}")

    print("\n多單出場條件:")
    for rule in strategy['exit_long']:
        print(f"  - {rule}")

    print("\n空單進場條件:")
    for rule in strategy['entry_short']:
        print(f"  - {rule}")

    print("\n空單出場條件:")
    for rule in strategy['exit_short']:
        print(f"  - {rule}")

    print("\n使用的特徵:")
    for name in strategy['features_used']:
        expr = strategy['feature_expressions'][name]
        print(f"  - {name}: {expr}")


# ============================================================================
# 6. Demo 5: 快速特徵演化（便利函數）
# ============================================================================

def demo_quick_evolution():
    """Demo: 快速特徵演化"""
    print("\n" + "="*60)
    print("Demo 5: 快速特徵演化（多次試驗）")
    print("="*60)

    data = generate_sample_data(n_samples=500)

    def fitness_func(feature_set: FeatureSet, data: pd.DataFrame) -> float:
        """簡單適應度函數"""
        return np.random.uniform(0.5, 2.5)

    # 執行 3 次試驗，取最佳結果
    best_result = quick_feature_evolution(
        data=data,
        fitness_function=fitness_func,
        n_trials=3,
        verbose=True
    )

    print(f"\n最終最佳結果 (Fitness: {best_result.fitness_score:.4f}):")
    for feature in best_result.features:
        print(f"  - {feature.name}: {feature.expression}")


# ============================================================================
# Main
# ============================================================================

def main():
    """執行所有 Demo"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*16 + "自動特徵工程 Demo" + " "*22 + "║")
    print("╚" + "="*58 + "╝")

    demo_random_features()
    demo_genetic_evolution()
    demo_feature_selection()
    demo_auto_strategy_generation()
    demo_quick_evolution()

    print("\n" + "="*60)
    print("所有 Demo 執行完成！")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
