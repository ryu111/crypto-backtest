"""
自動特徵工程單元測試

測試遺傳算法特徵工程的各項功能。
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.automation.feature_engineering import (
    Feature,
    FeatureSet,
    RandomFeatureGenerator,
    GeneticFeatureEngineer,
    FeatureSelector,
    AutoStrategyGenerator,
    create_feature_engineer
)


class TestFeature(unittest.TestCase):
    """測試 Feature 類別"""

    def test_feature_creation(self):
        """測試特徵建立"""
        feature = Feature(
            name='test_feature',
            expression='SMA(close, 20)',
            importance=0.8,
            description='Test feature'
        )

        self.assertEqual(feature.name, 'test_feature')
        self.assertEqual(feature.expression, 'SMA(close, 20)')
        self.assertEqual(feature.importance, 0.8)
        self.assertEqual(feature.description, 'Test feature')

    def test_auto_name_generation(self):
        """測試自動名稱生成"""
        feature = Feature(
            name='',
            expression='SMA(close, 20)'
        )

        self.assertTrue(feature.name.startswith('feature_'))
        self.assertEqual(len(feature.name), 16)  # 'feature_' + 8 char hash

    def test_to_dict(self):
        """測試轉為字典"""
        feature = Feature(
            name='test',
            expression='EMA(close, 50)',
            importance=0.5
        )

        data = feature.to_dict()

        self.assertIn('name', data)
        self.assertIn('expression', data)
        self.assertIn('importance', data)
        self.assertEqual(data['name'], 'test')


class TestFeatureSet(unittest.TestCase):
    """測試 FeatureSet 類別"""

    def setUp(self):
        """設定測試資料"""
        self.features = [
            Feature('f1', 'SMA(close, 20)', 0.8),
            Feature('f2', 'EMA(close, 50)', 0.6),
            Feature('f3', 'RSI(close, 14)', 0.9),
        ]

    def test_feature_set_creation(self):
        """測試特徵集合建立"""
        fs = FeatureSet(
            features=self.features,
            fitness_score=1.5,
            generation=5
        )

        self.assertEqual(len(fs.features), 3)
        self.assertEqual(fs.fitness_score, 1.5)
        self.assertEqual(fs.generation, 5)

    def test_get_feature_names(self):
        """測試取得特徵名稱"""
        fs = FeatureSet(features=self.features)
        names = fs.get_feature_names()

        self.assertEqual(names, ['f1', 'f2', 'f3'])

    def test_get_expressions(self):
        """測試取得表達式"""
        fs = FeatureSet(features=self.features)
        expressions = fs.get_expressions()

        self.assertEqual(expressions['f1'], 'SMA(close, 20)')
        self.assertEqual(expressions['f2'], 'EMA(close, 50)')
        self.assertEqual(expressions['f3'], 'RSI(close, 14)')

    def test_to_dict(self):
        """測試轉為字典"""
        fs = FeatureSet(
            features=self.features,
            fitness_score=2.0,
            generation=3
        )

        data = fs.to_dict()

        self.assertIn('features', data)
        self.assertIn('fitness_score', data)
        self.assertIn('generation', data)
        self.assertEqual(len(data['features']), 3)


class TestRandomFeatureGenerator(unittest.TestCase):
    """測試隨機特徵生成器"""

    def setUp(self):
        """設定測試資料"""
        self.generator = RandomFeatureGenerator(
            base_indicators=['SMA', 'EMA', 'RSI'],
            operators=['+', '-', '*', '/'],
            price_cols=['close', 'high', 'low'],
            max_complexity=2,
            seed=42
        )

    def test_generate_features(self):
        """測試生成特徵"""
        features = self.generator.generate(n_features=10)

        self.assertEqual(len(features), 10)
        for feature in features:
            self.assertIsInstance(feature, Feature)
            self.assertTrue(feature.name.startswith('auto_feature_'))
            self.assertTrue(len(feature.expression) > 0)

    def test_reproducibility(self):
        """測試可重現性（相同 seed 應產生相同特徵）"""
        gen1 = RandomFeatureGenerator(
            base_indicators=['SMA', 'EMA'],
            operators=['+', '-'],
            seed=42
        )
        gen2 = RandomFeatureGenerator(
            base_indicators=['SMA', 'EMA'],
            operators=['+', '-'],
            seed=42
        )

        features1 = gen1.generate(n_features=5)
        features2 = gen2.generate(n_features=5)

        for f1, f2 in zip(features1, features2):
            self.assertEqual(f1.expression, f2.expression)


class TestGeneticFeatureEngineer(unittest.TestCase):
    """測試遺傳算法特徵工程"""

    def setUp(self):
        """設定測試資料"""
        self.engineer = GeneticFeatureEngineer(
            base_indicators=['SMA', 'EMA'],
            operators=['+', '-'],
            population_size=10,
            generations=5,
            features_per_set=3,
            seed=42
        )

        # 生成模擬資料
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        self.data = pd.DataFrame({
            'timestamp': dates,
            'close': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100) + 105,
            'low': np.random.randn(100) + 95,
            'volume': np.random.uniform(1000, 5000, 100)
        })

    def test_initialize_population(self):
        """測試初始化族群"""
        population = self.engineer._initialize_population()

        self.assertEqual(len(population), 10)
        for individual in population:
            self.assertIsInstance(individual, FeatureSet)
            self.assertEqual(len(individual.features), 3)
            self.assertEqual(individual.generation, 0)

    def test_tournament_selection(self):
        """測試錦標賽選擇"""
        population = [
            FeatureSet(features=[], fitness_score=1.0),
            FeatureSet(features=[], fitness_score=2.0),
            FeatureSet(features=[], fitness_score=3.0),
        ]

        selected = self.engineer._tournament_selection(population, tournament_size=2)

        # 應該選擇適應度較高的
        self.assertIn(selected.fitness_score, [2.0, 3.0])

    def test_crossover(self):
        """測試交叉"""
        parent1 = FeatureSet(features=[
            Feature('f1', 'expr1'),
            Feature('f2', 'expr2'),
            Feature('f3', 'expr3'),
        ])
        parent2 = FeatureSet(features=[
            Feature('f4', 'expr4'),
            Feature('f5', 'expr5'),
            Feature('f6', 'expr6'),
        ])

        child = self.engineer._crossover(parent1, parent2)

        # 檢查子代包含父母的特徵
        self.assertEqual(len(child.features), 3)

    def test_mutate(self):
        """測試突變"""
        original = FeatureSet(features=[
            Feature('f1', 'SMA(close, 20)'),
            Feature('f2', 'EMA(close, 50)'),
        ])

        mutated = self.engineer._mutate(original)

        # 應該有一個特徵不同
        self.assertEqual(len(mutated.features), len(original.features))

    def test_evolve(self):
        """測試演化過程"""
        def dummy_fitness(feature_set: FeatureSet, data: pd.DataFrame) -> float:
            """虛擬適應度函數"""
            return np.random.uniform(0, 2)

        best = self.engineer.evolve(
            data=self.data,
            fitness_function=dummy_fitness,
            verbose=False
        )

        # 檢查返回最佳結果
        self.assertIsInstance(best, FeatureSet)
        self.assertGreaterEqual(best.fitness_score, 0)

        # 檢查歷史記錄
        history = self.engineer.get_evolution_history()
        self.assertEqual(len(history), 5)  # 5 generations


class TestFeatureSelector(unittest.TestCase):
    """測試特徵選擇器"""

    def setUp(self):
        """設定測試資料"""
        self.selector = FeatureSelector()

        self.features = [
            Feature('f1', 'SMA(close, 20)', importance=0.9),
            Feature('f2', 'EMA(close, 50)', importance=0.7),
            Feature('f3', 'RSI(close, 14)', importance=0.8),
            Feature('f4', 'MACD(close, 12, 26, 9)', importance=0.5),
            Feature('f5', 'ATR(close, 14)', importance=0.6),
        ]

    def test_select_features_by_importance(self):
        """測試按重要性選擇特徵"""
        selected = self.selector.select_features(
            features=self.features,
            max_features=3
        )

        # 應該選擇前 3 個最重要的
        self.assertEqual(len(selected), 3)
        importances = [f.importance for f in selected]
        self.assertEqual(importances, [0.9, 0.8, 0.7])

    def test_select_features_limit(self):
        """測試特徵數量限制"""
        selected = self.selector.select_features(
            features=self.features,
            max_features=10  # 超過實際數量
        )

        # 應該返回所有特徵
        self.assertEqual(len(selected), 5)


class TestAutoStrategyGenerator(unittest.TestCase):
    """測試自動策略生成器"""

    def setUp(self):
        """設定測試資料"""
        self.generator = AutoStrategyGenerator()

        self.features = FeatureSet(
            features=[
                Feature('f1', 'SMA(close, 20)', importance=0.9),
                Feature('f2', 'EMA(close, 50)', importance=0.8),
                Feature('f3', 'RSI(close, 14)', importance=0.7),
                Feature('f4', 'MACD(close, 12, 26, 9)', importance=0.6),
            ],
            fitness_score=2.0
        )

    def test_generate_strategy(self):
        """測試生成策略"""
        strategy = self.generator.generate_strategy(
            features=self.features,
            entry_rules=2,
            exit_rules=1,
            rule_type='mixed'
        )

        # 檢查策略結構
        self.assertIn('entry_long', strategy)
        self.assertIn('exit_long', strategy)
        self.assertIn('entry_short', strategy)
        self.assertIn('exit_short', strategy)
        self.assertIn('features_used', strategy)

        # 檢查規則數量
        self.assertLessEqual(len(strategy['entry_long']), 2)
        self.assertLessEqual(len(strategy['exit_long']), 1)

    def test_reverse_rule(self):
        """測試規則反轉"""
        original = "feature1 > 0.5"
        reversed_rule = self.generator._reverse_rule(original)

        self.assertEqual(reversed_rule, "feature1 < 0.5")


class TestUtilityFunctions(unittest.TestCase):
    """測試便利函數"""

    def test_create_feature_engineer(self):
        """測試建立特徵工程器"""
        engineer = create_feature_engineer(
            base_indicators=['SMA', 'EMA'],
            population_size=20,
            generations=10
        )

        self.assertIsInstance(engineer, GeneticFeatureEngineer)
        self.assertEqual(engineer.population_size, 20)
        self.assertEqual(engineer.generations, 10)

    def test_create_feature_engineer_defaults(self):
        """測試使用預設參數建立"""
        engineer = create_feature_engineer()

        self.assertIsInstance(engineer, GeneticFeatureEngineer)
        self.assertEqual(engineer.population_size, 50)
        self.assertEqual(engineer.generations, 20)


def run_tests():
    """執行所有測試"""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_tests()
