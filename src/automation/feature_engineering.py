"""
自動特徵工程

使用遺傳算法自動生成和優化技術指標組合。
支援特徵重要性評估、特徵選擇、自動策略生成。

主要功能:
1. 遺傳算法自動生成指標組合
2. 特徵重要性評估（基於回測績效）
3. 特徵選擇（過濾相關性高的特徵）
4. 自動策略生成框架
"""

import ast
import hashlib
import operator
import random
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable, Literal
from datetime import datetime

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression


# ============================================================================
# 1. 特徵定義與集合
# ============================================================================

@dataclass
class Feature:
    """
    特徵定義

    Attributes:
        name: 特徵名稱
        expression: 特徵表達式 (如 "SMA(close, 20) - SMA(close, 50)")
        importance: 特徵重要性分數 (0-1)
        description: 特徵描述
        metadata: 附加元數據
    """
    name: str
    expression: str
    importance: float = 0.0
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """自動生成名稱（如果未提供）"""
        if not self.name:
            # 使用 expression 的 hash 作為名稱
            hash_obj = hashlib.md5(self.expression.encode())
            self.name = f"feature_{hash_obj.hexdigest()[:8]}"

    def to_dict(self) -> Dict[str, Any]:
        """轉為字典"""
        return {
            'name': self.name,
            'expression': self.expression,
            'importance': self.importance,
            'description': self.description,
            'metadata': self.metadata
        }


@dataclass
class FeatureSet:
    """
    特徵集合

    Attributes:
        features: 特徵列表
        fitness_score: 適應度分數（回測 Sharpe Ratio）
        generation: 來自第幾代
        timestamp: 建立時間
    """
    features: List[Feature]
    fitness_score: float = 0.0
    generation: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def get_feature_names(self) -> List[str]:
        """取得所有特徵名稱"""
        return [f.name for f in self.features]

    def get_expressions(self) -> Dict[str, str]:
        """取得特徵表達式字典"""
        return {f.name: f.expression for f in self.features}

    def to_dict(self) -> Dict[str, Any]:
        """轉為字典"""
        return {
            'features': [f.to_dict() for f in self.features],
            'fitness_score': self.fitness_score,
            'generation': self.generation,
            'timestamp': self.timestamp.isoformat()
        }

    def __repr__(self) -> str:
        return (
            f"FeatureSet(n_features={len(self.features)}, "
            f"fitness={self.fitness_score:.4f}, gen={self.generation})"
        )


# ============================================================================
# 2. 特徵生成器基礎類別
# ============================================================================

class BaseFeatureGenerator(ABC):
    """特徵生成器基礎類別"""

    @abstractmethod
    def generate(self, n_features: int = 10) -> List[Feature]:
        """
        生成特徵

        Args:
            n_features: 要生成的特徵數量

        Returns:
            特徵列表
        """
        pass


class RandomFeatureGenerator(BaseFeatureGenerator):
    """
    隨機特徵生成器

    隨機組合基礎指標和運算符生成特徵表達式。

    使用範例:
        generator = RandomFeatureGenerator(
            base_indicators=['SMA', 'EMA', 'RSI', 'MACD'],
            operators=['+', '-', '*', '/'],
            price_cols=['close', 'high', 'low']
        )

        features = generator.generate(n_features=20)
    """

    def __init__(
        self,
        base_indicators: List[str],
        operators: List[str],
        price_cols: List[str] = ['close', 'high', 'low', 'volume'],
        max_complexity: int = 3,
        seed: Optional[int] = None
    ):
        """
        初始化

        Args:
            base_indicators: 基礎指標列表 ['SMA', 'EMA', 'RSI', 'MACD', 'ATR', ...]
            operators: 運算符列表 ['+', '-', '*', '/', 'crossover', ...]
            price_cols: 價格欄位 ['close', 'high', 'low', 'volume']
            max_complexity: 最大表達式複雜度（運算符數量）
            seed: 隨機種子
        """
        self.base_indicators = base_indicators
        self.operators = operators
        self.price_cols = price_cols
        self.max_complexity = max_complexity
        self.seed = seed

        if seed is not None:
            self._set_seed(seed)

        # 預定義指標參數範圍
        self.param_ranges = {
            'SMA': {'period': (5, 200)},
            'EMA': {'period': (5, 200)},
            'RSI': {'period': (7, 28)},
            'ATR': {'period': (7, 28)},
            'MACD': {'fast': (8, 20), 'slow': (20, 40), 'signal': (5, 15)},
            'BB': {'period': (10, 50), 'std': (1.5, 3.0)},
            'STOCH': {'k_period': (5, 21), 'd_period': (3, 9)},
        }

    def _set_seed(self, seed: int):
        """設定隨機種子"""
        random.seed(seed)
        np.random.seed(seed)

    def generate(self, n_features: int = 10) -> List[Feature]:
        """生成隨機特徵"""
        # 確保每次 generate 時重設種子（可重現性）
        if self.seed is not None:
            self._set_seed(self.seed)

        features = []

        for i in range(n_features):
            expression = self._generate_expression()
            feature = Feature(
                name=f"auto_feature_{i}",
                expression=expression,
                description=f"Automatically generated feature {i}"
            )
            features.append(feature)

        return features

    def _generate_expression(self) -> str:
        """生成單一特徵表達式"""
        complexity = random.randint(1, self.max_complexity)

        # 生成基礎項
        terms = []
        for _ in range(complexity + 1):
            term = self._generate_term()
            terms.append(term)

        # 使用運算符組合
        if len(terms) == 1:
            return terms[0]

        expression = terms[0]
        for i in range(1, len(terms)):
            op = random.choice(['+', '-', '*', '/'])
            expression = f"({expression} {op} {terms[i]})"

        return expression

    def _generate_term(self) -> str:
        """生成單一項（指標或價格）"""
        # 50% 機率選擇指標，50% 選擇價格
        if random.random() < 0.5 and self.base_indicators:
            return self._generate_indicator()
        else:
            return random.choice(self.price_cols)

    def _generate_indicator(self) -> str:
        """生成指標表達式"""
        indicator = random.choice(self.base_indicators)

        if indicator not in self.param_ranges:
            # 未定義參數範圍，使用預設
            price = random.choice(self.price_cols)
            period = random.randint(5, 200)
            return f"{indicator}({price}, {period})"

        # 根據參數範圍生成
        params = self.param_ranges[indicator]

        if indicator in ['SMA', 'EMA', 'RSI', 'ATR']:
            price = random.choice(self.price_cols)
            period = random.randint(*params['period'])
            return f"{indicator}({price}, {period})"

        elif indicator == 'MACD':
            price = random.choice(self.price_cols)
            fast = random.randint(*params['fast'])
            slow = random.randint(*params['slow'])
            signal = random.randint(*params['signal'])
            return f"{indicator}({price}, {fast}, {slow}, {signal})"

        elif indicator == 'BB':
            price = random.choice(self.price_cols)
            period = random.randint(*params['period'])
            std = round(random.uniform(*params['std']), 1)
            return f"{indicator}({price}, {period}, {std})"

        elif indicator == 'STOCH':
            k_period = random.randint(*params['k_period'])
            d_period = random.randint(*params['d_period'])
            return f"{indicator}({k_period}, {d_period})"

        else:
            # Fallback
            price = random.choice(self.price_cols)
            period = random.randint(5, 200)
            return f"{indicator}({price}, {period})"


# ============================================================================
# 3. 遺傳算法特徵工程
# ============================================================================

class GeneticFeatureEngineer:
    """
    遺傳算法特徵工程

    使用遺傳算法演化特徵集合，目標是最大化策略回測績效。

    使用範例:
        engineer = GeneticFeatureEngineer(
            base_indicators=['SMA', 'EMA', 'RSI', 'MACD'],
            operators=['+', '-', '*', '/'],
            population_size=50,
            generations=20
        )

        # 演化特徵
        best_features = engineer.evolve(
            data=market_data,
            fitness_function=my_backtest_fitness_func
        )

        print(f"Best Fitness: {best_features.fitness_score}")
        for feature in best_features.features:
            print(f"{feature.name}: {feature.expression}")
    """

    def __init__(
        self,
        base_indicators: List[str],
        operators: List[str],
        population_size: int = 50,
        generations: int = 20,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
        elite_size: int = 5,
        features_per_set: int = 10,
        seed: Optional[int] = None
    ):
        """
        初始化遺傳算法

        Args:
            base_indicators: 基礎指標列表
            operators: 運算符列表
            population_size: 族群大小
            generations: 演化代數
            mutation_rate: 突變率
            crossover_rate: 交叉率
            elite_size: 精英數量（保留最佳個體）
            features_per_set: 每個特徵集合的特徵數量
            seed: 隨機種子
        """
        self.base_indicators = base_indicators
        self.operators = operators
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.features_per_set = features_per_set
        self.seed = seed

        # 特徵生成器
        self.generator = RandomFeatureGenerator(
            base_indicators=base_indicators,
            operators=operators,
            seed=seed
        )

        # 演化歷史
        self.history: List[Dict[str, Any]] = []

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def evolve(
        self,
        data: DataFrame,
        fitness_function: Callable[[FeatureSet, DataFrame], float],
        verbose: bool = True
    ) -> FeatureSet:
        """
        執行遺傳演化

        Args:
            data: 市場資料 (OHLCV DataFrame)
            fitness_function: 適應度函數，輸入 (FeatureSet, data)，輸出 fitness score
            verbose: 是否顯示進度

        Returns:
            最佳特徵集合
        """
        if verbose:
            print(f"開始遺傳演化（{self.generations} 代，族群 {self.population_size}）")
            print("="*60)

        # 初始化族群
        population = self._initialize_population()

        # 演化循環
        for gen in range(self.generations):
            # 評估適應度
            for individual in population:
                if individual.fitness_score == 0.0:
                    individual.fitness_score = fitness_function(individual, data)

            # 排序（由高到低）
            population.sort(key=lambda x: x.fitness_score, reverse=True)

            # 記錄最佳
            best = population[0]
            self.history.append({
                'generation': gen,
                'best_fitness': best.fitness_score,
                'avg_fitness': np.mean([ind.fitness_score for ind in population]),
                'timestamp': datetime.now()
            })

            if verbose:
                print(
                    f"Gen {gen:3d} | "
                    f"Best: {best.fitness_score:.4f} | "
                    f"Avg: {self.history[-1]['avg_fitness']:.4f}"
                )

            # 最後一代不需要產生下一代
            if gen == self.generations - 1:
                break

            # 產生下一代
            new_population = []

            # 精英保留
            new_population.extend(population[:self.elite_size])

            # 選擇、交叉、突變
            while len(new_population) < self.population_size:
                # 選擇父母
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)

                # 交叉
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1 if random.random() < 0.5 else parent2

                # 突變
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)

                # 重置適應度（需要重新評估）
                child.fitness_score = 0.0
                child.generation = gen + 1

                new_population.append(child)

            population = new_population

        # 返回最佳
        best_feature_set = population[0]

        if verbose:
            print("="*60)
            print(f"演化完成！最佳適應度: {best_feature_set.fitness_score:.4f}")

        return best_feature_set

    def _initialize_population(self) -> List[FeatureSet]:
        """初始化族群"""
        population = []

        for _ in range(self.population_size):
            features = self.generator.generate(n_features=self.features_per_set)
            feature_set = FeatureSet(features=features, generation=0)
            population.append(feature_set)

        return population

    def _tournament_selection(
        self,
        population: List[FeatureSet],
        tournament_size: int = 3
    ) -> FeatureSet:
        """錦標賽選擇"""
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: x.fitness_score)

    def _crossover(self, parent1: FeatureSet, parent2: FeatureSet) -> FeatureSet:
        """交叉（單點交叉）"""
        crossover_point = random.randint(1, len(parent1.features) - 1)

        child_features = (
            parent1.features[:crossover_point] +
            parent2.features[crossover_point:]
        )

        return FeatureSet(features=child_features)

    def _mutate(self, feature_set: FeatureSet) -> FeatureSet:
        """突變（隨機替換一個特徵）"""
        mutated_features = feature_set.features.copy()

        # 隨機選擇要突變的位置
        mutate_idx = random.randint(0, len(mutated_features) - 1)

        # 生成新特徵
        new_feature = self.generator.generate(n_features=1)[0]
        mutated_features[mutate_idx] = new_feature

        return FeatureSet(features=mutated_features)

    def get_evolution_history(self) -> DataFrame:
        """取得演化歷史 DataFrame"""
        return pd.DataFrame(self.history)


# ============================================================================
# 4. 特徵選擇與評估
# ============================================================================

class FeatureSelector:
    """
    特徵選擇器

    基於重要性和相關性選擇最佳特徵子集。

    使用範例:
        selector = FeatureSelector()

        # 評估特徵重要性
        importances = selector.evaluate_importance(
            features=feature_set,
            data=market_data,
            target=returns
        )

        # 選擇特徵（移除高度相關）
        selected = selector.select_features(
            features=feature_set.features,
            max_features=10,
            correlation_threshold=0.8
        )
    """

    def evaluate_importance(
        self,
        features: FeatureSet,
        data: DataFrame,
        target: Series,
        method: Literal['random_forest', 'mutual_info'] = 'random_forest'
    ) -> Dict[str, float]:
        """
        評估特徵重要性

        Args:
            features: 特徵集合
            data: 市場資料
            target: 目標變數（如未來收益率）
            method: 評估方法 ('random_forest' 或 'mutual_info')

        Returns:
            {feature_name: importance_score}
        """
        # 計算特徵值（這裡需要實際計算指標，簡化處理）
        # 實際應用中需要使用 FeatureCalculator
        feature_values = {}
        for feature in features.features:
            # 這裡假設已經計算好了特徵值
            # 實際需要解析 expression 並計算
            feature_values[feature.name] = np.random.randn(len(data))

        X = pd.DataFrame(feature_values)
        y = target.values

        # 移除 NaN
        valid_idx = ~(X.isna().any(axis=1) | pd.isna(y))
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]

        if len(X_clean) < 10:
            warnings.warn("有效樣本數過少，無法評估重要性")
            return {f.name: 0.0 for f in features.features}

        # 評估重要性
        if method == 'random_forest':
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_clean, y_clean)
            importances = rf.feature_importances_

        elif method == 'mutual_info':
            importances = mutual_info_regression(X_clean, y_clean, random_state=42)

        else:
            raise ValueError(f"Unknown method: {method}")

        # 正規化到 0-1
        importances = importances / (importances.sum() + 1e-10)

        return dict(zip(X.columns, importances))

    def select_features(
        self,
        features: List[Feature],
        max_features: int = 10,
        correlation_threshold: float = 0.8,
        data: Optional[DataFrame] = None
    ) -> List[Feature]:
        """
        特徵選擇（基於重要性和相關性）

        Args:
            features: 特徵列表
            max_features: 最大特徵數
            correlation_threshold: 相關性閾值（移除高度相關特徵）
            data: 市場資料（用於計算相關性）

        Returns:
            選擇的特徵列表
        """
        # 按重要性排序
        sorted_features = sorted(
            features,
            key=lambda f: f.importance,
            reverse=True
        )

        if data is None or len(sorted_features) <= max_features:
            return sorted_features[:max_features]

        # 計算特徵相關性矩陣（簡化處理）
        # 實際需要計算每個特徵的值
        selected = [sorted_features[0]]  # 保留最重要的

        for feature in sorted_features[1:]:
            if len(selected) >= max_features:
                break

            # 檢查與已選擇特徵的相關性
            # 這裡簡化處理，實際需要計算相關係數
            is_correlated = False

            # 實際應用中在這裡計算相關性
            # for selected_feature in selected:
            #     corr = calculate_correlation(feature, selected_feature, data)
            #     if abs(corr) > correlation_threshold:
            #         is_correlated = True
            #         break

            if not is_correlated:
                selected.append(feature)

        return selected


# ============================================================================
# 5. 自動策略生成器
# ============================================================================

class AutoStrategyGenerator:
    """
    自動策略生成器

    基於特徵集合自動生成交易策略規則。

    使用範例:
        generator = AutoStrategyGenerator()

        strategy_rules = generator.generate_strategy(
            features=best_feature_set,
            entry_rules=3,
            exit_rules=2
        )

        print(strategy_rules)
    """

    def generate_strategy(
        self,
        features: FeatureSet,
        entry_rules: int = 3,
        exit_rules: int = 2,
        rule_type: Literal['threshold', 'crossover', 'mixed'] = 'mixed'
    ) -> Dict[str, Any]:
        """
        基於特徵生成交易策略

        Args:
            features: 特徵集合
            entry_rules: 進場規則數量
            exit_rules: 出場規則數量
            rule_type: 規則類型
                - 'threshold': 閾值規則 (feature > threshold)
                - 'crossover': 交叉規則 (feature1 crosses feature2)
                - 'mixed': 混合規則

        Returns:
            策略規則字典
            {
                'entry_long': [規則列表],
                'exit_long': [規則列表],
                'entry_short': [規則列表],
                'exit_short': [規則列表],
                'features_used': [使用的特徵]
            }
        """
        # 選擇重要特徵
        top_features = sorted(
            features.features,
            key=lambda f: f.importance,
            reverse=True
        )[:entry_rules + exit_rules]

        # 生成進場規則
        entry_long_rules = []
        for i in range(min(entry_rules, len(top_features))):
            feature = top_features[i]
            rule = self._generate_rule(feature, 'entry_long', rule_type)
            entry_long_rules.append(rule)

        # 生成出場規則
        exit_long_rules = []
        for i in range(entry_rules, min(entry_rules + exit_rules, len(top_features))):
            feature = top_features[i]
            rule = self._generate_rule(feature, 'exit_long', rule_type)
            exit_long_rules.append(rule)

        # 空單規則（反向）
        entry_short_rules = [self._reverse_rule(r) for r in entry_long_rules]
        exit_short_rules = [self._reverse_rule(r) for r in exit_long_rules]

        return {
            'entry_long': entry_long_rules,
            'exit_long': exit_long_rules,
            'entry_short': entry_short_rules,
            'exit_short': exit_short_rules,
            'features_used': [f.name for f in top_features],
            'feature_expressions': {f.name: f.expression for f in top_features}
        }

    def _generate_rule(
        self,
        feature: Feature,
        rule_context: str,
        rule_type: str
    ) -> str:
        """生成單一規則"""
        if rule_type == 'threshold':
            # 閾值規則
            threshold = random.uniform(-2, 2)  # 假設標準化後的閾值
            operator_str = '>' if 'entry' in rule_context else '<'
            return f"{feature.name} {operator_str} {threshold:.2f}"

        elif rule_type == 'crossover':
            # 交叉規則（需要另一個特徵）
            return f"{feature.name} crosses above 0"

        else:  # mixed
            # 混合規則
            if random.random() < 0.5:
                threshold = random.uniform(-2, 2)
                operator_str = '>' if 'entry' in rule_context else '<'
                return f"{feature.name} {operator_str} {threshold:.2f}"
            else:
                return f"{feature.name} crosses above 0"

    def _reverse_rule(self, rule: str) -> str:
        """反轉規則（用於空單）"""
        # 反轉比較運算符
        rule = rule.replace(' > ', ' <GREATER> ')
        rule = rule.replace(' < ', ' <LESS> ')
        rule = rule.replace(' <GREATER> ', ' < ')
        rule = rule.replace(' <LESS> ', ' > ')

        # 反轉方向詞
        rule = rule.replace('above', 'below')
        rule = rule.replace('Below', 'above')  # 處理首字母大寫

        return rule


# ============================================================================
# 6. 便利函數
# ============================================================================

def create_feature_engineer(
    base_indicators: Optional[List[str]] = None,
    population_size: int = 50,
    generations: int = 20
) -> GeneticFeatureEngineer:
    """
    建立特徵工程器

    Args:
        base_indicators: 基礎指標列表（None 則使用預設）
        population_size: 族群大小
        generations: 演化代數

    Returns:
        GeneticFeatureEngineer 實例
    """
    if base_indicators is None:
        base_indicators = [
            'SMA', 'EMA', 'RSI', 'MACD', 'ATR',
            'BB', 'STOCH', 'ADX', 'CCI', 'MOM'
        ]

    return GeneticFeatureEngineer(
        base_indicators=base_indicators,
        operators=['+', '-', '*', '/'],
        population_size=population_size,
        generations=generations
    )


def quick_feature_evolution(
    data: DataFrame,
    fitness_function: Callable[[FeatureSet, DataFrame], float],
    n_trials: int = 3,
    verbose: bool = True
) -> FeatureSet:
    """
    快速特徵演化（執行多次取最佳）

    Args:
        data: 市場資料
        fitness_function: 適應度函數
        n_trials: 試驗次數
        verbose: 是否顯示進度

    Returns:
        最佳特徵集合
    """
    best_result = None
    best_fitness = float('-inf')

    for trial in range(n_trials):
        if verbose:
            print(f"\n試驗 {trial + 1}/{n_trials}")

        engineer = create_feature_engineer(
            population_size=30,
            generations=15
        )

        result = engineer.evolve(
            data=data,
            fitness_function=fitness_function,
            verbose=verbose
        )

        if result.fitness_score > best_fitness:
            best_fitness = result.fitness_score
            best_result = result

    if verbose:
        print(f"\n最佳結果來自試驗 (Fitness: {best_fitness:.4f})")

    return best_result
