# DEAP GP Integration - Technical Design

## Context

### Background
合約交易回測系統需要持續發現新的交易策略。目前依賴人工設計策略邏輯，效率有限。Genetic Programming (GP) 可以自動從技術指標組合中演化出交易規則，突破人工設計的局限性。

### Stakeholders
- 量化交易開發者（使用演化策略）
- AI 自動化系統（執行演化循環）
- 回測引擎（評估策略績效）

### Constraints
- 必須與現有 BaseStrategy 介面相容
- 不能修改現有 12 個策略
- 不能替換現有 Optuna 優化器
- 演化時間需控制在合理範圍（< 1 小時）

---

## Goals / Non-Goals

### Goals
1. **自動策略生成**：從技術指標原語演化出可執行的交易策略
2. **過擬合防護**：內建複雜度控制和驗證機制
3. **可解釋輸出**：產生人類可讀的策略程式碼
4. **系統整合**：與現有回測、優化、學習系統無縫整合

### Non-Goals
- 替換現有 Optuna 參數優化器
- 替換現有 NSGA-II 多目標優化器
- 即時交易整合（只做回測階段）
- 高頻策略支援（專注於中低頻策略）

---

## Architecture Decisions

### Decision 1: GP 原語分層設計

**選擇**：三層原語架構

```
Layer 3: Signal Primitives (輸出布林訊號)
├── and_(bool, bool) -> bool
├── or_(bool, bool) -> bool
├── not_(bool) -> bool
└── if_then_else(bool, bool, bool) -> bool

Layer 2: Comparison Primitives (產生比較結果)
├── gt(float, float) -> bool      # >
├── lt(float, float) -> bool      # <
├── cross_above(float, float) -> bool
└── cross_below(float, float) -> bool

Layer 1: Indicator Primitives (終端節點)
├── close -> float
├── rsi(period) -> float
├── ma(period) -> float
├── atr(period) -> float
├── macd_line -> float
├── macd_signal -> float
├── bb_upper -> float
├── bb_lower -> float
└── constant(value) -> float
```

**原因**：
- 類型安全：使用 Typed GP 確保表達式合法
- 可控複雜度：分層限制組合方式
- 可解釋性：層次結構易於轉換為人類可讀規則

**替代方案（放棄）**：
- 單層原語：靈活但易產生無意義表達式
- 純數值原語：需要額外閾值參數增加搜索空間

---

### Decision 2: 適應度函數設計

**選擇**：多目標加權適應度

```python
fitness = (
    w1 * sharpe_ratio +
    w2 * (1 - max_drawdown) +
    w3 * win_rate +
    w4 * (1 - complexity_penalty)
)
```

**預設權重**：
- `w1 = 0.5`（Sharpe Ratio，主要目標）
- `w2 = 0.3`（Max Drawdown 懲罰）
- `w3 = 0.1`（Win Rate）
- `w4 = 0.1`（複雜度懲罰）

**原因**：
- Sharpe Ratio 是主要績效指標
- Max Drawdown 控制風險
- 複雜度懲罰防止過擬合

**替代方案（保留可選）**：
- 純 NSGA-II 多目標：可作為進階模式
- 純 Sharpe：簡單但易過擬合

---

### Decision 3: 過擬合防護機制

**選擇**：多層防護

```
1. 結構約束
   ├── max_depth = 17（樹最大深度）
   ├── max_nodes = 50（節點數量上限）
   └── min_trades = 30（最少交易次數）

2. 複雜度懲罰
   └── penalty = 0.01 * (node_count - 10) if node_count > 10 else 0

3. 驗證流程
   ├── Train/Test Split（70/30）
   ├── Walk-Forward Validation（5 folds）
   └── Monte Carlo Permutation Test
```

**原因**：
- 多層防護比單一措施更可靠
- 與現有驗證系統（`src/validator/`）整合

---

### Decision 4: 策略轉換架構

**選擇**：動態編譯 + 檔案生成

```
GP Expression Tree
       ↓
   Converter
       ↓
   ┌───────────────┐
   │ Dynamic Mode  │ ─── 即時編譯，測試用
   └───────────────┘
       ↓
   ┌───────────────┐
   │ Static Mode   │ ─── 生成 .py 檔案，生產用
   └───────────────┘
       ↓
   EvolvedStrategy
```

**動態模式**：
```python
strategy = EvolvedStrategy.from_expression(expr_tree)
result = engine.run(strategy, data)
```

**靜態模式**：
```python
converter.save_strategy(expr_tree, "src/strategies/gp/generated/evolved_001.py")
# 產生完整的 Python 檔案
```

**原因**：
- 動態模式：快速迭代，演化過程使用
- 靜態模式：持久化，生產部署使用

---

### Decision 5: 與現有系統的整合方式

**選擇**：Adapter 模式 + 事件驅動

```
┌──────────────────────────────────────────────────────────┐
│                   GPEvolutionEngine                       │
│                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ Primitives  │ ←→ │ Population  │ ←→ │ Constraints │  │
│  └─────────────┘    └─────────────┘    └─────────────┘  │
│                           ↓                              │
│                    FitnessAdapter                        │
│                           ↓                              │
└───────────────────────────┬──────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          ↓                 ↓                 ↓
    BacktestEngine    ValidationStages    LearningRecorder
    (評估績效)          (驗證策略)         (記錄結果)
```

**原因**：
- 不修改現有模組
- 透過 Adapter 封裝整合邏輯
- 事件驅動方便擴展

---

## Detailed Design

### GP Primitives Implementation

```python
# src/gp/primitives.py

from deap import gp
from typing import Any
import operator

def create_trading_pset() -> gp.PrimitiveSetTyped:
    """建立交易專用的 Typed Primitive Set"""

    # 定義類型
    pset = gp.PrimitiveSetTyped(
        "TradingGP",
        in_types=[],  # 終端節點提供輸入
        ret_type=bool  # 輸出布林訊號
    )

    # === Layer 1: Indicator Terminals ===
    # 這些將在運行時綁定到實際數據
    pset.addTerminal(name="close", terminal=0.0, ret_type=float)
    pset.addTerminal(name="rsi_14", terminal=0.0, ret_type=float)
    pset.addTerminal(name="rsi_7", terminal=0.0, ret_type=float)
    pset.addTerminal(name="ma_10", terminal=0.0, ret_type=float)
    pset.addTerminal(name="ma_20", terminal=0.0, ret_type=float)
    pset.addTerminal(name="ma_50", terminal=0.0, ret_type=float)
    pset.addTerminal(name="atr_14", terminal=0.0, ret_type=float)
    pset.addTerminal(name="bb_upper", terminal=0.0, ret_type=float)
    pset.addTerminal(name="bb_lower", terminal=0.0, ret_type=float)
    pset.addTerminal(name="macd_line", terminal=0.0, ret_type=float)
    pset.addTerminal(name="macd_signal", terminal=0.0, ret_type=float)

    # 常數
    pset.addEphemeralConstant(
        "const",
        lambda: random.choice([20, 30, 50, 70, 80]),
        ret_type=float
    )

    # === Layer 2: Comparison Primitives ===
    pset.addPrimitive(operator.gt, [float, float], bool, name="gt")
    pset.addPrimitive(operator.lt, [float, float], bool, name="lt")
    pset.addPrimitive(operator.ge, [float, float], bool, name="ge")
    pset.addPrimitive(operator.le, [float, float], bool, name="le")

    def cross_above(a: float, b: float) -> bool:
        """向上突破（需要前一根 K 棒數據，簡化為 a > b）"""
        return a > b

    def cross_below(a: float, b: float) -> bool:
        """向下跌破"""
        return a < b

    pset.addPrimitive(cross_above, [float, float], bool)
    pset.addPrimitive(cross_below, [float, float], bool)

    # === Layer 3: Logic Primitives ===
    pset.addPrimitive(operator.and_, [bool, bool], bool, name="and_")
    pset.addPrimitive(operator.or_, [bool, bool], bool, name="or_")
    pset.addPrimitive(operator.not_, [bool], bool, name="not_")

    def if_then_else(condition: bool, out1: bool, out2: bool) -> bool:
        return out1 if condition else out2

    pset.addPrimitive(if_then_else, [bool, bool, bool], bool)

    return pset
```

### Fitness Function

```python
# src/gp/fitness.py

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from ..backtester.engine import BacktestEngine, BacktestResult

@dataclass
class FitnessConfig:
    """適應度配置"""
    sharpe_weight: float = 0.5
    drawdown_weight: float = 0.3
    win_rate_weight: float = 0.1
    complexity_weight: float = 0.1

    min_trades: int = 30
    max_depth: int = 17
    max_nodes: int = 50


class TradingFitness:
    """交易策略適應度評估器"""

    def __init__(
        self,
        engine: BacktestEngine,
        data: pd.DataFrame,
        config: Optional[FitnessConfig] = None
    ):
        self.engine = engine
        self.data = data
        self.config = config or FitnessConfig()

    def evaluate(self, individual) -> Tuple[float,]:
        """評估個體適應度"""

        # 1. 約束檢查
        depth = individual.height
        nodes = len(individual)

        if depth > self.config.max_depth:
            return (-1000.0,)  # 懲罰過深的樹

        if nodes > self.config.max_nodes:
            return (-1000.0,)  # 懲罰過大的樹

        # 2. 編譯並執行回測
        try:
            strategy = self._compile_to_strategy(individual)
            result = self.engine.run(strategy, data=self.data)
        except Exception as e:
            return (-1000.0,)  # 無效策略

        # 3. 最少交易次數檢查
        if result.total_trades < self.config.min_trades:
            return (-500.0,)  # 交易次數不足

        # 4. 計算加權適應度
        fitness = self._calculate_fitness(result, nodes)

        return (fitness,)

    def _calculate_fitness(
        self,
        result: BacktestResult,
        node_count: int
    ) -> float:
        """計算加權適應度分數"""

        # Sharpe Ratio（越高越好）
        sharpe_score = result.sharpe_ratio

        # Max Drawdown（越低越好，轉為正向分數）
        dd_score = 1.0 - abs(result.max_drawdown)

        # Win Rate（越高越好）
        wr_score = result.win_rate

        # 複雜度懲罰（節點數越多懲罰越大）
        complexity_penalty = 0.0
        if node_count > 10:
            complexity_penalty = 0.01 * (node_count - 10)

        complexity_score = 1.0 - min(complexity_penalty, 0.5)

        # 加權總分
        fitness = (
            self.config.sharpe_weight * sharpe_score +
            self.config.drawdown_weight * dd_score +
            self.config.win_rate_weight * wr_score +
            self.config.complexity_weight * complexity_score
        )

        return fitness
```

### GP Engine

```python
# src/gp/engine.py

from deap import base, creator, tools, algorithms, gp
from typing import Optional, List, Dict, Any
import multiprocessing
import random

@dataclass
class EvolutionConfig:
    """演化配置"""
    population_size: int = 100
    n_generations: int = 50
    crossover_prob: float = 0.7
    mutation_prob: float = 0.2
    tournament_size: int = 3
    elite_size: int = 5
    n_jobs: int = -1  # -1 = 所有 CPU
    seed: Optional[int] = None
    early_stopping_patience: int = 10


class GPEvolutionEngine:
    """GP 演化引擎"""

    def __init__(
        self,
        fitness_evaluator: TradingFitness,
        pset: gp.PrimitiveSetTyped,
        config: Optional[EvolutionConfig] = None
    ):
        self.fitness_evaluator = fitness_evaluator
        self.pset = pset
        self.config = config or EvolutionConfig()

        self._setup_deap()

    def _setup_deap(self):
        """設定 DEAP 框架"""

        # 建立適應度和個體類型
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create(
            "Individual",
            gp.PrimitiveTree,
            fitness=creator.FitnessMax
        )

        # 工具箱
        self.toolbox = base.Toolbox()

        # 個體生成
        self.toolbox.register(
            "expr",
            gp.genHalfAndHalf,
            pset=self.pset,
            min_=1,
            max_=3
        )
        self.toolbox.register(
            "individual",
            tools.initIterate,
            creator.Individual,
            self.toolbox.expr
        )
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual
        )

        # 編譯
        self.toolbox.register("compile", gp.compile, pset=self.pset)

        # 評估
        self.toolbox.register(
            "evaluate",
            self.fitness_evaluator.evaluate
        )

        # 選擇
        self.toolbox.register(
            "select",
            tools.selTournament,
            tournsize=self.config.tournament_size
        )

        # 交叉
        self.toolbox.register("mate", gp.cxOnePoint)

        # 突變
        self.toolbox.register(
            "expr_mut",
            gp.genFull,
            min_=0,
            max_=2
        )
        self.toolbox.register(
            "mutate",
            gp.mutUniform,
            expr=self.toolbox.expr_mut,
            pset=self.pset
        )

        # 約束裝飾器（限制樹深度）
        self.toolbox.decorate(
            "mate",
            gp.staticLimit(
                key=operator.attrgetter("height"),
                max_value=17
            )
        )
        self.toolbox.decorate(
            "mutate",
            gp.staticLimit(
                key=operator.attrgetter("height"),
                max_value=17
            )
        )

    def evolve(self) -> Dict[str, Any]:
        """執行演化"""

        if self.config.seed is not None:
            random.seed(self.config.seed)

        # 初始化種群
        pop = self.toolbox.population(n=self.config.population_size)

        # Hall of Fame
        hof = tools.HallOfFame(self.config.elite_size)

        # 統計
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # 並行化
        if self.config.n_jobs != 1:
            n_workers = (
                multiprocessing.cpu_count()
                if self.config.n_jobs == -1
                else self.config.n_jobs
            )
            pool = multiprocessing.Pool(n_workers)
            self.toolbox.register("map", pool.map)

        # 演化
        pop, logbook = algorithms.eaSimple(
            pop,
            self.toolbox,
            cxpb=self.config.crossover_prob,
            mutpb=self.config.mutation_prob,
            ngen=self.config.n_generations,
            stats=stats,
            halloffame=hof,
            verbose=True
        )

        # 關閉 pool
        if self.config.n_jobs != 1:
            pool.close()
            pool.join()

        return {
            "best_individual": hof[0],
            "best_fitness": hof[0].fitness.values[0],
            "hall_of_fame": hof,
            "final_population": pop,
            "logbook": logbook
        }
```

---

## Migration Plan

### Phase 1: 安裝和驗證（Day 1）
1. 安裝 DEAP：`pip install deap`
2. 驗證基本 GP 功能可運作

### Phase 2: 核心開發（Day 2-3）
1. 實作原語系統
2. 實作適應度函數
3. 實作演化引擎

### Phase 3: 整合（Day 4）
1. 與 BacktestEngine 整合
2. 策略轉換器
3. 學習系統整合

### Phase 4: 測試驗證（Day 5）
1. 單元測試
2. 整合測試
3. 過擬合測試

### Rollback Plan
- 所有新增都在 `src/gp/` 目錄
- 不修改現有程式碼
- 若有問題，直接刪除 `src/gp/` 即可回滾

---

## Risks & Trade-offs

### Risk 1: 演化時間過長
- **風險**：100 代 × 100 個體 × 回測時間 = 數小時
- **緩解**：
  - 使用向量化回測（已有 `vectorized.py`）
  - 並行評估
  - 早期停止
  - 縮減訓練資料範圍

### Risk 2: 產生無意義策略
- **風險**：GP 可能產生 `and_(True, True)` 這種無意義表達式
- **緩解**：
  - 最少交易次數約束
  - 複雜度懲罰
  - 人工審查最終策略

### Risk 3: 過擬合
- **風險**：GP 策略可能過度適應歷史資料
- **緩解**：
  - 樹深度限制
  - Walk-Forward 驗證
  - 複雜度懲罰
  - Monte Carlo 測試

### Trade-off: 可解釋性 vs 績效
- 限制樹深度提高可解釋性，但可能犧牲績效
- 預設傾向可解釋性（max_depth=17）
- 可配置為更深的樹以探索更複雜策略
