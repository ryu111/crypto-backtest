# Integrate DEAP Genetic Programming

## Why

現有回測系統已具備完整的參數優化能力（Optuna Bayesian、NSGA-II 多目標優化），但**缺乏自動生成交易策略邏輯的能力**。目前所有策略都需要人工設計 `generate_signals()` 方法中的規則。

DEAP 的 **Genetic Programming (GP)** 提供了獨特價值：

1. **自動策略生成**：GP 可以從技術指標原語（RSI、MA、價格比較等）自動組合出交易規則
2. **表達式樹演化**：策略規則以樹狀結構表示，易於解釋和分析
3. **超越參數優化**：不只調整參數，而是發現全新的策略邏輯組合
4. **符號回歸能力**：找出指標與報酬之間的數學關係

### DEAP vs 現有 Optuna 對比

| 功能 | 現有 Optuna | DEAP GP |
|------|------------|---------|
| 參數優化 | Bayesian TPE | GA/ES（不採用，Optuna 更成熟）|
| 多目標優化 | NSGA-II（保留）| NSGA-II/III（不採用，已有）|
| **策略自動生成** | **無** | **GP 表達式樹（獨特價值）** |
| 策略結構搜索 | 無 | 可演化策略結構 |

**結論**：只整合 DEAP GP 策略生成功能，不替換現有優化器。

## What Changes

### 新增模組

1. **`src/gp/` 目錄**
   - `primitives.py` - 定義交易專用原語（技術指標、比較運算、邏輯運算）
   - `fitness.py` - 適應度函數（整合現有回測引擎）
   - `engine.py` - GP 演化引擎
   - `converter.py` - GP 表達式 → BaseStrategy 轉換器
   - `constraints.py` - 複雜度約束、過擬合防護

2. **`src/strategies/gp/` 目錄**
   - `evolved_strategy.py` - 演化策略基礎類別
   - `generated/` - 自動產生的策略檔案存放處

### 整合點

1. **與現有回測引擎整合**
   - `BacktestEngine` 作為 GP 適應度函數的評估後端
   - 使用 `BacktestResult.sharpe_ratio` 等作為適應度指標

2. **與現有策略系統整合**
   - 產生的策略繼承 `BaseStrategy`
   - 可與 `CompositeStrategy` 組合使用
   - 可加入 `StrategyRegistry`

3. **與學習系統整合**
   - 記錄演化過程到 `learning/insights.md`
   - 存儲最佳策略到 Memory MCP

## Impact

### Affected Code
- `src/strategies/base.py` - 需確保相容性（不修改）
- `src/backtester/engine.py` - 需確保相容性（不修改）
- `src/strategies/__init__.py` - 新增 GP 策略導出
- `requirements.txt` - 新增 `deap` 依賴

### Risks & Mitigations

| 風險 | 緩解措施 |
|------|----------|
| **過擬合** | 樹深度限制、複雜度懲罰、Walk-Forward 驗證 |
| **運算爆炸** | Bloat control、早期停止、並行化 |
| **不可解釋策略** | 強制簡化、人類可讀輸出 |
| **與現有系統不相容** | 使用 Adapter 模式、充分測試 |

### Non-Goals（明確排除）

- **不替換** Optuna 參數優化器
- **不替換** NSGA-II 多目標優化器
- **不修改** 現有 12 個策略
- **不改變** BacktestEngine API

## Expected Benefits

1. **自動發現新策略**：從指標組合中發現人類難以想到的規則
2. **減少人工設計工作**：讓 GP 探索策略空間
3. **與現有系統互補**：GP 生成策略 → Optuna 優化參數
4. **可解釋性**：表達式樹可轉為人類可讀的規則

## Technical Design

### GP 原語設計（核心）

```python
# 技術指標（終端節點）
pset.addTerminal("close", float)           # 當前收盤價
pset.addTerminal("rsi_14", float)          # RSI(14)
pset.addTerminal("ma_20", float)           # MA(20)
pset.addTerminal("atr_14", float)          # ATR(14)

# 比較運算（二元原語）
pset.addPrimitive(operator.gt, 2)          # >
pset.addPrimitive(operator.lt, 2)          # <
pset.addPrimitive(cross_above, 2)          # 突破
pset.addPrimitive(cross_below, 2)          # 跌破

# 邏輯運算
pset.addPrimitive(operator.and_, 2)        # AND
pset.addPrimitive(operator.or_, 2)         # OR
pset.addPrimitive(operator.not_, 1)        # NOT

# 數學運算
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protected_div, 2)
```

### 演化流程

```
1. 初始化種群（隨機表達式樹）
   ↓
2. 評估適應度（執行回測 → Sharpe Ratio）
   ↓
3. 選擇（Tournament Selection）
   ↓
4. 交叉（Subtree Crossover）
   ↓
5. 突變（Point Mutation, Subtree Mutation）
   ↓
6. 約束檢查（深度限制、複雜度限制）
   ↓
7. 重複 2-6 直到收斂或達到世代數
   ↓
8. 輸出最佳個體 → 轉換為 BaseStrategy
```

### 策略轉換範例

**GP 表達式**：
```
and_(gt(rsi_14, 70), lt(close, ma_20))
```

**轉換為策略**：
```python
class EvolvedStrategy_001(BaseStrategy):
    name = "evolved_001"

    def generate_signals(self, data):
        rsi = self.calculate_rsi(data['close'], 14)
        ma = data['close'].rolling(20).mean()

        long_entry = (rsi > 70) & (data['close'] < ma)
        # ...
```
