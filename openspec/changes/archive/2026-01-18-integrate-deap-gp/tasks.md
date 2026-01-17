# integrate-deap-gp Implementation Tasks

## Progress
- Total: 18 tasks
- Completed: 18
- Status: ✅ COMPLETED

---

## Phase 1: Foundation (sequential) ✅

建立基礎設施和依賴

- [x] 1.1 Add DEAP dependency to requirements.txt | files: requirements.txt
- [x] 1.2 Create src/gp/ directory structure | files: src/gp/__init__.py

---

## Phase 2: GP Primitives (sequential) ✅

設計交易專用的 GP 原語系統

- [x] 2.1 Create indicator primitives (RSI, MA, ATR, MACD, BB) | files: src/gp/primitives.py
- [x] 2.2 Create comparison primitives (gt, lt, cross_above, cross_below) | files: src/gp/primitives.py
- [x] 2.3 Create logic primitives (and_, or_, not_) | files: src/gp/primitives.py
- [x] 2.4 Create math primitives with protected operations | files: src/gp/primitives.py
- [x] 2.5 Create PrimitiveSetFactory for building typed primitive sets | files: src/gp/primitives.py

**Dependencies**: Phase 1

---

## Phase 3: Fitness & Constraints (parallel after 2.5) ✅

建立適應度函數和約束系統

- [x] 3.1 Create fitness function integrating BacktestEngine | files: src/gp/fitness.py
  - 使用 BacktestResult 作為適應度來源
  - 支援多目標（Sharpe, MaxDD, WinRate）
- [x] 3.2 Create constraint functions (depth limit, bloat control) | files: src/gp/constraints.py
  - 樹深度限制（max_depth=17）
  - 節點數量限制
  - 複雜度懲罰

**Dependencies**: Phase 2

---

## Phase 4: GP Engine (sequential) ✅

核心演化引擎

- [x] 4.1 Create GP engine with DEAP integration | files: src/gp/engine.py
  - 種群初始化（HalfAndHalf）
  - 選擇（Tournament）
  - 交叉（cxOnePoint）
  - 突變（mutUniform, mutNodeReplacement）
- [x] 4.2 Add evolution loop with early stopping | files: src/gp/engine.py
  - 收斂檢測
  - 最大世代限制
  - Hall of Fame
- [x] 4.3 Add parallelization support (multiprocessing) | files: src/gp/engine.py
  - 並行適應度評估
  - 可配置 worker 數量

**Dependencies**: Phase 3

---

## Phase 5: Strategy Converter (sequential) ✅

GP 表達式轉換為 BaseStrategy

- [x] 5.1 Create expression tree to Python code converter | files: src/gp/converter.py
  - 表達式樹遍歷
  - Python AST 生成
  - 程式碼格式化
- [x] 5.2 Create EvolvedStrategy base class | files: src/strategies/gp/evolved_strategy.py
  - 繼承 BaseStrategy
  - 動態編譯支援
  - 元資料（來源表達式、適應度）
- [x] 5.3 Create strategy file generator | files: src/gp/converter.py
  - 產生完整的策略 Python 檔案
  - 自動命名（evolved_001, evolved_002...）
  - 存放到 src/strategies/gp/generated/

**Dependencies**: Phase 4

---

## Phase 6: Integration (parallel after 5.3) ✅

與現有系統整合

- [x] 6.1 Update strategy registry to include GP strategies | files: src/strategies/registry.py, src/strategies/__init__.py
- [x] 6.2 Add GP evolution to automation loop | files: src/automation/gp_loop.py
  - 定期演化新策略
  - 與現有 Optuna 優化配合
- [x] 6.3 Add learning system integration | files: src/gp/learning.py
  - 記錄最佳策略到 insights.md
  - 存儲到 Memory MCP

**Dependencies**: Phase 5

---

## Phase 7: Testing (parallel) ✅

測試覆蓋

- [x] 7.1 Unit tests for primitives | files: tests/test_gp_primitives.py (54 tests)
- [x] 7.2 Integration tests for GP engine | files: tests/test_gp_engine.py (41 tests)
- [x] 7.3 End-to-end test: evolve and backtest strategy | files: tests/test_gp_e2e.py (8 tests)

**Dependencies**: Phase 6

---

## Task Metadata

### Estimated Effort
- Phase 1: 0.5 hours
- Phase 2: 2 hours
- Phase 3: 1.5 hours
- Phase 4: 3 hours
- Phase 5: 2.5 hours
- Phase 6: 2 hours
- Phase 7: 1.5 hours
- **Total**: ~13 hours

### D→R→T Cycles
每個任務完成後執行 DEVELOPER → REVIEWER → TESTER 循環。

預估 D→R→T 總數: 18 cycles

### Risk Tasks（需要特別注意）
- **4.1**: GP Engine 是核心，需確保與 DEAP 正確整合
- **5.1**: 表達式轉換邏輯複雜，需充分測試
- **3.1**: 適應度函數與回測引擎整合，需處理邊界情況

### Success Criteria
- [x] 可以從頭演化出新策略 ✅ GPEngine + evolve()
- [x] 演化策略可以正常回測 ✅ FitnessEvaluator 整合 BacktestEngine
- [x] 策略可以存檔並在下次載入 ✅ StrategyGenerator 生成 .py 檔案
- [x] 與現有 CompositeStrategy 相容 ✅ EvolvedStrategy 繼承 BaseStrategy
- [x] 有過擬合防護機制 ✅ Constraint 深度/大小限制 + 複雜度懲罰
