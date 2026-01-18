# integrate-gp-ultimate Implementation Tasks

## Progress
- Total: 12 tasks
- Completed: 12
- Status: ✅ COMPLETED

---

## 1. Foundation - Data Contracts (sequential)

- [x] 1.1 Define data contracts | files: src/automation/gp_integration.py
  - 建立 `GPExplorationRequest`、`GPExplorationResult`、`DynamicStrategyInfo` dataclasses
  - 確保所有欄位有 type hints 和 docstrings

- [x] 1.2 Extend StrategyRegistry for dynamic registration | files: src/strategies/registry.py
  - 新增 `_dynamic_strategies: Dict[str, DynamicStrategyInfo]`
  - 實作 `register_dynamic()`、`unregister_dynamic()`、`list_dynamic()`
  - 新增 `clear_dynamic()` 清理所有動態策略

## 2. Core Components (sequential, depends: 1)

- [x] 2.1 Implement GPStrategyAdapter | files: src/automation/gp_integration.py
  - 實作 `create_strategy_class()` 動態建立策略類別
  - 整合 `ExpressionConverter` 編譯表達式
  - 設定 `EvolvedStrategy` 類別屬性

- [x] 2.2 Implement GPExplorer | files: src/automation/gp_integration.py
  - 封裝 `GPLoop` 調用邏輯
  - 實作 `explore()` 方法
  - 處理錯誤和超時

## 3. Configuration Extension (parallel, depends: 1)

- [x] 3.1 Extend UltimateLoopConfig | files: src/automation/ultimate_config.py
  - 新增 GP 配置參數：
    - `gp_explore_enabled: bool = True`
    - `gp_explore_ratio: float = 0.2`
    - `gp_population_size: int = 50`
    - `gp_generations: int = 30`
    - `gp_top_n: int = 3`
  - 更新 `validate()` 方法驗證新參數
  - 更新 factory methods（`create_production_config` 等）

## 4. UltimateLoop Integration (sequential, depends: 2, 3)

- [x] 4.1 Initialize GPExplorer in UltimateLoop | files: src/automation/ultimate_loop.py
  - 新增 `_init_gp_explorer()` 方法
  - 條件性初始化（根據 `gp_explore_enabled`）
  - 更新 `_log_module_availability()` 顯示 GP 狀態

- [x] 4.2 Implement GP explore branch | files: src/automation/ultimate_loop.py
  - 新增 `_explore_with_gp()` 方法
  - 修改 `_select_strategies()` 加入 GP 分支
  - 實作二層隨機機制（exploit_ratio + gp_explore_ratio）

- [x] 4.3 Implement strategy registration | files: src/automation/ultimate_loop.py
  - 新增 `_register_gp_strategies()` 方法
  - 將 GP 生成的策略註冊到 Registry
  - 更新 `available_strategies` 列表

## 5. Learning Integration (parallel, depends: 4)

- [x] 5.1 Record GP exploration to learning system | files: src/automation/ultimate_loop.py
  - 修改 `_record_and_learn()` 識別 GP 策略
  - 記錄 GP 演化元資料（expression, generation, fitness）
  - 更新 `UltimateLoopSummary` 統計 GP 生成數量

## 6. Testing (parallel, depends: 4)

- [x] 6.1 Unit tests for GPExplorer | files: tests/unit/automation/test_gp_integration.py
  - 測試 `GPExplorationRequest` 驗證
  - 測試 `GPStrategyAdapter.create_strategy_class()`
  - 測試 `GPExplorer.explore()` 成功和失敗路徑
  - Mock GPLoop 避免實際演化

- [x] 6.2 Unit tests for dynamic registration | files: tests/unit/strategies/test_registry_dynamic.py
  - 測試 `register_dynamic()` 成功註冊
  - 測試名稱衝突處理
  - 測試 `unregister_dynamic()`
  - 測試 `list_dynamic()` 和 `clear_dynamic()`

- [x] 6.3 Integration tests | files: tests/integration/test_ultimate_gp_integration.py
  - 測試完整 GP explore 流程
  - 測試 GP 策略進入 5 階段驗證
  - 測試 GP 策略被 exploit 選中
  - 使用小規模配置（population=10, generations=5）

---

## Execution Notes

### Phase Dependencies
```
Phase 1 (Foundation) → Phase 2 (Core) → Phase 4 (Integration)
                    → Phase 3 (Config) ↗
Phase 4 → Phase 5 (Learning)
       → Phase 6 (Testing)
```

### Parallel Execution
- Phase 3 與 Phase 2 可並行
- Phase 5 與 Phase 6 可並行

### File Conflict Check
- `gp_integration.py` - 新檔案，無衝突
- `registry.py` - Task 1.2 獨佔
- `ultimate_config.py` - Task 3.1 獨佔
- `ultimate_loop.py` - Tasks 4.1-4.3, 5.1 需順序執行

### Estimated D->R->T Cycles
- Total: 12 tasks
- Estimated time: ~8-10 D->R->T cycles（部分 parallel）
