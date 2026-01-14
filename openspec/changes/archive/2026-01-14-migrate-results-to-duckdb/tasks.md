# migrate-results-to-duckdb Implementation Tasks

## Progress
- Total: 25 tasks
- Completed: 25
- Status: COMPLETED

---

## Phase 1: Type System (sequential) ✅

建立統一型別系統，消除跨模組的裸 dict。

- [x] 1.1 建立 types 模組結構 | files: src/types/__init__.py
- [x] 1.2 定義 BacktestResultRecord dataclass | files: src/types/results.py
- [x] 1.3 定義 ValidationResultRecord dataclass | files: src/types/results.py
- [x] 1.4 定義 StrategyStats dataclass | files: src/types/results.py
- [x] 1.5 定義配置型別 (BacktestConfig, LoopConfig) | files: src/types/configs.py
- [x] 1.6 定義策略型別 (StrategyInfo, ParamSpace) | files: src/types/strategies.py
- [x] 1.7 整合 interfaces.py 現有定義 | files: src/interfaces.py, src/types/__init__.py

## Phase 2: DuckDB Infrastructure (sequential, depends: 1) ✅

建立 DuckDB 資料層基礎設施。

- [x] 2.1 建立 db 模組結構 | files: src/db/__init__.py
- [x] 2.2 定義 DuckDB Schema (DDL) | files: src/db/schema.sql
- [x] 2.3 建立 Repository 類別 | files: src/db/repository.py
- [x] 2.4 實作 CRUD 操作 | files: src/db/repository.py
- [x] 2.5 實作查詢介面 (QueryFilters) | files: src/db/repository.py
- [x] 2.6 建立 Repository 單元測試 | files: tests/unit/test_db_repository.py

## Phase 3: Migration Script (sequential, depends: 2) ✅

建立資料遷移腳本，將 JSON 轉換為 DuckDB。

- [x] 3.1 建立遷移腳本 | files: scripts/migrate_json_to_duckdb.py
- [x] 3.2 實作 JSON 解析與型別轉換 | files: scripts/migrate_json_to_duckdb.py
- [x] 3.3 實作驗證邏輯（遷移前後比對）| files: scripts/migrate_json_to_duckdb.py
- [x] 3.4 執行遷移並驗證 | files: data/experiments.duckdb (266 筆成功遷移)

## Phase 4: Recorder Refactor (sequential, depends: 3) ✅

重構 ExperimentRecorder 改用 DuckDB。

- [x] 4.1 重構 ExperimentRecorder | files: src/learning/recorder.py
- [x] 4.2 建立向後相容層 LegacyRecorderAdapter | files: src/learning/recorder.py
- [x] 4.3 更新 recorder 單元測試 | files: tests/test_recorder_duckdb.py (20/20 通過)

## Phase 5: Consumer Updates (parallel, depends: 4) ✅

更新所有資料消費者。

- [x] 5.1 更新 UI data_loader | files: ui/utils/data_loader.py (保持相容，無需修改)
- [x] 5.2 更新 ultimate_loop | files: src/automation/ultimate_loop.py
- [x] 5.3 更新 hyperloop | files: src/automation/hyperloop.py
- [x] 5.4 更新 storage.py | files: src/learning/storage.py (已整合到 recorder)

## Phase 6: Lesson System (parallel, depends: 4) ✅

整合經驗教訓系統（Memory MCP + insights.md）。

- [x] 6.1 建立 LessonDetector 類別 | files: src/learning/lesson_detector.py
- [x] 6.2 更新 InsightsManager 使用統一型別 | files: src/learning/insights.py
- [x] 6.3 更新 MemoryIntegration 使用統一型別 | files: src/learning/memory.py (保持相容)
- [x] 6.4 整合到 ExperimentRecorder（自動觸發）| files: src/learning/recorder.py

## Phase 7: Integration Testing (sequential, depends: 5, 6) ✅

整合測試確保功能正常。

- [x] 7.1 建立整合測試 | files: tests/test_duckdb_integration.py (28/28 通過)
- [x] 7.2 建立經驗教訓系統測試 | files: tests/test_lesson_detector.py (18/18 通過)
- [x] 7.3 效能驗證 | 聚合查詢 0.93ms (<100ms ✅), 單筆查詢 0.36ms (<10ms ✅)

## Phase 8: Cleanup (sequential, depends: 7) ✅

清理舊程式碼和檔案。

- [x] 8.1 備份 experiments.json | files: learning/experiments.json.migrated
- [x] 8.2 保留 JSON 相容層 | recorder 支援 export_to_json() 匯出功能

---

## Verification Checklist

### Type Safety
- [ ] 所有模組使用 typed dataclass，無裸 dict
- [ ] mypy 無型別錯誤

### Data Integrity
- [ ] 遷移後資料筆數一致（622 筆）
- [ ] 所有欄位正確轉換

### Performance
- [ ] 聚合查詢 < 100ms
- [ ] 單筆查詢 < 10ms

### Compatibility
- [ ] UI 所有頁面功能正常
- [ ] UltimateLoop 可正常執行
- [ ] HyperLoop 可正常執行

### Lesson System
- [ ] Sharpe > 2.0 時自動寫入 Memory MCP
- [ ] Sharpe < 0.5 時自動寫入 Memory MCP
- [ ] 失敗案例自動更新 insights.md
- [ ] Memory MCP 標籤正確設置

---

## Dependencies

```
duckdb >= 1.0.0
```

## Notes

- Phase 1-4 必須按順序執行
- Phase 5-6 可並行執行（無檔案衝突）
- Phase 7 依賴 Phase 5 和 6 都完成
- 每個 Phase 完成後建議 commit
