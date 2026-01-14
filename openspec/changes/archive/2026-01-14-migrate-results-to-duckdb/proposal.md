# Migrate Results to DuckDB

## Why

目前策略結果儲存在 `learning/experiments.json`，造成三個問題：
1. **混合關注點**：JSON 同時存「策略結果」（結構化數據）和「經驗教訓」（語意內容），但兩者查詢模式完全不同
2. **型別混亂**：跨模組資料傳遞使用裸 dict，導致欄位不一致（如 `_strategy_name` vs `strategy_name`）
3. **效能瓶頸**：JSON 不適合大量查詢（已有 622 筆、26,000+ 行），每次查詢需載入全部資料

## What Changes

### 資料儲存重構
- **NEW** `src/db/results.duckdb` - 策略結果使用 DuckDB 儲存
- **NEW** `src/db/models.py` - DuckDB 表結構定義
- **NEW** `src/types/` - 統一型別定義模組
- **REMOVE** `learning/experiments.json` - 遷移後刪除
- **KEEP** `learning/insights.md` - 人讀的經驗總結保留
- **KEEP** Memory MCP - AI 可查詢的語意經驗保留

### 型別系統建立
- **NEW** `src/types/results.py` - BacktestResult, ValidationResult 型別
- **NEW** `src/types/configs.py` - 配置型別
- **NEW** `src/types/strategies.py` - 策略相關型別
- **UPDATE** 所有模組改用統一型別，不再使用裸 dict

### 模組更新
- **UPDATE** `src/learning/recorder.py` - 改寫入 DuckDB
- **UPDATE** `ui/utils/data_loader.py` - 改從 DuckDB 讀取
- **UPDATE** `src/automation/ultimate_loop.py` - 使用統一型別
- **UPDATE** `src/automation/hyperloop.py` - 使用統一型別

## Impact

### Affected Specs
- data-layer (新增)
- learning-system (更新查詢介面)

### Affected Code
```
src/db/                           # 新增
src/types/                        # 新增
src/learning/recorder.py          # 重構
src/learning/storage.py           # 更新
ui/utils/data_loader.py           # 更新
src/automation/ultimate_loop.py   # 更新
src/automation/hyperloop.py       # 更新
src/interfaces.py                 # 整合
```

### Breaking Changes
- `ExperimentRecorder.log_experiment()` 參數型別變更（dict → typed dataclass）
- `ExperimentRecorder.query_experiments()` 返回型別變更
- UI 需要重新整合 data_loader

### Migration
- 現有 622 筆 JSON 資料會自動遷移到 DuckDB
- 遷移期間保留 JSON 作為備份
- 遷移完成並驗證後刪除 JSON

## Decisions

### 選擇 DuckDB 而非 SQLite
- **原因**：分析型查詢更快（如聚合、排序、視窗函數）
- **優勢**：原生 Pandas 整合、列式儲存適合時間序列
- **風險緩解**：DuckDB 已穩定，API 與 SQLite 相似

### 型別系統使用 dataclass + Protocol
- **原因**：比 TypedDict 更嚴格，IDE 支援更好
- **優勢**：可驗證、可序列化、可繼承
- **相容性**：保留 `to_dict()` 方法供 JSON/DuckDB 序列化

### 保留三層儲存架構
```
DuckDB         ← 策略結果（結構化查詢）
insights.md    ← 人讀總結（Git 追蹤）
Memory MCP     ← AI 語意查詢（跨 session）
```

### 經驗教訓系統整合
- **UPDATE** `src/learning/insights.py` - 整合統一型別
- **UPDATE** `src/learning/memory.py` - 整合統一型別，自動觸發
- **NEW** `src/learning/lesson_detector.py` - 自動偵測值得記錄的洞察
- **UPDATE** `src/automation/ultimate_loop.py` - 回測後自動呼叫經驗教訓系統

### 經驗教訓觸發條件
| 類型 | 觸發條件 | 目標 |
|------|----------|------|
| 策略成功 | Sharpe > 2.0 | Memory MCP + insights.md |
| 策略失敗 | Sharpe < 0.5 | Memory MCP + insights.md |
| 過擬合警訊 | MC 失敗率 > 30% | Memory MCP + insights.md |
| 風險事件 | MaxDD > 25% | insights.md |
| 參數敏感 | 穩健性差異大 | insights.md |
