# Tasks: UltimateLoop 與 Skills 對齊

## 優先級說明

| 優先級 | 意義 | 預計任務數 |
|--------|------|-----------|
| P0 | 解決核心問題（Sharpe 相同） | 2 |
| P1 | 重要對齊項目 | 3 |
| P2 | 完善對齊 | 3 |

---

## P0: Walk-Forward 滾動窗口

### Task 1: WalkForwardAnalyzer 模組 ✅

**檔案**：`src/automation/walk_forward.py`

**需求**：
- [x] `WalkForwardWindow` dataclass
- [x] `WalkForwardResult` dataclass
- [x] `WalkForwardAnalyzer` 類別
  - [x] `__init__(is_ratio=0.7, n_windows=5, overlap=0.5)`
  - [x] `generate_windows(data: pd.DataFrame) -> List[WalkForwardWindow]`
  - [x] `calculate_efficiency(is_returns, oos_returns) -> float`
  - [x] `analyze(data, strategy_func, optimize_func) -> WalkForwardResult`

**參考**：`/.claude/skills/參數優化/references/walk-forward.md`

**驗收**：
- [x] 產生 5 個滾動窗口
- [x] IS/OOS 比例為 70/30
- [x] 窗口重疊 50%
- [x] 效率計算正確

---

### Task 2: 整合 Walk-Forward 到驗證階段 ✅

**檔案**：`src/automation/ultimate_loop.py`

**修改**：
- [x] `_init_validation_runner()` 新增 WalkForwardAnalyzer 初始化
- [x] 新增配置 `use_walk_forward: bool = True`
- [x] 配置 `wfa_is_ratio`, `wfa_n_windows`, `wfa_overlap`

**驗收**：
- [x] WalkForwardAnalyzer 正確初始化
- [x] 配置驗證通過

---

## P0: 交易筆數門檻

### Task 3: 交易筆數檢查 ✅

**檔案**：`src/automation/ultimate_loop.py`

**修改**：
- [x] `_validate_pareto_solutions()` 新增交易筆數檢查
- [x] 新增配置 `min_trades: int = 30`
- [x] 交易筆數 < 30 自動降級為 Grade D

**驗收**：
- [x] 低交易筆數策略被標記警告
- [x] Grade 正確降級

---

## P1: PBO 過擬合指標

### Task 4: OverfitDetector 模組 ✅

**檔案**：`src/automation/overfitting_detector.py`

**需求**：
- [x] `OverfitMetrics` dataclass
- [x] `OverfitDetector` 類別
  - [x] `calculate_pbo(returns_matrix, n_splits=8) -> float`
  - [x] `calculate_is_oos_ratio(is_results, oos_results) -> float`
  - [x] `calculate_param_sensitivity(sharpe_matrix) -> float`
  - [x] `assess_risk(metrics) -> str` ('LOW', 'MEDIUM', 'HIGH')

**參考**：`/.claude/skills/參數優化/references/overfitting-detection.md`

**驗收**：
- [x] PBO 計算符合 CSCV 方法
- [x] 風險評估邏輯正確
- [x] 模組導入測試通過

---

### Task 5: 整合 PBO 到驗證階段 ✅

**檔案**：`src/automation/ultimate_loop.py`

**修改**：
- [x] `_validate_pareto_solutions()` 整合 OverfitDetector
- [x] 新增配置 `max_pbo: float = 0.5`
- [x] 高過擬合風險自動降級

**驗收**：
- [x] OverfitDetector 正確初始化
- [x] 過擬合檢查已整合到驗證流程
- [x] 高風險策略會被降級

---

## P1: 兩階段參數搜索

### Task 6: 兩階段搜索實作 ✅

**檔案**：`src/automation/ultimate_loop.py`

**修改**：
- [x] `_optimize_strategy()` 支援兩階段搜索
- [x] `_two_stage_optimize()` 新方法
- [x] `_expand_param_space()` 擴展參數步長
- [x] `_find_promising_region()` 找有效區域

**配置**：
- [x] `two_stage_search: bool = True`
- [x] `coarse_trials: int = 20`
- [x] `coarse_step_multiplier: float = 3.0`
- [x] `fine_trials: int = 50`

**驗收**：
- [x] 兩階段正確執行
- [x] 粗搜索使用大步長
- [x] 細搜索在有效區域內（±30%）

---

## P2: 其他對齊項目

### Task 7: 市場狀態回傳值修復 ✅

**檔案**：`src/automation/ultimate_loop.py`

**修改**：
- [x] `_analyze_market_state()` 回傳實際狀態
- [x] 移除 `return None`

**驗收**：
- [x] 市場狀態影響策略選擇
- [x] 狀態被記錄到學習系統

---

### Task 8: 參數預生成機制 ✅

**檔案**：`src/automation/ultimate_loop.py`、`src/automation/ultimate_config.py`

**修改**：
- [x] 新增 `_narrow_param_space_from_history()` 方法
- [x] 在 `_optimize_strategy()` 中整合參數預生成
- [x] 基於現有最佳參數 ±30% 範圍
- [x] 新增配置 `use_param_pregeneration`、`param_pregeneration_ratio`

**參考**：`參數優化` Skill

**驗收**：
- [x] 參數在合理範圍內變異
- [x] 避免完全隨機

---

### Task 9: 洞察觸發條件驗證 ✅

**檔案**：`src/automation/ultimate_loop.py`

**修改**：
- [x] 更新 `_generate_insights()` 方法
- [x] 新增 CLAUDE.md 規定的所有觸發條件常數
- [x] 傳遞 `overfit_probability` 參數到洞察生成

**觸發條件**（CLAUDE.md 規定，已實作）：
```python
INSIGHT_HIGH_SHARPE_THRESHOLD = 2.0      # Sharpe > 2.0 → 記錄成功
INSIGHT_LOW_SHARPE_THRESHOLD = 0.5       # Sharpe < 0.5 → 記錄失敗
INSIGHT_HIGH_DRAWDOWN_THRESHOLD = 0.25   # MaxDD > 25% → 記錄風險
INSIGHT_OVERFIT_THRESHOLD = 0.30         # MC 失敗率 > 30% → 記錄過擬合
```

**驗收**：
- [x] 洞察正確觸發
- [x] 所有 CLAUDE.md 規定條件已實現

---

## 任務依賴

```
Task 1 (WalkForwardAnalyzer)
    ↓
Task 2 (整合 Walk-Forward)
    ↓
Task 3 (交易筆數門檻)
    ↓
Task 4 (OverfitDetector) ───→ Task 5 (整合 PBO)
    ↓
Task 6 (兩階段搜索)
    ↓
Task 7, 8, 9 (可並行)
```

## 驗收標準

| 指標 | 目標值 |
|------|--------|
| 迭代間 Sharpe 變異 | > 0 |
| WFA 效率 | >= 50% |
| 交易筆數門檻 | >= 30 |
| PBO 上限 | < 50% |
| 參數搜索覆蓋率 | 2x 原本 |
