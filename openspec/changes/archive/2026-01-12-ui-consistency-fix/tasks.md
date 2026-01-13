# ui-consistency-fix Implementation Tasks

## Progress
- Total: 24 tasks
- Completed: 24
- Status: COMPLETED

---

## 1. Foundation - Design System (sequential)
- [x] 1.1 建立 ui/design_tokens.py（Light + Dark 配色） | files: ui/design_tokens.py | agent: DESIGNER
- [x] 1.2 建立 ui/chart_config.py（Plotly 統一配置） | files: ui/chart_config.py | agent: DESIGNER
- [x] 1.3 建立 ui/theme_switcher.py（主題切換元件） | files: ui/theme_switcher.py | agent: DESIGNER

## 2. Core CSS Refactor (sequential, depends: 1)
- [x] 2.1 重構 ui/styles.py 使用 design tokens | files: ui/styles.py
- [x] 2.2 更新 ui/utils.py 的 render_sidebar_navigation | files: ui/utils.py
- [x] 2.3 移除 Home.py 內嵌 CSS，改用統一樣式 | files: ui/Home.py

## 3. Page Updates - Dashboard (sequential, depends: 2)
- [x] 3.1 更新 Dashboard 頁面套用 dark mode | files: ui/pages/1_📊_Dashboard.py
- [x] 3.2 Dashboard 圖表改用統一 chart_config | files: ui/pages/1_📊_Dashboard.py
- [x] 3.3 加入 loading 狀態和錯誤處理 | files: ui/pages/1_📊_Dashboard.py

## 4. Page Updates - Strategies (sequential, depends: 2)
- [x] 4.1 更新 Strategies 頁面套用 dark mode | files: ui/pages/2_Strategies.py
- [x] 4.2 Strategies 圖表改用統一 chart_config | files: ui/pages/2_Strategies.py
- [x] 4.3 移除模擬資料邏輯，改用標準錯誤處理 | files: ui/pages/2_Strategies.py
- [x] 4.4 權益曲線和月度報酬圖表標準化 | files: ui/pages/2_Strategies.py

## 5. Page Updates - Comparison (sequential, depends: 2)
- [x] 5.1 更新 Comparison 頁面套用 dark mode | files: ui/pages/3_Comparison.py
- [x] 5.2 移除所有模擬資料（load_strategy_results） | files: ui/pages/3_Comparison.py
- [x] 5.3 改用 data_loader 載入真實資料 | files: ui/pages/3_Comparison.py
- [x] 5.4 圖表改用統一 chart_config | files: ui/pages/3_Comparison.py

## 6. Page Updates - Validation (sequential, depends: 2)
- [x] 6.1 更新 Validation 頁面套用 dark mode | files: ui/pages/4_Validation.py
- [x] 6.2 Validation 圖表改用統一 chart_config | files: ui/pages/4_Validation.py
- [x] 6.3 整合真實驗證資料（移除模擬資料） | files: ui/pages/4_Validation.py

## 7. Page Updates - Risk Dashboard (sequential, depends: 2)
- [x] 7.1 更新 Risk Dashboard 套用 dark mode | files: ui/pages/5_RiskDashboard.py
- [x] 7.2 Risk 圖表改用統一 chart_config | files: ui/pages/5_RiskDashboard.py
- [x] 7.3 Kelly/Portfolio 圖表標準化 | files: ui/pages/5_RiskDashboard.py

## 8. Testing & Validation (parallel, depends: 7)
- [x] 8.1 測試 Light Mode 所有頁面顯示 | agent: TESTER
- [x] 8.2 測試 Dark Mode 所有頁面顯示 | agent: TESTER
- [x] 8.3 測試主題切換流暢性 | agent: TESTER
- [x] 8.4 測試圖表在兩種主題下的可讀性 | agent: TESTER

---

## 任務分配說明

- **Phase 1-2**: Foundation，必須按順序完成
- **Phase 3-7**: 各頁面獨立更新，可考慮並行（但依賴 Phase 2）
- **Phase 8**: 測試階段，4 個測試可並行執行

## 設計規範參考

- DESIGNER 任務需讀取：
  - `~/.claude/skills/ui/references/tokens.md`
  - `~/.claude/skills/ui/references/components.md`
  - `~/.claude/skills/ux/references/psychology.md`（主題切換 UX）

## 預估時間

- Foundation: 3 個 D→R→T 循環
- CSS Refactor: 3 個 D→R→T 循環
- 5 個頁面更新: 約 15 個 D→R→T 循環（每頁 2-4 個循環）
- Testing: 4 個並行測試

**總計約 21 個 D→R→T 循環**
