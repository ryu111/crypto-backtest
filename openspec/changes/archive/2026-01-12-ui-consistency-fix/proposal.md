# UI Consistency Fix - Proposal

## Why

當前 UI 存在多個一致性問題：

1. **設計系統分散**：`Home.py` 自訂 CSS 與 `styles.py` 重複，缺乏統一管理
2. **Dark Mode 缺失**：所有頁面硬編碼 light mode 顏色，無法切換主題
3. **數據處理不統一**：部分頁面使用模擬資料，錯誤處理不一致
4. **視覺不一致**：圖表配色、間距、字體未統一標準化

這些問題影響：
- 使用者體驗（無法切換 dark mode）
- 維護成本（樣式散落各處）
- 專業性（視覺不一致）

## What Changes

### 1. Design System 統一化
- 建立 `ui/design_tokens.py`（包含 light/dark 兩套配色）
- 移除 `Home.py` 內嵌 CSS
- 所有頁面統一使用 `get_common_css()`

### 2. Dark Mode 實作
- CSS Variables 定義（`:root` 和 `[data-theme="dark"]`）
- Streamlit Session State 儲存主題選擇
- Sidebar 主題切換器
- Plotly 圖表動態配色

### 3. 數據載入標準化
- 統一使用 `ui/utils/data_loader.py`
- 所有頁面加入 loading spinner
- 標準化錯誤提示（無資料時顯示引導而非模擬資料）

### 4. 視覺標準化
- Plotly 圖表統一配色（`ui/chart_config.py`）
- 間距使用 Design Tokens（`--spacing-*`）
- 字體大小標準化

## Impact

- Affected specs: `ui/` 下所有頁面
- Affected code:
  - `ui/Home.py`（移除內嵌 CSS）
  - `ui/styles.py`（增強為完整 design system）
  - `ui/pages/*.py`（統一套用樣式）
  - **新增**: `ui/design_tokens.py`
  - **新增**: `ui/chart_config.py`
  - **新增**: `ui/theme_switcher.py`

**Breaking Changes**: 無（向下相容，只是增強）
