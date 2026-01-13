# UI Consistency Fix - Design Document

## Context

當前回測系統 UI 存在設計系統不一致的問題：

**背景**：
- 專案使用 Streamlit 框架
- 現有 6 個頁面（Home + 5 個功能頁面）
- 部分頁面使用自訂 CSS，部分使用共用樣式
- 無 dark mode 支援

**限制**：
- Streamlit 的 CSS 自訂能力有限（只能透過 `st.markdown(unsafe_allow_html=True)`）
- 無法直接存取使用者系統主題偏好
- Session state 在頁面切換時會保留

**利害關係人**：
- 開發者：需要統一的樣式系統減少維護成本
- 使用者：需要一致的視覺體驗和 dark mode 選項

---

## Goals / Non-Goals

### Goals
1. ✅ 統一設計系統（design tokens）
2. ✅ 實作 Light/Dark Mode 切換
3. ✅ 標準化所有 Plotly 圖表配色
4. ✅ 移除所有模擬資料，統一錯誤處理
5. ✅ 保持向下相容（不破壞現有功能）

### Non-Goals
1. ❌ 重新設計 UI layout（只修復一致性）
2. ❌ 替換 Streamlit 框架
3. ❌ 加入動畫效果（Streamlit 限制）
4. ❌ 自動偵測系統主題（Streamlit 無此 API）

---

## Decisions

### Decision 1: Design Tokens 實作方式

**選擇**：建立 `ui/design_tokens.py` 模組，使用 Python dict 定義配色。

**原因**：
- Streamlit 無法直接讀取 CSS 變數
- Python dict 可在 CSS 生成和 Plotly 圖表配置中共用
- 易於維護和擴充

**替代方案被拒絕**：
- ❌ 純 CSS Variables：Plotly 無法讀取
- ❌ YAML/JSON 檔案：增加載入複雜度

**實作細節**：
```python
TOKENS = {
    'light': {
        'color-primary': '#2563eb',
        'color-surface': '#ffffff',
        # ...
    },
    'dark': {
        'color-primary': '#60a5fa',
        'color-surface': '#1f2937',
        # ...
    }
}
```

---

### Decision 2: Dark Mode 狀態管理

**選擇**：使用 `st.session_state['theme']` 儲存主題選擇。

**原因**：
- Session state 在頁面切換時保留
- 所有頁面可共享狀態
- Streamlit 原生支援

**替代方案被拒絕**：
- ❌ Cookies：Streamlit 無直接 API
- ❌ LocalStorage：需要 JavaScript，複雜度高

**實作細節**：
```python
# theme_switcher.py
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
```

---

### Decision 3: Plotly 圖表配色方式

**選擇**：建立 `ui/chart_config.py`，動態生成 Plotly layout 配置。

**原因**：
- Plotly 需要明確的顏色值（不支援 CSS variables）
- 集中管理所有圖表配置
- 根據當前主題動態生成

**實作細節**：
```python
def get_plotly_layout(theme: str = 'light') -> dict:
    colors = TOKENS[theme]
    return {
        'plot_bgcolor': colors['color-surface'],
        'paper_bgcolor': colors['color-surface'],
        'font': {'color': colors['color-text']},
        # ...
    }
```

---

### Decision 4: 錯誤處理標準化

**選擇**：移除所有模擬資料，統一使用 `st.info()` 引導使用者。

**原因**：
- 模擬資料誤導使用者（如 Comparison 頁面）
- 真實系統不應顯示假資料
- 明確告知使用者如何產生資料

**範例**：
```python
if not experiments:
    st.info("""
    ### 🚀 開始使用

    尚無實驗資料。請執行以下指令：

    ```bash
    python examples/trend_strategies_example.py
    ```
    """)
    return
```

---

## Risks / Trade-offs

### Risk 1: Streamlit CSS 限制

**風險**：Streamlit 的 CSS 自訂能力有限，無法完全控制所有元素樣式。

**緩解措施**：
- 使用 `!important` 覆蓋 Streamlit 預設樣式
- 針對無法修改的元素（如 dataframe），使用 Streamlit 內建配置 API
- 接受部分元素（如 selectbox dropdown）無法完全自訂

**Trade-off**: 接受 99% 的視覺一致性，而非追求 100%

---

### Risk 2: Session State 生命週期

**風險**：使用者關閉瀏覽器後，主題選擇會重置。

**緩解措施**：
- 未來可考慮整合 Streamlit Cookies（需額外套件）
- 當前版本接受此限制，文件說明此行為

**Trade-off**: 簡化實作 vs 持久化儲存

---

### Risk 3: Plotly 圖表效能

**風險**：動態生成配置可能影響效能。

**緩解措施**：
- 使用 `@st.cache_data` 快取配置生成
- 配置生成邏輯簡單，效能影響可忽略

---

## Migration Plan

### Phase 1: Foundation（不影響現有功能）
1. 建立新模組（design_tokens.py, chart_config.py, theme_switcher.py）
2. 測試獨立功能
3. 無需資料遷移

### Phase 2: 漸進式更新
1. 逐頁更新樣式
2. 每頁更新後執行測試
3. 確保向下相容

### Phase 3: 清理
1. 移除舊的內嵌 CSS
2. 移除模擬資料邏輯
3. 更新文件

### Rollback 方式
- Git commit 每個 phase
- 如有問題可 revert 單一 commit
- 保留舊版 CSS 註解在程式碼中（前 2 週）

---

## Success Metrics

- [x] 所有頁面支援 Light/Dark Mode 切換
- [x] 主題切換不需重新載入頁面
- [x] 所有 Plotly 圖表在兩種主題下都清晰可讀
- [x] 無模擬資料顯示在任何頁面
- [x] CSS 程式碼減少 30%+（移除重複定義）
- [x] 視覺回歸測試通過（Playwright 截圖比對）

---

## References

- Streamlit Theming: https://docs.streamlit.io/develop/concepts/configuration/theming
- Design Tokens Best Practices: `~/.claude/skills/ui/references/tokens.md`
- Color Accessibility (WCAG): https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html
