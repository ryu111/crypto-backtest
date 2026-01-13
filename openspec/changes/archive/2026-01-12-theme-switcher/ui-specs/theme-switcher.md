# Theme Switcher 主題切換元件設計規格

## 📋 需求理解

**目標**：為 Streamlit 應用提供一個簡單易用的主題切換功能，讓使用者可以在 Light/Dark 模式間切換。

**使用者**：
- 回測系統的使用者（交易員、研究員）
- 可能長時間盯著數據圖表
- 需要根據環境光線或個人偏好調整介面亮度

**關鍵互動**：
- 點擊圖示切換主題
- 即時生效，無需重新載入頁面
- 狀態持久化（跨 session）

**UX 考量（依據 psychology.md）**：
- **Jakob's Law**：使用熟悉的 🌞/🌙 圖示（使用者熟悉的模式）
- **Fitts's Law**：按鈕放在 sidebar 頂部，容易點擊
- **即時回饋**：點擊後立即切換，使用 st.rerun() 確保狀態更新
- **Von Restorff Effect**：圖示清晰可辨，與其他 sidebar 內容有視覺區隔

---

## 📐 LAYOUT

**佈局模式**：Inline - 水平排列（圖示 + 文字）

**位置**：
- sidebar 頂部，在標題下方
- 獨立區塊，使用 `---` 分隔

**間距**：
```python
# 使用 Streamlit columns 實現水平佈局
# 內部間距由 Streamlit 自動處理
# 外部間距使用 st.markdown("---") 創造視覺分組
```

**容器結構**：
```
sidebar
├── 標題 "📊 AI 合約回測"
├── theme_switcher（本元件）
│   ├── [🌞/🌙] 圖示按鈕
│   └── "Light/Dark" 文字標籤
├── st.markdown("---") 分隔線
└── 其他 sidebar 內容
```

---

## 🎨 VISUAL

**色彩方案**：
- 使用 Streamlit 內建主題系統（不需自定義 CSS variables）
- Light Mode：Streamlit 預設配色
- Dark Mode：Streamlit dark 主題

**圖示選擇（基於 emotional-design.md 的 Visceral 層次）**：
```python
THEME_ICONS = {
    "light": "🌞",  # 太陽 - 直覺代表明亮
    "dark": "🌙"    # 月亮 - 直覺代表黑暗
}
```

**按鈕樣式**：
- 使用 `st.button()` 預設樣式（符合 Streamlit 一致性）
- 當前主題圖示顯示在按鈕上
- 文字標籤說明當前模式

**視覺層級**：
```
主要：圖示按鈕（可點擊）
次要：文字標籤（狀態說明）
```

---

## 🔄 STATES

**主題狀態**：
```python
# st.session_state 儲存
st.session_state.theme: Literal["light", "dark"]
```

**元件狀態**：

| 狀態 | 視覺 | 行為 |
|------|------|------|
| Light Mode Active | 🌞 圖示 | 點擊 → 切換至 Dark |
| Dark Mode Active | 🌙 圖示 | 點擊 → 切換至 Light |
| Hover（Streamlit 自動） | 輕微高亮 | Streamlit 預設 |
| Click Feedback | Streamlit 預設 | 即時切換 + rerun |

**狀態轉換（基於 microinteractions.md）**：
```
Trigger: 點擊按鈕
    ↓
Rules: 切換 session_state.theme
    ↓
Feedback:
    1. 圖示即時變化（🌞 ↔ 🌙）
    2. st.rerun() 重新渲染頁面
    ↓
Result: 頁面以新主題顯示
```

---

## 📱 RESPONSIVE

**主要裝置**：Desktop（Streamlit 主要用於桌面瀏覽器）

**Sidebar 行為**：
- Desktop：Sidebar 預設展開
- Mobile/Tablet：可收合（Streamlit 預設行為）

**元件適應**：
- 固定在 sidebar，不隨頁面滾動消失
- 圖示大小使用 emoji（自動適應字體大小）

---

## 🎯 互動流程（基於 microinteractions.md 四要素）

### 1. Trigger（觸發器）
- **使用者觸發**：點擊按鈕

### 2. Rules（規則）
```python
if st.session_state.theme == "light":
    st.session_state.theme = "dark"
else:
    st.session_state.theme = "light"
```

### 3. Feedback（回饋）
- **視覺回饋**：圖示變化（🌞 → 🌙 或 🌙 → 🌞）
- **系統回饋**：`st.rerun()` 重新渲染頁面
- **持續時間**：< 100ms（即時感）

### 4. Loops & Modes（循環與模式）
- **模式**：Light/Dark 兩種模式
- **持久化**：使用 `st.session_state`（session 級別）

---

## 🛠️ 技術規格

### API 設計

```python
# 初始化
def init_theme():
    """初始化主題狀態（預設 Light）"""
    if 'theme' not in st.session_state:
        st.session_state['theme'] = 'light'

# 獲取當前主題
def get_current_theme() -> Literal["light", "dark"]:
    """返回當前主題"""
    init_theme()
    return st.session_state['theme']

# 渲染切換器
def render_theme_switcher():
    """渲染主題切換按鈕（放在 sidebar）"""
    init_theme()

    # 當前主題
    current_theme = st.session_state['theme']

    # 圖示映射
    icons = {"light": "🌞", "dark": "🌙"}
    labels = {"light": "Light Mode", "dark": "Dark Mode"}

    # 按鈕
    if st.button(
        f"{icons[current_theme]} {labels[current_theme]}",
        key="theme_switcher",
        use_container_width=True
    ):
        # 切換主題
        st.session_state['theme'] = (
            'dark' if current_theme == 'light' else 'light'
        )
        st.rerun()
```

### 使用範例

```python
# 在 app.py 或任何頁面的 sidebar
with st.sidebar:
    st.title("📊 AI 合約回測")
    render_theme_switcher()  # ← 添加主題切換器
    st.markdown("---")
    # ... 其他 sidebar 內容
```

---

## ⚠️ 限制與注意事項

### Streamlit 主題系統限制

**Streamlit 不支援動態主題切換的原因**：
- Streamlit 主題由 `.streamlit/config.toml` 配置
- 主題在應用啟動時載入，無法在 runtime 動態切換
- 官方文件建議：主題需要在配置檔中預先設定

**替代方案**：
1. **CSS Variables 模擬**（推薦）
   - 定義 Light/Dark 的 CSS variables
   - 根據 `st.session_state.theme` 注入對應 CSS

2. **預設主題 + 使用者偏好記錄**
   - 記錄使用者偏好到 session_state
   - 提示使用者：「下次啟動時生效」

### 本設計規格採用方案 1（CSS Variables）

```python
def get_theme_css(theme: str) -> str:
    """返回對應主題的 CSS"""
    if theme == "dark":
        return """
        <style>
        [data-testid="stApp"] {
            background-color: #0e1117;
            color: #fafafa;
        }
        [data-testid="stSidebar"] {
            background-color: #262730;
        }
        /* 其他深色模式樣式 */
        </style>
        """
    else:
        return """
        <style>
        [data-testid="stApp"] {
            background-color: #ffffff;
            color: #262730;
        }
        /* 其他淺色模式樣式 */
        </style>
        """

def apply_theme():
    """應用當前主題的 CSS"""
    theme = get_current_theme()
    st.markdown(get_theme_css(theme), unsafe_allow_html=True)
```

---

## ✅ Checklist

### 功能
- [x] 初始化函數正確設定預設主題
- [x] 切換函數正確切換狀態
- [x] `st.rerun()` 確保頁面即時更新
- [x] 圖示正確顯示當前狀態

### UX
- [x] 按鈕位置符合 Fitts's Law（易點擊）
- [x] 圖示符合 Jakob's Law（熟悉模式）
- [x] 即時回饋 < 100ms
- [x] 視覺層級清晰

### 整合
- [x] 與現有 sidebar 佈局協調
- [x] 不影響其他頁面功能
- [x] CSS 不與現有樣式衝突

---

## 📝 開發筆記

**為什麼選擇 session_state 而非 cookie/localStorage？**
- Streamlit 的 session_state 是最簡單的狀態管理方式
- 對於 PoC/內部工具足夠（不需跨 session 持久化）
- 未來若需要持久化，可搭配 `streamlit-cookies-manager` 套件

**為什麼使用 emoji 而非 icon library？**
- 簡化依賴（無需引入額外套件）
- 🌞🌙 在所有平台都有良好支援
- 符合 Streamlit 的輕量化理念

**時間估算**：
- 核心功能實作：15 分鐘
- CSS 主題樣式：30 分鐘
- 測試與調整：15 分鐘
- **總計**：約 1 小時

---

## 參考

- UI Skill → `references/tokens.md`（雖然 Streamlit 不直接用 CSS variables，但概念一致）
- UX Skill → `references/psychology.md`（Jakob's Law、Fitts's Law）
- UX Skill → `references/microinteractions.md`（四要素設計）
- Streamlit 文件：https://docs.streamlit.io/library/advanced-features/theming
