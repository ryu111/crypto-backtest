"""
主題切換元件

提供 Light/Dark 模式切換功能，使用 Streamlit session state 管理狀態。
"""

import streamlit as st
from typing import Literal

ThemeType = Literal["light", "dark"]


def init_theme() -> None:
    """初始化主題狀態

    自動偵測系統主題設定（Dark Mode / Light Mode）。
    優先順序：
    1. 用戶已選擇的主題（session_state）
    2. Streamlit 系統主題設定
    3. 預設為 light

    應在每個頁面開始時呼叫。
    """
    if "theme" not in st.session_state:
        # 嘗試從 Streamlit config 讀取系統主題
        try:
            streamlit_theme = st.get_option("theme.base")
            if streamlit_theme == "dark":
                st.session_state["theme"] = "dark"
            else:
                st.session_state["theme"] = "light"
        except Exception:
            # 如果無法讀取，預設為 light
            st.session_state["theme"] = "light"


def get_current_theme() -> ThemeType:
    """取得當前主題

    Returns:
        'light' 或 'dark'
    """
    init_theme()
    return st.session_state["theme"]


def set_theme(theme: ThemeType) -> None:
    """設定主題

    Args:
        theme: 'light' 或 'dark'
    """
    st.session_state["theme"] = theme


def toggle_theme() -> None:
    """切換主題

    如果當前是 light 就切換到 dark，反之亦然。
    """
    current = get_current_theme()
    new_theme = "dark" if current == "light" else "light"
    set_theme(new_theme)


def render_theme_switcher(location: str = "sidebar") -> None:
    """渲染主題切換按鈕

    Args:
        location: 'sidebar' 或 'main'（放置位置）

    使用 emoji 圖標讓用戶一目了然：
    - 🌞 Light Mode
    - 🌙 Dark Mode
    """
    init_theme()
    current = get_current_theme()

    # 圖標和標籤
    if current == "light":
        icon = "🌙"
        label = "Dark Mode"
    else:
        icon = "🌞"
        label = "Light Mode"

    # 根據位置渲染按鈕
    if location == "sidebar":
        col1, col2 = st.sidebar.columns([1, 3])
        with col1:
            if st.sidebar.button(icon, key="theme_toggle_btn", help=f"切換到 {label}"):
                toggle_theme()
                st.rerun()
        with col2:
            st.sidebar.caption(f"當前：{'☀️ 亮色' if current == 'light' else '🌙 暗色'}")
    else:
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button(icon, key="theme_toggle_btn", help=f"切換到 {label}"):
                toggle_theme()
                st.rerun()
        with col2:
            st.caption(f"當前：{'☀️ 亮色' if current == 'light' else '🌙 暗色'}")


def render_theme_toggle() -> None:
    """渲染簡潔的主題切換 toggle

    使用 checkbox 風格的 toggle，適合放在 sidebar 頂部。
    """
    init_theme()
    current = get_current_theme()

    # 使用 toggle
    is_dark = current == "dark"

    new_is_dark = st.sidebar.toggle(
        "🌙 Dark Mode",
        value=is_dark,
        key="theme_toggle",
        help="切換亮色/暗色模式"
    )

    # 如果狀態改變，更新並重新渲染
    if new_is_dark != is_dark:
        set_theme("dark" if new_is_dark else "light")
        st.rerun()


def apply_theme_css() -> str:
    """取得當前主題的 CSS

    Returns:
        CSS 字串，可用於 st.markdown

    範例:
        ```python
        st.markdown(f'<style>{apply_theme_css()}</style>', unsafe_allow_html=True)
        ```
    """
    from .design_tokens import get_css_variables

    theme = get_current_theme()
    return get_css_variables(theme)


def apply_theme() -> None:
    """應用當前主題的 CSS 到頁面

    在每個頁面開始時呼叫，自動注入主題 CSS。

    範例:
        ```python
        import streamlit as st
        from ui.theme_switcher import apply_theme, render_theme_toggle

        st.set_page_config(...)
        apply_theme()
        render_theme_toggle()
        ```
    """
    css = apply_theme_css()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ============================================================================
# 便利函數
# ============================================================================

def is_dark_mode() -> bool:
    """檢查是否為暗色模式

    Returns:
        True 如果當前是 dark mode
    """
    return get_current_theme() == "dark"


def is_light_mode() -> bool:
    """檢查是否為亮色模式

    Returns:
        True 如果當前是 light mode
    """
    return get_current_theme() == "light"
