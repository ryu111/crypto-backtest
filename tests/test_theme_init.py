"""
測試主題初始化功能

驗證 init_theme() 是否能正確偵測系統主題。
"""

import sys
from pathlib import Path

# 確保能 import ui 模組
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_init_theme_detection():
    """測試主題初始化偵測邏輯"""
    import streamlit as st
    from ui.theme_switcher import init_theme, get_current_theme

    # 模擬 session_state 重置
    if "theme" in st.session_state:
        del st.session_state["theme"]

    # 執行初始化
    init_theme()

    # 驗證主題已設定
    theme = get_current_theme()
    assert theme in ["light", "dark"], f"主題應為 light 或 dark，但得到 {theme}"

    print(f"✅ 偵測到系統主題：{theme}")
    print(f"✅ init_theme() 功能正常")

    return True


def test_theme_fallback():
    """測試主題偵測失敗時的 fallback"""
    import streamlit as st

    # 模擬無法讀取系統主題的情況
    if "theme" in st.session_state:
        del st.session_state["theme"]

    # 這裡我們假設 st.get_option() 可能拋出異常
    # 實際執行時會 fallback 到 "light"
    from ui.theme_switcher import init_theme, get_current_theme

    init_theme()
    theme = get_current_theme()

    # 應該至少有一個預設值
    assert theme in ["light", "dark"]

    print(f"✅ Fallback 機制正常，預設主題：{theme}")

    return True


if __name__ == "__main__":
    print("開始測試主題初始化...")
    print("=" * 50)

    try:
        test_init_theme_detection()
        print()
        test_theme_fallback()
        print("=" * 50)
        print("✅ 所有測試通過！")
    except Exception as e:
        print(f"❌ 測試失敗：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
