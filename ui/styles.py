"""UI 共用樣式和顏色常數

提供一致的視覺樣式，包括評級顏色、CSS 樣式等。
"""

from typing import Dict
from .design_tokens import get_css_variables

# 評級顏色配置
GRADE_COLORS: Dict[str, Dict[str, str]] = {
    "A": {
        "bg": "#d1fae5",
        "text": "#065f46",
        "hex": "#10b981"
    },
    "B": {
        "bg": "#dbeafe",
        "text": "#1e40af",
        "hex": "#3b82f6"
    },
    "C": {
        "bg": "#fef3c7",
        "text": "#92400e",
        "hex": "#f59e0b"
    },
    "D": {
        "bg": "#fed7aa",
        "text": "#9a3412",
        "hex": "#f97316"
    },
    "F": {
        "bg": "#fee2e2",
        "text": "#991b1b",
        "hex": "#ef4444"
    },
}


def get_grade_badge_style(grade: str) -> str:
    """
    取得評級徽章樣式

    Args:
        grade: 評級 (A/B/C/D/F)

    Returns:
        CSS class 字串
    """
    if grade not in GRADE_COLORS:
        grade = "F"

    return f"grade-{grade}"


def get_common_css(theme: str = "light") -> str:
    """
    返回共用 CSS 樣式

    Args:
        theme: 主題名稱 ("light" 或 "dark")

    Returns:
        CSS 字串
    """
    # 從 design_tokens 取得動態 CSS Variables
    css_vars = get_css_variables(theme)

    return f"""
<style>
/* 隱藏 Streamlit 自動產生的頁面導航（英文）*/
[data-testid="stSidebarNav"] {{
    display: none !important;
}}

/* Design Tokens（動態生成） */
{css_vars}

/* 評級徽章 */
.grade-badge {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: var(--radius-full);
    font-weight: 600;
    font-size: 0.875rem;
}}

.grade-A {{
    background: var(--grade-a-bg);
    color: var(--grade-a-text);
}}

.grade-B {{
    background: var(--grade-b-bg);
    color: var(--grade-b-text);
}}

.grade-C {{
    background: var(--grade-c-bg);
    color: var(--grade-c-text);
}}

.grade-D {{
    background: var(--grade-d-bg);
    color: var(--grade-d-text);
}}

.grade-F {{
    background: var(--grade-f-bg);
    color: var(--grade-f-text);
}}

/* 指標卡片 */
.metric-card {{
    background: var(--surface-raised);
    padding: var(--spacing-md);
    border-radius: var(--radius-lg);
    border: 1px solid var(--border);
}}

/* 表格樣式 */
.dataframe {{
    border-radius: var(--radius-md) !important;
    overflow: hidden;
}}

/* 滑桿樣式 */
.stSlider > div > div > div {{
    background: var(--primary);
}}
</style>
"""
