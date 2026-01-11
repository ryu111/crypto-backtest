"""UI 共用樣式和顏色常數

提供一致的視覺樣式，包括評級顏色、CSS 樣式等。
"""

from typing import Dict

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


def get_common_css() -> str:
    """
    返回共用 CSS 樣式

    Returns:
        CSS 字串
    """
    return """
<style>
/* 隱藏 Streamlit 自動產生的頁面導航（英文）*/
[data-testid="stSidebarNav"] {
    display: none !important;
}

/* Design Tokens */
:root {
    /* Colors */
    --color-primary: #2563eb;
    --color-primary-hover: #1d4ed8;
    --color-surface: #ffffff;
    --color-surface-raised: #f9fafb;
    --color-text: #111827;
    --color-text-secondary: #6b7280;
    --color-border: #e5e7eb;
    --color-success: #22c55e;
    --color-warning: #eab308;
    --color-error: #ef4444;

    /* Spacing */
    --spacing-sm: 0.5rem;
    --spacing-md: 1rem;
    --spacing-lg: 1.5rem;

    /* Radius */
    --radius-md: 0.375rem;
    --radius-lg: 0.5rem;
}

/* 評級徽章 */
.grade-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.875rem;
}

.grade-A {
    background: #d1fae5;
    color: #065f46;
}

.grade-B {
    background: #dbeafe;
    color: #1e40af;
}

.grade-C {
    background: #fef3c7;
    color: #92400e;
}

.grade-D {
    background: #fed7aa;
    color: #9a3412;
}

.grade-F {
    background: #fee2e2;
    color: #991b1b;
}

/* 指標卡片 */
.metric-card {
    background: var(--color-surface-raised);
    padding: var(--spacing-md);
    border-radius: var(--radius-lg);
    border: 1px solid var(--color-border);
}

/* 表格樣式 */
.dataframe {
    border-radius: var(--radius-md) !important;
    overflow: hidden;
}

/* 滑桿樣式 */
.stSlider > div > div > div {
    background: var(--color-primary);
}
</style>
"""
