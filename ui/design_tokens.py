"""設計 Token 系統

提供完整的設計 token，支援 Light/Dark 主題切換，
以及為 Plotly 圖表提供專用配色方案。

架構：
- Primitive Tokens（原始值）
- Semantic Tokens（語意化）
- Component Tokens（元件專用）
"""

from typing import Dict, Any


# ============================================================================
# Layer 1: Primitive Tokens（原始值）
# ============================================================================

PRIMITIVE_COLORS = {
    # Blue Scale（主色系）
    "blue-50": "#eff6ff",
    "blue-100": "#dbeafe",
    "blue-200": "#bfdbfe",
    "blue-300": "#93c5fd",
    "blue-400": "#60a5fa",
    "blue-500": "#3b82f6",
    "blue-600": "#2563eb",
    "blue-700": "#1d4ed8",
    "blue-800": "#1e40af",
    "blue-900": "#1e3a8a",
    "blue-950": "#172554",

    # Gray Scale（中性色）
    "gray-50": "#f9fafb",
    "gray-100": "#f3f4f6",
    "gray-200": "#e5e7eb",
    "gray-300": "#d1d5db",
    "gray-400": "#9ca3af",
    "gray-500": "#6b7280",
    "gray-600": "#4b5563",
    "gray-700": "#374151",
    "gray-800": "#1f2937",
    "gray-900": "#111827",
    "gray-950": "#030712",

    # Green Scale（成功）
    "green-50": "#f0fdf4",
    "green-100": "#dcfce7",
    "green-200": "#bbf7d0",
    "green-300": "#86efac",
    "green-400": "#4ade80",
    "green-500": "#22c55e",
    "green-600": "#16a34a",
    "green-700": "#15803d",
    "green-800": "#166534",
    "green-900": "#14532d",

    # Red Scale（錯誤/看跌）
    "red-50": "#fef2f2",
    "red-100": "#fee2e2",
    "red-200": "#fecaca",
    "red-300": "#fca5a5",
    "red-400": "#f87171",
    "red-500": "#ef4444",
    "red-600": "#dc2626",
    "red-700": "#b91c1c",
    "red-800": "#991b1b",
    "red-900": "#7f1d1d",

    # Yellow/Orange Scale（警告）
    "yellow-50": "#fefce8",
    "yellow-100": "#fef3c7",
    "yellow-200": "#fde68a",
    "yellow-300": "#fcd34d",
    "yellow-400": "#fbbf24",
    "yellow-500": "#f59e0b",
    "yellow-600": "#d97706",
    "yellow-700": "#b45309",
    "yellow-800": "#92400e",
    "yellow-900": "#78350f",

    "orange-50": "#fff7ed",
    "orange-100": "#ffedd5",
    "orange-200": "#fed7aa",
    "orange-300": "#fdba74",
    "orange-400": "#fb923c",
    "orange-500": "#f97316",
    "orange-600": "#ea580c",
    "orange-700": "#c2410c",
    "orange-800": "#9a3412",
    "orange-900": "#7c2d12",
}

PRIMITIVE_SPACING = {
    "0": "0",
    "px": "1px",
    "0.5": "0.125rem",  # 2px
    "1": "0.25rem",     # 4px
    "1.5": "0.375rem",  # 6px
    "2": "0.5rem",      # 8px
    "2.5": "0.625rem",  # 10px
    "3": "0.75rem",     # 12px
    "3.5": "0.875rem",  # 14px
    "4": "1rem",        # 16px
    "5": "1.25rem",     # 20px
    "6": "1.5rem",      # 24px
    "8": "2rem",        # 32px
    "10": "2.5rem",     # 40px
    "12": "3rem",       # 48px
    "16": "4rem",       # 64px
}

PRIMITIVE_RADIUS = {
    "none": "0",
    "sm": "0.125rem",   # 2px
    "md": "0.375rem",   # 6px
    "lg": "0.5rem",     # 8px
    "xl": "0.75rem",    # 12px
    "2xl": "1rem",      # 16px
    "full": "9999px",
}

PRIMITIVE_FONT = {
    "xs": "0.75rem",    # 12px
    "sm": "0.875rem",   # 14px
    "base": "1rem",     # 16px
    "lg": "1.125rem",   # 18px
    "xl": "1.25rem",    # 20px
    "2xl": "1.5rem",    # 24px
    "3xl": "1.875rem",  # 30px
    "4xl": "2.25rem",   # 36px
}


# ============================================================================
# Layer 2: Semantic Tokens（語意化）
# ============================================================================

def get_light_tokens() -> Dict[str, Any]:
    """Light 主題語意化 Token"""
    return {
        # 背景（60% - 主色）
        "background": "#ffffff",
        "surface": PRIMITIVE_COLORS["gray-50"],
        "surface-raised": "#ffffff",
        "surface-sunken": PRIMITIVE_COLORS["gray-100"],
        "surface-overlay": "rgba(0, 0, 0, 0.5)",

        # 文字
        "text-primary": PRIMITIVE_COLORS["gray-900"],
        "text-secondary": PRIMITIVE_COLORS["gray-600"],
        "text-muted": PRIMITIVE_COLORS["gray-500"],
        "text-placeholder": PRIMITIVE_COLORS["gray-400"],
        "text-inverse": "#ffffff",
        "text-on-primary": "#ffffff",

        # 邊框
        "border": PRIMITIVE_COLORS["gray-200"],
        "border-strong": PRIMITIVE_COLORS["gray-300"],
        "border-muted": PRIMITIVE_COLORS["gray-100"],
        "border-focus": PRIMITIVE_COLORS["blue-500"],

        # 品牌色（10% - 強調色）
        "primary": PRIMITIVE_COLORS["blue-600"],
        "primary-hover": PRIMITIVE_COLORS["blue-700"],
        "primary-active": PRIMITIVE_COLORS["blue-800"],
        "primary-light": PRIMITIVE_COLORS["blue-50"],
        "primary-muted": PRIMITIVE_COLORS["blue-100"],

        # 功能色
        "success": PRIMITIVE_COLORS["green-500"],
        "success-light": PRIMITIVE_COLORS["green-50"],
        "success-text": PRIMITIVE_COLORS["green-700"],

        "warning": PRIMITIVE_COLORS["yellow-500"],
        "warning-light": PRIMITIVE_COLORS["yellow-50"],
        "warning-text": PRIMITIVE_COLORS["yellow-800"],

        "error": PRIMITIVE_COLORS["red-500"],
        "error-light": PRIMITIVE_COLORS["red-50"],
        "error-text": PRIMITIVE_COLORS["red-700"],

        "info": PRIMITIVE_COLORS["blue-500"],
        "info-light": PRIMITIVE_COLORS["blue-50"],
        "info-text": PRIMITIVE_COLORS["blue-700"],

        # 交易特定
        "bullish": PRIMITIVE_COLORS["green-500"],  # 看漲
        "bearish": PRIMITIVE_COLORS["red-500"],    # 看跌

        # 評級顏色（保持與現有 styles.py 一致）
        "grade-a-bg": "#d1fae5",
        "grade-a-text": "#065f46",
        "grade-b-bg": "#dbeafe",
        "grade-b-text": "#1e40af",
        "grade-c-bg": "#fef3c7",
        "grade-c-text": "#92400e",
        "grade-d-bg": "#fed7aa",
        "grade-d-text": "#9a3412",
        "grade-f-bg": "#fee2e2",
        "grade-f-text": "#991b1b",

        # 間距（語意化）
        "spacing-xs": PRIMITIVE_SPACING["1"],
        "spacing-sm": PRIMITIVE_SPACING["2"],
        "spacing-md": PRIMITIVE_SPACING["4"],
        "spacing-lg": PRIMITIVE_SPACING["6"],
        "spacing-xl": PRIMITIVE_SPACING["8"],

        # 圓角
        "radius-sm": PRIMITIVE_RADIUS["sm"],
        "radius-md": PRIMITIVE_RADIUS["md"],
        "radius-lg": PRIMITIVE_RADIUS["lg"],
        "radius-xl": PRIMITIVE_RADIUS["xl"],
        "radius-full": PRIMITIVE_RADIUS["full"],

        # 陰影
        "shadow-sm": "0 1px 2px rgba(0, 0, 0, 0.05)",
        "shadow-md": "0 4px 6px rgba(0, 0, 0, 0.1)",
        "shadow-lg": "0 10px 15px rgba(0, 0, 0, 0.1)",
        "shadow-xl": "0 20px 25px rgba(0, 0, 0, 0.1)",
    }


def get_dark_tokens() -> Dict[str, Any]:
    """Dark 主題語意化 Token"""
    return {
        # 背景（60% - 主色）
        "background": "#0f172a",        # Slate-900
        "surface": "#1e293b",           # Slate-800
        "surface-raised": "#334155",    # Slate-700
        "surface-sunken": "#0a0f1a",    # 更深
        "surface-overlay": "rgba(0, 0, 0, 0.7)",

        # 文字
        "text-primary": "#f1f5f9",      # Slate-100
        "text-secondary": "#cbd5e1",    # Slate-300
        "text-muted": "#94a3b8",        # Slate-400
        "text-placeholder": "#64748b",  # Slate-500
        "text-inverse": PRIMITIVE_COLORS["gray-900"],
        "text-on-primary": "#ffffff",

        # 邊框
        "border": "#334155",            # Slate-700
        "border-strong": "#475569",     # Slate-600
        "border-muted": "#1e293b",      # Slate-800
        "border-focus": PRIMITIVE_COLORS["blue-400"],

        # 品牌色（深色模式稍微降低飽和度）
        "primary": PRIMITIVE_COLORS["blue-500"],
        "primary-hover": PRIMITIVE_COLORS["blue-400"],
        "primary-active": PRIMITIVE_COLORS["blue-300"],
        "primary-light": PRIMITIVE_COLORS["blue-950"],
        "primary-muted": PRIMITIVE_COLORS["blue-900"],

        # 功能色（深色模式使用較亮的變體）
        "success": PRIMITIVE_COLORS["green-400"],
        "success-light": PRIMITIVE_COLORS["green-900"],
        "success-text": PRIMITIVE_COLORS["green-300"],

        "warning": PRIMITIVE_COLORS["yellow-400"],
        "warning-light": PRIMITIVE_COLORS["yellow-900"],
        "warning-text": PRIMITIVE_COLORS["yellow-300"],

        "error": PRIMITIVE_COLORS["red-400"],
        "error-light": PRIMITIVE_COLORS["red-900"],
        "error-text": PRIMITIVE_COLORS["red-300"],

        "info": PRIMITIVE_COLORS["blue-400"],
        "info-light": PRIMITIVE_COLORS["blue-950"],
        "info-text": PRIMITIVE_COLORS["blue-300"],

        # 交易特定
        "bullish": PRIMITIVE_COLORS["green-400"],
        "bearish": PRIMITIVE_COLORS["red-400"],

        # 評級顏色（深色模式調整）
        "grade-a-bg": "#064e3b",
        "grade-a-text": "#6ee7b7",
        "grade-b-bg": "#1e3a8a",
        "grade-b-text": "#93c5fd",
        "grade-c-bg": "#78350f",
        "grade-c-text": "#fcd34d",
        "grade-d-bg": "#7c2d12",
        "grade-d-text": "#fdba74",
        "grade-f-bg": "#7f1d1d",
        "grade-f-text": "#fca5a5",

        # 間距（語意化）
        "spacing-xs": PRIMITIVE_SPACING["1"],
        "spacing-sm": PRIMITIVE_SPACING["2"],
        "spacing-md": PRIMITIVE_SPACING["4"],
        "spacing-lg": PRIMITIVE_SPACING["6"],
        "spacing-xl": PRIMITIVE_SPACING["8"],

        # 圓角
        "radius-sm": PRIMITIVE_RADIUS["sm"],
        "radius-md": PRIMITIVE_RADIUS["md"],
        "radius-lg": PRIMITIVE_RADIUS["lg"],
        "radius-xl": PRIMITIVE_RADIUS["xl"],
        "radius-full": PRIMITIVE_RADIUS["full"],

        # 陰影（深色模式使用更深的陰影）
        "shadow-sm": "0 1px 2px rgba(0, 0, 0, 0.3)",
        "shadow-md": "0 4px 6px rgba(0, 0, 0, 0.4)",
        "shadow-lg": "0 10px 15px rgba(0, 0, 0, 0.4)",
        "shadow-xl": "0 20px 25px rgba(0, 0, 0, 0.5)",
    }


# ============================================================================
# Plotly 圖表配色
# ============================================================================

def get_plotly_colors_light() -> Dict[str, Any]:
    """Plotly 圖表配色（Light 主題）"""
    return {
        # 背景
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": PRIMITIVE_COLORS["gray-50"],

        # 網格線
        "gridcolor": PRIMITIVE_COLORS["gray-200"],
        "zerolinecolor": PRIMITIVE_COLORS["gray-300"],

        # 文字
        "font_color": PRIMITIVE_COLORS["gray-900"],
        "title_color": PRIMITIVE_COLORS["gray-900"],

        # 分類色彩（8色，色盲友好）
        "categorical": [
            PRIMITIVE_COLORS["blue-600"],    # 藍
            PRIMITIVE_COLORS["orange-500"],  # 橙
            PRIMITIVE_COLORS["green-500"],   # 綠
            PRIMITIVE_COLORS["red-500"],     # 紅
            "#9333ea",                        # 紫
            "#eab308",                        # 黃
            "#06b6d4",                        # 青
            "#ec4899",                        # 粉
        ],

        # 連續色彩（數值大小）
        "sequential": [
            PRIMITIVE_COLORS["blue-100"],
            PRIMITIVE_COLORS["blue-300"],
            PRIMITIVE_COLORS["blue-500"],
            PRIMITIVE_COLORS["blue-700"],
            PRIMITIVE_COLORS["blue-900"],
        ],

        # 發散色彩（負-0-正）
        "diverging": [
            PRIMITIVE_COLORS["red-600"],     # 負值
            PRIMITIVE_COLORS["red-300"],
            "#ffffff",                        # 中點
            PRIMITIVE_COLORS["green-300"],
            PRIMITIVE_COLORS["green-600"],   # 正值
        ],

        # K線圖專用
        "candlestick_increasing": PRIMITIVE_COLORS["green-500"],
        "candlestick_decreasing": PRIMITIVE_COLORS["red-500"],

        # 成交量
        "volume_up": PRIMITIVE_COLORS["green-200"],
        "volume_down": PRIMITIVE_COLORS["red-200"],
    }


def get_plotly_colors_dark() -> Dict[str, Any]:
    """Plotly 圖表配色（Dark 主題）"""
    return {
        # 背景
        "paper_bgcolor": "#0f172a",
        "plot_bgcolor": "#1e293b",

        # 網格線
        "gridcolor": "#334155",
        "zerolinecolor": "#475569",

        # 文字
        "font_color": "#f1f5f9",
        "title_color": "#f1f5f9",

        # 分類色彩（8色，深色模式調整）
        "categorical": [
            PRIMITIVE_COLORS["blue-400"],
            PRIMITIVE_COLORS["orange-400"],
            PRIMITIVE_COLORS["green-400"],
            PRIMITIVE_COLORS["red-400"],
            "#a78bfa",  # 紫
            "#facc15",  # 黃
            "#22d3ee",  # 青
            "#f472b6",  # 粉
        ],

        # 連續色彩
        "sequential": [
            PRIMITIVE_COLORS["blue-900"],
            PRIMITIVE_COLORS["blue-700"],
            PRIMITIVE_COLORS["blue-500"],
            PRIMITIVE_COLORS["blue-300"],
            PRIMITIVE_COLORS["blue-100"],
        ],

        # 發散色彩
        "diverging": [
            PRIMITIVE_COLORS["red-400"],
            PRIMITIVE_COLORS["red-600"],
            "#1e293b",  # 深色背景作為中點
            PRIMITIVE_COLORS["green-600"],
            PRIMITIVE_COLORS["green-400"],
        ],

        # K線圖專用
        "candlestick_increasing": PRIMITIVE_COLORS["green-400"],
        "candlestick_decreasing": PRIMITIVE_COLORS["red-400"],

        # 成交量
        "volume_up": PRIMITIVE_COLORS["green-800"],
        "volume_down": PRIMITIVE_COLORS["red-800"],
    }


# ============================================================================
# 主要介面
# ============================================================================

def get_theme_tokens(theme: str = "light") -> Dict[str, Any]:
    """
    取得指定主題的完整 Token

    Args:
        theme: 主題名稱 ("light" 或 "dark")

    Returns:
        完整的設計 token 字典
    """
    if theme.lower() == "dark":
        return get_dark_tokens()
    return get_light_tokens()


def get_plotly_theme(theme: str = "light") -> Dict[str, Any]:
    """
    取得 Plotly 圖表主題配色

    Args:
        theme: 主題名稱 ("light" 或 "dark")

    Returns:
        Plotly 配色字典
    """
    if theme.lower() == "dark":
        return get_plotly_colors_dark()
    return get_plotly_colors_light()


def get_css_variables(theme: str = "light") -> str:
    """
    生成 CSS Variables 字串

    Args:
        theme: 主題名稱 ("light" 或 "dark")

    Returns:
        CSS :root 變數定義字串
    """
    tokens = get_theme_tokens(theme)

    css_vars = [":root {"]
    for key, value in tokens.items():
        css_key = f"--{key.replace('_', '-')}"
        css_vars.append(f"  {css_key}: {value};")
    css_vars.append("}")

    return "\n".join(css_vars)


# ============================================================================
# 使用範例
# ============================================================================

if __name__ == "__main__":
    # 範例 1: 取得主題 token
    light_theme = get_theme_tokens("light")
    print("Light Theme Primary:", light_theme["primary"])

    dark_theme = get_theme_tokens("dark")
    print("Dark Theme Primary:", dark_theme["primary"])

    # 範例 2: 取得 Plotly 配色
    plotly_light = get_plotly_theme("light")
    print("\nPlotly Light Categorical Colors:", plotly_light["categorical"])

    # 範例 3: 生成 CSS Variables
    css = get_css_variables("light")
    print("\nCSS Variables Preview:")
    print(css[:200] + "...")
