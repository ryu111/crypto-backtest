"""Plotly 圖表統一配置

提供一致的圖表視覺風格，支援 Light/Dark 主題切換。

設計原則：
1. 遵循 UI Design Tokens（來自 styles.py）
2. 提供預設配置，減少重複程式碼
3. 支援主題切換
4. 保持圖表清晰可讀
"""

from typing import Dict, List, Optional, Any


# ============================================================================
# Design Tokens（與 styles.py 保持一致）
# ============================================================================

TOKENS = {
    "light": {
        # 背景色
        "background": "#ffffff",
        "surface": "#f9fafb",
        "surface_raised": "#f3f4f6",

        # 文字色
        "text_primary": "#111827",
        "text_secondary": "#6b7280",
        "text_muted": "#9ca3af",

        # 邊框/網格
        "border": "#e5e7eb",
        "grid": "#f3f4f6",

        # 功能色
        "primary": "#2563eb",
        "primary_light": "#dbeafe",
        "success": "#22c55e",
        "warning": "#eab308",
        "error": "#ef4444",
        "info": "#3b82f6",

        # 圖表專用色板（60-30-10 法則）
        "chart_colors": [
            "#2563eb",  # primary blue (60%)
            "#10b981",  # success green (30%)
            "#f59e0b",  # warning orange (10%)
            "#8b5cf6",  # purple
            "#ec4899",  # pink
            "#06b6d4",  # cyan
        ],

        # 評級色
        "grade_A": "#10b981",
        "grade_B": "#3b82f6",
        "grade_C": "#f59e0b",
        "grade_D": "#f97316",
        "grade_F": "#ef4444",
    },

    "dark": {
        # 背景色（Material Design Dark Mode）
        "background": "#121212",
        "surface": "#1e1e1e",
        "surface_raised": "#252525",

        # 文字色
        "text_primary": "#f3f4f6",
        "text_secondary": "#9ca3af",
        "text_muted": "#6b7280",

        # 邊框/網格
        "border": "#2d2d2d",
        "grid": "#252525",

        # 功能色（深色模式調整飽和度）
        "primary": "#3b82f6",
        "primary_light": "#1e3a8a",
        "success": "#22c55e",
        "warning": "#fbbf24",
        "error": "#f87171",
        "info": "#60a5fa",

        # 圖表專用色板（深色模式亮度提高）
        "chart_colors": [
            "#3b82f6",  # primary blue
            "#22c55e",  # success green
            "#fbbf24",  # warning yellow
            "#a78bfa",  # purple
            "#f472b6",  # pink
            "#22d3ee",  # cyan
        ],

        # 評級色
        "grade_A": "#22c55e",
        "grade_B": "#60a5fa",
        "grade_C": "#fbbf24",
        "grade_D": "#fb923c",
        "grade_F": "#f87171",
    }
}


# ============================================================================
# 核心配置函數
# ============================================================================

def get_theme_tokens(theme: str = "light") -> Dict[str, Any]:
    """取得主題 tokens

    Args:
        theme: 主題名稱 ('light' 或 'dark')

    Returns:
        主題 token 字典
    """
    return TOKENS.get(theme, TOKENS["light"])


def get_chart_layout(theme: str = "light", **kwargs: Any) -> Dict[str, Any]:
    """取得圖表 layout 配置

    Args:
        theme: 主題 ('light' 或 'dark')
        **kwargs: layout 參數，常用的包括：
            - title: 圖表標題
            - xaxis_title: X 軸標題
            - yaxis_title: Y 軸標題
            - height: 圖表高度（px），預設 400
            - showlegend: 是否顯示圖例，預設 True

    Returns:
        Plotly layout dict

    範例:
        ```python
        fig = go.Figure()
        fig.update_layout(**get_chart_layout(
            theme='light',
            title='Sharpe Ratio 趨勢',
            height=500
        ))
        ```
    """
    # 預設值
    title = kwargs.pop("title", None)
    xaxis_title = kwargs.pop("xaxis_title", None)
    yaxis_title = kwargs.pop("yaxis_title", None)
    height = kwargs.pop("height", 400)
    showlegend = kwargs.pop("showlegend", True)

    tokens = get_theme_tokens(theme)

    base_layout = {
        # 主題模板
        "template": "plotly_white" if theme == "light" else "plotly_dark",

        # 背景色
        "paper_bgcolor": tokens["background"],
        "plot_bgcolor": tokens["surface"],

        # 字體
        "font": {
            "family": "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            "color": tokens["text_primary"],
            "size": 13,
        },

        # 標題
        "title": {
            "text": title,
            "font": {
                "size": 16,
                "weight": 600,
                "color": tokens["text_primary"],
            },
            "x": 0.02,  # 左對齊
            "xanchor": "left",
        } if title else None,

        # X 軸
        "xaxis": {
            "title": {
                "text": xaxis_title,
                "font": {"size": 12, "color": tokens["text_secondary"]},
            } if xaxis_title else None,
            "showgrid": True,
            "gridcolor": tokens["grid"],
            "gridwidth": 1,
            "zeroline": False,
            "linecolor": tokens["border"],
            "tickfont": {"size": 11, "color": tokens["text_secondary"]},
        },

        # Y 軸
        "yaxis": {
            "title": {
                "text": yaxis_title,
                "font": {"size": 12, "color": tokens["text_secondary"]},
            } if yaxis_title else None,
            "showgrid": True,
            "gridcolor": tokens["grid"],
            "gridwidth": 1,
            "zeroline": True,
            "zerolinecolor": tokens["border"],
            "zerolinewidth": 1,
            "linecolor": tokens["border"],
            "tickfont": {"size": 11, "color": tokens["text_secondary"]},
        },

        # 圖例
        "showlegend": showlegend,
        "legend": {
            "font": {"size": 11, "color": tokens["text_secondary"]},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": tokens["border"],
            "borderwidth": 0,
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },

        # Hover
        "hovermode": "closest",
        "hoverlabel": {
            "bgcolor": tokens["surface_raised"],
            "bordercolor": tokens["border"],
            "font": {"size": 11, "color": tokens["text_primary"]},
        },

        # 邊距
        "margin": {
            "l": 60,
            "r": 40,
            "t": 60 if title else 40,
            "b": 50,
        },

        # 高度
        "height": height,

        # 響應式
        "autosize": True,
    }

    # 合併自訂參數
    base_layout.update(kwargs)

    return base_layout


def get_chart_colors(theme: str = "light", n: Optional[int] = None) -> List[str]:
    """取得圖表顏色列表

    Args:
        theme: 主題 ('light' 或 'dark')
        n: 需要的顏色數量（None = 返回全部）

    Returns:
        顏色列表（Hex）

    範例:
        ```python
        colors = get_chart_colors('light', n=3)
        fig.add_trace(go.Bar(marker_color=colors[0]))
        ```
    """
    tokens = get_theme_tokens(theme)
    colors = tokens["chart_colors"]

    if n is None:
        return colors

    # 如果需要更多顏色，循環使用
    return [colors[i % len(colors)] for i in range(n)]


def get_grade_color(grade: str, theme: str = "light") -> str:
    """取得評級對應顏色

    Args:
        grade: 評級 (A/B/C/D/F)
        theme: 主題

    Returns:
        Hex 顏色碼

    範例:
        ```python
        color = get_grade_color('A', 'light')
        fig.add_trace(go.Bar(marker_color=color))
        ```
    """
    tokens = get_theme_tokens(theme)
    return tokens.get(f"grade_{grade}", tokens["text_muted"])


# ============================================================================
# 預設圖表配置
# ============================================================================

def get_line_chart_config(theme: str = "light", **layout_kwargs) -> Dict[str, Any]:
    """Line Chart 預設配置

    Args:
        theme: 主題
        **layout_kwargs: 額外 layout 參數

    Returns:
        完整 layout dict

    範例:
        ```python
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=values, mode='lines'))
        fig.update_layout(**get_line_chart_config(
            theme='light',
            title='時間趨勢'
        ))
        ```
    """
    tokens = get_theme_tokens(theme)

    defaults = {
        "hovermode": "x unified",
        "xaxis": {
            "showspikes": True,
            "spikecolor": tokens["border"],
            "spikethickness": 1,
            "spikedash": "dot",
            "spikemode": "across",
        },
    }

    defaults.update(layout_kwargs)
    return get_chart_layout(theme=theme, **defaults)


def get_bar_chart_config(theme: str = "light", **layout_kwargs) -> Dict[str, Any]:
    """Bar Chart 預設配置

    Args:
        theme: 主題
        **layout_kwargs: 額外 layout 參數

    Returns:
        完整 layout dict

    範例:
        ```python
        fig = go.Figure()
        fig.add_trace(go.Bar(x=categories, y=values))
        fig.update_layout(**get_bar_chart_config(theme='light'))
        ```
    """
    defaults = {
        "bargap": 0.15,
        "bargroupgap": 0.1,
    }

    defaults.update(layout_kwargs)
    return get_chart_layout(theme=theme, **defaults)


def get_histogram_config(theme: str = "light", **layout_kwargs) -> Dict[str, Any]:
    """Histogram 預設配置

    Args:
        theme: 主題
        **layout_kwargs: 額外 layout 參數

    Returns:
        完整 layout dict

    範例:
        ```python
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data))
        fig.update_layout(**get_histogram_config(
            theme='light',
            title='分布圖'
        ))
        ```
    """
    defaults = {
        "bargap": 0.05,
    }

    defaults.update(layout_kwargs)
    return get_chart_layout(theme=theme, **defaults)


def get_pie_chart_config(theme: str = "light", **layout_kwargs) -> Dict[str, Any]:
    """Pie Chart 預設配置

    Args:
        theme: 主題
        **layout_kwargs: 額外 layout 參數

    Returns:
        完整 layout dict

    範例:
        ```python
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=get_chart_colors('light', len(labels)))
        ))
        fig.update_layout(**get_pie_chart_config(theme='light'))
        ```
    """
    defaults = {
        "showlegend": True,
        "legend": {
            "orientation": "v",
            "yanchor": "middle",
            "y": 0.5,
            "xanchor": "left",
            "x": 1.05,
        },
    }

    defaults.update(layout_kwargs)
    return get_chart_layout(theme=theme, **defaults)


def get_heatmap_config(theme: str = "light", **layout_kwargs) -> Dict[str, Any]:
    """Heatmap 預設配置

    Args:
        theme: 主題
        **layout_kwargs: 額外 layout 參數

    Returns:
        完整 layout dict

    範例:
        ```python
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=correlation_matrix,
            x=labels,
            y=labels,
            colorscale='RdBu',
            zmid=0
        ))
        fig.update_layout(**get_heatmap_config(theme='light'))
        ```
    """
    defaults = {
        "xaxis": {
            "showgrid": False,
            "side": "bottom",
        },
        "yaxis": {
            "showgrid": False,
            "autorange": "reversed",  # Y 軸反轉（符合習慣）
        },
    }

    defaults.update(layout_kwargs)
    return get_chart_layout(theme=theme, **defaults)


# ============================================================================
# 常用顏色配置（Colorscale）
# ============================================================================

def get_diverging_colorscale(theme: str = "light") -> List:
    """取得發散型配色（用於 Heatmap）

    適用於有正負值的資料（如相關係數、報酬率）

    Args:
        theme: 主題

    Returns:
        Plotly colorscale 格式

    範例:
        ```python
        fig.add_trace(go.Heatmap(
            z=data,
            colorscale=get_diverging_colorscale('light'),
            zmid=0
        ))
        ```
    """
    if theme == "light":
        return [
            [0.0, "#ef4444"],   # 紅色（負值）
            [0.5, "#f3f4f6"],   # 灰白（0）
            [1.0, "#22c55e"],   # 綠色（正值）
        ]
    else:
        return [
            [0.0, "#f87171"],   # 亮紅色
            [0.5, "#252525"],   # 深灰
            [1.0, "#22c55e"],   # 綠色
        ]


def get_sequential_colorscale(theme: str = "light", color: str = "blue") -> List:
    """取得連續型配色（用於 Heatmap）

    適用於單一方向的數值（如交易量、權重）

    Args:
        theme: 主題
        color: 基礎顏色 ('blue', 'green', 'red')

    Returns:
        Plotly colorscale 格式

    範例:
        ```python
        fig.add_trace(go.Heatmap(
            z=weights,
            colorscale=get_sequential_colorscale('light', 'blue')
        ))
        ```
    """
    tokens = get_theme_tokens(theme)

    if color == "blue":
        base = tokens["primary"]
    elif color == "green":
        base = tokens["success"]
    elif color == "red":
        base = tokens["error"]
    else:
        base = tokens["primary"]

    bg = tokens["surface"]

    return [
        [0.0, bg],
        [1.0, base],
    ]


# ============================================================================
# 快速配置範例
# ============================================================================

# 使用範例見各函數的 docstring
