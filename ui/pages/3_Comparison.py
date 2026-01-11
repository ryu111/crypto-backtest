"""
策略比較頁面

功能：
- 選擇多個策略進行比較（最多 5 個）
- 指標對比表
- 視覺化圖表（權益曲線、回撤、月度報酬、雷達圖）
- AI 生成的比較結論
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Any
from pathlib import Path
import sys

# 加入專案根目錄到 sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 注意：避免 import src.validator.stages，因為會觸發 vectorbt/numba 載入
# 導致 NumPy 版本衝突（Numba 需要 NumPy <= 2.3，但系統有 NumPy 2.4）
from ui.utils import render_sidebar_navigation
from ui.styles import get_common_css


# ========== 設計 Token ==========
# 根據 ~/.claude/skills/ui/references/tokens.md

COLORS = {
    'primary': '#2563eb',
    'primary_light': '#dbeafe',
    'success': '#22c55e',
    'warning': '#eab308',
    'error': '#ef4444',
    'text': '#111827',
    'text_secondary': '#6b7280',
    'border': '#e5e7eb',
    'surface': '#ffffff',
    'surface_raised': '#f9fafb',
}

SPACING = {
    'xs': '4px',
    'sm': '8px',
    'md': '16px',
    'lg': '24px',
    'xl': '32px',
}


def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """將 hex 顏色轉換為 rgba 格式（Plotly 需要）"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r}, {g}, {b}, {alpha})'


# ========== 資料載入 ==========

@st.cache_data
def load_strategy_results() -> Dict[str, Dict[str, Any]]:
    """
    載入所有策略的回測結果

    Returns:
        Dict[策略名稱, 策略資料]
        策略資料包含：
        - metrics: 績效指標
        - equity_curve: 權益曲線
        - trades: 交易記錄
        - validation: 驗證結果
    """
    # TODO: 從實際儲存位置載入資料
    # 這裡提供模擬資料結構

    strategies = {}

    # 模擬資料
    for i in range(5):
        strategy_name = f"策略 {chr(65 + i)}"  # A, B, C, D, E

        # 模擬權益曲線
        np.random.seed(i * 100)
        days = 365
        returns = np.random.normal(0.001, 0.02, days)
        equity = 10000 * (1 + returns).cumprod()

        strategies[strategy_name] = {
            'metrics': {
                'total_return': (equity[-1] / equity[0] - 1) * 100,
                'sharpe_ratio': np.random.uniform(1.5, 2.5),
                'max_drawdown': -np.random.uniform(5, 20),
                'win_rate': np.random.uniform(45, 65),
                'total_trades': np.random.randint(80, 200),
                'profit_factor': np.random.uniform(1.2, 2.5),
                'calmar_ratio': np.random.uniform(1.0, 3.0),
                'validation_grade': np.random.choice(['A', 'B', 'C']),
            },
            'equity_curve': pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=days),
                'equity': equity,
            }).set_index('date'),
            'monthly_returns': pd.Series(
                np.random.normal(0.03, 0.05, 12),
                index=pd.date_range('2024-01', periods=12, freq='MS')
            ),
            'params': {
                'period': np.random.randint(10, 20),
                'threshold': np.random.uniform(0.01, 0.05),
            }
        }

    return strategies


def get_strategy_names(strategies: Dict) -> List[str]:
    """取得所有策略名稱"""
    return list(strategies.keys())


# ========== UI 元件 ==========

def render_strategy_selector(available_strategies: List[str]) -> List[str]:
    """
    策略選擇器

    Returns:
        選中的策略名稱列表
    """
    st.subheader("📊 選擇策略")

    col1, col2 = st.columns([3, 1])

    with col1:
        selected = st.multiselect(
            "選擇要比較的策略（最多 5 個）",
            options=available_strategies,
            default=available_strategies[:3] if len(available_strategies) >= 3 else available_strategies,
            max_selections=5,
            help="選擇 2-5 個策略進行比較"
        )

    with col2:
        st.write("")  # 對齊
        st.write("")
        quick_select = st.selectbox(
            "快速選擇",
            options=['手動選擇', 'Top 3', '最新 3 個'],
            help="快速選擇策略組合"
        )

        if quick_select == 'Top 3':
            # TODO: 根據評級或報酬排序
            selected = available_strategies[:3]
        elif quick_select == '最新 3 個':
            # TODO: 根據建立時間排序
            selected = available_strategies[-3:]

    if len(selected) < 2:
        st.warning("⚠️ 請至少選擇 2 個策略進行比較")
        return []

    return selected


def render_metrics_comparison_table(strategies: Dict, selected_names: List[str]):
    """渲染指標比較表"""
    st.subheader("📈 指標比較")

    # 準備資料
    metrics_data = []
    metric_labels = {
        'total_return': '總報酬率 (%)',
        'sharpe_ratio': 'Sharpe Ratio',
        'max_drawdown': '最大回撤 (%)',
        'win_rate': '勝率 (%)',
        'total_trades': '交易次數',
        'profit_factor': 'Profit Factor',
        'calmar_ratio': 'Calmar Ratio',
        'validation_grade': '驗證等級',
    }

    for metric_key, metric_label in metric_labels.items():
        row = {'指標': metric_label}

        for name in selected_names:
            value = strategies[name]['metrics'][metric_key]

            # 格式化數值
            if metric_key in ['total_return', 'max_drawdown', 'win_rate']:
                row[name] = f"{value:.2f}%"
            elif metric_key in ['sharpe_ratio', 'profit_factor', 'calmar_ratio']:
                row[name] = f"{value:.2f}"
            elif metric_key == 'total_trades':
                row[name] = f"{int(value)}"
            else:
                row[name] = str(value)

        # 標註最佳值
        if metric_key != 'validation_grade':
            values = [strategies[name]['metrics'][metric_key] for name in selected_names]

            # 最大回撤是越接近 0 越好（負數絕對值越小越好）
            if metric_key == 'max_drawdown':
                best_idx = np.argmax(values)  # 最接近 0
            else:
                best_idx = np.argmax(values)

            best_name = selected_names[best_idx]
            row['最佳'] = best_name
        else:
            # 驗證等級
            grades = [strategies[name]['metrics'][metric_key] for name in selected_names]
            grade_order = {'A': 3, 'B': 2, 'C': 1, 'D': 0, 'F': 0}
            best_idx = max(range(len(grades)), key=lambda i: grade_order.get(grades[i], 0))
            row['最佳'] = selected_names[best_idx]

        metrics_data.append(row)

    df = pd.DataFrame(metrics_data)

    # 使用 Streamlit 的表格渲染
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    # [C1] 指標對比表解讀
    st.caption("""
    **[C1] 指標對比表解讀**：
    - **總報酬率**：回測期間累積收益，>30% 優秀
    - **Sharpe Ratio**：每單位風險的收益，>2.0 卓越，>1.5 良好
    - **最大回撤**：歷史最大虧損幅度，<15% 優秀，>25% 需注意
    - **Profit Factor**：總獲利/總虧損，>2.0 優秀，>1.5 及格
    - **Calmar Ratio**：年化報酬/最大回撤，>2.0 表示風險報酬比良好
    - **最佳欄位**：標註各指標表現最佳的策略，幫助快速判斷
    """)


def render_equity_curves(strategies: Dict, selected_names: List[str]):
    """渲染權益曲線疊加圖"""
    st.subheader("📉 權益曲線對比")

    fig = go.Figure()

    # 配色方案
    colors = [
        COLORS['primary'],
        COLORS['success'],
        COLORS['warning'],
        COLORS['error'],
        '#8b5cf6',  # purple
    ]

    for i, name in enumerate(selected_names):
        equity_curve = strategies[name]['equity_curve']

        # 正規化到相同起點（100%）
        normalized = (equity_curve['equity'] / equity_curve['equity'].iloc[0]) * 100

        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=normalized,
            name=name,
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate='%{y:.2f}%<extra></extra>'
        ))

    fig.update_layout(
        xaxis_title='日期',
        yaxis_title='權益 (%)',
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor=COLORS['surface'],
        paper_bgcolor=COLORS['surface'],
    )

    st.plotly_chart(fig, use_container_width=True)

    # [C2] 權益曲線說明
    st.caption("""
    **[C2] 權益曲線解讀**：
    - **正規化起點**：所有策略從 100% 開始，方便比較相對表現
    - **曲線走勢**：持續上升且波動小 = 穩定成長；劇烈波動 = 風險較高
    - **曲線交叉**：當曲線交叉時，代表策略相對表現發生變化
    - **選擇建議**：優先選擇曲線平滑向上、回撤期間恢復快的策略
    """)


def render_drawdown_comparison(strategies: Dict, selected_names: List[str]):
    """渲染回撤比較圖"""
    st.subheader("📊 回撤對比")

    fig = go.Figure()

    colors = [
        COLORS['primary'],
        COLORS['success'],
        COLORS['warning'],
        COLORS['error'],
        '#8b5cf6',
    ]

    for i, name in enumerate(selected_names):
        equity_curve = strategies[name]['equity_curve']

        # 計算回撤
        running_max = equity_curve['equity'].expanding().max()
        drawdown = (equity_curve['equity'] - running_max) / running_max * 100

        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=drawdown,
            name=name,
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=2),
            fill='tozeroy',
            fillcolor=colors[i % len(colors)].replace(')', ', 0.1)').replace('rgb', 'rgba'),
            hovertemplate='%{y:.2f}%<extra></extra>'
        ))

    fig.update_layout(
        xaxis_title='日期',
        yaxis_title='回撤 (%)',
        hovermode='x unified',
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor=COLORS['surface'],
        paper_bgcolor=COLORS['surface'],
    )

    st.plotly_chart(fig, use_container_width=True)

    # [C3] 回撤圖解讀
    st.caption("""
    **[C3] 回撤圖解讀**：
    - **回撤深度**：負值越大代表虧損越深，-10% 表示從高點下跌 10%
    - **回撤頻率**：頻繁觸底代表策略波動大，需要更強心理素質
    - **恢復速度**：回撤後快速回到 0% = 恢復能力強
    - **重疊期間**：多策略同時回撤 = 系統性風險，需注意市場環境
    - **選擇建議**：優先選擇回撤淺、恢復快的策略
    """)


def render_monthly_returns_comparison(strategies: Dict, selected_names: List[str]):
    """渲染月度報酬對比（Group Bar Chart）"""
    st.subheader("📅 月度報酬對比")

    # 準備資料
    data = []
    for name in selected_names:
        monthly = strategies[name]['monthly_returns']
        for date, value in monthly.items():
            data.append({
                '月份': date.strftime('%Y-%m'),
                '策略': name,
                '報酬率': value * 100
            })

    df = pd.DataFrame(data)

    fig = px.bar(
        df,
        x='月份',
        y='報酬率',
        color='策略',
        barmode='group',
        color_discrete_sequence=[
            COLORS['primary'],
            COLORS['success'],
            COLORS['warning'],
            COLORS['error'],
            '#8b5cf6',
        ]
    )

    fig.update_layout(
        xaxis_title='月份',
        yaxis_title='報酬率 (%)',
        height=400,
        plot_bgcolor=COLORS['surface'],
        paper_bgcolor=COLORS['surface'],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)


def render_radar_chart(strategies: Dict, selected_names: List[str]):
    """渲染雷達圖（多維度比較）"""
    st.subheader("🎯 多維度雷達圖")

    # 定義維度（正規化到 0-100）
    dimensions = [
        '報酬率',
        'Sharpe Ratio',
        '穩定性',
        '勝率',
        '驗證等級'
    ]

    fig = go.Figure()

    colors = [
        COLORS['primary'],
        COLORS['success'],
        COLORS['warning'],
        COLORS['error'],
        '#8b5cf6',
    ]

    for i, name in enumerate(selected_names):
        metrics = strategies[name]['metrics']

        # 正規化各維度到 0-100
        values = [
            min(max(metrics['total_return'], 0), 100),  # 報酬率
            min(metrics['sharpe_ratio'] * 20, 100),  # Sharpe * 20
            (1 - abs(metrics['max_drawdown']) / 100) * 100,  # 穩定性
            metrics['win_rate'],  # 勝率
            {'A': 100, 'B': 80, 'C': 60, 'D': 40, 'F': 20}.get(metrics['validation_grade'], 50),  # 等級
        ]

        color = colors[i % len(colors)]
        # 將顏色轉換為 rgba 格式用於填充（Plotly 不支援 8 位 hex）
        fill_color = hex_to_rgba(color, 0.2) if color.startswith('#') else color.replace(')', ', 0.2)').replace('rgb', 'rgba')

        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # 閉合圖形
            theta=dimensions + [dimensions[0]],
            name=name,
            fill='toself',
            line=dict(color=color),
            fillcolor=fill_color,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # [C4] 雷達圖說明
    st.caption("""
    **[C4] 雷達圖解讀**：
    - **覆蓋面積**：面積越大代表綜合表現越好
    - **形狀平衡**：五邊形越均勻 = 各方面表現平衡；凸出/凹陷 = 優勢/劣勢明顯
    - **報酬率**：回測期間累積收益能力
    - **Sharpe**：風險調整後收益（正規化：Sharpe×20）
    - **穩定性**：基於最大回撤計算，越高越穩定
    - **勝率**：獲利交易佔比
    - **驗證等級**：A=100, B=80, C=60, D=40, F=20
    - **選擇建議**：選擇面積大且形狀平衡的策略
    """)


def render_parameter_comparison(strategies: Dict, selected_names: List[str]):
    """渲染參數差異對比"""
    st.subheader("⚙️ 參數對比")

    # 收集所有參數
    all_params = set()
    for name in selected_names:
        all_params.update(strategies[name]['params'].keys())

    # 準備資料
    param_data = []
    for param in sorted(all_params):
        row = {'參數': param}
        for name in selected_names:
            value = strategies[name]['params'].get(param, '-')
            if isinstance(value, float):
                row[name] = f"{value:.4f}"
            else:
                row[name] = str(value)
        param_data.append(row)

    df = pd.DataFrame(param_data)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


def render_validation_comparison(strategies: Dict, selected_names: List[str]):
    """渲染驗證階段對比"""
    st.subheader("✅ 驗證階段對比")

    # TODO: 從實際的 ValidationResult 載入
    # 這裡使用模擬資料

    stages = [
        '階段1_基礎回測',
        '階段2_統計檢驗',
        '階段3_穩健性',
        '階段4_WalkForward',
        '階段5_MonteCarlo'
    ]

    stage_data = []
    for stage in stages:
        row = {'階段': stage.split('_')[1]}
        for name in selected_names:
            # 模擬通過狀態
            passed = np.random.choice([True, False], p=[0.7, 0.3])
            row[name] = '✅ 通過' if passed else '❌ 未通過'
        stage_data.append(row)

    df = pd.DataFrame(stage_data)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


def render_ai_recommendation(strategies: Dict, selected_names: List[str]):
    """渲染 AI 生成的比較結論"""
    st.subheader("🤖 AI 比較結論")

    # 分析各策略表現
    best_return_name = max(
        selected_names,
        key=lambda n: strategies[n]['metrics']['total_return']
    )

    best_sharpe_name = max(
        selected_names,
        key=lambda n: strategies[n]['metrics']['sharpe_ratio']
    )

    best_stability_name = max(
        selected_names,
        key=lambda n: strategies[n]['metrics']['max_drawdown']
    )

    best_grade_name = max(
        selected_names,
        key=lambda n: {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'F': 1}.get(
            strategies[n]['metrics']['validation_grade'], 0
        )
    )

    # 生成結論
    st.markdown("### 整體評估")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 優勢分析")
        st.markdown(f"""
        - **最高報酬**: {best_return_name} ({strategies[best_return_name]['metrics']['total_return']:.2f}%)
        - **最佳風險調整**: {best_sharpe_name} (Sharpe: {strategies[best_sharpe_name]['metrics']['sharpe_ratio']:.2f})
        - **最穩定**: {best_stability_name} (最大回撤: {strategies[best_stability_name]['metrics']['max_drawdown']:.2f}%)
        - **最高驗證等級**: {best_grade_name} (等級: {strategies[best_grade_name]['metrics']['validation_grade']})
        """)

    with col2:
        st.markdown("#### 推薦選擇")

        # 綜合評分
        scores = {}
        for name in selected_names:
            m = strategies[name]['metrics']
            score = (
                m['total_return'] * 0.3 +
                m['sharpe_ratio'] * 10 * 0.3 +
                (1 - abs(m['max_drawdown']) / 100) * 100 * 0.2 +
                {'A': 100, 'B': 80, 'C': 60, 'D': 40, 'F': 20}.get(m['validation_grade'], 50) * 0.2
            )
            scores[name] = score

        recommended = max(scores, key=scores.get)

        st.success(f"""
        **推薦策略**: {recommended}

        **理由**:
        - 綜合評分最高
        - 平衡報酬與風險
        - 通過完整驗證
        """)

    st.markdown("---")
    st.markdown("### 各策略特點")

    for name in selected_names:
        m = strategies[name]['metrics']

        # 判斷特點
        strengths = []
        weaknesses = []

        if m['total_return'] > 30:
            strengths.append("高報酬率")
        elif m['total_return'] < 10:
            weaknesses.append("報酬率偏低")

        if m['sharpe_ratio'] > 2.0:
            strengths.append("優秀的風險調整報酬")
        elif m['sharpe_ratio'] < 1.0:
            weaknesses.append("風險調整報酬不佳")

        if abs(m['max_drawdown']) < 10:
            strengths.append("回撤控制良好")
        elif abs(m['max_drawdown']) > 20:
            weaknesses.append("回撤較大")

        if m['validation_grade'] in ['A', 'B']:
            strengths.append("高驗證等級")
        elif m['validation_grade'] in ['D', 'F']:
            weaknesses.append("驗證等級不足")

        with st.expander(f"**{name}** - {m['validation_grade']} 級"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**優勢**:")
                for s in strengths:
                    st.markdown(f"- ✅ {s}")

            with col2:
                st.markdown("**劣勢**:")
                for w in weaknesses:
                    st.markdown(f"- ⚠️ {w}")


# ========== 主程式 ==========

def main():
    """主程式"""
    st.set_page_config(
        page_title="策略比較",
        page_icon="⚖️",
        layout="wide"
    )

    # 共用樣式（包含隱藏英文導航）
    st.markdown(get_common_css(), unsafe_allow_html=True)

    # 渲染中文 sidebar 導航
    render_sidebar_navigation()

    st.title("⚖️ 策略比較")
    st.markdown("比較多個策略的績效指標，選擇最佳策略")

    # 載入資料
    with st.spinner("載入策略資料..."):
        strategies = load_strategy_results()

    available_names = get_strategy_names(strategies)

    if not available_names:
        st.error("❌ 沒有可用的策略資料")
        st.info("請先執行策略回測並儲存結果")
        return

    # 策略選擇器
    selected_names = render_strategy_selector(available_names)

    if len(selected_names) < 2:
        return

    # 建立 Tab
    tabs = st.tabs([
        "📊 指標對比",
        "📈 視覺化圖表",
        "⚙️ 詳細對比",
        "🤖 AI 結論"
    ])

    # Tab 1: 指標對比
    with tabs[0]:
        render_metrics_comparison_table(strategies, selected_names)

    # Tab 2: 視覺化圖表
    with tabs[1]:
        render_equity_curves(strategies, selected_names)
        st.markdown("---")
        render_drawdown_comparison(strategies, selected_names)
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            render_monthly_returns_comparison(strategies, selected_names)
        with col2:
            render_radar_chart(strategies, selected_names)

    # Tab 3: 詳細對比
    with tabs[2]:
        render_parameter_comparison(strategies, selected_names)
        st.markdown("---")
        render_validation_comparison(strategies, selected_names)

    # Tab 4: AI 結論
    with tabs[3]:
        render_ai_recommendation(strategies, selected_names)


if __name__ == "__main__":
    main()
