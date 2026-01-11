"""
策略驗證頁面

評估策略的統計顯著性與穩健性，防止過擬合。
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import sys

# 加入專案路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ui.styles import get_common_css, GRADE_COLORS
from ui.utils import render_sidebar_navigation


# ===== 設定頁面 =====
st.set_page_config(
    page_title="策略驗證 - 合約交易系統",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== 自訂樣式 =====
st.markdown(get_common_css(), unsafe_allow_html=True)


# ===== 資料載入函數 =====

@st.cache_data
def load_validation_results() -> Dict[str, Any]:
    """
    載入驗證結果

    TODO: 實際從檔案系統載入結果
    目前返回範例資料，結構符合設計規格預期
    """
    # 範例資料結構
    return {
        'bootstrap': {
            'ci_lower': 12.3,
            'ci_upper': 45.2,
            'pass': True,
            'confidence': 0.95,
            'distribution': np.random.normal(28.5, 8.2, 10000)  # Bootstrap 分布
        },
        'permutation': {
            'p_value': 0.032,
            'pass': True,
            'actual_value': 28.5,
            'null_distribution': np.random.normal(0, 10, 10000),
            'n_permutations': 10000
        },
        'cross_validation': {
            'folds': [
                {'fold_id': 1, 'is_return': 25.3, 'oos_return': 21.5, 'period': '2023-01 ~ 2023-03'},
                {'fold_id': 2, 'is_return': 30.2, 'oos_return': 28.1, 'period': '2023-04 ~ 2023-06'},
                {'fold_id': 3, 'is_return': 28.9, 'oos_return': 25.8, 'period': '2023-07 ~ 2023-09'},
                {'fold_id': 4, 'is_return': 27.5, 'oos_return': 24.2, 'period': '2023-10 ~ 2023-12'},
            ],
            'mean': 28.5,
            'std': 4.2,
            'stability': 0.87
        },
        'sharpe_correction': {
            'original': 2.15,
            'deflated': 1.45,
            'trials': 120,
            'pbo': 0.35,
            'adjustment_factor': 0.674
        },
        'stress_test': {
            'events': {
                'covid_crash_2020': {
                    'name': 'COVID-19 崩盤 (2020/03)',
                    'max_drawdown': -18.5,
                    'recovery_days': 45,
                    'sharpe': 0.82,
                    'equity_curve': 100 * (1 + np.cumsum(np.random.randn(90) * 0.03)),
                    'dates': pd.date_range('2020-02-15', periods=90, freq='D')
                },
                'luna_crash_2022': {
                    'name': 'LUNA 崩盤 (2022/05)',
                    'max_drawdown': -25.3,
                    'recovery_days': 68,
                    'sharpe': 0.65,
                    'equity_curve': 100 * (1 + np.cumsum(np.random.randn(90) * 0.04)),
                    'dates': pd.date_range('2022-04-20', periods=90, freq='D')
                },
                'ftx_collapse_2022': {
                    'name': 'FTX 崩盤 (2022/11)',
                    'max_drawdown': -15.2,
                    'recovery_days': 38,
                    'sharpe': 0.95,
                    'equity_curve': 100 * (1 + np.cumsum(np.random.randn(90) * 0.025)),
                    'dates': pd.date_range('2022-10-25', periods=90, freq='D')
                },
            }
        }
    }


@st.cache_data
def get_available_strategies() -> List[Dict[str, str]]:
    """取得可用的策略清單"""
    # TODO: 實際從檔案系統載入
    return [
        {'name': 'MA Cross (10/30)', 'symbol': 'BTCUSDT', 'timeframe': '4h'},
        {'name': 'RSI Mean Reversion', 'symbol': 'ETHUSDT', 'timeframe': '1h'},
        {'name': 'Supertrend Momentum', 'symbol': 'BTCUSDT', 'timeframe': '1d'},
    ]


# ===== 圖表繪製函數 =====

def get_chart_config() -> Dict:
    """Plotly 圖表統一配置"""
    return {
        'displayModeBar': True,
        'displaylogo': False,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'validation_chart',
            'height': 600,
            'width': 1200,
            'scale': 2
        }
    }


def apply_chart_theme(fig: go.Figure) -> go.Figure:
    """套用圖表主題"""
    fig.update_layout(
        font=dict(family="'Inter', sans-serif", size=14),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_family="'Inter', sans-serif"
        )
    )
    return fig


def plot_bootstrap_distribution(data: Dict) -> go.Figure:
    """繪製 Bootstrap 分布圖"""
    distribution = data['distribution']
    ci_lower = data['ci_lower']
    ci_upper = data['ci_upper']

    fig = go.Figure()

    # 直方圖
    fig.add_trace(go.Histogram(
        x=distribution,
        name='Bootstrap 分布',
        marker_color='#2563eb',
        opacity=0.7,
        nbinsx=50
    ))

    # 信賴區間下界
    fig.add_vline(
        x=ci_lower,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"95% CI 下界: {ci_lower:.1f}%",
        annotation_position="top left"
    )

    # 信賴區間上界
    fig.add_vline(
        x=ci_upper,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"95% CI 上界: {ci_upper:.1f}%",
        annotation_position="top right"
    )

    # 平均值
    mean_val = np.mean(distribution)
    fig.add_vline(
        x=mean_val,
        line_color="#22c55e",
        annotation_text=f"平均: {mean_val:.1f}%",
        annotation_position="top"
    )

    fig.update_layout(
        title='Bootstrap 報酬率分布',
        xaxis_title='報酬率 (%)',
        yaxis_title='頻率',
        height=400,
        showlegend=False
    )

    return apply_chart_theme(fig)


def plot_permutation_test(data: Dict) -> go.Figure:
    """繪製 Permutation Test 結果"""
    null_dist = data['null_distribution']
    actual = data['actual_value']
    p_value = data['p_value']

    fig = go.Figure()

    # 隨機分布
    fig.add_trace(go.Histogram(
        x=null_dist,
        name='隨機分布',
        marker_color='#9ca3af',
        opacity=0.7,
        nbinsx=50
    ))

    # 實際值
    fig.add_vline(
        x=actual,
        line_color="#22c55e",
        line_width=3,
        annotation_text=f"實際報酬: {actual:.1f}%<br>p-value: {p_value:.3f}",
        annotation_position="top right"
    )

    fig.update_layout(
        title='Permutation Test - 實際 vs 隨機分布',
        xaxis_title='報酬率 (%)',
        yaxis_title='頻率',
        height=400,
        showlegend=False
    )

    return apply_chart_theme(fig)


def plot_cv_performance(folds: List[Dict]) -> go.Figure:
    """繪製交叉驗證績效折線圖"""
    fold_ids = [f['fold_id'] for f in folds]
    is_returns = [f['is_return'] for f in folds]
    oos_returns = [f['oos_return'] for f in folds]
    periods = [f['period'] for f in folds]

    fig = go.Figure()

    # 樣本內（IS）
    fig.add_trace(go.Scatter(
        x=fold_ids,
        y=is_returns,
        mode='lines+markers',
        name='樣本內 (IS)',
        line=dict(color='#2563eb', width=2),
        marker=dict(size=8),
        hovertemplate='<b>Fold %{x}</b><br>IS 報酬: %{y:.1f}%<extra></extra>'
    ))

    # 樣本外（OOS）
    fig.add_trace(go.Scatter(
        x=fold_ids,
        y=oos_returns,
        mode='lines+markers',
        name='樣本外 (OOS)',
        line=dict(color='#f59e0b', width=2),
        marker=dict(size=8),
        hovertemplate='<b>Fold %{x}</b><br>OOS 報酬: %{y:.1f}%<extra></extra>'
    ))

    fig.update_layout(
        title='各 Fold 績效比較',
        xaxis_title='Fold 編號',
        yaxis_title='報酬率 (%)',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return apply_chart_theme(fig)


def plot_wfa_efficiency(folds: List[Dict]) -> go.Figure:
    """繪製 Walk-Forward 效率長條圖"""
    fold_ids = [f['fold_id'] for f in folds]
    efficiency = [f['oos_return'] / f['is_return'] if f['is_return'] != 0 else 0 for f in folds]

    # 顏色映射
    colors = []
    for eff in efficiency:
        if eff >= 0.7:
            colors.append('#22c55e')  # 綠色
        elif eff >= 0.5:
            colors.append('#eab308')  # 黃色
        else:
            colors.append('#ef4444')  # 紅色

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=fold_ids,
        y=efficiency,
        marker_color=colors,
        text=[f'{e:.2f}' for e in efficiency],
        textposition='outside',
        hovertemplate='<b>Fold %{x}</b><br>效率: %{y:.2f}<extra></extra>'
    ))

    # 基準線 (1.0)
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="#6b7280",
        annotation_text="完美效率 (1.0)",
        annotation_position="right"
    )

    # 警戒線 (0.7)
    fig.add_hline(
        y=0.7,
        line_dash="dot",
        line_color="#eab308",
        annotation_text="可接受 (0.7)",
        annotation_position="right"
    )

    fig.update_layout(
        title='Walk-Forward 效率（OOS / IS）',
        xaxis_title='Fold 編號',
        yaxis_title='效率比率',
        height=400,
        showlegend=False
    )

    return apply_chart_theme(fig)


def plot_sharpe_comparison(data: Dict) -> go.Figure:
    """繪製 Sharpe Ratio 比較長條圖"""
    original = data['original']
    deflated = data['deflated']

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=['原始 Sharpe', '校正後 Sharpe'],
        y=[original, deflated],
        marker_color=['#93c5fd', '#2563eb'],
        text=[f'{original:.2f}', f'{deflated:.2f}'],
        textposition='outside',
        hovertemplate='%{x}: %{y:.2f}<extra></extra>'
    ))

    # 顯著性基準線
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="#6b7280",
        annotation_text="顯著性基準 (1.0)",
        annotation_position="right"
    )

    # 標註變化百分比
    change_pct = ((deflated - original) / original * 100) if original != 0 else 0
    fig.add_annotation(
        x=1,
        y=deflated,
        text=f"{change_pct:+.0f}%",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#ef4444",
        ax=-40,
        ay=-40
    )

    fig.update_layout(
        title='Sharpe Ratio 比較',
        yaxis_title='Sharpe Ratio',
        height=400,
        showlegend=False
    )

    return apply_chart_theme(fig)


def plot_pbo_gauge(pbo: float) -> go.Figure:
    """繪製 PBO 儀表板"""
    # 顏色映射
    if pbo < 0.5:
        color = '#22c55e'
        status = '低風險'
    elif pbo < 0.7:
        color = '#eab308'
        status = '中風險'
    else:
        color = '#ef4444'
        status = '高風險'

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pbo * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "過擬合機率 (PBO)", 'font': {'size': 20}},
        delta={'reference': 50, 'suffix': '%'},
        gauge={
            'axis': {'range': [None, 100], 'ticksuffix': '%'},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': '#d1fae5'},
                {'range': [50, 70], 'color': '#fef9c3'},
                {'range': [70, 100], 'color': '#fee2e2'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))

    fig.update_layout(
        height=300,
        annotations=[
            dict(
                text=f'<b>{status}</b>',
                x=0.5,
                y=0.15,
                showarrow=False,
                font=dict(size=16, color=color)
            )
        ]
    )

    return fig


def plot_stress_equity_curve(event_data: Dict) -> go.Figure:
    """繪製壓力測試期間權益曲線"""
    equity = event_data['equity_curve']
    dates = event_data['dates']

    # 計算基準（買入持有）
    benchmark = 100 * np.ones_like(equity)
    benchmark[len(benchmark)//3:2*len(benchmark)//3] *= 0.7  # 模擬崩盤
    benchmark[2*len(benchmark)//3:] = benchmark[2*len(benchmark)//3-1] * (1 + np.cumsum(np.random.randn(len(benchmark)//3) * 0.01))

    fig = go.Figure()

    # 策略權益
    fig.add_trace(go.Scatter(
        x=dates,
        y=equity,
        mode='lines',
        name='策略',
        line=dict(color='#2563eb', width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>權益: $%{y:.2f}<extra></extra>'
    ))

    # 基準
    fig.add_trace(go.Scatter(
        x=dates,
        y=benchmark,
        mode='lines',
        name='基準',
        line=dict(color='#9ca3af', width=2, dash='dash'),
        hovertemplate='%{x|%Y-%m-%d}<br>權益: $%{y:.2f}<extra></extra>'
    ))

    # 標示事件期間（中間 1/3）
    event_start = dates[len(dates)//3]
    event_end = dates[2*len(dates)//3]

    fig.add_vrect(
        x0=event_start,
        x1=event_end,
        fillcolor="rgba(239, 68, 68, 0.1)",
        layer="below",
        line_width=0,
        annotation_text="事件期間",
        annotation_position="top left"
    )

    # 標註最大回撤點
    min_idx = np.argmin(equity)
    fig.add_annotation(
        x=dates[min_idx],
        y=equity[min_idx],
        text=f"最低點<br>${equity[min_idx]:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#ef4444",
        ax=40,
        ay=-40
    )

    fig.update_layout(
        title=f'{event_data["name"]} - 權益曲線',
        xaxis_title='日期',
        yaxis_title='權益 ($)',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return apply_chart_theme(fig)


def plot_drawdown_comparison(events: Dict) -> go.Figure:
    """繪製各事件回撤比較"""
    event_names = [e['name'] for e in events.values()]
    drawdowns = [e['max_drawdown'] for e in events.values()]

    # 顏色映射
    colors = []
    for dd in drawdowns:
        if dd > -15:
            colors.append('#22c55e')
        elif dd > -25:
            colors.append('#eab308')
        else:
            colors.append('#ef4444')

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=event_names,
        y=drawdowns,
        marker_color=colors,
        text=[f'{dd:.1f}%' for dd in drawdowns],
        textposition='outside',
        hovertemplate='%{x}<br>最大回撤: %{y:.1f}%<extra></extra>'
    ))

    fig.update_layout(
        title='各事件最大回撤比較',
        xaxis_title='事件',
        yaxis_title='最大回撤 (%)',
        height=400,
        showlegend=False
    )

    return apply_chart_theme(fig)


# ===== 主程式 =====

def main():
    # 渲染中文 sidebar 導航
    render_sidebar_navigation()

    st.title("📊 策略驗證")
    st.markdown("評估策略的統計顯著性與穩健性")

    # ===== 選擇器列 =====
    col_select, col_export = st.columns([3, 1])

    with col_select:
        strategies = get_available_strategies()
        strategy_options = [
            f"{s['name']} ({s['symbol']} {s['timeframe']})"
            for s in strategies
        ]
        selected_strategy = st.selectbox(
            "選擇策略",
            options=strategy_options,
            label_visibility="collapsed"
        )

    with col_export:
        if st.button("📥 匯出報告", type="primary", use_container_width=True):
            st.info("匯出功能開發中...")

    # ===== 載入資料 =====
    if not selected_strategy:
        st.info("""
        👈 請先從上方選擇一個策略

        驗證頁面將顯示：
        - 統計檢定結果
        - 交叉驗證分析
        - Sharpe Ratio 校正
        - 極端市況壓力測試
        """)
        st.stop()

    validation_result = load_validation_results()

    # ===== Tab 切換 =====
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 統計檢定",
        "🔄 交叉驗證",
        "📉 Sharpe 校正",
        "🔥 壓力測試"
    ])

    # ===== TAB 1: 統計檢定 =====
    with tab1:
        st.markdown("### 評估策略報酬的統計顯著性")

        bootstrap_data = validation_result['bootstrap']
        permutation_data = validation_result['permutation']

        # 指標卡片
        col1, col2, col3 = st.columns(3)

        with col1:
            bootstrap_status = "✅ 通過" if bootstrap_data['pass'] else "❌ 失敗"
            st.metric(
                label="Bootstrap Test",
                value=bootstrap_status,
                delta=f"95% CI: [{bootstrap_data['ci_lower']:.1f}%, {bootstrap_data['ci_upper']:.1f}%]"
            )

        with col2:
            perm_status = "✅ 顯著" if permutation_data['pass'] else "❌ 不顯著"
            st.metric(
                label="Permutation Test",
                value=perm_status,
                delta=f"p-value: {permutation_data['p_value']:.3f}"
            )

        with col3:
            st.metric(
                label="信賴水準",
                value=f"{bootstrap_data['confidence']*100:.0f}%",
                delta=f"CI 範圍: {bootstrap_data['ci_upper'] - bootstrap_data['ci_lower']:.1f}%"
            )

        # 狀態提示
        if bootstrap_data['pass'] and permutation_data['pass']:
            st.success("✅ 策略報酬具有統計顯著性，非隨機結果")
        elif bootstrap_data['pass'] or permutation_data['pass']:
            st.warning("⚠️ 部分檢定通過，建議謹慎評估")
        else:
            st.error("❌ 策略報酬無統計顯著性，可能為隨機結果")

        st.divider()

        # Bootstrap 分布圖
        st.plotly_chart(
            plot_bootstrap_distribution(bootstrap_data),
            use_container_width=True,
            config=get_chart_config()
        )

        # [D1] Bootstrap 分布解讀
        st.caption("""
        **[D1] Bootstrap 分布解讀**：
        - **藍色直方圖**：重抽樣 10,000 次後的報酬率分布
        - **紅色虛線**：95% 信賴區間上下界，區間內的報酬有 95% 可信度
        - **綠色實線**：平均報酬率
        - **判讀標準**：若信賴區間下界 > 0%，代表策略報酬顯著為正
        - **區間越窄**：估計越精確，結果越可靠
        """)

        # Permutation Test 結果
        st.plotly_chart(
            plot_permutation_test(permutation_data),
            use_container_width=True,
            config=get_chart_config()
        )

        # [D2] Permutation Test 解讀
        st.caption("""
        **[D2] Permutation Test 解讀**：
        - **灰色直方圖**：隨機打亂交易順序後的報酬分布（虛無假設）
        - **綠色實線**：實際策略報酬
        - **p-value**：實際報酬優於隨機的機率，<0.05 代表統計顯著
        - **判讀標準**：若實際報酬明顯超出隨機分布右側，代表策略非運氣
        - **意義**：排除「隨機交易也能達到此報酬」的可能性
        """)

    # ===== TAB 2: 交叉驗證 =====
    with tab2:
        st.markdown("### 評估策略在不同時期的穩定性")

        cv_data = validation_result['cross_validation']
        folds = cv_data['folds']

        # 指標卡片
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="CV 平均報酬",
                value=f"{cv_data['mean']:.1f}%",
                delta="穩定" if cv_data['stability'] > 0.7 else "波動"
            )

        with col2:
            st.metric(
                label="CV 標準差",
                value=f"±{cv_data['std']:.1f}%",
                delta="波動小" if cv_data['std'] < 5 else "波動大",
                delta_color="inverse"
            )

        with col3:
            stability_score = cv_data['stability']
            stability_status = "優秀" if stability_score > 0.8 else ("良好" if stability_score > 0.6 else "普通")
            st.metric(
                label="穩定性分數",
                value=f"{stability_score:.2f}/1.0",
                delta=f"{stability_status}"
            )

        # 狀態提示
        stability = cv_data['mean'] / cv_data['std'] if cv_data['std'] != 0 else 0

        if stability > 3.0:
            st.success("✅ 策略表現穩定，各時期一致")
        elif stability > 1.5:
            st.warning("⚠️ 策略穩定性中等，需持續監控")
        else:
            st.error("❌ 策略不穩定，不同時期差異大")

        st.divider()

        # Fold 績效折線圖
        st.plotly_chart(
            plot_cv_performance(folds),
            use_container_width=True,
            config=get_chart_config()
        )

        # [D3] 交叉驗證折線圖解讀
        st.caption("""
        **[D3] 交叉驗證折線圖解讀**：
        - **藍線 (IS)**：樣本內報酬，策略在訓練數據上的表現
        - **橘線 (OOS)**：樣本外報酬，策略在未見過數據上的表現
        - **兩線差距**：差距越小 = 過擬合風險越低
        - **理想狀態**：兩線接近且穩定，波動不大
        - **警訊**：OOS 大幅低於 IS = 過擬合警告
        """)

        # WFA 效率長條圖
        st.plotly_chart(
            plot_wfa_efficiency(folds),
            use_container_width=True,
            config=get_chart_config()
        )

        # [D4] WFA 效率解讀
        st.caption("""
        **[D4] Walk-Forward 效率解讀**：
        - **效率 = OOS報酬 / IS報酬**，衡量樣本外表現保持度
        - **綠色 (≥0.7)**：優秀，樣本外保持 70%+ 表現
        - **黃色 (0.5-0.7)**：可接受，有輕微過擬合
        - **紅色 (<0.5)**：過擬合嚴重，樣本外表現大幅衰退
        - **虛線 (1.0)**：完美效率，OOS = IS
        - **點線 (0.7)**：可接受門檻
        """)

    # ===== TAB 3: Sharpe 校正 =====
    with tab3:
        st.markdown("### 調整多重測試偏誤")

        sharpe_data = validation_result['sharpe_correction']

        # 指標卡片
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="原始 Sharpe",
                value=f"{sharpe_data['original']:.2f}",
                delta="⚠️ 可能過高" if sharpe_data['original'] > 2.0 else None
            )

        with col2:
            change_pct = ((sharpe_data['deflated'] - sharpe_data['original']) / sharpe_data['original'] * 100)
            st.metric(
                label="校正後 Sharpe",
                value=f"{sharpe_data['deflated']:.2f}",
                delta=f"{change_pct:+.0f}% 校正"
            )

        with col3:
            st.metric(
                label="測試次數",
                value=f"{sharpe_data['trials']} 次",
                delta=f"{abs(change_pct):.0f}% 懲罰"
            )

        st.divider()

        # Sharpe 比較長條圖
        st.plotly_chart(
            plot_sharpe_comparison(sharpe_data),
            use_container_width=True,
            config=get_chart_config()
        )

        # [D5] Sharpe 比較解讀
        st.caption("""
        **[D5] Sharpe Ratio 校正解讀**：
        - **淺藍色**：原始 Sharpe，未經調整
        - **深藍色**：校正後 Sharpe，考慮多重測試後的真實估計
        - **校正幅度**：測試次數越多，懲罰越大
        - **虛線 (1.0)**：顯著性基準，>1.0 才有統計意義
        - **判讀標準**：校正後 Sharpe >1.5 = 優秀，>1.0 = 及格
        - **意義**：避免因多次測試而高估策略表現
        """)

        # PBO 儀表板
        st.markdown("### 🎲 過擬合機率 (PBO)")

        pbo = sharpe_data['pbo']

        if pbo < 0.5:
            st.success("✅ 過擬合機率低，策略可靠")
        elif pbo < 0.7:
            st.warning("⚠️ 過擬合風險中等，需額外驗證")
        else:
            st.error("❌ 過擬合機率高，策略不可靠")

        st.plotly_chart(
            plot_pbo_gauge(pbo),
            use_container_width=True,
            config=get_chart_config()
        )

        # [D6] PBO 儀表板解讀
        st.caption("""
        **[D6] PBO 儀表板解讀**：
        - **PBO (Probability of Backtest Overfitting)**：過擬合機率
        - **綠色區域 (0-50%)**：過擬合風險低，策略可信
        - **黃色區域 (50-70%)**：中等風險，需謹慎使用
        - **紅色區域 (70-100%)**：高風險，策略可能不可靠
        - **紅色門檻線 (70%)**：警戒線，超過需重新評估策略
        - **數值意義**：PBO 35% = 有 35% 機率是過擬合產生的虛假績效
        """)

        st.info(
            f"**解讀說明**\n\n"
            f"PBO = {pbo:.2%} 表示有 {pbo:.0%} 的機率是因為過度測試而產生的虛假績效。\n\n"
            f"{'✅ 此策略的過擬合風險較低，可信度較高。' if pbo < 0.5 else '⚠️ 建議進行額外的樣本外驗證。'}"
        )

    # ===== TAB 4: 壓力測試 =====
    with tab4:
        st.markdown("### 評估極端市況下的表現")

        stress_data = validation_result['stress_test']
        events = stress_data['events']

        # 事件選擇器
        event_names = list(events.keys())
        event_labels = [events[k]['name'] for k in event_names]
        event_labels.append("全部")

        selected_event = st.radio(
            "選擇事件",
            options=event_labels,
            horizontal=True
        )

        # 如果選擇單一事件
        if selected_event != "全部":
            event_key = event_names[event_labels.index(selected_event)]
            event_data = events[event_key]

            # 指標卡片
            col1, col2, col3 = st.columns(3)

            with col1:
                dd = event_data['max_drawdown']
                dd_status = "抗壓強" if dd > -15 else ("中等" if dd > -25 else "風險高")
                st.metric(
                    label="最大回撤",
                    value=f"{dd:.1f}%",
                    delta=dd_status
                )

            with col2:
                recovery = event_data['recovery_days']
                recovery_status = "快速恢復" if recovery < 50 else ("中等" if recovery < 80 else "恢復慢")
                st.metric(
                    label="恢復天數",
                    value=f"{recovery} 天",
                    delta=recovery_status
                )

            with col3:
                sharpe = event_data['sharpe']
                sharpe_status = "維持良好" if sharpe > 0.8 else ("下降" if sharpe > 0.5 else "大幅下降")
                st.metric(
                    label="事件期 Sharpe",
                    value=f"{sharpe:.2f}",
                    delta=sharpe_status
                )

            # 狀態提示
            if dd > -15:
                st.success("✅ 抗壓能力強，極端市況影響小")
            elif dd > -25:
                st.warning("⚠️ 中等回撤，需注意風控")
            else:
                st.error("❌ 極端市況下風險過高")

            st.divider()

            # 權益曲線
            st.plotly_chart(
                plot_stress_equity_curve(event_data),
                use_container_width=True,
                config=get_chart_config()
            )

            # [D7] 壓力測試權益曲線解讀
            st.caption("""
            **[D7] 壓力測試權益曲線解讀**：
            - **藍線**：策略在極端事件期間的權益變化
            - **灰色虛線**：基準（買入持有）表現
            - **紅色陰影區域**：事件發生期間
            - **最低點標註**：權益最低時刻和金額
            - **判讀標準**：
              - 策略線在基準線上方 = 抗壓能力強
              - 最低點後快速回升 = 恢復能力佳
              - 回撤幅度 <15% = 優秀，<25% = 及格
            """)

        else:
            # 顯示所有事件的回撤比較
            st.plotly_chart(
                plot_drawdown_comparison(events),
                use_container_width=True,
                config=get_chart_config()
            )

            # [D8] 回撤比較解讀
            st.caption("""
            **[D8] 各事件回撤比較解讀**：
            - **綠色 (>-15%)**：抗壓能力強，極端市況影響有限
            - **黃色 (-15% ~ -25%)**：中等回撤，需注意風控
            - **紅色 (<-25%)**：風險過高，極端市況下損失嚴重
            - **事件類型**：不同黑天鵝事件代表不同市場壓力
              - COVID-19：快速暴跌後反彈
              - LUNA/FTX：信心崩潰型下跌
            - **選擇建議**：選擇各事件都保持綠/黃色的策略
            """)

            # 事件摘要表
            st.markdown("### 📋 事件摘要")

            summary_data = []
            for key, event in events.items():
                summary_data.append({
                    '事件': event['name'],
                    '最大回撤': f"{event['max_drawdown']:.1f}%",
                    '恢復天數': event['recovery_days'],
                    'Sharpe': f"{event['sharpe']:.2f}"
                })

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
