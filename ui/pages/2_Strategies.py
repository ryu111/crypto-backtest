"""
策略列表頁面

展示所有策略實驗結果，支援複雜篩選、排序、分頁、展開式詳情。
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Any
import json
import sys

# 加入專案路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ui.styles import get_common_css, GRADE_COLORS


# ===== 設定頁面 =====
st.set_page_config(
    page_title="策略列表 - 合約交易系統",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== 自訂樣式 =====
st.markdown(get_common_css(), unsafe_allow_html=True)


# ===== 資料載入函數 =====

@st.cache_data
def load_strategy_results() -> pd.DataFrame:
    """載入所有策略驗證結果"""
    # TODO: 實際從檔案系統載入結果
    # 目前返回範例資料

    sample_data = [
        {
            'strategy_name': 'MA Cross (10/30)',
            'strategy_type': '趨勢',
            'symbol': 'BTCUSDT',
            'timeframe': '4h',
            'total_return': 45.8,
            'annual_return': 28.2,
            'sharpe_ratio': 1.85,
            'max_drawdown': 12.5,
            'total_trades': 158,
            'win_rate': 62.5,
            'grade': 'A',
            'wfa_efficiency': 0.85,
            'params': {'fast_period': 10, 'slow_period': 30},
            'created_at': '2024-01-10 14:30:00'
        },
        {
            'strategy_name': 'RSI Mean Reversion',
            'strategy_type': '均值回歸',
            'symbol': 'ETHUSDT',
            'timeframe': '1h',
            'total_return': 32.1,
            'annual_return': 22.4,
            'sharpe_ratio': 1.62,
            'max_drawdown': 15.8,
            'total_trades': 245,
            'win_rate': 58.3,
            'grade': 'B',
            'wfa_efficiency': 0.72,
            'params': {'rsi_period': 14, 'oversold': 30, 'overbought': 70},
            'created_at': '2024-01-10 12:15:00'
        },
        {
            'strategy_name': 'Supertrend Momentum',
            'strategy_type': '動量',
            'symbol': 'BTCUSDT',
            'timeframe': '1d',
            'total_return': 68.5,
            'annual_return': 41.2,
            'sharpe_ratio': 2.15,
            'max_drawdown': 18.3,
            'total_trades': 89,
            'win_rate': 71.2,
            'grade': 'A',
            'wfa_efficiency': 0.91,
            'params': {'atr_period': 10, 'multiplier': 3.0},
            'created_at': '2024-01-09 16:45:00'
        },
        {
            'strategy_name': 'MACD Cross',
            'strategy_type': '動量',
            'symbol': 'ETHUSDT',
            'timeframe': '4h',
            'total_return': 18.9,
            'annual_return': 12.8,
            'sharpe_ratio': 1.12,
            'max_drawdown': 22.4,
            'total_trades': 167,
            'win_rate': 54.1,
            'grade': 'C',
            'wfa_efficiency': 0.58,
            'params': {'fast': 12, 'slow': 26, 'signal': 9},
            'created_at': '2024-01-09 10:20:00'
        },
    ]

    df = pd.DataFrame(sample_data)
    return df


def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """套用篩選條件"""
    filtered = df.copy()

    # 數值篩選
    if filters['min_sharpe'] > 0:
        filtered = filtered[filtered['sharpe_ratio'] >= filters['min_sharpe']]

    if filters['min_return'] > -50:
        filtered = filtered[filtered['total_return'] >= filters['min_return']]

    if filters['max_drawdown'] < 50:
        filtered = filtered[filtered['max_drawdown'] <= filters['max_drawdown']]

    if filters['min_trades'] > 0:
        filtered = filtered[filtered['total_trades'] >= filters['min_trades']]

    # 分類篩選
    if filters['grades']:
        filtered = filtered[filtered['grade'].isin(filters['grades'])]

    if filters['strategy_types']:
        filtered = filtered[filtered['strategy_type'].isin(filters['strategy_types'])]

    if filters['symbols']:
        filtered = filtered[filtered['symbol'].isin(filters['symbols'])]

    if filters['timeframes']:
        filtered = filtered[filtered['timeframe'].isin(filters['timeframes'])]

    return filtered


def sort_dataframe(df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    """排序資料"""
    sort_map = {
        'Sharpe Ratio (高→低)': ('sharpe_ratio', False),
        '報酬率 (高→低)': ('total_return', False),
        '回撤 (低→高)': ('max_drawdown', True),
        '時間 (新→舊)': ('created_at', False),
    }

    if sort_by in sort_map:
        column, ascending = sort_map[sort_by]
        return df.sort_values(column, ascending=ascending)

    return df


def render_grade_badge(grade: str) -> str:
    """渲染等級徽章"""
    return f'<span class="grade-badge grade-{grade}">{grade}</span>'


def render_metric_card(title: str, value: str, delta: str = None):
    """渲染指標卡片"""
    st.metric(label=title, value=value, delta=delta)


def plot_equity_curve(strategy_name: str) -> go.Figure:
    """繪製權益曲線（範例）"""
    # TODO: 實際從結果載入
    import numpy as np

    days = 100
    equity = 10000 * (1 + np.cumsum(np.random.randn(days) * 0.02))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=equity,
        mode='lines',
        name='權益',
        line=dict(color='var(--color-primary)', width=2)
    ))

    fig.update_layout(
        title=f'{strategy_name} - 權益曲線',
        xaxis_title='交易日',
        yaxis_title='權益 ($)',
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode='x unified'
    )

    return fig


def plot_monthly_heatmap(strategy_name: str) -> go.Figure:
    """繪製月度報酬熱力圖（範例）"""
    # TODO: 實際從結果載入
    import numpy as np

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    returns = np.random.randn(12) * 5 + 2

    fig = go.Figure(data=go.Heatmap(
        z=[returns],
        x=months,
        y=['2024'],
        colorscale='RdYlGn',
        text=[[f'{r:.1f}%' for r in returns]],
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title='報酬率 %')
    ))

    fig.update_layout(
        title=f'{strategy_name} - 月度報酬',
        height=200,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig


# ===== 主程式 =====

def main():
    st.title("📊 策略列表")
    st.markdown("篩選和查看所有策略實驗結果")

    # 載入資料
    df_all = load_strategy_results()

    # ===== 側邊欄：篩選器 =====
    with st.sidebar:
        st.header("🔍 篩選器")

        # 數值篩選
        st.subheader("數值篩選")
        min_sharpe = st.slider(
            "最小 Sharpe Ratio",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.1
        )

        min_return = st.slider(
            "最小報酬率 (%)",
            min_value=-50,
            max_value=200,
            value=-50,
            step=5
        )

        max_drawdown = st.slider(
            "最大回撤 (%)",
            min_value=0,
            max_value=50,
            value=50,
            step=5
        )

        min_trades = st.slider(
            "最小交易筆數",
            min_value=0,
            max_value=500,
            value=0,
            step=10
        )

        # 分類篩選
        st.subheader("分類篩選")

        grades = st.multiselect(
            "驗證等級",
            options=['A', 'B', 'C', 'D', 'F'],
            default=['A', 'B', 'C', 'D', 'F']
        )

        strategy_types = st.multiselect(
            "策略類型",
            options=['趨勢', '動量', '均值回歸'],
            default=['趨勢', '動量', '均值回歸']
        )

        symbols = st.multiselect(
            "標的",
            options=['BTCUSDT', 'ETHUSDT'],
            default=['BTCUSDT', 'ETHUSDT']
        )

        timeframes = st.multiselect(
            "時間框架",
            options=['1h', '4h', '1d'],
            default=['1h', '4h', '1d']
        )

        # 排序
        st.subheader("排序")
        sort_by = st.selectbox(
            "排序依據",
            options=[
                'Sharpe Ratio (高→低)',
                '報酬率 (高→低)',
                '回撤 (低→高)',
                '時間 (新→舊)'
            ]
        )

        # 重置按鈕
        if st.button("🔄 重置篩選", use_container_width=True):
            st.rerun()

    # ===== 套用篩選和排序 =====
    filters = {
        'min_sharpe': min_sharpe,
        'min_return': min_return,
        'max_drawdown': max_drawdown,
        'min_trades': min_trades,
        'grades': grades,
        'strategy_types': strategy_types,
        'symbols': symbols,
        'timeframes': timeframes
    }

    df_filtered = apply_filters(df_all, filters)
    df_sorted = sort_dataframe(df_filtered, sort_by)

    # ===== 概覽指標 =====
    st.subheader("📈 概覽")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric_card(
            "總策略數",
            f"{len(df_sorted)} / {len(df_all)}",
            f"{len(df_sorted) - len(df_all)}" if len(df_sorted) != len(df_all) else None
        )

    with col2:
        avg_sharpe = df_sorted['sharpe_ratio'].mean() if len(df_sorted) > 0 else 0
        render_metric_card("平均 Sharpe", f"{avg_sharpe:.2f}")

    with col3:
        avg_return = df_sorted['total_return'].mean() if len(df_sorted) > 0 else 0
        render_metric_card("平均報酬率", f"{avg_return:.1f}%")

    with col4:
        grade_a_count = len(df_sorted[df_sorted['grade'] == 'A'])
        render_metric_card("A 級策略", f"{grade_a_count}")

    st.divider()

    # ===== 結果表格 =====
    st.subheader("📋 策略列表")

    if len(df_sorted) == 0:
        st.warning("沒有符合篩選條件的策略")
        return

    # 分頁設定
    ITEMS_PER_PAGE = 20
    total_pages = (len(df_sorted) - 1) // ITEMS_PER_PAGE + 1

    # 分頁控制
    col_prev, col_page, col_next = st.columns([1, 2, 1])

    with col_page:
        current_page = st.number_input(
            "頁碼",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
            label_visibility="collapsed"
        )

    # 計算分頁範圍
    start_idx = (current_page - 1) * ITEMS_PER_PAGE
    end_idx = min(start_idx + ITEMS_PER_PAGE, len(df_sorted))
    df_page = df_sorted.iloc[start_idx:end_idx]

    # 顯示表格
    display_df = df_page[[
        'strategy_name', 'total_return', 'annual_return', 'sharpe_ratio',
        'max_drawdown', 'total_trades', 'win_rate', 'grade', 'wfa_efficiency'
    ]].copy()

    display_df.columns = [
        '策略名稱', '報酬率 (%)', '年化報酬 (%)', 'Sharpe',
        'MaxDD (%)', '交易筆數', '勝率 (%)', '等級', '過擬合率'
    ]

    # 格式化數值
    display_df['報酬率 (%)'] = display_df['報酬率 (%)'].apply(lambda x: f"{x:.1f}%")
    display_df['年化報酬 (%)'] = display_df['年化報酬 (%)'].apply(lambda x: f"{x:.1f}%")
    display_df['Sharpe'] = display_df['Sharpe'].apply(lambda x: f"{x:.2f}")
    display_df['MaxDD (%)'] = display_df['MaxDD (%)'].apply(lambda x: f"{x:.1f}%")
    display_df['勝率 (%)'] = display_df['勝率 (%)'].apply(lambda x: f"{x:.1f}%")
    display_df['過擬合率'] = display_df['過擬合率'].apply(lambda x: f"{x:.2f}")

    # 使用 dataframe 展示
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=600
    )

    st.caption(f"顯示第 {start_idx + 1}-{end_idx} 筆，共 {len(df_sorted)} 筆")

    # ===== 詳情展開 =====
    st.divider()
    st.subheader("🔍 策略詳情")

    selected_strategy = st.selectbox(
        "選擇策略查看詳情",
        options=df_page['strategy_name'].tolist(),
        label_visibility="collapsed"
    )

    if selected_strategy:
        strategy_data = df_page[df_page['strategy_name'] == selected_strategy].iloc[0]

        with st.expander(f"📊 {selected_strategy} - 完整資訊", expanded=True):
            # 基本資訊
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**基本資訊**")
                st.write(f"類型：{strategy_data['strategy_type']}")
                st.write(f"標的：{strategy_data['symbol']}")
                st.write(f"時間框架：{strategy_data['timeframe']}")
                st.write(f"建立時間：{strategy_data['created_at']}")

            with col2:
                st.markdown("**績效指標**")
                st.write(f"總報酬率：{strategy_data['total_return']:.1f}%")
                st.write(f"年化報酬：{strategy_data['annual_return']:.1f}%")
                st.write(f"Sharpe Ratio：{strategy_data['sharpe_ratio']:.2f}")
                st.write(f"最大回撤：{strategy_data['max_drawdown']:.1f}%")

            with col3:
                st.markdown("**交易統計**")
                st.write(f"交易筆數：{strategy_data['total_trades']}")
                st.write(f"勝率：{strategy_data['win_rate']:.1f}%")
                st.write(f"過擬合率：{strategy_data['wfa_efficiency']:.2f}")
                st.markdown(render_grade_badge(strategy_data['grade']), unsafe_allow_html=True)

            # 參數
            st.markdown("**策略參數**")
            params_json = json.dumps(strategy_data['params'], indent=2, ensure_ascii=False)
            st.code(params_json, language='json')

            # 權益曲線
            st.plotly_chart(
                plot_equity_curve(selected_strategy),
                use_container_width=True
            )

            # 月度報酬熱力圖
            st.plotly_chart(
                plot_monthly_heatmap(selected_strategy),
                use_container_width=True
            )

            # AI 洞察（範例）
            st.markdown("**🤖 AI 洞察**")
            st.info(
                f"此策略在 {strategy_data['timeframe']} 時間框架下表現{strategy_data['grade']}級，"
                f"Sharpe Ratio 為 {strategy_data['sharpe_ratio']:.2f}，"
                f"顯示出良好的風險調整後報酬。最大回撤為 {strategy_data['max_drawdown']:.1f}%，"
                f"處於可接受範圍。建議進一步監控過擬合率 ({strategy_data['wfa_efficiency']:.2f})。"
            )

    # ===== 匯出功能 =====
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 匯出篩選結果 (CSV)", use_container_width=True):
            csv = df_sorted.to_csv(index=False)
            st.download_button(
                label="下載 CSV",
                data=csv,
                file_name=f"strategies_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    with col2:
        if selected_strategy and st.button("📥 匯出選中策略詳情 (JSON)", use_container_width=True):
            strategy_json = strategy_data.to_json(indent=2, force_ascii=False)
            st.download_button(
                label="下載 JSON",
                data=strategy_json,
                file_name=f"{selected_strategy.replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
