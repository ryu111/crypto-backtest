"""
AI 回測系統 Dashboard

展示整體統計、趨勢圖表、Top 排行榜
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import sys

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ui.utils import render_sidebar_navigation, render_page_header
from ui.styles import get_common_css
from ui.theme_switcher import apply_theme, get_current_theme
from ui.chart_config import get_chart_layout, get_chart_colors

# 設定頁面配置
st.set_page_config(
    page_title="Dashboard - AI 回測系統",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 套用主題
apply_theme()
theme = get_current_theme()


# ============================================================================
# 資料載入
# ============================================================================

@st.cache_data(ttl=60)
def load_experiments() -> List[Dict]:
    """載入實驗資料"""
    experiments_file = Path(__file__).parent.parent.parent / "learning" / "experiments.json"

    if not experiments_file.exists():
        return []

    with open(experiments_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get('experiments', [])


def parse_timestamp(ts_str: str) -> datetime:
    """解析時間戳"""
    try:
        return datetime.fromisoformat(ts_str)
    except:
        return datetime.now()


def calculate_grade(sharpe: float) -> str:
    """計算評級 (基於 Sharpe Ratio)"""
    if sharpe >= 2.0:
        return 'A'
    elif sharpe >= 1.5:
        return 'B'
    elif sharpe >= 1.0:
        return 'C'
    elif sharpe >= 0.5:
        return 'D'
    else:
        return 'F'


# ============================================================================
# 資料處理
# ============================================================================

def prepare_dashboard_data(experiments: List[Dict]) -> Dict:
    """準備 Dashboard 資料"""
    if not experiments:
        return {
            'total_experiments': 0,
            'validated_count': 0,
            'best_sharpe': 0,
            'avg_sharpe': 0,
            'unique_strategies': 0,
            'experiments_df': pd.DataFrame(),
            'grade_counts': {},
            'strategy_type_stats': {}
        }

    # 轉換為 DataFrame
    df_list = []
    for exp in experiments:
        df_list.append({
            'id': exp['id'],
            'timestamp': parse_timestamp(exp['timestamp']),
            'strategy_name': exp['strategy']['name'],
            'strategy_type': exp['strategy']['type'],
            'version': exp['strategy'].get('version', '1.0'),
            'symbol': exp['config'].get('symbol', 'BTCUSDT'),
            'timeframe': exp['config'].get('timeframe', '1h'),
            'sharpe_ratio': exp['results'].get('sharpe_ratio', 0),
            'total_return': exp['results'].get('total_return', 0),
            'max_drawdown': exp['results'].get('max_drawdown', 0),
            'win_rate': exp['results'].get('win_rate', 0),
            'profit_factor': exp['results'].get('profit_factor', 0),
            'total_trades': exp['results'].get('total_trades', 0),
        })

    df = pd.DataFrame(df_list)

    # 計算評級
    df['grade'] = df['sharpe_ratio'].apply(calculate_grade)

    # 統計資料
    total_experiments = len(df)
    validated_count = len(df[df['grade'].isin(['A', 'B'])])
    best_sharpe = df['sharpe_ratio'].max()
    avg_sharpe = df['sharpe_ratio'].mean()
    unique_strategies = df['strategy_name'].nunique()

    # 評級分布
    grade_counts = df['grade'].value_counts().to_dict()

    # 策略類型統計
    strategy_type_stats = df.groupby('strategy_type').agg({
        'sharpe_ratio': 'mean',
        'strategy_name': 'count'
    }).rename(columns={'strategy_name': 'count'}).to_dict('index')

    return {
        'total_experiments': total_experiments,
        'validated_count': validated_count,
        'best_sharpe': best_sharpe,
        'avg_sharpe': avg_sharpe,
        'unique_strategies': unique_strategies,
        'experiments_df': df,
        'grade_counts': grade_counts,
        'strategy_type_stats': strategy_type_stats
    }


# ============================================================================
# UI 元件
# ============================================================================

def render_metric_card(label: str, value: str, delta: Optional[str] = None):
    """渲染指標卡片"""
    st.metric(label=label, value=value, delta=delta)


def render_sharpe_distribution(df: pd.DataFrame):
    """渲染 Sharpe 分布直方圖"""
    if df.empty:
        st.info("尚無資料")
        return

    fig = go.Figure()

    colors = get_chart_colors(theme, n=1)

    # 直方圖
    fig.add_trace(go.Histogram(
        x=df['sharpe_ratio'],
        nbinsx=20,
        name='Sharpe 分布',
        marker_color=colors[0],
        opacity=0.7
    ))

    # 門檻線
    fig.add_vline(x=1.0, line_dash="dash", line_color="orange",
                  annotation_text="門檻 1.0", annotation_position="top")
    fig.add_vline(x=2.0, line_dash="dash", line_color="green",
                  annotation_text="門檻 2.0", annotation_position="top")

    fig.update_layout(**get_chart_layout(
        theme=theme,
        title="Sharpe Ratio 分布",
        xaxis_title="Sharpe Ratio",
        yaxis_title="實驗數量",
        height=400,
        showlegend=False,
        hovermode='x unified'
    ))

    st.plotly_chart(fig, use_container_width=True)

    # [A2] Sharpe 分布解讀
    above_1 = len(df[df['sharpe_ratio'] >= 1.0])
    above_2 = len(df[df['sharpe_ratio'] >= 2.0])
    total = len(df)
    median_sharpe = df['sharpe_ratio'].median()

    st.caption(f"""
    **[A2] 圖表解讀**：此圖顯示所有策略的 Sharpe Ratio 分布。
    - **橘線（1.0）**：基本門檻，超過代表策略有正向風險調整收益
    - **綠線（2.0）**：優秀門檻，超過代表策略表現優異
    - **目前狀況**：{above_1}/{total} ({above_1/total*100:.0f}%) 策略達基本門檻，{above_2}/{total} ({above_2/total*100:.0f}%) 達優秀門檻
    - **中位數**：{median_sharpe:.2f}（{'✅ 良好' if median_sharpe >= 1.0 else '⚠️ 需改善'}）
    """)


def render_grade_distribution(grade_counts: Dict[str, int]):
    """渲染評級分布圓餅圖"""
    if not grade_counts:
        st.info("尚無資料")
        return

    # 定義顏色
    grade_colors = {
        'A': '#22c55e',  # green
        'B': '#3b82f6',  # blue
        'C': '#eab308',  # yellow
        'D': '#f97316',  # orange
        'F': '#ef4444'   # red
    }

    labels = list(grade_counts.keys())
    values = list(grade_counts.values())
    colors = [grade_colors.get(g, '#9ca3af') for g in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hole=0.4,
        textinfo='label+percent',
        textposition='auto'
    )])

    fig.update_layout(**get_chart_layout(
        theme=theme,
        title="評級分布",
        height=400,
        showlegend=True
    ))

    st.plotly_chart(fig, use_container_width=True)

    # [A3] 評級分布說明
    a_count = grade_counts.get('A', 0)
    b_count = grade_counts.get('B', 0)
    f_count = grade_counts.get('F', 0)
    total_count = sum(grade_counts.values())
    good_rate = (a_count + b_count) / total_count * 100 if total_count > 0 else 0

    st.caption(f"""
    **[A3] 評級說明**：基於 Sharpe Ratio 分級。
    - **A 級（≥2.0）**：卓越，可直接實盤考慮
    - **B 級（≥1.5）**：優良，建議進一步驗證後使用
    - **C 級（≥1.0）**：及格，需要優化參數
    - **D 級（≥0.5）**：偏弱，不建議使用
    - **F 級（<0.5）**：不及格，需重新設計
    - **目前狀況**：A+B 級佔 {good_rate:.0f}%（{a_count + b_count}/{total_count}），{'✅ 品質良好' if good_rate >= 30 else '⚠️ 需要更多優質策略'}
    """)


def render_time_trend(df: pd.DataFrame):
    """渲染時間趨勢圖"""
    if df.empty:
        st.info("尚無資料")
        return

    # 按日期分組，取最佳 Sharpe
    df_sorted = df.sort_values('timestamp')
    df_sorted['date'] = df_sorted['timestamp'].dt.date

    daily_best = df_sorted.groupby('date')['sharpe_ratio'].max().reset_index()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=daily_best['date'],
        y=daily_best['sharpe_ratio'],
        mode='lines+markers',
        name='每日最佳 Sharpe',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=8)
    ))

    fig.update_layout(**get_chart_layout(
        theme=theme,
        title="時間趨勢 - 每日最佳 Sharpe Ratio",
        xaxis_title="日期",
        yaxis_title="Sharpe Ratio",
        height=400,
        hovermode='x unified'
    ))

    st.plotly_chart(fig, use_container_width=True)


def render_top_strategies(df: pd.DataFrame, n: int = 10):
    """渲染 Top N 排行榜"""
    if df.empty:
        st.info("尚無資料")
        return

    # 排序取前 N
    top_n = df.nlargest(n, 'sharpe_ratio')[
        ['strategy_name', 'sharpe_ratio', 'total_return', 'max_drawdown', 'grade']
    ].copy()

    # 格式化數值
    top_n['sharpe_ratio'] = top_n['sharpe_ratio'].apply(lambda x: f"{x:.2f}")
    top_n['total_return'] = top_n['total_return'].apply(lambda x: f"{x*100:.1f}%")
    top_n['max_drawdown'] = top_n['max_drawdown'].apply(lambda x: f"{x*100:.1f}%")

    # 重新命名欄位
    top_n = top_n.rename(columns={
        'strategy_name': '策略',
        'sharpe_ratio': 'Sharpe',
        'total_return': '報酬率',
        'max_drawdown': 'MaxDD',
        'grade': '評級'
    })

    # 加入排名
    top_n.insert(0, '排名', range(1, len(top_n) + 1))

    # 顯示表格
    st.dataframe(
        top_n,
        use_container_width=True,
        hide_index=True,
        column_config={
            '排名': st.column_config.NumberColumn(width="small"),
            '評級': st.column_config.TextColumn(width="small")
        }
    )

    # [A4] 排行榜說明
    st.caption("""
    **[A4] 排行榜說明**：按 Sharpe Ratio 排序的前 10 名策略。
    - **Sharpe**：風險調整收益，>2.0 優秀，>1.0 及格
    - **報酬率**：總回報百分比，越高越好
    - **MaxDD**：最大回撤，代表歷史最大虧損幅度，越小越好（<20% 理想）
    - **評級**：綜合評分，A/B 級可考慮實盤
    - **建議**：選擇排行靠前且 MaxDD 可接受的策略
    """)


def render_strategy_type_analysis(stats: Dict):
    """渲染策略類型分析"""
    if not stats:
        st.info("尚無資料")
        return

    # 準備資料
    types = []
    avg_sharpe = []
    counts = []

    for strategy_type, data in stats.items():
        types.append(strategy_type)
        avg_sharpe.append(data['sharpe_ratio'])
        counts.append(data['count'])

    # 建立子圖
    fig = go.Figure()

    # Bar chart
    fig.add_trace(go.Bar(
        x=types,
        y=avg_sharpe,
        name='平均 Sharpe',
        marker_color='#3b82f6',
        text=[f"{s:.2f}" for s in avg_sharpe],
        textposition='auto'
    ))

    fig.update_layout(**get_chart_layout(
        theme=theme,
        title="策略類型平均表現",
        xaxis_title="策略類型",
        yaxis_title="平均 Sharpe Ratio",
        height=350,
        showlegend=False
    ))

    st.plotly_chart(fig, use_container_width=True)

    # 顯示實驗數量
    st.caption("各類型實驗數量:")
    cols = st.columns(len(types))
    for i, (t, c) in enumerate(zip(types, counts)):
        with cols[i]:
            st.metric(label=t, value=f"{c} 個")


def render_overall_recommendations(data: Dict):
    """[A5] 渲染整體建議"""
    total = data['total_experiments']
    validated = data['validated_count']
    avg_sharpe = data['avg_sharpe']
    best_sharpe = data['best_sharpe']
    df = data['experiments_df']

    recommendations = []

    # 基於驗證率的建議
    validation_rate = validated / total if total > 0 else 0
    if validation_rate < 0.1:
        recommendations.append("⚠️ **驗證率過低**：僅 {:.0f}% 策略通過驗證，建議重新審視策略設計或調整參數範圍".format(validation_rate * 100))
    elif validation_rate < 0.3:
        recommendations.append("📊 **驗證率一般**：{:.0f}% 策略通過驗證，可嘗試優化表現較好的策略".format(validation_rate * 100))
    else:
        recommendations.append("✅ **驗證率良好**：{:.0f}% 策略通過驗證，可進行實盤前的最終壓力測試".format(validation_rate * 100))

    # 基於 Sharpe 的建議
    if avg_sharpe < 0.5:
        recommendations.append("⚠️ **平均 Sharpe 過低**：{:.2f}，策略整體風險調整收益不佳，需重新設計".format(avg_sharpe))
    elif avg_sharpe < 1.0:
        recommendations.append("📊 **平均 Sharpe 普通**：{:.2f}，有改善空間，建議優化入場/出場邏輯".format(avg_sharpe))
    else:
        recommendations.append("✅ **平均 Sharpe 良好**：{:.2f}，策略品質不錯".format(avg_sharpe))

    # 基於最大回撤的建議
    if not df.empty:
        max_dd = df['max_drawdown'].max()
        if max_dd > 0.3:
            recommendations.append("⚠️ **回撤風險高**：最大回撤達 {:.1f}%，建議加強止損機制".format(max_dd * 100))
        elif max_dd > 0.2:
            recommendations.append("📊 **回撤風險中等**：最大回撤 {:.1f}%，可考慮降低槓桿".format(max_dd * 100))
        else:
            recommendations.append("✅ **回撤控制良好**：最大回撤 {:.1f}%，風險管理得當".format(max_dd * 100))

    # 下一步建議
    recommendations.append("")
    recommendations.append("**📋 下一步建議：**")
    if best_sharpe >= 2.0:
        recommendations.append("1. 對 A 級策略進行壓力測試（Validation 頁面）")
        recommendations.append("2. 檢查策略間相關性（Risk Dashboard）")
        recommendations.append("3. 準備小資金實盤測試")
    elif best_sharpe >= 1.0:
        recommendations.append("1. 優化 B/C 級策略的參數")
        recommendations.append("2. 增加更多回測樣本")
        recommendations.append("3. 分析失敗策略的共同點")
    else:
        recommendations.append("1. 重新檢視策略邏輯")
        recommendations.append("2. 考慮更換指標組合")
        recommendations.append("3. 增加訓練數據量")

    for rec in recommendations:
        if rec:
            st.markdown(rec)


def render_recent_activity(df: pd.DataFrame, n: int = 10):
    """渲染最近活動"""
    if df.empty:
        st.info("尚無資料")
        return

    # 排序取最近 N 筆
    recent = df.nlargest(n, 'timestamp')[
        ['timestamp', 'strategy_name', 'sharpe_ratio', 'grade']
    ].copy()

    # 格式化
    recent['timestamp'] = recent['timestamp'].dt.strftime('%m-%d %H:%M')
    recent['sharpe_ratio'] = recent['sharpe_ratio'].apply(lambda x: f"{x:.2f}")

    # 重新命名
    recent = recent.rename(columns={
        'timestamp': '時間',
        'strategy_name': '策略',
        'sharpe_ratio': 'Sharpe',
        'grade': '評級'
    })

    st.dataframe(
        recent,
        use_container_width=True,
        hide_index=True
    )


# ============================================================================
# 主要 UI
# ============================================================================

def render_summary_box(data: Dict):
    """[A1] 渲染整體績效摘要框"""
    total = data['total_experiments']
    validated = data['validated_count']
    avg_sharpe = data['avg_sharpe']
    best_sharpe = data['best_sharpe']

    # 判斷整體狀態
    validation_rate = validated / total if total > 0 else 0

    if validation_rate >= 0.3 and avg_sharpe >= 1.5:
        status = "✅ 優秀"
        status_color = "green"
        summary = f"有 {validation_rate*100:.0f}% 策略通過驗證，平均 Sharpe {avg_sharpe:.2f}，表現優於基準"
    elif validation_rate >= 0.1 or avg_sharpe >= 1.0:
        status = "⚠️ 普通"
        status_color = "orange"
        summary = f"有 {validation_rate*100:.0f}% 策略通過驗證，平均 Sharpe {avg_sharpe:.2f}，仍有改善空間"
    else:
        status = "❌ 需改善"
        status_color = "red"
        summary = f"僅 {validation_rate*100:.0f}% 策略通過驗證，建議調整策略參數或增加回測樣本"

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
                border-left: 4px solid {'#22c55e' if status_color == 'green' else '#eab308' if status_color == 'orange' else '#ef4444'};
                padding: 16px 20px; border-radius: 8px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 1.2em; font-weight: 600;">整體績效摘要</span>
                <span style="margin-left: 12px; padding: 4px 12px; border-radius: 12px;
                       background: {'#dcfce7' if status_color == 'green' else '#fef9c3' if status_color == 'orange' else '#fee2e2'};
                       color: {'#166534' if status_color == 'green' else '#854d0e' if status_color == 'orange' else '#991b1b'};">
                    {status}
                </span>
            </div>
            <div style="color: #6b7280; font-size: 0.9em;">
                共 {total} 個策略 | {validated} 個通過驗證 | 最佳 Sharpe {best_sharpe:.2f}
            </div>
        </div>
        <p style="margin: 8px 0 0 0; color: #374151;">{summary}</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """主要 Dashboard"""

    # 套用主題
    apply_theme()
    theme = get_current_theme()

    # 共用樣式（包含隱藏英文導航）
    st.markdown(get_common_css(theme), unsafe_allow_html=True)

    # 渲染中文 sidebar 導航
    render_sidebar_navigation()

    # 標題（右上角含主題切換）
    render_page_header("📊 AI 回測系統 Dashboard")

    # 載入資料
    experiments = load_experiments()
    data = prepare_dashboard_data(experiments)

    if data['total_experiments'] == 0:
        st.warning("尚未記錄任何實驗。請先執行回測並記錄結果。")
        st.info("💡 範例：執行 `examples/learning/record_experiment.py`")
        return

    # [A1] 整體績效摘要
    render_summary_box(data)

    # 核心指標卡片
    st.subheader("核心指標")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        render_metric_card("總實驗數", str(data['total_experiments']))

    with col2:
        render_metric_card(
            "驗證通過數",
            str(data['validated_count']),
            f"{data['validated_count']/data['total_experiments']*100:.1f}%"
        )

    with col3:
        render_metric_card("最佳 Sharpe", f"{data['best_sharpe']:.2f}")

    with col4:
        render_metric_card("平均 Sharpe", f"{data['avg_sharpe']:.2f}")

    with col5:
        render_metric_card("記錄策略數", str(data['unique_strategies']))

    st.markdown("---")

    # 圖表區
    st.subheader("績效分析")

    col1, col2 = st.columns(2)

    with col1:
        render_sharpe_distribution(data['experiments_df'])

    with col2:
        render_grade_distribution(data['grade_counts'])

    st.markdown("---")

    # 時間趨勢
    render_time_trend(data['experiments_df'])

    st.markdown("---")

    # Top 10 排行榜
    st.subheader("🏆 Top 10 排行榜")
    render_top_strategies(data['experiments_df'], n=10)

    st.markdown("---")

    # 策略類型分析
    st.subheader("策略類型分析")
    render_strategy_type_analysis(data['strategy_type_stats'])

    st.markdown("---")

    # 最近活動
    st.subheader("最近活動")
    render_recent_activity(data['experiments_df'], n=10)

    # [A5] 整體建議
    st.markdown("---")
    st.subheader("💡 整體建議")
    render_overall_recommendations(data)

    # Footer
    st.markdown("---")
    st.caption(f"最後更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
