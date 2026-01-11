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
from ui.utils import render_sidebar_navigation, render_page_header
from ui.utils.data_loader import load_equity_curve, load_daily_returns, calculate_monthly_returns
from ui.theme_switcher import apply_theme, get_current_theme
from ui.chart_config import get_chart_layout, get_chart_colors


# ===== 設定頁面 =====
st.set_page_config(
    page_title="策略列表 - 合約交易系統",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== 套用主題 =====
apply_theme()
theme = get_current_theme()

# ===== 自訂樣式 =====
st.markdown(get_common_css(theme), unsafe_allow_html=True)


# ===== 資料載入函數 =====

# DataFrame 欄位定義（避免重複）
STRATEGY_COLUMNS = [
    'experiment_id', 'strategy_name', 'strategy_type', 'symbol', 'timeframe',
    'total_return', 'annual_return', 'sharpe_ratio', 'max_drawdown',
    'total_trades', 'win_rate', 'grade', 'wfa_efficiency', 'params', 'created_at'
]


def calculate_grade(sharpe: float, max_dd: float, win_rate: float) -> str:
    """根據績效指標計算驗證等級"""
    if sharpe >= 2.0 and max_dd <= 15 and win_rate >= 60:
        return 'A'
    elif sharpe >= 1.5 and max_dd <= 20 and win_rate >= 55:
        return 'B'
    elif sharpe >= 1.0 and max_dd <= 25 and win_rate >= 50:
        return 'C'
    elif sharpe >= 0.5 and max_dd <= 30:
        return 'D'
    else:
        return 'F'


@st.cache_data
def load_strategy_results() -> pd.DataFrame:
    """
    載入所有策略驗證結果

    Returns:
        pd.DataFrame: 策略實驗結果，包含以下欄位：
            - experiment_id: 實驗 ID
            - strategy_name: 策略名稱
            - strategy_type: 策略類型
            - symbol: 交易標的
            - timeframe: 時間框架
            - total_return: 總報酬率 (%)
            - annual_return: 年化報酬率 (%)
            - sharpe_ratio: Sharpe Ratio
            - max_drawdown: 最大回撤 (%)
            - total_trades: 總交易筆數
            - win_rate: 勝率 (%)
            - grade: 驗證等級
            - wfa_efficiency: WFA 效率（從 validation 中提取，若無則為 0）
            - params: 策略參數 (dict)
            - created_at: 建立時間
    """
    from ui.utils.data_loader import get_all_experiments

    try:
        # 載入所有實驗
        experiments = get_all_experiments()

        # 處理空數據情況
        if not experiments:
            return pd.DataFrame(columns=STRATEGY_COLUMNS)

        # 轉換為 DataFrame 格式
        data = []
        for exp in experiments:
            try:
                # 提取策略資訊
                strategy = exp.strategy
                results = exp.results
                config = exp.config
                
                # 驗證必要欄位
                if not config.get('symbol') or not config.get('timeframe'):
                    st.warning(f"⚠️ 實驗 {exp.id} 缺少必要欄位（symbol 或 timeframe），已跳過")
                    continue

                # 提取數值（百分比轉換）
                total_return = results.get('total_return', 0.0) * 100
                annual_return = results.get('annual_return', 0.0) * 100
                sharpe_ratio = results.get('sharpe_ratio', 0.0)
                max_drawdown = abs(results.get('max_drawdown', 0.0)) * 100
                win_rate = results.get('win_rate', 0.0) * 100
                
                # 計算等級
                grade = calculate_grade(sharpe_ratio, max_drawdown, win_rate)
                
                # 提取 WFA 效率（如果有驗證結果）
                wfa_efficiency = 0.0
                if hasattr(exp, 'validation') and exp.validation:
                    wfa_efficiency = exp.validation.get('wfa_efficiency', 0.0)

                # 構建資料行
                row = {
                    'experiment_id': exp.id,
                    'strategy_name': strategy.get('name', 'Unknown'),
                    'strategy_type': strategy.get('type', '未分類'),
                    'symbol': config.get('symbol'),
                    'timeframe': config.get('timeframe'),
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'total_trades': results.get('total_trades', 0),
                    'win_rate': win_rate,
                    'grade': grade,
                    'wfa_efficiency': wfa_efficiency,
                    'params': exp.parameters if hasattr(exp, 'parameters') else {},
                    'created_at': exp.timestamp
                }
                data.append(row)

            except (AttributeError, KeyError) as e:
                # 數據格式錯誤
                exp_id = exp.id if hasattr(exp, 'id') else 'Unknown'
                st.warning(f"⚠️ 實驗 {exp_id} 數據格式錯誤：{str(e)}")
                continue
            except Exception as e:
                # 其他未預期錯誤
                exp_id = exp.id if hasattr(exp, 'id') else 'Unknown'
                st.warning(f"⚠️ 載入實驗 {exp_id} 時發生錯誤：{str(e)}")
                continue

        # 建立 DataFrame
        df = pd.DataFrame(data)

        # 如果所有實驗都轉換失敗，返回空 DataFrame
        if df.empty:
            return pd.DataFrame(columns=STRATEGY_COLUMNS)

        return df

    except Exception as e:
        # 處理載入失敗情況
        st.error(f"❌ 載入策略結果失敗：{str(e)}")
        st.info("請確認 experiments.json 檔案存在且格式正確。")

        # 返回空 DataFrame
        return pd.DataFrame(columns=STRATEGY_COLUMNS)



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


def plot_equity_curve(strategy_name: str, experiment_id: str) -> go.Figure:
    """
    繪製權益曲線

    Args:
        strategy_name: 策略名稱（用於標題）
        experiment_id: 實驗 ID（用於載入數據）

    Returns:
        Plotly Figure 物件，如果數據不存在則返回空白圖表並顯示提示
    """
    # 載入權益曲線數據
    equity_curve = load_equity_curve(experiment_id)

    # 處理數據缺失
    if equity_curve is None or len(equity_curve) == 0:
        st.info("""
📊 **權益曲線數據缺失**

此策略實驗未儲存詳細權益曲線資料。

**可能原因**：
- 實驗記錄於舊版本系統
- 回測未正常完成

**建議**：
- 重新執行回測
- 檢查實驗記錄完整性
        """)
        # 返回空白圖表
        fig = go.Figure()
        fig.update_layout(
            title=f'{strategy_name} - 權益曲線',
            xaxis_title='日期',
            yaxis_title='權益 ($)',
            height=400,
            margin=dict(l=60, r=40, t=60, b=60),
        )
        return fig

    # 數據驗證
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        st.error("❌ 權益曲線索引必須為日期格式")
        fig = go.Figure()
        return fig

    # 處理缺失值
    if equity_curve.isnull().any():
        st.warning("⚠️ 權益曲線包含缺失值，已自動填充")
        equity_curve = equity_curve.ffill()

    # 套用時間範圍篩選
    if 'chart_xrange' in st.session_state and st.session_state.chart_xrange:
        start_date, end_date = st.session_state.chart_xrange

        # 驗證時間範圍合理性
        if start_date > end_date:
            st.error("❌ 時間範圍錯誤：起始日期晚於結束日期")
            # 返回空白圖表
            fig = go.Figure()
            fig.update_layout(
                title=f'{strategy_name} - 權益曲線',
                xaxis_title='日期',
                yaxis_title='權益 ($)',
                height=400,
                margin=dict(l=60, r=40, t=60, b=60),
            )
            return fig

        equity_curve = equity_curve.loc[
            (equity_curve.index.date >= start_date) &
            (equity_curve.index.date <= end_date)
        ]

        # 檢查篩選後是否還有數據
        if len(equity_curve) == 0:
            st.warning("⚠️ 選擇的時間範圍內無數據，請調整時間範圍")
            # 返回空白圖表
            fig = go.Figure()
            fig.update_layout(
                title=f'{strategy_name} - 權益曲線',
                xaxis_title='日期',
                yaxis_title='權益 ($)',
                height=400,
                margin=dict(l=60, r=40, t=60, b=60),
            )
            return fig

    # 建立圖表
    fig = go.Figure()

    # 主線
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve.values,
        mode='lines',
        name='權益',
        line=dict(
            color='#2563eb',  # --color-primary from styles.py
            width=2
        ),
        hovertemplate='<b>日期</b>: %{x|%Y-%m-%d}<br>' +
                      '<b>權益</b>: $%{y:,.2f}<br>' +
                      '<extra></extra>'
    ))

    # 可選：填充區域
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve.values,
        fill='tozeroy',
        fillcolor='rgba(37, 99, 235, 0.1)',  # --color-primary 10% opacity
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    # 佈局配置
    fig.update_layout(**get_chart_layout(
        theme=theme,
        title=f'{strategy_name} - 權益曲線',
        xaxis_title='日期',
        yaxis_title='權益 ($)',
        height=400,
        hovermode='x unified'
    ))

    # 貨幣格式
    fig.update_yaxes(tickformat='$,.0f')

    return fig


def plot_monthly_heatmap(
    strategy_name: str,
    experiment_id: str,
    start_date: pd.Timestamp = None,
    end_date: pd.Timestamp = None
) -> go.Figure:
    """
    繪製月度報酬熱力圖

    Args:
        strategy_name: 策略名稱（用於標題）
        experiment_id: 實驗 ID（用於載入數據）
        start_date: 起始日期（可選，用於篩選）
        end_date: 結束日期（可選，用於篩選）

    Returns:
        Plotly Figure 物件，如果數據不存在則返回空白圖表並顯示提示
    """
    # 載入日報酬數據
    daily_returns = load_daily_returns(experiment_id)

    # 處理數據缺失
    if daily_returns is None or len(daily_returns) == 0:
        st.info("""
📊 **月度報酬數據缺失**

此策略實驗未儲存詳細報酬資料。

**可能原因**：
- 實驗記錄於舊版本系統
- 回測未正常完成

**建議**：
- 重新執行回測
- 檢查實驗記錄完整性
        """)
        # 返回空白圖表
        fig = go.Figure()
        fig.update_layout(
            title=f'{strategy_name} - 月度報酬',
            height=200,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        return fig

    # 套用時間範圍篩選
    if start_date and end_date:
        # 驗證時間範圍合理性
        if start_date > end_date:
            st.error("❌ 時間範圍錯誤：起始日期晚於結束日期")
            fig = go.Figure()
            fig.update_layout(
                title=f'{strategy_name} - 月度報酬',
                height=200,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            return fig

        daily_returns = daily_returns.loc[
            (daily_returns.index.date >= start_date) &
            (daily_returns.index.date <= end_date)
        ]

    # 檢查篩選後是否還有數據
    if len(daily_returns) == 0:
        st.warning("⚠️ 時間範圍內無數據")
        fig = go.Figure()
        fig.update_layout(
            title=f'{strategy_name} - 月度報酬',
            height=200,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        return fig

    # 計算月度報酬
    monthly_data = calculate_monthly_returns(daily_returns)

    # 檢查月度數據是否有效
    if len(monthly_data) == 0:
        st.warning("⚠️ 無法計算月度報酬")
        fig = go.Figure()
        fig.update_layout(
            title=f'{strategy_name} - 月度報酬',
            height=200,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        return fig

    # 檢查是否全為 NaN
    if 'return' in monthly_data.columns and monthly_data['return'].isna().all():
        st.warning("⚠️ 無法計算月度報酬（數據異常）")
        fig = go.Figure()
        fig.update_layout(
            title=f'{strategy_name} - 月度報酬',
            height=200,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        return fig

    # 準備熱力圖數據
    # 取得所有年份（由舊到新排列）
    years = sorted(monthly_data['year'].unique())
    months_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # 建立矩陣：rows = years, cols = months (1-12)
    z_matrix = []
    text_matrix = []

    for year in years:
        year_data = monthly_data[monthly_data['year'] == year]
        row_values = []
        row_text = []

        for month in range(1, 13):
            month_return = year_data[year_data['month'] == month]['return'].values
            if len(month_return) > 0:
                ret = month_return[0]
                row_values.append(ret)
                row_text.append(f'{ret:.1f}%')
            else:
                # 空缺月份（未來月份或資料不足）
                row_values.append(None)
                row_text.append('')

        z_matrix.append(row_values)
        text_matrix.append(row_text)

    # 自定義色盲友好配色（藍-橙漸層）
    # 根據設計規格：負報酬藍色系，正報酬綠色系
    colorscale = [
        [0.0, '#1d4ed8'],   # 深藍（-10% 以下）
        [0.35, '#60a5fa'],  # 中藍（-5%）
        [0.45, '#dbeafe'],  # 淺藍（-1%）
        [0.5, '#f3f4f6'],   # 中性灰（0%）
        [0.55, '#d1fae5'],  # 淺綠（+1%）
        [0.65, '#22c55e'],  # 中綠（+5%）
        [1.0, '#15803d']    # 深綠（+10% 以上）
    ]

    # 建立熱力圖
    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=months_abbr,
        y=[str(y) for y in years],
        colorscale=colorscale,
        text=text_matrix,
        texttemplate='%{text}',
        textfont=dict(size=10),
        colorbar=dict(
            title='報酬率 (%)',
            titleside='right',
            ticksuffix='%',
            thickness=15,
            len=0.7
        ),
        hovertemplate='<b>%{y}年 %{x}</b><br>月報酬: %{z:.2f}%<extra></extra>',
        zmid=0,  # 中點設為 0（中性色）
        zmin=-10,  # 最小值 -10%
        zmax=10    # 最大值 +10%
    ))

    # 佈局配置
    fig.update_layout(**get_chart_layout(
        theme=theme,
        title=f'{strategy_name} - 月度報酬',
        xaxis_title='月份',
        yaxis_title='年份',
        height=200
    ))

    # Y 軸反轉（最新年份在上方）
    fig.update_yaxes(autorange='reversed')

    return fig


# ===== 主程式 =====

def render_filter_summary(df_all: pd.DataFrame, df_filtered: pd.DataFrame, filters: dict):
    """[B1] 渲染篩選結果摘要"""
    total = len(df_all)
    filtered = len(df_filtered)
    filter_rate = filtered / total * 100 if total > 0 else 0

    # 計算篩選後的統計
    if filtered > 0:
        avg_sharpe = df_filtered['sharpe_ratio'].mean()
        a_count = len(df_filtered[df_filtered['grade'] == 'A'])
        b_count = len(df_filtered[df_filtered['grade'] == 'B'])
        good_rate = (a_count + b_count) / filtered * 100
    else:
        avg_sharpe = 0
        good_rate = 0

    # 判斷篩選結果品質
    if filter_rate < 10:
        status = "⚠️ 篩選條件過嚴"
        border_color = "var(--warning)"
    elif good_rate >= 50:
        status = "✅ 篩選結果優質"
        border_color = "var(--success)"
    else:
        status = "📊 篩選結果一般"
        border_color = "var(--info)"

    st.markdown(f"""
    <div style="background: var(--surface-raised);
                border-left: 4px solid {border_color};
                border: 1px solid var(--border);
                padding: 12px 16px;
                border-radius: var(--radius-lg);
                margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: 600; color: var(--text-primary);">[B1] 篩選結果摘要 {status}</span>
            <span style="color: var(--text-secondary); font-size: 0.9em;">
                符合 {filtered}/{total} 筆 ({filter_rate:.0f}%) | 平均 Sharpe {avg_sharpe:.2f} | A+B 級 {good_rate:.0f}%
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_column_explanation():
    """[B2] 渲染欄位說明"""
    with st.expander("📖 [B2] 欄位說明（點擊展開）"):
        st.markdown("""
        | 欄位 | 說明 | 判讀標準 |
        |------|------|----------|
        | **策略名稱** | 策略識別名稱 | - |
        | **報酬率** | 回測期間總報酬 | >30% 優秀，>10% 及格 |
        | **年化報酬** | 年化換算報酬率 | >20% 優秀，>10% 及格 |
        | **Sharpe** | 風險調整後收益 | >2.0 卓越，>1.5 優良，>1.0 及格 |
        | **MaxDD** | 最大回撤幅度 | <15% 優秀，<25% 及格，>30% 危險 |
        | **交易筆數** | 總交易次數 | >50 較可靠，<20 樣本不足 |
        | **勝率** | 獲利交易比例 | >55% 優秀，>50% 及格 |
        | **等級** | 綜合評分 | A/B 可實盤，C 需優化，D/F 不建議 |
        | **過擬合率** | WFA 效率指標 | >0.8 可靠，<0.6 可能過擬合 |
        """)


def render_quick_recommendations(df: pd.DataFrame):
    """[B4] 渲染頁尾快速建議"""
    if df.empty:
        return

    st.markdown("---")
    st.subheader("💡 [B4] 快速建議")

    recommendations = []

    # 分析當前篩選結果
    avg_sharpe = df['sharpe_ratio'].mean()
    avg_dd = df['max_drawdown'].mean()
    a_strategies = df[df['grade'] == 'A']['strategy_name'].tolist()

    if a_strategies:
        recommendations.append(f"✅ **推薦策略**：{', '.join(a_strategies[:3])} 表現優異，可優先考慮")

    if avg_sharpe < 1.0:
        recommendations.append("⚠️ **平均 Sharpe 偏低**：考慮放寬篩選條件或優化現有策略")

    if avg_dd > 25:
        recommendations.append("⚠️ **回撤風險較高**：建議降低最大回撤篩選門檻，或加強止損機制")

    best_type = df.groupby('strategy_type')['sharpe_ratio'].mean().idxmax() if len(df) > 0 else None
    if best_type:
        recommendations.append(f"📊 **最佳策略類型**：{best_type} 類型平均表現最好")

    for rec in recommendations:
        st.markdown(rec)


def main():
    # 渲染中文 sidebar 導航
    render_sidebar_navigation()

    # 標題（右上角含主題切換）
    render_page_header("📊 策略列表", "篩選和查看所有策略實驗結果")

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

    # [B1] 篩選結果摘要
    render_filter_summary(df_all, df_filtered, filters)

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

    # [B2] 欄位說明
    render_column_explanation()

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
                plot_equity_curve(selected_strategy, strategy_data['experiment_id']),
                use_container_width=True
            )

            # 月度報酬熱力圖
            # 獲取時間範圍（與權益曲線同步）
            start_date = None
            end_date = None
            if 'chart_xrange' in st.session_state and st.session_state.chart_xrange:
                start_date, end_date = st.session_state.chart_xrange

            st.plotly_chart(
                plot_monthly_heatmap(
                    selected_strategy,
                    strategy_data['experiment_id'],
                    start_date,
                    end_date
                ),
                use_container_width=True
            )

            # [B3] AI 洞察（增強版）
            st.markdown("**🤖 [B3] AI 洞察**")

            # 綜合評估
            sharpe = strategy_data['sharpe_ratio']
            dd = strategy_data['max_drawdown']
            wfa = strategy_data['wfa_efficiency']
            win_rate = strategy_data['win_rate']

            insights = []

            # Sharpe 評估
            if sharpe >= 2.0:
                insights.append(f"✅ **Sharpe 卓越** ({sharpe:.2f})：風險調整收益優異，可考慮實盤")
            elif sharpe >= 1.5:
                insights.append(f"✅ **Sharpe 良好** ({sharpe:.2f})：表現穩定，建議進一步驗證")
            elif sharpe >= 1.0:
                insights.append(f"📊 **Sharpe 及格** ({sharpe:.2f})：有改善空間，可優化入場時機")
            else:
                insights.append(f"⚠️ **Sharpe 偏低** ({sharpe:.2f})：需重新檢視策略邏輯")

            # 回撤評估
            if dd <= 15:
                insights.append(f"✅ **回撤控制優秀** ({dd:.1f}%)：風險管理得當")
            elif dd <= 25:
                insights.append(f"📊 **回撤可接受** ({dd:.1f}%)：建議設定止損保護")
            else:
                insights.append(f"⚠️ **回撤風險高** ({dd:.1f}%)：強烈建議降低槓桿或加強止損")

            # 過擬合評估
            if wfa >= 0.8:
                insights.append(f"✅ **樣本外表現穩定** (WFA {wfa:.2f})：過擬合風險低")
            elif wfa >= 0.6:
                insights.append(f"📊 **樣本外表現普通** (WFA {wfa:.2f})：可能存在輕微過擬合")
            else:
                insights.append(f"⚠️ **過擬合風險** (WFA {wfa:.2f})：建議簡化策略或增加訓練數據")

            # 勝率評估
            if win_rate >= 60:
                insights.append(f"✅ **勝率優秀** ({win_rate:.1f}%)：入場時機把握準確")
            elif win_rate >= 50:
                insights.append(f"📊 **勝率普通** ({win_rate:.1f}%)：可優化出場邏輯提升盈虧比")

            for insight in insights:
                st.markdown(insight)

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

    # [B4] 頁尾快速建議
    render_quick_recommendations(df_sorted)


if __name__ == "__main__":
    main()
