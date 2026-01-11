"""
風險管理儀表板

提供全面的風險管理工具：
1. Kelly Criterion 部位大小計算
2. 策略相關性分析
3. 投資組合優化
4. 風險指標監控（VaR, CVaR, MaxDD）
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import sys

# 加入專案路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 匯入風險管理模組
from src.risk.position_sizing import kelly_criterion, PositionSizeResult, KellyPositionSizer
from src.risk.correlation import CorrelationAnalyzer, CorrelationMatrix, RollingCorrelation
from src.optimizer.portfolio import PortfolioOptimizer, PortfolioWeights
from src.validator.stress_test import StressTestResult

# 頁面配置
st.set_page_config(
    page_title="風險管理儀表板",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自訂樣式（符合 Design Tokens）
st.markdown("""
<style>
/* 卡片容器 */
.stContainer {
    background: var(--color-surface-raised, #f9fafb);
    border-radius: 8px;
    padding: 24px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

/* 指標卡片 */
.stMetric {
    background: var(--color-surface, #ffffff);
    border: 1px solid var(--color-border, #e5e7eb);
    border-radius: 6px;
    padding: 16px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 6px 6px 0 0;
    padding: 8px 16px;
    font-weight: 500;
}

/* 標題分隔線 */
hr {
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Plotly 圖表配色（符合 Design Tokens）
# ============================================================================

CHART_COLORS = {
    'primary': '#2563eb',      # --primitive-blue-600
    'success': '#22c55e',      # --primitive-green-500
    'warning': '#eab308',      # --primitive-yellow-500
    'error': '#ef4444',        # --primitive-red-500
    'secondary': '#6b7280',    # --primitive-gray-500

    # 相關性熱圖漸層
    'heatmap': ['#3b82f6', '#93c5fd', '#f3f4f6', '#fca5a5', '#ef4444'],

    # 效率前緣漸層
    'frontier': ['#dbeafe', '#3b82f6', '#1e40af']
}

PLOTLY_LAYOUT = {
    'font': {'family': 'Inter, sans-serif', 'size': 14},
    'plot_bgcolor': '#ffffff',
    'paper_bgcolor': '#ffffff',
    'margin': {'l': 60, 'r': 40, 't': 60, 'b': 60},
}


# ============================================================================
# 資料載入
# ============================================================================

@st.cache_data(ttl=300)
def load_experiments() -> List[Dict]:
    """載入實驗資料"""
    experiments_file = project_root / "learning" / "experiments.json"

    if not experiments_file.exists():
        return []

    with open(experiments_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get('experiments', [])


def get_available_strategies(experiments: List[Dict]) -> List[str]:
    """取得可用的策略清單"""
    strategies = set()
    for exp in experiments:
        strategy_name = exp['strategy']['name']
        strategies.add(strategy_name)
    return sorted(list(strategies))


def prepare_strategy_returns(experiments: List[Dict], strategy_names: List[str]) -> pd.DataFrame:
    """
    準備策略收益率資料（簡化版本，實際應該從回測結果載入）

    注意：這是示範版本，實際應該從完整回測資料載入每日收益率
    """
    # 過濾選中的策略
    selected_exps = [
        exp for exp in experiments
        if exp['strategy']['name'] in strategy_names
    ]

    if not selected_exps:
        return pd.DataFrame()

    # 使用模擬資料（實際應該從回測結果載入）
    # 這裡基於策略的 Sharpe 和波動率來生成
    np.random.seed(42)
    n_days = 252  # 一年交易日

    returns_dict = {}
    for exp in selected_exps:
        strategy_name = exp['strategy']['name']
        sharpe = exp['results'].get('sharpe_ratio', 1.0)
        annual_return = exp['results'].get('total_return', 0.15)

        # 估計日波動率
        daily_vol = annual_return / (sharpe * np.sqrt(252)) if sharpe > 0 else 0.01
        daily_return = annual_return / 252

        # 生成隨機收益率
        returns = np.random.normal(daily_return, daily_vol, n_days)
        returns_dict[strategy_name] = returns

    df = pd.DataFrame(returns_dict)
    df.index = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

    return df


def calculate_strategy_stats(experiments: List[Dict], strategy_name: str) -> Dict:
    """計算單一策略的統計數據"""
    # 找到該策略的最新實驗
    strategy_exps = [
        exp for exp in experiments
        if exp['strategy']['name'] == strategy_name
    ]

    if not strategy_exps:
        return {}

    # 取最新的實驗
    latest_exp = max(strategy_exps, key=lambda x: x.get('timestamp', ''))
    results = latest_exp['results']

    return {
        'sharpe_ratio': results.get('sharpe_ratio', 0),
        'total_return': results.get('total_return', 0),
        'max_drawdown': results.get('max_drawdown', 0),
        'win_rate': results.get('win_rate', 0),
        'profit_factor': results.get('profit_factor', 0),
        'total_trades': results.get('total_trades', 0),
        'avg_win': results.get('avg_win', 0),
        'avg_loss': results.get('avg_loss', 0)
    }


# ============================================================================
# Tab 1: Kelly Criterion
# ============================================================================

def render_kelly_criterion_tab(experiments: List[Dict], selected_strategies: List[str], account_size: float):
    """渲染 Kelly Criterion 分析"""

    if not selected_strategies:
        st.info("""
        ### 👋 開始分析

        請從上方選擇至少一個策略來查看 Kelly Criterion 部位管理建議。

        **Kelly Criterion** 是一個數學公式，用於計算最大化長期資本成長的最佳賭注大小。
        """)
        return

    st.markdown("### 📊 Kelly Criterion 部位管理")

    # 計算所有策略的 Kelly
    kelly_results = []

    for strategy_name in selected_strategies:
        stats = calculate_strategy_stats(experiments, strategy_name)

        if not stats or stats.get('total_trades', 0) < 10:
            continue

        win_rate = stats['win_rate']
        avg_win = abs(stats['avg_win'])
        avg_loss = abs(stats['avg_loss'])

        if avg_loss == 0 or win_rate == 0:
            continue

        win_loss_ratio = avg_win / avg_loss

        # 計算三種 Kelly
        full_kelly = kelly_criterion(win_rate, win_loss_ratio)
        half_kelly = full_kelly / 2
        quarter_kelly = full_kelly / 4

        kelly_results.append({
            'strategy': strategy_name,
            'win_rate': win_rate,
            'win_loss_ratio': win_loss_ratio,
            'full_kelly': max(0, full_kelly),  # 避免負值
            'half_kelly': max(0, half_kelly),
            'quarter_kelly': max(0, quarter_kelly),
            'full_kelly_size': max(0, full_kelly) * account_size,
            'half_kelly_size': max(0, half_kelly) * account_size,
            'quarter_kelly_size': max(0, quarter_kelly) * account_size
        })

    if not kelly_results:
        st.warning("選中的策略資料不足，無法計算 Kelly Criterion（需要至少 10 筆交易）")
        return

    # 摘要指標卡片
    # 取平均值作為組合建議
    avg_full = np.mean([r['full_kelly'] for r in kelly_results])
    avg_half = np.mean([r['half_kelly'] for r in kelly_results])
    avg_quarter = np.mean([r['quarter_kelly'] for r in kelly_results])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Full Kelly (最激進)",
            value=f"{avg_full*100:.1f}%",
            delta=f"${avg_full * account_size:,.0f}",
            help="最大化成長，但波動較大"
        )

    with col2:
        st.metric(
            label="Half Kelly ⭐ (推薦)",
            value=f"{avg_half*100:.1f}%",
            delta=f"${avg_half * account_size:,.0f}",
            help="平衡成長與波動，適合大多數交易者"
        )

    with col3:
        st.metric(
            label="Quarter Kelly (保守)",
            value=f"{avg_quarter*100:.1f}%",
            delta=f"${avg_quarter * account_size:,.0f}",
            help="極保守，波動最小"
        )

    st.markdown("---")

    # Kelly 曲線圖
    render_kelly_curve(kelly_results[0])  # 使用第一個策略作為範例

    st.markdown("---")

    # Kelly 分配表格
    st.markdown("#### 策略 Kelly 分配表")

    df_kelly = pd.DataFrame(kelly_results)

    display_df = df_kelly[[
        'strategy', 'win_rate', 'win_loss_ratio',
        'full_kelly', 'half_kelly', 'quarter_kelly'
    ]].copy()

    # 格式化顯示（使用 pandas Series 方法）
    for col in ['win_rate', 'full_kelly', 'half_kelly', 'quarter_kelly']:
        display_df[col] = display_df[col].map(lambda x: f"{float(x)*100:.1f}%")
    display_df['win_loss_ratio'] = display_df['win_loss_ratio'].map(lambda x: f"{float(x):.2f}")

    display_df.columns = ['策略', '勝率', '盈虧比', 'Full Kelly', 'Half Kelly', 'Quarter Kelly']

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_kelly_curve(kelly_data: Dict):
    """繪製 Kelly 曲線"""

    # 模擬成長率曲線（簡化版本）
    fractions = np.linspace(0, 1, 100)

    # 使用簡化的成長率模型
    win_rate = kelly_data['win_rate']
    win_loss_ratio = kelly_data['win_loss_ratio']

    # 成長率 = win_rate * log(1 + f * win_loss_ratio) + (1 - win_rate) * log(1 - f)
    growth_rate = []
    ruin_risk = []

    for f in fractions:
        if f >= 1.0:
            g = -100  # 破產
            r = 100
        else:
            # 簡化的成長率計算
            win_term = win_rate * np.log(1 + f * win_loss_ratio) if (1 + f * win_loss_ratio) > 0 else -10
            loss_term = (1 - win_rate) * np.log(1 - f) if (1 - f) > 0 else -10
            g = (win_term + loss_term) * 252 * 100  # 年化成長率 (%)

            # 簡化的破產風險（指數關係）
            r = min(100, (f ** 2) * 100)

        growth_rate.append(g)
        ruin_risk.append(r)

    fig = go.Figure()

    # 成長率曲線
    fig.add_trace(go.Scatter(
        x=fractions * 100,
        y=growth_rate,
        name='預期成長率',
        line=dict(color=CHART_COLORS['primary'], width=3),
        yaxis='y'
    ))

    # 破產風險曲線
    fig.add_trace(go.Scatter(
        x=fractions * 100,
        y=ruin_risk,
        name='破產風險',
        line=dict(color=CHART_COLORS['error'], width=2, dash='dot'),
        yaxis='y2'
    ))

    # 標記 Full Kelly
    fig.add_vline(
        x=kelly_data['full_kelly'] * 100,
        line_dash="dash",
        line_color=CHART_COLORS['success'],
        annotation_text="Full Kelly",
        annotation_position="top"
    )

    # 標記 Half Kelly (推薦)
    fig.add_vline(
        x=kelly_data['half_kelly'] * 100,
        line_dash="solid",
        line_color=CHART_COLORS['warning'],
        line_width=3,
        annotation_text="Half Kelly ⭐",
        annotation_position="top"
    )

    # 標記 Quarter Kelly
    fig.add_vline(
        x=kelly_data['quarter_kelly'] * 100,
        line_dash="dash",
        line_color=CHART_COLORS['primary'],
        annotation_text="Quarter Kelly",
        annotation_position="bottom"
    )

    fig.update_layout(
        title="部位大小 vs 風險收益關係",
        xaxis_title="資金比例 (%)",
        yaxis=dict(title="預期成長率 (%)", side='left'),
        yaxis2=dict(
            title="破產風險 (%)",
            side='right',
            overlaying='y',
            range=[0, 100]
        ),
        height=500,
        hovermode='x unified',
        **PLOTLY_LAYOUT
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Tab 2: 相關性分析
# ============================================================================

def render_correlation_tab(experiments: List[Dict], selected_strategies: List[str]):
    """渲染相關性分析"""

    if len(selected_strategies) < 2:
        st.info("""
        ### 🔗 相關性分析

        請選擇至少 **2 個策略** 來分析相關性。

        **為什麼重要？**
        - 低相關性策略可以降低組合波動
        - 避免策略同時失效
        - 提升風險調整後報酬
        """)
        return

    st.markdown("### 🔗 策略相關性分析")

    # 準備收益率資料
    returns_df = prepare_strategy_returns(experiments, selected_strategies)

    if returns_df.empty:
        st.error("無法載入策略收益率資料")
        return

    # 計算相關性矩陣
    corr_matrix = returns_df.corr()

    # 計算平均相關性（排除對角線）
    mask = np.ones_like(corr_matrix, dtype=bool)
    np.fill_diagonal(mask, False)
    mean_corr = corr_matrix.where(mask).mean().mean()

    # 摘要指標
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="平均相關性",
            value=f"{mean_corr:.3f}",
            help="策略間平均相關係數，越低越好（< 0.3 為佳）"
        )

    with col2:
        max_corr = corr_matrix.where(mask).max().max()
        st.metric(
            label="最大相關性",
            value=f"{max_corr:.3f}",
            delta="⚠️" if max_corr > 0.7 else "✅",
            help="最高的兩兩相關性"
        )

    with col3:
        diversification_ratio = 1 - mean_corr
        st.metric(
            label="分散比率",
            value=f"{diversification_ratio:.3f}",
            delta="✅" if diversification_ratio > 0.7 else "⚠️",
            help="分散效果指標，越高越好"
        )

    st.markdown("---")

    # 佈局：左右兩欄
    col1, col2 = st.columns(2)

    with col1:
        render_correlation_heatmap(corr_matrix)

    with col2:
        render_rolling_correlation(returns_df, window=30)


def render_correlation_heatmap(corr_matrix: pd.DataFrame):
    """相關性矩陣熱圖"""

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale=[
            [0.0, CHART_COLORS['heatmap'][0]],   # 低相關：藍色
            [0.5, CHART_COLORS['heatmap'][2]],   # 中等：灰白
            [1.0, CHART_COLORS['heatmap'][4]]    # 高相關：紅色
        ],
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        colorbar=dict(
            title="相關係數",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=['-1.0', '-0.5', '0.0', '0.5', '1.0']
        )
    ))

    fig.update_layout(
        title="策略相關性矩陣",
        xaxis_title="策略",
        yaxis_title="策略",
        height=500,
        **PLOTLY_LAYOUT
    )

    st.plotly_chart(fig, use_container_width=True)


def render_rolling_correlation(returns_df: pd.DataFrame, window: int = 30):
    """滾動相關性時間序列"""

    if len(returns_df.columns) < 2:
        return

    # 計算所有策略對的滾動相關性
    fig = go.Figure()

    strategies = list(returns_df.columns)

    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            s1, s2 = strategies[i], strategies[j]

            # 滾動相關性
            rolling_corr = returns_df[s1].rolling(window).corr(returns_df[s2])

            fig.add_trace(go.Scatter(
                x=rolling_corr.index,
                y=rolling_corr,
                name=f'{s1} vs {s2}',
                mode='lines',
                line=dict(width=2)
            ))

    # 添加參考線
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="無相關",
        annotation_position="right"
    )
    fig.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="orange",
        annotation_text="中度相關",
        annotation_position="right"
    )

    fig.update_layout(
        title=f"滾動相關性（窗口：{window} 天）",
        xaxis_title="日期",
        yaxis_title="相關係數",
        yaxis_range=[-1, 1],
        height=400,
        hovermode='x unified',
        **PLOTLY_LAYOUT
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Tab 3: 組合優化
# ============================================================================

def render_portfolio_optimization_tab(experiments: List[Dict], selected_strategies: List[str]):
    """渲染組合優化分析"""

    if len(selected_strategies) < 2:
        st.info("""
        ### ⚖️ 投資組合優化

        請選擇至少 **2 個策略** 來進行組合優化。

        **優化目標**：
        - 最大化 Sharpe Ratio
        - 風險平價配置
        - 最小波動率
        """)
        return

    st.markdown("### ⚖️ 投資組合優化")

    # 準備收益率資料
    returns_df = prepare_strategy_returns(experiments, selected_strategies)

    if returns_df.empty:
        st.error("無法載入策略收益率資料")
        return

    # 優化方法選擇
    optimization_method = st.radio(
        "優化方法",
        ["最大 Sharpe Ratio", "風險平價", "最小波動"],
        horizontal=True,
        help="選擇組合優化的目標函數"
    )

    st.markdown("---")

    # 執行優化
    try:
        optimizer = PortfolioOptimizer(returns_df)

        if optimization_method == "最大 Sharpe Ratio":
            weights = optimizer.max_sharpe_optimize()
        elif optimization_method == "風險平價":
            weights = optimizer.risk_parity_optimize()
        else:  # 最小波動
            weights = optimizer.mean_variance_optimize(target_risk=0.0)

        # 計算組合指標
        weights_arr = np.array(list(weights.weights.values()))
        portfolio_return = (returns_df.mean() * weights_arr).sum() * 252
        portfolio_vol = np.sqrt(
            np.dot(weights_arr,
                   np.dot(returns_df.cov() * 252, weights_arr))
        )
        portfolio_sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0

        # 績效摘要卡片
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="年化報酬",
                value=f"{portfolio_return*100:.1f}%",
                help="組合預期年化報酬率"
            )

        with col2:
            st.metric(
                label="年化波動",
                value=f"{portfolio_vol*100:.1f}%",
                help="組合預期年化波動率"
            )

        with col3:
            st.metric(
                label="Sharpe Ratio",
                value=f"{portfolio_sharpe:.2f}",
                delta="✅" if portfolio_sharpe > 1.5 else "⚠️",
                help="風險調整後報酬"
            )

        st.markdown("---")

        # 圖表區
        col1, col2 = st.columns(2)

        with col1:
            render_efficient_frontier(optimizer, weights)

        with col2:
            render_weight_allocation(weights.weights)

        st.markdown("---")

        # 權重分配表格
        render_weight_table(optimizer, returns_df)

    except Exception as e:
        st.error(f"組合優化失敗: {str(e)}")


def render_efficient_frontier(optimizer: PortfolioOptimizer, optimal_weights: PortfolioWeights):
    """效率前緣圖"""

    # 生成效率前緣
    n_points = 50
    target_returns = np.linspace(0.05, 0.50, n_points)

    frontier_vols = []
    frontier_rets = []

    for target_return in target_returns:
        try:
            result = optimizer.efficient_return(target_return)
            frontier_vols.append(result.expected_volatility)
            frontier_rets.append(result.expected_return)
        except:
            continue

    fig = go.Figure()

    # 效率前緣曲線
    if frontier_vols:
        fig.add_trace(go.Scatter(
            x=np.array(frontier_vols) * 100,
            y=np.array(frontier_rets) * 100,
            mode='lines',
            name='效率前緣',
            line=dict(color=CHART_COLORS['primary'], width=3),
            fill='tonexty',
            fillcolor='rgba(59, 130, 246, 0.1)'
        ))

    # 最優點
    fig.add_trace(go.Scatter(
        x=[optimal_weights.expected_volatility * 100],
        y=[optimal_weights.expected_return * 100],
        mode='markers',
        name='最優組合',
        marker=dict(
            size=15,
            color=CHART_COLORS['success'],
            symbol='star',
            line=dict(color='white', width=2)
        )
    ))

    fig.update_layout(
        title="效率前緣與最優組合",
        xaxis_title="年化波動率 (%)",
        yaxis_title="年化報酬率 (%)",
        height=500,
        hovermode='closest',
        **PLOTLY_LAYOUT
    )

    st.plotly_chart(fig, use_container_width=True)


def render_weight_allocation(weights: Dict[str, float]):
    """權重分配圓餅圖"""

    labels = list(weights.keys())
    values = [w * 100 for w in weights.values()]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(
            colors=[CHART_COLORS['primary'], CHART_COLORS['success'],
                   CHART_COLORS['warning'], CHART_COLORS['error'],
                   CHART_COLORS['secondary']][:len(labels)]
        ),
        textinfo='label+percent',
        textposition='auto',
        textfont_size=14
    )])

    fig.update_layout(
        title="權重配置",
        height=500,
        showlegend=True,
        **PLOTLY_LAYOUT
    )

    st.plotly_chart(fig, use_container_width=True)


def render_weight_table(optimizer: PortfolioOptimizer, returns_df: pd.DataFrame):
    """權重分配表格（比較三種方法）"""

    st.markdown("#### 權重分配比較表")

    try:
        max_sharpe = optimizer.max_sharpe_optimize()
        risk_parity = optimizer.risk_parity_optimize()
        min_vol = optimizer.mean_variance_optimize(target_risk=0.0)

        strategies = list(returns_df.columns)

        table_data = []
        for strategy in strategies:
            table_data.append({
                '策略': strategy,
                '最大 Sharpe': f"{max_sharpe.weights.get(strategy, 0)*100:.1f}%",
                '風險平價': f"{risk_parity.weights.get(strategy, 0)*100:.1f}%",
                '最小波動': f"{min_vol.weights.get(strategy, 0)*100:.1f}%"
            })

        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True, hide_index=True)

    except Exception as e:
        st.warning(f"無法生成完整比較表: {str(e)}")


# ============================================================================
# Tab 4: 風險指標
# ============================================================================

def render_risk_metrics_tab(experiments: List[Dict], selected_strategies: List[str]):
    """渲染風險指標監控"""

    if not selected_strategies:
        st.info("""
        ### 📉 風險指標監控

        請選擇至少一個策略來查看風險指標。

        **監控指標**：
        - VaR (Value at Risk): 可能損失
        - CVaR (Conditional VaR): 尾部風險
        - 最大回撤: 歷史最大虧損
        - 恢復時間: 回撤恢復天數
        """)
        return

    st.markdown("### 📉 風險指標監控")

    # 準備收益率資料
    returns_df = prepare_strategy_returns(experiments, selected_strategies)

    if returns_df.empty:
        st.error("無法載入策略收益率資料")
        return

    # 計算組合收益率（等權重）
    portfolio_returns = returns_df.mean(axis=1)

    # 計算風險指標
    var_95 = portfolio_returns.quantile(0.05)
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

    # 計算回撤
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    # 恢復時間（簡化計算）
    recovery_days = 0
    if max_dd < 0:
        dd_series = drawdown[drawdown == max_dd]
        if len(dd_series) > 0:
            max_dd_date = dd_series.index[0]
            recovery_series = drawdown[drawdown.index > max_dd_date]
            recovery_dates = recovery_series[recovery_series >= -0.01]
            if len(recovery_dates) > 0:
                recovery_days = (recovery_dates.index[0] - max_dd_date).days

    # 風險指標卡片
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="VaR (95%)",
            value=f"{var_95*100:.2f}%",
            delta="⚠️" if var_95 < -0.05 else "✅",
            help="95% 信心水準下的單日最大損失"
        )

    with col2:
        st.metric(
            label="CVaR (95%)",
            value=f"{cvar_95*100:.2f}%",
            delta="⚠️" if cvar_95 < -0.10 else "✅",
            help="超過 VaR 時的平均損失（尾部風險）"
        )

    with col3:
        st.metric(
            label="最大回撤",
            value=f"{max_dd*100:.2f}%",
            delta="❌" if max_dd < -0.20 else "⚠️" if max_dd < -0.10 else "✅",
            help="歷史最大虧損幅度"
        )

    with col4:
        st.metric(
            label="恢復時間",
            value=f"{recovery_days} 天" if recovery_days > 0 else "N/A",
            delta="⚠️" if recovery_days > 60 else "✅",
            help="從最大回撤恢復到前高的天數"
        )

    st.markdown("---")

    # 回撤曲線
    render_drawdown_curve(cumulative, drawdown)

    st.markdown("---")

    # VaR 分布
    render_var_distribution(portfolio_returns, var_95, cvar_95)


def render_drawdown_curve(equity_curve: pd.Series, drawdown: pd.Series):
    """回撤曲線圖"""

    fig = go.Figure()

    # 回撤曲線
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown * 100,
        mode='lines',
        name='回撤',
        line=dict(color=CHART_COLORS['error'], width=2),
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.2)'
    ))

    # 標記最大回撤
    max_dd_idx = drawdown.idxmin()
    max_dd_value = drawdown.min()

    fig.add_trace(go.Scatter(
        x=[max_dd_idx],
        y=[max_dd_value * 100],
        mode='markers+text',
        name='最大回撤',
        marker=dict(size=12, color='#dc2626'),
        text=[f'{max_dd_value*100:.1f}%'],
        textposition='bottom center'
    ))

    fig.update_layout(
        title="歷史回撤曲線",
        xaxis_title="日期",
        yaxis_title="回撤 (%)",
        yaxis_range=[min(drawdown*100)*1.2, 5],
        height=400,
        hovermode='x unified',
        **PLOTLY_LAYOUT
    )

    st.plotly_chart(fig, use_container_width=True)


def render_var_distribution(returns: pd.Series, var_95: float, cvar_95: float):
    """VaR / CVaR 分布圖"""

    fig = go.Figure()

    # 收益率直方圖
    fig.add_trace(go.Histogram(
        x=returns * 100,
        nbinsx=50,
        name='收益分布',
        marker_color=CHART_COLORS['primary'],
        opacity=0.7
    ))

    # VaR 95% 線
    fig.add_vline(
        x=var_95 * 100,
        line_dash="dash",
        line_color=CHART_COLORS['warning'],
        line_width=2,
        annotation_text=f"VaR 95%: {var_95*100:.2f}%",
        annotation_position="top left"
    )

    # CVaR 95% 線
    fig.add_vline(
        x=cvar_95 * 100,
        line_dash="solid",
        line_color=CHART_COLORS['error'],
        line_width=2,
        annotation_text=f"CVaR 95%: {cvar_95*100:.2f}%",
        annotation_position="bottom left"
    )

    # 填充尾部區域
    fig.add_vrect(
        x0=returns.min() * 100,
        x1=var_95 * 100,
        fillcolor="#fef2f2",
        opacity=0.3,
        layer="below",
        line_width=0
    )

    fig.update_layout(
        title="收益分布與風險值",
        xaxis_title="收益率 (%)",
        yaxis_title="頻率",
        height=400,
        showlegend=False,
        **PLOTLY_LAYOUT
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# 主要 UI
# ============================================================================

def main():
    """主要 Dashboard"""

    # 標題
    st.title("🛡️ 風險管理儀表板")
    st.markdown("---")

    # 載入資料
    experiments = load_experiments()

    if not experiments:
        st.warning("""
        尚未記錄任何實驗。請先執行回測並記錄結果。

        💡 提示：執行 `examples/learning/record_experiment.py`
        """)
        return

    available_strategies = get_available_strategies(experiments)

    # 控制區
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        selected_strategies = st.multiselect(
            "選擇策略",
            options=available_strategies,
            default=available_strategies[:2] if len(available_strategies) >= 2 else available_strategies,
            help="選擇 2-5 個策略以進行風險分析"
        )

    with col2:
        account_size = st.number_input(
            "帳戶規模 (USD)",
            min_value=1000.0,
            max_value=10000000.0,
            value=100000.0,
            step=10000.0,
            help="用於計算部位大小"
        )

    with col3:
        if st.button("🔄 重新計算", use_container_width=True):
            st.rerun()

    st.markdown("---")

    # Tabs 內容區
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Kelly Criterion",
        "🔗 相關性分析",
        "⚖️ 組合優化",
        "📉 風險指標"
    ])

    with tab1:
        render_kelly_criterion_tab(experiments, selected_strategies, account_size)

    with tab2:
        render_correlation_tab(experiments, selected_strategies)

    with tab3:
        render_portfolio_optimization_tab(experiments, selected_strategies)

    with tab4:
        render_risk_metrics_tab(experiments, selected_strategies)

    # Footer
    st.markdown("---")
    st.caption(f"最後更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
