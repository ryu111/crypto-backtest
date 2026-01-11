"""AI 合約回測系統 - Streamlit UI 主入口

提供視覺化介面查看回測結果、策略表現和評級系統。
"""

import streamlit as st
from pathlib import Path
import sys

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.utils import (
    load_experiments,
    calculate_summary_stats,
    get_latest_experiments,
    format_percentage,
    format_sharpe,
    grade_color,
    format_timestamp,
    get_data_source_status,
    render_page_header,
)
from ui.styles import get_common_css, GRADE_COLORS


# 頁面配置
st.set_page_config(
    page_title="AI 合約回測系統",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# 自定義 CSS (包含共用樣式和頁面專用樣式)
common_css = get_common_css()
page_specific_css = """
<style>
    /* 隱藏 Streamlit 自動產生的頁面導航（英文）*/
    [data-testid="stSidebarNav"] {
        display: none !important;
    }

    /* 主要容器 */
    .main > div {
        padding-top: 2rem;
    }

    /* 統計卡片 */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }

    .stat-label {
        font-size: 0.875rem;
        opacity: 0.9;
    }

    /* 導航卡片 */
    .nav-card {
        padding: 1.5rem;
        border: 2px solid var(--color-border);
        border-radius: var(--radius-lg);
        cursor: pointer;
        transition: all 0.2s;
    }

    .nav-card:hover {
        border-color: var(--color-primary);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }

    /* 狀態指示器 */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }

    .status-online {
        background-color: var(--color-success);
    }

    .status-offline {
        background-color: var(--color-error);
    }
</style>
"""

st.markdown(common_css + page_specific_css, unsafe_allow_html=True)


def render_sidebar():
    """渲染側邊欄"""
    with st.sidebar:
        st.title("📊 AI 合約回測")
        st.markdown("---")

        # 頁面導航
        st.subheader("🧭 導航")
        st.page_link("app.py", label="首頁", icon="🏠")
        st.page_link("pages/1_📊_Dashboard.py", label="數據儀表板", icon="📈")
        st.page_link("pages/2_Strategies.py", label="策略列表", icon="📋")
        st.page_link("pages/3_Comparison.py", label="策略比較", icon="⚖️")
        st.page_link("pages/4_Validation.py", label="策略驗證", icon="🔬")
        st.page_link("pages/5_RiskDashboard.py", label="風險管理", icon="🛡️")

        st.markdown("---")

        # 資料來源狀態
        st.subheader("💾 資料狀態")
        status = get_data_source_status()

        if status["available"]:
            st.markdown(
                f'<span class="status-indicator status-online"></span>資料可用',
                unsafe_allow_html=True
            )
            st.caption(f"實驗數: {status['experiment_count']}")
            st.caption(f"更新: {status['last_updated']}")
        else:
            st.markdown(
                f'<span class="status-indicator status-offline"></span>資料不可用',
                unsafe_allow_html=True
            )
            if "error" in status:
                st.error(f"錯誤: {status['error']}")

        st.markdown("---")

        # AI Loop 狀態（未來功能）
        st.subheader("🤖 AI Loop")
        st.caption("狀態: 待開發")
        st.caption("下次執行: N/A")


def render_summary_stats(stats: dict):
    """渲染總體統計卡片"""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">總實驗數</div>
            <div class="stat-value">{stats['total_count']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">驗證通過</div>
            <div class="stat-value">{stats['validated_count']}</div>
            <div class="stat-label">{stats['validated_count'] / max(stats['total_count'], 1) * 100:.1f}% 通過率</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        best_sharpe = stats.get('best_sharpe')
        sharpe_display = format_sharpe(best_sharpe) if best_sharpe else "N/A"

        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">最佳 Sharpe</div>
            <div class="stat-value">{sharpe_display}</div>
        </div>
        """, unsafe_allow_html=True)


def render_recent_experiments(experiments: list):
    """渲染最近實驗列表"""
    st.subheader("🕐 最近實驗")

    if not experiments:
        st.info("目前沒有實驗記錄")
        return

    for exp in experiments:
        exp_id = exp.get("experiment_id", "N/A")
        grade = exp.get("grade", "F")
        sharpe = exp.get("sharpe_ratio")
        total_return = exp.get("total_return")
        timestamp = exp.get("timestamp", "")
        validated = exp.get("validation_pass", False)

        # 建立卡片
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

            with col1:
                st.markdown(f"**{exp_id}**")
                st.caption(format_timestamp(timestamp))

            with col2:
                grade_bg = grade_color(grade)
                st.markdown(
                    f'<span class="grade-badge" style="background: {grade_bg}; color: white;">{grade}</span>',
                    unsafe_allow_html=True
                )

            with col3:
                st.metric("Sharpe", format_sharpe(sharpe))

            with col4:
                st.metric("報酬率", format_percentage(total_return))

            # 驗證狀態
            if validated:
                st.success("✓ 驗證通過", icon="✅")
            else:
                st.warning("✗ 未通過驗證", icon="⚠️")

            st.markdown("---")


def render_navigation_cards():
    """渲染導航卡片"""
    st.subheader("🚀 快速導航")

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container():
            st.markdown("""
            <div class="nav-card">
                <h3>📈 數據儀表板</h3>
                <p>查看整體表現趨勢、評級分布和統計圖表</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("前往儀表板", key="nav_dashboard", width="stretch"):
                st.switch_page("pages/1_📊_Dashboard.py")

    with col2:
        with st.container():
            st.markdown("""
            <div class="nav-card">
                <h3>📋 策略列表</h3>
                <p>瀏覽所有策略、過濾排序、查看詳細資訊</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("查看策略", key="nav_strategies", width="stretch"):
                st.switch_page("pages/2_Strategies.py")

    with col3:
        with st.container():
            st.markdown("""
            <div class="nav-card">
                <h3>⚖️ 策略比較</h3>
                <p>並排比較多個策略的表現指標</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("開始比較", key="nav_comparison", width="stretch"):
                st.switch_page("pages/3_Comparison.py")


def main():
    """主函數"""
    # 渲染側邊欄
    render_sidebar()

    # 主標題（右上角含主題切換）
    render_page_header(
        "🤖 AI 合約回測系統",
        "歡迎使用 AI 驅動的合約交易策略回測系統。本系統透過遺傳演算法自動探索策略空間，並使用多層級驗證機制評估策略品質。"
    )

    st.markdown("---")

    # 載入實驗資料
    experiments = load_experiments()
    stats = calculate_summary_stats(experiments)

    # 渲染統計卡片
    render_summary_stats(stats)

    st.markdown("---")

    # 兩欄佈局
    col1, col2 = st.columns([2, 1])

    with col1:
        # 最近實驗
        recent = get_latest_experiments(experiments, count=5)
        render_recent_experiments(recent)

    with col2:
        # 評級分布
        st.subheader("📊 評級分布")
        grade_dist = stats.get("grade_distribution", {})

        for grade, count in grade_dist.items():
            if count > 0:
                color = grade_color(grade)
                percentage = count / max(stats['total_count'], 1) * 100

                st.markdown(
                    f'<span class="grade-badge" style="background: {color}; color: white;">{grade}</span> '
                    f'{count} ({percentage:.1f}%)',
                    unsafe_allow_html=True
                )
                st.progress(percentage / 100)

        st.markdown("---")

        # 系統資訊
        st.subheader("ℹ️ 系統資訊")
        st.caption(f"總實驗數: {stats['total_count']}")
        st.caption(f"平均 Sharpe: {format_sharpe(stats.get('avg_sharpe'))}")

    st.markdown("---")

    # 導航卡片
    render_navigation_cards()

    # 頁尾
    st.markdown("---")
    st.caption("AI 合約回測系統 v1.0 | Powered by Streamlit")


if __name__ == "__main__":
    main()
