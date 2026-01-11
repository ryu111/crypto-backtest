"""
AI 回測系統主頁面
"""

import streamlit as st

# 設定頁面配置
st.set_page_config(
    page_title="AI 回測系統",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自訂 CSS
st.markdown("""
<style>
    /* 主要色彩 */
    :root {
        --primary-color: #3b82f6;
        --success-color: #22c55e;
        --warning-color: #eab308;
        --error-color: #ef4444;
        --text-color: #111827;
        --text-secondary: #6b7280;
        --border-color: #e5e7eb;
        --surface: #ffffff;
        --surface-raised: #f9fafb;
    }

    /* 全域字體 */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* 標題 */
    h1 {
        font-weight: 700;
        color: var(--text-color);
    }

    h2, h3 {
        font-weight: 600;
        color: var(--text-color);
    }

    /* 指標卡片 */
    [data-testid="stMetric"] {
        background: var(--surface-raised);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid var(--border-color);
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.875rem;
        color: var(--text-secondary);
        font-weight: 500;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.875rem;
        font-weight: 700;
        color: var(--text-color);
    }

    /* 按鈕 */
    .stButton > button {
        background: var(--primary-color);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
        font-weight: 500;
        transition: all 150ms ease;
    }

    .stButton > button:hover {
        filter: brightness(0.95);
        transform: scale(0.98);
    }

    /* 表格 */
    [data-testid="stDataFrame"] {
        border-radius: 0.5rem;
        overflow: hidden;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--surface-raised);
    }

    /* 資訊框 */
    .stAlert {
        border-radius: 0.5rem;
        border-left: 4px solid var(--primary-color);
    }
</style>
""", unsafe_allow_html=True)


def main():
    """主頁面"""

    # Logo & 標題
    st.title("🤖 AI 回測系統")
    st.markdown("專業級量化交易策略回測與驗證平台")
    st.markdown("---")

    # 簡介
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("系統功能")

        st.markdown("""
        ### 📊 完整回測流程
        - **策略開發**: 趨勢、動量、均值回歸策略
        - **參數優化**: Walk-Forward Analysis
        - **嚴格驗證**: 5 階段驗證流程
        - **風險管理**: 倉位管理、止損止盈
        - **永續合約**: 資金費率、槓桿交易

        ### 🧪 驗證系統
        - **Stage 1**: 基本績效 (Sharpe > 1.0)
        - **Stage 2**: 樣本外測試
        - **Stage 3**: Walk-Forward Analysis
        - **Stage 4**: Monte Carlo 模擬
        - **Stage 5**: 跨標的驗證

        ### 📈 學習系統
        - **實驗記錄**: 自動記錄所有回測
        - **洞察累積**: 提取成功經驗
        - **策略演進**: 追蹤版本改進
        - **知識管理**: Memory MCP 整合
        """)

    with col2:
        st.subheader("快速開始")

        st.info("""
        **1. 查看 Dashboard**

        點選左側 `Dashboard` 查看總覽
        """)

        st.success("""
        **2. 執行範例回測**

        ```bash
        python examples/trend_strategies_example.py
        ```
        """)

        st.warning("""
        **3. 記錄實驗**

        ```bash
        python examples/learning/record_experiment.py
        ```
        """)

    st.markdown("---")

    # 系統狀態
    st.subheader("系統狀態")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="回測引擎", value="運行中", delta="正常")

    with col2:
        st.metric(label="驗證系統", value="運行中", delta="正常")

    with col3:
        st.metric(label="學習系統", value="運行中", delta="正常")

    with col4:
        st.metric(label="資料管道", value="運行中", delta="正常")

    st.markdown("---")

    # 文件連結
    st.subheader("📚 相關文件")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **回測核心**
        - [回測引擎](./docs)
        - [永續合約](./src/backtester/README_PERPETUAL.md)
        - [VectorBT 基礎](./docs)
        """)

    with col2:
        st.markdown("""
        **驗證與優化**
        - [策略驗證](./src/validator/README.md)
        - [Walk-Forward](./docs/optimizer/walk_forward.md)
        - [Monte Carlo](./docs/monte_carlo.md)
        """)

    with col3:
        st.markdown("""
        **學習系統**
        - [實驗記錄](./src/learning/README.md)
        - [Memory 整合](./src/learning/MEMORY_INTEGRATION.md)
        - [自動化](./src/automation/README.md)
        """)

    st.markdown("---")

    # Footer
    st.caption("AI 回測系統 v1.0 | 由 Claude Code 協助開發")


if __name__ == "__main__":
    main()
