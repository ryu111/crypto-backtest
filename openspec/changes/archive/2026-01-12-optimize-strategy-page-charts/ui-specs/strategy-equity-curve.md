# 策略詳情頁面 - 權益曲線設計規格

## 📋 需求理解

### 目標
在策略詳情展開區域中顯示**真實的權益曲線**和**月度報酬熱力圖**，取代現有的假數據。使用者可以通過時間範圍選擇器統一控制兩個圖表的顯示區間。

### 使用者
量化交易者，需要:
- 查看策略在回測期間的真實權益變化
- 了解月度報酬分布和波動性
- 選擇特定時間範圍進行細部分析
- 快速識別關鍵績效時期（高/低報酬月份）

### 關鍵互動
1. 展開策略列表項目，查看詳細圖表
2. 使用時間範圍滑桿同步縮放兩個圖表
3. Hover 圖表查看具體數值
4. 視覺化識別績效趨勢

### UX 考量（基於 psychology.md）

**視覺層級（Gestalt Principles）**:
- 權益曲線為主要焦點（佔據上方較大空間）
- 月度熱力圖為補充資訊（下方次要位置）
- 時間範圍選擇器放在圖表上方（Common Region 分組）

**認知負荷最小化（Miller's Law）**:
- 每個圖表專注單一資訊（權益 vs 月度報酬）
- 統一的時間軸減少認知負擔

**即時回饋（Fitts's Law）**:
- Hover 時立即顯示 tooltip
- 滑桿調整後即時更新圖表

**熟悉性（Jakob's Law）**:
- 遵循 Plotly 標準互動模式
- 時間選擇器使用 Streamlit 原生元件

---

## 📐 LAYOUT

### 展開區域結構

```
┌────────────────────────────────────────────────────────────┐
│ [策略名稱] MA Cross (10/30)                    [收合 ▲]    │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ 時間範圍選擇                                          │ │
│  │ [━━━━━●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━●━━━━━━]        │ │
│  │ 2024-01-01                              2024-12-31   │ │
│  │                            [重置範圍]                 │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ 📈 權益曲線                                           │ │
│  │                                                       │ │
│  │  [折線圖: 日期 vs 權益]                              │ │
│  │  高度: 400px                                         │ │
│  │                                                       │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ 📊 月度報酬熱力圖                                     │ │
│  │                                                       │ │
│  │  [熱力圖: 月份 x 年份]                               │ │
│  │  高度: 200px                                         │ │
│  │                                                       │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 佈局模式
- **容器**: Streamlit `st.expander()` 展開元件
- **時間選擇器**: `st.columns([4, 1])` - 左側滑桿，右側重置按鈕
- **圖表佈局**: 垂直堆疊（`st.container()`）
- **圖表寬度**: `use_container_width=True` 自適應

### 間距系統（基於 tokens.md + styles.py）

```css
/* 已定義於 styles.py */
--spacing-sm: 0.5rem;   /* 8px  - 圖表內元素間距 */
--spacing-md: 1rem;     /* 16px - 時間選擇器與圖表間距 */
--spacing-lg: 1.5rem;   /* 24px - 兩個圖表之間間距 */
```

### 容器尺寸

| 元素 | 高度 | 寬度 |
|------|------|------|
| 時間選擇器區塊 | 80px | 100% |
| 權益曲線圖表 | 400px | 100% |
| 月度報酬熱力圖 | 200px | 100% |
| 圖表間距 | 24px (`--spacing-lg`) | - |

---

## 🎨 VISUAL

### 顏色方案（基於 tokens.md + styles.py）

```css
/* 語意化色彩 - 已定義於 styles.py */
--color-primary: #2563eb;         /* 權益曲線主線 */
--color-success: #22c55e;         /* 正報酬（熱力圖綠色端） */
--color-error: #ef4444;           /* 負報酬（熱力圖紅色端） */
--color-text: #111827;            /* 標題、軸標籤 */
--color-text-secondary: #6b7280;  /* 輔助文字 */
--color-border: #e5e7eb;          /* 圖表邊框 */
--color-surface: #ffffff;         /* 圖表背景 */
--color-surface-raised: #f9fafb;  /* 時間選擇器背景 */
```

### 權益曲線顏色

```python
# 主線
line_color = '#2563eb'  # --color-primary
line_width = 2

# 填充區（可選：曲線下方淡色填充）
fill_color = 'rgba(37, 99, 235, 0.1)'  # --color-primary with 10% opacity
```

### 月度報酬熱力圖顏色

```python
# Plotly colorscale
colorscale = 'RdYlGn'  # 紅-黃-綠（負-中性-正）

# 色彩映射邏輯
# < 0%  紅色系 (#ef4444 系列)
# ≈ 0%  黃色系 (#eab308 系列)
# > 0%  綠色系 (#22c55e 系列)

# 色條（Colorbar）
colorbar = {
    'title': '報酬率 (%)',
    'titlefont': {'size': 12},
    'tickfont': {'size': 10},
    'thickness': 15,
    'len': 0.7
}
```

### 背景與表面（60-30-10 法則應用）

```
60% - 主背景
  圖表區域: --color-surface (#ffffff)

30% - 次要區域
  時間選擇器背景: --color-surface-raised (#f9fafb)

10% - 強調色
  權益曲線: --color-primary (#2563eb)
  熱力圖正值: --color-success (#22c55e)
  熱力圖負值: --color-error (#ef4444)
```

### 字體規範（基於 tokens.md）

```css
/* 圖表標題 */
title: 1.125rem (18px), font-weight: 600, color: var(--color-text)

/* 軸標籤 */
axis_labels: 0.875rem (14px), font-weight: 400, color: var(--color-text-secondary)

/* Tooltip 文字 */
tooltip: 0.875rem (14px), font-weight: 400, background: white

/* 熱力圖數值標註 */
heatmap_text: 0.75rem (12px), font-weight: 500
```

### 邊框與圓角（基於 styles.py）

```css
/* 時間選擇器容器 */
border-radius: var(--radius-lg);  /* 8px */
border: 1px solid var(--color-border);
padding: var(--spacing-md);

/* 圖表容器 */
border-radius: var(--radius-lg);  /* 8px */
```

---

## 🔄 STATES

### 圖表載入狀態

| 狀態 | 視覺 | 說明 |
|------|------|------|
| **載入中** | Spinner + Skeleton | 初次載入數據 |
| **數據缺失** | Empty State | 實驗未儲存 equity_curve |
| **正常顯示** | 完整圖表 | 成功載入數據 |
| **錯誤** | Error Alert | 數據格式錯誤 |

### 載入中（Loading）

```python
with st.spinner('載入權益曲線...'):
    # 載入數據
    equity_curve = load_equity_curve(experiment_id)
```

視覺元素：
- Streamlit 原生 spinner
- 位置：圖表中心
- 顏色：`--color-primary`

### 數據缺失（Empty State）

```python
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
```

視覺元素：
- 使用 `st.info()` 藍色提示框
- 圖示：📊
- 包含原因說明和建議操作

### 錯誤狀態（Error）

```python
st.error(f"""
❌ **載入失敗**

{error_message}

**詳細錯誤**：
```
{traceback_info}
```
""")
```

視覺元素：
- 使用 `st.error()` 紅色警告框
- 圖示：❌
- 顯示錯誤訊息和技術細節

### 時間選擇器狀態

**預設（Default）**:
- 顯示完整時間範圍
- 滑桿把手位於兩端

**選擇中（Active）**:
- 滑桿拖動時，即時更新日期標籤
- 圖表即時縮放（session_state 更新）

**重置（Reset）**:
- 點擊「重置範圍」按鈕
- 滑桿回到兩端
- 圖表顯示完整範圍

---

## 📱 RESPONSIVE

### 主要裝置
- **Desktop**: 1280px+ (主要使用場景)
- **Tablet**: 768px-1280px
- **Mobile**: < 768px (次要，可能隱藏或縮小圖表)

### 圖表響應式設計

```python
# Streamlit 自動處理響應式
st.plotly_chart(fig, use_container_width=True)

# 圖表高度根據螢幕寬度調整（可選）
import streamlit as st

# 偵測螢幕寬度（需要額外 JS，非必要）
# 或使用固定高度，依賴 Plotly 內建縮放
```

### 圖表高度調整

| 裝置 | 權益曲線高度 | 熱力圖高度 |
|------|--------------|------------|
| Desktop (≥1280px) | 400px | 200px |
| Tablet (768-1280px) | 350px | 180px |
| Mobile (<768px) | 300px | 150px |

> **實作方式**：Streamlit 的 `use_container_width=True` 自動處理寬度，高度使用固定值（Plotly layout.height）。行動裝置建議使用較小高度以減少捲動。

---

## 📊 時間範圍選擇器規格

### 元件配置

```python
# 佈局
col_slider, col_reset = st.columns([4, 1])

with col_slider:
    # 時間範圍滑桿
    date_range = st.slider(
        label="選擇時間範圍",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),  # 預設全範圍
        format="YYYY-MM-DD",
        key="equity_date_range"
    )

with col_reset:
    # 重置按鈕（垂直置中）
    st.markdown("<div style='padding-top: 1.5rem;'></div>", unsafe_allow_html=True)
    if st.button("🔄 重置範圍", key="reset_date_range"):
        st.session_state.equity_date_range = (min_date, max_date)
        st.rerun()
```

### 視覺樣式

**滑桿樣式**（已定義於 styles.py）:
```css
.stSlider > div > div > div {
    background: var(--color-primary);  /* #2563eb */
}
```

**重置按鈕樣式**:
- 類型：`type="secondary"` (Streamlit 預設次要按鈕)
- 圖示：🔄 表示重置動作
- 對齊：與滑桿垂直置中

### 互動邏輯

```python
# 1. 初始化 session_state
if 'chart_xrange' not in st.session_state:
    st.session_state.chart_xrange = (min_date, max_date)

# 2. 滑桿變化時更新
selected_range = st.slider(...)
if selected_range != st.session_state.chart_xrange:
    st.session_state.chart_xrange = selected_range

# 3. 圖表使用統一範圍
fig.update_xaxes(range=st.session_state.chart_xrange)
```

---

## 📈 權益曲線圖表規格

### 資料來源

```python
# 從 BacktestResult 載入
equity_curve = load_equity_curve(experiment_id)
# 預期格式: pd.Series
#   Index: DatetimeIndex (日期)
#   Values: float (權益值)
```

### Plotly 配置

```python
import plotly.graph_objects as go

fig = go.Figure()

# 主線
fig.add_trace(go.Scatter(
    x=equity_curve.index,
    y=equity_curve.values,
    mode='lines',
    name='權益',
    line=dict(
        color='#2563eb',  # --color-primary
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
fig.update_layout(
    title=dict(
        text='權益曲線',
        font=dict(size=18, weight=600, color='#111827')  # --color-text
    ),
    xaxis=dict(
        title='日期',
        titlefont=dict(size=14, color='#6b7280'),  # --color-text-secondary
        tickfont=dict(size=12),
        gridcolor='#f3f4f6',
        range=st.session_state.chart_xrange  # 應用時間範圍
    ),
    yaxis=dict(
        title='權益 ($)',
        titlefont=dict(size=14, color='#6b7280'),
        tickfont=dict(size=12),
        gridcolor='#f3f4f6',
        tickformat='$,.0f'  # 貨幣格式
    ),
    height=400,
    margin=dict(l=60, r=40, t=60, b=60),
    plot_bgcolor='white',
    paper_bgcolor='white',
    hovermode='x unified',
    hoverlabel=dict(
        bgcolor='white',
        font_size=13,
        font_family="'Inter', sans-serif"
    )
)

# 統一配置（工具列、縮放等）
fig.update_xaxes(showspikes=True, spikecolor='#d1d5db', spikethickness=1)
fig.update_yaxes(showspikes=True, spikecolor='#d1d5db', spikethickness=1)
```

### Hover Tooltip 規格

```
┌─────────────────────┐
│ 日期: 2024-05-15    │
│ 權益: $12,345.67    │
└─────────────────────┘
```

**格式規則**:
- 日期：`YYYY-MM-DD`
- 金額：千分位逗號 + 兩位小數

### 工具列配置

```python
config = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
    'toImageButtonOptions': {
        'format': 'png',
        'filename': f'equity_curve_{strategy_name}',
        'height': 800,
        'width': 1400,
        'scale': 2
    }
}

st.plotly_chart(fig, use_container_width=True, config=config)
```

---

## 📊 月度報酬熱力圖規格

### 資料來源與計算

```python
# 從 BacktestResult 載入日報酬
daily_returns = load_daily_returns(experiment_id)
# 預期格式: pd.Series
#   Index: DatetimeIndex (日期)
#   Values: float (日報酬率，小數形式如 0.02 = 2%)

# 按月聚合（複利計算）
monthly_returns = daily_returns.resample('M').apply(
    lambda x: (1 + x).prod() - 1
) * 100  # 轉為百分比

# 轉為熱力圖格式（年 x 月）
pivot_data = monthly_returns.to_frame('return')
pivot_data['year'] = pivot_data.index.year
pivot_data['month'] = pivot_data.index.month
heatmap_matrix = pivot_data.pivot_table(
    values='return',
    index='year',
    columns='month',
    fill_value=0
)
```

### Plotly 配置

```python
import plotly.graph_objects as go

# 月份標籤
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig = go.Figure(data=go.Heatmap(
    z=heatmap_matrix.values,
    x=month_labels,
    y=heatmap_matrix.index.astype(str),  # 年份
    colorscale='RdYlGn',  # 紅-黃-綠
    zmid=0,  # 中心點設為 0（黃色）
    text=heatmap_matrix.values,
    texttemplate='%{text:.1f}%',
    textfont=dict(size=12, weight=500),
    colorbar=dict(
        title='報酬率 (%)',
        titlefont=dict(size=12),
        tickfont=dict(size=10),
        thickness=15,
        len=0.7
    ),
    hovertemplate='<b>%{y} %{x}</b><br>' +
                  '報酬率: %{z:.2f}%<br>' +
                  '<extra></extra>'
))

# 佈局配置
fig.update_layout(
    title=dict(
        text='月度報酬熱力圖',
        font=dict(size=18, weight=600, color='#111827')
    ),
    xaxis=dict(
        title='',
        side='bottom',
        tickfont=dict(size=12),
        range=get_month_range_for_date_filter(st.session_state.chart_xrange)  # 根據選擇範圍篩選月份
    ),
    yaxis=dict(
        title='年份',
        tickfont=dict(size=12)
    ),
    height=200,
    margin=dict(l=60, r=40, t=60, b=40),
    plot_bgcolor='white',
    paper_bgcolor='white'
)
```

### Hover Tooltip 規格

```
┌────────────────────┐
│ 2024 May          │
│ 報酬率: 5.23%      │
└────────────────────┘
```

### 色彩映射邏輯

```python
# Plotly 'RdYlGn' colorscale 自動映射
# 負值範圍 → 紅色系（#ef4444 相近）
# 接近 0   → 黃色系（#eab308 相近）
# 正值範圍 → 綠色系（#22c55e 相近）

# zmid=0 確保 0% 對應黃色中心
```

### 時間範圍篩選邏輯

```python
def get_month_range_for_date_filter(date_range):
    """根據日期範圍篩選要顯示的月份"""
    start_date, end_date = date_range

    # 計算月份範圍
    start_month = start_date.month
    end_month = end_date.month
    start_year = start_date.year
    end_year = end_date.year

    # 如果跨年，顯示所有月份
    if start_year != end_year:
        return None  # 不設限制

    # 同一年內，僅顯示範圍內月份（Plotly 不直接支援，改用 x 軸範圍）
    # 實作時可選擇過濾資料或使用視覺提示
    return None  # 簡化實作：總是顯示全部月份
```

> **Note**: 熱力圖的時間篩選較複雜，建議簡化為「只要日期範圍包含該月，就顯示該月」，或總是顯示全部月份。

---

## 🚨 錯誤處理

### 數據載入錯誤

```python
try:
    equity_curve = load_equity_curve(experiment_id)
except FileNotFoundError:
    st.error("""
    ❌ **權益曲線數據不存在**

    實驗 ID: {experiment_id}

    **建議**：檢查實驗記錄或重新執行回測。
    """)
    st.stop()
except Exception as e:
    st.error(f"""
    ❌ **載入權益曲線失敗**

    錯誤訊息: {str(e)}

    請聯絡技術支援或檢查數據完整性。
    """)
    st.stop()
```

### 數據格式錯誤

```python
# 驗證數據格式
if not isinstance(equity_curve.index, pd.DatetimeIndex):
    st.error("❌ 權益曲線索引必須為日期格式")
    st.stop()

if equity_curve.isnull().any():
    st.warning("⚠️ 權益曲線包含缺失值，已自動填充")
    equity_curve = equity_curve.fillna(method='ffill')
```

### 空數據處理

```python
if len(equity_curve) == 0:
    st.info("""
    📊 **無權益曲線數據**

    此策略實驗未產生交易記錄。

    **可能原因**：
    - 策略條件未觸發
    - 回測期間過短
    - 策略參數設定錯誤
    """)
    st.stop()
```

---

## 🎯 完整實作範例

### Python 程式碼結構

```python
def render_strategy_detail_charts(strategy_data: dict):
    """
    渲染策略詳情圖表區域

    Args:
        strategy_data: 策略資料（包含 experiment_id）
    """

    # 1. 載入數據
    experiment_id = strategy_data['experiment_id']

    try:
        equity_curve = load_equity_curve(experiment_id)
        daily_returns = load_daily_returns(experiment_id)
    except Exception as e:
        st.error(f"❌ 載入數據失敗: {str(e)}")
        return

    # 2. 數據驗證
    if len(equity_curve) == 0:
        st.info("📊 無權益曲線數據")
        return

    # 3. 時間範圍選擇器
    min_date = equity_curve.index.min().date()
    max_date = equity_curve.index.max().date()

    # 初始化 session_state
    if 'chart_xrange' not in st.session_state:
        st.session_state.chart_xrange = (min_date, max_date)

    # 渲染選擇器
    col_slider, col_reset = st.columns([4, 1])

    with col_slider:
        date_range = st.slider(
            "選擇時間範圍",
            min_value=min_date,
            max_value=max_date,
            value=st.session_state.chart_xrange,
            format="YYYY-MM-DD",
            key=f"date_range_{experiment_id}"
        )

    with col_reset:
        st.markdown("<div style='padding-top: 1.5rem;'></div>", unsafe_allow_html=True)
        if st.button("🔄 重置範圍", key=f"reset_{experiment_id}"):
            st.session_state.chart_xrange = (min_date, max_date)
            st.rerun()

    # 更新 session_state
    st.session_state.chart_xrange = date_range

    # 4. 篩選數據（根據時間範圍）
    equity_filtered = equity_curve.loc[
        (equity_curve.index.date >= date_range[0]) &
        (equity_curve.index.date <= date_range[1])
    ]

    # 5. 繪製權益曲線
    equity_fig = plot_equity_curve(equity_filtered, strategy_data['strategy_name'])
    st.plotly_chart(equity_fig, use_container_width=True, key=f"equity_{experiment_id}")

    # 6. 繪製月度報酬熱力圖
    monthly_fig = plot_monthly_heatmap(daily_returns, date_range, strategy_data['strategy_name'])
    st.plotly_chart(monthly_fig, use_container_width=True, key=f"monthly_{experiment_id}")


def plot_equity_curve(equity_curve: pd.Series, strategy_name: str) -> go.Figure:
    """繪製權益曲線"""
    fig = go.Figure()

    # [完整實作如上述規格]

    return fig


def plot_monthly_heatmap(daily_returns: pd.Series, date_range: tuple, strategy_name: str) -> go.Figure:
    """繪製月度報酬熱力圖"""
    # 按月聚合
    monthly_returns = daily_returns.resample('M').apply(
        lambda x: (1 + x).prod() - 1
    ) * 100

    # [完整實作如上述規格]

    return fig
```

---

## 📝 Checklist

### 設計完整性
- [x] 遵循 60-30-10 色彩法則
- [x] Token 引用正確（styles.py）
- [x] 所有互動元素有明確狀態
- [x] 響應式設計考量
- [x] 載入/錯誤/空狀態處理

### UX 原則應用
- [x] Miller's Law: 每個圖表單一焦點
- [x] Jakob's Law: 遵循 Plotly/Streamlit 慣例
- [x] Fitts's Law: 時間選擇器易操作
- [x] Gestalt Principles: 視覺分組清晰
- [x] 即時回饋: Hover tooltip 即時顯示

### 技術實作
- [x] 使用 `get_common_css()` 載入樣式
- [x] 圖表配置統一（Plotly theme）
- [x] 時間範圍同步機制清晰
- [x] 數值格式化一致
- [x] 錯誤處理完整

---

## 🎓 開發者注意事項

### 從設計到程式碼

1. **讀取現有樣式**
   ```python
   from ui.styles import get_common_css
   st.markdown(get_common_css(), unsafe_allow_html=True)
   ```

2. **使用 CSS Variables（不要 hardcode）**
   ```python
   # ❌ 錯誤
   line_color = "#2563eb"

   # ✅ 正確（在註解中說明對應 token）
   line_color = '#2563eb'  # --color-primary from styles.py
   ```

3. **圖表顏色一致性**
   ```python
   # 所有圖表使用相同顏色定義
   CHART_COLORS = {
       'primary_line': '#2563eb',  # --color-primary
       'success': '#22c55e',        # --color-success
       'error': '#ef4444',          # --color-error
   }
   ```

4. **時間範圍同步**
   ```python
   # 使用 session_state 跨圖表共享
   st.session_state.chart_xrange = (start_date, end_date)

   # 所有圖表應用
   fig.update_xaxes(range=st.session_state.chart_xrange)
   ```

### 資料結構預期

```python
# equity_curve: pd.Series
# Index: DatetimeIndex
# Values: float (累積權益，如 10000, 10200, 10150...)
equity_curve = pd.Series(
    data=[10000, 10200, 10150, 10300, ...],
    index=pd.date_range('2024-01-01', periods=100, freq='D')
)

# daily_returns: pd.Series
# Index: DatetimeIndex
# Values: float (日報酬率小數，如 0.02 = 2%)
daily_returns = pd.Series(
    data=[0.02, -0.005, 0.015, ...],
    index=pd.date_range('2024-01-01', periods=100, freq='D')
)
```

---

## 🔗 相關規範

- **Design Tokens**: `~/.claude/skills/ui/references/tokens.md`
- **Component Specs**: `~/.claude/skills/ui/references/components.md`
- **UX Patterns**: `~/.claude/skills/ux/references/patterns.md`
- **Psychology**: `~/.claude/skills/ux/references/psychology.md`
- **現有樣式**: `/ui/styles.py`
- **參考規格**: `/openspec/changes/archive/.../ui-specs/validation-page.md`
- **當前頁面**: `/ui/pages/2_Strategies.py`
