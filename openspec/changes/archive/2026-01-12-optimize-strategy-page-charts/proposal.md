# 策略頁面圖表優化提案

## Why

策略頁面（`ui/pages/2_Strategies.py`）目前存在以下問題：

1. **假數據問題**：權益曲線和月度報酬使用硬編碼隨機數據（line 177-228）
2. **數據範圍不一致**：權益曲線顯示 100 天，月度報酬顯示 12 個月
3. **缺乏真實數據連接**：未從 `BacktestResult` 載入實際回測結果
4. **缺乏圖表聯動**：拖動/縮放時權益曲線和月度報酬無法同步
5. **時間軸不統一**：兩個圖表使用不同的時間基準

這導致用戶無法查看真實的策略表現，影響決策品質。

---

## What Changes

### 核心功能變更

- **[NEW]** 從回測結果載入真實數據
  - 讀取 `BacktestResult.equity_curve`（權益曲線）
  - 讀取 `BacktestResult.daily_returns`（日報酬率）
  - 計算月度報酬統計（從日報酬聚合）

- **[MODIFIED]** 統一時間範圍
  - 權益曲線：顯示完整回測期間
  - 月度報酬：基於相同期間，按月聚合
  - 兩者使用一致的日期軸

- **[NEW]** 圖表聯動機制
  - 使用 Plotly `relayoutData` 捕捉縮放事件
  - 使用 Streamlit `session_state` 同步時間範圍
  - 雙向同步：任一圖表縮放，另一圖表跟隨

- **[MODIFIED]** 數據載入邏輯
  - 從 `learning/experiments.json` 載入實驗記錄
  - 提取 `BacktestResult` 數據（權益曲線、交易記錄）
  - 建立數據 cache 避免重複載入

---

## Impact

### Affected Specs

**NEW SPECS (需建立)**：
- `openspec/changes/optimize-strategy-page-charts/specs/ui-data-integration/spec.md`
  - UI 如何載入和展示回測數據
  - 數據格式轉換規範

### Affected Code

**核心變更檔案**：
- `/ui/pages/2_Strategies.py`
  - `load_strategy_results()` - 改為載入真實數據
  - `plot_equity_curve()` - 使用真實 equity_curve
  - `plot_monthly_heatmap()` - 基於真實 daily_returns 計算
  - **[NEW]** `sync_chart_zoom()` - 圖表聯動函數

**數據來源檔案**：
- `/learning/experiments.json` - 讀取實驗記錄
- `BacktestResult` 結構（from `src/backtester/engine.py`）
  - `equity_curve: pd.Series` - 權益曲線
  - `daily_returns: pd.Series` - 日報酬率
  - `trades: pd.DataFrame` - 交易記錄

**參考設計規格**：
- `openspec/changes/archive/.../ui-specs/validation-page.md` - 圖表設計參考

---

## Technical Approach

### 1. 數據載入整合

```python
# 從 experiments.json 載入
def load_strategy_results() -> pd.DataFrame:
    experiments_file = project_root / 'learning' / 'experiments.json'
    with open(experiments_file, 'r') as f:
        data = json.load(f)

    # 轉換為 DataFrame
    records = []
    for exp in data['experiments']:
        record = {
            'strategy_name': exp['strategy']['name'],
            'strategy_type': exp['strategy']['type'],
            'symbol': exp['config']['symbol'],
            'timeframe': exp['config']['timeframe'],
            # ... 其他欄位
            'experiment_id': exp['id']  # 用於載入詳細數據
        }
        records.append(record)

    return pd.DataFrame(records)
```

### 2. 權益曲線真實數據

```python
def plot_equity_curve(strategy_name: str, experiment_id: str) -> go.Figure:
    # 從實驗記錄載入 equity_curve
    # 注意：experiments.json 儲存的是摘要，equity_curve 需另外儲存
    # 方案 A：擴充 experiments.json 包含 equity_curve（序列化為 list）
    # 方案 B：建立 results/{exp_id}/equity_curve.csv

    equity_curve = load_equity_curve(experiment_id)  # pd.Series with DatetimeIndex

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_curve.index,  # 日期軸
        y=equity_curve.values,
        mode='lines',
        name='權益',
        line=dict(color='#2563eb', width=2)
    ))

    # 統一時間範圍（從 session_state 讀取）
    if 'chart_xrange' in st.session_state:
        fig.update_xaxes(range=st.session_state.chart_xrange)

    return fig
```

### 3. 月度報酬計算

```python
def plot_monthly_heatmap(strategy_name: str, experiment_id: str) -> go.Figure:
    daily_returns = load_daily_returns(experiment_id)  # pd.Series with DatetimeIndex

    # 按月聚合
    monthly_returns = daily_returns.resample('M').apply(
        lambda x: (1 + x).prod() - 1
    ) * 100  # 轉為百分比

    # 轉為熱力圖格式（年 x 月）
    pivot_data = monthly_returns.groupby([
        monthly_returns.index.year,
        monthly_returns.index.month
    ]).first().unstack(fill_value=0)

    # 繪製熱力圖...
```

### 4. 圖表聯動機制

```python
# 在 main() 中
if 'chart_xrange' not in st.session_state:
    st.session_state.chart_xrange = None

# 權益曲線
equity_fig = plot_equity_curve(selected_strategy, exp_id)
equity_chart = st.plotly_chart(equity_fig, use_container_width=True, key='equity')

# 捕捉縮放事件（需要 Streamlit + Plotly 聯動）
# 注意：Streamlit 原生不支援 plotly 回傳事件，需要其他方案：
# 方案 A：使用時間範圍選擇器（slider）統一控制
# 方案 B：使用 streamlit-plotly-events（第三方套件）
# 方案 C：僅實現「重置範圍」按鈕

# 推薦方案 A（最簡單且可靠）：
date_range = st.slider(
    "時間範圍",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)

# 所有圖表使用相同的 date_range
```

---

## Implementation Phases

詳見 `tasks.md`

---

## Risks & Mitigation

### Risk 1: experiments.json 未儲存 equity_curve

**緩解**：
- 方案 A：修改 `ExperimentRecorder` 儲存 equity_curve（序列化為 list）
- 方案 B：建立 `results/{exp_id}/equity_curve.csv` 獨立檔案
- **推薦方案 B**（避免 JSON 過大）

### Risk 2: Streamlit 原生不支援 Plotly 事件聯動

**緩解**：
- 使用時間範圍選擇器（slider）統一控制兩個圖表
- 提供「重置範圍」按鈕
- 未來可選：整合 `streamlit-plotly-events` 套件

### Risk 3: 歷史實驗可能缺少 equity_curve 數據

**緩解**：
- 檢查數據是否存在，不存在則顯示提示訊息
- 提供「重新執行回測」按鈕

---

## Future Enhancements

1. 新增基準比較線（如 BTC Buy & Hold）
2. 新增回撤標記（紅點顯示最大回撤位置）
3. 支援多策略權益曲線疊加比較
4. 匯出圖表為 PNG/PDF
