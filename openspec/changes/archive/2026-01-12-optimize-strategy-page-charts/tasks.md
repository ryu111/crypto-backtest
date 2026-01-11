# optimize-strategy-page-charts Implementation Tasks

## Progress
- Total: 13 tasks
- Completed: 13
- Status: COMPLETED

---

## 1. Foundation - 數據架構準備 (sequential)

- [x] 1.1 建立 equity_curve 儲存機制 | files: src/learning/recorder.py, results/{exp_id}/equity_curve.csv
  - 修改 `ExperimentRecorder.log_experiment()` 儲存 equity_curve 和 daily_returns 到獨立 CSV
  - 新增 `load_equity_curve(exp_id)` 和 `load_daily_returns(exp_id)` 輔助函數
  - 驗證序列化/反序列化正確性（日期索引保留）

- [x] 1.2 建立數據載入輔助模組 | files: ui/utils/data_loader.py
  - 新增 `load_experiment_data(exp_id)` - 載入完整實驗數據
  - 新增 `load_equity_curve(exp_id)` - 載入權益曲線
  - 新增 `load_daily_returns(exp_id)` - 載入日報酬率
  - 新增 `calculate_monthly_returns(daily_returns)` - 計算月度報酬
  - 處理檔案不存在的情況（返回 None 或 raise）

---

## 2. UI Design (sequential, agent: DESIGNER)

- [x] 2.1 設計策略詳情權益曲線規格 | output: ui-specs/strategy-equity-curve.md
  - 權益曲線佈局、顏色、互動元素
  - 時間範圍選擇器設計
  - 數據缺失狀態提示

- [x] 2.2 設計月度報酬熱力圖規格 | output: ui-specs/strategy-monthly-heatmap.md
  - 熱力圖色彩映射（紅綠漸層）
  - 多年度資料的展示方式
  - 與權益曲線的視覺一致性

- [x] 2.3 設計時間範圍聯動控制器 | output: ui-specs/time-range-control.md
  - Slider 樣式和佈局
  - 重置按鈕設計
  - 與圖表的視覺距離和分組

---

## 3. Core Implementation (parallel, agent: DEVELOPER)

- [x] 3.1 實作真實數據載入 | files: ui/pages/2_Strategies.py | ui-spec: ui-specs/strategy-equity-curve.md
  - 修改 `load_strategy_results()` 從 experiments.json 載入
  - 新增 `experiment_id` 欄位到 DataFrame
  - 使用 `@st.cache_data` 優化載入效能
  - 處理空數據或載入失敗情況

- [x] 3.2 實作權益曲線真實圖表 | files: ui/pages/2_Strategies.py | ui-spec: ui-specs/strategy-equity-curve.md
  - 重寫 `plot_equity_curve(strategy_name, experiment_id)`
  - 使用真實 equity_curve 數據（from data_loader）
  - 套用時間範圍篩選（from session_state）
  - 處理數據缺失情況（顯示提示訊息）

- [x] 3.3 實作月度報酬真實圖表 | files: ui/pages/2_Strategies.py | ui-spec: ui-specs/strategy-monthly-heatmap.md
  - 重寫 `plot_monthly_heatmap(strategy_name, experiment_id)`
  - 從 daily_returns 計算月度報酬
  - 支援多年度資料（Y 軸顯示年份）
  - 套用時間範圍篩選

- [x] 3.4 實作時間範圍聯動控制 | files: ui/pages/2_Strategies.py | ui-spec: ui-specs/time-range-control.md
  - 新增 `st.slider` 時間範圍選擇器
  - 使用 `st.session_state` 儲存選中範圍
  - 兩個圖表讀取相同的時間範圍
  - 新增「重置範圍」按鈕

---

## 4. Error Handling & Edge Cases (parallel, agent: DEVELOPER)

- [x] 4.1 處理歷史實驗缺少數據 | files: ui/pages/2_Strategies.py
  - 檢查 equity_curve.csv 是否存在
  - 不存在時顯示友善提示訊息
  - 提供「查看其他策略」或「返回」按鈕
  - **新增**：時間範圍篩選後無數據檢查
  - **新增**：數據全為 NaN 檢查
  - **修正**：fillna() 棄用語法

- [x] 4.2 處理時間範圍異常情況 | files: ui/pages/2_Strategies.py
  - 時間範圍合理性驗證（start_date > end_date）
  - 篩選後數據為空的友善提示
  - 所有錯誤情況返回空白圖表（保持佈局一致）

---

## 5. Testing & Validation (sequential)

- [ ] 5.1 單元測試數據載入函數 | files: tests/ui/test_data_loader.py
  - 測試 `load_equity_curve()` 正確載入
  - 測試 `calculate_monthly_returns()` 計算正確
  - 測試檔案不存在的錯誤處理

- [ ] 5.2 整合測試策略頁面 | files: tests/ui/test_strategies_page.py
  - 測試選擇策略後圖表正確顯示
  - 測試時間範圍調整圖表同步更新
  - 測試數據缺失時提示訊息顯示

---

## Notes

### 數據儲存格式

```
results/
└── {exp_id}/
    ├── equity_curve.csv       # index=date, columns=['equity']
    ├── daily_returns.csv      # index=date, columns=['return']
    └── trades.csv             # 交易記錄（已存在）
```

### 時間範圍同步邏輯

```python
# 初始化
if 'time_range' not in st.session_state:
    st.session_state.time_range = (min_date, max_date)

# 控制器
time_range = st.slider("時間範圍", ..., key='time_range')

# 圖表讀取
def plot_equity_curve(...):
    start, end = st.session_state.time_range
    filtered_data = equity_curve[start:end]
    ...
```

### 月度報酬計算

```python
def calculate_monthly_returns(daily_returns: pd.Series) -> pd.DataFrame:
    """
    Args:
        daily_returns: 日報酬率 (index=date)

    Returns:
        DataFrame with columns=[year, month, return_pct]
    """
    monthly = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    monthly_pct = monthly * 100

    df = pd.DataFrame({
        'year': monthly_pct.index.year,
        'month': monthly_pct.index.month,
        'return': monthly_pct.values
    })

    return df
```

### 失敗處理範例

```python
try:
    equity_curve = load_equity_curve(exp_id)
except FileNotFoundError:
    st.warning("""
    ⚠️ 此實驗缺少權益曲線數據

    可能原因：
    - 此實驗在數據儲存機制更新前執行
    - 數據檔案已被刪除

    建議：
    - 重新執行此策略的回測
    - 或查看其他策略
    """)
    st.stop()
```

---

## Dependencies

- Phase 1 必須完成後才能開始 Phase 2
- Phase 2 完成後，Phase 3 可並行執行
- Phase 4 可在 Phase 3 完成後並行執行
- Phase 5 必須在 Phase 3, 4 完成後執行
