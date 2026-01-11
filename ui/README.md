# AI 回測系統 UI

基於 Streamlit 的視覺化介面，展示回測結果與策略分析。

## 啟動方式

### 1. 安裝依賴

```bash
pip install streamlit plotly pandas
```

### 2. 啟動 UI

```bash
# 從專案根目錄執行
streamlit run ui/Home.py

# 或指定 port
streamlit run ui/Home.py --server.port 8501
```

### 3. 瀏覽器訪問

```
http://localhost:8501
```

## 頁面結構

```
ui/
├── Home.py                    # 主頁面（系統總覽）
└── pages/
    ├── 1_📊_Dashboard.py      # Dashboard 頁面
    ├── 2_Strategies.py        # 策略列表頁面 ✅ 已實作
    └── 3_Comparison.py        # 績效比較（待實作）
```

## 頁面功能

### Home (主頁)
- 系統簡介
- 功能清單
- 快速開始指南
- 系統狀態
- 文件連結

### Dashboard (儀表板)
- **核心指標卡片**
  - 總實驗數
  - 驗證通過數
  - 最佳 Sharpe Ratio
  - 平均 Sharpe Ratio
  - 記錄策略數

- **績效分析圖表**
  - Sharpe Ratio 分布直方圖
  - 評級分布圓餅圖

- **時間趨勢**
  - 每日最佳 Sharpe 趨勢線

- **Top 10 排行榜**
  - 按 Sharpe Ratio 排序
  - 顯示策略、報酬率、最大回撤、評級

- **策略類型分析**
  - 各類型平均 Sharpe
  - 各類型實驗數量

- **最近活動**
  - 最近 10 個實驗記錄

### 2_Strategies (策略列表) ✅ 已實作

展示所有策略實驗結果，提供強大的篩選、排序、分頁功能。

#### 1. 側邊欄篩選器

**數值篩選（滑桿）**
- 最小 Sharpe Ratio (0.0 ~ 5.0)
- 最小報酬率 (-50% ~ 200%)
- 最大回撤 (0% ~ 50%)
- 最小交易筆數 (0 ~ 500)

**分類篩選（多選）**
- 驗證等級：A, B, C, D, F
- 策略類型：趨勢, 動量, 均值回歸
- 標的：BTCUSDT, ETHUSDT
- 時間框架：1h, 4h, 1d

**排序**
- Sharpe Ratio（高→低）
- 報酬率（高→低）
- 回撤（低→高）
- 時間（新→舊）

#### 2. 概覽儀表板
- 總策略數（篩選後/總數）
- 平均 Sharpe Ratio
- 平均報酬率
- A 級策略數量

#### 3. 策略列表表格
顯示欄位：
- 策略名稱
- 報酬率 (%)
- 年化報酬 (%)
- Sharpe Ratio
- 最大回撤 (%)
- 交易筆數
- 勝率 (%)
- 驗證等級（色彩徽章）
- 過擬合率（WFA Efficiency）

特色：
- 分頁：每頁 20 筆
- 互動式表格
- 即時篩選

#### 4. 策略詳情展開區

點擊策略後可查看：

**基本資訊**
- 策略類型
- 交易標的
- 時間框架
- 建立時間

**績效指標**
- 總報酬率
- 年化報酬
- Sharpe Ratio
- 最大回撤

**交易統計**
- 交易筆數
- 勝率
- 過擬合率
- 驗證等級徽章

**策略參數**
- JSON 格式展示
- 完整參數配置

**視覺化圖表**
- 權益曲線圖（Plotly 互動）
- 月度報酬熱力圖

**AI 洞察**
- 自動分析策略表現
- 風險評估
- 改進建議

#### 5. 匯出功能
- 匯出篩選結果為 CSV
- 匯出選中策略詳情為 JSON

## 資料來源

### Dashboard
讀取以下資料：
```
learning/experiments.json
```
格式請參考：`src/learning/README.md`

### 策略列表頁面

**目前狀態**：使用範例資料（位於 `load_strategy_results()` 函數）

**未來整合**：需從以下路徑載入實際策略驗證結果
```
results/strategies/*.json
```

**資料格式範例**：
```json
{
  "strategy_name": "MA Cross (10/30)",
  "strategy_type": "趨勢",
  "symbol": "BTCUSDT",
  "timeframe": "4h",
  "total_return": 45.8,
  "annual_return": 28.2,
  "sharpe_ratio": 1.85,
  "max_drawdown": 12.5,
  "total_trades": 158,
  "win_rate": 62.5,
  "grade": "A",
  "wfa_efficiency": 0.85,
  "params": {
    "fast_period": 10,
    "slow_period": 30
  },
  "created_at": "2024-01-10 14:30:00",
  "equity_curve": [10000, 10050, ...],
  "monthly_returns": [2.5, -1.2, ...]
}
```

**整合步驟**：
1. 將 `load_strategy_results()` 的 TODO 替換為實際實作
2. 從 `results/strategies/` 讀取所有 JSON 檔案
3. 解析 `ValidationResult` 物件
4. 轉換為 DataFrame 格式

## 設計規範

### 色彩系統

| 用途 | 顏色 | 變數 |
|------|------|------|
| 主色 | #3b82f6 | --primary-color |
| 成功 | #22c55e | --success-color |
| 警告 | #eab308 | --warning-color |
| 錯誤 | #ef4444 | --error-color |

### 評級顏色

| 評級 | 顏色 | 說明 |
|------|------|------|
| A | #22c55e | Sharpe >= 2.0 |
| B | #3b82f6 | Sharpe >= 1.5 |
| C | #eab308 | Sharpe >= 1.0 |
| D | #f97316 | Sharpe >= 0.5 |
| F | #ef4444 | Sharpe < 0.5 |

## 開發指南

### 新增頁面

1. 在 `pages/` 目錄建立檔案：`N_PageName.py`
2. 設定頁面配置：

```python
import streamlit as st

st.set_page_config(
    page_title="頁面標題",
    page_icon="📊",
    layout="wide"
)

st.title("📊 頁面標題")
```

3. Streamlit 會自動在側邊欄加入連結

### 使用範例資料

```python
from pathlib import Path
import json

def load_experiments():
    file_path = Path(__file__).parent.parent.parent / "learning" / "experiments.json"
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
```

### Plotly 圖表範例

```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Bar(x=['A', 'B', 'C'], y=[1, 2, 3]))
fig.update_layout(title="範例圖表", height=400)
st.plotly_chart(fig, use_container_width=True)
```

## 快取機制

使用 `@st.cache_data` 避免重複載入：

```python
@st.cache_data(ttl=60)
def load_experiments():
    # ... 載入資料
    return data
```

## 除錯模式

```bash
# 啟用除錯模式
streamlit run ui/Home.py --logger.level=debug

# 清除快取
streamlit cache clear
```

## 參考資源

- [Streamlit 文件](https://docs.streamlit.io)
- [Plotly 文件](https://plotly.com/python/)
- UI 設計規範：`~/.claude/skills/ui/SKILL.md`
