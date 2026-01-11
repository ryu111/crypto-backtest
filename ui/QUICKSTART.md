# 快速開始 - Dashboard UI

## 1. 安裝依賴

```bash
pip install -r ui/requirements.txt
```

或手動安裝：

```bash
pip install streamlit plotly pandas
```

## 2. 啟動 UI

### 方法一：使用啟動腳本（推薦）

```bash
./ui/start.sh
```

### 方法二：直接啟動

```bash
streamlit run ui/Home.py
```

## 3. 瀏覽器訪問

打開瀏覽器訪問：

```
http://localhost:8501
```

## 頁面導覽

- **Home**: 系統總覽、功能介紹
- **Dashboard**: 實驗統計、績效分析、排行榜

## 產生測試資料

如果看到「尚未記錄任何實驗」，請執行：

```bash
python examples/learning/record_experiment.py
```

這會產生測試資料到 `learning/experiments.json`

## 常見問題

### Q: 畫面空白或顯示「尚無資料」

A: 需要先執行回測並記錄實驗：

```bash
cd examples/learning
python record_experiment.py
```

### Q: 模組找不到

A: 確認已安裝所有依賴：

```bash
pip install streamlit plotly pandas
```

### Q: Port 衝突

A: 使用其他 port：

```bash
streamlit run ui/Home.py --server.port 8502
```

## 進階配置

### 自訂主題

建立 `.streamlit/config.toml`：

```toml
[theme]
primaryColor = "#3b82f6"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f9fafb"
textColor = "#111827"
font = "sans serif"
```

### 效能優化

```bash
# 增加快取時間
STREAMLIT_CACHE_TTL=300 streamlit run ui/Home.py
```

## 下一步

- 查看完整文件：`ui/README.md`
- 學習系統：`src/learning/README.md`
- 策略開發：`.claude/skills/策略開發/SKILL.md`
