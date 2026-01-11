# DataCleaner 使用指南

## 概述

`DataCleaner` 是一個專為永續合約回測系統設計的資料清理模組，用於處理資料缺失、異常值和品質問題。

## 主要功能

### 1. 資料缺失偵測

自動偵測時間序列中的缺失區間：

- **短期缺失** (< 1小時)：使用線性插值填補
- **長期缺失** (> 1小時)：標記為 NaN，不填補
- **維護期間**：自動識別（凌晨 0-4 點的長時間缺失）

### 2. 資料品質分析

提供詳細的資料品質報告：

- 缺失率統計
- 缺失區間列表
- OHLC 邏輯驗證
- 成交量異常檢查
- 品質評分（0-100）

### 3. 自動清理

一鍵完成所有清理工作：

- 移除重複時間戳
- 填補短期缺失
- 標記長期缺失
- 驗證資料邏輯

## 快速開始

### 基本使用

```python
from src.data import DataFetcher, DataCleaner

# 1. 獲取資料
fetcher = DataFetcher()
df = fetcher.fetch_ohlcv('BTCUSDT', '4h')

# 2. 建立清理器
cleaner = DataCleaner(timeframe='4h', verbose=True)

# 3. 分析品質
report = cleaner.analyze_quality(df)
print(report)

# 4. 執行清理
df_cleaned = cleaner.clean(df)
```

### 進階使用

#### 自定義填補策略

```python
# 只分析，不填補
report = cleaner.analyze_quality(df)

# 只填補短期缺失
df_short = cleaner.clean(df, fill_short_gaps=True, mark_long_gaps=False)

# 只標記，不填補
df_marked = cleaner.clean(df, fill_short_gaps=False, mark_long_gaps=True)
```

#### 調整參數

```python
# 更嚴格的 gap 偵測（閾值倍數降低）
cleaner = DataCleaner(
    timeframe='4h',
    gap_threshold_multiplier=1.5,  # 預設 2.0
    long_gap_hours=0.5,            # 預設 1.0
    verbose=True
)
```

## 輸出格式

### DataQualityReport

```python
DataQualityReport(
    total_records=1000,           # 總筆數
    missing_count=10,              # 缺失筆數
    missing_rate=0.01,             # 缺失率（1%）
    gap_count=2,                   # Gap 數量
    gaps=[...],                    # Gap 詳細資訊
    quality_score=95.5,            # 品質評分
    issues=['問題描述']            # 問題列表
)
```

### GapInfo

```python
GapInfo(
    start_time=datetime(...),      # 缺失開始時間
    end_time=datetime(...),        # 缺失結束時間
    duration=timedelta(hours=8),   # 缺失時長
    gap_size=2,                    # 缺失筆數
    is_maintenance=False,          # 是否為維護期間
    fill_strategy=GapFillStrategy.LINEAR  # 填補策略
)
```

### 清理後資料

```python
# 清理後的 DataFrame 包含原始欄位 + gap_flag
df_cleaned.columns
# ['open', 'high', 'low', 'close', 'volume', 'gap_flag']

# gap_flag 值
# 0: 正常資料
# 1: 短期缺失（已填補）
# 2: 長期缺失（未填補）
# 3: 維護期間
```

## 品質評分計算

品質評分（0-100）由三個因素組成：

```
品質評分 = 缺失率分數 × 40% + Gap分數 × 30% + 問題分數 × 30%
```

- **缺失率分數**：缺失率越低越好
- **Gap 分數**：Gap 數量佔總數的比例越低越好
- **問題分數**：每個問題扣 10 分

### 評分參考

| 評分 | 等級 | 說明 |
|------|------|------|
| 90-100 | 優秀 | 資料品質極佳，可直接使用 |
| 70-89 | 良好 | 資料品質良好，建議清理後使用 |
| 50-69 | 中等 | 資料品質一般，必須清理 |
| < 50 | 較差 | 資料品質較差，需仔細檢查 |

## 最佳實踐

### 1. 定期品質檢查

```python
# 下載資料後立即檢查
df = fetcher.fetch_ohlcv('BTCUSDT', '4h')
report = cleaner.analyze_quality(df)

if report.quality_score < 70:
    print(f"⚠ 資料品質較差：{report.quality_score:.2f}")
    print(report)
```

### 2. 保留原始資料

```python
# 清理前先備份
df_original = df.copy()

# 清理
df_cleaned = cleaner.clean(df)

# 對比
print(f"原始: {len(df_original)}, 清理後: {len(df_cleaned)}")
```

### 3. 使用 gap_flag 過濾

```python
# 只使用正常資料
df_normal = df_cleaned[df_cleaned['gap_flag'] == 0]

# 排除維護期間
df_no_maintenance = df_cleaned[df_cleaned['gap_flag'] != 3]
```

### 4. 批次處理多個交易對

```python
symbols = ['BTCUSDT', 'ETHUSDT']
reports = {}

for symbol in symbols:
    df = fetcher.fetch_ohlcv(symbol, '4h')
    df_cleaned = cleaner.clean(df)
    reports[symbol] = cleaner.analyze_quality(df_cleaned)

# 查看所有報告
for symbol, report in reports.items():
    print(f"\n{symbol}: 品質評分 {report.quality_score:.2f}")
```

## 效能優化

### 使用 Polars（可選）

安裝 Polars 以獲得更好的效能：

```bash
pip install polars>=0.20.0
```

`DataCleaner` 會自動偵測並使用 Polars 後端（如果可用）。

### 大數據處理

```python
# 分批處理
chunk_size = 10000
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    chunk_cleaned = cleaner.clean(chunk)
    # 處理 chunk_cleaned
```

## 常見問題

### Q1: 為什麼清理後資料筆數增加了？

A: 短期缺失（< 1小時）會被自動填補，因此筆數會增加。可以檢查 `gap_flag == 1` 的資料。

### Q2: 如何判斷是否需要清理？

A: 查看品質評分。通常評分 < 70 建議清理，< 50 必須清理。

### Q3: 長期缺失為什麼不填補？

A: 長期缺失（> 1小時）通常表示交易所維護或異常事件，填補會產生不準確的資料，建議標記並在回測時跳過。

### Q4: 如何自定義填補策略？

A: 可以繼承 `DataCleaner` 並覆寫 `_fill_gaps` 方法：

```python
class CustomCleaner(DataCleaner):
    def _fill_gaps(self, df, gaps, only_short=True):
        # 自定義填補邏輯
        return df
```

## 相關檔案

- 原始碼：`src/data/cleaner.py`
- 測試：`tests/test_data_cleaner.py`
- 範例：`examples/test_data_cleaner.py`

## 參考資料

- [Pandas 時間序列處理](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [Polars 效能優化](https://pola-rs.github.io/polars-book/)
- [資料品質最佳實踐](https://en.wikipedia.org/wiki/Data_quality)
