# Code Smells 分析報告

> 分析日期：2026-01-12
> 分析工具：基於 Martin Fowler 重構技術

## 概覽

| 檔案 | 行數 | 嚴重性 | 主要問題 |
|------|------|--------|----------|
| recorder.py | 952 | 🔴 高 | Large Class, Duplicate Code |
| orchestrator.py | 851 | 🟠 中高 | God Class, Long Method |
| stages.py | 792 | 🟠 中 | Duplicate Code, Long Method |
| engine.py | 633 | 🟢 低 | Duplicate Code (可接受) |

---

## 1. recorder.py (952 lines) - 🔴 高優先級

### Large Class (過大類別)
`ExperimentRecorder` 承擔了太多職責：

```
目前職責:
├── 實驗記錄 (log_experiment)
├── 實驗查詢 (query_experiments, get_best_experiments)
├── 演進追蹤 (get_strategy_evolution)
├── 洞察更新 (update_insights)
├── 標籤產生 (generate_tags)
├── 時間序列儲存 (_save_timeseries_data)
├── 時間序列載入 (load_equity_curve, load_daily_returns, load_trades)
└── 洞察文件解析 (_update_*_insights 方法)
```

**建議**: Extract Class
- `ExperimentRecorder` - 只負責記錄和查詢
- `InsightsManager` - 負責洞察文件更新
- `TimeSeriesStorage` - 負責時間序列資料

### Duplicate Code (重複程式碼)
`_update_trend_insights` 和 `_update_momentum_insights` 幾乎相同：

```python
# _update_trend_insights (L691-724)
section = "### 趨勢跟隨策略\n"
if section in content:
    start = content.find(section)
    next_section = content.find('\n### ', start + len(section))
    # ... 完全相同的邏輯

# _update_momentum_insights (L726-750)
section = "### 動量策略\n"
if section in content:
    start = content.find(section)
    next_section = content.find('\n### ', start + len(section))
    # ... 完全相同的邏輯
```

**建議**: Extract Method → `_update_section_insights(content, section_name, experiment)`

### Long Method (過長方法)
- `log_experiment` (L197-297): 100 行
- `query_experiments` (L317-396): 80 行

**建議**: Extract Method - 拆分過濾邏輯

---

## 2. orchestrator.py (851 lines) - 🟠 中高優先級

### God Class (上帝類別)
`Orchestrator` 做了太多事情：

```
目前職責:
├── 策略選擇 (_select_strategy)
├── 參數空間生成 (_generate_param_space)
├── 優化執行 (_optimize)
├── 驗證執行 (_validate)
├── 價值判斷 (_should_record)
├── 記錄到 JSON (_record)
├── 記錄到 Memory (_record_to_memory)
└── 統計更新 (_update_loop_summary)
```

**建議**: 使用 Strategy Pattern 或 Command Pattern
- 每個步驟可以是獨立的處理器
- Orchestrator 只負責協調

### Long Method (過長方法)
`run_iteration` (L209-351): 142 行，做了太多事情

**建議**: Extract Method - 拆分為更小的步驟方法

### Duplicate Code (重複程式碼)
`_optimize` 和 `_validate` 中重複建立 BacktestConfig：

```python
# _optimize (L544-554)
config = BacktestConfig(
    symbol=self.config['symbols'][0],
    timeframe=self.config['timeframes'][0],
    start_date=data.index[0],
    end_date=data.index[-1],
    initial_capital=self.config['initial_capital'],
    ...
)

# _validate (L603-613)
config = BacktestConfig(
    symbol='BTCUSDT',  # Hardcoded!
    timeframe=self.config['timeframes'][0],
    ...
)
```

**建議**: Extract Method → `_create_backtest_config(data)`

### Primitive Obsession (基本型別偏執)
`config` 使用 `Dict[str, Any]` 而非專用類別

**建議**: Replace Primitive with Object → `OrchestratorConfig` dataclass

---

## 3. stages.py (792 lines) - 🟠 中優先級

### Duplicate Code (重複程式碼)
所有 stage 方法都有相同的模式：

```python
def stageN_xxx(self, ...) -> StageResult:
    thresholds = self.thresholds['stageN']

    checks = {
        'check1': condition1,
        'check2': condition2,
        ...
    }

    passed = all(checks.values())
    score = sum(checks.values()) / len(checks) * 100

    details = {..., 'checks': checks}

    if passed:
        message = "通過訊息"
    else:
        failed = [k for k, v in checks.items() if not v]
        message = f"未通過: {', '.join(failed)}"

    return StageResult(passed, score, details, message, thresholds)
```

**建議**: Template Method Pattern
- 抽象出共同的評估流程
- 子類別只需實作 `_get_checks()` 和 `_get_thresholds()`

### Long Method (過長方法)
- `validate` (L137-231): 95 行，控制所有階段
- `_perform_walk_forward` (L559-626): 68 行

**建議**:
- 使用 Chain of Responsibility 或 Strategy Pattern
- 每個階段是獨立的 Validator 類別

### Shotgun Surgery (霰彈式修改)
添加新階段需要修改：
1. `_load_thresholds()`
2. `validate()`
3. `_calculate_grade()`
4. `_generate_recommendation()`

**建議**: 使用 Plugin 架構，新階段只需要實作介面

---

## 4. engine.py (633 lines) - 🟢 低優先級

### Duplicate Code (重複程式碼)
`_run_vectorized_pandas` 和 `_run_vectorized_polars` 有 70% 重複：

```python
# 兩個方法都有:
# 1. 產生訊號
# 2. 組合訊號
# 3. 計算部位
# 4. 計算損益
# 5. 建立 Portfolio
```

**建議**: Template Method
- 抽象出共同流程
- 子類別只需實作資料轉換部分

### Long Method (過長方法)
`_calculate_metrics` (L491-583): 92 行

**建議**: Extract Method
- `_calculate_basic_metrics()`
- `_calculate_trade_statistics()`
- `_calculate_advanced_metrics()`

---

## 重構優先級

| 優先級 | 檔案 | 重構 | 預期效果 |
|--------|------|------|----------|
| 1 | recorder.py | Extract Class | -300 行，職責清晰 |
| 2 | recorder.py | Extract Method (duplicate) | -50 行，減少重複 |
| 3 | orchestrator.py | Extract Method | 可讀性提升 |
| 4 | stages.py | Template Method | 擴展性提升 |
| 5 | engine.py | Template Method | -50 行，減少重複 |

---

## 重構風險評估

| 風險 | 評估 | 緩解措施 |
|------|------|----------|
| 破壞功能 | 中 | 確保測試覆蓋 |
| 引入新 bug | 低 | 小步驟 + 頻繁測試 |
| 影響其他模組 | 低 | 保持公開 API 不變 |
