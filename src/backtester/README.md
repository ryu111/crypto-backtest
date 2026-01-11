# 並行回測系統

利用 Apple M4 Max 16 核心進行高效能並行回測。

## 功能特性

### 1. 多核心並行執行
- 自動利用所有 CPU 核心（可自訂）
- ProcessPoolExecutor 實現真正並行
- 進度追蹤和錯誤處理

### 2. 參數掃描 (Parameter Sweep)
```python
from src.backtester.parallel import run_parameter_sweep

param_grid = {
    'period': [10, 20, 30],
    'threshold': [0.01, 0.02, 0.03]
}

results = run_parameter_sweep(
    data=your_data,
    strategy_fn=your_strategy,
    param_grid=param_grid,
    n_workers=8,  # 使用 8 核心
    progress=True
)
```

### 3. 多策略比較
```python
from src.backtester.parallel import run_multi_strategy

strategies = [
    trend_following_strategy,
    mean_reversion_strategy,
    momentum_strategy
]

results = run_multi_strategy(
    data=your_data,
    strategies=strategies,
    n_workers=4
)
```

### 4. 自訂並行任務
```python
from src.backtester.parallel import ParallelBacktester, ParallelTask

backtester = ParallelBacktester(n_workers=8)

tasks = [
    ParallelTask(
        task_id=f"task_{i}",
        params={'data': data, 'param1': value},
        strategy_name="my_strategy"
    )
    for i in range(100)
]

results = backtester.run_parallel(tasks, my_backtest_fn)
```

## 重要注意事項

### multiprocessing 限制

由於使用 `multiprocessing.ProcessPoolExecutor`，所有函數**必須**定義在模組頂層：

```python
# ✅ 正確：頂層函數
def my_strategy(data, period):
    return calculate_profit(data, period)

# ❌ 錯誤：局部函數（無法序列化）
def main():
    def my_strategy(data, period):  # 這會失敗！
        return calculate_profit(data, period)
```

### 策略函數簽名

#### 參數掃描策略
```python
def my_strategy(data, param1, param2, ...):
    """接收 data 和所有參數"""
    return result
```

#### 多策略回測
```python
def my_strategy(data):
    """只接收 data"""
    return result
```

#### 自訂任務
```python
def my_backtest_fn(task: ParallelTask):
    """接收 ParallelTask 物件"""
    params = task.params
    return result
```

## 效能考量

### 何時使用並行？

✅ **適合並行**：
- 參數掃描（> 10 組參數）
- 多策略比較（> 3 個策略）
- 長時間回測（> 1 秒/次）

❌ **不適合並行**：
- 快速計算（< 0.1 秒/次）
- 少量任務（< 5 個）
- multiprocessing overhead 會降低效能

### 最佳化建議

1. **調整工作程序數量**
   ```python
   # M4 Max 16 核心建議：
   n_workers = 8  # 留一半核心給系統
   ```

2. **批次處理**
   ```python
   # 將大量參數分批處理
   for batch in batches(all_params, batch_size=100):
       results = run_parameter_sweep(data, strategy, batch)
   ```

3. **進度追蹤**
   ```python
   def progress_callback(completed, total):
       print(f"Progress: {completed}/{total}")

   backtester.run_parallel(tasks, fn, progress_callback)
   ```

## 範例

完整範例請參考：`examples/parallel_example.py`

```bash
python examples/parallel_example.py
```

範例包含：
1. 參數掃描（5×5 = 25 組參數）
2. 多策略比較（4 個策略）
3. 自訂並行任務（12 個任務）
4. 錯誤處理示範

## API 參考

### ParallelBacktester

主要類別，負責並行執行管理。

```python
backtester = ParallelBacktester(
    n_workers=None,      # CPU 核心數（預設：全部）
    backend='concurrent'  # 'concurrent' 或 'multiprocessing'
)
```

### ParallelTask

任務定義。

```python
task = ParallelTask(
    task_id="unique_id",
    params={'data': data, 'param1': value},
    strategy_name="strategy_name"
)
```

### ParallelResult

執行結果。

```python
result = ParallelResult(
    task_id="unique_id",
    result=output,           # 策略回傳值
    execution_time=1.23,     # 執行時間（秒）
    worker_id=12345,         # 工作程序 PID
    success=True,            # 是否成功
    error_message=None       # 錯誤訊息（失敗時）
)
```

## 測試

執行測試：

```bash
pytest tests/test_parallel.py -v
```

測試涵蓋：
- 基本並行執行
- 參數掃描
- 多策略回測
- 錯誤處理
- 效能驗證

## 未來改進

- [ ] 支援 Ray 分散式執行
- [ ] 結果快取機制
- [ ] 更細緻的資源管理
- [ ] GPU 加速整合
- [ ] 動態負載平衡
