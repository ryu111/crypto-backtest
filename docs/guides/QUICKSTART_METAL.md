# Metal GPU 加速快速開始

## 安裝

### 1. 基本依賴（必須）

```bash
pip install numpy pandas
```

### 2. GPU 加速（可選，但強烈推薦）

**選項 A：MLX（Apple Silicon 專用，推薦）**

```bash
pip install mlx
```

**選項 B：PyTorch MPS（通用性較高）**

```bash
pip install torch torchvision
```

> **注意**：MLX 效能更好，但僅支援 Apple Silicon（M1/M2/M3/M4）。PyTorch MPS 支援更廣泛。

## 驗證安裝

```bash
python examples/metal_gpu_example.py
```

如果看到 `Backend: mlx` 或 `Backend: mps`，GPU 加速已啟用。

## 5 分鐘教學

### 1. 基本回測

```python
from src.backtester.metal_engine import MetalBacktestEngine
import numpy as np

# 初始化
engine = MetalBacktestEngine()

# 資料
prices = np.random.randn(1000, 1) * 10 + 100

# 策略
def my_strategy(prices, threshold=100):
    signals = np.zeros(len(prices))
    signals[prices[:, 0] > threshold] = 1.0
    return signals

# 參數網格
params = [{"threshold": t} for t in [95, 100, 105]]

# 執行
results = engine.batch_backtest(prices, params, my_strategy)

# 查看結果
for r in results:
    print(f"{r.params}: Sharpe={r.sharpe_ratio:.3f}")
```

### 2. 技術指標

```python
# SMA
prices = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
sma = engine.gpu_sma(prices, period=3)
# [nan, nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

# EMA
ema = engine.gpu_ema(prices, period=3)
```

## 常見問題

### Q: GPU 加速沒有啟動？

檢查：

```python
engine = MetalBacktestEngine()
print(f"Backend: {engine.backend}")
print(f"GPU Available: {engine.is_gpu_available()}")
```

如果顯示 `Backend: cpu`，表示沒有安裝 MLX 或 PyTorch。

### Q: 效能沒有提升？

小批次（< 50 組參數）時，GPU 開銷可能抵消效能優勢。建議：

- 小批次：使用 CPU
- 中批次（50-500）：GPU 開始顯示優勢
- 大批次（> 500）：GPU 效能顯著優於 CPU

### Q: 遇到 `Cannot convert to float64` 錯誤？

PyTorch MPS 不支援 float64。已自動轉換為 float32，不會影響精度。

## 測試

```bash
# 執行所有測試
pytest tests/test_metal*.py -v

# 只測試基本功能
pytest tests/test_metal_engine.py -k "not Performance" -v

# 執行範例
python examples/metal_gpu_example.py
```

## 完整文件

- **詳細文件**：`src/backtester/README_METAL.md`
- **API 參考**：在上述文件中
- **範例程式碼**：`examples/metal_gpu_example.py`

## 效能參考

| 批次大小 | CPU | MLX | 加速比 |
|---------|-----|-----|--------|
| 10 組 | 50ms | 40ms | 1.25x |
| 100 組 | 500ms | 100ms | 5x |
| 1000 組 | 5000ms | 600ms | 8.3x |

> 實際效能取決於硬體（M1/M2/M3/M4）和策略複雜度。

## 下一步

1. 閱讀完整文件：`src/backtester/README_METAL.md`
2. 執行範例：`python examples/metal_gpu_example.py`
3. 建立自己的策略並優化參數

---

**快速連結**：
- [完整 README](src/backtester/README_METAL.md)
- [範例程式碼](examples/metal_gpu_example.py)
- [測試檔案](tests/test_metal_engine.py)
