# GP 演化策略模組

本模組包含由 Genetic Programming (GP) 自動演化生成的交易策略。

## 目錄結構

```
src/strategies/gp/
├── __init__.py                    # 模組初始化
├── evolved_strategy.py            # GP 演化策略基類
├── generated/                     # 自動生成的策略檔案
│   ├── __init__.py
│   ├── evolved_001.py
│   ├── evolved_002.py
│   └── ...
└── README.md                      # 本檔案
```

## 核心組件

### EvolvedStrategy 基類

所有 GP 演化策略的基類，提供：
- 從 GP 表達式樹生成交易訊號
- 演化元資料（適應度、代數等）
- 與 BaseStrategy 的完整整合

```python
from src.strategies.gp import EvolvedStrategy

# 繼承 EvolvedStrategy 建立新策略
class MyEvolvedStrategy(EvolvedStrategy):
    expression = "gt(rsi(close, 14), 70)"
    fitness_score = 1.85
    generation = 100

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._signal_func = self._build_signal_func()

    def _build_signal_func(self):
        from src.gp.primitives import gt, rsi

        def signal_func(close, high, low):
            return gt(rsi(close, 14), 70)

        return signal_func
```

## 使用流程

### 1. GP 演化生成策略

使用 GPEngine 演化出最佳個體：

```python
from src.gp import GPEngine, EvolutionConfig, ExpressionConverter, StrategyGenerator

# 建立並執行 GP 演化
engine = GPEngine(config=EvolutionConfig())
result = engine.evolve(data=backtest_data, generations=100)

# 取得最佳個體
best_individual = result.best_individual
```

### 2. 轉換為策略檔案

使用 ExpressionConverter 和 StrategyGenerator：

```python
from src.gp import ExpressionConverter, StrategyGenerator, PrimitiveSetFactory

# 建立 PrimitiveSet 和 Converter
factory = PrimitiveSetFactory()
pset = factory.create_standard_set()
converter = ExpressionConverter(pset)

# 生成策略檔案
generator = StrategyGenerator(converter)
file_path = generator.generate(
    individual=best_individual,
    strategy_name="evolved_rsi_ma_001",
    fitness=result.best_fitness,
    metadata={'generation': result.generation}
)

print(f"策略檔案已生成: {file_path}")
```

### 3. Import 並使用策略

```python
from src.strategies.gp.generated.evolved_rsi_ma_001 import EvolvedRsiMa001

# 實例化策略
strategy = EvolvedRsiMa001()

# 查看策略資訊
print(f"策略名稱: {strategy.name}")
print(f"適應度: {strategy.fitness_score}")
print(f"表達式: {strategy.expression}")

# 生成交易訊號
import pandas as pd
data = pd.read_csv('data/BTCUSDT.csv')
long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(data)
```

### 4. 執行回測

與其他策略一樣，可以直接傳遞給 BacktestEngine：

```python
from src.backtest import BacktestEngine

engine = BacktestEngine(strategy=strategy)
result = engine.run(data=data, initial_capital=10000)

print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Total Return: {result.total_return:.2%}")
```

## 生成的策略檔案格式

每個生成的策略檔案包含：

```python
"""
GP 演化策略: evolved_rsi_ma_001

自動生成於: 2026-01-17T12:00:00.000000Z
適應度: 1.8500
表達式: gt(rsi(close, 14), 70)
"""

from src.strategies.gp.evolved_strategy import EvolvedStrategy
from src.gp.primitives import *
import numpy as np
import pandas as pd


class EvolvedRsiMa001(EvolvedStrategy):
    """GP 演化策略"""

    name = "evolved_rsi_ma_001"
    version = "1.0"
    description = "GP evolved strategy with fitness 1.8500"

    # 演化元資料
    expression = "gt(rsi(close, 14), 70)"
    fitness_score = 1.85
    generation = 100
    evolved_at = "2026-01-17T12:00:00.000000Z"

    def _build_signal_func(self):
        """建立訊號函數"""
        def signal_func(close, high, low):
            return gt(rsi(close, 14), 70)
        return signal_func
```

## 元資料欄位

每個演化策略包含以下元資料：

| 欄位 | 類型 | 說明 |
|------|------|------|
| `expression` | str | GP 表達式字串 |
| `fitness_score` | float | 適應度分數（如 Sharpe Ratio） |
| `generation` | int | 演化代數 |
| `evolved_at` | str | 生成時間（ISO 格式） |

## 注意事項

### 1. 生成的策略是獨立的

每個生成的策略檔案都是完全獨立的，不依賴 GP 引擎。可以：
- 直接複製到其他專案
- 手動修改表達式
- 作為新策略的基礎

### 2. 表達式使用的原語

生成的策略使用 `src/gp/primitives` 中定義的原語：

```python
from src.gp.primitives import *
```

確保這些原語可用：
- 指標：rsi, ma, ema, atr, macd, bb_upper, bb_lower
- 比較：gt, lt, cross_above, cross_below
- 邏輯：and_, or_, not_
- 數學：add, sub, mul, protected_div, protected_log

### 3. 訊號生成邏輯

生成的策略遵循以下規則：
- `long_entry = True` 當 GP 表達式為 True
- `long_exit = True` 當 GP 表達式為 False
- `short_entry = False` 不做空
- `short_exit = False` 不做空

如需修改這個邏輯，可以覆寫 `generate_signals` 方法。

## 範例

完整範例請參考：
- `examples/phase5_complete_example.py` - Phase 5 完整示範
- `tests/test_gp_converter.py` - 單元測試

## 相關模組

- `src/gp/converter.py` - 表達式轉換器
- `src/gp/primitives.py` - GP 原語定義
- `src/gp/engine.py` - GP 演化引擎
- `src/strategies/base.py` - 策略基類
