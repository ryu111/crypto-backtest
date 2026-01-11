# 策略基礎架構

交易策略的基礎類別、註冊表和管理工具。

## 架構概覽

```
src/strategies/
├── __init__.py       # 模組入口
├── base.py           # 策略基礎類別
├── registry.py       # 策略註冊表
├── trend/            # 趨勢策略
├── momentum/         # 動量策略
└── mean_reversion/   # 均值回歸策略
```

## 核心元件

### 1. BaseStrategy（基礎抽象類別）

所有策略必須繼承的基礎類別。

**必須實作的方法**：
- `calculate_indicators(data)` - 計算指標
- `generate_signals(data)` - 產生交易訊號

**內建功能**：
- `position_size()` - 部位大小計算
- `validate_params()` - 參數驗證
- `get_info()` - 取得策略資訊
- `apply_filters()` - 應用過濾器

### 2. 策略類型類別

提供特定策略類型的輔助方法：

| 類別 | 策略類型 | 內建方法 |
|------|----------|----------|
| `TrendStrategy` | 趨勢跟隨 | `apply_trend_filter()` |
| `MomentumStrategy` | 動量策略 | `calculate_rsi()`, `calculate_macd()` |
| `MeanReversionStrategy` | 均值回歸 | `calculate_bollinger_bands()` |

### 3. StrategyRegistry（註冊表）

統一管理所有策略。

**主要功能**：
- 策略註冊與查詢
- 參數空間管理
- 策略實例化
- 類型過濾

## 快速開始

### 1. 建立策略

```python
from src.strategies import TrendStrategy, register_strategy

@register_strategy('my_ma_cross')
class MyMACross(TrendStrategy):
    """我的均線交叉策略"""

    params = {
        'fast_period': 10,
        'slow_period': 30,
    }

    param_space = {
        'fast_period': {'type': 'int', 'low': 5, 'high': 20},
        'slow_period': {'type': 'int', 'low': 20, 'high': 100},
    }

    version = "1.0"
    description = "Simple MA crossover"

    def calculate_indicators(self, data):
        close = data['close']
        return {
            'sma_fast': close.rolling(self.params['fast_period']).mean(),
            'sma_slow': close.rolling(self.params['slow_period']).mean()
        }

    def generate_signals(self, data):
        ind = self.calculate_indicators(data)
        fast = ind['sma_fast']
        slow = ind['sma_slow']

        long_entry = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        long_exit = (fast < slow) & (fast.shift(1) >= slow.shift(1))

        return long_entry, long_exit, None, None
```

### 2. 使用策略

```python
from src.strategies import create_strategy

# 建立策略實例
strategy = create_strategy('my_ma_cross', fast_period=12, slow_period=26)

# 產生交易訊號
long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(data)

# 計算部位大小
size = strategy.position_size(
    capital=10000,
    entry_price=50000,
    stop_loss_price=49000,
    risk_per_trade=0.02
)
```

### 3. 查詢策略

```python
from src.strategies import list_strategies, get_strategy, StrategyRegistry

# 列出所有策略
all_strategies = list_strategies()

# 取得策略類別
strategy_class = get_strategy('my_ma_cross')

# 取得策略資訊
info = StrategyRegistry.get_info('my_ma_cross')

# 取得參數空間
param_space = StrategyRegistry.get_param_space('my_ma_cross')

# 按類型過濾
trend_strategies = StrategyRegistry.list_by_type('trend')
```

## 策略開發指南

### 策略結構

```python
class MyStrategy(BaseStrategy):
    # 1. 元資料
    name = "my_strategy"
    strategy_type = "trend"  # trend, momentum, mean_reversion
    version = "1.0"
    description = "策略描述"

    # 2. 預設參數
    params = {
        'param1': 10,
        'param2': 20,
    }

    # 3. 參數優化空間
    param_space = {
        'param1': {'type': 'int', 'low': 5, 'high': 20},
        'param2': {'type': 'float', 'low': 1.0, 'high': 3.0},
    }

    # 4. 指標計算
    def calculate_indicators(self, data):
        # 計算所需指標
        return {'indicator_name': series}

    # 5. 訊號產生
    def generate_signals(self, data):
        # 產生進出場訊號
        return long_entry, long_exit, short_entry, short_exit

    # 6. 參數驗證（可選）
    def validate_params(self):
        # 自訂驗證邏輯
        return True
```

### 參數空間定義

用於參數優化（Optuna）：

```python
param_space = {
    # 整數參數
    'period': {
        'type': 'int',
        'low': 10,
        'high': 100
    },

    # 浮點數參數
    'threshold': {
        'type': 'float',
        'low': 0.5,
        'high': 2.0
    },

    # 類別參數
    'mode': {
        'type': 'categorical',
        'choices': ['aggressive', 'moderate', 'conservative']
    }
}
```

### 訊號格式

所有訊號都是 boolean Series：

```python
# 進場訊號：True 表示進場
long_entry = pd.Series([False, True, False, ...])

# 出場訊號：True 表示出場
long_exit = pd.Series([False, False, True, ...])

# 不使用的訊號可以回傳 None 或 False Series
short_entry = None  # 不做空
```

## 最佳實踐

### 1. 避免前瞻偏差

```python
# 錯誤：使用未來資料
ma = data['close'].rolling(10).mean()
long_entry = data['close'] > ma

# 正確：使用 shift 避免前瞻
ma = data['close'].rolling(10).mean()
long_entry = data['close'] > ma.shift(1)
```

### 2. 參數數量限制

建議每個策略不超過 5 個參數，避免過度擬合。

### 3. 包含止損機制

```python
def generate_signals(self, data):
    # ... 進場邏輯

    # 計算 ATR 止損
    atr = calculate_atr(data, period=14)
    stop_loss = data['close'] - atr * self.params['stop_loss_atr']

    # 止損出場
    long_exit = data['low'] < stop_loss

    return long_entry, long_exit, short_entry, short_exit
```

### 4. 策略命名規範

```
{type}_{indicator}_{timeframe}_{version}

範例：
- trend_ma_cross_4h_v1
- momentum_rsi_1h_v2
- breakout_bb_4h_v1
```

## 進階功能

### 1. 使用過濾器

```python
def apply_filters(self, data, long_entry, long_exit, short_entry, short_exit):
    # 趨勢過濾
    uptrend, downtrend = self.apply_trend_filter(data)

    # 只在上升趨勢做多
    long_entry = long_entry & uptrend

    # 只在下降趨勢做空
    short_entry = short_entry & downtrend

    return long_entry, long_exit, short_entry, short_exit
```

### 2. 動態部位大小

```python
def position_size(self, capital, entry_price, stop_loss_price, risk_per_trade=0.02):
    # 基於波動率調整風險
    volatility = self.calculate_volatility(data)
    adjusted_risk = risk_per_trade * (1 - volatility)

    return super().position_size(capital, entry_price, stop_loss_price, adjusted_risk)
```

## 測試

執行測試：

```bash
# 基礎架構測試
python test_strategy_architecture.py

# 使用範例
python examples/strategy_usage_example.py
```

## 相關文件

- 策略開發指南：`.claude/skills/策略開發/SKILL.md`
- 策略範本：`.claude/skills/策略開發/templates/base_strategy.py`
- 參數優化：`.claude/skills/參數優化/SKILL.md`
- 策略驗證：`.claude/skills/策略驗證/SKILL.md`

## 範例策略

完整範例請參考：
- `examples/strategy_usage_example.py` - 基本使用範例
- `test_strategy_architecture.py` - 測試範例
