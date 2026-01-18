# Integrate GPLoop into UltimateLoop

## Why

目前 `UltimateLoop` 的 Explore 模式只能從 `StrategyRegistry` 中**隨機選擇已註冊策略**，無法**生成新策略**。這限制了策略探索的空間。

`GPLoop` 提供了獨特的策略自動生成能力：
- 從技術指標原語自動組合出新的交易規則
- 表達式樹演化產生人類難以想到的策略組合
- 已有完整的 `EvolvedStrategy` 基類和策略生成器

**整合價值**：
1. Explore 模式可以調用 GPLoop 生成全新策略
2. GP 生成的策略可納入 StrategyRegistry 供後續 Exploit 使用
3. GP 策略經過完整 5 階段驗證，確保品質
4. 學習系統記錄 GP 策略的演化過程和表現

## What Changes

### 新增模組

1. **`src/automation/gp_integration.py`**
   - `GPExplorer` 類別 - 封裝 GPLoop 供 UltimateLoop 調用
   - `GPStrategyAdapter` - 將 GP 生成策略適配到 StrategyRegistry

### 修改模組

2. **`src/automation/ultimate_loop.py`**
   - `_select_strategies()` - 新增 GP explore 分支
   - `_explore_with_gp()` - 新方法，調用 GPExplorer
   - `_register_gp_strategies()` - 將 GP 策略註冊到 Registry

3. **`src/automation/ultimate_config.py`**
   - 新增 GP 相關配置參數：
     - `gp_explore_enabled: bool = True`
     - `gp_explore_ratio: float = 0.2` (explore 時 20% 機率用 GP)
     - `gp_population_size: int = 50`
     - `gp_generations: int = 30`

4. **`src/strategies/registry.py`**
   - `register_dynamic()` - 新方法，支援運行時動態註冊策略

## Impact

### Affected Specs
- `gp-strategy-generation` - 已有 GP 生成規格，需擴展整合介面

### Affected Code
- `src/automation/ultimate_loop.py` - 主要修改
- `src/automation/ultimate_config.py` - 配置擴展
- `src/strategies/registry.py` - 動態註冊
- `src/automation/gp_loop.py` - 可能需要小修改以支援外部調用

### Breaking Changes
**無破壞性變更** - 所有新功能都是可選的：
- `gp_explore_enabled=False` 維持原有行為
- 現有策略和回測流程不受影響

## Data Contracts（資料契約）

### GPExplorer -> UltimateLoop

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class GPExplorationResult:
    """GP 探索結果"""
    success: bool                    # 是否成功生成策略
    strategies: List[str]            # 生成的策略名稱列表（已註冊）
    best_fitness: Optional[float]    # 最佳適應度分數
    generations_run: int             # 實際執行的演化代數
    error: Optional[str]             # 錯誤訊息（如失敗）
```

### UltimateLoop -> GPExplorer

```python
@dataclass
class GPExplorationRequest:
    """GP 探索請求"""
    symbol: str                      # 交易標的
    timeframe: str                   # 時間框架
    population_size: int             # 種群大小
    generations: int                 # 演化代數
    top_n: int = 3                   # 取前 N 個最佳策略
    market_state: Optional[Any] = None  # 市場狀態（用於適應度調整）
```

### StrategyRegistry 動態註冊介面

```python
@dataclass
class DynamicStrategyInfo:
    """動態策略資訊"""
    name: str                        # 策略唯一名稱
    strategy_class: Type[BaseStrategy]  # 策略類別
    source: str = "gp"               # 來源標記
    metadata: Optional[Dict] = None  # 額外元資料
```

## Expected Behavior

### Explore 模式流程

```
UltimateLoop._select_strategies(market_state)
    ↓
策略選擇模式 = 'regime_aware' 或 'exploit'
    ↓
if random() < explore_ratio:
    if random() < gp_explore_ratio:
        # 使用 GP 生成新策略
        result = GPExplorer.explore(request)
        register_gp_strategies(result.strategies)
        return result.strategies
    else:
        # 傳統隨機探索
        return random_select()
else:
    # Exploit 模式
    return select_best_strategies()
```

### GP 策略生命週期

```
1. GPLoop 演化 → 產生 Hall of Fame
2. StrategyGenerator 轉換 → Python 策略類別
3. StrategyRegistry.register_dynamic() → 動態註冊
4. 策略納入 available_strategies → 可被選擇
5. 經過 5 階段驗證 → 品質確認
6. 記錄到 Learning 系統 → 累積經驗
7. 高分策略被 Exploit → 持續優化
```

## Non-Goals

- **不修改** GPLoop 核心演化邏輯
- **不修改** 現有 12 個手寫策略
- **不修改** BacktestEngine API
- **不替換** 現有 Explore 機制，只是擴展

## Technical Design

詳見 `design.md`
