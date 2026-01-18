# GP Integration Specification

## Overview

整合 GPLoop 策略生成能力到 UltimateLoop 的 Explore 模式。

---

## ADDED Requirements

### Requirement: GP Exploration Capability

The system SHALL provide a GP-based strategy exploration mechanism that integrates with UltimateLoop's Explore mode.

#### Scenario: GP Explore Triggered
- **GIVEN** UltimateLoop is running with `gp_explore_enabled=True`
- **AND** strategy selection enters Explore mode (based on `exploit_ratio`)
- **AND** random check passes GP threshold (based on `gp_explore_ratio`)
- **WHEN** `_select_strategies()` is called
- **THEN** the system SHALL invoke `GPExplorer.explore()` to generate new strategies
- **AND** the generated strategies SHALL be dynamically registered to StrategyRegistry
- **AND** the strategy names SHALL be returned for subsequent optimization

#### Scenario: GP Explore Disabled
- **GIVEN** UltimateLoop is running with `gp_explore_enabled=False`
- **WHEN** strategy selection enters Explore mode
- **THEN** the system SHALL use traditional random strategy selection
- **AND** GPExplorer SHALL NOT be invoked

#### Scenario: GP Explore Failure
- **GIVEN** GPExplorer.explore() fails (e.g., timeout, invalid data)
- **WHEN** GP exploration is triggered
- **THEN** the system SHALL log the error
- **AND** the system SHALL fallback to traditional random selection
- **AND** the UltimateLoop SHALL continue execution without interruption

---

### Requirement: Dynamic Strategy Registration

The system SHALL support runtime dynamic registration of GP-generated strategies to StrategyRegistry.

#### Scenario: Successful Dynamic Registration
- **GIVEN** a valid GP-evolved strategy class
- **AND** a unique strategy name
- **WHEN** `StrategyRegistry.register_dynamic()` is called
- **THEN** the strategy SHALL be added to `_strategies` dictionary
- **AND** the strategy info SHALL be recorded in `_dynamic_strategies`
- **AND** the strategy SHALL be available via `get()` and `list_all()`

#### Scenario: Name Collision Handling
- **GIVEN** a strategy name that already exists in StrategyRegistry
- **WHEN** `register_dynamic()` is called with the same name
- **THEN** the system SHALL raise a `ValueError`
- **AND** the existing strategy SHALL NOT be overwritten

#### Scenario: Dynamic Strategy Removal
- **GIVEN** a dynamically registered strategy
- **WHEN** `StrategyRegistry.unregister_dynamic()` is called
- **THEN** the strategy SHALL be removed from `_strategies`
- **AND** the strategy SHALL be removed from `_dynamic_strategies`
- **AND** the strategy SHALL no longer be available via `get()` or `list_all()`

#### Scenario: Static Strategy Protection
- **GIVEN** a statically registered strategy (via decorator)
- **WHEN** `unregister_dynamic()` is called with its name
- **THEN** the system SHALL return `False`
- **AND** the strategy SHALL NOT be removed

---

### Requirement: GP Strategy Lifecycle

GP-generated strategies SHALL follow a complete lifecycle from generation to learning.

#### Scenario: Full GP Strategy Lifecycle
- **GIVEN** GPLoop evolves a population
- **WHEN** evolution completes with valid Hall of Fame
- **THEN** the top N strategies SHALL be converted to strategy classes
- **AND** the strategies SHALL be dynamically registered
- **AND** the strategies SHALL be eligible for optimization
- **AND** the strategies SHALL undergo 5-stage validation
- **AND** the validation results SHALL be recorded to Learning system

#### Scenario: GP Strategy Metadata Recording
- **GIVEN** a GP-generated strategy passes validation
- **WHEN** the strategy is recorded to Learning system
- **THEN** the record SHALL include:
  - `expression`: original GP expression string
  - `fitness_score`: GP fitness value
  - `generation`: evolution generation number
  - `source`: "gp" marker

---

### Requirement: GP Configuration

The system SHALL provide configurable parameters for GP exploration.

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gp_explore_enabled` | bool | True | Enable GP exploration |
| `gp_explore_ratio` | float | 0.2 | Probability of using GP when exploring |
| `gp_population_size` | int | 50 | GP population size |
| `gp_generations` | int | 30 | GP evolution generations |
| `gp_top_n` | int | 3 | Number of top strategies to register |

#### Validation Rules
- `gp_explore_ratio` SHALL be in range [0.0, 1.0]
- `gp_population_size` SHALL be >= 10
- `gp_generations` SHALL be >= 5
- `gp_top_n` SHALL be >= 1 and <= `gp_population_size`

---

## MODIFIED Requirements

### Requirement: Strategy Selection (Modified)

The existing `_select_strategies()` method SHALL be extended to support GP exploration.

#### Original Behavior (Preserved)
- `strategy_selection_mode='random'` → random selection from Registry
- `strategy_selection_mode='exploit'` → select best performing strategies
- `strategy_selection_mode='regime_aware'` → regime-based selection

#### New Behavior (Added)
- When in Explore mode (determined by `exploit_ratio`)
- AND `gp_explore_enabled=True`
- AND random check < `gp_explore_ratio`
- THEN use GP exploration instead of random selection

---

## Interface Specifications

### GPExplorationRequest

```python
@dataclass
class GPExplorationRequest:
    """GP 探索請求"""
    symbol: str                      # 交易標的（必填）
    timeframe: str                   # 時間框架（必填）
    population_size: int             # 種群大小
    generations: int                 # 演化代數
    top_n: int = 3                   # 取前 N 個最佳策略
    market_state: Optional[Any] = None  # 市場狀態
```

### GPExplorationResult

```python
@dataclass
class GPExplorationResult:
    """GP 探索結果"""
    success: bool                    # 是否成功
    strategies: List[str]            # 已註冊的策略名稱
    best_fitness: Optional[float]    # 最佳適應度
    generations_run: int             # 實際演化代數
    error: Optional[str]             # 錯誤訊息
```

### StrategyRegistry Extensions

```python
class StrategyRegistry:
    @classmethod
    def register_dynamic(
        cls,
        name: str,
        strategy_class: Type[BaseStrategy],
        source: str = "gp",
        metadata: Optional[Dict] = None
    ) -> None:
        """動態註冊策略"""

    @classmethod
    def unregister_dynamic(cls, name: str) -> bool:
        """移除動態策略"""

    @classmethod
    def list_dynamic(cls) -> List[str]:
        """列出動態策略"""

    @classmethod
    def clear_dynamic(cls) -> int:
        """清理所有動態策略，返回清理數量"""
```

---

## Error Handling

### GP Exploration Errors

| Error Type | Handling |
|------------|----------|
| GPLoop timeout | Log warning, fallback to random selection |
| Insufficient data | Log error, fallback to random selection |
| DEAP not available | Log warning at init, disable GP explore |
| Expression conversion failure | Skip individual, continue with others |
| Registration failure | Log error, skip strategy, continue |

### Graceful Degradation

The system SHALL degrade gracefully when GP components are unavailable:
1. `DEAP_AVAILABLE=False` → disable GP explore automatically
2. `GPLoop` initialization failure → disable GP explore
3. Individual strategy failure → skip, continue with remaining

---

## Performance Considerations

### GP Exploration Cost
- Expected duration: 30-120 seconds per exploration
- Memory: ~100MB per exploration (population in memory)
- CPU: High (fitness evaluation uses BacktestEngine)

### Optimization Strategies
1. Limit `gp_generations` for faster exploration
2. Use early stopping (already implemented in GPLoop)
3. Cache OHLCV data (already implemented in UltimateLoop)
4. Consider async execution for future optimization

### Dynamic Strategy Limits
- Maximum dynamic strategies: 50 (configurable)
- LRU eviction when limit reached
- Prefer keeping validated strategies
