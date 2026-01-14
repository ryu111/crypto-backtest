# Data Layer Specification

## ADDED Requirements

### Requirement: Type System

The system SHALL provide a unified type system for all data structures passed between modules.

#### Scenario: BacktestResultRecord Creation
- **GIVEN** a completed backtest execution
- **WHEN** the result is recorded
- **THEN** the result SHALL be a `BacktestResultRecord` dataclass with all required fields typed
- **AND** no raw `Dict[str, Any]` SHALL be used for structured data

#### Scenario: Type Validation
- **GIVEN** a data structure being passed between modules
- **WHEN** the data is received
- **THEN** the receiver SHALL validate the type at runtime using dataclass type hints
- **AND** type mismatches SHALL raise `TypeError` with descriptive message

### Requirement: DuckDB Storage

The system SHALL store backtest results in DuckDB for efficient querying.

#### Scenario: Result Persistence
- **GIVEN** a new `BacktestResultRecord`
- **WHEN** `ExperimentRecorder.log_experiment()` is called
- **THEN** the record SHALL be inserted into `backtest_results` table
- **AND** the insert SHALL be atomic (all-or-nothing)
- **AND** the experiment ID SHALL be returned

#### Scenario: Query Performance
- **GIVEN** 1000+ records in the database
- **WHEN** an aggregation query is executed (e.g., get best Sharpe by strategy)
- **THEN** the query SHALL complete in < 100ms
- **AND** the result SHALL be returned as typed `List[BacktestResultRecord]`

#### Scenario: Concurrent Access
- **GIVEN** multiple processes accessing the database
- **WHEN** simultaneous writes occur
- **THEN** DuckDB's transaction isolation SHALL prevent data corruption
- **AND** readers SHALL not block writers

### Requirement: Data Contracts

The system SHALL define explicit data contracts for module boundaries.

#### Scenario: Producer-Consumer Contract
- **GIVEN** a module producing data (e.g., `BacktestEngine`)
- **AND** a module consuming data (e.g., `ExperimentRecorder`)
- **WHEN** data is transferred
- **THEN** the data type SHALL match the contract defined in `src/types/`
- **AND** the contract SHALL be versioned for backward compatibility

#### Scenario: Field Naming Consistency
- **GIVEN** a field representing strategy name
- **WHEN** the field appears in any module
- **THEN** it SHALL be named `strategy_name` (not `_strategy_name` or `strategyName`)
- **AND** this convention SHALL be enforced by the type system

---

## ADDED Constraints

### Constraint: No Raw Dict for Structured Data

Modules SHALL NOT use `Dict[str, Any]` for structured data that crosses module boundaries.

**Allowed**:
```python
def log_experiment(self, record: BacktestResultRecord) -> str: ...
```

**Prohibited**:
```python
def log_experiment(self, result: Dict[str, Any]) -> str: ...  # VIOLATION
```

**Exception**: Dynamic parameters (e.g., strategy params) MAY use `Dict[str, Any]` but SHALL be wrapped in a typed container.

### Constraint: Single Source of Truth

- Backtest results: `src/db/results.duckdb` (source of truth)
- Time series: `results/{exp_id}/*.csv` (supplementary)
- Insights: `learning/insights.md` (human-readable summary)
- Semantic memory: Memory MCP (AI queryable)

---

## ADDED Interfaces

### Interface: IResultRepository

```python
class IResultRepository(Protocol):
    """結果儲存庫介面"""

    def insert(self, record: BacktestResultRecord) -> str:
        """插入記錄，返回 ID"""
        ...

    def get(self, id: str) -> Optional[BacktestResultRecord]:
        """取得單一記錄"""
        ...

    def query(
        self,
        filters: Optional[QueryFilters] = None,
        order_by: str = 'created_at',
        limit: int = 100
    ) -> List[BacktestResultRecord]:
        """查詢記錄"""
        ...

    def get_strategy_stats(self, strategy_name: str) -> Optional[StrategyStats]:
        """取得策略統計"""
        ...

    def delete(self, id: str) -> bool:
        """刪除記錄"""
        ...
```

### Interface: QueryFilters

```python
@dataclass
class QueryFilters:
    """查詢過濾條件"""

    strategy_name: Optional[str] = None
    strategy_type: Optional[str] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None

    min_sharpe: Optional[float] = None
    max_drawdown: Optional[float] = None

    validation_grade: Optional[List[str]] = None  # ['A', 'B']
    min_stages_passed: Optional[int] = None

    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None

    tags: Optional[List[str]] = None
```

---

## ADDED Data Types

### BacktestResultRecord

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| id | str | Yes | 實驗 ID (exp_YYYYMMDD_HHMMSS) |
| created_at | datetime | Yes | 建立時間 |
| strategy_name | str | Yes | 策略名稱 |
| strategy_type | str | Yes | 策略類型 (trend/momentum/volatility/composite) |
| strategy_version | str | No | 策略版本 (default: 1.0.0) |
| symbol | str | Yes | 交易標的 |
| timeframe | str | Yes | 時間框架 |
| start_date | datetime | Yes | 回測開始日期 |
| end_date | datetime | Yes | 回測結束日期 |
| leverage | int | No | 槓桿倍數 (default: 1) |
| initial_capital | float | No | 初始資金 (default: 10000.0) |
| parameters | Dict | Yes | 策略參數 |
| total_return | float | Yes | 總報酬率 |
| annual_return | float | Yes | 年化報酬率 |
| sharpe_ratio | float | Yes | 夏普比率 |
| sortino_ratio | float | Yes | 索提諾比率 |
| calmar_ratio | float | Yes | 卡爾馬比率 |
| max_drawdown | float | Yes | 最大回撤 |
| max_drawdown_duration | int | Yes | 最大回撤持續時間 |
| volatility | float | Yes | 波動率 |
| total_trades | int | Yes | 總交易次數 |
| win_rate | float | Yes | 勝率 |
| profit_factor | float | Yes | 獲利因子 |
| avg_win | float | Yes | 平均獲利 |
| avg_loss | float | Yes | 平均虧損 |
| avg_trade_duration | float | Yes | 平均交易持續時間 |
| expectancy | float | Yes | 期望值 |
| total_funding_fees | float | No | 總資金費率 (default: 0.0) |
| avg_leverage_used | float | No | 平均使用槓桿 (default: 1.0) |
| validation_grade | str | No | 驗證評級 (A/B/C/D/F) |
| validation_stages_passed | int | No | 通過階段數 (default: 0) |
| walk_forward_efficiency | float | No | Walk-Forward 效率 |
| monte_carlo_p5 | float | No | Monte Carlo P5 |
| tags | List[str] | No | 標籤 (default: []) |

---

## Migration Notes

### From JSON to DuckDB

1. `experiments[].id` -> `id`
2. `experiments[].timestamp` -> `created_at`
3. `experiments[].strategy.name` -> `strategy_name`
4. `experiments[].strategy.type` -> `strategy_type`
5. `experiments[].strategy.version` -> `strategy_version`
6. `experiments[].strategy.params` -> `parameters` (merged with config params)
7. `experiments[].config.*` -> `symbol`, `timeframe`, `start_date`, `end_date`
8. `experiments[].results.*` -> 對應績效欄位
9. `experiments[].validation.grade` -> `validation_grade`
10. `experiments[].validation.stages_passed` -> `validation_stages_passed`
11. `experiments[].tags` -> `tags`
