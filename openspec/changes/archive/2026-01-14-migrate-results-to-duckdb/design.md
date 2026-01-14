# Design: Migrate Results to DuckDB

## Context

### 現有架構

```
learning/experiments.json     # 622 筆，26,000+ 行
    ├── experiments[]         # 策略結果（結構化）
    │   ├── id, timestamp
    │   ├── strategy{name, type, version, params}
    │   ├── config{symbol, timeframe, start_date, end_date}
    │   ├── results{sharpe, return, drawdown, ...}
    │   └── validation{grade, stages_passed, ...}
    └── metadata              # 統計資訊

results/{exp_id}/             # 時間序列資料（已分離）
    ├── equity_curve.csv
    ├── daily_returns.csv
    └── trades.csv
```

### 問題分析

| 問題 | 現狀 | 影響 |
|------|------|------|
| 混合關注點 | JSON 存結構化資料 + 語意洞察 | 查詢模式衝突 |
| 型別混亂 | 跨模組使用 `Dict[str, Any]` | 欄位不一致、runtime 錯誤 |
| 效能瓶頸 | 每次查詢載入全部 JSON | 隨資料增長惡化 |
| 欄位不一致 | `_strategy_name` vs `strategy_name` | 資料遺失 |

## Goals / Non-Goals

### Goals
- 建立統一型別系統，消除裸 dict
- 遷移策略結果到 DuckDB，支援高效查詢
- 定義清晰的 Data Contracts，確保模組間一致性
- 保持 UI 功能不中斷

### Non-Goals
- 不重構時間序列儲存（results/{exp_id}/*.csv 保留）
- 不修改 Memory MCP 整合邏輯
- 不新增 UI 功能

---

## Data Contracts

### Contract 1: BacktestResultRecord

**寫入方**：`BacktestEngine`, `UltimateLoop`, `HyperLoop`
**讀取方**：`ExperimentRecorder`, `UI data_loader`

```python
@dataclass
class BacktestResultRecord:
    """回測結果記錄（DuckDB 主表）"""

    # 識別欄位
    id: str                           # exp_20260113_093205
    created_at: datetime              # 建立時間

    # 策略資訊
    strategy_name: str                # trend_ma_cross
    strategy_type: str                # trend | momentum | volatility | composite
    strategy_version: str             # 1.0.0

    # 配置
    symbol: str                       # BTCUSDT
    timeframe: str                    # 1h
    start_date: datetime
    end_date: datetime
    leverage: int = 1
    initial_capital: float = 10000.0

    # 參數（JSON 序列化）
    parameters: Dict[str, Any]

    # 績效指標
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    volatility: float

    # 交易統計
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade_duration: float
    expectancy: float

    # 永續合約特有
    total_funding_fees: float = 0.0
    avg_leverage_used: float = 1.0

    # 驗證結果
    validation_grade: Optional[str] = None      # A/B/C/D/F
    validation_stages_passed: int = 0
    walk_forward_efficiency: Optional[float] = None
    monte_carlo_p5: Optional[float] = None

    # 標籤（便於查詢）
    tags: List[str] = field(default_factory=list)
```

### Contract 2: StrategyStats

**寫入方**：`ExperimentRecorder`（聚合計算）
**讀取方**：`StrategySelector`, `UI`

```python
@dataclass
class StrategyStats:
    """策略統計（DuckDB 視圖或快取）"""

    strategy_name: str

    # 嘗試統計
    total_attempts: int
    successful_attempts: int          # 通過驗證次數
    success_rate: float

    # 績效統計
    avg_sharpe: float
    best_sharpe: float
    avg_return: float
    best_return: float
    avg_drawdown: float
    worst_drawdown: float

    # 時間追蹤
    first_attempt: datetime
    last_attempt: datetime

    # 最佳參數
    best_params: Optional[Dict[str, Any]] = None
```

### Contract 3: ValidationResultRecord

**寫入方**：`StageValidator`, `ValidationRunner`
**讀取方**：`ExperimentRecorder`, `UI`

```python
@dataclass
class ValidationResultRecord:
    """驗證結果記錄（DuckDB 子表）"""

    experiment_id: str                # FK to BacktestResultRecord.id

    grade: str                        # A/B/C/D/F
    passed_stages: int

    # 各階段結果
    stage1_passed: bool
    stage1_score: float
    stage1_message: str

    stage2_passed: bool
    stage2_score: float
    stage2_message: str

    stage3_passed: bool
    stage3_score: float
    stage3_message: str

    stage4_passed: bool
    stage4_score: float
    stage4_message: str
    stage4_efficiency: Optional[float] = None

    stage5_passed: bool
    stage5_score: float
    stage5_message: str
    stage5_p5: Optional[float] = None

    recommendation: str
```

---

## DuckDB Schema

```sql
-- 主表：回測結果
CREATE TABLE backtest_results (
    id VARCHAR PRIMARY KEY,
    created_at TIMESTAMP NOT NULL,

    -- 策略資訊
    strategy_name VARCHAR NOT NULL,
    strategy_type VARCHAR NOT NULL,
    strategy_version VARCHAR DEFAULT '1.0.0',

    -- 配置
    symbol VARCHAR NOT NULL,
    timeframe VARCHAR NOT NULL,
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    leverage INTEGER DEFAULT 1,
    initial_capital DOUBLE DEFAULT 10000.0,

    -- 參數（JSON）
    parameters JSON,

    -- 績效指標
    total_return DOUBLE NOT NULL,
    annual_return DOUBLE NOT NULL,
    sharpe_ratio DOUBLE NOT NULL,
    sortino_ratio DOUBLE NOT NULL,
    calmar_ratio DOUBLE NOT NULL,
    max_drawdown DOUBLE NOT NULL,
    max_drawdown_duration INTEGER NOT NULL,
    volatility DOUBLE NOT NULL,

    -- 交易統計
    total_trades INTEGER NOT NULL,
    win_rate DOUBLE NOT NULL,
    profit_factor DOUBLE NOT NULL,
    avg_win DOUBLE NOT NULL,
    avg_loss DOUBLE NOT NULL,
    avg_trade_duration DOUBLE NOT NULL,
    expectancy DOUBLE NOT NULL,

    -- 永續合約
    total_funding_fees DOUBLE DEFAULT 0.0,
    avg_leverage_used DOUBLE DEFAULT 1.0,

    -- 驗證
    validation_grade VARCHAR,
    validation_stages_passed INTEGER DEFAULT 0,
    walk_forward_efficiency DOUBLE,
    monte_carlo_p5 DOUBLE,

    -- 標籤
    tags VARCHAR[]
);

-- 索引
CREATE INDEX idx_results_strategy ON backtest_results(strategy_name);
CREATE INDEX idx_results_symbol ON backtest_results(symbol);
CREATE INDEX idx_results_created ON backtest_results(created_at);
CREATE INDEX idx_results_sharpe ON backtest_results(sharpe_ratio);
CREATE INDEX idx_results_grade ON backtest_results(validation_grade);

-- 視圖：策略統計
CREATE VIEW strategy_stats AS
SELECT
    strategy_name,
    COUNT(*) as total_attempts,
    COUNT(CASE WHEN validation_grade IN ('A', 'B') THEN 1 END) as successful_attempts,
    COUNT(CASE WHEN validation_grade IN ('A', 'B') THEN 1 END)::DOUBLE / COUNT(*) as success_rate,
    AVG(sharpe_ratio) as avg_sharpe,
    MAX(sharpe_ratio) as best_sharpe,
    AVG(total_return) as avg_return,
    MAX(total_return) as best_return,
    AVG(max_drawdown) as avg_drawdown,
    MIN(max_drawdown) as worst_drawdown,
    MIN(created_at) as first_attempt,
    MAX(created_at) as last_attempt
FROM backtest_results
GROUP BY strategy_name;
```

---

## API Changes

### ExperimentRecorder（重構後）

```python
class ExperimentRecorder:
    """實驗記錄器 - DuckDB 版本"""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or project_root / 'src' / 'db' / 'results.duckdb'
        self.conn = duckdb.connect(str(self.db_path))
        self._ensure_schema()

    def log_experiment(
        self,
        result: BacktestResultRecord,
        validation: Optional[ValidationResultRecord] = None
    ) -> str:
        """記錄實驗（型別安全）"""
        # INSERT INTO backtest_results ...

    def query_experiments(
        self,
        filters: Optional[QueryFilters] = None,
        order_by: str = 'created_at',
        limit: int = 100
    ) -> List[BacktestResultRecord]:
        """查詢實驗（返回型別化物件）"""

    def get_strategy_stats(self, strategy_name: str) -> Optional[StrategyStats]:
        """取得策略統計（從視圖查詢）"""

    def get_best_experiments(
        self,
        metric: str = 'sharpe_ratio',
        n: int = 10
    ) -> List[BacktestResultRecord]:
        """取得最佳實驗"""
```

### 向後相容層

```python
class LegacyRecorderAdapter:
    """提供舊 API 相容性（過渡期使用）"""

    def __init__(self, new_recorder: ExperimentRecorder):
        self._recorder = new_recorder

    def log_experiment(
        self,
        result: Any,  # 舊的 BacktestResult
        strategy_info: Dict[str, Any],
        config: Dict[str, Any],
        validation_result: Optional[Any] = None,
        insights: Optional[List[str]] = None,
        parent_experiment: Optional[str] = None
    ) -> str:
        """轉換舊格式到新格式"""
        record = self._convert_to_record(result, strategy_info, config, validation_result)
        return self._recorder.log_experiment(record)
```

---

## Migration Strategy

### Phase 1: 建立新架構（不影響現有功能）

1. 建立 `src/types/` 型別定義
2. 建立 `src/db/models.py` DuckDB schema
3. 建立 `src/db/repository.py` 資料存取層

### Phase 2: 雙寫模式（漸進遷移）

1. 新增 DuckDB 寫入，保留 JSON 寫入
2. 新增遷移腳本 `scripts/migrate_json_to_duckdb.py`
3. 執行遷移，驗證資料一致性

### Phase 3: 切換讀取源

1. `data_loader.py` 改從 DuckDB 讀取
2. 驗證 UI 功能正常
3. 監控效能改善

### Phase 4: 清理

1. 移除 JSON 寫入邏輯
2. 備份並刪除 `experiments.json`
3. 移除相容層（下一個 major 版本）

---

## Risks / Trade-offs

| 風險 | 機率 | 影響 | 緩解措施 |
|------|------|------|----------|
| 遷移資料遺失 | 低 | 高 | 遷移前備份、雙寫驗證 |
| DuckDB 相容性問題 | 低 | 中 | 使用穩定版本、CI 測試 |
| 型別轉換錯誤 | 中 | 中 | 完整測試覆蓋、漸進遷移 |
| UI 中斷 | 中 | 高 | 雙寫模式、相容層 |

---

## Rollback Plan

1. 保留 `experiments.json` 直到 Phase 4
2. 相容層允許隨時切回 JSON
3. DuckDB 檔案可直接刪除重建

---

## Success Metrics

| 指標 | 目標 |
|------|------|
| 查詢效能 | 聚合查詢 < 100ms（目前 > 1s） |
| 型別錯誤 | 0 runtime type errors |
| 測試覆蓋 | types/ 和 db/ > 90% |
| 資料一致性 | 遷移後 100% 資料可查詢 |
