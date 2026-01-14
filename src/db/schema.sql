-- 實驗記錄表
CREATE TABLE IF NOT EXISTS experiments (
    -- 主鍵
    id VARCHAR PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,

    -- 策略資訊
    strategy_name VARCHAR NOT NULL,
    strategy_type VARCHAR NOT NULL,
    strategy_version VARCHAR DEFAULT '1.0',
    params JSON,

    -- 配置
    symbol VARCHAR NOT NULL,
    timeframe VARCHAR NOT NULL,
    start_date VARCHAR,
    end_date VARCHAR,

    -- 結果（數值欄位便於查詢和聚合）
    sharpe_ratio DOUBLE,
    total_return DOUBLE,
    max_drawdown DOUBLE,
    win_rate DOUBLE,
    profit_factor DOUBLE,
    total_trades INTEGER,
    sortino_ratio DOUBLE,
    calmar_ratio DOUBLE,

    -- 驗證
    grade VARCHAR,
    stages_passed JSON,

    -- 狀態
    status VARCHAR DEFAULT 'completed',

    -- 元數據
    insights JSON,
    tags JSON,
    parent_experiment VARCHAR,
    improvement DOUBLE,

    -- 索引欄位
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 策略統計表
CREATE TABLE IF NOT EXISTS strategy_stats (
    name VARCHAR PRIMARY KEY,
    attempts INTEGER DEFAULT 0,
    successes INTEGER DEFAULT 0,
    avg_sharpe DOUBLE DEFAULT 0.0,
    best_sharpe DOUBLE DEFAULT 0.0,
    worst_sharpe DOUBLE DEFAULT 0.0,
    best_params JSON,
    last_params JSON,
    last_attempt TIMESTAMP,
    first_attempt TIMESTAMP,
    ucb_score DOUBLE DEFAULT 0.0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引：提升查詢效能
CREATE INDEX IF NOT EXISTS idx_experiments_symbol ON experiments(symbol);
CREATE INDEX IF NOT EXISTS idx_experiments_strategy ON experiments(strategy_name);
CREATE INDEX IF NOT EXISTS idx_experiments_grade ON experiments(grade);
CREATE INDEX IF NOT EXISTS idx_experiments_timestamp ON experiments(timestamp);
CREATE INDEX IF NOT EXISTS idx_experiments_sharpe ON experiments(sharpe_ratio);
