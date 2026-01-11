# Tasks: backtest-optimization-v1

## Phase 1: 資料品質

- [x] 1.1 資料缺失處理 | files: src/data/fetcher.py, src/data/cleaner.py | agent: DEVELOPER
- [x] 1.2 滑點模擬 | files: src/backtester/slippage.py, src/backtester/engine.py | agent: DEVELOPER
- [x] 1.3 流動性影響 | files: src/backtester/liquidity.py | agent: DEVELOPER

## Phase 2: 策略驗證增強

- [x] 2.1 Bootstrap & Permutation Test | files: src/validator/statistical_tests.py | agent: DEVELOPER
- [x] 2.2 Combinatorial Purged CV | files: src/validator/walk_forward.py | agent: DEVELOPER
- [x] 2.3 Deflated Sharpe Ratio | files: src/validator/sharpe_correction.py | agent: DEVELOPER

## Phase 3: 風險管理強化

- [x] 3.1 Kelly Criterion 部位管理 | files: src/risk/position_sizing.py | agent: DEVELOPER
- [x] 3.2 多策略相關性分析 | files: src/risk/correlation.py | agent: DEVELOPER
- [x] 3.3 黑天鵝壓力測試 | files: src/validator/stress_test.py | agent: DEVELOPER

## Phase 4: AI 自動化升級

- [x] 4.1 多目標優化 | files: src/optimizer/multi_objective.py | agent: DEVELOPER
- [x] 4.2 策略組合優化 | files: src/optimizer/portfolio.py | agent: DEVELOPER
- [x] 4.3 自動特徵工程 | files: src/automation/feature_engineering.py | agent: DEVELOPER

## Phase 5: 效能優化（Apple Silicon M4 Max）

- [x] 5.1 向量化 + Polars 優化 | files: src/backtester/engine.py | agent: DEVELOPER
- [x] 5.2 Metal GPU 加速 | files: src/backtester/metal_engine.py | agent: DEVELOPER
- [x] 5.3 多核心並行回測 | files: src/backtester/parallel.py | agent: DEVELOPER
- [x] 5.4 統一記憶體優化 | files: src/data/memory_manager.py | agent: DEVELOPER

## UI 更新

- [x] UI.1 驗證頁面更新 | files: ui/pages/4_Validation.py | agent: DESIGNER, DEVELOPER
- [x] UI.2 風險管理儀表板 | files: ui/pages/5_RiskDashboard.py | agent: DESIGNER, DEVELOPER
