# Backtest Loop System

## Why

現有系統有完整的回測元件（HyperLoop、策略、優化器、驗證器），但缺乏統一的使用者介面將這些元件整合為一個可配置的自動化回測循環。使用者需要手動編寫程式碼來執行完整的策略優化與驗證流程。

## What Changes

建立 `BacktestLoop` 作為使用者導向的回測循環系統，整合現有 HyperLoop 高效能基礎設施：

- **新增** `src/automation/backtest_loop.py` - 主要使用者介面
- **新增** `src/automation/loop_config.py` - 配置系統
- **新增** `src/automation/loop_runner.py` - 執行引擎
- **新增** `src/automation/validation_runner.py` - 驗證流程整合
- **修改** `src/automation/__init__.py` - 匯出新模組

## Impact

- Affected specs: automation, learning
- Affected code:
  - `src/automation/` - 新增 4 個模組
  - `src/learning/recorder.py` - 可能需要小幅擴展
- No breaking changes - 純新增功能

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    BacktestLoop (User API)                   │
│  - 配置策略、標的、時間框架                                   │
│  - 啟動/暫停/恢復 Loop                                        │
│  - 查看進度與結果                                             │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                     LoopRunner (Orchestration)               │
│  - 迭代控制                                                   │
│  - 策略選擇 (exploit/explore)                                 │
│  - 結果聚合                                                   │
└──────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  HyperLoop      │  │  Bayesian       │  │  Validation     │
│  Controller     │  │  Optimizer      │  │  Runner         │
│  (並行執行)      │  │  (參數優化)      │  │  (5階段驗證)    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                  ExperimentRecorder (Learning)               │
│  - 記錄實驗結果                                               │
│  - 更新 insights.md                                           │
│  - 同步 Memory MCP                                            │
└──────────────────────────────────────────────────────────────┘
```

## Key Features

1. **配置驅動**：YAML/Dict 配置，易於調整
2. **策略選擇**：支援 epsilon-greedy、UCB、Thompson Sampling
3. **多模式執行**：N 次迭代、時間限制、目標達成
4. **自動驗證**：Walk-Forward + Monte Carlo
5. **學習整合**：自動記錄洞察到 insights.md 和 Memory MCP
6. **斷點恢復**：支援中斷後繼續執行
