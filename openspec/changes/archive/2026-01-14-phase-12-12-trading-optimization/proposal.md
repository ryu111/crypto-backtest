# Phase 12.12: 交易優化框架 + 高效能並行

## 狀態

**已完成** - 2026-01-14

## 摘要

建立完整的交易優化框架，包含信號放大、過濾管道、動態風控、自適應槓桿，並優化為 M4 Max 高效能並行執行。

## 動機

用戶核心思路：

| 問題 | 解決方案 |
|------|----------|
| 交易筆數太少 | 放寬條件增加信號 → 用過濾器濾掉雜訊 |
| MaxDD 太大 | 用風控管理（部位大小、止損、回撤限制） |
| 報酬不高 | 適當使用槓桿放大 |

## 硬體配置

```
Apple M4 Max | 16 核心 (12P + 4E) | 64GB RAM
可用效能：70-80% → 11-13 核心 / 45-51GB RAM
```

## 變更範圍

### 新增檔案

| 檔案 | 功能 |
|------|------|
| `src/strategies/signal_amplifier.py` | 信號放大器（閾值放寬、提前預判、容忍度、靈敏度） |
| `src/strategies/filters/base_filter.py` | 過濾器基礎類別 |
| `src/strategies/filters/strength_filter.py` | 強度過濾器 |
| `src/strategies/filters/confirmation_filter.py` | 確認過濾器 |
| `src/strategies/filters/time_filter.py` | 時間過濾器（避開資金費率結算） |
| `src/strategies/filters/volume_filter.py` | 成交量過濾器 |
| `src/strategies/filters/pipeline.py` | 過濾管道 |
| `src/risk/dynamic_risk.py` | 動態風控（波動度縮放、回撤縮放、移動止損） |
| `src/risk/adaptive_leverage.py` | 自適應槓桿（波動度/回撤/連續表現調整） |

### 修改檔案

| 檔案 | 修改內容 |
|------|----------|
| `src/automation/ultimate_config.py` | 新增 Phase 12.12 配置 + `create_high_performance_config()` |

## 預估效能

| 指標 | 單核心 | 12 核心並行 | 提升 |
|------|--------|-------------|------|
| 100 iterations | ~55 min | ~5 min | 11x |
| 5000 total trials | - | ~5 min | - |
| 記憶體使用 | ~8GB | ~40GB | 5x 容量 |
