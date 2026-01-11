# Proposal: 回測系統優化 v1

## 概述

針對 BTC/ETH 永續合約回測系統進行全面優化，提升回測真實性、統計信心、風險控制和執行效能。

## 目標

1. **回測真實性** +20-30%：滑點、流動性模擬
2. **統計信心**：Bootstrap、Deflated Sharpe Ratio
3. **風險控制**：Kelly Criterion、相關性分析
4. **執行效能** 50-200x：Apple Silicon M4 Max 專屬優化

## 硬體配置

```
Mac Studio - Apple M4 Max
├── CPU: 16 核心
├── GPU: 40 核心 Metal
├── Neural Engine: 16 核心
├── 統一記憶體: 64GB
└── SSD: 2TB NVMe
```

## 範圍

### Phase 1: 資料品質（3 任務）
- 資料缺失處理
- 滑點模擬
- 流動性影響

### Phase 2: 策略驗證（3 任務）
- Bootstrap & Permutation Test
- Combinatorial Purged CV
- Deflated Sharpe Ratio

### Phase 3: 風險管理（3 任務）
- Kelly Criterion
- 多策略相關性
- 黑天鵝壓力測試

### Phase 4: AI 自動化（3 任務）
- 多目標優化
- 策略組合優化
- 自動特徵工程

### Phase 5: 效能優化（4 任務）
- 向量化 + Polars
- Metal GPU 加速（MLX）
- 多核心並行
- 統一記憶體優化

### UI 更新（2 任務）
- 驗證頁面
- 風險儀表板

## 預期成果

| 指標 | 預期提升 |
|------|----------|
| 回測真實性 | +20-30% |
| 統計信心 | 顯著提升 |
| 風險控制 | 動態調整 |
| 執行效能 | 50-200x |

## 風險評估

- **低風險**：Phase 1-3（增加功能，不影響現有）
- **中風險**：Phase 4（AI 功能複雜度）
- **高風險**：Phase 5（效能優化可能影響穩定性）

## 驗證策略

1. 每個 Task 完成後執行單元測試
2. Phase 完成後執行整合測試
3. 效能優化需要基準測試比較
