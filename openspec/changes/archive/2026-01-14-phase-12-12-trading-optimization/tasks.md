# Phase 12.12 Implementation Tasks

## 1. 啟用 5 階段驗證 (sequential)

- [x] 1.1 修改 `ultimate_config.py` 預設配置 | files: src/automation/ultimate_config.py

## 2. 信號放大器 (sequential)

- [x] 2.1 建立 SignalAmplifier 類別 | files: src/strategies/signal_amplifier.py
- [x] 2.2 實作 threshold_expand 模式（閾值放寬）
- [x] 2.3 實作 anticipation 模式（提前預判）
- [x] 2.4 實作 tolerance 模式（容忍度放寬）
- [x] 2.5 實作 sensitivity 模式（靈敏度調整）

## 3. 信號過濾管道 (parallel: 3.1-3.4, then 3.5)

- [x] 3.1 建立 BaseSignalFilter 基礎類別 | files: src/strategies/filters/base_filter.py
- [x] 3.2 建立 TimeFilter（避開資金費率結算）| files: src/strategies/filters/time_filter.py
- [x] 3.3 建立 StrengthFilter（只保留強信號）| files: src/strategies/filters/strength_filter.py
- [x] 3.4 建立 VolumeFilter（成交量確認）| files: src/strategies/filters/volume_filter.py
- [x] 3.5 建立 ConfirmationFilter（多指標確認）| files: src/strategies/filters/confirmation_filter.py
- [x] 3.6 建立 FilterPipeline（過濾管道）| files: src/strategies/filters/pipeline.py

## 4. 動態風控 (sequential)

- [x] 4.1 建立 DynamicRiskController 類別 | files: src/risk/dynamic_risk.py
- [x] 4.2 實作波動度縮放（高波動→小部位）
- [x] 4.3 實作回撤縮放（DD > 5/10/15% → 降低風險）
- [x] 4.4 實作移動止損（獲利 > 2% → 追蹤止損）
- [x] 4.5 修復 Reviewer 發現的問題（初始化旗標、PnL 計算、安全常數）

## 5. 自適應槓桿 (sequential)

- [x] 5.1 建立 AdaptiveLeverageManager 類別 | files: src/risk/adaptive_leverage.py
- [x] 5.2 實作波動度調整（低波動 × 1.5，高波動 × 0.5）
- [x] 5.3 實作回撤調整（DD > 5/10% → 降低槓桿）
- [x] 5.4 實作連續表現調整（連勝 × 1.2，連虧 × 0.8）
- [x] 5.5 修復 Reviewer 發現的問題（驗證、除零保護、邊界檢查）

## 6. 高效能配置 (sequential)

- [x] 6.1 新增 Phase 12.12 配置欄位 | files: src/automation/ultimate_config.py
- [x] 6.2 新增 `create_high_performance_config()` 方法
- [x] 6.3 更新 `validate()` 方法驗證新欄位
- [x] 6.4 更新 `to_dict()` 方法包含新欄位

## 7. 整合測試 (sequential)

- [x] 7.1 執行 dynamic_risk.py 單元測試（23/23 通過）
- [x] 7.2 執行 adaptive_leverage.py 單元測試（19/19 通過）
- [x] 7.3 執行整合測試（12/12 通過）
- [x] 7.4 修復所有 Pyright 型別錯誤

## 完成日期

2026-01-14
