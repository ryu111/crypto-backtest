# Tasks: 完整 Skills 系統對齊

## 優先級說明

| 優先級 | 意義 | 任務數 |
|--------|------|--------|
| P0 | 核心功能缺失 | 4 |
| P1 | 重要對齊項目 | 4 |
| P2 | 完善對齊 | 4 |

---

## P0: 核心功能補齊

### Task 1: 強平機制整合 ✅

**Skill 對齊**：風險管理、永續合約

**檔案**：`src/backtester/perpetual.py`

**需求**：
- [x] `LiquidationSimulator` 類別
  - [x] `calculate_liquidation_price(entry, leverage, direction)` 計算強平價格
  - [x] `check_liquidation(position, price)` 檢查是否觸發
  - [x] `execute_liquidation(position)` 執行強平、扣除罰金
- [x] 整合到 `PerpetualBacktester._process_bar()`
- [x] 新增配置 `enable_liquidation: bool = True`
- [x] 新增配置 `maintenance_margin_rate: float = 0.005`

**驗收**：
- [x] 10x 槓桿做多 $50,000，強平價約 $45,250
- [x] 觸發強平時記錄 `liquidation_event`
- [x] 強平計入績效統計

---

### Task 2: 強平安全檢查 ✅

**Skill 對齊**：風險管理

**檔案**：`src/risk/liquidation_safety.py`（新建）

**需求**：
- [x] `LiquidationSafetyChecker` 類別
  - [x] `check_stop_before_liquidation(entry, stop, leverage, direction)`
  - [x] `suggest_safe_stop(entry, leverage, direction, buffer=0.02)`
- [x] 整合到策略進場邏輯
- [x] 進場前警告：止損在強平之後

**驗收**：
- [x] 止損在強平之後時回傳警告
- [x] 提供建議的安全止損價格
- [x] 可配置安全緩衝比例

---

### Task 3: 動態槓桿管理 ✅

**Skill 對齊**：風險管理

**檔案**：`src/risk/dynamic_leverage.py`（新建）

**需求**：
- [x] `DynamicLeverageManager` 類別
  - [x] `__init__(base_leverage, max_leverage, atr_period=14)`
  - [x] `calculate_adjusted_leverage(current_atr, avg_atr)`
  - [x] `get_leverage_for_trade(data)`
- [x] 整合到 `UltimateLoopController._execute_trade()`
- [x] 新增配置 `use_dynamic_leverage: bool = False`

**驗收**：
- [x] 高波動時自動降低槓桿
- [x] 低波動時可提高槓桿（有上限）
- [x] 槓桿變化記錄到日誌

---

### Task 4: 資料驗證優先級 ✅

**Skill 對齊**：資料管道

**檔案**：`src/data/validator.py`（新建）

**需求**：
- [x] `DataValidator` 類別
  - [x] `validate(df) -> List[DataIssue]`
  - [x] `validate_before_backtest(df) -> bool` 回測前強制驗證
  - [x] `auto_fix(df, issues) -> pd.DataFrame` 自動修復
- [x] 驗證項目（參考資料管道 Skill）：
  - [x] 缺失值檢查
  - [x] 時間連續性
  - [x] OHLC 邏輯（H >= max(O,C), L <= min(O,C)）
  - [x] 成交量 > 0
  - [x] 重複時間戳
- [x] 整合到回測流程開始前

**驗收**：
- [x] FATAL 問題阻止回測啟動
- [x] WARNING 問題自動修復並記錄
- [x] 驗證結果寫入日誌

---

## P1: 重要對齊項目

### Task 5: Loop 狀態持久化 ✅

**Skill 對齊**：AI自動化

**檔案**：`src/automation/state_persistence.py`（新建）

**需求**：
- [x] `LoopStatePersistence` 類別
  - [x] `save_state(controller)` 保存當前狀態
  - [x] `load_state() -> LoopState` 載入狀態
  - [x] `clear_state()` 清除狀態檔案
- [x] 狀態內容：
  - [x] `iteration`: 當前迭代數
  - [x] `completed_strategies`: 已完成策略列表
  - [x] `best_results`: 最佳結果快照
  - [x] `timestamp`: 保存時間
- [x] 整合到 `UltimateLoopController`
  - [x] 每次迭代後自動保存
  - [x] 啟動時檢查是否有中斷狀態
  - [x] 新增 `--resume` CLI 參數

**驗收**：
- [x] 中斷後可恢復執行
- [x] 狀態檔案正確保存/載入
- [x] CLI 參數運作正常

---

### Task 6: 資金費率成本整合 ✅

**Skill 對齊**：永續合約

**檔案**：`src/backtester/perpetual.py`

**需求**：
- [x] `FundingRateHandler` 類別
  - [x] `load_funding_rates(symbol, start, end)` 載入費率數據
  - [x] `get_rate_at(timestamp)` 獲取特定時間費率
  - [x] `calculate_cost(position, timestamp)` 計算成本
- [x] 結算時機：每 8 小時（00:00, 08:00, 16:00 UTC）
- [x] 整合到 `PerpetualBacktester._process_bar()`
- [x] 新增配置 `include_funding_cost: bool = True`

**驗收**：
- [x] 持倉跨越結算時間時扣除費率
- [x] 費率成本記錄到交易記錄
- [x] 績效報告顯示總資金費率成本

---

### Task 7: 時間段穩健性測試 ✅

**Skill 對齊**：策略驗證

**檔案**：`src/validator/time_robustness.py`（新建）

**需求**：
- [x] `TimeRobustnessTest` 類別
  - [x] `split_by_segments(data, n_segments=4)`
  - [x] `test_each_segment(strategy, params, segments)`
  - [x] `calculate_consistency_score(results)`
- [x] 一致性標準：
  - [x] 所有時間段都獲利
  - [x] Sharpe 變異 < 50%
  - [x] 無單段重大虧損 (< -15%)
- [x] 整合到 5 階段驗證的階段 3

**驗收**：
- [x] 分成 4 段測試
- [x] 計算各段一致性分數
- [x] 輸出詳細的時間段報告

---

### Task 8: 多標的驗證自動化 ✅

**Skill 對齊**：策略驗證

**檔案**：`src/validator/multi_asset.py`（新建）

**需求**：
- [x] `MultiAssetValidator` 類別
  - [x] `validate_across_assets(strategy, params, assets=['BTC', 'ETH'])`
  - [x] `calculate_cross_asset_score(results)`
- [x] 驗證標準：
  - [x] 兩個標的都獲利
  - [x] Sharpe 差異 < 1.0
  - [x] 相關性分析
- [x] 整合到 5 階段驗證的階段 3

**驗收**：
- [x] 自動對 BTC/ETH 測試
- [x] 輸出跨標的一致性報告
- [x] 標記高相關性標的

---

## P2: 完善對齊

### Task 9: 策略過濾器整合 ✅

**Skill 對齊**：策略開發

**狀態**：已存在於現有架構中

**說明**：
- 策略過濾器（VolumeFilter, TimeFilter, StrengthFilter）
- 已在 `src/strategies/filters/` 目錄實作
- 可透過配置啟用

---

### Task 10: Kelly Criterion 部位計算 ✅

**Skill 對齊**：風險管理

**檔案**：`src/risk/position_sizing.py`

**狀態**：已完整實作

**已有功能**：
- [x] `kelly_criterion(win_rate, win_loss_ratio)` 函數
- [x] `KellyPositionSizer` 類別
  - [x] Full Kelly、Half Kelly、Quarter Kelly 支援
  - [x] `calculate_position_size(capital, win_rate, avg_win, avg_loss)`
  - [x] `calculate_from_trades(capital, winning_trades, losing_trades)`

---

### Task 11: 基差交易參考整合 ✅

**Skill 對齊**：永續合約

**狀態**：已存在於 `src/strategies/statistical_arb/basis_arb.py`

**已有功能**：
- [x] `BasisTrade` 類別
- [x] entry_threshold / exit_threshold 配置
- [x] 基差計算邏輯

---

### Task 12: 完整 PBO 計算 ✅

**Skill 對齊**：參數優化

**檔案**：`src/automation/overfitting_detector.py`

**狀態**：已完整實作

**已有功能**：
- [x] `OverfitDetector.calculate_pbo()` CSCV 方法
- [x] `OverfitDetector.deflated_sharpe_ratio()` DSR 計算
- [x] `OverfitMetrics` 詳細過擬合指標
- [x] `PBOResult` 詳細結果輸出

---

## 任務依賴

```
Task 1 (強平機制) ───→ Task 2 (強平安全檢查)
       ↓
Task 3 (動態槓桿) ───→ Task 10 (Kelly Criterion)
       ↓
Task 4 (資料驗證) ───→ Task 6 (資金費率)
       ↓
Task 5 (Loop持久化)
       ↓
Task 7,8 (穩健性/多標的) 可並行
       ↓
Task 9,11,12 (過濾器/基差/PBO) 可並行
```

## 驗收標準

| 指標 | 目標值 | 狀態 |
|------|--------|------|
| 風險管理對齊率 | >= 85% | ✅ |
| 永續合約對齊率 | >= 80% | ✅ |
| 資料管道對齊率 | >= 85% | ✅ |
| AI自動化對齊率 | >= 90% | ✅ |
| 整體對齊率 | >= 90% | ✅ |
| 現有測試通過率 | 100% | 待驗證 |

## 完成摘要

### 新建檔案
1. `src/risk/liquidation_safety.py` - 強平安全檢查
2. `src/risk/dynamic_leverage.py` - 動態槓桿管理
3. `src/data/validator.py` - 資料驗證優先級
4. `src/automation/state_persistence.py` - Loop 狀態持久化
5. `src/validator/time_robustness.py` - 時間段穩健性測試
6. `src/validator/multi_asset.py` - 多標的驗證

### 修改檔案
1. `src/backtester/perpetual.py` - 新增 LiquidationSimulator、FundingRateHandler
2. `src/backtester/engine.py` - 新增強平配置和統計欄位

### 已存在（確認對齊）
1. `src/risk/position_sizing.py` - Kelly Criterion
2. `src/automation/overfitting_detector.py` - PBO、DSR
3. `src/strategies/filters/` - 策略過濾器
4. `src/strategies/statistical_arb/basis_arb.py` - 基差交易
