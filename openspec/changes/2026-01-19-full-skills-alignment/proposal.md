# Change Proposal: 完整 Skills 系統對齊

## 概述

| 欄位 | 內容 |
|------|------|
| ID | 2026-01-19-full-skills-alignment |
| 狀態 | 📋 規劃中 |
| 優先級 | P0-P2 |
| 影響範圍 | 風險管理、永續合約、資料管道、AI自動化 |

## 背景

在完成第一階段 Skills 對齊（Walk-Forward、PBO、兩階段搜索）後，經過全面檢驗發現整體對齊率為 66%，主要缺口集中在：

| Skill | 對齊率 | 主要缺口 |
|-------|--------|----------|
| 風險管理 | 50% | 強平機制、動態槓桿、Kelly Criterion |
| 永續合約 | 40% | 資金費率策略、基差交易、強平級聯 |
| 資料管道 | 45% | 資料驗證優先級、時間連續性檢查 |
| AI自動化 | 70% | Loop 狀態持久化、暫停/恢復機制 |
| 策略驗證 | 75% | 時間段穩健性、多標的驗證 |

## 目標

1. **風險管理對齊**：實作強平機制、動態槓桿、強平安全檢查
2. **永續合約對齊**：整合資金費率到回測成本計算
3. **資料管道對齊**：實作資料驗證優先級和自動修復
4. **AI自動化完善**：Loop 狀態持久化和斷點恢復
5. **整體對齊率**：從 66% 提升至 90%+

## 技術方案

### 1. 強平機制整合 (P0)

**問題**：回測引擎未模擬強平，導致績效不真實

**方案**：
```python
# src/backtester/perpetual.py 新增
class LiquidationSimulator:
    def check_liquidation(self, position, current_price, leverage):
        """檢查是否觸發強平"""
        liq_price = self.calculate_liquidation_price(
            position.entry_price, leverage, position.direction
        )

        if position.direction == 1:  # 多單
            return current_price <= liq_price
        else:  # 空單
            return current_price >= liq_price
```

**整合點**：
- `BacktestEngine._process_bar()` 每根 K 線檢查強平
- 觸發強平時：關閉倉位、記錄強平事件、扣除強平罰金

### 2. 動態槓桿 (P0)

**問題**：固定槓桿不適應市場波動變化

**方案**：
```python
# src/risk/dynamic_leverage.py 新增
class DynamicLeverageManager:
    def calculate_leverage(self, base_leverage, current_atr, avg_atr):
        """根據波動率調整槓桿"""
        vol_ratio = current_atr / avg_atr

        if vol_ratio > 1.5:
            return base_leverage * 0.5
        elif vol_ratio > 1.2:
            return base_leverage * 0.75
        elif vol_ratio < 0.8:
            return min(base_leverage * 1.25, self.max_leverage)
        return base_leverage
```

### 3. 資料驗證優先級 (P0)

**問題**：資料品質問題在回測中才發現，浪費計算資源

**方案**：
```python
# src/data/validator.py 新增
class DataValidator:
    def validate_before_backtest(self, df):
        """回測前強制驗證"""
        issues = []

        # 優先級 1：致命錯誤（阻止回測）
        if df.isnull().any().any():
            issues.append(('FATAL', 'null_values', df.isnull().sum()))

        # 優先級 2：警告（記錄但繼續）
        if not self._check_time_continuity(df):
            issues.append(('WARNING', 'time_gaps', self._find_gaps(df)))

        return issues
```

### 4. Loop 狀態持久化 (P1)

**問題**：Loop 中斷後需要重新開始

**方案**：
```python
# src/automation/state_persistence.py 新增
class LoopStatePersistence:
    def save_state(self, controller):
        """保存 Loop 狀態"""
        state = {
            'iteration': controller.current_iteration,
            'completed_strategies': controller.completed,
            'best_results': controller.best_results,
            'timestamp': datetime.now().isoformat()
        }
        Path('loop_state.json').write_text(json.dumps(state))

    def load_state(self):
        """載入並恢復"""
        return json.loads(Path('loop_state.json').read_text())
```

### 5. 資金費率成本整合 (P1)

**問題**：回測未計算資金費率成本，高估收益

**方案**：
- 在持倉期間，每 8 小時扣除資金費率
- 需要資金費率歷史數據（已有 `src/data/fetcher.py` 支援）

```python
# 修改 src/backtester/perpetual.py
def apply_funding_cost(self, position, funding_rate, timestamp):
    """每 8 小時結算資金費率"""
    if self._is_funding_settlement(timestamp):
        cost = position.notional * funding_rate * position.direction
        position.realized_pnl -= cost
        return cost
    return 0
```

## 任務拆解

| 優先級 | 任務數 | 主要內容 |
|--------|--------|----------|
| P0 | 4 | 強平機制、動態槓桿、資料驗證、強平安全檢查 |
| P1 | 4 | Loop 持久化、資金費率、時間穩健性、多標的驗證 |
| P2 | 4 | 策略過濾器、Kelly Criterion、基差交易參考、完整 PBO |

## 風險評估

| 風險 | 影響 | 緩解 |
|------|------|------|
| 強平機制影響現有測試 | 中 | 可配置開關 |
| 動態槓桿增加複雜度 | 低 | 預設關閉 |
| 資金費率數據缺失 | 中 | 降級為估算 |

## 成功指標

| 指標 | 目標 |
|------|------|
| 整體 Skills 對齊率 | >= 90% |
| 風險管理對齊率 | >= 85% |
| 永續合約對齊率 | >= 80% |
| 資料管道對齊率 | >= 85% |
| 所有現有測試通過 | 100% |

## 時程估計

| 階段 | 任務數 | 預計 |
|------|--------|------|
| P0 | 4 | Loop 1-4 |
| P1 | 4 | Loop 5-8 |
| P2 | 4 | Loop 9-12 |

---

**下一步**：查看 [tasks.md](./tasks.md) 獲取詳細任務清單
