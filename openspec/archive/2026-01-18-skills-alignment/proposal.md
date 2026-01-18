# UltimateLoop 與 Skills 對齊

## Why

目前 `UltimateLoop` 的回測流程與專案 Skills 文檔存在差距：

| 階段 | 對齊度 | 主要缺口 |
|------|--------|---------|
| Phase 1: 市場分析 | 30% | 回傳 None，未實際影響後續流程 |
| Phase 2: 策略選擇 | 80% | 缺少參數預生成（±30%）機制 |
| Phase 3: 優化 | 85% | **無 Walk-Forward 滾動窗口** |
| Phase 4: 驗證 | 90% | Grade D 邏輯需優化 |
| Phase 5: 學習 | 70% | 洞察觸發條件未驗證 |

### 核心問題

**每次迭代使用相同歷史資料 → 所有 Sharpe 值相同**

Skills 文檔明確規定：
- Walk-Forward Analysis 使用 70/30 IS/OOS 分割
- 交易筆數 >= 30 才有統計意義
- PBO < 50% 才能信任策略
- 兩階段參數搜索避免局部最優

## What Changes

### 1. Walk-Forward 滾動窗口（高優先）

**來源 Skill**：`參數優化` → `references/walk-forward.md`

```
現狀：每次用完整 2017-2024 資料優化
改為：
  Window 1: IS=[2017-2019], OOS=[2019-2020]
  Window 2: IS=[2018-2020], OOS=[2020-2021]
  Window 3: IS=[2019-2021], OOS=[2021-2022]
  ...
```

**新增模組**：`src/automation/walk_forward.py`
- `WalkForwardAnalyzer` 類別
- `generate_windows()` - 生成滾動窗口
- `calculate_efficiency()` - 計算 WFA 效率

### 2. 交易筆數門檻（高優先）

**來源 Skill**：`參數優化` → `references/overfitting-detection.md`

```python
# 現狀：無檢查
# 改為：
if total_trades < 30:
    result.warning = "統計無效：交易筆數 < 30"
    result.grade = 'D'  # 降級
```

### 3. PBO 過擬合指標（中優先）

**來源 Skill**：`參數優化` → `references/overfitting-detection.md`

```python
def calculate_pbo(returns_matrix, n_splits=8):
    """
    PBO < 25%: 低風險 → 可信賴
    PBO 25-50%: 中風險 → 需謹慎
    PBO > 50%: 高風險 → 策略可能過擬合
    """
```

### 4. 兩階段參數搜索（中優先）

**來源 Skill**：`參數優化` → SKILL.md

```
Stage 1: 粗搜索（大步長）→ 找有效區域
Stage 2: 細搜索（小步長）→ 在有效區域內精細優化
```

### 5. 其他對齊項目

| 項目 | 階段 | 來源 Skill | 優先級 |
|------|------|-----------|--------|
| 市場狀態回傳值修復 | Phase 1 | 策略開發 | 低 |
| 參數預生成 ±30% | Phase 2 | 參數優化 | 中 |
| Grade D 邏輯優化 | Phase 4 | 策略驗證 | 低 |
| 洞察觸發條件驗證 | Phase 5 | 學習系統 | 低 |

## Impact

### Affected Specs
- `參數優化` - 主要參考
- `策略驗證` - Grade 判定
- `學習系統` - 洞察記錄

### Affected Code
- `src/automation/ultimate_loop.py` - 主要修改
- `src/automation/walk_forward.py` - 新增
- `src/automation/overfitting_detector.py` - 新增

### Breaking Changes
**無破壞性變更** - 所有新功能都是可選的：
- `use_walk_forward: bool = True`
- `min_trades: int = 30`
- `max_pbo: float = 0.5`

## Data Contracts

### WalkForwardWindow

```python
@dataclass
class WalkForwardWindow:
    window_id: int
    is_start: datetime
    is_end: datetime
    oos_start: datetime
    oos_end: datetime
    is_data: pd.DataFrame
    oos_data: pd.DataFrame
```

### WalkForwardResult

```python
@dataclass
class WalkForwardResult:
    windows: List[WalkForwardWindow]
    is_mean_return: float
    oos_mean_return: float
    efficiency: float  # OOS / IS
    oos_win_rate: float
    total_oos_return: float
```

### OverfitMetrics

```python
@dataclass
class OverfitMetrics:
    pbo: float                    # 0-1, <0.25 低風險
    is_oos_ratio: float           # IS/OOS 績效比
    param_sensitivity: float      # 參數敏感度
    trade_count: int              # 交易筆數
    overall_risk: str             # 'LOW', 'MEDIUM', 'HIGH'
```

## Expected Behavior

### Walk-Forward 流程

```
_run_optimization()
    ↓
生成滾動窗口（n_windows=5, is_ratio=0.7）
    ↓
for each window:
    1. 在 IS 優化參數
    2. 在 OOS 測試績效
    3. 記錄 IS/OOS 比較
    ↓
計算 WFA 效率
    ↓
if efficiency < 0.5:
    warning: "可能過擬合"
```

### 交易筆數檢查

```
回測完成
    ↓
if total_trades < 30:
    grade = max('D', grade)  # 降級
    add_warning("統計無效")
```

### PBO 計算

```
驗證階段
    ↓
收集所有參數組合的報酬序列
    ↓
calculate_pbo(returns_matrix)
    ↓
if pbo > 0.5:
    grade = 'D'
    add_warning("高過擬合風險")
```

## Non-Goals

- **不修改** 現有策略類別
- **不修改** BacktestEngine API
- **不修改** 資料管道
- **不替換** Optuna 優化器

## Success Criteria

1. **每次迭代產生不同 Sharpe** - Walk-Forward 滾動窗口生效
2. **WFA 效率 > 50%** - 策略可信賴
3. **交易筆數 >= 30** - 統計有效
4. **PBO < 50%** - 無嚴重過擬合
5. **兩階段搜索覆蓋更大參數空間**

## Technical Design

詳見 `design.md`（待建立）
