# 交易策略洞察彙整

> 最後更新：2026-01-14
> 總實驗數：622（含 4 個 Bayesian 優化實驗）
> Phase 6 Backtest Loop：✅ 已完成

---

## 🎓 核心知識

### 永續合約特殊性
| 特性 | 說明 | 回測影響 |
|------|------|----------|
| 資金費率 | 每 8 小時結算，多空成本不同 | 必須納入 P&L 計算 |
| 基差 | 永續 vs 現貨價差 | 可設計套利策略 |
| 強平機制 | 槓桿越高，維持保證金越高 | 需模擬強平風險 |
| 滑點 | 市價單衝擊成本 | 大單需考慮流動性 |

### 過擬合防護（5 階段驗證）
```
Stage 1: 基礎回測 → 邏輯正確？
    ↓
Stage 2: 統計檢定 → Sharpe 顯著 > 0？（p < 0.05）
    ↓
Stage 3: 穩健性測試 → 參數 ±10% 還能賺？
    ↓
Stage 4: Walk-Forward → 滾動訓練/測試通過？
    ↓
Stage 5: Monte Carlo → 1000 次模擬過擬合率 < 30%？
```

### 關鍵評估指標
| 指標 | 健康值 | 警戒值 | 說明 |
|------|--------|--------|------|
| Sharpe Ratio | > 1.5 | < 1.0 | 風險調整報酬 |
| MaxDD | < 20% | > 30% | 最大回撤 |
| 勝率 | > 50% | - | 需配合盈虧比 |
| 過擬合率 | < 20% | > 30% | Monte Carlo 失敗率 |
| 交易次數 | > 100 | < 30 | 統計顯著性 |

### AI 策略選擇
- **80% Exploit**：選擇歷史表現最好的策略
- **20% Explore**：嘗試新策略或參數組合
- **原因**：純 exploit 陷入局部最優，純 explore 浪費資源

---

## 策略類型洞察

### 趨勢跟隨策略

#### bayesian_ma_cross (2026-01-13) ⭐ 推薦
- **ETH 最佳參數**：fast_period=4, slow_period=76
  - 績效：Sharpe 1.49, Return 5271.7%, MaxDD 51.4%
  - **Walk-Forward Sharpe: 0.54** ✅ 通過驗證
  - 評級：B 級（穩健可用）
- **BTC 最佳參數**：fast_period=21, slow_period=72
  - 績效：Sharpe 0.97, Return 652.2%, MaxDD 58.5%
  - Walk-Forward Sharpe: 0.42
  - 評級：C 級
- **洞察**：較長的 slow_period (70+) 在長期趨勢中表現更佳

#### bayesian_supertrend (2026-01-13) ⚠️ 過擬合警訊
- **ETH 參數**：atr_period=15, multiplier=2.4
  - 整體 Sharpe 1.54, Return 6742.7%
  - **Walk-Forward Sharpe: 0.07** ❌ 未通過
  - **穩健性: 0.04** 極度不穩
- **BTC 參數**：atr_period=18, multiplier=3.8
  - 整體 Sharpe 1.43, Return 2235.0%
  - **Walk-Forward Sharpe: 0.12** ❌ 未通過
- **⚠️ 重要教訓**：高 Sharpe 不代表穩健！必須驗證 Walk-Forward！

#### ma_cross_4h_v1
- **最佳參數**：fast_period=10, slow_period=30
- **績效**：Sharpe 1.85, Return 45.6%
- **洞察**：ATR 2x 止損表現更好

### 動量策略

#### bayesian_rsi (2026-01-13)
- **ETH**：period=15, oversold=34, overbought=76
  - Sharpe 0.85, Return 716%
- **BTC**：period=16, oversold=34, overbought=78
  - Sharpe 0.68, Return 297%
- **洞察**：RSI 在趨勢市場表現較差，適合震盪市場

### 均值回歸策略
*尚無記錄*

---

## 標的特性

### BTCUSDT
- trend 策略表現良好（Sharpe: 1.85）
- ATR 2x 止損表現更好
- 高波動期適合趨勢策略

### ETHUSDT
- **最佳策略**：MA Cross (fast=4, slow=76)
  - Sharpe 1.49, Walk-Forward 0.54
  - 通過 Walk-Forward 驗證 ✅
- **標的特性**：波動性高於 BTC，趨勢策略表現更佳
- **風險**：MaxDD 較高（51.4%），需注意部位控制
- **發現**：ETH 的 MA Cross 比 Supertrend 穩健很多

---

## 風險管理洞察

### 部位大小
- Kelly Criterion 理論最優但實務上太激進
- 建議使用 Half-Kelly 或 Quarter-Kelly

### 止損策略
- ATR-based 止損比固定百分比更適應波動
- 2x ATR 是常見起點，需依策略調整

### 槓桿使用
- 永續合約最大槓桿雖高達 100x，但建議 < 10x
- 高槓桿 + 高波動 = 快速強平

---

## 過擬合教訓

### 教訓 1：參數過度優化
- **現象**：回測 Sharpe 3.0+，實盤虧損
- **原因**：過度擬合歷史資料
- **解法**：Walk-Forward + Monte Carlo 驗證

### 教訓 2：樣本不足
- **現象**：100 筆交易就宣稱策略有效
- **原因**：統計不顯著
- **解法**：至少 100+ 筆交易，t-test p < 0.05

### 教訓 3：忽略交易成本
- **現象**：回測賺錢，實盤虧手續費
- **原因**：未計入滑點和手續費
- **解法**：回測時加入真實交易成本

### 教訓 4：高 Sharpe ≠ 穩健（2026-01-13 新增）⭐
- **現象**：Supertrend 整體 Sharpe 1.54，但 Walk-Forward 只有 0.07
- **原因**：策略在特定市場週期表現極好，但不具普遍性
- **解法**：
  1. **必須執行 Walk-Forward 驗證**
  2. 計算穩健性 = WF_Sharpe / Std_Sharpe
  3. 穩健性 < 0.3 視為過擬合警訊
- **黃金法則**：整體 Sharpe 高但 WF Sharpe 低 = 過擬合！

---

## 🔄 工作流教訓

### 教訓 5：D→R→T 流程不可跳過（2026-01-13 新增）⭐
- **現象**：Developer 完成後直接修 bug，跳過 Reviewer
- **原因**：認為「簡單修復」不需要審查
- **後果**：未經審查的 bug 修復可能引入新問題
- **解法**：
  1. **所有程式碼產出必須經過 R→T**
  2. 即使是 Main Agent 直接修復，也要 R→T
  3. 「簡單」不是跳過的理由
- **黃金法則**：D→R→T 是品質保證，不是官僚程序！

### Phase 6 完成記錄（2026-01-13）

**新增模組**：
| 檔案 | 用途 | 測試狀態 |
|------|------|----------|
| `loop_config.py` | BacktestLoopConfig, LoopResult | ✅ 15/15 PASS |
| `validation_runner.py` | 5 階段驗證整合 | ✅ |
| `backtest_loop.py` | 使用者 API | ✅ |

**使用範例**：
```python
from src.automation import BacktestLoop, BacktestLoopConfig

config = BacktestLoopConfig(
    strategies=['ma_cross', 'rsi'],
    symbols=['BTCUSDT', 'ETHUSDT'],
    n_iterations=100,
    selection_mode='epsilon_greedy',
    validation_stages=[4, 5]  # WFA + MC
)

with BacktestLoop(config) as loop:
    result = loop.run()
    print(result.summary())
```

---

## 🔧 待優化項目

### 資料品質
- [ ] 處理交易所維護期間的資料缺失
- [ ] 加入滑點模擬（市價單 vs 限價單）
- [ ] 考慮流動性影響（大單衝擊成本）

### 策略驗證
- [ ] 加入更多統計檢定（Bootstrap, Permutation Test）
- [ ] 實作 Combinatorial Purged Cross-Validation
- [ ] 加入 Deflated Sharpe Ratio（多重檢定校正）

### 風險管理
- [ ] 動態部位調整（Kelly Criterion）
- [ ] 相關性風險（多策略同時虧損）
- [ ] 黑天鵝事件壓力測試

### AI 自動化
- [ ] 多目標優化（Sharpe + MaxDD 同時優化）
- [ ] 策略組合優化（Portfolio of Strategies）
- [ ] 自動特徵工程（指標組合生成）

### 市場狀態偵測（Regime Detection）- 2026-01-13 決定

**開發順序：先策略，後狀態偵測**

```
Phase 1: 策略開發（先做）← 現在
├── 開發多種策略（趨勢、均值回歸、網格等）
├── 各策略獨立回測
└── 確認策略本身有效

Phase 2: 狀態偵測驗證（後做）
├── 開發 MarketStateAnalyzer（方向×波動矩陣）
├── RegimeValidator 驗證準確度（>60%）
└── 調整參數直到通過

Phase 3: 整合切換（最後）
├── 策略放進 StrategySwitch
└── 驗證切換是否提升績效
```

**原因：**
- 策略是核心，沒有策略則狀態偵測沒東西可切換
- 策略可獨立運作，即使不做狀態切換也能用
- RegimeValidator 是優化層，是「錦上添花」不是必要條件

**相關文件：** `.claude/skills/指標庫/references/regime-detection.md`

### 效能優化
- [x] **DataFrameOps 統一層**（2026-01-13）✅
  - 建立 `src/strategies/utils/dataframe_ops.py`
  - 支援 Pandas/Polars 雙格式
  - 自動 fallback 機制：Polars 失敗時降級為 Pandas
  - 基類方法已重構使用 DataFrameOps
- [ ] 向量化計算優化
- [ ] GPU 加速（CuPy/RAPIDS）
- [ ] 分散式回測（多幣種並行）
