# 趨勢策略模組

提供基於趨勢跟隨的交易策略。

## 可用策略

### 1. MA Cross（雙均線交叉策略）

**策略 ID**: `trend_ma_cross`

**策略邏輯**:
- 進場：快線向上穿越慢線（金叉）做多，向下穿越（死叉）做空
- 出場：反向穿越或 ATR 止損
- 過濾：可選趨勢過濾器（200MA）

**參數**:
| 參數 | 類型 | 預設值 | 說明 | 優化範圍 |
|------|------|--------|------|----------|
| fast_period | int | 10 | 快線週期 | 5-20 |
| slow_period | int | 30 | 慢線週期 | 20-60 |
| stop_loss_atr | float | 2.0 | 止損 ATR 倍數 | 1.0-4.0 |
| use_trend_filter | bool | False | 是否使用趨勢過濾 | - |
| trend_filter_period | int | 200 | 趨勢過濾週期 | - |

**適用市場**:
- ✅ 趨勢市場（trending market）
- ✅ 中長期持倉（4H, 1D 時間框）
- ❌ 震盪市場（容易產生假訊號）

**使用範例**:
```python
from src.strategies import create_strategy

# 建立策略
ma_cross = create_strategy(
    'trend_ma_cross',
    fast_period=12,
    slow_period=26,
    stop_loss_atr=2.5,
)

# 產生訊號
long_entry, long_exit, short_entry, short_exit = ma_cross.generate_signals(data)

# 計算止損
stop_loss = ma_cross.calculate_stop_loss(data, entry_price, 'long')
```

**優點**:
- 簡單易懂，邏輯清晰
- 參數少，容易優化
- 適合大週期趨勢跟隨

**缺點**:
- 震盪市場容易虧損
- 訊號延遲（MA 是滯後指標）
- 需要搭配趨勢過濾器

---

### 2. Supertrend（Supertrend 指標策略）

**策略 ID**: `trend_supertrend`

**策略邏輯**:
- 進場：Supertrend 方向翻轉時進場
- 出場：Supertrend 反向翻轉
- 止損：Supertrend 線本身作為動態止損

**參數**:
| 參數 | 類型 | 預設值 | 說明 | 優化範圍 |
|------|------|--------|------|----------|
| period | int | 10 | ATR 計算週期 | 7-14 |
| multiplier | float | 3.0 | ATR 倍數 | 2.0-4.0 |
| use_volume_filter | bool | False | 是否使用成交量過濾 | - |
| volume_ma_period | int | 20 | 成交量均線週期 | - |
| volume_threshold | float | 1.0 | 成交量閾值倍數 | - |

**適用市場**:
- ✅ 趨勢市場（trending market）
- ✅ 中短期持倉（1H, 4H 時間框）
- ⚠️ 震盪市場需搭配過濾器

**使用範例**:
```python
from src.strategies import create_strategy

# 建立策略
supertrend = create_strategy(
    'trend_supertrend',
    period=10,
    multiplier=3.0,
    use_volume_filter=True,
)

# 產生訊號
long_entry, long_exit, short_entry, short_exit = supertrend.generate_signals(data)

# 取得動態止損
current_stop = supertrend.get_stop_loss(data, 'long')

# 計算目標價格（2:1 風險回報比）
target = supertrend.calculate_target_price(
    entry_price,
    risk_reward_ratio=2.0,
    position_type='long',
    data=data
)
```

**優點**:
- 提供動態止損位置
- 適應波動性變化
- 視覺化容易理解

**缺點**:
- 震盪市場容易反覆進出
- 需要選擇合適的 multiplier
- ATR 計算需要足夠資料

**Supertrend 計算說明**:
```
1. 計算 ATR（平均真實波幅）
2. 計算基礎帶：
   - 上軌 = (H+L)/2 + multiplier × ATR
   - 下軌 = (H+L)/2 - multiplier × ATR
3. 趨勢判斷：
   - 價格突破上軌 → 上升趨勢（做多）
   - 價格跌破下軌 → 下降趨勢（做空）
4. Supertrend 線跟隨當前趨勢方向
```

---

## 策略比較

| 特性 | MA Cross | Supertrend |
|------|----------|------------|
| 複雜度 | 低 | 中 |
| 訊號數量 | 少 | 中等 |
| 止損方式 | 固定 ATR | 動態追蹤 |
| 適合週期 | 4H, 1D | 1H, 4H |
| 參數敏感度 | 低 | 中 |
| 震盪市表現 | 差 | 中 |

## 使用建議

### MA Cross 策略
1. **參數選擇**：
   - 快線：5-12（短期）
   - 慢線：20-30（中期）或 50-60（長期）
   - 大週期用大參數（4H 用 20/50，1D 用 50/200）

2. **改進方向**：
   - 加入趨勢過濾器（200MA）
   - 加入成交量確認
   - 加入 RSI 過濾超買超賣

### Supertrend 策略
1. **參數選擇**：
   - Period：7-10（靈敏）或 12-14（穩定）
   - Multiplier：2.5-3.0（一般）或 3.5-4.0（寬鬆）
   - 高波動市場用大 multiplier

2. **改進方向**：
   - 加入成交量過濾（避免假突破）
   - 搭配趨勢確認指標（ADX）
   - 設定固定止盈（如 2:1 RR）

## 回測建議

1. **資料準備**：
   - 至少 1 年歷史資料
   - 包含不同市場狀態（趨勢、震盪、崩盤）
   - 使用真實成交量資料

2. **評估指標**：
   - 總報酬率（Return）
   - 最大回撤（Max Drawdown）
   - 夏普比率（Sharpe Ratio）
   - 勝率（Win Rate）
   - 獲利因子（Profit Factor）

3. **參數優化**：
   - 使用 Walk-Forward 驗證
   - 避免過度擬合（保持參數簡單）
   - 測試參數穩健性

## 相關資源

- 策略開發指南：`.claude/skills/策略開發/SKILL.md`
- 指標庫參考：`.claude/skills/指標庫/SKILL.md`
- 使用範例：`examples/trend_strategies_example.py`
- 測試檔案：`test_trend_strategies.py`

## 技術支援

如需自訂策略或參數優化協助，請參考：
- BaseStrategy 文件：`src/strategies/base.py`
- TrendStrategy 基礎類別：`src/strategies/base.py`
- 策略註冊表：`src/strategies/registry.py`
