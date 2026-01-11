# 合約交易回測系統 - 專案配置

## 專案概述

BTC/ETH 永續合約 AI 自動化回測系統。

## 自動學習機制

### 回測後自動記錄

**每次執行回測後，必須自動判斷是否有新洞察並更新 `learning/insights.md`**

```
回測完成
    ↓
分析結果是否有新發現？
    │
    ├── 有新洞察 → 更新 learning/insights.md
    │   - 策略表現異常好/差的原因
    │   - 參數敏感度發現
    │   - 市場狀態影響
    │   - 過擬合警訊
    │   - 風險管理教訓
    │
    └── 無新發現 → 不需更新
```

### 應記錄的洞察類型

| 類型 | 觸發條件 | 記錄位置 |
|------|----------|----------|
| 策略洞察 | Sharpe > 2.0 或 < 0.5 | `## 策略類型洞察` |
| 標的特性 | 特定幣種表現異常 | `## 標的特性` |
| 風險發現 | MaxDD > 25% 或強平事件 | `## 風險管理洞察` |
| 過擬合警訊 | Monte Carlo 失敗率 > 30% | `## 過擬合教訓` |
| 參數發現 | 穩健性測試揭示敏感參數 | `## 策略類型洞察 > 對應策略` |

### 記錄格式範例

```markdown
#### [策略名稱]_[版本]
- **最佳參數**：param1=X, param2=Y
- **績效**：Sharpe X.XX, Return XX.X%, MaxDD XX.X%
- **洞察**：[具體發現]
- **日期**：YYYY-MM-DD
```

### 自動判斷邏輯

```python
def should_record_insight(result):
    """判斷是否值得記錄"""

    # 異常好的表現
    if result.sharpe > 2.0:
        return True, "exceptional_performance"

    # 異常差的表現（可能有 bug 或市場變化）
    if result.sharpe < 0.5 and result.expected_sharpe > 1.0:
        return True, "unexpected_poor_performance"

    # 過擬合警訊
    if result.overfit_probability > 0.3:
        return True, "overfit_warning"

    # 風險事件
    if result.max_drawdown > 0.25:
        return True, "risk_event"

    # 參數敏感度發現
    if result.robustness_variance > 0.5:
        return True, "parameter_sensitivity"

    return False, None
```

## 10 個 Skills 位置

```
.claude/skills/
├── 資料管道/SKILL.md
├── 指標庫/SKILL.md
├── 策略開發/SKILL.md
├── 永續合約/SKILL.md
├── 參數優化/SKILL.md
├── 策略驗證/SKILL.md
├── 風險管理/SKILL.md
├── 學習系統/SKILL.md
├── 回測核心/SKILL.md
└── AI自動化/SKILL.md
```

## 關鍵檔案

| 檔案 | 用途 |
|------|------|
| `learning/insights.md` | 策略洞察彙整（自動更新） |
| `learning/experiments.json` | 實驗記錄（機器可讀） |
| `src/learning/recorder.py` | 實驗記錄器 |
| `src/learning/memory.py` | Memory MCP 整合 |
| `src/automation/loop.py` | AI 自動化循環 |

## 回測工作流

```
1. 選擇策略（80% exploit / 20% explore）
2. 生成參數（Bayesian 優化或隨機）
3. 執行 5 階段驗證
4. 記錄結果到 experiments.json
5. 【新增】判斷是否更新 insights.md
6. 更新策略評級
7. 繼續下一輪
```
