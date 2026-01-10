---
name: learning-system
description: AI 學習記錄系統。記錄實驗、萃取洞察、持續優化。當需要記錄回測結果、查詢歷史經驗、分析優化模式時使用。
---

# AI 學習記錄系統

記錄實驗、萃取洞察、持續優化的知識管理系統。

## 雙軌記錄架構

| 存儲位置 | 用途 | 格式 | 查詢方式 |
|----------|------|------|----------|
| Memory MCP | 跨專案洞察 | 語義化文字 | 語義搜尋 |
| 專案 JSON/MD | 詳細實驗數據 | 結構化 JSON | 精確查詢 |

## 學習循環

```
執行回測
    ↓
記錄實驗（JSON）
    ↓
分析績效
    ↓
萃取洞察（MD）
    ↓
存入 Memory MCP
    ↓
下次優化前查詢
    ↓
應用歷史洞察
    ↓
執行回測（循環）
```

## 實驗記錄格式

### JSON 結構

```json
{
  "id": "exp_20260111_001",
  "timestamp": "2026-01-11T10:30:00Z",

  "strategy": {
    "name": "trend_ma_cross_4h_v2",
    "type": "trend_following",
    "version": "2.0"
  },

  "config": {
    "symbol": "BTCUSDT",
    "timeframe": "4h",
    "period": {
      "start": "2024-01-01",
      "end": "2025-12-31"
    },
    "initial_capital": 10000,
    "leverage": 5
  },

  "parameters": {
    "fast_period": 10,
    "slow_period": 30,
    "stop_loss_atr": 2.0
  },

  "results": {
    "total_return": 0.456,
    "sharpe_ratio": 1.85,
    "max_drawdown": 0.10,
    "win_rate": 0.55,
    "profit_factor": 1.72,
    "total_trades": 124
  },

  "validation": {
    "walk_forward_efficiency": 0.68,
    "monte_carlo_p5": 0.18,
    "passed_stages": 5,
    "grade": "A"
  },

  "insights": [
    "止損 2x ATR 在高波動期間表現更好",
    "慢線 30 優於 20，減少假訊號"
  ],

  "tags": ["trend", "ma", "btc", "4h", "validated"],

  "parent_experiment": "exp_20260105_003",
  "improvement": 0.12
}
```

### 記錄程式碼

```python
import json
from datetime import datetime

def log_experiment(result, strategy_info, config, insights=None):
    """記錄實驗結果"""

    # 讀取現有記錄
    with open('learning/experiments.json', 'r') as f:
        data = json.load(f)

    # 建立新實驗記錄
    exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    experiment = {
        'id': exp_id,
        'timestamp': datetime.now().isoformat(),
        'strategy': strategy_info,
        'config': config,
        'parameters': result.get('parameters', {}),
        'results': {
            'total_return': result['total_return'],
            'sharpe_ratio': result['sharpe_ratio'],
            'max_drawdown': result['max_drawdown'],
            'win_rate': result['win_rate'],
            'profit_factor': result['profit_factor'],
            'total_trades': result['total_trades']
        },
        'validation': result.get('validation', {}),
        'insights': insights or [],
        'tags': generate_tags(strategy_info, config)
    }

    # 添加到記錄
    data['experiments'].append(experiment)
    data['metadata']['total_experiments'] += 1
    data['metadata']['last_updated'] = datetime.now().isoformat()

    # 更新最佳策略
    if is_better(experiment, data['metadata'].get('best_strategy')):
        data['metadata']['best_strategy'] = exp_id

    # 儲存
    with open('learning/experiments.json', 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return exp_id
```

## 洞察格式

### Markdown 結構

```markdown
# 交易策略洞察彙整

> 最後更新：2026-01-11
> 總實驗數：156

## 策略類型洞察

### 趨勢策略

#### MA 交叉
- **最佳時間框架**：4H
- **最佳參數範圍**：Fast 8-12, Slow 25-35（比例 1:3）
- **適用市場**：趨勢明確時期
- **失效情境**：長期震盪

#### Supertrend
- **最佳參數**：Period 10, Multiplier 3.0
- **優勢**：內建動態止損

### 動量策略

#### RSI
- 單用 RSI 效果差，需配合趨勢濾網
- 最佳組合：RSI + MA 方向確認

## 標的特性

### BTCUSDT
- 趨勢策略表現較好
- 4H 時間框架最穩定
- 極端行情需擴大止損

### ETHUSDT
- 波動性更高
- 與 BTC 相關性高

## 風險管理洞察

### 止損
- ATR 2x 優於固定百分比
- 極端行情需 ATR 3x

### 部位大小
- 單筆 1-2% 風險
- Kelly 的 1/4 最安全

## 過擬合教訓

### 失敗案例
1. exp_20260103_021：參數過度優化
2. exp_20260108_045：交易次數過少

### 避免方法
- 交易次數 >= 30
- WFA Efficiency >= 50%
```

### 更新洞察

```python
def update_insights(experiment, insights_file='learning/insights.md'):
    """更新洞察文件"""

    # 讀取現有洞察
    with open(insights_file, 'r') as f:
        content = f.read()

    # 解析並更新相關區塊
    # ...（根據實驗類型更新對應區塊）

    # 儲存
    with open(insights_file, 'w') as f:
        f.write(content)
```

## Memory MCP 整合

### 存儲重要洞察

```python
# 策略洞察
await mcp__memory_service__store_memory(
    content="""
    MA 交叉策略最佳實踐：
    - 時間框架：4H
    - 參數：Fast 10, Slow 30
    - 止損：ATR 2x
    - Sharpe: 1.85, WFA Efficiency: 68%
    """,
    metadata={
        "tags": "crypto,backtest,strategy,ma,trend",
        "type": "trading-insight"
    }
)

# 市場洞察
await mcp__memory_service__store_memory(
    content="""
    BTC 永續合約特性：
    - 資金費率 > 0.1% 時常見短期回調
    - 週末流動性差，假突破多
    """,
    metadata={
        "tags": "crypto,btc,perpetual,market",
        "type": "market-insight"
    }
)

# 失敗教訓
await mcp__memory_service__store_memory(
    content="""
    過擬合教訓：
    - 交易次數 < 30 統計無效
    - IS/OOS 比 > 2 需警惕
    """,
    metadata={
        "tags": "crypto,backtest,overfitting,lesson",
        "type": "trading-lesson"
    }
)
```

### 查詢歷史經驗

```python
# 查詢相關策略經驗
results = await mcp__memory_service__retrieve_memory(
    query="MA 交叉 BTC 4H 參數"
)

# 查詢特定標籤
results = await mcp__memory_service__search_by_tag(
    tags=["crypto", "trend", "validated"]
)

# 查詢近期記錄
results = await mcp__memory_service__recall_memory(
    query="上週的回測結果"
)
```

## 學習分析

### 策略演進追蹤

```python
def analyze_strategy_evolution(experiments, strategy_name):
    """分析策略版本演進"""

    # 過濾相關實驗
    related = [e for e in experiments
               if e['strategy']['name'].startswith(strategy_name)]

    # 按時間排序
    related.sort(key=lambda x: x['timestamp'])

    evolution = []
    for i, exp in enumerate(related):
        entry = {
            'version': exp['strategy'].get('version', i+1),
            'date': exp['timestamp'],
            'sharpe': exp['results']['sharpe_ratio'],
            'return': exp['results']['total_return'],
            'changes': exp.get('insights', [])
        }

        if i > 0:
            prev = related[i-1]
            entry['improvement'] = (
                exp['results']['sharpe_ratio'] -
                prev['results']['sharpe_ratio']
            )

        evolution.append(entry)

    return evolution
```

### 參數趨勢分析

```python
def analyze_parameter_trends(experiments, param_name):
    """分析參數最佳值趨勢"""

    data = []
    for exp in experiments:
        if param_name in exp.get('parameters', {}):
            data.append({
                'value': exp['parameters'][param_name],
                'sharpe': exp['results']['sharpe_ratio'],
                'date': exp['timestamp']
            })

    # 找出表現最佳的參數範圍
    best = max(data, key=lambda x: x['sharpe'])

    return {
        'best_value': best['value'],
        'best_sharpe': best['sharpe'],
        'value_range': (min(d['value'] for d in data),
                       max(d['value'] for d in data)),
        'trend': calculate_trend(data)
    }
```

## 使用流程

### 回測後記錄

```python
# 1. 執行回測
result = run_backtest(strategy, data, params)

# 2. 驗證策略
validation = validate_strategy(result)

# 3. 萃取洞察
insights = extract_insights(result, validation)

# 4. 記錄實驗
exp_id = log_experiment(result, strategy_info, config, insights)

# 5. 存入 Memory（重要洞察）
if validation['grade'] in ['A', 'B']:
    store_to_memory(insights)

# 6. 更新洞察文件
update_insights_file(insights)
```

### 優化前查詢

```python
# 1. 查詢相關歷史
history = query_experiments(strategy_type='trend', symbol='BTCUSDT')

# 2. 查詢 Memory 洞察
memory_insights = retrieve_memory("MA 交叉 BTC 最佳參數")

# 3. 分析歷史趨勢
trends = analyze_parameter_trends(history, 'slow_period')

# 4. 設定優化範圍
param_range = {
    'slow_period': (trends['best_value'] - 5, trends['best_value'] + 5)
}

# 5. 執行優化
best_params = optimize(strategy, data, param_range)
```

## 標籤系統

| 類別 | 標籤範例 |
|------|----------|
| 資產 | btc, eth, crypto |
| 策略類型 | trend, momentum, mean-reversion |
| 時間框架 | 1h, 4h, 1d |
| 驗證狀態 | validated, testing, failed |
| 市場狀態 | bull, bear, sideways |

For Memory 整合詳解 → read `references/memory-integration.md`
For 實驗追蹤詳解 → read `references/experiment-tracking.md`
