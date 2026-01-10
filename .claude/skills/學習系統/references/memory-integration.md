# Memory MCP 整合指南

## 概述

Memory MCP 提供持久化的語義記憶存儲，用於跨專案、跨對話的知識累積。

## 核心操作

### 存儲記憶

```python
# 存儲策略洞察
await mcp__memory_service__store_memory(
    content="""
    MA 交叉策略 BTC 4H 最佳實踐：
    - Fast Period: 10
    - Slow Period: 30
    - 止損: ATR 2x
    - Sharpe Ratio: 1.85
    - WFA Efficiency: 68%
    - 驗證等級: A
    """,
    metadata={
        "tags": "crypto,btc,strategy,ma,trend,validated",
        "type": "trading-insight"
    }
)
```

### 檢索記憶

```python
# 語義搜尋
results = await mcp__memory_service__retrieve_memory(
    query="BTC 趨勢策略 最佳參數",
    n_results=5
)

# 標籤搜尋
results = await mcp__memory_service__search_by_tag(
    tags=["crypto", "validated", "trend"]
)

# 時間範圍
results = await mcp__memory_service__recall_memory(
    query="上週的回測實驗"
)
```

### 更新記憶

```python
# 更新標籤
await mcp__memory_service__update_memory_metadata(
    content_hash="abc123...",
    updates={
        "tags": ["crypto", "btc", "validated", "production"]
    }
)

# 評分記憶
await mcp__memory_service__rate_memory(
    content_hash="abc123...",
    rating=1,  # 1=好, 0=中, -1=差
    feedback="策略實盤驗證有效"
)
```

## 標籤系統

### 標籤分類

| 類別 | 標籤 | 用途 |
|------|------|------|
| 資產 | crypto, btc, eth | 資產類型 |
| 策略 | trend, momentum, mean-reversion | 策略類型 |
| 時間 | 1h, 4h, 1d | 時間框架 |
| 狀態 | validated, testing, failed | 驗證狀態 |
| 類型 | insight, lesson, parameter | 內容類型 |

### 命名規範

```
# 策略洞察
crypto,{asset},{strategy-type},{timeframe},validated

# 市場洞察
crypto,{asset},market,{topic}

# 失敗教訓
crypto,backtest,{topic},lesson
```

## 工作流整合

### 回測後存儲

```python
async def post_backtest_storage(result, insights):
    """回測後存儲洞察到 Memory"""

    if result['validation']['grade'] in ['A', 'B']:
        # 存儲驗證通過的策略
        content = f"""
        策略: {result['strategy']['name']}
        標的: {result['config']['symbol']}
        時間框架: {result['config']['timeframe']}

        最佳參數:
        {format_params(result['parameters'])}

        績效:
        - Sharpe: {result['results']['sharpe_ratio']:.2f}
        - Return: {result['results']['total_return']:.2%}
        - Max DD: {result['results']['max_drawdown']:.2%}

        驗證:
        - WFA Efficiency: {result['validation']['walk_forward_efficiency']:.2%}
        - Grade: {result['validation']['grade']}

        洞察:
        {chr(10).join('- ' + i for i in insights)}
        """

        await mcp__memory_service__store_memory(
            content=content,
            metadata={
                "tags": generate_tags(result),
                "type": "validated-strategy"
            }
        )
```

### 優化前查詢

```python
async def pre_optimization_query(strategy_type, symbol):
    """優化前查詢歷史經驗"""

    # 查詢相似策略
    results = await mcp__memory_service__retrieve_memory(
        query=f"{strategy_type} {symbol} 最佳參數 validated",
        n_results=10
    )

    # 萃取參數範圍
    param_hints = extract_params_from_results(results)

    return param_hints
```

### 失敗記錄

```python
async def log_failure_lesson(experiment, failure_reason):
    """記錄失敗教訓"""

    content = f"""
    失敗實驗: {experiment['id']}
    策略: {experiment['strategy']['name']}

    問題:
    {failure_reason}

    避免方式:
    - [根據分析填寫]

    相關指標:
    - 交易次數: {experiment['results']['total_trades']}
    - IS/OOS 比: {experiment['is_oos_ratio']:.2f}
    """

    await mcp__memory_service__store_memory(
        content=content,
        metadata={
            "tags": "crypto,backtest,lesson,overfitting",
            "type": "failure-lesson"
        }
    )
```

## 查詢模式

### 模式 1：策略開發

```python
# 開發新策略前，查詢現有經驗
query = "RSI 均值回歸 ETH 參數 驗證"
```

### 模式 2：問題排查

```python
# 遇到問題時，查詢類似教訓
query = "過擬合 IS OOS 差距大 解決"
```

### 模式 3：參數優化

```python
# 優化參數前，查詢歷史最佳值
query = "MA 交叉 fast period 最佳範圍"
```

## 最佳實踐

### 存儲

1. **結構化內容**：使用清晰的格式
2. **充足標籤**：多維度標記
3. **量化數據**：包含具體數字
4. **時間戳**：自動記錄

### 檢索

1. **語義優先**：用自然語言描述需求
2. **標籤輔助**：用標籤縮小範圍
3. **評分參考**：優先高評分記憶

### 維護

1. **定期清理**：刪除過時記憶
2. **評分更新**：根據實際效果調整
3. **標籤統一**：保持標籤一致性

## 與專案 JSON 的分工

| 面向 | Memory MCP | 專案 JSON |
|------|------------|-----------|
| 用途 | 跨專案洞察 | 詳細實驗數據 |
| 查詢 | 語義搜尋 | 精確查詢 |
| 持久性 | 永久 | 版本控制 |
| 內容 | 精煉洞察 | 完整記錄 |
