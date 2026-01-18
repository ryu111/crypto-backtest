# Workflow Hooks 優化報告

## 優化完成時間

2026-01-18

## 核心規則

1. ✅ **閾值要低** - 警告閾值設定要低，不要等到問題嚴重才提醒
2. ✅ **數值驗證** - 所有從 JSON 讀取的數值都要驗證是否為有效數字
3. ✅ **常數提取** - 所有 magic numbers 都要提取為常數
4. ✅ **類型檢查** - 使用 jq 時要檢查類型
5. ✅ **錯誤處理** - 使用 `2>/dev/null || echo "default"` 模式

## 優化檔案清單

### ✅ 已完成優化

| 檔案 | 狀態 | 主要改進 |
|------|------|----------|
| `workflow-violation-tracker.js` | 已優化 | 1. 閾值從 3 降為 1<br>2. 提取常數 `WARNING_THRESHOLD_EDITS`<br>3. 移除重複的 `MAX_INPUT_SIZE` 定義 |
| `loop-heartbeat.sh` | 已優化 | 1. 新增常數定義<br>2. 所有變數加入數值驗證<br>3. 預設值使用常數 |
| `loop-continue-reminder.sh` | 已優化 | 1. 提取 `CANCEL_PATTERN` 常數<br>2. 加入錯誤處理<br>3. 所有變數數值驗證 |
| `delegation-logger.sh` | 已優化 | 1. 提取 `PROMPT_PREVIEW_LENGTH` 常數<br>2. 全面錯誤處理<br>3. 數值驗證 |
| `task-executor-logger.sh` | 已優化 | 1. 加入 `SAFE_AGENT_TYPE` 驗證<br>2. 防止特殊字元注入 |

### ✅ 無需修改（已符合規範）

| 檔案 | 原因 |
|------|------|
| `remind-review.sh` | 已在之前優化過，完全符合規範 |
| `tech-debt-reminder.sh` | 簡單提醒 hook，無複雜邏輯 |
| `check-core-skill.sh` | 已有良好錯誤處理和驗證 |

## 詳細修改內容

### 1. workflow-violation-tracker.js

**變更摘要**：
- 將警告閾值從 `> 3` 降為 `> 1`（及早提醒）
- 新增常數定義區塊
- 移除重複的 `MAX_INPUT_SIZE` 定義（提升到頂部）

**關鍵改進**：
```javascript
// 閾值常數（設定要低，及早提醒）
const WARNING_THRESHOLD_EDITS = 1;  // 有 1 個未審查編輯就警告
const MAX_INPUT_SIZE = 1024 * 1024; // 1MB 限制
```

**影響**：
- 現在只要有 1 個未審查的編輯就會觸發警告
- 更符合 D→R→T 工作流的嚴格要求

---

### 2. loop-heartbeat.sh

**變更摘要**：
- 新增常數定義：`DEFAULT_STATUS`, `DEFAULT_LOOP_ID`
- 所有從檔案讀取的變數都加入空值檢查
- 使用常數作為預設值

**關鍵改進**：
```bash
# 常數定義
DEFAULT_STATUS="running"
DEFAULT_LOOP_ID="unknown"

# 數值驗證範例
if [ -z "$LOOP_ID" ]; then
    LOOP_ID="$DEFAULT_LOOP_ID"
fi
```

**影響**：
- 即使狀態檔案損壞或讀取失敗，也能優雅降級
- 避免空值導致的 JSON 格式錯誤

---

### 3. loop-continue-reminder.sh

**變更摘要**：
- 提取取消模式為常數 `CANCEL_PATTERN`
- 所有輸入讀取都加入錯誤處理
- 使用預設值避免空輸入

**關鍵改進**：
```bash
# 常數定義
CANCEL_PATTERN="(cancel|取消|停止|stop).*loop|/ralph-loop:cancel"

# 錯誤處理
INPUT=$(cat 2>/dev/null || echo "{}")
if [ -z "$INPUT" ]; then
    INPUT="{}"
fi
```

**影響**：
- 更容易維護取消指令模式
- 輸入失敗時不會導致 hook 錯誤

---

### 4. delegation-logger.sh

**變更摘要**：
- 新增 `PROMPT_PREVIEW_LENGTH` 常數
- 所有命令都加入 `2>/dev/null` 錯誤處理
- 提前退出避免無效記錄

**關鍵改進**：
```bash
# 常數定義
PROMPT_PREVIEW_LENGTH=100

# 數值驗證
if [ -z "$SUBAGENT_TYPE" ]; then
    exit 0
fi
```

**影響**：
- 避免記錄無效的委派事件
- 日誌更乾淨、更有意義

---

### 5. task-executor-logger.sh

**變更摘要**：
- 加入 `SAFE_AGENT_TYPE` 變數清理
- 防止環境變數注入特殊字元

**關鍵改進**：
```bash
# 清理特殊字元
SAFE_AGENT_TYPE=$(echo "$CLAUDE_AGENT_TYPE" | tr -cd '[:alnum:]_-')
EXECUTOR="subagent:$SAFE_AGENT_TYPE"
```

**影響**：
- 防止潛在的 JSON 格式錯誤
- 更安全的日誌記錄

---

## 測試建議

### 自動化測試

```bash
# 測試數值驗證
echo '{"tool_name": "Edit", "parameters": {}}' | ~/.claude/hooks/core/workflow-violation-tracker.js

# 測試空輸入
echo '' | ~/.claude/hooks/workflow/loop-continue-reminder.sh

# 測試損壞的 JSON
echo 'invalid json' | ~/.claude/hooks/workflow/delegation-logger.sh
```

### 功能測試

1. **workflow-violation-tracker.js**
   - [ ] 執行 1 個編輯，檢查是否觸發警告
   - [ ] 執行 Task(reviewer)，檢查待審查計數是否清零

2. **loop-heartbeat.sh**
   - [ ] 刪除 loop state 檔案，檢查是否使用預設值
   - [ ] 檢查生成的 JSON 格式是否正確

3. **loop-continue-reminder.sh**
   - [ ] 輸入「繼續」，檢查是否顯示提醒
   - [ ] 輸入「cancel loop」，檢查是否靜默退出

4. **delegation/task-executor logger**
   - [ ] 呼叫 Task(developer)，檢查日誌格式
   - [ ] 檢查日誌檔案權限和大小

---

## 統計資料

| 項目 | 數量 |
|------|------|
| 檢查的檔案 | 8 |
| 優化的檔案 | 5 |
| 無需修改 | 3 |
| 新增常數 | 7 |
| 新增數值驗證 | 15+ |
| 新增錯誤處理 | 12+ |

---

## 遵循的最佳實踐

✅ **閾值設定低**
- `WARNING_THRESHOLD_EDITS = 1` 而非 3
- `WARNING_THRESHOLD_LIGHT = 1` 而非 3

✅ **數值驗證無所不在**
```bash
if ! [[ "$VALUE" =~ ^[0-9]+$ ]]; then
    VALUE=0
fi
```

✅ **常數優於 Magic Number**
```bash
head -c "$PROMPT_PREVIEW_LENGTH"  # ✅
head -c 100                        # ❌
```

✅ **錯誤處理模式**
```bash
command 2>/dev/null || echo "default"
```

✅ **類型檢查（jq）**
```bash
jq -r 'if (.field | type) == "array" then ... else "0" end'
```

---

## 後續維護指南

### 新增 Hook 時的檢查清單

- [ ] 所有 magic numbers 提取為常數
- [ ] 所有外部輸入加入驗證
- [ ] 所有命令加入錯誤處理（`2>/dev/null || default`）
- [ ] 使用 jq 時檢查型別
- [ ] 閾值設定偏低（寧可多提醒）
- [ ] 空值情況有預設值
- [ ] 特殊字元有清理機制

### 範本

```bash
#!/bin/bash
# Hook Name
# 觸發時機：XXX

# 常數定義
CONST_VALUE=100
DEFAULT_VALUE="default"

# 讀取輸入（帶錯誤處理）
INPUT=$(cat 2>/dev/null || echo "{}")

# 數值驗證
if [ -z "$INPUT" ]; then
    exit 0
fi

# 提取資料
VALUE=$(echo "$INPUT" | jq -r '.field // "default"' 2>/dev/null || echo "$DEFAULT_VALUE")

# 再次驗證
if [ -z "$VALUE" ]; then
    VALUE="$DEFAULT_VALUE"
fi

# 執行邏輯
if [ "$VALUE" != "$DEFAULT_VALUE" ]; then
    # 處理...
fi
```

---

## 結論

所有 workflow hooks 已完成優化，符合以下標準：

1. ✅ 閾值設定低，及早警告
2. ✅ 完整的數值驗證
3. ✅ 所有 magic numbers 提取為常數
4. ✅ 錯誤處理完善
5. ✅ 使用預設值避免空值問題

系統現在更加健壯，能夠優雅處理各種邊界情況和錯誤輸入。
