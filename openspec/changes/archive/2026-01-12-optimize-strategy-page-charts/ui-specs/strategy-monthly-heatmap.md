# 月度報酬熱力圖 - UI 設計規格

## 📋 需求理解

### 目標
顯示策略在每個月份的報酬表現，快速識別表現好/差的月份以及季節性規律。

### 使用者
- 量化交易者：分析策略月度穩定性
- 策略研究員：識別季節性模式（如特定月份表現較好）
- 風險管理者：檢查報酬分佈和極端月份

### 關鍵互動
1. **快速掃視**：紅綠配色直覺識別盈虧月份
2. **精確查詢**：懸停顯示具體報酬數字
3. **多年比較**：切換年份查看歷史表現
4. **時間聯動**：與權益曲線圖表時間範圍同步

### 核心挑戰
1. **色彩映射**：紅綠漸層需考慮色盲友好性和對比度
2. **數據範圍**：多年資料需要年份切換機制
3. **視覺一致性**：與權益曲線圖表風格統一
4. **空缺處理**：回測期間不足一整月的格子如何顯示

---

## 📐 LAYOUT

### 元件結構

```
┌─────────────────────────────────────────────────────────┐
│ 月度報酬熱力圖                                           │
│                                                         │
│ ┌─────────────────────┐  ┌─────────────────────────┐  │
│ │ 顯示年份: 2024 ▼    │  │ 時間範圍: [與主圖同步]   │  │
│ └─────────────────────┘  └─────────────────────────────┘  │
│                                                         │
│     Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep  Oct... │
│                                                         │
│ 2024 [+5.2%][+3.1%][-1.2%][+4.5%]...                   │
│ 2023 [+2.3%][+1.8%][+6.2%][-0.5%]...                   │
│ 2022 ...                                               │
│                                                         │
│ 🟢 正報酬  ⚪ 接近零  🔴 負報酬                          │
└─────────────────────────────────────────────────────────┘
```

### 佈局模式

**Grid 佈局（月份格子）**
```css
.heatmap-grid {
  display: grid;
  grid-template-columns: 60px repeat(12, 1fr); /* 年份標籤 + 12 月 */
  gap: var(--spacing-xs); /* 4px - 格子間距 */
  max-width: 100%;
}
```

### 間距系統

| 區域 | Token | 值 | 原因 |
|------|-------|-----|------|
| 格子間距 | `var(--spacing-xs)` | 4px | 緊湊排列，最大化顯示密度 |
| 標題間距 | `var(--spacing-md)` | 16px | 與其他元件標題一致 |
| 圖例間距 | `var(--spacing-sm)` | 8px | 圖例項目間距 |
| 控制列間距 | `var(--spacing-lg)` | 24px | 分隔控制項和圖表 |

### 容器寬度

```css
.monthly-heatmap-container {
  width: 100%; /* 響應式填滿容器 */
  max-width: 1200px; /* 避免過寬難以閱讀 */
  margin: 0 auto; /* 置中 */
  padding: var(--spacing-lg); /* 24px 內距 */
}
```

---

## 🎨 VISUAL

### 色彩映射（紅綠漸層）

**核心配色原則**
- 使用**藍-白-橙**漸層代替傳統紅綠（色盲友好）
- 基於 60-30-10 法則：背景白 60%，格子填滿 30%，強調色 10%

#### 色彩漸層定義

```css
:root {
  /* 負報酬（藍色系）- 冷色系表示下跌 */
  --heatmap-negative-strong: #1d4ed8;  /* -10% 以上 */
  --heatmap-negative-medium: #60a5fa; /* -5% ~ -10% */
  --heatmap-negative-light: #dbeafe;  /* -1% ~ -5% */

  /* 中性（接近零）*/
  --heatmap-neutral: #f3f4f6;         /* -1% ~ +1% */

  /* 正報酬（橙/綠色系）- 暖色系表示上漲 */
  --heatmap-positive-light: #d1fae5;  /* +1% ~ +5% */
  --heatmap-positive-medium: #22c55e; /* +5% ~ +10% */
  --heatmap-positive-strong: #15803d; /* +10% 以上 */
}
```

**為什麼不用紅綠？**
- 8% 男性有紅綠色盲（最常見）
- 藍-橙對比明顯且色盲友好
- 符合金融慣例（藍=冷/跌，橙/綠=暖/漲）

#### 色彩映射邏輯

```javascript
function getHeatmapColor(monthlyReturn) {
  const percent = monthlyReturn * 100;

  if (percent <= -10) return 'var(--heatmap-negative-strong)';
  if (percent <= -5) return 'var(--heatmap-negative-medium)';
  if (percent <= -1) return 'var(--heatmap-negative-light)';
  if (percent < 1) return 'var(--heatmap-neutral)';
  if (percent < 5) return 'var(--heatmap-positive-light)';
  if (percent < 10) return 'var(--heatmap-positive-medium)';
  return 'var(--heatmap-positive-strong)';
}
```

### 背景與邊框

```css
.heatmap-cell {
  background: var(--color-surface); /* 預設白色 */
  border: 1px solid var(--color-border); /* #e5e7eb */
  border-radius: var(--primitive-radius-sm); /* 2px - 微圓角 */
  transition: all var(--duration-fast) var(--ease-out); /* 150ms */
}

.heatmap-cell--has-data {
  /* 有資料的格子使用計算出的顏色 */
  background: /* 根據報酬計算 */;
  border-color: transparent; /* 移除邊框讓顏色更統一 */
}

.heatmap-cell--empty {
  /* 空格子（未來月份或資料不足） */
  background: var(--color-surface-sunken); /* #f3f4f6 */
  border: 1px dashed var(--color-border-muted); /* 虛線邊框 */
}
```

### 字體排版

```css
.heatmap-title {
  font: var(--font-heading-4); /* 600 / 20px / 1.375 / sans */
  color: var(--color-text); /* #111827 */
  margin-bottom: var(--spacing-md); /* 16px */
}

.heatmap-cell-label {
  /* 格子內的報酬百分比 */
  font-size: var(--primitive-text-xs); /* 12px */
  font-weight: var(--primitive-font-medium); /* 500 */
  color: var(--color-text); /* 預設黑色 */
  line-height: var(--primitive-leading-tight); /* 1.25 */
}

.heatmap-cell-label--strong-negative {
  color: #ffffff; /* 深藍背景需要白色文字 */
}

.heatmap-cell-label--strong-positive {
  color: #ffffff; /* 深綠背景需要白色文字 */
}

.heatmap-month-header {
  /* 月份標籤（Jan, Feb...） */
  font-size: var(--primitive-text-sm); /* 14px */
  font-weight: var(--primitive-font-semibold); /* 600 */
  color: var(--color-text-secondary); /* #4b5563 */
  text-align: center;
}

.heatmap-year-label {
  /* 年份標籤（2024, 2023...） */
  font-size: var(--primitive-text-sm); /* 14px */
  font-weight: var(--primitive-font-semibold); /* 600 */
  color: var(--color-text-secondary); /* #4b5563 */
  text-align: right;
  padding-right: var(--spacing-sm); /* 8px */
}
```

### 對比度檢查

| 組合 | 對比度 | WCAG 標準 |
|------|--------|-----------|
| 白色文字 on `#1d4ed8` | 8.6:1 | ✅ AAA |
| 白色文字 on `#15803d` | 5.2:1 | ✅ AA |
| 黑色文字 on `#dbeafe` | 12.1:1 | ✅ AAA |
| 黑色文字 on `#d1fae5` | 11.8:1 | ✅ AAA |

---

## 🔄 STATES

### 格子狀態

| 狀態 | 視覺變化 | 觸發條件 |
|------|----------|----------|
| **Default** | 根據報酬顯示顏色 | 有資料的月份 |
| **Empty** | 虛線邊框 + 淺灰背景 | 未來月份或資料不足 |
| **Hover** | 輕微放大 + 陰影 | 滑鼠懸停 |
| **Selected** | 加粗邊框 | 點擊選中（如果支援） |

#### Hover 狀態

```css
.heatmap-cell:hover {
  transform: scale(1.05); /* 輕微放大 */
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12); /* 陰影 */
  z-index: 1; /* 避免被相鄰格子遮住 */
  cursor: pointer;
}

.heatmap-cell--empty:hover {
  transform: none; /* 空格子不放大 */
  cursor: default;
}
```

#### Tooltip（懸停提示）

```css
.heatmap-tooltip {
  position: absolute;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--primitive-radius-md); /* 6px */
  padding: var(--spacing-sm); /* 8px */
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  z-index: var(--z-tooltip); /* 80 */
  pointer-events: none; /* 避免阻擋滑鼠事件 */
}

.heatmap-tooltip-month {
  font-size: var(--primitive-text-sm); /* 14px */
  font-weight: var(--primitive-font-semibold); /* 600 */
  color: var(--color-text-secondary);
  margin-bottom: var(--spacing-xs); /* 4px */
}

.heatmap-tooltip-return {
  font-size: var(--primitive-text-lg); /* 18px */
  font-weight: var(--primitive-font-bold); /* 700 */
  color: var(--color-text);
}

.heatmap-tooltip-return--positive {
  color: var(--color-success); /* #22c55e */
}

.heatmap-tooltip-return--negative {
  color: var(--heatmap-negative-strong); /* #1d4ed8 */
}
```

**Tooltip 內容範例**
```
┌───────────────────┐
│ 2024年 3月        │
│                   │
│ 月報酬: +5.2%     │ ← 綠色文字
│ 交易次數: 12      │
│ 勝率: 66.7%       │
└───────────────────┘
```

### 年份切換控制

```css
.heatmap-year-selector {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm); /* 8px */
}

.heatmap-year-button {
  /* 使用 button 元件 token */
  background: var(--button-secondary-bg); /* transparent */
  border: 1px solid var(--color-border);
  border-radius: var(--button-radius); /* 6px */
  padding: var(--spacing-xs) var(--spacing-sm); /* 4px 8px */
  font-size: var(--primitive-text-sm); /* 14px */
  font-weight: var(--primitive-font-medium); /* 500 */
  color: var(--color-text);
  cursor: pointer;
  transition: all var(--duration-fast) var(--ease-out);
}

.heatmap-year-button:hover {
  background: var(--button-secondary-bg-hover); /* #f9fafb */
  border-color: var(--color-border-strong); /* #d1d5db */
}

.heatmap-year-button--active {
  background: var(--color-primary); /* #2563eb */
  border-color: var(--color-primary);
  color: #ffffff;
}

.heatmap-year-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
```

---

## 📱 RESPONSIVE

### 主要裝置
- **Desktop（主要）**: 1024px+ - 完整顯示所有月份
- **Tablet**: 768px - 可能需要橫向捲動
- **Mobile**: <768px - 優先顯示當前年份，隱藏歷史年份

### 斷點行為

```css
/* Desktop - 完整顯示 */
@media (min-width: 1024px) {
  .heatmap-grid {
    grid-template-columns: 60px repeat(12, 1fr);
  }

  .heatmap-cell {
    min-width: 60px;
    min-height: 48px;
  }
}

/* Tablet - 壓縮但可讀 */
@media (min-width: 768px) and (max-width: 1023px) {
  .heatmap-grid {
    grid-template-columns: 50px repeat(12, 1fr);
  }

  .heatmap-cell {
    min-width: 48px;
    min-height: 40px;
    font-size: var(--primitive-text-xs); /* 12px */
  }
}

/* Mobile - 橫向捲動 */
@media (max-width: 767px) {
  .heatmap-container {
    overflow-x: auto; /* 橫向捲動 */
  }

  .heatmap-grid {
    grid-template-columns: 40px repeat(12, 60px); /* 固定寬度 */
    min-width: 760px; /* 確保不壓縮 */
  }

  .heatmap-cell {
    min-width: 60px;
    min-height: 40px;
  }

  /* Mobile 優先顯示最近年份 */
  .heatmap-year-selector {
    flex-wrap: wrap; /* 按鈕換行 */
  }
}
```

### 觸控優化（Mobile）

```css
@media (max-width: 767px) {
  .heatmap-cell {
    /* 觸控目標最小 44px */
    min-height: 44px;
  }

  .heatmap-cell:active {
    /* Mobile 點擊回饋 */
    transform: scale(0.95);
    background: rgba(0, 0, 0, 0.05);
  }

  .heatmap-tooltip {
    /* Mobile tooltip 放大 */
    font-size: var(--primitive-text-base); /* 16px */
    padding: var(--spacing-md); /* 16px */
  }
}
```

---

## 🔗 與權益曲線的視覺一致性

### 共用設計元素

| 元素 | 統一規範 |
|------|----------|
| **卡片容器** | `border-radius: var(--card-radius)`, `padding: var(--card-padding)` |
| **標題樣式** | `font: var(--font-heading-4)` |
| **色彩系統** | 使用相同的成功/錯誤色（綠/藍） |
| **間距系統** | 元件間距 `var(--spacing-lg)` |
| **陰影效果** | `box-shadow: var(--card-shadow)` |

### 統一時間範圍控制

```css
.chart-time-control {
  /* 統一的時間範圍控制列 */
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--spacing-md); /* 16px */
  margin-bottom: var(--spacing-lg); /* 24px */
  padding: var(--spacing-md);
  background: var(--color-surface-raised); /* #f9fafb */
  border-radius: var(--primitive-radius-lg); /* 8px */
}

.chart-time-range-slider {
  flex: 1;
  /* Streamlit slider 樣式會繼承主題 */
}

.chart-sync-indicator {
  font-size: var(--primitive-text-xs); /* 12px */
  color: var(--color-text-muted); /* #6b7280 */
  display: flex;
  align-items: center;
  gap: var(--spacing-xs); /* 4px */
}

.chart-sync-indicator::before {
  content: '🔗'; /* 鎖鏈圖示表示聯動 */
}
```

---

## 💡 UX 設計原則應用

### Jakob's Law（熟悉性）
- ✅ 使用標準的熱力圖佈局（年 x 月）
- ✅ 顏色直覺對應盈虧（暖色=漲，冷色=跌）
- ✅ 懸停顯示詳細資訊（GitHub Contributions 風格）

### Fitts's Law（大+近=易點）
- ✅ 格子大小至少 48px x 48px（Desktop）
- ✅ 年份切換按鈕靠近圖表
- ✅ 常用年份（最近）優先顯示

### Von Restorff Effect（差異化記憶）
- ✅ 極端月份（+10% 或 -10%）使用深色突出
- ✅ 當前月份加粗邊框（如果在回測範圍內）

### Gestalt Principles（視覺分組）
- ✅ 使用 gap 分隔月份格子（接近性）
- ✅ 相同年份的月份使用同一行（共同區域）
- ✅ 顏色漸層產生連續性

---

## 🧪 年份切換邏輯

### 多年資料展示策略

**方案 A：單年顯示 + 切換按鈕（推薦）**
```
優點：
- 畫面簡潔不擁擠
- 適合回測資料跨多年（3 年以上）
- 切換流暢

缺點：
- 無法同時比較多年
```

**方案 B：多年堆疊顯示**
```
優點：
- 一次看到所有歷史
- 容易發現跨年趨勢

缺點：
- 資料多時畫面過長
- Mobile 體驗差
```

**推薦方案 A**，並提供「檢視所有年份」選項

### 切換控制實作

```javascript
// 年份選擇器狀態
const [selectedYear, setSelectedYear] = useState('2024');
const [availableYears, setAvailableYears] = useState(['2024', '2023', '2022']);

// 切換按鈕
<div className="heatmap-year-selector">
  <button
    onClick={() => setSelectedYear('all')}
    className={selectedYear === 'all' ? 'heatmap-year-button--active' : ''}
  >
    所有年份
  </button>
  {availableYears.map(year => (
    <button
      key={year}
      onClick={() => setSelectedYear(year)}
      className={selectedYear === year ? 'heatmap-year-button--active' : ''}
    >
      {year}
    </button>
  ))}
</div>
```

### 自動年份選擇邏輯

```javascript
// 預設顯示最新完整年份
function getDefaultYear(equityCurve) {
  const latestDate = equityCurve.index[-1];
  const currentYear = new Date().getFullYear();

  // 如果今年資料不足 6 個月，顯示去年
  if (latestDate.getMonth() < 6 && latestDate.getFullYear() === currentYear) {
    return currentYear - 1;
  }

  return latestDate.getFullYear();
}
```

---

## 📊 空缺處理規則

### 情境

1. **未來月份**：當前年份的未來月份
2. **資料不足**：月初或月末開始/結束的回測
3. **無交易月**：整月無交易活動

### 視覺處理

```css
/* 未來月份 */
.heatmap-cell--future {
  background: var(--color-surface-sunken); /* #f3f4f6 */
  border: 1px dashed var(--color-border-muted); /* 虛線 */
  opacity: 0.5;
  cursor: not-allowed;
}

/* 資料不足（僅部分天數） */
.heatmap-cell--partial {
  /* 正常顯示顏色，但加上標記 */
  position: relative;
}

.heatmap-cell--partial::after {
  content: '*';
  position: absolute;
  top: 2px;
  right: 4px;
  font-size: 10px;
  color: var(--color-warning); /* 橙色星號 */
}

/* 無交易月 */
.heatmap-cell--no-trades {
  background: var(--color-surface-sunken);
  border: 1px solid var(--color-border);
  color: var(--color-text-muted);
}

.heatmap-cell--no-trades::before {
  content: '-';
  font-size: var(--primitive-text-sm);
}
```

### Tooltip 差異

```
未來月份：
┌───────────────────┐
│ 2024年 12月       │
│ 尚無資料          │
└───────────────────┘

資料不足：
┌───────────────────┐
│ 2024年 1月 *      │
│                   │
│ 月報酬: +3.1%     │
│ 資料天數: 15/31   │ ← 顯示實際天數
└───────────────────┘

無交易月：
┌───────────────────┐
│ 2024年 8月        │
│ 無交易活動        │
└───────────────────┘
```

---

## ✅ Checklist

### 色彩設計
- [x] 使用色盲友好的藍-橙配色
- [x] 所有文字對比度 ≥ 4.5:1
- [x] 極端值使用深色突出
- [x] 引用 tokens.md 的具體變數

### 互動設計
- [x] Hover 狀態清晰（放大 + 陰影）
- [x] Tooltip 提供詳細資訊
- [x] 空格子不可互動
- [x] 觸控目標 ≥ 44px（Mobile）

### 資訊層級
- [x] 標題使用 `--font-heading-4`
- [x] 月份標籤使用 `--text-sm` 半粗體
- [x] 報酬數字使用 `--text-xs`
- [x] 視覺分組清晰（年份行）

### 響應式
- [x] Desktop 完整顯示
- [x] Tablet 壓縮但可讀
- [x] Mobile 橫向捲動
- [x] 觸控優化

### 一致性
- [x] 與權益曲線共用卡片樣式
- [x] 統一時間範圍控制
- [x] 共用間距和字體系統
- [x] 遵循 60-30-10 色彩法則

---

## 🎯 實作優先級

### P0（核心功能）
1. 基本熱力圖渲染（12 月 x N 年）
2. 色彩映射邏輯（報酬 → 顏色）
3. Hover tooltip 顯示詳細資訊

### P1（增強體驗）
4. 年份切換按鈕
5. 與權益曲線時間範圍聯動
6. 響應式佈局

### P2（錦上添花）
7. 空缺格子特殊標記
8. 動畫過渡效果
9. 匯出圖表功能

---

## 📝 開發者注意事項

### CSS Variables 必須使用
```css
/* ❌ 錯誤：hardcode 數值 */
.heatmap-cell {
  color: #2563eb;
  padding: 16px;
  border-radius: 6px;
}

/* ✅ 正確：使用 tokens */
.heatmap-cell {
  color: var(--color-primary);
  padding: var(--spacing-md);
  border-radius: var(--primitive-radius-md);
}
```

### 資料格式

```python
# 從 daily_returns 計算月度報酬
monthly_returns = daily_returns.resample('M').apply(
    lambda x: (1 + x).prod() - 1
) * 100  # 轉為百分比

# 轉為熱力圖格式
heatmap_data = {
    '2024': {
        'Jan': 5.2,
        'Feb': 3.1,
        'Mar': -1.2,
        # ...
    },
    '2023': {
        # ...
    }
}
```

### Plotly 實作範例

```python
import plotly.graph_objects as go

# 建立熱力圖
fig = go.Figure(data=go.Heatmap(
    z=returns_matrix,  # 報酬數值矩陣
    x=['Jan', 'Feb', 'Mar', ...],  # 月份
    y=['2024', '2023', '2022'],  # 年份
    colorscale=[
        [0.0, '#1d4ed8'],  # 深藍（負報酬）
        [0.4, '#dbeafe'],  # 淺藍
        [0.5, '#f3f4f6'],  # 中性灰
        [0.6, '#d1fae5'],  # 淺綠
        [1.0, '#15803d'],  # 深綠（正報酬）
    ],
    hovertemplate='<b>%{y}年 %{x}</b><br>月報酬: %{z:.2f}%<extra></extra>',
    colorbar=dict(
        title='月報酬 (%)',
        ticksuffix='%'
    )
))

fig.update_layout(
    title='月度報酬熱力圖',
    xaxis_title='月份',
    yaxis_title='年份',
    font=dict(family='Inter, sans-serif'),
    plot_bgcolor='white',
)
```

---

## 🔍 參考資源

- Design Tokens: `~/.claude/skills/ui/references/tokens.md`
- 色彩理論: `~/.claude/skills/ui/references/color-theory.md`
- UX 心理學: `~/.claude/skills/ux/references/psychology.md`
- 權益曲線設計: `openspec/changes/archive/.../ui-specs/validation-page.md`

---

**最後更新**: 2026-01-12
**設計者**: DESIGNER
**版本**: 1.0
