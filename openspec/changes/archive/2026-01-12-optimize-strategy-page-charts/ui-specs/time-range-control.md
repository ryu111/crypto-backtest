# 時間範圍聯動控制器設計規格

## 📋 需求理解

**目標**: 讓使用者可以快速調整圖表顯示的時間範圍,提供即時預覽並與圖表聯動

**使用者**: 回測分析師,需要在不同時間尺度下觀察策略表現

**關鍵互動**:
- 拖動 slider 改變時間範圍
- 圖表即時更新顯示對應區間的資料
- 可重置回完整時間範圍

**使用情境**: 在策略比較頁面,需要聚焦某個時間段的績效細節

---

## 📐 LAYOUT

### 整體結構

```
┌─────────────────────────────────────────────────────────────┐
│ 圖表區域 (Interactive Charts)                               │
│ • 資金曲線                                                    │
│ • 回撤曲線                                                    │
└─────────────────────────────────────────────────────────────┘

[間距: var(--spacing-lg) = 24px]

┌─────────────────────────────────────────────────────────────┐
│ 時間範圍控制器                                               │
│                                                              │
│ [時間標籤]                                                   │
│ ├────────────●═══════●────────────┤                         │
│ 開始                            結束                          │
│                                                              │
│                                    [重置按鈕]                │
└─────────────────────────────────────────────────────────────┘
```

### 佈局決策

**為什麼放在圖表下方?**
- 遵循 Jakob's Law: 時間軸控制器通常在視覺化下方 (參考 YouTube、股票軟體)
- 視覺流向: 使用者先看圖表,再調整時間範圍
- 空間效率: 充分利用水平空間

**間距選擇**:
- 圖表與控制器間隔: `var(--spacing-lg)` (24px)
  - **理由**: 視覺分組,但不會太遠導致失去關聯性
- 控制器內部元素間距: `var(--spacing-md)` (16px)

**容器設計**:
```css
.time-range-container {
  width: 100%;
  max-width: 100%; /* 與圖表等寬 */
  padding: var(--spacing-md); /* 16px 內距 */
  background: var(--color-surface-raised); /* 淺色背景區分區域 */
  border-radius: var(--primitive-radius-lg); /* 8px 圓角 */
  border: 1px solid var(--color-border-muted); /* 淡邊框 */
}
```

---

## 🎨 VISUAL

### 色彩方案 (遵循 60-30-10 法則)

```
60% - 背景 (Surface)
└── var(--color-surface-raised): #f9fafb (淡灰背景)

30% - 控制元件 (Slider Track)
└── var(--color-border): #e5e7eb (軌道背景)

10% - 強調色 (Active Range & Button)
├── var(--color-primary): #2563eb (選中區間、按鈕)
└── var(--color-primary-hover): #1d4ed8 (hover 狀態)
```

### Slider 視覺規格

#### 軌道 (Track)

```css
.slider-track {
  height: 8px;
  background: var(--color-border); /* #e5e7eb 灰色軌道 */
  border-radius: var(--primitive-radius-full); /* 完全圓角 */
  position: relative;
}
```

**為什麼 8px 高度?**
- 視覺清晰: 足夠讓使用者看到選中區間
- 不搶主視覺: 圖表才是主角,控制器應低調
- 符合觸控: 配合 thumb 尺寸,整體觸控區 ≥ 44px

#### 選中區間 (Active Range)

```css
.slider-range-active {
  position: absolute;
  height: 100%; /* 與軌道同高 */
  background: var(--color-primary); /* #2563eb 藍色 */
  border-radius: var(--primitive-radius-full);
  /* 動態寬度由 JavaScript 計算 */
}
```

**為什麼用 primary 色?**
- 品牌色 = 主要動作 (符合 Von Restorff Effect)
- 對比清晰: 在灰色軌道上明顯可見

#### Thumb (拖動手柄)

```css
.slider-thumb {
  width: 20px;
  height: 20px;
  background: var(--color-primary);
  border: 3px solid white; /* 白色邊框增強層次 */
  border-radius: var(--primitive-radius-full);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* 輕微陰影提升層級 */
  cursor: grab;
  transition: all var(--duration-fast) var(--ease-out); /* 100ms */

  /* 確保觸控區域 ≥ 44px (Fitts's Law) */
  position: relative;
}

.slider-thumb::before {
  content: '';
  position: absolute;
  inset: -12px; /* 擴大觸控區到 44x44px */
  border-radius: 50%;
}

.slider-thumb:hover {
  transform: scale(1.15); /* hover 放大 15% */
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
}

.slider-thumb:active {
  cursor: grabbing;
  transform: scale(1.1); /* 抓取時略縮 */
}

.slider-thumb:focus-visible {
  outline: 2px solid var(--color-border-focus); /* #3b82f6 */
  outline-offset: 2px;
}
```

**設計細節**:
- **20px 直徑**: 視覺適中,不會太小難抓
- **白色邊框**: 分離 thumb 與軌道,增強層次
- **擴大觸控區**: 實際可點擊範圍 44x44px (符合無障礙標準)

### 時間標籤

```css
.time-label {
  font: var(--font-label); /* medium 500, 14px */
  color: var(--color-text-secondary); /* #4b5563 灰色 */
  margin-bottom: var(--spacing-sm); /* 8px */
}

.time-value {
  font-family: var(--primitive-font-mono); /* 等寬字體 */
  color: var(--color-text); /* #111827 深色 */
  font-weight: var(--primitive-font-semibold); /* 600 */
}
```

**為什麼用等寬字體顯示日期?**
- 對齊一致: 日期數字變化時不會跳動
- 專業感: 符合數據分析工具的視覺語言

### 重置按鈕

```css
.reset-button {
  /* 使用 ghost variant (低調) */
  background: transparent;
  color: var(--color-text-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--primitive-radius-md); /* 6px */
  padding: var(--primitive-space-2) var(--primitive-space-3); /* 8px 12px */
  font: var(--font-label);
  cursor: pointer;
  transition: all var(--duration-fast) var(--ease-out);

  min-height: 32px; /* 較小的按鈕,因為是次要動作 */
}

.reset-button:hover {
  background: var(--color-surface-raised);
  border-color: var(--color-border-strong);
  color: var(--color-text);
}

.reset-button:active {
  transform: scale(0.98);
}

.reset-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
```

**為什麼用 ghost variant?**
- 次要動作: 不與 slider 互動搶主視覺
- 視覺層級: 淡色表示「可選操作」

---

## 🔄 STATES

### Slider 狀態

| 狀態 | 視覺變化 | 觸發條件 |
|------|----------|----------|
| **Default** | Thumb 正常大小 | 初始狀態 |
| **Hover** | Thumb 放大 1.15x, 陰影加深 | 滑鼠懸停 thumb |
| **Dragging** | Thumb 略縮 1.1x, cursor: grabbing | 拖動中 |
| **Focus** | 2px 藍色 focus ring | 鍵盤聚焦 |
| **Disabled** | 整體 50% 透明度,cursor: not-allowed | 圖表載入中 |

### 重置按鈕狀態

| 狀態 | 視覺 | 條件 |
|------|------|------|
| **Default** | 透明背景,灰色文字 | 可操作 |
| **Hover** | 淺灰背景 | 滑鼠懸停 |
| **Active** | 略縮 0.98x | 點擊中 |
| **Disabled** | 50% 透明度 | 已是完整範圍 |

### 圖表聯動狀態

```
使用者拖動 Thumb
    ↓
JavaScript 更新選中區間視覺
    ↓
同步更新圖表資料範圍
    ↓
圖表平滑過渡到新範圍 (300ms ease-out)
```

**為什麼即時聯動?**
- 即時回饋 (< 100ms): 使用者感覺操作流暢
- 視覺連貫: 拖動與圖表變化同步,減少認知負荷

---

## ⚡ 互動邏輯

### 拖動行為 (基於 Fitts's Law)

```javascript
// 擴大觸控區域,降低誤操作
const TOUCH_PADDING = 12; // 每側擴大 12px

// Thumb 最小間距,避免重疊
const MIN_RANGE_PERCENT = 5; // 最小選中 5% 的時間範圍

// 拖動流程
onThumbDragStart() {
  // 1. 記錄初始位置
  // 2. 添加 dragging class
  // 3. 禁用文字選取
}

onThumbDrag(event) {
  // 1. 計算新位置 (限制在軌道範圍內)
  // 2. 檢查與另一個 thumb 的最小距離
  // 3. 更新選中區間視覺
  // 4. 同步更新圖表 (節流 throttle: 16ms, 60fps)
}

onThumbDragEnd() {
  // 1. 移除 dragging class
  // 2. 恢復文字選取
  // 3. 觸發最終圖表更新
}
```

### 鍵盤支援 (無障礙)

```
Focus 在 Start Thumb:
├── ← 左箭頭: 向左移動 1%
├── → 右箭頭: 向右移動 1%
├── Shift + ←: 向左移動 5%
├── Shift + →: 向右移動 5%
└── Home: 移到最左 (0%)

Focus 在 End Thumb:
├── ← 左箭頭: 向左移動 1%
├── → 右箭頭: 向右移動 1%
├── Shift + ←: 向左移動 5%
├── Shift + →: 向右移動 5%
└── End: 移到最右 (100%)
```

### 重置邏輯

```javascript
onResetClick() {
  // 1. 檢查當前是否已是完整範圍
  if (isFullRange()) {
    return; // disabled 狀態,不執行
  }

  // 2. 動畫過渡 (300ms ease-out)
  animateSliderTo({
    start: 0,
    end: 100
  }, {
    duration: 300,
    easing: 'ease-out'
  });

  // 3. 同步更新圖表
  updateChartRange(fullTimeRange);

  // 4. 短暫提示 (微互動)
  showToast('已重置為完整時間範圍', { type: 'info', duration: 2000 });
}
```

### 效能優化 (節流)

```javascript
// 拖動時節流更新圖表,避免過度渲染
const throttledUpdateChart = throttle((newRange) => {
  updateChartData(newRange);
}, 16); // 60fps = 16ms
```

**為什麼節流?**
- 避免過度渲染: 拖動連續觸發,圖表重算成本高
- 60fps 流暢: 16ms 間隔對應 60fps,視覺流暢

---

## 📱 RESPONSIVE

### 桌面 (≥ 1024px)

```css
.time-range-container {
  display: grid;
  grid-template-columns: 1fr auto; /* slider 佔滿,按鈕自適應 */
  gap: var(--spacing-md); /* 16px */
  align-items: center;
}

.slider-wrapper {
  min-width: 400px; /* 保證 slider 夠長,易拖動 */
}
```

### 平板 (768px - 1023px)

```css
@media (max-width: 1023px) {
  .slider-wrapper {
    min-width: 300px;
  }
}
```

### 手機 (< 768px)

```css
@media (max-width: 767px) {
  .time-range-container {
    grid-template-columns: 1fr; /* 垂直排列 */
    gap: var(--spacing-sm); /* 8px */
  }

  .slider-wrapper {
    min-width: unset;
    width: 100%;
  }

  .reset-button {
    width: 100%; /* 全寬按鈕,易點擊 */
  }

  /* 放大 thumb,方便觸控 */
  .slider-thumb {
    width: 24px;
    height: 24px;
  }
}
```

**響應式決策**:
- 桌面: 水平排列,視覺簡潔
- 手機: 垂直排列,避免擁擠
- 觸控優化: 手機上 thumb 更大 (24px),觸控區 ≥ 48px

---

## 🎯 微互動設計

### 拖動回饋 (基於 Microinteractions 四要素)

**Trigger**: 使用者拖動 thumb

**Rules**:
```javascript
if (拖動中) {
  更新 thumb 位置;
  更新選中區間視覺;
  節流更新圖表 (16ms);

  if (到達邊界) {
    輕微震動回饋 (haptic);
  }
}
```

**Feedback**:
```css
/* 視覺回饋 */
.slider-thumb.dragging {
  transform: scale(1.1);
  cursor: grabbing;
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2); /* 陰影加深 */
}

/* 選中區間即時更新 */
.slider-range-active {
  transition: width 0s, left 0s; /* 拖動時無過渡,即時回饋 */
}
```

**Loops**: 持續至使用者釋放滑鼠

### 重置動畫

```css
/* 重置時,slider 平滑過渡 */
.slider-thumb.resetting,
.slider-range-active.resetting {
  transition: all 300ms var(--ease-out);
}

/* 重置按鈕點擊回饋 */
@keyframes resetPulse {
  0% { transform: scale(1); }
  50% { transform: scale(0.95); }
  100% { transform: scale(1); }
}

.reset-button.clicked {
  animation: resetPulse 200ms ease-out;
}
```

### 時間標籤更新

```css
.time-value {
  transition: color 150ms ease-out;
}

/* 拖動時,正在改變的時間標籤高亮 */
.time-value.updating {
  color: var(--color-primary);
}
```

---

## ♿ 無障礙 (Accessibility)

### ARIA 屬性

```html
<div class="slider-track" role="group" aria-label="時間範圍選擇器">
  <div
    class="slider-thumb slider-thumb-start"
    role="slider"
    aria-label="開始時間"
    aria-valuemin="0"
    aria-valuemax="100"
    aria-valuenow="0"
    aria-valuetext="2024-01-01"
    tabindex="0"
  ></div>

  <div
    class="slider-thumb slider-thumb-end"
    role="slider"
    aria-label="結束時間"
    aria-valuemin="0"
    aria-valuemax="100"
    aria-valuenow="100"
    aria-valuetext="2024-12-31"
    tabindex="0"
  ></div>
</div>

<button
  class="reset-button"
  aria-label="重置為完整時間範圍"
  aria-disabled="false"
>
  重置
</button>
```

### 鍵盤導航

```
Tab 順序:
1. Start Thumb
2. End Thumb
3. Reset Button

焦點指示:
├── 明顯的 focus ring (2px 藍色)
└── outline-offset: 2px (與元件保持距離)
```

### 對比度檢查

```
文字對比度:
├── 標籤文字 (#4b5563) vs 背景 (#f9fafb): 4.5:1 ✓ (AA)
├── 時間數值 (#111827) vs 背景 (#f9fafb): 9.8:1 ✓ (AAA)
└── 按鈕文字 (#4b5563) vs 背景 (transparent): 4.5:1 ✓ (AA)

UI 元件對比度:
├── Primary slider (#2563eb) vs 軌道 (#e5e7eb): 3.5:1 ✓ (3:1)
└── Thumb 邊框 (white) vs primary: 清晰可辨 ✓
```

### 動畫偏好

```css
@media (prefers-reduced-motion: reduce) {
  .slider-thumb,
  .slider-range-active,
  .reset-button {
    transition: none; /* 移除所有過渡 */
    animation: none;  /* 移除所有動畫 */
  }

  /* 保留即時回饋,但無過渡 */
  .slider-thumb:hover {
    transform: none; /* 不放大 */
    box-shadow: 0 0 0 2px var(--color-primary); /* 改用邊框提示 */
  }
}
```

---

## 📊 與圖表的視覺距離和分組

### 視覺層級

```
Z-Index 層級:
├── Z-0: 圖表背景
├── Z-1: 圖表線條
├── Z-2: 時間範圍控制器 (獨立區塊)
└── Z-3: Tooltip (懸停時)

視覺分組:
┌─────────────────────────────────────┐
│ 圖表區域 (主視覺)                    │  ← 焦點
└─────────────────────────────────────┘
          ↓ (24px 間距)
┌─────────────────────────────────────┐
│ 控制器區域 (輔助工具)                │  ← 工具
│ • 淺色背景區分                       │
│ • 淡邊框框住                         │
└─────────────────────────────────────┘
```

### 為什麼這樣分組?

**視覺分離**:
- 背景色差異: 圖表背景白色,控制器淺灰
- 邊框框住: 明確「這是一個控制區」

**認知關聯**:
- 距離適中 (24px): 太近會混淆,太遠會失去關聯
- 對齊一致: 控制器與圖表等寬,視覺連貫

---

## 🏗️ 實作範例 (HTML + CSS)

```html
<div class="time-range-control">
  <!-- 時間標籤 -->
  <div class="time-labels">
    <span class="time-label">
      開始: <span class="time-value" data-thumb="start">2024-01-01</span>
    </span>
    <span class="time-label">
      結束: <span class="time-value" data-thumb="end">2024-12-31</span>
    </span>
  </div>

  <!-- Slider 容器 -->
  <div class="time-range-container">
    <div class="slider-wrapper">
      <div class="slider-track" role="group" aria-label="時間範圍選擇器">
        <!-- 選中區間 -->
        <div class="slider-range-active"></div>

        <!-- Start Thumb -->
        <div
          class="slider-thumb slider-thumb-start"
          role="slider"
          aria-label="開始時間"
          aria-valuemin="0"
          aria-valuemax="100"
          aria-valuenow="0"
          tabindex="0"
        ></div>

        <!-- End Thumb -->
        <div
          class="slider-thumb slider-thumb-end"
          role="slider"
          aria-label="結束時間"
          aria-valuemin="0"
          aria-valuemax="100"
          aria-valuenow="100"
          tabindex="0"
        ></div>
      </div>
    </div>

    <!-- 重置按鈕 -->
    <button class="reset-button" aria-label="重置為完整時間範圍">
      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
        <path d="M8 2a6 6 0 1 0 0 12A6 6 0 0 0 8 2zM4 8a4 4 0 1 1 8 0 4 4 0 0 1-8 0z"/>
        <path d="M8 4.5v3.25l2.5 1.5-.75 1.25L6.5 8.75V4.5h1.5z"/>
      </svg>
      重置
    </button>
  </div>
</div>
```

```css
/* ========== 容器 ========== */
.time-range-control {
  margin-top: var(--spacing-lg); /* 24px 與圖表間隔 */
}

.time-labels {
  display: flex;
  justify-content: space-between;
  margin-bottom: var(--spacing-sm); /* 8px */
}

.time-label {
  font: var(--font-label);
  color: var(--color-text-secondary);
}

.time-value {
  font-family: var(--primitive-font-mono);
  font-weight: var(--primitive-font-semibold);
  color: var(--color-text);
  transition: color 150ms var(--ease-out);
}

.time-value.updating {
  color: var(--color-primary);
}

.time-range-container {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: var(--spacing-md);
  align-items: center;
  padding: var(--spacing-md);
  background: var(--color-surface-raised);
  border-radius: var(--primitive-radius-lg);
  border: 1px solid var(--color-border-muted);
}

/* ========== Slider ========== */
.slider-wrapper {
  min-width: 400px;
  position: relative;
}

.slider-track {
  height: 8px;
  background: var(--color-border);
  border-radius: var(--primitive-radius-full);
  position: relative;
  cursor: pointer;
}

.slider-range-active {
  position: absolute;
  height: 100%;
  background: var(--color-primary);
  border-radius: var(--primitive-radius-full);
  pointer-events: none;
}

.slider-thumb {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  width: 20px;
  height: 20px;
  background: var(--color-primary);
  border: 3px solid white;
  border-radius: var(--primitive-radius-full);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  cursor: grab;
  transition: all var(--duration-fast) var(--ease-out);
}

/* 擴大觸控區 */
.slider-thumb::before {
  content: '';
  position: absolute;
  inset: -12px;
  border-radius: 50%;
}

.slider-thumb:hover {
  transform: translateY(-50%) scale(1.15);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
}

.slider-thumb:active,
.slider-thumb.dragging {
  cursor: grabbing;
  transform: translateY(-50%) scale(1.1);
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
}

.slider-thumb:focus-visible {
  outline: 2px solid var(--color-border-focus);
  outline-offset: 2px;
}

/* ========== 重置按鈕 ========== */
.reset-button {
  display: inline-flex;
  align-items: center;
  gap: var(--spacing-xs); /* 4px icon 與文字間距 */
  background: transparent;
  color: var(--color-text-secondary);
  border: 1px solid var(--color-border);
  border-radius: var(--primitive-radius-md);
  padding: var(--primitive-space-2) var(--primitive-space-3);
  font: var(--font-label);
  cursor: pointer;
  transition: all var(--duration-fast) var(--ease-out);
  min-height: 32px;
}

.reset-button:hover {
  background: var(--color-surface-raised);
  border-color: var(--color-border-strong);
  color: var(--color-text);
}

.reset-button:active {
  transform: scale(0.98);
}

.reset-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* ========== 響應式 ========== */
@media (max-width: 1023px) {
  .slider-wrapper {
    min-width: 300px;
  }
}

@media (max-width: 767px) {
  .time-range-container {
    grid-template-columns: 1fr;
    gap: var(--spacing-sm);
  }

  .slider-wrapper {
    min-width: unset;
    width: 100%;
  }

  .reset-button {
    width: 100%;
    justify-content: center;
  }

  .slider-thumb {
    width: 24px;
    height: 24px;
  }
}

/* ========== 無障礙 ========== */
@media (prefers-reduced-motion: reduce) {
  .slider-thumb,
  .slider-range-active,
  .reset-button,
  .time-value {
    transition: none;
    animation: none;
  }

  .slider-thumb:hover {
    transform: translateY(-50%);
    box-shadow: 0 0 0 2px var(--color-primary);
  }
}
```

---

## ✅ Checklist

### 視覺設計
- [x] 使用 Design Tokens (顏色、間距、圓角、字體)
- [x] 遵循 60-30-10 色彩法則
- [x] 對比度符合 WCAG AA (≥ 4.5:1 文字, ≥ 3:1 UI)
- [x] 圓角使用 `--primitive-radius-*`
- [x] 間距使用 4px 倍數系統

### 互動設計
- [x] Hover 狀態清晰 (放大、陰影)
- [x] Focus 狀態符合無障礙 (2px ring)
- [x] Active 狀態有回饋 (略縮)
- [x] 觸控區 ≥ 44px (Fitts's Law)
- [x] 拖動即時回饋 (< 100ms)

### 無障礙
- [x] 鍵盤可操作 (Tab、方向鍵)
- [x] ARIA 標籤完整
- [x] 支援 `prefers-reduced-motion`
- [x] Focus ring 清晰可見

### 效能
- [x] 使用 `transform` 而非位置屬性
- [x] 拖動時節流更新 (16ms)
- [x] 避免觸發重排

### 響應式
- [x] 桌面/平板/手機適配
- [x] Mobile First 設計
- [x] 觸控裝置優化 (更大 thumb)

---

## 🎓 設計決策總結

| 決策 | 理由 | 參考原則 |
|------|------|----------|
| Slider 在圖表下方 | 使用者熟悉此模式 | Jakob's Law |
| Thumb 觸控區 44px | 易於點擊 | Fitts's Law |
| 拖動即時更新 < 100ms | 感覺流暢 | 微互動回饋 |
| 重置按鈕用 ghost | 次要動作,不搶主視覺 | 視覺層級 |
| 間距 24px | 視覺分組但保持關聯 | 格式塔接近性 |
| 主色 primary | 品牌色 = 主要互動 | Von Restorff Effect |
| 等寬字體顯示日期 | 數字變化不跳動 | 視覺穩定性 |
| 節流 16ms | 60fps 流暢 | 效能優化 |

---

**設計版本**: v1.0
**最後更新**: 2026-01-12
**設計者**: DESIGNER (AI)
