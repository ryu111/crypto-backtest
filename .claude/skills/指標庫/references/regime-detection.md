# 市場狀態偵測與策略切換

Market Regime Detection - 識別市場狀態並動態切換策略。

## 方法選擇

| 方法 | 定位 | 加密市場適用性 |
|------|------|----------------|
| **方向×波動矩陣** | ⭐ 主力方法 | ✅ 專為加密設計 |
| HMM/HSMM | 📚 學術參考 | ⚠️ 需重新訓練 |
| Jump Model | 📚 學術參考 | ⚠️ 需重新訓練 |

> **為什麼可解釋指標法是主力？**
> - 學術方法多基於股票市場（S&P 500），不適用加密
> - 加密市場波動率是股票 3-5 倍，狀態轉換更頻繁
> - 可解釋指標可即時調整，不依賴歷史假設

## 核心概念

```
市場狀態矩陣 = 方向維度 × 波動維度

方向（Direction）: -10 到 +10
- -10：極度熊市
- 0：中性/盤整
- +10：極度牛市

波動（Volatility）: 0 到 10
- 0：極低波動（盤整）
- 5：正常波動
- 10：極高波動（恐慌/狂熱）
```

### 策略適用區域

```
波動 ↑
10 │ ┌───────────┬───────────┬───────────┐
   │ │ 做空突破  │ 高波動    │ 做多突破  │
   │ │ 趨勢策略  │ 雙向策略  │ 趨勢策略  │
 7 │ ├───────────┼───────────┼───────────┤
   │ │ 震盪放空  │ 網格策略  │ 震盪做多  │
   │ │ RSI策略   │ 區間策略  │ RSI策略   │
 3 │ ├───────────┼───────────┼───────────┤
   │ │ 等待或    │ 低波動    │ 等待或    │
   │ │ 減倉      │ 盤整策略  │ 減倉      │
 0 │ └───────────┴───────────┴───────────┘
  -10         -3    0    +3         +10   → 方向
```

## ⭐ 主力方法：可解釋指標法

**直觀、可調整、專為加密市場設計。**

### 方向分數計算

```python
import numpy as np
import pandas as pd

def calculate_direction_score(
    data: pd.DataFrame,
    ma_short: int = 20,
    ma_long: int = 50,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    scale: int = 10
) -> pd.Series:
    """
    計算方向分數 (-scale 到 +scale)

    組合多個指標的綜合方向判斷
    """
    close = data['close']

    # 1. MA 位置 (-1 到 +1)
    ma_s = close.rolling(ma_short).mean()
    ma_l = close.rolling(ma_long).mean()
    ma_position = np.where(close > ma_s, 0.5, -0.5)
    ma_position += np.where(ma_s > ma_l, 0.5, -0.5)

    # 2. MA 斜率 (-1 到 +1)
    ma_slope = (ma_s - ma_s.shift(5)) / ma_s.shift(5)
    ma_slope_score = np.clip(ma_slope * 20, -1, 1)

    # 3. RSI 偏離 (-1 到 +1)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_score = (rsi - 50) / 50

    # 4. MACD 柱狀 (-1 到 +1)
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_hist = macd_line - macd_line.ewm(span=9, adjust=False).mean()
    macd_score = np.clip(macd_hist / close * 100, -1, 1)

    # 綜合分數（加權平均）
    weights = {'ma_position': 0.3, 'ma_slope': 0.2, 'rsi': 0.25, 'macd': 0.25}

    composite = (
        ma_position * weights['ma_position'] +
        ma_slope_score * weights['ma_slope'] +
        rsi_score * weights['rsi'] +
        macd_score * weights['macd']
    )

    return pd.Series(composite * scale, index=data.index, name='direction_score')


def adx_direction_score(
    data: pd.DataFrame,
    period: int = 14,
    scale: int = 10
) -> pd.Series:
    """
    使用 ADX 的 +DI/-DI 計算方向

    優點：直接基於價格動量，較少延遲
    """
    high, low, close = data['high'], data['low'], data['close']

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / (atr + 1e-10)

    # 方向分數：+DI 和 -DI 的差異標準化
    di_diff = plus_di - minus_di
    di_sum = plus_di + minus_di + 1e-10

    direction = (di_diff / di_sum) * scale
    return pd.Series(direction.values, index=data.index, name='adx_direction')


def elder_power_score(
    data: pd.DataFrame,
    ema_period: int = 13,
    scale: int = 10
) -> pd.Series:
    """
    Elder 的 Bull/Bear Power 方向分數

    Bull Power = High - EMA（多頭力量）
    Bear Power = Low - EMA（空頭力量）
    """
    close = data['close']
    high = data['high']
    low = data['low']

    ema = close.ewm(span=ema_period, adjust=False).mean()

    bull_power = high - ema
    bear_power = low - ema

    # 標準化
    power_range = (high - low).rolling(20).mean()
    bull_norm = bull_power / (power_range + 1e-10)
    bear_norm = bear_power / (power_range + 1e-10)

    # 綜合：Bull + Bear
    net_power = (bull_norm + bear_norm) / 2
    direction = np.clip(net_power * scale, -scale, scale)

    return pd.Series(direction, index=data.index, name='elder_power')
```

### 波動分數計算

```python
def volatility_score_atr(
    data: pd.DataFrame,
    atr_period: int = 14,
    lookback: int = 100,
    scale: int = 10
) -> pd.Series:
    """
    基於 ATR 的波動分數 (0 到 scale)

    將當前 ATR 對比歷史百分位
    """
    high, low, close = data['high'], data['low'], data['close']

    # ATR 計算
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    # 標準化 ATR (相對於價格)
    natr = atr / close

    # 百分位排名
    def percentile_rank(x):
        return pd.Series(x).rank(pct=True).iloc[-1]

    volatility = natr.rolling(lookback).apply(percentile_rank, raw=False)
    return pd.Series(volatility * scale, index=data.index, name='volatility_atr')


def volatility_score_bbw(
    data: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    lookback: int = 100,
    scale: int = 10
) -> pd.Series:
    """
    基於 Bollinger Band Width 的波動分數

    BBW = (Upper - Lower) / Middle
    """
    close = data['close']

    # Bollinger Bands
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std

    # Band Width
    bbw = (upper - lower) / middle

    # 百分位排名
    def percentile_rank(x):
        return pd.Series(x).rank(pct=True).iloc[-1]

    volatility = bbw.rolling(lookback).apply(percentile_rank, raw=False)
    return pd.Series(volatility * scale, index=data.index, name='volatility_bbw')


def choppiness_index(
    data: pd.DataFrame,
    period: int = 14,
    scale: int = 10
) -> pd.Series:
    """
    Choppiness Index - 趨勢 vs 區間

    高值 = 盤整/震盪
    低值 = 趨勢明確

    回傳：0 = 強趨勢, scale = 強震盪
    """
    high, low, close = data['high'], data['low'], data['close']

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR sum
    atr_sum = tr.rolling(period).sum()

    # Highest High - Lowest Low
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    range_hl = hh - ll

    # Choppiness Index
    ci = 100 * np.log10(atr_sum / (range_hl + 1e-10)) / np.log10(period)

    # 標準化到 0-scale（CI 通常在 38.2 到 61.8 之間）
    ci_norm = (ci - 38.2) / (61.8 - 38.2)
    ci_norm = np.clip(ci_norm, 0, 1) * scale

    return pd.Series(ci_norm, index=data.index, name='choppiness')
```

### 市場狀態分析器

```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

class MarketRegime(Enum):
    """市場狀態枚舉"""
    STRONG_BULL_HIGH_VOL = "strong_bull_high_vol"     # 強勢上漲，高波動
    STRONG_BULL_LOW_VOL = "strong_bull_low_vol"       # 強勢上漲，低波動
    WEAK_BULL_HIGH_VOL = "weak_bull_high_vol"         # 弱勢上漲，高波動
    WEAK_BULL_LOW_VOL = "weak_bull_low_vol"           # 弱勢上漲，低波動
    NEUTRAL_HIGH_VOL = "neutral_high_vol"             # 中性，高波動
    NEUTRAL_LOW_VOL = "neutral_low_vol"               # 中性，低波動
    WEAK_BEAR_HIGH_VOL = "weak_bear_high_vol"         # 弱勢下跌，高波動
    WEAK_BEAR_LOW_VOL = "weak_bear_low_vol"           # 弱勢下跌，低波動
    STRONG_BEAR_HIGH_VOL = "strong_bear_high_vol"     # 強勢下跌，高波動
    STRONG_BEAR_LOW_VOL = "strong_bear_low_vol"       # 強勢下跌，低波動


@dataclass
class MarketState:
    """市場狀態數據類"""
    direction: float      # -10 到 +10
    volatility: float     # 0 到 10
    regime: MarketRegime
    timestamp: datetime

    def to_dict(self) -> dict:
        return {
            'direction': self.direction,
            'volatility': self.volatility,
            'regime': self.regime.value,
            'timestamp': self.timestamp.isoformat()
        }


class MarketStateAnalyzer:
    """市場狀態分析器"""

    def __init__(
        self,
        direction_threshold_strong: float = 5.0,
        direction_threshold_weak: float = 2.0,
        volatility_threshold: float = 5.0,
        direction_method: str = 'composite'  # 'composite', 'adx', 'elder'
    ):
        self.dir_strong = direction_threshold_strong
        self.dir_weak = direction_threshold_weak
        self.vol_threshold = volatility_threshold
        self.direction_method = direction_method

    def calculate_state(self, data: pd.DataFrame) -> MarketState:
        """計算當前市場狀態"""
        direction = self._calculate_direction(data)
        volatility = self._calculate_volatility(data)
        regime = self._determine_regime(direction, volatility)

        return MarketState(
            direction=direction,
            volatility=volatility,
            regime=regime,
            timestamp=data.index[-1] if isinstance(data.index[-1], datetime)
                      else datetime.now()
        )

    def _calculate_direction(self, data: pd.DataFrame) -> float:
        """計算方向分數"""
        if self.direction_method == 'composite':
            score = calculate_direction_score(data)
        elif self.direction_method == 'adx':
            score = adx_direction_score(data)
        elif self.direction_method == 'elder':
            score = elder_power_score(data)
        else:
            score = calculate_direction_score(data)

        return float(score.iloc[-1])

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """計算波動分數"""
        # 綜合 ATR 和 BBW
        vol_atr = volatility_score_atr(data)
        vol_bbw = volatility_score_bbw(data)

        # 加權平均
        volatility = vol_atr.iloc[-1] * 0.6 + vol_bbw.iloc[-1] * 0.4
        return float(volatility)

    def _determine_regime(self, direction: float, volatility: float) -> MarketRegime:
        """判斷市場狀態"""
        # 方向分類
        if direction > self.dir_strong:
            dir_class = 'strong_bull'
        elif direction > self.dir_weak:
            dir_class = 'weak_bull'
        elif direction < -self.dir_strong:
            dir_class = 'strong_bear'
        elif direction < -self.dir_weak:
            dir_class = 'weak_bear'
        else:
            dir_class = 'neutral'

        # 波動分類
        vol_class = 'high_vol' if volatility > self.vol_threshold else 'low_vol'

        # 組合
        regime_name = f"{dir_class}_{vol_class}"
        return MarketRegime(regime_name)
```

### ⚠️ 狀態偵測準確度驗證（必做！）

> **重要：先驗證狀態偵測準確度，再做策略匹配！**
>
> 如果狀態偵測本身不準，策略切換就沒有意義。

```
傳統回測：
策略規則 → 跑歷史數據 → 看績效
（一步到位）

狀態切換回測（三層驗證）：
1️⃣ 驗證狀態偵測準確度 → 狀態真的準嗎？
         ↓
2️⃣ 匹配策略 → 哪個策略適合哪個狀態？
         ↓
3️⃣ 整體績效 → 切換機制有效嗎？
```

#### 驗證流程

| 步驟 | 驗證內容 | 通過標準 | 說明 |
|------|----------|----------|------|
| 1️⃣ | 方向偵測準確度 | > 60% | 預測牛市後真的漲 |
| 2️⃣ | 波動偵測準確度 | > 60% | 預測高波動後真的波動大 |
| 3️⃣ | 狀態穩定性 | 翻轉 < 20%/日 | 不會頻繁來回切換 |
| 4️⃣ | **通過後才做** | 策略匹配回測 | - |

#### 驗證程式碼

```python
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

class RegimeValidator:
    """狀態偵測準確度驗證器"""

    def __init__(
        self,
        forward_periods: int = 20,  # 看未來多少根 K 線
        direction_threshold: float = 0.03,  # 方向判定閾值 (3%)
        volatility_threshold: float = 1.5   # 波動判定閾值 (1.5倍)
    ):
        self.forward_periods = forward_periods
        self.dir_threshold = direction_threshold
        self.vol_threshold = volatility_threshold

    def validate_direction(
        self,
        data: pd.DataFrame,
        states: List[MarketState]
    ) -> Dict:
        """
        驗證方向偵測準確度

        邏輯：
        - 預測 direction > 5（強牛）→ 未來應該漲 > 3%
        - 預測 direction < -5（強熊）→ 未來應該跌 > 3%
        - 預測 |direction| < 3（中性）→ 未來應該盤整 < 3%
        """
        results = []

        for i, state in enumerate(states):
            if i + self.forward_periods >= len(data):
                break

            # 計算未來報酬
            future_return = (
                data['close'].iloc[i + self.forward_periods] /
                data['close'].iloc[i] - 1
            )

            # 判斷是否準確
            if state.direction > 5:  # 預測強牛
                accurate = future_return > self.dir_threshold
                prediction = 'strong_bull'
            elif state.direction < -5:  # 預測強熊
                accurate = future_return < -self.dir_threshold
                prediction = 'strong_bear'
            elif state.direction > 2:  # 預測弱牛
                accurate = future_return > 0
                prediction = 'weak_bull'
            elif state.direction < -2:  # 預測弱熊
                accurate = future_return < 0
                prediction = 'weak_bear'
            else:  # 預測中性
                accurate = abs(future_return) < self.dir_threshold
                prediction = 'neutral'

            results.append({
                'timestamp': state.timestamp,
                'direction_score': state.direction,
                'prediction': prediction,
                'future_return': future_return,
                'accurate': accurate
            })

        df = pd.DataFrame(results)
        accuracy = df['accurate'].mean()

        # 分類別準確度
        by_prediction = df.groupby('prediction')['accurate'].mean().to_dict()

        return {
            'overall_accuracy': accuracy,
            'by_prediction': by_prediction,
            'n_samples': len(results),
            'passed': accuracy > 0.6,
            'details': df
        }

    def validate_volatility(
        self,
        data: pd.DataFrame,
        states: List[MarketState]
    ) -> Dict:
        """
        驗證波動偵測準確度

        邏輯：
        - 預測 volatility > 7（高波動）→ 未來波動應該大於平均
        - 預測 volatility < 3（低波動）→ 未來波動應該小於平均
        """
        results = []

        # 計算歷史平均波動
        returns = data['close'].pct_change()
        avg_vol = returns.rolling(100).std().mean()

        for i, state in enumerate(states):
            if i + self.forward_periods >= len(data):
                break

            # 計算未來波動
            future_vol = returns.iloc[i:i+self.forward_periods].std()
            vol_ratio = future_vol / avg_vol

            # 判斷是否準確
            if state.volatility > 7:  # 預測高波動
                accurate = vol_ratio > self.vol_threshold
                prediction = 'high_vol'
            elif state.volatility < 3:  # 預測低波動
                accurate = vol_ratio < 1 / self.vol_threshold
                prediction = 'low_vol'
            else:  # 預測中等波動
                accurate = 0.7 < vol_ratio < 1.5
                prediction = 'mid_vol'

            results.append({
                'timestamp': state.timestamp,
                'volatility_score': state.volatility,
                'prediction': prediction,
                'future_vol_ratio': vol_ratio,
                'accurate': accurate
            })

        df = pd.DataFrame(results)
        accuracy = df['accurate'].mean()

        return {
            'overall_accuracy': accuracy,
            'by_prediction': df.groupby('prediction')['accurate'].mean().to_dict(),
            'n_samples': len(results),
            'passed': accuracy > 0.6,
            'details': df
        }

    def validate_stability(
        self,
        states: List[MarketState],
        max_flip_rate: float = 0.2
    ) -> Dict:
        """
        驗證狀態穩定性（不會頻繁翻轉）

        邏輯：
        - 狀態翻轉太頻繁 = 噪音太多，不可靠
        - 每日翻轉率應 < 20%
        """
        if len(states) < 2:
            return {'passed': False, 'reason': 'insufficient_data'}

        flips = 0
        for i in range(1, len(states)):
            prev_dir = 'bull' if states[i-1].direction > 2 else \
                      ('bear' if states[i-1].direction < -2 else 'neutral')
            curr_dir = 'bull' if states[i].direction > 2 else \
                      ('bear' if states[i].direction < -2 else 'neutral')

            if prev_dir != curr_dir:
                flips += 1

        flip_rate = flips / len(states)

        return {
            'flip_rate': flip_rate,
            'total_flips': flips,
            'total_states': len(states),
            'passed': flip_rate < max_flip_rate
        }

    def full_validation(
        self,
        data: pd.DataFrame,
        states: List[MarketState]
    ) -> Dict:
        """完整驗證報告"""
        dir_result = self.validate_direction(data, states)
        vol_result = self.validate_volatility(data, states)
        stability_result = self.validate_stability(states)

        all_passed = (
            dir_result['passed'] and
            vol_result['passed'] and
            stability_result['passed']
        )

        return {
            'direction': dir_result,
            'volatility': vol_result,
            'stability': stability_result,
            'all_passed': all_passed,
            'recommendation': (
                '✅ 可進行策略匹配' if all_passed
                else '❌ 需調整狀態偵測參數'
            )
        }


# 使用範例
def validate_before_strategy_matching(data: pd.DataFrame):
    """驗證流程範例"""
    # 1. 計算狀態
    analyzer = MarketStateAnalyzer()
    states = []
    for i in range(100, len(data)):
        state = analyzer.calculate_state(data.iloc[:i])
        states.append(state)

    # 2. 驗證準確度
    validator = RegimeValidator()
    report = validator.full_validation(data, states)

    print(f"方向準確度: {report['direction']['overall_accuracy']:.1%}")
    print(f"波動準確度: {report['volatility']['overall_accuracy']:.1%}")
    print(f"狀態翻轉率: {report['stability']['flip_rate']:.1%}")
    print(f"\n{report['recommendation']}")

    # 3. 只有通過才繼續
    if report['all_passed']:
        print("\n✅ 開始策略匹配回測...")
        # switch = setup_strategy_switch()
        # ...
    else:
        print("\n❌ 請先調整狀態偵測參數")

    return report
```

#### 不通過時的調整建議

| 問題 | 可能原因 | 調整方向 |
|------|----------|----------|
| 方向準確度低 | 指標太敏感 | 增加 MA 週期、降低權重 |
| 波動準確度低 | ATR 週期不適合 | 調整 lookback、換用 BBW |
| 翻轉太頻繁 | 閾值太窄 | 增加 direction_threshold |
| 強牛/強熊不準 | 閾值設太低 | 提高 dir_strong 到 6-7 |

### 策略切換器

```python
@dataclass
class StrategyConfig:
    """策略配置"""
    name: str
    direction_range: tuple  # (min, max)
    volatility_range: tuple  # (min, max)
    weight: float = 1.0  # 權重

    def is_active(self, direction: float, volatility: float) -> bool:
        """檢查策略是否應該啟用"""
        dir_ok = self.direction_range[0] <= direction <= self.direction_range[1]
        vol_ok = self.volatility_range[0] <= volatility <= self.volatility_range[1]
        return dir_ok and vol_ok


class StrategySwitch:
    """策略切換管理器"""

    def __init__(self):
        self.strategies: Dict[str, StrategyConfig] = {}

    def register_strategy(
        self,
        name: str,
        direction_range: tuple,
        volatility_range: tuple,
        weight: float = 1.0
    ):
        """註冊策略及其適用範圍"""
        self.strategies[name] = StrategyConfig(
            name=name,
            direction_range=direction_range,
            volatility_range=volatility_range,
            weight=weight
        )

    def get_active_strategies(self, state: MarketState) -> List[str]:
        """獲取當前應啟用的策略"""
        active = []
        for name, config in self.strategies.items():
            if config.is_active(state.direction, state.volatility):
                active.append(name)
        return active

    def get_strategy_weights(self, state: MarketState) -> Dict[str, float]:
        """獲取策略權重分配"""
        active = self.get_active_strategies(state)
        if not active:
            return {}

        weights = {name: self.strategies[name].weight for name in active}
        total = sum(weights.values())

        # 標準化權重
        return {name: w / total for name, w in weights.items()}


# 使用範例
def setup_strategy_switch() -> StrategySwitch:
    """設定策略切換器"""
    switch = StrategySwitch()

    # 趨勢策略：需要明確方向，中高波動
    switch.register_strategy(
        "trend_following_long",
        direction_range=(3, 10),
        volatility_range=(3, 10),
        weight=1.0
    )
    switch.register_strategy(
        "trend_following_short",
        direction_range=(-10, -3),
        volatility_range=(3, 10),
        weight=1.0
    )

    # 均值回歸：中性方向，低波動
    switch.register_strategy(
        "mean_reversion",
        direction_range=(-3, 3),
        volatility_range=(0, 5),
        weight=0.8
    )

    # 突破策略：低波動後準備突破
    switch.register_strategy(
        "breakout",
        direction_range=(-5, 5),
        volatility_range=(0, 3),
        weight=0.6
    )

    # 網格策略：高波動震盪
    switch.register_strategy(
        "grid_trading",
        direction_range=(-3, 3),
        volatility_range=(5, 10),
        weight=0.7
    )

    # 資金費率套利：任何市場狀態
    switch.register_strategy(
        "funding_rate_arb",
        direction_range=(-10, 10),
        volatility_range=(0, 10),
        weight=0.5
    )

    return switch
```

---

## 📚 學術參考（僅供參考）

> ⚠️ **加密市場適用性警告**
>
> 以下方法主要基於傳統股票市場研究，直接套用可能不準確：
> - 訓練數據：S&P 500、債券（非加密）
> - 波動假設：年化 15-20%（加密 60-100%+）
> - 狀態持續：傳統市場狀態持續較長
>
> **如需使用，必須用加密數據重新訓練。**

### Hidden Markov Model (HMM)

**核心概念：**
- 市場存在「隱藏狀態」（如牛市、熊市、震盪）
- 只能觀察到價格/報酬，無法直接觀察狀態
- HMM 根據觀察值推斷隱藏狀態

```python
from hmmlearn import hmm
import numpy as np

class HMMRegimeDetector:
    """HMM 市場狀態偵測器"""

    def __init__(self, n_states: int = 3):
        """
        Args:
            n_states: 狀態數量（通常 2-4）
                2: 牛市/熊市
                3: 牛市/盤整/熊市
                4: 強牛/弱牛/弱熊/強熊
        """
        self.n_states = n_states
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )

    def fit(self, returns: np.ndarray):
        """訓練 HMM 模型"""
        # HMM 需要 2D 輸入
        X = returns.reshape(-1, 1)
        self.model.fit(X)

        # 排序狀態（按平均報酬）
        self._sort_states()

    def _sort_states(self):
        """按平均報酬排序狀態"""
        means = self.model.means_.flatten()
        order = np.argsort(means)  # 低到高

        # 重新排序
        self.model.means_ = self.model.means_[order]
        self.model.covars_ = self.model.covars_[order]
        self.model.transmat_ = self.model.transmat_[order][:, order]
        self.model.startprob_ = self.model.startprob_[order]

    def predict(self, returns: np.ndarray) -> np.ndarray:
        """預測市場狀態"""
        X = returns.reshape(-1, 1)
        return self.model.predict(X)

    def predict_proba(self, returns: np.ndarray) -> np.ndarray:
        """預測狀態機率"""
        X = returns.reshape(-1, 1)
        return self.model.predict_proba(X)

    def get_state_stats(self) -> dict:
        """獲取狀態統計"""
        return {
            'means': self.model.means_.flatten(),
            'stds': np.sqrt(self.model.covars_.flatten()),
            'transition_matrix': self.model.transmat_
        }


# 使用範例
def hmm_regime_example(returns: pd.Series):
    """HMM 狀態偵測範例"""
    detector = HMMRegimeDetector(n_states=3)
    detector.fit(returns.values)

    states = detector.predict(returns.values)
    probs = detector.predict_proba(returns.values)

    # 狀態解讀
    state_names = ['熊市', '盤整', '牛市']  # 按報酬排序

    return pd.DataFrame({
        'return': returns,
        'state': [state_names[s] for s in states],
        'prob_bear': probs[:, 0],
        'prob_neutral': probs[:, 1],
        'prob_bull': probs[:, 2]
    })
```

### Hidden Semi-Markov Model (HSMM)

**與 HMM 差異：**
- HMM：狀態持續時間服從幾何分佈
- HSMM：可指定任意持續時間分佈

```python
# 概念示例（實際需要專門庫如 pyhsmm）
class HSMMConcept:
    """HSMM 概念說明"""

    def __init__(self):
        """
        HSMM 關鍵參數：
        - emission_distributions: 每個狀態的觀測分佈
        - duration_distributions: 每個狀態的持續時間分佈
        - transition_matrix: 狀態轉換機率
        """
        pass

    @staticmethod
    def duration_modeling():
        """
        持續時間建模選項：

        1. Poisson: 適合短期狀態
           - 參數：lambda (平均持續時間)

        2. Negative Binomial: 更靈活
           - 參數：r, p

        3. Empirical: 從數據學習
           - 非參數方法
        """
        pass
```

### Statistical Jump Model

**2024 最新研究：**
- 結合統計跳躍偵測和動態資產配置
- 比 HMM 更適合金融市場的突變特性

```python
class StatisticalJumpModel:
    """統計跳躍模型概念"""

    def __init__(self, threshold: float = 2.0):
        """
        Args:
            threshold: 跳躍偵測閾值（標準差倍數）
        """
        self.threshold = threshold

    def detect_jumps(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """
        偵測統計跳躍（狀態轉換點）

        基於 Z-score 異常偵測
        """
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()

        z_score = (returns - rolling_mean) / (rolling_std + 1e-10)

        # 標記跳躍點
        jumps = abs(z_score) > self.threshold

        return jumps

    def segment_regimes(self, returns: pd.Series) -> pd.Series:
        """
        根據跳躍點分割市場狀態
        """
        jumps = self.detect_jumps(returns)
        jump_idx = jumps[jumps].index

        # 創建狀態標籤
        regimes = pd.Series(0, index=returns.index)
        current_regime = 0

        for i, idx in enumerate(returns.index):
            if idx in jump_idx:
                current_regime += 1
            regimes[idx] = current_regime

        return regimes
```

## 方法比較

| 方法 | 定位 | 加密適用 | 即時性 | 可解釋 |
|------|------|----------|--------|--------|
| **方向×波動矩陣** | ⭐ 主力 | ✅ 是 | ✅ 高 | ✅ 高 |
| HMM | 📚 參考 | ⚠️ 需訓練 | ❌ 低 | ❌ 低 |
| HSMM | 📚 參考 | ⚠️ 需訓練 | ❌ 低 | ❌ 低 |
| Jump Model | 📚 參考 | ⚠️ 需訓練 | ⚠️ 中 | ⚠️ 中 |

### 結論

**直接使用可解釋指標法**，學術方法僅作為概念參考或未來研究方向。

如果未來要嘗試學術方法：
1. 必須用 BTC/ETH 數據重新訓練
2. 驗證狀態轉換是否符合加密市場特性
3. 與可解釋指標法比較，看哪個更準確

## 視覺化

```python
import matplotlib.pyplot as plt

def visualize_market_state(
    data: pd.DataFrame,
    direction: pd.Series,
    volatility: pd.Series,
    figsize: tuple = (14, 10)
):
    """視覺化市場狀態"""
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # 價格
    axes[0].plot(data.index, data['close'], label='Price')
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].set_title('Price with Market State Analysis')

    # 方向分數
    axes[1].plot(data.index, direction, label='Direction', color='blue')
    axes[1].axhline(y=0, color='gray', linestyle='--')
    axes[1].axhline(y=5, color='green', linestyle=':', alpha=0.5)
    axes[1].axhline(y=-5, color='red', linestyle=':', alpha=0.5)
    axes[1].fill_between(data.index, direction, 0,
                         where=direction > 0, color='green', alpha=0.3)
    axes[1].fill_between(data.index, direction, 0,
                         where=direction < 0, color='red', alpha=0.3)
    axes[1].set_ylabel('Direction (-10 to +10)')
    axes[1].set_ylim(-12, 12)
    axes[1].legend()

    # 波動分數
    axes[2].plot(data.index, volatility, label='Volatility', color='orange')
    axes[2].axhline(y=5, color='red', linestyle='--', alpha=0.5)
    axes[2].fill_between(data.index, volatility, 0, color='orange', alpha=0.3)
    axes[2].set_ylabel('Volatility (0 to 10)')
    axes[2].set_ylim(0, 12)
    axes[2].legend()

    plt.tight_layout()
    return fig


def plot_regime_scatter(states: List[MarketState], figsize: tuple = (10, 8)):
    """繪製狀態散佈圖"""
    directions = [s.direction for s in states]
    volatilities = [s.volatility for s in states]

    plt.figure(figsize=figsize)
    scatter = plt.scatter(directions, volatilities, c=range(len(states)),
                         cmap='viridis', alpha=0.6)

    plt.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=3, color='green', linestyle=':', alpha=0.3)
    plt.axvline(x=-3, color='red', linestyle=':', alpha=0.3)

    plt.xlabel('Direction (-10 to +10)')
    plt.ylabel('Volatility (0 to 10)')
    plt.title('Market State Distribution')
    plt.colorbar(scatter, label='Time')
    plt.xlim(-12, 12)
    plt.ylim(0, 12)

    # 添加象限標籤
    plt.text(6, 8, 'Bull + High Vol\n(Trend Long)', ha='center')
    plt.text(-6, 8, 'Bear + High Vol\n(Trend Short)', ha='center')
    plt.text(6, 2, 'Bull + Low Vol\n(Accumulation)', ha='center')
    plt.text(-6, 2, 'Bear + Low Vol\n(Distribution)', ha='center')
    plt.text(0, 8, 'Neutral + High Vol\n(Choppy)', ha='center')
    plt.text(0, 2, 'Neutral + Low Vol\n(Range)', ha='center')

    return plt.gcf()
```

## 學術參考

### 核心論文

| 論文 | 方法 | 關鍵發現 |
|------|------|----------|
| Bailey et al. (2024) | Statistical Jump Model | 比 HMM 提升 1-4% 報酬 |
| MDPI (2024) | Regime-Switching Factor | 因子策略需適應市場狀態 |
| Nystrup et al. | 5-State HSMM | 5 狀態比 2/3 狀態更準確 |
| State Street (2025) | ML + HMM Ensemble | 機器學習增強傳統方法 |

### 參考連結

1. **Statistical Jump Model**
   - arXiv: https://arxiv.org/abs/2411.08730

2. **Regime-Switching Factor Investing**
   - MDPI: https://www.mdpi.com/2227-7390/12/19/3011

3. **HMM vs HSMM Comparison**
   - SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4556048

4. **ML-Enhanced Regime Detection**
   - State Street: https://www.statestreet.com/content/dam/statestreet/documents/ss_associates/Decoding-market-regimes-part-2.pdf

5. **R Package for HMM**
   - CRAN: https://cran.r-project.org/web/packages/MSwM/MSwM.pdf

## 實作建議

### 快速開始

```python
# 1. 初始化
analyzer = MarketStateAnalyzer()
switch = setup_strategy_switch()

# 2. 計算狀態
state = analyzer.calculate_state(data)
print(f"方向: {state.direction:.1f}, 波動: {state.volatility:.1f}")
print(f"狀態: {state.regime.value}")

# 3. 獲取活躍策略
active = switch.get_active_strategies(state)
weights = switch.get_strategy_weights(state)
print(f"活躍策略: {active}")
print(f"權重分配: {weights}")
```

### 回測整合

```python
def backtest_with_regime(data: pd.DataFrame, strategies: dict) -> pd.DataFrame:
    """帶狀態切換的回測"""
    analyzer = MarketStateAnalyzer()
    switch = setup_strategy_switch()

    results = []

    for i in range(100, len(data)):
        # 計算當前狀態
        current_data = data.iloc[:i]
        state = analyzer.calculate_state(current_data)

        # 獲取活躍策略
        active = switch.get_active_strategies(state)
        weights = switch.get_strategy_weights(state)

        # 執行策略並記錄
        for name, weight in weights.items():
            if name in strategies:
                signal = strategies[name].generate_signal(current_data)
                results.append({
                    'timestamp': data.index[i],
                    'strategy': name,
                    'weight': weight,
                    'signal': signal,
                    'regime': state.regime.value
                })

    return pd.DataFrame(results)
```

## 後續研究方向

### 近期（可解釋指標優化）

1. **參數自適應**：根據市場變化動態調整閾值
2. **多資產聯動**：考慮 BTC/ETH 相關性
3. **時間尺度**：不同時框的狀態可能不同
4. **加密特有指標**：整合資金費率、Open Interest、鏈上數據

### 遠期（學術方法驗證）

5. **用加密數據訓練 HMM**：驗證是否比指標法更準確
6. **狀態持續時間分析**：加密市場的狀態轉換規律
7. **跨市場比較**：學術方法在不同市場的表現差異
