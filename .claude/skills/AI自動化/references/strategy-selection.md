# 策略選擇理論

AI 自動化回測循環中的策略選擇方法。

## Exploitation vs Exploration 問題

### 核心矛盾

| 行為 | 優點 | 缺點 |
|------|------|------|
| **Exploitation（利用）** | 獲得已知最佳結果 | 可能錯過更優解 |
| **Exploration（探索）** | 發現新的可能性 | 短期報酬較低 |

### 為什麼需要平衡？

```
純利用：陷入局部最優，無法發現更好策略
純探索：不斷嘗試新事物，無法累積最佳策略的優勢
```

## Epsilon-Greedy 算法

### 原理

```
以機率 ε 隨機選擇（探索）
以機率 1-ε 選擇當前最佳（利用）
```

### 本系統預設：80%/20%

| 參數 | 值 | 說明 |
|------|-----|------|
| ε (epsilon) | 0.2 | 探索機率 |
| 1-ε | 0.8 | 利用機率 |

### 實作程式碼

```python
import random
from typing import List, Optional

class EpsilonGreedySelector:
    """Epsilon-Greedy 策略選擇器"""

    def __init__(self, epsilon: float = 0.2):
        self.epsilon = epsilon
        self.strategy_stats = {}  # {strategy_name: {'count': n, 'avg_sharpe': x}}

    def select(self, available_strategies: List[str]) -> str:
        """選擇策略"""
        if random.random() < self.epsilon:
            # 探索：隨機選擇
            return random.choice(available_strategies)
        else:
            # 利用：選擇最佳
            return self._get_best_strategy(available_strategies)

    def _get_best_strategy(self, strategies: List[str]) -> str:
        """獲取歷史表現最佳的策略"""
        best_strategy = None
        best_sharpe = -float('inf')

        for strategy in strategies:
            stats = self.strategy_stats.get(strategy)
            if stats and stats['avg_sharpe'] > best_sharpe:
                best_sharpe = stats['avg_sharpe']
                best_strategy = strategy

        # 如果沒有歷史數據，隨機選擇
        return best_strategy or random.choice(strategies)

    def update(self, strategy: str, sharpe: float):
        """更新策略統計"""
        if strategy not in self.strategy_stats:
            self.strategy_stats[strategy] = {'count': 0, 'total_sharpe': 0}

        stats = self.strategy_stats[strategy]
        stats['count'] += 1
        stats['total_sharpe'] += sharpe
        stats['avg_sharpe'] = stats['total_sharpe'] / stats['count']
```

### Epsilon 衰減

隨著學習進行，逐漸降低探索比例：

```python
class DecayingEpsilonGreedy:
    """衰減式 Epsilon-Greedy"""

    def __init__(
        self,
        initial_epsilon: float = 0.5,
        min_epsilon: float = 0.1,
        decay_rate: float = 0.995
    ):
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate

    def select(self, strategies: List[str], best_strategy: Optional[str]) -> str:
        if random.random() < self.epsilon:
            return random.choice(strategies)
        return best_strategy or random.choice(strategies)

    def decay(self):
        """每次迭代後衰減"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
```

**衰減建議：**

| 階段 | 迭代次數 | ε 值 | 行為 |
|------|----------|------|------|
| 早期 | 0-50 | 0.5-0.3 | 大量探索 |
| 中期 | 50-200 | 0.3-0.15 | 平衡 |
| 後期 | 200+ | 0.15-0.1 | 主要利用 |

## UCB (Upper Confidence Bound)

### 原理

同時考慮：
1. **期望報酬**：歷史平均表現
2. **不確定性**：嘗試次數越少，不確定性越高

### 公式

```
UCB = 平均報酬 + c × √(ln(N) / n_i)

其中：
- N = 總嘗試次數
- n_i = 策略 i 的嘗試次數
- c = 探索係數（通常 √2）
```

### 實作程式碼

```python
import math
from typing import Dict, List

class UCBSelector:
    """UCB 策略選擇器"""

    def __init__(self, c: float = 1.414):  # √2
        self.c = c
        self.total_count = 0
        self.strategy_stats: Dict[str, dict] = {}

    def select(self, strategies: List[str]) -> str:
        """選擇 UCB 值最高的策略"""
        # 確保每個策略至少嘗試一次
        for s in strategies:
            if s not in self.strategy_stats:
                return s

        best_strategy = None
        best_ucb = -float('inf')

        for strategy in strategies:
            ucb = self._calculate_ucb(strategy)
            if ucb > best_ucb:
                best_ucb = ucb
                best_strategy = strategy

        return best_strategy

    def _calculate_ucb(self, strategy: str) -> float:
        stats = self.strategy_stats[strategy]
        n = stats['count']

        exploitation = stats['avg_sharpe']
        exploration = self.c * math.sqrt(math.log(self.total_count) / n)

        return exploitation + exploration

    def update(self, strategy: str, sharpe: float):
        """更新統計"""
        self.total_count += 1

        if strategy not in self.strategy_stats:
            self.strategy_stats[strategy] = {
                'count': 0,
                'total_sharpe': 0,
                'avg_sharpe': 0
            }

        stats = self.strategy_stats[strategy]
        stats['count'] += 1
        stats['total_sharpe'] += sharpe
        stats['avg_sharpe'] = stats['total_sharpe'] / stats['count']
```

### UCB 優勢

| 優勢 | 說明 |
|------|------|
| 自動平衡 | 不需手動設定 ε |
| 系統性探索 | 優先探索不確定性高的策略 |
| 收斂性保證 | 理論上保證找到最優 |

## Thompson Sampling

### 原理

基於貝葉斯統計：
1. 為每個策略維護一個機率分佈
2. 從分佈中抽樣
3. 選擇抽樣值最高的策略

### Beta 分佈版本（適用於成功/失敗）

```python
import numpy as np
from typing import Dict, List

class ThompsonSamplingSelector:
    """Thompson Sampling 選擇器"""

    def __init__(self, success_threshold: float = 1.0):
        """
        Args:
            success_threshold: Sharpe > 此值視為成功
        """
        self.success_threshold = success_threshold
        # Beta 分佈參數：{strategy: (alpha, beta)}
        self.priors: Dict[str, tuple] = {}

    def select(self, strategies: List[str]) -> str:
        """Thompson Sampling 選擇"""
        best_strategy = None
        best_sample = -float('inf')

        for strategy in strategies:
            alpha, beta = self.priors.get(strategy, (1, 1))  # 均勻先驗
            sample = np.random.beta(alpha, beta)

            if sample > best_sample:
                best_sample = sample
                best_strategy = strategy

        return best_strategy

    def update(self, strategy: str, sharpe: float):
        """更新先驗"""
        alpha, beta = self.priors.get(strategy, (1, 1))

        if sharpe > self.success_threshold:
            alpha += 1  # 成功
        else:
            beta += 1   # 失敗

        self.priors[strategy] = (alpha, beta)
```

### 常態分佈版本（適用於連續報酬）

```python
class GaussianThompsonSampling:
    """高斯 Thompson Sampling"""

    def __init__(self):
        # {strategy: {'mu': mean, 'sigma': std, 'n': count}}
        self.posteriors: Dict[str, dict] = {}

    def select(self, strategies: List[str]) -> str:
        best_strategy = None
        best_sample = -float('inf')

        for strategy in strategies:
            post = self.posteriors.get(strategy, {'mu': 0, 'sigma': 1, 'n': 0})

            # 從後驗分佈抽樣
            sample = np.random.normal(post['mu'], post['sigma'])

            if sample > best_sample:
                best_sample = sample
                best_strategy = strategy

        return best_strategy

    def update(self, strategy: str, sharpe: float):
        """貝葉斯更新"""
        if strategy not in self.posteriors:
            self.posteriors[strategy] = {'mu': 0, 'sigma': 1, 'n': 0, 'sum': 0, 'sum_sq': 0}

        post = self.posteriors[strategy]
        post['n'] += 1
        post['sum'] += sharpe
        post['sum_sq'] += sharpe ** 2

        n = post['n']
        post['mu'] = post['sum'] / n

        if n > 1:
            variance = (post['sum_sq'] - post['sum']**2/n) / (n-1)
            post['sigma'] = max(0.01, np.sqrt(variance / n))  # 標準誤差
```

## 方法比較

| 方法 | 複雜度 | 探索方式 | 適用場景 |
|------|--------|----------|----------|
| Epsilon-Greedy | 低 | 隨機 | 預設推薦 |
| UCB | 中 | 系統性 | 策略數量多時 |
| Thompson Sampling | 高 | 自適應 | 需要機率估計時 |

## 何時調整比例？

### 增加探索 (↑ε)

| 情況 | 理由 |
|------|------|
| 新增策略類型 | 需要評估新策略 |
| 市場狀態改變 | 舊策略可能失效 |
| 長期無進展 | 可能陷入局部最優 |

### 減少探索 (↓ε)

| 情況 | 理由 |
|------|------|
| 找到穩定優秀策略 | 專注利用 |
| 資源有限 | 減少浪費 |
| 策略池穩定 | 不需大量探索 |

## 本系統整合

```python
from src.automation.selector import StrategySelector

# 建立選擇器
selector = StrategySelector(method='epsilon_greedy', epsilon=0.2)

# 執行選擇
strategy = selector.select(available_strategies)

# 回測後更新
selector.update(strategy, result.sharpe_ratio)
```

## 參考資料

- [Multi-armed Bandit - Wikipedia](https://en.wikipedia.org/wiki/Multi-armed_bandit)
- [Epsilon-Greedy Q-learning](https://www.baeldung.com/cs/epsilon-greedy-q-learning)
- [arXiv: Optimization of Epsilon-Greedy Exploration](https://arxiv.org/abs/2506.03324)
