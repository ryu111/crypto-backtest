# 策略選擇器使用指南

## 概述

`StrategySelector` 實作 Explore/Exploit 平衡機制，智能選擇下一個要優化的策略。

## 核心概念

### Explore vs Exploit

- **Exploit（利用）**: 選擇歷史最佳策略，最大化已知收益
- **Explore（探索）**: 嘗試新策略或表現未知的策略，發現潛在更佳選項

### 三種選擇策略

#### 1. Epsilon-Greedy

最簡單直觀的方法：

- 80% 機率選擇歷史最佳策略（Exploit）
- 20% 機率隨機探索（Explore）

**優點**：
- 簡單易理解
- 性能穩定
- 適合快速迭代

**缺點**：
- 探索效率較低
- 不考慮策略的不確定性

#### 2. UCB (Upper Confidence Bound)

考慮不確定性的選擇：

```
UCB = avg_sharpe + c * sqrt(ln(total_attempts) / strategy_attempts)
         ↑              ↑
      期望值        探索獎勵
```

**特點**：
- 對嘗試次數少的策略有獎勵
- 自動平衡探索和利用
- 理論保證（後悔界限）

**適用場景**：
- 策略數量較多
- 需要系統性探索
- 長期優化

#### 3. Thompson Sampling

基於貝葉斯的機率選擇：

```
成功率 ~ Beta(alpha + successes, beta + failures)
```

**特點**：
- 機率性選擇
- 自然平衡探索利用
- 適應性強

**適用場景**：
- 需要高度自適應
- 策略性能波動大
- 追求最優解

## 基本使用

### 初始化

```python
from src.automation.selector import StrategySelector
from src.automation.registry import StrategyRegistry
from src.automation.experiment import ExperimentRecorder

# 建立依賴
registry = StrategyRegistry()
recorder = ExperimentRecorder(db_path='experiments.db')

# 建立選擇器
selector = StrategySelector(
    strategy_registry=registry,
    experiment_recorder=recorder,
    config={
        'epsilon': 0.2,      # 20% 探索
        'ucb_c': 2.0,        # UCB 探索常數
        'min_attempts': 3,   # 最小嘗試次數
    }
)
```

### 選擇策略

```python
# 使用 Epsilon-Greedy
strategy = selector.select(method='epsilon_greedy')

# 使用 UCB
strategy = selector.select(method='ucb')

# 使用 Thompson Sampling
strategy = selector.select(method='thompson_sampling')
```

### 更新統計

```python
# 優化結果
result = {
    'passed': True,        # 是否通過驗證
    'sharpe_ratio': 2.5,   # Sharpe Ratio
    'params': {            # 最佳參數
        'window': 20,
        'threshold': 0.02
    }
}

# 更新統計
selector.update_stats('momentum', result)
```

### 取得推薦

```python
# 綜合三種方法的推薦
recommendation = selector.get_recommendation()

print(f"推薦策略: {recommendation['strategy']}")
print(f"推薦原因: {recommendation['reason']}")
print(f"替代選項: {recommendation['alternatives']}")
```

## 進階使用

### 自訂配置

```python
config = {
    # Epsilon-Greedy 參數
    'epsilon': 0.15,  # 降低探索率到 15%

    # UCB 參數
    'ucb_c': 1.5,     # 降低探索獎勵

    # 通用參數
    'min_attempts': 5,  # 提高最小嘗試次數

    # Thompson Sampling 參數
    'thompson_alpha': 1.0,
    'thompson_beta': 1.0,
}

selector = StrategySelector(
    strategy_registry=registry,
    experiment_recorder=recorder,
    config=config
)
```

### 查看探索統計

```python
stats = selector.get_exploration_stats()

print(f"總嘗試次數: {stats['total_attempts']}")
print(f"已嘗試策略: {stats['strategies_tried']}/{stats['strategies_available']}")
print(f"探索率: {stats['exploration_rate']:.1%}")
print(f"最佳策略: {stats['best_strategy']} (Sharpe: {stats['best_sharpe']:.2f})")
```

### 查看策略統計

```python
all_stats = selector.get_strategy_stats()

for name, stat in all_stats.items():
    print(f"\n策略: {name}")
    print(f"  嘗試次數: {stat.attempts}")
    print(f"  成功率: {stat.success_rate:.1%}")
    print(f"  平均 Sharpe: {stat.avg_sharpe:.2f}")
    print(f"  最佳 Sharpe: {stat.best_sharpe:.2f}")
    print(f"  最後嘗試: {stat.last_attempt}")
```

## 完整範例

### AI Loop 整合

```python
from src.automation.selector import StrategySelector
from src.automation.optimizer import StrategyOptimizer
from src.automation.validator import StrategyValidator

class AILoop:
    def __init__(self):
        self.selector = StrategySelector(...)
        self.optimizer = StrategyOptimizer(...)
        self.validator = StrategyValidator(...)

    def run_iteration(self):
        # 1. 選擇策略
        strategy_name = self.selector.select(method='ucb')
        print(f"選擇策略: {strategy_name}")

        # 2. 優化策略
        best_params, metrics = self.optimizer.optimize(strategy_name)

        # 3. 驗證策略
        validation = self.validator.validate(
            strategy_name,
            best_params,
            metrics
        )

        # 4. 更新統計
        result = {
            'passed': validation['passed'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'params': best_params
        }
        self.selector.update_stats(strategy_name, result)

        return validation

    def run(self, iterations=100):
        for i in range(iterations):
            print(f"\n=== Iteration {i+1} ===")

            validation = self.run_iteration()

            if validation['passed']:
                print(f"✅ 策略通過驗證")
            else:
                print(f"❌ 策略未通過: {validation['reason']}")

            # 每 10 次顯示探索統計
            if (i + 1) % 10 == 0:
                stats = self.selector.get_exploration_stats()
                print(f"\n探索進度: {stats['strategies_tried']}/{stats['strategies_available']}")
                print(f"最佳策略: {stats['best_strategy']} (Sharpe: {stats['best_sharpe']:.2f})")
```

### 多策略比較

```python
def compare_selection_methods(selector, iterations=100):
    """比較不同選擇方法的效果"""
    methods = ['epsilon_greedy', 'ucb', 'thompson_sampling']
    results = {method: [] for method in methods}

    for method in methods:
        print(f"\n測試方法: {method}")

        # 重置快取
        selector.reset_cache()

        for i in range(iterations):
            # 選擇策略
            strategy = selector.select(method=method)

            # 模擬優化結果（實際應該執行優化）
            simulated_sharpe = simulate_optimization(strategy)

            result = {
                'passed': simulated_sharpe > 1.0,
                'sharpe_ratio': simulated_sharpe,
                'params': {}
            }

            selector.update_stats(strategy, result)
            results[method].append(simulated_sharpe)

        # 統計
        avg_sharpe = sum(results[method]) / len(results[method])
        print(f"平均 Sharpe: {avg_sharpe:.2f}")

    return results
```

## 參數調整建議

### Epsilon (ε)

| 值 | 特性 | 適用場景 |
|----|------|---------|
| 0.1 | 高度利用 | 策略已充分探索，聚焦優化 |
| 0.2 | 平衡 | 一般情況（預設） |
| 0.3 | 高度探索 | 策略空間大，需要多探索 |

### UCB 常數 (c)

| 值 | 特性 | 適用場景 |
|----|------|---------|
| 1.0 | 保守探索 | 已有穩定策略，微調 |
| 2.0 | 平衡（預設） | 一般情況 |
| 3.0 | 激進探索 | 需要快速找到新策略 |

### 最小嘗試次數 (min_attempts)

| 值 | 特性 | 適用場景 |
|----|------|---------|
| 1 | 快速利用 | 快速迭代，即時反饋 |
| 3 | 平衡（預設） | 一般情況 |
| 5+ | 謹慎利用 | 需要充分驗證再利用 |

## 最佳實踐

### 1. 動態調整 Epsilon

```python
# 初期高探索，後期高利用
def adaptive_epsilon(iteration, total_iterations):
    return 0.3 * (1 - iteration / total_iterations) + 0.1

selector.config['epsilon'] = adaptive_epsilon(current_iter, total_iters)
```

### 2. 定期重置統計

```python
# 每 N 次迭代重置，避免過度依賴歷史
if iteration % 50 == 0:
    selector.reset_cache()
```

### 3. 組合使用多種方法

```python
# 使用推薦系統（綜合三種方法）
recommendation = selector.get_recommendation()
strategy = recommendation['strategy']
```

### 4. 監控探索率

```python
# 確保有足夠探索
stats = selector.get_exploration_stats()
if stats['exploration_rate'] < 0.5:
    # 提高探索率
    selector.config['epsilon'] += 0.1
```

## 故障排除

### 問題：總是選擇同一個策略

**可能原因**：
- Epsilon 太低
- UCB 常數太低
- min_attempts 太低

**解決**：
```python
# 提高探索率
selector.config['epsilon'] = 0.3
selector.config['ucb_c'] = 3.0
```

### 問題：探索過度，無法收斂

**可能原因**：
- Epsilon 太高
- UCB 常數太高

**解決**：
```python
# 降低探索率
selector.config['epsilon'] = 0.1
selector.config['ucb_c'] = 1.0
```

### 問題：新策略從不被選中

**可能原因**：
- UCB 方法會自動獎勵未嘗試策略
- Epsilon-Greedy 需要隨機探索

**解決**：
```python
# 使用 UCB 或提高 epsilon
strategy = selector.select(method='ucb')
```

## 參考資料

- [Multi-Armed Bandit Problem](https://en.wikipedia.org/wiki/Multi-armed_bandit)
- [UCB Algorithm](https://en.wikipedia.org/wiki/Thompson_sampling)
- [Thompson Sampling](https://proceedings.neurips.cc/paper/2011/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf)
