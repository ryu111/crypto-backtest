# 黑天鵝壓力測試指南

## 概述

`StressTester` 模組用於測試交易策略在極端市場事件（黑天鵝事件）下的表現和穩健性。

## 核心功能

### 1. 歷史事件重播

內建 4 個歷史黑天鵝事件：

| 事件 | 時間 | 跌幅 | 說明 |
|------|------|------|------|
| COVID-19 崩盤 | 2020/03/12-13 | -40% | 疫情爆發引發市場崩盤 |
| 中國禁令 | 2021/05/19-23 | -30% | 中國政府加密貨幣禁令 |
| LUNA 崩盤 | 2022/05/09-12 | -50% | Terra/LUNA 生態系統崩潰 |
| FTX 倒閉 | 2022/11/08-10 | -25% | FTX 交易所破產危機 |

### 2. 自定義情境測試

可以建立任意極端情境進行測試。

### 3. 完整報告生成

自動測試所有情境並產生分析報告。

## 使用方法

### 基本使用

```python
from src.validator import StressTester, HISTORICAL_EVENTS
import pandas as pd

# 1. 準備策略報酬序列（日報酬率）
strategy_returns = pd.Series([0.01, -0.02, 0.015, ...])

# 2. 建立測試器
tester = StressTester(
    survival_threshold=-0.5,  # 虧損 50% 視為爆倉
    risk_free_rate=0.02       # 2% 無風險利率
)

# 3. 重播歷史事件
result = tester.replay_historical_event(
    strategy_returns=strategy_returns,
    event_name='covid_crash_2020'
)

# 4. 查看結果
print(f"總報酬: {result.total_return:.2%}")
print(f"最大回撤: {result.max_drawdown:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
```

### 自定義情境

```python
# 建立自定義情境
scenario = {
    'name': '極端崩盤',
    'drop': -0.70,      # 70% 下跌
    'duration': 14,     # 持續 14 天
    'description': '假設性極端黑天鵝事件'
}

# 執行測試
result = tester.run_scenario(
    strategy_returns=strategy_returns,
    scenario=scenario
)
```

### 完整報告

```python
# 產生完整壓力測試報告
report = tester.generate_stress_report(
    strategy_returns=strategy_returns,
    custom_scenarios=[
        {'name': '輕微', 'drop': -0.15, 'duration': 3},
        {'name': '中等', 'drop': -0.35, 'duration': 7},
        {'name': '嚴重', 'drop': -0.70, 'duration': 14},
    ]
)

# 格式化輸出
StressTester.print_report(report)

# 查看統計
print(f"測試情境數: {report.n_scenarios}")
print(f"存活率: {report.survival_rate:.1%}")
print(f"獲利率: {report.profit_rate:.1%}")
print(f"最差情境: {report.worst_scenario}")
```

## 輸出指標

### StressTestResult（單一測試結果）

| 欄位 | 說明 |
|------|------|
| `event_name` | 事件名稱 |
| `description` | 事件描述 |
| `drop_percentage` | 下跌幅度 |
| `duration_days` | 持續天數 |
| `total_return` | 總報酬 |
| `max_drawdown` | 最大回撤 |
| `sharpe_ratio` | Sharpe Ratio |
| `var_95` | 95% VaR |
| `cvar_95` | 95% CVaR |
| `recovery_days` | 恢復天數（None 表示未恢復） |
| `time_underwater` | 水下時間（天數） |
| `total_trades` | 總交易數 |
| `winning_trades` | 獲利交易數 |
| `losing_trades` | 虧損交易數 |
| `win_rate` | 勝率 |
| `returns` | 壓力測試後的報酬序列 |
| `equity_curve` | 權益曲線 |

### StressTestReport（完整報告）

| 欄位 | 說明 |
|------|------|
| `n_scenarios` | 測試情境總數 |
| `test_results` | 所有測試結果列表 |
| `average_return` | 平均報酬 |
| `average_max_drawdown` | 平均最大回撤 |
| `average_recovery_days` | 平均恢復天數 |
| `worst_scenario` | 最差情境名稱 |
| `worst_return` | 最差報酬 |
| `worst_drawdown` | 最差回撤 |
| `best_scenario` | 最佳情境名稱 |
| `best_return` | 最佳報酬 |
| `survival_rate` | 存活率（不爆倉比例） |
| `profit_rate` | 獲利率（獲利比例） |

## 評估標準

### 存活率（Survival Rate）

- **≥ 80%**: 優秀，策略在大多數黑天鵝中存活
- **50-80%**: 普通，需要改善風控
- **< 50%**: 危險，策略不夠穩健

### 獲利率（Profit Rate）

- **≥ 50%**: 策略在多數極端情境仍能獲利
- **30-50%**: 部分情境仍能獲利
- **< 30%**: 極端情境下容易虧損

### 恢復天數（Recovery Days）

- **≤ 30 天**: 快速恢復
- **30-90 天**: 中等恢復
- **> 90 天**: 恢復緩慢
- **無法恢復**: 策略受創嚴重

## 範例

完整範例請參考 `examples/stress_test_example.py`

```bash
python examples/stress_test_example.py
```

## 注意事項

1. **輸入格式**：`strategy_returns` 必須是 pandas Series，包含日報酬率
2. **時間範圍**：報酬序列至少需包含足夠長度以容納衝擊持續時間
3. **衝擊注入**：黑天鵝事件會隨機注入到報酬序列中（避免開頭/結尾）
4. **存活閾值**：預設 -50%，可根據策略風險承受能力調整
5. **恢復計算**：從最大回撤點到恢復新高的天數

## API 參考

### StressTester

```python
class StressTester:
    def __init__(
        self,
        survival_threshold: float = -0.5,
        risk_free_rate: float = 0.0
    ):
        """初始化壓力測試器"""

    def replay_historical_event(
        self,
        strategy_returns: pd.Series,
        event_name: str,
        custom_event: Optional[Dict] = None
    ) -> StressTestResult:
        """重播歷史事件"""

    def run_scenario(
        self,
        strategy_returns: pd.Series,
        scenario: Dict
    ) -> StressTestResult:
        """執行自定義情境"""

    def generate_stress_report(
        self,
        strategy_returns: pd.Series,
        custom_scenarios: Optional[List[Dict]] = None
    ) -> StressTestReport:
        """產生完整報告"""

    @staticmethod
    def print_result(result: StressTestResult) -> None:
        """輸出單一結果"""

    @staticmethod
    def print_report(report: StressTestReport) -> None:
        """輸出完整報告"""
```

## 測試

執行單元測試：

```bash
pytest tests/test_stress_test.py -v
```

所有測試應通過（32 個測試）。
