# 實驗追蹤

回測實驗的記錄、追蹤與分析系統。

## 為什麼需要實驗追蹤？

```
問題：
- 昨天測的參數是什麼？
- 哪個版本的策略表現最好？
- 這個結果是用什麼資料測的？

實驗追蹤解決：
- 所有實驗可追溯
- 結果可重現
- 決策有依據
```

## 核心概念

### 三層結構

```
Experiment（實驗）
├── Run 1（執行）
│   ├── Parameters（參數）
│   ├── Metrics（指標）
│   └── Artifacts（產物）
├── Run 2
│   ├── Parameters
│   ├── Metrics
│   └── Artifacts
└── ...
```

### 概念對應

| 概念 | 定義 | 本專案範例 |
|------|------|------------|
| Experiment | 策略類型分組 | `trend_ma_cross_4h` |
| Run | 單次回測執行 | `run_20260112_143000` |
| Parameters | 超參數配置 | `fast=10, slow=30` |
| Metrics | 績效指標 | `sharpe=1.5, return=25%` |
| Artifacts | 產生的檔案 | 圖表、報告、模型 |

## 資料結構設計

### experiments.json 格式

```json
{
  "experiments": {
    "trend_ma_cross_4h": {
      "description": "趨勢策略：雙均線交叉 4H",
      "created_at": "2026-01-01T00:00:00Z",
      "tags": ["trend", "ma", "4h"],
      "runs": [
        {
          "run_id": "run_20260112_143000",
          "timestamp": "2026-01-12T14:30:00Z",
          "status": "completed",
          "parameters": {
            "fast_period": 10,
            "slow_period": 30,
            "stop_loss_atr": 2.0,
            "leverage": 3
          },
          "metrics": {
            "sharpe_ratio": 1.52,
            "total_return": 0.25,
            "max_drawdown": -0.12,
            "win_rate": 0.45,
            "profit_factor": 1.8,
            "n_trades": 87
          },
          "validation": {
            "passed_stages": 4,
            "grade": "B",
            "overfit_probability": 0.18
          },
          "data_version": "btc_eth_2024_v2",
          "code_version": "abc1234",
          "duration_seconds": 125,
          "artifacts": [
            "results/run_20260112_143000/equity_curve.png",
            "results/run_20260112_143000/report.html"
          ]
        }
      ],
      "best_run": "run_20260112_143000",
      "run_count": 1
    }
  },
  "metadata": {
    "version": "1.0",
    "last_updated": "2026-01-12T14:32:00Z",
    "total_runs": 1
  }
}
```

### Run 狀態

| 狀態 | 說明 |
|------|------|
| `running` | 執行中 |
| `completed` | 成功完成 |
| `failed` | 執行失敗 |
| `cancelled` | 被取消 |

## Python 實作

### ExperimentTracker 類別

```python
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import hashlib

@dataclass
class RunMetrics:
    """執行指標"""
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    n_trades: int

@dataclass
class RunValidation:
    """驗證結果"""
    passed_stages: int
    grade: str
    overfit_probability: float

@dataclass
class Run:
    """單次執行"""
    run_id: str
    timestamp: str
    status: str
    parameters: Dict[str, Any]
    metrics: Optional[RunMetrics] = None
    validation: Optional[RunValidation] = None
    data_version: Optional[str] = None
    code_version: Optional[str] = None
    duration_seconds: Optional[float] = None
    artifacts: List[str] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = []

class ExperimentTracker:
    """實驗追蹤器"""

    def __init__(self, experiments_path: str = "learning/experiments.json"):
        self.experiments_path = Path(experiments_path)
        self.experiments_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        """載入實驗資料"""
        if self.experiments_path.exists():
            with open(self.experiments_path) as f:
                self.data = json.load(f)
        else:
            self.data = {
                "experiments": {},
                "metadata": {
                    "version": "1.0",
                    "last_updated": datetime.now().isoformat(),
                    "total_runs": 0
                }
            }

    def _save(self):
        """保存實驗資料"""
        self.data["metadata"]["last_updated"] = datetime.now().isoformat()
        with open(self.experiments_path, 'w') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def create_experiment(
        self,
        experiment_name: str,
        description: str = "",
        tags: List[str] = None
    ):
        """建立新實驗"""
        if experiment_name not in self.data["experiments"]:
            self.data["experiments"][experiment_name] = {
                "description": description,
                "created_at": datetime.now().isoformat(),
                "tags": tags or [],
                "runs": [],
                "best_run": None,
                "run_count": 0
            }
            self._save()

    def start_run(
        self,
        experiment_name: str,
        parameters: Dict[str, Any],
        data_version: str = None,
        code_version: str = None
    ) -> str:
        """開始新執行"""
        if experiment_name not in self.data["experiments"]:
            self.create_experiment(experiment_name)

        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        run = Run(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            status="running",
            parameters=parameters,
            data_version=data_version,
            code_version=code_version
        )

        self.data["experiments"][experiment_name]["runs"].append(asdict(run))
        self.data["experiments"][experiment_name]["run_count"] += 1
        self.data["metadata"]["total_runs"] += 1
        self._save()

        return run_id

    def log_metrics(
        self,
        experiment_name: str,
        run_id: str,
        metrics: Dict[str, float]
    ):
        """記錄指標"""
        runs = self.data["experiments"][experiment_name]["runs"]
        for run in runs:
            if run["run_id"] == run_id:
                run["metrics"] = metrics
                break
        self._save()

    def log_validation(
        self,
        experiment_name: str,
        run_id: str,
        validation: Dict[str, Any]
    ):
        """記錄驗證結果"""
        runs = self.data["experiments"][experiment_name]["runs"]
        for run in runs:
            if run["run_id"] == run_id:
                run["validation"] = validation
                break
        self._save()

    def log_artifact(
        self,
        experiment_name: str,
        run_id: str,
        artifact_path: str
    ):
        """記錄產物"""
        runs = self.data["experiments"][experiment_name]["runs"]
        for run in runs:
            if run["run_id"] == run_id:
                if "artifacts" not in run:
                    run["artifacts"] = []
                run["artifacts"].append(artifact_path)
                break
        self._save()

    def end_run(
        self,
        experiment_name: str,
        run_id: str,
        status: str = "completed",
        duration_seconds: float = None,
        error: str = None
    ):
        """結束執行"""
        runs = self.data["experiments"][experiment_name]["runs"]
        for run in runs:
            if run["run_id"] == run_id:
                run["status"] = status
                if duration_seconds:
                    run["duration_seconds"] = duration_seconds
                if error:
                    run["error"] = error
                break

        # 更新最佳執行
        self._update_best_run(experiment_name)
        self._save()

    def _update_best_run(self, experiment_name: str):
        """更新最佳執行"""
        runs = self.data["experiments"][experiment_name]["runs"]
        completed_runs = [r for r in runs if r["status"] == "completed" and r.get("metrics")]

        if completed_runs:
            best = max(completed_runs, key=lambda r: r["metrics"].get("sharpe_ratio", 0))
            self.data["experiments"][experiment_name]["best_run"] = best["run_id"]

    def get_best_run(self, experiment_name: str) -> Optional[Dict]:
        """獲取最佳執行"""
        exp = self.data["experiments"].get(experiment_name)
        if not exp or not exp["best_run"]:
            return None

        for run in exp["runs"]:
            if run["run_id"] == exp["best_run"]:
                return run
        return None

    def compare_runs(
        self,
        experiment_name: str,
        run_ids: List[str] = None,
        metrics: List[str] = None
    ) -> List[Dict]:
        """比較多個執行"""
        exp = self.data["experiments"].get(experiment_name)
        if not exp:
            return []

        runs = exp["runs"]
        if run_ids:
            runs = [r for r in runs if r["run_id"] in run_ids]

        if metrics is None:
            metrics = ["sharpe_ratio", "total_return", "max_drawdown", "win_rate"]

        comparison = []
        for run in runs:
            if run.get("metrics"):
                row = {"run_id": run["run_id"]}
                row.update({m: run["metrics"].get(m) for m in metrics})
                comparison.append(row)

        return sorted(comparison, key=lambda x: x.get("sharpe_ratio", 0), reverse=True)
```

### 使用範例

```python
# 初始化
tracker = ExperimentTracker()

# 開始實驗
run_id = tracker.start_run(
    experiment_name="trend_ma_cross_4h",
    parameters={
        "fast_period": 10,
        "slow_period": 30,
        "stop_loss_atr": 2.0
    },
    data_version="btc_eth_2024_v2",
    code_version="abc1234"
)

# 執行回測...
result = run_backtest(...)

# 記錄指標
tracker.log_metrics(
    "trend_ma_cross_4h",
    run_id,
    {
        "sharpe_ratio": result.sharpe,
        "total_return": result.total_return,
        "max_drawdown": result.max_drawdown,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "n_trades": result.n_trades
    }
)

# 記錄驗證結果
tracker.log_validation(
    "trend_ma_cross_4h",
    run_id,
    {
        "passed_stages": 4,
        "grade": "B",
        "overfit_probability": 0.18
    }
)

# 記錄圖表
tracker.log_artifact(
    "trend_ma_cross_4h",
    run_id,
    f"results/{run_id}/equity_curve.png"
)

# 結束執行
tracker.end_run(
    "trend_ma_cross_4h",
    run_id,
    status="completed",
    duration_seconds=125.5
)
```

## 版本控制整合

### 程式碼版本

```python
import subprocess

def get_git_version() -> str:
    """獲取當前 git commit hash"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except:
        return "unknown"
```

### 資料版本

```python
import hashlib

def get_data_version(data_path: str) -> str:
    """計算資料檔案的 hash"""
    hasher = hashlib.md5()
    with open(data_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()[:8]
```

## 查詢與分析

### 查詢最佳參數

```python
def find_best_parameters(
    tracker: ExperimentTracker,
    experiment_name: str,
    min_sharpe: float = 1.0
) -> List[Dict]:
    """
    找出表現優異的參數組合
    """
    exp = tracker.data["experiments"].get(experiment_name)
    if not exp:
        return []

    good_runs = []
    for run in exp["runs"]:
        if (run["status"] == "completed" and
            run.get("metrics", {}).get("sharpe_ratio", 0) >= min_sharpe):
            good_runs.append({
                "run_id": run["run_id"],
                "parameters": run["parameters"],
                "sharpe": run["metrics"]["sharpe_ratio"],
                "return": run["metrics"]["total_return"]
            })

    return sorted(good_runs, key=lambda x: x["sharpe"], reverse=True)
```

### 參數敏感度分析

```python
import numpy as np
from typing import Tuple

def parameter_sensitivity(
    tracker: ExperimentTracker,
    experiment_name: str,
    parameter_name: str
) -> Dict:
    """
    分析單一參數對績效的影響
    """
    exp = tracker.data["experiments"].get(experiment_name)
    if not exp:
        return {}

    param_values = []
    sharpe_values = []

    for run in exp["runs"]:
        if run["status"] == "completed" and run.get("metrics"):
            param_val = run["parameters"].get(parameter_name)
            if param_val is not None:
                param_values.append(param_val)
                sharpe_values.append(run["metrics"]["sharpe_ratio"])

    if len(param_values) < 3:
        return {"error": "insufficient data"}

    # 計算相關性
    correlation = np.corrcoef(param_values, sharpe_values)[0, 1]

    # 找最佳值
    best_idx = np.argmax(sharpe_values)

    return {
        "parameter": parameter_name,
        "correlation_with_sharpe": correlation,
        "best_value": param_values[best_idx],
        "best_sharpe": sharpe_values[best_idx],
        "n_samples": len(param_values),
        "sensitivity": "high" if abs(correlation) > 0.5 else "low"
    }
```

## 與 insights.md 整合

### 自動更新洞察

```python
from pathlib import Path

def update_insights(
    tracker: ExperimentTracker,
    insights_path: str = "learning/insights.md"
) -> None:
    """
    根據實驗結果更新 insights.md
    """
    insights_file = Path(insights_path)

    # 讀取現有內容
    if insights_file.exists():
        content = insights_file.read_text()
    else:
        content = "# 策略洞察彙整\n\n"

    # 找出值得記錄的新發現
    new_insights = []

    for exp_name, exp_data in tracker.data["experiments"].items():
        best_run = tracker.get_best_run(exp_name)
        if not best_run:
            continue

        sharpe = best_run["metrics"]["sharpe_ratio"]

        # 記錄條件：Sharpe > 2.0
        if sharpe > 2.0:
            insight = f"""
#### {exp_name}
- **最佳參數**：{best_run['parameters']}
- **績效**：Sharpe {sharpe:.2f}, Return {best_run['metrics']['total_return']*100:.1f}%
- **驗證**：{best_run.get('validation', {}).get('grade', 'N/A')}
- **日期**：{best_run['timestamp'][:10]}
"""
            if insight not in content:
                new_insights.append(insight)

    if new_insights:
        # 找到適當位置插入
        if "## 策略類型洞察" in content:
            insert_pos = content.find("## 策略類型洞察") + len("## 策略類型洞察")
            content = content[:insert_pos] + "\n" + "\n".join(new_insights) + content[insert_pos:]
        else:
            content += "\n## 策略類型洞察\n" + "\n".join(new_insights)

        insights_file.write_text(content)
```

## Memory MCP 整合

### 語義搜尋歷史實驗

```python
async def search_similar_experiments(
    query: str,
    n_results: int = 5
) -> List[Dict]:
    """
    使用 Memory MCP 搜尋相似實驗

    範例查詢：
    - "高 Sharpe 的趨勢策略"
    - "低回撤的參數組合"
    - "BTC 上表現好的策略"
    """
    # 呼叫 Memory MCP
    # retrieve_memory(query=query, n_results=n_results)
    pass
```

### 儲存重要發現

```python
async def store_experiment_insight(
    content: str,
    tags: List[str]
) -> None:
    """
    將重要發現存入 Memory MCP

    範例：
    content = "trend_ma_cross_4h 在高波動期表現優異，Sharpe 2.3"
    tags = ["strategy", "trend", "high_performance"]
    """
    # 呼叫 Memory MCP
    # store_memory(content=content, metadata={"tags": tags})
    pass
```

## 最佳實踐

### 命名規範

| 項目 | 規範 | 範例 |
|------|------|------|
| Experiment | `{type}_{indicator}_{timeframe}` | `trend_ma_cross_4h` |
| Run | `run_{YYYYMMDD}_{HHMMSS}` | `run_20260112_143000` |
| Artifact | `{run_id}/{type}.{ext}` | `run_xxx/equity_curve.png` |

### 記錄什麼

| 必記錄 | 選擇性 |
|--------|--------|
| 所有參數 | 中間狀態 |
| 最終指標 | 詳細日誌 |
| 驗證結果 | 除錯資訊 |
| 資料/程式版本 | 執行環境 |

### 清理策略

```python
def cleanup_old_runs(
    tracker: ExperimentTracker,
    experiment_name: str,
    keep_best_n: int = 10,
    keep_recent_days: int = 30
) -> int:
    """
    清理舊的執行記錄

    保留：
    - 最佳 N 個執行
    - 最近 N 天的執行
    """
    from datetime import datetime, timedelta

    exp = tracker.data["experiments"].get(experiment_name)
    if not exp:
        return 0

    cutoff_date = datetime.now() - timedelta(days=keep_recent_days)

    # 排序找最佳
    completed_runs = [r for r in exp["runs"] if r["status"] == "completed"]
    sorted_runs = sorted(
        completed_runs,
        key=lambda r: r.get("metrics", {}).get("sharpe_ratio", 0),
        reverse=True
    )
    best_run_ids = {r["run_id"] for r in sorted_runs[:keep_best_n]}

    # 過濾保留
    original_count = len(exp["runs"])
    exp["runs"] = [
        r for r in exp["runs"]
        if (r["run_id"] in best_run_ids or
            datetime.fromisoformat(r["timestamp"]) > cutoff_date)
    ]

    removed = original_count - len(exp["runs"])
    tracker._save()

    return removed
```

## 參考資料

- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [Weights & Biases Experiments](https://docs.wandb.ai/guides/track/)
- [Neptune Experiment Tracking](https://docs.neptune.ai/usage/experiment_tracking/)
- [Azure ML: Log Metrics](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-log-view-metrics)
