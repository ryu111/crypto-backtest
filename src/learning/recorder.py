"""
實驗記錄器

記錄回測實驗結果、更新洞察、支援查詢分析。
參考：.claude/skills/學習系統/SKILL.md
"""

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING
from collections import defaultdict
import pandas as pd

# 使用 TYPE_CHECKING 避免執行時 import（避免 vectorbt 依賴問題）
if TYPE_CHECKING:
    from ..validator.stages import ValidationResult


@dataclass
class Experiment:
    """實驗記錄"""

    # 基本資訊
    id: str  # exp_20260111_120000
    timestamp: datetime

    # 策略資訊
    strategy: Dict[str, Any]  # name, type, version

    # 配置
    config: Dict[str, Any]  # symbol, timeframe, period, capital, leverage

    # 參數
    parameters: Dict[str, Any]

    # 績效結果
    results: Dict[str, float]  # return, sharpe, drawdown, win_rate, etc.

    # 驗證結果
    validation: Dict[str, Any]  # grade, passed_stages, efficiency

    # 洞察
    insights: List[str] = field(default_factory=list)

    # 標籤
    tags: List[str] = field(default_factory=list)

    # 演進追蹤
    parent_experiment: Optional[str] = None
    improvement: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典（JSON 序列化）"""
        data = asdict(self)
        # 轉換 datetime 為 ISO 格式
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experiment':
        """從字典建立"""
        data = data.copy()
        # 轉換 ISO 格式為 datetime
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class ExperimentRecorder:
    """
    實驗記錄器

    功能：
    - 記錄實驗到 experiments.json
    - 更新洞察到 insights.md
    - 查詢歷史實驗
    - 分析策略演進

    使用範例:
        recorder = ExperimentRecorder()

        # 記錄實驗
        exp_id = recorder.log_experiment(
            result=backtest_result,
            strategy_info={'name': 'ma_cross', 'type': 'trend'},
            config={'symbol': 'BTCUSDT', 'timeframe': '4h'},
            validation_result=validation_result,
            insights=['ATR 2x 止損表現更好']
        )

        # 查詢最佳策略
        best = recorder.get_best_experiments('sharpe_ratio', n=5)

        # 追蹤演進
        evolution = recorder.get_strategy_evolution('ma_cross')
    """

    def __init__(
        self,
        experiments_file: Optional[Path] = None,
        insights_file: Optional[Path] = None
    ):
        """
        初始化記錄器

        Args:
            experiments_file: 實驗 JSON 檔案路徑（預設: learning/experiments.json）
            insights_file: 洞察 MD 檔案路徑（預設: learning/insights.md）
        """
        # 確定專案根目錄
        current_file = Path(__file__)
        self.project_root = current_file.parent.parent.parent

        self.experiments_file = self._validate_path(
            experiments_file or self.project_root / 'learning' / 'experiments.json'
        )
        self.insights_file = self._validate_path(
            insights_file or self.project_root / 'learning' / 'insights.md'
        )

        # 確保目錄存在
        self.experiments_file.parent.mkdir(parents=True, exist_ok=True)

        # 初始化檔案（如果不存在）
        self._init_files()

    def _validate_path(self, path: Path) -> Path:
        """
        驗證路徑在專案目錄內

        Args:
            path: 要驗證的路徑

        Returns:
            Path: 驗證後的路徑

        Raises:
            ValueError: 路徑在專案目錄外
        """
        resolved = path.resolve()
        project_resolved = self.project_root.resolve()

        if not str(resolved).startswith(str(project_resolved)):
            raise ValueError(f"Path {path} is outside project directory")

        return resolved

    def _init_files(self):
        """初始化檔案"""
        # 初始化 experiments.json
        if not self.experiments_file.exists():
            initial_data = {
                'version': '1.0',
                'metadata': {
                    'total_experiments': 0,
                    'last_updated': None,
                    'best_strategy': None
                },
                'experiments': []
            }
            self._save_experiments(initial_data)

        # 初始化 insights.md
        if not self.insights_file.exists():
            initial_insights = """# 交易策略洞察彙整

> 最後更新：待更新
> 總實驗數：0

## 策略類型洞察

### 趨勢跟隨策略
*尚無記錄*

### 動量策略
*尚無記錄*

## 標的特性

### BTCUSDT
*尚無記錄*

### ETHUSDT
*尚無記錄*

## 風險管理洞察
*尚無記錄*

## 過擬合教訓
*尚無記錄*
"""
            self.insights_file.write_text(initial_insights, encoding='utf-8')

    def log_experiment(
        self,
        result: Any,  # BacktestResult
        strategy_info: Dict[str, Any],
        config: Dict[str, Any],
        validation_result: Optional[Any] = None,  # ValidationResult
        insights: Optional[List[str]] = None,
        parent_experiment: Optional[str] = None
    ) -> str:
        """
        記錄實驗結果

        Args:
            result: BacktestResult 物件
            strategy_info: 策略資訊 {'name': str, 'type': str, 'version': str}
            config: 回測配置 {'symbol': str, 'timeframe': str, ...}
            validation_result: ValidationResult 物件（可選）
            insights: 洞察列表（可選）
            parent_experiment: 父實驗 ID（用於追蹤演進）

        Returns:
            str: 實驗 ID
        """
        # 生成實驗 ID
        exp_id = self._generate_experiment_id()

        # 提取結果數據
        results = {
            'total_return': result.total_return,
            'annual_return': result.annual_return,
            'sharpe_ratio': result.sharpe_ratio,
            'sortino_ratio': result.sortino_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'profit_factor': result.profit_factor,
            'total_trades': result.total_trades,
            'avg_trade_duration': result.avg_trade_duration,
            'expectancy': result.expectancy,
        }

        # 提取驗證結果
        validation = {}
        if validation_result:
            validation = {
                'grade': validation_result.grade.value,
                'passed_stages': validation_result.passed_stages,
                'stage_results': {
                    name: {
                        'passed': stage.passed,
                        'score': stage.score,
                        'message': stage.message
                    }
                    for name, stage in validation_result.stage_results.items()
                }
            }

            # 提取關鍵指標
            if '階段4_WalkForward' in validation_result.stage_results:
                wfa = validation_result.stage_results['階段4_WalkForward']
                validation['walk_forward_efficiency'] = wfa.details.get('efficiency', 0)

            if '階段5_MonteCarlo' in validation_result.stage_results:
                mc = validation_result.stage_results['階段5_MonteCarlo']
                validation['monte_carlo_p5'] = mc.details.get('p5', 0)

        # 生成標籤
        tags = self.generate_tags(strategy_info, config, validation)

        # 計算改進程度
        improvement = None
        if parent_experiment:
            improvement = self._calculate_improvement(
                exp_id, parent_experiment, results
            )

        # 建立實驗物件
        experiment = Experiment(
            id=exp_id,
            timestamp=datetime.now(),
            strategy=strategy_info,
            config=config,
            parameters=getattr(result, 'parameters', {}),
            results=results,
            validation=validation,
            insights=insights or [],
            tags=tags,
            parent_experiment=parent_experiment,
            improvement=improvement
        )

        # 儲存實驗
        self._append_experiment(experiment)

        # 儲存時間序列資料（equity_curve, daily_returns, trades）
        self._save_timeseries_data(exp_id, result)

        # 更新洞察文件
        if insights:
            self.update_insights(experiment)

        return exp_id

    def get_experiment(self, exp_id: str) -> Optional[Experiment]:
        """
        取得單一實驗

        Args:
            exp_id: 實驗 ID

        Returns:
            Experiment 或 None
        """
        data = self._load_experiments()

        for exp_data in data['experiments']:
            if exp_data['id'] == exp_id:
                return Experiment.from_dict(exp_data)

        return None

    def query_experiments(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Experiment]:
        """
        查詢實驗

        Args:
            filters: 過濾條件
                {
                    'strategy_type': 'trend',
                    'symbol': 'BTCUSDT',
                    'min_sharpe': 1.0,
                    'max_drawdown': 0.20,
                    'grade': ['A', 'B'],
                    'tags': ['validated'],
                    'date_range': ('2026-01-01', '2026-01-11')
                }

        Returns:
            List[Experiment]: 符合條件的實驗列表
        """
        data = self._load_experiments()
        experiments = [Experiment.from_dict(e) for e in data['experiments']]

        if not filters:
            return experiments

        # 應用過濾器
        filtered = experiments

        if 'strategy_type' in filters:
            filtered = [
                e for e in filtered
                if e.strategy.get('type') == filters['strategy_type']
            ]

        if 'symbol' in filters:
            filtered = [
                e for e in filtered
                if e.config.get('symbol') == filters['symbol']
            ]

        if 'min_sharpe' in filters:
            filtered = [
                e for e in filtered
                if e.results.get('sharpe_ratio', 0) >= filters['min_sharpe']
            ]

        if 'max_drawdown' in filters:
            filtered = [
                e for e in filtered
                if abs(e.results.get('max_drawdown', 1)) <= filters['max_drawdown']
            ]

        if 'grade' in filters:
            grades = filters['grade'] if isinstance(filters['grade'], list) else [filters['grade']]
            filtered = [
                e for e in filtered
                if e.validation.get('grade') in grades
            ]

        if 'tags' in filters:
            required_tags = set(filters['tags'])
            filtered = [
                e for e in filtered
                if required_tags.issubset(set(e.tags))
            ]

        if 'date_range' in filters:
            start_str, end_str = filters['date_range']
            start_date = datetime.fromisoformat(start_str) if isinstance(start_str, str) else start_str
            end_date = datetime.fromisoformat(end_str) if isinstance(end_str, str) else end_str

            filtered = [
                e for e in filtered
                if start_date <= e.timestamp <= end_date
            ]

        return filtered

    def get_best_experiments(
        self,
        metric: str = 'sharpe_ratio',
        n: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Experiment]:
        """
        取得最佳 N 個實驗

        Args:
            metric: 排序指標（sharpe_ratio, total_return, profit_factor 等）
            n: 取得數量
            filters: 額外過濾條件

        Returns:
            List[Experiment]: 最佳實驗列表
        """
        experiments = self.query_experiments(filters)

        # 排序
        sorted_experiments = sorted(
            experiments,
            key=lambda e: e.results.get(metric, float('-inf')),
            reverse=True
        )

        return sorted_experiments[:n]

    def get_strategy_evolution(
        self,
        strategy_name: str
    ) -> List[Dict[str, Any]]:
        """
        追蹤策略演進

        Args:
            strategy_name: 策略名稱（可為前綴，如 'ma_cross'）

        Returns:
            List[Dict]: 演進歷史
                [
                    {
                        'version': '1.0',
                        'date': datetime,
                        'exp_id': 'exp_...',
                        'sharpe': 1.5,
                        'return': 0.45,
                        'changes': ['...'],
                        'improvement': 0.12
                    },
                    ...
                ]
        """
        # 過濾相關實驗
        data = self._load_experiments()
        related = [
            Experiment.from_dict(e)
            for e in data['experiments']
            if e['strategy']['name'].startswith(strategy_name)
        ]

        # 按時間排序
        related.sort(key=lambda e: e.timestamp)

        evolution = []
        for i, exp in enumerate(related):
            entry = {
                'version': exp.strategy.get('version', f'{i+1}.0'),
                'date': exp.timestamp,
                'exp_id': exp.id,
                'sharpe': exp.results.get('sharpe_ratio', 0),
                'return': exp.results.get('total_return', 0),
                'changes': exp.insights,
                'improvement': exp.improvement
            }

            evolution.append(entry)

        return evolution

    def update_insights(self, experiment: Experiment):
        """
        更新洞察文件

        Args:
            experiment: 實驗物件
        """
        # 讀取現有洞察
        content = self.insights_file.read_text(encoding='utf-8')

        # 更新標題資訊
        data = self._load_experiments()
        total = data['metadata']['total_experiments']

        content = re.sub(
            r'> 最後更新：.*\n',
            f"> 最後更新：{datetime.now().strftime('%Y-%m-%d')}\n",
            content
        )
        content = re.sub(
            r'> 總實驗數：\d+',
            f"> 總實驗數：{total}",
            content
        )

        # 根據策略類型更新相應區塊
        strategy_type = experiment.strategy.get('type', 'unknown')

        if strategy_type == 'trend' or strategy_type == 'trend_following':
            content = self._update_trend_insights(content, experiment)
        elif strategy_type == 'momentum':
            content = self._update_momentum_insights(content, experiment)

        # 更新標的特性
        symbol = experiment.config.get('symbol', '')
        if symbol:
            content = self._update_asset_insights(content, experiment, symbol)

        # 更新失敗教訓（如果驗證不通過）
        if experiment.validation.get('grade') in ['D', 'F']:
            content = self._update_failure_lessons(content, experiment)

        # 儲存
        self.insights_file.write_text(content, encoding='utf-8')

    def generate_tags(
        self,
        strategy_info: Dict[str, Any],
        config: Dict[str, Any],
        validation: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        自動產生標籤

        Args:
            strategy_info: 策略資訊
            config: 配置
            validation: 驗證結果（可選）

        Returns:
            List[str]: 標籤列表
        """
        tags = []

        # 資產類別
        tags.append('crypto')

        # 標的
        symbol = config.get('symbol', '').lower()
        if 'btc' in symbol:
            tags.append('btc')
        if 'eth' in symbol:
            tags.append('eth')

        # 策略類型
        strategy_type = strategy_info.get('type', '')
        if strategy_type:
            tags.append(strategy_type)

        # 策略名稱關鍵字
        strategy_name = strategy_info.get('name', '').lower()
        if 'ma' in strategy_name:
            tags.append('ma')
        if 'rsi' in strategy_name:
            tags.append('rsi')
        if 'macd' in strategy_name:
            tags.append('macd')
        if 'supertrend' in strategy_name:
            tags.append('supertrend')

        # 時間框架
        timeframe = config.get('timeframe', '')
        if timeframe:
            tags.append(timeframe)

        # 驗證狀態
        if validation:
            grade = validation.get('grade')
            if grade in ['A', 'B']:
                tags.append('validated')
            elif grade == 'C':
                tags.append('testing')
            else:
                tags.append('failed')

        return list(set(tags))  # 去重

    # ===== 私有方法 =====

    def _generate_experiment_id(self) -> str:
        """生成實驗 ID"""
        return f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _load_experiments(self) -> Dict[str, Any]:
        """載入實驗數據"""
        if not self.experiments_file.exists():
            return self._get_empty_data()

        try:
            with open(self.experiments_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"JSON 解析錯誤: {e}")
            return self._get_empty_data()

    def _get_empty_data(self) -> Dict[str, Any]:
        """返回空的實驗數據結構"""
        return {
            'version': '1.0',
            'metadata': {
                'total_experiments': 0,
                'last_updated': None,
                'best_strategy': None
            },
            'experiments': []
        }

    def _save_experiments(self, data: Dict[str, Any]):
        """儲存實驗數據"""
        with open(self.experiments_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _append_experiment(self, experiment: Experiment):
        """添加實驗到記錄"""
        data = self._load_experiments()

        # 添加實驗
        data['experiments'].append(experiment.to_dict())

        # 更新 metadata
        data['metadata']['total_experiments'] += 1
        data['metadata']['last_updated'] = datetime.now().isoformat()

        # 更新最佳策略
        current_best_id = data['metadata'].get('best_strategy')
        if self._is_better_than_current_best(experiment, current_best_id, data):
            data['metadata']['best_strategy'] = experiment.id

        # 儲存
        self._save_experiments(data)

    def _is_better_than_current_best(
        self,
        experiment: Experiment,
        current_best_id: Optional[str],
        data: Dict[str, Any]
    ) -> bool:
        """判斷是否優於當前最佳"""
        # 必須通過驗證
        if experiment.validation.get('grade') not in ['A', 'B']:
            return False

        # 如果沒有最佳策略，直接成為最佳
        if not current_best_id:
            return True

        # 比較 Sharpe Ratio
        current_best = next(
            (e for e in data['experiments'] if e['id'] == current_best_id),
            None
        )

        if not current_best:
            return True

        return (
            experiment.results.get('sharpe_ratio', 0) >
            current_best['results'].get('sharpe_ratio', 0)
        )

    def _calculate_improvement(
        self,
        exp_id: str,
        parent_id: str,
        results: Dict[str, float]
    ) -> Optional[float]:
        """計算相對於父實驗的改進"""
        parent = self.get_experiment(parent_id)

        if not parent:
            return None

        # 比較 Sharpe Ratio
        current_sharpe = results.get('sharpe_ratio', 0)
        parent_sharpe = parent.results.get('sharpe_ratio', 0)

        if parent_sharpe == 0:
            return None

        return (current_sharpe - parent_sharpe) / abs(parent_sharpe)

    def _update_trend_insights(
        self,
        content: str,
        experiment: Experiment
    ) -> str:
        """更新趨勢策略洞察"""
        section = "### 趨勢跟隨策略\n"

        strategy_name = experiment.strategy.get('name', '')

        # 檢查是否已有該策略的記錄
        if section in content:
            # 找到區塊位置
            start = content.find(section)
            next_section = content.find('\n### ', start + len(section))
            if next_section == -1:
                next_section = content.find('\n## ', start + len(section))

            current_section = content[start:next_section if next_section != -1 else len(content)]

            # 如果是空的，替換
            if '*尚無記錄*' in current_section:
                new_content = self._format_strategy_insight(experiment)
                content = content.replace(
                    current_section,
                    section + new_content
                )
            else:
                # 添加到現有內容
                new_insight = f"\n{self._format_strategy_insight(experiment)}"
                insert_pos = next_section if next_section != -1 else len(content)
                content = content[:insert_pos] + new_insight + content[insert_pos:]

        return content

    def _update_momentum_insights(
        self,
        content: str,
        experiment: Experiment
    ) -> str:
        """更新動量策略洞察"""
        # 類似 _update_trend_insights
        section = "### 動量策略\n"

        if section in content:
            start = content.find(section)
            next_section = content.find('\n### ', start + len(section))
            if next_section == -1:
                next_section = content.find('\n## ', start + len(section))

            current_section = content[start:next_section if next_section != -1 else len(content)]

            if '*尚無記錄*' in current_section:
                new_content = self._format_strategy_insight(experiment)
                content = content.replace(
                    current_section,
                    section + new_content
                )

        return content

    def _update_asset_insights(
        self,
        content: str,
        experiment: Experiment,
        symbol: str
    ) -> str:
        """更新標的特性洞察"""
        section = f"### {symbol}\n"

        if section in content:
            start = content.find(section)
            next_section = content.find('\n### ', start + len(section))
            if next_section == -1:
                next_section = content.find('\n## ', start + len(section))

            current_section = content[start:next_section if next_section != -1 else len(content)]

            if '*尚無記錄*' in current_section:
                new_content = f"- {experiment.strategy['type']} 策略表現良好（Sharpe: {experiment.results['sharpe_ratio']:.2f}）\n"
                if experiment.insights:
                    new_content += f"- {experiment.insights[0]}\n"

                content = content.replace(
                    current_section,
                    section + new_content
                )

        return content

    def _update_failure_lessons(
        self,
        content: str,
        experiment: Experiment
    ) -> str:
        """更新失敗教訓"""
        section = "## 過擬合教訓\n"

        if section in content:
            start = content.find(section)

            # 產生教訓內容
            lesson = f"\n### 失敗案例：{experiment.id}\n"
            lesson += f"- 策略：{experiment.strategy['name']}\n"
            lesson += f"- 問題：{self._diagnose_failure(experiment)}\n"

            # 插入
            insert_pos = start + len(section)
            if '*尚無記錄*' in content[insert_pos:insert_pos+100]:
                content = content.replace('*尚無記錄*', lesson.strip())
            else:
                content = content[:insert_pos] + lesson + content[insert_pos:]

        return content

    def _format_strategy_insight(self, experiment: Experiment) -> str:
        """格式化策略洞察"""
        lines = [
            f"\n#### {experiment.strategy['name']}",
            f"- **最佳參數**：{self._format_params(experiment.parameters)}",
            f"- **績效**：Sharpe {experiment.results['sharpe_ratio']:.2f}, Return {experiment.results['total_return']:.1%}",
        ]

        if experiment.validation.get('grade'):
            lines.append(f"- **驗證等級**：{experiment.validation['grade']}")

        if experiment.insights:
            lines.append(f"- **洞察**：{experiment.insights[0]}")

        return '\n'.join(lines) + '\n'

    def _format_params(self, params: Dict[str, Any]) -> str:
        """格式化參數"""
        if not params:
            return '無'

        return ', '.join(f"{k}={v}" for k, v in params.items())

    def _diagnose_failure(self, experiment: Experiment) -> str:
        """診斷失敗原因"""
        validation = experiment.validation

        if validation.get('passed_stages', 0) == 0:
            return "基礎績效不達標"
        elif validation.get('passed_stages', 0) == 1:
            return "統計檢驗失敗，可能為隨機結果"
        elif validation.get('passed_stages', 0) == 2:
            return "穩健性不足（參數敏感或時間/標的不一致）"
        else:
            return "Walk-Forward 失敗，可能過擬合"

    def _save_timeseries_data(self, exp_id: str, result: Any):
        """
        儲存時間序列資料到獨立 CSV 檔案

        Args:
            exp_id: 實驗 ID
            result: BacktestResult 物件

        儲存結構:
            results/{exp_id}/
                ├── equity_curve.csv    (index=date, columns=['equity'])
                ├── daily_returns.csv   (index=date, columns=['return'])
                └── trades.csv          (交易記錄)
        """
        # 建立實驗專屬目錄
        exp_dir = self.project_root / 'results' / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # 1. 儲存 equity_curve
        if hasattr(result, 'equity_curve') and result.equity_curve is not None:
            equity_df = pd.DataFrame({
                'equity': result.equity_curve
            })
            # 確保索引名稱為 'date'
            equity_df.index.name = 'date'
            equity_df.to_csv(exp_dir / 'equity_curve.csv')

        # 2. 儲存 daily_returns
        if hasattr(result, 'daily_returns') and result.daily_returns is not None:
            returns_df = pd.DataFrame({
                'return': result.daily_returns
            })
            # 確保索引名稱為 'date'
            returns_df.index.name = 'date'
            returns_df.to_csv(exp_dir / 'daily_returns.csv')

        # 3. 儲存 trades（如果存在）
        if hasattr(result, 'trades') and result.trades is not None and len(result.trades) > 0:
            result.trades.to_csv(exp_dir / 'trades.csv', index=False)

    def load_equity_curve(self, exp_id: str) -> Optional[pd.Series]:
        """
        載入實驗的權益曲線

        Args:
            exp_id: 實驗 ID

        Returns:
            pd.Series: 權益曲線（index 為日期），如果不存在則返回 None
        """
        equity_file = self.project_root / 'results' / exp_id / 'equity_curve.csv'

        if not equity_file.exists():
            return None

        try:
            df = pd.read_csv(equity_file, index_col='date', parse_dates=True)
            return df['equity']
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"載入 equity_curve 失敗 ({exp_id}): {e}")
            return None

    def load_daily_returns(self, exp_id: str) -> Optional[pd.Series]:
        """
        載入實驗的每日收益率

        Args:
            exp_id: 實驗 ID

        Returns:
            pd.Series: 每日收益率（index 為日期），如果不存在則返回 None
        """
        returns_file = self.project_root / 'results' / exp_id / 'daily_returns.csv'

        if not returns_file.exists():
            return None

        try:
            df = pd.read_csv(returns_file, index_col='date', parse_dates=True)
            return df['return']
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"載入 daily_returns 失敗 ({exp_id}): {e}")
            return None

    def load_trades(self, exp_id: str) -> Optional[pd.DataFrame]:
        """
        載入實驗的交易記錄

        Args:
            exp_id: 實驗 ID

        Returns:
            pd.DataFrame: 交易記錄，如果不存在則返回 None
        """
        trades_file = self.project_root / 'results' / exp_id / 'trades.csv'

        if not trades_file.exists():
            return None

        try:
            return pd.read_csv(trades_file)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"載入 trades 失敗 ({exp_id}): {e}")
            return None
