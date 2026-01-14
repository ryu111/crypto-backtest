"""
洞察文件管理

負責 learning/insights.md 的更新與維護。
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from src.types import ExperimentRecord


class InsightsManager:
    """
    洞察文件管理器

    職責：
    - 更新 insights.md 文件
    - 根據策略類型更新相應區塊
    - 記錄成功案例和失敗教訓

    使用範例:
        manager = InsightsManager(insights_file)
        manager.update(experiment, total_experiments)
    """

    def __init__(self, insights_file: Path):
        """
        初始化 InsightsManager

        Args:
            insights_file: insights.md 檔案路徑
        """
        self.insights_file = insights_file

        # 確保檔案存在
        if not self.insights_file.exists():
            self._init_file()

    def _init_file(self):
        """初始化 insights.md 檔案"""
        initial_content = """# 交易策略洞察彙整

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
        self.insights_file.write_text(initial_content, encoding='utf-8')

    def update(self, experiment: ExperimentRecord, total_experiments: int):
        """
        更新洞察文件

        Args:
            experiment: 實驗記錄
            total_experiments: 總實驗數
        """
        content = self.insights_file.read_text(encoding='utf-8')

        # 更新標題資訊
        content = self._update_header(content, total_experiments)

        # 根據策略類型更新相應區塊
        strategy_type = experiment.strategy_type
        content = self._update_strategy_section(content, experiment, strategy_type)

        # 更新標的特性
        symbol = experiment.symbol
        if symbol:
            content = self._update_asset_section(content, experiment, symbol)

        # 更新失敗教訓（如果驗證不通過）
        if experiment.grade in ['D', 'F']:
            content = self._update_failure_section(content, experiment)

        # 儲存
        self.insights_file.write_text(content, encoding='utf-8')

    def _update_header(self, content: str, total_experiments: int) -> str:
        """更新標題資訊"""
        content = re.sub(
            r'> 最後更新：.*\n',
            f"> 最後更新：{datetime.now().strftime('%Y-%m-%d')}\n",
            content
        )
        content = re.sub(
            r'> 總實驗數：\d+',
            f"> 總實驗數：{total_experiments}",
            content
        )
        return content

    def _update_strategy_section(
        self,
        content: str,
        experiment: ExperimentRecord,
        strategy_type: str
    ) -> str:
        """
        更新策略區塊（泛用方法，消除重複）

        Args:
            content: 當前內容
            experiment: 實驗記錄
            strategy_type: 策略類型（trend/trend_following/momentum 等）
        """
        # 映射策略類型到區塊標題
        section_map = {
            'trend': '### 趨勢跟隨策略\n',
            'trend_following': '### 趨勢跟隨策略\n',
            'momentum': '### 動量策略\n',
        }

        section = section_map.get(strategy_type)
        if not section or section not in content:
            return content

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

    def _update_asset_section(
        self,
        content: str,
        experiment: ExperimentRecord,
        symbol: str
    ) -> str:
        """更新標的特性區塊"""
        section = f"### {symbol}\n"

        if section not in content:
            return content

        start = content.find(section)
        next_section = content.find('\n### ', start + len(section))
        if next_section == -1:
            next_section = content.find('\n## ', start + len(section))

        current_section = content[start:next_section if next_section != -1 else len(content)]

        if '*尚無記錄*' in current_section:
            new_content = f"- {experiment.strategy_type} 策略表現良好（Sharpe: {experiment.sharpe_ratio:.2f}）\n"
            if experiment.insights:
                new_content += f"- {experiment.insights[0]}\n"

            content = content.replace(
                current_section,
                section + new_content
            )

        return content

    def _update_failure_section(
        self,
        content: str,
        experiment: ExperimentRecord
    ) -> str:
        """更新失敗教訓區塊"""
        section = "## 過擬合教訓\n"

        if section not in content:
            return content

        start = content.find(section)

        # 產生教訓內容
        lesson = f"\n### 失敗案例：{experiment.id}\n"
        lesson += f"- 策略：{experiment.strategy_name}\n"
        lesson += f"- 問題：{self._diagnose_failure(experiment)}\n"

        # 插入
        insert_pos = start + len(section)
        if '*尚無記錄*' in content[insert_pos:insert_pos+100]:
            content = content.replace('*尚無記錄*', lesson.strip())
        else:
            content = content[:insert_pos] + lesson + content[insert_pos:]

        return content

    def _format_strategy_insight(self, experiment: ExperimentRecord) -> str:
        """格式化策略洞察"""
        lines = [
            f"\n#### {experiment.strategy_name}",
            f"- **最佳參數**：{self._format_params(experiment.params)}",
            f"- **績效**：Sharpe {experiment.sharpe_ratio:.2f}, Return {experiment.total_return:.1%}",
        ]

        if experiment.grade:
            lines.append(f"- **驗證等級**：{experiment.grade}")

        if experiment.insights:
            lines.append(f"- **洞察**：{experiment.insights[0]}")

        return '\n'.join(lines) + '\n'

    def _format_params(self, params: Dict[str, Any]) -> str:
        """格式化參數"""
        if not params:
            return '無'

        return ', '.join(f"{k}={v}" for k, v in params.items())

    def _diagnose_failure(self, experiment: ExperimentRecord) -> str:
        """診斷失敗原因"""
        stages_passed = experiment.validation.get('stages_passed', [])
        num_stages = len(stages_passed)

        if num_stages == 0:
            return "基礎績效不達標"
        elif num_stages == 1:
            return "統計檢驗失敗，可能為隨機結果"
        elif num_stages == 2:
            return "穩健性不足（參數敏感或時間/標的不一致）"
        else:
            return "Walk-Forward 失敗，可能過擬合"
