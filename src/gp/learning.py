"""
GP 學習系統整合

記錄 GP 演化實驗到 learning 系統。

使用範例:
    integrator = GPLearningIntegration()

    # 記錄演化實驗
    exp_id = integrator.record_evolution(
        result=evolution_result,
        metadata={'symbol': 'BTCUSDT'}
    )

    # 記錄到洞察
    integrator.record_to_insights(evolution_result)
"""

from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import logging

from ..learning.recorder import ExperimentRecorder
from .engine import EvolutionResult

logger = logging.getLogger(__name__)


# ============================================================================
# 常數定義
# ============================================================================

# 值得記錄到 insights.md 的適應度閾值
INSIGHT_FITNESS_THRESHOLD = 1.0


# ============================================================================
# GP 學習系統整合
# ============================================================================

class GPLearningIntegration:
    """
    GP 學習系統整合

    記錄演化實驗結果，包括：
    - 最佳個體和適應度
    - 演化統計（代數、早停）
    - Hall of Fame

    使用範例:
        integrator = GPLearningIntegration()

        # 記錄演化實驗
        exp_id = integrator.record_evolution(
            result=evolution_result,
            metadata={'symbol': 'BTCUSDT'}
        )

        # 記錄到洞察
        integrator.record_to_insights(evolution_result)
    """

    def __init__(self, recorder: Optional[ExperimentRecorder] = None):
        """
        初始化學習系統整合

        Args:
            recorder: ExperimentRecorder 實例（可選，未提供則建立新實例）
        """
        self.recorder = recorder

    def record_evolution(
        self,
        result: EvolutionResult,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        記錄演化實驗

        Args:
            result: EvolutionResult 演化結果
            metadata: 額外元資料
                - symbol: 交易標的
                - timeframe: 時間框架
                - primitive_set: 原語集類型

        Returns:
            str: 實驗 ID

        Note:
            記錄到 experiments.duckdb，不是 insights.md（除非呼叫 record_to_insights）
        """
        # 確保 recorder 存在
        if self.recorder is None:
            self.recorder = ExperimentRecorder()

        # 準備實驗資料
        experiment_data = self._prepare_experiment_data(result, metadata)

        # 記錄到資料庫
        # 注意：ExperimentRecorder.log_experiment 需要 BacktestResult 和 ValidationResult
        # 但 GP 演化結果不是標準回測，我們需要適配

        # 建立簡化的記錄（使用 JSON 格式儲存到 insights.md）
        # 或者擴展 ExperimentRecorder 支援 GP 演化記錄

        # 目前簡化實作：生成實驗 ID 並記錄到日誌
        exp_id = self._generate_experiment_id(metadata)

        logger.info(
            f"記錄 GP 演化實驗: {exp_id} "
            f"(fitness={result.best_fitness:.4f}, "
            f"generations={result.generations_run})"
        )

        # TODO: 整合到 ExperimentRecorder
        # 需要擴展 ExperimentRecorder 支援 GP 演化記錄類型
        # 或者建立獨立的 GP 演化記錄系統

        return exp_id

    def record_to_insights(
        self,
        result: EvolutionResult,
        insight_type: str = 'gp_evolution'
    ):
        """
        記錄到 insights.md

        只有當結果優於閾值時才記錄。

        Args:
            result: EvolutionResult 演化結果
            insight_type: 洞察類型（預設 'gp_evolution'）

        Note:
            - 適應度 >= INSIGHT_FITNESS_THRESHOLD 才記錄
            - 寫入 learning/insights.md
        """
        # 檢查是否值得記錄
        if result.best_fitness < INSIGHT_FITNESS_THRESHOLD:
            logger.info(
                f"適應度 {result.best_fitness:.4f} < {INSIGHT_FITNESS_THRESHOLD}，"
                f"不記錄到 insights.md"
            )
            return

        # 準備洞察內容
        insight = self._format_insight(result, insight_type)

        # 寫入 insights.md
        insights_file = self._get_insights_file()

        try:
            # 追加到檔案末尾
            with open(insights_file, 'a', encoding='utf-8') as f:
                f.write('\n\n')
                f.write(insight)

            logger.info(f"已記錄洞察到 {insights_file}")

        except Exception as e:
            logger.error(f"記錄洞察失敗: {e}", exc_info=True)

    def _prepare_experiment_data(
        self,
        result: EvolutionResult,
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        準備實驗資料

        Args:
            result: 演化結果
            metadata: 額外元資料

        Returns:
            Dict: 實驗資料字典
        """
        data = {
            'type': 'gp_evolution',
            'best_fitness': float(result.best_fitness),
            'generations_run': int(result.generations_run),
            'stopped_early': bool(result.stopped_early),
            'elapsed_time': float(result.elapsed_time),
            'population_size': int(result.config.population_size),
            'best_expression': str(result.best_individual),
            'hall_of_fame_size': len(result.hall_of_fame),
            'timestamp': datetime.now().isoformat()
        }

        # 合併額外元資料
        if metadata:
            data.update(metadata)

        return data

    def _generate_experiment_id(self, metadata: Optional[Dict[str, Any]]) -> str:
        """
        生成實驗 ID

        Args:
            metadata: 元資料（包含 symbol 等）

        Returns:
            str: 實驗 ID（格式: exp_gp_YYYYMMDD_HHMMSS_symbol）
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        parts = ['exp', 'gp', timestamp]

        if metadata and 'symbol' in metadata:
            symbol = metadata['symbol'].replace('USDT', '').lower()
            parts.append(symbol)

        return '_'.join(parts)

    def _format_insight(self, result: EvolutionResult, insight_type: str) -> str:
        """
        格式化洞察內容

        Args:
            result: 演化結果
            insight_type: 洞察類型

        Returns:
            str: Markdown 格式的洞察
        """
        # 格式化最佳表達式
        best_expr = str(result.best_individual)

        # 計算改善統計
        fitness_history = result.fitness_history
        initial_fitness = fitness_history[0] if fitness_history else 0.0
        final_fitness = result.best_fitness
        improvement = final_fitness - initial_fitness

        # 生成洞察內容
        insight = f"""#### GP 演化洞察

- **類型**: {insight_type}
- **最佳適應度**: {result.best_fitness:.4f}
- **演化代數**: {result.generations_run}
- **早停**: {'是' if result.stopped_early else '否'}
- **執行時間**: {result.elapsed_time:.2f}s
- **改善幅度**: {improvement:+.4f} ({initial_fitness:.4f} → {final_fitness:.4f})
- **最佳表達式**: `{best_expr}`
- **日期**: {datetime.now().strftime('%Y-%m-%d')}

**洞察**:
- GP 演化成功找到適應度 {result.best_fitness:.4f} 的策略
- 經過 {result.generations_run} 代演化，相比初始族群改善 {improvement:.4f}
- {'提前停止（無顯著改善）' if result.stopped_early else '完成全部演化'}

**Hall of Fame** (前 5 名):
"""

        # 添加 Hall of Fame
        for i, individual in enumerate(result.hall_of_fame[:5]):
            fitness = individual.fitness.values[0]
            expr = str(individual)
            insight += f"\n{i+1}. 適應度 {fitness:.4f}: `{expr}`"

        return insight

    def _get_insights_file(self) -> Path:
        """
        取得 insights.md 檔案路徑

        Returns:
            Path: insights.md 路徑
        """
        # 專案根目錄
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent

        insights_file = project_root / 'learning' / 'insights.md'

        # 確保目錄存在
        insights_file.parent.mkdir(parents=True, exist_ok=True)

        # 確保檔案存在
        if not insights_file.exists():
            insights_file.write_text(
                "# 策略洞察彙整\n\n自動記錄優秀策略的洞察和發現。\n",
                encoding='utf-8'
            )

        return insights_file


# 公開 API
__all__ = [
    'GPLearningIntegration',
]
