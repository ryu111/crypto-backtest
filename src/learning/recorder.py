"""
實驗記錄器（DuckDB 版本）

記錄回測實驗結果、更新洞察、支援查詢分析。
參考：.claude/skills/學習系統/SKILL.md

遷移說明:
    - 使用 DuckDB Repository 取代 JSON 儲存
    - 保留向後相容的介面
    - 自動遷移 experiments.json → DuckDB
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from datetime import datetime

import numpy as np
import pandas as pd

# 使用 TYPE_CHECKING 避免執行時 import（避免 vectorbt 依賴問題）
if TYPE_CHECKING:
    from ..validator.stages import ValidationResult

# Repository 和型別
from src.db import Repository, QueryFilters
from src.types import ExperimentRecord

# 使用新的類別
from .storage import TimeSeriesStorage
from .insights import InsightsManager

logger = logging.getLogger(__name__)

# 通過驗證的評級（A/B 為成功）
PASSING_GRADES = ['A', 'B']

# 查詢限制常數
MAX_QUERY_EXPERIMENTS = 10000  # 單次查詢最大實驗數
MAX_EXPORT_EXPERIMENTS = 100000  # 匯出最大實驗數


class ExperimentRecorder:
    """
    實驗記錄器（DuckDB 版本）

    功能：
    - 記錄實驗到 DuckDB
    - 更新洞察到 insights.md
    - 查詢歷史實驗
    - 分析策略演進

    遷移說明:
        自動檢測並遷移舊的 experiments.json 到 DuckDB。
        如需回退，可使用 export_to_json() 方法。

    使用範例:
        # 推薦：使用 context manager（自動關閉資源）
        with ExperimentRecorder() as recorder:
            exp_id = recorder.log_experiment(
                result=backtest_result,
                strategy_info={'name': 'ma_cross', 'type': 'trend'},
                config={'symbol': 'BTCUSDT', 'timeframe': '4h'},
                validation_result=validation_result,
                insights=['ATR 2x 止損表現更好']
            )
            best = recorder.get_best_experiments('sharpe_ratio', n=5)

        # 或手動管理資源
        recorder = ExperimentRecorder()
        try:
            recorder.log_experiment(...)
        finally:
            recorder.close()
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        insights_file: Optional[Path] = None,
        experiments_file: Optional[Path] = None  # 僅用於遷移
    ):
        """
        初始化記錄器

        Args:
            db_path: DuckDB 資料庫路徑（預設: data/experiments.duckdb）
            insights_file: 洞察 MD 檔案路徑（預設: learning/insights.md）
            experiments_file: 舊的 JSON 檔案路徑（用於自動遷移）
        """
        # 確定專案根目錄
        current_file = Path(__file__)
        self.project_root = current_file.parent.parent.parent

        # DuckDB 路徑
        self.db_path = self._validate_path(
            db_path or self.project_root / 'data' / 'experiments.duckdb'
        )

        # Insights 路徑
        insights_file_path = self._validate_path(
            insights_file or self.project_root / 'learning' / 'insights.md'
        )

        # 舊 JSON 檔案路徑（用於遷移）
        self.legacy_json_file = self._validate_path(
            experiments_file or self.project_root / 'learning' / 'experiments.json'
        )

        # 確保目錄存在
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 初始化 Repository
        self.repo = Repository(str(self.db_path))

        # 初始化子元件
        self.storage = TimeSeriesStorage(self.project_root)
        self.insights_manager = InsightsManager(insights_file_path)

        # 自動遷移（如果 JSON 存在且 DB 是空的）
        self._auto_migrate_if_needed()

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

    def _auto_migrate_if_needed(self):
        """自動遷移 JSON → DuckDB"""
        # 檢查是否需要遷移
        if not self.legacy_json_file.exists():
            return  # 無 JSON 檔案，不需遷移

        # 檢查 DB 是否有資料
        try:
            count = self.repo.conn.execute(
                "SELECT COUNT(*) FROM experiments"
            ).fetchone()[0]

            if count > 0:
                return  # DB 已有資料，不需遷移
        except Exception:
            pass  # DB 可能還沒建立 table，繼續遷移

        # 執行遷移
        logger.info(f"檢測到 {self.legacy_json_file}，開始遷移到 DuckDB...")
        try:
            migrated = self.migrate_from_json()
            logger.info(f"成功遷移 {migrated} 筆實驗記錄")

            # 遷移成功後，重命名原檔案作為備份
            backup_path = self.legacy_json_file.with_suffix('.json.migrated')
            self.legacy_json_file.rename(backup_path)
            logger.info(f"已將 {self.legacy_json_file.name} 備份至 {backup_path.name}")

        except Exception as e:
            logger.error(f"遷移失敗: {e}", exc_info=True)
            logger.warning("將繼續使用 DuckDB，但可能無歷史資料。請檢查錯誤日誌。")

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
        exp_id = self._generate_experiment_id(config.get('symbol', ''), strategy_info.get('name', ''))

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
                'stages_passed': validation_result.passed_stages,
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
            parent = self.repo.get_experiment(parent_experiment)
            if parent:
                current_sharpe = results.get('sharpe_ratio', 0)
                parent_sharpe = parent.sharpe_ratio
                if parent_sharpe != 0:
                    improvement = (current_sharpe - parent_sharpe) / abs(parent_sharpe)

        # 建立 ExperimentRecord
        experiment = ExperimentRecord(
            id=exp_id,
            timestamp=datetime.now(),
            strategy={
                'name': strategy_info.get('name', ''),
                'type': strategy_info.get('type', ''),
                'version': strategy_info.get('version', '1.0'),
                'params': self._extract_params(result),
            },
            config={
                'symbol': config.get('symbol', ''),
                'timeframe': config.get('timeframe', ''),
                'start_date': config.get('start_date'),
                'end_date': config.get('end_date'),
            },
            results=results,
            validation={
                'grade': validation.get('grade', 'F'),
                'stages_passed': validation.get('stages_passed', []),
            },
            status='completed',
            insights=insights or [],
            tags=tags,
            parent_experiment=parent_experiment,
            improvement=improvement,
        )

        # 儲存到 DuckDB
        self.repo.insert_experiment(experiment)

        # 儲存時間序列資料（委派給 storage）
        self.storage.save(exp_id, result)

        # 更新洞察文件（委派給 insights_manager）
        if insights:
            # 取得總實驗數
            total = self.repo.conn.execute(
                "SELECT COUNT(*) FROM experiments"
            ).fetchone()[0]

            # 直接使用 ExperimentRecord
            self.insights_manager.update(experiment, total)

        return exp_id

    def get_experiment(self, exp_id: str) -> Optional[ExperimentRecord]:
        """
        取得單一實驗

        Args:
            exp_id: 實驗 ID

        Returns:
            ExperimentRecord 或 None
        """
        return self.repo.get_experiment(exp_id)

    def query_experiments(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ExperimentRecord]:
        """
        查詢實驗

        Args:
            filters: 過濾條件（舊格式，向後相容）
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
            List[ExperimentRecord]: 符合條件的實驗列表
        """
        # 轉換舊格式 filters → QueryFilters
        query_filters = self._convert_filters(filters or {})
        return self.repo.query_experiments(query_filters)

    def get_best_experiments(
        self,
        metric: str = 'sharpe_ratio',
        n: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ExperimentRecord]:
        """
        取得最佳 N 個實驗

        Args:
            metric: 排序指標（sharpe_ratio, total_return, profit_factor 等）
            n: 取得數量
            filters: 額外過濾條件

        Returns:
            List[ExperimentRecord]: 最佳實驗列表
        """
        query_filters = self._convert_filters(filters or {})
        return self.repo.get_best_experiments(metric, n, query_filters)

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
        # 使用 LIKE 查詢（效能優於全表掃描）
        related = self.repo.query_experiments_by_strategy_prefix(
            strategy_name_prefix=strategy_name,
            limit=MAX_QUERY_EXPERIMENTS
        )

        # 按時間排序
        related.sort(key=lambda e: e.timestamp)

        evolution = []
        for i, exp in enumerate(related):
            entry = {
                'version': exp.strategy.get('version', f'{i+1}.0'),
                'date': exp.timestamp,
                'exp_id': exp.id,
                'sharpe': exp.sharpe_ratio,
                'return': exp.total_return,
                'changes': exp.insights,
                'improvement': exp.improvement
            }

            evolution.append(entry)

        return evolution

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

    # ========== 時間序列資料（委派） ==========

    def load_equity_curve(self, exp_id: str) -> Optional[pd.Series]:
        """
        載入實驗的權益曲線（委派給 storage）

        Args:
            exp_id: 實驗 ID

        Returns:
            pd.Series: 權益曲線（index 為日期），如果不存在則返回 None
        """
        return self.storage.load_equity_curve(exp_id)

    def load_daily_returns(self, exp_id: str) -> Optional[pd.Series]:
        """
        載入實驗的每日收益率（委派給 storage）

        Args:
            exp_id: 實驗 ID

        Returns:
            pd.Series: 每日收益率（index 為日期），如果不存在則返回 None
        """
        return self.storage.load_daily_returns(exp_id)

    def load_trades(self, exp_id: str) -> Optional[pd.DataFrame]:
        """
        載入實驗的交易記錄（委派給 storage）

        Args:
            exp_id: 實驗 ID

        Returns:
            pd.DataFrame: 交易記錄，如果不存在則返回 None
        """
        return self.storage.load_trades(exp_id)

    # ========== 策略統計 ==========

    def get_strategy_stats(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """
        從歷史實驗中提取策略統計

        Args:
            strategy_name: 策略名稱（可為前綴匹配，如 'ma_cross'）

        Returns:
            Optional[Dict]: 策略統計，如果無歷史記錄則返回 None
            {
                'name': str,
                'attempts': int,        # 嘗試次數
                'successes': int,       # 成功次數（A/B 評級）
                'avg_sharpe': float,    # 平均 Sharpe
                'best_sharpe': float,   # 最佳 Sharpe
                'last_updated': datetime  # 最後更新時間
            }
        """
        # 使用 LIKE 查詢（效能優於全表掃描）
        related = self.repo.query_experiments_by_strategy_prefix(
            strategy_name_prefix=strategy_name,
            limit=MAX_QUERY_EXPERIMENTS
        )

        # 無歷史記錄
        if not related:
            return None

        # 計算統計
        attempts = len(related)

        # 成功次數（A/B 評級）
        successes = sum(
            1 for e in related
            if e.grade in PASSING_GRADES
        )

        # Sharpe 比率列表
        sharpe_list = [e.sharpe_ratio for e in related]

        avg_sharpe = float(np.mean(sharpe_list))
        best_sharpe = float(np.max(sharpe_list))

        # 最後更新時間
        last_updated = max(e.timestamp for e in related)

        return {
            'name': strategy_name,
            'attempts': attempts,
            'successes': successes,
            'avg_sharpe': avg_sharpe,
            'best_sharpe': best_sharpe,
            'last_updated': last_updated
        }

    def update_strategy_stats(
        self,
        strategy_name: str,
        stats: Dict[str, Any]
    ) -> bool:
        """
        更新策略的最近一筆實驗記錄

        此方法用於 StrategySelector 更新策略績效追蹤。它會找到指定策略的
        最近一筆實驗記錄，並更新其 results 或 validation 欄位。

        Args:
            strategy_name: 策略名稱（需完全匹配，如 'trend_ma_cross'）
            stats: 要更新的欄位
                - results 欄位：'sharpe_ratio', 'total_return', 'max_drawdown' 等
                - validation 欄位：'grade'（會自動識別並放入 validation）

        Returns:
            bool: 是否成功更新

        Note:
            DuckDB 不支援 UPDATE，此方法僅保留向後相容性。
            實際上會記錄 warning 並返回 False。

        Example:
            >>> recorder.update_strategy_stats('trend_ma_cross', {
            ...     'sharpe_ratio': 1.5,
            ...     'grade': 'B'
            ... })
        """
        logger.warning(
            f"update_strategy_stats() 不支援 DuckDB 模式。"
            f"請使用 log_experiment() 記錄新實驗。"
            f"策略: {strategy_name}, 統計: {stats}"
        )
        return False

    # ========== 遷移和匯出 ==========

    def migrate_from_json(self) -> int:
        """
        從 experiments.json 遷移到 DuckDB

        Returns:
            int: 遷移的實驗數量
        """
        if not self.legacy_json_file.exists():
            logger.warning(f"找不到 {self.legacy_json_file}")
            return 0

        # 讀取 JSON
        with open(self.legacy_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        experiments = data.get('experiments', [])

        # 逐筆插入
        migrated = 0
        for exp_data in experiments:
            try:
                # 轉換為 ExperimentRecord
                record = self._legacy_to_experiment_record(exp_data)
                self.repo.insert_experiment(record)
                migrated += 1
            except Exception as e:
                logger.error(f"遷移實驗 {exp_data.get('id')} 失敗: {e}")

        logger.info(f"成功遷移 {migrated}/{len(experiments)} 筆實驗")
        return migrated

    def export_to_json(self, output_file: Optional[Path] = None) -> Path:
        """
        匯出 DuckDB → JSON（用於備份或回退）

        Args:
            output_file: 輸出檔案路徑（預設: learning/experiments_backup.json）

        Returns:
            Path: 輸出檔案路徑
        """
        output = output_file or self.project_root / 'learning' / 'experiments_backup.json'

        # 查詢所有實驗
        all_experiments = self.repo.query_experiments(
            QueryFilters(limit=MAX_EXPORT_EXPERIMENTS)
        )

        # 轉換為 JSON 格式
        experiments_data = [exp.to_dict() for exp in all_experiments]

        # 組織結構
        data = {
            'version': '1.0',
            'metadata': {
                'total_experiments': len(all_experiments),
                'last_updated': datetime.now().isoformat(),
                'best_strategy': None  # TODO: 計算最佳策略
            },
            'experiments': experiments_data
        }

        # 寫入檔案
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"已匯出 {len(all_experiments)} 筆實驗到 {output}")
        return output

    # ========== 私有方法 ==========

    def _generate_experiment_id(self, symbol: str = '', strategy_name: str = '') -> str:
        """生成實驗 ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        parts = ['exp', timestamp]

        if symbol:
            parts.append(symbol.replace('USDT', '').lower())
        if strategy_name:
            parts.append(strategy_name.lower())

        return '_'.join(parts)

    def _convert_filters(self, old_filters: Dict[str, Any]) -> QueryFilters:
        """
        轉換舊格式 filters → QueryFilters

        Args:
            old_filters: 舊格式過濾條件

        Returns:
            QueryFilters
        """
        kwargs = {}

        # 直接對應的欄位
        for key in ['strategy_type', 'symbol', 'timeframe', 'min_sharpe', 'max_drawdown', 'grade', 'tags']:
            if key in old_filters:
                kwargs[key] = old_filters[key]

        # 日期範圍
        if 'date_range' in old_filters:
            start, end = old_filters['date_range']
            kwargs['start_date'] = start if isinstance(start, str) else start.isoformat()
            kwargs['end_date'] = end if isinstance(end, str) else end.isoformat()

        return QueryFilters(**kwargs)

    def _extract_params(self, result: Any) -> Dict[str, Any]:
        """
        提取 BacktestResult 的參數（處理新舊格式）

        Args:
            result: BacktestResult 物件

        Returns:
            參數字典
        """
        # 優先使用新格式 'params'
        if hasattr(result, 'params'):
            return result.params

        # 向後相容舊格式 'parameters'
        if hasattr(result, 'parameters'):
            logger.warning(
                "BacktestResult 使用已棄用的 'parameters' 屬性，"
                "請改用 'params'"
            )
            return result.parameters

        # 無參數
        return {}

    def _legacy_to_experiment_record(self, exp_data: Dict[str, Any]) -> ExperimentRecord:
        """
        轉換舊格式實驗記錄 → ExperimentRecord

        Args:
            exp_data: 舊格式實驗資料（JSON）

        Returns:
            ExperimentRecord
        """
        # 使用 ExperimentRecord.from_dict（已處理向後相容）
        return ExperimentRecord.from_dict(exp_data)

    def __enter__(self):
        """Context manager 入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 出口（自動關閉資源）"""
        self.close()
        return False  # 不吞掉異常

    def close(self):
        """明確關閉資源（建議使用 context manager 或手動呼叫）"""
        if hasattr(self, 'repo') and self.repo:
            try:
                self.repo.close()
            except Exception as e:
                logger.warning(f"關閉 Repository 時發生錯誤: {e}")

    def __del__(self):
        """析構函數（fallback，不保證呼叫）"""
        # 嘗試清理，但不依賴此方法
        try:
            self.close()
        except Exception:
            pass  # 忽略析構時的錯誤
