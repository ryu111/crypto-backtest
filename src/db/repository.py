"""
DuckDB Repository 實作

提供實驗記錄和策略統計的資料庫操作。
"""

import json
import logging
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

import duckdb

from src.types import ExperimentRecord, StrategyStats

logger = logging.getLogger(__name__)

# 安全的排序欄位白名單（防止 SQL Injection）
VALID_ORDER_COLUMNS = frozenset([
    'sharpe_ratio', 'total_return', 'sortino_ratio',
    'calmar_ratio', 'profit_factor', 'win_rate'
])


@dataclass
class QueryFilters:
    """
    實驗查詢過濾器

    使用範例:
        filters = QueryFilters(
            strategy_name="ma_cross",
            min_sharpe=1.5,
            grade=["A", "B"],
            limit=50
        )
    """

    strategy_name: Optional[str] = None
    strategy_type: Optional[str] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    min_sharpe: Optional[float] = None
    max_drawdown: Optional[float] = None
    grade: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    limit: int = 100
    offset: int = 0


class Repository:
    """
    DuckDB Repository

    管理實驗記錄和策略統計的資料庫操作。

    使用範例:
        # 使用 context manager（推薦）
        with Repository("data/experiments.duckdb") as repo:
            repo.insert_experiment(record)
            experiments = repo.query_experiments(filters)

        # 或手動管理
        repo = Repository("data/experiments.duckdb")
        try:
            repo.insert_experiment(record)
        finally:
            repo.close()
    """

    def __init__(self, db_path: str):
        """
        初始化 Repository

        Args:
            db_path: 資料庫檔案路徑（會自動建立目錄）
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 連接資料庫
        self.conn = duckdb.connect(str(self.db_path))

        # 初始化 schema
        self._init_schema()

    def _init_schema(self) -> None:
        """執行 schema.sql 初始化表結構"""
        schema_file = Path(__file__).parent / "schema.sql"
        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_file}")

        schema_sql = schema_file.read_text()
        self.conn.execute(schema_sql)

    def _build_where_clause(self, filters: 'QueryFilters') -> Tuple[str, List[Any]]:
        """
        建構 WHERE 子句（消除重複程式碼）

        Args:
            filters: 查詢過濾器

        Returns:
            (WHERE SQL 字串, 參數列表)
        """
        where_clauses = []
        params = []

        if filters.strategy_name:
            where_clauses.append("strategy_name = ?")
            params.append(filters.strategy_name)

        if filters.strategy_type:
            where_clauses.append("strategy_type = ?")
            params.append(filters.strategy_type)

        if filters.symbol:
            where_clauses.append("symbol = ?")
            params.append(filters.symbol)

        if filters.timeframe:
            where_clauses.append("timeframe = ?")
            params.append(filters.timeframe)

        if filters.min_sharpe is not None:
            where_clauses.append("sharpe_ratio >= ?")
            params.append(filters.min_sharpe)

        if filters.max_drawdown is not None:
            where_clauses.append("max_drawdown <= ?")
            params.append(filters.max_drawdown)

        if filters.grade:
            if not isinstance(filters.grade, list):
                raise ValueError("grade must be a list")
            if not all(isinstance(g, str) for g in filters.grade):
                raise ValueError("all grades must be strings")
            placeholders = ','.join(['?' for _ in filters.grade])
            where_clauses.append(f"grade IN ({placeholders})")
            params.extend(filters.grade)

        if filters.start_date:
            # 使用日期開始時間（00:00:00）
            where_clauses.append("timestamp >= ?")
            params.append(f"{filters.start_date} 00:00:00" if len(filters.start_date) == 10 else filters.start_date)

        if filters.end_date:
            # 使用日期結束時間（23:59:59）以包含整天
            where_clauses.append("timestamp <= ?")
            params.append(f"{filters.end_date} 23:59:59" if len(filters.end_date) == 10 else filters.end_date)

        # Tags 查詢（JSON 陣列包含）
        if filters.tags:
            if not isinstance(filters.tags, list):
                raise ValueError("tags must be a list")
            if not all(isinstance(tag, str) for tag in filters.tags):
                raise ValueError("all tags must be strings")
            for tag in filters.tags:
                # DuckDB 的 JSON 函數：檢查 JSON 是否包含指定值
                where_clauses.append("json_contains(tags::JSON, ?)")
                params.append(f'"{tag}"')  # JSON 字串需要加引號

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        return where_sql, params

    def __enter__(self):
        """Context manager 入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 出口"""
        self.close()

    def close(self) -> None:
        """關閉資料庫連接"""
        if self.conn:
            self.conn.close()

    def insert_experiment(self, record: ExperimentRecord) -> None:
        """
        插入實驗記錄

        Args:
            record: 實驗記錄

        Raises:
            ValueError: 如果記錄格式錯誤
        """
        # 準備資料
        data = {
            'id': record.id,
            'timestamp': record.timestamp,
            'strategy_name': record.strategy_name,
            'strategy_type': record.strategy_type,
            'strategy_version': record.strategy.get('version', '1.0'),
            'params': json.dumps(record.params),
            'symbol': record.symbol,
            'timeframe': record.timeframe,
            'start_date': record.config.get('start_date'),
            'end_date': record.config.get('end_date'),
            'sharpe_ratio': record.sharpe_ratio,
            'total_return': record.total_return,
            'max_drawdown': record.max_drawdown,
            'win_rate': record.results.get('win_rate'),
            'profit_factor': record.results.get('profit_factor'),
            'total_trades': record.results.get('total_trades'),
            'sortino_ratio': record.results.get('sortino_ratio'),
            'calmar_ratio': record.results.get('calmar_ratio'),
            'grade': record.grade,
            'stages_passed': json.dumps(record.validation.get('stages_passed', [])),
            'status': record.status,
            'insights': json.dumps(record.insights),
            'tags': json.dumps(record.tags),
            'parent_experiment': record.parent_experiment,
            'improvement': record.improvement,
        }

        # 插入資料
        self.conn.execute("""
            INSERT INTO experiments (
                id, timestamp, strategy_name, strategy_type, strategy_version,
                params, symbol, timeframe, start_date, end_date,
                sharpe_ratio, total_return, max_drawdown, win_rate,
                profit_factor, total_trades, sortino_ratio, calmar_ratio,
                grade, stages_passed, status, insights, tags,
                parent_experiment, improvement
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?
            )
        """, list(data.values()))

    def get_experiment(self, id: str) -> Optional[ExperimentRecord]:
        """
        取得單一實驗

        Args:
            id: 實驗 ID

        Returns:
            實驗記錄，如果不存在則回傳 None
        """
        result = self.conn.execute(
            "SELECT * FROM experiments WHERE id = ?", [id]
        ).fetchone()

        if not result:
            return None

        return self._row_to_experiment(result)

    def query_experiments(self, filters: QueryFilters) -> List[ExperimentRecord]:
        """
        查詢實驗記錄

        Args:
            filters: 查詢過濾器

        Returns:
            符合條件的實驗記錄列表
        """
        # 使用共用方法建構 WHERE 子句
        where_sql, params = self._build_where_clause(filters)

        # 組合 SQL
        sql = f"""
            SELECT * FROM experiments
            WHERE {where_sql}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """
        params.extend([filters.limit, filters.offset])

        # 執行查詢
        results = self.conn.execute(sql, params).fetchall()
        return [self._row_to_experiment(row) for row in results]

    def query_experiments_by_strategy_prefix(
        self,
        strategy_name_prefix: str,
        limit: int = 1000
    ) -> List[ExperimentRecord]:
        """
        查詢策略名稱前綴匹配的實驗（使用 LIKE 查詢，效能優於全表掃描）

        Args:
            strategy_name_prefix: 策略名稱前綴（如 'ma_cross'）
            limit: 結果數量限制

        Returns:
            符合條件的實驗記錄列表
        """
        sql = """
            SELECT * FROM experiments
            WHERE strategy_name LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        results = self.conn.execute(
            sql,
            [f"{strategy_name_prefix}%", limit]
        ).fetchall()
        return [self._row_to_experiment(row) for row in results]

    def get_best_experiments(
        self,
        metric: str = "sharpe_ratio",
        n: int = 10,
        filters: Optional[QueryFilters] = None
    ) -> List[ExperimentRecord]:
        """
        取得最佳 N 個實驗

        Args:
            metric: 排序指標（sharpe_ratio, total_return, calmar_ratio 等）
            n: 取得數量
            filters: 額外過濾條件

        Returns:
            最佳實驗記錄列表
        """
        # 使用白名單常數驗證 metric（防止 SQL Injection）
        if metric not in VALID_ORDER_COLUMNS:
            raise ValueError(f"Invalid metric: {metric}. Must be one of {list(VALID_ORDER_COLUMNS)}")

        # 使用 filters 建構基礎查詢
        if filters:
            base_filters = replace(filters, limit=n, offset=0)
        else:
            base_filters = QueryFilters(limit=n)

        # 使用共用方法建構 WHERE 子句
        where_sql, params = self._build_where_clause(base_filters)

        # 組合 SQL（metric 已通過白名單驗證，安全使用）
        sql = f"""
            SELECT * FROM experiments
            WHERE {where_sql}
            ORDER BY {metric} DESC NULLS LAST
            LIMIT ?
        """
        params.append(n)

        # 執行查詢
        results = self.conn.execute(sql, params).fetchall()
        return [self._row_to_experiment(row) for row in results]

    def update_strategy_stats(self, stats: StrategyStats) -> None:
        """
        更新策略統計

        Args:
            stats: 策略統計資料
        """
        data = {
            'name': stats.name,
            'attempts': stats.attempts,
            'successes': stats.successes,
            'avg_sharpe': stats.avg_sharpe,
            'best_sharpe': stats.best_sharpe,
            'worst_sharpe': stats.worst_sharpe,
            'best_params': json.dumps(stats.best_params) if stats.best_params else None,
            'last_params': json.dumps(stats.last_params) if stats.last_params else None,
            'last_attempt': stats.last_attempt,
            'first_attempt': stats.first_attempt,
            'ucb_score': stats.ucb_score,
        }

        # Upsert（插入或更新）
        # 注意：DuckDB 在 VALUES 中使用 CURRENT_TIMESTAMP，UPDATE SET 中需要提供值
        now = datetime.now()

        self.conn.execute("""
            INSERT INTO strategy_stats (
                name, attempts, successes, avg_sharpe, best_sharpe,
                worst_sharpe, best_params, last_params, last_attempt,
                first_attempt, ucb_score, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (name) DO UPDATE SET
                attempts = EXCLUDED.attempts,
                successes = EXCLUDED.successes,
                avg_sharpe = EXCLUDED.avg_sharpe,
                best_sharpe = EXCLUDED.best_sharpe,
                worst_sharpe = EXCLUDED.worst_sharpe,
                best_params = EXCLUDED.best_params,
                last_params = EXCLUDED.last_params,
                last_attempt = EXCLUDED.last_attempt,
                first_attempt = EXCLUDED.first_attempt,
                ucb_score = EXCLUDED.ucb_score,
                updated_at = EXCLUDED.updated_at
        """, [*list(data.values()), now])

    def get_strategy_stats(self, name: str) -> Optional[StrategyStats]:
        """
        取得策略統計

        Args:
            name: 策略名稱

        Returns:
            策略統計，如果不存在則回傳 None
        """
        result = self.conn.execute(
            "SELECT * FROM strategy_stats WHERE name = ?", [name]
        ).fetchone()

        if not result:
            return None

        return self._row_to_strategy_stats(result)

    def get_all_strategy_stats(self) -> List[StrategyStats]:
        """
        取得所有策略統計

        Returns:
            策略統計列表
        """
        results = self.conn.execute(
            "SELECT * FROM strategy_stats ORDER BY ucb_score DESC"
        ).fetchall()

        return [self._row_to_strategy_stats(row) for row in results]

    def _safe_json_loads(self, value: Any, default: Any, field_name: str) -> Any:
        """
        安全的 JSON 反序列化

        Args:
            value: 要解析的值
            default: 解析失敗時的預設值
            field_name: 欄位名稱（用於錯誤日誌）

        Returns:
            解析結果或預設值
        """
        if not value:
            return default
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"JSON 解析失敗 [{field_name}]: {e}")
            return default

    def _row_to_experiment(self, row) -> ExperimentRecord:
        """
        將資料庫 row 轉換為 ExperimentRecord

        Args:
            row: DuckDB 查詢結果

        Returns:
            實驗記錄
        """
        # DuckDB 回傳的是 tuple，需要對應欄位名稱
        columns = [desc[0] for desc in self.conn.description]
        data = dict(zip(columns, row))

        # 安全解析 JSON 欄位
        params = self._safe_json_loads(data['params'], {}, 'params')
        stages_passed = self._safe_json_loads(data['stages_passed'], [], 'stages_passed')
        insights = self._safe_json_loads(data['insights'], [], 'insights')
        tags = self._safe_json_loads(data['tags'], [], 'tags')

        # 重建 ExperimentRecord 結構
        return ExperimentRecord(
            id=data['id'],
            timestamp=data['timestamp'],
            strategy={
                'name': data['strategy_name'],
                'type': data['strategy_type'],
                'version': data['strategy_version'],
                'params': params,
            },
            config={
                'symbol': data['symbol'],
                'timeframe': data['timeframe'],
                'start_date': data['start_date'],
                'end_date': data['end_date'],
            },
            results={
                'sharpe_ratio': data['sharpe_ratio'],
                'total_return': data['total_return'],
                'max_drawdown': data['max_drawdown'],
                'win_rate': data['win_rate'],
                'profit_factor': data['profit_factor'],
                'total_trades': data['total_trades'],
                'sortino_ratio': data['sortino_ratio'],
                'calmar_ratio': data['calmar_ratio'],
            },
            validation={
                'grade': data['grade'],
                'stages_passed': stages_passed,
            },
            status=data['status'],
            insights=insights,
            tags=tags,
            parent_experiment=data['parent_experiment'],
            improvement=data['improvement'],
        )

    def _row_to_strategy_stats(self, row) -> StrategyStats:
        """
        將資料庫 row 轉換為 StrategyStats

        Args:
            row: DuckDB 查詢結果

        Returns:
            策略統計
        """
        columns = [desc[0] for desc in self.conn.description]
        data = dict(zip(columns, row))

        # 安全解析 JSON 欄位
        best_params = self._safe_json_loads(data['best_params'], None, 'best_params')
        last_params = self._safe_json_loads(data['last_params'], None, 'last_params')

        return StrategyStats(
            name=data['name'],
            attempts=data['attempts'],
            successes=data['successes'],
            avg_sharpe=data['avg_sharpe'],
            best_sharpe=data['best_sharpe'],
            worst_sharpe=data['worst_sharpe'],
            best_params=best_params,
            last_params=last_params,
            last_attempt=data['last_attempt'],
            first_attempt=data['first_attempt'],
            ucb_score=data['ucb_score'],
        )
