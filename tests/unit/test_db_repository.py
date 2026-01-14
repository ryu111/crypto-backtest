"""
DuckDB Repository 測試

測試項目：
1. 基本 CRUD 操作
2. QueryFilters 查詢功能
3. get_best_experiments 排序功能
4. SQL Injection 防護
5. JSON 解析錯誤處理
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import tempfile
from datetime import datetime
from src.db.repository import Repository, QueryFilters, VALID_ORDER_COLUMNS
from src.types import ExperimentRecord, StrategyStats


# ========== Fixtures ==========

@pytest.fixture
def temp_db():
    """建立臨時資料庫"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/test.duckdb"
        yield db_path


@pytest.fixture
def repo(temp_db):
    """建立 Repository 實例"""
    with Repository(temp_db) as r:
        yield r


@pytest.fixture
def sample_experiment():
    """建立範例實驗記錄"""
    return ExperimentRecord(
        id="exp_test_001",
        timestamp=datetime(2024, 1, 1, 12, 0),
        strategy={
            'name': 'ma_cross',
            'type': 'trend',
            'version': '1.0',
            'params': {'fast_period': 10, 'slow_period': 30}
        },
        config={
            'symbol': 'BTCUSDT',
            'timeframe': '4h',
            'start_date': '2020-01-01',
            'end_date': '2024-01-01'
        },
        results={
            'sharpe_ratio': 1.85,
            'total_return': 0.92,
            'max_drawdown': 0.18,
            'win_rate': 0.58,
            'profit_factor': 2.1,
            'total_trades': 245,
            'sortino_ratio': 2.2,
            'calmar_ratio': 10.2
        },
        validation={
            'grade': 'A',
            'stages_passed': [1, 2, 3, 4, 5]
        },
        status='completed',
        insights=['MA Cross works well on BTC 4h'],
        tags=['validated', 'production'],
    )


@pytest.fixture
def sample_strategy_stats():
    """建立範例策略統計"""
    return StrategyStats(
        name='ma_cross',
        attempts=10,
        successes=3,
        avg_sharpe=1.2,
        best_sharpe=2.1,
        worst_sharpe=0.5,
        best_params={'fast_period': 10, 'slow_period': 30},
        last_params={'fast_period': 12, 'slow_period': 28},
        last_attempt=datetime(2024, 1, 1, 12, 0),
        first_attempt=datetime(2023, 1, 1, 12, 0),
        ucb_score=1.5,
    )


# ========== Test 1: 基本 CRUD 操作 ==========

def test_insert_and_get_experiment(repo, sample_experiment):
    """測試插入和讀取實驗記錄"""
    # Insert
    repo.insert_experiment(sample_experiment)

    # Get
    retrieved = repo.get_experiment(sample_experiment.id)

    # Verify
    assert retrieved is not None
    assert retrieved.id == sample_experiment.id
    assert retrieved.strategy_name == 'ma_cross'
    assert retrieved.sharpe_ratio == 1.85
    assert retrieved.grade == 'A'
    assert 'validated' in retrieved.tags
    assert 'MA Cross works well on BTC 4h' in retrieved.insights


def test_get_nonexistent_experiment(repo):
    """測試讀取不存在的實驗"""
    result = repo.get_experiment("nonexistent_id")
    assert result is None


def test_insert_multiple_experiments(repo, sample_experiment):
    """測試插入多筆實驗"""
    # 插入 3 筆不同的實驗
    for i in range(3):
        exp = ExperimentRecord(
            id=f"exp_test_{i:03d}",
            timestamp=datetime(2024, 1, i+1, 12, 0),
            strategy=sample_experiment.strategy.copy(),
            config=sample_experiment.config.copy(),
            results=sample_experiment.results.copy(),
            validation=sample_experiment.validation.copy(),
            status='completed',
        )
        repo.insert_experiment(exp)

    # 驗證全部插入成功
    for i in range(3):
        result = repo.get_experiment(f"exp_test_{i:03d}")
        assert result is not None


def test_update_strategy_stats(repo, sample_strategy_stats):
    """測試更新策略統計"""
    # Insert
    repo.update_strategy_stats(sample_strategy_stats)

    # Get
    retrieved = repo.get_strategy_stats('ma_cross')

    # Verify
    assert retrieved is not None
    assert retrieved.name == 'ma_cross'
    assert retrieved.attempts == 10
    assert retrieved.successes == 3
    assert retrieved.avg_sharpe == 1.2
    assert retrieved.best_params == {'fast_period': 10, 'slow_period': 30}


def test_upsert_strategy_stats(repo, sample_strategy_stats):
    """測試策略統計的 Upsert 功能"""
    # 第一次插入
    repo.update_strategy_stats(sample_strategy_stats)

    # 修改並更新
    sample_strategy_stats.attempts = 15
    sample_strategy_stats.successes = 5
    sample_strategy_stats.avg_sharpe = 1.5
    repo.update_strategy_stats(sample_strategy_stats)

    # 驗證更新成功
    retrieved = repo.get_strategy_stats('ma_cross')
    assert retrieved.attempts == 15
    assert retrieved.successes == 5
    assert retrieved.avg_sharpe == 1.5


def test_get_all_strategy_stats(repo, sample_strategy_stats):
    """測試取得所有策略統計"""
    # 插入多個策略
    for name in ['ma_cross', 'rsi_bb', 'breakout']:
        stats = StrategyStats(
            name=name,
            attempts=10,
            avg_sharpe=1.0 + hash(name) % 10 / 10,
            ucb_score=1.0 + hash(name) % 10 / 10,
        )
        repo.update_strategy_stats(stats)

    # 取得全部
    all_stats = repo.get_all_strategy_stats()

    # 驗證
    assert len(all_stats) == 3
    names = {s.name for s in all_stats}
    assert names == {'ma_cross', 'rsi_bb', 'breakout'}

    # 驗證按 ucb_score 排序
    for i in range(len(all_stats) - 1):
        assert all_stats[i].ucb_score >= all_stats[i+1].ucb_score


def test_get_nonexistent_strategy_stats(repo):
    """測試取得不存在的策略統計"""
    result = repo.get_strategy_stats("nonexistent_strategy")
    assert result is None


# ========== Test 2: QueryFilters 查詢功能 ==========

def test_query_by_strategy_name(repo, sample_experiment):
    """測試按策略名稱查詢"""
    # 插入不同策略
    repo.insert_experiment(sample_experiment)

    exp2 = ExperimentRecord(
        id="exp_test_002",
        timestamp=datetime(2024, 1, 2, 12, 0),
        strategy={'name': 'rsi_bb', 'type': 'mean_reversion', 'version': '1.0'},
        config=sample_experiment.config.copy(),
        results=sample_experiment.results.copy(),
        validation=sample_experiment.validation.copy(),
    )
    repo.insert_experiment(exp2)

    # 查詢 ma_cross
    filters = QueryFilters(strategy_name='ma_cross')
    results = repo.query_experiments(filters)

    # 驗證
    assert len(results) == 1
    assert results[0].strategy_name == 'ma_cross'


def test_query_by_symbol(repo, sample_experiment):
    """測試按標的查詢"""
    repo.insert_experiment(sample_experiment)

    exp2 = ExperimentRecord(
        id="exp_test_002",
        timestamp=datetime(2024, 1, 2, 12, 0),
        strategy=sample_experiment.strategy.copy(),
        config={'symbol': 'ETHUSDT', 'timeframe': '4h'},
        results=sample_experiment.results.copy(),
        validation=sample_experiment.validation.copy(),
    )
    repo.insert_experiment(exp2)

    # 查詢 BTC
    filters = QueryFilters(symbol='BTCUSDT')
    results = repo.query_experiments(filters)

    assert len(results) == 1
    assert results[0].config['symbol'] == 'BTCUSDT'


def test_query_by_strategy_type(repo, sample_experiment):
    """測試按策略類型查詢"""
    repo.insert_experiment(sample_experiment)

    exp2 = ExperimentRecord(
        id="exp_test_002",
        timestamp=datetime(2024, 1, 2, 12, 0),
        strategy={'name': 'rsi_bb', 'type': 'mean_reversion', 'version': '1.0'},
        config=sample_experiment.config.copy(),
        results=sample_experiment.results.copy(),
        validation=sample_experiment.validation.copy(),
    )
    repo.insert_experiment(exp2)

    # 查詢 trend 類型
    filters = QueryFilters(strategy_type='trend')
    results = repo.query_experiments(filters)

    assert len(results) == 1
    assert results[0].strategy_type == 'trend'


def test_query_by_timeframe(repo, sample_experiment):
    """測試按時間框架查詢"""
    repo.insert_experiment(sample_experiment)

    exp2 = ExperimentRecord(
        id="exp_test_002",
        timestamp=datetime(2024, 1, 2, 12, 0),
        strategy=sample_experiment.strategy.copy(),
        config={'symbol': 'BTCUSDT', 'timeframe': '1h'},
        results=sample_experiment.results.copy(),
        validation=sample_experiment.validation.copy(),
    )
    repo.insert_experiment(exp2)

    # 查詢 4h
    filters = QueryFilters(timeframe='4h')
    results = repo.query_experiments(filters)

    assert len(results) == 1
    assert results[0].config['timeframe'] == '4h'


def test_query_by_max_drawdown(repo, sample_experiment):
    """測試按最大回撤查詢"""
    # 插入不同回撤的實驗
    for i, dd in enumerate([0.10, 0.15, 0.20, 0.25, 0.30]):
        exp = ExperimentRecord(
            id=f"exp_test_{i:03d}",
            timestamp=datetime(2024, 1, i+1, 12, 0),
            strategy=sample_experiment.strategy.copy(),
            config=sample_experiment.config.copy(),
            results={'sharpe_ratio': 1.5, 'total_return': 0.5, 'max_drawdown': dd},
            validation=sample_experiment.validation.copy(),
        )
        repo.insert_experiment(exp)

    # 查詢 MaxDD <= 0.20
    filters = QueryFilters(max_drawdown=0.20)
    results = repo.query_experiments(filters)

    assert len(results) == 3  # 0.10, 0.15, 0.20
    for r in results:
        assert r.max_drawdown <= 0.20


def test_query_by_min_sharpe(repo, sample_experiment):
    """測試按最小 Sharpe Ratio 查詢"""
    # 插入不同 Sharpe 的實驗
    for i, sharpe in enumerate([0.5, 1.0, 1.5, 2.0, 2.5]):
        exp = ExperimentRecord(
            id=f"exp_test_{i:03d}",
            timestamp=datetime(2024, 1, i+1, 12, 0),
            strategy=sample_experiment.strategy.copy(),
            config=sample_experiment.config.copy(),
            results={'sharpe_ratio': sharpe, 'total_return': 0.5, 'max_drawdown': 0.2},
            validation=sample_experiment.validation.copy(),
        )
        repo.insert_experiment(exp)

    # 查詢 Sharpe >= 1.5
    filters = QueryFilters(min_sharpe=1.5)
    results = repo.query_experiments(filters)

    # 驗證
    assert len(results) == 3  # 1.5, 2.0, 2.5
    for r in results:
        assert r.sharpe_ratio >= 1.5


def test_query_by_grade(repo, sample_experiment):
    """測試按等級查詢"""
    # 插入不同等級
    for i, grade in enumerate(['A', 'B', 'C', 'D', 'F']):
        exp = ExperimentRecord(
            id=f"exp_test_{i:03d}",
            timestamp=datetime(2024, 1, i+1, 12, 0),
            strategy=sample_experiment.strategy.copy(),
            config=sample_experiment.config.copy(),
            results=sample_experiment.results.copy(),
            validation={'grade': grade, 'stages_passed': []},
        )
        repo.insert_experiment(exp)

    # 查詢 A 或 B
    filters = QueryFilters(grade=['A', 'B'])
    results = repo.query_experiments(filters)

    assert len(results) == 2
    for r in results:
        assert r.grade in ['A', 'B']


def test_query_by_tags(repo, sample_experiment):
    """測試按標籤查詢"""
    # 插入不同標籤
    exp1 = sample_experiment
    exp1.id = "exp_001"
    exp1.tags = ['production', 'validated']
    repo.insert_experiment(exp1)

    exp2 = ExperimentRecord(
        id="exp_002",
        timestamp=datetime(2024, 1, 2, 12, 0),
        strategy=sample_experiment.strategy.copy(),
        config=sample_experiment.config.copy(),
        results=sample_experiment.results.copy(),
        validation=sample_experiment.validation.copy(),
        tags=['experimental', 'testing']
    )
    repo.insert_experiment(exp2)

    # 查詢包含 'production' 標籤
    filters = QueryFilters(tags=['production'])
    results = repo.query_experiments(filters)

    assert len(results) == 1
    assert 'production' in results[0].tags


def test_query_by_date_range(repo, sample_experiment):
    """測試按時間範圍查詢"""
    # 插入不同日期
    for i in range(5):
        exp = ExperimentRecord(
            id=f"exp_test_{i:03d}",
            timestamp=datetime(2024, 1, i+1, 12, 0),
            strategy=sample_experiment.strategy.copy(),
            config=sample_experiment.config.copy(),
            results=sample_experiment.results.copy(),
            validation=sample_experiment.validation.copy(),
        )
        repo.insert_experiment(exp)

    # 查詢 2024-01-02 到 2024-01-04
    filters = QueryFilters(
        start_date='2024-01-02',
        end_date='2024-01-04'
    )
    results = repo.query_experiments(filters)

    assert len(results) == 3


def test_query_pagination(repo, sample_experiment):
    """測試分頁查詢"""
    # 插入 10 筆
    for i in range(10):
        exp = ExperimentRecord(
            id=f"exp_test_{i:03d}",
            timestamp=datetime(2024, 1, i+1, 12, 0),
            strategy=sample_experiment.strategy.copy(),
            config=sample_experiment.config.copy(),
            results=sample_experiment.results.copy(),
            validation=sample_experiment.validation.copy(),
        )
        repo.insert_experiment(exp)

    # 第一頁（前 5 筆）
    filters = QueryFilters(limit=5, offset=0)
    page1 = repo.query_experiments(filters)
    assert len(page1) == 5

    # 第二頁（後 5 筆）
    filters = QueryFilters(limit=5, offset=5)
    page2 = repo.query_experiments(filters)
    assert len(page2) == 5

    # 確保不重複
    ids1 = {e.id for e in page1}
    ids2 = {e.id for e in page2}
    assert len(ids1 & ids2) == 0


def test_query_combined_filters(repo, sample_experiment):
    """測試組合過濾條件"""
    # 插入多筆數據
    for i in range(5):
        exp = ExperimentRecord(
            id=f"exp_test_{i:03d}",
            timestamp=datetime(2024, 1, i+1, 12, 0),
            strategy={
                'name': 'ma_cross' if i % 2 == 0 else 'rsi_bb',
                'type': 'trend',
                'version': '1.0'
            },
            config={'symbol': 'BTCUSDT', 'timeframe': '4h'},
            results={
                'sharpe_ratio': 1.0 + i * 0.5,
                'total_return': 0.5,
                'max_drawdown': 0.2
            },
            validation={'grade': 'A' if i >= 2 else 'C', 'stages_passed': []},
        )
        repo.insert_experiment(exp)

    # 組合查詢: ma_cross + Sharpe >= 1.5 + Grade A
    filters = QueryFilters(
        strategy_name='ma_cross',
        min_sharpe=1.5,
        grade=['A']
    )
    results = repo.query_experiments(filters)

    # 驗證
    for r in results:
        assert r.strategy_name == 'ma_cross'
        assert r.sharpe_ratio >= 1.5
        assert r.grade == 'A'


# ========== Test 3: get_best_experiments 排序功能 ==========

def test_get_best_by_sharpe(repo, sample_experiment):
    """測試按 Sharpe Ratio 排序"""
    # 插入 5 筆不同 Sharpe
    sharpes = [0.5, 1.2, 2.1, 1.8, 3.0]
    for i, sharpe in enumerate(sharpes):
        exp = ExperimentRecord(
            id=f"exp_test_{i:03d}",
            timestamp=datetime(2024, 1, i+1, 12, 0),
            strategy=sample_experiment.strategy.copy(),
            config=sample_experiment.config.copy(),
            results={'sharpe_ratio': sharpe, 'total_return': 0.5, 'max_drawdown': 0.2},
            validation=sample_experiment.validation.copy(),
        )
        repo.insert_experiment(exp)

    # 取得最佳 3 個
    best = repo.get_best_experiments(metric='sharpe_ratio', n=3)

    # 驗證排序
    assert len(best) == 3
    assert best[0].sharpe_ratio == 3.0
    assert best[1].sharpe_ratio == 2.1
    assert best[2].sharpe_ratio == 1.8


def test_get_best_by_total_return(repo, sample_experiment):
    """測試按總報酬排序"""
    returns = [0.3, 0.7, 0.5, 1.2, 0.9]
    for i, ret in enumerate(returns):
        exp = ExperimentRecord(
            id=f"exp_test_{i:03d}",
            timestamp=datetime(2024, 1, i+1, 12, 0),
            strategy=sample_experiment.strategy.copy(),
            config=sample_experiment.config.copy(),
            results={'sharpe_ratio': 1.0, 'total_return': ret, 'max_drawdown': 0.2},
            validation=sample_experiment.validation.copy(),
        )
        repo.insert_experiment(exp)

    best = repo.get_best_experiments(metric='total_return', n=2)

    assert len(best) == 2
    assert best[0].total_return == 1.2
    assert best[1].total_return == 0.9


def test_get_best_with_filters(repo, sample_experiment):
    """測試帶過濾條件的排序"""
    # 插入不同策略和 Sharpe
    for i in range(5):
        exp = ExperimentRecord(
            id=f"exp_test_{i:03d}",
            timestamp=datetime(2024, 1, i+1, 12, 0),
            strategy={
                'name': 'ma_cross' if i < 3 else 'rsi_bb',
                'type': 'trend',
                'version': '1.0'
            },
            config=sample_experiment.config.copy(),
            results={'sharpe_ratio': 1.0 + i * 0.5, 'total_return': 0.5, 'max_drawdown': 0.2},
            validation=sample_experiment.validation.copy(),
        )
        repo.insert_experiment(exp)

    # 只查詢 ma_cross 的最佳結果
    filters = QueryFilters(strategy_name='ma_cross')
    best = repo.get_best_experiments(metric='sharpe_ratio', n=2, filters=filters)

    assert len(best) <= 2
    for r in best:
        assert r.strategy_name == 'ma_cross'

    # 驗證排序
    if len(best) == 2:
        assert best[0].sharpe_ratio >= best[1].sharpe_ratio


def test_get_best_handles_null_values(repo, sample_experiment):
    """測試處理 NULL 值（NULLS LAST）"""
    # 插入包含 NULL sharpe 的記錄
    for i in range(3):
        exp = ExperimentRecord(
            id=f"exp_test_{i:03d}",
            timestamp=datetime(2024, 1, i+1, 12, 0),
            strategy=sample_experiment.strategy.copy(),
            config=sample_experiment.config.copy(),
            results={
                'sharpe_ratio': None if i == 0 else 1.0 + i,
                'total_return': 0.5,
                'max_drawdown': 0.2
            },
            validation=sample_experiment.validation.copy(),
        )
        repo.insert_experiment(exp)

    best = repo.get_best_experiments(metric='sharpe_ratio', n=3)

    # NULL 應該排在最後
    assert best[-1].sharpe_ratio is None or best[-1].sharpe_ratio == 0.0


# ========== Test 4: SQL Injection 防護 ==========

def test_sql_injection_invalid_metric(repo):
    """測試 SQL Injection 防護：無效的 metric"""
    # 嘗試 SQL Injection
    with pytest.raises(ValueError, match="Invalid metric"):
        repo.get_best_experiments(
            metric="sharpe_ratio; DROP TABLE experiments; --",
            n=10
        )


def test_sql_injection_valid_metric_only(repo):
    """測試只接受白名單中的 metric"""
    # 驗證白名單
    for metric in VALID_ORDER_COLUMNS:
        try:
            repo.get_best_experiments(metric=metric, n=1)
        except Exception as e:
            # 如果沒有資料是正常的，但不應該是 ValueError
            assert not isinstance(e, ValueError)

    # 非白名單應該拋出錯誤
    invalid_metrics = ['invalid_column', 'id', 'timestamp', 'params']
    for metric in invalid_metrics:
        with pytest.raises(ValueError):
            repo.get_best_experiments(metric=metric, n=1)


# ========== Test 5: JSON 解析錯誤處理 ==========

def test_safe_json_loads_invalid_json(repo):
    """
    測試 JSON 解析錯誤處理

    注意：DuckDB 在插入時就會驗證 JSON 格式，所以無法插入無效 JSON。
    這裡測試的是 _safe_json_loads 如何處理非 JSON 類型的值。
    """
    # DuckDB 會在插入時拋錯，所以我們測試插入失敗的情況
    with pytest.raises(Exception):  # 預期會拋出錯誤
        repo.conn.execute("""
            INSERT INTO experiments (
                id, timestamp, strategy_name, strategy_type, symbol, timeframe,
                params, sharpe_ratio, total_return, max_drawdown, grade, stages_passed
            ) VALUES (
                'exp_invalid_json',
                '2024-01-01 12:00:00',
                'test_strategy',
                'trend',
                'BTCUSDT',
                '4h',
                'INVALID_JSON{]',
                1.5,
                0.5,
                0.2,
                'A',
                '[1, 2, 3]'
            )
        """)


def test_safe_json_loads_null_values(repo):
    """測試 NULL JSON 欄位處理"""
    # 插入 NULL JSON 欄位
    repo.conn.execute("""
        INSERT INTO experiments (
            id, timestamp, strategy_name, strategy_type, symbol, timeframe,
            params, sharpe_ratio, total_return, max_drawdown, grade
        ) VALUES (
            'exp_null_json',
            '2024-01-01 12:00:00',
            'test_strategy',
            'trend',
            'BTCUSDT',
            '4h',
            NULL,
            1.5,
            0.5,
            0.2,
            'A'
        )
    """)

    result = repo.get_experiment('exp_null_json')

    # 驗證預設值
    assert result is not None
    assert result.strategy.get('params') == {}
    assert result.insights == []
    assert result.tags == []


def test_safe_json_loads_empty_string(repo):
    """
    測試空字串 JSON 欄位處理

    注意：DuckDB 不接受空字串作為 JSON，會拋錯。
    """
    # DuckDB 會在插入時拋錯
    with pytest.raises(Exception):
        repo.conn.execute("""
            INSERT INTO experiments (
                id, timestamp, strategy_name, strategy_type, symbol, timeframe,
                params, sharpe_ratio, total_return, max_drawdown, grade, stages_passed
            ) VALUES (
                'exp_empty_json',
                '2024-01-01 12:00:00',
                'test_strategy',
                'trend',
                'BTCUSDT',
                '4h',
                '',
                1.5,
                0.5,
                0.2,
                'A',
                '[]'
            )
        """)


# ========== Test 6: 型別驗證測試 ==========

def test_query_tags_type_validation(repo, sample_experiment):
    """測試 tags 參數的型別驗證"""
    # 插入測試數據
    repo.insert_experiment(sample_experiment)

    # 測試 1: tags 必須是 list，不能是 string
    with pytest.raises(ValueError, match="tags must be a list"):
        filters = QueryFilters(tags="production")
        repo.query_experiments(filters)

    # 測試 2: tags 的元素必須都是 string
    with pytest.raises(ValueError, match="all tags must be strings"):
        filters = QueryFilters(tags=["production", 123, "validated"])
        repo.query_experiments(filters)

    # 測試 3: 正確的 tags 應該可以查詢成功
    filters = QueryFilters(tags=["production"])
    results = repo.query_experiments(filters)
    # 不會拋出錯誤即為成功


def test_query_grade_type_validation(repo, sample_experiment):
    """測試 grade 參數的型別驗證"""
    # 插入測試數據
    repo.insert_experiment(sample_experiment)

    # 測試 1: grade 必須是 list，不能是 string
    with pytest.raises(ValueError, match="grade must be a list"):
        filters = QueryFilters(grade="A")
        repo.query_experiments(filters)

    # 測試 2: grade 的元素必須都是 string
    with pytest.raises(ValueError, match="all grades must be strings"):
        filters = QueryFilters(grade=["A", 123, "B"])
        repo.query_experiments(filters)

    # 測試 3: 正確的 grade 應該可以查詢成功
    filters = QueryFilters(grade=["A", "B"])
    results = repo.query_experiments(filters)
    # 不會拋出錯誤即為成功


def test_query_combined_type_validation(repo, sample_experiment):
    """測試組合查詢時的型別驗證"""
    repo.insert_experiment(sample_experiment)

    # 測試：同時使用錯誤型別的 tags 和 grade
    with pytest.raises(ValueError, match="tags must be a list"):
        filters = QueryFilters(
            strategy_name="ma_cross",
            tags="production",  # 錯誤：應該是 list
            grade=["A"]
        )
        repo.query_experiments(filters)

    with pytest.raises(ValueError, match="grade must be a list"):
        filters = QueryFilters(
            strategy_name="ma_cross",
            tags=["production"],
            grade="A"  # 錯誤：應該是 list
        )
        repo.query_experiments(filters)


def test_get_best_with_invalid_tags(repo, sample_experiment):
    """測試 get_best_experiments 使用無效 tags"""
    repo.insert_experiment(sample_experiment)

    # 測試：在 get_best_experiments 中使用錯誤型別的 tags
    with pytest.raises(ValueError, match="tags must be a list"):
        filters = QueryFilters(tags="production")
        repo.get_best_experiments(
            metric='sharpe_ratio',
            n=10,
            filters=filters
        )


def test_get_best_with_invalid_grade(repo, sample_experiment):
    """測試 get_best_experiments 使用無效 grade"""
    repo.insert_experiment(sample_experiment)

    # 測試：在 get_best_experiments 中使用錯誤型別的 grade
    with pytest.raises(ValueError, match="grade must be a list"):
        filters = QueryFilters(grade="A")
        repo.get_best_experiments(
            metric='sharpe_ratio',
            n=10,
            filters=filters
        )


# ========== 執行測試 ==========

if __name__ == "__main__":
    pytest.main([__file__, '-v', '-s'])
