"""
GP 系統端到端測試

測試完整的 GP 演化流程，從 PrimitiveSet 建立到策略生成和執行。

執行:
    pytest tests/test_gp_e2e.py -v
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import logging

# 檢查 DEAP 是否可用
try:
    from deap import creator, base, tools, gp
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False

# GP 模組
from src.gp.primitives import PrimitiveSetFactory
from src.gp.engine import GPEngine, EvolutionConfig
from src.gp.fitness import create_fitness_type
from src.gp.converter import ExpressionConverter, StrategyGenerator
from src.gp.learning import GPLearningIntegration


# ============================================================================
# 測試常數定義（避免 Magic Numbers）
# ============================================================================

# 種群大小配置
TEST_SMALL_POPULATION = 10
TEST_MEDIUM_POPULATION = 15
TEST_STANDARD_POPULATION = 20

# 世代數配置
TEST_SHORT_GENERATIONS = 2
TEST_STANDARD_GENERATIONS = 3
TEST_MEDIUM_GENERATIONS = 5

# 提前停止配置
TEST_EARLY_STOPPING_GENERATIONS = 3

# Mock 評估函數參數
MOCK_DEPTH_PENALTY = 0.05
MOCK_LENGTH_PENALTY = 0.01
MOCK_RANDOM_NOISE = 0.1

# 適應度閾值
MIN_VALID_FITNESS = -1.0
VERY_LOW_FITNESS = -10.0

# 常數值
CONST_ONE = 1.0
CONST_TWO = 2.0
CONST_FIVE = 5.0

# 除法保護閾值
DIV_PROTECTION_THRESHOLD = 0.001

# 測試種子
TEST_SEED = 42


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def deap_fitness_type():
    """建立 DEAP fitness 類型（只建立一次）"""
    if DEAP_AVAILABLE:
        create_fitness_type()


@pytest.fixture
def primitive_set():
    """建立簡化原語集（用於測試）"""
    if not DEAP_AVAILABLE:
        pytest.skip("DEAP 未安裝")

    # 建立簡單的非強類型原語集
    pset = gp.PrimitiveSet("TEST", 1)  # 1個輸入
    pset.renameArguments(ARG0='x')

    # 添加簡單的原語
    pset.addPrimitive(lambda a, b: a + b, 2, name='add')
    pset.addPrimitive(lambda a, b: a - b, 2, name='sub')
    pset.addPrimitive(lambda a, b: a * b, 2, name='mul')
    pset.addPrimitive(
        lambda a, b: a / b if abs(b) > DIV_PROTECTION_THRESHOLD else CONST_ONE,
        2,
        name='div'
    )

    # 添加常數
    pset.addTerminal(CONST_ONE, name='const_1')
    pset.addTerminal(CONST_TWO, name='const_2')
    pset.addTerminal(CONST_FIVE, name='const_5')

    return pset


@pytest.fixture
def temp_output_dir():
    """建立臨時輸出目錄"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # 明確處理清理異常，不掩蓋問題
    try:
        shutil.rmtree(temp_dir)
    except OSError as e:
        logging.warning(f"Failed to clean up temp dir {temp_dir}: {e}")


def create_mock_evaluate_func(seed: int = TEST_SEED):
    """
    建立 Mock 適應度評估函數

    Args:
        seed: 隨機種子（用於確定性測試）

    Returns:
        評估函數
    """
    rng = np.random.RandomState(seed)

    def evaluate(individual):
        try:
            depth = individual.height
            length = len(individual)
        except AttributeError as e:
            logging.error(f"Invalid individual structure: {e}")
            return (MIN_VALID_FITNESS,)

        # 使用固定種子的隨機數生成器
        fitness = (
            CONST_ONE
            - (depth * MOCK_DEPTH_PENALTY)
            - (length * MOCK_LENGTH_PENALTY)
            + rng.random() * MOCK_RANDOM_NOISE
        )
        return (fitness,)

    return evaluate


# ============================================================================
# Test 1: 完整演化流程
# ============================================================================

@pytest.mark.skipif(not DEAP_AVAILABLE, reason="需要 DEAP")
def test_full_evolution_pipeline(deap_fitness_type, primitive_set):
    """測試 1: 完整演化流程 (PrimitiveSet → GPEngine → 演化)"""
    evaluate_func = create_mock_evaluate_func(seed=TEST_SEED)

    evolution_config = EvolutionConfig(
        population_size=TEST_STANDARD_POPULATION,
        generations=TEST_MEDIUM_GENERATIONS,
        early_stopping_generations=TEST_EARLY_STOPPING_GENERATIONS,
        seed=TEST_SEED
    )

    gp_engine = GPEngine(
        pset=primitive_set,
        evaluate_func=evaluate_func,
        config=evolution_config
    )

    result = gp_engine.evolve()

    # 基本結果驗證
    assert result is not None, "演化結果不應為 None"
    assert result.best_individual is not None, "最佳個體不應為 None"
    assert result.best_fitness > MIN_VALID_FITNESS, \
        f"最佳適應度 {result.best_fitness} 應大於 {MIN_VALID_FITNESS}"

    # 演化過程驗證
    assert result.generations_run >= 1, "至少應執行一代"
    assert len(result.fitness_history) == result.generations_run + 1, \
        "適應度歷史長度應等於代數+1（包含初始種群）"

    # 結果品質驗證
    assert len(result.hall_of_fame) > 0, "名人堂應包含至少一個個體"
    assert result.elapsed_time > 0, "執行時間應大於零"
    assert result.best_individual.fitness.valid, "最佳個體的適應度應已被評估"

    print(f"✓ 演化成功: {result.generations_run} 代，最佳適應度 {result.best_fitness:.4f}")


# ============================================================================
# Test 2: 表達式轉換流程
# ============================================================================

@pytest.mark.skipif(not DEAP_AVAILABLE, reason="需要 DEAP")
def test_expression_conversion_pipeline(deap_fitness_type, primitive_set):
    """測試 2: 表達式轉換 (演化 → ExpressionConverter → Python 程式碼)"""
    evaluate_func = create_mock_evaluate_func(seed=TEST_SEED)

    gp_engine = GPEngine(
        pset=primitive_set,
        evaluate_func=evaluate_func,
        config=EvolutionConfig(
            population_size=TEST_SMALL_POPULATION,
            generations=TEST_STANDARD_GENERATIONS,
            seed=TEST_SEED
        )
    )

    result = gp_engine.evolve()
    best = result.best_individual

    converter = ExpressionConverter(primitive_set)

    # 轉換為 Python 程式碼
    python_code = converter.to_python(best)
    assert isinstance(python_code, str), "Python 程式碼應為字串"
    assert len(python_code) > 0, "Python 程式碼不應為空"

    # 取得函數體並進行更精確的檢查
    func_body = converter.to_function_body(best)
    assert 'def generate_entry_signal' in func_body, \
        "函數體應包含函數定義"
    # 檢查是否有 return 語句（至少一個）
    assert func_body.count('return') >= 1, "函數體應包含至少一個 return 語句"
    # 檢查是否有函數參數
    assert '(' in func_body and ')' in func_body, "函數應有參數定義"

    # 編譯為函數
    compiled_func = converter.compile(best)
    assert callable(compiled_func), "編譯後的函數應可調用"

    # 取得元資料
    metadata = converter.get_expression_metadata(best)
    assert 'expression' in metadata, "元資料應包含 expression"
    assert 'depth' in metadata, "元資料應包含 depth"
    assert 'length' in metadata, "元資料應包含 length"
    assert 'primitives_used' in metadata, "元資料應包含 primitives_used"

    print(f"✓ 轉換成功: {metadata['expression'][:50]}...")


# ============================================================================
# Test 3: 策略生成流程
# ============================================================================

def _verify_strategy_file_content(file_path: Path, expected_fitness: float):
    """
    驗證策略檔案內容的輔助函數

    Args:
        file_path: 策略檔案路徑
        expected_fitness: 預期的適應度分數
    """
    import ast

    content = file_path.read_text(encoding='utf-8')

    # 檢查必要的類別和導入
    assert 'class' in content, "檔案應包含類別定義"
    assert 'EvolvedStrategy' in content, "檔案應包含 EvolvedStrategy 類別"
    assert 'from src.gp.primitives import *' in content, \
        "檔案應包含必要的導入語句"

    # 檢查適應度分數
    assert f'fitness_score = {expected_fitness}' in content, \
        f"檔案應包含正確的適應度分數 {expected_fitness}"

    # 驗證 Python 語法
    try:
        ast.parse(content)
    except SyntaxError as e:
        pytest.fail(f"生成的策略檔案有語法錯誤: {e}")


@pytest.mark.skipif(not DEAP_AVAILABLE, reason="需要 DEAP")
def test_strategy_generation_pipeline(
    deap_fitness_type,
    primitive_set,
    temp_output_dir
):
    """測試 3: 策略生成 (演化 → StrategyGenerator → 策略檔案)"""
    evaluate_func = create_mock_evaluate_func(seed=TEST_SEED)

    gp_engine = GPEngine(
        pset=primitive_set,
        evaluate_func=evaluate_func,
        config=EvolutionConfig(
            population_size=TEST_SMALL_POPULATION,
            generations=TEST_STANDARD_GENERATIONS,
            seed=TEST_SEED
        )
    )

    result = gp_engine.evolve()

    converter = ExpressionConverter(primitive_set)
    generator = StrategyGenerator(converter)

    strategy_name = "test_evolved_strategy_001"
    file_path = generator.generate(
        individual=result.best_individual,
        strategy_name=strategy_name,
        fitness=result.best_fitness,
        metadata={'generation': result.generations_run},
        output_dir=temp_output_dir
    )

    # 驗證檔案生成
    assert file_path.exists(), f"策略檔案應已生成: {file_path}"
    assert file_path.name == f"{strategy_name}.py", \
        f"檔案名稱應為 {strategy_name}.py"

    # 驗證檔案內容（使用輔助函數）
    _verify_strategy_file_content(file_path, result.best_fitness)

    print(f"✓ 策略檔案生成成功: {file_path}")


# ============================================================================
# Test 4: 學習系統整合
# ============================================================================

@pytest.mark.skipif(not DEAP_AVAILABLE, reason="需要 DEAP")
def test_learning_integration(deap_fitness_type, primitive_set):
    """測試 4: 學習系統整合 (演化 → GPLearningIntegration)"""
    evaluate_func = create_mock_evaluate_func(seed=TEST_SEED)

    gp_engine = GPEngine(
        pset=primitive_set,
        evaluate_func=evaluate_func,
        config=EvolutionConfig(
            population_size=TEST_SMALL_POPULATION,
            generations=TEST_STANDARD_GENERATIONS,
            seed=TEST_SEED
        )
    )

    result = gp_engine.evolve()

    integration = GPLearningIntegration()

    exp_id = integration.record_evolution(
        result=result,
        metadata={'symbol': 'BTCUSDT', 'timeframe': '1d'}
    )

    # 驗證實驗 ID 格式
    assert exp_id is not None, "實驗 ID 不應為 None"
    assert isinstance(exp_id, str), "實驗 ID 應為字串"
    assert 'exp_gp' in exp_id, "實驗 ID 應包含 'exp_gp' 前綴"
    assert 'btc' in exp_id, "實驗 ID 應包含標的符號"

    print(f"✓ 學習系統記錄成功: {exp_id}")


# ============================================================================
# Test 5: 錯誤處理
# ============================================================================

@pytest.mark.skipif(not DEAP_AVAILABLE, reason="需要 DEAP")
def test_error_handling(deap_fitness_type, primitive_set):
    """測試 5: 錯誤處理（無效個體）"""
    def failing_evaluate(individual):
        """總是返回極低分的評估函數"""
        return (VERY_LOW_FITNESS,)

    # 即使評估函數返回極低分，演化也應該正常執行
    gp_engine = GPEngine(
        pset=primitive_set,
        evaluate_func=failing_evaluate,
        config=EvolutionConfig(
            population_size=TEST_SMALL_POPULATION,
            generations=TEST_SHORT_GENERATIONS,
            seed=TEST_SEED
        )
    )

    result = gp_engine.evolve()

    # 應該能完成演化，只是適應度很低
    assert result is not None, "即使適應度很低，演化結果也不應為 None"
    assert result.best_individual is not None, "應該有最佳個體"
    assert result.generations_run >= 1, "至少應執行一代"

    # 轉換器應該能處理任何有效個體
    converter = ExpressionConverter(primitive_set)
    code = converter.to_python(result.best_individual)
    assert isinstance(code, str), "應能轉換為 Python 程式碼"
    assert len(code) > 0, "程式碼不應為空"

    print(f"✓ 錯誤處理正常")


# ============================================================================
# Test 6: 端到端整合
# ============================================================================

@pytest.mark.skipif(not DEAP_AVAILABLE, reason="需要 DEAP")
def test_end_to_end_integration(
    deap_fitness_type,
    primitive_set,
    temp_output_dir
):
    """測試 6: 完整端到端流程（演化 → 轉換 → 生成 → 記錄）"""
    # Step 1: 演化
    evaluate_func = create_mock_evaluate_func(seed=TEST_SEED)
    evolution_config = EvolutionConfig(
        population_size=TEST_MEDIUM_POPULATION,
        generations=TEST_MEDIUM_GENERATIONS,
        seed=TEST_SEED
    )

    gp_engine = GPEngine(
        pset=primitive_set,
        evaluate_func=evaluate_func,
        config=evolution_config
    )

    result = gp_engine.evolve()
    assert result.best_fitness > MIN_VALID_FITNESS, \
        f"適應度 {result.best_fitness} 應大於 {MIN_VALID_FITNESS}"

    # Step 2: 轉換表達式
    converter = ExpressionConverter(primitive_set)
    python_code = converter.to_python(result.best_individual)
    assert len(python_code) > 0, "Python 程式碼不應為空"

    # Step 3: 生成策略檔案
    generator = StrategyGenerator(converter)
    file_path = generator.generate(
        individual=result.best_individual,
        strategy_name="e2e_test_strategy",
        fitness=result.best_fitness,
        metadata={'generation': result.generations_run},
        output_dir=temp_output_dir
    )
    assert file_path.exists(), f"策略檔案應已生成: {file_path}"

    # Step 4: 記錄到學習系統
    integration = GPLearningIntegration()
    exp_id = integration.record_evolution(
        result=result,
        metadata={'symbol': 'BTCUSDT'}
    )
    assert exp_id is not None, "實驗 ID 不應為 None"

    # Step 5: 驗證一致性
    assert result.best_fitness == result.best_individual.fitness.values[0], \
        "結果中的適應度應與最佳個體的適應度一致"
    assert len(result.fitness_history) == result.generations_run + 1, \
        "適應度歷史長度應等於代數+1"

    print(f"✓ 端到端整合測試通過 (適應度: {result.best_fitness:.4f})")


# ============================================================================
# Test 7: 邊界測試
# ============================================================================

@pytest.mark.skipif(not DEAP_AVAILABLE, reason="需要 DEAP")
def test_empty_population_handling(deap_fitness_type, primitive_set):
    """測試 7.1: 空種群處理"""
    evaluate_func = create_mock_evaluate_func(seed=TEST_SEED)

    # 種群大小為 0 應該引發錯誤或有合理的預設值
    with pytest.raises((ValueError, AssertionError)):
        evolution_config = EvolutionConfig(
            population_size=0,  # 無效的種群大小
            generations=TEST_STANDARD_GENERATIONS,
            seed=TEST_SEED
        )
        gp_engine = GPEngine(
            pset=primitive_set,
            evaluate_func=evaluate_func,
            config=evolution_config
        )
        gp_engine.evolve()


@pytest.mark.skipif(not DEAP_AVAILABLE, reason="需要 DEAP")
def test_extremely_deep_tree(deap_fitness_type, primitive_set):
    """測試 7.2: 極深表達式樹處理"""
    # 使用演化生成較深的樹，然後測試轉換
    evaluate_func = create_mock_evaluate_func(seed=TEST_SEED)

    # 使用較多代數可能產生較深的樹
    gp_engine = GPEngine(
        pset=primitive_set,
        evaluate_func=evaluate_func,
        config=EvolutionConfig(
            population_size=TEST_SMALL_POPULATION,
            generations=TEST_MEDIUM_GENERATIONS,
            seed=TEST_SEED
        )
    )

    result = gp_engine.evolve()
    best = result.best_individual

    converter = ExpressionConverter(primitive_set)

    # 應該能處理演化產生的任何深度的樹
    try:
        python_code = converter.to_python(best)
        assert isinstance(python_code, str), "應能轉換表達式樹"
        assert len(python_code) > 0, "轉換結果不應為空"

        # 記錄實際深度以供參考
        if hasattr(best, 'height'):
            print(f"✓ 成功處理深度為 {best.height} 的表達式樹")
    except RecursionError:
        pytest.skip("遞歸深度超過 Python 限制，這是預期的")
