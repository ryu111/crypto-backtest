"""
測試 GP Converter

測試 ExpressionConverter 和 StrategyGenerator 的功能。
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from deap import creator, base, gp

from src.gp.primitives import PrimitiveSetFactory
from src.gp.converter import ExpressionConverter, StrategyGenerator


@pytest.fixture
def pset():
    """建立 PrimitiveSet"""
    factory = PrimitiveSetFactory()
    return factory.create_standard_set()


@pytest.fixture
def individual(pset):
    """
    建立測試個體

    使用 DEAP 的 compile 逆向工程建立簡單個體。
    """
    # 建立 DEAP 類型
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    # 建立工具箱
    toolbox = base.Toolbox()

    # 使用 genGrow 生成表達式（允許深度為 0-2）
    toolbox.register("expr", gp.genGrow, pset=pset, min_=0, max_=2)
    toolbox.register("individual", gp.tools.initIterate, creator.Individual, toolbox.expr)

    # 生成個體（多次嘗試）
    for _ in range(100):
        try:
            ind = toolbox.individual()
            # 檢查是否可以字串化（避免空個體）
            _ = str(ind)
            return ind
        except (IndexError, AttributeError):
            continue

    # 如果還是失敗，返回一個非常簡單的個體（只有終端）
    # 這是一個 fallback，正常情況應該能在上面生成成功
    raise RuntimeError("無法生成有效個體（PrimitiveSet 可能缺少必要的終端）")


def test_expression_converter_to_python(pset, individual):
    """測試表達式轉 Python 程式碼"""
    converter = ExpressionConverter(pset)

    python_code = converter.to_python(individual)

    # 驗證
    assert isinstance(python_code, str)
    assert len(python_code) > 0
    # 不應該包含 ARG0, ARG1 等（應該被替換為 close, high, low）
    assert 'ARG0' not in python_code
    assert 'ARG1' not in python_code
    assert 'ARG2' not in python_code
    # 應該包含參數名稱
    assert 'close' in python_code or 'high' in python_code or 'low' in python_code

    print(f"Python code: {python_code}")


def test_expression_converter_to_function_body(pset, individual):
    """測試生成函數體"""
    converter = ExpressionConverter(pset)

    func_body = converter.to_function_body(individual)

    # 驗證
    assert isinstance(func_body, str)
    assert 'def generate_entry_signal' in func_body
    assert 'return' in func_body

    print(f"Function body:\n{func_body}")


def test_expression_converter_compile(pset, individual):
    """測試編譯"""
    converter = ExpressionConverter(pset)

    compiled_func = converter.compile(individual)

    # 驗證：編譯後的函數可以呼叫
    assert callable(compiled_func)

    # 建立測試資料
    close = np.array([100, 102, 101, 103, 105])
    high = close + 1
    low = close - 1

    # 執行函數（不應該出錯）
    try:
        result = compiled_func(close, high, low)
        assert result is not None
        print(f"Compiled result: {result}")
    except Exception as e:
        pytest.fail(f"Compiled function failed: {e}")


def test_expression_converter_metadata(pset, individual):
    """測試元資料提取"""
    converter = ExpressionConverter(pset)

    metadata = converter.get_expression_metadata(individual)

    # 驗證
    assert 'expression' in metadata
    assert 'depth' in metadata
    assert 'length' in metadata
    assert 'primitives_used' in metadata

    assert isinstance(metadata['expression'], str)
    assert isinstance(metadata['depth'], int)
    assert isinstance(metadata['length'], int)
    assert isinstance(metadata['primitives_used'], list)

    print(f"Metadata: {metadata}")


def test_strategy_generator(pset, individual, tmp_path):
    """測試策略生成器"""
    converter = ExpressionConverter(pset)
    generator = StrategyGenerator(converter)

    # 生成策略
    file_path = generator.generate(
        individual=individual,
        strategy_name="test_evolved_001",
        fitness=1.85,
        metadata={'generation': 10},
        output_dir=tmp_path
    )

    # 驗證
    assert file_path.exists()
    assert file_path.name == "test_evolved_001.py"

    # 讀取並驗證內容
    content = file_path.read_text()
    assert "class TestEvolved001(EvolvedStrategy)" in content
    assert "fitness_score = 1.85" in content
    assert "generation = 10" in content

    print(f"Generated file: {file_path}")
    print(f"Content preview:\n{content[:500]}")


def test_generated_strategy_can_be_imported(pset, individual, tmp_path):
    """測試生成的策略可以被 import"""
    import sys

    converter = ExpressionConverter(pset)
    generator = StrategyGenerator(converter)

    # 生成策略
    file_path = generator.generate(
        individual=individual,
        strategy_name="test_evolved_import",
        fitness=2.0,
        output_dir=tmp_path
    )

    # 將 tmp_path 加入 sys.path
    sys.path.insert(0, str(tmp_path))

    try:
        # 動態 import
        module = __import__('test_evolved_import')

        # 驗證類別存在
        assert hasattr(module, 'TestEvolvedImport')

        # 實例化策略
        strategy_class = getattr(module, 'TestEvolvedImport')
        strategy = strategy_class()

        # 驗證屬性
        assert strategy.name == "test_evolved_import"
        assert strategy.fitness_score == 2.0
        assert strategy.strategy_type == "evolved"

        # 測試 generate_signals（需要真實資料）
        data = pd.DataFrame({
            'close': np.random.rand(100) * 100,
            'high': np.random.rand(100) * 100 + 1,
            'low': np.random.rand(100) * 100 - 1,
        })

        long_entry, long_exit, short_entry, short_exit = strategy.generate_signals(data)

        # 驗證訊號
        assert isinstance(long_entry, pd.Series)
        assert isinstance(long_exit, pd.Series)
        assert len(long_entry) == len(data)

        print(f"Strategy {strategy.name} successfully instantiated and tested!")

    finally:
        sys.path.remove(str(tmp_path))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
