"""
GP 表達式轉換器

將 DEAP GP 表達式樹轉換為可執行的 Python 程式碼。
"""

from typing import Callable, Dict, Optional, Any
from pathlib import Path
from datetime import datetime
import re
import logging

try:
    from deap import gp
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    gp = None


class ExpressionConverter:
    """
    將 GP 個體轉換為 Python 程式碼

    使用範例:
        converter = ExpressionConverter(pset)
        code = converter.to_python(individual)
        func = converter.compile(individual)
    """

    def __init__(self, pset):
        """
        初始化轉換器

        Args:
            pset: DEAP PrimitiveSetTyped 實例
        """
        if not DEAP_AVAILABLE:
            raise ImportError(
                "DEAP is required for ExpressionConverter. "
                "Install it with: pip install deap"
            )

        self.pset = pset

        # 建立引數映射（ARG0 -> close, ARG1 -> high, etc.）
        self._arg_mapping = {}
        for i, arg_name in enumerate(pset.arguments):
            self._arg_mapping[f'ARG{i}'] = arg_name

        # 建立原語名稱映射（DEAP 名稱 -> Python 函數名稱）
        # DEAP 使用大寫名稱（RSI, MA），但實際函數是小寫（rsi, ma）
        self._primitive_mapping = self._build_primitive_mapping()

    def _build_primitive_mapping(self) -> Dict[str, str]:
        """
        建立原語名稱映射

        從 pset 中提取所有原語的名稱，並轉為小寫。

        Returns:
            dict: {DEAP名稱: Python函數名稱}
        """
        mapping = {}

        # 遍歷所有原語
        for prim in self.pset.primitives.values():
            for p in prim:
                # DEAP 名稱（如 "RSI"）
                deap_name = p.name
                # Python 函數名稱（如 "rsi"）
                python_name = deap_name.lower()
                mapping[deap_name] = python_name

        # 添加終端名稱映射（period_14 等保持不變）
        for term_list in self.pset.terminals.values():
            for term in term_list:
                if hasattr(term, 'name'):
                    mapping[term.name] = term.name

        return mapping

    def _sanitize_expression(self, expr_str: str) -> str:
        """
        清理並驗證表達式字串

        確保只包含允許的函數和運算符，防止程式碼注入攻擊。

        Args:
            expr_str: 表達式字串

        Returns:
            str: 驗證通過的表達式

        Raises:
            ValueError: 如果表達式包含危險關鍵字或語法錯誤
        """
        import ast

        # 1. 檢查是否包含危險關鍵字
        dangerous_keywords = [
            '__import__', 'eval', 'exec', 'compile',
            'globals', 'locals', 'open', 'os', 'sys',
            '__builtins__', '__file__', '__name__'
        ]
        expr_lower = expr_str.lower()
        for keyword in dangerous_keywords:
            # 使用正則表達式的單詞邊界（\b）確保精確匹配
            # 例如：\bos\b 只匹配單獨的 "os"，不會匹配 "close"
            pattern = rf'\b{re.escape(keyword)}\b'
            if re.search(pattern, expr_lower):
                raise ValueError(f"Dangerous keyword detected: {keyword}")

        # 2. 使用 ast.parse 驗證語法
        try:
            ast.parse(expr_str)
        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {expr_str}") from e

        return expr_str

    def to_python(self, individual, replace_terminals: bool = False) -> str:
        """
        將表達式樹轉為 Python 程式碼字串

        Args:
            individual: DEAP GP 個體
            replace_terminals: 是否將終端名稱替換為實際值（預設 False）

        Returns:
            str: Python 程式碼字串

        Example:
            輸入: AND(GT(RSI(ARG0, 14), 70), LT(MA(ARG0, 20), ARG0))
            輸出: "and_(gt(rsi(close, 14), 70), lt(ma(close, 20), close))"
        """
        # 將表達式轉為字串
        expr_str = str(individual)

        # 替換引數名稱（ARG0 -> close, ARG1 -> high, ARG2 -> low）
        for arg_code, arg_name in self._arg_mapping.items():
            expr_str = expr_str.replace(arg_code, arg_name)

        # 替換原語名稱（RSI -> rsi, MA -> ma）
        for deap_name, python_name in self._primitive_mapping.items():
            # 使用正則替換，確保只替換完整單詞（避免 MACD 被替換為 macd_d）
            expr_str = re.sub(
                rf'\b{deap_name}\b',
                python_name,
                expr_str
            )

        # 替換終端名稱為實際值（如果需要）
        if replace_terminals:
            expr_str = self._replace_terminal_names(expr_str)

        return expr_str

    def _replace_terminal_names(self, expr_str: str) -> str:
        """
        將終端名稱替換為實際值

        Args:
            expr_str: 表達式字串

        Returns:
            str: 替換後的表達式

        Example:
            輸入: "rsi(close, period_14)"
            輸出: "rsi(close, 14)"
        """
        # 從 pset.context 中獲取終端的實際值
        for term_name, term_value in self.pset.context.items():
            # 跳過內建項目和原語（只處理常數終端）
            if term_name.startswith('__') or callable(term_value):
                continue

            # 替換終端名稱為實際值
            # 使用單詞邊界確保只替換完整單詞
            expr_str = re.sub(
                rf'\b{term_name}\b',
                str(term_value),
                expr_str
            )

        return expr_str

    def to_function_body(self, individual) -> str:
        """
        生成完整的函數體程式碼

        Args:
            individual: DEAP GP 個體

        Returns:
            str: 函數體程式碼

        Example:
            def generate_entry_signal(close, high, low):
                return and_(gt(rsi(close, 14), 70), lt(ma(close, 20), close))
        """
        # 取得參數名稱列表
        param_names = ', '.join(self.pset.arguments)

        # 取得表達式程式碼
        expr_code = self.to_python(individual)

        # 組裝函數
        func_code = f"""def generate_entry_signal({param_names}):
    return {expr_code}"""

        return func_code

    def compile(self, individual) -> Callable:
        """
        編譯表達式樹為可呼叫函數

        Args:
            individual: DEAP GP 個體

        Returns:
            Callable: 可呼叫的訊號函數

        Raises:
            RuntimeError: 如果 DEAP 未安裝
        """
        if not DEAP_AVAILABLE or gp is None:
            raise RuntimeError("DEAP is required but not available")
        return gp.compile(individual, self.pset)

    def get_expression_metadata(self, individual) -> Dict[str, Any]:
        """
        取得表達式元資料

        Args:
            individual: DEAP GP 個體

        Returns:
            dict: 元資料
                - expression: 表達式字串
                - depth: 深度
                - length: 節點數量
                - primitives_used: 使用的原語列表
        """
        expr_str = self.to_python(individual)

        # 計算深度和長度
        from deap.gp import PrimitiveTree
        if isinstance(individual, PrimitiveTree):
            depth = individual.height
            length = len(individual)
        else:
            depth = 0
            length = 0

        # 提取使用的原語
        primitives_used = self._extract_primitives(expr_str)

        return {
            'expression': expr_str,
            'depth': depth,
            'length': length,
            'primitives_used': primitives_used
        }

    def _extract_primitives(self, expr_str: str) -> list:
        """
        從表達式字串中提取使用的原語

        Args:
            expr_str: 表達式字串

        Returns:
            list: 原語名稱列表
        """
        # 使用正則表達式提取函數名稱
        pattern = r'\b([a-z_]+)\('
        primitives = re.findall(pattern, expr_str)

        # 去重並排序
        return sorted(set(primitives))


class StrategyGenerator:
    """
    生成完整的策略 Python 檔案

    使用範例:
        generator = StrategyGenerator(converter)
        file_path = generator.generate(
            individual=best,
            strategy_name="evolved_rsi_ma_001",
            fitness=1.85
        )
    """

    def __init__(self, converter: ExpressionConverter):
        """
        初始化策略生成器

        Args:
            converter: ExpressionConverter 實例
        """
        self.converter = converter

    def generate(
        self,
        individual,
        strategy_name: str,
        fitness: float,
        metadata: Optional[Dict] = None,
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        生成策略檔案

        Args:
            individual: GP 個體
            strategy_name: 策略名稱
            fitness: 適應度分數
            metadata: 額外元資料（可選）
                - generation: 代數
                - population_size: 族群大小
                - mutation_rate: 突變率
                - crossover_rate: 交叉率
            output_dir: 輸出目錄（預設為 src/strategies/gp/generated）

        Returns:
            Path: 生成的檔案路徑
        """
        # 預設輸出目錄
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'strategies' / 'gp' / 'generated'

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 取得表達式（替換終端為實際值）
        expression = self.converter.to_python(individual, replace_terminals=True)

        # 驗證表達式安全性
        expression = self.converter._sanitize_expression(expression)

        # 合併元資料
        if metadata is None:
            metadata = {}

        generation = metadata.get('generation', 0)

        # 生成類別名稱（Evolved001, Evolved002, ...）
        class_name = self._to_class_name(strategy_name)

        # 取得當前時間
        generated_at = datetime.utcnow().isoformat() + 'Z'

        # 生成策略程式碼
        strategy_code = self._generate_strategy_code(
            strategy_name=strategy_name,
            class_name=class_name,
            expression=expression,
            fitness=fitness,
            generation=generation,
            generated_at=generated_at,
            metadata=metadata
        )

        # 寫入檔案
        file_path = output_dir / f"{strategy_name}.py"

        try:
            if file_path.exists():
                logger = logging.getLogger(__name__)
                logger.warning(f"策略檔案已存在，將覆蓋: {file_path}")

            file_path.write_text(strategy_code, encoding='utf-8')
        except OSError as e:
            raise RuntimeError(f"無法寫入策略檔案 {file_path}: {e}") from e

        return file_path

    def _to_class_name(self, strategy_name: str) -> str:
        """
        將策略名稱轉為類別名稱

        Args:
            strategy_name: 策略名稱（如 "evolved_rsi_ma_001"）

        Returns:
            str: 類別名稱（如 "EvolvedRsiMa001"）
        """
        # 分割並首字母大寫
        parts = strategy_name.split('_')
        class_name = ''.join(word.capitalize() for word in parts)
        return class_name

    def _generate_strategy_code(
        self,
        strategy_name: str,
        class_name: str,
        expression: str,
        fitness: float,
        generation: int,
        generated_at: str,
        metadata: Dict
    ) -> str:
        """
        生成策略程式碼

        Args:
            strategy_name: 策略名稱
            class_name: 類別名稱
            expression: GP 表達式字串
            fitness: 適應度分數
            generation: 演化代數
            generated_at: 生成時間（ISO 格式）
            metadata: 額外元資料

        Returns:
            str: 完整的策略程式碼
        """
        # 轉義表達式中的引號
        expression_escaped = expression.replace('"', r'\"')

        # 生成策略程式碼
        code = f'''"""
GP 演化策略: {strategy_name}

自動生成於: {generated_at}
適應度: {fitness:.4f}
表達式: {expression}
"""

from src.strategies.gp.evolved_strategy import EvolvedStrategy
from src.gp.primitives import *
import numpy as np
import pandas as pd


class {class_name}(EvolvedStrategy):
    """
    GP 演化策略

    表達式: {expression}
    適應度: {fitness:.4f}
    代數: {generation}
    """

    name = "{strategy_name}"
    version = "1.0"
    description = "GP evolved strategy with fitness {fitness:.4f}"

    # 演化元資料
    expression = "{expression_escaped}"
    fitness_score = {fitness}
    generation = {generation}
    evolved_at = "{generated_at}"

    def __init__(self, **kwargs):
        """初始化演化策略"""
        super().__init__(**kwargs)
        # 編譯訊號函數
        self._signal_func = self._build_signal_func()

    def _build_signal_func(self):
        """建立訊號函數"""
        def signal_func(close, high, low):
            """
            GP 演化的訊號函數

            Args:
                close: 收盤價序列
                high: 最高價序列
                low: 最低價序列

            Returns:
                布林陣列：True 表示進場訊號
            """
            # GP 表達式
            return {expression}

        return signal_func
'''

        return code


# 公開 API
__all__ = [
    'ExpressionConverter',
    'StrategyGenerator',
]
