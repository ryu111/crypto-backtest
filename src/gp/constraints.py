"""
GP 約束函數

控制表達式樹的結構，避免過度複雜或無效的個體。

使用範例:
    from src.gp.constraints import (
        ConstraintConfig,
        apply_constraints,
        calculate_complexity_penalty
    )

    # 應用約束到 DEAP toolbox
    config = ConstraintConfig(max_depth=15, max_size=80)
    apply_constraints(toolbox, config)

    # 計算複雜度懲罰
    penalty = calculate_complexity_penalty(individual, alpha=0.01)
"""

from typing import Callable, Optional, List, Any
from dataclasses import dataclass
import operator

try:
    from deap import gp, tools
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    gp = tools = None


# ============================================================================
# 約束配置
# ============================================================================

@dataclass
class ConstraintConfig:
    """GP 約束配置

    定義表達式樹的結構限制。

    Attributes:
        max_depth: 最大樹深度（預設 17，DEAP 預設值）
        max_size: 最大節點數量（預設 100）
        min_depth: 最小樹深度（預設 2，避免過於簡單）
        max_constants: 最多常數數量（預設 10）
    """

    max_depth: int = 17  # DEAP 預設值
    max_size: int = 100
    min_depth: int = 2
    max_constants: int = 10

    def __post_init__(self):
        """驗證配置"""
        if self.max_depth < self.min_depth:
            raise ValueError(
                f"max_depth ({self.max_depth}) 不能小於 min_depth ({self.min_depth})"
            )

        if self.max_size < 1:
            raise ValueError("max_size 必須大於 0")


# ============================================================================
# 深度限制
# ============================================================================

def limit_depth(max_depth: int = 17) -> Callable:
    """
    建立深度限制裝飾器

    DEAP GP 預設最大深度為 17，這通常足夠。
    過深的樹會導致：
        - 過擬合
        - 計算緩慢
        - 難以解釋

    Args:
        max_depth: 最大樹深度

    Returns:
        Callable: 可用於 toolbox.decorate 的裝飾器

    使用範例:
        decorator = limit_depth(15)
        toolbox.decorate("mate", decorator)
        toolbox.decorate("mutate", decorator)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)

            # 記錄原始型別（在轉換為列表之前）
            is_list = isinstance(offspring, list)

            # offspring 可能是單一個體或列表，統一轉為列表處理
            if not is_list:
                offspring = [offspring]

            # 檢查每個後代的深度
            valid_offspring = []
            for child in offspring:
                if hasattr(child, 'height') and child.height > max_depth:
                    # 深度超過限制，不加入
                    continue
                valid_offspring.append(child)

            # 返回有效後代
            # 單一個體：返回有效個體或原個體（避免返回 None）
            # 列表：返回有效列表（可能是空列表，DEAP 會處理）
            if is_list:
                return valid_offspring
            else:
                return valid_offspring[0] if valid_offspring else offspring[0]

        return wrapper
    return decorator


# ============================================================================
# 節點數量限制（膨脹控制）
# ============================================================================

def limit_size(max_size: int = 100) -> Callable:
    """
    建立節點數量限制裝飾器

    防止「程式碼膨脹」(Bloat) - GP 常見問題。
    未限制的 GP 會產生越來越大的樹，而不增加準確度。

    Args:
        max_size: 最大節點數量

    Returns:
        Callable: 可用於 toolbox.decorate 的裝飾器

    使用範例:
        decorator = limit_size(80)
        toolbox.decorate("mate", decorator)
        toolbox.decorate("mutate", decorator)

    Note:
        Bloat 現象：
        - 樹大小隨代數增長
        - 不帶來準確度提升
        - 計算成本增加
        - 解釋性下降
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)

            # 記錄原始型別（在轉換為列表之前）
            is_list = isinstance(offspring, list)

            # offspring 可能是單一個體或列表，統一轉為列表處理
            if not is_list:
                offspring = [offspring]

            # 檢查每個後代的大小
            valid_offspring = []
            for child in offspring:
                if len(child) > max_size:
                    # 節點數超過限制，不加入
                    continue
                valid_offspring.append(child)

            # 返回有效後代
            # 單一個體：返回有效個體或原個體（避免返回 None）
            # 列表：返回有效列表（可能是空列表，DEAP 會處理）
            if is_list:
                return valid_offspring
            else:
                return valid_offspring[0] if valid_offspring else offspring[0]

        return wrapper
    return decorator


# ============================================================================
# 應用所有約束
# ============================================================================

def apply_constraints(
    toolbox: 'tools.Toolbox',
    config: Optional[ConstraintConfig] = None
):
    """
    應用所有約束到 DEAP toolbox

    使用 DEAP 的 staticLimit 機制限制交叉和突變後代。

    Args:
        toolbox: DEAP toolbox（需已註冊 mate、mutate）
        config: 約束配置（使用預設值如果未提供）

    使用範例:
        toolbox = base.Toolbox()
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        # 應用約束
        apply_constraints(toolbox, ConstraintConfig(max_depth=15))

    Note:
        必須在註冊 mate/mutate 之後呼叫。
    """
    if not DEAP_AVAILABLE:
        raise ImportError("需要安裝 DEAP: pip install deap")

    cfg = config or ConstraintConfig()

    # 使用 DEAP 內建的 staticLimit
    # 這比自訂裝飾器更高效，因為它在 C 層級實作

    # 限制交叉後代的深度
    if hasattr(toolbox, "mate"):
        toolbox.decorate(
            "mate",
            gp.staticLimit(
                key=operator.attrgetter("height"),
                max_value=cfg.max_depth
            )
        )

    # 限制突變後代的深度
    if hasattr(toolbox, "mutate"):
        toolbox.decorate(
            "mutate",
            gp.staticLimit(
                key=operator.attrgetter("height"),
                max_value=cfg.max_depth
            )
        )

    # 限制交叉後代的節點數量
    if hasattr(toolbox, "mate"):
        toolbox.decorate(
            "mate",
            gp.staticLimit(
                key=len,
                max_value=cfg.max_size
            )
        )

    # 限制突變後代的節點數量
    if hasattr(toolbox, "mutate"):
        toolbox.decorate(
            "mutate",
            gp.staticLimit(
                key=len,
                max_value=cfg.max_size
            )
        )


# ============================================================================
# 複雜度懲罰計算
# ============================================================================

def calculate_complexity_penalty(
    individual: 'gp.PrimitiveTree',
    alpha: float = 0.01,
    depth_weight: float = 0.5
) -> float:
    """
    計算複雜度懲罰

    複雜度懲罰用於適應度函數，鼓勵較簡單的解。

    公式:
        penalty = alpha * (size + depth * depth_weight)

    Args:
        individual: GP 個體
        alpha: 懲罰係數（預設 0.01）
        depth_weight: 深度權重（預設 0.5）

    Returns:
        float: 懲罰值（應從適應度中減去）

    使用範例:
        penalty = calculate_complexity_penalty(individual, alpha=0.02)
        fitness = raw_fitness - penalty

    Note:
        - 線性懲罰節點數量
        - 深度有額外權重（深度對計算成本影響更大）
        - alpha 越大，越傾向簡單解
    """
    size = len(individual)
    depth = individual.height

    # 線性懲罰 + 深度加權
    penalty = alpha * (size + depth * depth_weight)

    return penalty


def count_constants(individual: 'gp.PrimitiveTree') -> int:
    """
    計算個體中的常數數量

    Args:
        individual: GP 個體

    Returns:
        int: 常數數量

    使用範例:
        num_constants = count_constants(individual)
        if num_constants > 10:
            # 常數太多，可能過擬合
            pass
    """
    constants = 0

    for node in individual:
        # DEAP 的常數節點是 Terminal 且非 ARG
        if isinstance(node, gp.Terminal) and not node.name.startswith("ARG"):
            constants += 1

    return constants


def validate_individual(
    individual: 'gp.PrimitiveTree',
    config: Optional[ConstraintConfig] = None
) -> bool:
    """
    驗證個體是否符合所有約束

    Args:
        individual: GP 個體
        config: 約束配置

    Returns:
        bool: True 如果符合所有約束

    使用範例:
        if not validate_individual(individual, config):
            # 個體無效，重新生成
            individual = toolbox.individual()
    """
    cfg = config or ConstraintConfig()

    # 檢查深度
    if individual.height > cfg.max_depth:
        return False

    if individual.height < cfg.min_depth:
        return False

    # 檢查節點數量
    if len(individual) > cfg.max_size:
        return False

    # 檢查常數數量
    if count_constants(individual) > cfg.max_constants:
        return False

    return True


# ============================================================================
# 修剪過大個體（可選）
# ============================================================================

def prune_individual(
    individual: 'gp.PrimitiveTree',
    max_depth: int
) -> 'gp.PrimitiveTree':
    """
    修剪過深的個體

    將超過最大深度的子樹替換為 Terminal。

    Args:
        individual: GP 個體
        max_depth: 最大深度

    Returns:
        修剪後的個體

    Note:
        這是破壞性操作，會修改原個體。
        通常直接拒絕無效個體比修剪更好。

    Warning:
        此功能實驗性質，可能產生無效個體。
        建議使用 apply_constraints 阻止產生過深個體。
    """
    # 簡單實作：如果超過深度，隨機選擇子樹替換為 Terminal
    # 完整實作需要遞迴遍歷並修剪

    if individual.height <= max_depth:
        return individual

    # TODO: 實作智慧修剪
    # 目前只返回原個體（交由 apply_constraints 處理）
    return individual
