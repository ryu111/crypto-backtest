# Technical Design: GPLoop + UltimateLoop Integration

## Context

### 現有架構

```
UltimateLoopController
    │
    ├── _select_strategies()
    │       ↓
    │   StrategyRegistry.list_all() → 靜態策略列表
    │       ↓
    │   exploit/explore 選擇 → 只能選已有策略
    │
    ├── _run_optimization()
    │       ↓
    │   MultiObjectiveOptimizer / HyperLoop
    │
    └── _validate_pareto_solutions()
            ↓
        ValidationRunner → 5 階段驗證


GPLoop (獨立運行)
    │
    ├── GPEngine.evolve() → EvolutionResult
    │
    └── StrategyGenerator.generate() → Python 檔案
            ↓
        generated/*.py (需手動導入)
```

### 整合後架構

```
UltimateLoopController
    │
    ├── _select_strategies()
    │       ↓
    │   ┌─────────────────────────────────────────┐
    │   │ Explore?                                 │
    │   │   ├── GP Explore (20%) ──┐              │
    │   │   │                      ↓              │
    │   │   │   GPExplorer.explore()              │
    │   │   │       ↓                             │
    │   │   │   GPLoop 演化                       │
    │   │   │       ↓                             │
    │   │   │   動態註冊到 Registry                │
    │   │   │       ↓                             │
    │   │   │   返回新策略名稱                     │
    │   │   │                                     │
    │   │   └── Random Explore (80%)              │
    │   │           ↓                             │
    │   │       傳統隨機選擇                       │
    │   │                                         │
    │   └── Exploit?                              │
    │           ↓                                 │
    │       選擇歷史最佳（含 GP 策略）             │
    │                                             │
    └───────────────────────────────────────────┘
            ↓
    _run_optimization() → 優化所選策略
            ↓
    _validate_pareto_solutions() → 驗證
            ↓
    _record_and_learn() → 記錄（含 GP 元資料）
```

## Goals

1. **無縫整合**：GPLoop 作為 UltimateLoop 的 Explore 選項之一
2. **動態註冊**：GP 生成的策略可被後續迭代重複使用
3. **完整驗證**：GP 策略經過相同的 5 階段驗證流程
4. **學習整合**：GP 策略的演化歷程記錄到 Learning 系統

## Non-Goals

- 修改 GPLoop 核心演化邏輯
- 替換現有 Explore 機制
- 支援 GP 策略的增量演化（每次都是全新演化）

## Decisions

### Decision 1: GPExplorer 封裝模式

**決策**：使用獨立的 `GPExplorer` 類別封裝 GPLoop 調用

**原因**：
- 關注點分離：UltimateLoop 不需要了解 GP 細節
- 易於測試：可獨立測試 GPExplorer
- 易於擴展：未來可支援其他策略生成方法

**實作**：
```python
class GPExplorer:
    """GP 策略探索器 - 封裝 GPLoop 供 UltimateLoop 使用"""

    def __init__(self, config: GPExplorerConfig):
        self.config = config

    def explore(self, request: GPExplorationRequest) -> GPExplorationResult:
        """執行 GP 探索，生成並註冊新策略"""
        # 1. 建立 GPLoopConfig
        # 2. 執行 GPLoop
        # 3. 生成策略類別（記憶體中）
        # 4. 動態註冊到 Registry
        # 5. 返回結果
```

### Decision 2: 策略記憶體生成 vs 檔案生成

**決策**：優先使用**記憶體生成**，可選保存到檔案

**原因**：
- 效能：避免 I/O 開銷
- 簡化：不需要動態 import
- 彈性：檔案保存可選（用於持久化最佳策略）

**實作**：
```python
class GPStrategyAdapter:
    """將 GP 表達式轉為可註冊的策略類別"""

    def create_strategy_class(
        self,
        individual: Any,
        name: str,
        pset: Any,
        fitness: float,
        metadata: Dict
    ) -> Type[EvolvedStrategy]:
        """動態建立策略類別（不寫檔案）"""
        # 使用 type() 動態建立類別
        # 設定類別屬性（expression, fitness_score 等）
        # 編譯 signal_func
        # 返回類別
```

### Decision 3: 動態註冊機制

**決策**：擴展 `StrategyRegistry` 支援動態註冊

**原因**：
- 現有 `register()` 是裝飾器，不適合運行時使用
- 需要追蹤動態策略的來源和元資料
- 需要支援移除/替換動態策略

**實作**：
```python
class StrategyRegistry:
    # 現有靜態註冊（不變）
    _strategies: Dict[str, Type[BaseStrategy]] = {}

    # 新增：動態策略追蹤
    _dynamic_strategies: Dict[str, DynamicStrategyInfo] = {}

    @classmethod
    def register_dynamic(
        cls,
        name: str,
        strategy_class: Type[BaseStrategy],
        source: str = "gp",
        metadata: Optional[Dict] = None
    ) -> None:
        """動態註冊策略（運行時）"""
        # 驗證策略類別
        # 註冊到 _strategies
        # 記錄到 _dynamic_strategies
        # 記錄日誌

    @classmethod
    def unregister_dynamic(cls, name: str) -> bool:
        """移除動態策略"""
        # 只能移除動態策略，不能移除靜態策略

    @classmethod
    def list_dynamic(cls) -> List[str]:
        """列出所有動態策略"""
```

### Decision 4: GP Explore 觸發條件

**決策**：二層隨機機制

1. **第一層**：`exploit_ratio` 決定是否 explore（現有）
2. **第二層**：`gp_explore_ratio` 決定 explore 時是否用 GP

**原因**：
- 保持與現有 explore/exploit 機制相容
- GP 演化成本較高，不應每次 explore 都觸發
- 可調整比例平衡效率和創新

**配置**：
```python
@dataclass
class UltimateLoopConfig:
    # 現有
    exploit_ratio: float = 0.8  # 80% exploit

    # 新增
    gp_explore_enabled: bool = True
    gp_explore_ratio: float = 0.2  # explore 時 20% 用 GP
    gp_population_size: int = 50
    gp_generations: int = 30
    gp_top_n: int = 3  # 取前 N 個最佳策略
```

**計算**：
- 總體 GP 觸發機率 = (1 - exploit_ratio) * gp_explore_ratio
- 預設 = 0.2 * 0.2 = 4% 的迭代會觸發 GP

## Risks & Trade-offs

### Risk 1: GP 演化時間過長

**問題**：GP 演化可能需要幾分鐘，影響迭代效率

**緩解**：
- 限制 `gp_generations`（預設 30）
- 使用早停機制（已有）
- 非同步執行（未來優化）
- 調低 `gp_explore_ratio`

### Risk 2: 動態策略名稱衝突

**問題**：多次 GP 生成可能產生相同名稱

**緩解**：
- 使用 UUID 後綴：`gp_evolved_btcusdt_001_a1b2c3`
- 檢查名稱唯一性後再註冊
- 提供 `unregister_dynamic` 清理機制

### Risk 3: 記憶體累積

**問題**：動態策略累積過多占用記憶體

**緩解**：
- 設定動態策略上限（如 50 個）
- LRU 清理：移除最久未使用的動態策略
- 只保留通過驗證的策略

### Risk 4: GP 策略品質不穩定

**問題**：GP 可能生成無效或低品質策略

**緩解**：
- GP 已有適應度篩選
- 5 階段驗證會過濾低品質策略
- 未通過驗證的策略不會被 exploit 選中

## Migration Plan

**無需遷移** - 新功能是可選的：
- `gp_explore_enabled=False` 維持原有行為
- 現有測試不受影響
- 可漸進式啟用

## Alternatives Considered

### Alternative 1: 修改 GPLoop 直接整合

**描述**：讓 GPLoop 直接調用 UltimateLoop 的驗證流程

**拒絕原因**：
- 耦合過緊
- 破壞 GPLoop 的獨立性
- 不符合關注點分離原則

### Alternative 2: 使用檔案作為中介

**描述**：GP 生成檔案 → 動態 import

**拒絕原因**：
- I/O 開銷
- 需要處理模組重載
- 清理檔案複雜

### Alternative 3: 完全替換 Explore 機制

**描述**：所有 Explore 都用 GP

**拒絕原因**：
- GP 成本太高
- 傳統隨機 Explore 仍有價值
- 不符合漸進式整合原則
