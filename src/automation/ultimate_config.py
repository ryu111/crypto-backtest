"""
UltimateLoopConfig - UltimateLoop 終極配置類別

整合所有進階功能的配置：
1. HyperLoop（GPU + 三層並行）
2. Regime Detection（市場狀態識別）
3. CompositeStrategy（動態策略組合）
4. Multi-Objective Optimization（NSGA-II 多目標）
5. 5 階段驗證
6. Memory MCP 自動學習

使用範例：
    # 生產環境配置
    config = UltimateLoopConfig.create_production_config()

    # 開發測試配置
    config = UltimateLoopConfig.create_development_config()

    # 快速測試配置
    config = UltimateLoopConfig.create_quick_test_config()

    # 自訂配置
    config = UltimateLoopConfig(
        max_workers=16,
        use_gpu=True,
        regime_detection=True,
        validation_enabled=True
    )

    # 驗證配置
    config.validate()
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from src.types.enums import (
    DirectionMethod,
    StrategySelectionMode,
    AggregationMode,
    ObjectiveMetric,
    ParetoSelectMethod,
)

logger = logging.getLogger(__name__)


@dataclass
class UltimateLoopConfig:
    """
    UltimateLoop 終極配置類別

    整合所有進階功能，提供靈活的配置選項。

    Attributes:
        # 基礎設定
        max_workers (int): CPU 並行工作數
        use_gpu (bool): 是否使用 GPU 加速
        gpu_batch_size (int): GPU 批次大小
        symbols (List[str]): 交易標的列表
        timeframes (List[str]): 時間框架列表
        data_dir (str): 資料目錄

        # HyperLoop 設定
        hyperloop_enabled (bool): 是否啟用高效能 HyperLoop 引擎
        param_sweep_threshold (int): 超過此參數組合數使用 GPU

        # Regime Detection 設定
        regime_detection (bool): 是否啟用市場狀態識別
        direction_method (str): 方向性計算方法
        direction_threshold_strong (float): 強趨勢閾值
        direction_threshold_weak (float): 弱趨勢閾值
        volatility_threshold (float): 波動度閾值

        # 策略組合設定
        strategy_selection_mode (str): 策略選擇模式
        exploit_ratio (float): 利用 vs 探索比例
        aggregation_mode (str): 信號聚合模式
        enabled_strategies (Optional[List[str]]): 啟用的策略列表

        # 多目標優化設定
        objectives (List[Tuple[str, str]]): 優化目標列表
        n_trials (int): 優化試驗次數
        pareto_select_method (str): Pareto 解選擇方法
        pareto_top_n (int): 選擇 top N 個 Pareto 解

        # 驗證設定
        validation_enabled (bool): 是否啟用驗證
        min_stages (int): 最少通過驗證階段數
        min_sharpe (float): 最低 Sharpe Ratio
        max_overfit (float): 最大過擬合度

        # 學習系統設定
        learning_enabled (bool): 是否啟用學習系統
        memory_mcp_enabled (bool): 是否啟用 Memory MCP
        auto_insights (bool): 是否自動更新 insights.md
        experiment_dir (str): 實驗記錄目錄

        # 交易設定
        leverage (int): 槓桿倍數
        initial_capital (float): 初始資金
        maker_fee (float): Maker 手續費
        taker_fee (float): Taker 手續費

        # 執行設定
        timeout_per_iteration (int): 每次迭代超時時間（秒）
        max_retries (int): 最大重試次數
        checkpoint_enabled (bool): 是否啟用檢查點
        checkpoint_interval (int): 檢查點間隔
        checkpoint_dir (str): 檢查點目錄

        # GP 探索設定（Phase 13.x）
        gp_explore_enabled (bool): 是否啟用 GP 探索
        gp_explore_ratio (float): explore 中使用 GP 的比例（0-1）
        gp_population_size (int): GP 族群大小
        gp_generations (int): GP 演化代數
        gp_top_n (int): 每次 GP 探索產生的策略數
    """

    # ===== 基礎設定 =====
    max_workers: int = 8
    use_gpu: bool = True
    gpu_batch_size: int = 50
    symbols: List[str] = field(default_factory=lambda: ['BTCUSDT', 'ETHUSDT'])
    timeframes: List[str] = field(default_factory=lambda: [
        '1m', '3m', '5m', '15m', '30m',      # 短線
        '1h', '2h', '4h', '6h', '8h',        # 中線（8h 對齊資金費率）
        '12h', '1d', '3d', '1w'              # 長線
    ])
    data_dir: str = "data"

    # ===== HyperLoop 設定 =====
    hyperloop_enabled: bool = True  # 是否啟用高效能 HyperLoop 引擎
    param_sweep_threshold: int = 100  # 超過此參數組合數使用 GPU

    # ===== Regime Detection 設定 =====
    regime_detection: bool = True
    direction_method: DirectionMethod = DirectionMethod.COMPOSITE
    direction_threshold_strong: float = 5.0
    direction_threshold_weak: float = 2.0
    volatility_threshold: float = 5.0

    # ===== 策略組合設定 =====
    strategy_selection_mode: StrategySelectionMode = StrategySelectionMode.REGIME_AWARE
    exploit_ratio: float = 0.8  # 80% exploit / 20% explore
    aggregation_mode: AggregationMode = AggregationMode.WEIGHTED
    enabled_strategies: Optional[List[str]] = None  # None = 使用所有註冊策略

    # ===== 多目標優化設定 =====
    objectives: List[Tuple[ObjectiveMetric, str]] = field(default_factory=lambda: [
        (ObjectiveMetric.SHARPE_RATIO, 'maximize'),
        (ObjectiveMetric.MAX_DRAWDOWN, 'minimize'),
        (ObjectiveMetric.WIN_RATE, 'maximize')
    ])
    n_trials: int = 100
    pareto_select_method: ParetoSelectMethod = ParetoSelectMethod.KNEE
    pareto_top_n: int = 3  # 選擇 top N 個 Pareto 解進行驗證

    # ===== 驗證設定 =====
    validation_enabled: bool = True
    min_stages: int = 5  # 最少通過幾個驗證階段（全 5 階段）
    min_sharpe: float = 1.0
    max_overfit: float = 0.5

    # ===== 學習系統設定 =====
    learning_enabled: bool = True
    memory_mcp_enabled: bool = True
    auto_insights: bool = True  # 自動更新 insights.md
    experiment_dir: str = "learning"

    # Memory MCP 進階設定（Phase 12.8）
    # 注意：啟用開關使用上方的 memory_mcp_enabled
    memory_min_sharpe: float = 1.0       # 最低 Sharpe 才存入 Memory
    memory_store_failures: bool = True    # 是否存儲失敗教訓

    # ===== 交易設定 =====
    leverage: int = 5
    initial_capital: float = 10000.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0004

    # ===== 執行設定 =====
    timeout_per_iteration: int = 600  # 10 分鐘
    max_retries: int = 3
    checkpoint_enabled: bool = True
    checkpoint_interval: int = 10  # 每 10 次迭代存一次
    checkpoint_dir: str = "checkpoints"

    # ===== 高效能並行設定（Phase 12.12） =====
    batch_size: int = 50                      # 每批回測數量
    use_shared_memory: bool = True            # 使用共享記憶體
    data_pool_max_gb: float = 20.0            # 資料池最大 GB

    # ===== 交易優化功能（Phase 12.12） =====
    signal_amplification_enabled: bool = False   # 信號放大器
    signal_filter_enabled: bool = False          # 信號過濾管道
    dynamic_risk_enabled: bool = False           # 動態風控
    adaptive_leverage_enabled: bool = False      # 自適應槓桿

    # ===== 迭代設定（Phase 12.12） =====
    n_iterations: int = 50                       # 總迭代次數
    trials_per_iteration: int = 20               # 每輪迭代的 trials 數

    # ===== GP 探索設定（Phase 13.x） =====
    gp_explore_enabled: bool = True              # 是否啟用 GP 探索
    gp_explore_ratio: float = 0.2                # explore 中使用 GP 的比例（0-1）
    gp_population_size: int = 50                 # GP 族群大小
    gp_generations: int = 30                     # GP 演化代數
    gp_top_n: int = 3                            # 每次 GP 探索產生的策略數

    # ===== Walk-Forward 設定（Skills 對齊） =====
    use_walk_forward: bool = True                # 是否啟用 Walk-Forward 分析
    wfa_is_ratio: float = 0.7                    # In-Sample 比例
    wfa_n_windows: int = 5                       # 滾動窗口數量
    wfa_overlap: float = 0.5                     # 窗口重疊比例
    wfa_min_efficiency: float = 0.5              # 最低可接受效率

    # ===== 過擬合偵測設定（Skills 對齊） =====
    min_trades: int = 30                         # 最低交易筆數（統計有效性）
    max_pbo: float = 0.5                         # 最大可接受 PBO
    max_is_oos_ratio: float = 2.0                # 最大可接受 IS/OOS 比
    max_param_sensitivity: float = 0.3           # 最大可接受參數敏感度

    # ===== 兩階段參數搜索設定（Skills 對齊） =====
    two_stage_search: bool = True                # 是否啟用兩階段搜索
    coarse_trials: int = 20                      # 粗搜索試驗數
    coarse_step_multiplier: float = 3.0          # 粗搜索步長倍數
    fine_trials: int = 50                        # 細搜索試驗數

    # ===== 參數預生成設定（Skills 對齊） =====
    use_param_pregeneration: bool = True         # 是否基於歷史最佳參數預生成搜索範圍
    param_pregeneration_ratio: float = 0.3       # 預生成範圍比例（±30%）

    def validate(self) -> bool:
        """
        驗證配置有效性

        Returns:
            bool: 配置是否有效

        Raises:
            ValueError: 配置無效時拋出
        """
        errors = []

        # 1. 基礎設定驗證
        if self.max_workers < 1:
            errors.append("max_workers 必須 >= 1")

        if self.gpu_batch_size < 1:
            errors.append("gpu_batch_size 必須 >= 1")

        if not self.symbols:
            errors.append("symbols 不能為空")

        if not self.timeframes:
            errors.append("timeframes 不能為空")

        # 2. Regime Detection 設定驗證
        if not isinstance(self.direction_method, DirectionMethod):
            errors.append(f"direction_method 必須是 DirectionMethod 類型，收到: {type(self.direction_method)}")

        if not (0 <= self.direction_threshold_strong <= 10):
            errors.append("direction_threshold_strong 必須在 0-10 之間")

        if not (0 <= self.direction_threshold_weak <= 10):
            errors.append("direction_threshold_weak 必須在 0-10 之間")

        if not (0 <= self.volatility_threshold <= 10):
            errors.append("volatility_threshold 必須在 0-10 之間")

        # 3. 策略組合設定驗證
        if not isinstance(self.strategy_selection_mode, StrategySelectionMode):
            errors.append(f"strategy_selection_mode 必須是 StrategySelectionMode 類型，收到: {type(self.strategy_selection_mode)}")

        if not (0 <= self.exploit_ratio <= 1):
            errors.append("exploit_ratio 必須在 0-1 之間")

        if not isinstance(self.aggregation_mode, AggregationMode):
            errors.append(f"aggregation_mode 必須是 AggregationMode 類型，收到: {type(self.aggregation_mode)}")

        # 4. 多目標優化設定驗證
        if not self.objectives:
            errors.append("objectives 不能為空")

        for i, (metric, direction) in enumerate(self.objectives):
            # 驗證 metric 是否為 ObjectiveMetric 類型
            if not isinstance(metric, ObjectiveMetric):
                errors.append(f"objectives[{i}] metric 必須是 ObjectiveMetric 類型，收到: {type(metric)}")
            # 驗證 direction
            if direction not in ['maximize', 'minimize']:
                errors.append(f"objectives[{i}] 方向必須是 'maximize' 或 'minimize'，收到: {direction}")

        if self.n_trials < 1:
            errors.append("n_trials 必須 >= 1")

        if not isinstance(self.pareto_select_method, ParetoSelectMethod):
            errors.append(f"pareto_select_method 必須是 ParetoSelectMethod 類型，收到: {type(self.pareto_select_method)}")

        if self.pareto_top_n < 1:
            errors.append("pareto_top_n 必須 >= 1")

        # 5. 驗證設定驗證
        if not (0 <= self.min_stages <= 5):
            errors.append("min_stages 必須在 0-5 之間")

        if self.min_sharpe < 0:
            errors.append("min_sharpe 必須 >= 0")

        if not (0 <= self.max_overfit <= 1):
            errors.append("max_overfit 必須在 0-1 之間")

        # 8. Memory MCP 設定驗證（Phase 12.8）
        if self.memory_min_sharpe < 0:
            errors.append("memory_min_sharpe 必須 >= 0")

        # 6. 交易設定驗證
        if self.leverage < 1:
            errors.append("leverage 必須 >= 1")

        if self.initial_capital <= 0:
            errors.append("initial_capital 必須 > 0")

        if not (0 <= self.maker_fee <= 0.01):
            errors.append("maker_fee 必須在 0-0.01 之間")

        if not (0 <= self.taker_fee <= 0.01):
            errors.append("taker_fee 必須在 0-0.01 之間")

        # 9. 執行設定驗證
        if self.timeout_per_iteration < 1:
            errors.append("timeout_per_iteration 必須 >= 1")

        if self.max_retries < 0:
            errors.append("max_retries 必須 >= 0")

        if self.checkpoint_interval < 1:
            errors.append("checkpoint_interval 必須 >= 1")

        # 10. 高效能並行設定驗證（Phase 12.12）
        if self.batch_size < 1:
            errors.append("batch_size 必須 >= 1")

        if self.data_pool_max_gb <= 0:
            errors.append("data_pool_max_gb 必須 > 0")

        if self.n_iterations < 1:
            errors.append("n_iterations 必須 >= 1")

        if self.trials_per_iteration < 1:
            errors.append("trials_per_iteration 必須 >= 1")

        # 11. GP 探索設定驗證（Phase 13.x）
        if not (0 <= self.gp_explore_ratio <= 1):
            errors.append("gp_explore_ratio 必須在 0-1 之間")

        if self.gp_population_size < 1:
            errors.append("gp_population_size 必須 >= 1")

        if self.gp_generations < 1:
            errors.append("gp_generations 必須 >= 1")

        if self.gp_top_n < 1:
            errors.append("gp_top_n 必須 >= 1")

        # 12. Walk-Forward 設定驗證（Skills 對齊）
        if not (0.5 <= self.wfa_is_ratio <= 0.9):
            errors.append("wfa_is_ratio 必須在 0.5-0.9 之間")

        if self.wfa_n_windows < 2:
            errors.append("wfa_n_windows 必須 >= 2")

        if not (0 <= self.wfa_overlap < 1):
            errors.append("wfa_overlap 必須在 0-1 之間")

        if not (0 <= self.wfa_min_efficiency <= 1):
            errors.append("wfa_min_efficiency 必須在 0-1 之間")

        # 13. 過擬合偵測設定驗證（Skills 對齊）
        if self.min_trades < 1:
            errors.append("min_trades 必須 >= 1")

        if not (0 <= self.max_pbo <= 1):
            errors.append("max_pbo 必須在 0-1 之間")

        if self.max_is_oos_ratio < 1:
            errors.append("max_is_oos_ratio 必須 >= 1")

        if not (0 <= self.max_param_sensitivity <= 1):
            errors.append("max_param_sensitivity 必須在 0-1 之間")

        # 14. 兩階段搜索設定驗證（Skills 對齊）
        if self.coarse_trials < 1:
            errors.append("coarse_trials 必須 >= 1")

        if self.coarse_step_multiplier < 1:
            errors.append("coarse_step_multiplier 必須 >= 1")

        if self.fine_trials < 1:
            errors.append("fine_trials 必須 >= 1")

        # 12. 路徑安全驗證
        for path_name, path_value in [
            ('data_dir', self.data_dir),
            ('experiment_dir', self.experiment_dir),
            ('checkpoint_dir', self.checkpoint_dir)
        ]:
            if '..' in path_value:
                errors.append(f"{path_name} 路徑不安全（包含 ..）: {path_value}")

        # 如果有錯誤，拋出異常
        if errors:
            error_msg = "\n".join([f"  - {err}" for err in errors])
            raise ValueError(f"配置驗證失敗：\n{error_msg}")

        return True

    @classmethod
    def create_production_config(cls) -> 'UltimateLoopConfig':
        """
        建立生產環境配置

        特點：
        - 高並行（16 workers）
        - 啟用 GPU
        - 完整驗證（5 階段）
        - 啟用所有進階功能
        - 嚴格的品質標準

        Returns:
            UltimateLoopConfig: 生產環境配置
        """
        return cls(
            # 高效能設定
            max_workers=16,
            use_gpu=True,
            gpu_batch_size=100,

            # 完整標的和時間框架
            symbols=['BTCUSDT', 'ETHUSDT'],
            timeframes=[
                '1m', '3m', '5m', '15m', '30m',
                '1h', '2h', '4h', '6h', '8h',
                '12h', '1d', '3d', '1w'
            ],

            # 啟用所有進階功能
            regime_detection=True,
            strategy_selection_mode=StrategySelectionMode.REGIME_AWARE,
            exploit_ratio=0.9,  # 90% exploit（生產環境更保守）

            # 多目標優化
            objectives=[
                (ObjectiveMetric.SHARPE_RATIO, 'maximize'),
                (ObjectiveMetric.MAX_DRAWDOWN, 'minimize'),
                (ObjectiveMetric.WIN_RATE, 'maximize'),
                (ObjectiveMetric.PROFIT_FACTOR, 'maximize')
            ],
            n_trials=200,  # 更多試驗
            pareto_top_n=5,

            # 嚴格驗證
            validation_enabled=True,
            min_stages=5,  # 必須通過所有階段
            min_sharpe=1.5,  # 更高要求
            max_overfit=0.3,  # 更嚴格的過擬合控制

            # 啟用學習系統
            learning_enabled=True,
            memory_mcp_enabled=True,
            auto_insights=True,

            # Memory MCP 進階設定（Phase 12.8）
            memory_min_sharpe=1.5,  # 生產環境更高標準
            memory_store_failures=True,

            # 生產級交易設定
            leverage=5,
            initial_capital=10000.0,

            # 執行設定
            timeout_per_iteration=900,  # 15 分鐘
            max_retries=3,
            checkpoint_enabled=True,
            checkpoint_interval=5,  # 更頻繁的檢查點

            # GP 探索設定（Phase 13.x）
            gp_explore_enabled=True,
            gp_explore_ratio=0.15,  # 生產環境保守（15% GP）
            gp_population_size=100,  # 大族群
            gp_generations=50,       # 更多演化代數
            gp_top_n=5               # 產生 5 個優質策略
        )

    @classmethod
    def create_development_config(cls) -> 'UltimateLoopConfig':
        """
        建立開發測試配置

        特點：
        - 中等並行（8 workers）
        - 啟用 GPU
        - 基礎驗證（3 階段）
        - 啟用主要進階功能
        - 適中的品質標準

        Returns:
            UltimateLoopConfig: 開發測試配置
        """
        return cls(
            # 中等效能設定
            max_workers=8,
            use_gpu=True,
            gpu_batch_size=50,

            # 部分標的和時間框架
            symbols=['BTCUSDT'],
            timeframes=['1h', '4h', '1d'],

            # 啟用主要功能
            regime_detection=True,
            strategy_selection_mode=StrategySelectionMode.REGIME_AWARE,
            exploit_ratio=0.8,

            # 多目標優化
            objectives=[
                (ObjectiveMetric.SHARPE_RATIO, 'maximize'),
                (ObjectiveMetric.MAX_DRAWDOWN, 'minimize')
            ],
            n_trials=100,
            pareto_top_n=3,

            # 基礎驗證
            validation_enabled=True,
            min_stages=3,
            min_sharpe=1.0,
            max_overfit=0.5,

            # 啟用學習系統
            learning_enabled=True,
            memory_mcp_enabled=True,
            auto_insights=True,

            # Memory MCP 進階設定（Phase 12.8）
            memory_min_sharpe=1.0,  # 開發環境標準
            memory_store_failures=True,

            # 開發級交易設定
            leverage=5,
            initial_capital=10000.0,

            # 執行設定
            timeout_per_iteration=600,  # 10 分鐘
            max_retries=3,
            checkpoint_enabled=True,
            checkpoint_interval=10,

            # GP 探索設定（Phase 13.x）
            gp_explore_enabled=True,
            gp_explore_ratio=0.2,    # 標準 20% GP
            gp_population_size=50,   # 中等族群
            gp_generations=30,       # 標準演化代數
            gp_top_n=3               # 產生 3 個策略
        )

    @classmethod
    def create_quick_test_config(cls) -> 'UltimateLoopConfig':
        """
        建立快速測試配置

        特點：
        - 低並行（2 workers）
        - 不使用 GPU
        - 最少驗證（1 階段）
        - 關閉部分進階功能
        - 寬鬆的品質標準

        Returns:
            UltimateLoopConfig: 快速測試配置
        """
        return cls(
            # 最小效能設定
            max_workers=2,
            use_gpu=False,
            gpu_batch_size=10,

            # 單一標的和時間框架
            symbols=['BTCUSDT'],
            timeframes=['1h'],

            # 簡化功能
            regime_detection=False,
            strategy_selection_mode=StrategySelectionMode.RANDOM,
            exploit_ratio=0.5,

            # 單目標優化
            objectives=[
                (ObjectiveMetric.SHARPE_RATIO, 'maximize')
            ],
            n_trials=20,  # 快速試驗
            pareto_top_n=1,

            # 最少驗證
            validation_enabled=True,
            min_stages=1,
            min_sharpe=0.5,
            max_overfit=0.8,

            # 關閉學習系統
            learning_enabled=False,
            memory_mcp_enabled=False,
            auto_insights=False,

            # Memory MCP 進階設定（Phase 12.8）
            memory_min_sharpe=0.5,
            memory_store_failures=False,

            # 測試級交易設定
            leverage=5,
            initial_capital=10000.0,

            # 執行設定
            timeout_per_iteration=300,  # 5 分鐘
            max_retries=1,
            checkpoint_enabled=False,
            checkpoint_interval=100,

            # GP 探索設定（Phase 13.x）
            gp_explore_enabled=False,  # 快速測試關閉 GP
            gp_explore_ratio=0.1,      # 小比例
            gp_population_size=20,     # 小族群
            gp_generations=10,         # 少代數
            gp_top_n=1                 # 只產生 1 個策略
        )

    @classmethod
    def create_high_performance_config(cls) -> 'UltimateLoopConfig':
        """
        建立高效能配置（M4 Max 優化）

        專為 Apple M4 Max（16 核心、64GB RAM）設計：
        - 使用 70-80% 系統資源
        - 12 核心並行處理
        - 最大 40GB 資料池
        - 啟用所有交易優化功能

        預估效能：
        - 100 iterations：約 5-10 分鐘
        - 5000 total trials
        - 11x 並行加速

        Returns:
            UltimateLoopConfig: 高效能配置
        """
        return cls(
            # ===== 高效能並行設定 =====
            max_workers=12,                # 12 核心並行（70-80% of 16 cores）
            use_gpu=True,
            gpu_batch_size=100,            # 大批次 GPU 處理
            batch_size=100,                # 每批 100 個回測
            use_shared_memory=True,        # 共享記憶體零拷貝
            data_pool_max_gb=40.0,         # 最大 40GB 資料池

            # ===== 迭代設定 =====
            n_iterations=100,              # 100 輪迭代
            trials_per_iteration=50,       # 每輪 50 trials
            n_trials=50,                   # Optuna trials per iteration

            # ===== 完整標的和時間框架 =====
            symbols=['BTCUSDT', 'ETHUSDT'],
            timeframes=[
                '15m', '30m',           # 短線
                '1h', '4h',             # 中線
                '1d'                    # 長線
            ],

            # ===== 啟用所有交易優化功能 =====
            signal_amplification_enabled=True,   # 信號放大器
            signal_filter_enabled=True,          # 信號過濾管道
            dynamic_risk_enabled=True,           # 動態風控
            adaptive_leverage_enabled=True,      # 自適應槓桿

            # ===== 完整 Regime Detection =====
            regime_detection=True,
            strategy_selection_mode=StrategySelectionMode.REGIME_AWARE,
            exploit_ratio=0.8,

            # ===== 多目標優化 =====
            objectives=[
                (ObjectiveMetric.SHARPE_RATIO, 'maximize'),
                (ObjectiveMetric.MAX_DRAWDOWN, 'minimize'),
                (ObjectiveMetric.WIN_RATE, 'maximize')
            ],
            pareto_select_method=ParetoSelectMethod.KNEE,
            pareto_top_n=5,

            # ===== 全 5 階段驗證 =====
            validation_enabled=True,
            min_stages=5,
            min_sharpe=1.0,
            max_overfit=0.5,

            # ===== 啟用學習系統 =====
            learning_enabled=True,
            memory_mcp_enabled=True,
            auto_insights=True,
            memory_min_sharpe=1.0,
            memory_store_failures=True,

            # ===== 交易設定 =====
            leverage=5,
            initial_capital=10000.0,

            # ===== 執行設定 =====
            timeout_per_iteration=600,     # 10 分鐘超時
            max_retries=3,
            checkpoint_enabled=True,
            checkpoint_interval=10,        # 每 10 次迭代存檢查點

            # ===== GP 探索設定（Phase 13.x） =====
            gp_explore_enabled=True,
            gp_explore_ratio=0.25,         # 高效能模式可以多嘗試（25% GP）
            gp_population_size=100,        # 大族群
            gp_generations=50,             # 充分演化
            gp_top_n=5                     # 產生 5 個優質策略
        )

    def to_dict(self) -> dict:
        """
        轉換為字典格式

        Returns:
            dict: 配置字典
        """
        return {
            # 基礎設定
            'max_workers': self.max_workers,
            'use_gpu': self.use_gpu,
            'gpu_batch_size': self.gpu_batch_size,
            'symbols': self.symbols,
            'timeframes': self.timeframes,
            'data_dir': self.data_dir,

            # Regime Detection 設定
            'regime_detection': self.regime_detection,
            'direction_method': self.direction_method.value,
            'direction_threshold_strong': self.direction_threshold_strong,
            'direction_threshold_weak': self.direction_threshold_weak,
            'volatility_threshold': self.volatility_threshold,

            # 策略組合設定
            'strategy_selection_mode': self.strategy_selection_mode.value,
            'exploit_ratio': self.exploit_ratio,
            'aggregation_mode': self.aggregation_mode.value,
            'enabled_strategies': self.enabled_strategies,

            # 多目標優化設定
            'objectives': [(metric.value, direction) for metric, direction in self.objectives],
            'n_trials': self.n_trials,
            'pareto_select_method': self.pareto_select_method.value,
            'pareto_top_n': self.pareto_top_n,

            # 驗證設定
            'validation_enabled': self.validation_enabled,
            'min_stages': self.min_stages,
            'min_sharpe': self.min_sharpe,
            'max_overfit': self.max_overfit,

            # 學習系統設定
            'learning_enabled': self.learning_enabled,
            'memory_mcp_enabled': self.memory_mcp_enabled,
            'auto_insights': self.auto_insights,
            'experiment_dir': self.experiment_dir,

            # Memory MCP 進階設定（Phase 12.8）
            'memory_min_sharpe': self.memory_min_sharpe,
            'memory_store_failures': self.memory_store_failures,

            # 交易設定
            'leverage': self.leverage,
            'initial_capital': self.initial_capital,
            'maker_fee': self.maker_fee,
            'taker_fee': self.taker_fee,

            # 執行設定
            'timeout_per_iteration': self.timeout_per_iteration,
            'max_retries': self.max_retries,
            'checkpoint_enabled': self.checkpoint_enabled,
            'checkpoint_interval': self.checkpoint_interval,
            'checkpoint_dir': self.checkpoint_dir,

            # 高效能並行設定（Phase 12.12）
            'batch_size': self.batch_size,
            'use_shared_memory': self.use_shared_memory,
            'data_pool_max_gb': self.data_pool_max_gb,

            # 交易優化功能（Phase 12.12）
            'signal_amplification_enabled': self.signal_amplification_enabled,
            'signal_filter_enabled': self.signal_filter_enabled,
            'dynamic_risk_enabled': self.dynamic_risk_enabled,
            'adaptive_leverage_enabled': self.adaptive_leverage_enabled,

            # 迭代設定（Phase 12.12）
            'n_iterations': self.n_iterations,
            'trials_per_iteration': self.trials_per_iteration,

            # GP 探索設定（Phase 13.x）
            'gp_explore_enabled': self.gp_explore_enabled,
            'gp_explore_ratio': self.gp_explore_ratio,
            'gp_population_size': self.gp_population_size,
            'gp_generations': self.gp_generations,
            'gp_top_n': self.gp_top_n,

            # Walk-Forward 設定（Skills 對齊）
            'use_walk_forward': self.use_walk_forward,
            'wfa_is_ratio': self.wfa_is_ratio,
            'wfa_n_windows': self.wfa_n_windows,
            'wfa_overlap': self.wfa_overlap,
            'wfa_min_efficiency': self.wfa_min_efficiency,

            # 過擬合偵測設定（Skills 對齊）
            'min_trades': self.min_trades,
            'max_pbo': self.max_pbo,
            'max_is_oos_ratio': self.max_is_oos_ratio,
            'max_param_sensitivity': self.max_param_sensitivity,

            # 兩階段搜索設定（Skills 對齊）
            'two_stage_search': self.two_stage_search,
            'coarse_trials': self.coarse_trials,
            'coarse_step_multiplier': self.coarse_step_multiplier,
            'fine_trials': self.fine_trials
        }

    def __repr__(self) -> str:
        """字串表示"""
        return (
            f"UltimateLoopConfig("
            f"workers={self.max_workers}, "
            f"gpu={self.use_gpu}, "
            f"regime={self.regime_detection}, "
            f"validation={self.validation_enabled})"
        )


if __name__ == "__main__":
    """測試配置類別"""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 70)
    print("測試 UltimateLoopConfig")
    print("=" * 70)

    # 測試生產配置
    print("\n1. 生產環境配置:")
    prod_config = UltimateLoopConfig.create_production_config()
    prod_config.validate()
    print(f"   ✓ {prod_config}")
    print(f"   Workers: {prod_config.max_workers}")
    print(f"   Objectives: {prod_config.objectives}")
    print(f"   Min Stages: {prod_config.min_stages}")

    # 測試開發配置
    print("\n2. 開發測試配置:")
    dev_config = UltimateLoopConfig.create_development_config()
    dev_config.validate()
    print(f"   ✓ {dev_config}")
    print(f"   Workers: {dev_config.max_workers}")
    print(f"   Objectives: {dev_config.objectives}")
    print(f"   Min Stages: {dev_config.min_stages}")

    # 測試快速測試配置
    print("\n3. 快速測試配置:")
    quick_config = UltimateLoopConfig.create_quick_test_config()
    quick_config.validate()
    print(f"   ✓ {quick_config}")
    print(f"   Workers: {quick_config.max_workers}")
    print(f"   Objectives: {quick_config.objectives}")
    print(f"   Min Stages: {quick_config.min_stages}")

    # 測試高效能配置（Phase 12.12）
    print("\n4. 高效能配置 (M4 Max):")
    hp_config = UltimateLoopConfig.create_high_performance_config()
    hp_config.validate()
    print(f"   ✓ {hp_config}")
    print(f"   Workers: {hp_config.max_workers}")
    print(f"   Batch Size: {hp_config.batch_size}")
    print(f"   Data Pool: {hp_config.data_pool_max_gb} GB")
    print(f"   Iterations: {hp_config.n_iterations}")
    print(f"   Trials/Iteration: {hp_config.trials_per_iteration}")
    print(f"   Signal Amplification: {hp_config.signal_amplification_enabled}")
    print(f"   Signal Filter: {hp_config.signal_filter_enabled}")
    print(f"   Dynamic Risk: {hp_config.dynamic_risk_enabled}")
    print(f"   Adaptive Leverage: {hp_config.adaptive_leverage_enabled}")

    # 測試無效配置
    print("\n5. 測試無效配置:")
    try:
        invalid_config = UltimateLoopConfig(
            max_workers=-1,  # 無效
            direction_threshold_strong=15.0  # 無效
        )
        invalid_config.validate()
        print("   ✗ 應該拋出異常")
    except ValueError as e:
        print(f"   ✓ 正確捕獲異常:")
        print(f"   {e}")

    # 測試 to_dict
    print("\n6. 測試 to_dict:")
    config_dict = dev_config.to_dict()
    print(f"   ✓ 轉換為字典：{len(config_dict)} 個鍵")
    print(f"   鍵: {list(config_dict.keys())[:5]}...")

    print("\n✅ UltimateLoopConfig 測試完成")
