"""
配置型別定義

包含回測配置、循環配置、優化配置等。
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, Any, List

from .enums import OptimizationMethod, ObjectiveMetric


@dataclass
class BacktestConfig:
    """
    回測配置（對應 experiments.json 的 config 欄位）

    使用範例:
        config = BacktestConfig(
            symbol="BTCUSDT",
            timeframe="4h",
            start_date="2020-01-01",
            end_date="2024-01-01",
        )
    """

    # 標的和時間框架
    symbol: str  # BTCUSDT, ETHUSDT
    timeframe: str  # 1m, 5m, 15m, 1h, 4h, 1d

    # 回測時間範圍
    start_date: str  # ISO 格式或 YYYY-MM-DD
    end_date: str

    # 交易配置
    initial_capital: float = 10000.0
    leverage: int = 1
    commission: float = 0.0004  # 0.04% Binance taker fee
    slippage: float = 0.0001  # 0.01%

    # 資金費率（永續合約）
    include_funding: bool = True
    funding_rate: float = 0.0001  # 平均資金費率

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典（JSON 序列化）"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BacktestConfig':
        """從字典建立"""
        # 過濾掉未知欄位
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class OptimizationConfig:
    """
    參數優化配置

    使用範例:
        config = OptimizationConfig(
            method="bayesian",
            n_iterations=50,
            n_random_starts=10,
            objective="sharpe_ratio",
        )
    """

    # 優化方法
    method: OptimizationMethod = OptimizationMethod.BAYESIAN  # bayesian / grid / random

    # 迭代次數
    n_iterations: int = 50
    n_random_starts: int = 10  # Bayesian 隨機起始點

    # 目標函數
    objective: ObjectiveMetric = ObjectiveMetric.SHARPE_RATIO  # sharpe_ratio / sortino_ratio / calmar_ratio

    # Walk-Forward 設定
    wf_windows: int = 5  # Walk-Forward 窗口數
    wf_train_ratio: float = 0.7  # 訓練集比例

    # 穩健性測試
    robustness_samples: int = 20  # 參數擾動樣本數
    robustness_std: float = 0.1  # 參數擾動標準差（相對）

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典（JSON 序列化）"""
        data = asdict(self)
        # Enum 轉字串
        data['method'] = self.method.value if isinstance(self.method, OptimizationMethod) else self.method
        data['objective'] = self.objective.value if isinstance(self.objective, ObjectiveMetric) else self.objective
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationConfig':
        """從字典建立"""
        data = data.copy()
        # 字串轉 Enum
        if 'method' in data and isinstance(data['method'], str):
            try:
                data['method'] = OptimizationMethod(data['method'])
            except ValueError:
                pass  # 保持原值，讓 dataclass 驗證處理
        if 'objective' in data and isinstance(data['objective'], str):
            try:
                data['objective'] = ObjectiveMetric(data['objective'])
            except ValueError:
                pass  # 保持原值，讓 dataclass 驗證處理
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class LoopConfig:
    """
    自動化循環配置（AI Loop）

    使用範例:
        config = LoopConfig(
            max_iterations=100,
            target_sharpe=2.0,
            symbols=["BTCUSDT", "ETHUSDT"],
            exploit_ratio=0.8,
        )
    """

    # 循環控制
    max_iterations: int = 100
    stop_on_target: bool = True  # 達標後停止
    target_sharpe: float = 2.0  # 目標 Sharpe Ratio

    # 策略選擇
    exploit_ratio: float = 0.8  # 80% exploit / 20% explore
    strategy_types: Optional[List[str]] = None  # None = 所有類型

    # 標的設定
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])
    timeframes: List[str] = field(default_factory=lambda: ["4h"])

    # 優化設定
    optimization: Optional[OptimizationConfig] = None

    # 記錄設定
    log_interval: int = 10  # 每 N 次記錄一次
    save_checkpoints: bool = True

    def __post_init__(self):
        """預設值處理"""
        if self.optimization is None:
            self.optimization = OptimizationConfig()

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典（JSON 序列化）"""
        data = asdict(self)
        # 處理嵌套的 dataclass
        if self.optimization:
            data['optimization'] = self.optimization.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LoopConfig':
        """從字典建立"""
        data = data.copy()
        # 處理嵌套的 dataclass
        if 'optimization' in data and isinstance(data['optimization'], dict):
            data['optimization'] = OptimizationConfig.from_dict(data['optimization'])
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)
