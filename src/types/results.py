"""
回測和驗證結果型別定義

對應 learning/experiments.json 的結構。
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import pandas as pd

from .enums import ExperimentStatus, Grade


@dataclass
class PerformanceMetrics:
    """績效指標"""

    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None

    # 進階指標（選填）
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    max_consecutive_wins: Optional[int] = None
    max_consecutive_losses: Optional[int] = None
    expectancy: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典（JSON 序列化）"""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """從字典建立"""
        # 過濾掉未知欄位
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class BacktestResult:
    """
    回測結果（對應 experiments.json 的 results 欄位）

    包含績效指標和時間序列資料（equity curve, daily returns）
    """

    # 核心績效指標
    metrics: PerformanceMetrics

    # 時間序列資料（不序列化到 JSON）
    daily_returns: Optional[pd.Series] = None
    equity_curve: Optional[pd.Series] = None

    # 額外資訊
    execution_time: Optional[float] = None  # 執行時間（秒）

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典（JSON 序列化）"""
        return {
            **self.metrics.to_dict(),
            'execution_time': self.execution_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BacktestResult':
        """從字典建立"""
        metrics = PerformanceMetrics.from_dict(data)
        execution_time = data.get('execution_time')
        # 嘗試還原時間序列資料（如果存在於記憶體傳遞中）
        daily_returns = data.get('daily_returns')
        equity_curve = data.get('equity_curve')
        return cls(
            metrics=metrics,
            execution_time=execution_time,
            daily_returns=daily_returns,
            equity_curve=equity_curve,
        )


@dataclass
class ValidationResult:
    """
    驗證結果（對應 experiments.json 的 validation 欄位）

    5 階段驗證：
    1. 基本績效（Sharpe > 1.0）
    2. 穩健性測試（參數敏感度）
    3. Monte Carlo（隨機性檢驗）
    4. Walk-Forward（時間穩定性）
    5. 過擬合檢測
    """

    grade: str  # A/B/C/D/F
    stages_passed: List[int]  # 通過的階段編號 [1, 2, 3]

    # 詳細資訊（選填）
    efficiency: Optional[float] = None  # Sharpe 效率（vs. 調整後 Sharpe）
    overfit_probability: Optional[float] = None  # 過擬合機率
    details: Optional[Dict[str, Any]] = None  # 其他詳細資訊

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典（JSON 序列化）"""
        result = {
            'grade': self.grade,
            'stages_passed': self.stages_passed,
        }
        if self.efficiency is not None:
            result['efficiency'] = self.efficiency
        if self.overfit_probability is not None:
            result['overfit_probability'] = self.overfit_probability
        if self.details:
            result['details'] = self.details
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationResult':
        """從字典建立"""
        return cls(
            grade=data['grade'],
            stages_passed=data['stages_passed'],
            efficiency=data.get('efficiency'),
            overfit_probability=data.get('overfit_probability'),
            details=data.get('details'),
        )

    @property
    def is_passing(self) -> bool:
        """是否通過驗證（A/B 為通過）"""
        return self.grade in [Grade.A.value, Grade.B.value]


@dataclass
class ExperimentRecord:
    """
    完整實驗記錄（對應 experiments.json 的結構）

    設計說明：
        strategy, config, results, validation 欄位使用 Dict[str, Any] 而非強型別，
        原因是需要與 experiments.json 格式完全相容。使用 property 存取器提供
        型別安全的存取方式。

    使用範例:
        # 從 JSON 還原
        experiment = ExperimentRecord.from_dict(json_data)

        # 型別安全存取
        sharpe = experiment.sharpe_ratio  # float
        grade = experiment.grade  # str

        # 序列化
        data = experiment.to_dict()
        json.dump(data, f)
    """

    # 基本資訊
    id: str  # exp_{timestamp}_{symbol}_{strategy_name}
    timestamp: datetime

    # 策略資訊 {name, type, version, params}
    strategy: Dict[str, Any]

    # 配置 {symbol, timeframe, start_date, end_date, ...}
    config: Dict[str, Any]

    # 結果 {sharpe_ratio, total_return, max_drawdown, ...}
    results: Dict[str, float]

    # 驗證 {grade, stages_passed, ...}
    validation: Dict[str, Any]

    # 狀態（支援 Enum 或字串，向後相容）
    status: Union[ExperimentStatus, str] = ExperimentStatus.COMPLETED

    # 洞察和標籤
    insights: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    # 演進追蹤
    parent_experiment: Optional[str] = None
    improvement: Optional[float] = None  # vs. parent

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典（JSON 序列化）"""
        data = asdict(self)
        # 轉換 datetime 為 ISO 格式
        data['timestamp'] = self.timestamp.isoformat()
        # 轉換 Enum 為字串值（向後相容：支援字串輸入）
        data['status'] = self.status.value if isinstance(self.status, ExperimentStatus) else self.status
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentRecord':
        """
        從字典建立

        向後相容性處理：
        - 舊格式使用 'parameters' → 自動轉為 'params'
        - 字串 status → 轉為 ExperimentStatus Enum
        - 忽略未知欄位（如 'notes'）
        """
        data = data.copy()

        # 轉換 ISO 格式為 datetime
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])

        # 轉換字串 status 為 ExperimentStatus Enum（向後相容）
        if 'status' in data and isinstance(data['status'], str):
            # 支援大寫和小寫（舊 JSON 可能用大寫）
            status_str = data['status'].lower()
            data['status'] = ExperimentStatus(status_str)

        # 向後相容：舊格式使用 'parameters'，現在改用 'params'
        if 'parameters' in data and 'params' not in data.get('strategy', {}):
            # 將頂層 parameters 移到 strategy.params
            if 'strategy' not in data:
                data['strategy'] = {}
            data['strategy']['params'] = data.pop('parameters')

        # 過濾掉未知欄位（避免 __init__ 錯誤）
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}

        return cls(**filtered_data)

    # ========== 型別安全 Property 存取器 ==========

    @property
    def sharpe_ratio(self) -> float:
        """快速訪問 Sharpe Ratio"""
        return self.results.get('sharpe_ratio', 0.0)

    @property
    def total_return(self) -> float:
        """快速訪問總報酬率"""
        return self.results.get('total_return', 0.0)

    @property
    def max_drawdown(self) -> float:
        """快速訪問最大回撤"""
        return self.results.get('max_drawdown', 0.0)

    @property
    def grade(self) -> str:
        """快速訪問驗證等級"""
        return self.validation.get('grade', 'F')

    @property
    def strategy_name(self) -> str:
        """快速訪問策略名稱"""
        return self.strategy.get('name', '')

    @property
    def strategy_type(self) -> str:
        """快速訪問策略類型"""
        return self.strategy.get('type', '')

    @property
    def symbol(self) -> str:
        """快速訪問交易標的"""
        return self.config.get('symbol', '')

    @property
    def timeframe(self) -> str:
        """快速訪問時間框架"""
        return self.config.get('timeframe', '')

    @property
    def params(self) -> Dict[str, Any]:
        """快速訪問策略參數"""
        return self.strategy.get('params', {})

    @property
    def is_success(self) -> bool:
        """是否為成功實驗（驗證通過）"""
        return self.grade in [Grade.A.value, Grade.B.value]
