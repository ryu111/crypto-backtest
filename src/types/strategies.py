"""
策略相關型別定義

包含策略資訊、參數空間、統計資料等。
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union

from .enums import StrategyType, ParamType


@dataclass
class ParamSpace:
    """
    參數空間定義

    使用範例:
        space = ParamSpace(
            params={
                'fast_period': (5, 50, ParamType.INT),  # 或使用字串 'int'（向後相容）
                'slow_period': (20, 200, ParamType.INT),
                'rsi_period': (10, 30, ParamType.INT),
            },
            constraints=[
                lambda p: p['fast_period'] < p['slow_period']
            ]
        )
    """

    # 參數範圍 {param_name: (min, max, type)}
    # type 可以是 ParamType Enum 或字串（向後相容）
    params: Dict[str, Tuple[float, float, Union[ParamType, str]]]

    # 約束條件（lambda 函數列表）
    constraints: Optional[List[Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典（JSON 序列化）"""
        return {
            'params': {
                k: {
                    'min': v[0],
                    'max': v[1],
                    'type': v[2].value if isinstance(v[2], ParamType) else v[2]
                }
                for k, v in self.params.items()
            },
            # constraints 無法序列化，略過
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParamSpace':
        """從字典建立"""
        params = {}
        for k, v in data['params'].items():
            param_type = v['type']
            # 嘗試轉換為 ParamType Enum，失敗則保持字串
            if isinstance(param_type, str):
                try:
                    param_type = ParamType(param_type)
                except ValueError:
                    pass  # 保持原字串
            params[k] = (v['min'], v['max'], param_type)

        return cls(params=params, constraints=None)

    def sample_random(self, max_retries: int = 100) -> Dict[str, Any]:
        """
        隨機採樣參數

        Args:
            max_retries: 最大重試次數（避免無限遞迴）

        Returns:
            符合約束的參數字典

        Raises:
            ValueError: 超過最大重試次數仍無法滿足約束
        """
        import random

        for attempt in range(max_retries):
            params = {}
            for name, (min_val, max_val, param_type) in self.params.items():
                # 正規化 param_type（支援 Enum 或字串）
                type_str = param_type.value if isinstance(param_type, ParamType) else param_type

                if type_str == 'int':
                    params[name] = random.randint(int(min_val), int(max_val))
                elif type_str == 'float':
                    params[name] = random.uniform(min_val, max_val)
                elif type_str == 'log':
                    # 對數空間
                    import math
                    log_min = math.log(min_val)
                    log_max = math.log(max_val)
                    params[name] = math.exp(random.uniform(log_min, log_max))

            # 驗證約束
            if not self.constraints or all(c(params) for c in self.constraints):
                return params

        raise ValueError(f"無法在 {max_retries} 次嘗試內滿足約束條件")


@dataclass
class StrategyInfo:
    """
    策略資訊（對應 experiments.json 的 strategy 欄位）

    使用範例:
        info = StrategyInfo(
            name="trend_ma_cross",
            type=StrategyType.TREND,  # 使用 Enum
            version="1.0",
            params={'fast_period': 10, 'slow_period': 30}
        )
    """

    name: str  # trend_ma_cross, mean_rsi_bb
    type: Union[StrategyType, str]  # 使用 Enum，向後相容字串
    version: str = "1.0"
    params: Dict[str, Any] = field(default_factory=dict)

    # 策略描述（選填）
    description: Optional[str] = None
    param_space: Optional[ParamSpace] = None

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典（JSON 序列化）"""
        result = {
            'name': self.name,
            'type': self.type.value if isinstance(self.type, StrategyType) else self.type,
            'version': self.version,
            'params': self.params,
        }
        if self.description:
            result['description'] = self.description
        if self.param_space:
            result['param_space'] = self.param_space.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyInfo':
        """從字典建立"""
        data = data.copy()

        # 嘗試轉換 type 為 Enum，失敗則保持字串（向後相容）
        if 'type' in data and isinstance(data['type'], str):
            try:
                data['type'] = StrategyType(data['type'])
            except ValueError:
                # 保持原字串（例如舊資料或未定義的類型）
                pass

        # 處理嵌套的 dataclass
        if 'param_space' in data and isinstance(data['param_space'], dict):
            data['param_space'] = ParamSpace.from_dict(data['param_space'])

        return cls(**data)


@dataclass
class StrategyStats:
    """
    策略統計資料（整合自 interfaces.py 的 StrategyStatsData）

    追蹤策略的歷史表現，用於 Exploit/Explore 決策。

    使用範例:
        stats = StrategyStats(
            name="trend_ma_cross",
            attempts=10,
            successes=3,
            avg_sharpe=1.2,
            best_sharpe=2.1,
        )
    """

    name: str
    attempts: int = 0
    successes: int = 0  # 通過驗證的次數（A/B）

    # 績效統計
    avg_sharpe: float = 0.0
    best_sharpe: float = 0.0
    worst_sharpe: float = 0.0

    # 最佳參數
    best_params: Optional[Dict[str, Any]] = None
    last_params: Optional[Dict[str, Any]] = None  # 最近使用的參數（相容 StrategyStatsData）

    # 時間追蹤
    last_attempt: Optional[datetime] = None
    first_attempt: Optional[datetime] = None

    # UCB 評分（用於 Exploit/Explore）
    ucb_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典（JSON 序列化）"""
        data = asdict(self)
        # 轉換 datetime 為 ISO 格式
        if self.last_attempt:
            data['last_attempt'] = self.last_attempt.isoformat()
        if self.first_attempt:
            data['first_attempt'] = self.first_attempt.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyStats':
        """從字典建立"""
        data = data.copy()
        # 轉換 ISO 格式為 datetime
        if 'last_attempt' in data and isinstance(data['last_attempt'], str):
            data['last_attempt'] = datetime.fromisoformat(data['last_attempt'])
        if 'first_attempt' in data and isinstance(data['first_attempt'], str):
            data['first_attempt'] = datetime.fromisoformat(data['first_attempt'])
        return cls(**data)

    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts

    def update_from_experiment(
        self,
        sharpe: float,
        passed: bool,
        params: Dict[str, Any],
    ) -> None:
        """從實驗結果更新統計"""
        self.attempts += 1
        if passed:
            self.successes += 1

        # 更新績效統計（使用增量更新公式提高精度）
        if self.attempts == 1:
            self.avg_sharpe = sharpe
            self.best_sharpe = sharpe
            self.worst_sharpe = sharpe
            self.best_params = params
        else:
            # 增量平均值更新：avg += (new - avg) / n
            self.avg_sharpe += (sharpe - self.avg_sharpe) / self.attempts
            if sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                self.best_params = params
            if sharpe < self.worst_sharpe:
                self.worst_sharpe = sharpe

        # 更新最近使用的參數（相容 StrategyStatsData）
        self.last_params = params

        # 更新時間
        now = datetime.now()
        self.last_attempt = now
        if self.first_attempt is None:
            self.first_attempt = now

    def calculate_ucb(self, total_attempts: int, exploration_weight: float = 2.0) -> float:
        """
        計算 UCB (Upper Confidence Bound) 評分

        用於 Exploit/Explore 平衡：
        UCB = avg_reward + exploration_weight * sqrt(ln(total) / attempts)

        Args:
            total_attempts: 所有策略的總嘗試次數
            exploration_weight: 探索權重（越大越傾向探索）

        Returns:
            UCB 評分（越高越應該嘗試）
        """
        if self.attempts == 0:
            return float('inf')  # 未嘗試的策略優先

        import math
        exploitation = self.avg_sharpe
        exploration = exploration_weight * math.sqrt(
            math.log(total_attempts) / self.attempts
        )
        self.ucb_score = exploitation + exploration
        return self.ucb_score
