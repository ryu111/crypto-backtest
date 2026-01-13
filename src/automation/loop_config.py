"""
BacktestLoop 配置與結果類別

定義 BacktestLoop 系統的配置、執行模式、驗證階段和結果摘要類別。
參考：.claude/skills/AI自動化/SKILL.md
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Dict, List, Any, Optional


# ===== 列舉定義 =====

class SelectionMode(str, Enum):
    """策略選擇模式

    定義不同的策略選擇算法，用於決定下次迭代執行哪個策略。
    """

    EPSILON_GREEDY = "epsilon_greedy"    # ε-貪婪：平衡探索與利用
    UCB = "ucb"                          # Upper Confidence Bound：置信上界
    THOMPSON_SAMPLING = "thompson_sampling"  # Thompson Sampling：貝葉斯採樣
    ROUND_ROBIN = "round_robin"          # 輪詢：依序執行所有策略
    SINGLE = "single"                    # 單一策略：固定執行一個策略


class ValidationStage(IntEnum):
    """驗證階段

    對應 src/validation/stages.py 的五個驗證階段。
    使用 IntEnum 便於比較和排序。
    """

    BASIC = 1           # 基礎指標驗證（Sharpe > 1.0, MaxDD < 30%）
    STATISTICAL = 2     # 統計檢驗（t-test, normality）
    STABILITY = 3       # 穩健性測試（參數敏感度）
    WALK_FORWARD = 4    # Walk-Forward 分析
    MONTE_CARLO = 5     # Monte Carlo 模擬


# ===== 配置類別 =====

@dataclass
class BacktestLoopConfig:
    """BacktestLoop 系統配置

    定義完整的回測循環配置，包括策略選擇、驗證、效能等設定。

    範例:
        config = BacktestLoopConfig(
            strategies=['ma_cross', 'rsi', 'supertrend'],
            symbols=['BTCUSDT', 'ETHUSDT'],
            timeframes=['1h', '4h'],
            n_iterations=100,
            selection_mode='epsilon_greedy',
            validation_stages=[4, 5],  # 執行 WF + MC
            max_workers=8,
            use_gpu=True
        )
    """

    # ===== 策略配置 =====

    strategies: List[str] = field(default_factory=list)
    """執行的策略列表

    範例: ['ma_cross', 'rsi', 'supertrend', 'macd']
    空列表表示執行所有已註冊策略
    """

    symbols: List[str] = field(default_factory=lambda: ['BTCUSDT', 'ETHUSDT'])
    """交易標的列表

    範例: ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    """

    timeframes: List[str] = field(default_factory=lambda: ['1h'])
    """時間框架列表

    範例: ['5m', '15m', '1h', '4h', '1d']
    """

    # ===== 執行控制 =====

    n_iterations: int = 100
    """總迭代次數

    每次迭代執行一個策略的完整優化→驗證→記錄流程
    """

    selection_mode: str = 'epsilon_greedy'
    """策略選擇模式

    可選值: 'epsilon_greedy', 'ucb', 'thompson_sampling', 'round_robin', 'single'
    """

    epsilon: float = 0.2
    """ε-貪婪的探索率（僅當 selection_mode='epsilon_greedy'）

    0.2 表示 20% 機率隨機探索，80% 利用最佳策略
    """

    ucb_c: float = 2.0
    """UCB 的探索係數（僅當 selection_mode='ucb'）

    較大值更傾向探索，較小值更傾向利用
    """

    # ===== 驗證配置 =====

    validation_stages: List[int] = field(default_factory=lambda: [4, 5])
    """執行的驗證階段列表

    可選值: [1, 2, 3, 4, 5] 對應五個驗證階段
    預設 [4, 5] 執行 Walk-Forward + Monte Carlo
    """

    min_sharpe: float = 1.0
    """最低 Sharpe Ratio 要求

    低於此值的策略會被標記為失敗
    """

    max_drawdown: float = 0.30
    """最大回撤上限

    超過此值的策略會被標記為失敗（0.30 = 30%）
    """

    min_trades: int = 30
    """最少交易次數

    低於此值的策略會被標記為樣本不足
    """

    # ===== 效能配置 =====

    max_workers: int = 8
    """CPU 並行工作數

    控制 ProcessPoolExecutor 的最大 worker 數量
    """

    use_gpu: bool = True
    """是否啟用 GPU 加速

    啟用時使用 GPUBatchOptimizer 進行參數優化
    """

    gpu_batch_size: int = 50
    """GPU 批次大小

    一次處理的參數組合數量，越大效能越好但記憶體需求越高
    """

    # ===== 優化配置 =====

    n_trials: int = 100
    """參數優化的試驗次數

    使用 Bayesian 優化時的採樣次數
    """

    timeout_per_iteration: int = 600
    """每次迭代的超時時間（秒）

    超過此時間的迭代會被終止
    """

    # ===== 交易配置 =====

    leverage: int = 5
    """槓桿倍數

    永續合約的槓桿設定
    """

    initial_capital: float = 10000.0
    """初始資金

    回測時的起始資金量（USD）
    """

    maker_fee: float = 0.0002
    """Maker 手續費率

    掛單成交的手續費率（0.0002 = 0.02%）
    """

    taker_fee: float = 0.0004
    """Taker 手續費率

    吃單成交的手續費率（0.0004 = 0.04%）
    """

    def validate(self) -> None:
        """驗證配置有效性

        Raises:
            ValueError: 配置無效時拋出異常
        """
        # 檢查策略列表（允許空列表，表示使用所有策略）
        if not isinstance(self.strategies, list):
            raise ValueError("strategies 必須是列表")

        # 檢查標的列表
        if not self.symbols:
            raise ValueError("symbols 不能為空")

        # 檢查時間框架列表
        if not self.timeframes:
            raise ValueError("timeframes 不能為空")

        # 檢查迭代次數
        if self.n_iterations <= 0:
            raise ValueError("n_iterations 必須大於 0")

        # 檢查選擇模式
        try:
            SelectionMode(self.selection_mode)
        except ValueError:
            valid_modes = [m.value for m in SelectionMode]
            raise ValueError(
                f"selection_mode 必須是 {valid_modes} 之一，"
                f"got: {self.selection_mode}"
            )

        # 檢查 epsilon
        if not 0 <= self.epsilon <= 1:
            raise ValueError("epsilon 必須在 [0, 1] 範圍內")

        # 檢查 UCB 係數
        if self.ucb_c < 0:
            raise ValueError("ucb_c 必須大於等於 0")

        # 檢查驗證階段
        valid_stages = {1, 2, 3, 4, 5}
        invalid_stages = set(self.validation_stages) - valid_stages
        if invalid_stages:
            raise ValueError(
                f"validation_stages 包含無效階段: {invalid_stages}, "
                f"有效階段為: {valid_stages}"
            )

        # 檢查閾值
        if self.min_sharpe < 0:
            raise ValueError("min_sharpe 必須大於等於 0")

        if not 0 < self.max_drawdown <= 1:
            raise ValueError("max_drawdown 必須在 (0, 1] 範圍內")

        if self.min_trades < 0:
            raise ValueError("min_trades 必須大於等於 0")

        # 檢查效能設定
        if self.max_workers <= 0:
            raise ValueError("max_workers 必須大於 0")

        if self.gpu_batch_size <= 0:
            raise ValueError("gpu_batch_size 必須大於 0")

        # 檢查優化設定
        if self.n_trials <= 0:
            raise ValueError("n_trials 必須大於 0")

        if self.timeout_per_iteration <= 0:
            raise ValueError("timeout_per_iteration 必須大於 0")

        # 檢查交易設定
        if self.leverage <= 0:
            raise ValueError("leverage 必須大於 0")

        if self.initial_capital <= 0:
            raise ValueError("initial_capital 必須大於 0")

        if not 0 <= self.maker_fee <= 1:
            raise ValueError("maker_fee 必須在 [0, 1] 範圍內")

        if not 0 <= self.taker_fee <= 1:
            raise ValueError("taker_fee 必須在 [0, 1] 範圍內")


# ===== 結果類別 =====

@dataclass
class IterationSummary:
    """單次迭代摘要

    記錄單次迭代（一個策略的完整優化→驗證流程）的結果。

    範例:
        summary = IterationSummary(
            iteration=1,
            strategy_name='ma_cross',
            symbol='BTCUSDT',
            timeframe='1h',
            best_params={'fast_period': 10, 'slow_period': 30},
            sharpe_ratio=2.3,
            total_return=0.45,
            max_drawdown=0.12,
            validation_grade='A',
            wf_sharpe=2.1,
            passed=True,
            duration_seconds=45.2,
            timestamp=datetime.now()
        )
    """

    # ===== 基本資訊 =====

    iteration: int
    """迭代編號（從 1 開始）"""

    strategy_name: str
    """策略名稱（如 'ma_cross'）"""

    symbol: str
    """交易標的（如 'BTCUSDT'）"""

    timeframe: str
    """時間框架（如 '1h'）"""

    # ===== 參數與績效 =====

    best_params: Dict[str, Any]
    """最佳參數組合

    範例: {'fast_period': 10, 'slow_period': 30, 'atr_multiplier': 2.0}
    """

    sharpe_ratio: float
    """Sharpe Ratio（越高越好）"""

    total_return: float
    """總報酬率（小數形式，0.45 = 45%）"""

    max_drawdown: float
    """最大回撤（小數形式，0.12 = 12%）"""

    # ===== 驗證結果 =====

    validation_grade: str
    """驗證等級

    可能值: 'A', 'B', 'C', 'D', 'F'
    A/B 表示通過，C 表示邊緣，D/F 表示失敗
    """

    # ===== 執行資訊 =====

    duration_seconds: float
    """執行時間（秒）"""

    timestamp: datetime
    """完成時間"""

    # ===== 有預設值的欄位 =====

    wf_sharpe: Optional[float] = None
    """Walk-Forward Sharpe Ratio（若執行階段 4）

    樣本外績效，用於判斷是否過擬合
    """

    mc_p5: Optional[float] = None
    """Monte Carlo P5 值（若執行階段 5）

    5% 分位數的 Sharpe，用於判斷穩健性
    """

    passed: bool = False
    """是否通過驗證

    True 表示通過所有驗證階段，False 表示失敗
    """

    experiment_id: Optional[str] = None
    """對應的實驗 ID（如 'exp_20260112_120000'）

    用於連結到 learning/experiments.json 的記錄
    """

    error: Optional[str] = None
    """錯誤訊息（若失敗）

    None 表示成功，非 None 表示執行過程中發生錯誤
    """


@dataclass
class LoopResult:
    """Loop 完整執行結果

    彙總所有迭代的結果，提供統計分析和摘要報告。

    範例:
        result = LoopResult(
            iterations_completed=100,
            total_iterations=100,
            best_strategies=[...],
            failed_strategies=[...],
            experiment_ids=['exp_...', 'exp_...'],
            duration_seconds=3600.0,
            avg_sharpe=1.5,
            best_sharpe=2.8,
            avg_wf_sharpe=1.3,
            pass_rate=0.65
        )

        print(result.summary())
    """

    # ===== 執行統計 =====

    iterations_completed: int
    """完成的迭代次數"""

    total_iterations: int
    """總迭代次數"""

    best_strategies: List[IterationSummary] = field(default_factory=list)
    """表現最佳的策略列表（按 Sharpe 排序）

    通常取前 10 名
    """

    failed_strategies: List[IterationSummary] = field(default_factory=list)
    """失敗的策略列表（未通過驗證）"""

    experiment_ids: List[str] = field(default_factory=list)
    """所有實驗 ID 列表

    用於後續查詢和分析
    """

    duration_seconds: float = 0.0
    """總執行時間（秒）"""

    # ===== 績效統計 =====

    avg_sharpe: float = 0.0
    """平均 Sharpe Ratio（所有通過的策略）"""

    best_sharpe: float = 0.0
    """最佳 Sharpe Ratio"""

    avg_wf_sharpe: float = 0.0
    """平均 Walk-Forward Sharpe（若執行階段 4）"""

    pass_rate: float = 0.0
    """通過率（通過數 / 總數）

    範例: 0.65 = 65% 的策略通過驗證
    """

    # ===== 策略分布 =====

    strategy_counts: Dict[str, int] = field(default_factory=dict)
    """各策略執行次數統計

    範例: {'ma_cross': 25, 'rsi': 30, 'supertrend': 20}
    """

    strategy_win_rates: Dict[str, float] = field(default_factory=dict)
    """各策略通過率

    範例: {'ma_cross': 0.8, 'rsi': 0.6, 'supertrend': 0.7}
    """

    def summary(self) -> str:
        """生成人類可讀的摘要報告

        Returns:
            str: 多行摘要文字

        範例:
            ================================================================
            BacktestLoop 執行摘要
            ================================================================
            完成迭代: 100 / 100 (100.0%)
            執行時間: 60.0 分鐘
            通過率: 65.0% (65 / 100)

            績效統計:
            ----------------------------------------------------------------
            平均 Sharpe: 1.50
            最佳 Sharpe: 2.80 (ma_cross @ BTCUSDT 1h)
            平均 WF Sharpe: 1.30

            策略統計:
            ----------------------------------------------------------------
            ma_cross: 25 次執行, 80.0% 通過率
            rsi: 30 次執行, 60.0% 通過率
            supertrend: 20 次執行, 70.0% 通過率
            ================================================================
        """
        lines = [
            "",
            "=" * 70,
            "BacktestLoop 執行摘要",
            "=" * 70,
        ]

        # 執行統計
        completion_rate = (
            self.iterations_completed / self.total_iterations * 100
            if self.total_iterations > 0 else 0
        )

        lines.extend([
            f"完成迭代: {self.iterations_completed} / {self.total_iterations} "
            f"({completion_rate:.1f}%)",
            f"執行時間: {self.duration_seconds / 60:.1f} 分鐘",
            f"通過率: {self.pass_rate * 100:.1f}% "
            f"({len(self.best_strategies)} / {self.iterations_completed})",
            "",
        ])

        # 績效統計
        lines.extend([
            "績效統計:",
            "-" * 70,
            f"平均 Sharpe: {self.avg_sharpe:.2f}",
        ])

        # 最佳策略
        if self.best_strategies:
            best = self.best_strategies[0]
            lines.append(
                f"最佳 Sharpe: {self.best_sharpe:.2f} "
                f"({best.strategy_name} @ {best.symbol} {best.timeframe})"
            )
        else:
            lines.append("最佳 Sharpe: N/A")

        if self.avg_wf_sharpe > 0:
            lines.append(f"平均 WF Sharpe: {self.avg_wf_sharpe:.2f}")

        lines.append("")

        # 策略統計
        if self.strategy_counts:
            lines.extend([
                "策略統計:",
                "-" * 70,
            ])

            for strategy_name in sorted(self.strategy_counts.keys()):
                count = self.strategy_counts[strategy_name]
                win_rate = self.strategy_win_rates.get(strategy_name, 0)

                lines.append(
                    f"{strategy_name}: {count} 次執行, "
                    f"{win_rate * 100:.1f}% 通過率"
                )

            lines.append("")

        # 失敗統計
        if self.failed_strategies:
            lines.extend([
                f"失敗策略: {len(self.failed_strategies)} 個",
                ""
            ])

        lines.append("=" * 70)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典（JSON 序列化）

        Returns:
            Dict: 可序列化的字典

        範例:
            {
                'iterations_completed': 100,
                'total_iterations': 100,
                'duration_seconds': 3600.0,
                'avg_sharpe': 1.5,
                'best_sharpe': 2.8,
                'pass_rate': 0.65,
                'best_strategies': [
                    {
                        'iteration': 1,
                        'strategy_name': 'ma_cross',
                        'sharpe_ratio': 2.8,
                        ...
                    }
                ],
                ...
            }
        """
        return {
            'iterations_completed': self.iterations_completed,
            'total_iterations': self.total_iterations,
            'duration_seconds': self.duration_seconds,
            'avg_sharpe': self.avg_sharpe,
            'best_sharpe': self.best_sharpe,
            'avg_wf_sharpe': self.avg_wf_sharpe,
            'pass_rate': self.pass_rate,
            'best_strategies': [
                {
                    'iteration': s.iteration,
                    'strategy_name': s.strategy_name,
                    'symbol': s.symbol,
                    'timeframe': s.timeframe,
                    'best_params': s.best_params,
                    'sharpe_ratio': s.sharpe_ratio,
                    'total_return': s.total_return,
                    'max_drawdown': s.max_drawdown,
                    'validation_grade': s.validation_grade,
                    'wf_sharpe': s.wf_sharpe,
                    'mc_p5': s.mc_p5,
                    'passed': s.passed,
                    'duration_seconds': s.duration_seconds,
                    'timestamp': s.timestamp.isoformat(),
                    'experiment_id': s.experiment_id,
                }
                for s in self.best_strategies
            ],
            'failed_strategies': [
                {
                    'iteration': s.iteration,
                    'strategy_name': s.strategy_name,
                    'symbol': s.symbol,
                    'timeframe': s.timeframe,
                    'error': s.error,
                }
                for s in self.failed_strategies
            ],
            'experiment_ids': self.experiment_ids,
            'strategy_counts': self.strategy_counts,
            'strategy_win_rates': self.strategy_win_rates,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LoopResult':
        """從字典建立（反序列化）

        Args:
            data: 字典數據

        Returns:
            LoopResult: 結果物件
        """
        # 重建 IterationSummary
        best_strategies = [
            IterationSummary(
                iteration=s['iteration'],
                strategy_name=s['strategy_name'],
                symbol=s['symbol'],
                timeframe=s['timeframe'],
                best_params=s['best_params'],
                sharpe_ratio=s['sharpe_ratio'],
                total_return=s['total_return'],
                max_drawdown=s['max_drawdown'],
                validation_grade=s['validation_grade'],
                wf_sharpe=s.get('wf_sharpe'),
                mc_p5=s.get('mc_p5'),
                passed=s['passed'],
                duration_seconds=s['duration_seconds'],
                timestamp=datetime.fromisoformat(s['timestamp']),
                experiment_id=s.get('experiment_id'),
            )
            for s in data.get('best_strategies', [])
        ]

        failed_strategies = [
            IterationSummary(
                iteration=s['iteration'],
                strategy_name=s['strategy_name'],
                symbol=s['symbol'],
                timeframe=s['timeframe'],
                best_params={},
                sharpe_ratio=0.0,
                total_return=0.0,
                max_drawdown=1.0,
                validation_grade='F',
                passed=False,
                duration_seconds=0.0,
                timestamp=datetime.now(),
                error=s.get('error'),
            )
            for s in data.get('failed_strategies', [])
        ]

        return cls(
            iterations_completed=data['iterations_completed'],
            total_iterations=data['total_iterations'],
            best_strategies=best_strategies,
            failed_strategies=failed_strategies,
            experiment_ids=data.get('experiment_ids', []),
            duration_seconds=data['duration_seconds'],
            avg_sharpe=data['avg_sharpe'],
            best_sharpe=data['best_sharpe'],
            avg_wf_sharpe=data.get('avg_wf_sharpe', 0.0),
            pass_rate=data['pass_rate'],
            strategy_counts=data.get('strategy_counts', {}),
            strategy_win_rates=data.get('strategy_win_rates', {}),
        )


# ===== 便利函數 =====

def create_default_config() -> BacktestLoopConfig:
    """建立預設配置

    Returns:
        BacktestLoopConfig: 預設配置物件

    範例:
        config = create_default_config()
        config.strategies = ['ma_cross', 'rsi']
        config.n_iterations = 50
    """
    return BacktestLoopConfig()


def create_quick_config(
    strategies: Optional[List[str]] = None,
    n_iterations: int = 50,
    use_gpu: bool = False
) -> BacktestLoopConfig:
    """建立快速測試配置

    Args:
        strategies: 策略列表（None 則使用所有策略）
        n_iterations: 迭代次數
        use_gpu: 是否使用 GPU

    Returns:
        BacktestLoopConfig: 配置物件

    範例:
        config = create_quick_config(
            strategies=['ma_cross'],
            n_iterations=10,
            use_gpu=False
        )
    """
    return BacktestLoopConfig(
        strategies=strategies or [],
        symbols=['BTCUSDT'],
        timeframes=['1h'],
        n_iterations=n_iterations,
        selection_mode='round_robin',
        validation_stages=[1, 4],  # 只執行基礎 + WF
        max_workers=4,
        use_gpu=use_gpu,
        gpu_batch_size=20,
        n_trials=50,
        timeout_per_iteration=300,
    )


def create_production_config(
    strategies: Optional[List[str]] = None,
    n_iterations: int = 100
) -> BacktestLoopConfig:
    """建立生產級配置

    Args:
        strategies: 策略列表（None 則使用所有策略）
        n_iterations: 迭代次數

    Returns:
        BacktestLoopConfig: 配置物件

    範例:
        config = create_production_config(
            strategies=['ma_cross', 'rsi', 'supertrend'],
            n_iterations=100
        )
    """
    return BacktestLoopConfig(
        strategies=strategies or [],
        symbols=['BTCUSDT', 'ETHUSDT'],
        timeframes=['1h', '4h', '1d'],
        n_iterations=n_iterations,
        selection_mode='ucb',
        validation_stages=[1, 2, 3, 4, 5],  # 全部五個階段
        min_sharpe=1.5,
        max_drawdown=0.25,
        min_trades=50,
        max_workers=8,
        use_gpu=True,
        gpu_batch_size=50,
        n_trials=100,
        timeout_per_iteration=600,
    )
