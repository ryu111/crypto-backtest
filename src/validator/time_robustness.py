"""
時間段穩健性測試模組

將回測資料分成多個時間段，分別測試策略表現，
評估策略在不同市場環境下的穩定性。

參考：
- .claude/skills/策略驗證/SKILL.md

使用範例：
    test = TimeRobustnessTest()

    # 執行測試
    result = test.run(
        strategy=my_strategy,
        data=ohlcv_data,
        params={'period': 14},
        n_segments=4
    )

    print(result.summary())
    print(f"一致性分數: {result.consistency_score:.2%}")
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SegmentResult:
    """單一時間段測試結果"""
    segment_id: int
    start_date: datetime
    end_date: datetime
    n_bars: int

    # 績效指標
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int

    # 是否通過
    is_profitable: bool
    is_acceptable: bool  # 符合最低標準

    def to_dict(self) -> dict:
        """轉為字典"""
        return {
            'segment_id': self.segment_id,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'n_bars': self.n_bars,
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'n_trades': self.n_trades,
            'is_profitable': self.is_profitable,
            'is_acceptable': self.is_acceptable
        }


@dataclass
class TimeRobustnessResult:
    """時間段穩健性測試結果"""
    passed: bool
    consistency_score: float  # 0-1, 一致性分數
    segment_results: List[SegmentResult]

    # 統計數據
    n_segments: int
    profitable_segments: int
    acceptable_segments: int

    # Sharpe 統計
    sharpe_mean: float
    sharpe_std: float
    sharpe_min: float
    sharpe_max: float

    # 報酬統計
    return_mean: float
    return_std: float
    return_min: float
    return_max: float

    # 驗證詳情
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """生成摘要報告"""
        status = "PASS" if self.passed else "FAIL"
        lines = [
            "",
            "=" * 60,
            f"時間段穩健性測試: {status}",
            "=" * 60,
            "",
            f"一致性分數: {self.consistency_score:.1%}",
            f"總時間段數: {self.n_segments}",
            f"獲利時間段: {self.profitable_segments} ({self.profitable_segments/self.n_segments*100:.0f}%)",
            f"合格時間段: {self.acceptable_segments} ({self.acceptable_segments/self.n_segments*100:.0f}%)",
            "",
            "Sharpe 統計:",
            f"  平均: {self.sharpe_mean:.2f}",
            f"  標準差: {self.sharpe_std:.2f}",
            f"  範圍: [{self.sharpe_min:.2f}, {self.sharpe_max:.2f}]",
            "",
            "報酬統計:",
            f"  平均: {self.return_mean:.2%}",
            f"  標準差: {self.return_std:.2%}",
            f"  範圍: [{self.return_min:.2%}, {self.return_max:.2%}]",
            "",
            "各時間段詳情:",
            "-" * 60
        ]

        for seg in self.segment_results:
            status_icon = "" if seg.is_acceptable else ""
            lines.append(
                f"  段 {seg.segment_id}: {seg.start_date.strftime('%Y-%m-%d')} ~ "
                f"{seg.end_date.strftime('%Y-%m-%d')} | "
                f"Return: {seg.total_return:+.2%} | "
                f"Sharpe: {seg.sharpe_ratio:.2f} | "
                f"MaxDD: {seg.max_drawdown:.2%} {status_icon}"
            )

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """轉為字典"""
        return {
            'passed': self.passed,
            'consistency_score': self.consistency_score,
            'n_segments': self.n_segments,
            'profitable_segments': self.profitable_segments,
            'acceptable_segments': self.acceptable_segments,
            'sharpe_mean': self.sharpe_mean,
            'sharpe_std': self.sharpe_std,
            'sharpe_min': self.sharpe_min,
            'sharpe_max': self.sharpe_max,
            'return_mean': self.return_mean,
            'return_std': self.return_std,
            'return_min': self.return_min,
            'return_max': self.return_max,
            'segment_results': [s.to_dict() for s in self.segment_results]
        }


class TimeRobustnessTest:
    """
    時間段穩健性測試

    將回測期間分成多個時間段，分別測試策略表現，
    評估策略在不同時期的穩定性。

    一致性標準：
    - 所有時間段都獲利
    - Sharpe 變異係數 < 50%
    - 無單段重大虧損 (< -15%)

    使用範例：
        test = TimeRobustnessTest()

        result = test.run(
            strategy=my_strategy,
            data=ohlcv_data,
            params={'period': 14},
            n_segments=4
        )

        if result.passed:
            print("策略通過時間段穩健性測試")
    """

    # 預設一致性標準
    MIN_PROFITABLE_RATIO = 1.0   # 所有時間段必須獲利
    MAX_SHARPE_CV = 0.50         # Sharpe 變異係數上限
    MAX_SINGLE_LOSS = -0.15      # 單段最大虧損上限
    MIN_ACCEPTABLE_SHARPE = 0.5  # 最低可接受 Sharpe

    def __init__(
        self,
        min_profitable_ratio: float = None,
        max_sharpe_cv: float = None,
        max_single_loss: float = None,
        min_acceptable_sharpe: float = None
    ):
        """初始化時間段穩健性測試

        Args:
            min_profitable_ratio: 最低獲利時間段比例（預設 1.0 = 100%）
            max_sharpe_cv: Sharpe 變異係數上限（預設 0.5 = 50%）
            max_single_loss: 單段最大虧損上限（預設 -0.15 = -15%）
            min_acceptable_sharpe: 最低可接受 Sharpe（預設 0.5）
        """
        self.min_profitable_ratio = min_profitable_ratio or self.MIN_PROFITABLE_RATIO
        self.max_sharpe_cv = max_sharpe_cv or self.MAX_SHARPE_CV
        self.max_single_loss = max_single_loss or self.MAX_SINGLE_LOSS
        self.min_acceptable_sharpe = min_acceptable_sharpe or self.MIN_ACCEPTABLE_SHARPE

    def split_by_segments(
        self,
        data: pd.DataFrame,
        n_segments: int = 4
    ) -> List[pd.DataFrame]:
        """將資料分成多個時間段

        Args:
            data: OHLCV DataFrame（需有 datetime index）
            n_segments: 時間段數量

        Returns:
            List[pd.DataFrame]: 時間段資料列表
        """
        if len(data) < n_segments * 10:
            raise ValueError(f"資料量不足: {len(data)} bars，至少需要 {n_segments * 10} bars")

        # 計算每段的大小
        segment_size = len(data) // n_segments

        segments = []
        for i in range(n_segments):
            start_idx = i * segment_size
            # 最後一段包含所有剩餘資料
            end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(data)
            segment = data.iloc[start_idx:end_idx].copy()
            segments.append(segment)

        return segments

    def test_each_segment(
        self,
        strategy: Any,
        params: Dict[str, Any],
        segments: List[pd.DataFrame],
        backtest_func: Optional[Callable] = None
    ) -> List[SegmentResult]:
        """測試每個時間段

        Args:
            strategy: 策略物件或類別
            params: 策略參數
            segments: 時間段資料列表
            backtest_func: 自訂回測函數（可選）

        Returns:
            List[SegmentResult]: 各時間段測試結果
        """
        results = []

        for i, segment in enumerate(segments):
            try:
                # 執行回測
                if backtest_func:
                    backtest_result = backtest_func(strategy, segment, params)
                else:
                    backtest_result = self._default_backtest(strategy, segment, params)

                # 建立結果
                result = SegmentResult(
                    segment_id=i + 1,
                    start_date=segment.index[0].to_pydatetime() if hasattr(segment.index[0], 'to_pydatetime') else segment.index[0],
                    end_date=segment.index[-1].to_pydatetime() if hasattr(segment.index[-1], 'to_pydatetime') else segment.index[-1],
                    n_bars=len(segment),
                    total_return=backtest_result.get('total_return', 0.0),
                    sharpe_ratio=backtest_result.get('sharpe_ratio', 0.0),
                    max_drawdown=backtest_result.get('max_drawdown', 0.0),
                    win_rate=backtest_result.get('win_rate', 0.0),
                    n_trades=backtest_result.get('n_trades', 0),
                    is_profitable=backtest_result.get('total_return', 0.0) > 0,
                    is_acceptable=backtest_result.get('sharpe_ratio', 0.0) >= self.min_acceptable_sharpe
                )
                results.append(result)

            except Exception as e:
                logger.error(f"測試時間段 {i+1} 失敗: {e}")
                # 建立失敗結果
                results.append(SegmentResult(
                    segment_id=i + 1,
                    start_date=segment.index[0].to_pydatetime() if hasattr(segment.index[0], 'to_pydatetime') else segment.index[0],
                    end_date=segment.index[-1].to_pydatetime() if hasattr(segment.index[-1], 'to_pydatetime') else segment.index[-1],
                    n_bars=len(segment),
                    total_return=-1.0,  # 標記為失敗
                    sharpe_ratio=-999.0,
                    max_drawdown=1.0,
                    win_rate=0.0,
                    n_trades=0,
                    is_profitable=False,
                    is_acceptable=False
                ))

        return results

    def _default_backtest(
        self,
        strategy: Any,
        data: pd.DataFrame,
        params: Dict[str, Any]
    ) -> Dict[str, float]:
        """預設回測函數

        嘗試使用 BacktestEngine 執行回測。
        """
        try:
            from ..backtester.engine import BacktestEngine, BacktestConfig

            # 設定策略參數
            if hasattr(strategy, 'set_params'):
                strategy.set_params(**params)
            elif hasattr(strategy, 'params'):
                for k, v in params.items():
                    if hasattr(strategy.params, k):
                        setattr(strategy.params, k, v)

            # 執行回測
            config = BacktestConfig(
                initial_capital=100000,
                leverage=1
            )
            engine = BacktestEngine(config)
            result = engine.run(strategy, data)

            return {
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'n_trades': result.n_trades
            }

        except Exception as e:
            logger.warning(f"預設回測失敗: {e}，返回空結果")
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'n_trades': 0
            }

    def calculate_consistency_score(
        self,
        segment_results: List[SegmentResult]
    ) -> float:
        """計算一致性分數

        基於三個指標計算綜合分數：
        1. 獲利比例 (40%)
        2. Sharpe 穩定性 (40%)
        3. 最大虧損控制 (20%)

        Args:
            segment_results: 各時間段測試結果

        Returns:
            float: 0-1 之間的一致性分數
        """
        if not segment_results:
            return 0.0

        returns = [r.total_return for r in segment_results]
        sharpes = [r.sharpe_ratio for r in segment_results if r.sharpe_ratio > -900]

        # 1. 獲利比例分數 (0-1)
        profitable_ratio = sum(1 for r in returns if r > 0) / len(returns)
        profit_score = profitable_ratio

        # 2. Sharpe 穩定性分數 (0-1)
        if len(sharpes) > 1 and np.mean(sharpes) != 0:
            sharpe_cv = np.std(sharpes) / abs(np.mean(sharpes))
            # 變異係數越低越好，上限為 1.0
            stability_score = max(0, 1 - min(sharpe_cv / self.max_sharpe_cv, 1))
        else:
            stability_score = 0.5  # 無法計算時給中等分數

        # 3. 最大虧損控制分數 (0-1)
        min_return = min(returns)
        if min_return >= 0:
            loss_score = 1.0
        elif min_return >= self.max_single_loss:
            # 線性映射: max_single_loss -> 0.5, 0 -> 1.0
            loss_score = 0.5 + 0.5 * (min_return - self.max_single_loss) / (-self.max_single_loss)
        else:
            # 超過上限，給低分
            loss_score = max(0, 0.5 * (1 + min_return / self.max_single_loss))

        # 加權平均
        consistency_score = (
            0.40 * profit_score +
            0.40 * stability_score +
            0.20 * loss_score
        )

        return consistency_score

    def run(
        self,
        strategy: Any,
        data: pd.DataFrame,
        params: Dict[str, Any],
        n_segments: int = 4,
        backtest_func: Optional[Callable] = None
    ) -> TimeRobustnessResult:
        """執行時間段穩健性測試

        Args:
            strategy: 策略物件或類別
            data: OHLCV DataFrame
            params: 策略參數
            n_segments: 時間段數量（預設 4）
            backtest_func: 自訂回測函數（可選）

        Returns:
            TimeRobustnessResult: 測試結果
        """
        logger.info(f"開始時間段穩健性測試: {n_segments} 個時間段")

        # 1. 分割時間段
        segments = self.split_by_segments(data, n_segments)

        # 2. 測試每個時間段
        segment_results = self.test_each_segment(
            strategy, params, segments, backtest_func
        )

        # 3. 計算統計數據
        returns = [r.total_return for r in segment_results]
        sharpes = [r.sharpe_ratio for r in segment_results if r.sharpe_ratio > -900]

        # 4. 計算一致性分數
        consistency_score = self.calculate_consistency_score(segment_results)

        # 5. 判斷是否通過
        profitable_count = sum(1 for r in returns if r > 0)
        acceptable_count = sum(1 for r in segment_results if r.is_acceptable)

        # 通過條件：
        # - 獲利比例 >= 設定值
        # - Sharpe 變異係數 <= 設定值
        # - 無單段重大虧損
        profitable_ratio = profitable_count / len(returns)
        sharpe_cv = np.std(sharpes) / abs(np.mean(sharpes)) if sharpes and np.mean(sharpes) != 0 else float('inf')
        min_return = min(returns)

        passed = (
            profitable_ratio >= self.min_profitable_ratio and
            sharpe_cv <= self.max_sharpe_cv and
            min_return >= self.max_single_loss
        )

        result = TimeRobustnessResult(
            passed=passed,
            consistency_score=consistency_score,
            segment_results=segment_results,
            n_segments=n_segments,
            profitable_segments=profitable_count,
            acceptable_segments=acceptable_count,
            sharpe_mean=np.mean(sharpes) if sharpes else 0.0,
            sharpe_std=np.std(sharpes) if sharpes else 0.0,
            sharpe_min=min(sharpes) if sharpes else 0.0,
            sharpe_max=max(sharpes) if sharpes else 0.0,
            return_mean=np.mean(returns),
            return_std=np.std(returns),
            return_min=min(returns),
            return_max=max(returns),
            details={
                'profitable_ratio': profitable_ratio,
                'sharpe_cv': sharpe_cv,
                'min_return': min_return,
                'thresholds': {
                    'min_profitable_ratio': self.min_profitable_ratio,
                    'max_sharpe_cv': self.max_sharpe_cv,
                    'max_single_loss': self.max_single_loss
                }
            }
        )

        status = "PASS" if passed else "FAIL"
        logger.info(f"時間段穩健性測試完成: {status}, 一致性分數: {consistency_score:.2%}")

        return result


def test_time_robustness(
    strategy: Any,
    data: pd.DataFrame,
    params: Dict[str, Any],
    n_segments: int = 4,
    backtest_func: Optional[Callable] = None
) -> TimeRobustnessResult:
    """便捷函數：執行時間段穩健性測試

    Args:
        strategy: 策略物件
        data: OHLCV DataFrame
        params: 策略參數
        n_segments: 時間段數量
        backtest_func: 自訂回測函數

    Returns:
        TimeRobustnessResult: 測試結果
    """
    test = TimeRobustnessTest()
    return test.run(strategy, data, params, n_segments, backtest_func)
