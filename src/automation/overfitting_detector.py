"""過擬合偵測模組

實作 PBO、參數敏感度、交易筆數檢查等過擬合指標。

參考：/.claude/skills/參數優化/references/overfitting-detection.md
"""

from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.stats import rankdata


class RiskLevel(str, Enum):
    """過擬合風險等級"""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class OverfitThresholds:
    """過擬合門檻常數"""

    # 交易筆數
    MIN_TRADES_VALID = 30  # 統計有效
    MIN_TRADES_CONFIDENT = 50  # 統計信心

    # IS/OOS 比
    IS_OOS_RATIO_HIGH = 2.0  # 高風險
    IS_OOS_RATIO_MEDIUM = 1.5  # 中風險

    # 參數敏感度
    PARAM_SENSITIVITY_HIGH = 0.30  # 高風險
    PARAM_SENSITIVITY_MEDIUM = 0.20  # 中風險

    # PBO
    PBO_HIGH = 0.50  # 高風險
    PBO_MEDIUM = 0.25  # 中風險

    # 風險判定
    HIGH_RISK_COUNT_THRESHOLD = 2  # >= 2 個高風險指標 → HIGH


@dataclass
class OverfitMetrics:
    """過擬合指標"""

    pbo: float = 0.0  # Probability of Backtest Overfitting (0-1)
    is_oos_ratio: float = 1.0  # IS/OOS 績效比
    param_sensitivity: float = 0.0  # 參數敏感度 (0-1)
    trade_count: int = 0  # 交易筆數
    degradation: float = 0.0  # IS 到 OOS 的績效衰退

    @property
    def overall_risk(self) -> RiskLevel:
        """綜合風險等級

        判斷標準：
        - HIGH: >= 2 個高風險指標
        - MEDIUM: >= 1 個高風險或 >= 2 個中風險
        - LOW: 其他
        """
        risks = []
        T = OverfitThresholds

        # IS/OOS 比
        if self.is_oos_ratio > T.IS_OOS_RATIO_HIGH:
            risks.append(("is_oos_ratio", RiskLevel.HIGH))
        elif self.is_oos_ratio > T.IS_OOS_RATIO_MEDIUM:
            risks.append(("is_oos_ratio", RiskLevel.MEDIUM))

        # 交易次數
        if self.trade_count < T.MIN_TRADES_VALID:
            risks.append(("trade_count", RiskLevel.HIGH))
        elif self.trade_count < T.MIN_TRADES_CONFIDENT:
            risks.append(("trade_count", RiskLevel.MEDIUM))

        # 參數敏感度
        if self.param_sensitivity > T.PARAM_SENSITIVITY_HIGH:
            risks.append(("param_sensitivity", RiskLevel.HIGH))
        elif self.param_sensitivity > T.PARAM_SENSITIVITY_MEDIUM:
            risks.append(("param_sensitivity", RiskLevel.MEDIUM))

        # PBO
        if self.pbo > T.PBO_HIGH:
            risks.append(("pbo", RiskLevel.HIGH))
        elif self.pbo > T.PBO_MEDIUM:
            risks.append(("pbo", RiskLevel.MEDIUM))

        high_risks = sum(1 for _, level in risks if level == RiskLevel.HIGH)
        medium_risks = sum(1 for _, level in risks if level == RiskLevel.MEDIUM)

        if high_risks >= T.HIGH_RISK_COUNT_THRESHOLD:
            return RiskLevel.HIGH
        elif high_risks >= 1 or medium_risks >= T.HIGH_RISK_COUNT_THRESHOLD:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    @property
    def recommendation(self) -> str:
        """建議"""
        recommendations = {
            RiskLevel.LOW: "策略可靠，可進行下一步驗證",
            RiskLevel.MEDIUM: "需謹慎，建議增加樣本或簡化參數",
            RiskLevel.HIGH: "高過擬合風險，建議重新設計策略",
        }
        return recommendations.get(self.overall_risk, "")

    @property
    def warnings(self) -> List[str]:
        """警告訊息列表"""
        warnings = []
        T = OverfitThresholds

        if self.pbo > T.PBO_HIGH:
            warnings.append(f"PBO {self.pbo:.1%} 超過 {T.PBO_HIGH:.0%}，高過擬合風險")
        elif self.pbo > T.PBO_MEDIUM:
            warnings.append(f"PBO {self.pbo:.1%} 介於 {T.PBO_MEDIUM:.0%}-{T.PBO_HIGH:.0%}，需謹慎")

        if self.is_oos_ratio > T.IS_OOS_RATIO_HIGH:
            warnings.append(f"IS/OOS 比 {self.is_oos_ratio:.2f} 超過 {T.IS_OOS_RATIO_HIGH}，績效衰退過大")
        elif self.is_oos_ratio > T.IS_OOS_RATIO_MEDIUM:
            warnings.append(f"IS/OOS 比 {self.is_oos_ratio:.2f} 介於 {T.IS_OOS_RATIO_MEDIUM}-{T.IS_OOS_RATIO_HIGH}")

        if self.trade_count < T.MIN_TRADES_VALID:
            warnings.append(f"交易筆數 {self.trade_count} < {T.MIN_TRADES_VALID}，統計無效")
        elif self.trade_count < T.MIN_TRADES_CONFIDENT:
            warnings.append(f"交易筆數 {self.trade_count} < {T.MIN_TRADES_CONFIDENT}，統計信心較低")

        if self.param_sensitivity > T.PARAM_SENSITIVITY_HIGH:
            warnings.append(f"參數敏感度 {self.param_sensitivity:.1%} > {T.PARAM_SENSITIVITY_HIGH:.0%}，參數不穩定")
        elif self.param_sensitivity > T.PARAM_SENSITIVITY_MEDIUM:
            warnings.append(f"參數敏感度 {self.param_sensitivity:.1%} 介於 {T.PARAM_SENSITIVITY_MEDIUM:.0%}-{T.PARAM_SENSITIVITY_HIGH:.0%}")

        return warnings

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "pbo": self.pbo,
            "is_oos_ratio": self.is_oos_ratio,
            "param_sensitivity": self.param_sensitivity,
            "trade_count": self.trade_count,
            "degradation": self.degradation,
            "overall_risk": self.overall_risk.value,
            "recommendation": self.recommendation,
            "warnings": self.warnings,
        }


@dataclass
class PBOResult:
    """PBO 計算詳細結果"""

    pbo: float
    degradation: float
    n_combinations: int
    logits: List[float] = field(default_factory=list)
    avg_oos_rank: float = 0.0


class OverfitDetector:
    """過擬合偵測器

    提供多種過擬合檢測方法：
    - PBO (Probability of Backtest Overfitting)
    - Deflated Sharpe Ratio
    - 參數敏感度分析
    - 交易筆數檢查
    """

    def __init__(
        self,
        min_trades: int = 30,
        max_pbo: float = 0.5,
        max_is_oos_ratio: float = 2.0,
        max_param_sensitivity: float = 0.3,
    ):
        """初始化偵測器

        Args:
            min_trades: 最小交易筆數
            max_pbo: 最大可接受 PBO
            max_is_oos_ratio: 最大可接受 IS/OOS 比
            max_param_sensitivity: 最大可接受參數敏感度
        """
        self.min_trades = min_trades
        self.max_pbo = max_pbo
        self.max_is_oos_ratio = max_is_oos_ratio
        self.max_param_sensitivity = max_param_sensitivity

    def calculate_pbo(
        self,
        returns_matrix: np.ndarray,
        n_splits: int = 8,
        metric: str = "sharpe",
    ) -> PBOResult:
        """計算 PBO (Probability of Backtest Overfitting)

        使用 CSCV (Combinatorially Symmetric Cross-Validation) 方法。

        Args:
            returns_matrix: 每個參數組合的報酬序列，shape: (n_periods, n_trials)
            n_splits: 分割數
            metric: 評估指標 ('sharpe', 'return', 'omega')

        Returns:
            PBOResult: PBO 結果

        PBO 解讀：
        - < 25%: 低風險，策略可靠
        - 25-50%: 中風險，需謹慎
        - 50-75%: 高風險，很可能過擬合
        - > 75%: 極高風險，重新設計
        """
        n_periods, n_trials = returns_matrix.shape
        split_size = n_periods // n_splits

        if split_size < 10:
            # 資料太少，無法進行有效分割
            return PBOResult(
                pbo=0.5,  # 無法判斷，回傳中性值
                degradation=0.0,
                n_combinations=0,
            )

        # 產生所有組合
        all_splits = list(range(n_splits))
        is_combos = list(combinations(all_splits, n_splits // 2))

        logits = []
        oos_ranks = []

        for is_indices in is_combos:
            oos_indices = [i for i in all_splits if i not in is_indices]

            # 建立 IS 和 OOS 資料
            is_data = np.concatenate(
                [
                    returns_matrix[i * split_size : (i + 1) * split_size]
                    for i in is_indices
                ]
            )

            oos_data = np.concatenate(
                [
                    returns_matrix[i * split_size : (i + 1) * split_size]
                    for i in oos_indices
                ]
            )

            # 計算每個試驗的績效
            is_scores = self._calculate_metric(is_data, metric)
            oos_scores = self._calculate_metric(oos_data, metric)

            # IS 最佳試驗
            best_trial = np.argmax(is_scores)

            # 該試驗在 OOS 的排名
            ranks = rankdata(-oos_scores)  # 高分低排名
            best_oos_rank = ranks[best_trial]
            oos_ranks.append(best_oos_rank)

            # 計算 logit
            w = best_oos_rank / n_trials
            if 0 < w < 1:
                logit = np.log(w / (1 - w))
                logits.append(logit)

        # PBO = 排名在中位數以下的比例
        pbo = np.mean([logit > 0 for logit in logits]) if logits else 0.5

        # 績效衰退
        degradation = self._calculate_degradation(returns_matrix, n_splits, metric)

        # 平均 OOS 排名
        avg_oos_rank = np.mean(oos_ranks) / n_trials if oos_ranks else 0.5

        return PBOResult(
            pbo=pbo,
            degradation=degradation,
            n_combinations=len(is_combos),
            logits=logits,
            avg_oos_rank=avg_oos_rank,
        )

    def _calculate_metric(self, returns: np.ndarray, metric: str) -> np.ndarray:
        """計算每個試驗的績效指標"""
        if metric == "sharpe":
            mean_returns = np.mean(returns, axis=0)
            std_returns = np.std(returns, axis=0) + 1e-10
            return mean_returns / std_returns * np.sqrt(252)
        elif metric == "return":
            return np.sum(returns, axis=0)
        elif metric == "omega":
            threshold = 0
            gains = np.sum(np.maximum(returns - threshold, 0), axis=0)
            losses = np.sum(np.maximum(threshold - returns, 0), axis=0)
            return gains / (losses + 1e-10)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _calculate_degradation(
        self,
        returns: np.ndarray,
        n_splits: int,
        metric: str,
    ) -> float:
        """計算 IS 到 OOS 的績效衰退"""
        # 使用前半和後半
        mid = len(returns) // 2
        is_data = returns[:mid]
        oos_data = returns[mid:]

        is_scores = self._calculate_metric(is_data, metric)
        oos_scores = self._calculate_metric(oos_data, metric)

        best_is = np.argmax(is_scores)

        is_perf = is_scores[best_is]
        oos_perf = oos_scores[best_is]

        if is_perf > 0:
            return 1 - oos_perf / is_perf
        return 0.0

    def calculate_is_oos_ratio(
        self,
        is_sharpe: float,
        oos_sharpe: float,
    ) -> float:
        """計算 IS/OOS Sharpe 比

        Args:
            is_sharpe: In-Sample Sharpe
            oos_sharpe: Out-of-Sample Sharpe

        Returns:
            float: IS/OOS 比（> 2.0 為警戒值）
        """
        if oos_sharpe > 0:
            return is_sharpe / oos_sharpe
        elif is_sharpe > 0:
            return float("inf")
        else:
            return 1.0

    def calculate_param_sensitivity(
        self,
        sharpe_matrix: np.ndarray,
        threshold: float = 0.3,
    ) -> Tuple[float, bool]:
        """計算參數敏感度

        分析相鄰參數的績效變異，高變異表示參數不穩定。

        Args:
            sharpe_matrix: 參數組合的 Sharpe 矩陣（2D: param1 x param2）
            threshold: 敏感度閾值

        Returns:
            Tuple[float, bool]: (敏感度分數, 是否過度敏感)
        """
        if sharpe_matrix.size == 0:
            return 0.0, False

        # 計算相鄰參數的變異
        diff_x = np.abs(np.diff(sharpe_matrix, axis=0)) if sharpe_matrix.shape[0] > 1 else np.array([])
        diff_y = np.abs(np.diff(sharpe_matrix, axis=1)) if sharpe_matrix.shape[1] > 1 else np.array([])

        # 相對變異
        mean_sharpe = np.mean(sharpe_matrix)
        if mean_sharpe > 0:
            rel_diff_x = diff_x / mean_sharpe if diff_x.size > 0 else np.array([0])
            rel_diff_y = diff_y / mean_sharpe if diff_y.size > 0 else np.array([0])
        else:
            rel_diff_x = diff_x if diff_x.size > 0 else np.array([0])
            rel_diff_y = diff_y if diff_y.size > 0 else np.array([0])

        # 計算平均敏感度
        sensitivities = []
        if rel_diff_x.size > 0:
            sensitivities.append(np.mean(rel_diff_x))
        if rel_diff_y.size > 0:
            sensitivities.append(np.mean(rel_diff_y))

        sensitivity = np.mean(sensitivities) if sensitivities else 0.0

        return float(sensitivity), sensitivity > threshold

    def deflated_sharpe_ratio(
        self,
        observed_sharpe: float,
        n_trials: int,
        sample_length: int,
        skewness: float = 0,
        kurtosis: float = 3,
    ) -> Dict[str, float]:
        """計算 Deflated Sharpe Ratio

        多次測試會膨脹 Sharpe Ratio，DSR 校正這個偏誤。

        Args:
            observed_sharpe: 觀察到的 Sharpe
            n_trials: 測試的參數組合數
            sample_length: 樣本長度（交易次數或時間點）
            skewness: 報酬偏態
            kurtosis: 報酬峰態

        Returns:
            Dict: {'dsr': float, 'psr': float, 'haircut': float, ...}
        """
        # 計算 Sharpe 標準誤差
        sr_std = np.sqrt(
            (
                1
                + 0.5 * observed_sharpe**2
                - skewness * observed_sharpe
                + (kurtosis - 3) / 4 * observed_sharpe**2
            )
            / sample_length
        )

        # 預期最大 Sharpe（在 n_trials 中）
        expected_max_sharpe = self._expected_maximum(n_trials)

        # PSR：觀察 SR 超過基準的機率
        if sr_std > 0:
            psr = stats.norm.cdf((observed_sharpe - expected_max_sharpe) / sr_std)
        else:
            psr = 0.5

        # Haircut
        haircut = (
            observed_sharpe / expected_max_sharpe if expected_max_sharpe > 0 else 1.0
        )

        # DSR
        dsr = psr * haircut

        return {
            "dsr": dsr,
            "psr": psr,
            "haircut": haircut,
            "sr_std": sr_std,
            "expected_max_sharpe": expected_max_sharpe,
        }

    def _expected_maximum(self, n_trials: int) -> float:
        """預期最大值（標準常態分佈）"""
        if n_trials <= 1:
            return 0

        # 使用 Euler-Mascheroni 近似
        gamma = 0.5772156649
        z1 = stats.norm.ppf(1 - 1 / n_trials)
        z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))

        return (1 - gamma) * z1 + gamma * z2

    def assess(
        self,
        is_sharpe: float,
        oos_sharpe: float,
        trade_count: int,
        param_sensitivity: float = 0.0,
        returns_matrix: Optional[np.ndarray] = None,
    ) -> OverfitMetrics:
        """綜合過擬合風險評估

        Args:
            is_sharpe: In-Sample Sharpe
            oos_sharpe: Out-of-Sample Sharpe
            trade_count: 交易筆數
            param_sensitivity: 參數敏感度（如已計算）
            returns_matrix: 報酬矩陣（用於 PBO 計算）

        Returns:
            OverfitMetrics: 過擬合指標
        """
        # 計算 IS/OOS 比
        is_oos_ratio = self.calculate_is_oos_ratio(is_sharpe, oos_sharpe)

        # 計算 PBO
        pbo = 0.0
        degradation = 0.0
        if returns_matrix is not None and returns_matrix.size > 0:
            pbo_result = self.calculate_pbo(returns_matrix)
            pbo = pbo_result.pbo
            degradation = pbo_result.degradation

        return OverfitMetrics(
            pbo=pbo,
            is_oos_ratio=is_oos_ratio,
            param_sensitivity=param_sensitivity,
            trade_count=trade_count,
            degradation=degradation,
        )

    def check_trade_count(self, trade_count: int) -> Tuple[bool, str]:
        """檢查交易筆數是否足夠

        Args:
            trade_count: 交易筆數

        Returns:
            Tuple[bool, str]: (是否通過, 訊息)
        """
        if trade_count < self.min_trades:
            return False, f"交易筆數 {trade_count} < {self.min_trades}，統計無效"
        elif trade_count < 50:
            return True, f"交易筆數 {trade_count} < 50，統計信心較低"
        else:
            return True, f"交易筆數 {trade_count}，統計有效"

    def should_reject_strategy(self, metrics: OverfitMetrics) -> Tuple[bool, str]:
        """判斷是否應拒絕策略

        Args:
            metrics: 過擬合指標

        Returns:
            Tuple[bool, str]: (是否拒絕, 原因)
        """
        if metrics.pbo > self.max_pbo:
            return True, f"PBO {metrics.pbo:.1%} > {self.max_pbo:.0%}"

        if metrics.is_oos_ratio > self.max_is_oos_ratio:
            return True, f"IS/OOS 比 {metrics.is_oos_ratio:.2f} > {self.max_is_oos_ratio}"

        if metrics.trade_count < self.min_trades:
            return True, f"交易筆數 {metrics.trade_count} < {self.min_trades}"

        if metrics.param_sensitivity > self.max_param_sensitivity:
            return True, f"參數敏感度 {metrics.param_sensitivity:.1%} > {self.max_param_sensitivity:.0%}"

        return False, "策略通過過擬合檢測"
