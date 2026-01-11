"""
策略選擇器 - 使用 Explore/Exploit 平衡選擇下一個要優化的策略

實作三種選擇策略：
1. Epsilon-Greedy: 80% 利用最佳，20% 探索
2. UCB (Upper Confidence Bound): 平衡期望值和不確定性
3. Thompson Sampling: 貝葉斯後驗抽樣
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
import random
import math
import numpy as np


@dataclass
class StrategyStats:
    """策略統計資訊"""
    name: str
    attempts: int = 0
    successes: int = 0  # 通過驗證次數
    avg_sharpe: float = 0.0
    best_sharpe: float = 0.0
    last_attempt: Optional[datetime] = None
    last_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """成功率"""
        return self.successes / self.attempts if self.attempts > 0 else 0.0

    @property
    def failure_rate(self) -> float:
        """失敗率"""
        return 1 - self.success_rate


class StrategySelector:
    """策略選擇器"""

    def __init__(
        self,
        strategy_registry: 'StrategyRegistry',
        experiment_recorder: 'ExperimentRecorder',
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化策略選擇器

        Args:
            strategy_registry: 策略註冊表
            experiment_recorder: 實驗記錄器
            config: 配置參數
        """
        self.registry = strategy_registry
        self.recorder = experiment_recorder

        # 預設配置
        default_config = {
            'epsilon': 0.2,  # 探索機率（20%）
            'ucb_c': 2.0,    # UCB 探索常數
            'min_attempts': 3,  # 最小嘗試次數才納入 exploit
            'thompson_alpha': 1.0,  # Thompson Sampling alpha 參數
            'thompson_beta': 1.0,   # Thompson Sampling beta 參數
        }
        self.config = {**default_config, **(config or {})}

        # 策略統計（快取）
        self._stats_cache: Dict[str, StrategyStats] = {}
        self._cache_updated = False

    def select(self, method: str = 'epsilon_greedy') -> str:
        """
        選擇下一個要優化的策略

        Args:
            method: 選擇方法 ('epsilon_greedy', 'ucb', 'thompson_sampling')

        Returns:
            策略名稱
        """
        # 更新統計快取
        if not self._cache_updated:
            self._update_stats_cache()

        # 根據方法選擇
        if method == 'epsilon_greedy':
            return self._epsilon_greedy()
        elif method == 'ucb':
            return self._ucb()
        elif method == 'thompson_sampling':
            return self._thompson_sampling()
        else:
            raise ValueError(f"Unknown selection method: {method}")

    def _epsilon_greedy(self) -> str:
        """
        Epsilon-Greedy 選擇

        - 80% 機率選擇歷史最佳（Exploit）
        - 20% 機率隨機探索（Explore）
        """
        epsilon = self.config['epsilon']

        if random.random() < epsilon:
            # Explore: 隨機選擇
            return self._random_select()
        else:
            # Exploit: 選擇最佳
            return self._best_select()

    def _ucb(self) -> str:
        """
        UCB (Upper Confidence Bound) 選擇

        UCB = avg_sharpe + c * sqrt(ln(total_attempts) / strategy_attempts)

        平衡期望值和不確定性，對嘗試次數少的策略有獎勵
        """
        c = self.config['ucb_c']
        stats = self.get_strategy_stats()

        # 計算總嘗試次數
        total_attempts = sum(s.attempts for s in stats.values())
        if total_attempts == 0:
            return self._random_select()

        # 計算每個策略的 UCB 值
        ucb_scores = {}
        for name, stat in stats.items():
            if stat.attempts == 0:
                # 未嘗試過的策略給予最大優先權
                ucb_scores[name] = float('inf')
            else:
                exploration_bonus = c * math.sqrt(
                    math.log(total_attempts) / stat.attempts
                )
                ucb_scores[name] = stat.avg_sharpe + exploration_bonus

        # 選擇 UCB 值最高的策略
        return max(ucb_scores.items(), key=lambda x: x[1])[0]

    def _thompson_sampling(self) -> str:
        """
        Thompson Sampling 選擇

        基於貝葉斯後驗抽樣，自然平衡探索利用
        使用 Beta 分佈建模成功率
        """
        alpha = self.config['thompson_alpha']
        beta = self.config['thompson_beta']
        stats = self.get_strategy_stats()

        # 為每個策略從 Beta 分佈抽樣
        samples = {}
        for name, stat in stats.items():
            # Beta(alpha + successes, beta + failures)
            posterior_alpha = alpha + stat.successes
            posterior_beta = beta + (stat.attempts - stat.successes)

            # 從後驗分佈抽樣
            samples[name] = np.random.beta(posterior_alpha, posterior_beta)

        # 選擇抽樣值最高的策略
        return max(samples.items(), key=lambda x: x[1])[0]

    def _random_select(self) -> str:
        """隨機選擇策略"""
        strategies = self.registry.list_strategies()
        return random.choice(strategies)

    def _best_select(self) -> str:
        """選擇歷史最佳策略"""
        stats = self.get_strategy_stats()
        min_attempts = self.config['min_attempts']

        # 過濾出嘗試次數足夠的策略
        eligible = {
            name: stat for name, stat in stats.items()
            if stat.attempts >= min_attempts
        }

        if not eligible:
            # 沒有足夠嘗試的策略，隨機選擇
            return self._random_select()

        # 選擇平均 Sharpe 最高的策略
        return max(eligible.items(), key=lambda x: x[1].avg_sharpe)[0]

    def update_stats(
        self,
        strategy_name: str,
        result: Dict[str, Any]
    ):
        """
        更新策略統計

        Args:
            strategy_name: 策略名稱
            result: 優化結果
                {
                    'passed': bool,
                    'sharpe_ratio': float,
                    'params': dict
                }
        """
        # 從快取或建立新統計
        if strategy_name not in self._stats_cache:
            self._stats_cache[strategy_name] = StrategyStats(name=strategy_name)

        stat = self._stats_cache[strategy_name]

        # 更新統計
        stat.attempts += 1
        if result.get('passed', False):
            stat.successes += 1

        sharpe = result.get('sharpe_ratio', 0.0)

        # 更新平均 Sharpe（增量計算）
        stat.avg_sharpe = (
            (stat.avg_sharpe * (stat.attempts - 1) + sharpe) / stat.attempts
        )

        # 更新最佳 Sharpe
        if sharpe > stat.best_sharpe:
            stat.best_sharpe = sharpe

        stat.last_attempt = datetime.now()
        stat.last_params = result.get('params', {})

        # 同步到 ExperimentRecorder
        self.recorder.record_strategy_stats(strategy_name, stat)

    def get_strategy_stats(self) -> Dict[str, StrategyStats]:
        """
        取得所有策略的統計資訊

        Returns:
            策略名稱 -> StrategyStats
        """
        if not self._cache_updated:
            self._update_stats_cache()

        return self._stats_cache.copy()

    def _update_stats_cache(self):
        """從 ExperimentRecorder 更新統計快取"""
        strategies = self.registry.list_strategies()

        for name in strategies:
            # 從記錄器載入歷史資料
            stats = self.recorder.get_strategy_stats(name)

            if stats:
                self._stats_cache[name] = stats
            else:
                # 新策略，建立空統計
                self._stats_cache[name] = StrategyStats(name=name)

        self._cache_updated = True

    def get_exploration_stats(self) -> Dict[str, Any]:
        """
        取得探索統計

        Returns:
            {
                'total_attempts': int,
                'strategies_tried': int,
                'strategies_available': int,
                'exploration_rate': float,
                'best_strategy': str,
                'best_sharpe': float
            }
        """
        stats = self.get_strategy_stats()
        total_strategies = len(self.registry.list_strategies())

        total_attempts = sum(s.attempts for s in stats.values())
        strategies_tried = sum(1 for s in stats.values() if s.attempts > 0)

        best_stat = max(
            stats.values(),
            key=lambda s: s.best_sharpe,
            default=None
        )

        return {
            'total_attempts': total_attempts,
            'strategies_tried': strategies_tried,
            'strategies_available': total_strategies,
            'exploration_rate': strategies_tried / total_strategies if total_strategies > 0 else 0.0,
            'best_strategy': best_stat.name if best_stat else None,
            'best_sharpe': best_stat.best_sharpe if best_stat else 0.0,
        }

    def reset_cache(self):
        """重置快取，強制重新載入"""
        self._stats_cache.clear()
        self._cache_updated = False

    def get_recommendation(self) -> Dict[str, Any]:
        """
        取得推薦策略及原因

        Returns:
            {
                'strategy': str,
                'method': str,
                'reason': str,
                'stats': StrategyStats,
                'alternatives': List[str]
            }
        """
        # 使用三種方法各選一個
        epsilon_choice = self._epsilon_greedy()
        ucb_choice = self._ucb()
        thompson_choice = self._thompson_sampling()

        # 投票選出最佳
        votes = [epsilon_choice, ucb_choice, thompson_choice]
        vote_counts = {s: votes.count(s) for s in set(votes)}
        recommended = max(vote_counts.items(), key=lambda x: x[1])[0]

        stats = self._stats_cache.get(recommended)

        # 產生推薦原因
        reasons = []
        if recommended == epsilon_choice:
            reasons.append("epsilon-greedy 推薦")
        if recommended == ucb_choice:
            reasons.append("UCB 推薦（高潛力）")
        if recommended == thompson_choice:
            reasons.append("Thompson Sampling 推薦")

        reason = " + ".join(reasons)

        # 取得替代選項
        alternatives = [s for s in set(votes) if s != recommended]

        return {
            'strategy': recommended,
            'method': 'ensemble',
            'reason': reason,
            'stats': stats,
            'alternatives': alternatives,
        }
