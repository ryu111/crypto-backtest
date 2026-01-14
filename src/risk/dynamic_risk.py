"""
動態風控模組

根據市場狀態動態調整風控參數，包含：
1. 波動度調整的部位大小
2. 回撤調整的風險縮減
3. 移動止損機制
4. 日/週風險限制

設計原則：
- 高波動 → 小部位（降低風險）
- 高回撤 → 小部位或暫停交易（保護資金）
- 獲利後啟動移動止損（鎖定利潤）
- 所有計算向量化（numpy）
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ============= 常數定義 =============
EPSILON = 1e-8                    # 浮點數比較容差
DEFAULT_RISK_PERCENT = 0.01       # 預設風險百分比
MIN_VOL_ADJUSTMENT = 0.5          # 最小波動度調整係數
MAX_VOL_ADJUSTMENT = 2.0          # 最大波動度調整係數
MIN_VOLATILITY = 0.005            # 最小波動率閾值（低於此值使用保守係數）
MINIMUM_EQUITY = 100.0            # 最小權益保護閾值


@dataclass
class DynamicRiskConfig:
    """動態風控配置"""
    # 基礎風控
    base_risk_per_trade: float = 0.02      # 基礎單筆風險 2%
    max_risk_per_trade: float = 0.03       # 最大單筆風險 3%
    min_risk_per_trade: float = 0.01       # 最小單筆風險 1%

    # 波動度調整
    volatility_scaling: bool = True
    volatility_lookback: int = 20
    target_volatility: float = 0.02        # 目標日波動率 2%

    # 回撤調整
    drawdown_scaling: bool = True
    dd_threshold_1: float = 0.05           # 5% DD 開始降低
    dd_threshold_2: float = 0.10           # 10% DD 大幅降低
    dd_threshold_3: float = 0.15           # 15% DD 暫停交易

    # 移動止損
    trailing_stop_enabled: bool = True
    trailing_stop_atr_mult: float = 2.0
    trailing_activation: float = 0.02      # 獲利 2% 後啟動

    # 日/週風險限制
    daily_loss_limit: float = 0.05         # 單日最大虧損 5%
    weekly_loss_limit: float = 0.10        # 單週最大虧損 10%

    def __post_init__(self):
        """驗證配置參數"""
        # 驗證風險參數
        if not 0 < self.min_risk_per_trade <= self.base_risk_per_trade <= self.max_risk_per_trade <= 1.0:
            raise ValueError(
                f"風險參數必須滿足: 0 < min <= base <= max <= 1.0，"
                f"得到: min={self.min_risk_per_trade}, "
                f"base={self.base_risk_per_trade}, "
                f"max={self.max_risk_per_trade}"
            )

        # 驗證回撤門檻
        if not 0 < self.dd_threshold_1 < self.dd_threshold_2 < self.dd_threshold_3 < 1.0:
            raise ValueError(
                f"回撤門檻必須滿足: 0 < t1 < t2 < t3 < 1.0，"
                f"得到: t1={self.dd_threshold_1}, "
                f"t2={self.dd_threshold_2}, "
                f"t3={self.dd_threshold_3}"
            )

        # 驗證目標波動率
        if not 0 < self.target_volatility < 1.0:
            raise ValueError(f"target_volatility 必須介於 0 和 1 之間，得到: {self.target_volatility}")

        # 驗證移動止損
        if self.trailing_stop_atr_mult <= 0:
            raise ValueError(f"trailing_stop_atr_mult 必須大於 0，得到: {self.trailing_stop_atr_mult}")

        logger.debug(f"動態風控配置已驗證: {self}")


@dataclass
class RiskState:
    """風控狀態記錄"""
    timestamp: datetime
    current_equity: float
    peak_equity: float
    current_dd: float
    current_dd_pct: float
    daily_pnl: float
    weekly_pnl: float
    volatility_adjustment: float
    dd_adjustment: float
    final_risk_pct: float
    trading_allowed: bool
    reason: str = ""

    def to_dict(self) -> Dict:
        """轉換為字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'equity': self.current_equity,
            'peak': self.peak_equity,
            'dd': self.current_dd,
            'dd_pct': self.current_dd_pct,
            'daily_pnl': self.daily_pnl,
            'weekly_pnl': self.weekly_pnl,
            'vol_adj': self.volatility_adjustment,
            'dd_adj': self.dd_adjustment,
            'risk_pct': self.final_risk_pct,
            'trading_ok': self.trading_allowed,
            'reason': self.reason
        }


class DynamicRiskController:
    """
    動態風控控制器

    功能：
    1. 基於波動度的部位大小
    2. 基於回撤的風險調整
    3. 移動止損機制
    4. 日/週風險限制
    """

    def __init__(self, config: Optional[DynamicRiskConfig] = None):
        """
        初始化動態風控控制器

        Args:
            config: 動態風控配置（如不提供則使用預設值）
        """
        self.config = config or DynamicRiskConfig()

        # 權益追蹤
        self._peak_equity = 0.0
        self._current_equity = 0.0
        self._current_dd = 0.0
        self._current_dd_pct = 0.0
        self._initialized = False  # 標記是否已初始化權益

        # 日/週 PnL 追蹤
        self._daily_pnl = 0.0
        self._weekly_pnl = 0.0
        self._daily_start_equity = 0.0
        self._weekly_start_equity = 0.0
        self._last_daily_reset = datetime.now().date()
        self._last_weekly_reset = datetime.now().isocalendar()[:2]  # (year, week)

        # 歷史狀態記錄
        self._state_history: List[RiskState] = []

        logger.info(f"動態風控控制器已初始化: {self.config}")

    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss_price: float,
        current_volatility: Optional[float] = None,
        atr: Optional[float] = None
    ) -> Tuple[float, Dict]:
        """
        計算動態部位大小

        考慮因素：
        1. 當前波動度（高波動 → 小部位）
        2. 當前回撤（高回撤 → 小部位）
        3. 基礎風險參數

        Args:
            capital: 當前可用資金
            entry_price: 入場價格
            stop_loss_price: 止損價格
            current_volatility: 當前波動率（可選，用於波動度調整）
            atr: 當前 ATR（可選，用於止損計算）

        Returns:
            position_size: 建議的部位大小（單位數量）
            details: 計算詳情

        Raises:
            ValueError: 當參數不合法時
        """
        # 參數驗證
        if capital <= 0:
            raise ValueError(f"capital 必須大於 0，得到: {capital}")

        if entry_price <= 0:
            raise ValueError(f"entry_price 必須大於 0，得到: {entry_price}")

        if stop_loss_price <= 0:
            raise ValueError(f"stop_loss_price 必須大於 0，得到: {stop_loss_price}")

        # 檢查是否已初始化權益
        if not self._initialized:
            raise RuntimeError(
                "必須先呼叫 update_equity() 初始化權益狀態才能計算部位大小"
            )

        # 檢查是否允許交易
        trading_allowed, reason = self.should_stop_trading()
        if not trading_allowed:
            logger.warning(f"交易已暫停: {reason}")
            return 0.0, {
                'trading_allowed': False,
                'reason': reason,
                'base_risk_pct': 0.0,
                'vol_adjustment': 1.0,
                'dd_adjustment': 1.0,
                'final_risk_pct': 0.0,
                'position_size': 0.0
            }

        # 1. 計算基礎風險金額
        base_risk_amount = capital * self.config.base_risk_per_trade

        # 2. 波動度調整係數
        vol_adjustment = 1.0
        if self.config.volatility_scaling and current_volatility is not None:
            vol_adjustment = self._get_volatility_adjustment(current_volatility)

        # 3. 回撤調整係數
        dd_adjustment = self._get_dd_adjustment()

        # 4. 計算調整後的風險金額
        adjusted_risk_amount = base_risk_amount * vol_adjustment * dd_adjustment

        # 5. 限制在最小/最大範圍內
        min_risk_amount = capital * self.config.min_risk_per_trade
        max_risk_amount = capital * self.config.max_risk_per_trade
        final_risk_amount = np.clip(adjusted_risk_amount, min_risk_amount, max_risk_amount)

        # 6. 計算每單位的風險
        risk_per_unit = abs(entry_price - stop_loss_price)

        # 處理止損價格等於入場價格的極端情況
        if risk_per_unit < EPSILON:
            logger.warning(
                f"止損價格接近入場價格 (entry={entry_price}, stop={stop_loss_price})，"
                f"使用 ATR 或預設風險"
            )
            if atr is not None and atr > 0:
                risk_per_unit = atr * self.config.trailing_stop_atr_mult
            else:
                risk_per_unit = entry_price * DEFAULT_RISK_PERCENT  # 預設 1% 風險

        # 7. 計算部位大小
        position_size = final_risk_amount / risk_per_unit

        # 8. 組裝計算詳情
        final_risk_pct = final_risk_amount / capital
        details = {
            'trading_allowed': True,
            'reason': 'OK',
            'base_risk_pct': self.config.base_risk_per_trade,
            'vol_adjustment': vol_adjustment,
            'dd_adjustment': dd_adjustment,
            'final_risk_pct': final_risk_pct,
            'final_risk_amount': final_risk_amount,
            'risk_per_unit': risk_per_unit,
            'position_size': position_size,
            'entry_price': entry_price,
            'stop_loss_price': stop_loss_price
        }

        logger.debug(
            f"部位計算完成: 資金={capital:.2f}, "
            f"風險={final_risk_pct:.2%} (vol_adj={vol_adjustment:.2f}, dd_adj={dd_adjustment:.2f}), "
            f"部位={position_size:.4f}"
        )

        return position_size, details

    def update_equity(self, current_equity: float, force_reset_peak: bool = False):
        """
        更新權益和回撤狀態

        Args:
            current_equity: 當前權益
            force_reset_peak: 是否強制重設高點（用於新週期開始）

        Raises:
            ValueError: 當權益為負數或低於最小閾值時
        """
        if current_equity < 0:
            raise ValueError(f"權益不能為負數，得到: {current_equity}")

        # 最小權益保護
        if current_equity < MINIMUM_EQUITY:
            logger.warning(
                f"權益過低: {current_equity:.2f} < {MINIMUM_EQUITY:.2f}，"
                f"建議檢查資金管理或停止交易"
            )

        # 更新當前權益
        self._current_equity = current_equity

        # 初始化高點
        if self._peak_equity == 0.0 or force_reset_peak:
            self._peak_equity = current_equity
            self._current_dd = 0.0
            self._current_dd_pct = 0.0
            self._initialized = True  # 標記已初始化

            # 初始化日/週起始權益（修復：不能跳過 PnL 初始化）
            self._daily_start_equity = current_equity
            self._weekly_start_equity = current_equity

            logger.info(f"權益高點已設定: {self._peak_equity:.2f}")
            return

        # 更新高點
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
            self._current_dd = 0.0
            self._current_dd_pct = 0.0
            logger.debug(f"新權益高點: {self._peak_equity:.2f}")
        else:
            # 計算回撤
            self._current_dd = self._peak_equity - current_equity
            self._current_dd_pct = self._current_dd / self._peak_equity if self._peak_equity > 0 else 0.0

            if self._current_dd_pct >= self.config.dd_threshold_1:
                logger.warning(
                    f"當前回撤: {self._current_dd_pct:.2%} "
                    f"(${self._current_dd:.2f} from peak ${self._peak_equity:.2f})"
                )

        # 更新日/週 PnL
        self._update_period_pnl(current_equity)

        # 記錄狀態
        self._record_state()

    def _update_period_pnl(self, current_equity: float):
        """
        更新日/週 PnL（內部方法）

        Args:
            current_equity: 當前權益（傳入參數，而非使用 self._current_equity）
        """
        now = datetime.now()
        current_date = now.date()
        current_week = now.isocalendar()[:2]

        # 檢查是否需要重設日 PnL
        if current_date != self._last_daily_reset:
            logger.debug(f"日 PnL 重設: {current_date}")
            self._daily_start_equity = current_equity  # 使用傳入的 current_equity
            self._daily_pnl = 0.0
            self._last_daily_reset = current_date

        # 檢查是否需要重設週 PnL
        if current_week != self._last_weekly_reset:
            logger.debug(f"週 PnL 重設: Week {current_week[1]}, {current_week[0]}")
            self._weekly_start_equity = current_equity  # 使用傳入的 current_equity
            self._weekly_pnl = 0.0
            self._last_weekly_reset = current_week

        # 計算 PnL
        if self._daily_start_equity > 0:
            self._daily_pnl = (current_equity - self._daily_start_equity) / self._daily_start_equity

        if self._weekly_start_equity > 0:
            self._weekly_pnl = (current_equity - self._weekly_start_equity) / self._weekly_start_equity

    def _get_volatility_adjustment(self, current_volatility: float) -> float:
        """
        根據波動度獲取調整係數

        公式：adjustment = target_vol / current_vol
        - 當前波動高於目標 → 係數 < 1 → 減少部位
        - 當前波動低於目標 → 係數 > 1 → 增加部位

        限制在 [MIN_VOL_ADJUSTMENT, MAX_VOL_ADJUSTMENT] 範圍內避免極端值

        Args:
            current_volatility: 當前波動率

        Returns:
            adjustment: 調整係數
        """
        if current_volatility <= 0:
            logger.warning(f"波動率無效: {current_volatility}，使用係數 1.0")
            return 1.0

        # 極低波動度保護（可能是數據異常或市場停滯）
        if current_volatility <= MIN_VOLATILITY:
            logger.warning(
                f"波動率過低: {current_volatility:.6f} <= {MIN_VOLATILITY:.6f}，"
                f"使用保守係數 {MIN_VOL_ADJUSTMENT}"
            )
            return MIN_VOL_ADJUSTMENT

        # 計算調整係數
        adjustment = self.config.target_volatility / current_volatility

        # 限制在合理範圍
        adjustment = np.clip(adjustment, MIN_VOL_ADJUSTMENT, MAX_VOL_ADJUSTMENT)

        logger.debug(
            f"波動度調整: current_vol={current_volatility:.4f}, "
            f"target_vol={self.config.target_volatility:.4f}, "
            f"adjustment={adjustment:.2f}"
        )

        return adjustment

    def _get_dd_adjustment(self) -> float:
        """
        根據回撤獲取調整係數

        回撤等級：
        - < 5%: 正常交易 (1.0)
        - 5-10%: 降低風險 (0.5)
        - 10-15%: 大幅降低 (0.25)
        - >= 15%: 暫停交易 (0.0)

        Returns:
            adjustment: 調整係數 [0.0, 1.0]
        """
        if not self.config.drawdown_scaling:
            return 1.0

        dd = self._current_dd_pct

        # 分級調整
        if dd < self.config.dd_threshold_1:
            adjustment = 1.0
        elif dd < self.config.dd_threshold_2:
            # 線性插值: 5% → 1.0, 10% → 0.5
            t = (dd - self.config.dd_threshold_1) / (self.config.dd_threshold_2 - self.config.dd_threshold_1)
            adjustment = 1.0 - (0.5 * t)
        elif dd < self.config.dd_threshold_3:
            # 線性插值: 10% → 0.5, 15% → 0.25
            t = (dd - self.config.dd_threshold_2) / (self.config.dd_threshold_3 - self.config.dd_threshold_2)
            adjustment = 0.5 - (0.25 * t)
        else:
            # >= 15%: 停止交易
            adjustment = 0.0

        if adjustment < 1.0:
            logger.warning(
                f"回撤調整啟動: DD={dd:.2%}, adjustment={adjustment:.2f}"
            )

        return adjustment

    def update_trailing_stop(
        self,
        current_price: float,
        entry_price: float,
        current_stop: float,
        atr: float,
        direction: int  # 1=long, -1=short
    ) -> float:
        """
        更新移動止損

        邏輯：
        1. 計算當前盈虧百分比
        2. 如果達到啟動條件（預設獲利 2%），啟動移動止損
        3. 移動止損 = 當前價 - (方向 * ATR * multiplier)
        4. 止損只能往有利方向移動（long: 向上, short: 向下）

        Args:
            current_price: 當前價格
            entry_price: 入場價格
            current_stop: 當前止損價格
            atr: 當前 ATR 值
            direction: 方向 (1=做多, -1=做空)

        Returns:
            new_stop: 更新後的止損價格

        Raises:
            ValueError: 當參數不合法時
        """
        # 參數驗證
        if direction not in [1, -1]:
            raise ValueError(f"direction 必須是 1 (long) 或 -1 (short)，得到: {direction}")

        if not self.config.trailing_stop_enabled:
            return current_stop

        if atr <= 0:
            logger.warning(f"ATR 無效: {atr}，維持原止損")
            return current_stop

        if current_price <= 0 or entry_price <= 0:
            raise ValueError(f"價格必須大於 0，得到: current={current_price}, entry={entry_price}")

        # 計算盈虧百分比
        if direction == 1:  # Long
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # Short
            pnl_pct = (entry_price - current_price) / entry_price

        # 檢查是否達到啟動條件
        if pnl_pct < self.config.trailing_activation:
            logger.debug(
                f"移動止損尚未啟動: pnl={pnl_pct:.2%} < "
                f"activation={self.config.trailing_activation:.2%}"
            )
            return current_stop

        # 計算新止損位置
        trailing_distance = atr * self.config.trailing_stop_atr_mult

        if direction == 1:  # Long
            new_stop = current_price - trailing_distance
            # 止損只能向上移動
            new_stop = max(new_stop, current_stop)
        else:  # Short
            new_stop = current_price + trailing_distance
            # 止損只能向下移動
            new_stop = min(new_stop, current_stop)

        if new_stop != current_stop:
            logger.info(
                f"移動止損更新: {current_stop:.2f} → {new_stop:.2f} "
                f"(盈虧={pnl_pct:.2%}, ATR={atr:.2f})"
            )

        return new_stop

    def should_stop_trading(self) -> Tuple[bool, str]:
        """
        檢查是否應該暫停交易

        暫停條件：
        1. 回撤 >= 15%
        2. 日虧損 >= 5%
        3. 週虧損 >= 10%

        Returns:
            (should_continue, reason)
            - should_continue: True=可以繼續, False=應暫停
            - reason: 原因說明
        """
        # 檢查回撤
        if self._current_dd_pct >= self.config.dd_threshold_3:
            return False, f"回撤過大: {self._current_dd_pct:.2%} >= {self.config.dd_threshold_3:.2%}"

        # 檢查日虧損
        if self._daily_pnl <= -self.config.daily_loss_limit:
            return False, f"日虧損超限: {self._daily_pnl:.2%} <= -{self.config.daily_loss_limit:.2%}"

        # 檢查週虧損
        if self._weekly_pnl <= -self.config.weekly_loss_limit:
            return False, f"週虧損超限: {self._weekly_pnl:.2%} <= -{self.config.weekly_loss_limit:.2%}"

        return True, "OK"

    def get_risk_report(self) -> Dict:
        """
        獲取風控報告

        Returns:
            dict: 包含所有風控狀態的報告
        """
        trading_allowed, reason = self.should_stop_trading()
        vol_adj = 1.0  # 沒有實時波動率時的預設值
        dd_adj = self._get_dd_adjustment()

        return {
            'equity': {
                'current': self._current_equity,
                'peak': self._peak_equity,
                'drawdown': self._current_dd,
                'drawdown_pct': self._current_dd_pct
            },
            'period_pnl': {
                'daily': self._daily_pnl,
                'weekly': self._weekly_pnl
            },
            'adjustments': {
                'volatility': vol_adj,
                'drawdown': dd_adj,
                'combined': vol_adj * dd_adj
            },
            'risk_limits': {
                'base_risk_pct': self.config.base_risk_per_trade,
                'current_risk_pct': self.config.base_risk_per_trade * vol_adj * dd_adj,
                'max_risk_pct': self.config.max_risk_per_trade,
                'min_risk_pct': self.config.min_risk_per_trade
            },
            'trading_status': {
                'allowed': trading_allowed,
                'reason': reason
            },
            'thresholds': {
                'dd_threshold_1': self.config.dd_threshold_1,
                'dd_threshold_2': self.config.dd_threshold_2,
                'dd_threshold_3': self.config.dd_threshold_3,
                'daily_loss_limit': self.config.daily_loss_limit,
                'weekly_loss_limit': self.config.weekly_loss_limit
            },
            'state_history_count': len(self._state_history)
        }

    def _record_state(self):
        """記錄當前風控狀態（內部方法）"""
        trading_allowed, reason = self.should_stop_trading()
        vol_adj = 1.0  # 預設值
        dd_adj = self._get_dd_adjustment()
        final_risk_pct = self.config.base_risk_per_trade * vol_adj * dd_adj

        state = RiskState(
            timestamp=datetime.now(),
            current_equity=self._current_equity,
            peak_equity=self._peak_equity,
            current_dd=self._current_dd,
            current_dd_pct=self._current_dd_pct,
            daily_pnl=self._daily_pnl,
            weekly_pnl=self._weekly_pnl,
            volatility_adjustment=vol_adj,
            dd_adjustment=dd_adj,
            final_risk_pct=final_risk_pct,
            trading_allowed=trading_allowed,
            reason=reason
        )

        self._state_history.append(state)

    def get_state_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        獲取歷史狀態記錄

        Args:
            limit: 限制返回數量（None=全部）

        Returns:
            List[Dict]: 狀態記錄列表
        """
        if limit is None:
            return [state.to_dict() for state in self._state_history]
        else:
            return [state.to_dict() for state in self._state_history[-limit:]]

    def reset(self, initial_equity: Optional[float] = None):
        """
        重置所有狀態

        Args:
            initial_equity: 初始權益（可選）
        """
        if initial_equity is not None:
            if initial_equity <= 0:
                raise ValueError(f"initial_equity 必須大於 0，得到: {initial_equity}")
            self._peak_equity = initial_equity
            self._current_equity = initial_equity
            self._daily_start_equity = initial_equity
            self._weekly_start_equity = initial_equity
            self._initialized = True  # 標記已初始化
        else:
            self._peak_equity = 0.0
            self._current_equity = 0.0
            self._daily_start_equity = 0.0
            self._weekly_start_equity = 0.0
            self._initialized = False  # 未提供初始權益則標記為未初始化

        self._current_dd = 0.0
        self._current_dd_pct = 0.0
        self._daily_pnl = 0.0
        self._weekly_pnl = 0.0
        self._last_daily_reset = datetime.now().date()
        self._last_weekly_reset = datetime.now().isocalendar()[:2]
        self._state_history.clear()

        logger.info(f"風控狀態已重置，初始權益: {initial_equity or 0.0}")

    def __repr__(self) -> str:
        return (
            f"DynamicRiskController("
            f"equity={self._current_equity:.2f}, "
            f"dd={self._current_dd_pct:.2%}, "
            f"daily_pnl={self._daily_pnl:.2%}, "
            f"weekly_pnl={self._weekly_pnl:.2%})"
        )


# ============= 便利函數 =============

def create_conservative_controller() -> DynamicRiskController:
    """建立保守型風控控制器"""
    config = DynamicRiskConfig(
        base_risk_per_trade=0.01,
        max_risk_per_trade=0.02,
        min_risk_per_trade=0.005,
        dd_threshold_1=0.03,
        dd_threshold_2=0.05,
        dd_threshold_3=0.10,
        daily_loss_limit=0.03,
        weekly_loss_limit=0.08
    )
    return DynamicRiskController(config)


def create_aggressive_controller() -> DynamicRiskController:
    """建立激進型風控控制器"""
    config = DynamicRiskConfig(
        base_risk_per_trade=0.03,
        max_risk_per_trade=0.05,
        min_risk_per_trade=0.02,
        dd_threshold_1=0.08,
        dd_threshold_2=0.15,
        dd_threshold_3=0.25,
        daily_loss_limit=0.08,
        weekly_loss_limit=0.15
    )
    return DynamicRiskController(config)


# ============= 測試程式 =============

if __name__ == "__main__":
    """測試動態風控功能"""

    # 設定 logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 70)
    print("測試 1: 基本部位計算")
    print("=" * 70)

    controller = DynamicRiskController()
    controller.update_equity(10000.0)  # 初始資金 $10,000

    # 計算部位（無波動度調整）
    position, details = controller.calculate_position_size(
        capital=10000.0,
        entry_price=50000.0,
        stop_loss_price=49000.0  # 2% 止損
    )

    print(f"部位大小: {position:.4f}")
    print(f"詳情: {details}")
    print("✓ 測試 1 通過\n")

    print("=" * 70)
    print("測試 2: 波動度調整")
    print("=" * 70)

    # 高波動（應減少部位）
    position_high_vol, details_high = controller.calculate_position_size(
        capital=10000.0,
        entry_price=50000.0,
        stop_loss_price=49000.0,
        current_volatility=0.04  # 4% 波動（目標 2%）
    )

    # 低波動（應增加部位）
    position_low_vol, details_low = controller.calculate_position_size(
        capital=10000.0,
        entry_price=50000.0,
        stop_loss_price=49000.0,
        current_volatility=0.01  # 1% 波動（目標 2%）
    )

    print(f"高波動部位: {position_high_vol:.4f} (調整係數={details_high['vol_adjustment']:.2f})")
    print(f"低波動部位: {position_low_vol:.4f} (調整係數={details_low['vol_adjustment']:.2f})")
    assert position_high_vol < position_low_vol, "高波動應該有更小的部位"
    print("✓ 測試 2 通過\n")

    print("=" * 70)
    print("測試 3: 回撤調整")
    print("=" * 70)

    # 模擬虧損
    controller.update_equity(9300.0)  # -7% 虧損

    position_dd, details_dd = controller.calculate_position_size(
        capital=9300.0,
        entry_price=50000.0,
        stop_loss_price=49000.0
    )

    print(f"回撤部位: {position_dd:.4f}")
    print(f"回撤調整係數: {details_dd['dd_adjustment']:.2f}")
    print(f"當前回撤: {controller._current_dd_pct:.2%}")
    assert details_dd['dd_adjustment'] < 1.0, "回撤時應該降低風險"
    print("✓ 測試 3 通過\n")

    print("=" * 70)
    print("測試 4: 移動止損")
    print("=" * 70)

    # 獲利 3% 後啟動移動止損
    entry = 50000.0
    current = 51500.0  # +3%
    current_stop = 49000.0
    atr = 500.0

    new_stop = controller.update_trailing_stop(
        current_price=current,
        entry_price=entry,
        current_stop=current_stop,
        atr=atr,
        direction=1  # Long
    )

    print(f"原止損: {current_stop:.2f}")
    print(f"新止損: {new_stop:.2f}")
    print(f"移動距離: {new_stop - current_stop:.2f}")
    assert new_stop > current_stop, "做多時止損應該向上移動"
    print("✓ 測試 4 通過\n")

    print("=" * 70)
    print("測試 5: 交易暫停機制")
    print("=" * 70)

    # 模擬大幅回撤（16%）
    controller.update_equity(8400.0)

    trading_ok, reason = controller.should_stop_trading()
    print(f"允許交易: {trading_ok}")
    print(f"原因: {reason}")
    print(f"當前回撤: {controller._current_dd_pct:.2%}")
    assert not trading_ok, "大幅回撤時應該停止交易"
    print("✓ 測試 5 通過\n")

    print("=" * 70)
    print("測試 6: 風控報告")
    print("=" * 70)

    controller.reset(10000.0)
    controller.update_equity(9500.0)

    report = controller.get_risk_report()
    print("風控報告:")
    for key, value in report.items():
        print(f"  {key}: {value}")

    assert 'equity' in report
    assert 'trading_status' in report
    print("✓ 測試 6 通過\n")

    print("=" * 70)
    print("測試 7: 保守/激進控制器")
    print("=" * 70)

    conservative = create_conservative_controller()
    aggressive = create_aggressive_controller()

    conservative.update_equity(10000.0)
    aggressive.update_equity(10000.0)

    pos_con, _ = conservative.calculate_position_size(10000, 50000, 49000)
    pos_agg, _ = aggressive.calculate_position_size(10000, 50000, 49000)

    print(f"保守部位: {pos_con:.4f}")
    print(f"激進部位: {pos_agg:.4f}")
    assert pos_agg > pos_con, "激進控制器應該有更大的部位"
    print("✓ 測試 7 通過\n")

    print("=" * 70)
    print("所有測試完成！")
    print("=" * 70)
