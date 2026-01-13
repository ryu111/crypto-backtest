"""
回測引擎核心

基於 VectorBT Pro 的永續合約回測引擎。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, Union
import pandas as pd
import numpy as np
import logging

try:
    import vectorbtpro as vbt
except ImportError:
    import vectorbt as vbt

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None

from .metrics import MetricsCalculator
from .vectorized import (
    ensure_pandas,
    ensure_polars,
    vectorized_positions,
    vectorized_pnl,
)

logger = logging.getLogger(__name__)

# 常數定義
DAYS_PER_YEAR = 365
TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestConfig:
    """回測配置"""

    # 基本設定
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime

    # 資金設定
    initial_capital: float = 10000.0
    leverage: int = 1

    # 交易成本
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0004  # 0.04%
    slippage: float = 0.0001  # 0.01%

    # 永續合約特有
    funding_rate: float = 0.0001  # 預設資金費率（每 8 小時）
    funding_interval_hours: int = 8

    # 風險控制
    max_leverage: int = 10
    position_mode: str = "one-way"  # "one-way" or "hedge"

    # 效能設定
    use_polars: bool = True  # 使用 Polars 後端（更快）
    vectorized: bool = True  # 使用向量化計算

    # 其他
    commission_type: str = "percentage"  # "percentage" or "fixed"


    def __post_init__(self):
        """驗證配置"""
        if self.leverage > self.max_leverage:
            raise ValueError(f"槓桿 {self.leverage} 超過最大值 {self.max_leverage}")

        if self.leverage < 1:
            raise ValueError("槓桿不能小於 1")

        if self.initial_capital <= 0:
            raise ValueError("初始資金必須大於 0")

        if self.position_mode not in ["one-way", "hedge"]:
            raise ValueError("position_mode 必須是 'one-way' 或 'hedge'")

        # 檢查 Polars 可用性
        if self.use_polars and not POLARS_AVAILABLE:
            logger.warning("Polars not available, falling back to Pandas")
            self.use_polars = False


@dataclass
class BacktestResult:
    """回測結果"""

    # 整體績效
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # 風險指標
    max_drawdown: float
    max_drawdown_duration: int
    volatility: float

    # 交易統計
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade_duration: float

    # 進階指標
    expectancy: float
    recovery_factor: float
    ulcer_index: float

    # 原始資料
    equity_curve: pd.Series
    trades: pd.DataFrame
    daily_returns: pd.Series

    # 永續合約特有
    total_funding_fees: float = 0.0
    avg_leverage_used: float = 1.0

    # 額外資訊
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """轉為字典"""
        return {
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'volatility': self.volatility,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'avg_trade_duration': self.avg_trade_duration,
            'expectancy': self.expectancy,
            'recovery_factor': self.recovery_factor,
            'ulcer_index': self.ulcer_index,
            'total_funding_fees': self.total_funding_fees,
            'avg_leverage_used': self.avg_leverage_used,
            **self.metadata
        }

    def summary(self) -> str:
        """產生摘要報告"""
        return f"""
回測結果摘要
{'='*50}
總報酬率: {self.total_return:.2%}
年化報酬率: {self.annual_return:.2%}
夏普比率: {self.sharpe_ratio:.2f}
索提諾比率: {self.sortino_ratio:.2f}
卡爾馬比率: {self.calmar_ratio:.2f}

風險指標
{'-'*50}
最大回撤: {self.max_drawdown:.2%}
回撤持續: {self.max_drawdown_duration} 天
波動率: {self.volatility:.2%}
潰瘍指數: {self.ulcer_index:.2f}

交易統計
{'-'*50}
總交易次數: {self.total_trades}
勝率: {self.win_rate:.2%}
獲利因子: {self.profit_factor:.2f}
平均獲利: {self.avg_win:.2f}
平均虧損: {self.avg_loss:.2f}
期望值: {self.expectancy:.2f}

永續合約
{'-'*50}
總資金費用: {self.total_funding_fees:.2f}
平均槓桿: {self.avg_leverage_used:.2f}x
"""


class BacktestEngine:
    """
    回測引擎

    基於 VectorBT Pro 的永續合約回測引擎。
    支援資金費率、槓桿、完整績效分析。

    使用範例：
        config = BacktestConfig(
            symbol='BTCUSDT',
            timeframe='1h',
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=10000,
            leverage=3
        )

        engine = BacktestEngine(config)
        result = engine.run(strategy, params)
        print(result.summary())
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data: Optional[Union[pd.DataFrame, pl.DataFrame]] = None
        self.metrics_calc = MetricsCalculator()
        self._use_polars = config.use_polars and POLARS_AVAILABLE

    def load_data(
        self,
        data: Optional[Union[pd.DataFrame, pl.DataFrame]] = None
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        載入市場資料

        Args:
            data: OHLCV DataFrame (Pandas or Polars)

        Returns:
            載入的資料
        """
        if data is not None:
            # 自動轉換為對應後端格式
            if self._use_polars and isinstance(data, pd.DataFrame):
                self.data = ensure_polars(data)
            elif not self._use_polars and isinstance(data, pl.DataFrame):
                self.data = ensure_pandas(data)
            else:
                self.data = data
        else:
            raise NotImplementedError("請提供 data 或實作自動載入功能")

        # 驗證資料格式
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        data_cols = self.data.columns if isinstance(self.data, pd.DataFrame) else self.data.columns
        missing = set(required_cols) - set(data_cols)
        if missing:
            raise ValueError(f"缺少必要欄位: {missing}")

        return self.data

    def run(
        self,
        strategy: Any,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Union[pd.DataFrame, pl.DataFrame]] = None
    ) -> BacktestResult:
        """
        執行回測

        Args:
            strategy: 策略物件（需有 generate_signals 方法）
            params: 策略參數（會覆蓋策略預設值）
            data: 市場資料（Pandas or Polars）

        Returns:
            回測結果
        """
        # 載入資料
        if data is None and self.data is None:
            raise ValueError("請先提供市場資料")
        elif data is not None:
            self.load_data(data)

        # 更新策略參數
        if params:
            strategy.params.update(params)

        # 選擇執行路徑
        if self.config.vectorized and self._use_polars:
            result = self._run_vectorized_polars(strategy)
        elif self.config.vectorized:
            result = self._run_vectorized_pandas(strategy)
        else:
            result = self._run_vectorbt(strategy)

        return result

    def _run_vectorbt(self, strategy: Any) -> BacktestResult:
        """使用 VectorBT 執行回測（原始方法）"""
        # 確保資料為 Pandas 格式
        data_pandas = ensure_pandas(self.data)

        # 產生交易訊號
        signals = strategy.generate_signals(data_pandas)
        long_entry, long_exit, short_entry, short_exit = signals

        # 建立 VectorBT Portfolio
        portfolio = self._create_portfolio(
            long_entry, long_exit, short_entry, short_exit
        )

        # 計算績效指標
        result = self._calculate_metrics(portfolio, strategy)

        return result

    def _run_vectorized_pandas(self, strategy: Any) -> BacktestResult:
        """使用向量化 Pandas 執行回測"""
        data_pandas = ensure_pandas(self.data)

        # 產生交易訊號
        signals = strategy.generate_signals(data_pandas)
        long_entry, long_exit, short_entry, short_exit = signals

        # 向量化計算部位
        signal_combined = pd.Series(0, index=data_pandas.index)
        signal_combined[long_entry] = 1
        signal_combined[short_entry] = -1

        positions = vectorized_positions(signal_combined, self.config.position_mode)

        # 向量化計算損益
        pnl = vectorized_pnl(
            positions,
            data_pandas['close'],
            self.config.leverage,
            self.config.taker_fee
        )

        # 建立 VectorBT Portfolio 用於完整指標計算
        portfolio = self._create_portfolio(
            long_entry, long_exit, short_entry, short_exit
        )

        result = self._calculate_metrics(portfolio, strategy)
        return result

    def _run_vectorized_polars(self, strategy: Any) -> BacktestResult:
        """使用向量化 Polars 執行回測（最快）"""
        data_polars = ensure_polars(self.data)

        # 策略需支援 Polars，否則轉為 Pandas
        try:
            signals = strategy.generate_signals(data_polars)
        except Exception as e:
            logger.warning(f"Strategy doesn't support Polars: {e}, falling back to Pandas")
            return self._run_vectorized_pandas(strategy)

        long_entry, long_exit, short_entry, short_exit = signals

        # 轉回 Pandas 用於 VectorBT（Polars Series indexing 不同）
        data_pandas = data_polars.to_pandas()

        # 將 Polars Series 轉為 Pandas Series
        if hasattr(long_entry, 'to_pandas'):
            long_entry_pd = long_entry.to_pandas()
            long_exit_pd = long_exit.to_pandas()
            short_entry_pd = short_entry.to_pandas()
            short_exit_pd = short_exit.to_pandas()
        else:
            long_entry_pd = pd.Series(long_entry.to_list(), index=data_pandas.index)
            long_exit_pd = pd.Series(long_exit.to_list(), index=data_pandas.index)
            short_entry_pd = pd.Series(short_entry.to_list(), index=data_pandas.index)
            short_exit_pd = pd.Series(short_exit.to_list(), index=data_pandas.index)

        # 向量化計算部位
        signal_combined = pd.Series(0, index=data_pandas.index)
        signal_combined[long_entry_pd] = 1
        signal_combined[short_entry_pd] = -1

        positions = vectorized_positions(signal_combined, self.config.position_mode)

        pnl = vectorized_pnl(
            positions,
            data_pandas['close'],
            self.config.leverage,
            self.config.taker_fee
        )

        # 建立 VectorBT Portfolio
        portfolio = self._create_portfolio(
            long_entry_pd, long_exit_pd, short_entry_pd, short_exit_pd
        )

        result = self._calculate_metrics(portfolio, strategy)
        return result

    def _create_portfolio(
        self,
        long_entry: pd.Series,
        long_exit: pd.Series,
        short_entry: pd.Series,
        short_exit: pd.Series
    ):
        """建立 VectorBT 投資組合"""

        # 確保使用 Pandas DataFrame（Polars 不支援 .index）
        data_pandas = ensure_pandas(self.data)
        close = data_pandas['close']

        # VectorBT 使用 size 參數模擬槓桿效果
        # size=np.inf + size_type='value' = 使用全部可用資金
        # 槓桿效果通過調整 init_cash 實現（init_cash * leverage）
        effective_cash = self.config.initial_capital * self.config.leverage

        # 統一使用 short_entries/short_exits 方式建立 Portfolio
        # 這是 VectorBT 正確的多空訊號處理方式
        # position_mode 的差異在於風控邏輯，而非 VectorBT 建立方式
        pf = vbt.Portfolio.from_signals(
            close,
            entries=long_entry,
            exits=long_exit,
            short_entries=short_entry,
            short_exits=short_exit,
            init_cash=effective_cash,
            size=np.inf,
            size_type='value',
            fees=self.config.taker_fee,
            slippage=self.config.slippage,
            freq=self.config.timeframe
        )

        return pf

    def _calculate_funding_fees(self, portfolio) -> Tuple[float, float]:
        """
        計算資金費率費用

        Returns:
            (總資金費用, 平均槓桿)
        """
        trades = portfolio.trades.records_readable

        if len(trades) == 0:
            return 0.0, 0.0

        # 計算每筆交易的持倉時間（防禦性處理：確保時間戳為 datetime 類型）
        try:
            entry_ts = pd.to_datetime(trades['Entry Timestamp'])
            exit_ts = pd.to_datetime(trades['Exit Timestamp'])
            trades['duration_hours'] = (exit_ts - entry_ts).dt.total_seconds() / 3600
        except (KeyError, TypeError):
            trades['duration_hours'] = 0.0

        # 計算資金費率次數
        trades['funding_count'] = (
            trades['duration_hours'] / self.config.funding_interval_hours
        ).apply(np.floor)

        # 計算資金費用
        trades['funding_fee'] = (
            trades['Size'] *
            trades['Avg Entry Price'] *
            self.config.funding_rate *
            trades['funding_count']
        )

        total_funding_fees = trades['funding_fee'].sum()

        # 計算平均槓桿
        avg_leverage = (
            trades['Size'] * trades['Avg Entry Price'] /
            self.config.initial_capital
        ).mean()

        return total_funding_fees, avg_leverage

    def _calculate_trade_statistics(
        self,
        trades_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        計算交易統計

        Args:
            trades_df: 交易記錄 DataFrame

        Returns:
            包含交易統計的字典：
            - total_trades
            - win_rate
            - avg_win
            - avg_loss
            - profit_factor
            - expectancy
            - avg_duration
        """
        total_trades = len(trades_df)

        if total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0,
                'avg_duration': 0.0,
            }

        win_trades = trades_df[trades_df['PnL'] > 0]
        loss_trades = trades_df[trades_df['PnL'] < 0]

        win_rate = len(win_trades) / total_trades
        avg_win = win_trades['PnL'].mean() if len(win_trades) > 0 else 0.0
        avg_loss = loss_trades['PnL'].mean() if len(loss_trades) > 0 else 0.0

        # 獲利因子
        total_wins = win_trades['PnL'].sum() if len(win_trades) > 0 else 0.0
        total_losses = abs(loss_trades['PnL'].sum()) if len(loss_trades) > 0 else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        # 期望值
        expectancy = avg_win * win_rate + avg_loss * (1 - win_rate)

        # 平均持倉時間（防禦性處理：確保時間戳為 datetime 類型）
        try:
            entry_ts = pd.to_datetime(trades_df['Entry Timestamp'])
            exit_ts = pd.to_datetime(trades_df['Exit Timestamp'])
            duration_hours = (exit_ts - entry_ts).dt.total_seconds() / 3600
            avg_duration = duration_hours.mean()
        except (KeyError, TypeError):
            avg_duration = 0.0

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_duration': avg_duration,
        }

    def _calculate_metrics(self, portfolio, strategy) -> BacktestResult:
        """計算完整績效指標"""

        stats = portfolio.stats()
        trades_df = portfolio.trades.records_readable

        # 基本績效
        total_return = portfolio.total_return()
        equity_curve = portfolio.value()
        daily_returns = portfolio.returns()

        # 計算年化報酬
        total_days = (self.config.end_date - self.config.start_date).days
        annual_return = (1 + total_return) ** (DAYS_PER_YEAR / total_days) - 1

        # 風險指標
        sharpe = self.metrics_calc.calculate_sharpe(daily_returns)
        sortino = self.metrics_calc.calculate_sortino(daily_returns)
        max_dd = portfolio.max_drawdown()
        volatility = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        # 交易統計
        trade_stats = self._calculate_trade_statistics(trades_df)
        total_trades = trade_stats['total_trades']
        win_rate = trade_stats['win_rate']
        avg_win = trade_stats['avg_win']
        avg_loss = trade_stats['avg_loss']
        profit_factor = trade_stats['profit_factor']
        expectancy = trade_stats['expectancy']
        avg_duration = trade_stats['avg_duration']

        # 進階指標
        calmar = self.metrics_calc.calculate_calmar(annual_return, max_dd)
        ulcer = self.metrics_calc.calculate_ulcer_index(equity_curve)
        recovery_factor = abs(total_return / max_dd) if max_dd != 0 else 0

        # 最大回撤持續時間
        dd_series = portfolio.drawdown()
        dd_duration = self.metrics_calc.calculate_max_dd_duration(dd_series)

        # 永續合約特有指標
        total_funding, avg_leverage = self._calculate_funding_fees(portfolio)

        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_duration=dd_duration,
            volatility=volatility,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade_duration=avg_duration,
            expectancy=expectancy,
            recovery_factor=recovery_factor,
            ulcer_index=ulcer,
            equity_curve=equity_curve,
            trades=trades_df,
            daily_returns=daily_returns,
            total_funding_fees=total_funding,
            avg_leverage_used=avg_leverage,
            metadata={
                'strategy_name': strategy.name,
                'strategy_params': strategy.params,
                'config': self.config.__dict__
            }
        )

    def optimize(
        self,
        strategy: Any,
        param_grid: Dict[str, list],
        metric: str = 'sharpe_ratio',
        data: Optional[pd.DataFrame] = None
    ) -> Tuple[Dict, BacktestResult]:
        """
        參數優化

        Args:
            strategy: 策略物件
            param_grid: 參數網格 {'param_name': [val1, val2, ...]}
            metric: 優化目標指標
            data: 市場資料

        Returns:
            (最佳參數, 最佳結果)
        """
        if data is not None:
            self.load_data(data)

        best_metric = float('-inf')
        best_params = None
        best_result = None

        # 產生所有參數組合
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        for values in product(*param_values):
            params = dict(zip(param_names, values))

            try:
                result = self.run(strategy, params)
                current_metric = getattr(result, metric)

                if current_metric > best_metric:
                    best_metric = current_metric
                    best_params = params
                    best_result = result

            except Exception as e:
                logger.warning(f"參數組合 {params} 失敗: {e}", exc_info=True)
                continue

        return best_params, best_result
