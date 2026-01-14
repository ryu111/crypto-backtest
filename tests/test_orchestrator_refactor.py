"""
測試 Orchestrator 重構結果

驗證：
1. OrchestratorConfig dataclass 功能
2. __init__ 支援多種輸入方式
3. _create_backtest_config 正確建立配置
4. 向後兼容性
"""

import pytest
from dataclasses import asdict
from datetime import datetime
import pandas as pd
import numpy as np

from src.automation.orchestrator import (
    Orchestrator,
    OrchestratorConfig
)


class TestOrchestratorConfig:
    """測試 OrchestratorConfig dataclass"""

    def test_default_config(self):
        """測試預設配置"""
        config = OrchestratorConfig()

        assert config.n_trials == 50
        assert config.min_sharpe == 1.0
        assert config.min_stages == 3
        assert config.max_overfit == 0.5
        assert config.symbols == ['BTCUSDT', 'ETHUSDT']
        # 預設包含多個時間框架
        expected_timeframes = [
            '1m', '3m', '5m', '15m', '30m',      # 短線
            '1h', '2h', '4h', '6h', '8h',        # 中線（8h 對齊資金費率）
            '12h', '1d', '3d', '1w'              # 長線
        ]
        assert config.timeframes == expected_timeframes
        assert config.leverage == 5
        assert config.initial_capital == 10000.0
        assert config.maker_fee == 0.0002
        assert config.taker_fee == 0.0004
        assert config.slippage == 0.0001

    def test_custom_config(self):
        """測試自訂配置"""
        config = OrchestratorConfig(
            n_trials=100,
            min_sharpe=1.5,
            symbols=['BTCUSDT'],
            leverage=10
        )

        assert config.n_trials == 100
        assert config.min_sharpe == 1.5
        assert config.symbols == ['BTCUSDT']
        assert config.leverage == 10

        # 其他參數使用預設值
        assert config.min_stages == 3
        # timeframes 使用預設的多時間框架列表
        expected_timeframes = [
            '1m', '3m', '5m', '15m', '30m',      # 短線
            '1h', '2h', '4h', '6h', '8h',        # 中線（8h 對齊資金費率）
            '12h', '1d', '3d', '1w'              # 長線
        ]
        assert config.timeframes == expected_timeframes

    def test_config_is_dataclass(self):
        """測試是否為 dataclass"""
        config = OrchestratorConfig()
        config_dict = asdict(config)

        assert isinstance(config_dict, dict)
        assert 'n_trials' in config_dict
        assert 'symbols' in config_dict


class TestOrchestratorInit:
    """測試 Orchestrator __init__ 多種輸入方式"""

    def test_init_with_none(self):
        """測試使用 None (預設配置)"""
        orch = Orchestrator(config=None)

        assert isinstance(orch.config, OrchestratorConfig)
        assert orch.config.n_trials == 50
        assert orch.config.symbols == ['BTCUSDT', 'ETHUSDT']

    def test_init_with_dict(self):
        """測試使用 dict (向後兼容)"""
        config_dict = {
            'n_trials': 100,
            'min_sharpe': 1.5,
            'symbols': ['BTCUSDT'],
            'leverage': 10
        }

        orch = Orchestrator(config=config_dict)

        assert isinstance(orch.config, OrchestratorConfig)
        assert orch.config.n_trials == 100
        assert orch.config.min_sharpe == 1.5
        assert orch.config.symbols == ['BTCUSDT']
        assert orch.config.leverage == 10

    def test_init_with_orchestrator_config(self):
        """測試使用 OrchestratorConfig 實例"""
        config = OrchestratorConfig(
            n_trials=200,
            min_sharpe=2.0,
            symbols=['ETHUSDT']
        )

        orch = Orchestrator(config=config)

        assert isinstance(orch.config, OrchestratorConfig)
        assert orch.config.n_trials == 200
        assert orch.config.min_sharpe == 2.0
        assert orch.config.symbols == ['ETHUSDT']

    def test_backward_compatibility(self):
        """測試向後兼容性：舊程式碼仍能運作"""
        # 模擬舊程式碼的使用方式
        old_style_config = {
            'n_trials': 50,
            'min_sharpe': 1.0,
            'symbols': ['BTCUSDT', 'ETHUSDT'],
            'timeframes': ['4h']
        }

        orch = Orchestrator(config=old_style_config, verbose=False)

        # 確認可以正常初始化
        assert orch.config.n_trials == 50
        assert orch.config.symbols == ['BTCUSDT', 'ETHUSDT']


class TestCreateBacktestConfig:
    """測試 _create_backtest_config 方法"""

    def test_create_backtest_config(self):
        """測試建立 BacktestConfig"""
        # 建立測試資料
        dates = pd.date_range('2024-01-01', periods=100, freq='4h')
        data = pd.DataFrame({
            'open': np.random.randn(100),
            'high': np.random.randn(100),
            'low': np.random.randn(100),
            'close': np.random.randn(100),
            'volume': np.random.randn(100)
        }, index=dates)

        # 建立 Orchestrator
        orch = Orchestrator(
            config={
                'symbols': ['BTCUSDT'],
                'timeframes': ['4h'],
                'leverage': 5,
                'initial_capital': 10000.0,
                'maker_fee': 0.0002,
                'taker_fee': 0.0004,
                'slippage': 0.0001
            },
            verbose=False
        )

        # 呼叫 _create_backtest_config
        backtest_config = orch._create_backtest_config(
            symbol='BTCUSDT',
            data=data
        )

        # 驗證配置
        assert backtest_config.symbol == 'BTCUSDT'
        assert backtest_config.timeframe == '4h'
        assert backtest_config.start_date == dates[0]
        assert backtest_config.end_date == dates[-1]
        assert backtest_config.initial_capital == 10000.0
        assert backtest_config.leverage == 5
        assert backtest_config.maker_fee == 0.0002
        assert backtest_config.taker_fee == 0.0004
        assert backtest_config.slippage == 0.0001

    def test_config_consistency(self):
        """測試多次呼叫產生一致的配置"""
        dates = pd.date_range('2024-01-01', periods=100, freq='4h')
        data = pd.DataFrame({
            'close': np.random.randn(100)
        }, index=dates)

        orch = Orchestrator(verbose=False)

        # 呼叫兩次
        config1 = orch._create_backtest_config('BTCUSDT', data)
        config2 = orch._create_backtest_config('BTCUSDT', data)

        # 驗證配置一致
        assert config1.symbol == config2.symbol
        assert config1.timeframe == config2.timeframe
        assert config1.leverage == config2.leverage
        assert config1.initial_capital == config2.initial_capital


class TestAttributeAccess:
    """測試屬性存取（確保沒有 dict 存取）"""

    def test_config_attribute_access(self):
        """測試所有配置都使用屬性存取"""
        config = OrchestratorConfig()

        # 確認可以用屬性存取
        assert config.n_trials == 50
        assert config.min_sharpe == 1.0
        assert config.symbols == ['BTCUSDT', 'ETHUSDT']

        # 確認 dataclass 的屬性存在
        assert hasattr(config, 'n_trials')
        assert hasattr(config, 'min_sharpe')
        assert hasattr(config, 'symbols')
        assert hasattr(config, 'timeframes')
        assert hasattr(config, 'leverage')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
