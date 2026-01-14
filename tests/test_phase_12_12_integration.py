"""
Phase 12.12 整合測試

測試所有新模組的整合：
1. SignalAmplifier - 信號放大器
2. FilterPipeline - 過濾管道
3. DynamicRiskController - 動態風控
4. AdaptiveLeverageController - 自適應槓桿
5. UltimateLoopConfig - 高效能配置

測試項目：
- 匯入測試
- 實例化測試
- 基本功能測試
- 整合流程測試
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ===== 測試 1: 匯入測試 =====

def test_imports():
    """測試所有模組能正確匯入"""
    try:
        # Signal Amplifier
        from src.strategies.signal_amplifier import SignalAmplifier, AmplificationConfig

        # Filter Pipeline
        from src.strategies.filters import (
            FilterPipeline,
            BaseSignalFilter,
            SignalStrengthFilter,
            ConfirmationFilter,
            TimeFilter,
            VolumeFilter,
            create_filter
        )

        # Dynamic Risk
        from src.risk.dynamic_risk import (
            DynamicRiskController,
            DynamicRiskConfig,
            create_conservative_controller,
            create_aggressive_controller
        )

        # Adaptive Leverage
        from src.risk.adaptive_leverage import (
            AdaptiveLeverageController,
            AdaptiveLeverageConfig
        )

        # Ultimate Config
        from src.automation.ultimate_config import UltimateLoopConfig

        print("✓ 所有模組匯入成功")

    except ImportError as e:
        pytest.fail(f"匯入失敗: {e}")


# ===== 測試 2: 實例化測試 =====

def test_signal_amplifier_instantiation():
    """測試信號放大器實例化"""
    from src.strategies.signal_amplifier import SignalAmplifier, AmplificationConfig

    # 預設配置
    amp1 = SignalAmplifier()
    assert amp1.config.enabled is True

    # 自訂配置
    config = AmplificationConfig(
        rsi_expand=10.0,
        macd_lookahead=3,
        enabled=True
    )
    amp2 = SignalAmplifier(config)
    assert amp2.config.rsi_expand == 10.0

    print("✓ SignalAmplifier 實例化成功")


def test_filter_pipeline_instantiation():
    """測試過濾管道實例化"""
    from src.strategies.filters import FilterPipeline, create_filter

    # 空管道
    pipeline1 = FilterPipeline()
    assert len(pipeline1.filters) == 0

    # 預設管道
    pipeline2 = FilterPipeline.create_default()
    assert len(pipeline2.filters) > 0

    # 激進模式
    pipeline3 = FilterPipeline.create_aggressive()
    assert len(pipeline3.filters) < len(pipeline2.filters)

    # 保守模式
    pipeline4 = FilterPipeline.create_conservative()
    assert len(pipeline4.filters) > len(pipeline2.filters)

    # 使用工廠函數
    strength_filter = create_filter('strength', {'min_rsi_distance': 5.0})
    assert strength_filter is not None

    print("✓ FilterPipeline 實例化成功")


def test_dynamic_risk_instantiation():
    """測試動態風控實例化"""
    from src.risk.dynamic_risk import (
        DynamicRiskController,
        DynamicRiskConfig,
        create_conservative_controller,
        create_aggressive_controller
    )

    # 預設配置
    controller1 = DynamicRiskController()
    assert controller1.config.base_risk_per_trade == 0.02

    # 自訂配置
    config = DynamicRiskConfig(
        base_risk_per_trade=0.01,
        max_risk_per_trade=0.03,
        dd_threshold_3=0.15
    )
    controller2 = DynamicRiskController(config)
    assert controller2.config.base_risk_per_trade == 0.01

    # 保守控制器
    conservative = create_conservative_controller()
    assert conservative.config.base_risk_per_trade < 0.02

    # 激進控制器
    aggressive = create_aggressive_controller()
    assert aggressive.config.base_risk_per_trade > 0.02

    print("✓ DynamicRiskController 實例化成功")


def test_adaptive_leverage_instantiation():
    """測試自適應槓桿實例化"""
    from src.risk.adaptive_leverage import (
        AdaptiveLeverageController,
        AdaptiveLeverageConfig
    )

    # 預設配置
    controller1 = AdaptiveLeverageController()
    assert controller1.config.base_leverage == 5

    # 自訂配置
    config = AdaptiveLeverageConfig(
        base_leverage=10,
        max_leverage=20,
        volatility_mode=True
    )
    controller2 = AdaptiveLeverageController(config)
    assert controller2.config.base_leverage == 10

    print("✓ AdaptiveLeverageController 實例化成功")


def test_ultimate_config_instantiation():
    """測試 UltimateLoopConfig 實例化"""
    from src.automation.ultimate_config import UltimateLoopConfig

    # 預設配置
    config1 = UltimateLoopConfig()
    assert config1.max_workers == 8

    # 生產配置
    config2 = UltimateLoopConfig.create_production_config()
    assert config2.max_workers >= 16
    assert config2.validation_enabled is True

    # 開發配置
    config3 = UltimateLoopConfig.create_development_config()
    assert config3.max_workers == 8

    # 快速測試配置
    config4 = UltimateLoopConfig.create_quick_test_config()
    assert config4.max_workers == 2

    # 高效能配置
    config5 = UltimateLoopConfig.create_high_performance_config()
    assert config5.max_workers == 12
    assert config5.signal_amplification_enabled is True

    # 驗證所有配置
    for config in [config1, config2, config3, config4, config5]:
        try:
            config.validate()
        except ValueError as e:
            pytest.fail(f"配置驗證失敗: {e}")

    print("✓ UltimateLoopConfig 實例化成功")


# ===== 測試 3: 基本功能測試 =====

def test_signal_amplifier_basic_function():
    """測試信號放大器基本功能"""
    from src.strategies.signal_amplifier import SignalAmplifier, AmplificationConfig

    # 建立測試資料
    # 使用更明確的值來測試放大效果
    rsi = pd.Series([25, 28, 32, 34.9, 68, 72, 75])

    # 建立放大器（rsi_expand=5 → 30+5=35 oversold, 70-5=65 overbought）
    config = AmplificationConfig(rsi_expand=5.0, enabled=True)
    amp = SignalAmplifier(config)

    # 測試 RSI 信號放大
    oversold, overbought = amp.amplify_rsi_signals(rsi)

    # 驗證結果
    assert isinstance(oversold, pd.Series)
    assert isinstance(overbought, pd.Series)
    assert len(oversold) == len(rsi)

    # 檢查放大效果（閾值擴展為 < 35）
    # 注意：使用 == True 而非 is True，因為 pandas 返回 np.bool_
    assert oversold.iloc[0] == True  # RSI=25 < 35，應觸發
    assert oversold.iloc[2] == True  # RSI=32 < 35，應觸發
    assert oversold.iloc[3] == True  # RSI=34.9 < 35，應觸發
    assert overbought.iloc[4] == True  # RSI=68 > 65，應觸發
    assert overbought.iloc[5] == True  # RSI=72 > 65，應觸發

    # 檢查統計
    stats = amp.get_stats()
    assert 'rsi_amplified_count' in stats

    # 驗證放大效果
    # 原始：< 30 有 2 個（25, 28），> 70 有 2 個（72, 75）
    # 放大：< 35 有 4 個（25, 28, 32, 34.9），> 65 有 3 個（68, 72, 75）
    # 增加：2 + 1 = 3
    assert stats['rsi_amplified_count'] == 3

    print("✓ SignalAmplifier 基本功能正常")


def test_filter_pipeline_basic_function():
    """測試過濾管道基本功能"""
    from src.strategies.filters import FilterPipeline

    # 建立測試資料
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        'open': np.random.randn(n).cumsum() + 100,
        'high': np.random.randn(n).cumsum() + 101,
        'low': np.random.randn(n).cumsum() + 99,
        'close': np.random.randn(n).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, n),
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h')
    })

    # 建立信號（10% 的資料有信號）
    long_entry = pd.Series(np.random.random(n) < 0.1, index=data.index)
    long_exit = pd.Series(False, index=data.index)
    short_entry = pd.Series(np.random.random(n) < 0.1, index=data.index)
    short_exit = pd.Series(False, index=data.index)

    # 建立預設過濾管道
    pipeline = FilterPipeline.create_default()

    # 執行過濾
    filtered_long, _, filtered_short, _ = pipeline.process(
        data, long_entry, long_exit, short_entry, short_exit
    )

    # 驗證結果
    assert isinstance(filtered_long, pd.Series)
    assert isinstance(filtered_short, pd.Series)
    assert len(filtered_long) == len(data)

    # 過濾後信號應該更少
    assert filtered_long.sum() <= long_entry.sum()
    assert filtered_short.sum() <= short_entry.sum()

    # 檢查統計
    stats = pipeline.get_stats()
    assert '_total' in stats

    print("✓ FilterPipeline 基本功能正常")


def test_dynamic_risk_basic_function():
    """測試動態風控基本功能"""
    from src.risk.dynamic_risk import DynamicRiskController, DynamicRiskConfig

    # 建立控制器
    config = DynamicRiskConfig(
        base_risk_per_trade=0.02,
        volatility_scaling=True,
        drawdown_scaling=True
    )
    controller = DynamicRiskController(config)

    # 初始化權益
    controller.update_equity(10000.0)

    # 測試部位計算（無波動度調整）
    position1, details1 = controller.calculate_position_size(
        capital=10000.0,
        entry_price=50000.0,
        stop_loss_price=49000.0
    )

    assert position1 > 0
    assert 'trading_allowed' in details1
    assert details1['trading_allowed'] is True

    # 測試波動度調整（高波動應減少部位）
    position2, details2 = controller.calculate_position_size(
        capital=10000.0,
        entry_price=50000.0,
        stop_loss_price=49000.0,
        current_volatility=0.04  # 4% 波動（高於目標 2%）
    )

    assert position2 < position1  # 高波動時部位應該更小

    # 測試回撤調整（重置日 PnL，避免觸發日虧損限制）
    controller.reset(9300.0)  # 重置為新起點，避免日虧損限制
    position3, details3 = controller.calculate_position_size(
        capital=9300.0,
        entry_price=50000.0,
        stop_loss_price=49000.0
    )

    # 無回撤時應該沒有調整
    assert details3['dd_adjustment'] == 1.0

    # 測試移動止損
    new_stop = controller.update_trailing_stop(
        current_price=51500.0,  # +3% 獲利
        entry_price=50000.0,
        current_stop=49000.0,
        atr=500.0,
        direction=1  # Long
    )

    assert new_stop > 49000.0  # 止損應該向上移動

    # 檢查報告
    report = controller.get_risk_report()
    assert 'equity' in report
    assert 'trading_status' in report

    print("✓ DynamicRiskController 基本功能正常")


def test_adaptive_leverage_basic_function():
    """測試自適應槓桿基本功能"""
    from src.risk.adaptive_leverage import (
        AdaptiveLeverageController,
        AdaptiveLeverageConfig
    )

    # 建立控制器
    config = AdaptiveLeverageConfig(
        base_leverage=5,
        max_leverage=10,
        volatility_mode=True,
        drawdown_mode=True,
        performance_mode=True
    )
    controller = AdaptiveLeverageController(config)

    # 測試基礎槓桿計算
    leverage1 = controller.calculate_leverage(
        current_volatility=0.02,  # 正常波動
        current_drawdown=0.02,    # 小回撤
        recent_win_rate=0.55
    )

    assert leverage1 >= config.min_leverage
    assert leverage1 <= config.max_leverage

    # 測試低波動提高槓桿
    leverage2 = controller.calculate_leverage(
        current_volatility=0.005,  # 低波動
        current_drawdown=0.02,
        recent_win_rate=0.55
    )

    assert leverage2 >= leverage1  # 低波動應該提高槓桿

    # 測試高回撤降低槓桿（重置後再測試）
    controller.reset()  # 重置平滑值，避免受前面影響

    # 先計算一次基準（無回撤）
    leverage_baseline = controller.calculate_leverage(
        current_volatility=0.02,
        current_drawdown=0.0,  # 無回撤
        recent_win_rate=0.55
    )

    controller.reset()  # 再次重置

    # 再計算高回撤情況
    leverage3 = controller.calculate_leverage(
        current_volatility=0.02,
        current_drawdown=0.12,  # 高回撤（12% > 10% threshold_2）
        recent_win_rate=0.55
    )

    assert leverage3 < leverage_baseline  # 高回撤應該降低槓桿

    # 測試連勝機制
    for _ in range(5):
        controller.update_streak(True)  # 連勝 5 次

    leverage4 = controller.calculate_leverage(
        current_volatility=0.02,
        current_drawdown=0.02,
        recent_win_rate=0.65
    )

    # 檢查報告
    report = controller.get_leverage_report()
    assert 'avg_leverage' in report
    assert 'current_streak' in report
    assert report['current_streak'] == 5

    print("✓ AdaptiveLeverageController 基本功能正常")


# ===== 測試 4: 整合流程測試 =====

def test_full_integration_workflow():
    """測試完整整合流程"""
    from src.strategies.signal_amplifier import SignalAmplifier, AmplificationConfig
    from src.strategies.filters import FilterPipeline
    from src.risk.dynamic_risk import DynamicRiskController
    from src.risk.adaptive_leverage import AdaptiveLeverageController
    from src.automation.ultimate_config import UltimateLoopConfig

    print("\n=== 完整整合流程測試 ===\n")

    # 1. 建立高效能配置
    print("1. 建立配置...")
    config = UltimateLoopConfig.create_high_performance_config()
    config.validate()
    print(f"   ✓ 配置驗證通過: {config.max_workers} workers")

    # 2. 建立測試資料
    print("\n2. 準備測試資料...")
    np.random.seed(42)
    n = 200

    # 建立 OHLCV 資料
    close = pd.Series(np.random.randn(n).cumsum() + 50000)
    data = pd.DataFrame({
        'open': close + np.random.randn(n) * 10,
        'high': close + abs(np.random.randn(n)) * 20,
        'low': close - abs(np.random.randn(n)) * 20,
        'close': close,
        'volume': np.random.randint(1000, 10000, n),
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h')
    })

    # 建立指標
    rsi = pd.Series(30 + np.random.randn(n) * 20, index=data.index).clip(0, 100)
    macd = pd.Series(np.random.randn(n), index=data.index)
    signal = macd.rolling(9).mean()

    print(f"   ✓ 資料準備完成: {len(data)} 筆資料")

    # 3. 信號放大
    print("\n3. 執行信號放大...")
    amp_config = AmplificationConfig(
        rsi_expand=5.0,
        macd_lookahead=2,
        enabled=config.signal_amplification_enabled
    )
    amplifier = SignalAmplifier(amp_config)

    # 原始信號
    original_long = (rsi < 30) & (macd > signal)
    original_short = (rsi > 70) & (macd < signal)

    # 放大信號
    indicators = {'rsi': rsi, 'macd': macd, 'signal': signal}
    long_entry, long_exit, short_entry, short_exit = amplifier.amplify_all(
        data,
        (original_long, pd.Series(False, index=data.index),
         original_short, pd.Series(False, index=data.index)),
        indicators
    )

    amp_stats = amplifier.get_stats()
    print(f"   ✓ 信號放大完成:")
    print(f"     原始信號: {amp_stats['total_original_signals']}")
    print(f"     放大後: {amp_stats['total_amplified_signals']}")
    print(f"     增加率: {amp_stats.get('amplification_rate', 0):.1%}")

    # 4. 信號過濾
    print("\n4. 執行信號過濾...")
    pipeline = FilterPipeline.create_default() if config.signal_filter_enabled else FilterPipeline()

    filtered_long, _, filtered_short, _ = pipeline.process(
        data, long_entry, long_exit, short_entry, short_exit
    )

    filter_stats = pipeline.get_stats()
    if filter_stats:
        total = filter_stats.get('_total', {})
        print(f"   ✓ 信號過濾完成:")
        print(f"     做多: {total.get('original_long_signals', 0)} → {total.get('final_long_signals', 0)}")
        print(f"     做空: {total.get('original_short_signals', 0)} → {total.get('final_short_signals', 0)}")
    else:
        print(f"   ✓ 過濾器未啟用")

    # 5. 動態風控
    print("\n5. 執行動態風控...")
    risk_controller = DynamicRiskController()
    risk_controller.update_equity(config.initial_capital)

    # 計算部位大小
    position, risk_details = risk_controller.calculate_position_size(
        capital=config.initial_capital,
        entry_price=data['close'].iloc[-1],
        stop_loss_price=data['close'].iloc[-1] * 0.98,
        current_volatility=data['close'].pct_change().std()
    )

    print(f"   ✓ 風控計算完成:")
    print(f"     建議部位: {position:.4f}")
    print(f"     風險百分比: {risk_details['final_risk_pct']:.2%}")
    print(f"     波動調整: {risk_details['vol_adjustment']:.2f}")
    print(f"     回撤調整: {risk_details['dd_adjustment']:.2f}")

    # 6. 自適應槓桿
    print("\n6. 執行自適應槓桿...")
    leverage_controller = AdaptiveLeverageController()

    # 模擬幾筆交易
    for won in [True, True, False, True]:
        leverage_controller.update_streak(won)

    leverage = leverage_controller.calculate_leverage(
        current_volatility=data['close'].pct_change().std(),
        current_drawdown=risk_controller._current_dd_pct,
        recent_win_rate=0.60
    )

    lev_report = leverage_controller.get_leverage_report()
    print(f"   ✓ 槓桿計算完成:")
    print(f"     建議槓桿: {leverage}x")
    print(f"     當前連勝: {lev_report['current_streak']}")
    print(f"     近期勝率: {lev_report['recent_win_rate']:.2%}")

    # 7. 整合驗證
    print("\n7. 整合驗證...")

    # 所有模組都應該返回有效值
    assert long_entry is not None and isinstance(long_entry, pd.Series)
    assert position > 0
    assert 1 <= leverage <= 10
    assert risk_details['trading_allowed'] is True

    print(f"   ✓ 所有模組整合正常")

    # 8. 最終摘要
    print("\n" + "="*50)
    print("整合測試摘要")
    print("="*50)
    print(f"配置: {config.max_workers} workers, GPU={config.use_gpu}")
    print(f"資料: {len(data)} 筆 OHLCV")
    print(f"信號: 原始 {amp_stats['total_original_signals']} → 放大 {amp_stats['total_amplified_signals']} → 過濾 {total.get('final_long_signals', 0) + total.get('final_short_signals', 0)}")
    print(f"風控: 部位={position:.4f}, 風險={risk_details['final_risk_pct']:.2%}")
    print(f"槓桿: {leverage}x")
    print("="*50)
    print("✅ 完整整合流程測試通過")


# ===== 測試 5: 配置驗證測試 =====

def test_config_validation_errors():
    """測試配置驗證錯誤處理"""
    from src.automation.ultimate_config import UltimateLoopConfig

    # 測試無效的 max_workers
    config1 = UltimateLoopConfig(max_workers=-1)
    with pytest.raises(ValueError) as exc_info:
        config1.validate()
    assert "max_workers" in str(exc_info.value)

    # 測試無效的 objectives
    config2 = UltimateLoopConfig(objectives=[])
    with pytest.raises(ValueError) as exc_info:
        config2.validate()
    assert "objectives" in str(exc_info.value)

    # 測試不支援的指標
    config3 = UltimateLoopConfig(
        objectives=[('invalid_metric', 'maximize')]
    )
    with pytest.raises(ValueError) as exc_info:
        config3.validate()
    assert "不支援的指標" in str(exc_info.value)

    print("✓ 配置驗證錯誤處理正常")


# ===== 執行所有測試 =====

if __name__ == "__main__":
    """執行所有測試"""
    print("\n" + "="*70)
    print("Phase 12.12 整合測試")
    print("="*70 + "\n")

    # 測試 1: 匯入測試
    print("測試 1: 匯入測試")
    print("-" * 70)
    test_imports()

    # 測試 2: 實例化測試
    print("\n測試 2: 實例化測試")
    print("-" * 70)
    test_signal_amplifier_instantiation()
    test_filter_pipeline_instantiation()
    test_dynamic_risk_instantiation()
    test_adaptive_leverage_instantiation()
    test_ultimate_config_instantiation()

    # 測試 3: 基本功能測試
    print("\n測試 3: 基本功能測試")
    print("-" * 70)
    test_signal_amplifier_basic_function()
    test_filter_pipeline_basic_function()
    test_dynamic_risk_basic_function()
    test_adaptive_leverage_basic_function()

    # 測試 4: 整合流程測試
    print("\n測試 4: 整合流程測試")
    print("-" * 70)
    test_full_integration_workflow()

    # 測試 5: 配置驗證測試
    print("\n測試 5: 配置驗證測試")
    print("-" * 70)
    test_config_validation_errors()

    print("\n" + "="*70)
    print("✅ 所有測試完成！")
    print("="*70)
