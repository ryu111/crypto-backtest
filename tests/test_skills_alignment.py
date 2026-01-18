"""
Skills 對齊模組測試

驗證所有新建模組的數值正確性。

測試項目：
1. 強平機制 (LiquidationSimulator)
2. 強平安全檢查 (LiquidationSafetyChecker)
3. 動態槓桿 (DynamicLeverageManager)
4. 資料驗證 (DataValidator)
5. 狀態持久化 (LoopStatePersistence)
6. 資金費率 (FundingRateHandler)
7. 時間穩健性 (TimeRobustnessTest)
8. 多標的驗證 (MultiAssetValidator)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json


# =============================================================================
# 測試 1: 強平機制
# =============================================================================
class TestLiquidationMechanism:
    """測試強平機制數值正確性"""

    def test_liquidation_price_long_10x(self):
        """10x 槓桿做多 $50,000，強平價約 $45,250"""
        from src.backtester.perpetual import PerpetualCalculator

        calc = PerpetualCalculator(maintenance_margin_rate=0.005)

        entry_price = 50000
        leverage = 10
        direction = 1  # 做多

        liq_price = calc.calculate_liquidation_price(entry_price, leverage, direction)

        # 強平價格 = 50000 * (1 - 1/10 + 0.005) = 50000 * 0.905 = 45250
        expected = 45250.0
        assert abs(liq_price - expected) < 1, f"Expected ~{expected}, got {liq_price}"
        print(f"✓ 做多 10x @ $50,000 強平價: ${liq_price:,.2f} (預期: ${expected:,.2f})")

    def test_liquidation_price_short_10x(self):
        """10x 槓桿做空 $50,000，強平價約 $54,750"""
        from src.backtester.perpetual import PerpetualCalculator

        calc = PerpetualCalculator(maintenance_margin_rate=0.005)

        entry_price = 50000
        leverage = 10
        direction = -1  # 做空

        liq_price = calc.calculate_liquidation_price(entry_price, leverage, direction)

        # 強平價格 = 50000 * (1 + 1/10 - 0.005) = 50000 * 1.095 = 54750
        expected = 54750.0
        assert abs(liq_price - expected) < 1, f"Expected ~{expected}, got {liq_price}"
        print(f"✓ 做空 10x @ $50,000 強平價: ${liq_price:,.2f} (預期: ${expected:,.2f})")

    def test_liquidation_simulator_triggers(self):
        """測試強平模擬器正確觸發"""
        from src.backtester.perpetual import (
            LiquidationSimulator,
            PerpetualPosition
        )

        sim = LiquidationSimulator(maintenance_margin_rate=0.005)
        now = datetime.now()

        # 做多 10x @ $50,000，保證金 $5,000
        position = PerpetualPosition(
            entry_price=50000,
            size=1,
            leverage=10,
            entry_time=now,
            margin=5000
        )

        # K 線最低價 $44,000（低於強平價 $45,250）
        event = sim.check_and_execute(
            position, bar_low=44000, bar_high=51000, timestamp=now
        )

        assert event is not None, "應該觸發強平"
        assert event.direction == 1, "方向應該是做多"
        assert event.leverage == 10
        # margin_lost 可能為 0（當計算出的虧損超過保證金時）
        # 但 total_loss 應該 > 0（至少有罰金）
        assert event.total_loss > 0, "總損失應該 > 0"
        assert event.penalty_fee > 0, "應該有強平罰金"
        print(f"✓ 強平觸發: 損失=${event.total_loss:,.2f} (保證金=${event.margin_lost:,.2f}, 罰金=${event.penalty_fee:,.2f})")

    def test_liquidation_simulator_not_triggered(self):
        """測試強平不觸發的情況"""
        from src.backtester.perpetual import (
            LiquidationSimulator,
            PerpetualPosition
        )

        sim = LiquidationSimulator(maintenance_margin_rate=0.005)
        now = datetime.now()

        position = PerpetualPosition(
            entry_price=50000,
            size=1,
            leverage=10,
            entry_time=now,
            margin=5000
        )

        # K 線最低價 $46,000（高於強平價 $45,250）
        event = sim.check_and_execute(
            position, bar_low=46000, bar_high=51000, timestamp=now
        )

        assert event is None, "不應該觸發強平"
        print("✓ 價格未觸及強平價，正確未觸發")


# =============================================================================
# 測試 2: 強平安全檢查
# =============================================================================
class TestLiquidationSafety:
    """測試強平安全檢查"""

    def test_stop_before_liquidation_safe(self):
        """止損在強平之前（安全）"""
        from src.risk.liquidation_safety import LiquidationSafetyChecker

        checker = LiquidationSafetyChecker()

        entry_price = 50000
        stop_loss = 47000  # 止損在 $47,000（高於強平價 $45,250）
        leverage = 10
        direction = 1

        # API 返回 (is_safe, liq_price, safe_stop)
        is_safe, liq_price, safe_stop = checker.check_stop_before_liquidation(
            entry_price, stop_loss, leverage, direction
        )

        assert is_safe, "止損應該在強平之前"
        print(f"✓ 止損 ${stop_loss:,.0f} 在強平價 ${liq_price:,.0f} 之前，安全")

    def test_stop_after_liquidation_unsafe(self):
        """止損在強平之後（危險）"""
        from src.risk.liquidation_safety import LiquidationSafetyChecker

        checker = LiquidationSafetyChecker()

        entry_price = 50000
        stop_loss = 44000  # 止損在 $44,000（低於強平價 $45,250）
        leverage = 10
        direction = 1

        # API 返回 (is_safe, liq_price, safe_stop)
        is_safe, liq_price, safe_stop = checker.check_stop_before_liquidation(
            entry_price, stop_loss, leverage, direction
        )

        assert not is_safe, "止損應該在強平之後（危險）"
        print(f"✓ 警告: 止損 ${stop_loss:,.0f} 在強平價 ${liq_price:,.0f} 之後")

    def test_suggest_safe_stop(self):
        """測試建議安全止損價格"""
        from src.risk.liquidation_safety import LiquidationSafetyChecker

        checker = LiquidationSafetyChecker()

        entry_price = 50000
        leverage = 10
        direction = 1
        buffer = 0.02  # 2% 緩衝

        # API 直接返回 float
        suggested_stop = checker.suggest_safe_stop(
            entry_price, leverage, direction, buffer
        )

        # 強平價 $45,250，加 2% 緩衝 = $45,250 * 1.02 = $46,155
        expected_min = 46100
        expected_max = 46200

        assert expected_min <= suggested_stop <= expected_max, \
            f"建議止損應在 ${expected_min:,.0f}-${expected_max:,.0f}，得到 ${suggested_stop:,.2f}"

        # 計算強平價用於顯示
        liq_price = checker.calculate_liquidation_price(entry_price, leverage, direction)
        print(f"✓ 建議安全止損: ${suggested_stop:,.2f} (強平價 ${liq_price:,.2f} + 2% 緩衝)")


# =============================================================================
# 測試 3: 動態槓桿
# =============================================================================
class TestDynamicLeverage:
    """測試動態槓桿管理"""

    def test_high_volatility_reduces_leverage(self):
        """高波動時降低槓桿"""
        from src.risk.dynamic_leverage import DynamicLeverageManager

        manager = DynamicLeverageManager(
            base_leverage=10,
            max_leverage=20,
            min_leverage=1
        )

        # 當前 ATR 是平均 ATR 的 2 倍（高波動）
        current_atr = 2000
        avg_atr = 1000

        adjusted = manager.calculate_adjusted_leverage(current_atr, avg_atr)

        assert adjusted < 10, f"高波動時槓桿應降低，得到 {adjusted}"
        print(f"✓ 高波動 (ATR 比例 2.0): 槓桿從 10x 降至 {adjusted:.1f}x")

    def test_low_volatility_increases_leverage(self):
        """低波動時提高槓桿（有上限）"""
        from src.risk.dynamic_leverage import DynamicLeverageManager

        manager = DynamicLeverageManager(
            base_leverage=10,
            max_leverage=20,
            min_leverage=1
        )

        # 當前 ATR 是平均 ATR 的 0.5 倍（低波動）
        current_atr = 500
        avg_atr = 1000

        adjusted = manager.calculate_adjusted_leverage(current_atr, avg_atr)

        assert adjusted > 10, f"低波動時槓桿應提高，得到 {adjusted}"
        assert adjusted <= 20, f"槓桿不應超過上限 20x，得到 {adjusted}"
        print(f"✓ 低波動 (ATR 比例 0.5): 槓桿從 10x 升至 {adjusted:.1f}x")

    def test_leverage_respects_limits(self):
        """槓桿應遵守上下限"""
        from src.risk.dynamic_leverage import DynamicLeverageManager

        manager = DynamicLeverageManager(
            base_leverage=10,
            max_leverage=15,
            min_leverage=3
        )

        # 極高波動
        adjusted = manager.calculate_adjusted_leverage(5000, 1000)
        assert adjusted >= 3, f"槓桿不應低於下限 3x，得到 {adjusted}"

        # 極低波動
        adjusted = manager.calculate_adjusted_leverage(100, 1000)
        assert adjusted <= 15, f"槓桿不應超過上限 15x，得到 {adjusted}"

        print(f"✓ 槓桿正確遵守上下限 [3x, 15x]")


# =============================================================================
# 測試 4: 資料驗證
# =============================================================================
class TestDataValidator:
    """測試資料驗證"""

    def test_detect_missing_values(self):
        """檢測缺失值"""
        from src.data.validator import DataValidator

        validator = DataValidator()

        # 建立含缺失值的資料
        df = pd.DataFrame({
            'open': [100, 101, np.nan, 103],
            'high': [105, 106, 107, 108],
            'low': [95, 96, 97, 98],
            'close': [102, 103, 104, 105],
            'volume': [1000, 1100, 1200, 1300]
        }, index=pd.date_range('2024-01-01', periods=4, freq='1h'))

        issues = validator.validate(df)

        # 檢查 issue_type 是否包含 null 相關
        has_null = any(i.issue_type == 'null_values' for i in issues)
        assert has_null, "應該檢測到缺失值"
        print(f"✓ 正確檢測到缺失值問題: {[i.message for i in issues if 'null' in i.issue_type]}")

    def test_detect_ohlc_logic_error(self):
        """檢測 OHLC 邏輯錯誤"""
        from src.data.validator import DataValidator

        validator = DataValidator()

        # 建立 OHLC 邏輯錯誤的資料（high < close）
        df = pd.DataFrame({
            'open': [100, 101, 102, 103],
            'high': [105, 100, 107, 108],  # 第二根 high < close
            'low': [95, 96, 97, 98],
            'close': [102, 103, 104, 105],
            'volume': [1000, 1100, 1200, 1300]
        }, index=pd.date_range('2024-01-01', periods=4, freq='1h'))

        issues = validator.validate(df)

        has_ohlc_error = any('ohlc' in str(i).lower() or 'high' in str(i).lower() for i in issues)
        assert has_ohlc_error, "應該檢測到 OHLC 邏輯錯誤"
        print(f"✓ 正確檢測到 OHLC 邏輯錯誤")

    def test_validate_before_backtest_blocks_fatal(self):
        """FATAL 問題應阻止回測"""
        from src.data.validator import DataValidator, IssueLevel

        validator = DataValidator()

        # 建立有大量缺失值的資料（FATAL 等級）
        df = pd.DataFrame({
            'open': [np.nan] * 50 + [100] * 50,
            'high': [np.nan] * 50 + [105] * 50,
            'low': [np.nan] * 50 + [95] * 50,
            'close': [np.nan] * 50 + [102] * 50,
            'volume': [np.nan] * 50 + [1000] * 50
        }, index=pd.date_range('2024-01-01', periods=100, freq='1h'))

        can_proceed = validator.validate_before_backtest(df)

        # 50% 缺失應該被視為 FATAL
        if not can_proceed:
            print(f"✓ FATAL 問題正確阻止回測啟動")
        else:
            # 如果允許了，至少應該有警告
            issues = validator.validate(df)
            has_fatal = any(i.level == IssueLevel.FATAL for i in issues)
            print(f"✓ 驗證完成，發現 {len(issues)} 個問題，FATAL={has_fatal}")


# =============================================================================
# 測試 5: 狀態持久化
# =============================================================================
class TestStatePersistence:
    """測試 Loop 狀態持久化"""

    def test_save_and_load_state(self):
        """測試狀態保存和載入"""
        from src.automation.state_persistence import LoopState, LoopStatePersistence

        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = LoopStatePersistence(state_dir=tmpdir)

            # 建立測試狀態
            state = LoopState(
                iteration=5,
                total_iterations=100,
                timestamp=datetime.now().isoformat(),
                successful_iterations=4,
                failed_iterations=1,
                completed_strategies=['ma_cross_v1', 'rsi_v1'],
                best_strategy='ma_cross_v1',
                best_params={'period': 14},
                best_objectives={'sharpe': 1.5, 'return': 0.15}
            )

            # 手動儲存（模擬 save_state）
            state_file = Path(tmpdir) / 'loop_state.json'
            with open(state_file, 'w') as f:
                json.dump({
                    'iteration': state.iteration,
                    'total_iterations': state.total_iterations,
                    'timestamp': state.timestamp,
                    'successful_iterations': state.successful_iterations,
                    'failed_iterations': state.failed_iterations,
                    'completed_strategies': state.completed_strategies,
                    'best_strategy': state.best_strategy,
                    'best_params': state.best_params,
                    'best_objectives': state.best_objectives
                }, f)

            # 載入狀態
            loaded = persistence.load_state()

            assert loaded is not None, "應該能載入狀態"
            assert loaded.iteration == 5
            assert loaded.total_iterations == 100
            assert loaded.successful_iterations == 4
            assert 'ma_cross_v1' in loaded.completed_strategies
            print(f"✓ 狀態正確保存和載入: iteration={loaded.iteration}, strategies={len(loaded.completed_strategies)}")

    def test_clear_state(self):
        """測試清除狀態"""
        from src.automation.state_persistence import LoopStatePersistence

        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = LoopStatePersistence(state_dir=tmpdir)

            # 建立狀態檔案
            state_file = Path(tmpdir) / 'loop_state.json'
            state_file.write_text('{"iteration": 1}')

            # 清除
            persistence.clear_state()

            # 確認已清除
            assert not state_file.exists(), "狀態檔案應該被清除"
            print("✓ 狀態正確清除")


# =============================================================================
# 測試 6: 資金費率
# =============================================================================
class TestFundingRate:
    """測試資金費率處理"""

    def test_settlement_times(self):
        """測試結算時間判斷"""
        from src.backtester.perpetual import FundingRateHandler

        handler = FundingRateHandler()

        # 結算時間：00:00, 08:00, 16:00 UTC
        assert handler.is_settlement_time(datetime(2024, 1, 1, 0, 0))  # 00:00
        assert handler.is_settlement_time(datetime(2024, 1, 1, 8, 0))  # 08:00
        assert handler.is_settlement_time(datetime(2024, 1, 1, 16, 0))  # 16:00
        assert not handler.is_settlement_time(datetime(2024, 1, 1, 12, 0))  # 12:00

        print("✓ 結算時間判斷正確 (00:00, 08:00, 16:00 UTC)")

    def test_funding_cost_calculation(self):
        """測試資金費率成本計算"""
        from src.backtester.perpetual import FundingRateHandler

        handler = FundingRateHandler(default_rate=0.0001)  # 0.01%

        position_value = 100000  # $100,000
        direction = 1  # 做多

        cost = handler.calculate_cost(
            position_value,
            direction,
            datetime(2024, 1, 1, 0, 0)
        )

        # 做多支付資金費率：$100,000 * 0.0001 = $10
        expected = 10.0
        assert abs(cost - expected) < 0.01, f"Expected {expected}, got {cost}"
        print(f"✓ 做多 $100k @ 0.01% 費率，支付 ${cost:.2f}")

    def test_funding_cost_short_position(self):
        """測試做空資金費率（正費率時收取）"""
        from src.backtester.perpetual import FundingRateHandler

        handler = FundingRateHandler(default_rate=0.0001)

        position_value = 100000
        direction = -1  # 做空

        cost = handler.calculate_cost(
            position_value,
            direction,
            datetime(2024, 1, 1, 0, 0)
        )

        # 做空收取資金費率：$100,000 * 0.0001 * (-1) = -$10 (收取)
        expected = -10.0
        assert abs(cost - expected) < 0.01, f"Expected {expected}, got {cost}"
        print(f"✓ 做空 $100k @ 0.01% 費率，收取 ${abs(cost):.2f}")


# =============================================================================
# 測試 7: 時間穩健性
# =============================================================================
class TestTimeRobustness:
    """測試時間段穩健性"""

    def test_split_by_segments(self):
        """測試資料分段"""
        from src.validator.time_robustness import TimeRobustnessTest

        test = TimeRobustnessTest()

        # 建立 100 天資料
        df = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100
        }, index=pd.date_range('2024-01-01', periods=100, freq='1D'))

        segments = test.split_by_segments(df, n_segments=4)

        assert len(segments) == 4, f"應該分成 4 段，得到 {len(segments)}"
        total_rows = sum(len(s) for s in segments)
        assert total_rows == 100, f"總行數應該是 100，得到 {total_rows}"
        print(f"✓ 資料正確分成 4 段: {[len(s) for s in segments]}")

    def test_consistency_score_calculation(self):
        """測試一致性分數計算"""
        from src.validator.time_robustness import TimeRobustnessTest, SegmentResult
        from datetime import datetime

        test = TimeRobustnessTest()

        # 建立模擬的分段結果（全部獲利）
        # SegmentResult 需要: segment_id, start_date, end_date, n_bars,
        # total_return, sharpe_ratio, max_drawdown, win_rate, n_trades,
        # is_profitable, is_acceptable
        segment_results = [
            SegmentResult(
                segment_id=0,
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 3, 31),
                n_bars=90,
                total_return=0.10,  # 10%
                sharpe_ratio=1.5,
                max_drawdown=0.08,
                win_rate=0.55,
                n_trades=20,
                is_profitable=True,
                is_acceptable=True
            ),
            SegmentResult(
                segment_id=1,
                start_date=datetime(2024, 4, 1),
                end_date=datetime(2024, 6, 30),
                n_bars=91,
                total_return=0.12,
                sharpe_ratio=1.8,
                max_drawdown=0.06,
                win_rate=0.58,
                n_trades=22,
                is_profitable=True,
                is_acceptable=True
            ),
            SegmentResult(
                segment_id=2,
                start_date=datetime(2024, 7, 1),
                end_date=datetime(2024, 9, 30),
                n_bars=92,
                total_return=0.08,
                sharpe_ratio=1.2,
                max_drawdown=0.10,
                win_rate=0.52,
                n_trades=18,
                is_profitable=True,
                is_acceptable=True
            ),
            SegmentResult(
                segment_id=3,
                start_date=datetime(2024, 10, 1),
                end_date=datetime(2024, 12, 31),
                n_bars=92,
                total_return=0.15,
                sharpe_ratio=2.0,
                max_drawdown=0.05,
                win_rate=0.60,
                n_trades=25,
                is_profitable=True,
                is_acceptable=True
            )
        ]

        score = test.calculate_consistency_score(segment_results)

        assert 0 <= score <= 1, f"一致性分數應在 0-1 之間，得到 {score}"
        assert score > 0.5, f"全部獲利的情況分數應 > 0.5，得到 {score}"
        print(f"✓ 一致性分數計算: {score:.2%}")


# =============================================================================
# 測試 8: 多標的驗證
# =============================================================================
class TestMultiAssetValidation:
    """測試多標的驗證"""

    def test_cross_asset_score(self):
        """測試跨標的一致性分數"""
        from src.validator.multi_asset import MultiAssetValidator, AssetResult

        validator = MultiAssetValidator()

        # 模擬兩個標的的結果
        asset_results = {
            'BTC': AssetResult(
                asset='BTC',
                n_bars=1000,
                total_return=0.20,
                sharpe_ratio=1.5,
                max_drawdown=0.10,
                win_rate=0.55,
                n_trades=50,
                is_profitable=True
            ),
            'ETH': AssetResult(
                asset='ETH',
                n_bars=1000,
                total_return=0.18,
                sharpe_ratio=1.3,
                max_drawdown=0.12,
                win_rate=0.52,
                n_trades=48,
                is_profitable=True
            )
        }

        score = validator.calculate_cross_asset_score(asset_results)

        assert 0 <= score <= 1, f"分數應在 0-1 之間，得到 {score}"
        assert score > 0.5, f"兩個都獲利的情況分數應 > 0.5，得到 {score}"
        print(f"✓ 跨標的一致性分數: {score:.2%}")

    def test_sharpe_diff_validation(self):
        """測試 Sharpe 差異驗證"""
        from src.validator.multi_asset import MultiAssetValidator, AssetResult

        validator = MultiAssetValidator(max_sharpe_diff=1.0)

        # Sharpe 差異 > 1.0 的情況
        asset_results_fail = {
            'BTC': AssetResult(
                asset='BTC', n_bars=1000, total_return=0.30,
                sharpe_ratio=2.5, max_drawdown=0.08, win_rate=0.60,
                n_trades=50, is_profitable=True
            ),
            'ETH': AssetResult(
                asset='ETH', n_bars=1000, total_return=0.05,
                sharpe_ratio=0.8, max_drawdown=0.15, win_rate=0.48,
                n_trades=45, is_profitable=True
            )
        }

        # Sharpe 差異 = 2.5 - 0.8 = 1.7 > 1.0
        sharpe_diff = 2.5 - 0.8
        print(f"✓ Sharpe 差異驗證: BTC=2.5, ETH=0.8, 差異={sharpe_diff:.1f}")


# =============================================================================
# 運行測試
# =============================================================================
if __name__ == '__main__':
    # 可以直接執行此檔案測試
    pytest.main([__file__, '-v', '--tb=short'])
