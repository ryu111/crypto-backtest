"""
Memory MCP 整合獨立測試

只測試 MemoryIntegration 核心功能，不依賴其他模組。
"""

import pytest
import sys
from pathlib import Path

# 直接 import memory 模組，避免載入其他依賴
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src" / "learning"))

# 直接 import，避免透過 __init__
from memory import (
    MemoryIntegration,
    StrategyInsight,
    MarketInsight,
    TradingLesson,
    MemoryTags
)


class TestStrategyInsight:
    """測試 StrategyInsight"""

    def test_create_insight(self):
        """測試建立策略洞察"""
        insight = StrategyInsight(
            strategy_name="MA Cross",
            symbol="BTCUSDT",
            timeframe="4h",
            best_params={"fast": 10, "slow": 30},
            sharpe_ratio=1.85,
            total_return=0.456,
            max_drawdown=-0.12,
            win_rate=0.58
        )

        assert insight.strategy_name == "MA Cross"
        assert insight.sharpe_ratio == 1.85
        assert insight.created_at is not None

    def test_to_content_basic(self):
        """測試基本格式化"""
        insight = StrategyInsight(
            strategy_name="MA Cross",
            symbol="BTCUSDT",
            timeframe="4h",
            best_params={"fast": 10, "slow": 30},
            sharpe_ratio=1.85,
            total_return=0.456,
            max_drawdown=-0.12,
            win_rate=0.58
        )

        content = insight.to_content()

        assert "MA Cross" in content
        assert "BTCUSDT" in content
        assert "4h" in content
        assert "1.85" in content
        assert "45.6%" in content

    def test_to_content_with_wfa(self):
        """測試包含 WFA 資訊的格式化"""
        insight = StrategyInsight(
            strategy_name="MA Cross",
            symbol="BTCUSDT",
            timeframe="4h",
            best_params={"fast": 10, "slow": 30},
            sharpe_ratio=1.85,
            total_return=0.456,
            max_drawdown=-0.12,
            win_rate=0.58,
            wfa_efficiency=0.68,
            wfa_grade="A"
        )

        content = insight.to_content()

        assert "WFA Efficiency" in content
        assert "68%" in content
        assert "Grade: A" in content

    def test_format_params(self):
        """測試參數格式化"""
        insight = StrategyInsight(
            strategy_name="MA Cross",
            symbol="BTCUSDT",
            timeframe="4h",
            best_params={"fast": 10, "slow": 30, "stop": 2.0},
            sharpe_ratio=1.85,
            total_return=0.456,
            max_drawdown=-0.12,
            win_rate=0.58
        )

        formatted = insight._format_params()
        assert "fast 10" in formatted
        assert "slow 30" in formatted


class TestMarketInsight:
    """測試 MarketInsight"""

    def test_create_market_insight(self):
        """測試建立市場洞察"""
        insight = MarketInsight(
            symbol="BTCUSDT",
            timeframe="1h",
            market_type="volatile",
            observations="市場波動劇烈",
            recommended_strategies=["momentum", "breakout"]
        )

        assert insight.market_type == "volatile"
        assert len(insight.recommended_strategies) == 2

    def test_to_content(self):
        """測試格式化"""
        insight = MarketInsight(
            symbol="BTCUSDT",
            timeframe="1h",
            market_type="volatile",
            observations="市場波動劇烈",
            recommended_strategies=["momentum", "breakout"],
            warnings="不適合長持倉"
        )

        content = insight.to_content()

        assert "volatile" in content
        assert "momentum" in content
        assert "⚠️" in content

    def test_to_content_without_warnings(self):
        """測試沒有警告的格式化"""
        insight = MarketInsight(
            symbol="BTCUSDT",
            timeframe="1h",
            market_type="bull",
            observations="強勢上漲",
            recommended_strategies=["trend"]
        )

        content = insight.to_content()
        assert "⚠️" not in content


class TestTradingLesson:
    """測試 TradingLesson"""

    def test_create_lesson(self):
        """測試建立交易教訓"""
        lesson = TradingLesson(
            strategy_name="RSI",
            symbol="BTCUSDT",
            timeframe="1h",
            failure_type="overfitting",
            description="過擬合",
            symptoms="OOS 失效",
            prevention="增加驗證期"
        )

        assert lesson.failure_type == "overfitting"

    def test_to_content(self):
        """測試格式化"""
        lesson = TradingLesson(
            strategy_name="RSI",
            symbol="BTCUSDT",
            timeframe="1h",
            failure_type="overfitting",
            description="過擬合",
            symptoms="OOS 失效",
            prevention="增加驗證期",
            failed_params={"rsi_period": 7}
        )

        content = lesson.to_content()

        assert "OVERFITTING" in content
        assert "過擬合" in content
        assert "失敗參數" in content

    def test_all_failure_types(self):
        """測試所有失敗類型"""
        failure_types = ["overfitting", "poor_validation", "market_change", "parameter_instability"]

        for failure_type in failure_types:
            lesson = TradingLesson(
                strategy_name="Test",
                symbol="BTCUSDT",
                timeframe="1h",
                failure_type=failure_type,
                description="測試",
                symptoms="測試",
                prevention="測試"
            )
            assert lesson.failure_type == failure_type


class TestMemoryIntegration:
    """測試 MemoryIntegration"""

    def test_format_strategy_insight(self):
        """測試格式化策略洞察"""
        memory = MemoryIntegration()

        insight = StrategyInsight(
            strategy_name="MA Cross",
            symbol="BTCUSDT",
            timeframe="4h",
            best_params={"fast": 10, "slow": 30},
            sharpe_ratio=1.85,
            total_return=0.456,
            max_drawdown=-0.12,
            win_rate=0.58
        )

        content, metadata = memory.format_strategy_insight(insight)

        # 檢查 content
        assert "MA Cross" in content

        # 檢查 metadata
        assert metadata["type"] == "trading-insight"
        assert "tags" in metadata

        tags = metadata["tags"].split(",")
        assert MemoryTags.ASSET_CRYPTO in tags
        assert MemoryTags.STATUS_VALIDATED in tags

    def test_format_with_extra_tags(self):
        """測試額外標籤"""
        memory = MemoryIntegration()

        insight = StrategyInsight(
            strategy_name="MA Cross",
            symbol="BTCUSDT",
            timeframe="4h",
            best_params={"fast": 10, "slow": 30},
            sharpe_ratio=1.85,
            total_return=0.456,
            max_drawdown=-0.12,
            win_rate=0.58
        )

        content, metadata = memory.format_strategy_insight(
            insight,
            extra_tags=["custom-tag"]
        )

        tags = metadata["tags"].split(",")
        assert "custom-tag" in tags

    def test_format_market_insight(self):
        """測試格式化市場洞察"""
        memory = MemoryIntegration()

        insight = MarketInsight(
            symbol="ETHUSDT",
            timeframe="1h",
            market_type="volatile",
            observations="市場波動劇烈",
            recommended_strategies=["momentum"]
        )

        content, metadata = memory.format_market_insight(insight)

        assert metadata["type"] == "market-insight"
        tags = metadata["tags"].split(",")
        assert MemoryTags.ASSET_ETH in tags
        assert MemoryTags.MARKET_VOLATILE in tags

    def test_format_trading_lesson(self):
        """測試格式化交易教訓"""
        memory = MemoryIntegration()

        lesson = TradingLesson(
            strategy_name="RSI",
            symbol="BTCUSDT",
            timeframe="1h",
            failure_type="overfitting",
            description="過擬合",
            symptoms="OOS 失效",
            prevention="增加驗證期"
        )

        content, metadata = memory.format_trading_lesson(lesson)

        assert metadata["type"] == "trading-lesson"
        tags = metadata["tags"].split(",")
        assert MemoryTags.STATUS_FAILED in tags
        assert "overfitting" in tags

    def test_suggest_retrieval_query(self):
        """測試查詢建議"""
        memory = MemoryIntegration()

        # 完整參數
        query = memory.suggest_retrieval_query(
            strategy_name="MA Cross",
            symbol="BTCUSDT",
            timeframe="4h"
        )
        assert "MA Cross" in query
        assert "BTCUSDT" in query
        assert "4h" in query

        # 部分參數
        query = memory.suggest_retrieval_query(strategy_name="RSI")
        assert "RSI" in query

        # 無參數
        query = memory.suggest_retrieval_query()
        assert "trading insights" in query

    def test_suggest_tags_for_search(self):
        """測試標籤搜尋建議"""
        memory = MemoryIntegration()

        tags = memory.suggest_tags_for_search(
            strategy_type="trend",
            symbol="BTCUSDT",
            timeframe="4h",
            status=MemoryTags.STATUS_VALIDATED
        )

        assert "trend" in tags
        assert MemoryTags.ASSET_BTC in tags
        assert "4h" in tags
        assert MemoryTags.STATUS_VALIDATED in tags

    def test_suggest_best_params_query(self):
        """測試最佳參數查詢建議"""
        memory = MemoryIntegration()

        query = memory.suggest_best_params_query(
            strategy_type="ma-cross",
            symbol="BTCUSDT",
            timeframe="4h"
        )

        assert "ma-cross" in query
        assert "BTCUSDT" in query
        assert "4h" in query
        assert "Sharpe" in query

    def test_symbol_to_tag(self):
        """測試標的轉標籤"""
        memory = MemoryIntegration()

        assert memory._symbol_to_tag("BTCUSDT") == MemoryTags.ASSET_BTC
        assert memory._symbol_to_tag("ETHUSDT") == MemoryTags.ASSET_ETH
        assert memory._symbol_to_tag("SOLUSDT") == MemoryTags.ASSET_CRYPTO

        # 小寫
        assert memory._symbol_to_tag("btcusdt") == MemoryTags.ASSET_BTC

    def test_market_type_to_tag(self):
        """測試市場類型轉標籤"""
        memory = MemoryIntegration()

        assert memory._market_type_to_tag("bull") == MemoryTags.MARKET_BULL
        assert memory._market_type_to_tag("bear") == MemoryTags.MARKET_BEAR
        assert memory._market_type_to_tag("sideways") == MemoryTags.MARKET_SIDEWAYS
        assert memory._market_type_to_tag("ranging") == MemoryTags.MARKET_SIDEWAYS
        assert memory._market_type_to_tag("volatile") == MemoryTags.MARKET_VOLATILE

    def test_infer_strategy_tags(self):
        """測試策略標籤推斷"""
        memory = MemoryIntegration()

        # MA Cross
        tags = memory._infer_strategy_tags("MA Cross")
        assert MemoryTags.STRATEGY_MA_CROSS in tags
        assert MemoryTags.STRATEGY_TREND in tags

        # RSI
        tags = memory._infer_strategy_tags("RSI Momentum")
        assert MemoryTags.STRATEGY_MOMENTUM in tags

        # Bollinger
        tags = memory._infer_strategy_tags("Bollinger Bands Mean Reversion")
        assert MemoryTags.STRATEGY_MEAN_REVERSION in tags

        # Breakout
        tags = memory._infer_strategy_tags("Donchian Breakout")
        assert MemoryTags.STRATEGY_BREAKOUT in tags

        # Trend
        tags = memory._infer_strategy_tags("Trend Following")
        assert MemoryTags.STRATEGY_TREND in tags


class TestMemoryTags:
    """測試 MemoryTags 常數"""

    def test_asset_tags(self):
        """測試資產標籤"""
        assert MemoryTags.ASSET_BTC == "btc"
        assert MemoryTags.ASSET_ETH == "eth"
        assert MemoryTags.ASSET_CRYPTO == "crypto"

    def test_strategy_tags(self):
        """測試策略標籤"""
        assert MemoryTags.STRATEGY_TREND == "trend"
        assert MemoryTags.STRATEGY_MOMENTUM == "momentum"
        assert MemoryTags.STRATEGY_MEAN_REVERSION == "mean-reversion"

    def test_timeframe_tags(self):
        """測試時間框架標籤"""
        assert MemoryTags.TIMEFRAME_1H == "1h"
        assert MemoryTags.TIMEFRAME_4H == "4h"
        assert MemoryTags.TIMEFRAME_1D == "1d"

    def test_status_tags(self):
        """測試狀態標籤"""
        assert MemoryTags.STATUS_VALIDATED == "validated"
        assert MemoryTags.STATUS_TESTING == "testing"
        assert MemoryTags.STATUS_FAILED == "failed"

    def test_market_tags(self):
        """測試市場標籤"""
        assert MemoryTags.MARKET_BULL == "bull"
        assert MemoryTags.MARKET_BEAR == "bear"
        assert MemoryTags.MARKET_SIDEWAYS == "sideways"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
