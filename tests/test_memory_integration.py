"""
Memory MCP 整合測試

測試 MemoryIntegration 的格式化和查詢建議功能。
"""

import pytest
from datetime import datetime

from src.learning.memory import (
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

    def test_suggest_retrieval_query(self):
        """測試查詢建議"""
        memory = MemoryIntegration()

        query = memory.suggest_retrieval_query(
            strategy_name="MA Cross",
            symbol="BTCUSDT",
            timeframe="4h"
        )

        assert "MA Cross" in query
        assert "BTCUSDT" in query
        assert "4h" in query

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

    def test_market_type_to_tag(self):
        """測試市場類型轉標籤"""
        memory = MemoryIntegration()

        assert memory._market_type_to_tag("bull") == MemoryTags.MARKET_BULL
        assert memory._market_type_to_tag("bear") == MemoryTags.MARKET_BEAR
        assert memory._market_type_to_tag("sideways") == MemoryTags.MARKET_SIDEWAYS
        assert memory._market_type_to_tag("volatile") == MemoryTags.MARKET_VOLATILE

    def test_infer_strategy_tags(self):
        """測試策略標籤推斷"""
        memory = MemoryIntegration()

        # MA Cross 應該推斷為 trend + ma-cross
        tags = memory._infer_strategy_tags("MA Cross")
        assert MemoryTags.STRATEGY_MA_CROSS in tags
        assert MemoryTags.STRATEGY_TREND in tags

        # RSI 應該推斷為 momentum
        tags = memory._infer_strategy_tags("RSI Momentum")
        assert MemoryTags.STRATEGY_MOMENTUM in tags

        # Bollinger 應該推斷為 mean-reversion
        tags = memory._infer_strategy_tags("Bollinger Bands")
        assert MemoryTags.STRATEGY_MEAN_REVERSION in tags


class TestConvenienceFunctions:
    """測試便利函數"""

    def test_create_memory_integration(self):
        """測試建立 Memory 實例"""
        from src.learning.memory import create_memory_integration

        memory = create_memory_integration()
        assert isinstance(memory, MemoryIntegration)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
