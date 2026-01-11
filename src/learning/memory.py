"""
Memory MCP 整合 - AI 學習系統的跨專案知識存儲

透過 Memory MCP 服務進行語義化存儲和檢索交易洞察、市場經驗、失敗教訓。
這是 wrapper 類別，實際 MCP 呼叫由外部 Claude 執行。
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Literal, TYPE_CHECKING
from datetime import datetime
import json

# 使用 TYPE_CHECKING 避免循環 import，並支援獨立測試
if TYPE_CHECKING:
    from ..optimizer.bayesian import OptimizationResult
    from ..validator.walk_forward import WalkForwardResult


# 標籤常數
class MemoryTags:
    """記憶體標籤定義"""

    # 資產類型
    ASSET_BTC = "btc"
    ASSET_ETH = "eth"
    ASSET_CRYPTO = "crypto"

    # 策略類型
    STRATEGY_TREND = "trend"
    STRATEGY_MOMENTUM = "momentum"
    STRATEGY_MEAN_REVERSION = "mean-reversion"
    STRATEGY_BREAKOUT = "breakout"
    STRATEGY_MA_CROSS = "ma-cross"

    # 時間框架
    TIMEFRAME_1H = "1h"
    TIMEFRAME_4H = "4h"
    TIMEFRAME_1D = "1d"

    # 狀態
    STATUS_VALIDATED = "validated"
    STATUS_TESTING = "testing"
    STATUS_FAILED = "failed"
    STATUS_OVERFITTED = "overfitted"

    # 市場狀態
    MARKET_BULL = "bull"
    MARKET_BEAR = "bear"
    MARKET_SIDEWAYS = "sideways"
    MARKET_VOLATILE = "volatile"


@dataclass
class StrategyInsight:
    """策略洞察"""

    strategy_name: str
    symbol: str
    timeframe: str

    # 最佳參數
    best_params: Dict[str, Any]

    # 績效指標
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float

    # WFA 驗證
    wfa_efficiency: Optional[float] = None
    wfa_grade: Optional[str] = None

    # 適用條件
    market_conditions: Optional[str] = None
    notes: Optional[str] = None

    # 元數據
    created_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

    def to_content(self) -> str:
        """格式化為存儲內容"""
        parts = [
            f"{self.strategy_name} 策略最佳實踐 ({self.symbol} {self.timeframe}):",
            f"- 參數: {self._format_params()}",
            f"- Sharpe: {self.sharpe_ratio:.2f}, Return: {self.total_return:.1%}, MDD: {self.max_drawdown:.1%}",
            f"- 勝率: {self.win_rate:.1%}"
        ]

        if self.wfa_efficiency and self.wfa_grade:
            parts.append(f"- WFA Efficiency: {self.wfa_efficiency:.0%}, Grade: {self.wfa_grade}")

        if self.market_conditions:
            parts.append(f"- 適用: {self.market_conditions}")

        if self.notes:
            parts.append(f"- 備註: {self.notes}")

        return "\n".join(parts)

    def _format_params(self) -> str:
        """格式化參數為可讀字串"""
        return ", ".join(f"{k} {v}" for k, v in self.best_params.items())


@dataclass
class MarketInsight:
    """市場洞察"""

    symbol: str
    timeframe: str

    # 市場特性
    market_type: str  # bull/bear/sideways/volatile

    # 觀察
    observations: str

    # 建議策略
    recommended_strategies: List[str]

    # 警告
    warnings: Optional[str] = None

    created_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

    def to_content(self) -> str:
        """格式化為存儲內容"""
        parts = [
            f"市場洞察 ({self.symbol} {self.timeframe} - {self.market_type}):",
            f"- 觀察: {self.observations}",
            f"- 建議策略: {', '.join(self.recommended_strategies)}"
        ]

        if self.warnings:
            parts.append(f"- ⚠️ 警告: {self.warnings}")

        return "\n".join(parts)


@dataclass
class TradingLesson:
    """交易教訓"""

    strategy_name: str
    symbol: str
    timeframe: str

    # 失敗原因
    failure_type: Literal["overfitting", "poor_validation", "market_change", "parameter_instability"]
    description: str

    # 症狀
    symptoms: str

    # 避免方法
    prevention: str

    # 相關參數（可選）
    failed_params: Optional[Dict[str, Any]] = None

    created_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

    def to_content(self) -> str:
        """格式化為存儲內容"""
        parts = [
            f"交易教訓 - {self.failure_type.upper()} ({self.strategy_name} {self.symbol} {self.timeframe}):",
            f"- 問題: {self.description}",
            f"- 症狀: {self.symptoms}",
            f"- 預防: {self.prevention}"
        ]

        if self.failed_params:
            parts.append(f"- 失敗參數: {self.failed_params}")

        return "\n".join(parts)


class MemoryIntegration:
    """
    Memory MCP 整合

    提供格式化方法和查詢建議，實際 MCP 呼叫由外部 Claude 執行。

    使用範例:
        memory = MemoryIntegration()

        # 存儲策略洞察
        insight = StrategyInsight(
            strategy_name="MA Cross",
            symbol="BTCUSDT",
            timeframe="4h",
            best_params={"fast": 10, "slow": 30, "atr_stop": 2.0},
            sharpe_ratio=1.85,
            total_return=0.456,
            max_drawdown=-0.12,
            win_rate=0.58,
            wfa_efficiency=0.68,
            wfa_grade="A",
            market_conditions="趨勢明確市場"
        )

        content, metadata = memory.format_strategy_insight(insight)
        print(f"請使用 Memory MCP 存儲:")
        print(f"content: {content}")
        print(f"metadata: {metadata}")

        # 檢索建議
        query = memory.suggest_retrieval_query("MA Cross", "BTCUSDT", "4h")
        print(f"建議查詢: {query}")
    """

    def __init__(self):
        """初始化 Memory 整合"""
        pass

    # ========== 存儲方法 ==========

    def format_strategy_insight(
        self,
        insight: StrategyInsight,
        extra_tags: Optional[List[str]] = None
    ) -> tuple[str, Dict[str, Any]]:
        """
        格式化策略洞察為 Memory MCP 存儲格式

        Args:
            insight: StrategyInsight 物件
            extra_tags: 額外標籤

        Returns:
            (content, metadata) - 適合 store_memory 的格式
        """
        content = insight.to_content()

        # 建立標籤
        tags = [
            MemoryTags.ASSET_CRYPTO,
            self._symbol_to_tag(insight.symbol),
            insight.timeframe,
            MemoryTags.STATUS_VALIDATED
        ]

        # 根據策略名稱推斷策略類型
        strategy_tags = self._infer_strategy_tags(insight.strategy_name)
        tags.extend(strategy_tags)

        if extra_tags:
            tags.extend(extra_tags)

        metadata = {
            "tags": ",".join(tags),
            "type": "trading-insight"
        }

        return content, metadata

    def format_market_insight(
        self,
        insight: MarketInsight,
        extra_tags: Optional[List[str]] = None
    ) -> tuple[str, Dict[str, Any]]:
        """
        格式化市場洞察為 Memory MCP 存儲格式

        Args:
            insight: MarketInsight 物件
            extra_tags: 額外標籤

        Returns:
            (content, metadata)
        """
        content = insight.to_content()

        tags = [
            MemoryTags.ASSET_CRYPTO,
            self._symbol_to_tag(insight.symbol),
            insight.timeframe,
            self._market_type_to_tag(insight.market_type)
        ]

        if extra_tags:
            tags.extend(extra_tags)

        metadata = {
            "tags": ",".join(tags),
            "type": "market-insight"
        }

        return content, metadata

    def format_trading_lesson(
        self,
        lesson: TradingLesson,
        extra_tags: Optional[List[str]] = None
    ) -> tuple[str, Dict[str, Any]]:
        """
        格式化交易教訓為 Memory MCP 存儲格式

        Args:
            lesson: TradingLesson 物件
            extra_tags: 額外標籤

        Returns:
            (content, metadata)
        """
        content = lesson.to_content()

        tags = [
            MemoryTags.ASSET_CRYPTO,
            self._symbol_to_tag(lesson.symbol),
            lesson.timeframe,
            MemoryTags.STATUS_FAILED,
            lesson.failure_type
        ]

        strategy_tags = self._infer_strategy_tags(lesson.strategy_name)
        tags.extend(strategy_tags)

        if extra_tags:
            tags.extend(extra_tags)

        metadata = {
            "tags": ",".join(tags),
            "type": "trading-lesson"
        }

        return content, metadata

    # ========== 便利方法：從實驗結果建立 ==========

    def create_insight_from_optimization(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        opt_result: "OptimizationResult",
        wfa_result: Optional["WalkForwardResult"] = None,
        market_conditions: Optional[str] = None,
        notes: Optional[str] = None
    ) -> StrategyInsight:
        """
        從優化結果建立策略洞察

        Args:
            strategy_name: 策略名稱
            symbol: 交易標的
            timeframe: 時間框架
            opt_result: 優化結果
            wfa_result: Walk-Forward 驗證結果（可選）
            market_conditions: 市場條件描述
            notes: 額外備註

        Returns:
            StrategyInsight 物件
        """
        backtest = opt_result.best_backtest_result

        insight = StrategyInsight(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            best_params=opt_result.best_params,
            sharpe_ratio=backtest.sharpe_ratio,
            total_return=backtest.total_return,
            max_drawdown=backtest.max_drawdown,
            win_rate=backtest.win_rate,
            market_conditions=market_conditions,
            notes=notes
        )

        # 如果有 WFA 結果
        if wfa_result:
            insight.wfa_efficiency = wfa_result.oos_efficiency
            insight.wfa_grade = wfa_result.grade

        return insight

    def create_lesson_from_validation_failure(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        opt_result: "OptimizationResult",
        wfa_result: "WalkForwardResult",
        failure_description: str,
        prevention_advice: str
    ) -> TradingLesson:
        """
        從驗證失敗建立交易教訓

        Args:
            strategy_name: 策略名稱
            symbol: 交易標的
            timeframe: 時間框架
            opt_result: 優化結果
            wfa_result: Walk-Forward 驗證結果
            failure_description: 失敗描述
            prevention_advice: 預防建議

        Returns:
            TradingLesson 物件
        """
        # 判斷失敗類型
        if wfa_result.oos_efficiency < 0.3:
            failure_type = "overfitting"
            symptoms = f"OOS Efficiency 僅 {wfa_result.oos_efficiency:.1%}，嚴重過擬合"
        elif wfa_result.grade in ["D", "F"]:
            failure_type = "poor_validation"
            symptoms = f"WFA Grade {wfa_result.grade}，樣本外績效不佳"
        else:
            failure_type = "parameter_instability"
            symptoms = "參數在不同時期表現不穩定"

        lesson = TradingLesson(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            failure_type=failure_type,
            description=failure_description,
            symptoms=symptoms,
            prevention=prevention_advice,
            failed_params=opt_result.best_params
        )

        return lesson

    # ========== 檢索建議方法 ==========

    def suggest_retrieval_query(
        self,
        strategy_name: Optional[str] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        market_type: Optional[str] = None
    ) -> str:
        """
        產生檢索查詢建議

        Args:
            strategy_name: 策略名稱
            symbol: 交易標的
            timeframe: 時間框架
            market_type: 市場類型

        Returns:
            建議的查詢字串
        """
        parts = []

        if strategy_name:
            parts.append(strategy_name)

        if symbol:
            parts.append(symbol)

        if timeframe:
            parts.append(f"{timeframe} timeframe")

        if market_type:
            parts.append(f"{market_type} market")

        if not parts:
            return "trading insights best practices"

        return " ".join(parts) + " best parameters validated"

    def suggest_tags_for_search(
        self,
        strategy_type: Optional[str] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[str]:
        """
        產生標籤搜尋建議

        Args:
            strategy_type: 策略類型（trend/momentum/mean-reversion）
            symbol: 交易標的
            timeframe: 時間框架
            status: 狀態（validated/failed）

        Returns:
            建議的標籤列表
        """
        tags = []

        if strategy_type:
            tags.append(strategy_type)

        if symbol:
            tags.append(self._symbol_to_tag(symbol))

        if timeframe:
            tags.append(timeframe)

        if status:
            tags.append(status)

        return tags

    def suggest_best_params_query(
        self,
        strategy_type: str,
        symbol: str,
        timeframe: str
    ) -> str:
        """
        產生「取得最佳參數」的查詢建議

        Args:
            strategy_type: 策略類型
            symbol: 交易標的
            timeframe: 時間框架

        Returns:
            建議的查詢字串
        """
        return (
            f"Best validated parameters for {strategy_type} strategy "
            f"on {symbol} {timeframe} with high Sharpe ratio"
        )

    # ========== 輔助方法 ==========

    def _symbol_to_tag(self, symbol: str) -> str:
        """將交易標的轉為標籤"""
        symbol_upper = symbol.upper()

        if "BTC" in symbol_upper:
            return MemoryTags.ASSET_BTC
        elif "ETH" in symbol_upper:
            return MemoryTags.ASSET_ETH
        else:
            return MemoryTags.ASSET_CRYPTO

    def _market_type_to_tag(self, market_type: str) -> str:
        """將市場類型轉為標籤"""
        market_lower = market_type.lower()

        if "bull" in market_lower:
            return MemoryTags.MARKET_BULL
        elif "bear" in market_lower:
            return MemoryTags.MARKET_BEAR
        elif "sideways" in market_lower or "ranging" in market_lower:
            return MemoryTags.MARKET_SIDEWAYS
        elif "volatile" in market_lower:
            return MemoryTags.MARKET_VOLATILE
        else:
            return market_type.lower()

    def _infer_strategy_tags(self, strategy_name: str) -> List[str]:
        """從策略名稱推斷策略類型標籤"""
        name_lower = strategy_name.lower()
        tags = []

        if "ma" in name_lower or "moving average" in name_lower or "cross" in name_lower:
            tags.append(MemoryTags.STRATEGY_MA_CROSS)
            tags.append(MemoryTags.STRATEGY_TREND)

        if "trend" in name_lower:
            tags.append(MemoryTags.STRATEGY_TREND)

        if "momentum" in name_lower or "rsi" in name_lower:
            tags.append(MemoryTags.STRATEGY_MOMENTUM)

        if "mean" in name_lower or "reversion" in name_lower or "bollinger" in name_lower:
            tags.append(MemoryTags.STRATEGY_MEAN_REVERSION)

        if "breakout" in name_lower or "donchian" in name_lower:
            tags.append(MemoryTags.STRATEGY_BREAKOUT)

        return tags

    # ========== 使用範例產生方法 ==========

    def print_storage_example(
        self,
        content: str,
        metadata: Dict[str, Any]
    ):
        """
        印出存儲範例（供 Claude 參考）

        Args:
            content: 記憶內容
            metadata: 元數據
        """
        print("="*60)
        print("請使用 Memory MCP 存儲以下內容:")
        print("="*60)
        print(f"Content:\n{content}\n")
        print(f"Metadata:\n{json.dumps(metadata, indent=2)}\n")
        print("MCP 呼叫範例:")
        print(f"""
mcp__memory-service__store_memory(
    content='''{content}''',
    metadata={json.dumps(metadata)}
)
        """)
        print("="*60)

    def print_retrieval_example(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        n_results: int = 5
    ):
        """
        印出檢索範例（供 Claude 參考）

        Args:
            query: 查詢字串
            tags: 標籤列表（可選）
            n_results: 結果數量
        """
        print("="*60)
        print("建議的 Memory MCP 檢索方式:")
        print("="*60)

        if tags:
            print(f"方法 1: 標籤搜尋")
            print(f"""
mcp__memory-service__search_by_tag(
    tags={json.dumps(tags)}
)
            """)

        print(f"方法 2: 語義搜尋")
        print(f"""
mcp__memory-service__retrieve_memory(
    query="{query}",
    n_results={n_results}
)
        """)

        print(f"方法 3: 品質加權檢索（推薦）")
        print(f"""
mcp__memory-service__retrieve_with_quality_boost(
    query="{query}",
    n_results={n_results},
    quality_weight=0.3
)
        """)
        print("="*60)


# 便利函數

def create_memory_integration() -> MemoryIntegration:
    """建立 Memory 整合實例"""
    return MemoryIntegration()


def store_successful_experiment(
    memory: MemoryIntegration,
    strategy_name: str,
    symbol: str,
    timeframe: str,
    opt_result: "OptimizationResult",
    wfa_result: "WalkForwardResult",
    market_conditions: Optional[str] = None
) -> tuple[str, Dict[str, Any]]:
    """
    存儲成功的實驗結果

    Args:
        memory: MemoryIntegration 實例
        strategy_name: 策略名稱
        symbol: 交易標的
        timeframe: 時間框架
        opt_result: 優化結果
        wfa_result: Walk-Forward 驗證結果
        market_conditions: 市場條件

    Returns:
        (content, metadata) - 適合 Memory MCP 存儲
    """
    insight = memory.create_insight_from_optimization(
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        opt_result=opt_result,
        wfa_result=wfa_result,
        market_conditions=market_conditions
    )

    content, metadata = memory.format_strategy_insight(insight)

    # 印出範例供 Claude 使用
    memory.print_storage_example(content, metadata)

    return content, metadata


def store_failed_experiment(
    memory: MemoryIntegration,
    strategy_name: str,
    symbol: str,
    timeframe: str,
    opt_result: "OptimizationResult",
    wfa_result: "WalkForwardResult",
    failure_description: str,
    prevention_advice: str
) -> tuple[str, Dict[str, Any]]:
    """
    存儲失敗的實驗教訓

    Args:
        memory: MemoryIntegration 實例
        strategy_name: 策略名稱
        symbol: 交易標的
        timeframe: 時間框架
        opt_result: 優化結果
        wfa_result: Walk-Forward 驗證結果
        failure_description: 失敗描述
        prevention_advice: 預防建議

    Returns:
        (content, metadata) - 適合 Memory MCP 存儲
    """
    lesson = memory.create_lesson_from_validation_failure(
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        opt_result=opt_result,
        wfa_result=wfa_result,
        failure_description=failure_description,
        prevention_advice=prevention_advice
    )

    content, metadata = memory.format_trading_lesson(lesson)

    # 印出範例供 Claude 使用
    memory.print_storage_example(content, metadata)

    return content, metadata


def retrieve_best_params_guide(
    memory: MemoryIntegration,
    strategy_type: str,
    symbol: str,
    timeframe: str
):
    """
    印出「取得最佳參數」的檢索指南

    Args:
        memory: MemoryIntegration 實例
        strategy_type: 策略類型
        symbol: 交易標的
        timeframe: 時間框架
    """
    query = memory.suggest_best_params_query(strategy_type, symbol, timeframe)
    tags = memory.suggest_tags_for_search(
        strategy_type=strategy_type,
        symbol=symbol,
        timeframe=timeframe,
        status=MemoryTags.STATUS_VALIDATED
    )

    memory.print_retrieval_example(query, tags)
