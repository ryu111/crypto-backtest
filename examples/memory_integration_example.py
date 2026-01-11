"""
Memory MCP 整合使用範例

展示如何在 AI Loop 中整合 Memory MCP 進行知識存儲與檢索。
"""

import sys
from pathlib import Path

# 加入專案根目錄到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.learning import (
    MemoryIntegration,
    StrategyInsight,
    MarketInsight,
    TradingLesson,
    MemoryTags,
    create_memory_integration,
    store_successful_experiment,
    store_failed_experiment,
    retrieve_best_params_guide
)


def example_1_manual_store_insight():
    """範例 1: 手動建立並存儲策略洞察"""
    print("\n" + "="*60)
    print("範例 1: 手動建立策略洞察")
    print("="*60)

    memory = create_memory_integration()

    # 建立策略洞察
    insight = StrategyInsight(
        strategy_name="MA Cross",
        symbol="BTCUSDT",
        timeframe="4h",
        best_params={
            "fast_period": 10,
            "slow_period": 30,
            "atr_stop": 2.0
        },
        sharpe_ratio=1.85,
        total_return=0.456,
        max_drawdown=-0.12,
        win_rate=0.58,
        wfa_efficiency=0.68,
        wfa_grade="A",
        market_conditions="趨勢明確市場",
        notes="在牛市中表現特別好"
    )

    # 格式化為存儲格式
    content, metadata = memory.format_strategy_insight(insight)

    # 印出存儲範例
    memory.print_storage_example(content, metadata)


def example_2_store_market_insight():
    """範例 2: 存儲市場洞察"""
    print("\n" + "="*60)
    print("範例 2: 存儲市場洞察")
    print("="*60)

    memory = create_memory_integration()

    # 建立市場洞察
    insight = MarketInsight(
        symbol="ETHUSDT",
        timeframe="1h",
        market_type="volatile",
        observations="市場波動劇烈，資金費率頻繁變化，適合短線策略",
        recommended_strategies=["momentum", "breakout"],
        warnings="不適合長持倉，建議嚴格止損"
    )

    content, metadata = memory.format_market_insight(insight)
    memory.print_storage_example(content, metadata)


def example_3_store_trading_lesson():
    """範例 3: 存儲交易教訓"""
    print("\n" + "="*60)
    print("範例 3: 存儲失敗教訓")
    print("="*60)

    memory = create_memory_integration()

    # 建立交易教訓
    lesson = TradingLesson(
        strategy_name="RSI Mean Reversion",
        symbol="BTCUSDT",
        timeframe="1h",
        failure_type="overfitting",
        description="參數在訓練期表現優異（Sharpe 2.5），但 OOS 完全失效",
        symptoms="WFA Efficiency 僅 15%，Grade F，樣本外虧損",
        prevention="使用更長的驗證期，增加 WFA 窗口數量，檢查參數穩定性",
        failed_params={"rsi_period": 7, "oversold": 25, "overbought": 75}
    )

    content, metadata = memory.format_trading_lesson(lesson)
    memory.print_storage_example(content, metadata)


def example_4_retrieval_suggestions():
    """範例 4: 檢索建議"""
    print("\n" + "="*60)
    print("範例 4: 檢索建議")
    print("="*60)

    memory = create_memory_integration()

    # 產生查詢建議
    query = memory.suggest_retrieval_query(
        strategy_name="MA Cross",
        symbol="BTCUSDT",
        timeframe="4h"
    )

    tags = memory.suggest_tags_for_search(
        strategy_type="trend",
        symbol="BTCUSDT",
        timeframe="4h",
        status=MemoryTags.STATUS_VALIDATED
    )

    memory.print_retrieval_example(query, tags)


def example_5_ai_loop_integration():
    """範例 5: AI Loop 整合流程"""
    print("\n" + "="*60)
    print("範例 5: AI Loop 整合流程")
    print("="*60)

    memory = create_memory_integration()

    print("""
AI Loop 整合流程:

1. 優化前 - 查詢歷史最佳參數
   ========================================
   """)

    retrieve_best_params_guide(
        memory,
        strategy_type="ma-cross",
        symbol="BTCUSDT",
        timeframe="4h"
    )

    print("""
2. 優化後 - 驗證通過，存儲新洞察
   ========================================
   使用 store_successful_experiment() 或 store_failed_experiment()
   根據 WFA 結果自動判斷成功/失敗

3. 驗證失敗 - 存儲教訓
   ========================================
   如果 WFA Grade < C 或 Efficiency < 50%，存儲失敗教訓
   """)


def example_6_complete_workflow():
    """範例 6: 完整工作流程（含模擬資料）"""
    print("\n" + "="*60)
    print("範例 6: 完整工作流程模擬")
    print("="*60)

    from dataclasses import dataclass
    from src.backtester.engine import BacktestResult
    from src.optimizer.bayesian import OptimizationResult
    from src.validator.walk_forward import WalkForwardResult
    import pandas as pd

    # 模擬優化結果
    @dataclass
    class MockOptResult:
        best_params: dict
        best_backtest_result: object

    @dataclass
    class MockBacktest:
        sharpe_ratio: float = 1.85
        total_return: float = 0.456
        max_drawdown: float = -0.12
        win_rate: float = 0.58

    @dataclass
    class MockWFAResult:
        oos_efficiency: float = 0.68
        grade: str = "A"

    mock_opt = MockOptResult(
        best_params={"fast": 10, "slow": 30},
        best_backtest_result=MockBacktest()
    )

    mock_wfa = MockWFAResult()

    memory = create_memory_integration()

    print("\n情境: 策略驗證通過，存儲成功經驗\n")

    # 建立 insight
    insight = memory.create_insight_from_optimization(
        strategy_name="MA Cross",
        symbol="BTCUSDT",
        timeframe="4h",
        opt_result=mock_opt,
        wfa_result=mock_wfa,
        market_conditions="趨勢明確的牛市"
    )

    content, metadata = memory.format_strategy_insight(insight)
    memory.print_storage_example(content, metadata)


if __name__ == "__main__":
    """執行所有範例"""

    print("""
╔═══════════════════════════════════════════════════════════╗
║           Memory MCP 整合使用範例                          ║
╚═══════════════════════════════════════════════════════════╝
    """)

    # 執行範例
    example_1_manual_store_insight()
    example_2_store_market_insight()
    example_3_store_trading_lesson()
    example_4_retrieval_suggestions()
    example_5_ai_loop_integration()
    example_6_complete_workflow()

    print("""
╔═══════════════════════════════════════════════════════════╗
║                    範例執行完成                            ║
║                                                           ║
║  下一步:                                                  ║
║  1. 在 AI Loop 優化前呼叫檢索方法                           ║
║  2. 在驗證通過後呼叫存儲方法                                ║
║  3. 在失敗時存儲教訓避免重複錯誤                             ║
╚═══════════════════════════════════════════════════════════╝
    """)
