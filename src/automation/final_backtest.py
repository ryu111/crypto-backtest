"""
最終回測模式

只測試通過驗證(PASSED/OPTIMIZABLE)的變化,用於正式發布前的最終驗證。

設計目標：
1. 從 VariationTracker 取得候選變化
2. 使用更嚴格的驗證標準
3. 輸出最終部署建議報告
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from .variation_tracker import VariationTracker, VariationStatus, VariationRecord

logger = logging.getLogger(__name__)


class FinalBacktest:
    """
    最終回測系統

    使用場景：
    1. AI Loop 完成 N 次迭代後
    2. 只測試 PASSED (A 級) 或 OPTIMIZABLE (B 級) 的變化
    3. 使用更嚴格的驗證標準
    4. 輸出最終報告

    使用範例：
        tracker = VariationTracker()
        final = FinalBacktest(tracker)

        # 取得待測變化
        variations = final.get_final_variations(min_sharpe=1.5)

        # 顯示待測清單
        final.print_summary()

        # 執行最終回測
        results = final.run_final_backtest(variations)

        # 生成報告
        final.generate_report(results)
    """

    # 預設配置
    DEFAULT_MIN_SHARPE = 1.5
    DEFAULT_INCLUDE_OPTIMIZABLE = True

    def __init__(
        self,
        tracker: VariationTracker,
        report_path: Optional[Path] = None
    ):
        """
        初始化最終回測

        Args:
            tracker: 變化追蹤器實例
            report_path: 報告輸出路徑（預設: learning/final_backtest_report.md）
        """
        self.tracker = tracker
        self.report_path = report_path or (
            tracker.project_root / 'learning' / 'final_backtest_report.md'
        )

        # 確保目錄存在
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"FinalBacktest 已初始化 (報告路徑: {self.report_path})")

    def get_final_variations(
        self,
        min_sharpe: float = DEFAULT_MIN_SHARPE,
        include_optimizable: bool = DEFAULT_INCLUDE_OPTIMIZABLE,
        limit: Optional[int] = None
    ) -> List[VariationRecord]:
        """
        取得最終回測變化清單

        Args:
            min_sharpe: 最小 Sharpe 要求（預設 1.5）
            include_optimizable: 是否包含 B 級變化
            limit: 限制數量（選填）

        Returns:
            變化清單（按 Sharpe 降序）
        """
        # 決定包含的狀態
        statuses = [VariationStatus.PASSED]
        if include_optimizable:
            statuses.append(VariationStatus.OPTIMIZABLE)

        # 從追蹤器取得變化
        all_variations = []
        for status in statuses:
            variations = self.tracker._get_variations_by_status(status)
            all_variations.extend(variations)

        # 過濾 Sharpe
        filtered = [
            v for v in all_variations
            if v.metrics and v.metrics.get('sharpe_ratio', 0) >= min_sharpe
        ]

        # 排序（Sharpe 降序）
        filtered = self.tracker._sort_by_sharpe(filtered)

        # 限制數量
        if limit:
            filtered = filtered[:limit]

        logger.info(
            f"取得 {len(filtered)} 個變化進行最終回測 "
            f"(min_sharpe={min_sharpe}, include_optimizable={include_optimizable})"
        )

        return filtered

    def print_summary(
        self,
        min_sharpe: float = DEFAULT_MIN_SHARPE,
        include_optimizable: bool = DEFAULT_INCLUDE_OPTIMIZABLE
    ):
        """
        列印待測變化摘要

        Args:
            min_sharpe: 最小 Sharpe 要求
            include_optimizable: 是否包含 B 級變化
        """
        variations = self.get_final_variations(
            min_sharpe=min_sharpe,
            include_optimizable=include_optimizable
        )

        print("\n" + "=" * 60)
        print("📋 最終回測候選清單")
        print("=" * 60)
        print(f"條件: Sharpe >= {min_sharpe}, 包含 B 級: {include_optimizable}")
        print(f"總計: {len(variations)} 個變化")
        print("-" * 60)

        if not variations:
            print("❌ 無符合條件的變化")
            print("=" * 60)
            return

        # 按策略分組
        by_strategy: Dict[str, List[VariationRecord]] = {}
        for v in variations:
            if v.strategy_name not in by_strategy:
                by_strategy[v.strategy_name] = []
            by_strategy[v.strategy_name].append(v)

        # 列印每個策略的變化
        for strategy_name, strategy_variations in sorted(by_strategy.items()):
            print(f"\n📊 {strategy_name} ({len(strategy_variations)} 個變化)")
            print("-" * 40)

            for i, v in enumerate(strategy_variations, 1):
                metrics = v.metrics or {}
                sharpe = metrics.get('sharpe_ratio', 0)
                max_dd = metrics.get('max_drawdown', 0)
                grade = v.grade or 'N/A'
                status_icon = "✅" if v.status == VariationStatus.PASSED else "🔄"

                print(f"  {i}. {status_icon} {v.variation_hash[:12]}...")
                print(f"     Grade: {grade} | Sharpe: {sharpe:.2f} | MaxDD: {max_dd*100:.1f}%")
                print(f"     參數: {v.params}")

        print("\n" + "=" * 60)

    def get_variation_by_hash(self, variation_hash: str) -> Optional[VariationRecord]:
        """
        根據 hash 取得變化記錄

        Args:
            variation_hash: 變化 hash

        Returns:
            VariationRecord 或 None
        """
        return self.tracker.variations.get(variation_hash)

    def generate_report(
        self,
        variations: List[VariationRecord],
        title: str = "最終回測報告"
    ) -> Path:
        """
        生成最終回測報告

        Args:
            variations: 變化列表
            title: 報告標題

        Returns:
            Path: 報告檔案路徑
        """
        lines = [
            f"# {title}",
            "",
            f"**生成時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**總變化數**: {len(variations)}",
            "",
            "---",
            "",
            "## 概覽",
            ""
        ]

        # 統計
        passed_count = sum(1 for v in variations if v.status == VariationStatus.PASSED)
        optimizable_count = sum(1 for v in variations if v.status == VariationStatus.OPTIMIZABLE)

        lines.extend([
            f"- **通過 (A)**: {passed_count}",
            f"- **可優化 (B)**: {optimizable_count}",
            ""
        ])

        # 按策略分組
        by_strategy: Dict[str, List[VariationRecord]] = {}
        for v in variations:
            if v.strategy_name not in by_strategy:
                by_strategy[v.strategy_name] = []
            by_strategy[v.strategy_name].append(v)

        lines.extend([
            "## 各策略變化",
            ""
        ])

        for strategy_name, strategy_variations in sorted(by_strategy.items()):
            lines.extend([
                f"### {strategy_name}",
                "",
                "| Hash | Grade | Sharpe | Return | MaxDD | 參數摘要 |",
                "|------|-------|--------|--------|-------|----------|"
            ])

            for v in strategy_variations:
                metrics = v.metrics or {}
                sharpe = metrics.get('sharpe_ratio', 0)
                total_return = metrics.get('total_return', 0)
                max_dd = metrics.get('max_drawdown', 0)
                grade = v.grade or 'N/A'

                # 參數摘要（只顯示前 3 個）
                params_items = list(v.params.items())[:3]
                params_summary = ", ".join(f"{k}={v}" for k, v in params_items)
                if len(v.params) > 3:
                    params_summary += "..."

                lines.append(
                    f"| {v.variation_hash[:12]}... "
                    f"| {grade} "
                    f"| {sharpe:.2f} "
                    f"| {total_return*100:.1f}% "
                    f"| {max_dd*100:.1f}% "
                    f"| {params_summary} |"
                )

            lines.append("")

        # 部署建議
        lines.extend([
            "---",
            "",
            "## 部署建議",
            "",
            "### 建議優先部署的變化",
            ""
        ])

        # 取 Sharpe 最高的 3 個 PASSED 變化
        top_passed = [
            v for v in variations
            if v.status == VariationStatus.PASSED
        ][:3]

        if top_passed:
            for i, v in enumerate(top_passed, 1):
                metrics = v.metrics or {}
                sharpe = metrics.get('sharpe_ratio', 0)
                lines.append(
                    f"{i}. **{v.strategy_name}** - `{v.variation_hash[:12]}...` "
                    f"(Sharpe {sharpe:.2f})"
                )
        else:
            lines.append("*無建議部署的變化*")

        lines.append("")

        # 寫入檔案
        try:
            with open(self.report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            logger.info(f"已生成最終回測報告: {self.report_path}")
        except (PermissionError, IOError) as e:
            logger.error(f"寫入報告失敗: {self.report_path}: {e}")

        return self.report_path
