"""
策略變化追蹤系統

管理策略參數變化的生命週期：註冊 → 測試 → 狀態更新 → 最終回測清單

設計決策：
- 參數相似度檢測：開啟（閾值 5%）
- 狀態判定：Grade 導向（A→PASSED, B→OPTIMIZABLE, C/D/F→FAILED）
- 最終回測 min_sharpe：1.5
"""

import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# ========== 型別定義 ==========

class VariationStatus(str, Enum):
    """
    策略變化狀態

    狀態流轉：
        UNTESTED → FAILED/OPTIMIZABLE/PASSED
        PASSED/OPTIMIZABLE → DEPRECATED（當有更好的變化時）
    """
    UNTESTED = "UNTESTED"           # 未測試
    FAILED = "FAILED"               # 未達標 (Grade C/D/F)
    OPTIMIZABLE = "OPTIMIZABLE"     # 可優化 (Grade B)
    PASSED = "PASSED"               # 通過 (Grade A)
    DEPRECATED = "DEPRECATED"       # 已淘汰


@dataclass
class VariationRecord:
    """
    策略變化記錄

    用於追蹤策略參數組合的完整生命週期
    """
    variation_hash: str                     # 變化 hash（唯一識別）
    strategy_name: str                      # 策略名稱（如 "trend_ma_cross"）
    strategy_type: str                      # 策略類型（如 "trend", "momentum"）
    params: Dict[str, Any]                  # 參數字典
    status: VariationStatus                 # 當前狀態
    grade: Optional[str] = None             # 驗證評級 (A/B/C/D/F)
    metrics: Optional[Dict[str, float]] = None  # 績效指標
    validation: Optional[Dict[str, Any]] = None # 驗證資訊
    experiment_id: Optional[str] = None     # 對應的實驗 ID
    tested_at: Optional[datetime] = None    # 測試時間
    registered_at: datetime = field(default_factory=datetime.now)  # 註冊時間
    failure_reason: Optional[str] = None    # 失敗原因
    tags: List[str] = field(default_factory=list)  # 標籤

    def to_dict(self) -> Dict[str, Any]:
        """
        序列化為 JSON

        Returns:
            Dict: JSON 可序列化的字典
        """
        data = asdict(self)

        # 轉換 Enum
        data['status'] = self.status.value

        # 轉換 datetime
        if self.tested_at:
            data['tested_at'] = self.tested_at.isoformat()
        data['registered_at'] = self.registered_at.isoformat()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VariationRecord':
        """
        從 JSON 反序列化

        Args:
            data: JSON 字典

        Returns:
            VariationRecord: 變化記錄
        """
        # 轉換 status
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = VariationStatus(data['status'])

        # 轉換 datetime
        if 'tested_at' in data and data['tested_at']:
            if isinstance(data['tested_at'], str):
                data['tested_at'] = datetime.fromisoformat(data['tested_at'])

        if 'registered_at' in data:
            if isinstance(data['registered_at'], str):
                data['registered_at'] = datetime.fromisoformat(data['registered_at'])

        return cls(**data)


# ========== 主類別 ==========

class VariationTracker:
    """
    策略變化追蹤器

    功能：
    1. 註冊新變化（產生 hash, 標記 UNTESTED）
    2. 檢測參數相似度（避免重複測試）
    3. 從實驗結果更新狀態（Grade → Status 轉換）
    4. 生成最終回測清單（PASSED/OPTIMIZABLE, Sharpe >= 1.5）
    5. 持久化 JSON + 生成 Markdown 報告

    使用範例：
        tracker = VariationTracker()

        # 註冊新變化
        var_hash = tracker.register_variation(
            strategy_name="trend_ma_cross",
            strategy_type="trend",
            params={"fast": 10, "slow": 30}
        )

        # 更新狀態（從實驗結果）
        tracker.update_from_experiment(
            variation_hash=var_hash,
            experiment_id="exp_123",
            grade="A",
            metrics={"sharpe_ratio": 2.1, ...}
        )

        # 取得最終回測清單
        final_list = tracker.get_final_backtest_list(min_sharpe=1.5)
    """

    # ========== 常數定義 ==========
    HASH_LENGTH = 16              # SHA256 hash 使用長度
    HASH_DISPLAY_LENGTH = 12      # Markdown 顯示長度
    ZERO_THRESHOLD = 1e-10        # 零值判定閾值

    # 狀態標籤（用於 Markdown）
    STATUS_LABELS = {
        'UNTESTED': '未測試',
        'PASSED': '通過 (A)',
        'OPTIMIZABLE': '可優化 (B)',
        'FAILED': '失敗 (C/D/F)',
        'DEPRECATED': '已淘汰'
    }

    def __init__(
        self,
        json_path: Optional[Path] = None,
        md_path: Optional[Path] = None,
        float_precision: int = 2,
        similarity_threshold: float = 0.05
    ):
        """
        初始化追蹤器

        Args:
            json_path: JSON 儲存路徑（預設: learning/variations.json）
            md_path: Markdown 報告路徑（預設: learning/variations.md）
            float_precision: 浮點數精度（用於 hash 計算）
            similarity_threshold: 相似度閾值（5% = 0.05）
        """
        # 確定專案根目錄（安全搜尋）
        self.project_root = self._find_project_root(Path(__file__).parent)

        # 預設路徑
        self.json_path = json_path or self.project_root / 'learning' / 'variations.json'
        self.md_path = md_path or self.project_root / 'learning' / 'variations.md'

        # 配置
        self.float_precision = float_precision
        self.similarity_threshold = similarity_threshold

        # 變化記錄（hash → VariationRecord）
        self.variations: Dict[str, VariationRecord] = {}

        # 確保目錄存在
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.md_path.parent.mkdir(parents=True, exist_ok=True)

        # 載入現有資料
        self._load()

        logger.info(
            f"VariationTracker 已初始化 "
            f"(已載入 {len(self.variations)} 個變化, "
            f"相似度閾值 {similarity_threshold*100}%)"
        )

    @staticmethod
    def _find_project_root(start_path: Path) -> Path:
        """
        安全搜尋專案根目錄

        往上搜尋包含 .git 或 pyproject.toml 的目錄

        Args:
            start_path: 起始路徑

        Returns:
            Path: 專案根目錄

        Raises:
            RuntimeError: 找不到專案根目錄
        """
        current = start_path.resolve()

        while current != current.parent:
            if (current / '.git').exists() or (current / 'pyproject.toml').exists():
                return current
            current = current.parent

        # 如果找不到，使用 start_path 往上兩層（src/automation → 根目錄）
        fallback = start_path.parent.parent
        logger.warning(f"找不到專案根目錄標記，使用 fallback: {fallback}")
        return fallback

    # ========== Hash 計算 ==========

    def compute_hash(self, strategy_name: str, params: Dict[str, Any]) -> str:
        """
        計算參數組合的唯一 hash

        流程：
        1. 標準化參數（浮點數捨入）
        2. 按字典序排序
        3. 生成 SHA256 hash

        Args:
            strategy_name: 策略名稱
            params: 參數字典

        Returns:
            str: 格式為 "var_xxxxxxxxxxxxxxxx"（前 16 碼）

        Example:
            >>> compute_hash("ma_cross", {"fast": 10.12345, "slow": 30})
            "var_a1b2c3d4e5f6g7h8"
        """
        # 標準化參數
        normalized = self._normalize_params(params)

        # 組合鍵值對（字典序）
        sorted_items = sorted(normalized.items())

        # 建立唯一字串
        unique_str = f"{strategy_name}:" + ":".join(
            f"{k}={v}" for k, v in sorted_items
        )

        # SHA256 hash（前 HASH_LENGTH 碼）
        hash_obj = hashlib.sha256(unique_str.encode('utf-8'))
        short_hash = hash_obj.hexdigest()[:self.HASH_LENGTH]

        return f"var_{short_hash}"

    def _normalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        標準化參數（浮點數捨入）

        Args:
            params: 原始參數

        Returns:
            Dict: 標準化後的參數
        """
        normalized = {}

        for key, value in params.items():
            if isinstance(value, float):
                # 浮點數捨入
                normalized[key] = round(value, self.float_precision)
            elif isinstance(value, (int, str, bool, type(None))):
                # 基本型別直接保留
                normalized[key] = value
            else:
                # 其他型別轉字串（如 dict, list）
                normalized[key] = str(value)

        return normalized

    # ========== 變化管理 ==========

    def register_variation(
        self,
        strategy_name: str,
        strategy_type: str,
        params: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> str:
        """
        註冊新變化

        Args:
            strategy_name: 策略名稱
            strategy_type: 策略類型
            params: 參數字典
            tags: 標籤列表（可選）

        Returns:
            str: variation_hash
        """
        # 計算 hash
        var_hash = self.compute_hash(strategy_name, params)

        # 檢查是否已存在
        if var_hash in self.variations:
            logger.debug(f"變化 {var_hash} 已存在，跳過註冊")
            return var_hash

        # 建立新記錄
        record = VariationRecord(
            variation_hash=var_hash,
            strategy_name=strategy_name,
            strategy_type=strategy_type,
            params=params,
            status=VariationStatus.UNTESTED,
            tags=tags or []
        )

        self.variations[var_hash] = record

        # 持久化
        self._save()

        logger.info(f"已註冊變化 {var_hash} ({strategy_name})")

        return var_hash

    def is_tested(self, variation_hash: str) -> bool:
        """
        檢查變化是否已測試

        Args:
            variation_hash: 變化 hash

        Returns:
            bool: True 表示已測試（狀態非 UNTESTED）
        """
        record = self.variations.get(variation_hash)
        if not record:
            return False

        return record.status != VariationStatus.UNTESTED

    def get_status(self, variation_hash: str) -> Optional[VariationStatus]:
        """
        取得變化狀態

        Args:
            variation_hash: 變化 hash

        Returns:
            VariationStatus 或 None（如果不存在）
        """
        record = self.variations.get(variation_hash)
        return record.status if record else None

    # ========== 實驗結果更新 ==========

    def update_from_experiment(
        self,
        variation_hash: str,
        experiment_id: str,
        grade: str,
        metrics: Dict[str, float],
        validation: Optional[Dict[str, Any]] = None,
        failure_reason: Optional[str] = None
    ):
        """
        從實驗結果更新變化狀態

        狀態判定規則：
            - Grade A → PASSED
            - Grade B → OPTIMIZABLE
            - Grade C/D/F → FAILED

        Args:
            variation_hash: 變化 hash
            experiment_id: 實驗 ID
            grade: 驗證評級（A/B/C/D/F）
            metrics: 績效指標
                {
                    'sharpe_ratio': float,
                    'total_return': float,
                    'max_drawdown': float,
                    'win_rate': float,
                    'total_trades': int
                }
            validation: 驗證詳細資訊（可選）
            failure_reason: 失敗原因（可選，僅 FAILED 時使用）
        """
        # 檢查變化是否存在
        if variation_hash not in self.variations:
            logger.warning(f"變化 {variation_hash} 不存在，無法更新")
            return

        record = self.variations[variation_hash]

        # 根據 Grade 判定狀態
        if grade == 'A':
            new_status = VariationStatus.PASSED
        elif grade == 'B':
            new_status = VariationStatus.OPTIMIZABLE
        else:  # C/D/F
            new_status = VariationStatus.FAILED

        # 更新記錄
        record.status = new_status
        record.grade = grade
        record.metrics = metrics
        record.validation = validation
        record.experiment_id = experiment_id
        record.tested_at = datetime.now()
        record.failure_reason = failure_reason

        # 持久化
        self._save()

        logger.info(
            f"已更新變化 {variation_hash}: "
            f"{grade} → {new_status.value} "
            f"(Sharpe {metrics.get('sharpe_ratio', 0):.2f})"
        )

    # ========== 查詢方法 ==========

    def get_untested_variations(
        self,
        strategy_name: Optional[str] = None,
        strategy_type: Optional[str] = None
    ) -> List[VariationRecord]:
        """
        取得未測試變化

        Args:
            strategy_name: 過濾策略名稱（可選）
            strategy_type: 過濾策略類型（可選）

        Returns:
            List[VariationRecord]: 未測試變化列表
        """
        results = []

        for record in self.variations.values():
            # 狀態過濾
            if record.status != VariationStatus.UNTESTED:
                continue

            # 策略名稱過濾
            if strategy_name and record.strategy_name != strategy_name:
                continue

            # 策略類型過濾
            if strategy_type and record.strategy_type != strategy_type:
                continue

            results.append(record)

        # 按註冊時間排序（最舊的優先）
        results.sort(key=lambda r: r.registered_at)

        return results

    def get_final_backtest_list(
        self,
        min_sharpe: float = 1.5,
        limit: Optional[int] = None
    ) -> List[VariationRecord]:
        """
        取得最終回測清單

        過濾條件：
        1. 狀態為 PASSED 或 OPTIMIZABLE
        2. Sharpe Ratio >= min_sharpe

        Args:
            min_sharpe: Sharpe 最小值（預設 1.5）
            limit: 限制數量（可選）

        Returns:
            List[VariationRecord]: 最終回測清單（按 Sharpe 降序）
        """
        results = []

        for record in self.variations.values():
            # 狀態過濾
            if record.status not in [VariationStatus.PASSED, VariationStatus.OPTIMIZABLE]:
                continue

            # Sharpe 過濾
            if not record.metrics:
                continue

            sharpe = record.metrics.get('sharpe_ratio', 0)
            if sharpe < min_sharpe:
                continue

            results.append(record)

        # 按 Sharpe 降序排列
        results.sort(
            key=lambda r: r.metrics.get('sharpe_ratio', 0),
            reverse=True
        )

        # 限制數量
        if limit:
            results = results[:limit]

        return results

    def find_similar_variations(
        self,
        params: Dict[str, Any],
        strategy_name: str
    ) -> List[VariationRecord]:
        """
        找出相似的已測試變化

        用於避免重複測試相似參數組合

        Args:
            params: 參數字典
            strategy_name: 策略名稱

        Returns:
            List[VariationRecord]: 相似變化列表（僅已測試的）
        """
        results = []

        for record in self.variations.values():
            # 只檢查同策略
            if record.strategy_name != strategy_name:
                continue

            # 只檢查已測試的
            if record.status == VariationStatus.UNTESTED:
                continue

            # 檢查相似度
            if self._is_similar_params(params, record.params):
                results.append(record)

        return results

    def _is_similar_params(
        self,
        params1: Dict[str, Any],
        params2: Dict[str, Any]
    ) -> bool:
        """
        檢查兩組參數是否相似

        相似度判定：
        - 數值參數：相對差異 < similarity_threshold（5%）
        - 非數值參數：必須完全相等

        Args:
            params1: 參數組 1
            params2: 參數組 2

        Returns:
            bool: True 表示相似
        """
        # 標準化（浮點數捨入）
        norm1 = self._normalize_params(params1)
        norm2 = self._normalize_params(params2)

        # 鍵值必須相同
        if set(norm1.keys()) != set(norm2.keys()):
            return False

        # 逐個比較
        for key in norm1.keys():
            val1 = norm1[key]
            val2 = norm2[key]

            # 數值型別：檢查相對差異
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # 轉為 float 確保除法正確
                v1, v2 = float(val1), float(val2)

                # 使用更安全的零值檢查
                if abs(v2) < self.ZERO_THRESHOLD:
                    if abs(v1) > self.similarity_threshold:
                        return False
                else:
                    relative_diff = abs(v1 - v2) / abs(v2)
                    if relative_diff > self.similarity_threshold:
                        return False
            else:
                # 非數值型別：必須完全相等
                if val1 != val2:
                    return False

        return True

    # ========== 持久化 ==========

    def _load(self):
        """從 JSON 載入資料"""
        if not self.json_path.exists():
            logger.debug(f"找不到 {self.json_path}，使用空資料")
            return

        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 反序列化
            variations_data = data.get('variations', [])

            for var_data in variations_data:
                record = VariationRecord.from_dict(var_data)
                self.variations[record.variation_hash] = record

            logger.info(f"已載入 {len(self.variations)} 個變化")

        except Exception as e:
            logger.error(f"載入 {self.json_path} 失敗: {e}", exc_info=True)

    def _save(self):
        """儲存到 JSON 並生成 Markdown"""
        # 序列化
        variations_data = [
            record.to_dict()
            for record in self.variations.values()
        ]

        # 統計資訊
        stats = self._compute_statistics()

        # JSON 結構
        data = {
            'version': '1.0',
            'updated_at': datetime.now().isoformat(),
            'statistics': stats,
            'variations': variations_data
        }

        # 寫入 JSON（明確的錯誤處理）
        try:
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"已儲存 {len(self.variations)} 個變化到 {self.json_path}")
        except PermissionError as e:
            logger.error(f"沒有寫入權限: {self.json_path}: {e}")
            raise
        except IOError as e:
            logger.error(f"寫入 JSON 失敗: {self.json_path}: {e}")
            raise

        # 生成 Markdown（非致命性，失敗僅記錄）
        try:
            self._generate_markdown()
        except Exception as e:
            logger.warning(f"生成 Markdown 報告失敗（非致命）: {e}")

    def _compute_statistics(self) -> Dict[str, Any]:
        """
        計算統計資訊

        Returns:
            Dict: 統計資料
                {
                    'total': int,
                    'by_status': {...},
                    'by_strategy': {...}
                }
        """
        # 總數
        total = len(self.variations)

        # 按狀態統計
        by_status = {}
        for status in VariationStatus:
            count = sum(
                1 for r in self.variations.values()
                if r.status == status
            )
            by_status[status.value] = count

        # 按策略統計
        by_strategy = {}
        for record in self.variations.values():
            strategy_name = record.strategy_name

            if strategy_name not in by_strategy:
                by_strategy[strategy_name] = {
                    'total': 0,
                    'untested': 0,
                    'passed': 0,
                    'optimizable': 0,
                    'failed': 0
                }

            by_strategy[strategy_name]['total'] += 1

            if record.status == VariationStatus.UNTESTED:
                by_strategy[strategy_name]['untested'] += 1
            elif record.status == VariationStatus.PASSED:
                by_strategy[strategy_name]['passed'] += 1
            elif record.status == VariationStatus.OPTIMIZABLE:
                by_strategy[strategy_name]['optimizable'] += 1
            elif record.status == VariationStatus.FAILED:
                by_strategy[strategy_name]['failed'] += 1

        return {
            'total': total,
            'by_status': by_status,
            'by_strategy': by_strategy
        }

    # ========== Markdown 生成輔助方法 ==========

    def _get_variations_by_status(self, status: VariationStatus) -> List[VariationRecord]:
        """取得特定狀態的變化列表"""
        return [r for r in self.variations.values() if r.status == status]

    def _sort_by_sharpe(self, variations: List[VariationRecord]) -> List[VariationRecord]:
        """按 Sharpe Ratio 降序排列"""
        return sorted(
            variations,
            key=lambda r: r.metrics.get('sharpe_ratio', 0) if r.metrics else 0,
            reverse=True
        )

    def _format_variation_table(self, variations: List[VariationRecord]) -> List[str]:
        """
        格式化變化列表為 Markdown 表格

        Args:
            variations: 變化列表

        Returns:
            List[str]: Markdown 行列表
        """
        if not variations:
            return ["*無記錄*", ""]

        lines = [
            "| Hash | 策略 | Sharpe | Return | MaxDD | 測試時間 |",
            "|------|------|--------|--------|-------|----------|"
        ]

        for record in variations:
            if not record.metrics:
                continue

            sharpe = record.metrics.get('sharpe_ratio', 0.0)
            total_return = record.metrics.get('total_return', 0.0)
            max_dd = record.metrics.get('max_drawdown', 0.0)
            tested_at = record.tested_at.strftime('%Y-%m-%d') if record.tested_at else 'N/A'

            lines.append(
                f"| {record.variation_hash[:self.HASH_DISPLAY_LENGTH]}... "
                f"| {record.strategy_name} "
                f"| {sharpe:.2f} "
                f"| {total_return*100:.1f}% "
                f"| {max_dd*100:.1f}% "
                f"| {tested_at} |"
            )

        lines.append("")  # 空行
        return lines

    def _generate_header(self, stats: Dict[str, Any]) -> List[str]:
        """生成標題和概覽統計"""
        return [
            "# 策略變化追蹤報告",
            "",
            f"**最後更新**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 概覽",
            "",
            f"- **總變化數**: {stats['total']}",
            f"- **{self.STATUS_LABELS['UNTESTED']}**: {stats['by_status'].get('UNTESTED', 0)}",
            f"- **{self.STATUS_LABELS['PASSED']}**: {stats['by_status'].get('PASSED', 0)}",
            f"- **{self.STATUS_LABELS['OPTIMIZABLE']}**: {stats['by_status'].get('OPTIMIZABLE', 0)}",
            f"- **{self.STATUS_LABELS['FAILED']}**: {stats['by_status'].get('FAILED', 0)}",
            ""
        ]

    def _generate_strategy_stats_table(self, stats: Dict[str, Any]) -> List[str]:
        """生成策略統計表格"""
        lines = [
            "## 各策略統計",
            "",
            "| 策略 | 總數 | 未測試 | 通過 (A) | 可優化 (B) | 失敗 |",
            "|------|------|--------|----------|------------|------|"
        ]

        for strategy_name, strategy_stats in sorted(stats['by_strategy'].items()):
            lines.append(
                f"| {strategy_name} "
                f"| {strategy_stats['total']} "
                f"| {strategy_stats['untested']} "
                f"| {strategy_stats['passed']} "
                f"| {strategy_stats['optimizable']} "
                f"| {strategy_stats['failed']} |"
            )

        lines.append("")
        return lines

    def _generate_passed_section(self) -> List[str]:
        """生成通過變化章節"""
        passed = self._get_variations_by_status(VariationStatus.PASSED)
        passed = self._sort_by_sharpe(passed)

        lines = ["## 通過變化 (Grade A)", ""]
        lines.extend(self._format_variation_table(passed))
        return lines

    def _generate_optimizable_section(self) -> List[str]:
        """生成可優化變化章節"""
        optimizable = self._get_variations_by_status(VariationStatus.OPTIMIZABLE)
        optimizable = self._sort_by_sharpe(optimizable)

        lines = ["## 可優化變化 (Grade B)", ""]
        lines.extend(self._format_variation_table(optimizable))
        return lines

    def _write_markdown(self, lines: List[str]):
        """寫入 Markdown 檔案"""
        try:
            with open(self.md_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            logger.debug(f"已生成 Markdown 報告: {self.md_path}")
        except PermissionError as e:
            logger.warning(f"沒有寫入權限: {self.md_path}: {e}")
        except IOError as e:
            logger.warning(f"寫入 Markdown 失敗: {self.md_path}: {e}")

    def _generate_markdown(self):
        """生成 Markdown 報告（調度方法）"""
        stats = self._compute_statistics()

        lines = []
        lines.extend(self._generate_header(stats))
        lines.extend(self._generate_strategy_stats_table(stats))
        lines.extend(self._generate_passed_section())
        lines.extend(self._generate_optimizable_section())

        self._write_markdown(lines)
