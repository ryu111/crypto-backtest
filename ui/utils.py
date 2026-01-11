"""UI 共用工具函數

提供實驗載入、過濾、格式化等功能。
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# 確定專案根目錄
UI_DIR = Path(__file__).parent
PROJECT_ROOT = UI_DIR.parent


def load_experiments() -> List[Dict]:
    """載入所有實驗記錄

    Returns:
        實驗列表，按時間戳排序（新到舊）
    """
    experiments_file = PROJECT_ROOT / "learning" / "experiments.json"

    if not experiments_file.exists():
        return []

    try:
        with open(experiments_file, "r", encoding="utf-8") as f:
            experiments = json.load(f)

        # 按時間戳排序（新到舊）
        experiments.sort(
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )

        return experiments
    except Exception as e:
        print(f"載入實驗失敗: {e}")
        return []


def filter_experiments(
    experiments: List[Dict],
    filters: Optional[Dict] = None
) -> List[Dict]:
    """過濾實驗列表

    Args:
        experiments: 實驗列表
        filters: 過濾條件
            - grade: str 評級篩選
            - min_sharpe: float 最小 Sharpe
            - max_sharpe: float 最大 Sharpe
            - validated_only: bool 只顯示驗證通過

    Returns:
        過濾後的實驗列表
    """
    if not filters:
        return experiments

    filtered = experiments

    # 評級篩選
    if "grade" in filters and filters["grade"] != "全部":
        filtered = [
            exp for exp in filtered
            if exp.get("grade") == filters["grade"]
        ]

    # Sharpe 範圍
    if "min_sharpe" in filters:
        filtered = [
            exp for exp in filtered
            if exp.get("sharpe_ratio", float('-inf')) >= filters["min_sharpe"]
        ]

    if "max_sharpe" in filters:
        filtered = [
            exp for exp in filtered
            if exp.get("sharpe_ratio", float('inf')) <= filters["max_sharpe"]
        ]

    # 只顯示驗證通過
    if filters.get("validated_only"):
        filtered = [
            exp for exp in filtered
            if exp.get("validation_pass", False)
        ]

    return filtered


def format_percentage(value: Optional[float], decimals: int = 2) -> str:
    """格式化百分比

    Args:
        value: 數值（0.05 = 5%）
        decimals: 小數位數

    Returns:
        格式化字串，例如 "+5.23%"
    """
    if value is None:
        return "N/A"

    percent = value * 100
    sign = "+" if percent > 0 else ""

    return f"{sign}{percent:.{decimals}f}%"


def format_sharpe(value: Optional[float], with_color: bool = False) -> str:
    """格式化 Sharpe Ratio

    Args:
        value: Sharpe 值
        with_color: 是否返回帶顏色的 HTML（用於 st.markdown）

    Returns:
        格式化字串或 HTML
    """
    if value is None:
        return "N/A"

    formatted = f"{value:.2f}"

    if not with_color:
        return formatted

    # 根據 Sharpe 值返回顏色
    if value >= 2.0:
        color = "#10b981"  # 綠色 - 優秀
    elif value >= 1.0:
        color = "#22d3ee"  # 青色 - 良好
    elif value >= 0:
        color = "#f59e0b"  # 橘色 - 普通
    else:
        color = "#ef4444"  # 紅色 - 不佳

    return f'<span style="color: {color}; font-weight: 600;">{formatted}</span>'


def grade_color(grade: str) -> str:
    """取得評級對應顏色

    Args:
        grade: 評級（S/A/B/C/D/F）

    Returns:
        Hex 顏色碼
    """
    colors = {
        "S": "#a855f7",  # 紫色
        "A": "#10b981",  # 綠色
        "B": "#22d3ee",  # 青色
        "C": "#f59e0b",  # 橘色
        "D": "#fb923c",  # 深橘色
        "F": "#ef4444",  # 紅色
    }

    return colors.get(grade, "#6b7280")  # 預設灰色


def get_grade_stats(experiments: List[Dict]) -> Dict[str, int]:
    """統計各評級數量

    Args:
        experiments: 實驗列表

    Returns:
        評級統計字典 {"S": 5, "A": 10, ...}
    """
    stats = {"S": 0, "A": 0, "B": 0, "C": 0, "D": 0, "F": 0}

    for exp in experiments:
        grade = exp.get("grade", "F")
        if grade in stats:
            stats[grade] += 1

    return stats


def format_timestamp(timestamp: str) -> str:
    """格式化時間戳為可讀格式

    Args:
        timestamp: ISO 格式時間戳

    Returns:
        格式化字串，例如 "2026-01-11 14:30"
    """
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return timestamp


def get_latest_experiments(experiments: List[Dict], count: int = 5) -> List[Dict]:
    """取得最近的實驗

    Args:
        experiments: 實驗列表（假設已按時間排序）
        count: 取得數量

    Returns:
        最近的 N 個實驗
    """
    return experiments[:count]


def calculate_summary_stats(experiments: List[Dict]) -> Dict:
    """計算總體統計數據

    Args:
        experiments: 實驗列表

    Returns:
        統計字典，包含總數、驗證通過數、最佳 Sharpe 等
    """
    if not experiments:
        return {
            "total_count": 0,
            "validated_count": 0,
            "best_sharpe": None,
            "avg_sharpe": None,
            "grade_distribution": {},
        }

    validated = [exp for exp in experiments if exp.get("validation_pass")]
    sharpes = [
        exp["sharpe_ratio"]
        for exp in experiments
        if exp.get("sharpe_ratio") is not None
    ]

    return {
        "total_count": len(experiments),
        "validated_count": len(validated),
        "best_sharpe": max(sharpes) if sharpes else None,
        "avg_sharpe": sum(sharpes) / len(sharpes) if sharpes else None,
        "grade_distribution": get_grade_stats(experiments),
    }


def get_data_source_status() -> Dict:
    """檢查資料來源狀態

    Returns:
        狀態字典，包含是否可用、最後更新時間等
    """
    experiments_file = PROJECT_ROOT / "learning" / "experiments.json"

    if not experiments_file.exists():
        return {
            "available": False,
            "last_updated": None,
            "experiment_count": 0,
        }

    try:
        # 取得檔案修改時間
        mtime = experiments_file.stat().st_mtime
        last_updated = datetime.fromtimestamp(mtime)

        # 讀取實驗數量
        with open(experiments_file, "r", encoding="utf-8") as f:
            experiments = json.load(f)

        return {
            "available": True,
            "last_updated": last_updated.strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_count": len(experiments),
        }
    except Exception as e:
        return {
            "available": False,
            "last_updated": None,
            "experiment_count": 0,
            "error": str(e),
        }
