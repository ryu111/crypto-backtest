"""
Pytest 全域設定

自動安裝進程守護，防止測試中斷時留下孤兒進程。
"""

import pytest
import atexit

from src.utils.process_guard import ProcessGuard, cleanup_orphan_processes


# 測試開始前安裝進程守護
ProcessGuard.install()


@pytest.fixture(scope="session", autouse=True)
def cleanup_processes_on_exit():
    """測試結束後清理孤兒進程"""
    yield
    # 測試結束後清理
    cleaned = cleanup_orphan_processes()
    if cleaned > 0:
        print(f"\n[ProcessGuard] 清理了 {cleaned} 個孤兒進程")


def pytest_configure(config):
    """Pytest 配置階段"""
    # 註冊 atexit 確保即使異常退出也會清理
    atexit.register(cleanup_orphan_processes)


def pytest_unconfigure(config):
    """Pytest 結束階段"""
    cleanup_orphan_processes()
