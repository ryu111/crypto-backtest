"""
合約交易回測系統

自動安裝進程守護，防止孤兒進程。
"""

from src.utils.process_guard import install_process_guard

# 全域安裝進程守護
install_process_guard()
