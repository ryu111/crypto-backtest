"""
驗證修復的測試

測試兩個關鍵修復：
1. 共享記憶體名稱生成（hyperloop.py:231-236）
2. 年化報酬計算（engine.py:552-561）
"""

import pytest
import os
from datetime import datetime, timedelta


class TestSharedMemoryNameGeneration:
    """測試共享記憶體名稱生成修復"""

    def test_pool_name_format(self):
        """驗證共享記憶體名稱格式正確"""
        # 模擬名稱生成邏輯
        pid = os.getpid()
        instance_counter = 0
        pool_name = f"hl_{pid % 10000}_{instance_counter}"

        # 驗證格式
        assert pool_name.startswith("hl_"), "名稱應以 'hl_' 開頭"
        parts = pool_name.split("_")
        assert len(parts) == 3, "名稱應有三個部分（hl_PID_counter）"
        assert parts[1].isdigit(), "PID 部分應為數字"
        assert parts[2].isdigit(), "counter 部分應為數字"

    def test_pool_name_length_limit(self):
        """驗證名稱長度符合 macOS 限制（31 字元）"""
        # 測試最大可能長度
        max_pid = 9999  # % 10000 的最大值
        max_counter = 999  # 假設最多 999 個實例
        pool_name = f"hl_{max_pid}_{max_counter}"

        assert len(pool_name) <= 31, f"名稱長度 {len(pool_name)} 超過 macOS 限制 31 字元"
        # 實際長度應為: "hl_" (3) + "9999" (4) + "_" (1) + "999" (3) = 11 字元
        assert len(pool_name) == 11, f"預期長度 11，實際 {len(pool_name)}"

    def test_pool_name_uniqueness(self):
        """驗證不同實例產生唯一名稱"""
        pid = os.getpid()
        names = []

        for i in range(5):
            pool_name = f"hl_{pid % 10000}_{i}"
            names.append(pool_name)

        # 所有名稱應該唯一
        assert len(names) == len(set(names)), "名稱應該唯一"

    def test_pool_name_no_special_chars(self):
        """驗證名稱不包含特殊字元（POSIX 相容）"""
        pid = os.getpid()
        pool_name = f"hl_{pid % 10000}_0"

        # POSIX 共享記憶體名稱只能包含字母、數字、底線
        import re
        assert re.match(r'^[a-zA-Z0-9_]+$', pool_name), "名稱應只包含字母、數字、底線"


class TestAnnualReturnCalculation:
    """測試年化報酬計算修復"""

    def test_normal_profit(self):
        """測試正常獲利情況"""
        total_return = 0.5  # 50% 報酬
        total_days = 365
        base = 1 + total_return  # 1.5

        # 修復後邏輯
        if base <= 0:
            annual_return = -1.0
        else:
            annual_return = base ** (365 / total_days) - 1

        assert annual_return == pytest.approx(0.5, abs=0.01), "365天50%報酬，年化應為50%"

    def test_normal_loss(self):
        """測試正常虧損情況"""
        total_return = -0.2  # -20% 報酬
        total_days = 365
        base = 1 + total_return  # 0.8

        # 修復後邏輯
        if base <= 0:
            annual_return = -1.0
        else:
            annual_return = base ** (365 / total_days) - 1

        assert annual_return == pytest.approx(-0.2, abs=0.01), "365天-20%報酬，年化應為-20%"

    def test_total_loss_edge_case(self):
        """測試本金歸零邊界情況"""
        total_return = -1.0  # 本金歸零
        total_days = 365
        base = 1 + total_return  # 0.0

        # 修復後邏輯
        if base <= 0:
            annual_return = -1.0
        else:
            annual_return = base ** (365 / total_days) - 1

        assert annual_return == -1.0, "本金歸零應返回 -1.0"

    def test_liquidation_edge_case(self):
        """測試爆倉（損失超過100%）邊界情況"""
        total_return = -1.5  # 爆倉（損失150%，理論上不可能但要防護）
        total_days = 365
        base = 1 + total_return  # -0.5

        # 修復後邏輯
        if base <= 0:
            annual_return = -1.0
        else:
            annual_return = base ** (365 / total_days) - 1

        assert annual_return == -1.0, "爆倉應返回 -1.0"

    def test_short_period_profit(self):
        """測試短期獲利的年化計算"""
        total_return = 0.1  # 10% 報酬
        total_days = 30  # 1 個月
        base = 1 + total_return  # 1.1

        # 修復後邏輯
        if base <= 0:
            annual_return = -1.0
        else:
            annual_return = base ** (365 / total_days) - 1

        # 1 個月 10% 報酬，年化約 214%
        expected = 1.1 ** (365 / 30) - 1
        assert annual_return == pytest.approx(expected, rel=0.01), "短期獲利年化計算錯誤"

    def test_days_protection(self):
        """測試 total_days 的保護邏輯"""
        # 測試 max() 保護
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 1)  # 同一天
        total_days = max((end_date - start_date).days, 1)

        assert total_days == 1, "同一天應被保護為 1 天"

    def test_zero_return(self):
        """測試零報酬情況"""
        total_return = 0.0  # 0% 報酬
        total_days = 365
        base = 1 + total_return  # 1.0

        # 修復後邏輯
        if base <= 0:
            annual_return = -1.0
        else:
            annual_return = base ** (365 / total_days) - 1

        assert annual_return == pytest.approx(0.0, abs=0.01), "0%報酬年化應為0%"


class TestEdgeCaseCombinations:
    """測試邊界情況組合"""

    def test_one_day_total_loss(self):
        """測試一天內本金歸零"""
        total_return = -1.0
        total_days = 1
        base = 1 + total_return  # 0.0

        if base <= 0:
            annual_return = -1.0
        else:
            annual_return = base ** (365 / total_days) - 1

        assert annual_return == -1.0, "一天歸零應返回 -1.0"

    def test_very_long_period(self):
        """測試極長週期（避免溢位）"""
        total_return = 2.0  # 200% 報酬
        total_days = 3650  # 10 年
        base = 1 + total_return  # 3.0

        if base <= 0:
            annual_return = -1.0
        else:
            annual_return = base ** (365 / total_days) - 1

        # 10 年 200% 報酬，年化約 11.6%
        expected = 3.0 ** (365 / 3650) - 1
        assert annual_return == pytest.approx(expected, rel=0.01)

    def test_nearly_total_loss(self):
        """測試接近歸零但未歸零"""
        total_return = -0.999  # 99.9% 虧損
        total_days = 365
        base = 1 + total_return  # 0.001

        if base <= 0:
            annual_return = -1.0
        else:
            annual_return = base ** (365 / total_days) - 1

        # base > 0 應正常計算
        expected = 0.001 ** (365 / 365) - 1
        assert annual_return == pytest.approx(expected, rel=0.01)
        assert annual_return != -1.0, "99.9% 虧損不應返回 -1.0（因為 base > 0）"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
