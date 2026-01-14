"""
測試 UltimateLoopController 與 DuckDB 整合

驗證：
1. ExperimentRecorder 使用 DuckDB 儲存
2. 資源正確清理（close() 方法）
3. Context manager 支援
"""

import tempfile
import shutil
from pathlib import Path

from src.automation.ultimate_loop import UltimateLoopController
from src.automation.ultimate_config import UltimateLoopConfig
from src.learning.recorder import ExperimentRecorder


# 專案根目錄
PROJECT_ROOT = Path(__file__).parent


def test_recorder_context_manager():
    """測試 ExperimentRecorder context manager"""
    print("\n=== Test 1: Context Manager ===")

    # 使用專案內的測試目錄（避免 _validate_path 錯誤）
    test_dir = PROJECT_ROOT / "data" / "test_db"
    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        db_path = test_dir / "test_experiments.duckdb"

        # 使用 context manager
        with ExperimentRecorder(db_path=db_path) as recorder:
            print(f"✅ Recorder created (DB: {db_path})")
            print(f"   DB exists: {db_path.exists()}")

        # 檢查資源是否釋放（DB 應該存在）
        print(f"✅ Context manager exited")
        print(f"   DB still exists: {db_path.exists()}")

    finally:
        # 清理測試檔案
        if db_path.exists():
            db_path.unlink()
        if test_dir.exists() and not any(test_dir.iterdir()):
            test_dir.rmdir()


def test_recorder_manual_close():
    """測試手動呼叫 close()"""
    print("\n=== Test 2: Manual Close ===")

    test_dir = PROJECT_ROOT / "data" / "test_db"
    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        db_path = test_dir / "test_experiments2.duckdb"

        # 手動管理
        recorder = ExperimentRecorder(db_path=db_path)
        print(f"✅ Recorder created (DB: {db_path})")

        try:
            # 檢查 close 方法存在
            assert hasattr(recorder, 'close'), "Missing close() method"
            print("✅ close() method available")
        finally:
            recorder.close()
            print("✅ close() called successfully")

    finally:
        # 清理測試檔案
        if db_path.exists():
            db_path.unlink()
        if test_dir.exists() and not any(test_dir.iterdir()):
            test_dir.rmdir()


def test_ultimate_loop_cleanup():
    """測試 UltimateLoopController 資源清理"""
    print("\n=== Test 3: UltimateLoopController Cleanup ===")

    # 使用預設配置（避免參數不匹配）
    config = UltimateLoopConfig()
    config.learning_enabled = True
    config.validation_enabled = False
    config.regime_detection = False
    config.hyperloop_enabled = False

    controller = UltimateLoopController(config, verbose=True)
    print("✅ Controller created")

    # 檢查 recorder 是否初始化
    if controller.recorder:
        print("✅ Recorder initialized")

        # 檢查 cleanup 方法
        assert hasattr(controller, '_cleanup'), "Missing _cleanup() method"
        print("✅ _cleanup() method available")

        # 執行清理
        controller._cleanup()
        print("✅ _cleanup() executed successfully")
    else:
        print("⚠️  Recorder not initialized (module might not be available)")


def test_ultimate_loop_context_manager():
    """測試 UltimateLoopController context manager"""
    print("\n=== Test 4: Controller Context Manager ===")

    config = UltimateLoopConfig()
    config.learning_enabled = True
    config.validation_enabled = False
    config.regime_detection = False
    config.hyperloop_enabled = False

    with UltimateLoopController(config) as controller:
        print("✅ Controller entered context")
        if controller.recorder:
            print("✅ Recorder available")

    print("✅ Controller exited context (auto cleanup)")


def test_duckdb_persistence():
    """測試 DuckDB 持久化"""
    print("\n=== Test 5: DuckDB Persistence ===")

    from src.types import ExperimentRecord
    from datetime import datetime

    test_dir = PROJECT_ROOT / "data" / "test_db"
    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        db_path = test_dir / "test_persistence.duckdb"

        # 建立測試實驗
        test_exp = ExperimentRecord(
            id='test_001',
            timestamp=datetime.now(),
            strategy={'name': 'test_strategy', 'type': 'trend', 'params': {}},
            config={'symbol': 'BTCUSDT', 'timeframe': '1h'},
            results={
                'sharpe_ratio': 1.5,
                'total_return': 0.25,
                'max_drawdown': 0.10,
            },
            validation={'grade': 'B', 'stages_passed': [1, 2, 3]},
            status='completed',
        )

        # 寫入
        with ExperimentRecorder(db_path=db_path) as recorder:
            recorder.repo.insert_experiment(test_exp)
            print(f"✅ Experiment inserted: {test_exp.id}")

        # 讀取（新連線）
        with ExperimentRecorder(db_path=db_path) as recorder:
            retrieved = recorder.get_experiment('test_001')
            assert retrieved is not None, "Experiment not found"
            assert retrieved.id == 'test_001', "ID mismatch"
            assert retrieved.sharpe_ratio == 1.5, "Sharpe ratio mismatch"
            print(f"✅ Experiment retrieved: {retrieved.id}")
            print(f"   Sharpe: {retrieved.sharpe_ratio}, Return: {retrieved.total_return}")

    finally:
        # 清理測試檔案
        if db_path.exists():
            db_path.unlink()
        if test_dir.exists() and not any(test_dir.iterdir()):
            test_dir.rmdir()


def main():
    """執行所有測試"""
    print("=" * 60)
    print("UltimateLoopController DuckDB Integration Test")
    print("=" * 60)

    try:
        test_recorder_context_manager()
        test_recorder_manual_close()
        test_ultimate_loop_cleanup()
        test_ultimate_loop_context_manager()
        test_duckdb_persistence()

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
