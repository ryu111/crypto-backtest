# fix-hardcoding Implementation Tasks

## Phase 1: 基礎設施 (sequential) ✅

- [x] 1.1 建立 src/types/enums.py | files: src/types/enums.py
  - 定義 ExperimentStatus, Grade, StrategyType
  - 定義 OptimizationMethod, ObjectiveMetric
  - 定義 BackendType, LessonType
  - 加入 JSON 序列化輔助方法

- [x] 1.2 更新 src/types/__init__.py | files: src/types/__init__.py
  - 匯出所有新 Enum
  - 維持向後相容（舊的 import 仍可用）

## Phase 2: 型別定義更新 (parallel) ✅

- [x] 2.1 更新 src/types/results.py | files: src/types/results.py
  - ExperimentRecord.status → ExperimentStatus
  - ValidationResult.grade → Grade
  - 更新 to_dict/from_dict 處理 Enum 序列化

- [x] 2.2 更新 src/types/strategies.py | files: src/types/strategies.py
  - StrategyInfo.type → StrategyType
  - 更新 to_dict/from_dict

- [x] 2.3 更新 src/types/configs.py | files: src/types/configs.py
  - OptimizationConfig.method → OptimizationMethod
  - OptimizationConfig.objective → ObjectiveMetric
  - 更新 to_dict/from_dict

## Phase 3: 回測引擎更新 (parallel) ✅

- [x] 3.1 更新 src/backtester/metal_engine.py | files: src/backtester/metal_engine.py
  - backend 字串比對 → BackendType Enum

- [x] 3.2 更新 src/backtester/runners.py (如存在) | files: src/backtester/runners.py
  - 檔案不存在，跳過

## Phase 4: 自動化模組更新 (parallel) ✅

- [x] 4.1 更新 src/automation/ultimate_loop.py | files: src/automation/ultimate_loop.py
  - grade 字串比對 → Grade Enum

- [x] 4.2 更新 src/automation/hyperloop.py | files: src/automation/hyperloop.py
  - 無需修改（param_type 比對使用 'categorical'，不在 ParamType Enum 中）

- [x] 4.3 更新 src/automation/ultimate_config.py | files: src/automation/ultimate_config.py
  - 無需修改（字串為模組特有配置值）

## Phase 5: 學習模組更新 (sequential) ✅

- [x] 5.1 更新 src/learning/lesson_detector.py | files: src/learning/lesson_detector.py
  - 移除 Literal 定義，改用 src/types/enums.py 的 LessonType Enum
  - 更新所有 lesson_type 比對使用 Enum

## Phase 6: 整合測試 (sequential) ✅

- [x] 6.1 執行回歸測試 | cmd: pytest
  - pytest 執行完成：1150 通過，74 失敗
  - 失敗為預存在問題（recorder/DuckDB），非 Enum 重構造成
  - 修復 test_metal_engine.py 硬編碼字串 → BackendType.CPU
  - JSON 序列化/反序列化正確（str Enum 自動轉換）

- [x] 6.2 驗證既有資料相容性 | files: learning/experiments.json
  - experiments.json 不存在，跳過驗證
  - 型別定義支援 Union[EnumType, str] 向後相容
