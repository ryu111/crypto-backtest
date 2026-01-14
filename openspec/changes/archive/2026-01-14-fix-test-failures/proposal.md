# fix-test-failures

## Why

專案有 71 個測試失敗，主要原因：

1. **API 變更** (36 tests): `ExperimentRecorder` 從 JSON 遷移到 DuckDB，測試仍使用舊 API
2. **浮點數精度** (12 tests): 使用 `==` 而非 `pytest.approx()` 比較浮點數
3. **效能假設** (7 tests): 測試假設 Polars 比 Pandas 快，實際不一定成立
4. **邏輯變更** (5 tests): 業務邏輯變更但測試未更新
5. **參數缺失** (4 tests): 策略測試缺少必要參數
6. **其他** (7 tests): 各種小問題

## What Changes

### 1. 更新 Recorder 測試 (36 tests)
- 移除對 `experiments_file` 屬性的依賴
- 改用 DuckDB Repository API
- 更新 fixture 以建立 DuckDB 測試環境

### 2. 修復浮點數比較 (12 tests)
- 使用 `pytest.approx()` 或 `assertAlmostEqual()`
- 涵蓋 selector、perpetual、cleaner 測試

### 3. 調整效能測試 (7 tests)
- 移除硬編碼的效能閾值
- 改為驗證功能正確性，而非效能
- 修復 Polars `min_periods` → `min_samples` deprecation

### 4. 更新邏輯測試 (5 tests)
- 根據實際業務邏輯調整期望值
- 修復 quality_score 計算邏輯

### 5. 修復策略測試 (4 tests)
- 補充缺失的 params
- 修正 param_space 定義

### 6. 其他修復 (7 tests)
- orchestrator、portfolio、walk-forward 等
