# UI Data Integration - 策略頁面數據整合規格

## ADDED Requirements

### Requirement: UI SHALL load real backtest results

The system SHALL integrate real backtest results into the strategy page, replacing hardcoded mock data.

#### Scenario: Display equity curve from backtest result

- **GIVEN** a user selects a strategy from the strategy list
- **AND** the strategy has a completed backtest with experiment ID
- **WHEN** the user expands the strategy details
- **THEN** the system SHALL load the equity curve from `results/{exp_id}/equity_curve.csv`
- **AND** the system SHALL display the equity curve with real datetime index
- **AND** the system SHALL show the complete backtest period (no 100-day limitation)

#### Scenario: Display monthly returns from daily returns

- **GIVEN** a user views strategy details
- **WHEN** the monthly returns heatmap is rendered
- **THEN** the system SHALL load daily returns from `results/{exp_id}/daily_returns.csv`
- **AND** the system SHALL calculate monthly returns by resampling daily returns
- **AND** the system SHALL display returns for all months in the backtest period
- **AND** the system SHALL use consistent time range with equity curve

#### Scenario: Handle missing data gracefully

- **GIVEN** a user selects a strategy
- **WHEN** equity curve data file does not exist
- **THEN** the system SHALL display a warning message
- **AND** the system SHALL explain possible reasons (legacy experiment, deleted file)
- **AND** the system SHALL provide actionable suggestions (re-run backtest, view other strategies)
- **AND** the system SHALL NOT crash or show error traces

---

### Requirement: Time range synchronization

The system SHALL provide synchronized time range control across all charts.

#### Scenario: User adjusts time range

- **GIVEN** equity curve and monthly returns are both displayed
- **WHEN** user adjusts the time range slider
- **THEN** both charts SHALL update to show only the selected time range
- **AND** the update SHALL be immediate (no page reload required)
- **AND** the selected range SHALL persist until user changes it or resets

#### Scenario: User resets time range

- **GIVEN** user has zoomed into a specific time period
- **WHEN** user clicks "Reset Range" button
- **THEN** both charts SHALL return to full backtest period
- **AND** the time slider SHALL reset to min/max dates

#### Scenario: Time range validation

- **GIVEN** user adjusts time range
- **WHEN** selected range is less than 7 days
- **THEN** the system SHALL display a warning "Time range too short for meaningful analysis"
- **WHEN** selected range spans more than 2 years
- **THEN** the system SHALL apply sampling to improve performance
- **AND** the system SHALL display a note "Data sampled for performance (every N days)"

---

### Requirement: Data caching and performance

The system SHALL optimize data loading to prevent unnecessary reloads.

#### Scenario: Cache experiment data

- **GIVEN** a user views strategy details
- **WHEN** equity curve data is loaded
- **THEN** the system SHALL cache the data using `@st.cache_data`
- **AND** subsequent views of the same strategy SHALL use cached data
- **AND** cache SHALL invalidate when experiment data is updated

#### Scenario: Lazy load detailed data

- **GIVEN** strategy list page displays summary metrics
- **WHEN** page initially loads
- **THEN** the system SHALL only load summary data (from experiments.json)
- **AND** the system SHALL NOT load equity curves for all strategies
- **WHEN** user expands a specific strategy
- **THEN** the system SHALL load detailed data only for that strategy

---

### Requirement: Data structure consistency

The system SHALL maintain consistent data structures between backend and UI.

#### Scenario: Equity curve format

- **GIVEN** equity curve is saved by backtest engine
- **WHEN** UI loads the data
- **THEN** the data SHALL be a CSV with columns: `['date', 'equity']`
- **AND** `date` column SHALL be parseable as datetime
- **AND** `equity` column SHALL be numeric (float)
- **AND** data SHALL be sorted by date ascending

#### Scenario: Daily returns format

- **GIVEN** daily returns are saved by backtest engine
- **WHEN** UI loads the data
- **THEN** the data SHALL be a CSV with columns: `['date', 'return']`
- **AND** `return` SHALL be daily percentage change (e.g., 0.02 for 2%)
- **AND** data SHALL align with equity curve dates

---

## MODIFIED Requirements

### Requirement: Strategy list data source

**BEFORE**:
- Load strategy results from hardcoded mock data
- Fixed 4 sample strategies

**AFTER**:
- Load strategy results from `learning/experiments.json`
- Include all recorded experiments
- Add `experiment_id` field for loading detailed data
- Preserve existing summary metrics (sharpe, return, drawdown, etc.)

**Migration**:
- Update `load_strategy_results()` function in `ui/pages/2_Strategies.py`
- Map experiment JSON structure to DataFrame columns
- Maintain backward compatibility with existing filter/sort logic

---

### Requirement: Chart rendering functions

**BEFORE**:
- `plot_equity_curve()` generates random equity data (100 days)
- `plot_monthly_heatmap()` generates random monthly returns (12 months)
- No connection to real backtest results

**AFTER**:
- `plot_equity_curve(strategy_name, experiment_id)` loads real equity curve
- `plot_monthly_heatmap(strategy_name, experiment_id)` calculates from real daily returns
- Accept time range parameter from session state
- Handle data loading errors gracefully

**Migration**:
- Refactor both functions to accept `experiment_id` parameter
- Add data loading logic using `ui/utils/data_loader.py`
- Add error handling for missing files
- Preserve existing Plotly styling and layout

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Strategy List Page Load                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  load_strategy_results()                                        │
│  ├─ Read learning/experiments.json                             │
│  ├─ Extract summary metrics                                    │
│  └─ Return DataFrame with experiment_id                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  User selects strategy → Expand details                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  plot_equity_curve(strategy_name, experiment_id)               │
│  ├─ load_equity_curve(experiment_id)                           │
│  │   └─ Read results/{exp_id}/equity_curve.csv                │
│  ├─ Apply time_range filter from session_state                │
│  └─ Render Plotly chart                                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  plot_monthly_heatmap(strategy_name, experiment_id)            │
│  ├─ load_daily_returns(experiment_id)                          │
│  │   └─ Read results/{exp_id}/daily_returns.csv               │
│  ├─ calculate_monthly_returns(daily_returns)                   │
│  ├─ Apply time_range filter from session_state                │
│  └─ Render Plotly heatmap                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
learning/
├── experiments.json          # Summary data (existing)
└── results/
    └── {exp_id}/
        ├── equity_curve.csv   # NEW: Equity curve time series
        ├── daily_returns.csv  # NEW: Daily returns time series
        └── trades.csv         # Existing: Trade records

ui/
├── pages/
│   └── 2_Strategies.py       # MODIFIED: Load real data
└── utils/
    └── data_loader.py        # NEW: Data loading helpers
```

---

## API Specifications

### data_loader.py

```python
def load_equity_curve(exp_id: str) -> pd.Series:
    """
    Load equity curve for an experiment.

    Args:
        exp_id: Experiment ID (e.g., 'exp_20260111_120000')

    Returns:
        pd.Series with DatetimeIndex and equity values

    Raises:
        FileNotFoundError: If equity curve file does not exist
        ValueError: If data format is invalid
    """
    pass

def load_daily_returns(exp_id: str) -> pd.Series:
    """
    Load daily returns for an experiment.

    Args:
        exp_id: Experiment ID

    Returns:
        pd.Series with DatetimeIndex and return values (as decimals, e.g., 0.02)

    Raises:
        FileNotFoundError: If daily returns file does not exist
        ValueError: If data format is invalid
    """
    pass

def calculate_monthly_returns(daily_returns: pd.Series) -> pd.DataFrame:
    """
    Calculate monthly returns from daily returns.

    Args:
        daily_returns: Daily returns with DatetimeIndex

    Returns:
        DataFrame with columns ['year', 'month', 'return_pct']
        where return_pct is percentage (e.g., 5.2 for 5.2%)
    """
    pass

def load_experiment_data(exp_id: str) -> Dict[str, Any]:
    """
    Load complete experiment data including equity and returns.

    Args:
        exp_id: Experiment ID

    Returns:
        {
            'equity_curve': pd.Series,
            'daily_returns': pd.Series,
            'summary': dict  # from experiments.json
        }

    Raises:
        FileNotFoundError: If experiment data not found
    """
    pass
```

---

## CSV Data Formats

### equity_curve.csv

```csv
date,equity
2024-01-01,10000.00
2024-01-02,10125.50
2024-01-03,10085.30
...
```

- **date**: ISO format (YYYY-MM-DD) or datetime string
- **equity**: Numeric, portfolio value in USD

### daily_returns.csv

```csv
date,return
2024-01-01,0.0000
2024-01-02,0.0125
2024-01-03,-0.0039
...
```

- **date**: ISO format (YYYY-MM-DD) or datetime string
- **return**: Decimal format (0.0125 = 1.25% return)

---

## Error Handling Specifications

### Missing Data Errors

```python
# When equity_curve.csv does not exist
try:
    equity_curve = load_equity_curve(exp_id)
except FileNotFoundError:
    st.warning("""
    ⚠️ 權益曲線數據缺失

    此實驗可能在數據儲存機制更新前執行，或數據檔案已被刪除。

    **建議操作**：
    - 重新執行此策略的回測
    - 查看其他已有完整數據的策略
    """)
    st.stop()
```

### Invalid Data Format

```python
try:
    equity_curve = load_equity_curve(exp_id)
except ValueError as e:
    st.error(f"""
    ❌ 數據格式錯誤

    無法解析權益曲線數據：{str(e)}

    請聯繫系統管理員檢查數據完整性。
    """)
    st.stop()
```

---

## Testing Requirements

### Unit Tests

```python
def test_load_equity_curve_success():
    """Test loading valid equity curve"""
    exp_id = "exp_20260111_120000"
    equity = load_equity_curve(exp_id)

    assert isinstance(equity, pd.Series)
    assert isinstance(equity.index, pd.DatetimeIndex)
    assert len(equity) > 0
    assert equity.dtype == float

def test_load_equity_curve_not_found():
    """Test handling of missing equity curve"""
    with pytest.raises(FileNotFoundError):
        load_equity_curve("exp_nonexistent")

def test_calculate_monthly_returns():
    """Test monthly returns calculation"""
    daily_returns = pd.Series([0.01, 0.02, -0.01], index=pd.date_range('2024-01-01', periods=3))
    monthly = calculate_monthly_returns(daily_returns)

    assert 'year' in monthly.columns
    assert 'month' in monthly.columns
    assert 'return_pct' in monthly.columns
```

### Integration Tests

```python
def test_strategy_page_loads_real_data(streamlit_app):
    """Test strategy page displays real equity curve"""
    # Select a strategy
    select_strategy("MA Cross (10/30)")

    # Expand details
    expand_details()

    # Verify equity curve chart exists
    assert chart_exists("權益曲線")

    # Verify chart has real data (not random)
    chart_data = get_chart_data("權益曲線")
    assert len(chart_data) > 100  # Real backtest should have more than 100 days

def test_time_range_sync(streamlit_app):
    """Test time range synchronization"""
    select_strategy("MA Cross (10/30)")
    expand_details()

    # Adjust time range
    set_time_range("2024-01-01", "2024-06-30")

    # Verify both charts updated
    equity_range = get_chart_time_range("權益曲線")
    monthly_range = get_chart_time_range("月度報酬")

    assert equity_range == ("2024-01-01", "2024-06-30")
    assert monthly_range == ("2024-01-01", "2024-06-30")
```

---

## Performance Requirements

- **Load time**: Strategy summary data SHALL load within 1 second
- **Detail load time**: Equity curve and returns SHALL load within 2 seconds
- **Chart update**: Time range changes SHALL update charts within 500ms
- **Memory**: UI SHALL NOT load more than 10 equity curves simultaneously (use lazy loading)

---

## Security & Validation

- **Path validation**: All file paths SHALL be validated to prevent directory traversal attacks
- **Data validation**: Loaded CSV SHALL be validated for expected columns and data types
- **Error exposure**: Error messages SHALL NOT expose internal file paths or system details to users

---

## Acceptance Criteria

✅ Strategy page loads real backtest results from experiments.json
✅ Equity curve displays complete backtest period (not limited to 100 days)
✅ Monthly returns calculated from real daily returns (not hardcoded 12 months)
✅ Time range slider controls both charts simultaneously
✅ Missing data handled gracefully with user-friendly messages
✅ Data loading uses caching to improve performance
✅ All existing filter/sort/pagination features continue to work
✅ Unit tests cover all data loading functions
✅ Integration tests verify end-to-end data flow
