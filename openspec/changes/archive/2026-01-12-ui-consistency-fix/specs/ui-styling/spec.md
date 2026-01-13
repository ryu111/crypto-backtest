# UI Styling Capability - Specification (Delta)

## ADDED Requirements

### Requirement: Dark Mode Support
The system SHALL provide a dark mode theme option that can be toggled by the user.

#### Scenario: User toggles dark mode
- **WHEN** user clicks the theme switcher in the sidebar
- **THEN** all UI elements SHALL switch to dark mode colors
- **AND** the selection SHALL persist across page navigation
- **AND** Plotly charts SHALL update with dark-appropriate colors

#### Scenario: Initial theme selection
- **WHEN** user first visits the application
- **THEN** the system SHALL default to light mode
- **AND** display a theme switcher button in the sidebar

---

### Requirement: Design Token System
The system SHALL use a centralized design token system for all visual styling.

#### Scenario: Developer adds new UI element
- **WHEN** developer creates a new component
- **THEN** they SHALL use design tokens from `ui/design_tokens.py`
- **AND NOT** hardcode color values
- **AND** the component SHALL automatically support light/dark themes

#### Scenario: Color token usage
- **GIVEN** a design token `color-primary`
- **WHEN** used in light mode
- **THEN** it SHALL resolve to `#2563eb`
- **WHEN** used in dark mode
- **THEN** it SHALL resolve to `#60a5fa`

---

### Requirement: Plotly Chart Standardization
All Plotly charts SHALL use standardized configuration from `ui/chart_config.py`.

#### Scenario: Chart rendering in light mode
- **WHEN** a chart is rendered with `get_plotly_layout('light')`
- **THEN** it SHALL use:
  - `plot_bgcolor`: `#ffffff`
  - `paper_bgcolor`: `#ffffff`
  - `font.color`: `#111827`

#### Scenario: Chart rendering in dark mode
- **WHEN** a chart is rendered with `get_plotly_layout('dark')`
- **THEN** it SHALL use:
  - `plot_bgcolor`: `#1f2937`
  - `paper_bgcolor`: `#1f2937`
  - `font.color`: `#f9fafb`

---

## MODIFIED Requirements

### Requirement: CSS Management (Enhanced)
The system SHALL manage CSS through a unified `get_common_css()` function.

**Changes**:
- **REMOVED**: Individual CSS definitions in page files
- **ADDED**: CSS variable support for theming
- **ADDED**: Dynamic CSS generation based on theme

#### Scenario: Page loads with theme-aware CSS
- **WHEN** a page calls `st.markdown(get_common_css(), unsafe_allow_html=True)`
- **THEN** it SHALL inject CSS variables matching current theme
- **AND** apply consistent spacing using `--spacing-*` tokens
- **AND** apply consistent colors using `--color-*` tokens

---

### Requirement: Error Handling (Enhanced)
The system SHALL display helpful guidance when data is missing.

**Changes**:
- **REMOVED**: Mock/simulated data display
- **ADDED**: Instructional `st.info()` messages

#### Scenario: No experiment data available
- **WHEN** user navigates to a page requiring experiment data
- **AND** no data exists
- **THEN** the system SHALL display:
  ```
  🚀 開始使用

  尚無實驗資料。請執行以下指令：

  ```bash
  python examples/trend_strategies_example.py
  ```
  ```
- **AND NOT** display simulated/mock data

---

## REMOVED Requirements

### Requirement: Inline CSS in Home.py
**Reason**: Duplicates functionality in `styles.py`, causes inconsistency

**Migration**:
- Replace with `get_common_css()` call
- Remove lines 16-99 in `ui/Home.py`

---

### Requirement: Mock Data in Comparison Page
**Reason**: Misleads users, creates false expectations

**Migration**:
- Remove `load_strategy_results()` mock implementation
- Replace with real data loader or guidance message

---

## Non-Functional Requirements

### NFR-1: Performance
- Theme switching SHALL complete within 100ms
- CSS injection SHALL NOT block page rendering
- Plotly config generation SHALL be cached

### NFR-2: Accessibility
- Color contrast SHALL meet WCAG AA standards (4.5:1) in both themes
- Theme switcher SHALL be keyboard accessible
- All charts SHALL be readable in both themes

### NFR-3: Maintainability
- All colors SHALL be defined in `design_tokens.py`
- No color SHALL be hardcoded in component files
- Design token updates SHALL propagate automatically

---

## Testing Requirements

### Test Coverage
- [x] Light mode visual regression (all pages)
- [x] Dark mode visual regression (all pages)
- [x] Theme toggle functionality
- [x] Plotly chart readability (both themes)
- [x] CSS variable resolution
- [x] Session state persistence across pages

### Manual Testing Checklist
- [x] Navigate through all 6 pages in light mode
- [x] Toggle to dark mode on each page
- [x] Verify all text is readable
- [x] Verify all charts update correctly
- [x] Check color contrast with browser tools
- [x] Test on different screen sizes
