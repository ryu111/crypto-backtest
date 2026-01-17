# GP Strategy Generation Specification

## ADDED Requirements

### Requirement: GP Primitive System

The system SHALL provide a typed primitive set for constructing trading strategy expressions.

#### Scenario: Indicator Terminals
- **GIVEN** a GP evolution session is initialized
- **WHEN** the primitive set is created
- **THEN** the following indicator terminals SHALL be available:
  - `close` (float): Current closing price
  - `rsi_7`, `rsi_14` (float): RSI with different periods
  - `ma_10`, `ma_20`, `ma_50` (float): Moving averages
  - `atr_14` (float): Average True Range
  - `bb_upper`, `bb_lower` (float): Bollinger Bands
  - `macd_line`, `macd_signal` (float): MACD components

#### Scenario: Comparison Primitives
- **GIVEN** indicator values are available
- **WHEN** comparison primitives are applied
- **THEN** the following operations SHALL be supported:
  - `gt(a, b)`: Returns True if a > b
  - `lt(a, b)`: Returns True if a < b
  - `ge(a, b)`: Returns True if a >= b
  - `le(a, b)`: Returns True if a <= b
  - `cross_above(a, b)`: Returns True if a crosses above b
  - `cross_below(a, b)`: Returns True if a crosses below b

#### Scenario: Logic Primitives
- **GIVEN** boolean comparison results
- **WHEN** logic primitives are applied
- **THEN** the following operations SHALL be supported:
  - `and_(a, b)`: Logical AND
  - `or_(a, b)`: Logical OR
  - `not_(a)`: Logical NOT
  - `if_then_else(cond, t, f)`: Conditional selection

---

### Requirement: Fitness Evaluation

The system SHALL evaluate GP individuals using the existing backtest engine.

#### Scenario: Fitness Calculation
- **GIVEN** a valid GP expression tree
- **WHEN** fitness is evaluated
- **THEN** the system SHALL:
  1. Compile the expression to executable strategy
  2. Run backtest using `BacktestEngine`
  3. Calculate weighted fitness score from:
     - Sharpe Ratio (weight: 0.5)
     - Max Drawdown (weight: 0.3)
     - Win Rate (weight: 0.1)
     - Complexity penalty (weight: 0.1)

#### Scenario: Invalid Individual Handling
- **GIVEN** an invalid GP expression tree
- **WHEN** fitness evaluation fails
- **THEN** the system SHALL return a penalty fitness value (-1000.0)

#### Scenario: Minimum Trade Requirement
- **GIVEN** a valid strategy with less than 30 trades
- **WHEN** fitness is evaluated
- **THEN** the system SHALL return a penalty fitness value (-500.0)

---

### Requirement: Overfitting Protection

The system SHALL implement multiple layers of overfitting protection.

#### Scenario: Tree Depth Limit
- **GIVEN** a GP crossover or mutation operation
- **WHEN** the resulting tree exceeds depth 17
- **THEN** the operation SHALL be rejected and original tree preserved

#### Scenario: Node Count Limit
- **GIVEN** a GP individual
- **WHEN** the node count exceeds 50
- **THEN** the fitness evaluation SHALL return a penalty value

#### Scenario: Complexity Penalty
- **GIVEN** a GP individual with N nodes (N > 10)
- **WHEN** fitness is calculated
- **THEN** a penalty of `0.01 * (N - 10)` SHALL be subtracted

---

### Requirement: Evolution Engine

The system SHALL provide a complete GP evolution engine.

#### Scenario: Population Initialization
- **GIVEN** evolution is started
- **WHEN** initial population is created
- **THEN** the system SHALL:
  - Create 100 individuals (configurable)
  - Use Half-and-Half initialization
  - Ensure all individuals are valid typed trees

#### Scenario: Evolution Loop
- **GIVEN** an initialized population
- **WHEN** evolution proceeds
- **THEN** for each generation the system SHALL:
  1. Evaluate fitness of all individuals
  2. Select parents using tournament selection (size 3)
  3. Apply crossover with probability 0.7
  4. Apply mutation with probability 0.2
  5. Update Hall of Fame (top 5)
  6. Record statistics

#### Scenario: Parallel Evaluation
- **GIVEN** `n_jobs != 1` in configuration
- **WHEN** fitness evaluation is performed
- **THEN** the system SHALL use multiprocessing for parallel evaluation

#### Scenario: Early Stopping
- **GIVEN** evolution is running
- **WHEN** best fitness has not improved for 10 generations
- **THEN** the system MAY terminate evolution early

---

### Requirement: Strategy Conversion

The system SHALL convert GP expression trees to executable BaseStrategy instances.

#### Scenario: Dynamic Compilation
- **GIVEN** a GP expression tree
- **WHEN** `EvolvedStrategy.from_expression(tree)` is called
- **THEN** the system SHALL:
  1. Compile expression to callable function
  2. Create EvolvedStrategy instance
  3. Store original expression as metadata

#### Scenario: Static File Generation
- **GIVEN** a GP expression tree and output path
- **WHEN** `Converter.save_strategy(tree, path)` is called
- **THEN** the system SHALL:
  1. Generate Python code for the strategy
  2. Include complete class definition inheriting BaseStrategy
  3. Include `calculate_indicators()` method
  4. Include `generate_signals()` method
  5. Include original expression in docstring

#### Scenario: Generated Strategy Compatibility
- **GIVEN** a generated strategy file
- **WHEN** the strategy is loaded and used
- **THEN** it SHALL:
  - Inherit from BaseStrategy
  - Implement all required abstract methods
  - Be compatible with BacktestEngine
  - Be compatible with CompositeStrategy
  - Be registerable in StrategyRegistry

---

### Requirement: System Integration

The system SHALL integrate with existing modules.

#### Scenario: Backtest Engine Integration
- **GIVEN** an evolved strategy
- **WHEN** backtest is requested
- **THEN** the system SHALL use `BacktestEngine.run()` without modifications

#### Scenario: Learning System Integration
- **GIVEN** evolution completes successfully
- **WHEN** best strategies are identified
- **THEN** the system SHALL:
  1. Record to `learning/insights.md`
  2. Store to Memory MCP with tags: `strategy,evolved,gp`

#### Scenario: Automation Loop Integration
- **GIVEN** AI automation loop is running
- **WHEN** GP evolution is triggered
- **THEN** the system SHALL:
  1. Run evolution with configured parameters
  2. Validate top strategies with Walk-Forward
  3. Save validated strategies to registry
  4. Record experiment results

---

## Acceptance Criteria

### AC1: End-to-End Evolution
- [ ] Can evolve a population from random initialization
- [ ] Produces at least one strategy with Sharpe > 1.0
- [ ] Evolution completes within 1 hour for 50 generations

### AC2: Strategy Usability
- [ ] Generated strategies can be loaded and imported
- [ ] Generated strategies pass all BaseStrategy validation
- [ ] Generated strategies produce valid backtest results

### AC3: Overfitting Protection
- [ ] No strategy exceeds max tree depth
- [ ] Complexity penalty is correctly applied
- [ ] Walk-Forward validation can be executed

### AC4: Integration
- [ ] Works with existing BacktestEngine
- [ ] Works with CompositeStrategy
- [ ] Records to learning system
