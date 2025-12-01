# Texas Hold'em Poker Test Suite

## Overview

This test suite uses **pytest** to verify that the poker simulation correctly implements all fundamental rules of Texas Hold'em. The tests follow industry best practices for maintainability and scalability.

## Test Methodology

### Design Principles

1. **AAA Pattern (Arrange-Act-Assert)**
   - Each test clearly separates setup, execution, and verification
   - Makes tests easy to read and understand

2. **Given-When-Then Naming**
   - Test names describe: initial state, action, expected result
   - Example: `test_player_can_fold` → "Given a player in a hand, When they fold, Then they should be marked as folded"

3. **Isolated Tests**
   - Each test is independent and can run in any order
   - Uses fixtures for common setup to avoid duplication

4. **Fixtures Over Inheritance**
   - pytest fixtures provide reusable test setup
   - More flexible than class-based inheritance

5. **Single Responsibility**
   - Each test verifies one specific rule or behavior
   - Easier to debug when tests fail

## Running Tests

### Run All Tests
```bash
pytest tests/test_poker_rules.py -v
```

### Run Specific Test
```bash
pytest tests/test_poker_rules.py::test_blinds_posted_correctly -v
```

### Run with Coverage
```bash
pytest tests/test_poker_rules.py --cov=agents --cov=simulation --cov-report=html
```

### Run Tests Matching Pattern
```bash
pytest tests/test_poker_rules.py -k "blind" -v  # All blind-related tests
```

## Test Coverage

### Rules Tested (25 Core Rules)

#### Card Dealing (4 rules)
- ✅ Each player receives 2 hole cards
- ✅ Hole cards are unique (no duplicates)
- ✅ Community cards: 3 flop, 1 turn, 1 river
- ✅ No overlap between player cards and board

#### Blinds (2 rules)
- ✅ Small blind and big blind posted correctly
- ✅ Blind positions rotate with dealer

#### Player Actions (5 rules)
- ✅ Players can fold
- ✅ Players can call
- ✅ Players can raise
- ✅ All-in when betting all chips
- ✅ Cannot bet more than available chips

#### Betting Rules (4 rules)
- ✅ Betting round ends when all players match
- ✅ Minimum raise equals previous raise size
- ✅ Cannot check when facing a bet
- ✅ Must call, raise, or fold when facing bet

#### Pot & Side Pots (4 rules)
- ✅ All bets go into pot
- ✅ Side pots created for all-in scenarios
- ✅ Players only eligible for pots they invested in
- ✅ Equal investments create single pot

#### Hand Rankings & Showdown (4 rules)
- ✅ Best 5-card hand from 7 cards
- ✅ Lower rank number = better hand (treys)
- ✅ Best hand wins pot
- ✅ Ties split pot equally

#### Chip Management (3 rules)
- ✅ Players eliminated at zero chips
- ✅ Chip stacks persist across hands
- ✅ Cannot bet negative amounts

#### Integration Tests (3 scenarios)
- ✅ Complete hand from deal to showdown
- ✅ Pre-flop betting with blinds
- ✅ Multiple all-ins create correct side pots

## Test Categories

### Unit Tests
Test individual components in isolation:
- Card dealing logic
- Betting mechanics
- Pot calculations
- Hand evaluation

### Integration Tests
Test complete workflows:
- Full hand simulation
- Complex side pot scenarios
- Multi-round betting

### Edge Cases
Test boundary conditions:
- All players all-in
- Negative bet amounts
- Zero chip stacks
- Tie scenarios

## Fixtures

### `basic_agents`
4 players with 1000 chips each - standard game setup

### `game`
Pre-configured game with 4 players, blinds 5/10

### `small_stack_agents`
Players with varying stacks (1000, 500, 100, 20) for testing all-in scenarios

## Future Test Additions

As the codebase evolves, add tests for:

1. **AI Decision Making**
   - Equity calculations
   - Bluffing strategies
   - Position-based decisions

2. **Tournament Features**
   - Blind increases
   - Player elimination
   - Final table dynamics

3. **Edge Cases**
   - Network disconnections (if multiplayer)
   - Invalid input handling
   - Concurrent actions

4. **Performance Tests**
   - Monte Carlo simulation speed
   - Large tournament scalability
   - Memory usage

## Bug Fixes from Tests

During test development, we found and fixed:

1. **Negative Bet Bug**: `place_bet(-100)` would add chips instead of rejecting
   - Fixed: Added validation to return 0 for negative amounts

2. **Method Name**: Test revealed missing `determine_winners()` method
   - Fixed: Used existing `get_winner()` method instead

## Contributing

When adding new features:

1. Write tests first (TDD)
2. Follow AAA pattern
3. Use descriptive test names
4. Add to this README
5. Ensure all tests pass before committing

## Test Results

Last run: All 29 tests passing ✅

```
===================== 29 passed in 0.36s =====================
```
