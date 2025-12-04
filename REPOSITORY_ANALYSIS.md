# Texas Hold'em Repository Analysis

## Table of Contents
1. [Repository Analysis](#1-repository-analysis)
2. [Quality Review](#2-quality-review)
3. [Feature Review](#3-feature-review)
4. [Task Breakdown](#4-task-breakdown)

---

## 1. Repository Analysis

### Main Components and Modules

The repository is organized into the following key directories:

| Directory | Purpose |
|-----------|---------|
| `agents/` | Player implementations (AI agents, human player, RL agent) |
| `simulation/` | Game engine and Gymnasium environment for RL training |
| `training/` | PPO model architecture and training loop |
| `tests/` | Comprehensive test suite (119 tests) |
| `utils/` | Utility functions (device selection, TensorBoard reader) |
| `connect/` | Tournament connection code (for external competitions) |
| `data/` | Data storage for training datasets |
| `tools/` | Visualization and configuration tools |

### Overall Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ENTRY POINTS                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  main.py           - Interactive play vs AI agents                          │
│  play_vs_rl.py     - Play against trained RL model                          │
│  train_rl_model.py - Train PPO agent                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AGENTS LAYER                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  PokerAgent (agent.py)                                                      │
│    ├── Base class for all players                                           │
│    ├── Equity calculations (Monte Carlo simulation)                         │
│    └── Default decision-making strategy                                     │
│                                                                             │
│  PokerPlayer (poker_player.py) - Abstract interface                         │
│    ├── MonteCarloAgent - MC simulation-based decisions                      │
│    ├── RandomAgent - Random decisions (testing)                             │
│    └── CallStationAgent - Always calls (baseline)                           │
│                                                                             │
│  HumanPlayer (human_player.py) - Interactive console player                 │
│  RLAgent (rl_agent.py) - Wrapper for trained PPO model                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SIMULATION LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  TexasHoldemSimulation (poker_simulator.py)                                 │
│    ├── Game state management (deck, board, pot)                             │
│    ├── Betting round execution                                              │
│    ├── Side pot calculation                                                 │
│    └── Hand evaluation & winner determination                               │
│                                                                             │
│  PokerEnv (poker_env.py) - Gymnasium Environment                            │
│    ├── Observation encoding (cards, hand features, numeric state)           │
│    ├── Action interpretation (fold/check/call/raise)                        │
│    ├── Reward calculation (normalized chip gain/loss)                       │
│    └── Episode management (reset, step)                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TRAINING LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  PokerPPOModel (ppo_model.py)                                               │
│    ├── Card embedding: 53-dim → 64-dim (per card, 7 cards total)            │
│    ├── Hand embedding: 10-dim → 32-dim                                      │
│    ├── Numeric embedding: 42-dim → 64-dim                                   │
│    ├── Shared trunk: 544 → 256 → 256                                        │
│    └── Three heads:                                                         │
│        ├── Fold head: Bernoulli logit                                       │
│        ├── Bet head: Beta distribution (α, β)                               │
│        └── Value head: V(s) estimate                                        │
│                                                                             │
│  PPOTrainer (train_rl_model.py)                                             │
│    ├── Rollout buffer for experience collection                             │
│    ├── GAE (Generalized Advantage Estimation)                               │
│    ├── Clipped objective + value clipping                                   │
│    └── TensorBoard logging & checkpointing                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Game Flow Implementation

The game flow follows standard Texas Hold'em rules:

```
1. SETUP
   ├── Create agents with starting chips
   ├── Create TexasHoldemSimulation instance
   └── Initialize dealer position

2. NEW HAND
   ├── reset_for_new_hand() - Reset game state
   ├── post_blinds() - Post small/big blinds
   └── deal_hole_cards() - Deal 2 cards per player

3. BETTING ROUNDS (4 total)
   ├── Pre-flop: No community cards
   │   └── run_betting_round("Pre-flop")
   ├── Flop: deal_flop() → 3 community cards
   │   └── run_betting_round("Flop")
   ├── Turn: deal_turn() → 4th community card
   │   └── run_betting_round("Turn")
   └── River: deal_river() → 5th community card
       └── run_betting_round("River")

4. SHOWDOWN
   ├── evaluate_hands() - Rank all hands
   ├── create_side_pots() - Handle all-in scenarios
   └── award_pot() - Distribute winnings

5. REPEAT from step 2 (or end tournament)
```

### Card/Hand Evaluation

The repository uses the `treys` library for card and hand evaluation:

1. **Card Representation**: Treys uses prime-based integer encoding
2. **Hand Evaluation**: `Evaluator.evaluate(board, hole_cards)` returns a score (lower = better)
3. **Hand Classes**: Royal Flush (1) → High Card (10)
4. **Equity Calculation**: Monte Carlo simulation samples possible outcomes

**Decision Making Flow:**

```python
# For PokerAgent (rule-based)
1. If pre-flop → _preflop_decision() based on hand strength heuristics
2. If post-flop → evaluate_hand() to get hand rank percentage
3. Compare percentage to thresholds (0.3 = strong, 0.6 = medium)
4. Return action: fold/check/call/raise

# For RL Agent (neural network)
1. Build observation tensor (423 dimensions)
2. Forward pass through PokerPPOModel
3. Sample action from distributions (Bernoulli for fold, Beta for bet)
4. interpret_action() converts to PokerAction
```

### Entry Points and Important Functions

| Entry Point | File | Description |
|-------------|------|-------------|
| `play_interactive()` | main.py | Human vs AI interactive mode |
| `run_tournament()` | main.py | AI tournament simulation |
| `run_full_game()` | main.py | Single hand AI vs AI |
| `play_vs_rl()` | play_vs_rl.py | Human vs trained RL model |
| `PPOTrainer.train()` | train_rl_model.py | PPO training loop |
| `PPOTrainer.debug_run()` | train_rl_model.py | Debug mode for diagnostics |

**Critical Functions:**

| Function | File | Importance |
|----------|------|------------|
| `PokerAgent.make_decision()` | agent.py | Core AI decision logic |
| `PokerAgent.calculate_equity()` | agent.py | Monte Carlo equity estimation |
| `TexasHoldemSimulation.run_betting_round()` | poker_simulator.py | Betting round controller |
| `TexasHoldemSimulation.create_side_pots()` | poker_simulator.py | Side pot logic |
| `PokerEnv.step()` | poker_env.py | RL environment step |
| `PokerEnv._get_observation()` | poker_env.py | Observation encoding |
| `interpret_action()` | poker_env.py | Action decoding |
| `PokerPPOModel.forward()` | ppo_model.py | Neural network forward pass |
| `PPOTrainer._update_policy()` | train_rl_model.py | PPO update step |

---

## 2. Quality Review

### Bugs and Issues Found

| Issue | Severity | Location | Description |
|-------|----------|----------|-------------|
| ✅ Fixed | High | poker_env.py | Player money wasn't reset between episodes |
| ✅ Fixed | High | poker_env.py | Reward calculation used wrong baseline |
| ✅ Fixed | Medium | poker_env.py | Illegal fold when nothing to call |
| ✅ Fixed | Medium | ppo_model.py | Saturated fold probabilities (binary outputs) |
| ⚠️ Potential | Low | poker_simulator.py | Betting round may not handle edge cases with multiple all-ins |
| ⚠️ Potential | Low | agent.py | Pre-flop decision doesn't consider position |

### Code Smells and Inconsistent Patterns

1. **Dual Agent Interfaces**: Both `PokerAgent` (agent.py) and `PokerPlayer` (poker_player.py) exist with overlapping functionality
   - `PokerAgent`: Used by TexasHoldemSimulation
   - `PokerPlayer`: Used by PokerEnv
   - This creates confusion and code duplication

2. **Inconsistent Action Representation**:
   - `PokerAgent.make_decision()` returns `(str, int)` tuple
   - `PokerPlayer.get_action()` returns `PokerAction` object
   - `interpret_action()` converts `(p_fold, bet_scalar)` to `PokerAction`

3. **Magic Numbers**:
   - `poker_simulator.py` line 133: `agent.get_chips() > 0` should use constant
   - `monte_carlo_agent.py`: Multiple hardcoded thresholds (0.3, 0.6, 0.25, etc.)

4. **Duplicated Equity Calculation**:
   - `PokerAgent._calculate_all_equities()` in agent.py
   - `MonteCarloAgent._calculate_equity()` in monte_carlo_agent.py
   - Similar but not identical implementations

### Missing Documentation

| Area | Issue |
|------|-------|
| API Docs | No docstrings in `__init__.py` files |
| Architecture | No high-level design document |
| Training Guide | No guide for hyperparameter tuning |
| Tournament Guide | No documentation for `connect/` module |
| Data Format | `data/datasets.md` exists but raw/processed folders are empty |

### Missing Tests

| Area | Missing Coverage |
|------|------------------|
| RLAgent | No tests for `rl_agent.py` |
| Human Player | No tests for `human_player.py` |
| Training | No tests for `train_rl_model.py` |
| Integration | No end-to-end training tests |

### Poor Abstractions

1. **No Game State Object**: Game state is spread across multiple properties in TexasHoldemSimulation instead of a single state object

2. **No Action Validator**: Actions are validated inline in betting round logic rather than in a dedicated validator

3. **No Tournament Manager**: Tournament logic is duplicated in `main.py` and needs a proper abstraction

### Duplicated Code

| Location 1 | Location 2 | Issue |
|------------|------------|-------|
| `main.py:run_single_hand()` | `play_vs_rl.py:play_vs_rl()` | Nearly identical hand execution logic |
| `agent.py:_calculate_all_equities()` | `monte_carlo_agent.py:_calculate_equity()` | Similar MC simulation |
| `agent.py:reset_for_new_hand()` | `poker_player.py:reset()` | Same reset logic |

### Unused Code

| File | Code | Status |
|------|------|--------|
| `tests/test_simulation.py` | Empty file | Can be removed |
| `tests/test_gameplay.py` | Empty file | Can be removed |
| `data/raw/` | Empty directory | Placeholder |
| `data/processed/` | Empty directory | Placeholder |

### Confusing Naming

| Current | Suggested | Reason |
|---------|-----------|--------|
| `PokerAgent` | `BasePokerAgent` | Clarify it's the base class |
| `PokerPlayer` | `IPokerPlayer` | Interface convention |
| `poker_simulator.py` | `game_engine.py` | More accurate name |
| `poker_env.py` | `rl_environment.py` | Clarify RL-specific |

---

## 3. Feature Review

### Existing Features ✅

| Feature | Status | Implementation |
|---------|--------|----------------|
| Full Texas Hold'em game rules | ✅ Complete | poker_simulator.py |
| Blinds and dealer rotation | ✅ Complete | poker_simulator.py |
| All betting actions (fold/check/call/raise) | ✅ Complete | Multiple files |
| Side pot calculation | ✅ Complete | create_side_pots() |
| Hand evaluation (using treys) | ✅ Complete | agent.py |
| Equity calculation (Monte Carlo) | ✅ Complete | agent.py |
| Human player interface | ✅ Complete | human_player.py |
| AI opponent (Monte Carlo) | ✅ Complete | monte_carlo_agent.py |
| Random/CallStation agents | ✅ Complete | monte_carlo_agent.py |
| PPO model architecture | ✅ Complete | ppo_model.py |
| Gymnasium environment | ✅ Complete | poker_env.py |
| PPO training loop | ✅ Complete | train_rl_model.py |
| TensorBoard logging | ✅ Complete | train_rl_model.py |
| Model checkpointing | ✅ Complete | train_rl_model.py |
| Play vs trained model | ✅ Complete | play_vs_rl.py |
| Debug mode for training | ✅ Complete | train_rl_model.py |
| Test suite (119 tests) | ✅ Complete | tests/ |

### Missing Features ❌

| Feature | Priority | Impact | Description |
|---------|----------|--------|-------------|
| **Multi-GPU training** | Medium | Training speed | Currently CPU/single GPU only |
| **Self-play training** | High | Model quality | Train against past versions of itself |
| **Opponent modeling** | High | Strategy | Track opponent tendencies, adjust strategy |
| **Position-aware decisions** | Medium | Strategy | Consider position (button, blinds) |
| **Action history encoding** | High | Strategy | Include betting history in observations |
| **Multiple table support** | Low | Scalability | Parallel environment vectorization |
| **Hand history logging** | Medium | Analysis | Log games for post-analysis |
| **Web UI** | Low | Usability | Browser-based interface |
| **Tournament bracket** | Low | Feature | Multi-table tournament support |
| **Model comparison tool** | Medium | Evaluation | Compare different model checkpoints |
| **Hyperparameter search** | Medium | Training | Automated HP optimization |
| **Pre-trained models** | High | Usability | Ship with ready-to-use models |

### Partially Implemented Features 🚧

| Feature | Status | Missing Parts |
|---------|--------|---------------|
| **Data generation** | 🚧 Partial | `simulation/generate_dataset.py` referenced in README but not present |
| **Model evaluation** | 🚧 Partial | `training/evaluate_model.py` referenced in README but not present |
| **Supervised training** | 🚧 Partial | `training/train_model.py` referenced in README but not present |
| **Visualization** | 🚧 Partial | `tools/visualize.py` exists but is empty stub |
| **Configuration** | 🚧 Partial | `tools/config.py` exists but not integrated |
| **Tournament connection** | 🚧 Partial | `connect/player1.py` exists but undocumented |

---

## 4. Task Breakdown

### Priority 1: Critical Improvements

---

#### Task 1.1: Unify Agent Interfaces

**Why it's needed:** Two overlapping agent interfaces (`PokerAgent` and `PokerPlayer`) cause confusion and code duplication. A unified interface will improve maintainability.

**Difficulty:** Medium

**Files involved:**
- `agents/agent.py`
- `agents/poker_player.py`
- `agents/monte_carlo_agent.py`
- `simulation/poker_simulator.py`
- `simulation/poker_env.py`

**Steps:**
1. Create `agents/base_player.py` with unified abstract interface
2. Define common attributes: `id`, `money`, `folded`, `all_in`, `bet`, `hole_cards`
3. Define abstract method: `get_action(game_state) -> Action`
4. Make `PokerAgent` inherit from base class
5. Make `MonteCarloAgent` inherit from base class
6. Update `TexasHoldemSimulation` to use unified interface
7. Update `PokerEnv` to use unified interface
8. Add adapter tests for backward compatibility

**Dependencies:** None

---

#### Task 1.2: Add Action History to Observations

**Why it's needed:** The current observation lacks betting history, which is crucial for reading opponents. Without it, the RL agent can't learn bluffing or opponent exploitation.

**Difficulty:** Medium

**Files involved:**
- `simulation/poker_env.py`
- `training/ppo_model.py`

**Steps:**
1. Define action history format: last N actions per player as `(action_type, amount, position)`
2. Add `action_history` list to `PokerEnv` state
3. Update `_apply_action()` to append to history
4. Update `_get_observation()` to encode action history
5. Add action history embedding module to `PokerPPOModel`
6. Update observation dimension constants
7. Add tests for history encoding/decoding

**Dependencies:** None

---

#### Task 1.3: Implement Self-Play Training

**Why it's needed:** Training against fixed opponents leads to overfitting. Self-play (training against past versions) improves generalization and strategy diversity.

**Difficulty:** Hard

**Files involved:**
- `training/train_rl_model.py`
- `agents/monte_carlo_agent.py` (for opponent interface)

**Steps:**
1. Create `SelfPlayBuffer` class to store past model checkpoints
2. Implement opponent sampling strategy (e.g., last N checkpoints)
3. Create `RLOpponent` class that uses saved model for opponent actions
4. Add `--self-play` flag to training script
5. Update `_create_envs()` to use `RLOpponent` when self-play enabled
6. Add checkpoint rotation (keep last K models)
7. Add TensorBoard metrics for opponent diversity
8. Test training convergence with self-play

**Dependencies:** Task 1.2 (better observations help self-play)

---

### Priority 2: Important Enhancements

---

#### Task 2.1: Add Position-Aware Decision Making

**Why it's needed:** Position (button, blinds, early/late) significantly affects optimal strategy. Current agents ignore position entirely.

**Difficulty:** Easy

**Files involved:**
- `agents/agent.py`
- `agents/monte_carlo_agent.py`
- `simulation/poker_env.py`

**Steps:**
1. Add `relative_position` property (0-1 scale, 0=early, 1=button)
2. Update observation encoding to include position features
3. Modify `_preflop_decision()` to adjust thresholds by position
4. Add position-aware tests
5. Document position strategy in agent docstrings

**Dependencies:** None

---

#### Task 2.2: Create Model Comparison Tool

**Why it's needed:** No way to compare trained models against each other to measure improvement.

**Difficulty:** Easy

**Files involved:**
- `tools/compare_models.py` (new file)
- `play_vs_rl.py` (reference implementation)

**Steps:**
1. Create `compare_models.py` with CLI interface
2. Load two model checkpoints
3. Run N hands of model1 vs model2
4. Track win rate, average profit, action distribution
5. Output comparison report
6. Add to README documentation

**Dependencies:** None

---

#### Task 2.3: Implement Hand History Logging

**Why it's needed:** No way to review games for debugging or analysis. Hand histories enable post-game analysis and training data generation.

**Difficulty:** Medium

**Files involved:**
- `simulation/poker_simulator.py`
- `utils/hand_history.py` (new file)

**Steps:**
1. Create `HandHistory` dataclass with all game events
2. Add `hand_history` attribute to `TexasHoldemSimulation`
3. Log events: deal, action, showdown, pot award
4. Implement JSON serialization for history
5. Add `--log-hands` flag to main.py
6. Create simple hand history viewer
7. Add tests for serialization/deserialization

**Dependencies:** None

---

#### Task 2.4: Add Opponent Modeling

**Why it's needed:** Current agents don't track opponent behavior. Modeling opponents enables exploitation strategies.

**Difficulty:** Hard

**Files involved:**
- `agents/opponent_model.py` (new file)
- `agents/monte_carlo_agent.py`
- `simulation/poker_env.py`

**Steps:**
1. Create `OpponentModel` class with action frequency tracking
2. Track per-opponent: fold%, raise%, check%, aggression factor
3. Integrate opponent model into `MonteCarloAgent.get_action()`
4. Adjust thresholds based on opponent tendencies
5. Add opponent features to RL observation space
6. Test opponent model updates correctly
7. Document exploitation strategies

**Dependencies:** Task 2.3 (hand history helps track opponents)

---

### Priority 3: Code Quality

---

#### Task 3.1: Remove Empty Test Files

**Why it's needed:** Empty test files (`test_simulation.py`, `test_gameplay.py`) clutter the codebase.

**Difficulty:** Easy

**Files involved:**
- `tests/test_simulation.py`
- `tests/test_gameplay.py`

**Steps:**
1. Delete `tests/test_simulation.py`
2. Delete `tests/test_gameplay.py`
3. Verify all 119 tests still pass
4. Update `.gitignore` if needed

**Dependencies:** None

---

#### Task 3.2: Add Missing Documentation

**Why it's needed:** Poor documentation makes onboarding difficult. Key modules lack docstrings.

**Difficulty:** Easy

**Files involved:**
- `agents/__init__.py`
- `simulation/__init__.py`
- `training/__init__.py`
- `utils/__init__.py`
- `tools/__init__.py`

**Steps:**
1. Add module docstrings to all `__init__.py` files
2. Document exported classes/functions
3. Add usage examples in docstrings
4. Update README with detailed API reference
5. Add docstring coverage check to CI (optional)

**Dependencies:** None

---

#### Task 3.3: Extract Magic Numbers to Constants

**Why it's needed:** Magic numbers make code harder to understand and modify.

**Difficulty:** Easy

**Files involved:**
- `agents/agent.py`
- `agents/monte_carlo_agent.py`
- `simulation/poker_env.py`

**Steps:**
1. Create `config/constants.py` with all game constants
2. Define: `STRONG_HAND_THRESHOLD`, `MEDIUM_HAND_THRESHOLD`, etc.
3. Replace magic numbers in agent files
4. Replace magic numbers in env files
5. Add tests for constant values

**Dependencies:** None

---

#### Task 3.4: Consolidate Equity Calculation

**Why it's needed:** Duplicate equity calculation code in `agent.py` and `monte_carlo_agent.py`.

**Difficulty:** Medium

**Files involved:**
- `agents/agent.py`
- `agents/monte_carlo_agent.py`
- `utils/equity.py` (new file)

**Steps:**
1. Create `utils/equity.py` with shared `calculate_equity()` function
2. Implement single Monte Carlo simulation function
3. Update `PokerAgent` to use shared function
4. Update `MonteCarloAgent` to use shared function
5. Add comprehensive equity tests
6. Benchmark equity calculation performance

**Dependencies:** None

---

### Priority 4: Missing Files

---

#### Task 4.1: Implement generate_dataset.py

**Why it's needed:** README references this file but it doesn't exist. Needed for supervised learning data.

**Difficulty:** Medium

**Files involved:**
- `simulation/generate_dataset.py` (new file)

**Steps:**
1. Create script to simulate N hands
2. Record: hole cards, board, actions, outcomes
3. Save as CSV or JSON
4. Add CLI arguments: num_hands, output_path
5. Add progress bar for long generations
6. Document output format in `data/datasets.md`
7. Add unit tests

**Dependencies:** Task 2.3 (hand history format)

---

#### Task 4.2: Implement evaluate_model.py

**Why it's needed:** README references this file but it doesn't exist. Needed to evaluate trained models.

**Difficulty:** Medium

**Files involved:**
- `training/evaluate_model.py` (new file)

**Steps:**
1. Create script to load model checkpoint
2. Run N evaluation games against benchmark opponents
3. Calculate metrics: win rate, profit/loss, action distribution
4. Output evaluation report
5. Add CLI arguments: model_path, num_episodes, opponent_type
6. Add tests for evaluation logic

**Dependencies:** None

---

#### Task 4.3: Complete tools/visualize.py

**Why it's needed:** File exists but is incomplete. Visualization helps understand model behavior.

**Difficulty:** Medium

**Files involved:**
- `tools/visualize.py`

**Steps:**
1. Add function to plot training curves from TensorBoard logs
2. Add function to visualize action distributions
3. Add function to plot equity vs decision
4. Create hand visualization (ASCII art cards)
5. Add matplotlib/plotly dependencies
6. Document visualization functions
7. Add example usage in docstrings

**Dependencies:** None

---

### Priority 5: Advanced Features

---

#### Task 5.1: Add Vectorized Environments

**Why it's needed:** Single environment training is slow. Vectorized envs enable parallel rollouts.

**Difficulty:** Hard

**Files involved:**
- `simulation/poker_env.py`
- `training/train_rl_model.py`

**Steps:**
1. Make `PokerEnv` compatible with gymnasium's `AsyncVectorEnv`
2. Update `PPOTrainer._create_envs()` to use vectorized wrapper
3. Update `_collect_rollouts()` for batched stepping
4. Test with 4, 8, 16 parallel environments
5. Benchmark speedup
6. Add `--num-envs` CLI argument

**Dependencies:** None

---

#### Task 5.2: Implement Pre-flop Range Charts

**Why it's needed:** Pre-flop decisions currently use simple heuristics. Range charts provide GTO-optimal starting hand decisions.

**Difficulty:** Hard

**Files involved:**
- `agents/preflop_ranges.py` (new file)
- `agents/agent.py`
- `agents/monte_carlo_agent.py`

**Steps:**
1. Research pre-flop range charts for different positions
2. Encode charts as Python data structure
3. Create `get_preflop_action(hand, position, facing_bet)` function
4. Integrate into `_preflop_decision()`
5. Add position-dependent ranges
6. Test against known optimal plays
7. Document range chart sources

**Dependencies:** Task 2.1 (position awareness)

---

#### Task 5.3: Add Web UI

**Why it's needed:** Console interface is limited. Web UI enables better visualization and mobile access.

**Difficulty:** Hard

**Files involved:**
- `web/` (new directory)
- `web/app.py` (Flask/FastAPI backend)
- `web/static/` (frontend assets)

**Steps:**
1. Create Flask/FastAPI backend
2. Add REST API for game actions
3. Create simple HTML/CSS frontend
4. Implement WebSocket for real-time updates
5. Add card visualization (SVG or images)
6. Deploy instructions for local/cloud
7. Add authentication (optional)

**Dependencies:** Task 2.3 (game state API)

---

## Summary Table

| ID | Task | Priority | Difficulty | Dependencies |
|----|------|----------|------------|--------------|
| 1.1 | Unify Agent Interfaces | P1 | Medium | None |
| 1.2 | Add Action History to Observations | P1 | Medium | None |
| 1.3 | Implement Self-Play Training | P1 | Hard | 1.2 |
| 2.1 | Add Position-Aware Decision Making | P2 | Easy | None |
| 2.2 | Create Model Comparison Tool | P2 | Easy | None |
| 2.3 | Implement Hand History Logging | P2 | Medium | None |
| 2.4 | Add Opponent Modeling | P2 | Hard | 2.3 |
| 3.1 | Remove Empty Test Files | P3 | Easy | None |
| 3.2 | Add Missing Documentation | P3 | Easy | None |
| 3.3 | Extract Magic Numbers to Constants | P3 | Easy | None |
| 3.4 | Consolidate Equity Calculation | P3 | Medium | None |
| 4.1 | Implement generate_dataset.py | P4 | Medium | 2.3 |
| 4.2 | Implement evaluate_model.py | P4 | Medium | None |
| 4.3 | Complete tools/visualize.py | P4 | Medium | None |
| 5.1 | Add Vectorized Environments | P5 | Hard | None |
| 5.2 | Implement Pre-flop Range Charts | P5 | Hard | 2.1 |
| 5.3 | Add Web UI | P5 | Hard | 2.3 |

---

## Recommended Order of Implementation

1. **Quick Wins (1-2 days)**
   - Task 3.1: Remove Empty Test Files
   - Task 3.2: Add Missing Documentation
   - Task 3.3: Extract Magic Numbers

2. **Foundation (1 week)**
   - Task 1.1: Unify Agent Interfaces
   - Task 2.1: Add Position-Aware Decision Making
   - Task 2.3: Implement Hand History Logging

3. **Core Improvements (2 weeks)**
   - Task 1.2: Add Action History to Observations
   - Task 2.2: Create Model Comparison Tool
   - Task 4.2: Implement evaluate_model.py

4. **Advanced Training (2-3 weeks)**
   - Task 1.3: Implement Self-Play Training
   - Task 2.4: Add Opponent Modeling
   - Task 5.1: Add Vectorized Environments

5. **Polish (1 week)**
   - Task 4.1: Implement generate_dataset.py
   - Task 4.3: Complete tools/visualize.py
   - Task 5.2: Implement Pre-flop Range Charts

6. **Optional Extras**
   - Task 5.3: Add Web UI
