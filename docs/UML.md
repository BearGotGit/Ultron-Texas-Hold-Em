# Texas Hold'em System UML Diagrams

This document contains UML diagrams that illustrate the architecture and design of the Texas Hold'em poker simulation and AI system.

## Table of Contents

1. [Class Diagram Overview](#class-diagram-overview)
2. [Agents Module](#agents-module)
3. [Simulation Module](#simulation-module)
4. [Training Module](#training-module)
5. [Data Classes and Enums](#data-classes-and-enums)
6. [Component Diagram](#component-diagram)
7. [Sequence Diagrams](#sequence-diagrams)

---

## Class Diagram Overview

This high-level class diagram shows the main classes and their relationships across all modules.

```mermaid
classDiagram
    %% Abstract Base Class
    class PokerPlayer {
        <<abstract>>
        +id: str
        +money: int
        +folded: bool
        +all_in: bool
        +bet: int
        +total_invested: int
        -_private_cards: List~int~
        +get_public_info() PokerPlayerPublic
        +reset()
        +reset_money(starting_money: int)
        +reset_bet()
        +deal_hand(dealt_cards: Tuple)
        +get_hole_cards() List~int~
        +move_money(pot: Pot, amount: int) int
        +add_winnings(amount: int)
        +fold()
        +is_active() bool
        +get_action()* PokerAction
    }

    %% Concrete Agent Classes
    class PokerAgent {
        +name: str
        +hole_cards: List
        +evaluator: Evaluator
        +chips: int
        +current_bet: int
        +is_folded: bool
        +is_all_in: bool
        +receive_cards(cards)
        +reset_for_new_hand()
        +reset_current_bet()
        +get_hole_cards() List
        +get_chips() int
        +add_chips(amount)
        +place_bet(amount) int
        +fold()
        +make_decision(board, pot_size, current_bet_to_call, min_raise) Tuple
        +evaluate_hand(board) Tuple
        +calculate_equity(board, opponent_hands, remaining_deck_cards, num_simulations) float
    }

    class HumanPlayer {
        +make_decision(board, pot_size, current_bet_to_call, min_raise) Tuple
    }

    class MonteCarloAgent {
        +num_simulations: int
        +aggression: float
        +bluff_frequency: float
        +evaluator: Evaluator
        +get_action(hole_cards, board, pot, current_bet, min_raise, players, my_idx) PokerAction
        -_preflop_decision(hole_cards, to_call, min_raise, my_money, pot) PokerAction
        -_calculate_equity(hole_cards, board, num_opponents) float
        -_make_decision(equity, pot_odds, to_call, pot, min_raise, my_money, board_stage) PokerAction
    }

    class RandomAgent {
        +fold_prob: float
        +raise_prob: float
        +get_action(hole_cards, board, pot, current_bet, min_raise, players, my_idx) PokerAction
    }

    class CallStationAgent {
        +get_action(hole_cards, board, pot, current_bet, min_raise, players, my_idx) PokerAction
    }

    %% Simulation Classes
    class TexasHoldemSimulation {
        +agents: List~PokerAgent~
        +small_blind: int
        +big_blind: int
        +deck: Deck
        +evaluator: Evaluator
        +board: List
        +flop: List
        +turn: List
        +river: List
        +pot: int
        +pots: List
        +current_bet: int
        +min_raise: int
        +dealer_position: int
        +reset_for_new_hand()
        +deal_hole_cards()
        +post_blinds() Tuple
        +deal_flop() List
        +deal_turn() List
        +deal_river() List
        +run_betting_round(round_name) bool
        +calculate_all_equities() List
        +evaluate_hands() List
        +get_winner() List
        +create_side_pots() List
        +award_pot() List
        +get_pot_size() int
        +print_board()
        +print_game_state()
    }

    class PokerEnv {
        +config: PokerEnvConfig
        +players: List~PokerPlayer~
        +num_players: int
        +hero_idx: int
        +render_mode: str
        +deck: Deck
        +evaluator: Evaluator
        +pot: Pot
        +board: List~int~
        +round_stage: RoundStage
        +current_bet: int
        +min_raise: int
        +dealer_position: int
        +active_player_idx: int
        +game_over: bool
        +hand_complete: bool
        +reset(seed, options) Tuple
        +step(action) Tuple
        -_post_blinds()
        -_apply_action(player_idx, action) bool
        -_advance_to_hero_or_end()
        -_is_betting_round_complete() bool
        -_move_to_next_active_player()
        -_advance_round()
        -_end_hand()
        -_run_showdown()
        -_calculate_reward() float
        -_get_observation() ndarray
        -_get_info() Dict
        +render()
    }

    %% Training Classes
    class PokerPPOModel {
        +card_embed_dim: int
        +hidden_dim: int
        +card_embedding: CardEmbedding
        +hand_embedding: Sequential
        +numeric_embedding: Sequential
        +shared_trunk: Sequential
        +fold_head: Sequential
        +bet_head: Sequential
        +value_head: Sequential
        +forward(obs) Tuple
        +get_action_and_value(obs, deterministic) Tuple
        +evaluate_actions(obs, actions) Tuple
        +get_value(obs) Tensor
    }

    class CardEmbedding {
        +net: Sequential
        +forward(x) Tensor
    }

    class RLPokerPlayer {
        +id: str
        +model: PokerPPOModel
        +device: device
        +deterministic: bool
        +get_action_from_obs(obs) Tuple
    }

    %% Data Classes
    class PokerAction {
        <<dataclass>>
        +action_type: ActionType
        +amount: int
        +fold()$ PokerAction
        +check()$ PokerAction
        +call(amount)$ PokerAction
        +raise_to(amount)$ PokerAction
    }

    class PokerPlayerPublic {
        <<dataclass>>
        +id: str
        +money: int
        +folded: bool
        +all_in: bool
        +bet: int
        +to_dict() dict
    }

    class PokerGameState {
        <<dataclass>>
        +players: List~PokerPlayerPublic~
        +pot: int
        +round: RoundStage
        +board: List~int~
        +current_bet: int
        +min_raise: int
        +dealer_position: int
        +active_player_idx: int
        +game_over: bool
        +invalid: bool
        +to_dict() dict
    }

    class Pot {
        <<dataclass>>
        +money: int
        +add(amount)
    }

    class PokerEnvConfig {
        <<dataclass>>
        +big_blind: int
        +small_blind: int
        +starting_stack: int
        +max_players: int
    }

    class ActionType {
        <<enumeration>>
        FOLD
        CHECK
        CALL
        RAISE
    }

    %% Inheritance Relationships
    PokerPlayer <|-- MonteCarloAgent
    PokerPlayer <|-- RandomAgent
    PokerPlayer <|-- CallStationAgent
    PokerAgent <|-- HumanPlayer

    %% Composition & Association Relationships
    TexasHoldemSimulation o-- PokerAgent : contains
    PokerEnv o-- PokerPlayer : contains
    PokerEnv *-- Pot : has
    PokerEnv --> PokerEnvConfig : uses
    PokerEnv --> PokerGameState : creates
    PokerPlayer --> PokerPlayerPublic : creates
    PokerPlayer --> PokerAction : uses
    MonteCarloAgent --> PokerAction : returns
    RandomAgent --> PokerAction : returns
    CallStationAgent --> PokerAction : returns
    PokerAction --> ActionType : has
    PokerGameState o-- PokerPlayerPublic : contains
    PokerPPOModel *-- CardEmbedding : has
    RLPokerPlayer --> PokerPPOModel : uses
```

---

## Agents Module

Detailed view of the agents package showing the class hierarchy for different player types.

```mermaid
classDiagram
    class PokerPlayer {
        <<abstract>>
        +id: str
        +money: int
        +folded: bool
        +all_in: bool
        +bet: int
        +total_invested: int
        -_private_cards: List~int~
        +get_public_info() PokerPlayerPublic
        +reset()
        +reset_money(starting_money)
        +reset_bet()
        +deal_hand(dealt_cards)
        +get_hole_cards() List~int~
        +move_money(pot, amount) int
        +add_winnings(amount)
        +fold()
        +is_active() bool
        +get_action()* PokerAction
    }

    class PokerAgent {
        +name: str
        +hole_cards: List
        +evaluator: Evaluator
        +chips: int
        +current_bet: int
        +total_invested: int
        +is_folded: bool
        +is_all_in: bool
        +receive_cards(cards)
        +reset_for_new_hand()
        +reset_current_bet()
        +get_hole_cards() List
        +get_chips() int
        +add_chips(amount)
        +place_bet(amount) int
        +fold()
        +make_decision(board, pot_size, current_bet_to_call, min_raise) Tuple
        -_preflop_decision(current_bet_to_call, min_raise) Tuple
        +evaluate_hand(board) Tuple
        +calculate_equity(...) float
        -_calculate_all_equities(board, hands, remaining_deck_cards, num_simulations) List
    }

    class HumanPlayer {
        +make_decision(board, pot_size, current_bet_to_call, min_raise) Tuple
    }

    class MonteCarloAgent {
        +num_simulations: int
        +aggression: float
        +bluff_frequency: float
        +evaluator: Evaluator
        +get_action(...) PokerAction
        -_preflop_decision(...) PokerAction
        -_calculate_equity(hole_cards, board, num_opponents) float
        -_make_decision(...) PokerAction
    }

    class RandomAgent {
        +fold_prob: float
        +raise_prob: float
        +get_action(...) PokerAction
    }

    class CallStationAgent {
        +get_action(...) PokerAction
    }

    PokerPlayer <|-- MonteCarloAgent : extends
    PokerPlayer <|-- RandomAgent : extends
    PokerPlayer <|-- CallStationAgent : extends
    PokerAgent <|-- HumanPlayer : extends

    note for PokerPlayer "Abstract base class for RL training environment.\nDefines standard interface for all poker players."
    note for PokerAgent "Original agent class used in simulation.\nUses Evaluator from treys library."
    note for MonteCarloAgent "Uses Monte Carlo simulation for\nequity calculation and decision making."
```

---

## Simulation Module

Classes responsible for game simulation and the RL training environment.

```mermaid
classDiagram
    class TexasHoldemSimulation {
        +agents: List~PokerAgent~
        +small_blind: int
        +big_blind: int
        +deck: Deck
        +evaluator: Evaluator
        +board: List
        +flop: List
        +turn: List
        +river: List
        +pot: int
        +pots: List
        +current_bet: int
        +min_raise: int
        +dealer_position: int
        +reset_for_new_hand()
        +deal_hole_cards()
        +post_blinds() Tuple
        +deal_flop() List
        +deal_turn() List
        +deal_river() List
        +run_betting_round(round_name) bool
        +calculate_all_equities() List
        +evaluate_hands() List
        +get_winner() List
        +create_side_pots() List
        +award_pot() List
        +get_pot_size() int
        +print_board()
        +print_game_state()
    }

    class PokerEnv {
        <<Gymnasium Environment>>
        +config: PokerEnvConfig
        +players: List~PokerPlayer~
        +num_players: int
        +hero_idx: int
        +render_mode: str
        +deck: Deck
        +evaluator: Evaluator
        +pot: Pot
        +board: List~int~
        +round_stage: RoundStage
        +current_bet: int
        +min_raise: int
        +dealer_position: int
        +active_player_idx: int
        +game_over: bool
        +hand_complete: bool
        +last_aggressor_idx: Optional~int~
        +players_acted_this_round: set
        +hero_hand_start_chips: int
        +observation_space: Box
        +action_space: Box
        +reset(seed, options) Tuple
        +step(action) Tuple
        +render()
        -_post_blinds()
        -_apply_action(player_idx, action) bool
        -_advance_to_hero_or_end()
        -_is_betting_round_complete() bool
        -_move_to_next_active_player()
        -_advance_round()
        -_end_hand()
        -_run_showdown()
        -_calculate_reward() float
        -_get_observation() ndarray
        -_get_info() Dict
    }

    class Deck {
        <<treys>>
        +cards: List
        +draw(n) List
    }

    class Evaluator {
        <<treys>>
        +evaluate(board, hand) int
        +get_rank_class(score) int
        +class_to_string(hand_class) str
        +get_five_card_rank_percentage(score) float
    }

    TexasHoldemSimulation *-- Deck : uses
    TexasHoldemSimulation *-- Evaluator : uses
    PokerEnv *-- Deck : uses
    PokerEnv *-- Evaluator : uses

    note for TexasHoldemSimulation "Main game simulation for interactive play.\nManages dealing, betting rounds, and pot distribution."
    note for PokerEnv "Gymnasium-compatible RL environment.\nObservation: card encodings + game state.\nAction: [p_fold, bet_scalar]"
```

---

## Training Module

Neural network architecture for PPO-based reinforcement learning.

```mermaid
classDiagram
    class PokerPPOModel {
        <<nn.Module>>
        +card_embed_dim: int
        +hidden_dim: int
        +card_embedding: CardEmbedding
        +hand_embedding: Sequential
        +numeric_embedding: Sequential
        +card_features: int
        +hand_embed_features: int
        +numeric_embed_features: int
        +combined_features: int
        +shared_trunk: Sequential
        +fold_head: Sequential
        +bet_head: Sequential
        +value_head: Sequential
        -_init_weights()
        -_embed_cards(obs) Tensor
        -_extract_hand_features(obs) Tensor
        -_extract_numeric_features(obs) Tensor
        +forward(obs) Tuple~Tensor~
        +get_action_and_value(obs, deterministic) Tuple
        +evaluate_actions(obs, actions) Tuple
        +get_value(obs) Tensor
    }

    class CardEmbedding {
        <<nn.Module>>
        +net: Sequential
        +forward(x) Tensor
    }

    class RLPokerPlayer {
        +id: str
        +model: PokerPPOModel
        +device: device
        +deterministic: bool
        +get_action_from_obs(obs) Tuple
    }

    PokerPPOModel *-- CardEmbedding : contains
    RLPokerPlayer o-- PokerPPOModel : uses

    note for PokerPPOModel "Actor-Critic PPO Model\n\nArchitecture:\n- Card embedding: 53-dim → 64 → 64\n- Hand embedding: 10 → 32 → 32\n- Numeric embedding: 42 → 64 → 64\n- Combined: 544 → 256 → 256\n- Three heads: fold, bet, value"
    note for CardEmbedding "Embeds 53-dim one-hot card vector\ninto hidden_dim representation"
```

---

## Data Classes and Enums

Supporting data structures used throughout the system.

```mermaid
classDiagram
    class ActionType {
        <<enumeration>>
        FOLD
        CHECK
        CALL
        RAISE
    }

    class RoundStage {
        <<type alias>>
        pre-flop
        flop
        turn
        river
        showdown
    }

    class PokerAction {
        <<dataclass>>
        +action_type: ActionType
        +amount: int
        +fold()$ PokerAction
        +check()$ PokerAction
        +call(amount: int)$ PokerAction
        +raise_to(amount: int)$ PokerAction
    }

    class PokerPlayerPublic {
        <<dataclass>>
        +id: str
        +money: int
        +folded: bool
        +all_in: bool
        +bet: int
        +to_dict() dict
    }

    class PokerGameState {
        <<dataclass>>
        +players: List~PokerPlayerPublic~
        +pot: int
        +round: RoundStage
        +board: List~int~
        +current_bet: int
        +min_raise: int
        +dealer_position: int
        +active_player_idx: int
        +game_over: bool
        +invalid: bool
        +to_dict() dict
    }

    class Pot {
        <<dataclass>>
        +money: int
        +add(amount: int)
    }

    class PokerEnvConfig {
        <<dataclass>>
        +big_blind: int = 10
        +small_blind: int = 5
        +starting_stack: int = 1000
        +max_players: int = 9
    }

    PokerAction --> ActionType : has
    PokerGameState o-- PokerPlayerPublic : contains
    PokerGameState --> RoundStage : uses

    note for PokerAction "Represents a player action.\nFactory methods for creating specific actions."
    note for PokerPlayerPublic "Public information visible to all players.\nUsed for opponent modeling."
    note for PokerGameState "Complete game state snapshot.\nUsed for RL observations."
```

---

## Component Diagram

High-level view of system modules and their dependencies.

```mermaid
flowchart TB
    subgraph Main["Main Entry Point"]
        main[main.py]
    end

    subgraph Agents["agents/"]
        PokerAgent[PokerAgent]
        HumanPlayer[HumanPlayer]
        PokerPlayer[PokerPlayer ABC]
        MonteCarloAgent[MonteCarloAgent]
        RandomAgent[RandomAgent]
        CallStationAgent[CallStationAgent]
        DataClasses[Data Classes]
    end

    subgraph Simulation["simulation/"]
        TexasHoldemSimulation[TexasHoldemSimulation]
        PokerEnv[PokerEnv]
        CardUtils[card_utils.py]
    end

    subgraph Training["training/"]
        PPOModel[PokerPPOModel]
        TrainRL[train_rl_model.py]
        Evaluate[evaluate_model.py]
    end

    subgraph Utils["utils/"]
        Device[device.py]
    end

    subgraph External["External Libraries"]
        Treys[treys]
        Gymnasium[gymnasium]
        PyTorch[torch]
    end

    main --> PokerAgent
    main --> HumanPlayer
    main --> TexasHoldemSimulation

    HumanPlayer --> PokerAgent
    MonteCarloAgent --> PokerPlayer
    RandomAgent --> PokerPlayer
    CallStationAgent --> PokerPlayer

    TexasHoldemSimulation --> PokerAgent
    PokerEnv --> PokerPlayer
    PokerEnv --> DataClasses

    TrainRL --> PPOModel
    TrainRL --> PokerEnv
    TrainRL --> MonteCarloAgent
    Evaluate --> PPOModel

    PokerAgent --> Treys
    TexasHoldemSimulation --> Treys
    PokerEnv --> Treys
    PokerEnv --> Gymnasium
    PPOModel --> PyTorch
    TrainRL --> Device
```

---

## Sequence Diagrams

### Game Hand Sequence

Shows the flow of a complete poker hand in the simulation.

```mermaid
sequenceDiagram
    participant Main
    participant Game as TexasHoldemSimulation
    participant Deck
    participant Agent1 as PokerAgent 1
    participant Agent2 as PokerAgent 2
    participant Evaluator

    Main->>Game: reset_for_new_hand()
    Game->>Agent1: reset_for_new_hand()
    Game->>Agent2: reset_for_new_hand()
    Game->>Deck: new Deck()

    Main->>Game: deal_hole_cards()
    Game->>Deck: draw(2)
    Game->>Agent1: receive_cards(cards)
    Game->>Deck: draw(2)
    Game->>Agent2: receive_cards(cards)

    Main->>Game: post_blinds()
    Game->>Agent1: place_bet(small_blind)
    Game->>Agent2: place_bet(big_blind)

    Main->>Game: run_betting_round("Pre-flop")
    loop Each active player
        Game->>Agent1: make_decision(board, pot, bet, min_raise)
        Agent1-->>Game: (action, amount)
        Game->>Game: process_action()
    end

    Main->>Game: deal_flop()
    Game->>Deck: draw(3)
    Game-->>Main: flop cards

    Main->>Game: run_betting_round("Flop")
    Note over Game: Repeat betting round logic

    Main->>Game: deal_turn()
    Main->>Game: run_betting_round("Turn")
    Main->>Game: deal_river()
    Main->>Game: run_betting_round("River")

    Main->>Game: evaluate_hands()
    Game->>Evaluator: evaluate(board, hand)
    Evaluator-->>Game: score

    Main->>Game: award_pot()
    Game->>Agent1: add_chips(winnings)
```

### RL Training Step Sequence

Shows how the PPO training loop interacts with the environment.

```mermaid
sequenceDiagram
    participant Trainer as train_rl_model.py
    participant Model as PokerPPOModel
    participant Env as PokerEnv
    participant Hero as Hero (RL Agent)
    participant Opponent as MonteCarloAgent

    Trainer->>Env: reset()
    Env->>Hero: deal_hand()
    Env->>Opponent: deal_hand()
    Env->>Env: post_blinds()
    Env-->>Trainer: observation, info

    loop Episode steps
        Trainer->>Model: get_action_and_value(obs)
        Model-->>Trainer: action, log_prob, entropy, value

        Trainer->>Env: step(action)
        Note over Env: Apply hero action

        loop Opponent turns
            Env->>Opponent: get_action(hole_cards, board, ...)
            Opponent-->>Env: PokerAction
            Env->>Env: apply_action()
        end

        Env-->>Trainer: obs, reward, terminated, truncated, info

        alt terminated or truncated
            Trainer->>Trainer: store_transition()
            Trainer->>Trainer: compute_advantages()
            Trainer->>Model: evaluate_actions(obs_batch, actions_batch)
            Trainer->>Trainer: compute_ppo_loss()
            Trainer->>Model: update_weights()
        end
    end
```

---

## Observation Space Structure

Visual representation of the RL observation tensor structure.

```mermaid
flowchart LR
    subgraph Obs["Observation Vector (423 dims)"]
        subgraph Cards["Card Encodings (371 dims)"]
            HC1["Hole Card 1<br/>53-dim one-hot"]
            HC2["Hole Card 2<br/>53-dim one-hot"]
            B1["Board 1<br/>53-dim one-hot"]
            B2["Board 2<br/>53-dim one-hot"]
            B3["Board 3<br/>53-dim one-hot"]
            B4["Board 4<br/>53-dim one-hot"]
            B5["Board 5<br/>53-dim one-hot"]
        end

        subgraph Hand["Hand Features (10 dims)"]
            HF["10 binary flags<br/>(hand type indicators)"]
        end

        subgraph Players["Player Features (36 dims = MAX_PLAYERS × 4)"]
            P1["Player 1: money, bet, folded, all_in"]
            P2["Player 2: money, bet, folded, all_in"]
            PN["... (MAX_PLAYERS=9, 4 features each)"]
        end

        subgraph Global["Global Features (6 dims)"]
            GF["pot, current_bet, min_raise,<br/>round_stage, my_position, dealer_position"]
        end
    end

    Cards --> Hand --> Players --> Global
```

---

## Action Space Structure

Visualization of how actions are interpreted.

```mermaid
flowchart TB
    subgraph Action["Action Vector [p_fold, bet_scalar]"]
        PF["p_fold ∈ [0, 1]<br/>Bernoulli probability"]
        BS["bet_scalar ∈ [0, 1]<br/>Beta distribution sample"]
    end

    PF -->|"≥ 0.5"| Fold["FOLD (or CHECK if to_call=0)"]
    PF -->|"< 0.5"| Continue["Continue to bet interpretation"]

    Continue --> BSCheck{bet_scalar value}
    BSCheck -->|"< ε"| CheckCall["CHECK / CALL"]
    BSCheck -->|"> 1-ε"| AllIn["ALL-IN"]
    BSCheck -->|"ε to 1-ε"| ScaledRaise["Scaled RAISE<br/>(min_raise to all-in)"]
```

---

## Notes

### Key Design Patterns

1. **Template Method Pattern**: `PokerPlayer` defines the abstract `get_action()` method that concrete agents implement
2. **Factory Pattern**: `PokerAction` uses class methods as factories for creating specific action types
3. **Observer Pattern (implicit)**: The environment observes and tracks player states
4. **Strategy Pattern**: Different agent implementations provide different decision-making strategies

### External Dependencies

- **treys**: Card evaluation library (Deck, Card, Evaluator)
- **gymnasium**: RL environment interface
- **torch**: Neural network implementation
- **numpy**: Numerical operations

### File Organization

```
Ultron-Texas-Hold-Em/
├── agents/
│   ├── __init__.py         # Exports PokerAgent
│   ├── agent.py            # PokerAgent class
│   ├── poker_player.py     # ABC, data classes, Pot
│   ├── human_player.py     # HumanPlayer
│   ├── monte_carlo_agent.py # MonteCarloAgent, RandomAgent, CallStationAgent
│   ├── game_state.py       # (empty)
│   └── opponent_models.py  # (empty)
├── simulation/
│   ├── __init__.py         # Exports TexasHoldemSimulation
│   ├── poker_simulator.py  # TexasHoldemSimulation
│   ├── poker_env.py        # PokerEnv (Gymnasium)
│   ├── card_utils.py       # (empty)
│   └── generate_dataset.py # (empty)
├── training/
│   ├── __init__.py
│   ├── ppo_model.py        # PokerPPOModel, CardEmbedding, RLPokerPlayer
│   ├── train_rl_model.py   # Training loop
│   ├── evaluate_model.py
│   └── losses.py
├── utils/
│   ├── __init__.py
│   └── device.py           # PyTorch device selection
├── tools/
│   ├── __init__.py
│   ├── config.py           # (empty)
│   ├── utils.py            # (empty)
│   └── visualize.py
├── main.py                 # Entry point
└── README.md
```
