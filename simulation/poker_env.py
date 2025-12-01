"""
PokerEnv: Gymnasium-compatible Texas Hold'em environment for RL training.

Follows OpenAI Gymnasium interface:
    observation, reward, terminated, truncated, info = env.step(action)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from treys import Card, Deck, Evaluator

from agents.poker_player import (
    PokerPlayer,
    PokerPlayerPublic,
    PokerAction,
    ActionType,
    PokerGameState,
    Pot,
    RoundStage,
)


# ============================================================
# Constants
# ============================================================

# Card encoding: 53 dimensions (0 = no card, 1-52 = Treys card IDs)
CARD_ENCODING_DIM = 53
NUM_CARD_SLOTS = 7  # 2 hole cards + 5 board cards
CARD_OBS_DIM = NUM_CARD_SLOTS * CARD_ENCODING_DIM  # 371

# Hand feature flags (10 binary features)
NUM_HAND_FEATURES = 10

# Hand rankings from Treys (lower = better)
HAND_CLASS_ROYAL_FLUSH = 1
HAND_CLASS_STRAIGHT_FLUSH = 2
HAND_CLASS_FOUR_KIND = 3
HAND_CLASS_FULL_HOUSE = 4
HAND_CLASS_FLUSH = 5
HAND_CLASS_STRAIGHT = 6
HAND_CLASS_TRIPS = 7
HAND_CLASS_TWO_PAIR = 8
HAND_CLASS_PAIR = 9
HAND_CLASS_HIGH_CARD = 10

# Max players supported
MAX_PLAYERS = 9

# Numeric state features per player: money, bet, folded, all_in
FEATURES_PER_PLAYER = 4

# Global numeric features: pot, current_bet, min_raise, round_stage, my_position, dealer_position
GLOBAL_NUMERIC_FEATURES = 6


@dataclass
class PokerEnvConfig:
    """Configuration for poker environment."""
    big_blind: int = 10
    small_blind: int = 5
    starting_stack: int = 1000
    max_players: int = MAX_PLAYERS


# ============================================================
# Observation Encoding Helpers
# ============================================================

def encode_card_one_hot(card: Optional[int]) -> np.ndarray:
    """
    Encode a single card as 53-dim one-hot vector.
    
    Args:
        card: Treys card integer or None for empty slot
        
    Returns:
        53-dim numpy array
    """
    vec = np.zeros(CARD_ENCODING_DIM, dtype=np.float32)
    if card is None or card == 0:
        vec[0] = 1.0  # "no card" flag
    else:
        # Map Treys card to index 1-52
        # Treys cards are integers; we'll use a deterministic mapping
        idx = card_to_index(card)
        vec[idx] = 1.0
    return vec


def card_to_index(card: int) -> int:
    """
    Convert Treys card integer to index 1-52.
    
    Treys uses a prime-based encoding. We extract rank and suit.
    """
    # Extract rank (0-12) and suit (0-3) from Treys card
    rank = Card.get_rank_int(card)  # 0-12 (2 to A)
    suit = Card.get_suit_int(card)  # 1, 2, 4, 8 for s, h, d, c
    
    # Map suit to 0-3
    suit_map = {1: 0, 2: 1, 4: 2, 8: 3}
    suit_idx = suit_map.get(suit, 0)
    
    # Index: 1 + rank * 4 + suit_idx (gives 1-52)
    return 1 + rank * 4 + suit_idx


def index_to_card_str(idx: int) -> str:
    """Convert index 1-52 back to card string (for debugging)."""
    if idx == 0:
        return "None"
    idx -= 1
    rank = idx // 4
    suit = idx % 4
    rank_chars = "23456789TJQKA"
    suit_chars = "shdc"
    return rank_chars[rank] + suit_chars[suit]


def encode_hand_features(hole_cards: List[int], board: List[int]) -> np.ndarray:
    """
    Compute 10 binary hand-feature flags using Treys.
    
    Args:
        hole_cards: List of 2 Treys card integers
        board: List of 0-5 Treys card integers
        
    Returns:
        10-dim numpy array of binary flags
    """
    features = np.zeros(NUM_HAND_FEATURES, dtype=np.float32)
    
    # Need at least 3 board cards to evaluate
    if len(hole_cards) < 2 or len(board) < 3:
        return features
    
    evaluator = Evaluator()
    try:
        score = evaluator.evaluate(board[:5], hole_cards[:2])
        hand_class = evaluator.get_rank_class(score)
        
        # Set the appropriate flag (only one should be 1)
        # Classes: 1=royal flush, 2=straight flush, ..., 10=high card
        if hand_class <= 10:
            features[hand_class - 1] = 1.0
    except:
        pass  # If evaluation fails, return zeros
    
    return features


def encode_round_stage(round_stage: RoundStage) -> float:
    """Encode round stage as a float 0-4."""
    mapping = {
        "pre-flop": 0.0,
        "flop": 1.0,
        "turn": 2.0,
        "river": 3.0,
        "showdown": 4.0,
    }
    return mapping.get(round_stage, 0.0)


# ============================================================
# Action Interpretation
# ============================================================

# Epsilon thresholds for action interpretation
ACTION_EPSILON = 0.1


def interpret_action(
    p_fold: float,
    bet_scalar: float,
    current_bet: int,
    my_bet: int,
    min_raise: int,
    my_money: int,
) -> PokerAction:
    """
    Interpret model outputs into a PokerAction.
    
    Args:
        p_fold: Probability of folding (0-1)
        bet_scalar: Betting scalar (0-1)
        current_bet: Current bet to match
        my_bet: My current bet this round
        min_raise: Minimum raise amount
        my_money: My remaining chips
        
    Returns:
        PokerAction
    """
    # Fold decision (Bernoulli)
    if p_fold > 0.5:
        return PokerAction.fold()
    
    # Calculate amount to call
    to_call = current_bet - my_bet
    
    # Interpret bet_scalar
    if bet_scalar < ACTION_EPSILON:
        # Check or call
        if to_call <= 0:
            return PokerAction.check()
        else:
            call_amount = min(to_call, my_money)
            return PokerAction.call(call_amount)
    
    elif bet_scalar > 1.0 - ACTION_EPSILON:
        # All-in
        return PokerAction.raise_to(my_money)
    
    else:
        # Scaled raise
        # Map bet_scalar from (ε, 1-ε) to (min_raise, my_money)
        normalized = (bet_scalar - ACTION_EPSILON) / (1.0 - 2 * ACTION_EPSILON)
        max_raise = my_money
        raise_amount = int(min_raise + normalized * (max_raise - min_raise))
        raise_amount = max(min_raise, min(raise_amount, my_money))
        return PokerAction.raise_to(raise_amount)


# ============================================================
# PokerEnv Gymnasium Environment
# ============================================================

class PokerEnv(gym.Env):
    """
    Gymnasium-compatible Texas Hold'em environment.
    
    Observation Space:
        - 7 x 53 card one-hot encodings (hole cards + board)
        - 10 binary hand features
        - Numeric state: pot, bets, stacks, positions, round stage
        
    Action Space:
        Box(2,): [p_fold, bet_scalar] both in [0, 1]
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(
        self,
        players: List[PokerPlayer],
        config: Optional[PokerEnvConfig] = None,
        hero_idx: int = 0,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize poker environment.
        
        Args:
            players: List of PokerPlayer instances
            config: Environment configuration
            hero_idx: Index of the RL agent being trained
            render_mode: Rendering mode
        """
        super().__init__()
        
        self.config = config or PokerEnvConfig()
        self.players = players
        self.num_players = len(players)
        self.hero_idx = hero_idx
        self.render_mode = render_mode
        
        # Treys components
        self.deck = Deck()
        self.evaluator = Evaluator()
        
        # Game state
        self.pot = Pot()
        self.board: List[int] = []
        self.round_stage: RoundStage = "pre-flop"
        self.current_bet = 0
        self.min_raise = self.config.big_blind
        self.dealer_position = 0
        self.active_player_idx = 0
        self.game_over = False
        self.hand_complete = False
        
        # Betting round tracking
        self.last_aggressor_idx: Optional[int] = None
        self.players_acted_this_round: set = set()
        
        # Track hero's chips at hand start for reward calculation
        self.hero_hand_start_chips: int = 0
        
        # Calculate observation size
        self.card_obs_dim = NUM_CARD_SLOTS * CARD_ENCODING_DIM
        self.hand_features_dim = NUM_HAND_FEATURES
        self.player_features_dim = MAX_PLAYERS * FEATURES_PER_PLAYER
        self.global_features_dim = GLOBAL_NUMERIC_FEATURES
        
        self.obs_dim = (
            self.card_obs_dim +
            self.hand_features_dim +
            self.player_features_dim +
            self.global_features_dim
        )
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        
        # Action space: [p_fold, bet_scalar]
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment for a new hand.
        
        Returns:
            observation, info
        """
        super().reset(seed=seed)
        
        # Reset deck
        self.deck = Deck()
        
        # Reset players (including money for episodic reset)
        for player in self.players:
            player.reset()
            player.reset_money(self.config.starting_stack)
        
        # Reset game state
        self.pot = Pot()
        self.board = []
        self.round_stage = "pre-flop"
        self.current_bet = 0
        self.min_raise = self.config.big_blind
        self.game_over = False
        self.hand_complete = False
        self.last_aggressor_idx = None
        self.players_acted_this_round = set()
        
        # Deal hole cards
        for player in self.players:
            cards = self.deck.draw(2)
            player.deal_hand(tuple(cards))
        
        # Post blinds
        self._post_blinds()
        
        # Track hero's chips at hand start (after blinds posted) for reward calculation
        self.hero_hand_start_chips = self.players[self.hero_idx].money
        
        # Set starting player (left of big blind for pre-flop)
        self.active_player_idx = (self.dealer_position + 3) % self.num_players
        
        # Skip to hero's turn or advance game
        self._advance_to_hero_or_end()
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: [p_fold, bet_scalar] array
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.game_over or self.hand_complete:
            # Game already over
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        hero = self.players[self.hero_idx]
        
        # Check if it's hero's turn
        if self.active_player_idx != self.hero_idx:
            # Shouldn't happen if advance logic is correct
            return self._get_observation(), 0.0, False, True, self._get_info()
        
        # Interpret action
        p_fold = float(action[0])
        bet_scalar = float(action[1])
        
        poker_action = interpret_action(
            p_fold=p_fold,
            bet_scalar=bet_scalar,
            current_bet=self.current_bet,
            my_bet=hero.bet,
            min_raise=self.min_raise,
            my_money=hero.money,
        )
        
        # Apply action
        truncated = not self._apply_action(self.hero_idx, poker_action)
        
        # Advance game state
        self._advance_to_hero_or_end()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination
        terminated = self.game_over or self.hand_complete
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _post_blinds(self):
        """Post small and big blinds."""
        sb_idx = (self.dealer_position + 1) % self.num_players
        bb_idx = (self.dealer_position + 2) % self.num_players
        
        sb_player = self.players[sb_idx]
        bb_player = self.players[bb_idx]
        
        # Post blinds
        sb_player.move_money(self.pot, self.config.small_blind)
        bb_player.move_money(self.pot, self.config.big_blind)
        
        self.current_bet = self.config.big_blind
        self.last_aggressor_idx = bb_idx
    
    def _apply_action(self, player_idx: int, action: PokerAction) -> bool:
        """
        Apply a player's action.
        
        Returns:
            True if action was valid, False otherwise
        """
        player = self.players[player_idx]
        
        if player.folded or player.all_in:
            return False
        
        to_call = self.current_bet - player.bet
        
        if action.action_type == ActionType.FOLD:
            player.fold()
            
        elif action.action_type == ActionType.CHECK:
            if to_call > 0:
                # Invalid: can't check when there's a bet
                player.fold()  # Force fold
                return False
            
        elif action.action_type == ActionType.CALL:
            actual = player.move_money(self.pot, to_call)
            
        elif action.action_type == ActionType.RAISE:
            # First call, then raise
            total_to_put = to_call + action.amount
            actual = player.move_money(self.pot, total_to_put)
            
            if player.bet > self.current_bet:
                raise_amount = player.bet - self.current_bet
                self.current_bet = player.bet
                self.min_raise = max(self.min_raise, raise_amount)
                self.last_aggressor_idx = player_idx
                # Reset action tracking for raises
                self.players_acted_this_round = {player_idx}
        
        self.players_acted_this_round.add(player_idx)
        return True
    
    def _advance_to_hero_or_end(self):
        """
        Advance game state, having opponents act until it's hero's turn or hand ends.
        """
        max_iterations = 100  # Safety limit
        
        for _ in range(max_iterations):
            # Check if hand is over
            active_players = [p for p in self.players if not p.folded]
            if len(active_players) <= 1:
                self._end_hand()
                return
            
            # Check if betting round is complete
            if self._is_betting_round_complete():
                self._advance_round()
                if self.hand_complete:
                    return
                continue
            
            # Find next player to act
            self._move_to_next_active_player()
            
            # If it's hero's turn, stop
            if self.active_player_idx == self.hero_idx:
                return
            
            # Opponent's turn - get their action
            opponent = self.players[self.active_player_idx]
            if opponent.is_active():
                action = opponent.get_action(
                    hole_cards=opponent.get_hole_cards(),
                    board=self.board,
                    pot=self.pot.money,
                    current_bet=self.current_bet,
                    min_raise=self.min_raise,
                    players=[p.get_public_info() for p in self.players],
                    my_idx=self.active_player_idx,
                )
                self._apply_action(self.active_player_idx, action)
        
        # Safety: if we hit max iterations, end the hand
        self._end_hand()
    
    def _is_betting_round_complete(self) -> bool:
        """Check if the current betting round is complete."""
        active_players = [
            (i, p) for i, p in enumerate(self.players)
            if not p.folded and not p.all_in
        ]
        
        if len(active_players) == 0:
            return True
        
        if len(active_players) == 1:
            # Only one active player - check if they've matched
            idx, player = active_players[0]
            if player.bet >= self.current_bet:
                return True
            # They need to act
            return False
        
        # Check if all active players have acted and matched the bet
        all_matched = all(p.bet >= self.current_bet for _, p in active_players)
        all_acted = all(i in self.players_acted_this_round for i, _ in active_players)
        
        return all_matched and all_acted
    
    def _move_to_next_active_player(self):
        """Move to the next player who can act."""
        for _ in range(self.num_players):
            self.active_player_idx = (self.active_player_idx + 1) % self.num_players
            player = self.players[self.active_player_idx]
            if player.is_active():
                return
        # No active players found
    
    def _advance_round(self):
        """Advance to the next betting round."""
        # Reset bets
        for player in self.players:
            player.reset_bet()
        self.current_bet = 0
        self.players_acted_this_round = set()
        self.last_aggressor_idx = None
        
        # Deal community cards and advance stage
        if self.round_stage == "pre-flop":
            self.board = self.deck.draw(3)
            self.round_stage = "flop"
        elif self.round_stage == "flop":
            self.board.extend(self.deck.draw(1))
            self.round_stage = "turn"
        elif self.round_stage == "turn":
            self.board.extend(self.deck.draw(1))
            self.round_stage = "river"
        elif self.round_stage == "river":
            self.round_stage = "showdown"
            self._end_hand()
            return
        
        # Set starting player (left of dealer for post-flop)
        self.active_player_idx = self.dealer_position
        self._move_to_next_active_player()
    
    def _end_hand(self):
        """End the current hand and award pot."""
        self.hand_complete = True
        self.game_over = True
        self.round_stage = "showdown"
        
        # Find winners
        active_players = [(i, p) for i, p in enumerate(self.players) if not p.folded]
        
        if len(active_players) == 1:
            # Everyone folded to one player
            winner = active_players[0][1]
            winner.add_winnings(self.pot.money)
        else:
            # Showdown - evaluate hands
            self._run_showdown()
    
    def _run_showdown(self):
        """Evaluate hands and award pot to winner(s)."""
        active_players = [(i, p) for i, p in enumerate(self.players) if not p.folded]
        
        if len(self.board) < 3:
            # Can't evaluate without enough board cards
            # Split pot equally
            split = self.pot.money // len(active_players)
            for _, p in active_players:
                p.add_winnings(split)
            return
        
        # Evaluate each hand
        scores = []
        for idx, player in active_players:
            try:
                score = self.evaluator.evaluate(self.board, player.get_hole_cards())
                scores.append((idx, player, score))
            except:
                scores.append((idx, player, float('inf')))
        
        # Find best score (lowest in Treys)
        best_score = min(s[2] for s in scores)
        winners = [s[1] for s in scores if s[2] == best_score]
        
        # Split pot among winners
        split = self.pot.money // len(winners)
        for winner in winners:
            winner.add_winnings(split)
    
    def _calculate_reward(self) -> float:
        """Calculate reward for the hero."""
        if not self.hand_complete:
            return 0.0
        
        hero = self.players[self.hero_idx]
        
        # Reward = chips won/lost this hand, normalized by starting_stack
        # Using starting_stack (constant) instead of hand-start chips (variable)
        # ensures consistent reward scaling across hands, making rewards
        # comparable regardless of hero's current chip count.
        initial_chips = self.hero_hand_start_chips
        starting_stack = self.config.starting_stack
        
        if starting_stack <= 0:
            # Edge case: invalid starting stack
            return 0.0
        
        # Calculate chip delta from hand start
        chip_delta = hero.money - initial_chips
        
        # Normalize by starting_stack for consistent scaling
        reward = chip_delta / starting_stack
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Build the observation tensor."""
        hero = self.players[self.hero_idx]
        hole_cards = hero.get_hole_cards()
        
        obs_parts = []
        
        # 1. Card encodings (7 x 53)
        # Hole cards
        for i in range(2):
            card = hole_cards[i] if i < len(hole_cards) else None
            obs_parts.append(encode_card_one_hot(card))
        
        # Board cards (5 slots)
        for i in range(5):
            card = self.board[i] if i < len(self.board) else None
            obs_parts.append(encode_card_one_hot(card))
        
        # 2. Hand features (10 binary flags)
        hand_features = encode_hand_features(hole_cards, self.board)
        obs_parts.append(hand_features)
        
        # 3. Player features (MAX_PLAYERS x 4)
        player_features = np.zeros(self.player_features_dim, dtype=np.float32)
        for i, player in enumerate(self.players):
            if i >= MAX_PLAYERS:
                break
            base = i * FEATURES_PER_PLAYER
            # Normalize money: log(money+1) / log(starting_stack+1)
            stack_normalizer = np.log1p(self.config.starting_stack)
            player_features[base] = np.log1p(player.money) / stack_normalizer
            # Normalize bet: log(bet+1) / log(big_blind+1)
            bb_normalizer = np.log1p(self.config.big_blind) if self.config.big_blind > 0 else 1.0
            player_features[base + 1] = np.log1p(player.bet) / bb_normalizer
            # Binary flags (no normalization needed)
            player_features[base + 2] = float(player.folded)
            player_features[base + 3] = float(player.all_in)
        obs_parts.append(player_features)
        
        # 4. Global features
        # Normalizers
        total_starting_money = self.config.starting_stack * self.num_players
        stack_normalizer = np.log1p(total_starting_money)
        bb_normalizer = np.log1p(self.config.big_blind) if self.config.big_blind > 0 else 1.0
        num_players_normalizer = np.log1p(self.num_players)
        
        global_features = np.array([
            # Pot: log(pot+1) / log(total_starting_money+1)
            np.log1p(self.pot.money) / stack_normalizer,
            # Current bet: log(bet+1) / log(big_blind+1)
            np.log1p(self.current_bet) / bb_normalizer,
            # Min raise: log(raise+1) / log(big_blind+1)
            np.log1p(self.min_raise) / bb_normalizer,
            # Round stage: normalize to [0, 1]
            encode_round_stage(self.round_stage) / 4.0,
            # Hero position: normalize to [0, 1]
            self.hero_idx / max(self.num_players - 1, 1),
            # Dealer position: normalize to [0, 1]
            self.dealer_position / max(self.num_players - 1, 1),
        ], dtype=np.float32)
        obs_parts.append(global_features)
        
        return np.concatenate(obs_parts)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info dictionary."""
        return {
            "round": self.round_stage,
            "pot": self.pot.money,
            "hero_money": self.players[self.hero_idx].money,
            "active_players": sum(1 for p in self.players if not p.folded),
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "ansi":
            return self._render_ansi()
    
    def _render_human(self):
        """Print game state to console."""
        print(f"\n{'='*50}")
        print(f"Round: {self.round_stage} | Pot: ${self.pot.money}")
        print(f"Board: {[Card.int_to_pretty_str(c) for c in self.board]}")
        print(f"{'='*50}")
        
        for i, player in enumerate(self.players):
            marker = " *" if i == self.active_player_idx else ""
            hero_marker = " [HERO]" if i == self.hero_idx else ""
            status = "FOLD" if player.folded else "ALL-IN" if player.all_in else ""
            print(f"{player.id}: ${player.money} (bet: ${player.bet}) {status}{marker}{hero_marker}")
    
    def _render_ansi(self) -> str:
        """Return ANSI string representation."""
        lines = [
            f"Round: {self.round_stage} | Pot: ${self.pot.money}",
            f"Board: {[Card.int_to_pretty_str(c) for c in self.board]}",
        ]
        for i, player in enumerate(self.players):
            marker = " *" if i == self.active_player_idx else ""
            lines.append(f"{player.id}: ${player.money}{marker}")
        return "\n".join(lines)
