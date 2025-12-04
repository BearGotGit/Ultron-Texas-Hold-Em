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
    # Penalty coefficient applied to normalized max raise fraction per hand.
    # Reward penalty = all_in_penalty_alpha * max_raise_fraction
    # Set to 0.0 to disable. Small positive values (e.g., 0.05) discourage reckless all-ins.
    all_in_penalty_alpha: float = 0.05
    # Optional potential-based shaping using equity estimates.
    # Set `use_equity_shaping=True` to enable. `equity_mc_samples` controls
    # how many Monte-Carlo samples to estimate hand equity (performance cost).
    # `equity_shaping_coef` scales the shaping reward added per step.
    use_equity_shaping: bool = False
    equity_mc_samples: int = 50
    equity_shaping_coef: float = 1.0
    # Discount used inside potential-based shaping (should match trainer gamma if possible)
    equity_shaping_gamma: float = 1.0
    # Simple per-step chip-delta shaping (very low-cost). If True, each step
    # returns normalized chip delta in addition to terminal reward (can be noisy).
    use_step_chip_delta: bool = False


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

import random

# Tunable thresholds for action interpretation.
#
# The goal is to get a *reasonable* baseline mix when the network
# outputs are roughly uniform:
#   - FOLD: 15–30%
#   - CHECK: 25–40%
#   - CALL: 20–35%
#   - RAISE: 10–25%
#
# You can tweak these four knobs to shift the mix:
#
#  - FOLD_THRESHOLD: how big p_fold must be to actually fold (when facing a bet)
#  - CHECK_THRESHOLD: when there is *no* bet to call, how often we check vs raise
#  - RAISE_THRESHOLD: when facing a bet, how large bet_scalar must be to raise
#  - POT_FRACTION: how big raises are relative to remaining chips
#
# Start here; if you log action counts and see too many folds/raises,
# adjust thresholds slightly.

FOLD_THRESHOLD = 0.75     #↑ require higher fold-prob to fold (reduce illegal fold->check conversions)
CHECK_THRESHOLD = 0.65    # ↓ more raises when checked to (25–35% raises)
RAISE_THRESHOLD = 0.60    # ↓ more raises when facing a bet (15–25%)
POT_FRACTION = 0.15       # small raises (non-all-in) so model can learn


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
        p_fold:     model output in [0, 1] (we treat it as a generic scalar)
        bet_scalar: model output in [0, 1] used for sizing / aggressiveness
        current_bet: current bet everyone must match
        my_bet:     my current bet in this round
        min_raise:  minimum legal raise amount
        my_money:   my remaining chips

    Returns:
        PokerAction (fold / check / call / raise_to)
    """
    # Amount to call
    to_call = max(0, current_bet - my_bet)

    # ----------------------------------------------------------------
    # CASE 1: Nothing to call (checked to us / we are BB facing no raise)
    # ----------------------------------------------------------------
    if to_call <= 0:
        # Folding is never correct here; map folds to check.
        # We use CHECK_THRESHOLD to bias toward check vs raise.
        if bet_scalar < CHECK_THRESHOLD:
            # CHECK about CHECK_THRESHOLD of the time
            return PokerAction.check()

        # Remaining ~ (1 - CHECK_THRESHOLD) of the time we RAISE
        # Compute a sensible raise size:
        #   - base on a fraction of our stack
        #   - at least min_raise
        #   - never more than our stack (all-in cap)
        max_additional = max(0, my_money)
        if max_additional <= 0:
            return PokerAction.check()

        # bet_scalar ∈ [CHECK_THRESHOLD, 1]; normalize to [0, 1]
        norm = (bet_scalar - CHECK_THRESHOLD) / max(1e-6, (1.0 - CHECK_THRESHOLD))
        target_additional = int(norm * (POT_FRACTION * max_additional))
        raise_amount = max(min_raise, min(max_additional, target_additional))

        if raise_amount <= 0:
            return PokerAction.check()

        return PokerAction.raise_to(raise_amount)

    # ----------------------------------------------------------------
    # CASE 2: There is a bet to call (to_call > 0)
    # ----------------------------------------------------------------

    # Use p_fold to handle fold vs “continue”.
    # If the model is confident on folding and there *is* something to call,
    # allow fold. FOLD_THRESHOLD ~ 0.6–0.7 tends to give 15–30% folds.
    if p_fold > FOLD_THRESHOLD:
        return PokerAction.fold()

    # We’re not folding. Decide between CALL and RAISE using bet_scalar.
    #
    #   bet_scalar < RAISE_THRESHOLD  -> CALL
    #   bet_scalar >= RAISE_THRESHOLD -> RAISE
    #
    # With uniform bet_scalar, that gives:
    #   P(call | not fold) ≈ RAISE_THRESHOLD
    #   P(raise | not fold) ≈ 1 - RAISE_THRESHOLD
    #
    # Example: FOLD_THRESHOLD=0.65, RAISE_THRESHOLD=0.70
    #   P(fold)  ≈ 0.35
    #   P(call)  ≈ 0.65 * 0.70 ≈ 0.455
    #   P(raise) ≈ 0.65 * 0.30 ≈ 0.195
    #
    # After training, the network will skew these, but this gives a
    # *healthy* starting prior rather than “raise 95%”.

    if bet_scalar < RAISE_THRESHOLD:
        # CALL
        call_amount = min(to_call, my_money)
        if call_amount <= 0:
            # If somehow busted / no chips, just check (env will handle).
            return PokerAction.check()
        return PokerAction.call(call_amount)

    # RAISE path
    max_additional = max(0, my_money - to_call)
    if max_additional <= 0:
        # Can't really raise; fall back to call
        call_amount = min(to_call, my_money)
        if call_amount <= 0:
            return PokerAction.check()
        return PokerAction.call(call_amount)

    # Normalize bet_scalar within [RAISE_THRESHOLD, 1] to [0, 1]
    norm = (bet_scalar - RAISE_THRESHOLD) / max(1e-6, (1.0 - RAISE_THRESHOLD))

    # Target additional amount scales with remaining stack and POT_FRACTION
    target_additional = int(norm * (POT_FRACTION * max_additional))

    raise_amount = max(min_raise, min(max_additional, target_additional))

    # If after clamping the raise is effectively 0, just call.
    if raise_amount <= 0:
        call_amount = min(to_call, my_money)
        if call_amount <= 0:
            return PokerAction.check()
        return PokerAction.call(call_amount)

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

        # Track the maximum raise fraction applied by the hero during the hand
        # (raise amount divided by starting stack). Used for reward shaping.
        self.current_hand_max_raise_fraction = 0.0

        # Track per-step hero money for optional per-step shaping
        self._last_hero_money = self.players[self.hero_idx].money

        # Compute initial potential (equity) if shaping enabled
        if getattr(self.config, 'use_equity_shaping', False):
            try:
                self._last_potential = self._estimate_equity(mc_samples=self.config.equity_mc_samples)
            except Exception:
                self._last_potential = 0.0
        else:
            self._last_potential = 0.0
        
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
        
        # Optional: compute potential (equity) before action for shaping
        phi_before = 0.0
        if getattr(self.config, 'use_equity_shaping', False):
            try:
                phi_before = self._estimate_equity(mc_samples=self.config.equity_mc_samples)
            except Exception:
                phi_before = 0.0

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
        
        # Calculate reward (terminal component)
        reward = self._calculate_reward()

        # Per-step chip delta shaping
        if getattr(self.config, 'use_step_chip_delta', False):
            hero_money = self.players[self.hero_idx].money
            chip_delta = (hero_money - getattr(self, '_last_hero_money', hero_money)) / max(1.0, float(self.config.starting_stack))
            reward += float(chip_delta)
            self._last_hero_money = hero_money

        # Equity-based potential shaping: add gamma * Phi(s') - Phi(s)
        if getattr(self.config, 'use_equity_shaping', False):
            try:
                phi_after = self._estimate_equity(mc_samples=self.config.equity_mc_samples)
            except Exception:
                phi_after = 0.0

            shaping = (self.config.equity_shaping_gamma * float(phi_after) - float(phi_before))
            reward += float(self.config.equity_shaping_coef) * shaping
            # update last potential
            self._last_potential = phi_after
        
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
                # Track raise fraction for the hero (or any player) relative to starting stack
                try:
                    raise_frac = float(raise_amount) / float(self.config.starting_stack)
                except Exception:
                    raise_frac = 0.0
                if raise_frac > getattr(self, 'current_hand_max_raise_fraction', 0.0):
                    self.current_hand_max_raise_fraction = raise_frac
        
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
        
        # Reward = chips won/lost this hand
        # Compare final chips to chips at hand start (after blinds posted)
        initial_chips = self.hero_hand_start_chips
        if initial_chips <= 0:
            # Edge case: hero had no chips at hand start
            return 0.0
        reward = (hero.money - initial_chips) / initial_chips  # Normalized

        # Apply small penalty proportional to the maximum raise fraction observed
        # during the hand to discourage reckless all-ins. The penalty coefficient
        # is configured in `PokerEnvConfig.all_in_penalty_alpha` and can be 0.0
        # to disable this behavior.
        alpha = getattr(self.config, 'all_in_penalty_alpha', 0.0)
        max_raise_frac = getattr(self, 'current_hand_max_raise_fraction', 0.0)
        penalty = float(alpha) * float(max_raise_frac)

        if penalty != 0.0:
            reward = reward - penalty

        return reward


    def _estimate_equity(self, mc_samples: int = 50) -> float:
        """
        Estimate hero's win equity (probability of winning at showdown)
        against the current active opponents by Monte-Carlo sampling.

        Returns a float in [0, 1]. Ties count as fractional wins.
        """
        # If hero has no hole cards or board is invalid, return 0.5 fallback
        hero = self.players[self.hero_idx]
        hero_hole = hero.get_hole_cards()
        if not hero_hole or len(hero_hole) < 2:
            return 0.5

        # Opponent count: active opponents (not folded) excluding hero
        opponents = [p for i, p in enumerate(self.players) if i != self.hero_idx and not p.folded]
        num_opponents = max(0, len(opponents))

        # If no opponents (already won), equity is 1.0
        if num_opponents == 0:
            return 1.0

        # Known cards: hero hole + current board
        known = set()
        for c in hero_hole:
            known.add(c)
        for c in self.board:
            known.add(c)

        # Build pool of remaining cards
        full_deck = Deck().cards.copy()
        remaining = [c for c in full_deck if c not in known]
        if len(remaining) < 2:
            return 0.5

        wins = 0.0
        ties = 0.0
        samples = max(1, int(mc_samples))

        for _ in range(samples):
            # sample without replacement enough cards for rest of board + opponents
            needed_board = max(0, 5 - len(self.board))
            need = needed_board + 2 * num_opponents
            if need > len(remaining):
                # not enough cards to simulate; fallback
                break
            draw = random.sample(remaining, need)
            idx = 0

            # construct full board
            sim_board = list(self.board)
            if needed_board > 0:
                sim_board.extend(draw[idx: idx + needed_board])
                idx += needed_board

            # opponent hands
            opp_hands = []
            for _ in range(num_opponents):
                opp_hands.append([draw[idx], draw[idx + 1]])
                idx += 2

            # Evaluate hero vs opponents
            try:
                hero_score = self.evaluator.evaluate(sim_board, hero_hole)
            except Exception:
                hero_score = float('inf')

            opp_scores = []
            for h in opp_hands:
                try:
                    s = self.evaluator.evaluate(sim_board, h)
                except Exception:
                    s = float('inf')
                opp_scores.append(s)

            best_opp = min(opp_scores) if opp_scores else float('inf')
            if hero_score < best_opp:
                wins += 1.0
            elif hero_score == best_opp:
                # tie: fraction = 1 / (num winners)
                # count ties as fractional win among tied parties
                winners = 1 + sum(1 for s in opp_scores if s == hero_score)
                ties += 1.0 / float(winners)
            # else loss

        total = float(samples)
        if total <= 0:
            return 0.5

        equity = (wins + ties) / total
        return float(equity)
    
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
