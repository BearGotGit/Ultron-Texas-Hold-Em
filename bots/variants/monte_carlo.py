"""
Monte Carlo Agent implementing PokerPlayer interface.

This agent uses Monte Carlo simulation to estimate hand equity
and makes decisions based on pot odds and equity thresholds.
Used for bootstrapping RL training (initial opponent pool).
"""

from typing import List, Tuple
from itertools import combinations
import random

from treys import Card, Evaluator, Deck

from bots.player_abc import PokerPlayer
from bots.interact import ActType, Obs
from game.game_state import PokerPlayerPublic



# ============================================================
# Pre-flop Decision Constants
# ============================================================

# When pot odds are favorable (< 0.33), reduce call threshold by this factor
POT_ODDS_THRESHOLD_REDUCTION = 0.8

# Aggression contributes to calling chance with this weight
AGGRESSION_CALL_FACTOR = 0.2


class MonteCarloBot(PokerPlayer):
    """
    Monte Carlo-based poker agent.
    
    Uses equity calculations to make betting decisions.
    Implements the PokerPlayer interface for use in PokerEnv.
    """
    
    def __init__(
        self,
        player_id: str = "MonteCarloBot",
        starting_money: int = 1000,
        num_simulations: int = 500,
        aggression: float = 0.5,
        bluff_frequency: float = 0.1,
    ):
        """
        Initialize Monte Carlo agent.
        
        Args:
            player_id: Unique identifier for this player
            starting_money: Starting chip stack
            num_simulations: Number of Monte Carlo simulations for equity
            aggression: Aggression factor (0-1), higher = more raises
            bluff_frequency: How often to bluff with weak hands (0-1)
        """
        super().__init__(player_id, starting_money)
        self.num_simulations = num_simulations
        self.aggression = aggression
        self.bluff_frequency = bluff_frequency
        self.evaluator = Evaluator()
    
    def __call__(
        self,
        obs: Obs
    ) -> Tuple[ActType, int]:
        """
        Decide on an action using Monte Carlo equity estimation.
        
        Args:
            obs: Observation containing game state
            
        Returns:
            Tuple of (action_type, amount)
        """
        hole_cards = self._private_cards
        board = obs.public_cards
        pot = obs.pot
        to_call = obs.to_call
        
        # Pre-flop: use simple hand strength heuristics
        if len(board) == 0:
            return self._preflop_decision(hole_cards, to_call, pot)
        
        # Calculate equity via Monte Carlo
        # Estimate active opponents from bets (simplified)
        active_opponents = max(1, len([b for b in obs.players_bets if b > 0]))
        
        if active_opponents == 0:
            # Everyone folded - just check/call
            if to_call <= 0:
                return ("CHECK", 0)
            return ("CALL", 0)
        
        equity = self._calculate_equity(hole_cards, board, active_opponents)
        
        # Calculate pot odds
        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0
        
        return self._make_decision(
            equity=equity,
            pot_odds=pot_odds,
            to_call=to_call,
            pot=pot,
            board_stage=len(board),
        )
    
    def _preflop_decision(
        self,
        hole_cards: List[int],
        to_call: int,
        pot: int = 0,
    ) -> Tuple[ActType, int]:
        """
        Pre-flop decision based on hand categories and pot odds.
        
        Uses hand strength heuristics combined with pot odds to make
        more realistic calling decisions that don't fold too often.
        """
        if len(hole_cards) < 2:
            if to_call <= 0:
                return ("CHECK", 0)
            return ("FOLD", 0)
        
        # Get card ranks (0-12, where 12 = Ace)
        rank1 = Card.get_rank_int(hole_cards[0])
        rank2 = Card.get_rank_int(hole_cards[1])
        
        # Check if suited
        suit1 = Card.get_suit_int(hole_cards[0])
        suit2 = Card.get_suit_int(hole_cards[1])
        is_suited = suit1 == suit2
        
        # Check if pair
        is_pair = rank1 == rank2
        
        # Hand strength score (rough heuristic)
        high_rank = max(rank1, rank2)
        low_rank = min(rank1, rank2)
        gap = high_rank - low_rank
        
        hand_strength = 0.0
        
        # Premium pairs
        if is_pair:
            if rank1 >= 10:  # TT+
                hand_strength = 0.9
            elif rank1 >= 7:  # 77-99
                hand_strength = 0.7
            else:  # 22-66
                hand_strength = 0.5
        
        # High cards
        elif high_rank >= 11:  # Q+ high card
            if low_rank >= 10:  # Both broadway
                hand_strength = 0.75 + (0.1 if is_suited else 0)
            elif low_rank >= 8:  # One broadway, one medium
                hand_strength = 0.55 + (0.1 if is_suited else 0)
            else:
                hand_strength = 0.35 + (0.1 if is_suited else 0)
        
        # Suited connectors
        elif is_suited and gap <= 1:
            hand_strength = 0.45 + 0.02 * high_rank
        
        # Connected cards
        elif gap <= 1:
            hand_strength = 0.35 + 0.02 * high_rank
        
        # Suited cards
        elif is_suited:
            hand_strength = 0.3 + 0.01 * high_rank
        
        # Junk
        else:
            hand_strength = 0.15 + 0.01 * high_rank
        
        # Decision based on hand strength
        call_threshold = 0.25
        raise_threshold = 0.6
        
        # Adjust for bet size relative to pot - use reduced scaling
        # to prevent folding too quickly to large bets
        if pot > 0:
            bet_fraction = to_call / pot
            call_threshold += bet_fraction * 0.1
            raise_threshold += bet_fraction * 0.1
        
        # Cap the thresholds to prevent them from getting too high
        call_threshold = min(call_threshold, 0.5)
        raise_threshold = min(raise_threshold, 0.75)
        
        # Consider pot odds - if pot is offering good odds, lower call threshold
        if pot > 0 and to_call > 0:
            pot_odds = to_call / (pot + to_call)
            # If pot odds are favorable (< 0.33 means 2:1 or better), be more willing to call
            if pot_odds < 0.33:
                call_threshold *= POT_ODDS_THRESHOLD_REDUCTION
        
        if to_call <= 0:
            # No bet to call
            if hand_strength >= raise_threshold and random.random() < self.aggression:
                # Default raise amount (simplified without min_raise)
                raise_amount = int(pot * 0.5) if pot > 0 else 10
                return ("RAISE", raise_amount)
            return ("CHECK", 0)
        
        if hand_strength >= raise_threshold:
            # Strong hand - raise
            raise_amount = int(to_call + pot * 0.3) if pot > 0 else to_call + 10
            if random.random() < self.aggression:
                return ("RAISE", raise_amount)
            return ("CALL", 0)
        
        elif hand_strength >= call_threshold:
            # Medium hand - call
            return ("CALL", 0)
        
        else:
            # Weak hand - consider calling sometimes based on aggression and bluff frequency
            # Combined chance to call = bluff_frequency + (aggression * AGGRESSION_CALL_FACTOR)
            # This makes aggressive players more likely to defend with weak hands
            call_chance = self.bluff_frequency + (self.aggression * AGGRESSION_CALL_FACTOR)
            if random.random() < call_chance:
                return ("CALL", 0)
            return ("FOLD", 0)
    
    def _calculate_equity(
        self,
        hole_cards: List[int],
        board: List[int],
        num_opponents: int,
    ) -> float:
        """
        Calculate equity using Monte Carlo simulation.
        
        Args:
            hole_cards: Our hole cards
            board: Community cards
            num_opponents: Number of active opponents
            
        Returns:
            Estimated equity (0.0 to 1.0)
        """
        if len(hole_cards) < 2 or len(board) < 3:
            return 0.5  # Default equity
        
        # Get remaining deck cards
        used_cards = set(hole_cards) | set(board)
        full_deck = Deck().cards
        remaining = [c for c in full_deck if c not in used_cards]
        
        cards_to_deal = 5 - len(board)
        wins = 0
        ties = 0
        total = 0
        
        for _ in range(self.num_simulations):
            # Sample opponent hands and remaining board
            sampled = random.sample(remaining, 2 * num_opponents + cards_to_deal)
            
            # Assign opponent hands
            opponent_hands = [
                sampled[i*2:(i+1)*2] for i in range(num_opponents)
            ]
            
            # Complete the board
            full_board = board + sampled[2*num_opponents:]
            
            # Evaluate all hands
            my_score = self.evaluator.evaluate(full_board, hole_cards)
            opponent_scores = [
                self.evaluator.evaluate(full_board, hand)
                for hand in opponent_hands
            ]
            
            best_opponent = min(opponent_scores) if opponent_scores else float('inf')
            
            if my_score < best_opponent:
                wins += 1
            elif my_score == best_opponent:
                ties += 0.5
            
            total += 1
        
        return (wins + ties) / total if total > 0 else 0.5
    
    def _make_decision(
        self,
        equity: float,
        pot_odds: float,
        to_call: int,
        pot: int,
        board_stage: int,
    ) -> Tuple[ActType, int]:
        """
        Make decision based on equity and pot odds.
        
        Args:
            equity: Estimated win probability
            pot_odds: Pot odds for calling
            to_call: Amount to call
            pot: Current pot
            board_stage: Number of community cards (3, 4, or 5)
            
        Returns:
            Tuple of (action_type, amount)
        """
        # Adjust thresholds based on street
        stage_factor = 1.0 + (board_stage - 3) * 0.1  # Tighter on later streets
        
        # Strong hand threshold
        raise_threshold = 0.6 * stage_factor
        call_threshold = max(pot_odds, 0.25 * stage_factor)
        
        # Bluff consideration
        should_bluff = random.random() < self.bluff_frequency
        
        if to_call <= 0:
            # No bet to call
            if equity >= raise_threshold or (should_bluff and random.random() < 0.3):
                if random.random() < self.aggression:
                    raise_amount = int(pot * 0.5) if pot > 0 else 10
                    return ("RAISE", raise_amount)
            return ("CHECK", 0)
        
        # Check if call is profitable (equity > pot odds)
        is_profitable = equity > pot_odds
        
        if equity >= raise_threshold:
            # Strong hand - consider raising
            if random.random() < self.aggression:
                # Size the raise based on equity
                extra = int((equity - 0.5) * pot)
                raise_amount = to_call + max(10, extra)
                return ("RAISE", raise_amount)
            return ("CALL", 0)
        
        elif is_profitable or equity >= call_threshold:
            # Profitable or decent hand - call
            return ("CALL", 0)
        
        else:
            # Unprofitable - fold (unless bluffing)
            if should_bluff:
                return ("CALL", 0)
            return ("FOLD", 0)
    
    def __repr__(self) -> str:
        return f"MonteCarloAgent(sims={self.num_simulations}, agg={self.aggression}, bluff={self.bluff_frequency})"
