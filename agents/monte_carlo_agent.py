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

from agents.poker_player import (
    PokerPlayer,
    PokerPlayerPublic,
    PokerAction,
    ActionType,
)


class MonteCarloAgent(PokerPlayer):
    """
    Monte Carlo-based poker agent.
    
    Uses equity calculations to make betting decisions.
    Implements the PokerPlayer interface for use in PokerEnv.
    """
    
    def __init__(
        self,
        player_id: str,
        starting_money: int = 1000,
        num_simulations: int = 500,
        aggression: float = 0.5,
        bluff_frequency: float = 0.1,
    ):
        """
        Initialize Monte Carlo agent.
        
        Args:
            player_id: Unique identifier
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
    
    def get_action(
        self,
        hole_cards: List[int],
        board: List[int],
        pot: int,
        current_bet: int,
        min_raise: int,
        players: List[PokerPlayerPublic],
        my_idx: int,
    ) -> PokerAction:
        """
        Decide on an action using Monte Carlo equity estimation.
        
        Args:
            hole_cards: This player's hole cards (Treys integers)
            board: Community cards (Treys integers)
            pot: Current pot size
            current_bet: Current bet to match
            min_raise: Minimum raise amount
            players: Public info for all players
            my_idx: This player's index in players list
            
        Returns:
            PokerAction describing the chosen action
        """
        my_info = players[my_idx]
        to_call = current_bet - my_info.bet
        
        # Can't act if folded or all-in
        if my_info.folded or my_info.all_in:
            return PokerAction.check()
        
        # Pre-flop: use simple hand strength heuristics
        if len(board) == 0:
            return self._preflop_decision(hole_cards, to_call, min_raise, my_info.money, pot)
        
        # Calculate equity via Monte Carlo
        active_opponents = sum(
            1 for i, p in enumerate(players)
            if i != my_idx and not p.folded
        )
        
        if active_opponents == 0:
            # Everyone folded - just check/call
            if to_call <= 0:
                return PokerAction.check()
            return PokerAction.call(min(to_call, my_info.money))
        
        equity = self._calculate_equity(hole_cards, board, active_opponents)
        
        # Calculate pot odds
        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0
        
        return self._make_decision(
            equity=equity,
            pot_odds=pot_odds,
            to_call=to_call,
            pot=pot,
            min_raise=min_raise,
            my_money=my_info.money,
            board_stage=len(board),
        )
    
    def _preflop_decision(
        self,
        hole_cards: List[int],
        to_call: int,
        min_raise: int,
        my_money: int,
        pot: int = 0,
    ) -> PokerAction:
        """
        Pre-flop decision based on hand categories and pot odds.
        
        Uses hand strength heuristics combined with pot odds to make
        more realistic calling decisions that don't fold too often.
        """
        if len(hole_cards) < 2:
            if to_call <= 0:
                return PokerAction.check()
            return PokerAction.fold()
        
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
        
        # Adjust for bet size - use reduced scaling (0.15 instead of 0.3)
        # to prevent folding too quickly to large bets
        bet_fraction = to_call / my_money if my_money > 0 else 1.0
        call_threshold += bet_fraction * 0.15
        raise_threshold += bet_fraction * 0.15
        
        # Cap the thresholds to prevent them from getting too high
        call_threshold = min(call_threshold, 0.5)
        raise_threshold = min(raise_threshold, 0.75)
        
        # Consider pot odds - if pot is offering good odds, lower call threshold
        if pot > 0 and to_call > 0:
            pot_odds = to_call / (pot + to_call)
            # If pot odds are favorable (< 0.33 means 2:1 or better), be more willing to call
            if pot_odds < 0.33:
                call_threshold *= 0.8  # 20% reduction in threshold
        
        if to_call <= 0:
            # No bet to call
            if hand_strength >= raise_threshold and random.random() < self.aggression:
                raise_amount = min(min_raise, my_money)
                return PokerAction.raise_to(raise_amount)
            return PokerAction.check()
        
        if hand_strength >= raise_threshold:
            # Strong hand - raise
            raise_amount = min(to_call + min_raise, my_money)
            if random.random() < self.aggression:
                return PokerAction.raise_to(raise_amount)
            return PokerAction.call(min(to_call, my_money))
        
        elif hand_strength >= call_threshold:
            # Medium hand - call
            return PokerAction.call(min(to_call, my_money))
        
        else:
            # Weak hand - consider calling sometimes based on aggression and bluff frequency
            # Combined chance to call = bluff_frequency + (aggression * 0.2)
            # This makes aggressive players more likely to defend with weak hands
            call_chance = self.bluff_frequency + (self.aggression * 0.2)
            if random.random() < call_chance:
                return PokerAction.call(min(to_call, my_money))
            return PokerAction.fold()
    
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
        min_raise: int,
        my_money: int,
        board_stage: int,
    ) -> PokerAction:
        """
        Make decision based on equity and pot odds.
        
        Args:
            equity: Estimated win probability
            pot_odds: Pot odds for calling
            to_call: Amount to call
            pot: Current pot
            min_raise: Minimum raise amount
            my_money: Available chips
            board_stage: Number of community cards (3, 4, or 5)
            
        Returns:
            PokerAction
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
                    raise_amount = min(min_raise, my_money)
                    return PokerAction.raise_to(raise_amount)
            return PokerAction.check()
        
        # Check if call is profitable (equity > pot odds)
        is_profitable = equity > pot_odds
        
        if equity >= raise_threshold:
            # Strong hand - consider raising
            if random.random() < self.aggression:
                # Size the raise based on equity
                extra = int((equity - 0.5) * pot)
                raise_amount = min(to_call + max(min_raise, extra), my_money)
                return PokerAction.raise_to(raise_amount)
            return PokerAction.call(min(to_call, my_money))
        
        elif is_profitable or equity >= call_threshold:
            # Profitable or decent hand - call
            return PokerAction.call(min(to_call, my_money))
        
        else:
            # Unprofitable - fold (unless bluffing)
            if should_bluff:
                return PokerAction.call(min(to_call, my_money))
            return PokerAction.fold()
    
    def __repr__(self) -> str:
        return f"MonteCarloAgent({self.id}, ${self.money}, sims={self.num_simulations})"


class RandomAgent(PokerPlayer):
    """
    Simple random agent for testing.
    Makes random decisions with configurable fold probability.
    """
    
    def __init__(
        self,
        player_id: str,
        starting_money: int = 1000,
        fold_prob: float = 0.3,
        raise_prob: float = 0.2,
    ):
        super().__init__(player_id, starting_money)
        self.fold_prob = fold_prob
        self.raise_prob = raise_prob
    
    def get_action(
        self,
        hole_cards: List[int],
        board: List[int],
        pot: int,
        current_bet: int,
        min_raise: int,
        players: List[PokerPlayerPublic],
        my_idx: int,
    ) -> PokerAction:
        my_info = players[my_idx]
        to_call = current_bet - my_info.bet
        
        if my_info.folded or my_info.all_in:
            return PokerAction.check()
        
        r = random.random()
        
        if to_call <= 0:
            # No bet to call
            if r < self.raise_prob:
                raise_amount = min(min_raise, my_info.money)
                return PokerAction.raise_to(raise_amount)
            return PokerAction.check()
        
        if r < self.fold_prob:
            return PokerAction.fold()
        elif r < self.fold_prob + self.raise_prob:
            raise_amount = min(to_call + min_raise, my_info.money)
            return PokerAction.raise_to(raise_amount)
        else:
            return PokerAction.call(min(to_call, my_info.money))


class CallStationAgent(PokerPlayer):
    """
    Agent that always calls (never folds, never raises).
    Useful as a baseline opponent.
    """
    
    def __init__(self, player_id: str, starting_money: int = 1000):
        super().__init__(player_id, starting_money)
    
    def get_action(
        self,
        hole_cards: List[int],
        board: List[int],
        pot: int,
        current_bet: int,
        min_raise: int,
        players: List[PokerPlayerPublic],
        my_idx: int,
    ) -> PokerAction:
        my_info = players[my_idx]
        to_call = current_bet - my_info.bet
        
        if my_info.folded or my_info.all_in:
            return PokerAction.check()
        
        if to_call <= 0:
            return PokerAction.check()
        
        return PokerAction.call(min(to_call, my_info.money))


if __name__ == "__main__":
    # Test the agents
    from treys import Deck
    
    print("Testing MonteCarloAgent...")
    
    # Create agents
    mca = MonteCarloAgent("MCA-1", starting_money=1000, num_simulations=100)
    random_agent = RandomAgent("Random-1")
    call_station = CallStationAgent("CallStation-1")
    
    # Create test scenario
    deck = Deck()
    
    # Deal hole cards
    mca.deal_hand(tuple(deck.draw(2)))
    random_agent.deal_hand(tuple(deck.draw(2)))
    call_station.deal_hand(tuple(deck.draw(2)))
    
    # Deal board
    board = deck.draw(3)
    
    # Create public info
    players = [
        mca.get_public_info(),
        random_agent.get_public_info(),
        call_station.get_public_info(),
    ]
    
    # Test each agent
    for i, agent in enumerate([mca, random_agent, call_station]):
        action = agent.get_action(
            hole_cards=agent.get_hole_cards(),
            board=board,
            pot=100,
            current_bet=20,
            min_raise=10,
            players=players,
            my_idx=i,
        )
        print(f"  {agent.id}: {action.action_type.value} (amount={action.amount})")
    
    print("\n✓ Agent tests passed!")
