"""
Agent class for Texas Hold'em poker players.
Handles card management, betting, and equity calculations.
"""

from treys import Card, Evaluator
from itertools import combinations
import random


class PokerAgent:
    """
    Represents a poker player with hole cards, chip stack, and decision-making capabilities.
    """
    
    def __init__(self, name=None, starting_chips=1000):
        """
        Initialize a poker agent.
        
        Args:
            name: Optional name for the agent (default: None)
            starting_chips: Starting chip stack (default: 1000)
        """
        self.name = name
        self.hole_cards = []
        self.evaluator = Evaluator()
        
        # Betting state
        self.chips = starting_chips
        self.current_bet = 0  # Amount bet in current betting round
        self.total_invested = 0  # Total amount invested in this hand
        self.is_folded = False
        self.is_all_in = False
    
    def receive_cards(self, cards):
        """
        Receive hole cards.
        
        Args:
            cards: List of 2 card integers
        """
        self.hole_cards = cards
    
    def reset_for_new_hand(self):
        """Reset agent state for a new hand."""
        self.hole_cards = []
        self.current_bet = 0
        self.total_invested = 0
        self.is_folded = False
        self.is_all_in = False
    
    def reset_current_bet(self):
        """Reset current bet for a new betting round."""
        self.current_bet = 0
    
    def get_hole_cards(self):
        """
        Get the agent's hole cards.
        
        Returns:
            List of 2 card integers
        """
        return self.hole_cards
    
    def get_chips(self):
        """Get current chip stack."""
        return self.chips
    
    def add_chips(self, amount):
        """
        Add chips to stack (when winning).
        
        Args:
            amount: Number of chips to add
        """
        self.chips += amount
    
    def place_bet(self, amount):
        """
        Place a bet (internal method - deducts chips and tracks bet).
        
        Args:
            amount: Amount to bet
            
        Returns:
            Actual amount bet (may be less if all-in)
        """
        actual_bet = min(amount, self.chips)
        self.chips -= actual_bet
        self.current_bet += actual_bet
        self.total_invested += actual_bet
        
        if self.chips == 0:
            self.is_all_in = True
        
        return actual_bet
    
    def fold(self):
        """Fold the hand."""
        self.is_folded = True
    
    def make_decision(self, board, pot_size, current_bet_to_call, min_raise):
        """
        Make a betting decision. Override this method to implement custom strategies.
        
        Args:
            board: Current community cards
            pot_size: Current pot size
            current_bet_to_call: Amount needed to call
            min_raise: Minimum raise amount
            
        Returns:
            Tuple of (action, amount) where:
                action: 'fold', 'call', 'raise', 'check'
                amount: Amount to bet (0 for fold/check/call, raise amount for raise)
        """
        # Default strategy: simple equity-based decisions
        if self.is_folded or self.is_all_in:
            return ('check', 0)
        
        # If no board cards yet, use simple pre-flop strategy
        if len(board) == 0:
            return self._preflop_decision(current_bet_to_call, min_raise)
        
        # Calculate hand strength
        score, _, hand_name, percentage = self.evaluate_hand(board)
        
        # Simple strategy based on hand strength (percentage is 0=best, 1=worst)
        if percentage < 0.3:  # Strong hand
            if current_bet_to_call == 0:
                return ('raise', min_raise)
            elif current_bet_to_call <= pot_size * 0.5:
                return ('raise', current_bet_to_call + min_raise)
            else:
                return ('call', current_bet_to_call)
        elif percentage < 0.6:  # Medium hand
            if current_bet_to_call == 0:
                return ('check', 0)
            elif current_bet_to_call <= pot_size * 0.3:
                return ('call', current_bet_to_call)
            else:
                return ('fold', 0)
        else:  # Weak hand
            if current_bet_to_call == 0:
                return ('check', 0)
            else:
                return ('fold', 0)
    
    def _preflop_decision(self, current_bet_to_call, min_raise):
        """Simple pre-flop decision based on hole cards."""
        if current_bet_to_call == 0:
            return ('check', 0)
        elif current_bet_to_call <= self.chips * 0.1:
            return ('call', current_bet_to_call)
        else:
            return ('fold', 0)
    
    def evaluate_hand(self, board):
        """
        Evaluate the strength of the current hand given the board.
        
        Args:
            board: List of community cards
            
        Returns:
            Tuple of (score, hand_class, hand_name, percentage)
        """
        if len(board) < 3:
            return None, None, None, None
        
        score = self.evaluator.evaluate(board, self.hole_cards)
        hand_class = self.evaluator.get_rank_class(score)
        hand_name = self.evaluator.class_to_string(hand_class)
        percentage = self.evaluator.get_five_card_rank_percentage(score)
        
        return score, hand_class, hand_name, percentage
    
    def calculate_equity(self, board, opponent_hands, remaining_deck_cards, num_simulations=1000):
        """
        Calculate equity (win probability) against opponents using Monte Carlo simulation.
        
        Args:
            board: Current community cards
            opponent_hands: List of opponent hole cards
            remaining_deck_cards: Cards still in the deck
            num_simulations: Number of simulations to run
            
        Returns:
            Float representing win probability (0.0 to 1.0)
        """
        all_hands = [self.hole_cards] + opponent_hands
        equities = self._calculate_all_equities(board, all_hands, remaining_deck_cards, num_simulations)
        return equities[0]  # Return equity for this agent (first in list)
    
    def _calculate_all_equities(self, board, hands, remaining_deck_cards, num_simulations=1000):
        """
        Calculate equity for all hands in the game.
        
        Args:
            board: Current community cards
            hands: List of all player hands
            remaining_deck_cards: Cards still in the deck
            num_simulations: Number of simulations to run
            
        Returns:
            List of win probabilities for each hand
        """
        cards_needed = 5 - len(board)
        
        if cards_needed == 0:
            # All community cards dealt, just evaluate once
            scores = [self.evaluator.evaluate(board, hand) for hand in hands]
            best_score = min(scores)
            wins = [1 if score == best_score else 0 for score in scores]
            num_winners = sum(wins)
            return [w / num_winners for w in wins]
        
        # Monte Carlo simulation
        wins = [0] * len(hands)
        ties = [0] * len(hands)
        
        # Generate possible future boards
        possible_boards = list(combinations(remaining_deck_cards, cards_needed))
        actual_simulations = min(num_simulations, len(possible_boards))
        sampled_boards = random.sample(possible_boards, actual_simulations) if len(possible_boards) > num_simulations else possible_boards
        
        for future_cards in sampled_boards:
            full_board = board + list(future_cards)
            scores = [self.evaluator.evaluate(full_board, hand) for hand in hands]
            best_score = min(scores)
            winners = [i for i, score in enumerate(scores) if score == best_score]
            
            if len(winners) == 1:
                wins[winners[0]] += 1
            else:
                for winner in winners:
                    ties[winner] += 1 / len(winners)
        
        total_sims = len(sampled_boards)
        equity = [(wins[i] + ties[i]) / total_sims for i in range(len(hands))]
        
        return equity
    
    def __str__(self):
        """String representation of the agent."""
        return self.name if self.name else "Agent"