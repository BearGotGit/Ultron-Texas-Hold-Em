"""
Poker utility functions.
Provides standalone utility functions for poker calculations.
"""

from itertools import combinations
import random
from treys import Evaluator


def calculate_hand_equity(hole_cards, board, opponent_hands, remaining_deck_cards, num_simulations=1000):
    """
    Calculate equity (win probability) against opponents using Monte Carlo simulation.
    
    Args:
        hole_cards: List of 2 card integers for the player
        board: Current community cards
        opponent_hands: List of opponent hole cards
        remaining_deck_cards: Cards still in the deck
        num_simulations: Number of simulations to run
        
    Returns:
        Float representing win probability (0.0 to 1.0)
    """
    all_hands = [hole_cards] + opponent_hands
    equities = calculate_all_equities(board, all_hands, remaining_deck_cards, num_simulations)
    return equities[0]  # Return equity for first hand in list


def calculate_all_equities(board, hands, remaining_deck_cards, num_simulations=1000):
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
    evaluator = Evaluator()
    cards_needed = 5 - len(board)
    
    if cards_needed == 0:
        # All community cards dealt, just evaluate once
        scores = [evaluator.evaluate(board, hand) for hand in hands]
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
        scores = [evaluator.evaluate(full_board, hand) for hand in hands]
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
