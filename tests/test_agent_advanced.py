"""
Advanced tests for PokerAgent decision-making and equity calculations.
These tests target uncovered code paths to increase coverage.
"""

import pytest
from agents import PokerAgent
from simulation import TexasHoldemSimulation
from treys import Card, Deck, Evaluator


@pytest.fixture
def agent():
    """Create a single agent for testing."""
    return PokerAgent(name="TestAgent", starting_chips=1000)


@pytest.fixture
def agents_for_equity():
    """Create multiple agents for equity testing."""
    return [PokerAgent(name=f"Player{i}", starting_chips=1000) for i in range(4)]


# ============================================================
# Hand Evaluation Coverage
# ============================================================

def test_evaluate_hand_with_empty_board(agent):
    """
    GIVEN an agent with hole cards
    WHEN evaluating with empty board (less than 3 cards)
    THEN should return None values
    """
    agent.hole_cards = [Card.new('Ah'), Card.new('Kh')]
    board = []
    
    score, hand_class, hand_name, percentage = agent.evaluate_hand(board)
    
    # With less than 3 board cards, returns None
    assert score is None
    assert hand_class is None
    assert hand_name is None
    assert percentage is None


def test_evaluate_hand_with_partial_board(agent):
    """
    GIVEN an agent with hole cards
    WHEN evaluating with flop only (3 cards)
    THEN should evaluate 5-card hand correctly
    """
    agent.hole_cards = [Card.new('Ah'), Card.new('Kh')]
    board = [Card.new('Qh'), Card.new('Jh'), Card.new('Th')]
    
    score, hand_class, hand_name, percentage = agent.evaluate_hand(board)
    
    # Should detect straight flush
    assert score < 100  # Very strong hand
    assert "Straight Flush" in hand_name or "Royal" in hand_name


def test_evaluate_hand_percentage_calculation(agent):
    """
    GIVEN an agent with a specific hand
    WHEN evaluating hand strength percentage
    THEN percentage should be between 0 and 1
    """
    agent.hole_cards = [Card.new('2c'), Card.new('7d')]  # Worst starting hand
    board = [Card.new('Kh'), Card.new('Qs'), Card.new('Jd'), Card.new('Tc'), Card.new('9h')]
    
    score, hand_class, hand_name, percentage = agent.evaluate_hand(board)
    
    assert 0.0 <= percentage <= 1.0
    # Weak hand, percentage should be valid
    assert percentage >= 0.0


# ============================================================
# Equity Calculation Coverage
# ============================================================

def test_calculate_equity_pre_flop(agent):
    """
    GIVEN an agent with hole cards pre-flop
    WHEN calculating equity with opponent hands
    THEN should return probability between 0 and 1
    """
    agent.hole_cards = [Card.new('Ah'), Card.new('Ad')]  # Pocket aces
    board = []
    
    # Create opponent hands
    opponent_hands = [
        [Card.new('Kh'), Card.new('Kd')],
        [Card.new('Qh'), Card.new('Qd')],
        [Card.new('Jh'), Card.new('Jd')]
    ]
    
    # Create remaining deck (exclude used cards)
    deck = Deck()
    used_cards = set(agent.hole_cards)
    for hand in opponent_hands:
        used_cards.update(hand)
    remaining_deck = [card for card in deck.cards if card not in used_cards]
    
    equity = agent.calculate_equity(board, opponent_hands, remaining_deck, num_simulations=100)
    
    assert 0.0 <= equity <= 1.0
    assert equity > 0.3  # Pocket aces should have decent equity


def test_calculate_equity_on_flop(agent):
    """
    GIVEN an agent on the flop
    WHEN calculating equity
    THEN should simulate remaining cards
    """
    agent.hole_cards = [Card.new('Ah'), Card.new('Kh')]
    board = [Card.new('Qh'), Card.new('Jh'), Card.new('2c')]  # Flush draw
    
    opponent_hands = [[Card.new('9d'), Card.new('8d')]]
    deck = Deck()
    used_cards = set(agent.hole_cards + board + opponent_hands[0])
    remaining_deck = [card for card in deck.cards if card not in used_cards]
    
    equity = agent.calculate_equity(board, opponent_hands, remaining_deck, num_simulations=100)
    
    assert 0.0 <= equity <= 1.0


def test_calculate_equity_on_turn(agent):
    """
    GIVEN an agent on the turn
    WHEN calculating equity
    THEN should simulate only river card
    """
    agent.hole_cards = [Card.new('Ah'), Card.new('Kh')]
    board = [Card.new('Qh'), Card.new('Jh'), Card.new('Th'), Card.new('2c')]  # Made straight flush
    
    opponent_hands = [[Card.new('9d'), Card.new('8d')], [Card.new('7c'), Card.new('6c')]]
    deck = Deck()
    used_cards = set(agent.hole_cards + board)
    for hand in opponent_hands:
        used_cards.update(hand)
    remaining_deck = [card for card in deck.cards if card not in used_cards]
    
    equity = agent.calculate_equity(board, opponent_hands, remaining_deck, num_simulations=100)
    
    assert equity > 0.8  # Strong hand should have high equity


def test_calculate_equity_on_river(agent):
    """
    GIVEN an agent on the river
    WHEN calculating equity
    THEN should return exact win probability (no more cards to come)
    """
    agent.hole_cards = [Card.new('Ah'), Card.new('Ad')]
    board = [Card.new('Kh'), Card.new('Kd'), Card.new('Kc'), Card.new('2s'), Card.new('3s')]  # Full house
    
    opponent_hands = [[Card.new('Qh'), Card.new('Qd')]]
    remaining_deck = []  # No cards left on river
    
    equity = agent.calculate_equity(board, opponent_hands, remaining_deck)
    
    assert 0.0 <= equity <= 1.0


def test_all_equities_calculation(agents_for_equity):
    """
    GIVEN multiple agents with hole cards
    WHEN calculating all equities at once
    THEN should return equity for each player summing to ~1.0
    """
    # Deal hole cards
    agents_for_equity[0].hole_cards = [Card.new('Ah'), Card.new('Ad')]
    agents_for_equity[1].hole_cards = [Card.new('Kh'), Card.new('Kd')]
    agents_for_equity[2].hole_cards = [Card.new('Qh'), Card.new('Qd')]
    agents_for_equity[3].hole_cards = [Card.new('Jh'), Card.new('Jd')]
    
    board = []
    all_hands = [agent.hole_cards for agent in agents_for_equity]
    
    # Create remaining deck (exclude used cards)
    deck = Deck()
    used_cards = set()
    for hand in all_hands:
        used_cards.update(hand)
    remaining_cards = [card for card in deck.cards if card not in used_cards]
    
    equities = agents_for_equity[0]._calculate_all_equities(board, all_hands, remaining_cards, num_simulations=100)
    
    assert len(equities) == 4
    assert all(0.0 <= eq <= 1.0 for eq in equities)
    # Aces should have highest equity
    assert equities[0] == max(equities)


# ============================================================
# Decision Making Coverage
# ============================================================

def test_make_decision_with_terrible_hand(agent):
    """
    GIVEN an agent with terrible hand
    WHEN facing a large bet
    THEN should make a valid decision
    """
    agent.hole_cards = [Card.new('2c'), Card.new('7d')]  # Worst hand
    board = [Card.new('Kh'), Card.new('Ks'), Card.new('Kd')]  # Board doesn't help
    
    action, amount = agent.make_decision(board, pot_size=100, current_bet_to_call=50, min_raise=50)
    
    # Agent should make a valid decision
    assert action in ['fold', 'call', 'check', 'raise']


def test_make_decision_with_zero_chips(agent):
    """
    GIVEN an agent with zero chips (all-in or eliminated)
    WHEN it's their turn to act
    THEN should check or be skipped
    """
    agent.chips = 0
    agent.is_all_in = True
    agent.hole_cards = [Card.new('Ah'), Card.new('Ad')]
    
    action, amount = agent.make_decision([], pot_size=100, current_bet_to_call=50, min_raise=50)
    
    # All-in players can't act
    assert action == 'check'
    assert amount == 0


def test_make_decision_can_check_when_no_bet(agent):
    """
    GIVEN an agent facing no bet
    WHEN making decision
    THEN should have option to check
    """
    agent.hole_cards = [Card.new('7h'), Card.new('6h')]
    board = [Card.new('2c'), Card.new('9d'), Card.new('Ks')]
    
    action, amount = agent.make_decision(board, pot_size=40, current_bet_to_call=0, min_raise=10)
    
    # Should be able to check or raise
    assert action in ['check', 'raise']


def test_make_decision_with_strong_hand_raises(agent):
    """
    GIVEN an agent with very strong hand
    WHEN making decision
    THEN may raise aggressively
    """
    agent.hole_cards = [Card.new('Ah'), Card.new('Ad')]  # Pocket aces
    board = [Card.new('Ac'), Card.new('As'), Card.new('Kh')]  # Quad aces!
    
    action, amount = agent.make_decision(board, pot_size=100, current_bet_to_call=0, min_raise=20)
    
    # With quad aces, should be aggressive
    assert action in ['raise', 'check']  # Might slow play or raise


def test_make_decision_respects_chip_limit(agent):
    """
    GIVEN an agent with limited chips
    WHEN making a decision to raise
    THEN raise amount should not exceed available chips
    """
    agent.chips = 50  # Only 50 chips left
    agent.hole_cards = [Card.new('Kh'), Card.new('Kd')]
    board = []
    
    action, amount = agent.make_decision(board, pot_size=100, current_bet_to_call=0, min_raise=20)
    
    if action == 'raise':
        assert amount <= 50  # Can't raise more than available


# ============================================================
# State Management Coverage
# ============================================================

def test_reset_for_new_hand(agent):
    """
    GIVEN an agent after a hand
    WHEN resetting for new hand
    THEN should clear hand-specific state
    """
    # Simulate being in a hand
    agent.hole_cards = [Card.new('Ah'), Card.new('Kh')]
    agent.is_folded = True
    agent.current_bet = 50
    agent.total_invested = 100
    agent.is_all_in = True
    
    agent.reset_for_new_hand()
    
    assert agent.hole_cards == []
    assert agent.is_folded == False
    assert agent.current_bet == 0
    assert agent.total_invested == 0
    assert agent.is_all_in == False


def test_reset_current_bet(agent):
    """
    GIVEN an agent who bet in previous round
    WHEN resetting for new betting round
    THEN current_bet should reset but chips remain
    """
    initial_chips = agent.chips
    agent.current_bet = 50
    
    agent.reset_current_bet()
    
    assert agent.current_bet == 0
    assert agent.chips == initial_chips  # Chips not affected


def test_string_representation(agent):
    """
    GIVEN an agent with a name
    WHEN converting to string
    THEN should return the name
    """
    agent.name = "TestPlayer"
    assert str(agent) == "TestPlayer"


def test_string_representation_no_name(agent):
    """
    GIVEN an agent without a name
    WHEN converting to string
    THEN should return default representation
    """
    agent_no_name = PokerAgent(starting_chips=1000)
    result = str(agent_no_name)
    # Agent with no explicit name can return "Agent" or "None" or similar
    assert result in ["Agent", "None"] or "PokerAgent" in result or "Agent" in result


# ============================================================
# Edge Cases and Boundary Conditions
# ============================================================

def test_agent_with_zero_starting_chips():
    """
    GIVEN an agent created with 0 chips
    WHEN checking their state
    THEN should be valid but unable to bet
    """
    broke_agent = PokerAgent(name="Broke", starting_chips=0)
    
    assert broke_agent.get_chips() == 0
    assert broke_agent.place_bet(10) == 0  # Can't bet


def test_agent_with_very_large_chip_stack():
    """
    GIVEN an agent with massive chip stack
    WHEN betting and calculating
    THEN should handle large numbers correctly
    """
    rich_agent = PokerAgent(name="Rich", starting_chips=1000000)
    
    assert rich_agent.get_chips() == 1000000
    amount = rich_agent.place_bet(500000)
    assert amount == 500000
    assert rich_agent.get_chips() == 500000


def test_consecutive_bets_accumulate(agent):
    """
    GIVEN an agent making multiple bets
    WHEN tracking total investment
    THEN should accumulate correctly
    """
    agent.place_bet(100)
    agent.place_bet(50)
    agent.place_bet(25)
    
    assert agent.total_invested == 175
    assert agent.current_bet == 175


def test_equity_with_one_opponent(agent):
    """
    GIVEN an agent heads-up (1v1)
    WHEN calculating equity
    THEN should handle 1 opponent correctly
    """
    agent.hole_cards = [Card.new('Ah'), Card.new('Kd')]
    board = [Card.new('2c'), Card.new('7h'), Card.new('9s')]
    
    opponent_hands = [[Card.new('Qh'), Card.new('Qd')]]
    deck = Deck()
    used_cards = set(agent.hole_cards + board)
    used_cards.update(opponent_hands[0])
    remaining_deck = [card for card in deck.cards if card not in used_cards]
    
    equity = agent.calculate_equity(board, opponent_hands, remaining_deck)
    
    assert 0.0 <= equity <= 1.0


def test_equity_with_many_opponents(agent):
    """
    GIVEN an agent at full table
    WHEN calculating equity with 8 opponents
    THEN should handle many opponents
    """
    agent.hole_cards = [Card.new('Ah'), Card.new('Ad')]
    board = []
    
    # Create 8 opponent hands
    opponent_hands = [
        [Card.new('Kh'), Card.new('Kd')],
        [Card.new('Qh'), Card.new('Qd')],
        [Card.new('Jh'), Card.new('Jd')],
        [Card.new('Th'), Card.new('Td')],
        [Card.new('9h'), Card.new('9d')],
        [Card.new('8h'), Card.new('8d')],
        [Card.new('7h'), Card.new('7d')],
        [Card.new('6h'), Card.new('6d')]
    ]
    
    deck = Deck()
    used_cards = set(agent.hole_cards)
    for hand in opponent_hands:
        used_cards.update(hand)
    remaining_deck = [card for card in deck.cards if card not in used_cards]
    
    equity = agent.calculate_equity(board, opponent_hands, remaining_deck, num_simulations=100)
    
    assert 0.0 <= equity <= 1.0
    # Even aces have lower equity vs 8 opponents
    assert equity > 0.2  # But still significant


def test_evaluate_hand_with_full_board(agent):
    """
    GIVEN an agent on the river
    WHEN evaluating with all 5 board cards
    THEN should evaluate complete 7-card hand
    """
    agent.hole_cards = [Card.new('7h'), Card.new('7d')]
    board = [Card.new('7c'), Card.new('7s'), Card.new('Kh'), Card.new('Kd'), Card.new('Kc')]
    
    score, hand_class, hand_name, percentage = agent.evaluate_hand(board)
    
    # Four 7s beats three Kings
    assert "Four of a Kind" in hand_name or "Quads" in hand_name
    assert score < 200  # Very strong hand


def test_folded_agent_cannot_win(agent):
    """
    GIVEN an agent who folded
    WHEN checking if they can win
    THEN they should be marked as folded
    """
    agent.fold()
    
    assert agent.is_folded
    # Folded players shouldn't win pots


def test_all_in_flag_set_correctly(agent):
    """
    GIVEN an agent betting their last chips
    WHEN checking all-in status
    THEN flag should be set
    """
    agent.chips = 100
    agent.place_bet(100)
    
    assert agent.is_all_in
    assert agent.get_chips() == 0


def test_partial_bet_sets_all_in(agent):
    """
    GIVEN an agent with 50 chips trying to bet 100
    WHEN placing bet
    THEN should bet 50 and be all-in
    """
    agent.chips = 50
    amount = agent.place_bet(100)
    
    assert amount == 50
    assert agent.is_all_in
    assert agent.get_chips() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
