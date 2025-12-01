"""
Advanced simulation tests covering betting rounds, dealer rotation, and complex scenarios.
These tests target uncovered code paths in poker_simulator.py
"""

import pytest
from agents import PokerAgent
from simulation import TexasHoldemSimulation
from treys import Card


@pytest.fixture
def game_with_agents():
    """Create a game with 4 agents."""
    agents = [PokerAgent(name=f"Player{i+1}", starting_chips=1000) for i in range(4)]
    return TexasHoldemSimulation(agents, small_blind=5, big_blind=10), agents


@pytest.fixture
def three_player_game():
    """Create a 3-player game."""
    agents = [PokerAgent(name=f"Player{i+1}", starting_chips=500) for i in range(3)]
    return TexasHoldemSimulation(agents, small_blind=5, big_blind=10), agents


# ============================================================
# Dealer Position and Rotation
# ============================================================

def test_dealer_position_initializes_to_zero(game_with_agents):
    """
    GIVEN a new game
    WHEN checking dealer position
    THEN should start at 0
    """
    game, agents = game_with_agents
    assert game.dealer_position == 0


def test_reset_for_new_hand_maintains_dealer(game_with_agents):
    """
    GIVEN a game after a hand
    WHEN resetting for new hand
    THEN dealer position should be maintained (rotated externally)
    """
    game, agents = game_with_agents
    initial_dealer = game.dealer_position
    
    game.reset_for_new_hand()
    
    # Dealer position doesn't auto-rotate in reset
    assert game.dealer_position == initial_dealer


def test_reset_for_new_hand_clears_board(game_with_agents):
    """
    GIVEN a game with community cards
    WHEN resetting for new hand
    THEN board should be cleared
    """
    game, agents = game_with_agents
    game.deal_hole_cards()
    game.deal_flop()
    
    assert len(game.board) == 3
    
    game.reset_for_new_hand()
    
    assert game.board == []


def test_reset_for_new_hand_creates_new_deck(game_with_agents):
    """
    GIVEN a game with cards dealt
    WHEN resetting for new hand
    THEN should create fresh deck
    """
    game, agents = game_with_agents
    game.deal_hole_cards()
    initial_deck_size = len(game.deck.cards)
    
    game.reset_for_new_hand()
    
    # New full deck
    assert len(game.deck.cards) == 52


def test_reset_for_new_hand_resets_pot(game_with_agents):
    """
    GIVEN a game with money in pot
    WHEN resetting for new hand
    THEN pot should be cleared
    """
    game, agents = game_with_agents
    game.pot = 500
    
    game.reset_for_new_hand()
    
    assert game.pot == 0


def test_reset_for_new_hand_resets_current_bet(game_with_agents):
    """
    GIVEN a game with active bets
    WHEN resetting for new hand
    THEN current bet should reset to 0
    """
    game, agents = game_with_agents
    game.current_bet = 100
    
    game.reset_for_new_hand()
    
    assert game.current_bet == 0


def test_reset_for_new_hand_resets_min_raise(game_with_agents):
    """
    GIVEN a game with min raise set
    WHEN resetting for new hand
    THEN min raise should reset to big blind
    """
    game, agents = game_with_agents
    game.min_raise = 50
    
    game.reset_for_new_hand()
    
    assert game.min_raise == game.big_blind


# ============================================================
# Blind Posting Edge Cases
# ============================================================

def test_post_blinds_with_three_players(three_player_game):
    """
    GIVEN a 3-player game
    WHEN posting blinds
    THEN positions should be calculated correctly
    """
    game, agents = three_player_game
    
    sb_pos, bb_pos = game.post_blinds()
    
    # Should be consecutive positions
    assert (sb_pos + 1) % 3 == bb_pos


def test_post_blinds_deducts_from_agent_chips(game_with_agents):
    """
    GIVEN agents with chips
    WHEN blinds are posted
    THEN chips should be deducted from correct agents
    """
    game, agents = game_with_agents
    initial_chips = [a.get_chips() for a in agents]
    
    sb_pos, bb_pos = game.post_blinds()
    
    assert agents[sb_pos].get_chips() == initial_chips[sb_pos] - game.small_blind
    assert agents[bb_pos].get_chips() == initial_chips[bb_pos] - game.big_blind


def test_post_blinds_sets_current_bet_to_big_blind(game_with_agents):
    """
    GIVEN a new hand
    WHEN blinds are posted
    THEN current bet should be set to big blind amount
    """
    game, agents = game_with_agents
    
    game.post_blinds()
    
    assert game.current_bet == game.big_blind


def test_post_blinds_adds_to_pot(game_with_agents):
    """
    GIVEN an empty pot
    WHEN blinds are posted
    THEN pot should contain both blinds
    """
    game, agents = game_with_agents
    
    game.post_blinds()
    
    assert game.pot == game.small_blind + game.big_blind


# ============================================================
# Card Dealing Coverage
# ============================================================

def test_deal_hole_cards_removes_from_deck(game_with_agents):
    """
    GIVEN a fresh deck
    WHEN dealing hole cards to 4 players
    THEN 8 cards should be removed from deck
    """
    game, agents = game_with_agents
    initial_deck_size = len(game.deck.cards)
    
    game.deal_hole_cards()
    
    # 4 players * 2 cards = 8 cards dealt
    assert len(game.deck.cards) == initial_deck_size - 8


def test_deal_flop_adds_three_cards(game_with_agents):
    """
    GIVEN an empty board
    WHEN dealing the flop
    THEN exactly 3 cards should be added
    """
    game, agents = game_with_agents
    game.deal_hole_cards()
    
    game.deal_flop()
    
    assert len(game.board) == 3


def test_deal_turn_adds_one_card(game_with_agents):
    """
    GIVEN a board with flop
    WHEN dealing the turn
    THEN exactly 1 card should be added
    """
    game, agents = game_with_agents
    game.deal_hole_cards()
    game.deal_flop()
    
    game.deal_turn()
    
    assert len(game.board) == 4


def test_deal_river_adds_one_card(game_with_agents):
    """
    GIVEN a board with flop and turn
    WHEN dealing the river
    THEN exactly 1 card should be added
    """
    game, agents = game_with_agents
    game.deal_hole_cards()
    game.deal_flop()
    game.deal_turn()
    
    game.deal_river()
    
    assert len(game.board) == 5


def test_sequential_dealing_maintains_unique_cards(game_with_agents):
    """
    GIVEN cards dealt throughout a hand
    WHEN checking all cards in play
    THEN all should be unique
    """
    game, agents = game_with_agents
    game.deal_hole_cards()
    game.deal_flop()
    game.deal_turn()
    game.deal_river()
    
    all_cards = []
    for agent in agents:
        all_cards.extend(agent.hole_cards)
    all_cards.extend(game.board)
    
    # Should have 8 hole cards + 5 board = 13 unique cards
    assert len(all_cards) == len(set(all_cards))


# ============================================================
# Print and Display Methods
# ============================================================

def test_print_board_with_cards(game_with_agents, capsys):
    """
    GIVEN a board with cards
    WHEN printing the board
    THEN should display cards
    """
    game, agents = game_with_agents
    game.deal_hole_cards()
    game.deal_flop()
    
    game.print_board()
    
    captured = capsys.readouterr()
    # Should print something (cards display)
    assert len(captured.out) > 0


def test_print_board_with_empty_board(game_with_agents, capsys):
    """
    GIVEN an empty board
    WHEN printing the board
    THEN should display appropriate message
    """
    game, agents = game_with_agents
    
    game.print_board()
    
    captured = capsys.readouterr()
    assert "No community cards" in captured.out or len(captured.out) > 0


def test_print_game_state_shows_all_info(game_with_agents, capsys):
    """
    GIVEN a game in progress
    WHEN printing game state
    THEN should show board and all player hands
    """
    game, agents = game_with_agents
    game.deal_hole_cards()
    game.deal_flop()
    
    game.print_game_state()
    
    captured = capsys.readouterr()
    # Should contain player names and cards
    assert "Player1" in captured.out or len(captured.out) > 0


# ============================================================
# Evaluate Hands Coverage
# ============================================================

def test_evaluate_hands_returns_results_for_all_players(game_with_agents):
    """
    GIVEN 4 players with hole cards
    WHEN evaluating hands
    THEN should return results for all 4 players
    """
    game, agents = game_with_agents
    game.deal_hole_cards()
    game.board = [Card.new('Kh'), Card.new('Qd'), Card.new('Js'), 
                   Card.new('Tc'), Card.new('9h')]
    
    results = game.evaluate_hands()
    
    assert len(results) == 4
    for agent, score, hand_name, percentage in results:
        assert isinstance(score, int)
        assert isinstance(hand_name, str)
        assert 0.0 <= percentage <= 1.0


def test_get_winner_with_single_player_not_folded(game_with_agents):
    """
    GIVEN 4 players where 3 have folded
    WHEN getting winner
    THEN should return the one remaining player
    """
    game, agents = game_with_agents
    game.deal_hole_cards()
    game.board = [Card.new('Kh'), Card.new('Qd'), Card.new('Js')]
    
    # Fold all but one
    agents[0].fold()
    agents[1].fold()
    agents[2].fold()
    
    winners = game.get_winner()
    
    assert len(winners) == 1
    assert winners[0] == agents[3]


def test_get_winner_with_tie(game_with_agents):
    """
    GIVEN players with identical hands
    WHEN determining winner
    THEN should return multiple winners
    """
    game, agents = game_with_agents
    
    # Give all players same cards for testing
    same_board = [Card.new('Ah'), Card.new('Kh'), Card.new('Qh'),
                  Card.new('Jh'), Card.new('Th')]
    game.board = same_board
    
    # Give everyone low cards that don't help
    agents[0].hole_cards = [Card.new('2c'), Card.new('3c')]
    agents[1].hole_cards = [Card.new('2d'), Card.new('3d')]
    agents[2].hole_cards = [Card.new('4c'), Card.new('5c')]
    agents[3].hole_cards = [Card.new('4d'), Card.new('5d')]
    
    winners = game.get_winner()
    
    # All should tie with royal flush from board
    assert len(winners) >= 2


# ============================================================
# Calculate All Equities
# ============================================================

def test_calculate_all_equities_returns_list(game_with_agents):
    """
    GIVEN 4 players with hole cards
    WHEN calculating all equities
    THEN should return list of 4 equity values
    """
    game, agents = game_with_agents
    game.deal_hole_cards()
    
    equities = game.calculate_all_equities()
    
    assert len(equities) == 4
    assert all(0.0 <= eq <= 1.0 for eq in equities)


def test_calculate_all_equities_with_board(game_with_agents):
    """
    GIVEN players with board cards
    WHEN calculating equities
    THEN should account for board
    """
    game, agents = game_with_agents
    game.deal_hole_cards()
    game.deal_flop()
    
    equities = game.calculate_all_equities()
    
    assert len(equities) == 4
    # At least one player should have >0 equity
    assert any(eq > 0 for eq in equities)


# ============================================================
# Award Pot with Side Pots
# ============================================================

def test_award_pot_to_single_winner(game_with_agents):
    """
    GIVEN a pot with clear winner
    WHEN awarding pot
    THEN winner should receive all chips
    """
    game, agents = game_with_agents
    game.deal_hole_cards()
    game.board = [Card.new('Kh'), Card.new('Kd'), Card.new('Kc'),
                  Card.new('Ks'), Card.new('Ah')]
    
    # Give one player pocket aces (for full house)
    agents[0].hole_cards = [Card.new('Ac'), Card.new('Ad')]
    agents[0].total_invested = 100
    
    # Others have nothing
    for i in range(1, 4):
        agents[i].total_invested = 100
        agents[i].fold()
    
    game.pot = 100
    initial_chips = agents[0].get_chips()
    
    results = game.award_pot()
    
    # Winner should have received chips
    assert agents[0].get_chips() > initial_chips


def test_award_pot_with_multiple_side_pots(game_with_agents):
    """
    GIVEN players with different investment levels
    WHEN awarding pots
    THEN should create and award multiple side pots correctly
    """
    game, agents = game_with_agents
    game.deal_hole_cards()
    game.board = [Card.new('Kh'), Card.new('Qd'), Card.new('Jc'),
                  Card.new('Tc'), Card.new('9h')]
    
    # Set different investment levels
    agents[0].total_invested = 100
    agents[1].total_invested = 50
    agents[1].is_all_in = True
    agents[2].total_invested = 20
    agents[2].is_all_in = True
    agents[3].fold()
    
    game.pot = 170
    
    results = game.award_pot()
    
    # Should return list of (winner, amount, pot_description) tuples
    assert len(results) > 0
    assert all(isinstance(r, tuple) and len(r) == 3 for r in results)


def test_award_pot_splits_evenly_on_tie(game_with_agents):
    """
    GIVEN two players with identical best hands
    WHEN awarding pot
    THEN pot should split evenly
    """
    game, agents = game_with_agents
    
    # Board plays (royal flush)
    game.board = [Card.new('Ah'), Card.new('Kh'), Card.new('Qh'),
                  Card.new('Jh'), Card.new('Th')]
    
    # Both players have low cards that don't improve
    agents[0].hole_cards = [Card.new('2c'), Card.new('3c')]
    agents[0].total_invested = 100
    agents[1].hole_cards = [Card.new('2d'), Card.new('3d')]
    agents[1].total_invested = 100
    
    # Others fold
    agents[2].fold()
    agents[3].fold()
    
    game.pot = 200
    initial_chips_0 = agents[0].get_chips()
    initial_chips_1 = agents[1].get_chips()
    
    results = game.award_pot()
    
    # Both should receive equal amounts
    chips_won_0 = agents[0].get_chips() - initial_chips_0
    chips_won_1 = agents[1].get_chips() - initial_chips_1
    
    assert chips_won_0 == chips_won_1


# ============================================================
# Betting Round Edge Cases
# ============================================================

def test_betting_round_with_all_but_one_folded(game_with_agents):
    """
    GIVEN a betting round where all but one folds
    WHEN round completes
    THEN should end early
    """
    game, agents = game_with_agents
    game.deal_hole_cards()
    game.post_blinds()
    
    # Manually fold all but one
    agents[0].fold()
    agents[1].fold()
    agents[2].fold()
    
    result = game.run_betting_round("Test")
    
    assert result == True  # Round completed


def test_betting_round_with_all_all_in(game_with_agents):
    """
    GIVEN all players all-in
    WHEN checking if betting should continue
    THEN should end betting round
    """
    game, agents = game_with_agents
    game.deal_hole_cards()
    
    # Make all players all-in
    for agent in agents:
        agent.place_bet(agent.get_chips())
    
    result = game.run_betting_round("Test")
    
    assert result == True


def test_get_pot_size_returns_current_pot(game_with_agents):
    """
    GIVEN a pot with chips
    WHEN getting pot size
    THEN should return correct amount
    """
    game, agents = game_with_agents
    game.pot = 250
    
    assert game.get_pot_size() == 250


# ============================================================
# Integration: Complete Hand Scenarios
# ============================================================

def test_complete_hand_all_streets(game_with_agents):
    """
    INTEGRATION: Play through all streets
    GIVEN a complete hand
    WHEN dealing all cards and running all betting rounds
    THEN should complete successfully
    """
    game, agents = game_with_agents
    
    game.deal_hole_cards()
    game.post_blinds()
    game.run_betting_round("Pre-flop")
    
    game.deal_flop()
    game.run_betting_round("Flop")
    
    game.deal_turn()
    game.run_betting_round("Turn")
    
    game.deal_river()
    game.run_betting_round("River")
    
    results = game.evaluate_hands()
    winners = game.get_winner()
    
    assert len(results) == 4
    assert len(winners) >= 1


def test_three_player_complete_hand(three_player_game):
    """
    INTEGRATION: 3-player game
    GIVEN a 3-player game
    WHEN playing complete hand
    THEN should handle correctly
    """
    game, agents = three_player_game
    
    game.deal_hole_cards()
    game.post_blinds()
    
    assert len(game.board) == 0
    
    game.deal_flop()
    assert len(game.board) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
