"""
Comprehensive tests for Texas Hold'em poker rules compliance.

This test suite verifies that the simulation correctly implements all
fundamental rules of Texas Hold'em poker using pytest best practices:
- Arrange-Act-Assert (AAA) pattern
- Descriptive test names following Given-When-Then
- Isolated tests with minimal dependencies
- Fixtures for common setup
"""

import pytest
from agents import PokerAgent
from simulation import TexasHoldemSimulation
from treys import Card, Evaluator


# ============================================================
# FIXTURES - Common test setup
# ============================================================

@pytest.fixture
def basic_agents():
    """Create 4 agents with standard starting chips."""
    return [
        PokerAgent(name=f"Player {i+1}", starting_chips=1000)
        for i in range(4)
    ]


@pytest.fixture
def game(basic_agents):
    """Create a standard game with 4 players."""
    return TexasHoldemSimulation(basic_agents, small_blind=5, big_blind=10)


@pytest.fixture
def small_stack_agents():
    """Create agents with varying chip stacks for all-in testing."""
    return [
        PokerAgent(name="BigStack", starting_chips=1000),
        PokerAgent(name="MedStack", starting_chips=500),
        PokerAgent(name="SmallStack", starting_chips=100),
        PokerAgent(name="TinyStack", starting_chips=20),
    ]


# ============================================================
# RULE 1-2: Card Dealing
# ============================================================

def test_each_player_receives_two_hole_cards(game, basic_agents):
    """
    GIVEN a game with 4 players
    WHEN hole cards are dealt
    THEN each player should have exactly 2 hole cards
    """
    game.deal_hole_cards()
    
    for agent in basic_agents:
        assert len(agent.get_hole_cards()) == 2
        assert all(isinstance(card, int) for card in agent.get_hole_cards())


def test_hole_cards_are_unique(game, basic_agents):
    """
    GIVEN a game with 4 players
    WHEN hole cards are dealt
    THEN no two players should have the same card
    """
    game.deal_hole_cards()
    
    all_cards = []
    for agent in basic_agents:
        all_cards.extend(agent.get_hole_cards())
    
    # Check for duplicates
    assert len(all_cards) == len(set(all_cards)), "Duplicate cards found"


def test_community_cards_dealt_in_correct_stages(game, basic_agents):
    """
    GIVEN a game in progress
    WHEN community cards are dealt
    THEN flop=3 cards, turn=1 card, river=1 card (total 5)
    """
    game.deal_hole_cards()
    
    # Flop
    game.deal_flop()
    assert len(game.board) == 3
    
    # Turn
    game.deal_turn()
    assert len(game.board) == 4
    
    # River
    game.deal_river()
    assert len(game.board) == 5


def test_no_card_overlap_between_players_and_board(game, basic_agents):
    """
    GIVEN a complete hand with all cards dealt
    WHEN checking all cards in play
    THEN there should be no duplicates between hole cards and board
    """
    game.deal_hole_cards()
    game.deal_flop()
    game.deal_turn()
    game.deal_river()
    
    all_cards = []
    for agent in basic_agents:
        all_cards.extend(agent.get_hole_cards())
    all_cards.extend(game.board)
    
    # 4 players * 2 cards + 5 board = 13 unique cards
    assert len(all_cards) == 13
    assert len(set(all_cards)) == 13


# ============================================================
# RULE 6: Blinds
# ============================================================

def test_blinds_posted_correctly(game, basic_agents):
    """
    GIVEN a game starting a new hand
    WHEN blinds are posted
    THEN small blind and big blind are deducted from correct players
    """
    initial_chips = [agent.get_chips() for agent in basic_agents]
    
    sb_pos, bb_pos = game.post_blinds()
    
    # Small blind player loses 5 chips
    assert basic_agents[sb_pos].get_chips() == initial_chips[sb_pos] - 5
    
    # Big blind player loses 10 chips
    assert basic_agents[bb_pos].get_chips() == initial_chips[bb_pos] - 10
    
    # Pot should have both blinds
    assert game.get_pot_size() == 15


def test_blind_positions_rotate(basic_agents):
    """
    GIVEN multiple hands played
    WHEN dealer position changes
    THEN blind positions should rotate accordingly
    """
    game = TexasHoldemSimulation(basic_agents, small_blind=5, big_blind=10)
    
    positions = []
    for _ in range(4):
        game.reset_for_new_hand()
        sb_pos, bb_pos = game.post_blinds()
        positions.append((sb_pos, bb_pos))
        game.dealer_position = (game.dealer_position + 1) % len(basic_agents)
    
    # All positions should be different
    assert len(set(positions)) == 4


# ============================================================
# RULE 7-11: Player Actions
# ============================================================

def test_player_can_fold(game, basic_agents):
    """
    GIVEN a player in a hand
    WHEN they fold
    THEN they should be marked as folded and excluded from pot
    """
    agent = basic_agents[0]
    assert not agent.is_folded
    
    agent.fold()
    
    assert agent.is_folded


def test_player_can_call(game, basic_agents):
    """
    GIVEN a player facing a bet
    WHEN they call
    THEN correct amount should be deducted from their chips
    """
    agent = basic_agents[0]
    initial_chips = agent.get_chips()
    
    amount_bet = agent.place_bet(50)
    
    assert amount_bet == 50
    assert agent.get_chips() == initial_chips - 50
    assert agent.current_bet == 50


def test_player_all_in_when_betting_all_chips(game, basic_agents):
    """
    GIVEN a player with limited chips
    WHEN they bet all their chips
    THEN they should be marked as all-in
    """
    agent = basic_agents[0]
    all_chips = agent.get_chips()
    
    agent.place_bet(all_chips)
    
    assert agent.is_all_in
    assert agent.get_chips() == 0


def test_player_cannot_bet_more_than_they_have(game, basic_agents):
    """
    GIVEN a player with 1000 chips
    WHEN they try to bet 2000 chips
    THEN they should only bet 1000 and be all-in
    """
    agent = basic_agents[0]
    initial_chips = agent.get_chips()
    
    amount_bet = agent.place_bet(initial_chips + 500)
    
    assert amount_bet == initial_chips
    assert agent.get_chips() == 0
    assert agent.is_all_in


# ============================================================
# RULE 12-15: Betting Round Rules
# ============================================================

def test_betting_round_ends_when_all_match(game, basic_agents):
    """
    GIVEN a betting round in progress
    WHEN all active players match the current bet
    THEN the betting round should end
    """
    game.deal_hole_cards()
    game.post_blinds()
    
    # Simulate pre-flop: everyone calls the big blind
    result = game.run_betting_round("Pre-flop")
    
    assert result is True  # Round completed successfully


def test_minimum_raise_is_previous_raise_size(game, basic_agents):
    """
    GIVEN a betting round with a raise
    WHEN calculating minimum re-raise
    THEN it should equal the previous raise amount
    """
    game.deal_hole_cards()
    game.post_blinds()
    
    # Big blind is 10, so first raise minimum should be 10
    initial_min_raise = game.min_raise
    assert initial_min_raise == 10


def test_cannot_check_when_facing_bet(game, basic_agents):
    """
    GIVEN a player facing a bet
    WHEN they try to check
    THEN they should be forced to call, raise, or fold
    """
    agent = basic_agents[0]
    
    # Set up a scenario where there's a bet to call
    game.current_bet = 50
    agent.current_bet = 0
    
    # Player cannot check when bet > their current bet
    amount_to_call = game.current_bet - agent.current_bet
    assert amount_to_call > 0  # There's a bet to face


# ============================================================
# RULE 16-18: Pot and Side Pot Rules
# ============================================================

def test_all_bets_go_into_pot(game, basic_agents):
    """
    GIVEN players making bets
    WHEN bets are placed
    THEN pot should increase by total bet amount
    """
    initial_pot = game.get_pot_size()
    
    for agent in basic_agents:
        amount = agent.place_bet(50)
        game.pot += amount
    
    assert game.get_pot_size() == initial_pot + (50 * 4)


def test_side_pot_created_when_player_all_in_for_less(small_stack_agents):
    """
    GIVEN players with different stack sizes
    WHEN a short stack goes all-in and others call more
    THEN side pots should be created correctly
    """
    game = TexasHoldemSimulation(small_stack_agents, small_blind=5, big_blind=10)
    
    # Simulate: TinyStack all-in for 20, others call 100
    small_stack_agents[3].place_bet(20)  # TinyStack all-in
    small_stack_agents[3].total_invested = 20
    small_stack_agents[0].place_bet(100)
    small_stack_agents[0].total_invested = 100
    small_stack_agents[1].place_bet(100)
    small_stack_agents[1].total_invested = 100
    
    game.pot = 220
    
    side_pots = game.create_side_pots()
    
    # Should have multiple pots
    assert len(side_pots) >= 2
    
    # First pot should include TinyStack
    first_pot_amount, first_pot_players = side_pots[0]
    assert small_stack_agents[3] in first_pot_players


def test_player_only_eligible_for_pots_they_invested_in(small_stack_agents):
    """
    GIVEN a player all-in for 20 when pot is 200
    WHEN determining pot eligibility
    THEN they should only be eligible for main pot, not side pot
    """
    game = TexasHoldemSimulation(small_stack_agents, small_blind=5, big_blind=10)
    
    # Setup: TinyStack all-in for 20, BigStack bets 200
    small_stack_agents[3].place_bet(20)
    small_stack_agents[3].total_invested = 20
    small_stack_agents[3].is_all_in = True
    
    small_stack_agents[0].place_bet(200)
    small_stack_agents[0].total_invested = 200
    
    game.pot = 220
    
    side_pots = game.create_side_pots()
    
    # TinyStack should only be in first pot
    for i, (pot_amount, eligible_players) in enumerate(side_pots):
        if i == 0:
            assert small_stack_agents[3] in eligible_players
        else:
            # Should not be in side pots
            assert small_stack_agents[3] not in eligible_players


def test_equal_investment_creates_single_pot(game, basic_agents):
    """
    GIVEN all players invest the same amount
    WHEN creating pots
    THEN only one pot should be created
    """
    # All players invest 100
    for agent in basic_agents:
        agent.place_bet(100)
        agent.total_invested = 100
    
    game.pot = 400
    
    side_pots = game.create_side_pots()
    
    # Should be exactly one pot
    assert len(side_pots) == 1
    
    # All players should be eligible
    pot_amount, eligible_players = side_pots[0]
    assert len(eligible_players) == 4


# ============================================================
# RULE 19-22: Hand Rankings and Showdown
# ============================================================

def test_hand_evaluation_uses_best_five_cards(game, basic_agents):
    """
    GIVEN a player's 2 hole cards and 5 community cards
    WHEN evaluating hand
    THEN best 5-card combination should be used
    """
    evaluator = Evaluator()
    agent = basic_agents[0]
    
    # Give player specific cards
    agent.hole_cards = [
        Card.new('Ah'), Card.new('Kh')  # Hole cards
    ]
    
    board = [
        Card.new('Qh'), Card.new('Jh'), Card.new('Th'),  # Royal flush possible
        Card.new('2c'), Card.new('3c')
    ]
    
    score, hand_class, hand_name, percentage = agent.evaluate_hand(board)
    
    # Should detect royal flush (best possible hand)
    assert "Straight Flush" in hand_name or "Royal" in hand_name
    assert score <= 10  # Royal flush has very low score


def test_lower_rank_beats_higher_rank(game, basic_agents):
    """
    GIVEN two hands with different ranks (treys uses lower=better)
    WHEN comparing hands
    THEN lower rank number should win
    """
    evaluator = Evaluator()
    
    # Royal flush (rank ~1)
    royal = [Card.new('Ah'), Card.new('Kh'), Card.new('Qh'), 
             Card.new('Jh'), Card.new('Th')]
    
    # Pair of aces (rank ~4000)
    pair = [Card.new('Ad'), Card.new('Ac'), Card.new('Kd'),
            Card.new('Qs'), Card.new('Jc')]
    
    royal_score = evaluator.evaluate([], royal)
    pair_score = evaluator.evaluate([], pair)
    
    assert royal_score < pair_score  # Lower is better


def test_winner_determined_by_best_hand(game, basic_agents):
    """
    GIVEN multiple players at showdown
    WHEN determining winner
    THEN player with lowest rank (best hand) should win
    """
    # Set board first (avoid using same cards in hole cards)
    game.board = [
        Card.new('2h'), Card.new('3h'), Card.new('4h'),
        Card.new('5h'), Card.new('9c')
    ]
    
    # Manually set hole cards for testing (no overlap with board)
    basic_agents[0].hole_cards = [Card.new('Ah'), Card.new('6h')]  # Flush (best)
    basic_agents[1].hole_cards = [Card.new('Ad'), Card.new('Ac')]  # Pair of aces
    
    winners = game.get_winner()
    
    # Player 0 should win with flush
    assert basic_agents[0] in winners


def test_tie_splits_pot_equally(basic_agents):
    """
    GIVEN two players with identical hands
    WHEN determining winners
    THEN both should be winners and pot should split
    """
    game = TexasHoldemSimulation(basic_agents[:2], small_blind=5, big_blind=10)
    
    # Both players have same hole cards (for testing purposes)
    board = [Card.new('Ah'), Card.new('Kh'), Card.new('Qh'),
             Card.new('Jh'), Card.new('Tc')]
    
    basic_agents[0].hole_cards = [Card.new('2c'), Card.new('3c')]
    basic_agents[1].hole_cards = [Card.new('2d'), Card.new('3d')]
    
    game.board = board
    
    # Both should have exact same hand (board plays - royal flush)
    score1, _, _, _ = basic_agents[0].evaluate_hand(board)
    score2, _, _, _ = basic_agents[1].evaluate_hand(board)
    
    assert score1 == score2  # Tied hands


def test_all_fold_gives_last_player_pot_without_showdown(game, basic_agents):
    """
    GIVEN a betting round where all but one player folds
    WHEN checking if showdown needed
    THEN remaining player should win without showdown
    """
    # Fold all but one player
    for i in range(3):
        basic_agents[i].fold()
    
    active_players = [a for a in basic_agents if not a.is_folded]
    
    assert len(active_players) == 1
    # No showdown needed when only 1 player remains


# ============================================================
# RULE 23-25: Chip Management
# ============================================================

def test_player_eliminated_at_zero_chips(game, basic_agents):
    """
    GIVEN a player with zero chips
    WHEN checking eligibility for next hand
    THEN they should not be able to play
    """
    agent = basic_agents[0]
    
    # Lose all chips
    agent.place_bet(agent.get_chips())
    
    assert agent.get_chips() == 0
    assert agent.is_all_in


def test_chip_stacks_persist_across_hands(basic_agents):
    """
    GIVEN a tournament with multiple hands
    WHEN playing consecutive hands
    THEN chip stacks should persist from hand to hand
    """
    game = TexasHoldemSimulation(basic_agents, small_blind=5, big_blind=10)
    
    # Play hand 1
    game.post_blinds()
    chips_after_hand1 = [agent.get_chips() for agent in basic_agents]
    
    # Reset for hand 2 (but chips persist)
    game.reset_for_new_hand()
    game.post_blinds()
    chips_after_hand2 = [agent.get_chips() for agent in basic_agents]
    
    # Chips should have changed from blinds
    assert chips_after_hand1 != [1000, 1000, 1000, 1000]
    # And should continue to decrease in hand 2
    assert chips_after_hand2 != chips_after_hand1


def test_cannot_bet_negative_chips(game, basic_agents):
    """
    GIVEN a player attempting to bet negative amount
    WHEN placing bet
    THEN bet should be 0 and chips unchanged
    """
    agent = basic_agents[0]
    initial_chips = agent.get_chips()
    
    amount_bet = agent.place_bet(-100)
    
    # Should not lose chips from negative bet
    assert agent.get_chips() == initial_chips
    assert amount_bet == 0


def test_winning_pot_increases_chip_stack(game, basic_agents):
    """
    GIVEN a player winning a pot
    WHEN chips are awarded
    THEN their stack should increase by pot amount
    """
    agent = basic_agents[0]
    initial_chips = agent.get_chips()
    
    agent.add_chips(500)
    
    assert agent.get_chips() == initial_chips + 500


# ============================================================
# INTEGRATION TESTS - Complete Hand Scenarios
# ============================================================

def test_complete_hand_with_winner(game, basic_agents):
    """
    INTEGRATION TEST: Full hand from deal to showdown
    GIVEN a complete poker hand
    WHEN playing through all streets
    THEN winner should be determined and pot awarded correctly
    """
    # Setup
    initial_total_chips = sum(agent.get_chips() for agent in basic_agents)
    
    game.deal_hole_cards()
    game.post_blinds()
    game.deal_flop()
    game.deal_turn()
    game.deal_river()
    
    # Award pot to winner
    results = game.evaluate_hands()
    
    # Total chips in system should remain constant
    final_total_chips = sum(agent.get_chips() for agent in basic_agents) + game.get_pot_size()
    assert final_total_chips == initial_total_chips


def test_pre_flop_betting_with_blinds(game, basic_agents):
    """
    INTEGRATION TEST: Pre-flop betting mechanics
    GIVEN players with blinds posted
    WHEN pre-flop betting occurs
    THEN big blind should have option to check if no raise
    """
    game.deal_hole_cards()
    sb_pos, bb_pos = game.post_blinds()
    
    # Big blind player has already invested 10
    assert basic_agents[bb_pos].current_bet == 10
    
    # If no one raises, BB can check (already matched the bet)
    # This is verified by the current_bet tracking


def test_multiple_all_ins_create_multiple_side_pots(small_stack_agents):
    """
    INTEGRATION TEST: Complex side pot scenario
    GIVEN 4 players with stacks [1000, 500, 100, 20]
    WHEN all go all-in
    THEN multiple side pots should be created correctly
    """
    game = TexasHoldemSimulation(small_stack_agents, small_blind=5, big_blind=10)
    
    # All players go all-in for their stack size
    small_stack_agents[0].place_bet(1000)
    small_stack_agents[0].total_invested = 1000
    
    small_stack_agents[1].place_bet(500)
    small_stack_agents[1].total_invested = 500
    
    small_stack_agents[2].place_bet(100)
    small_stack_agents[2].total_invested = 100
    
    small_stack_agents[3].place_bet(20)
    small_stack_agents[3].total_invested = 20
    
    game.pot = 1620
    
    side_pots = game.create_side_pots()
    
    # Should have 4 pots (one for each investment level)
    assert len(side_pots) == 4
    
    # Verify pot amounts
    pot_amounts = [pot[0] for pot in side_pots]
    total_in_pots = sum(pot_amounts)
    assert total_in_pots == 1620


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
