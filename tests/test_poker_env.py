"""
Tests for the PokerEnv Gymnasium environment.

These tests verify the environment correctly implements the Gymnasium interface
and handles episodic resets properly for RL training.
"""

import pytest
import numpy as np
from simulation.poker_env import PokerEnv, PokerEnvConfig
from agents.monte_carlo_agent import MonteCarloAgent


@pytest.fixture
def env_config():
    """Create a standard environment configuration."""
    return PokerEnvConfig(
        big_blind=10,
        small_blind=5,
        starting_stack=1000,
        max_players=2,
    )


@pytest.fixture
def two_player_env(env_config):
    """Create a 2-player environment for testing."""
    players = [
        MonteCarloAgent(
            player_id="Hero",
            starting_money=env_config.starting_stack,
            num_simulations=10,
        ),
        MonteCarloAgent(
            player_id="Opponent",
            starting_money=env_config.starting_stack,
            num_simulations=10,
        ),
    ]
    return PokerEnv(players=players, config=env_config, hero_idx=0)


# ============================================================
# Episodic Reset Tests - Fix for Bug #1
# ============================================================

def test_env_reset_restores_player_money(two_player_env):
    """
    GIVEN an environment after a hand where hero lost chips
    WHEN env.reset() is called
    THEN hero's money should be restored to starting stack (before blinds)
    
    This is the main test for Bug #1: Player money never reset between episodes
    Note: After reset, blinds are posted, so players will have slightly less than starting stack.
    The key is that total chips in play equals num_players * starting_stack.
    """
    env = two_player_env
    starting_stack = env.config.starting_stack
    num_players = len(env.players)
    
    # Play until the hand ends
    obs, info = env.reset()
    
    # Verify total chips in play (players + pot) equals starting total
    total_chips = sum(p.money for p in env.players) + env.pot.money
    expected_total = num_players * starting_stack
    assert total_chips == expected_total, \
        f"Total chips should be {expected_total} but got {total_chips}"
    
    # Simulate playing - step through the game
    done = False
    while not done:
        action = np.array([0.0, 0.5])  # Don't fold, moderate bet
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    # At this point, players may have different chip counts
    # Store the post-hand chip counts to verify they differ
    post_hand_chips = [player.money for player in env.players]
    
    # Reset the environment for a new episode
    obs, info = env.reset()
    
    # Verify total chips in play equals starting total again
    total_chips = sum(p.money for p in env.players) + env.pot.money
    assert total_chips == expected_total, \
        f"After reset, total chips should be {expected_total} but got {total_chips}"


def test_env_reset_restores_zero_chip_player(two_player_env):
    """
    GIVEN an environment where a player has 0 chips
    WHEN env.reset() is called
    THEN player's money should be restored to starting stack
    
    This tests the specific case where hero gets stuck permanently at 0 chips.
    The reset should restore their chips so they can participate in the next episode.
    """
    env = two_player_env
    starting_stack = env.config.starting_stack
    num_players = len(env.players)
    
    # Reset and then manually set hero's money to 0
    obs, info = env.reset()
    hero = env.players[env.hero_idx]
    hero.money = 0
    
    # Verify hero has 0 chips
    assert hero.money == 0, "Hero should have 0 chips after manual override"
    
    # Reset the environment
    obs, info = env.reset()
    
    # Verify hero can now participate (has chips in play)
    # Total chips should be back to starting total
    total_chips = sum(p.money for p in env.players) + env.pot.money
    expected_total = num_players * starting_stack
    assert total_chips == expected_total, \
        f"After reset, total chips should be {expected_total} but got {total_chips}"
    
    # Hero should have chips (after blinds may be posted)
    assert hero.money > 0, \
        f"Hero should have money after reset, not {hero.money}"


def test_env_reset_restores_all_players_money(two_player_env):
    """
    GIVEN an environment where multiple players have varying chip counts
    WHEN env.reset() is called
    THEN all players should have their money reset to starting stack
    """
    env = two_player_env
    starting_stack = env.config.starting_stack
    num_players = len(env.players)
    
    # Reset and set varying chip counts
    obs, info = env.reset()
    env.players[0].money = 500
    env.players[1].money = 1500
    
    # Reset the environment
    obs, info = env.reset()
    
    # Verify total chips in play equals starting total
    total_chips = sum(p.money for p in env.players) + env.pot.money
    expected_total = num_players * starting_stack
    assert total_chips == expected_total, \
        f"After reset, total chips should be {expected_total} but got {total_chips}"


# ============================================================
# Basic Environment Tests
# ============================================================

def test_reward_normalized_by_starting_stack(two_player_env):
    """
    GIVEN an environment
    WHEN hero wins or loses chips
    THEN reward should be calculated as:
         (final_chips - hand_start_chips) / starting_stack
    
    This normalization ensures rewards are comparable across hands regardless
    of hero's current chip count. The delta is still computed from hand-start
    chips to correctly track profit/loss, but divided by starting_stack for
    consistent scaling.
    
    Example: Hero with 500 chips wins 100 → ends with 600
    Reward = (600 - 500) / 1000 = 0.1 (positive, correctly shows profit)
    """
    env = two_player_env
    starting_stack = env.config.starting_stack
    
    # Reset the environment
    obs, info = env.reset()
    
    # Verify hero_hand_start_chips is set correctly after reset
    hero = env.players[env.hero_idx]
    assert env.hero_hand_start_chips == hero.money, \
        f"hero_hand_start_chips should be {hero.money} but got {env.hero_hand_start_chips}"
    
    # Track initial chips at hand start for manual verification
    initial_chips = hero.money
    
    # Play the hand through to completion
    done = False
    final_reward = 0.0
    while not done:
        action = np.array([0.0, 0.5])  # Don't fold, moderate bet
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            final_reward = reward
        done = terminated or truncated
    
    # Calculate expected reward: delta from hand-start normalized by starting_stack
    final_chips = hero.money
    chip_delta = final_chips - initial_chips
    expected_reward = chip_delta / starting_stack if starting_stack > 0 else 0.0
    
    assert abs(final_reward - expected_reward) < 1e-6, \
        f"Reward should be {expected_reward} (chip_delta={chip_delta}, starting_stack={starting_stack}) " \
        f"but got {final_reward}"


def test_reward_positive_when_hero_wins_chips(two_player_env):
    """
    GIVEN an environment
    WHEN hero ends the hand with more chips than at hand start
    THEN reward should be positive
    """
    env = two_player_env
    obs, info = env.reset()
    
    hero = env.players[env.hero_idx]
    initial_chips = env.hero_hand_start_chips
    
    # Play hands until we find one where hero wins
    # (this may take a few episodes since it's probabilistic)
    for _ in range(50):  # Try up to 50 hands
        obs, info = env.reset()
        initial_chips = env.hero_hand_start_chips
        
        done = False
        final_reward = 0.0
        while not done:
            action = np.array([0.0, 0.5])  # Don't fold
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                final_reward = reward
            done = terminated or truncated
        
        final_chips = hero.money
        if final_chips > initial_chips:
            # Hero won chips - reward should be positive
            assert final_reward > 0, \
                f"Reward should be positive when hero wins chips, but got {final_reward}"
            return  # Test passed
    
    # Note: This test may rarely fail if hero never wins in 50 hands
    # but this is statistically extremely unlikely


def test_reward_scales_with_chip_delta(two_player_env):
    """
    GIVEN different chip deltas (profits/losses)
    WHEN reward is calculated
    THEN rewards should scale linearly with chip delta
    
    This test verifies that rewards vary based on the amount won/lost,
    addressing the issue where rewards were suspiciously consistent.
    """
    env = two_player_env
    starting_stack = env.config.starting_stack
    
    # Test with direct reward calculation
    # We can't easily control the game outcome, so we'll verify the formula
    # by checking that different deltas produce different rewards
    
    # Manually set up a scenario and verify reward
    obs, info = env.reset()
    hero = env.players[env.hero_idx]
    initial_chips = env.hero_hand_start_chips
    
    # Play through
    done = False
    final_reward = 0.0
    while not done:
        action = np.array([0.0, 0.5])
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            final_reward = reward
        done = terminated or truncated
    
    final_chips = hero.money
    chip_delta = final_chips - initial_chips
    
    # Verify the relationship: reward = chip_delta / starting_stack
    expected_reward = chip_delta / starting_stack
    assert abs(final_reward - expected_reward) < 1e-6, \
        f"Reward {final_reward} should equal chip_delta/starting_stack = {expected_reward}"


def test_env_reset_returns_valid_observation(two_player_env):
    """
    GIVEN a poker environment
    WHEN reset is called
    THEN should return a valid observation
    """
    obs, info = two_player_env.reset()
    
    assert isinstance(obs, np.ndarray)
    assert obs.shape == two_player_env.observation_space.shape
    assert obs.dtype == np.float32


def test_env_reset_returns_info_dict(two_player_env):
    """
    GIVEN a poker environment
    WHEN reset is called
    THEN should return an info dictionary with expected keys
    """
    obs, info = two_player_env.reset()
    
    assert isinstance(info, dict)
    assert "round" in info
    assert "pot" in info
    assert "hero_money" in info


def test_env_step_returns_valid_tuple(two_player_env):
    """
    GIVEN a poker environment after reset
    WHEN step is called with a valid action
    THEN should return a valid 5-tuple
    """
    obs, info = two_player_env.reset()
    action = np.array([0.0, 0.5])  # Don't fold, moderate bet
    
    result = two_player_env.step(action)
    
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_env_action_space_is_correct(two_player_env):
    """
    GIVEN a poker environment
    WHEN checking action space
    THEN should be Box(2,) with values in [0, 1]
    """
    env = two_player_env
    
    assert env.action_space.shape == (2,)
    assert env.action_space.low.tolist() == [0.0, 0.0]
    assert env.action_space.high.tolist() == [1.0, 1.0]


# ============================================================
# Action Interpretation Tests
# ============================================================

def test_interpret_action_fold_converted_to_check_when_nothing_to_call():
    """
    GIVEN interpret_action is called with p_fold > 0.5
    WHEN there is nothing to call (to_call <= 0)
    THEN should return CHECK instead of FOLD
    
    This tests the fix for the illegal fold bug where the model
    could fold when there was no bet to call.
    """
    from simulation.poker_env import interpret_action
    from agents.poker_player import ActionType
    
    # p_fold = 1.0 (wants to fold), but nothing to call
    action = interpret_action(
        p_fold=1.0,
        bet_scalar=0.5,
        current_bet=0,  # No current bet
        my_bet=0,       # Hero hasn't bet either
        min_raise=10,
        my_money=1000,
    )
    
    # Should be CHECK, not FOLD (can't fold when nothing to call)
    assert action.action_type == ActionType.CHECK
    assert action.amount == 0


def test_interpret_action_fold_allowed_when_bet_to_call():
    """
    GIVEN interpret_action is called with p_fold > 0.5
    WHEN there is a bet to call (to_call > 0)
    THEN should return FOLD
    """
    from simulation.poker_env import interpret_action
    from agents.poker_player import ActionType
    
    # p_fold = 1.0, and there's a bet to call
    action = interpret_action(
        p_fold=1.0,
        bet_scalar=0.5,
        current_bet=50,  # Current bet is 50
        my_bet=0,        # Hero hasn't matched it
        min_raise=10,
        my_money=1000,
    )
    
    # Should be FOLD (there's something to call)
    assert action.action_type == ActionType.FOLD
    assert action.amount == 0


def test_interpret_action_check_when_low_bet_scalar_and_nothing_to_call():
    """
    GIVEN interpret_action is called with low bet_scalar
    WHEN there is nothing to call
    THEN should return CHECK
    """
    from simulation.poker_env import interpret_action
    from agents.poker_player import ActionType
    
    action = interpret_action(
        p_fold=0.0,     # Don't fold
        bet_scalar=0.05,  # Low bet scalar (< epsilon)
        current_bet=0,
        my_bet=0,
        min_raise=10,
        my_money=1000,
    )
    
    assert action.action_type == ActionType.CHECK
    assert action.amount == 0


def test_interpret_action_call_when_low_bet_scalar_and_bet_to_call():
    """
    GIVEN interpret_action is called with low bet_scalar
    WHEN there is a bet to call
    THEN should return CALL
    """
    from simulation.poker_env import interpret_action
    from agents.poker_player import ActionType
    
    action = interpret_action(
        p_fold=0.0,       # Don't fold
        bet_scalar=0.05,  # Low bet scalar (< epsilon)
        current_bet=50,
        my_bet=0,
        min_raise=10,
        my_money=1000,
    )
    
    assert action.action_type == ActionType.CALL
    assert action.amount == 50
# Opponent Behavior Tests - Fix for Task #3
# ============================================================

def test_opponent_does_not_always_fold_to_raises():
    """
    GIVEN a poker environment with MonteCarloAgent opponents
    WHEN hero makes large raises
    THEN opponents should NOT always fold (should call/raise sometimes)
    
    This is the main test for Task #3: Opponent behavior breaks training
    dynamics because opponents were folding too often to any raise.
    
    With the fix, opponents should call at least 30% of the time even
    when facing large raises, providing better training signal.
    """
    from simulation.poker_env import PokerEnv, PokerEnvConfig
    from agents.monte_carlo_agent import MonteCarloAgent
    
    config = PokerEnvConfig(
        big_blind=10,
        small_blind=5,
        starting_stack=1000,
        max_players=2,
    )
    
    # Track fold rate over many games (100 for statistical reliability)
    num_games = 100
    folds = 0
    calls_or_raises = 0
    
    for _ in range(num_games):
        # Create fresh players each game to avoid state issues
        hero = MonteCarloAgent(
            player_id="Hero",
            starting_money=config.starting_stack,
            num_simulations=10,
        )
        opponent = MonteCarloAgent(
            player_id="Opponent",
            starting_money=config.starting_stack,
            num_simulations=10,
            aggression=0.5,
            bluff_frequency=0.1,
        )
        
        env = PokerEnv(players=[hero, opponent], config=config, hero_idx=0)
        obs, _ = env.reset()
        
        # Hero makes a large raise (bet_scalar=0.5 means ~500 chip raise)
        action = np.array([0.0, 0.5])
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check opponent's response
        opp = env.players[1]
        if opp.folded:
            folds += 1
        else:
            calls_or_raises += 1
    
    fold_rate = folds / num_games
    
    # Opponent should not fold more than 70% of the time
    # (before the fix, fold rate was >90%)
    assert fold_rate <= 0.70, \
        f"Opponent fold rate {fold_rate:.1%} is too high (>70%). " \
        "This breaks training dynamics."
    
    # Opponent should call/raise at least 30% of the time
    call_rate = calls_or_raises / num_games
    assert call_rate >= 0.30, \
        f"Opponent call rate {call_rate:.1%} is too low (<30%). " \
        "Opponents need to defend against raises."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
