# RL Training Bug Report

## Executive Summary

After thorough investigation of the `rl-foundation` branch, I identified **two critical bugs** that explain why the RL model went from 20% success to 0% success after 2 million training iterations.

The root cause is that **player chip stacks are never reset between episodes**, causing the hero to go bankrupt and stay bankrupt for the remainder of training. Combined with a flawed reward calculation, this creates a catastrophic training failure.

---

## Bug #1: Player Money Not Reset Between Episodes (Critical)

### Location
- `simulation/poker_env.py` - `reset()` method (lines 335-337)
- `agents/poker_player.py` - `reset()` method (lines 174-180)

### Description
When the environment's `reset()` method is called to start a new episode, it calls `player.reset()` for each player. However, `player.reset()` is designed for tournament-style play where money persists across hands - it does NOT reset the player's chip stack.

```python
# agents/poker_player.py
def reset(self):
    """Reset for a new hand (keeps money and id)."""  # <-- Note: money is NOT reset
    self.folded = False
    self.all_in = False
    self.bet = 0
    self.total_invested = 0
    self._private_cards = []
```

### Impact
1. **Hero goes bankrupt over time**: When training over millions of iterations, the hero will eventually lose all their chips
2. **Hero stays at 0 chips forever**: Once bankrupt, the hero can never recover
3. **Training becomes meaningless**: A hero with 0 chips cannot post blinds or make meaningful bets
4. **Model learns helplessness**: The model learns that folding always happens (when you have no chips)

### Evidence
```python
# Before first episode:
#   Env 0 Player 0: 1000 chips

# After first rollout (multiple episodes):
#   Env 0 Player 0: 819 chips

# After second rollout:
#   Env 0 Player 0: 5 chips
```

---

## Bug #2: Incorrect Reward Calculation (Critical)

### Location
- `simulation/poker_env.py` - `_calculate_reward()` method (lines 632-645)

### Description
The reward calculation compares the hero's final chip count to `config.starting_stack` instead of tracking how many chips the hero had at the START of the current hand.

```python
def _calculate_reward(self) -> float:
    """Calculate reward for the hero."""
    if not self.hand_complete:
        return 0.0
    
    hero = self.players[self.hero_idx]
    
    # BUG: Uses starting_stack (1000) instead of hero's chips at hand start
    initial_chips = self.config.starting_stack
    reward = (hero.money - initial_chips) / initial_chips
    
    return reward
```

### Impact
1. **Wrong reward signals**: If hero has 500 chips and wins 100, reward = (600-1000)/1000 = -0.4 (should be positive!)
2. **Punishment for winning**: The model is punished for winning small pots when already below starting stack
3. **Gradient confusion**: The model receives incorrect learning signals throughout training

### Example Scenarios
| Hero's Chips Before | Hero's Chips After | Actual Change | Calculated Reward | Expected Reward |
|---------------------|-------------------|---------------|-------------------|-----------------|
| 1000 | 1100 | +100 | +0.10 | +0.10 ✓ |
| 500 | 600 | +100 | -0.40 | +0.20 ✗ |
| 500 | 400 | -100 | -0.60 | -0.20 ✗ |
| 200 | 300 | +100 | -0.70 | +0.50 ✗ |
| 0 | 0 | 0 | -1.00 | 0.00 ✗ |

---

## Why This Leads to 0% Success Rate

### Training Progression
1. **Early Training (0-500K iterations)**: Hero has chips, learns somewhat correctly, achieves ~20% success
2. **Mid Training**: Hero experiences variance, loses chips over time
3. **Late Training (1.5M+ iterations)**: Hero is bankrupt (0 chips)
4. **Remaining Training**: Hero cannot play, all rewards are -1.0, model learns to always fold

### The Death Spiral
1. Hero loses a few hands → chip stack decreases
2. Reward calculation becomes increasingly negative (comparing to 1000 starting)
3. Model gets confused signals → makes suboptimal decisions
4. More losses → more chip decrease → worse rewards
5. Eventually hits 0 chips → can't play → 0% success rate

---

## Recommended Fixes

### Fix #1: Reset Player Money in Environment
```python
# simulation/poker_env.py - In reset() method
def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    
    # Reset deck
    self.deck = Deck()
    
    # Reset players INCLUDING their money
    for player in self.players:
        player.reset()
        player.money = self.config.starting_stack  # <-- ADD THIS LINE
    
    # ... rest of reset code
```

### Fix #2: Track Chips at Hand Start for Reward
```python
# simulation/poker_env.py - Track initial chips
def reset(self, seed=None, options=None):
    # ... existing code ...
    
    # Reset players
    for player in self.players:
        player.reset()
        player.money = self.config.starting_stack
    
    # Track hero's starting chips for reward calculation
    self._hero_initial_chips = self.players[self.hero_idx].money  # <-- ADD THIS
    
    # ... rest of reset code

def _calculate_reward(self) -> float:
    """Calculate reward for the hero."""
    if not self.hand_complete:
        return 0.0
    
    hero = self.players[self.hero_idx]
    
    # Use hero's chips at start of hand, not config starting stack
    initial_chips = self._hero_initial_chips  # <-- CHANGE THIS
    if initial_chips == 0:
        return 0.0
    
    reward = (hero.money - initial_chips) / initial_chips
    return reward
```

### Alternative Fix: Create New Players Each Episode
```python
# training/train_rl_model.py - In _collect_rollouts or environment creation
# Instead of reusing players, create fresh players each episode
```

---

## Additional Observations

### Minor Issues Found
1. **Episode length is very short** (~1.2 steps average): This suggests hands end quickly, possibly due to aggressive all-in behavior or opponent folding patterns

2. **No exploration decay**: The entropy coefficient is fixed at 0.01, which may not provide enough exploration early in training

3. **Potential MPS fallback issues**: The code disables MPS due to Beta distribution support, but this may cause unexpected behavior if not all operations properly fall back

### Suggestions for Better Training
1. Add **reward shaping** to give intermediate rewards for good plays (not just final outcome)
2. Implement **curriculum learning** - start against weak opponents and gradually increase difficulty
3. Add **self-play** so the model plays against copies of itself
4. Monitor **entropy** during training to ensure sufficient exploration

---

## Verification Commands

To verify the bugs yourself, run these commands on the `rl-foundation` branch:

```bash
# Test 1: Show that player money isn't reset
python -c "
from simulation.poker_env import PokerEnv, PokerEnvConfig
from agents.monte_carlo_agent import MonteCarloAgent
import numpy as np

config = PokerEnvConfig()
hero = MonteCarloAgent('Hero', 1000, 50)
opponent = MonteCarloAgent('Opp', 1000, 50)
env = PokerEnv([hero, opponent], config, hero_idx=0)

for ep in range(5):
    obs, _ = env.reset()
    print(f'Episode {ep+1}: Hero starts with {hero.money} chips')
    done = False
    while not done:
        obs, reward, term, trunc, _ = env.step(np.array([0.2, 0.5]))
        done = term or trunc
    print(f'  Ended with {hero.money} chips, reward={reward:.3f}')
"
```

---

## Conclusion

The RL training failure from 20% to 0% success rate is caused by two critical bugs:

1. **Player money not being reset** - causes hero to go bankrupt permanently
2. **Incorrect reward calculation** - gives wrong signals when hero's chips differ from starting stack

Both bugs compound to create a "death spiral" where the model eventually gets stuck with 0 chips and cannot learn anything meaningful. Fixing these two issues should restore proper training behavior.

