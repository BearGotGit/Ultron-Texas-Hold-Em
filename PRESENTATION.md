# 🃏 Ultron Texas Hold'em - Mock Presentation

## Team: Anthony • Berend • Daniel • Dina • Eby • Aaron

---

## 🎯 Slide 1: Project Overview

### What We Built
- **An AI poker bot** that learns to play Texas Hold'em
- Uses **Reinforcement Learning (RL)** to improve through self-play
- Can compete against humans and other bots

### Why It's Cool
- Poker is a game of **incomplete information** (you don't see opponents' cards)
- Requires **strategic decision-making** under uncertainty
- Our bot learns to bluff, fold, call, and raise like a human!

---

## 🔧 Slide 2: Key Approach

### Our Strategy: Three Main Components

1. **Monte Carlo Simulation**
   - Randomly simulates thousands of possible game outcomes
   - Estimates hand strength (equity) based on win probability

2. **Reinforcement Learning (PPO)**
   - The bot learns by playing games against itself
   - Gets rewarded for winning chips, penalized for losing
   - Gradually improves its strategy over time

3. **Neural Network (Brain)**
   - Processes the game state (cards, bets, pot)
   - Outputs decisions: fold probability + bet sizing

---

## 📥 Slide 3: Input - What the Bot "Sees"

### Game State Observation (423 features total)

| Input Type | Description | Size |
|------------|-------------|------|
| **Cards** | Hole cards (2) + Board (5) encoded as one-hot vectors | 371 dims |
| **Hand Features** | Binary flags for hand type (pair, flush, etc.) | 10 dims |
| **Player Info** | Stack sizes, bets, fold/all-in status | 36 dims |
| **Game State** | Pot size, current bet, round stage, positions | 6 dims |

**Key Point:** The bot sees everything a human would see at the table, encoded numerically.

---

## 📤 Slide 4: Output - What the Bot Decides

### Action Space (2 outputs)

1. **Fold Probability (p_fold)**: 0 to 1
   - If > 0.5 → Fold
   - If ≤ 0.5 → Continue to bet decision

2. **Bet Scalar**: 0 to 1
   - ~0.0 → Check/Call (minimum action)
   - ~0.5 → Medium raise
   - ~1.0 → All-in

### Interpreted Actions
- **Fold** - Give up the hand
- **Check** - Pass (no bet required)
- **Call** - Match the current bet
- **Raise** - Increase the bet

---

## ⚙️ Slide 5: Process - How It Works

### Game Flow

```
1. DEAL: Bot receives 2 hole cards
           ↓
2. PRE-FLOP: First betting round
           ↓
3. FLOP: 3 community cards revealed → betting
           ↓
4. TURN: 4th card revealed → betting
           ↓
5. RIVER: 5th card revealed → betting
           ↓
6. SHOWDOWN: Best hand wins the pot!
```

### At Each Decision Point:
1. Observe game state → Encode as 423-dim vector
2. Feed to neural network
3. Get fold probability + bet scalar
4. Execute action (fold/check/call/raise)
5. Receive reward at end of hand

---

## 🧠 Slide 6: Model Architecture

### Neural Network Design

```
INPUT (423 dims)
      ↓
┌─────┴─────────────────┬──────────────────┐
│                       │                  │
▼                       ▼                  ▼
Card Embedding    Hand Embedding    Numeric Embedding
(53→64→64) x7     (10→32→32)        (42→64→64)
= 448 dims        = 32 dims         = 64 dims
│                       │                  │
└───────────┬───────────┴──────────────────┘
            │
            ▼
      Concatenate (544 dims)
            ↓
      Shared Layers (256→256)
            ↓
┌───────────┼───────────┐
│           │           │
▼           ▼           ▼
Fold Head  Bet Head   Value Head
(→128→1)   (→128→2)   (→128→1)
│           │           │
▼           ▼           ▼
p_fold     α,β for    V(s)
(Bernoulli) Beta dist (Critic)
```

**Total Parameters:** ~320,000

---

## 🎓 Slide 7: Training

### Proximal Policy Optimization (PPO)

**What is PPO?**
- A state-of-the-art RL algorithm
- Stable and efficient for continuous action spaces
- Used by OpenAI for training ChatGPT!

### Training Process

| Step | Description |
|------|-------------|
| 1. **Collect Experience** | Play 128 hands, recording states, actions, rewards |
| 2. **Compute Advantages** | Calculate how much better each action was vs. expected |
| 3. **Update Policy** | Adjust neural network weights to favor good actions |
| 4. **Repeat** | Loop until 1M+ timesteps |

### Key Hyperparameters
- Learning Rate: 0.0003
- Discount Factor (γ): 0.99
- Entropy Coefficient: 0.01 (encourages exploration)

---

## 📊 Slide 8: Testing & Evaluation

### Evaluation Metrics

1. **Win Rate**: % of hands where bot profits
2. **Average Profit**: Mean chips won per hand
3. **Episode Length**: How many decisions per hand

### Testing Against Different Opponents

| Opponent Type | Description |
|---------------|-------------|
| **Random Agent** | Makes random decisions |
| **Call Station** | Always calls, never folds |
| **Monte Carlo Agent** | Uses equity calculations |

### How We Know It's Working
- Win rate > 50% against baseline opponents
- Episode length > 1 (not folding immediately)
- Stable learning curves in TensorBoard

---

## 🔬 Slide 9: Technical Highlights

### Smart Design Choices

1. **Log Normalization**
   - Money values (100-10,000) normalized to ~0-10
   - Prevents gradient explosion during training

2. **Separate Embeddings**
   - Cards, hands, and numeric features processed separately
   - Allows specialized learning for each input type

3. **Beta Distribution for Bets**
   - More flexible than simple [0,1] output
   - Can represent different betting styles (aggressive, passive)

4. **Bernoulli Fold Decision**
   - Binary fold/don't fold makes training cleaner
   - Simpler than trying to learn fold as part of bet sizing

---

## 🎮 Slide 10: Demo - Play Against the Bot!

### Running the Interactive Mode

```bash
# Play against trained AI
python main.py

# Or play against specific RL model
python play_vs_rl.py checkpoints/final.pt
```

### What You'll See
- Your cards displayed
- Pot size and bets
- AI makes decisions in real-time
- Hand evaluation at showdown

---

## 📈 Slide 11: Results & Future Work

### Current Status
✅ Functional poker environment  
✅ Working PPO training pipeline  
✅ Monte Carlo equity calculation  
✅ Multiple opponent types for training  
✅ TensorBoard metrics logging  

### Future Improvements
- [ ] Train for more timesteps (10M+)
- [ ] Multi-agent self-play
- [ ] Opponent modeling (predict opponent strategy)
- [ ] Connect to online poker platforms
- [ ] Add position-aware strategy

---

## 🤔 Slide 12: Q&A

### Potential Questions & Answers

**Q: Why reinforcement learning instead of supervised learning?**
> A: We don't have labeled "correct" poker moves. RL discovers optimal strategy through trial and error.

**Q: How long does training take?**
> A: ~1-2 hours for 1M timesteps on CPU. GPU can speed this up significantly.

**Q: Can it beat professional players?**
> A: Our bot is competitive against basic strategies. Beating pros requires billions of hands of training (like Libratus/Pluribus).

**Q: Why use Monte Carlo opponents?**
> A: They provide a reasonable baseline that uses equity calculations, making training more realistic than pure random opponents.

---

## 📚 Slide 13: Key Takeaways

### Summary

1. **Poker is Hard** - Incomplete information + multiple betting rounds = complex strategy

2. **RL is Powerful** - The bot learns without being told what's "good" or "bad"

3. **Architecture Matters** - Separate embeddings + proper normalization = stable training

4. **Testing is Critical** - Multiple opponent types ensure robust strategy

### Thank You! 🎉

---

## Appendix: Quick Reference

### Commands to Remember

```bash
# Quick demo training (30 seconds)
PYTHONPATH=. python training/train_rl_model.py --total-timesteps 10000

# Full training run (1-2 hours)
PYTHONPATH=. python training/train_rl_model.py --total-timesteps 1000000

# Play interactively
python main.py

# Monitor training
tensorboard --logdir runs
```

### Files to Know

| File | Purpose |
|------|---------|
| `training/ppo_model.py` | Neural network architecture |
| `simulation/poker_env.py` | Game environment |
| `training/train_rl_model.py` | Training loop |
| `agents/monte_carlo_agent.py` | Baseline opponent |
