# 🃏 Texas-Hold-Em Ultron Take Over

Texas Hold'em is a popular poker game played in all casinos.  
For our **4444 class project**, our group is building a **competitive Machine Learning poker bot** capable of playing Texas Hold'em at a high level — and hopefully better than you or I!

## 👥 Team Members

Anthony • Berend • Daniel • Dina • Eby • Aaron

![unnamed](https://github.com/user-attachments/assets/fab873ac-b36c-495e-88cb-53aca8cb5ca3)

---

## 🎯 Project Goal

Build an AI agent that can:

- Understand game state  
- Evaluate hand strength & equity  
- Predict opponent ranges  
- Make decisions (fold/call/raise) in real time  
- Compete against other teams' bots in a class tournament  

Our approach uses **Monte Carlo simulation**, **supervised learning**, and **neural networks** to train a competitive Texas Hold'em model.

---

## Project Structure

```bash
Ultron-Texas-Hold-Em/
│
├── data/
│ ├── raw/
│ ├── processed/
│ └── datasets.md
│
├── models/
│ ├── saved/
│ └── architecture/
│
├── simulation/
│ ├── generate_dataset.py
│ ├── card_utils.py
│ └── poker_simulator.py
│
├── training/
│ ├── train_model.py
│ ├── evaluate_model.py
│ └── losses.py
│
├── gameplay/
│ ├── ai_agent.py
│ ├── opponent_models.py
│ └── game_state.py
│
├── tools/
│ ├── visualize.py
│ ├── utils.py
│ └── config.py
│
├── tests/
│ ├── test_simulation.py
│ ├── test_model.py
│ └── test_gameplay.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/BearGotGit/Ultron-Texas-Hold-Em
cd Ultron-Texas-Hold-Em

python -m venv .venv
# activate windows or mac way, then...
pip install -r requirements.txt
```

---

## Running the Project

### 🎮 Play Against the Bot
```bash
python main.py
```

### 🧠 Train the RL Agent (PPO)

Train a poker bot using Proximal Policy Optimization:

```bash
# Quick test run (10k timesteps, ~30 seconds)
PYTHONPATH=. python training/train_rl_model.py --total-timesteps 10000

# Full training run (1M timesteps, ~1-2 hours on CPU)
PYTHONPATH=. python training/train_rl_model.py --total-timesteps 1000000

# Custom training options
PYTHONPATH=. python training/train_rl_model.py \
    --total-timesteps 500000 \
    --num-players 2 \
    --lr 0.0003 \
    --hidden-dim 256 \
    --run-name my_training_run

# Resume from checkpoint
PYTHONPATH=. python training/train_rl_model.py --resume checkpoints/checkpoint_100.pt
```

**Training Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--total-timesteps` | 100,000 | Total training decisions |
| `--num-players` | 2 | Players per table (hero + opponents) |
| `--lr` | 0.0003 | Learning rate |
| `--hidden-dim` | 256 | Neural network hidden layer size |
| `--run-name` | auto | Name for TensorBoard logs |
| `--resume` | None | Checkpoint path to resume from |

**Monitor Training with TensorBoard:**
```bash
tensorboard --logdir runs
```
Then open http://localhost:6006 in your browser.

**Checkpoints:** Saved to `checkpoints/` directory.

---

### 📊 Other Commands

1️⃣ Generate Training Data (Monte Carlo simulation):
```bash
python simulation/generate_dataset.py
```

2️⃣ Train Supervised Model (deprecated, use RL instead):
```bash
python training/train_model.py
```

3️⃣ Evaluate Model Performance:
```bash
python training/evaluate_model.py
```
