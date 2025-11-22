# 🃏 Texas-Hold-Em Ultron Take Over

Texas Hold'em is a popular poker game played in all casinos.  
For our **4444 class project**, our group is building a **competitive Machine Learning poker bot** capable of playing Texas Hold'em at a high level — and hopefully better than you or I!

## 👥 Team Members
Anthony • Berend • Daniel • Dina • Eby • Aaron

![unnamed](https://github.com/user-attachments/assets/fab873ac-b36c-495e-88cb-53aca8cb5ca3)

---

# 🎯 Project Goal
Build an AI agent that can:
- Understand game state  
- Evaluate hand strength & equity  
- Predict opponent ranges  
- Make decisions (fold/call/raise) in real time  
- Compete against other teams' bots in a class tournament  

Our approach uses **Monte Carlo simulation**, **supervised learning**, and **neural networks** to train a competitive Texas Hold'em model.

---

# Project Structure
```
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

# Running the Project
1️⃣ Generate Training Data
Monte Carlo simulation:
python simulation/generate_dataset.py

2️⃣ Train the Model
python training/train_model.py

3️⃣ Evaluate Model Performance
python training/evaluate_model.py

4️⃣ Run the Poker Agent
python gameplay/ai_agent.py

---

# ⚙️ Installation

```bash
git clone https://github.com/BearGotGit/Ultron-Texas-Hold-Em
cd Ultron-Texas-Hold-Em
pip install -r requirements.txt
```
