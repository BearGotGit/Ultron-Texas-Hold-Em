# train_ppo_holdem.py
import gymnasium as gym
from stable_baselines3 import PPO

from training.holdem_env import TexasHoldemEnv

def main():
    env = TexasHoldemEnv(starting_chips=1000)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        # You can tune these:
        n_steps=256,
        batch_size=256,
        learning_rate=3e-4,
    )

#    model.learn(total_timesteps=100_000)
    model.learn(total_timesteps=1)

    # Quick test run
    obs, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()

    print("Episode total reward:", total_reward)
    #model.save("poker_ppo_model") #-> Uncomment this line to save data

if __name__ == "__main__":
    main()
 