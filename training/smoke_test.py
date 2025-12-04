"""Smoke test: instantiate PPOTrainer, collect a tiny rollout, save to rollouts_test/"""

from training.train_rl_model import PPOConfig, PPOTrainer

# Small config for smoke test
config = PPOConfig(
    num_envs=1,
    num_players=2,
    total_timesteps=100,
    num_steps=4,
    save_rollouts=True,
    rollout_dir="rollouts_test",
    run_name="smoke_test",
)

trainer = PPOTrainer(config)
print("Envs created:", len(trainer.envs))
stats = trainer._collect_rollouts()
print("Rollout stats:", stats)
print("Global step:", trainer.global_step)
print("Rollouts saved to 'rollouts_test/' (if enabled)")
