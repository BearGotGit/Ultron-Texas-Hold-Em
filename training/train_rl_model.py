"""
PPO Training Loop for Texas Hold'em.

Implements Proximal Policy Optimization (PPO) for training poker agents.
Features:
    - Rollout buffer for experience collection
    - GAE (Generalized Advantage Estimation)
    - Clipped objective for stable policy updates
    - Value function clipping
    - Entropy bonus for exploration
"""

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from simulation.poker_env import PokerEnv, PokerEnvConfig
from training.ppo_model import PokerPPOModel
from agents.poker_player import PokerPlayer
from agents.monte_carlo_agent import MonteCarloAgent, RandomAgent
from utils.device import DEVICE


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    # Environment
    num_envs: int = 1  # Single environment (single-threaded)
    num_players: int = 2  # Players per table (hero + opponents)
    starting_stack: int = 1000
    big_blind: int = 10
    small_blind: int = 5
    
    # Training
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    
    # PPO specific
    num_steps: int = 128  # Steps per rollout per env
    num_epochs: int = 4  # PPO epochs per update
    num_minibatches: int = 4  # Minibatches per epoch
    clip_coef: float = 0.2  # PPO clipping coefficient
    clip_vloss: bool = True  # Clip value loss
    ent_coef: float = 0.01  # Entropy coefficient
    vf_coef: float = 0.5  # Value function coefficient
    max_grad_norm: float = 0.5  # Gradient clipping
    target_kl: Optional[float] = None  # Target KL divergence
    
    # Model
    card_embed_dim: int = 64
    hidden_dim: int = 256
    num_shared_layers: int = 2
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 50
    eval_episodes: int = 100
    
    # Paths
    run_name: str = field(default_factory=lambda: f"ppo_{int(time.time())}")
    save_dir: str = "checkpoints"
    log_dir: str = "runs"
    
    # Debug
    debug: bool = False
    debug_episodes: int = 3  # Number of episodes to run in debug mode


class RolloutBuffer:
    """Buffer for storing rollout experience."""
    
    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, device: torch.device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.device = device
        
        # Allocate storage
        self.observations = torch.zeros((num_steps, num_envs, obs_dim), device=device)
        self.actions = torch.zeros((num_steps, num_envs, 2), device=device)
        self.log_probs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        
        # For GAE
        self.advantages = torch.zeros((num_steps, num_envs), device=device)
        self.returns = torch.zeros((num_steps, num_envs), device=device)
        
        self.pos = 0
    
    def reset(self):
        self.pos = 0
    
    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ):
        """Add a transition to the buffer."""
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value.squeeze(-1)
        self.pos += 1
    
    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ):
        """Compute returns and GAE advantages."""
        last_value = last_value.squeeze(-1)
        last_gae = 0
        
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
        self.returns = self.advantages + self.values
    
    def get_minibatches(self, num_minibatches: int):
        """Get shuffled minibatches for PPO update."""
        batch_size = self.num_steps * self.num_envs
        minibatch_size = batch_size // num_minibatches
        
        # Flatten
        b_obs = self.observations.reshape(batch_size, -1)
        b_actions = self.actions.reshape(batch_size, -1)
        b_log_probs = self.log_probs.reshape(batch_size)
        b_advantages = self.advantages.reshape(batch_size)
        b_returns = self.returns.reshape(batch_size)
        b_values = self.values.reshape(batch_size)
        
        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        # Shuffle indices
        indices = torch.randperm(batch_size, device=self.device)
        
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_indices = indices[start:end]
            
            yield (
                b_obs[mb_indices],
                b_actions[mb_indices],
                b_log_probs[mb_indices],
                b_advantages[mb_indices],
                b_returns[mb_indices],
                b_values[mb_indices],
            )


class RLPokerAgent(PokerPlayer):
    """
    PokerPlayer wrapper for the PPO model.
    Used during environment interaction.
    """
    
    def __init__(
        self,
        player_id: str,
        model: PokerPPOModel,
        env: PokerEnv,
        device: torch.device,
        starting_money: int = 1000,
    ):
        super().__init__(player_id, starting_money)
        self.model = model
        self.env = env
        self.device = device
    
    def get_action(
        self,
        hole_cards,
        board,
        pot,
        current_bet,
        min_raise,
        players,
        my_idx,
    ):
        """Get action from the model (not used directly - env handles this)."""
        # This is called for opponents, not the hero
        # The hero's actions come from the training loop
        from agents.poker_player import PokerAction
        return PokerAction.check()


class PPOTrainer:
    """PPO Trainer for poker agents."""
    
    def __init__(self, config: PPOConfig, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or DEVICE
        
        # Create model
        self.model = PokerPPOModel(
            card_embed_dim=config.card_embed_dim,
            hidden_dim=config.hidden_dim,
            num_shared_layers=config.num_shared_layers,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        
        # Create environments
        self.envs = self._create_envs()
        
        # Rollout buffer
        obs_dim = self.envs[0].observation_space.shape[0]
        self.buffer = RolloutBuffer(
            num_steps=config.num_steps,
            num_envs=config.num_envs,
            obs_dim=obs_dim,
            device=self.device,
        )
        
        # Logging
        self.writer = SummaryWriter(os.path.join(config.log_dir, config.run_name))
        self.global_step = 0
        self.num_updates = 0
        
        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _create_envs(self) -> List[PokerEnv]:
        """Create parallel environments with opponent agents."""
        envs = []
        
        for i in range(self.config.num_envs):
            # Create opponents (Monte Carlo agents with varying parameters)
            opponents = []
            for j in range(1, self.config.num_players):
                aggression = 0.3 + 0.4 * (j / self.config.num_players)
                opponent = MonteCarloAgent(
                    player_id=f"Opponent-{i}-{j}",
                    starting_money=self.config.starting_stack,
                    num_simulations=100,
                    aggression=aggression,
                )
                opponents.append(opponent)
            
            # Create a placeholder for hero (will be controlled by training loop)
            hero = MonteCarloAgent(
                player_id=f"Hero-{i}",
                starting_money=self.config.starting_stack,
                num_simulations=50,
            )
            
            players = [hero] + opponents
            
            env_config = PokerEnvConfig(
                big_blind=self.config.big_blind,
                small_blind=self.config.small_blind,
                starting_stack=self.config.starting_stack,
                max_players=self.config.num_players,
            )
            
            env = PokerEnv(
                players=players,
                config=env_config,
                hero_idx=0,
            )
            envs.append(env)
        
        return envs
    
    def _collect_rollouts(self) -> Dict[str, float]:
        """Collect rollout experience."""
        self.model.eval()
        self.buffer.reset()
        
        # Reset environments
        observations = []
        for env in self.envs:
            obs, _ = env.reset()
            observations.append(obs)
        
        observations = torch.tensor(np.array(observations), dtype=torch.float32, device=self.device)
        
        episode_rewards_batch = []
        episode_lengths_batch = []
        current_rewards = [0.0] * self.config.num_envs
        current_lengths = [0] * self.config.num_envs
        
        for step in range(self.config.num_steps):
            with torch.no_grad():
                actions, log_probs, _, values = self.model.get_action_and_value(observations)
            
            # Step environments
            next_observations = []
            rewards = []
            dones = []
            
            for i, env in enumerate(self.envs):
                action = actions[i].cpu().numpy()
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                current_rewards[i] += reward
                current_lengths[i] += 1
                
                if terminated or truncated:
                    episode_rewards_batch.append(current_rewards[i])
                    episode_lengths_batch.append(current_lengths[i])
                    current_rewards[i] = 0.0
                    current_lengths[i] = 0
                    next_obs, _ = env.reset()
                
                next_observations.append(next_obs)
                rewards.append(reward)
                dones.append(float(terminated or truncated))
            
            # Convert to tensors
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
            next_observations = torch.tensor(np.array(next_observations), dtype=torch.float32, device=self.device)
            
            # Add to buffer
            self.buffer.add(
                obs=observations,
                action=actions,
                log_prob=log_probs,
                reward=rewards_t,
                done=dones_t,
                value=values,
            )
            
            observations = next_observations
            self.global_step += self.config.num_envs
        
        # Compute final value for bootstrap
        with torch.no_grad():
            last_value = self.model.get_value(observations)
        
        self.buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        
        # Store episode stats
        self.episode_rewards.extend(episode_rewards_batch)
        self.episode_lengths.extend(episode_lengths_batch)
        
        return {
            "mean_reward": np.mean(episode_rewards_batch) if episode_rewards_batch else 0.0,
            "mean_length": np.mean(episode_lengths_batch) if episode_lengths_batch else 0.0,
        }
    
    def _update_policy(self) -> Dict[str, float]:
        """Update policy using PPO."""
        self.model.train()
        
        clip_losses = []
        value_losses = []
        entropy_losses = []
        kl_divs = []
        
        for epoch in range(self.config.num_epochs):
            for batch in self.buffer.get_minibatches(self.config.num_minibatches):
                mb_obs, mb_actions, mb_log_probs, mb_advantages, mb_returns, mb_values = batch
                
                # Get new log probs and values
                new_log_probs, entropy, new_values = self.model.evaluate_actions(mb_obs, mb_actions)
                
                # Ratio for PPO
                ratio = torch.exp(new_log_probs - mb_log_probs)
                
                # Clipped surrogate objective
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                new_values = new_values.squeeze(-1)
                if self.config.clip_vloss:
                    v_loss_unclipped = (new_values - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(
                        new_values - mb_values,
                        -self.config.clip_coef,
                        self.config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss - self.config.ent_coef * entropy_loss + self.config.vf_coef * v_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                clip_losses.append(pg_loss.item())
                value_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                with torch.no_grad():
                    kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    kl_divs.append(kl)
            
            # Early stopping based on KL
            if self.config.target_kl is not None and np.mean(kl_divs) > self.config.target_kl:
                break
        
        self.num_updates += 1
        
        return {
            "pg_loss": np.mean(clip_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropy_losses),
            "kl_div": np.mean(kl_divs),
        }
    
    def train(self, save_path: Optional[str] = None):
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Total timesteps: {self.config.total_timesteps:,}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        num_iterations = self.config.total_timesteps // (self.config.num_steps * self.config.num_envs)
        
        start_time = time.time()
        
        # Progress bar
        pbar = tqdm(range(1, num_iterations + 1), desc="Training", unit="iter")
        
        for iteration in pbar:
            # Collect rollouts
            rollout_stats = self._collect_rollouts()
            
            # Update policy
            update_stats = self._update_policy()
            
            # Update progress bar
            pbar.set_postfix({
                "reward": f"{rollout_stats['mean_reward']:.2f}",
                "ep_len": f"{rollout_stats['mean_length']:.1f}",
                "pg_loss": f"{update_stats['pg_loss']:.3f}",
            })
            
            # Logging
            if iteration % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                fps = self.global_step / elapsed
                
                # TensorBoard logging
                self.writer.add_scalar("rollout/mean_reward", rollout_stats["mean_reward"], self.global_step)
                self.writer.add_scalar("rollout/mean_length", rollout_stats["mean_length"], self.global_step)
                self.writer.add_scalar("losses/pg_loss", update_stats["pg_loss"], self.global_step)
                self.writer.add_scalar("losses/value_loss", update_stats["value_loss"], self.global_step)
                self.writer.add_scalar("losses/entropy", update_stats["entropy"], self.global_step)
                self.writer.add_scalar("losses/kl_div", update_stats["kl_div"], self.global_step)
                self.writer.add_scalar("charts/fps", fps, self.global_step)
            
            # Save checkpoint
            if iteration % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_{iteration}.pt")
            
            # Evaluation
            if iteration % self.config.eval_interval == 0:
                eval_stats = self.evaluate(self.config.eval_episodes)
                tqdm.write(f"\nEvaluation: Win rate = {eval_stats['win_rate']:.2%}, "
                           f"Avg profit = {eval_stats['avg_profit']:.2f}")
                self.writer.add_scalar("eval/win_rate", eval_stats["win_rate"], self.global_step)
                self.writer.add_scalar("eval/avg_profit", eval_stats["avg_profit"], self.global_step)
        
        pbar.close()
        
        # Final save
        final_path = save_path or "final.pt"
        self.save_checkpoint(final_path)
        print(f"\nTraining complete! Total time: {time.time() - start_time:.1f}s")
    
    def evaluate(self, num_episodes: int) -> Dict[str, float]:
        """Evaluate the current policy."""
        self.model.eval()
        
        wins = 0
        total_profit = 0.0
        
        # Use first environment for evaluation
        env = self.envs[0]
        
        for _ in range(num_episodes):
            obs, _ = env.reset()
            initial_chips = env.players[env.hero_idx].money
            done = False
            
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action, _, _, _ = self.model.get_action_and_value(obs_t, deterministic=True)
                action = action.squeeze(0).cpu().numpy()
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            final_chips = env.players[env.hero_idx].money
            profit = final_chips - initial_chips
            
            if profit > 0:
                wins += 1
            total_profit += profit
        
        return {
            "win_rate": wins / num_episodes,
            "avg_profit": total_profit / num_episodes,
        }
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        path = save_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "num_updates": self.num_updates,
            "config": self.config,
        }, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.num_updates = checkpoint["num_updates"]
        print(f"Loaded checkpoint from {path}")
    
    def debug_run(self, num_episodes: int = 3):
        """
        Run detailed debug episodes to understand what's happening.
        
        Prints detailed info about:
        - Opponent setup (what kind of bots)
        - Hero's actions (is it folding?)
        - Rewards and game flow
        - Step-by-step progression
        """
        from treys import Card
        from simulation.poker_env import interpret_action
        from agents.poker_player import ActionType
        
        print("\n" + "="*80)
        print("DEBUG MODE: Running detailed analysis")
        print("="*80)
        
        # Print environment setup
        print("\n--- ENVIRONMENT SETUP ---")
        print(f"Number of players: {self.config.num_players}")
        print(f"Starting stack: {self.config.starting_stack}")
        print(f"Big blind: {self.config.big_blind}")
        print(f"Small blind: {self.config.small_blind}")
        
        # Print opponent types
        print("\n--- PLAYER CONFIGURATION ---")
        env = self.envs[0]
        for i, player in enumerate(env.players):
            player_type = type(player).__name__
            if i == env.hero_idx:
                # Hero is controlled by the RL model (PokerPPOModel), not the placeholder agent type
                print(f"  Player {i} [HERO - RL Agent]: {player.id}")
                print(f"    - Actions controlled by: PokerPPOModel")
            else:
                print(f"  Player {i} [OPPONENT]: {player.id} - Type: {player_type}")
                if hasattr(player, 'aggression'):
                    print(f"    - Aggression: {player.aggression}")
                if hasattr(player, 'num_simulations'):
                    print(f"    - MC Simulations: {player.num_simulations}")
        
        print("\n--- RUNNING DEBUG EPISODES ---")
        self.model.eval()
        
        # Track statistics - use ActionType enum values as keys
        action_counts = {at.value: 0 for at in ActionType}
        total_steps = 0
        total_rewards = 0.0
        saturated_fold_count = 0  # Track saturated p_fold outputs
        fold_to_nothing_count = 0  # Track fold attempts when nothing to call
        
        # Thresholds for issue detection
        FOLD_RATE_THRESHOLD = 0.5  # Warn if folding more than 50%
        MIN_AVG_EPISODE_LENGTH = 1.5  # Warn if avg episode length <= this
        MIN_REWARD_THRESHOLD = 0.001  # Warn if avg reward is below this
        SATURATION_THRESHOLD = 0.99  # p_fold > 0.99 or < 0.01 is saturated
        
        for episode in range(num_episodes):
            print(f"\n{'='*60}")
            print(f"EPISODE {episode + 1}/{num_episodes}")
            print(f"{'='*60}")
            
            obs, info = env.reset()
            
            # Print initial state
            hero = env.players[env.hero_idx]
            hole_cards = hero.get_hole_cards()
            print(f"\nInitial State:")
            print(f"  Hero's hole cards: {[Card.int_to_pretty_str(c) for c in hole_cards]}")
            print(f"  Hero's money: ${hero.money}")
            print(f"  Pot: ${env.pot.money}")
            print(f"  Round stage: {env.round_stage}")
            print(f"  Active player: {env.active_player_idx} (Hero is {env.hero_idx})")
            
            done = False
            step = 0
            episode_reward = 0.0
            
            while not done:
                step += 1
                total_steps += 1
                
                # Get model's action
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action, log_prob, _, value = self.model.get_action_and_value(obs_t)
                action_np = action.squeeze(0).cpu().numpy()
                
                p_fold = float(action_np[0])
                bet_scalar = float(action_np[1])
                
                # Calculate amount to call for debugging
                to_call = env.current_bet - hero.bet
                
                # Interpret the action
                poker_action = interpret_action(
                    p_fold=p_fold,
                    bet_scalar=bet_scalar,
                    current_bet=env.current_bet,
                    my_bet=hero.bet,
                    min_raise=env.min_raise,
                    my_money=hero.money,
                )
                
                # Track action type
                action_type = poker_action.action_type.value
                action_counts[action_type] += 1
                
                # Track saturation and fold-to-nothing
                # Saturation means p_fold is near 0 or near 1 (extreme values)
                is_saturated = (p_fold > SATURATION_THRESHOLD) or (p_fold < (1 - SATURATION_THRESHOLD))
                if is_saturated:
                    saturated_fold_count += 1
                if p_fold > 0.5 and to_call <= 0:
                    fold_to_nothing_count += 1
                
                print(f"\n  Step {step}:")
                print(f"    Round: {env.round_stage}")
                print(f"    Board: {[Card.int_to_pretty_str(c) for c in env.board]}")
                print(f"    Pot: ${env.pot.money}")
                print(f"    Current bet: ${env.current_bet}, Hero's bet: ${hero.bet}, To call: ${to_call}")
                print(f"    Model output: p_fold={p_fold:.3f}, bet_scalar={bet_scalar:.3f}")
                
                # Warn about binary/saturated probabilities
                if is_saturated:
                    print(f"    ⚠️  WARNING: p_fold is saturated (extreme value)")
                
                print(f"    Value estimate: {value.item():.3f}")
                print(f"    Interpreted action: {action_type.upper()} (amount={poker_action.amount})")
                
                # Warn if model wanted to fold but couldn't (nothing to call)
                if p_fold > 0.5 and to_call <= 0:
                    print(f"    ⚠️  NOTE: Model wanted to FOLD but there's nothing to call → converted to CHECK")
                
                # Take step
                next_obs, reward, terminated, truncated, info = env.step(action_np)
                episode_reward += reward
                total_rewards += reward
                
                print(f"    Reward: {reward:.4f}")
                print(f"    Terminated: {terminated}, Truncated: {truncated}")
                
                if terminated or truncated:
                    print(f"\n  Episode ended at step {step}")
                    print(f"  Final hero money: ${env.players[env.hero_idx].money}")
                    print(f"  Total episode reward: {episode_reward:.4f}")
                    done = True
                else:
                    obs = next_obs
            
        print("\n" + "="*80)
        print("DEBUG SUMMARY")
        print("="*80)
        print(f"\nAction distribution over {total_steps} steps:")
        for action_type, count in action_counts.items():
            pct = (count / total_steps * 100) if total_steps > 0 else 0
            print(f"  {action_type.upper()}: {count} ({pct:.1f}%)")
        
        print(f"\nAverage steps per episode: {total_steps / num_episodes:.1f}")
        print(f"Average reward per episode: {total_rewards / num_episodes:.4f}")
        
        # Check for problems
        print("\n--- POTENTIAL ISSUES ---")
        issues_found = False
        
        # Check for saturated fold probabilities
        saturation_rate = saturated_fold_count / max(total_steps, 1)
        if saturation_rate > 0.5:
            print(f"⚠️  {saturation_rate*100:.0f}% of p_fold outputs are saturated (≈0 or ≈1)")
            print("    This indicates the network's fold head weights may need smaller initialization.")
            print("    Nuanced fold probabilities (0.3, 0.6, 0.8) are better for learning.")
            issues_found = True
        
        # Check for fold-to-nothing attempts
        if fold_to_nothing_count > 0:
            print(f"⚠️  Model attempted to FOLD with nothing to call {fold_to_nothing_count} time(s)")
            print("    These were converted to CHECK (folding when no bet is illegal).")
            issues_found = True
        
        fold_rate = action_counts.get(ActionType.FOLD.value, 0) / max(total_steps, 1)
        if fold_rate > FOLD_RATE_THRESHOLD:
            print(f"⚠️  Hero is folding more than {FOLD_RATE_THRESHOLD*100:.0f}% of the time!")
            issues_found = True
        
        avg_episode_length = total_steps / num_episodes
        if avg_episode_length <= MIN_AVG_EPISODE_LENGTH:
            print(f"⚠️  Episodes are ending very quickly (avg <= {MIN_AVG_EPISODE_LENGTH} steps)")
            print("    This suggests hero might be folding immediately or")
            print("    there's an issue with the game flow.")
            issues_found = True
        
        avg_reward = abs(total_rewards / num_episodes)
        if avg_reward < MIN_REWARD_THRESHOLD:
            print(f"⚠️  Rewards are very close to zero (< {MIN_REWARD_THRESHOLD})")
            print("    This could mean the reward function isn't working properly.")
            issues_found = True
            
        if not issues_found:
            print("✓ No obvious issues detected.")
        
        print("\n" + "="*80)
        print("END DEBUG MODE")
        print("="*80 + "\n")


def main():
    """Main entry point for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PPO poker agent")
    parser.add_argument("--total-timesteps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--num-players", type=int, default=2, help="Players per table")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden layer dimension")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for logging")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--save-path", type=str, default=None, help="Custom path for final model save")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (no training, just analysis)")
    parser.add_argument("--debug-episodes", type=int, default=3, help="Number of episodes to run in debug mode")
    args = parser.parse_args()
    
    config = PPOConfig(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_players=args.num_players,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        run_name=args.run_name or f"ppo_{int(time.time())}",
        debug=args.debug,
        debug_episodes=args.debug_episodes,
    )
    
    trainer = PPOTrainer(config)
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    if config.debug:
        trainer.debug_run(num_episodes=config.debug_episodes)
    else:
        trainer.train(save_path=args.save_path)


if __name__ == "__main__":
    main()


