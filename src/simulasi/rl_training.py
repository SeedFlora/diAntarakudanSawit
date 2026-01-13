"""
Training Script untuk RL Agent pada Simulasi Sawit-Banjir
Menggunakan Stable-Baselines3 (DQN, PPO, A2C)
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os

# Check if stable-baselines3 is available
try:
    from stable_baselines3 import DQN, PPO, A2C
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("stable-baselines3 not installed. Run: pip install stable-baselines3")

from .gym_env import PalmFloodFlatEnv, PalmFloodGymEnv


class TrainingCallback(BaseCallback):
    """Callback untuk tracking training progress."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.flood_cells_history = []
        
    def _on_step(self) -> bool:
        # Get info from environment
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "flood_cells" in info:
                    self.flood_cells_history.append(info["flood_cells"])
        return True
    
    def _on_rollout_end(self) -> None:
        if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            if "r" in ep_info:
                self.episode_rewards.append(ep_info["r"])
            if "l" in ep_info:
                self.episode_lengths.append(ep_info["l"])


class RLTrainer:
    """Trainer untuk RL agents pada simulasi sawit-banjir."""
    
    def __init__(
        self,
        grid_size: int = 8,
        max_steps: int = 50,
        algorithm: str = "DQN",
        rainfall_mm: float = 15.0,
        drainage_eff: float = 0.2
    ):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.algorithm = algorithm
        self.rainfall_mm = rainfall_mm
        self.drainage_eff = drainage_eff
        
        # Create environment
        self.env = PalmFloodFlatEnv(
            grid_size=grid_size,
            max_steps=max_steps,
            rainfall_mm=rainfall_mm,
            drainage_eff=drainage_eff
        )
        
        self.model = None
        self.callback = None
        self.training_history = {
            "rewards": [],
            "flood_cells": [],
            "episodes": []
        }
        
    def create_model(self, **kwargs):
        """Create RL model."""
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 required. Run: pip install stable-baselines3")
        
        # Wrap environment
        env = Monitor(self.env)
        
        if self.algorithm == "DQN":
            self.model = DQN(
                "MlpPolicy",
                env,
                learning_rate=1e-3,
                buffer_size=10000,
                learning_starts=1000,
                batch_size=32,
                gamma=0.99,
                exploration_fraction=0.3,
                exploration_final_eps=0.05,
                verbose=1,
                **kwargs
            )
        elif self.algorithm == "PPO":
            self.model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=256,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                verbose=1,
                **kwargs
            )
        elif self.algorithm == "A2C":
            self.model = A2C(
                "MlpPolicy",
                env,
                learning_rate=7e-4,
                n_steps=5,
                gamma=0.99,
                verbose=1,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        return self.model
    
    def train(self, total_timesteps: int = 10000):
        """Train the model."""
        if self.model is None:
            self.create_model()
        
        self.callback = TrainingCallback()
        
        print(f"\n{'='*60}")
        print(f"Training {self.algorithm} untuk {total_timesteps} timesteps")
        print(f"Grid: {self.grid_size}x{self.grid_size}")
        print(f"Hujan: {self.rainfall_mm} mm/jam, Drainase: {self.drainage_eff*100}%")
        print(f"{'='*60}\n")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.callback,
            progress_bar=True
        )
        
        self.training_history["rewards"] = self.callback.episode_rewards
        self.training_history["flood_cells"] = self.callback.flood_cells_history
        
        return self.model
    
    def evaluate(self, n_episodes: int = 10, render: bool = False) -> Dict:
        """Evaluate trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        results = {
            "episode_rewards": [],
            "episode_lengths": [],
            "final_flood_cells": [],
            "final_palm_cells": [],
            "final_forest_cells": []
        }
        
        for ep in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated
                
                if render:
                    self.env.render()
            
            results["episode_rewards"].append(total_reward)
            results["episode_lengths"].append(steps)
            results["final_flood_cells"].append(info["flood_cells"])
            results["final_palm_cells"].append(info["palm_cells"])
            results["final_forest_cells"].append(info["forest_cells"])
        
        # Calculate statistics
        results["mean_reward"] = np.mean(results["episode_rewards"])
        results["std_reward"] = np.std(results["episode_rewards"])
        results["mean_flood"] = np.mean(results["final_flood_cells"])
        
        return results
    
    def compare_with_baseline(self, n_episodes: int = 10) -> Dict:
        """Compare trained agent with random baseline."""
        # Trained agent
        trained_results = self.evaluate(n_episodes)
        
        # Random baseline
        random_rewards = []
        random_floods = []
        
        for _ in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = self.env.action_space.sample()  # Random action
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
            
            random_rewards.append(total_reward)
            random_floods.append(info["flood_cells"])
        
        comparison = {
            "trained": {
                "mean_reward": trained_results["mean_reward"],
                "std_reward": trained_results["std_reward"],
                "mean_flood": trained_results["mean_flood"]
            },
            "random": {
                "mean_reward": np.mean(random_rewards),
                "std_reward": np.std(random_rewards),
                "mean_flood": np.mean(random_floods)
            },
            "improvement": {
                "reward": trained_results["mean_reward"] - np.mean(random_rewards),
                "flood_reduction": np.mean(random_floods) - trained_results["mean_flood"]
            }
        }
        
        return comparison
    
    def plot_training(self):
        """Plot training progress."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Training Progress: {self.algorithm}", fontsize=14, fontweight='bold')
        
        # Episode rewards
        if self.training_history["rewards"]:
            ax = axes[0, 0]
            rewards = self.training_history["rewards"]
            ax.plot(rewards, alpha=0.6, label='Raw')
            
            # Moving average
            window = min(20, len(rewards) // 5) if len(rewards) > 5 else 1
            if window > 1:
                ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(rewards)), ma, 'r-', linewidth=2, label=f'MA({window})')
            
            ax.set_xlabel("Episode")
            ax.set_ylabel("Total Reward")
            ax.set_title("Episode Rewards")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Flood cells over time
        if self.training_history["flood_cells"]:
            ax = axes[0, 1]
            floods = self.training_history["flood_cells"]
            ax.plot(floods, alpha=0.3)
            
            # Moving average
            window = min(100, len(floods) // 10) if len(floods) > 10 else 1
            if window > 1:
                ma = np.convolve(floods, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(floods)), ma, 'r-', linewidth=2)
            
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Flood Cells")
            ax.set_title("Flood Cells During Training")
            ax.grid(True, alpha=0.3)
        
        # Comparison with baseline
        ax = axes[1, 0]
        comparison = self.compare_with_baseline(n_episodes=5)
        
        labels = ['Trained Agent', 'Random Baseline']
        rewards = [comparison["trained"]["mean_reward"], comparison["random"]["mean_reward"]]
        stds = [comparison["trained"]["std_reward"], comparison["random"]["std_reward"]]
        
        x = np.arange(len(labels))
        bars = ax.bar(x, rewards, yerr=stds, capsize=5, color=['green', 'gray'])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean Reward")
        ax.set_title("Trained vs Random Policy")
        ax.grid(True, alpha=0.3, axis='y')
        
        # Improvement percentage
        if comparison["random"]["mean_reward"] != 0:
            pct = (comparison["improvement"]["reward"] / abs(comparison["random"]["mean_reward"])) * 100
            ax.annotate(f"+{pct:.1f}%", xy=(0, rewards[0]), ha='center', va='bottom', fontweight='bold')
        
        # Summary text
        ax = axes[1, 1]
        ax.axis('off')
        
        summary = f"""
╔══════════════════════════════════════════════════╗
║         HASIL TRAINING RL AGENT                  ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  Algorithm    : {self.algorithm:<30} ║
║  Grid Size    : {self.grid_size}x{self.grid_size:<27} ║
║  Episodes     : {len(self.training_history['rewards']):<30} ║
║                                                  ║
║  TRAINED AGENT:                                  ║
║    Mean Reward : {comparison['trained']['mean_reward']:>8.3f}                    ║
║    Mean Flood  : {comparison['trained']['mean_flood']:>8.1f} sel                 ║
║                                                  ║
║  RANDOM BASELINE:                                ║
║    Mean Reward : {comparison['random']['mean_reward']:>8.3f}                    ║
║    Mean Flood  : {comparison['random']['mean_flood']:>8.1f} sel                 ║
║                                                  ║
║  IMPROVEMENT:                                    ║
║    Reward      : +{comparison['improvement']['reward']:>7.3f}                    ║
║    Flood       : -{comparison['improvement']['flood_reduction']:>7.1f} sel                ║
║                                                  ║
╚══════════════════════════════════════════════════╝
        """
        ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def save_model(self, path: str):
        """Save trained model."""
        if self.model is not None:
            self.model.save(path)
            print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model."""
        if self.algorithm == "DQN":
            self.model = DQN.load(path, env=self.env)
        elif self.algorithm == "PPO":
            self.model = PPO.load(path, env=self.env)
        elif self.algorithm == "A2C":
            self.model = A2C.load(path, env=self.env)
        print(f"Model loaded from {path}")


def demo_rl_training():
    """Demo training RL agent."""
    print("\n" + "="*60)
    print("     DEMO: Reinforcement Learning untuk Sawit-Banjir")
    print("="*60)
    
    if not SB3_AVAILABLE:
        print("\n⚠️  stable-baselines3 belum terinstall!")
        print("   Jalankan: pip install stable-baselines3")
        print("\n   Alternatif: gunakan simple Q-learning di bawah ini...")
        
        # Simple Q-learning demo
        return demo_simple_qlearning()
    
    # Create trainer
    trainer = RLTrainer(
        grid_size=6,
        max_steps=30,
        algorithm="DQN",
        rainfall_mm=15.0,
        drainage_eff=0.2
    )
    
    # Train
    print("\n[1/3] Training DQN agent...")
    trainer.train(total_timesteps=5000)
    
    # Evaluate
    print("\n[2/3] Evaluating...")
    results = trainer.evaluate(n_episodes=5)
    
    print(f"\nHasil Evaluasi:")
    print(f"  Mean Reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"  Mean Flood Cells: {results['mean_flood']:.1f}")
    
    # Compare with baseline
    print("\n[3/3] Comparing with random baseline...")
    comparison = trainer.compare_with_baseline(n_episodes=5)
    
    print(f"\nPerbandingan:")
    print(f"  Trained:  Reward={comparison['trained']['mean_reward']:.3f}, Flood={comparison['trained']['mean_flood']:.1f}")
    print(f"  Random:   Reward={comparison['random']['mean_reward']:.3f}, Flood={comparison['random']['mean_flood']:.1f}")
    print(f"  Improve:  Reward=+{comparison['improvement']['reward']:.3f}, Flood=-{comparison['improvement']['flood_reduction']:.1f}")
    
    # Plot
    fig = trainer.plot_training()
    plt.show()
    
    return trainer


def demo_simple_qlearning():
    """Simple Q-learning tanpa stable-baselines3."""
    print("\n" + "="*60)
    print("     DEMO: Simple Q-Learning (tanpa SB3)")
    print("="*60)
    
    # Create environment
    env = PalmFloodFlatEnv(grid_size=5, max_steps=30)
    
    # Q-table (simplified - just for demo)
    n_actions = env.action_space.n
    
    # Hyperparameters
    alpha = 0.1  # Learning rate
    gamma = 0.99  # Discount factor
    epsilon = 1.0  # Exploration rate
    epsilon_decay = 0.995
    epsilon_min = 0.01
    n_episodes = 200
    
    # Simple state discretization (just use flood count as state)
    def get_state(obs, info):
        return min(info.get("flood_cells", 0), 20)  # Cap at 20
    
    # Initialize Q-table
    q_table = np.zeros((21, n_actions))
    
    rewards_history = []
    flood_history = []
    
    print(f"\nTraining Q-Learning untuk {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        state = get_state(obs, info)
        total_reward = 0
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = get_state(obs, info)
            done = terminated or truncated
            
            # Q-learning update
            best_next = np.max(q_table[next_state])
            q_table[state, action] += alpha * (
                reward + gamma * best_next - q_table[state, action]
            )
            
            state = next_state
            total_reward += reward
        
        rewards_history.append(total_reward)
        flood_history.append(info["flood_cells"])
        
        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            avg_flood = np.mean(flood_history[-50:])
            print(f"  Episode {episode+1}: Avg Reward={avg_reward:.3f}, Avg Flood={avg_flood:.1f}")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Q-Learning Training Results", fontsize=14, fontweight='bold')
    
    # Rewards
    ax = axes[0]
    ax.plot(rewards_history, alpha=0.3)
    window = 20
    ma = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, len(rewards_history)), ma, 'r-', linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Episode Rewards")
    ax.grid(True, alpha=0.3)
    
    # Flood cells
    ax = axes[1]
    ax.plot(flood_history, alpha=0.3)
    ma = np.convolve(flood_history, np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, len(flood_history)), ma, 'r-', linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Final Flood Cells")
    ax.set_title("Flood Cells (Final)")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("Q-Learning selesai!")
    print(f"  Final Avg Reward: {np.mean(rewards_history[-20:]):.3f}")
    print(f"  Final Avg Flood:  {np.mean(flood_history[-20:]):.1f}")
    print("="*60)
    
    return q_table, rewards_history


if __name__ == "__main__":
    demo_rl_training()
