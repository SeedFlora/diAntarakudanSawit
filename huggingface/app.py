"""
Hugging Face Space - Sawit-Banjir RL Training
==============================================
Reinforcement Learning demo untuk simulasi sawit-banjir.
Deploy ke Hugging Face Spaces untuk training DQN.
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import io
import base64

# ============================================================
# SIMULASI ENVIRONMENT (standalone version)
# ============================================================

class PalmFloodEnvSimple:
    """Simplified environment for HF deployment."""
    
    def __init__(self, grid_size=10, palm_ratio=0.6, rainfall=15.0, 
                 drainage=0.2, flood_threshold=50.0):
        self.grid_size = grid_size
        self.palm_ratio = palm_ratio
        self.rainfall = rainfall
        self.drainage = drainage
        self.flood_threshold = flood_threshold
        self.reset()
    
    def reset(self):
        # Initialize land use
        self.land_use = np.full((self.grid_size, self.grid_size), "forest", dtype=object)
        n_cells = self.grid_size * self.grid_size
        n_palm = int(n_cells * self.palm_ratio)
        palm_indices = np.random.choice(n_cells, n_palm, replace=False)
        for idx in palm_indices:
            y, x = divmod(idx, self.grid_size)
            self.land_use[y, x] = "palm"
        
        self.water_level = np.zeros((self.grid_size, self.grid_size))
        self.step_count = 0
        return self._get_obs()
    
    def _get_obs(self):
        land_numeric = np.where(self.land_use == "palm", 1, 0).flatten()
        water_flat = self.water_level.flatten() / self.flood_threshold
        return np.concatenate([land_numeric, water_flat]).astype(np.float32)
    
    def step(self, action=None):
        # Apply action (simplified)
        if action is not None and action > 0:
            y, x = divmod(action - 1, self.grid_size)
            if y < self.grid_size and x < self.grid_size:
                if self.land_use[y, x] == "palm":
                    self.land_use[y, x] = "forest"
                else:
                    self.land_use[y, x] = "palm"
        
        # Rainfall
        self.water_level += self.rainfall
        
        # Infiltration (forest = 1.5x, palm = 0.5x base rate)
        base_infil = 5.0
        infil = np.where(self.land_use == "forest", base_infil * 1.5, base_infil * 0.5)
        self.water_level = np.maximum(0, self.water_level - infil)
        
        # Drainage
        self.water_level *= (1 - self.drainage)
        
        self.step_count += 1
        
        # Reward
        flooded = self.water_level >= self.flood_threshold
        flood_cells = flooded.sum()
        palm_cells = (self.land_use == "palm").sum()
        forest_cells = (self.land_use == "forest").sum()
        
        reward = palm_cells * 1.0 - flood_cells * 2.0 + forest_cells * 0.2
        done = self.step_count >= 30
        
        return self._get_obs(), reward, done, {"flood_cells": flood_cells}
    
    def get_flood_count(self):
        return (self.water_level >= self.flood_threshold).sum()


# ============================================================
# SIMPLE Q-LEARNING (no PyTorch needed)
# ============================================================

class SimpleQLearning:
    """Tabular Q-learning - no deep learning required."""
    
    def __init__(self, n_actions, learning_rate=0.1, gamma=0.95, epsilon=1.0):
        self.q_table = {}
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def _state_key(self, state):
        # Discretize state for tabular Q-learning
        return tuple((state * 10).astype(int))
    
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        key = self._state_key(state)
        if key not in self.q_table:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[key])
    
    def update(self, state, action, reward, next_state, done):
        key = self._state_key(state)
        next_key = self._state_key(next_state)
        
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.n_actions)
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_key])
        
        self.q_table[key][action] += self.lr * (target - self.q_table[key][action])
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ============================================================
# VISUALIZATION
# ============================================================

def create_visualization(env):
    """Create matplotlib figure for the environment."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Land use
    ax = axes[0]
    land_numeric = np.where(env.land_use == "palm", 1, 0)
    cmap = LinearSegmentedColormap.from_list("land", ["#228B22", "#FFD700"])
    ax.imshow(land_numeric, cmap=cmap, vmin=0, vmax=1)
    ax.set_title("Tutupan Lahan", fontweight='bold')
    ax.axis('off')
    forest_patch = mpatches.Patch(color='#228B22', label='Hutan')
    palm_patch = mpatches.Patch(color='#FFD700', label='Sawit')
    ax.legend(handles=[forest_patch, palm_patch], loc='upper right', fontsize=8)
    
    # Water/Flood
    ax = axes[1]
    flooded = env.water_level >= env.flood_threshold
    display = np.zeros((*env.water_level.shape, 3))
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            if flooded[i, j]:
                display[i, j] = [0.8, 0.2, 0.2]
            else:
                intensity = min(env.water_level[i, j] / env.flood_threshold, 1.0)
                display[i, j] = [0.2, 0.4, 0.6 + 0.4 * intensity]
    ax.imshow(display)
    ax.set_title(f"Banjir: {flooded.sum()} sel", fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    return fig


# ============================================================
# GRADIO INTERFACE
# ============================================================

def run_simulation(grid_size, palm_ratio, rainfall, drainage, n_steps):
    """Run simulation and return results."""
    env = PalmFloodEnvSimple(
        grid_size=int(grid_size),
        palm_ratio=palm_ratio/100,
        rainfall=rainfall,
        drainage=drainage/100
    )
    
    history = []
    for _ in range(int(n_steps)):
        _, reward, done, info = env.step(action=0)  # No action
        history.append({
            'flood': info['flood_cells'],
            'reward': reward
        })
        if done:
            env.reset()
    
    fig = create_visualization(env)
    
    # Summary
    avg_flood = np.mean([h['flood'] for h in history])
    max_flood = max([h['flood'] for h in history])
    
    summary = f"""
### Hasil Simulasi
- **Grid**: {grid_size}×{grid_size} sel
- **Sawit**: {palm_ratio}%
- **Curah Hujan**: {rainfall} mm/jam
- **Drainase**: {drainage}%

### Statistik Banjir
- **Rata-rata sel banjir**: {avg_flood:.1f}
- **Maksimum sel banjir**: {max_flood}
- **Total langkah**: {n_steps}
"""
    
    return fig, summary


def run_rl_training(grid_size, palm_ratio, rainfall, drainage, n_episodes):
    """Run Q-Learning training."""
    env = PalmFloodEnvSimple(
        grid_size=int(grid_size),
        palm_ratio=palm_ratio/100,
        rainfall=rainfall,
        drainage=drainage/100
    )
    
    n_actions = grid_size * grid_size + 1  # +1 for no-op
    agent = SimpleQLearning(n_actions)
    
    rewards_history = []
    
    for ep in range(int(n_episodes)):
        state = env.reset()
        total_reward = 0
        
        for _ in range(30):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        
        rewards_history.append(total_reward)
    
    # Plot training curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rewards_history, alpha=0.3, label='Per Episode')
    # Moving average
    window = min(10, len(rewards_history))
    if window > 1:
        ma = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards_history)), ma, 'r-', linewidth=2, label=f'MA-{window}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Q-Learning Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary
    final_avg = np.mean(rewards_history[-10:]) if len(rewards_history) >= 10 else np.mean(rewards_history)
    initial_avg = np.mean(rewards_history[:10]) if len(rewards_history) >= 10 else rewards_history[0]
    improvement = ((final_avg - initial_avg) / abs(initial_avg)) * 100 if initial_avg != 0 else 0
    
    summary = f"""
### Hasil Training Q-Learning
- **Episodes**: {n_episodes}
- **Grid**: {grid_size}×{grid_size}
- **Actions**: {n_actions} (konversi + no-op)

### Performa
- **Reward Awal**: {initial_avg:.2f}
- **Reward Akhir**: {final_avg:.2f}
- **Improvement**: {improvement:+.1f}%
- **Epsilon Final**: {agent.epsilon:.4f}
- **Q-table size**: {len(agent.q_table)} states
"""
    
    return fig, summary


# ============================================================
# GRADIO APP
# ============================================================

with gr.Blocks(title="🌴 Simulasi Sawit-Banjir RL", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌴 Simulasi Sawit & Banjir Sumatera - RL Demo
    
    Reinforcement Learning untuk optimasi tata guna lahan (sawit vs hutan) 
    terhadap risiko banjir di Sumatera.
    
    **Repository**: [GitHub](https://github.com/SeedFlora/diAntarakudanSawit)
    """)
    
    with gr.Tabs():
        with gr.TabItem("🎮 Simulasi"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Parameter")
                    sim_grid = gr.Slider(5, 20, value=10, step=1, label="Ukuran Grid")
                    sim_palm = gr.Slider(10, 90, value=60, step=5, label="Rasio Sawit (%)")
                    sim_rain = gr.Slider(5, 50, value=15, step=1, label="Curah Hujan (mm/jam)")
                    sim_drain = gr.Slider(5, 80, value=20, step=5, label="Drainase (%)")
                    sim_steps = gr.Slider(10, 100, value=30, step=5, label="Langkah Simulasi")
                    sim_btn = gr.Button("▶️ Jalankan Simulasi", variant="primary")
                
                with gr.Column(scale=2):
                    sim_plot = gr.Plot(label="Visualisasi")
                    sim_summary = gr.Markdown()
            
            sim_btn.click(
                run_simulation,
                inputs=[sim_grid, sim_palm, sim_rain, sim_drain, sim_steps],
                outputs=[sim_plot, sim_summary]
            )
        
        with gr.TabItem("🤖 RL Training"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Parameter Training")
                    rl_grid = gr.Slider(5, 15, value=8, step=1, label="Ukuran Grid")
                    rl_palm = gr.Slider(10, 90, value=60, step=5, label="Rasio Sawit (%)")
                    rl_rain = gr.Slider(5, 50, value=15, step=1, label="Curah Hujan (mm/jam)")
                    rl_drain = gr.Slider(5, 80, value=20, step=5, label="Drainase (%)")
                    rl_episodes = gr.Slider(50, 500, value=100, step=50, label="Episodes")
                    rl_btn = gr.Button("🚀 Train Q-Learning", variant="primary")
                
                with gr.Column(scale=2):
                    rl_plot = gr.Plot(label="Training Progress")
                    rl_summary = gr.Markdown()
            
            rl_btn.click(
                run_rl_training,
                inputs=[rl_grid, rl_palm, rl_rain, rl_drain, rl_episodes],
                outputs=[rl_plot, rl_summary]
            )
        
        with gr.TabItem("📚 Info"):
            gr.Markdown("""
            ## Tentang Simulasi
            
            ### 🎯 Tujuan
            Mengeksplorasi hubungan antara ekspansi perkebunan sawit dan risiko banjir
            di wilayah Sumatera menggunakan simulasi berbasis grid.
            
            ### 🔬 Model Hidrologi
            - **Curah Hujan**: Menambah level air di setiap sel
            - **Infiltrasi**: Hutan menyerap air 3× lebih baik dari sawit
            - **Drainase**: Persentase air yang dialirkan keluar per langkah
            - **Banjir**: Terjadi saat level air ≥ threshold (50mm default)
            
            ### 🤖 Reinforcement Learning
            - **State**: Grid tutupan lahan + level air
            - **Actions**: Konversi lahan (sawit↔hutan) atau no-op
            - **Reward**: +profit sawit − penalti banjir + bonus biodiversitas
            - **Algorithm**: Tabular Q-Learning (tanpa deep learning)
            
            ### 📖 Referensi
            - Lubis et al. (2024) - Sawit & banjir di Aceh
            - Lupascu et al. (2020) - Banjir & degradasi gambut SE Asia
            - Tarigan et al. (2020) - Perubahan hidrologi akibat sawit
            
            [📄 Dokumentasi lengkap](https://github.com/SeedFlora/diAntarakudanSawit)
            """)

if __name__ == "__main__":
    demo.launch()
