"""
Reinforcement Learning Environment untuk Simulasi Sawit-Banjir
Gymnasium-compatible wrapper untuk training RL agents
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from .config import SimulationConfig, GridConfig, HydroConfig, EconomicConfig
from .environment import PalmFloodEnv


class PalmFloodGymEnv(gym.Env):
    """
    Gymnasium Environment untuk simulasi sawit-banjir.
    
    Observation Space:
        - land_use: Grid 2D (0=forest, 1=palm) 
        - water_level: Grid 2D kedalaman air (normalized)
        
    Action Space:
        - Discrete: Pilih sel untuk dikonversi (atau no-op)
        - MultiDiscrete: Konversi multiple sel sekaligus
        
    Reward:
        - Profit dari sawit
        - Penalti dari area banjir
        - Bonus biodiversitas dari hutan
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        grid_size: int = 10,
        max_steps: int = 100,
        rainfall_mm: float = 15.0,
        drainage_eff: float = 0.2,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Config
        self.config = SimulationConfig(
            grid=GridConfig(width=grid_size, height=grid_size),
            hydro=HydroConfig(
                rainfall_mm_per_hr=rainfall_mm,
                drainage_efficiency=drainage_eff
            ),
            max_steps=max_steps
        )
        
        # Environment
        self.env = PalmFloodEnv(self.config)
        self.current_step = 0
        
        # Action space: choose cell to convert + action type
        # 0 = no-op, 1 to N*N = convert cell i to opposite type
        n_cells = grid_size * grid_size
        self.action_space = spaces.Discrete(n_cells + 1)  # +1 for no-op
        
        # Observation space: land_use + water_level (flattened)
        self.observation_space = spaces.Dict({
            "land_use": spaces.Box(
                low=0, high=1, shape=(grid_size, grid_size), dtype=np.float32
            ),
            "water_level": spaces.Box(
                low=0, high=200, shape=(grid_size, grid_size), dtype=np.float32
            ),
            "step": spaces.Box(low=0, high=max_steps, shape=(1,), dtype=np.float32)
        })
        
        # For rendering
        self.fig = None
        self.axes = None
        
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Convert environment state to observation."""
        land_use = np.where(
            self.env.state.land_use == "palm", 1.0, 0.0
        ).astype(np.float32)
        
        water_level = self.env.state.water_mm.astype(np.float32)
        
        return {
            "land_use": land_use,
            "water_level": water_level,
            "step": np.array([self.current_step], dtype=np.float32)
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info."""
        metrics = self.env._metrics()
        return {
            "flood_cells": metrics["flood_cells"],
            "palm_cells": metrics["palm_cells"],
            "forest_cells": metrics["forest_cells"],
            "reward_breakdown": {
                "profit": metrics["palm_cells"] * self.config.economic.palm_profit_per_cell,
                "penalty": metrics["flood_cells"] * self.config.economic.flood_penalty_per_cell,
                "biodiversity": metrics["forest_cells"] * self.config.economic.biodiversity_bonus_forest
            }
        }
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)
        
        self.env = PalmFloodEnv(self.config)
        self.current_step = 0
        
        return self._get_obs(), self._get_info()
    
    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step.
        
        Args:
            action: 0 = no-op, 1-N*N = convert cell (action-1) to opposite type
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        actions = []
        
        if action > 0:
            # Convert cell
            cell_idx = action - 1
            y = cell_idx // self.grid_size
            x = cell_idx % self.grid_size
            
            current_type = self.env.state.land_use[y, x]
            new_type = "forest" if current_type == "palm" else "palm"
            actions.append((y, x, new_type))
        
        # Step environment
        metrics = self.env.step(actions)
        self.current_step += 1
        
        # Calculate reward
        reward = float(metrics["reward"])
        
        # Normalize reward
        max_possible = self.grid_size ** 2 * max(
            self.config.economic.palm_profit_per_cell,
            self.config.economic.biodiversity_bonus_forest
        )
        reward = reward / max_possible  # Normalize to roughly [-1, 1]
        
        # Check termination
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            import matplotlib.pyplot as plt
            
            if self.fig is None:
                self.fig, self.axes = plt.subplots(1, 3, figsize=(12, 4))
                plt.ion()
            
            for ax in self.axes:
                ax.clear()
            
            # Land use
            land_map = np.where(self.env.state.land_use == "palm", 1, 0)
            self.axes[0].imshow(land_map, cmap='RdYlGn_r', vmin=0, vmax=1)
            self.axes[0].set_title(f"Tutupan Lahan (Step {self.current_step})")
            
            # Water level
            im = self.axes[1].imshow(self.env.state.water_mm, cmap='Blues')
            self.axes[1].set_title("Kedalaman Air (mm)")
            
            # Flood map
            flooded = self.env.state.water_mm >= self.config.hydro.flood_threshold_mm
            self.axes[2].imshow(flooded, cmap='RdBu_r', vmin=0, vmax=1)
            self.axes[2].set_title(f"Banjir: {flooded.sum()} sel")
            
            plt.tight_layout()
            plt.pause(0.1)
            
        elif self.render_mode == "rgb_array":
            # Return RGB array for video recording
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            land_map = np.where(self.env.state.land_use == "palm", 1, 0)
            axes[0].imshow(land_map, cmap='RdYlGn_r', vmin=0, vmax=1)
            axes[0].set_title(f"Land Use (Step {self.current_step})")
            
            axes[1].imshow(self.env.state.water_mm, cmap='Blues')
            axes[1].set_title("Water (mm)")
            
            flooded = self.env.state.water_mm >= self.config.hydro.flood_threshold_mm
            axes[2].imshow(flooded, cmap='RdBu_r', vmin=0, vmax=1)
            axes[2].set_title(f"Flood: {flooded.sum()}")
            
            plt.tight_layout()
            
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = canvas.buffer_rgba()
            rgb_array = np.asarray(buf)
            plt.close(fig)
            
            return rgb_array
    
    def close(self):
        """Clean up."""
        if self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            self.fig = None


class PalmFloodFlatEnv(PalmFloodGymEnv):
    """
    Flat observation version untuk compatibility dengan standard RL algorithms.
    Observation adalah vector 1D.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Flatten observation space
        n = self.grid_size * self.grid_size
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n * 2 + 1,),  # land_use + water + step
            dtype=np.float32
        )
    
    def _get_obs(self) -> np.ndarray:
        """Flatten observation."""
        obs_dict = super()._get_obs()
        
        land_flat = obs_dict["land_use"].flatten()
        water_flat = obs_dict["water_level"].flatten() / 100.0  # Normalize
        step_norm = obs_dict["step"] / self.max_steps
        
        return np.concatenate([land_flat, water_flat, step_norm]).astype(np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        _, reward, terminated, truncated, info = super().step(action)
        return self._get_obs(), reward, terminated, truncated, info


# Register environments
gym.register(
    id="PalmFlood-v0",
    entry_point="simulasi.gym_env:PalmFloodGymEnv",
    max_episode_steps=100,
)

gym.register(
    id="PalmFloodFlat-v0", 
    entry_point="simulasi.gym_env:PalmFloodFlatEnv",
    max_episode_steps=100,
)
