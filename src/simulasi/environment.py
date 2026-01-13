from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from .config import SimulationConfig


CellType = str  # "palm", "forest", "settlement"


@dataclass
class EnvState:
    land_use: np.ndarray  # shape (h, w) storing CellType codes as strings
    water_mm: np.ndarray  # shape (h, w) water depth


class PalmFloodEnv:
    """Minimal grid hydrology toy model."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        h, w = config.grid.height, config.grid.width
        
        # Initialize land use based on initial_palm_ratio
        palm_ratio = getattr(config.grid, 'initial_palm_ratio', 0.6)
        land_use = np.full((h, w), "forest", dtype=object)
        
        # Randomly assign palm cells
        n_cells = h * w
        n_palm = int(n_cells * palm_ratio)
        palm_indices = np.random.choice(n_cells, n_palm, replace=False)
        for idx in palm_indices:
            y, x = divmod(idx, w)
            land_use[y, x] = "palm"
        
        self.land_use = land_use
        self.water_level_mm = np.zeros((h, w), dtype=float)
        self.state = EnvState(land_use=land_use, water_mm=self.water_level_mm)
        self.step_count = 0

    def step(self, actions: List[Tuple[int, int, CellType]] = None):
        """Apply actions (cell conversions), then update water balance."""
        if actions:
            for y, x, land in actions:
                if 0 <= y < self.land_use.shape[0] and 0 <= x < self.land_use.shape[1]:
                    self.land_use[y, x] = land
        self._apply_rainfall()
        self._apply_infiltration()
        self._drain()
        self.step_count += 1
        self.state = EnvState(land_use=self.land_use, water_mm=self.water_level_mm)
        return self._metrics()

    def _apply_rainfall(self):
        # Support both naming conventions
        rainfall = getattr(self.config.hydro, 'rainfall_mm_per_hour', None)
        if rainfall is None:
            rainfall = self.config.hydro.rainfall_mm_per_hr
        self.water_level_mm += rainfall

    def _apply_infiltration(self):
        infil = np.where(
            self.land_use == "forest",
            self.config.hydro.base_infiltration_mm_per_hr * 1.5,
            self.config.hydro.base_infiltration_mm_per_hr * 0.5,
        )
        self.water_level_mm = np.maximum(0.0, self.water_level_mm - infil)

    def _drain(self):
        self.water_level_mm *= 1.0 - self.config.hydro.drainage_efficiency

    def get_flood_count(self) -> int:
        """Return number of flooded cells."""
        flooded = self.water_level_mm >= self.config.hydro.flood_threshold_mm
        return int(flooded.sum())

    def calculate_revenue(self) -> float:
        """Calculate economic revenue."""
        palm_cells = int((self.land_use == "palm").sum())
        flood_cells = self.get_flood_count()
        
        revenue_per_ha = getattr(self.config.economic, 'palm_revenue_per_ha', 25.0)
        damage_per_ha = getattr(self.config.economic, 'flood_damage_per_ha', 10.0)
        
        revenue = palm_cells * revenue_per_ha - flood_cells * damage_per_ha
        return revenue

    def _metrics(self):
        flooded = self.water_level_mm >= self.config.hydro.flood_threshold_mm
        flood_cells = int(flooded.sum())
        palm_cells = int((self.land_use == "palm").sum())
        forest_cells = int((self.land_use == "forest").sum())
        profit = palm_cells * self.config.economic.palm_profit_per_cell
        penalty = flood_cells * self.config.economic.flood_penalty_per_cell
        biodiversity = forest_cells * self.config.economic.biodiversity_bonus_forest
        reward = profit - penalty + biodiversity
        return {
            "step": self.step_count,
            "flood_cells": flood_cells,
            "palm_cells": palm_cells,
            "forest_cells": forest_cells,
            "reward": reward,
        }

    def reset(self):
        self.__init__(self.config)
        return self._metrics()

