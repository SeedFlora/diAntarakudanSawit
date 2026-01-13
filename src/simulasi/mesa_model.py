from typing import Tuple
import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation

from .environment import PalmFloodEnv
from .config import SimulationConfig


class LandCell(Agent):
    def __init__(self, unique_id: int, model: Model, pos: Tuple[int, int]):
        super().__init__(unique_id, model)
        self.pos = pos

    def step(self):
        y, x = self.pos
        # Simple heuristic: if flooded, convert to forest to improve infiltration
        flooded = self.model.env.state.water_mm[y, x] >= self.model.config.hydro.flood_threshold_mm
        land_use = self.model.env.state.land_use[y, x]
        if flooded and land_use == "palm":
            self.model.actions.append((y, x, "forest"))
        elif not flooded and land_use == "forest":
            # convert a fraction back to palm for profit if safe
            if np.random.rand() < 0.05:
                self.model.actions.append((y, x, "palm"))


class PalmFloodMesaModel(Model):
    def __init__(self, config: SimulationConfig):
        super().__init__()
        self.config = config
        self.env = PalmFloodEnv(config)
        self.schedule = RandomActivation(self)
        self.actions = []
        uid = 0
        for y in range(config.grid.height):
            for x in range(config.grid.width):
                agent = LandCell(uid, self, (y, x))
                self.schedule.add(agent)
                uid += 1

    def step(self):
        self.actions = []
        self.schedule.step()
        metrics = self.env.step(self.actions)
        return metrics
