import simpy

from .environment import PalmFloodEnv
from .config import SimulationConfig


class RainEvent:
    def __init__(self, env: simpy.Environment, model_env: PalmFloodEnv):
        self.env = env
        self.model_env = model_env
        self.action = env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(1)  # hourly steps
            self.model_env._apply_rainfall()
            self.model_env._apply_infiltration()
            self.model_env._drain()


class DecisionProcess:
    def __init__(self, env: simpy.Environment, model_env: PalmFloodEnv):
        self.env = env
        self.model_env = model_env
        self.action = env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(1)
            # Example: convert 1 random flooded cell to forest per hour
            flooded_positions = (self.model_env.state.water_mm >= self.model_env.config.hydro.flood_threshold_mm).nonzero()
            if len(flooded_positions[0]) > 0:
                y, x = flooded_positions[0][0], flooded_positions[1][0]
                self.model_env.state.land_use[y, x] = "forest"


class SimPyFloodModel:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.env = simpy.Environment()
        self.model_env = PalmFloodEnv(config)
        self.rain = RainEvent(self.env, self.model_env)
        self.decisions = DecisionProcess(self.env, self.model_env)

    def run(self, hours: int):
        self.env.run(until=hours)
        return self.model_env._metrics()
