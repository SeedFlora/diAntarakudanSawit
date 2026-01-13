from .config import SimulationConfig
from .simpy_model import SimPyFloodModel


def run():
    cfg = SimulationConfig(max_steps=24)
    model = SimPyFloodModel(cfg)
    metrics = model.run(hours=cfg.max_steps)
    print(metrics)


if __name__ == "__main__":
    run()
